"""Thread-safe multi-agent shared-pool sandbox scheduler (`SandboxPool`).

Multiplexes N concurrent agents across C warm RSIBox `Lab` (or ready `Box`) slots with
strict priority queueing (FIFO tie-break), role routing, fenced leases, failover/quarantine, and JSONL telemetry.
"""
from __future__ import annotations

from collections import deque, namedtuple
import contextlib, dataclasses, hashlib, json, math, operator, os, threading, time, types
from typing import Any, Callable, Iterator, Sequence
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None
from .box import Box

__all__ = ["PoolClosedError", "PoolError", "PoolSlot", "PoolTimeoutError", "SandboxPool", "SlotLease", "verify_trace"]
_GENESIS_HASH = "genesis"
_CleanupOutcome = namedtuple("_CleanupOutcome", "healthy cleanup_s recovery_s recovered status_after error")
_RUN_RESULT_KEYS = ("queue_wait_s", "hold_s", "cleanup_s", "recovery_s", "recovered", "status_after", "error")

class PoolError(RuntimeError): """Base error raised by `SandboxPool`."""
class PoolTimeoutError(TimeoutError, PoolError): """Raised when waiting for a healthy matching slot times out."""
class PoolClosedError(PoolError): """Raised when attempting to acquire a slot from a closed `SandboxPool`."""

def _format_error(error: str | BaseException | None) -> str | None:
    if error is None:
        return None
    return f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else (str(error).strip() or None)
def _close_quietly(lab: Any) -> None:
    if lab is not None:
        with contextlib.suppress(Exception):
            lab.close()
def _validate_priority(priority: int) -> int:
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise TypeError(f"priority must be an int, not {type(priority).__name__}")
    return priority
def _validate_timeout(timeout_s: float | None, name: str = "timeout_s") -> float | None:
    if timeout_s is None:
        return None
    if isinstance(timeout_s, bool):
        raise TypeError(f"{name} must be a real number or None, not bool")
    value = float(timeout_s)
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    return (None if value > 0 else 0.0) if math.isinf(value) else max(0.0, value)
def _safe_lab_call(lab: Any, method_name: str, *args: Any) -> dict[str, Any]:
    method = getattr(lab, method_name, None)
    if not callable(method):
        return {"ok": False, "error": f"lab does not implement {method_name}()"}
    try:
        reply = method(*args)
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return reply if isinstance(reply, dict) else {"ok": False, "error": f"{method_name}() returned {type(reply).__name__}"}
def _is_clean_base(status_reply: dict[str, Any]) -> bool:
    return (bool(status_reply.get("ok")) and status_reply.get("state") == "base"
            and not status_reply.get("txns") and not status_reply.get("poison"))
def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))])
def _hash_trace_record(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
def _validate_trace_record(record: Any, expected_sequence: int | None, expected_previous: str | None) -> tuple[str | None, int, str]:
    if not isinstance(record, dict):
        return f"expected JSON object, got {type(record).__name__}", 0, ""
    sequence, digest = record.get("seq"), record.get("record_sha256")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1 or (
        expected_sequence is not None and sequence != expected_sequence
    ):
        return "sequence mismatch", 0, ""
    if expected_previous is not None and record.get("prev_sha256") != expected_previous:
        return "prev_sha256 mismatch", 0, ""
    if not isinstance(digest, str) or digest != _hash_trace_record(record):
        return "record digest mismatch", 0, ""
    return None, sequence, digest
def _chain_record(record: dict[str, Any], last_sequence: int, last_hash: str) -> tuple[dict[str, Any], int, str]:
    next_sequence = last_sequence + 1
    chained = {**record, "seq": next_sequence, "prev_sha256": last_hash}
    chained["record_sha256"] = digest = _hash_trace_record(chained)
    return chained, next_sequence, digest
def _read_trace_tail(handle: Any) -> tuple[int, str]:
    handle.seek(0)
    last_line = next((line for line in reversed(handle.readlines()) if line.strip()), "")
    if not last_line:
        return 0, _GENESIS_HASH
    try:
        error_msg, last_sequence, last_hash = _validate_trace_record(json.loads(last_line), None, None)
    except ValueError as error:
        raise PoolError(f"corrupt trace tail: {error}") from error
    if error_msg is not None:
        raise PoolError(f"corrupt trace tail: {error_msg}")
    return last_sequence, last_hash
def verify_trace(trace_path: str, *, expected_head: tuple[int, str] | None = None) -> dict[str, Any]:
    """Verify JSONL hash-chain integrity; pass `expected_head` to also detect tail truncation."""
    records: list[dict[str, Any]] = []
    expected_previous, current_line = _GENESIS_HASH, 0
    try:
        with open(trace_path, encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                current_line = line_number
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                error_msg, _, digest = _validate_trace_record(record, len(records) + 1, expected_previous)
                if error_msg is not None:
                    return {"ok": False, "count": len(records), "line": line_number, "error": error_msg}
                expected_previous = digest
                records.append(record)
    except (OSError, ValueError, TypeError) as error:
        failure: dict[str, Any] = {"ok": False, "count": len(records), "error": f"{type(error).__name__}: {error}"}
        return {**failure, "line": current_line} if current_line else failure
    head = (len(records), expected_previous)
    if expected_head is not None and head != expected_head:
        return {"ok": False, "count": len(records), "head": head,
                "error": f"trace head mismatch: expected {expected_head!r}, got {head!r}"}
    return {"ok": True, "count": len(records), "head": head, "records": records}

class _BoxLabAdapter:
    """Thin 5-verb `Lab` proxy over a ready `Box`, preserving its call gate and event log."""
    def __init__(self, box: Box) -> None: self._box = box
    def install(self, patch: dict[str, Any]) -> dict[str, Any]: return self._box.call("install", {"patch": patch})[1]
    def check(self, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._box.call("check", {"workload": workload} if workload is not None else {})[1]
    def reset(self, kind: str = "state") -> dict[str, Any]: return self._box.call("reset", {"kind": kind})[1]
    def uninstall(self, id: str | None = None) -> dict[str, Any]: return self._box.call("uninstall", {"id": id} if id else {})[1]
    def status(self) -> dict[str, Any]: return self._box.call("status")[1]
    def source(self, module_name: str) -> dict[str, Any]: return self._box.source(module_name)[1]
    def receipts(self) -> dict[str, Any]: return self._box.read_receipts()
    def close(self) -> dict[str, Any]: return self._box.close()
    def __getattr__(self, attribute_name: str) -> Any:
        if self._box.lab is None:
            raise AttributeError(f"box is not ready; cannot access {attribute_name!r}")
        return getattr(self._box.lab, attribute_name)

class _FencedLabProxy:
    """Fenced proxy around a leased `Lab` that tracks in-flight calls and rejects post-release access."""
    def __init__(self, lease: SlotLease, target: Any) -> None:
        object.__setattr__(self, "_lease", lease)
        object.__setattr__(self, "_target", target)
    def _run_fenced(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        lease = self._lease
        with lease._call_cond:
            if lease._released:
                raise PoolError(f"lease for slot {lease._slot_id!r} has already been released")
            lease._active_calls += 1
        try:
            return operation(*args, **kwargs)
        finally:
            with lease._call_cond:
                lease._active_calls -= 1
                if lease._active_calls == 0:
                    lease._call_cond.notify_all()
    def close(self) -> None:
        self._run_fenced(lambda: None)
        raise PoolError("cannot close a pooled lab via SlotLease; call SandboxPool.close()")
    def __getattr__(self, attribute_name: str) -> Any:
        value = self._run_fenced(getattr, self._target, attribute_name)
        if not callable(value):
            return value
        return lambda *args, **kwargs: self._run_fenced(value, *args, **kwargs)
    def __setattr__(self, attribute_name: str, value: Any) -> None:
        self._run_fenced(setattr, self._target, attribute_name, value)

@dataclasses.dataclass(frozen=True)
class _SlotView:
    """Immutable view of slot identity and metadata exposed via `SlotLease.slot`."""
    slot_id: str
    role: str | None
    metadata: types.MappingProxyType

class PoolSlot:
    """One warm RSIBox `Lab` (or ready `Box`, `state == "ready"`) bound to a named slot and role."""
    def __init__(self, slot_id: str, lab: Any, role: str | None = None,
                 metadata: dict[str, Any] | None = None) -> None:
        if not isinstance(slot_id, str) or not slot_id.strip():
            raise ValueError("slot_id must be a non-empty string")
        if role is not None and (not isinstance(role, str) or not role.strip()):
            raise ValueError("role must be None or a non-empty string")
        if isinstance(lab, Box) and lab.state != "ready":
            raise ValueError(f"Box for slot {slot_id!r} must be ready (state={lab.state!r})")
        self.slot_id, self.role, self.metadata = slot_id, role, dict(metadata or {})
        self.box: Box | None = lab if isinstance(lab, Box) else None
        self.lab: Any = _BoxLabAdapter(lab) if isinstance(lab, Box) else lab
        self.quarantined, self.quarantine_reason = False, None

class _SlotState:
    __slots__ = ("slot", "slot_id", "role", "lab", "view", "quarantined", "quarantine_reason",
                 "in_use", "active_since", "dirty", "busy_s", "lab_closed")
    def __init__(self, slot: PoolSlot) -> None:
        self.slot, self.slot_id, self.role, self.lab = slot, slot.slot_id, slot.role, slot.lab
        self.view = _SlotView(slot.slot_id, slot.role, types.MappingProxyType(dict(slot.metadata)))
        self.quarantined, self.quarantine_reason = False, None
        self.in_use, self.active_since, self.dirty, self.busy_s, self.lab_closed = False, None, True, 0.0, False

class _Waiter:
    __slots__ = ("role", "priority", "event", "assigned_state", "assigned_dirty", "cancelled", "failure", "closed")
    def __init__(self, role: str | None, priority: int = 0) -> None:
        self.role, self.priority = role, priority
        self.event, self.assigned_state, self.assigned_dirty = threading.Event(), None, True
        self.cancelled, self.failure, self.closed = False, None, False

class SlotLease:
    """Exclusive lease over one `PoolSlot`; supports `with` and idempotent `release()`."""
    slot = property(operator.attrgetter("_slot_view"))
    slot_id = property(operator.attrgetter("_slot_id"))
    agent_id = property(operator.attrgetter("_agent_id"))
    role = property(operator.attrgetter("_role"))
    priority = property(operator.attrgetter("_priority"))
    idea_id = property(operator.attrgetter("_idea_id"))
    acquired_at = property(operator.attrgetter("_acquired_at"))
    queue_wait_s = property(operator.attrgetter("_queue_wait_s"))
    lab = property(operator.attrgetter("_lab_proxy"))
    released = property(operator.attrgetter("_released"))

    def __init__(self, pool: SandboxPool, slot_state: _SlotState, agent_id: str, role: str | None,
                 idea_id: str | None, acquired_at: float, acquired_mono: float,
                 queue_wait_s: float, preparation: _CleanupOutcome, *, priority: int = 0) -> None:
        self._pool, self._lab, self._slot_view, self._slot_id = (
            pool, slot_state.lab, slot_state.view, slot_state.slot_id,
        )
        self._agent_id, self._role, self._priority, self._idea_id = agent_id, role, priority, idea_id
        self._acquired_at, self._acquired_mono, self._queue_wait_s = acquired_at, acquired_mono, queue_wait_s
        self._preparation = preparation
        self.cleanup_s, self.recovery_s, self.recovered = preparation.cleanup_s, preparation.recovery_s, preparation.recovered
        self.released_at = self.hold_s = self.error = self.status_after = None
        self._released, self._release_lock, self._call_cond, self._active_calls = (
            False, threading.Lock(), threading.Condition(), 0,
        )
        self._trace_record, self._trace_error = None, None
        self._lab_proxy: Any = _FencedLabProxy(self, slot_state.lab)

    def release(self, error: str | BaseException | None = None) -> dict[str, Any]:
        return self._pool._release_lease(self, error=error)
    def __enter__(self) -> SlotLease: return self
    def __exit__(self, exc_type: Any, exc_value: BaseException | None, exc_traceback: Any) -> bool:
        self.release(error=exc_value)
        return False

class SandboxPool:
    """Thread-safe priority-and-FIFO scheduler multiplexing N agents over C warm RSIBox slots."""

    def __init__(self, slots: Sequence[PoolSlot], *, trace_path: str | None = None,
                 clock: Callable[[], float] = time.monotonic, wall_clock: Callable[[], float] = time.time,
                 close_labs: bool = False, drain_timeout_s: float | None = 5.0) -> None:
        if not slots:
            raise ValueError("SandboxPool requires at least one PoolSlot")
        self._drain_timeout_s = _validate_timeout(drain_timeout_s, "drain_timeout_s")
        self._slots: dict[str, _SlotState] = {}
        seen_targets: set[int] = set()
        for slot in slots:
            if not isinstance(slot, PoolSlot):
                raise TypeError(f"expected PoolSlot, got {type(slot).__name__}")
            if slot.slot_id in self._slots:
                raise ValueError(f"duplicate slot_id: {slot.slot_id!r}")
            raw_target = slot.box.lab if (slot.box is not None and slot.box.lab is not None) else (slot.box or slot.lab)
            if id(raw_target) in seen_targets:
                raise ValueError(f"duplicate underlying lab/box for slot_id: {slot.slot_id!r}")
            seen_targets.add(id(raw_target))
            self._slots[slot.slot_id] = _SlotState(slot)
        self._trace_path, self._clock, self._wall_clock = trace_path, clock, wall_clock
        self._close_labs, self._created_mono = close_labs, clock()
        self._lock, self._trace_lock, self._waiters = threading.Lock(), threading.Lock(), deque()
        self._closed, self._total_leases, self._recoveries, self._total_recovery_s = False, 0, 0, 0.0
        self._queue_waits, self._hold_times, self._trace_sequence, self._trace_previous_hash = [], [], 0, _GENESIS_HASH
        if trace_path and os.path.isfile(trace_path):
            verified = verify_trace(trace_path)
            if not verified["ok"]:
                raise PoolError(f"existing trace {trace_path!r} is invalid: {verified.get('error')}")
            self._trace_sequence, self._trace_previous_hash = verified["head"]

    def _safe_clock(self, fallback: float | None = None) -> float:
        try:
            return self._clock()
        except Exception:
            return self._created_mono if fallback is None else fallback
    def _has_healthy_slot(self, role: str | None) -> bool:
        return any(not slot_state.quarantined and (role is None or slot_state.role == role)
                   for slot_state in self._slots.values())
    def _mark_lab_to_close_locked(self, slot_state: _SlotState) -> Any | None:
        if self._closed and self._close_labs and not slot_state.lab_closed:
            slot_state.lab_closed = True
            return slot_state.lab
        return None

    def _complete_slot_locked(self, slot_id: str, finish_mono: float, healthy: bool, reason: str | None,
                              *, preserve_dirty: bool = False, skip_close: bool = False) -> Any | None:
        slot_state = self._slots[slot_id]
        if slot_state.active_since is not None:
            slot_state.busy_s += max(0.0, finish_mono - slot_state.active_since)
        slot_state.quarantined, slot_state.dirty = not healthy, (True if not healthy else preserve_dirty)
        slot_state.quarantine_reason = None if healthy else (reason or "recovery failed")
        slot_state.slot.quarantined, slot_state.slot.quarantine_reason = slot_state.quarantined, slot_state.quarantine_reason
        if not self._closed and healthy:
            best_index: int | None = None
            best_priority = 0
            for index, waiter in enumerate(self._waiters):
                if waiter.cancelled or (waiter.role is not None and waiter.role != slot_state.role):
                    continue
                if best_index is None or waiter.priority > best_priority:
                    best_index, best_priority = index, waiter.priority
            if best_index is not None:
                waiter = self._waiters[best_index]
                del self._waiters[best_index]
                waiter.assigned_state, waiter.assigned_dirty = slot_state, slot_state.dirty
                slot_state.in_use, slot_state.active_since = True, finish_mono
                waiter.event.set()
                return None
        slot_state.in_use, slot_state.active_since = False, None
        if not healthy:
            for waiter in self._waiters:
                if not waiter.cancelled and not self._has_healthy_slot(waiter.role):
                    waiter.failure = f"no healthy slot available for role={waiter.role!r}"
                    waiter.event.set()
            self._waiters = deque(waiter for waiter in self._waiters if not waiter.cancelled and waiter.failure is None)
        return None if skip_close else self._mark_lab_to_close_locked(slot_state)
    def _finish_slot(self, slot_id: str, finish_mono: float, healthy: bool, reason: str | None,
                     *, preserve_dirty: bool = False, extra_recovery_s: float = 0.0,
                     skip_close: bool = False) -> None:
        with self._lock:
            self._total_recovery_s += extra_recovery_s
            lab_to_close = self._complete_slot_locked(slot_id, finish_mono, healthy, reason,
                                                      preserve_dirty=preserve_dirty, skip_close=skip_close)
        _close_quietly(lab_to_close)
    def _cancel_waiter_locked(self, waiter: _Waiter) -> Any | None:
        waiter.cancelled = True
        with contextlib.suppress(ValueError):
            self._waiters.remove(waiter)
        if waiter.assigned_state is None:
            return None
        return self._complete_slot_locked(waiter.assigned_state.slot_id, self._safe_clock(), True, None,
                                          preserve_dirty=waiter.assigned_dirty)

    def _claim_slot(self, role: str | None, deadline: float | None, *, priority: int = 0) -> tuple[_SlotState, bool]:
        with self._lock:
            if self._closed:
                raise PoolClosedError("SandboxPool is closed")
            if not self._has_healthy_slot(role):
                raise PoolTimeoutError(f"no healthy slot available for role={role!r}")
            for slot_state in self._slots.values():
                if slot_state.in_use or slot_state.quarantined or (role is not None and slot_state.role != role):
                    continue
                if any(not waiter.cancelled and (waiter.role is None or waiter.role == slot_state.role)
                       and waiter.priority >= priority for waiter in self._waiters):
                    continue
                slot_state.in_use, slot_state.active_since = True, self._clock()
                return slot_state, slot_state.dirty
            if deadline is not None and self._clock() >= deadline:
                raise PoolTimeoutError(f"timed out waiting for slot (role={role!r})")
            waiter = _Waiter(role, priority)
            self._waiters.append(waiter)
        try:
            while True:
                remaining_s = None if deadline is None else deadline - self._clock()
                if remaining_s is not None and remaining_s <= 0.0:
                    break
                if waiter.event.wait(timeout=None if remaining_s is None else min(remaining_s, threading.TIMEOUT_MAX)):
                    break
        except BaseException:
            with self._lock:
                lab_to_close = self._cancel_waiter_locked(waiter)
            _close_quietly(lab_to_close)
            raise
        with self._lock:
            if waiter.assigned_state is not None and not self._closed:
                return waiter.assigned_state, waiter.assigned_dirty
            closed_error = self._closed or waiter.closed
            lab_to_close = self._cancel_waiter_locked(waiter)
        _close_quietly(lab_to_close)
        if closed_error:
            raise PoolClosedError("SandboxPool closed while waiting for slot")
        raise PoolTimeoutError(waiter.failure or f"timed out waiting for slot (role={role!r})")

    def _cleanup_in_place(self, lab: Any, *, force_reset_state: bool) -> str | None:
        status = _safe_lab_call(lab, "status")
        if not status.get("ok") or status.get("state") == "poisoned" or status.get("poison"):
            return str(status.get("error") or "slot poisoned")
        remaining = list(status.get("txns") or ())
        had_transactions = bool(remaining)
        for _ in range(len(remaining)):
            reply = _safe_lab_call(lab, "uninstall", remaining[0])
            if not reply.get("ok") or reply.get("state") == "poisoned":
                return str(reply.get("error") or "uninstall failed")
            if reply.get("state") == "base":
                remaining = []
                break
            recheck = _safe_lab_call(lab, "status")
            if not recheck.get("ok") or recheck.get("state") == "poisoned":
                return str(recheck.get("error") or "slot poisoned after uninstall")
            remaining = list(recheck.get("txns") or ())
            if not remaining:
                break
        if remaining:
            return "uninstall did not clear transaction stack"
        if (force_reset_state or had_transactions) and not (reset_reply := _safe_lab_call(lab, "reset", "state")).get("ok"):
            return str(reset_reply.get("error") or 'reset("state") failed')
        after = _safe_lab_call(lab, "status")
        return None if _is_clean_base(after) else str(after.get("error") or f"post-cleanup state was {after.get('state', 'unknown')!r}")

    def _clean_and_recover_slot(self, lab: Any, *, force_reset_state: bool) -> _CleanupOutcome:
        cleanup_start = self._clock()
        cleanup_error = self._cleanup_in_place(lab, force_reset_state=force_reset_state)
        cleanup_s = max(0.0, self._clock() - cleanup_start)
        if cleanup_error is None:
            return _CleanupOutcome(True, cleanup_s, 0.0, False, "base", None)
        recovery_start = self._clock()
        respawn = _safe_lab_call(lab, "reset", "respawn")
        recovery_s = max(0.0, self._clock() - recovery_start)
        if respawn.get("ok") and _is_clean_base(verified := _safe_lab_call(lab, "status")):
            return _CleanupOutcome(True, cleanup_s, recovery_s, True, "base", None)
        reason = (respawn.get("error") or cleanup_error or 'reset("respawn") failed'
                  if not respawn.get("ok") else
                  verified.get("error") or f"not clean after respawn (state={verified.get('state')!r})")
        return _CleanupOutcome(False, cleanup_s, recovery_s, False, "quarantined", str(reason))

    def acquire(self, agent_id: str, *, role: str | None = None,
                idea_id: str | None = None, timeout_s: float | None = None,
                priority: int = 0) -> SlotLease:
        """Acquire an exclusive, clean-slate `SlotLease` (highest priority first, FIFO tie-break)."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        normalized_priority = _validate_priority(priority)
        normalized_timeout = _validate_timeout(timeout_s)
        queue_start = self._clock()
        deadline = None if normalized_timeout is None else queue_start + normalized_timeout
        while True:
            slot_state, slot_dirty = self._claim_slot(role, deadline, priority=normalized_priority)
            slot_returned = False
            try:
                claim_mono = self._clock()
                preparation = (_CleanupOutcome(True, 0.0, 0.0, False, "base", None)
                               if (not slot_dirty and _is_clean_base(_safe_lab_call(slot_state.lab, "status")))
                               else self._clean_and_recover_slot(slot_state.lab, force_reset_state=True))
                if preparation.healthy:
                    ready_mono, ready_wall = self._clock(), self._wall_clock()
                    with self._lock:
                        if not self._closed:
                            slot_state.dirty = False
                            return SlotLease(self, slot_state, agent_id, role, idea_id, ready_wall, ready_mono,
                                             round(max(0.0, claim_mono - queue_start), 6), preparation,
                                             priority=normalized_priority)
                        lab_to_close = self._complete_slot_locked(slot_state.slot_id, ready_mono, True, None)
                        slot_returned = True
                    _close_quietly(lab_to_close)
                    raise PoolClosedError("SandboxPool closed while preparing slot")
                self._finish_slot(slot_state.slot_id, self._safe_clock(fallback=claim_mono), False,
                                  preparation.error or "pre-acquire recovery failed",
                                  extra_recovery_s=preparation.recovery_s)
                slot_returned = True
            except BaseException:
                if not slot_returned:
                    self._finish_slot(slot_state.slot_id, self._safe_clock(fallback=queue_start),
                                      True, None, preserve_dirty=True)
                raise

    @contextlib.contextmanager
    def lease(self, agent_id: str, *, role: str | None = None,
              idea_id: str | None = None, timeout_s: float | None = None,
              priority: int = 0) -> Iterator[SlotLease]:
        """Context manager acquiring (by highest `priority`, FIFO tie-break) and releasing a `SlotLease`."""
        slot_lease = self.acquire(agent_id, role=role, idea_id=idea_id, timeout_s=timeout_s, priority=priority)
        caught_error: BaseException | None = None
        try:
            yield slot_lease
        except BaseException as error:
            caught_error = error
            raise
        finally:
            slot_lease.release(error=caught_error)

    def _release_lease(self, lease: SlotLease, *, error: str | BaseException | None = None) -> dict[str, Any]:
        with lease._release_lock:
            with lease._call_cond:
                if lease._released:
                    if lease._trace_error is not None:
                        raise lease._trace_error
                    return dict(lease._trace_record or {})
                lease._released = True
            release_mono = lease._acquired_mono
            undrained, cleanup, finish_mono, finalized = True, None, None, None
            try:
                release_mono = self._safe_clock(fallback=lease._acquired_mono)
                with lease._call_cond:
                    drain_deadline = None if self._drain_timeout_s is None else time.monotonic() + self._drain_timeout_s
                    while lease._active_calls > 0:
                        remaining_s = None if drain_deadline is None else drain_deadline - time.monotonic()
                        if remaining_s is not None and remaining_s <= 0.0:
                            break
                        lease._call_cond.wait(
                            timeout=None if remaining_s is None else min(remaining_s, threading.TIMEOUT_MAX)
                        )
                    undrained = lease._active_calls > 0
                hold_s, caller_error = max(0.0, release_mono - lease._acquired_mono), _format_error(error)
                released_at = self._wall_clock()
                cleanup = (
                    _CleanupOutcome(False, 0.0, 0.0, False, "quarantined",
                                    "in-flight fenced call did not drain before release")
                    if undrained else self._clean_and_recover_slot(lease._lab, force_reset_state=True)
                )
                finish_mono = self._clock()
                raw_cleanup_s = lease._preparation.cleanup_s + cleanup.cleanup_s
                raw_recovery_s = lease._preparation.recovery_s + cleanup.recovery_s
                recovered = lease._preparation.recovered or cleanup.recovered
                combined_error = (f"{caller_error}; cleanup: {cleanup.error}"
                                  if (caller_error and cleanup.error) else (caller_error or cleanup.error))
                lease.released_at, lease.hold_s = round(released_at, 6), round(hold_s, 6)
                lease.cleanup_s, lease.recovery_s = round(raw_cleanup_s, 6), round(raw_recovery_s, 6)
                lease.recovered, lease.error, lease.status_after = recovered, combined_error, cleanup.status_after
                record = {
                    "agent_id": lease._agent_id, "role": lease._role, "slot_role": lease._slot_view.role,
                    "priority": lease._priority, "idea_id": lease._idea_id, "slot_id": lease._slot_id,
                    "acquired_at": round(lease._acquired_at, 6), "released_at": lease.released_at,
                    "queue_wait_s": lease._queue_wait_s, "hold_s": lease.hold_s,
                    "cleanup_s": lease.cleanup_s, "recovery_s": lease.recovery_s,
                    "recovered": recovered, "error": combined_error, "status_after": cleanup.status_after,
                }
                with self._lock:
                    self._total_leases, self._recoveries = self._total_leases + 1, self._recoveries + int(recovered)
                    self._total_recovery_s += raw_recovery_s
                    self._queue_waits.append(lease._queue_wait_s)
                    self._hold_times.append(hold_s)
                try:
                    finalized = lease._trace_record = self._append_trace(record)
                except BaseException as trace_error:
                    lease._trace_error, lease._trace_record = trace_error, record
                    raise
            except BaseException as release_error:
                if lease._trace_error is None:
                    lease._trace_error = release_error
                if lease.status_after is None:
                    lease.status_after = "quarantined"
                raise
            finally:
                done_mono = finish_mono if finish_mono is not None else self._safe_clock(fallback=release_mono)
                self._finish_slot(
                    lease._slot_id, done_mono, bool(cleanup.healthy) if cleanup is not None else False,
                    cleanup.error if cleanup is not None else "lease release interrupted",
                    skip_close=undrained,
                )
            return dict(finalized)
    def _append_trace(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._trace_lock:
            if self._trace_path:
                os.makedirs(os.path.dirname(os.path.abspath(self._trace_path)), exist_ok=True)
                with open(self._trace_path, "a+", encoding="utf-8") as handle:
                    fcntl and fcntl.flock(handle, fcntl.LOCK_EX)
                    try:
                        last_sequence, last_hash = _read_trace_tail(handle)
                        chained, next_seq, digest = _chain_record(record, last_sequence, last_hash)
                        handle.seek(0, os.SEEK_END)
                        handle.write(json.dumps(chained, sort_keys=True, default=str) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    finally:
                        fcntl and fcntl.flock(handle, fcntl.LOCK_UN)
            else:
                chained, next_seq, digest = _chain_record(record, self._trace_sequence, self._trace_previous_hash)
            self._trace_sequence, self._trace_previous_hash = next_seq, digest
            return chained

    def run_isolated(self, agent_id: str, action_fn: Callable[[Any, SlotLease], Any], *,
                     role: str | None = None, idea_id: str | None = None,
                     timeout_s: float | None = None, priority: int = 0) -> dict[str, Any]:
        """Acquire a lease (highest `priority` first, FIFO tie-break), run `action_fn`, and release cleanly."""
        start_mono = self._clock()
        try:
            slot_lease = self.acquire(agent_id, role=role, idea_id=idea_id, timeout_s=timeout_s, priority=priority)
        except PoolError as error:
            return {"ok": False, "result": None, "slot_id": None,
                    "queue_wait_s": round(max(0.0, self._clock() - start_mono), 6),
                    "hold_s": 0.0, "cleanup_s": 0.0, "recovery_s": 0.0,
                    "recovered": False, "status_after": None, "error": _format_error(error)}
        action_result, caught_error = None, None
        try:
            action_result = action_fn(slot_lease._lab_proxy, slot_lease)
        except BaseException as error:
            caught_error = error
            if not isinstance(error, Exception):
                raise
        finally:
            trace_record = slot_lease.release(error=caught_error)
        return {"ok": caught_error is None, "result": action_result if caught_error is None else None,
                "slot_id": slot_lease._slot_id, **{key: trace_record[key] for key in _RUN_RESULT_KEYS}}

    def metrics(self) -> dict[str, Any]:
        """Compute exact aggregate queue, hold, recovery, quarantine, and slot utilization stats."""
        with self._trace_lock:
            trace_head = (self._trace_sequence, self._trace_previous_hash)
        with self._lock:
            now_mono = self._clock()
            elapsed_s, total_leases = max(0.0, now_mono - self._created_mono), self._total_leases
            per_slot_utilization: dict[str, float] = {}
            for slot_id, slot_state in self._slots.items():
                busy = slot_state.busy_s + (max(0.0, now_mono - slot_state.active_since)
                                            if slot_state.active_since is not None else 0.0)
                per_slot_utilization[slot_id] = round(0.0 if (elapsed_s <= 0.0 or busy <= 0.0) else min(1.0, busy / elapsed_s), 6)
            return {
                "total_leases": total_leases,
                "mean_queue_wait_s": round(sum(self._queue_waits) / total_leases, 6) if total_leases else 0.0,
                "p95_queue_wait_s": round(_percentile(self._queue_waits, 0.95), 6),
                "mean_hold_s": round(sum(self._hold_times) / total_leases, 6) if total_leases else 0.0,
                "total_recovery_s": round(self._total_recovery_s, 6), "recoveries": self._recoveries,
                "quarantined_slots": [slot_id for slot_id, state in self._slots.items() if state.quarantined],
                "per_slot_utilization": per_slot_utilization, "trace_head": trace_head,
            }

    def close(self, *, close_labs: bool | None = None) -> None:
        """Idempotently close the pool, fail queued waiters, and optionally close idle labs."""
        labs_to_close: list[Any] = []
        with self._lock:
            if close_labs is True or (close_labs is False and not self._closed):
                self._close_labs = close_labs
            self._closed = True
            while self._waiters:
                waiter = self._waiters.popleft()
                if not waiter.cancelled:
                    waiter.closed = True
                    waiter.event.set()
            if self._close_labs:
                labs_to_close = [lab for state in self._slots.values() if not state.in_use
                                 for lab in (self._mark_lab_to_close_locked(state),) if lab is not None]
        for lab in labs_to_close:
            _close_quietly(lab)

    def __enter__(self) -> SandboxPool: return self
    def __exit__(self, exc_type: Any, exc_value: BaseException | None, exc_traceback: Any) -> bool:
        self.close()
        return False
