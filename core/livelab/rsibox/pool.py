"""Thread-safe multi-agent shared-pool sandbox scheduler (`SandboxPool`).

Multiplexes N concurrent agents across C warm RSIBox `Lab` (or ready `Box`) slots with
strict priority queueing (FIFO tie-break), role routing, fenced leases, failover/quarantine, and JSONL telemetry.
"""
from __future__ import annotations

from collections import deque, namedtuple
from fractions import Fraction
import contextlib, dataclasses, hashlib, json, math, operator, os, threading, time, types
from typing import Any, Callable, Iterator, Sequence
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None
from .box import Box

__all__ = [
    "AgentQuarantinedError", "PoolClosedError", "PoolError", "PoolSlot",
    "PoolTimeoutError", "SandboxPool", "SlotLease", "verify_trace",
]
_GENESIS_HASH = "genesis"
_CleanupOutcome = namedtuple("_CleanupOutcome", "healthy cleanup_s recovery_s recovered status_after error")
_RUN_RESULT_KEYS = ("queue_wait_s", "hold_s", "cleanup_s", "recovery_s", "recovered", "borrowed", "status_after", "error")

class PoolError(RuntimeError): """Base error raised by `SandboxPool`."""
class PoolTimeoutError(TimeoutError, PoolError): """Raised when waiting for a healthy matching slot times out."""
class PoolClosedError(PoolError): """Raised when attempting to acquire a slot from a closed `SandboxPool`."""
class AgentQuarantinedError(PoolError):
    """Raised when an agent's per-agent fault circuit breaker is open."""

    def __init__(self, agent_id: str, *, fault_count: int, retry_after_s: float | None = None) -> None:
        suffix = (
            f"; retry after {retry_after_s:.3f}s"
            if retry_after_s is not None
            else "; call reset_agent_circuit() to re-enable"
        )
        super().__init__(f"agent {agent_id!r} quarantined after {fault_count} slot fault(s){suffix}")
        self.agent_id = agent_id
        self.fault_count = fault_count
        self.retry_after_s = retry_after_s

def _format_error(error: str | BaseException | None) -> str | None:
    if error is None:
        return None
    return f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else (str(error).strip() or None)
def _close_quietly(lab: Any) -> None:
    if lab is not None:
        with contextlib.suppress(Exception):
            lab.close()
def _validate_fallback_roles(role: str | None, fallback_roles: Sequence[str] | None) -> tuple[str, ...]:
    if fallback_roles is None:
        return ()
    if isinstance(fallback_roles, (str, bytes, bytearray, memoryview)) or not isinstance(fallback_roles, Sequence):
        raise TypeError(f"fallback_roles must be a sequence of role strings or None, not {type(fallback_roles).__name__}")
    if not fallback_roles:
        return ()
    if role is None:
        raise ValueError("fallback_roles requires an explicit primary role (role=None already matches all slots)")
    if not isinstance(role, str):
        raise TypeError(f"role must be a non-empty str when fallback_roles is set, not {type(role).__name__}")
    if not role.strip():
        raise ValueError("role must be a non-empty string when fallback_roles is set")
    seen = {role}
    normalized: list[str] = []
    for entry in fallback_roles:
        if not isinstance(entry, str):
            raise TypeError(f"fallback_roles entries must be str, not {type(entry).__name__}")
        if not entry.strip():
            raise ValueError("fallback_roles entries must be non-empty strings")
        if entry not in seen:
            seen.add(entry)
            normalized.append(entry)
    return tuple(normalized)
def _validate_priority(priority: int) -> int:
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise TypeError(f"priority must be an int, not {type(priority).__name__}")
    return priority
def _validate_positive_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number or None, not {type(value).__name__}")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return numeric
def _validate_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive int or None, not {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return value
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
                 "in_use", "owner_agent_id", "active_since", "dirty", "busy_s", "lab_closed")
    def __init__(self, slot: PoolSlot) -> None:
        self.slot, self.slot_id, self.role, self.lab = slot, slot.slot_id, slot.role, slot.lab
        self.view = _SlotView(slot.slot_id, slot.role, types.MappingProxyType(dict(slot.metadata)))
        self.quarantined, self.quarantine_reason = False, None
        self.in_use, self.owner_agent_id, self.active_since, self.dirty, self.busy_s, self.lab_closed = (
            False, None, None, True, 0.0, False,
        )

class _Waiter:
    __slots__ = ("agent_id", "role", "fallback_roles", "priority", "enqueued_mono", "event",
                 "assigned_state", "assigned_dirty", "cancelled", "failure", "quarantined_error", "closed")
    def __init__(self, role: str | None, priority: int = 0, enqueued_mono: float = 0.0,
                 fallback_roles: tuple[str, ...] = (), agent_id: str | None = None) -> None:
        self.agent_id, self.role, self.fallback_roles, self.priority, self.enqueued_mono = (
            agent_id, role, fallback_roles, priority, enqueued_mono,
        )
        self.event, self.assigned_state, self.assigned_dirty = threading.Event(), None, True
        self.cancelled, self.failure, self.quarantined_error, self.closed = False, None, None, False

class SlotLease:
    """Exclusive lease over one `PoolSlot`; supports `with` and idempotent `release()`."""
    slot = property(operator.attrgetter("_slot_view"))
    slot_id = property(operator.attrgetter("_slot_id"))
    agent_id = property(operator.attrgetter("_agent_id"))
    role = property(operator.attrgetter("_role"))
    fallback_roles = property(operator.attrgetter("_fallback_roles"))
    borrowed = property(operator.attrgetter("_borrowed"))
    priority = property(operator.attrgetter("_priority"))
    idea_id = property(operator.attrgetter("_idea_id"))
    acquired_at = property(operator.attrgetter("_acquired_at"))
    queue_wait_s = property(operator.attrgetter("_queue_wait_s"))
    lab = property(operator.attrgetter("_lab_proxy"))
    released = property(operator.attrgetter("_released"))

    def __init__(self, pool: SandboxPool, slot_state: _SlotState, agent_id: str, role: str | None,
                 idea_id: str | None, acquired_at: float, acquired_mono: float,
                 queue_wait_s: float, preparation: _CleanupOutcome, *, priority: int = 0,
                 fallback_roles: tuple[str, ...] = ()) -> None:
        self._pool, self._lab, self._slot_view, self._slot_id = (
            pool, slot_state.lab, slot_state.view, slot_state.slot_id,
        )
        self._agent_id, self._role, self._fallback_roles = agent_id, role, fallback_roles
        self._borrowed = role is not None and slot_state.role != role
        self._priority, self._idea_id = priority, idea_id
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
                 close_labs: bool = False, drain_timeout_s: float | None = 5.0,
                 priority_aging_s: float | None = None,
                 max_agent_faults: int | None = None,
                 agent_fault_window_s: float | None = None,
                 agent_cooldown_s: float | None = None,
                 max_slots_per_agent: int | None = None) -> None:
        if not slots:
            raise ValueError("SandboxPool requires at least one PoolSlot")
        self._drain_timeout_s = _validate_timeout(drain_timeout_s, "drain_timeout_s")
        self._priority_aging_s = _validate_positive_finite(priority_aging_s, "priority_aging_s")
        self._max_agent_faults = _validate_positive_int(max_agent_faults, "max_agent_faults")
        self._agent_fault_window_s = _validate_positive_finite(agent_fault_window_s, "agent_fault_window_s")
        self._agent_cooldown_s = _validate_positive_finite(agent_cooldown_s, "agent_cooldown_s")
        self._max_slots_per_agent = _validate_positive_int(max_slots_per_agent, "max_slots_per_agent")
        if self._max_agent_faults is None and (self._agent_fault_window_s is not None or self._agent_cooldown_s is not None):
            raise ValueError("agent_fault_window_s and agent_cooldown_s require max_agent_faults")
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
        self._closed, self._total_leases, self._borrowed_leases, self._recoveries, self._healed_slots, self._total_recovery_s = (
            False, 0, 0, 0, 0, 0.0,
        )
        self._queue_waits, self._hold_times, self._trace_sequence, self._trace_previous_hash = [], [], 0, _GENESIS_HASH
        self._agent_active_slots: dict[str, int] = {}
        self._inflight_heal_targets: set[int] = set()
        self._agent_faults: dict[str, deque[float]] = {}
        self._agent_quarantined_until: dict[str, float | None] = {}
        self._agent_last_reset_mono: dict[str, float] = {}
        self._circuit_breaker_rejections = 0
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
    def _has_healthy_slot(self, role: str | None, fallback_roles: tuple[str, ...] = ()) -> bool:
        return any(
            not slot_state.quarantined
            and (role is None or slot_state.role == role or slot_state.role in fallback_roles)
            for slot_state in self._slots.values()
        )
    def _mark_lab_to_close_locked(self, slot_state: _SlotState) -> Any | None:
        if self._closed and self._close_labs and not slot_state.lab_closed:
            slot_state.lab_closed = True
            return slot_state.lab
        return None

    def _effective_priority(self, waiter: _Waiter, now_mono: float) -> int | float | Fraction:
        if self._priority_aging_s is None:
            return waiter.priority
        elapsed = max(0.0, now_mono - waiter.enqueued_mono)
        if not math.isfinite(elapsed):
            return float("inf")
        try:
            bonus = elapsed / self._priority_aging_s
            if math.isfinite(bonus) and abs(waiter.priority) <= (1 << 24) and bonus <= (1 << 24):
                return waiter.priority + bonus
        except OverflowError:
            pass
        return Fraction(waiter.priority) + Fraction(elapsed) / Fraction(self._priority_aging_s)

    def _agent_under_cap_locked(self, agent_id: str | None) -> bool:
        if self._max_slots_per_agent is None or agent_id is None:
            return True
        return self._agent_active_slots.get(agent_id, 0) < self._max_slots_per_agent

    def _assign_slot_locked(self, slot_state: _SlotState, agent_id: str | None, now_mono: float) -> None:
        slot_state.in_use, slot_state.active_since, slot_state.owner_agent_id = True, now_mono, agent_id
        if agent_id is not None:
            self._agent_active_slots[agent_id] = self._agent_active_slots.get(agent_id, 0) + 1

    def _release_slot_owner_locked(self, slot_state: _SlotState) -> str | None:
        prev_owner = slot_state.owner_agent_id
        slot_state.owner_agent_id = None
        if prev_owner is not None:
            remaining = self._agent_active_slots.get(prev_owner, 0) - 1
            if remaining > 0:
                self._agent_active_slots[prev_owner] = remaining
            else:
                self._agent_active_slots.pop(prev_owner, None)
        return prev_owner

    def _best_waiter_for_slot_locked(
        self, slot_state: _SlotState, now_mono: float
    ) -> tuple[int, int, int | float | Fraction] | None:
        """Select highest-priority eligible waiter for `slot_state` -> `(tier_rank, waiter_index, eff_priority)`."""
        best_index: int | None = None
        best_priority: int | float | Fraction = 0
        # Tier 1: uncapped primary-role (or generalist) waiters take precedence on their home slot.
        for index, waiter in enumerate(self._waiters):
            if (
                waiter.cancelled
                or (waiter.role is not None and waiter.role != slot_state.role)
                or not self._agent_under_cap_locked(waiter.agent_id)
            ):
                continue
            effective_priority = self._effective_priority(waiter, now_mono)
            if best_index is None or effective_priority > best_priority:
                best_index, best_priority = index, effective_priority
        if best_index is not None:
            return (1, best_index, best_priority)
        # Tier 2: elastic role borrowing when no eligible primary waiter can use this slot.
        if slot_state.role is not None:
            for index, waiter in enumerate(self._waiters):
                if (
                    waiter.cancelled
                    or slot_state.role not in waiter.fallback_roles
                    or not self._agent_under_cap_locked(waiter.agent_id)
                ):
                    continue
                effective_priority = self._effective_priority(waiter, now_mono)
                if best_index is None or effective_priority > best_priority:
                    best_index, best_priority = index, effective_priority
            if best_index is not None:
                return (0, best_index, best_priority)
        return None

    def _handoff_waiter_locked(self, slot_state: _SlotState, waiter_index: int, now_mono: float) -> None:
        waiter = self._waiters[waiter_index]
        del self._waiters[waiter_index]
        waiter.assigned_state, waiter.assigned_dirty = slot_state, slot_state.dirty
        self._assign_slot_locked(slot_state, waiter.agent_id, now_mono)
        waiter.event.set()

    def _dispatch_idle_slots_locked(self, now_mono: float, preferred_slot: _SlotState | None = None) -> None:
        if self._closed or not self._waiters:
            return
        idle_slots: list[_SlotState] = []
        if preferred_slot is not None and not preferred_slot.in_use and not preferred_slot.quarantined:
            idle_slots.append(preferred_slot)
        for slot_state in self._slots.values():
            if slot_state is not preferred_slot and not slot_state.in_use and not slot_state.quarantined:
                idle_slots.append(slot_state)
        if not idle_slots:
            return
        # Order non-cancelled waiters once by strict effective priority (with aging) and FIFO tie-break.
        ranked = [
            (self._effective_priority(waiter, now_mono), -index, waiter)
            for index, waiter in enumerate(self._waiters)
            if not waiter.cancelled
        ]
        if not ranked:
            return
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        assigned_ids: set[int] = set()
        # Pass 1 (Tier 1): primary-role (or generalist) matches take unconditional precedence across all idle slots.
        for _, _, waiter in ranked:
            if not idle_slots:
                break
            if not self._agent_under_cap_locked(waiter.agent_id):
                continue
            matched = next(
                (slot for slot in idle_slots if not slot.in_use and (waiter.role is None or slot.role == waiter.role)),
                None,
            )
            if matched is None:
                continue
            assigned_ids.add(id(waiter))
            waiter.assigned_state, waiter.assigned_dirty = matched, matched.dirty
            self._assign_slot_locked(matched, waiter.agent_id, now_mono)
            waiter.event.set()
            idle_slots.remove(matched)
        # Pass 2 (Tier 2): elastic role borrowing in caller-specified fallback_roles order for remaining idle slots.
        for _, _, waiter in ranked:
            if not idle_slots:
                break
            if id(waiter) in assigned_ids or not waiter.fallback_roles or not self._agent_under_cap_locked(waiter.agent_id):
                continue
            matched = None
            for fb_role in waiter.fallback_roles:
                matched = next((slot for slot in idle_slots if not slot.in_use and slot.role == fb_role), None)
                if matched is not None:
                    break
            if matched is None:
                continue
            assigned_ids.add(id(waiter))
            waiter.assigned_state, waiter.assigned_dirty = matched, matched.dirty
            self._assign_slot_locked(matched, waiter.agent_id, now_mono)
            waiter.event.set()
            idle_slots.remove(matched)
        if assigned_ids:
            self._waiters = deque(waiter for waiter in self._waiters if id(waiter) not in assigned_ids)

    def _complete_slot_locked(self, slot_id: str, finish_mono: float, healthy: bool, reason: str | None,
                              *, preserve_dirty: bool = False, skip_close: bool = False) -> Any | None:
        slot_state = self._slots[slot_id]
        if slot_state.active_since is not None:
            slot_state.busy_s += max(0.0, finish_mono - slot_state.active_since)
        prev_owner = self._release_slot_owner_locked(slot_state)
        slot_state.in_use, slot_state.active_since = False, None
        slot_state.quarantined, slot_state.dirty = not healthy, (True if not healthy else preserve_dirty)
        slot_state.quarantine_reason = None if healthy else (reason or "recovery failed")
        slot_state.slot.quarantined, slot_state.slot.quarantine_reason = slot_state.quarantined, slot_state.quarantine_reason
        if not healthy:
            for waiter in self._waiters:
                if not waiter.cancelled and not self._has_healthy_slot(waiter.role, waiter.fallback_roles):
                    waiter.failure = f"no healthy slot available for role={waiter.role!r}"
                    waiter.event.set()
            self._waiters = deque(waiter for waiter in self._waiters if not waiter.cancelled and waiter.failure is None)
        if not self._closed and self._waiters:
            if self._max_slots_per_agent is not None and prev_owner is not None:
                # Select globally across idle healthy slots so a freed permit goes to the highest-priority
                # eligible waiter (preventing cross-role priority inversion or lost cross-role wakeups).
                self._dispatch_idle_slots_locked(finish_mono, preferred_slot=slot_state if healthy else None)
            elif healthy:
                candidate = self._best_waiter_for_slot_locked(slot_state, finish_mono)
                if candidate is not None:
                    self._handoff_waiter_locked(slot_state, candidate[1], finish_mono)
        if slot_state.in_use:
            return None
        return None if skip_close else self._mark_lab_to_close_locked(slot_state)
    def _prune_agent_state_locked(self, agent_id: str, now_mono: float) -> None:
        if agent_id in self._agent_quarantined_until:
            until = self._agent_quarantined_until[agent_id]
            if until is not None and now_mono >= until:
                del self._agent_quarantined_until[agent_id]
                self._agent_faults.pop(agent_id, None)
                self._agent_last_reset_mono[agent_id] = max(
                    self._agent_last_reset_mono.get(agent_id, -math.inf), until
                )
            else:
                # Preserve recorded fault history while actively quarantined so fault_count
                # and metrics()["agent_fault_counts"] never shrink below max_agent_faults.
                return
        if self._agent_fault_window_s is not None and agent_id in self._agent_faults:
            kept = deque(
                ts for ts in self._agent_faults[agent_id]
                if max(0.0, now_mono - ts) <= self._agent_fault_window_s
            )
            if kept:
                self._agent_faults[agent_id] = kept
            else:
                del self._agent_faults[agent_id]

    def _build_quarantined_error_locked(self, agent_id: str, now_mono: float) -> AgentQuarantinedError | None:
        if self._max_agent_faults is None:
            return None
        self._prune_agent_state_locked(agent_id, now_mono)
        if agent_id not in self._agent_quarantined_until:
            return None
        until = self._agent_quarantined_until[agent_id]
        retry_after = None if until is None else round(max(0.0, until - now_mono), 6)
        fault_count = max(len(self._agent_faults.get(agent_id, ())), self._max_agent_faults)
        return AgentQuarantinedError(agent_id, fault_count=fault_count, retry_after_s=retry_after)

    def _record_agent_fault_locked(self, agent_id: str, fault_mono: float) -> None:
        if self._max_agent_faults is None:
            return
        if fault_mono < self._agent_last_reset_mono.get(agent_id, -math.inf):
            return
        self._prune_agent_state_locked(agent_id, fault_mono)
        faults = self._agent_faults.setdefault(agent_id, deque())
        faults.append(fault_mono)
        if len(faults) < self._max_agent_faults:
            return
        if self._agent_cooldown_s is None:
            until: float | None = None
        else:
            raw_until = fault_mono + self._agent_cooldown_s
            if not math.isfinite(raw_until):
                until = None
            elif raw_until <= fault_mono:
                next_val = math.nextafter(fault_mono, math.inf)
                until = next_val if math.isfinite(next_val) else None
            else:
                until = raw_until
        self._agent_quarantined_until[agent_id] = until
        fault_count = max(len(faults), self._max_agent_faults)
        retry_after = None if until is None else round(max(0.0, until - fault_mono), 6)
        # Immediately reject any already-queued waiters from this tripped agent before slot handoff.
        remaining_waiters: deque[_Waiter] = deque()
        for waiter in self._waiters:
            if not waiter.cancelled and waiter.agent_id == agent_id:
                self._circuit_breaker_rejections += 1
                waiter.quarantined_error = AgentQuarantinedError(
                    agent_id, fault_count=fault_count, retry_after_s=retry_after
                )
                waiter.event.set()
            else:
                remaining_waiters.append(waiter)
        self._waiters = remaining_waiters

    def reset_agent_circuit(self, agent_id: str) -> bool:
        """Clear recorded slot faults and re-enable a quarantined agent. Returns True if it was quarantined."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        with self._lock:
            now_mono = self._safe_clock()
            self._prune_agent_state_locked(agent_id, now_mono)
            was_quarantined = self._agent_quarantined_until.pop(agent_id, False) is not False
            had_faults = self._agent_faults.pop(agent_id, None) is not None
            if was_quarantined or had_faults:
                self._agent_last_reset_mono[agent_id] = max(
                    self._agent_last_reset_mono.get(agent_id, -math.inf), now_mono
                )
            return was_quarantined

    @staticmethod
    def _slot_raw_target(slot: PoolSlot) -> Any:
        return slot.box.lab if (slot.box is not None and slot.box.lab is not None) else (slot.box or slot.lab)

    def heal_slot(self, slot_id: str, *, lab: Any = None, close_replaced: bool = False) -> dict[str, Any]:
        """Rehabilitate a quarantined slot online (in-place recovery or replacement `lab`/`Box`).

        Runs cleanup/respawn outside the pool lock so healthy sibling slots continue serving
        uninterrupted, and immediately dispatches the healed slot to any waiting agents.
        """
        if not isinstance(slot_id, str) or not slot_id.strip():
            raise ValueError("slot_id must be a non-empty string")
        if not isinstance(close_replaced, bool):
            raise TypeError(f"close_replaced must be a bool, not {type(close_replaced).__name__}")
        new_box: Box | None = None
        candidate_lab: Any = None
        new_raw_id: int | None = None
        if lab is not None:
            if isinstance(lab, _FencedLabProxy):
                raise TypeError("cannot pass a fenced SlotLease.lab proxy to heal_slot")
            if isinstance(lab, _BoxLabAdapter):
                lab = lab._box
            if isinstance(lab, Box):
                if lab.state != "ready":
                    raise ValueError(f"Box for slot {slot_id!r} must be ready (state={lab.state!r})")
                new_box = lab
                candidate_lab = _BoxLabAdapter(lab)
                new_raw_id = id(lab.lab if lab.lab is not None else lab)
            else:
                candidate_lab = lab
                new_raw_id = id(lab)

        reserved_target_id: int | None = None
        with self._lock:
            if self._closed:
                raise PoolClosedError("SandboxPool is closed")
            slot_state = self._slots.get(slot_id)
            if slot_state is None:
                raise ValueError(f"unknown slot_id: {slot_id!r}")
            if slot_state.in_use:
                raise PoolError(f"slot {slot_id!r} is currently in use")
            if not slot_state.quarantined:
                if lab is not None:
                    raise PoolError(f"slot {slot_id!r} is not quarantined; cannot replace healthy slot lab")
                return {
                    "ok": True,
                    "slot_id": slot_id,
                    "healed": False,
                    "status": "healthy",
                    "recovery_s": 0.0,
                    "error": None,
                }
            if new_raw_id is not None:
                if new_raw_id in self._inflight_heal_targets:
                    raise ValueError(f"duplicate underlying lab/box for slot_id: {slot_id!r}")
                for other_state in self._slots.values():
                    if id(self._slot_raw_target(other_state.slot)) == new_raw_id:
                        raise ValueError(f"duplicate underlying lab/box for slot_id: {slot_id!r}")
            claim_mono = self._clock()
            if new_raw_id is not None:
                self._inflight_heal_targets.add(new_raw_id)
                reserved_target_id = new_raw_id
            slot_state.in_use = True
            slot_state.active_since = None
            slot_state.owner_agent_id = None
            target_lab = candidate_lab if candidate_lab is not None else slot_state.lab

        finalized = False
        old_lab_to_close: Any | None = None
        lab_to_close: Any | None = None
        try:
            outcome = self._clean_and_recover_slot(target_lab, force_reset_state=True)
            finish_mono = self._safe_clock(fallback=claim_mono)
            with self._lock:
                if reserved_target_id is not None:
                    self._inflight_heal_targets.discard(reserved_target_id)
                    reserved_target_id = None
                self._total_recovery_s += outcome.recovery_s
                if self._closed:
                    lab_to_close = self._complete_slot_locked(
                        slot_id, finish_mono, False, "SandboxPool closed during heal_slot",
                    )
                    finalized = True
                    result = {
                        "ok": False,
                        "slot_id": slot_id,
                        "healed": False,
                        "status": "closed",
                        "recovery_s": round(outcome.recovery_s, 6),
                        "error": "SandboxPool closed during heal_slot",
                    }
                elif outcome.healthy:
                    if candidate_lab is not None:
                        prev_lab = slot_state.lab
                        prev_raw_id = id(self._slot_raw_target(slot_state.slot))
                        slot_state.lab = candidate_lab
                        slot_state.slot.lab = candidate_lab
                        slot_state.slot.box = new_box
                        if close_replaced and prev_lab is not candidate_lab and prev_raw_id != new_raw_id:
                            old_lab_to_close = prev_lab
                    self._healed_slots += 1
                    self._recoveries += int(outcome.recovered)
                    lab_to_close = self._complete_slot_locked(
                        slot_id, finish_mono, True, None, preserve_dirty=False,
                    )
                    finalized = True
                    result = {
                        "ok": True,
                        "slot_id": slot_id,
                        "healed": True,
                        "status": "base",
                        "recovery_s": round(outcome.recovery_s, 6),
                        "error": None,
                    }
                else:
                    reason = outcome.error or "slot heal failed"
                    lab_to_close = self._complete_slot_locked(slot_id, finish_mono, False, reason)
                    finalized = True
                    result = {
                        "ok": False,
                        "slot_id": slot_id,
                        "healed": False,
                        "status": "quarantined",
                        "recovery_s": round(outcome.recovery_s, 6),
                        "error": reason,
                    }
        finally:
            if reserved_target_id is not None:
                with self._lock:
                    self._inflight_heal_targets.discard(reserved_target_id)
            if not finalized:
                self._finish_slot(
                    slot_id, self._safe_clock(fallback=claim_mono), False, "heal_slot interrupted",
                )
            _close_quietly(old_lab_to_close)
            _close_quietly(lab_to_close)
        return result

    def _finish_slot(self, slot_id: str, finish_mono: float, healthy: bool, reason: str | None,
                     *, preserve_dirty: bool = False, extra_recovery_s: float = 0.0,
                     skip_close: bool = False, faulted_agent_id: str | None = None) -> None:
        with self._lock:
            if faulted_agent_id is not None:
                self._record_agent_fault_locked(faulted_agent_id, finish_mono)
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

    def _claim_slot(self, agent_id: str, role: str | None, deadline: float | None, *, priority: int = 0,
                    fallback_roles: tuple[str, ...] = ()) -> tuple[_SlotState, bool]:
        with self._lock:
            if self._closed:
                raise PoolClosedError("SandboxPool is closed")
            now_mono = self._clock() if (
                self._max_agent_faults is not None or (self._priority_aging_s is not None and self._waiters)
            ) else None
            if self._max_agent_faults is not None:
                assert now_mono is not None
                quarantined_err = self._build_quarantined_error_locked(agent_id, now_mono)
                if quarantined_err is not None:
                    self._circuit_breaker_rejections += 1
                    raise quarantined_err
            if not self._has_healthy_slot(role, fallback_roles):
                raise PoolTimeoutError(f"no healthy slot available for role={role!r}")
            # Prefer primary-role slots first, then fallback roles in caller-specified order (only if under per-agent cap).
            if self._agent_under_cap_locked(agent_id):
                candidate_roles: tuple[tuple[str | None, bool], ...] = ((role, False),) + tuple(
                    (fb_role, True) for fb_role in fallback_roles
                )
                for target_role, is_borrow in candidate_roles:
                    for slot_state in self._slots.values():
                        if slot_state.in_use or slot_state.quarantined or (target_role is not None and slot_state.role != target_role):
                            continue
                        # Eligible (uncapped) primary waiters block borrowers unconditionally and block equal/lower primary arrivals.
                        if is_borrow:
                            if any(
                                not waiter.cancelled and self._agent_under_cap_locked(waiter.agent_id) and (
                                    waiter.role is None or waiter.role == slot_state.role or (
                                        slot_state.role is not None
                                        and slot_state.role in waiter.fallback_roles
                                        and self._effective_priority(waiter, 0.0 if now_mono is None else now_mono) >= priority
                                    )
                                )
                                for waiter in self._waiters
                            ):
                                continue
                        elif any(
                            not waiter.cancelled and self._agent_under_cap_locked(waiter.agent_id)
                            and (waiter.role is None or waiter.role == slot_state.role)
                            and self._effective_priority(waiter, 0.0 if now_mono is None else now_mono) >= priority
                            for waiter in self._waiters
                        ):
                            continue
                        self._assign_slot_locked(slot_state, agent_id, now_mono if now_mono is not None else self._clock())
                        return slot_state, slot_state.dirty
            if deadline is not None or self._priority_aging_s is not None:
                now_mono = now_mono if now_mono is not None else self._clock()
                if deadline is not None and now_mono >= deadline:
                    raise PoolTimeoutError(f"timed out waiting for slot (role={role!r})")
            waiter = _Waiter(role, priority)
            waiter.agent_id = agent_id
            if now_mono is not None:
                waiter.enqueued_mono = now_mono
            if fallback_roles:
                waiter.fallback_roles = fallback_roles
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
            quarantined_error = waiter.quarantined_error
            lab_to_close = self._cancel_waiter_locked(waiter)
        _close_quietly(lab_to_close)
        if closed_error:
            raise PoolClosedError("SandboxPool closed while waiting for slot")
        if quarantined_error is not None:
            raise quarantined_error
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
                priority: int = 0, fallback_roles: Sequence[str] | None = None) -> SlotLease:
        """Acquire an exclusive, clean-slate `SlotLease` (primary role first, optional elastic borrowing)."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        normalized_priority = _validate_priority(priority)
        normalized_fallbacks = _validate_fallback_roles(role, fallback_roles)
        normalized_timeout = _validate_timeout(timeout_s)
        queue_start = self._clock()
        deadline = None if normalized_timeout is None else queue_start + normalized_timeout
        while True:
            slot_state, slot_dirty = self._claim_slot(
                agent_id, role, deadline, priority=normalized_priority, fallback_roles=normalized_fallbacks,
            )
            slot_returned = False
            try:
                claim_mono = self._clock()
                preparation = (_CleanupOutcome(True, 0.0, 0.0, False, "base", None)
                               if (not slot_dirty and _is_clean_base(_safe_lab_call(slot_state.lab, "status")))
                               else self._clean_and_recover_slot(slot_state.lab, force_reset_state=True))
                if preparation.healthy:
                    ready_mono, ready_wall = self._clock(), self._wall_clock()
                    quarantined_err: AgentQuarantinedError | None = None
                    with self._lock:
                        if not self._closed:
                            quarantined_err = (
                                self._build_quarantined_error_locked(agent_id, ready_mono)
                                if self._max_agent_faults is not None else None
                            )
                            if quarantined_err is None:
                                slot_state.dirty = False
                                return SlotLease(self, slot_state, agent_id, role, idea_id, ready_wall, ready_mono,
                                                 round(max(0.0, claim_mono - queue_start), 6), preparation,
                                                 priority=normalized_priority, fallback_roles=normalized_fallbacks)
                            self._circuit_breaker_rejections += 1
                        lab_to_close = self._complete_slot_locked(slot_state.slot_id, ready_mono, True, None)
                        slot_returned = True
                    _close_quietly(lab_to_close)
                    if quarantined_err is not None:
                        raise quarantined_err
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
              priority: int = 0, fallback_roles: Sequence[str] | None = None) -> Iterator[SlotLease]:
        """Context manager acquiring and releasing a `SlotLease` (with optional `fallback_roles` borrowing)."""
        slot_lease = self.acquire(
            agent_id, role=role, idea_id=idea_id, timeout_s=timeout_s,
            priority=priority, fallback_roles=fallback_roles,
        )
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
                    "borrowed": lease._borrowed, "priority": lease._priority, "idea_id": lease._idea_id,
                    "slot_id": lease._slot_id, "acquired_at": round(lease._acquired_at, 6),
                    "released_at": lease.released_at, "queue_wait_s": lease._queue_wait_s, "hold_s": lease.hold_s,
                    "cleanup_s": lease.cleanup_s, "recovery_s": lease.recovery_s,
                    "recovered": recovered, "error": combined_error, "status_after": cleanup.status_after,
                }
                with self._lock:
                    self._total_leases = self._total_leases + 1
                    self._borrowed_leases = self._borrowed_leases + int(lease._borrowed)
                    self._recoveries = self._recoveries + int(recovered)
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
                caused_slot_fault = (cleanup is None) or cleanup.recovered or (not cleanup.healthy)
                self._finish_slot(
                    lease._slot_id, done_mono, bool(cleanup.healthy) if cleanup is not None else False,
                    cleanup.error if cleanup is not None else "lease release interrupted",
                    skip_close=undrained,
                    faulted_agent_id=lease._agent_id if caused_slot_fault else None,
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
                     timeout_s: float | None = None, priority: int = 0,
                     fallback_roles: Sequence[str] | None = None) -> dict[str, Any]:
        """Acquire a lease (with optional `fallback_roles` borrowing), run `action_fn`, and release cleanly."""
        start_mono = self._clock()
        try:
            slot_lease = self.acquire(
                agent_id, role=role, idea_id=idea_id, timeout_s=timeout_s,
                priority=priority, fallback_roles=fallback_roles,
            )
        except PoolError as error:
            return {"ok": False, "result": None, "slot_id": None,
                    "queue_wait_s": round(max(0.0, self._clock() - start_mono), 6),
                    "hold_s": 0.0, "cleanup_s": 0.0, "recovery_s": 0.0,
                    "recovered": False, "borrowed": False, "status_after": None, "error": _format_error(error)}
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
            for aid in list(set(self._agent_faults) | set(self._agent_quarantined_until)):
                self._prune_agent_state_locked(aid, now_mono)
            return {
                "total_leases": total_leases,
                "borrowed_leases": self._borrowed_leases,
                "mean_queue_wait_s": round(sum(self._queue_waits) / total_leases, 6) if total_leases else 0.0,
                "p95_queue_wait_s": round(_percentile(self._queue_waits, 0.95), 6),
                "mean_hold_s": round(sum(self._hold_times) / total_leases, 6) if total_leases else 0.0,
                "total_recovery_s": round(self._total_recovery_s, 6), "recoveries": self._recoveries,
                "healed_slots": self._healed_slots,
                "quarantined_slots": [slot_id for slot_id, state in self._slots.items() if state.quarantined],
                "quarantined_agents": sorted(self._agent_quarantined_until),
                "circuit_breaker_rejections": self._circuit_breaker_rejections,
                "agent_fault_counts": {aid: len(ts) for aid, ts in sorted(self._agent_faults.items()) if ts},
                "max_slots_per_agent": self._max_slots_per_agent,
                "agent_active_slots": {aid: cnt for aid, cnt in sorted(self._agent_active_slots.items()) if cnt},
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
