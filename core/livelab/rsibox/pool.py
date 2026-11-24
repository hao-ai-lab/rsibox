"""Thread-safe multi-agent shared-pool sandbox scheduler (`SandboxPool`).

Multiplexes N concurrent agents across C warm RSIBox `Lab` (or `Box`) slots with strict
FIFO queueing, role-aware routing (`dev` / `toolsmith` / `eval` / any), clean-slate
isolation on release (and defensively on acquire if dirty), automatic `reset("respawn")`
failover with fail-closed quarantine, hash-chained JSONL telemetry, and exact metrics.
"""
from __future__ import annotations

from collections import deque, namedtuple
import contextlib
import hashlib
import json
import math
import os
import threading
import time
from typing import Any, Callable, Iterator, Sequence
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None
from .box import Box

__all__ = ["PoolClosedError", "PoolError", "PoolSlot", "PoolTimeoutError", "SandboxPool", "SlotLease", "verify_trace"]
_GENESIS_HASH = "genesis"
_CleanupOutcome = namedtuple("_CleanupOutcome", "healthy cleanup_s recovery_s recovered status_after error")

class PoolError(RuntimeError):
    """Base error raised by `SandboxPool`."""
class PoolTimeoutError(TimeoutError, PoolError):
    """Raised when waiting for a healthy matching slot times out or none exists."""
class PoolClosedError(PoolError):
    """Raised when attempting to acquire a slot from a closed `SandboxPool`."""

def _format_error(error: str | BaseException | None) -> str | None:
    if error is None:
        return None
    return f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else (str(error).strip() or None)

def _close_quietly(lab: Any) -> None:
    if lab is not None:
        with contextlib.suppress(Exception):
            lab.close()

def _append_jsonl_line(trace_path: str, line: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)
    with open(trace_path, "a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)

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
    return bool(status_reply.get("ok")) and status_reply.get("state") == "base" and not status_reply.get("txns") and not status_reply.get("poison")

def _percentile(values: Sequence[float], fraction: float) -> float:
    return 0.0 if not values else float(sorted(values)[min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))])

def _hash_trace_record(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()

def verify_trace(trace_path: str) -> dict[str, Any]:
    """Verify hash-chain integrity and sequential ordering of a `SandboxPool` JSONL trace."""
    records: list[dict[str, Any]] = []
    expected_previous = _GENESIS_HASH
    try:
        with open(trace_path, encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                expected_hash = _hash_trace_record(record)
                for failed, reason in ((record.get("seq") != len(records) + 1, "sequence mismatch"),
                                       (record.get("prev_sha256") != expected_previous, "prev_sha256 mismatch"),
                                       (record.get("record_sha256") != expected_hash, "record digest mismatch")):
                    if failed:
                        return {"ok": False, "count": len(records), "line": line_number, "error": reason}
                expected_previous = expected_hash
                records.append(record)
    except (OSError, ValueError, TypeError) as error:
        return {"ok": False, "count": len(records), "error": f"{type(error).__name__}: {error}"}
    return {"ok": True, "count": len(records), "records": records}


class _BoxLabAdapter:
    """Thin 5-verb `Lab` proxy over a `Box`, preserving its call gate and event log."""

    def __init__(self, box: Box) -> None: self._box = box
    def install(self, patch: dict[str, Any]) -> dict[str, Any]: return self._box.call("install", {"patch": patch})[1]
    def check(self, workload: dict[str, Any] | None = None) -> dict[str, Any]: return self._box.call("check", {"workload": workload} if workload is not None else {})[1]
    def reset(self, kind: str = "state") -> dict[str, Any]: return self._box.call("reset", {"kind": kind})[1]
    def uninstall(self, id: str | None = None) -> dict[str, Any]: return self._box.call("uninstall", {"id": id} if id else {})[1]
    def status(self) -> dict[str, Any]: return self._box.call("status")[1]
    def source(self, module_name: str) -> dict[str, Any]: return self._box.source(module_name)[1]
    def close(self) -> dict[str, Any]: return self._box.close()
    def __getattr__(self, attribute_name: str) -> Any:
        if self._box.lab is None:
            raise AttributeError(f"box is not ready; cannot access {attribute_name!r}")
        return getattr(self._box.lab, attribute_name)


class PoolSlot:
    """One warm RSIBox `Lab` (or `Box`) bound to a named slot and optional role."""

    def __init__(self, slot_id: str, lab: Any, role: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        if not isinstance(slot_id, str) or not slot_id.strip():
            raise ValueError("slot_id must be a non-empty string")
        if role is not None and (not isinstance(role, str) or not role.strip()):
            raise ValueError("role must be None or a non-empty string")
        self.slot_id, self.role, self.metadata = slot_id, role, dict(metadata or {})
        self.box: Box | None = lab if isinstance(lab, Box) else None
        self.lab: Any = _BoxLabAdapter(lab) if isinstance(lab, Box) else lab
        self.quarantined, self.quarantine_reason = False, None


class _SlotState:
    __slots__ = ("slot", "in_use", "active_since", "dirty", "busy_s")
    def __init__(self, slot: PoolSlot) -> None:
        self.slot, self.in_use, self.active_since, self.dirty, self.busy_s = slot, False, None, False, 0.0


class _Waiter:
    __slots__ = ("role", "event", "assigned_slot", "assigned_dirty", "cancelled", "failure", "closed")
    def __init__(self, role: str | None) -> None:
        self.role, self.event = role, threading.Event()
        self.assigned_slot, self.assigned_dirty, self.cancelled, self.failure, self.closed = None, False, False, None, False


class SlotLease:
    """Exclusive lease over one `PoolSlot`; supports `with` and idempotent `release()`."""

    def __init__(self, pool: SandboxPool, slot: PoolSlot, agent_id: str, role: str | None, idea_id: str | None,
                 acquired_at: float, acquired_mono: float, queue_wait_s: float, preparation: _CleanupOutcome) -> None:
        self._pool, self.slot, self.slot_id, self.lab = pool, slot, slot.slot_id, slot.lab
        self.agent_id, self.role, self.idea_id = agent_id, role, idea_id
        self.acquired_at, self._acquired_mono, self.queue_wait_s = acquired_at, acquired_mono, queue_wait_s
        self._pre_cleanup_s, self._pre_recovery_s, self._pre_recovered = preparation.cleanup_s, preparation.recovery_s, preparation.recovered
        self.cleanup_s, self.recovery_s, self.recovered = preparation.cleanup_s, preparation.recovery_s, preparation.recovered
        self.released_at, self.hold_s, self.error, self.status_after, self.released = None, None, None, None, False
        self._release_lock, self._trace_record = threading.Lock(), None

    def release(self, error: str | BaseException | None = None) -> dict[str, Any]:
        return self._pool._release_lease(self, error=error)
    def __enter__(self) -> SlotLease:
        return self
    def __exit__(self, exc_type: Any, exc_value: BaseException | None, exc_traceback: Any) -> bool:
        self.release(error=exc_value)
        return False


class SandboxPool:
    """Thread-safe FIFO scheduler multiplexing N agents over C warm RSIBox slots."""

    def __init__(self, slots: Sequence[PoolSlot], *, trace_path: str | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time, close_labs: bool = False) -> None:
        if not slots:
            raise ValueError("SandboxPool requires at least one PoolSlot")
        self._slots: dict[str, _SlotState] = {}
        self._slot_order: list[str] = []
        for slot in slots:
            if not isinstance(slot, PoolSlot):
                raise TypeError(f"expected PoolSlot, got {type(slot).__name__}")
            if slot.slot_id in self._slots:
                raise ValueError(f"duplicate slot_id: {slot.slot_id!r}")
            self._slots[slot.slot_id] = _SlotState(slot)
            self._slot_order.append(slot.slot_id)
        self._trace_path, self._clock, self._wall_clock = trace_path, clock, wall_clock
        self._close_labs, self._created_mono = close_labs, clock()
        self._lock, self._trace_lock, self._waiters = threading.Lock(), threading.Lock(), deque()
        self._closed, self._total_leases, self._recoveries, self._total_recovery_s = False, 0, 0, 0.0
        self._queue_waits, self._hold_times, self._trace_sequence, self._trace_previous_hash = [], [], 0, _GENESIS_HASH
        if trace_path and os.path.exists(trace_path):
            with open(trace_path, encoding="utf-8") as handle:
                for raw_line in handle:
                    if raw_line.strip():
                        record = json.loads(raw_line)
                        self._trace_sequence = int(record.get("seq", self._trace_sequence + 1))
                        self._trace_previous_hash = str(record.get("record_sha256") or _hash_trace_record(record))

    def _has_healthy_slot(self, role: str | None) -> bool:
        return any(not state.slot.quarantined and (role is None or state.slot.role == role) for state in self._slots.values())

    def _claim_slot(self, role: str | None, deadline: float | None) -> tuple[PoolSlot, bool]:
        with self._lock:
            if self._closed:
                raise PoolClosedError("SandboxPool is closed")
            if not self._has_healthy_slot(role):
                raise PoolTimeoutError(f"no healthy slot available for role={role!r}")
            for slot_id in self._slot_order:
                slot_state = self._slots[slot_id]
                slot_role = slot_state.slot.role
                if (not slot_state.in_use and not slot_state.slot.quarantined and (role is None or slot_role == role)
                        and not any(not waiter.cancelled and (waiter.role is None or waiter.role == slot_role) for waiter in self._waiters)):
                    slot_state.in_use, slot_state.active_since = True, self._clock()
                    return slot_state.slot, slot_state.dirty
            if deadline is not None and self._clock() >= deadline:
                raise PoolTimeoutError(f"timed out waiting for slot (role={role!r})")
            waiter = _Waiter(role)
            self._waiters.append(waiter)
        waiter.event.wait(timeout=None if deadline is None else max(0.0, deadline - self._clock()))
        with self._lock:
            if waiter.assigned_slot is not None:
                return waiter.assigned_slot, waiter.assigned_dirty
            waiter.cancelled = True
            with contextlib.suppress(ValueError):
                self._waiters.remove(waiter)
            if waiter.closed:
                raise PoolClosedError("SandboxPool closed while waiting for slot")
            raise PoolTimeoutError(waiter.failure or f"timed out waiting for slot (role={role!r})")

    def _complete_slot_locked(self, slot_id: str, finish_mono: float, healthy: bool, reason: str | None) -> Any | None:
        slot_state = self._slots[slot_id]
        if slot_state.active_since is not None:
            slot_state.busy_s += max(0.0, finish_mono - slot_state.active_since)
        slot_state.slot.quarantined, slot_state.dirty = not healthy, not healthy
        slot_state.slot.quarantine_reason = None if healthy else (reason or "recovery failed")
        if not self._closed and healthy:
            slot_role = slot_state.slot.role
            for index, waiter in enumerate(self._waiters):
                if not waiter.cancelled and (waiter.role is None or waiter.role == slot_role):
                    del self._waiters[index]
                    waiter.assigned_slot, waiter.assigned_dirty = slot_state.slot, False
                    slot_state.in_use, slot_state.active_since = True, finish_mono
                    waiter.event.set()
                    return None
        slot_state.in_use, slot_state.active_since = False, None
        if not healthy:
            remaining: deque[_Waiter] = deque()
            for waiter in self._waiters:
                if not waiter.cancelled and not self._has_healthy_slot(waiter.role):
                    waiter.failure = f"no healthy slot available for role={waiter.role!r}"
                    waiter.event.set()
                elif not waiter.cancelled:
                    remaining.append(waiter)
            self._waiters = remaining
        return slot_state.slot.lab if (self._closed and self._close_labs) else None

    def _clean_and_recover_slot(self, slot: PoolSlot, *, force_reset_state: bool) -> _CleanupOutcome:
        status_reply = _safe_lab_call(slot.lab, "status")
        status_ok = bool(status_reply.get("ok"))
        remaining = list(status_reply.get("txns") or ()) if status_ok else []
        needs_respawn = (not status_ok) or (status_reply.get("state") == "poisoned") or bool(status_reply.get("poison"))
        cleanup_error, cleanup_s = (None if status_ok else status_reply.get("error")), 0.0
        if not needs_respawn:
            cleanup_start, had_transactions = self._clock(), bool(remaining)
            for _ in range(len(remaining) + 1):
                if not remaining:
                    break
                reply = _safe_lab_call(slot.lab, "uninstall", remaining[0])
                if not reply.get("ok") or reply.get("state") == "poisoned":
                    needs_respawn, cleanup_error = True, reply.get("error") or "uninstall failed"
                    break
                if reply.get("state") == "base":
                    remaining = []
                    break
                recheck = _safe_lab_call(slot.lab, "status")
                if not recheck.get("ok") or recheck.get("state") == "poisoned":
                    needs_respawn, cleanup_error = True, recheck.get("error") or "slot poisoned after uninstall"
                    break
                remaining = list(recheck.get("txns") or ())
            if not needs_respawn and remaining:
                needs_respawn, cleanup_error = True, "uninstall did not clear transaction stack"
            if not needs_respawn and (force_reset_state or had_transactions):
                reset_reply = _safe_lab_call(slot.lab, "reset", "state")
                if not reset_reply.get("ok"):
                    needs_respawn, cleanup_error = True, reset_reply.get("error") or 'reset("state") failed'
            if not needs_respawn and not _is_clean_base(after_reply := _safe_lab_call(slot.lab, "status")):
                needs_respawn, cleanup_error = True, after_reply.get("error") or f"post-cleanup state was {after_reply.get('state', 'unknown')!r}"
            cleanup_s = max(0.0, self._clock() - cleanup_start)
        if not needs_respawn:
            return _CleanupOutcome(True, cleanup_s, 0.0, False, "base", None)

        recovery_start = self._clock()
        respawn_reply = _safe_lab_call(slot.lab, "reset", "respawn")
        recovery_s = max(0.0, self._clock() - recovery_start)
        if respawn_reply.get("ok") and _is_clean_base(verified := _safe_lab_call(slot.lab, "status")):
            return _CleanupOutcome(True, cleanup_s, recovery_s, True, "base", None)
        reason = (respawn_reply.get("error") or cleanup_error or 'reset("respawn") failed'
                  if not respawn_reply.get("ok") else
                  verified.get("error") or f"slot not clean after respawn (state={verified.get('state')!r})")
        return _CleanupOutcome(False, cleanup_s, recovery_s, False, "quarantined", str(reason))

    def acquire(self, agent_id: str, *, role: str | None = None, idea_id: str | None = None, timeout_s: float | None = None) -> SlotLease:
        """Acquire an exclusive, clean-slate `SlotLease` in strict FIFO order."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        queue_start = self._clock()
        deadline = None if timeout_s is None else queue_start + max(0.0, float(timeout_s))
        while True:
            slot, slot_dirty = self._claim_slot(role, deadline)
            claim_mono = self._clock()
            preparation = (_CleanupOutcome(True, 0.0, 0.0, False, "base", None)
                           if (not slot_dirty and _is_clean_base(_safe_lab_call(slot.lab, "status")))
                           else self._clean_and_recover_slot(slot, force_reset_state=True))
            if preparation.healthy:
                with self._lock:
                    self._slots[slot.slot_id].dirty = False
                return SlotLease(self, slot, agent_id, role, idea_id, self._wall_clock(), self._clock(),
                                 round(max(0.0, claim_mono - queue_start), 6), preparation)
            finish_mono = self._clock()
            with self._lock:
                self._total_recovery_s += preparation.recovery_s
                lab_to_close = self._complete_slot_locked(slot.slot_id, finish_mono, False, preparation.error or "pre-acquire recovery failed")
            _close_quietly(lab_to_close)

    @contextlib.contextmanager
    def lease(self, agent_id: str, *, role: str | None = None, idea_id: str | None = None, timeout_s: float | None = None) -> Iterator[SlotLease]:
        """Context manager acquiring and cleanly releasing a `SlotLease`."""
        slot_lease = self.acquire(agent_id, role=role, idea_id=idea_id, timeout_s=timeout_s)
        try:
            yield slot_lease
        except BaseException as error:
            slot_lease.release(error=error)
            raise
        finally:
            slot_lease.release()

    def _release_lease(self, lease: SlotLease, *, error: str | BaseException | None = None) -> dict[str, Any]:
        with lease._release_lock:
            if lease.released and lease._trace_record is not None:
                return dict(lease._trace_record)
            hold_s, released_at = max(0.0, self._clock() - lease._acquired_mono), self._wall_clock()
            caller_error = _format_error(error)
            cleanup = self._clean_and_recover_slot(lease.slot, force_reset_state=True)
            finish_mono = self._clock()
            raw_cleanup_s, raw_recovery_s = lease._pre_cleanup_s + cleanup.cleanup_s, lease._pre_recovery_s + cleanup.recovery_s
            recovered = lease._pre_recovered or cleanup.recovered
            combined_error = f"{caller_error}; cleanup: {cleanup.error}" if (caller_error and cleanup.error) else (caller_error or cleanup.error)
            lease.released_at, lease.hold_s = round(released_at, 6), round(hold_s, 6)
            lease.cleanup_s, lease.recovery_s, lease.recovered = round(raw_cleanup_s, 6), round(raw_recovery_s, 6), recovered
            lease.error, lease.status_after, lease.released = combined_error, cleanup.status_after, True
            record = {
                "agent_id": lease.agent_id, "role": lease.role, "slot_role": lease.slot.role, "idea_id": lease.idea_id,
                "slot_id": lease.slot_id, "acquired_at": round(lease.acquired_at, 6), "released_at": lease.released_at,
                "queue_wait_s": lease.queue_wait_s, "hold_s": lease.hold_s, "cleanup_s": lease.cleanup_s,
                "recovery_s": lease.recovery_s, "recovered": recovered, "error": combined_error, "status_after": cleanup.status_after,
            }
            with self._lock:
                self._total_leases, self._recoveries, self._total_recovery_s = (
                    self._total_leases + 1, self._recoveries + int(recovered), self._total_recovery_s + raw_recovery_s,
                )
                self._queue_waits.append(lease.queue_wait_s)
                self._hold_times.append(hold_s)
                lab_to_close = self._complete_slot_locked(lease.slot_id, finish_mono, cleanup.healthy, cleanup.error)
            _close_quietly(lab_to_close)
            lease._trace_record = self._append_trace(record)
            return dict(lease._trace_record)

    def _append_trace(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._trace_lock:
            next_sequence = self._trace_sequence + 1
            chained = {**record, "seq": next_sequence, "prev_sha256": self._trace_previous_hash}
            digest = _hash_trace_record(chained)
            chained["record_sha256"] = digest
            if self._trace_path:
                _append_jsonl_line(self._trace_path, json.dumps(chained, sort_keys=True, default=str) + "\n")
            self._trace_sequence, self._trace_previous_hash = next_sequence, digest
            return chained

    def run_isolated(self, agent_id: str, action_fn: Callable[[Any, SlotLease], Any], *,
                     role: str | None = None, idea_id: str | None = None, timeout_s: float | None = None) -> dict[str, Any]:
        """Acquire a lease, run `action_fn(slot.lab, lease)`, and guarantee clean release."""
        start_mono = self._clock()
        try:
            slot_lease = self.acquire(agent_id, role=role, idea_id=idea_id, timeout_s=timeout_s)
        except PoolError as error:
            return {"ok": False, "result": None, "slot_id": None, "queue_wait_s": round(max(0.0, self._clock() - start_mono), 6),
                    "hold_s": 0.0, "cleanup_s": 0.0, "recovery_s": 0.0, "recovered": False, "status_after": None, "error": _format_error(error)}
        action_result, caught_error = None, None
        try:
            action_result = action_fn(slot_lease.lab, slot_lease)
        except BaseException as error:
            caught_error = error
            if not isinstance(error, Exception):
                raise
        finally:
            trace_record = slot_lease.release(error=caught_error)
        return {
            "ok": caught_error is None, "result": action_result if caught_error is None else None,
            "slot_id": slot_lease.slot_id,
            **{key: trace_record[key] for key in ("queue_wait_s", "hold_s", "cleanup_s", "recovery_s", "recovered", "status_after", "error")},
        }

    def metrics(self) -> dict[str, Any]:
        """Compute exact aggregate queue, hold, recovery, quarantine, and slot utilization stats."""
        with self._lock:
            now_mono = self._clock()
            elapsed_s, total_leases = max(0.0, now_mono - self._created_mono), self._total_leases
            per_slot_utilization = {
                slot_id: round(0.0 if (elapsed_s <= 0.0 or total_busy <= 0.0) else min(1.0, total_busy / elapsed_s), 6)
                for slot_id in self._slot_order
                for total_busy in [self._slots[slot_id].busy_s + (max(0.0, now_mono - self._slots[slot_id].active_since)
                                                                   if self._slots[slot_id].active_since is not None else 0.0)]
            }
            return {
                "total_leases": total_leases,
                "mean_queue_wait_s": round(sum(self._queue_waits) / total_leases, 6) if total_leases else 0.0,
                "p95_queue_wait_s": round(_percentile(self._queue_waits, 0.95), 6),
                "mean_hold_s": round(sum(self._hold_times) / total_leases, 6) if total_leases else 0.0,
                "total_recovery_s": round(self._total_recovery_s, 6), "recoveries": self._recoveries,
                "quarantined_slots": [slot_id for slot_id in self._slot_order if self._slots[slot_id].slot.quarantined],
                "per_slot_utilization": per_slot_utilization,
            }

    def close(self, *, close_labs: bool | None = None) -> None:
        """Close the pool, fail queued waiters, and optionally close idle labs."""
        should_close_labs = self._close_labs if close_labs is None else close_labs
        labs_to_close: list[Any] = []
        with self._lock:
            self._closed, self._close_labs = True, should_close_labs
            while self._waiters:
                waiter = self._waiters.popleft()
                if not waiter.cancelled:
                    waiter.closed = True
                    waiter.event.set()
            if should_close_labs:
                labs_to_close = [self._slots[slot_id].slot.lab for slot_id in self._slot_order if not self._slots[slot_id].in_use]
        for lab in labs_to_close:
            _close_quietly(lab)

    def __enter__(self) -> SandboxPool:
        return self
    def __exit__(self, exc_type: Any, exc_value: BaseException | None, exc_traceback: Any) -> bool:
        self.close()
        return False
