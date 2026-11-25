"""Self-healing autoscaling controller (`PoolAutoScaler`) for `SandboxPool`.

Closes the operational control loop over `SandboxPool.add_slot()`, `drain_slot()`, and
`heal_slot()` using pure Python stdlib:
- Priority 1 (Self-healing): detects quarantined slots, rehabilitates them online via
  `heal_slot()` with capped exponential backoff, and optionally drains + replaces
  persistently dead slots once `max_heal_attempts` is exhausted.
- Priority 2 (Scale-up): repairs `min_slots` deficits immediately and adds slots when
  matching queue backlog reaches `scale_up_queue_depth` (bounded by `max_slots` and
  `scale_up_cooldown_s`), provisioning outside internal locks and closing newly created
  labs if stopped or rejected.
- Priority 3 (Scale-down): drains continuously idle excess slots above `min_slots` after
  `scale_down_idle_s` with `scale_down_cooldown_s` hysteresis and zero aborted leases.
"""
from __future__ import annotations

from collections import deque
import contextlib
import dataclasses
import math
import threading
import time
from typing import Any, Callable

from .pool import (
    Box,
    PoolClosedError,
    PoolError,
    PoolSlot,
    PoolTimeoutError,
    SandboxPool,
    _BoxLabAdapter,
)

__all__ = ["AutoScalerPolicy", "PoolAutoScaler"]


def _check_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return value


def _check_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def _check_finite_float(value: Any, name: str, *, minimum: float = 0.0, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, not {type(value).__name__}")
    try:
        numeric = float(value)
    except OverflowError as err:
        raise ValueError(f"{name} must be finite, got {value!r}") from err
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if strictly_positive and numeric <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    if not strictly_positive and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return numeric


def _unwrap_raw_target(target: Any) -> Any:
    if isinstance(target, _BoxLabAdapter):
        target = target._box
    if isinstance(target, Box):
        return target.lab if target.lab is not None else target
    return target


def _pool_owns_target(pool: Any, target: Any) -> bool:
    if pool is None or target is None:
        return False
    lock = getattr(pool, "_lock", None)
    slots = getattr(pool, "_slots", None)
    if lock is None or not isinstance(slots, dict):
        return False
    raw_target = _unwrap_raw_target(target)
    raw_id = id(raw_target)
    with lock:
        inflight = getattr(pool, "_inflight_heal_targets", None)
        if inflight and raw_id in inflight:
            return True
        for st in slots.values():
            st_slot = getattr(st, "slot", None)
            if st_slot is None:
                continue
            if st_slot.box is target or st_slot.lab is target:
                return True
            if _unwrap_raw_target(st_slot.box or st_slot.lab) is raw_target:
                return True
            if id(SandboxPool._slot_raw_target(st_slot)) == raw_id:
                return True
    return False


def _close_slot_quietly(slot: PoolSlot | None, pool: Any = None) -> None:
    if slot is None:
        return
    target = slot.box if slot.box is not None else slot.lab
    if target is not None and not _pool_owns_target(pool, target):
        with contextlib.suppress(Exception):
            target.close()


@dataclasses.dataclass(frozen=True)
class AutoScalerPolicy:
    """Validated thresholds, cooldowns, and self-healing bounds for `PoolAutoScaler`."""

    min_slots: int = 1
    max_slots: int = 4
    role: str | None = None
    scale_up_queue_depth: int = 2
    scale_down_idle_s: float = 30.0
    scale_up_cooldown_s: float = 5.0
    scale_down_cooldown_s: float = 10.0
    auto_heal: bool = True
    heal_backoff_s: float = 5.0
    max_heal_backoff_s: float = 60.0
    max_heal_attempts: int = 3
    replace_on_heal_exhausted: bool = False
    poll_interval_s: float = 0.5
    drain_timeout_s: float | None = 30.0
    close_on_drain: bool = True
    max_history: int = 64

    def __post_init__(self) -> None:
        _check_int(self.min_slots, "min_slots", minimum=0)
        _check_int(self.max_slots, "max_slots", minimum=1)
        if self.max_slots < self.min_slots:
            raise ValueError(f"max_slots ({self.max_slots}) must be >= min_slots ({self.min_slots})")
        if self.role is not None and (not isinstance(self.role, str) or not self.role.strip()):
            raise ValueError("role must be None or a non-empty string")
        _check_int(self.scale_up_queue_depth, "scale_up_queue_depth", minimum=1)
        _check_finite_float(self.scale_down_idle_s, "scale_down_idle_s", minimum=0.0)
        _check_finite_float(self.scale_up_cooldown_s, "scale_up_cooldown_s", minimum=0.0)
        _check_finite_float(self.scale_down_cooldown_s, "scale_down_cooldown_s", minimum=0.0)
        _check_bool(self.auto_heal, "auto_heal")
        hb = _check_finite_float(self.heal_backoff_s, "heal_backoff_s", minimum=0.0)
        mhb = _check_finite_float(self.max_heal_backoff_s, "max_heal_backoff_s", minimum=0.0)
        if mhb < hb:
            raise ValueError(f"max_heal_backoff_s ({mhb}) must be >= heal_backoff_s ({hb})")
        _check_int(self.max_heal_attempts, "max_heal_attempts", minimum=1)
        _check_bool(self.replace_on_heal_exhausted, "replace_on_heal_exhausted")
        _check_finite_float(self.poll_interval_s, "poll_interval_s", strictly_positive=True)
        if self.drain_timeout_s is not None:
            _check_finite_float(self.drain_timeout_s, "drain_timeout_s", minimum=0.0)
        _check_bool(self.close_on_drain, "close_on_drain")
        _check_int(self.max_history, "max_history", minimum=0)


class PoolAutoScaler:
    """Thread-safe self-healing and elastic scaling controller for a `SandboxPool`.

    Supports both deterministic single-step reconciliation (`reconcile_once()`) and
    daemonized background control (`start()` / `stop()`, or `with PoolAutoScaler(...):`).
    """

    def __init__(
        self,
        pool: SandboxPool,
        slot_factory: Callable[[str, str | None], PoolSlot] | None = None,
        *,
        policy: AutoScalerPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        queue_depth_fn: Callable[[], int] | None = None,
        slot_id_prefix: str = "auto",
    ) -> None:
        if pool is None:
            raise TypeError("pool must not be None")
        if slot_factory is not None and not callable(slot_factory):
            raise TypeError(f"slot_factory must be callable or None, not {type(slot_factory).__name__}")
        if not callable(clock):
            raise TypeError(f"clock must be callable, not {type(clock).__name__}")
        if queue_depth_fn is not None and not callable(queue_depth_fn):
            raise TypeError(f"queue_depth_fn must be callable or None, not {type(queue_depth_fn).__name__}")
        if policy is not None and not isinstance(policy, AutoScalerPolicy):
            raise TypeError(f"policy must be AutoScalerPolicy or None, not {type(policy).__name__}")
        if not isinstance(slot_id_prefix, str) or not slot_id_prefix.strip():
            raise ValueError("slot_id_prefix must be a non-empty string")

        self._pool = pool
        self._slot_factory = slot_factory
        self._policy = policy or AutoScalerPolicy()
        self._clock = clock
        self._queue_depth_fn = queue_depth_fn
        self._slot_id_prefix = slot_id_prefix.strip()

        self._lock = threading.Lock()
        self._reconcile_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._slot_seq = 0
        self._reconciles = 0
        self._scale_ups = 0
        self._scale_downs = 0
        self._heals_attempted = 0
        self._heals_succeeded = 0
        self._heals_failed = 0
        self._replacements = 0
        self._errors = 0
        self._last_scale_up_mono: float | None = None
        self._last_scale_down_mono: float | None = None
        self._idle_since: dict[str, float] = {}
        self._auto_slots: set[str] = set()
        self._heal_failures: dict[str, int] = {}
        self._heal_backoff: dict[str, tuple[float, float]] = {}
        self._exhausted_slots: set[str] = set()
        self._replace_pending: set[str] = set()
        self._replace_heal_errors: dict[str, str] = {}
        self._replace_failures: dict[str, int] = {}
        self._last_action: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=self._policy.max_history)

    @property
    def policy(self) -> AutoScalerPolicy:
        return self._policy

    def _safe_clock(self, fallback: float = 0.0) -> float:
        try:
            return float(self._clock())
        except Exception:
            return fallback

    def _snapshot_pool(self) -> tuple[bool, list[dict[str, Any]], int, set[str]]:
        """Return `(closed, managed_slots, matching_queue_depth, all_slot_ids)`."""
        pool = self._pool
        target_role = self._policy.role
        slot_states_fn = getattr(pool, "slot_states", None)
        if callable(slot_states_fn):
            raw_slots = list(slot_states_fn())
            all_ids = {str(row["slot_id"]) for row in raw_slots}
            closed = bool(getattr(pool, "_closed", False))
            if self._queue_depth_fn is not None:
                queue_depth = max(0, int(self._queue_depth_fn()))
            elif hasattr(pool, "_lock") and hasattr(pool, "_waiters"):
                with pool._lock:
                    closed = bool(getattr(pool, "_closed", False))
                    queue_depth = self._count_matching_waiters_locked(pool, target_role)
            else:
                queue_depth = max(0, int(pool.metrics().get("queue_depth", 0)))
        elif hasattr(pool, "_lock") and hasattr(pool, "_slots"):
            with pool._lock:
                closed = bool(getattr(pool, "_closed", False))
                raw_slots = [
                    {
                        "slot_id": sid,
                        "role": st.role,
                        "in_use": bool(st.in_use),
                        "quarantined": bool(st.quarantined),
                        "draining": bool(st.draining),
                    }
                    for sid, st in pool._slots.items()
                ]
                all_ids = {row["slot_id"] for row in raw_slots}
                queue_depth = (
                    0
                    if self._queue_depth_fn is not None
                    else self._count_matching_waiters_locked(pool, target_role)
                )
            if self._queue_depth_fn is not None:
                queue_depth = max(0, int(self._queue_depth_fn()))
        else:
            m = pool.metrics()
            closed = bool(getattr(pool, "_closed", False))
            q_set = set(m.get("quarantined_slots") or ())
            d_set = set(m.get("draining_slots") or ())
            all_ids = set((m.get("per_slot_utilization") or {}).keys())
            raw_slots = [
                {
                    "slot_id": sid,
                    "role": target_role,
                    "in_use": False,
                    "quarantined": sid in q_set,
                    "draining": sid in d_set,
                }
                for sid in sorted(all_ids)
            ]
            queue_depth = (
                max(0, int(self._queue_depth_fn()))
                if self._queue_depth_fn is not None
                else max(0, int(m.get("queue_depth", 0)))
            )

        if target_role is None:
            managed = raw_slots
        else:
            managed = [row for row in raw_slots if row.get("role") == target_role]
        return closed, managed, queue_depth, all_ids

    @staticmethod
    def _count_matching_waiters_locked(pool: Any, target_role: str | None) -> int:
        waiters = getattr(pool, "_waiters", ())
        under_cap = getattr(pool, "_agent_under_cap_locked", None)
        check_cap = callable(under_cap)
        count = 0
        for waiter in waiters:
            if getattr(waiter, "cancelled", False):
                continue
            if check_cap and not under_cap(getattr(waiter, "agent_id", None)):
                continue
            if target_role is None:
                count += 1
            else:
                w_role = getattr(waiter, "role", None)
                w_fb = getattr(waiter, "fallback_roles", ())
                if w_role is None or w_role == target_role or target_role in w_fb:
                    count += 1
        return count

    def _record_event(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._last_action = dict(event)
            if self._policy.max_history > 0:
                self._events.append(dict(event))
        return event

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._errors += 1
            self._last_error = message

    def _next_slot_id(self, existing_ids: set[str]) -> str:
        with self._lock:
            while True:
                self._slot_seq += 1
                candidate = f"{self._slot_id_prefix}-{self._slot_seq}"
                if candidate not in existing_ids:
                    return candidate

    def _provision_and_add(self, reason: str, now_mono: float) -> dict[str, Any]:
        assert self._slot_factory is not None
        if self._stop_event.is_set():
            return self._record_event({
                "action": "scale_up_aborted",
                "reason": "stopped",
                "cost_s": 0.0,
            })
        closed_pre, managed_pre, _, fresh_ids = self._snapshot_pool()
        if closed_pre:
            return self._record_event({"action": "noop", "reason": "pool_closed"})
        active_pre = sum(1 for row in managed_pre if not row.get("draining"))
        if active_pre >= self._policy.max_slots:
            return self._record_event({
                "action": "scale_up_aborted",
                "reason": "max_slots_reached",
                "cost_s": 0.0,
            })

        slot_id = self._next_slot_id(fresh_ids)
        t0 = self._safe_clock(fallback=now_mono)
        new_slot: PoolSlot | None = None
        try:
            new_slot = self._slot_factory(slot_id, self._policy.role)
            if not isinstance(new_slot, PoolSlot):
                raise TypeError(f"slot_factory must return PoolSlot, got {type(new_slot).__name__}")
            if self._stop_event.is_set():
                _close_slot_quietly(new_slot, self._pool)
                return self._record_event({
                    "action": "scale_up_aborted",
                    "slot_id": new_slot.slot_id,
                    "reason": "stopped",
                    "cost_s": round(max(0.0, self._safe_clock(fallback=t0) - t0), 6),
                })
            closed_post, managed_post, _, _ = self._snapshot_pool()
            if closed_post:
                _close_slot_quietly(new_slot, self._pool)
                return self._record_event({"action": "noop", "reason": "pool_closed"})
            active_post = sum(1 for row in managed_post if not row.get("draining"))
            if active_post >= self._policy.max_slots:
                _close_slot_quietly(new_slot, self._pool)
                return self._record_event({
                    "action": "scale_up_aborted",
                    "slot_id": new_slot.slot_id,
                    "reason": "max_slots_reached",
                    "cost_s": round(max(0.0, self._safe_clock(fallback=t0) - t0), 6),
                })
            added = self._pool.add_slot(new_slot)
        except PoolClosedError as err:
            _close_slot_quietly(new_slot, self._pool)
            return self._record_event({"action": "noop", "reason": f"pool_closed: {err}"})
        except BaseException as err:
            _close_slot_quietly(new_slot, self._pool)
            if not isinstance(err, Exception):
                raise
            msg = f"{type(err).__name__}: {err}"
            self._record_error(msg)
            return self._record_event({
                "action": "scale_up_failed",
                "slot_id": slot_id,
                "reason": reason,
                "error": msg,
                "cost_s": round(max(0.0, self._safe_clock(fallback=t0) - t0), 6),
            })

        finish_mono = self._safe_clock(fallback=t0)
        added_id = str(added["slot_id"])
        with self._lock:
            self._scale_ups += 1
            self._last_scale_up_mono = finish_mono
            self._idle_since.pop(added_id, None)
            self._auto_slots.add(added_id)
        return self._record_event({
            "action": "scale_up",
            "slot_id": added_id,
            "role": added.get("role"),
            "slots": added.get("slots"),
            "reason": reason,
            "cost_s": round(max(0.0, finish_mono - t0), 6),
        })

    def _try_heal_or_replace(
        self, quarantined_candidates: list[dict[str, Any]], now_mono: float
    ) -> dict[str, Any] | None:
        policy = self._policy
        for row in quarantined_candidates:
            sid = str(row["slot_id"])
            with self._lock:
                exhausted = sid in self._exhausted_slots
                replace_pending = sid in self._replace_pending
                backoff_entry = self._heal_backoff.get(sid)
                if backoff_entry is not None and now_mono < backoff_entry[0]:
                    backoff_entry = (now_mono, backoff_entry[1])
                    self._heal_backoff[sid] = backoff_entry
            if exhausted and not replace_pending:
                continue
            if backoff_entry is not None and max(0.0, now_mono - backoff_entry[0]) < backoff_entry[1]:
                continue
            if replace_pending:
                with self._lock:
                    prev_err = self._replace_heal_errors.get(sid, "heal exhausted")
                return self._replace_exhausted_slot(sid, now_mono, prev_err)

            t0 = self._safe_clock(fallback=now_mono)
            try:
                reply = self._pool.heal_slot(sid)
            except PoolClosedError as err:
                return self._record_event({"action": "noop", "reason": f"pool_closed: {err}"})
            except (PoolError, ValueError):
                # Slot has an active undrained call, raced into use/draining, or disappeared; back off cleanly.
                finish_mono = self._safe_clock(fallback=t0)
                with self._lock:
                    self._heal_backoff[sid] = (finish_mono, policy.heal_backoff_s)
                continue
            except Exception as err:
                reply = {"ok": False, "healed": False, "status": "error", "error": f"{type(err).__name__}: {err}"}

            finish_mono = self._safe_clock(fallback=t0)
            cost_s = round(max(0.0, finish_mono - t0), 6)
            if reply.get("status") == "closed":
                return self._record_event({
                    "action": "noop",
                    "reason": f"pool_closed: {reply.get('error') or 'closed during heal_slot'}",
                })
            if reply.get("ok") and reply.get("healed"):
                with self._lock:
                    self._heals_attempted += 1
                    self._heals_succeeded += 1
                    self._heal_failures.pop(sid, None)
                    self._heal_backoff.pop(sid, None)
                    self._exhausted_slots.discard(sid)
                    self._replace_pending.discard(sid)
                    self._replace_heal_errors.pop(sid, None)
                    self._replace_failures.pop(sid, None)
                    self._idle_since.pop(sid, None)
                return self._record_event({
                    "action": "heal",
                    "slot_id": sid,
                    "ok": True,
                    "recovery_s": reply.get("recovery_s", cost_s),
                    "cost_s": cost_s,
                })
            if (reply.get("ok") and not reply.get("healed")) or reply.get("status") == "draining":
                # Already healthy or concurrently draining externally; clear stale tracking and check next slot.
                with self._lock:
                    self._heal_failures.pop(sid, None)
                    self._heal_backoff.pop(sid, None)
                    self._exhausted_slots.discard(sid)
                    self._replace_pending.discard(sid)
                    self._replace_heal_errors.pop(sid, None)
                    self._replace_failures.pop(sid, None)
                continue

            err_msg = str(reply.get("error") or f"heal failed (status={reply.get('status')!r})")
            self._record_error(err_msg)
            should_replace = bool(policy.replace_on_heal_exhausted and self._slot_factory is not None)
            with self._lock:
                self._heals_attempted += 1
                self._heals_failed += 1
                attempts = self._heal_failures.get(sid, 0) + 1
                self._heal_failures[sid] = attempts
                if attempts >= policy.max_heal_attempts:
                    self._exhausted_slots.add(sid)
                    self._heal_backoff.pop(sid, None)
                    if should_replace:
                        self._replace_pending.add(sid)
                        self._replace_heal_errors[sid] = err_msg
                    exhausted_now = True
                else:
                    delay = min(policy.max_heal_backoff_s, policy.heal_backoff_s * (2 ** min(attempts - 1, 30)))
                    self._heal_backoff[sid] = (finish_mono, delay)
                    exhausted_now = False

            if exhausted_now and should_replace:
                return self._replace_exhausted_slot(sid, finish_mono, err_msg)

            return self._record_event({
                "action": "heal_exhausted" if exhausted_now else "heal_failed",
                "slot_id": sid,
                "attempts": attempts,
                "error": err_msg,
                "cost_s": cost_s,
            })
        return None

    def _replace_exhausted_slot(
        self, dead_slot_id: str, now_mono: float, heal_error: str
    ) -> dict[str, Any]:
        policy = self._policy
        t0 = self._safe_clock(fallback=now_mono)
        drain_err: str | None = None
        try:
            dr = self._pool.drain_slot(
                dead_slot_id,
                timeout_s=policy.drain_timeout_s,
                close_lab=policy.close_on_drain,
            )
            if dr.get("status") == "closed":
                return self._record_event({
                    "action": "noop",
                    "reason": f"pool_closed: {dr.get('error') or 'closed during drain_slot'}",
                })
            if not dr.get("ok") or not dr.get("drained"):
                drain_err = str(dr.get("error") or "drain_slot returned not drained")
        except PoolClosedError as err:
            return self._record_event({"action": "noop", "reason": f"pool_closed: {err}"})
        except ValueError as err:
            with self._lock:
                self._heal_failures.pop(dead_slot_id, None)
                self._heal_backoff.pop(dead_slot_id, None)
                self._exhausted_slots.discard(dead_slot_id)
                self._replace_pending.discard(dead_slot_id)
                self._replace_heal_errors.pop(dead_slot_id, None)
                self._replace_failures.pop(dead_slot_id, None)
                self._idle_since.pop(dead_slot_id, None)
                self._auto_slots.discard(dead_slot_id)
            return self._record_event({"action": "noop", "reason": f"slot_removed: {err}"})
        except Exception as err:
            drain_err = f"{type(err).__name__}: {err}"

        if drain_err is not None:
            finish_mono = self._safe_clock(fallback=t0)
            self._record_error(drain_err)
            with self._lock:
                r_attempts = self._replace_failures.get(dead_slot_id, 0) + 1
                self._replace_failures[dead_slot_id] = r_attempts
                delay = min(policy.max_heal_backoff_s, policy.heal_backoff_s * (2 ** min(r_attempts - 1, 30)))
                self._heal_backoff[dead_slot_id] = (finish_mono, delay)
                self._exhausted_slots.add(dead_slot_id)
                self._replace_pending.add(dead_slot_id)
                self._replace_heal_errors[dead_slot_id] = heal_error
            return self._record_event({
                "action": "replace_failed",
                "slot_id": dead_slot_id,
                "error": drain_err,
            })

        with self._lock:
            self._heal_failures.pop(dead_slot_id, None)
            self._heal_backoff.pop(dead_slot_id, None)
            self._exhausted_slots.discard(dead_slot_id)
            self._replace_pending.discard(dead_slot_id)
            self._replace_heal_errors.pop(dead_slot_id, None)
            self._replace_failures.pop(dead_slot_id, None)
            self._idle_since.pop(dead_slot_id, None)
            self._auto_slots.discard(dead_slot_id)

        prov = self._provision_and_add("replace_exhausted", self._safe_clock(fallback=t0))
        if prov.get("action") == "scale_up":
            with self._lock:
                self._replacements += 1
            return self._record_event({
                "action": "replace",
                "replaced_slot_id": dead_slot_id,
                "new_slot_id": prov["slot_id"],
                "heal_error": heal_error,
                "cost_s": round(max(0.0, self._safe_clock(fallback=t0) - t0), 6),
            })
        return prov

    def _cooldown_elapsed(self, last_mono: float | None, cooldown_s: float, now_mono: float) -> bool:
        if last_mono is None or cooldown_s <= 0.0:
            return True
        return max(0.0, now_mono - last_mono) >= cooldown_s

    def reconcile_once(self) -> dict[str, Any]:
        """Execute one deterministic control-loop pass (heal -> scale-up -> scale-down)."""
        with self._reconcile_lock:
            now_mono = self._safe_clock()
            with self._lock:
                self._reconciles += 1
            closed, managed, queue_depth, all_ids = self._snapshot_pool()
            if closed:
                return self._record_event({"action": "noop", "reason": "pool_closed"})

            # Prune stale tracking for slots removed or manually healed externally, and clamp clock skew.
            quarantined_ids = {row["slot_id"] for row in managed if row.get("quarantined")}
            managed_ids = {row["slot_id"] for row in managed}
            with self._lock:
                if self._last_scale_up_mono is not None and now_mono < self._last_scale_up_mono:
                    self._last_scale_up_mono = now_mono
                if self._last_scale_down_mono is not None and now_mono < self._last_scale_down_mono:
                    self._last_scale_down_mono = now_mono
                for sid in list(self._heal_failures):
                    if sid not in quarantined_ids:
                        self._heal_failures.pop(sid, None)
                for sid in list(self._heal_backoff):
                    if sid not in quarantined_ids:
                        self._heal_backoff.pop(sid, None)
                    elif now_mono < self._heal_backoff[sid][0]:
                        self._heal_backoff[sid] = (now_mono, self._heal_backoff[sid][1])
                self._exhausted_slots.intersection_update(quarantined_ids)
                self._replace_pending.intersection_update(quarantined_ids)
                for sid in list(self._replace_heal_errors):
                    if sid not in quarantined_ids:
                        self._replace_heal_errors.pop(sid, None)
                for sid in list(self._replace_failures):
                    if sid not in quarantined_ids:
                        self._replace_failures.pop(sid, None)
                self._auto_slots.intersection_update(managed_ids)
                for sid in list(self._idle_since):
                    if sid not in managed_ids:
                        self._idle_since.pop(sid, None)
                    elif now_mono < self._idle_since[sid]:
                        self._idle_since[sid] = now_mono

            # Update continuous idle timers (any active work or queued waiter resets idle clocks).
            idle_healthy: list[dict[str, Any]] = []
            healthy_slots: list[dict[str, Any]] = []
            active_managed: list[dict[str, Any]] = []
            quarantined_idle: list[dict[str, Any]] = []
            with self._lock:
                for row in managed:
                    sid = row["slot_id"]
                    is_draining = bool(row.get("draining"))
                    is_quarantined = bool(row.get("quarantined"))
                    is_in_use = bool(row.get("in_use"))
                    if not is_draining:
                        active_managed.append(row)
                    if is_quarantined and not is_in_use and not is_draining:
                        quarantined_idle.append(row)
                    if not is_quarantined and not is_draining:
                        healthy_slots.append(row)
                        if not is_in_use and queue_depth == 0:
                            self._idle_since.setdefault(sid, now_mono)
                            idle_healthy.append(row)
                        else:
                            self._idle_since.pop(sid, None)
                    else:
                        self._idle_since.pop(sid, None)

            policy = self._policy

            # Priority 1: Self-heal quarantined slots (outside lock, exponential backoff).
            if policy.auto_heal and quarantined_idle:
                heal_event = self._try_heal_or_replace(quarantined_idle, now_mono)
                if heal_event is not None:
                    return heal_event

            # Priority 2: Scale up for min_slots deficit or queue pressure.
            if self._slot_factory is not None and len(active_managed) < policy.max_slots:
                if len(healthy_slots) < policy.min_slots:
                    return self._provision_and_add("below_min_slots", now_mono)
                with self._lock:
                    last_up = self._last_scale_up_mono
                if queue_depth >= policy.scale_up_queue_depth and self._cooldown_elapsed(
                    last_up, policy.scale_up_cooldown_s, now_mono
                ):
                    return self._provision_and_add(f"queue_depth={queue_depth}", now_mono)

            # Priority 3: Scale down one continuously idle excess slot above min_slots.
            if len(healthy_slots) > policy.min_slots and queue_depth == 0 and idle_healthy:
                with self._lock:
                    last_down = self._last_scale_down_mono
                    auto_slots = set(self._auto_slots)
                    idle_snapshot = {row["slot_id"]: self._idle_since.get(row["slot_id"], now_mono) for row in idle_healthy}
                if self._cooldown_elapsed(last_down, policy.scale_down_cooldown_s, now_mono):
                    eligible = [
                        (0 if sid in auto_slots else 1, started, sid)
                        for sid, started in idle_snapshot.items()
                        if max(0.0, now_mono - started) >= policy.scale_down_idle_s
                    ]
                    if eligible:
                        eligible.sort()
                        _, started, victim_id = eligible[0]
                        idle_for_s = round(max(0.0, now_mono - started), 6)
                        t0 = self._safe_clock(fallback=now_mono)
                        try:
                            dr = self._pool.drain_slot(
                                victim_id,
                                timeout_s=policy.drain_timeout_s,
                                close_lab=policy.close_on_drain,
                            )
                        except (PoolTimeoutError, ValueError) as err:
                            with self._lock:
                                self._idle_since.pop(victim_id, None)
                            return self._record_event({
                                "action": "scale_down_skipped",
                                "slot_id": victim_id,
                                "reason": str(err),
                            })
                        except PoolClosedError as err:
                            return self._record_event({"action": "noop", "reason": f"pool_closed: {err}"})
                        except Exception as err:
                            msg = f"{type(err).__name__}: {err}"
                            self._record_error(msg)
                            with self._lock:
                                self._idle_since.pop(victim_id, None)
                            return self._record_event({
                                "action": "scale_down_failed",
                                "slot_id": victim_id,
                                "error": msg,
                            })

                        if dr.get("status") == "closed":
                            return self._record_event({
                                "action": "noop",
                                "reason": f"pool_closed: {dr.get('error') or 'closed during drain_slot'}",
                            })
                        finish_mono = self._safe_clock(fallback=t0)
                        if dr.get("ok") and dr.get("drained"):
                            with self._lock:
                                self._scale_downs += 1
                                self._last_scale_down_mono = finish_mono
                                self._idle_since.pop(victim_id, None)
                                self._auto_slots.discard(victim_id)
                            return self._record_event({
                                "action": "scale_down",
                                "slot_id": victim_id,
                                "idle_s": idle_for_s,
                                "drain_s": dr.get("drain_s", 0.0),
                                "slots": dr.get("slots"),
                                "cost_s": round(max(0.0, finish_mono - t0), 6),
                            })
                        with self._lock:
                            self._idle_since.pop(victim_id, None)
                        return self._record_event({
                            "action": "scale_down_skipped",
                            "slot_id": victim_id,
                            "reason": str(dr.get("error") or dr.get("status") or "not drained"),
                        })

            return self._record_event({
                "action": "noop",
                "healthy_slots": len(healthy_slots),
                "quarantined_slots": len(quarantined_ids),
                "queue_depth": queue_depth,
            })

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ev = self.reconcile_once()
                if ev.get("action") == "noop" and str(ev.get("reason", "")).startswith("pool_closed"):
                    break
            except Exception as err:
                self._record_error(f"{type(err).__name__}: {err}")
            self._stop_event.wait(min(self._policy.poll_interval_s, threading.TIMEOUT_MAX))

    def start(self) -> PoolAutoScaler:
        """Start the background reconciliation daemon thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name=f"rsibox-autoscaler-{self._slot_id_prefix}",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self, *, timeout_s: float | None = 5.0) -> None:
        """Signal the background thread to stop and wait for clean exit (idempotent)."""
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            join_timeout = None if timeout_s is None else min(timeout_s, threading.TIMEOUT_MAX)
            thread.join(timeout=join_timeout)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def status(self) -> dict[str, Any]:
        """Return an atomic non-blocking telemetry snapshot of autoscaler state."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "running": running,
                "role": self._policy.role,
                "min_slots": self._policy.min_slots,
                "max_slots": self._policy.max_slots,
                "reconciles": self._reconciles,
                "scale_ups": self._scale_ups,
                "scale_downs": self._scale_downs,
                "heals_attempted": self._heals_attempted,
                "heals_succeeded": self._heals_succeeded,
                "heals_failed": self._heals_failed,
                "replacements": self._replacements,
                "errors": self._errors,
                "exhausted_slots": sorted(self._exhausted_slots),
                "last_action": dict(self._last_action) if self._last_action is not None else None,
                "last_error": self._last_error,
                "recent_events": [dict(ev) for ev in self._events],
            }

    def __enter__(self) -> PoolAutoScaler:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: BaseException | None, exc_tb: Any) -> bool:
        self.stop()
        return False
