"""Process-level recovery drivers and engine quiesce hooks (stdlib only).

Separates two responsibilities that were previously tangled inside `BaseLab.reset("respawn")`:
1. `QuiesceHooks`: the minimal 3-step engine lifecycle contract (`warmup`, `quiesce`, `resume`)
   that lets any RSIBox engine cooperate with process snapshots, shadow failover, or clean respawn.
2. `RecoveryDriver`: how a poisoned or crashed worker group is brought back (`DefaultRespawnDriver`,
   `StandbyShadowDriver`, or `DynamoK8sSnapshotDriver` in `snapshot_contract.py`).
"""
from __future__ import annotations

from collections import deque
import contextlib
import inspect
import logging
import os
import threading
import time
from typing import Any, Callable, Protocol, runtime_checkable

from .types import Mutability, QuiesceReceipt
from .weight_store import WeightStore

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class QuiesceHooks(Protocol):
    """Minimal engine lifecycle contract for checkpoint/restore and clean failover.

    Any RSIBox engine (vLLM, SGLang, FastVideo, Megatron-LM, veRL, Miles) implementing
    these three methods can plug into `ai-dynamo/snapshot` (`$SNAPSHOT_CONTROL_DIR`),
    shadow promotion, or coordinated warm respawns without engine-specific orchestration.
    """

    mutability: Mutability

    def warmup_for_snapshot(self) -> None:
        """Run one real step so lazy CUDA contexts, autotuning, and CUDA graphs exist."""
        ...

    def quiesce_for_snapshot(self) -> QuiesceReceipt:
        """Quiesce at a step boundary, drain in-flight work, and park disposable GPU memory."""
        ...

    def resume_after_restore(self, receipt: QuiesceReceipt) -> dict[str, Any]:
        """Re-attach persistent weights (e.g. GMS) and restore volatile runtime memory."""
        ...


@runtime_checkable
class RecoveryDriver(Protocol):
    """Single owner of `BaseLab.reset(kind='respawn')` execution."""

    name: str

    def respawn(
        self,
        lab: Any,
        store: WeightStore | None = None,
        *,
        reason: str = "reset_respawn",
    ) -> dict[str, Any]:
        """Recover `lab` from a crash, poison state, or startup-flag change."""
        ...


def _resolve_lab_tp(lab: Any) -> int | None:
    """Resolve TP degree across VllmLab (`tp`), SglangLab (`ranks` / `config['tp_size']`), etc."""
    for attr in ("tp", "ranks"):
        val = getattr(lab, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    cfg = getattr(lab, "config", None)
    if isinstance(cfg, dict):
        cfg_tp = cfg.get("tp_size")
        if isinstance(cfg_tp, int) and cfg_tp > 0:
            return cfg_tp
    return None


class DefaultRespawnDriver:
    """Default respawn path: verify `WeightStore` (fail closed), stop worker group, respawn."""

    name = "default_respawn"

    def respawn(
        self,
        lab: Any,
        store: WeightStore | None = None,
        *,
        reason: str = "reset_respawn",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        store_probe = None
        if store is not None and store.name != "none":
            probe_kw: dict[str, Any] = {
                "tp": _resolve_lab_tp(lab),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            if store.name == "dynamo_gms" and getattr(lab, "supports_gms_uuid_shim", False):
                probe_kw["allow_remapped"] = True
            store_probe = store.probe(**probe_kw)
            if not store_probe.ok:
                return {
                    "ok": False,
                    "reset": [],
                    "driver": self.name,
                    "reason": reason,
                    "error": store_probe.error or f"weight store {store.name!r} probe failed",
                    "weight_store": store_probe.to_dict(),
                    "cost_s": round(time.perf_counter() - t0, 6),
                }
        with contextlib.suppress(Exception):
            lab._stop_engine()
        lab._spawn()
        out: dict[str, Any] = {
            "ok": not lab.session.poisoned(),
            "reset": ["worker group", "session"],
            "driver": self.name,
            "reason": reason,
            "cost_s": round(time.perf_counter() - t0, 6),
        }
        if store_probe is not None:
            out["weight_store"] = store_probe.to_dict()
        return out


def _build_worker_invoker(
    replenish_worker: Callable[..., Any],
    cancel_event: threading.Event,
) -> Callable[[Any, WeightStore | None], Any]:
    """Build a signature-aware invoker supporting 0/1/2/3-arg and `cancel_event` workers."""
    try:
        parameters = list(inspect.signature(replenish_worker).parameters.values())
    except (TypeError, ValueError):
        return lambda lab, store: replenish_worker(lab, store)
    has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters)
    pos_count = sum(
        param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for param in parameters
    )
    kwonly_names = {param.name for param in parameters if param.kind == inspect.Parameter.KEYWORD_ONLY}
    if "cancel_event" in kwonly_names:
        if pos_count == 0 and not has_varargs:
            return lambda lab, store: replenish_worker(cancel_event=cancel_event)
        return lambda lab, store: replenish_worker(lab, store, cancel_event=cancel_event)
    if has_varargs or pos_count == 2:
        return lambda lab, store: replenish_worker(lab, store)
    if pos_count == 0:
        return lambda lab, store: replenish_worker()
    if pos_count == 1:
        return lambda lab, store: replenish_worker(cancel_event)
    return lambda lab, store: replenish_worker(lab, store, cancel_event)


def _resolve_async_replenisher(replenish_fn: Any) -> Any | None:
    """Return the async replenisher backing `replenish_fn` (instance, duck-typed, or bound `.trigger`)."""
    for candidate in (replenish_fn, getattr(replenish_fn, "__self__", None)):
        if candidate is None:
            continue
        if isinstance(candidate, AsyncShadowReplenisher) or (
            callable(getattr(candidate, "status", None)) and callable(getattr(candidate, "wait", None))
        ):
            return candidate
    return None


class AsyncShadowReplenisher:
    """Off-critical-path single-flight background replenisher for `StandbyShadowDriver`.

    Runs `replenish_worker` in a daemon thread so standby promotion returns immediately.
    Guarantees single-flight execution (coalescing at most 1 pending request when
    `queue_when_busy=True`, or returning `"busy"` when False), exposes `cancel_event` and
    optional `dispose_fn` for late post-close outcomes, records bounded telemetry, and
    catches all background worker errors without leaking thread exceptions.
    """

    def __init__(
        self,
        replenish_worker: Callable[..., Any],
        *,
        queue_when_busy: bool = True,
        max_history: int = 64,
        thread_name_prefix: str = "rsibox-shadow-replenish",
        dispose_fn: Callable[[Any], None] | None = None,
    ):
        if not callable(replenish_worker):
            raise TypeError(f"replenish_worker must be callable, got {type(replenish_worker).__name__}")
        if dispose_fn is not None and not callable(dispose_fn):
            raise TypeError(f"dispose_fn must be callable or None, got {type(dispose_fn).__name__}")
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 0:
            raise ValueError(f"max_history must be a non-negative integer, got {max_history!r}")
        self._replenish_worker = replenish_worker
        self._dispose_fn = dispose_fn
        self._queue_when_busy = bool(queue_when_busy)
        self._max_history = int(max_history)
        self._thread_name_prefix = str(thread_name_prefix or "rsibox-shadow-replenish")
        self._cancel_event = threading.Event()
        self._invoke_worker = _build_worker_invoker(replenish_worker, self._cancel_event)
        self._condition = threading.Condition(threading.Lock())
        self._in_flight = self._closed = self._closed_with_live_worker = False
        self._pending_request: tuple[Any, WeightStore | None] | None = None
        self._active_thread: threading.Thread | None = None
        self._thread_seq = self._attempts = self._succeeded = self._failed = 0
        self._coalesced = self._busy_rejected = self._disposed_after_close = 0
        self._last_ok: bool | None = None
        self._last_cost_s: float | None = None
        self._last_error: str | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=self._max_history)

    def _read_attr(self, attr_name: str) -> Any:
        with self._condition:
            return getattr(self, attr_name)

    attempts = property(lambda self: self._read_attr("_attempts"))
    succeeded = property(lambda self: self._read_attr("_succeeded"))
    failed = property(lambda self: self._read_attr("_failed"))
    coalesced = property(lambda self: self._read_attr("_coalesced"))
    busy_rejected = property(lambda self: self._read_attr("_busy_rejected"))
    disposed_after_close = property(lambda self: self._read_attr("_disposed_after_close"))
    last_ok = property(lambda self: self._read_attr("_last_ok"))
    last_cost_s = property(lambda self: self._read_attr("_last_cost_s"))
    last_error = property(lambda self: self._read_attr("_last_error"))
    in_flight = property(lambda self: self._read_attr("_in_flight"))
    closed = property(lambda self: self._read_attr("_closed"))
    closed_with_live_worker = property(lambda self: self._read_attr("_closed_with_live_worker"))

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    @property
    def pending(self) -> bool:
        return self._read_attr("_pending_request") is not None

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        with self._condition:
            return tuple(dict(entry) for entry in self._history)

    def status(self) -> dict[str, Any]:
        """Return an atomic, JSON-serializable telemetry snapshot."""
        with self._condition:
            return {
                "attempts": self._attempts,
                "succeeded": self._succeeded,
                "failed": self._failed,
                "coalesced": self._coalesced,
                "busy_rejected": self._busy_rejected,
                "disposed_after_close": self._disposed_after_close,
                "last_ok": self._last_ok,
                "last_cost_s": self._last_cost_s,
                "last_error": self._last_error,
                "in_flight": self._in_flight,
                "pending": self._pending_request is not None,
                "closed": self._closed,
                "closed_with_live_worker": self._closed_with_live_worker,
                "history": [dict(entry) for entry in self._history],
            }

    def _spawn_locked(self, lab: Any, store: WeightStore | None) -> str:
        self._in_flight = True
        self._thread_seq += 1
        worker_thread = threading.Thread(
            target=self._run_loop, args=(lab, store),
            name=f"{self._thread_name_prefix}-{self._thread_seq}", daemon=True,
        )
        self._active_thread = worker_thread
        try:
            worker_thread.start()
        except RuntimeError as error:
            now_wall = time.time()
            self._record_attempt_locked(
                attempt_ok=False, cost_s=0.0, started_wall=now_wall, finished_wall=now_wall,
                error_message=f"{type(error).__name__}: {error}",
            )
            self._mark_idle_locked()
            return "failed"
        except BaseException:
            self._mark_idle_locked()
            raise
        return "started"

    def trigger(self, lab: Any, store: WeightStore | None = None) -> str:
        """Schedule off-critical-path shadow replenishment and return dispatch status."""
        with self._condition:
            if self._closed:
                return "closed"
            if self._in_flight:
                if not self._queue_when_busy:
                    self._busy_rejected += 1
                    return "busy"
                already_pending = self._pending_request is not None
                self._pending_request = (lab, store)
                if already_pending:
                    self._coalesced += 1
                    return "coalesced"
                return "queued"
            return self._spawn_locked(lab, store)

    __call__ = trigger

    def _run_loop(self, initial_lab: Any, initial_store: WeightStore | None) -> None:
        current_lab, current_store = initial_lab, initial_store
        try:
            while True:
                started_wall, started_monotonic = time.time(), time.perf_counter()
                attempt_ok, error_message, outcome = False, None, None
                try:
                    outcome = self._invoke_worker(current_lab, current_store)
                    if isinstance(outcome, dict) and outcome.get("ok") is False:
                        error_message = str(outcome.get("error") or "replenish worker returned ok=False")
                    else:
                        attempt_ok = True
                except BaseException as error:
                    error_message = f"{type(error).__name__}: {error}"

                cost_s = round(time.perf_counter() - started_monotonic, 6)
                finished_wall = time.time()
                disposed = dispose_attempted = False
                while True:
                    with self._condition:
                        need_dispose = (not dispose_attempted) and self._closed and attempt_ok and outcome is not None and self._dispose_fn is not None
                        if not need_dispose:
                            self._record_attempt_locked(
                                attempt_ok=attempt_ok, cost_s=cost_s, started_wall=started_wall,
                                finished_wall=finished_wall, error_message=error_message, disposed=disposed,
                            )
                            if not self._closed and self._pending_request is not None:
                                current_lab, current_store = self._pending_request
                                self._pending_request = None
                                self._condition.notify_all()
                                break
                            self._closed_with_live_worker = False
                            self._mark_idle_locked()
                            return
                    dispose_attempted = True
                    try:
                        self._dispose_fn(outcome)
                        disposed = True
                    except BaseException as dispose_error:
                        attempt_ok = False
                        error_message = f"dispose_fn failed: {type(dispose_error).__name__}: {dispose_error}"
        except BaseException:  # pragma: no cover - defensive state cleanup
            with self._condition:
                self._closed_with_live_worker = False
                self._mark_idle_locked()

    def _record_attempt_locked(
        self,
        *,
        attempt_ok: bool,
        cost_s: float,
        started_wall: float,
        finished_wall: float,
        error_message: str | None,
        disposed: bool = False,
    ) -> None:
        self._attempts += 1
        self._succeeded += int(attempt_ok)
        self._failed += int(not attempt_ok)
        self._disposed_after_close += int(disposed)
        self._last_ok = attempt_ok
        self._last_cost_s = cost_s
        self._last_error = error_message
        if self._max_history > 0:
            entry: dict[str, Any] = {
                "attempt": self._attempts,
                "ok": attempt_ok,
                "cost_s": cost_s,
                "started_at": round(started_wall, 6),
                "finished_at": round(finished_wall, 6),
            }
            if disposed:
                entry["disposed"] = True
            if error_message is not None:
                entry["error"] = error_message
            self._history.append(entry)

    def _mark_idle_locked(self) -> None:
        self._pending_request = None
        self._in_flight = False
        self._active_thread = None
        self._condition.notify_all()

    def wait(self, timeout_s: float | None = None) -> bool:
        """Block until all in-flight and pending replenishments complete."""
        deadline = None if timeout_s is None else (time.monotonic() + max(0.0, float(timeout_s)))
        with self._condition:
            if self._active_thread is threading.current_thread():
                raise RuntimeError("cannot call wait() from inside the active replenishment thread")
            while self._in_flight or self._pending_request is not None:
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._condition.wait(timeout=remaining)
            return True

    def close(self, timeout_s: float | None = 5.0) -> bool:
        """Signal cancellation, drop pending work, and join any active worker."""
        with self._condition:
            self._closed = True
            self._pending_request = None
            self._cancel_event.set()
            active_thread = self._active_thread
            self._condition.notify_all()
        if active_thread is not None and active_thread is not threading.current_thread():
            active_thread.join(timeout=timeout_s)
            with self._condition:
                still_alive = self._in_flight and active_thread.is_alive()
                self._closed_with_live_worker = still_alive
            return not still_alive
        with self._condition:
            still_in_flight = self._in_flight
            self._closed_with_live_worker = still_in_flight
        return not still_in_flight

    def __enter__(self) -> AsyncShadowReplenisher:
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        if not self.close():
            _LOGGER.warning(
                "AsyncShadowReplenisher.close() timed out with an active background worker still running"
            )


class StandbyShadowDriver:
    """Promote a pre-initialized standby worker sharing read-only `WeightStore` weights.

    Designed for both single-GPU GMS Shadow Engines (e.g. Dynamo GMS + scratch KV) and
    Phase-2 parallel RL verifier pools where a poisoned worker must fail over in seconds
    while replenishment runs out of the critical path.
    """

    name = "standby_shadow"

    def __init__(
        self,
        promote_fn: Callable[[Any, WeightStore | None], dict[str, Any]],
        replenish_fn: Callable[[Any, WeightStore | None], Any] | None = None,
        *,
        replenish_wait_timeout_s: float = 0.0,
        fallback_driver: RecoveryDriver | None = None,
    ):
        if isinstance(replenish_wait_timeout_s, bool) or replenish_wait_timeout_s < 0:
            raise ValueError(
                f"replenish_wait_timeout_s must be a non-negative float, got {replenish_wait_timeout_s!r}"
            )
        self._promote_fn = promote_fn
        self._replenish_fn = replenish_fn
        self._replenish_wait_timeout_s = float(replenish_wait_timeout_s)
        self._fallback_driver = fallback_driver

    def _standby_guard(
        self, async_replenisher: Any, lab: Any, store: WeightStore | None, *, reason: str, started_at: float
    ) -> dict[str, Any] | None:
        """Verify standby readiness before stopping the primary; wait or fall back if not ready."""
        replenisher_state = async_replenisher.status()
        if (replenisher_state.get("in_flight") or replenisher_state.get("pending")) and self._replenish_wait_timeout_s > 0.0:
            async_replenisher.wait(timeout_s=self._replenish_wait_timeout_s)
            replenisher_state = async_replenisher.status()

        is_closed = bool(replenisher_state.get("closed"))
        still_busy = bool(replenisher_state.get("in_flight") or replenisher_state.get("pending"))
        last_failed = replenisher_state.get("attempts", 0) > 0 and (
            replenisher_state.get("last_ok") is False or replenisher_state.get("last_error") is not None
        )
        if not is_closed and not still_busy and not last_failed:
            return None

        if self._fallback_driver is not None:
            fallback_result = self._fallback_driver.respawn(lab, store, reason=reason)
            fallback_result["fallback_from"] = self.name
            if last_failed and not still_busy and not is_closed and fallback_result.get("ok") and self._replenish_fn is not None:
                fallback_result["replenish"] = self._replenish_fn(lab, store)
            return fallback_result

        if is_closed:
            error_text, replenish_status = "standby_not_ready: replenisher is closed", "closed"
        elif still_busy:
            error_text, replenish_status = "standby_not_ready: replenishment still in flight", "in_flight"
        else:
            error_text = f"standby_not_ready: last replenishment failed ({replenisher_state.get('last_error')})"
            replenish_status = self._replenish_fn(lab, store) if self._replenish_fn is not None else None
        return {
            "ok": False, "reset": [], "driver": self.name, "recovery_mode": self.name, "reason": reason,
            "error": error_text, "replenish": replenish_status, "replenisher_status": replenisher_state,
            "cost_s": round(time.perf_counter() - started_at, 6),
        }

    def respawn(
        self,
        lab: Any,
        store: WeightStore | None = None,
        *,
        reason: str = "reset_respawn",
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        async_replenisher = _resolve_async_replenisher(self._replenish_fn)
        if async_replenisher is not None:
            guard_result = self._standby_guard(
                async_replenisher, lab, store, reason=reason, started_at=started_at
            )
            if guard_result is not None:
                return guard_result

        with contextlib.suppress(Exception):
            lab._stop_engine()
        promoted = self._promote_fn(lab, store) or {}
        promote_cost_s = round(time.perf_counter() - started_at, 6)
        ok = bool(promoted.get("ok", True)) and not lab.session.poisoned()
        replenish_outcome: Any = None
        if ok and self._replenish_fn is not None:
            replenish_outcome = self._replenish_fn(lab, store)
        reserved = {"ok", "reset", "driver", "reason", "cost_s", "replenish"}
        result = {
            "ok": ok,
            "reset": list(promoted.get("reset", ["shadow promoted", "session"])) if ok else [],
            "driver": self.name,
            "reason": reason,
            "cost_s": promote_cost_s,
            **{key: value for key, value in promoted.items() if key not in reserved},
        }
        if ok and async_replenisher is not None:
            result["replenish"] = replenish_outcome
        return result
