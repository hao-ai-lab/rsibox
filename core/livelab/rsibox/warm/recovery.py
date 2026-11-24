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
import os
import threading
import time
from typing import Any, Callable, Protocol, runtime_checkable

from .types import Mutability, QuiesceReceipt
from .weight_store import WeightStore


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


class AsyncShadowReplenisher:
    """Off-critical-path single-flight background replenisher for `StandbyShadowDriver`.

    Runs `replenish_worker(lab, store) -> Any` in a daemon thread so standby promotion
    returns immediately. Guarantees single-flight execution (coalescing at most 1 pending
    request when `queue_when_busy=True`, or returning `"busy"` when False), records bounded
    telemetry, and catches all background worker errors without leaking thread exceptions.
    """

    def __init__(
        self,
        replenish_worker: Callable[[Any, WeightStore | None], Any],
        *,
        queue_when_busy: bool = True,
        max_history: int = 64,
        thread_name_prefix: str = "rsibox-shadow-replenish",
    ):
        if not callable(replenish_worker):
            raise TypeError(f"replenish_worker must be callable, got {type(replenish_worker).__name__}")
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 0:
            raise ValueError(f"max_history must be a non-negative integer, got {max_history!r}")
        self._replenish_worker = replenish_worker
        self._queue_when_busy = bool(queue_when_busy)
        self._max_history = int(max_history)
        self._thread_name_prefix = str(thread_name_prefix or "rsibox-shadow-replenish")
        self._condition = threading.Condition(threading.Lock())
        self._in_flight = False
        self._pending_request: tuple[Any, WeightStore | None] | None = None
        self._active_thread: threading.Thread | None = None
        self._closed = False
        self._thread_seq = 0
        self._attempts = self._succeeded = self._failed = 0
        self._coalesced = self._busy_rejected = 0
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
    last_cost_s = property(lambda self: self._read_attr("_last_cost_s"))
    last_error = property(lambda self: self._read_attr("_last_error"))
    in_flight = property(lambda self: self._read_attr("_in_flight"))
    closed = property(lambda self: self._read_attr("_closed"))

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
                "last_cost_s": self._last_cost_s,
                "last_error": self._last_error,
                "in_flight": self._in_flight,
                "pending": self._pending_request is not None,
                "closed": self._closed,
                "history": [dict(entry) for entry in self._history],
            }

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
            self._in_flight = True
            self._thread_seq += 1
            worker_thread = threading.Thread(
                target=self._run_loop,
                args=(lab, store),
                name=f"{self._thread_name_prefix}-{self._thread_seq}",
                daemon=True,
            )
            self._active_thread = worker_thread
            try:
                worker_thread.start()
            except BaseException:
                self._in_flight = False
                self._active_thread = None
                self._condition.notify_all()
                raise
            return "started"

    __call__ = trigger

    def _run_loop(self, initial_lab: Any, initial_store: WeightStore | None) -> None:
        current_lab, current_store = initial_lab, initial_store
        try:
            while True:
                started_wall, started_monotonic = time.time(), time.perf_counter()
                attempt_ok, error_message = False, None
                try:
                    outcome = self._replenish_worker(current_lab, current_store)
                    if isinstance(outcome, dict) and outcome.get("ok") is False:
                        error_message = str(outcome.get("error") or "replenish worker returned ok=False")
                    else:
                        attempt_ok = True
                except BaseException as error:
                    error_message = f"{type(error).__name__}: {error}"
                cost_s = round(time.perf_counter() - started_monotonic, 6)
                finished_wall = time.time()

                with self._condition:
                    self._attempts += 1
                    self._succeeded += int(attempt_ok)
                    self._failed += int(not attempt_ok)
                    self._last_cost_s = cost_s
                    self._last_error = error_message
                    entry: dict[str, Any] = {
                        "attempt": self._attempts,
                        "ok": attempt_ok,
                        "cost_s": cost_s,
                        "started_at": round(started_wall, 6),
                        "finished_at": round(finished_wall, 6),
                    }
                    if error_message is not None:
                        entry["error"] = error_message
                    if self._max_history > 0:
                        self._history.append(entry)
                    if not self._closed and self._pending_request is not None:
                        current_lab, current_store = self._pending_request
                        self._pending_request = None
                        self._condition.notify_all()
                        continue
                    self._mark_idle_locked()
                    return
        except BaseException:  # pragma: no cover - defensive state cleanup
            with self._condition:
                self._mark_idle_locked()

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
        """Reject new triggers, drop unstarted pending work, and join any active worker."""
        with self._condition:
            self._closed = True
            self._pending_request = None
            active_thread = self._active_thread
            self._condition.notify_all()
        if active_thread is not None and active_thread is not threading.current_thread():
            active_thread.join(timeout=timeout_s)
            return not active_thread.is_alive()
        return not self.in_flight

    def __enter__(self) -> AsyncShadowReplenisher:
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()


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
    ):
        self._promote_fn = promote_fn
        self._replenish_fn = replenish_fn

    def respawn(
        self,
        lab: Any,
        store: WeightStore | None = None,
        *,
        reason: str = "reset_respawn",
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        with contextlib.suppress(Exception):
            lab._stop_engine()
        promoted = self._promote_fn(lab, store) or {}
        promote_cost_s = round(time.perf_counter() - started_at, 6)
        ok = bool(promoted.get("ok", True)) and not lab.session.poisoned()
        replenish_outcome: Any = None
        if ok and self._replenish_fn is not None:
            replenish_outcome = self._replenish_fn(lab, store)
        reserved = {"ok", "reset", "driver", "reason", "cost_s"}
        result = {
            "ok": ok,
            "reset": list(promoted.get("reset", ["shadow promoted", "session"])) if ok else [],
            "driver": self.name,
            "reason": reason,
            "cost_s": promote_cost_s,
            **{key: value for key, value in promoted.items() if key not in reserved},
        }
        if ok and isinstance(self._replenish_fn, AsyncShadowReplenisher):
            result["replenish"] = replenish_outcome
        return result
