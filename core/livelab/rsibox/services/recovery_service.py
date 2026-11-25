"""S6: Unified Recovery & Health Service (`RecoveryService`, pure Python stdlib).

Coordinates RSIBox recovery tiers (`DefaultRespawnDriver`, `StandbyShadowDriver`,
`DynamoK8sSnapshotDriver`), `QuiesceHooks` snapshot handshakes, `SandboxPool` /
`PoolAutoScaler` health aggregation, per-endpoint single-flight fencing, and
owner-scoped teardown without importing `torch`, `numpy`, `kernel`, or `repos`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import threading
import time
from typing import Any, Callable

from ..autoscaler import PoolAutoScaler
from ..pool import SandboxPool
from ..warm.recovery import DefaultRespawnDriver, QuiesceHooks, RecoveryDriver, StandbyShadowDriver
from ..warm.snapshot_contract import DynamoK8sSnapshotDriver
from ..warm.types import QuiesceReceipt
from ..warm.weight_store import WeightStore


class RecoveryTier(StrEnum):
    """Supported RSIBox process/state recovery tiers."""
    COLD_RESPAWN = "cold_respawn"; STANDBY_SHADOW = "standby_shadow"; SNAPSHOT_RESTORE = "snapshot_restore"


class ServiceHealthStatus(StrEnum):
    """Lifecycle and fault status for a managed recovery endpoint."""
    HEALTHY = "healthy"; DEGRADED = "degraded"; RECOVERING = "recovering"
    QUARANTINED = "quarantined"; CLOSED = "closed"


class RecoveryServiceError(RuntimeError): """Base error raised by `RecoveryService`."""
class RecoveryClosedError(RecoveryServiceError): """Raised fail-closed when operating on a closed `RecoveryService`."""
class EndpointNotFoundError(RecoveryServiceError): """Raised when referencing an unregistered `endpoint_id`."""
class RecoveryInProgressError(RecoveryServiceError): """Raised when concurrent recovery or quiesce targets an active endpoint."""
class EscalationFailedError(RecoveryServiceError): """Raised when primary recovery and fallback escalation both fail."""


@dataclass(frozen=True)
class ServiceHealth:
    """Immutable health snapshot for a single registered endpoint."""
    endpoint_id: str; engine: str; tier: RecoveryTier; status: ServiceHealthStatus
    incarnation: int; consecutive_failures: int
    last_error: str | None = None; last_recovered_s: float | None = None; recovery_count: int = 0


@dataclass(frozen=True)
class RecoveryReceipt:
    """Structured outcome returned by `RecoveryService.recover()`."""
    endpoint_id: str; tier_attempted: RecoveryTier; tier_succeeded: RecoveryTier | None
    ok: bool; fallback_used: bool; cost_s: float; incarnation: int
    driver_receipt: dict[str, Any] = field(default_factory=dict); error: str | None = None


@dataclass
class _EndpointEntry:
    endpoint_id: str; engine: str; tier: RecoveryTier; driver: RecoveryDriver
    fallback_driver: RecoveryDriver | None; store: WeightStore | None
    lab: Any; lab_factory: Callable[[], Any] | None; quiesce_hooks: QuiesceHooks | None
    metadata: dict[str, Any]; own_driver: bool; own_store: bool; quarantine_threshold: int
    owns_lab: bool = False; status: ServiceHealthStatus = ServiceHealthStatus.HEALTHY
    incarnation: int = 1; consecutive_failures: int = 0; last_error: str | None = None
    last_recovered_s: float | None = None; recovery_count: int = 0; recovering: bool = False


@dataclass
class _PoolEntry:
    pool_id: str; pool: SandboxPool; autoscaler: PoolAutoScaler | None; own: bool


_DRIVER_NAME_TO_TIER = {
    "default_respawn": RecoveryTier.COLD_RESPAWN,
    "standby_shadow": RecoveryTier.STANDBY_SHADOW,
    "dynamo_k8s_snapshot": RecoveryTier.SNAPSHOT_RESTORE,
}


def _require_str(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip(): raise ValueError(f"{name} must be a non-empty string, got {v!r}")
    return v
def _require_bool(v: Any, name: str) -> bool:
    if not isinstance(v, bool): raise TypeError(f"{name} must be a bool, not {type(v).__name__}")
    return v
def _require_pos_int(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int): raise TypeError(f"{name} must be an int, not {type(v).__name__}")
    if v < 1: raise ValueError(f"{name} must be >= 1, got {v!r}")
    return v
def _format_error(err: str | BaseException) -> str:
    if isinstance(err, BaseException): return f"{type(err).__name__}: {err}".strip(": ")
    if isinstance(err, str) and err.strip(): return err.strip()
    raise ValueError(f"error must be a non-empty string or BaseException, got {err!r}")
def _validate_driver(drv: Any, name: str) -> RecoveryDriver:
    if not callable(getattr(drv, "respawn", None)): raise TypeError(f"{name} must implement respawn(lab, store=..., reason=...)")
    return drv
def _validate_hooks(hooks: Any) -> QuiesceHooks:
    for m in ("warmup_for_snapshot", "quiesce_for_snapshot", "resume_after_restore"):
        if not callable(getattr(hooks, m, None)): raise TypeError(f"quiesce_hooks missing callable {m}()")
    return hooks
def _infer_tier(driver: Any, default: RecoveryTier) -> RecoveryTier:
    if isinstance(driver, StandbyShadowDriver): return RecoveryTier.STANDBY_SHADOW
    if isinstance(driver, DynamoK8sSnapshotDriver): return RecoveryTier.SNAPSHOT_RESTORE
    if isinstance(driver, DefaultRespawnDriver): return RecoveryTier.COLD_RESPAWN
    return _DRIVER_NAME_TO_TIER.get(getattr(driver, "name", ""), default)
def _snapshot_health(e: _EndpointEntry) -> ServiceHealth:
    return ServiceHealth(e.endpoint_id, e.engine, e.tier, e.status, e.incarnation,
                         e.consecutive_failures, e.last_error, e.last_recovered_s, e.recovery_count)


def _close_labs(labs: list[Any], seen: set[int] | None = None, errors: list[str] | None = None) -> None:
    id_set = seen if seen is not None else set()
    for lab in labs:
        if lab is None or id(lab) in id_set: continue
        closer = getattr(lab, "close", None)
        if not callable(closer): continue
        id_set.add(id(lab))
        try: closer()
        except Exception as exc:
            if errors is not None: errors.append(f"lab({type(lab).__name__}): {_format_error(exc)}")


def _invoke_driver(
    driver: RecoveryDriver, lab: Any, store: WeightStore | None, reason: str, driver_kwargs: dict[str, Any],
) -> tuple[bool, dict[str, Any], str | None]:
    try:
        raw = driver.respawn(lab, store=store, reason=reason, **driver_kwargs) if driver_kwargs else driver.respawn(lab, store=store, reason=reason)
    except Exception as exc:
        err = _format_error(exc)
        return False, {"ok": False, "driver": getattr(driver, "name", type(driver).__name__), "error": err}, err
    if not isinstance(raw, dict):
        err = f"{getattr(driver, 'name', type(driver).__name__)}.respawn() returned {type(raw).__name__}"
        return False, {"ok": False, "error": err}, err
    ok = bool(raw.get("ok", False))
    return ok, dict(raw), None if ok else str(raw.get("error") or f"{raw.get('driver', 'driver')} returned ok=False")


def _close_driver_resources(driver: Any, seen: set[int], errors: list[str]) -> None:
    repl = getattr(driver, "_replenish_fn", None)
    for obj in (driver, repl, getattr(repl, "__self__", None)):
        if obj is None or id(obj) in seen: continue
        closer = getattr(obj, "close", None) if callable(getattr(obj, "close", None)) else getattr(obj, "stop", None)
        if callable(closer):
            seen.add(id(obj))
            try: closer()
            except Exception as exc: errors.append(f"driver({type(obj).__name__}): {_format_error(exc)}")


class RecoveryService:
    """Thread-safe S6 coordinator unifying recovery drivers, quiesce hooks, and pool health."""

    def __init__(self, *, quarantine_threshold: int = 3) -> None:
        self._quarantine_threshold = _require_pos_int(quarantine_threshold, "quarantine_threshold")
        self._lock = threading.RLock()
        self._teardown_done = threading.Event()
        self._closed = self._torn_down = False
        self._endpoints: dict[str, _EndpointEntry] = {}
        self._pools: dict[str, _PoolEntry] = {}

    def __enter__(self) -> RecoveryService:
        with self._lock: self._require_open_locked()
        return self
    def __exit__(self, *_exc: Any) -> None: self.close()
    def _require_open_locked(self) -> None:
        if self._closed: raise RecoveryClosedError("RecoveryService is closed")
    def _get_entry_locked(self, endpoint_id: str) -> _EndpointEntry:
        entry = self._endpoints.get(endpoint_id)
        if entry is None: raise EndpointNotFoundError(f"unknown endpoint {endpoint_id!r}")
        return entry

    @staticmethod
    def _apply_fault_locked(entry: _EndpointEntry, err_text: str | None) -> None:
        entry.consecutive_failures += 1
        entry.last_error = err_text
        if entry.status != ServiceHealthStatus.RECOVERING:
            entry.status = ServiceHealthStatus.QUARANTINED if entry.consecutive_failures >= entry.quarantine_threshold else ServiceHealthStatus.DEGRADED

    def _begin_op_locked(self, eid: str, *, set_recovering: bool) -> tuple[_EndpointEntry, ServiceHealthStatus]:
        self._require_open_locked()
        entry = self._get_entry_locked(eid)
        if entry.recovering: raise RecoveryInProgressError(f"recovery already in progress for endpoint {eid!r}")
        entry.recovering = True
        prev_status = entry.status
        if set_recovering: entry.status = ServiceHealthStatus.RECOVERING
        return entry, prev_status

    def register_endpoint(
        self, endpoint_id: str, *, engine: str, lab: Any = None, lab_factory: Callable[[], Any] | None = None,
        store: WeightStore | None = None, driver: RecoveryDriver | None = None,
        tier: RecoveryTier | str | None = None, fallback_driver: RecoveryDriver | None = None,
        quiesce_hooks: QuiesceHooks | None = None, metadata: dict[str, Any] | None = None,
        own_driver: bool = True, own_store: bool = False, own_lab: bool = False, quarantine_threshold: int | None = None,
    ) -> ServiceHealth:
        """Register an endpoint with its primary recovery driver, optional fallback, and hooks."""
        eid, eng = _require_str(endpoint_id, "endpoint_id"), _require_str(engine, "engine")
        _require_bool(own_driver, "own_driver"); _require_bool(own_store, "own_store"); _require_bool(own_lab, "own_lab")
        if lab_factory is not None and not callable(lab_factory):
            raise TypeError(f"lab_factory must be callable or None, not {type(lab_factory).__name__}")
        if store is not None and not callable(getattr(store, "probe", None)):
            raise TypeError("store must implement probe() when provided")
        if store is not None and own_store and not callable(getattr(store, "stop", None)):
            raise TypeError("store must implement stop() when own_store=True")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict or None, not {type(metadata).__name__}")
        resolved_tier = (
            (_infer_tier(driver, RecoveryTier.COLD_RESPAWN) if driver is not None else RecoveryTier.COLD_RESPAWN)
            if tier is None else RecoveryTier(tier)
        )
        threshold = self._quarantine_threshold if quarantine_threshold is None else _require_pos_int(quarantine_threshold, "quarantine_threshold")
        if driver is not None: resolved_driver = _validate_driver(driver, "driver")
        elif resolved_tier == RecoveryTier.COLD_RESPAWN: resolved_driver = DefaultRespawnDriver()
        elif resolved_tier == RecoveryTier.SNAPSHOT_RESTORE: resolved_driver = DynamoK8sSnapshotDriver(engine_name=eng)
        else: raise ValueError("driver is required when registering tier='standby_shadow'")
        fb_drv = _validate_driver(fallback_driver, "fallback_driver") if fallback_driver is not None else None
        hooks = _validate_hooks(quiesce_hooks) if quiesce_hooks is not None else None
        with self._lock:
            self._require_open_locked()
            if eid in self._endpoints: raise RecoveryServiceError(f"endpoint {eid!r} is already registered")
            entry = _EndpointEntry(eid, eng, resolved_tier, resolved_driver, fb_drv, store, lab, lab_factory, hooks,
                                   dict(metadata) if metadata is not None else {}, own_driver, own_store, threshold,
                                   owns_lab=(own_lab and lab is not None))
            self._endpoints[eid] = entry
            return _snapshot_health(entry)

    def attach_pool(
        self, pool_id: str, pool: SandboxPool, *, autoscaler: PoolAutoScaler | None = None, own: bool = False,
    ) -> dict[str, Any]:
        """Attach a `SandboxPool` (and optional `PoolAutoScaler`) for health aggregation and teardown."""
        pid = _require_str(pool_id, "pool_id"); _require_bool(own, "own")
        if pool is None or not callable(getattr(pool, "slot_states", None)) or not callable(getattr(pool, "metrics", None)):
            raise TypeError("pool must implement slot_states() and metrics()")
        if autoscaler is not None and not callable(getattr(autoscaler, "status", None)):
            raise TypeError("autoscaler must implement status() when provided")
        with self._lock:
            self._require_open_locked()
            if pid in self._pools: raise RecoveryServiceError(f"pool {pid!r} is already attached")
            self._pools[pid] = _PoolEntry(pid, pool, autoscaler, own)
        return {"pool_id": pid, "own": own, "has_autoscaler": autoscaler is not None}

    def record_failure(self, endpoint_id: str, error: str | BaseException) -> ServiceHealth:
        """Record an endpoint fault and transition health to degraded or quarantined."""
        eid, err_text = _require_str(endpoint_id, "endpoint_id"), _format_error(error)
        with self._lock:
            self._require_open_locked()
            entry = self._get_entry_locked(eid)
            self._apply_fault_locked(entry, err_text)
            return _snapshot_health(entry)

    def recover(
        self, endpoint_id: str, *, reason: str = "fault", allow_fallback: bool = True, reset_after: bool = True, **driver_kwargs: Any,
    ) -> RecoveryReceipt:
        """Execute single-flight recovery with optional fallback escalation outside internal locks."""
        eid, clean_reason = _require_str(endpoint_id, "endpoint_id"), _require_str(reason, "reason")
        _require_bool(allow_fallback, "allow_fallback"); _require_bool(reset_after, "reset_after")
        with self._lock:
            entry, prev_status = self._begin_op_locked(eid, set_recovering=True)
            driver, fallback_driver, primary_tier = entry.driver, entry.fallback_driver, entry.tier
            store, old_lab, old_owns_lab, lab_factory = entry.store, entry.lab, entry.owns_lab, entry.lab_factory

        committed, started, target_lab, created_lab, extra_labs = False, time.perf_counter(), old_lab, None, []

        def _track_candidate(cand: Any) -> None:
            if cand is not None and cand is not old_lab and cand is not created_lab:
                extra_labs.append(cand)

        try:
            lab_err: str | None = None
            if target_lab is None and lab_factory is not None:
                try: created_lab = lab_factory(); target_lab = created_lab
                except Exception as exc: lab_err = f"lab_factory failed: {_format_error(exc)}"
            ok1, raw1, err1 = ((False, {"ok": False, "error": lab_err}, lab_err) if lab_err is not None
                               else _invoke_driver(driver, target_lab, store, clean_reason, driver_kwargs))
            _track_candidate(raw1.get("lab"))
            fallback_used = bool(raw1.get("fallback_from"))
            if fallback_used and raw1.get("promote_error") and not ok1:
                err1 = f"promote: {raw1['promote_error']}; fallback: {err1}"
            tier_succeeded: RecoveryTier | None = None
            final_ok, final_raw, final_err = ok1, raw1, err1
            if ok1:
                tier_succeeded = (
                    _DRIVER_NAME_TO_TIER.get(str(raw1.get("driver", "")), RecoveryTier.COLD_RESPAWN)
                    if fallback_used else _infer_tier(driver, default=primary_tier)
                )
                if fallback_used:
                    final_err = str(raw1.get("promote_error") or f"fallback from {raw1.get('fallback_from')}")
            elif allow_fallback and fallback_driver is not None:
                with self._lock: self._require_open_locked()
                fb_lab_err: str | None = None
                if target_lab is None and lab_factory is not None:
                    try: created_lab = lab_factory(); target_lab = created_lab
                    except Exception as exc: fb_lab_err = f"lab_factory failed: {_format_error(exc)}"
                fb_tier = _infer_tier(fallback_driver, default=RecoveryTier.COLD_RESPAWN)
                fb_ok, fb_raw, fb_err = ((False, {"ok": False, "error": fb_lab_err}, fb_lab_err) if fb_lab_err is not None
                                         else _invoke_driver(fallback_driver, target_lab, store, clean_reason, driver_kwargs))
                _track_candidate(fb_raw.get("lab"))
                if fb_ok:
                    fallback_used, final_ok, tier_succeeded = True, True, fb_tier
                    final_raw, final_err = {**fb_raw, "primary_error": err1}, err1
                else:
                    combined = f"primary ({primary_tier.value}) failed: {err1}; fallback ({fb_tier.value}) failed: {fb_err}"
                    with self._lock:
                        entry.recovering, committed = False, True
                        self._require_open_locked()
                        entry.status = prev_status
                        self._apply_fault_locked(entry, combined)
                    raise EscalationFailedError(combined)

            cost_s = round(time.perf_counter() - started, 6)
            labs_to_close: list[Any] = []
            with self._lock:
                entry.recovering, committed = False, True
                self._require_open_locked()
                if final_ok:
                    entry.incarnation += 1; entry.recovery_count += 1; entry.last_recovered_s = cost_s
                    returned_lab = final_raw.get("lab")
                    new_lab = returned_lab if returned_lab is not None else target_lab
                    if new_lab is not None:
                        entry.lab = new_lab
                        entry.owns_lab = old_owns_lab or (new_lab is created_lab) or (returned_lab is not None and returned_lab is not old_lab)
                    extra_labs = [x for x in extra_labs if x is not new_lab]
                    if created_lab is not None and created_lab is not new_lab: labs_to_close.append(created_lab)
                    if old_owns_lab and old_lab is not None and old_lab is not new_lab: labs_to_close.append(old_lab)
                    if fallback_used: entry.last_error = final_err
                    elif reset_after: entry.last_error = None
                    if reset_after: entry.consecutive_failures, entry.status = 0, ServiceHealthStatus.HEALTHY
                    else: entry.status = (ServiceHealthStatus.QUARANTINED if entry.consecutive_failures >= entry.quarantine_threshold
                                          else (ServiceHealthStatus.DEGRADED if entry.consecutive_failures > 0 else ServiceHealthStatus.HEALTHY))
                else:
                    if created_lab is not None: labs_to_close.append(created_lab)
                    entry.status = prev_status
                    self._apply_fault_locked(entry, final_err)
                incarnation = entry.incarnation
            _close_labs([*labs_to_close, *extra_labs])
            return RecoveryReceipt(eid, primary_tier, tier_succeeded, final_ok, fallback_used, cost_s, incarnation, final_raw, final_err)
        except BaseException:
            _close_labs([created_lab, *extra_labs])
            raise
        finally:
            if not committed:
                with self._lock:
                    entry.recovering = False
                    if not self._closed and entry.status == ServiceHealthStatus.RECOVERING:
                        entry.status = (ServiceHealthStatus.QUARANTINED if entry.consecutive_failures >= entry.quarantine_threshold
                                        else (ServiceHealthStatus.DEGRADED if entry.consecutive_failures > 0 and prev_status == ServiceHealthStatus.HEALTHY else prev_status))

    def quiesce_and_checkpoint(self, endpoint_id: str, *, skip_warmup: bool = False) -> QuiesceReceipt:
        """Run `warmup_for_snapshot()` -> `quiesce_for_snapshot()` (or K8s snapshot) fail-closed."""
        eid = _require_str(endpoint_id, "endpoint_id"); _require_bool(skip_warmup, "skip_warmup")
        with self._lock:
            entry, _ = self._begin_op_locked(eid, set_recovering=False)
            hooks, driver, lab = entry.quiesce_hooks, entry.driver, entry.lab
        committed = False
        try:
            raw_hooks: Any = hooks
            if raw_hooks is None and lab is not None and callable(getattr(lab, "quiesce_hooks", None)):
                raw_hooks = lab.quiesce_hooks()
            if raw_hooks is None: raise RecoveryServiceError(f"endpoint {eid!r} has no quiesce_hooks configured")
            validated_hooks = _validate_hooks(raw_hooks)
            if isinstance(driver, DynamoK8sSnapshotDriver):
                res = driver.checkpoint_and_wait(validated_hooks, skip_warmup=skip_warmup)
                if not res.ok: raise RecoveryServiceError(res.error or "snapshot checkpoint failed")
                receipt = res.quiesce
            else:
                if not skip_warmup: validated_hooks.warmup_for_snapshot()
                receipt = validated_hooks.quiesce_for_snapshot()
                if not isinstance(receipt, QuiesceReceipt):
                    raise RecoveryServiceError(f"quiesce_for_snapshot() returned {type(receipt).__name__}, expected QuiesceReceipt")
                if not receipt.ok: raise RecoveryServiceError(receipt.error or "quiesce_for_snapshot failed")
            with self._lock:
                entry.recovering, committed = False, True
                self._require_open_locked()
                return receipt
        except RecoveryClosedError: raise
        except Exception as exc:
            err_text = _format_error(exc)
            with self._lock:
                entry.recovering, committed = False, True
                if self._closed: raise RecoveryClosedError("RecoveryService is closed") from exc
                self._apply_fault_locked(entry, err_text)
            if isinstance(exc, RecoveryServiceError): raise
            raise RecoveryServiceError(err_text) from exc
        finally:
            if not committed:
                with self._lock: entry.recovering = False

    def get_health(self, endpoint_id: str) -> ServiceHealth:
        """Return an immutable `ServiceHealth` snapshot for `endpoint_id`."""
        eid = _require_str(endpoint_id, "endpoint_id")
        with self._lock: return _snapshot_health(self._get_entry_locked(eid))

    def health_snapshot(self) -> dict[str, Any]:
        """Aggregate endpoint health and attached pool/autoscaler state without holding internal locks."""
        with self._lock:
            closed = self._closed
            ep_rows = [(_snapshot_health(e), dict(e.metadata)) for e in self._endpoints.values()]
            pool_rows = [(p.pool_id, p.pool, p.autoscaler, p.own) for p in self._pools.values()]
        endpoints_out: dict[str, dict[str, Any]] = {}
        endpoints_ok = True
        for h, meta in ep_rows:
            if h.status != ServiceHealthStatus.HEALTHY: endpoints_ok = False
            endpoints_out[h.endpoint_id] = {**asdict(h), "tier": h.tier.value, "status": h.status.value, "metadata": meta}
        pools_out: dict[str, dict[str, Any]] = {}
        pools_ok = True
        for pid, pool, autoscaler, own in pool_rows:
            pool_closed = bool(getattr(pool, "_closed", False))
            slots, metrics, scaler_status, pool_err = [], {}, None, None
            try:
                slots, metrics = list(pool.slot_states()), dict(pool.metrics())
                if autoscaler is not None: scaler_status = dict(autoscaler.status())
            except Exception as exc:
                pool_err = _format_error(exc)
            pool_ok = (not pool_closed) and (pool_err is None) and (not any(s.get("quarantined") for s in slots))
            if not pool_ok: pools_ok = False
            p_out: dict[str, Any] = {"pool_id": pid, "own": own, "ok": pool_ok, "closed": pool_closed,
                                     "slot_states": slots, "metrics": metrics, "autoscaler": scaler_status}
            if pool_err is not None: p_out["error"] = pool_err
            pools_out[pid] = p_out
        return {"ok": (not closed) and endpoints_ok and pools_ok, "closed": closed,
                "endpoint_count": len(endpoints_out), "pool_count": len(pools_out), "endpoints": endpoints_out, "pools": pools_out}

    def probe_health(self, *, probe_stores: bool = True) -> dict[str, Any]:
        """Aggregate `health_snapshot()` plus optional live `WeightStore` probes outside locks."""
        _require_bool(probe_stores, "probe_stores")
        snapshot = self.health_snapshot()
        with self._lock:
            if self._closed:
                snapshot["closed"], snapshot["ok"], snapshot["stores"] = True, False, {}
                return snapshot
            store_rows = [(e.endpoint_id, e.store) for e in self._endpoints.values() if e.store is not None]
        stores_out, stores_ok = {}, True
        if probe_stores:
            for eid, store in store_rows:
                assert store is not None
                try: probe = store.probe(); probe_dict, stores_ok = probe.to_dict(), stores_ok and bool(probe.ok)
                except Exception as exc:
                    stores_ok, probe_dict = False, {"ok": False, "backend": getattr(store, "name", "unknown"), "error": _format_error(exc)}
                stores_out[eid] = probe_dict
        snapshot["stores"], snapshot["ok"] = stores_out, bool(snapshot["ok"]) and stores_ok
        return snapshot

    def teardown(self, close_owned_drivers: bool = True) -> dict[str, Any]:
        """Idempotently close the service and stop owned pools, drivers, labs, and stores outside locks."""
        _require_bool(close_owned_drivers, "close_owned_drivers")
        with self._lock:
            already = self._torn_down
            if not already:
                self._closed = self._torn_down = True
                for entry in self._endpoints.values(): entry.status = ServiceHealthStatus.CLOSED
                owned_drivers = ([d for e in self._endpoints.values() if e.own_driver for d in (e.driver, e.fallback_driver) if d is not None]
                                 if close_owned_drivers else [])
                owned_stores = [e.store for e in self._endpoints.values() if e.own_store and e.store is not None]
                owned_labs = [e.lab for e in self._endpoints.values() if e.owns_lab and e.lab is not None]
                owned_pools = [p for p in self._pools.values() if p.own]
            else:
                owned_drivers, owned_stores, owned_labs, owned_pools = [], [], [], []
        if already:
            self._teardown_done.wait(timeout=5.0)
            return {"already_closed": True, "stopped_pools": [], "stopped_drivers": 0, "stopped_stores": [], "errors": []}

        stopped_pools, stopped_stores, errors = [], [], []
        seen_driver_ids, seen_store_ids, seen_lab_ids = set(), set(), set()
        try:
            for pentry in owned_pools:
                for label, obj, method in (("autoscaler", pentry.autoscaler, "stop"), ("pool", pentry.pool, "close")):
                    if obj is not None:
                        try:
                            getattr(obj, method)()
                            if label == "pool": stopped_pools.append(pentry.pool_id)
                        except Exception as exc: errors.append(f"{label}({pentry.pool_id}): {_format_error(exc)}")
            for drv in owned_drivers: _close_driver_resources(drv, seen_driver_ids, errors)
            _close_labs(owned_labs, seen_lab_ids, errors)
            for store in owned_stores:
                if id(store) not in seen_store_ids:
                    seen_store_ids.add(id(store))
                    try: store.stop(); stopped_stores.append(getattr(store, "name", type(store).__name__))
                    except Exception as exc: errors.append(f"store({getattr(store, 'name', 'unknown')}): {_format_error(exc)}")
        finally:
            self._teardown_done.set()
        return {"already_closed": False, "stopped_pools": stopped_pools,
                "stopped_drivers": len(seen_driver_ids), "stopped_stores": stopped_stores, "errors": errors}

    def close(self) -> None:
        """Close the service idempotently, releasing all owned resources."""
        self.teardown()


__all__ = [
    "EndpointNotFoundError", "EscalationFailedError", "RecoveryClosedError",
    "RecoveryInProgressError", "RecoveryReceipt", "RecoveryService",
    "RecoveryServiceError", "RecoveryTier", "ServiceHealth", "ServiceHealthStatus",
]
