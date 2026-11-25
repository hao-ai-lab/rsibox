"""S4: Shared Auxiliary & Frozen Stage Pool (`AuxiliaryModelService`, pure Python stdlib).

Manages shared, frozen, non-updated auxiliary model stages (VAE, text encoder,
reward model, reference policy, judge) serving concurrent inference/RL workers
with read-only S1 weight-handle fencing, per-stage bounded LRU result caching,
atomic version invalidation, and owner-scoped teardown.
"""
from __future__ import annotations

from collections import OrderedDict
import contextlib
import copy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import PurePath
import threading
import time
from typing import Any, Callable, Sequence

from .memory import MemoryHandle, MemoryPoolKind, MemoryService, SnapshotLease

_ALLOWED_AUX_POOLS: frozenset[MemoryPoolKind] = frozenset({MemoryPoolKind.RO_AUX_WEIGHTS, MemoryPoolKind.RO_WEIGHTS})
_MAX_INVOKE_RETRIES = 8


class AuxStageKind(StrEnum):
    """Canonical frozen auxiliary stage categories."""
    VAE = "vae"; TEXT_ENCODER = "text_encoder"; REWARD_MODEL = "reward_model"
    REFERENCE_MODEL = "reference_model"; JUDGE = "judge"; CUSTOM = "custom"


class AuxServiceError(RuntimeError): """Base error raised by `AuxiliaryModelService`."""
class AuxClosedError(AuxServiceError): """Raised fail-closed when operating on a closed `AuxiliaryModelService`."""
class StageNotFoundError(AuxServiceError): """Raised when referencing an unregistered `stage_id`."""
class StageExistsError(AuxServiceError): """Raised when registering a duplicate `stage_id`."""
class AuxInvocationError(AuxServiceError): """Raised when stage invocation or digest key extraction fails."""


@dataclass(frozen=True)
class AuxStageSpec:
    """Immutable descriptor for a registered frozen auxiliary model stage."""
    stage_id: str; kind: AuxStageKind; engine: str; model_id: str; version: str
    handle_id: str | None = None; gpus: tuple[int, ...] = (); read_only: bool = True
    max_cache_entries: int = 256; metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuxInvocationReceipt:
    """Immutable outcome receipt returned by `AuxiliaryModelService.invoke()`."""
    stage_id: str; kind: AuxStageKind; version: str; caller_id: str; input_digest: str
    cache_hit: bool; cost_s: float; lease_id: str | None; output: Any


@dataclass
class _StageEntry:
    spec: AuxStageSpec; infer_fn: Callable[[Any, dict[str, Any]], Any]
    memory_service: MemoryService | None; cache_key_fn: Callable[[Any], Any] | None
    close_fn: Callable[[], Any] | None; own_handle: bool; epoch: int = 0
    cache: OrderedDict[tuple[str, str, str], Any] = field(default_factory=OrderedDict)
    cache_hits: int = 0; cache_misses: int = 0; error_count: int = 0


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return value


def _validate_int(value: Any, name: str, *, min_value: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _normalize_gpus(gpus: Sequence[int]) -> tuple[int, ...]:
    if isinstance(gpus, (str, bytes, bytearray)):
        raise TypeError(f"gpus must be a sequence of ints, not {type(gpus).__name__}")
    for gpu in gpus:
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise ValueError(f"GPU ordinal must be a non-negative int, got {gpu!r}")
    return tuple(gpus)


def _validate_memory_service(mem: Any) -> MemoryService:
    for method in ("get_handle", "acquire_lease", "release_lease"):
        if not callable(getattr(mem, method, None)):
            raise TypeError(f"memory_service must implement {method}()")
    return mem


def _canonicalize(val: Any) -> Any:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return ["bool", val]
    if isinstance(val, int):
        return ["int", val]
    if isinstance(val, float):
        if not math.isfinite(val):
            raise AuxInvocationError(f"non-finite float in payload: {val!r}")
        return ["float", repr(val)]
    if isinstance(val, str):
        return ["str", val]
    if isinstance(val, (bytes, bytearray, memoryview)):
        return ["bytes", hashlib.sha256(bytes(val)).hexdigest()]
    if isinstance(val, PurePath):
        return ["path", os.fspath(val)]
    if is_dataclass(val) and not isinstance(val, type):
        return ["dataclass", type(val).__qualname__, [[f.name, _canonicalize(getattr(val, f.name))] for f in fields(val)]]
    if isinstance(val, list):
        return ["list", [_canonicalize(x) for x in val]]
    if isinstance(val, tuple):
        return ["tuple", [_canonicalize(x) for x in val]]
    if isinstance(val, (set, frozenset)):
        items = [_canonicalize(x) for x in val]
        items.sort(key=lambda c: json.dumps(c, separators=(",", ":"), ensure_ascii=True))
        return ["set", items]
    if isinstance(val, dict):
        pairs = [[_canonicalize(k), _canonicalize(v)] for k, v in val.items()]
        pairs.sort(key=lambda kv: json.dumps(kv[0], separators=(",", ":"), ensure_ascii=True))
        return ["dict", pairs]
    raise AuxInvocationError(f"unsupported payload type for canonical digest: {type(val).__qualname__}")


def _compute_digest(payload: Any, cache_key_fn: Callable[[Any], Any] | None) -> str:
    key_obj = payload
    if cache_key_fn is not None:
        try:
            key_obj = cache_key_fn(payload)
        except Exception as exc:
            raise AuxInvocationError(f"cache_key_fn failed: {type(exc).__name__}: {exc}") from exc
    encoded = json.dumps(_canonicalize(key_obj), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_spec(spec: AuxStageSpec) -> AuxStageSpec:
    return replace(spec, metadata=copy.deepcopy(spec.metadata))


def _verify_ro_handle(mem: MemoryService, handle_id: str) -> MemoryHandle:
    handle = mem.get_handle(handle_id)
    if not handle.read_only or handle.pool_kind not in _ALLOWED_AUX_POOLS:
        allowed = sorted(p.value for p in _ALLOWED_AUX_POOLS)
        raise AuxServiceError(
            f"auxiliary stage requires read-only handle in {allowed}; "
            f"got {handle_id!r} (pool_kind={handle.pool_kind.value!r}, read_only={handle.read_only})"
        )
    return handle


def _release_lease_quiet(mem: MemoryService | None, lease: SnapshotLease | None) -> None:
    if lease is not None and mem is not None:
        with contextlib.suppress(Exception):
            mem.release_lease(lease.lease_id)


def _close_fn_key(fn: Callable[[], Any]) -> tuple[int, int]:
    return (id(getattr(fn, "__self__", None)), id(getattr(fn, "__func__", fn)))


class AuxiliaryModelService:
    """Thread-safe S4 pool for frozen auxiliary stages with RO leases and LRU caching."""

    def __init__(self, *, memory_service: MemoryService | None = None, default_cache_entries: int = 256) -> None:
        self._memory_service = _validate_memory_service(memory_service) if memory_service is not None else None
        self._default_cache_entries = _validate_int(default_cache_entries, "default_cache_entries", min_value=0)
        self._lock = threading.RLock()
        self._teardown_done = threading.Event()
        self._closed = self._torn_down = False
        self._stages: dict[str, _StageEntry] = {}
        self._total_hits = self._total_misses = self._total_errors = 0

    def __enter__(self) -> AuxiliaryModelService:
        with self._lock:
            self._require_open_locked()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _require_open_locked(self) -> None:
        if self._closed:
            raise AuxClosedError("AuxiliaryModelService is closed")

    def _get_entry_locked(self, stage_id: str) -> _StageEntry:
        entry = self._stages.get(stage_id)
        if entry is None:
            raise StageNotFoundError(f"unknown stage {stage_id!r}")
        return entry

    def register_stage(
        self, stage_id: str, *, kind: AuxStageKind | str, engine: str, model_id: str, version: str = "v1",
        infer_fn: Callable[[Any, dict[str, Any]], Any], handle_id: str | None = None,
        memory_service: MemoryService | None = None, cache_key_fn: Callable[[Any], Any] | None = None,
        max_cache_entries: int | None = None, close_fn: Callable[[], Any] | None = None,
        own_handle: bool = False, gpus: Sequence[int] = (), metadata: dict[str, Any] | None = None,
    ) -> AuxStageSpec:
        """Register a frozen auxiliary stage after validating inputs and optional RO handle."""
        sid, resolved_kind = _require_str(stage_id, "stage_id"), AuxStageKind(kind)
        eng, mid, ver = _require_str(engine, "engine"), _require_str(model_id, "model_id"), _require_str(version, "version")
        if not callable(infer_fn):
            raise TypeError(f"infer_fn must be callable, not {type(infer_fn).__name__}")
        for fn, fname in ((cache_key_fn, "cache_key_fn"), (close_fn, "close_fn")):
            if fn is not None and not callable(fn):
                raise TypeError(f"{fname} must be callable or None, not {type(fn).__name__}")
        _require_bool(own_handle, "own_handle")
        cache_cap = (self._default_cache_entries if max_cache_entries is None
                     else _validate_int(max_cache_entries, "max_cache_entries", min_value=0))
        norm_gpus = _normalize_gpus(gpus)
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict or None, not {type(metadata).__name__}")
        meta = copy.deepcopy(metadata) if metadata is not None else {}
        mem = _validate_memory_service(memory_service) if memory_service is not None else self._memory_service
        clean_hid = _require_str(handle_id, "handle_id") if handle_id is not None else None
        if clean_hid is not None and mem is None:
            raise AuxServiceError("handle_id requires a memory_service")
        if own_handle and (clean_hid is None or mem is None):
            raise AuxServiceError("own_handle=True requires both handle_id and memory_service")

        with self._lock:
            self._require_open_locked()
            if sid in self._stages:
                raise StageExistsError(f"stage {sid!r} is already registered")
        if clean_hid is not None and mem is not None:
            handle = _verify_ro_handle(mem, clean_hid)
            if not norm_gpus:
                norm_gpus = handle.gpus

        spec = AuxStageSpec(sid, resolved_kind, eng, mid, ver, clean_hid, norm_gpus, True, cache_cap, meta)
        with self._lock:
            self._require_open_locked()
            if sid in self._stages:
                raise StageExistsError(f"stage {sid!r} is already registered")
            self._stages[sid] = _StageEntry(spec, infer_fn, mem, cache_key_fn, close_fn, own_handle)
            return _snapshot_spec(spec)

    def invoke(
        self, stage_id: str, caller_id: str, payload: Any, *, use_cache: bool = True,
        expected_stamp: dict[str, Any] | None = None, hold_lease: bool = False,
    ) -> AuxInvocationReceipt:
        """Invoke a frozen auxiliary stage under an RO weight lease with bounded LRU caching."""
        sid, caller = _require_str(stage_id, "stage_id"), _require_str(caller_id, "caller_id")
        _require_bool(use_cache, "use_cache")
        _require_bool(hold_lease, "hold_lease")
        if expected_stamp is not None and not isinstance(expected_stamp, dict):
            raise TypeError(f"expected_stamp must be a dict or None, not {type(expected_stamp).__name__}")

        started = time.perf_counter()
        lease: SnapshotLease | None = None
        mem: MemoryService | None = None
        keep_lease = False
        try:
            for _attempt in range(_MAX_INVOKE_RETRIES):
                with self._lock:
                    self._require_open_locked()
                    entry = self._get_entry_locked(sid)
                    spec, infer_fn, cache_key_fn, mem = (
                        _snapshot_spec(entry.spec), entry.infer_fn, entry.cache_key_fn, entry.memory_service
                    )
                    start_epoch = entry.epoch
                if expected_stamp is not None and (mem is None or spec.handle_id is None):
                    raise AuxServiceError(f"stage {sid!r} has no memory_service/handle_id for expected_stamp check")
                if hold_lease and (mem is None or spec.handle_id is None):
                    raise AuxServiceError(f"stage {sid!r} requires memory_service and handle_id when hold_lease=True")

                digest = _compute_digest(payload, cache_key_fn)
                if mem is not None and spec.handle_id is not None:
                    lease = mem.acquire_lease(spec.handle_id, caller, writable=False, expected_stamp=expected_stamp)

                closed_now = not_found = stale = False
                with self._lock:
                    if self._closed:
                        closed_now = True
                    else:
                        cur = self._stages.get(sid)
                        if cur is None:
                            not_found = True
                        elif (cur is not entry or cur.epoch != start_epoch
                                or cur.spec.version != spec.version or cur.spec.handle_id != spec.handle_id):
                            stale = True
                        elif use_cache and cur.spec.max_cache_entries > 0:
                            key = (sid, spec.version, digest)
                            if key in cur.cache:
                                cur.cache.move_to_end(key)
                                out = cur.cache[key]
                                cur.cache_hits += 1
                                self._total_hits += 1
                                keep_lease = hold_lease
                                return AuxInvocationReceipt(
                                    sid, spec.kind, spec.version, caller, digest, True,
                                    round(time.perf_counter() - started, 6),
                                    lease.lease_id if lease is not None else None, out,
                                )
                if closed_now:
                    _release_lease_quiet(mem, lease)
                    lease = None
                    self._teardown_done.wait(timeout=5.0)
                    raise AuxClosedError("AuxiliaryModelService is closed")
                if not_found:
                    _release_lease_quiet(mem, lease)
                    lease = None
                    raise StageNotFoundError(f"unknown stage {sid!r}")
                if stale:
                    _release_lease_quiet(mem, lease)
                    lease = None
                    continue
                break
            else:
                raise AuxServiceError(f"stage {sid!r} version/handle changed repeatedly during invoke")

            lease_id = lease.lease_id if lease is not None else None
            context = {
                "stage_id": sid, "kind": spec.kind.value, "engine": spec.engine, "model_id": spec.model_id,
                "version": spec.version, "caller_id": caller, "handle_id": spec.handle_id, "lease_id": lease_id,
                "gpus": spec.gpus, "input_digest": digest, "metadata": copy.deepcopy(spec.metadata),
            }
            try:
                output = infer_fn(payload, context)
            except Exception as exc:
                closed_on_err = False
                with self._lock:
                    if self._closed:
                        closed_on_err = True
                    else:
                        self._total_errors += 1
                        cur_err = self._stages.get(sid)
                        if cur_err is not None and cur_err is entry:
                            cur_err.error_count += 1
                if closed_on_err:
                    _release_lease_quiet(mem, lease)
                    lease = None
                    self._teardown_done.wait(timeout=5.0)
                    raise AuxClosedError("AuxiliaryModelService is closed") from exc
                if isinstance(exc, AuxServiceError):
                    raise
                raise AuxInvocationError(f"stage {sid!r} invocation failed: {type(exc).__name__}: {exc}") from exc

            cost_s = round(time.perf_counter() - started, 6)
            closed_after = False
            with self._lock:
                if self._closed:
                    closed_after = True
                else:
                    cur = self._stages.get(sid)
                    if cur is not None and cur is entry and cur.epoch == start_epoch:
                        self._total_misses += 1
                        cur.cache_misses += 1
                        if use_cache and cur.spec.max_cache_entries > 0:
                            key = (sid, spec.version, digest)
                            cur.cache[key] = output
                            cur.cache.move_to_end(key)
                            while len(cur.cache) > cur.spec.max_cache_entries:
                                cur.cache.popitem(last=False)
                        keep_lease = hold_lease
                    else:
                        keep_lease = hold_lease and (cur is not None and cur is entry)
            if closed_after:
                _release_lease_quiet(mem, lease)
                lease = None
                self._teardown_done.wait(timeout=5.0)
                raise AuxClosedError("AuxiliaryModelService is closed")
            return AuxInvocationReceipt(sid, spec.kind, spec.version, caller, digest, False, cost_s, lease_id, output)
        finally:
            if not keep_lease:
                _release_lease_quiet(mem, lease)

    def bump_version(self, stage_id: str, new_version: str, *, new_handle_id: str | None = None) -> AuxStageSpec:
        """Atomically update stage version (and optional RO handle), bump epoch, and invalidate cache."""
        sid, ver = _require_str(stage_id, "stage_id"), _require_str(new_version, "new_version")
        clean_hid = _require_str(new_handle_id, "new_handle_id") if new_handle_id is not None else None
        with self._lock:
            self._require_open_locked()
            orig_entry = self._get_entry_locked(sid)
            mem = orig_entry.memory_service
        if clean_hid is not None and mem is None:
            raise AuxServiceError("new_handle_id requires a memory_service")
        resolved_gpus = _verify_ro_handle(mem, clean_hid).gpus if (clean_hid is not None and mem is not None) else None
        with self._lock:
            self._require_open_locked()
            entry = self._get_entry_locked(sid)
            if entry is not orig_entry:
                raise StageNotFoundError(f"stage {sid!r} was replaced during bump_version")
            target_hid = clean_hid if clean_hid is not None else entry.spec.handle_id
            target_gpus = resolved_gpus if resolved_gpus is not None else entry.spec.gpus
            entry.epoch += 1
            entry.cache.clear()
            entry.spec = replace(entry.spec, version=ver, handle_id=target_hid, gpus=target_gpus)
            return _snapshot_spec(entry.spec)

    def unregister_stage(self, stage_id: str) -> AuxStageSpec:
        """Remove a stage, clear its cache, and clean up `close_fn` / stage-owned `memory_service`."""
        sid = _require_str(stage_id, "stage_id")
        with self._lock:
            self._require_open_locked()
            entry = self._stages.pop(sid, None)
            if entry is None:
                raise StageNotFoundError(f"unknown stage {sid!r}")
            entry.cache.clear()
            spec, close_fn, mem = _snapshot_spec(entry.spec), entry.close_fn, entry.memory_service
            should_teardown_mem = (
                entry.own_handle and mem is not None and mem is not self._memory_service
                and not any(e.memory_service is mem for e in self._stages.values())
            )
        try:
            if close_fn is not None:
                close_fn()
        finally:
            if should_teardown_mem and mem is not None and callable(getattr(mem, "teardown", None)):
                mem.teardown()
        return spec

    def list_stages(self, *, kind: AuxStageKind | str | None = None) -> tuple[AuxStageSpec, ...]:
        """Return immutable snapshots of registered stages, optionally filtered by `kind`."""
        resolved_kind = AuxStageKind(kind) if kind is not None else None
        with self._lock:
            return tuple(
                _snapshot_spec(e.spec) for e in self._stages.values()
                if resolved_kind is None or e.spec.kind == resolved_kind
            )

    def get_stage(self, stage_id: str) -> AuxStageSpec:
        """Return an immutable `AuxStageSpec` snapshot for `stage_id`."""
        sid = _require_str(stage_id, "stage_id")
        with self._lock:
            self._require_open_locked()
            return _snapshot_spec(self._get_entry_locked(sid).spec)

    def stats(self) -> dict[str, Any]:
        """Return service-wide and per-stage cache/invocation statistics."""
        with self._lock:
            stages_out = {
                sid: {
                    "kind": e.spec.kind.value, "engine": e.spec.engine, "model_id": e.spec.model_id,
                    "version": e.spec.version, "handle_id": e.spec.handle_id, "cache_entries": len(e.cache),
                    "max_cache_entries": e.spec.max_cache_entries, "cache_hits": e.cache_hits,
                    "cache_misses": e.cache_misses, "error_count": e.error_count,
                }
                for sid, e in self._stages.items()
            }
            return {
                "closed": self._closed, "stage_count": len(self._stages),
                "total_cache_entries": sum(len(e.cache) for e in self._stages.values()),
                "cache_hits": self._total_hits, "cache_misses": self._total_misses,
                "error_count": self._total_errors, "stages": stages_out,
            }

    def teardown(self) -> dict[str, Any]:
        """Idempotently close the service, clear caches, and run deduped cleanup outside locks."""
        with self._lock:
            already = self._torn_down
            if not already:
                self._closed = self._torn_down = True
                rows = [(sid, e.close_fn, e.memory_service if e.own_handle else None, len(e.cache))
                        for sid, e in self._stages.items()]
                for e in self._stages.values():
                    e.cache.clear()
                self._stages.clear()
            else:
                rows = []
        if already:
            self._teardown_done.wait(timeout=5.0)
            return {"already_closed": True, "closed_stages": [], "cleared_cache_entries": 0, "errors": []}

        closed_stages, cleared_entries, errors = [], sum(cnt for _, _, _, cnt in rows), []
        seen_close_fns: set[tuple[int, int]] = set()
        seen_mem_ids: set[int] = set()
        try:
            for sid, close_fn, stage_mem, _ in rows:
                closed_stages.append(sid)
                if close_fn is not None:
                    ckey = _close_fn_key(close_fn)
                    if ckey not in seen_close_fns:
                        seen_close_fns.add(ckey)
                        try:
                            close_fn()
                        except Exception as exc:
                            errors.append(f"{sid}: {type(exc).__name__}: {exc}")
                if stage_mem is not None and stage_mem is not self._memory_service and id(stage_mem) not in seen_mem_ids:
                    seen_mem_ids.add(id(stage_mem))
                    if callable(getattr(stage_mem, "teardown", None)):
                        try:
                            stage_mem.teardown()
                        except Exception as exc:
                            errors.append(f"{sid}.memory_service: {type(exc).__name__}: {exc}")
        finally:
            self._teardown_done.set()
        return {"already_closed": False, "closed_stages": closed_stages, "cleared_cache_entries": cleared_entries, "errors": errors}

    def close(self) -> None:
        """Close the service idempotently, releasing all stage resources."""
        self.teardown()


__all__ = [
    "AuxClosedError", "AuxInvocationError", "AuxInvocationReceipt", "AuxServiceError",
    "AuxStageKind", "AuxStageSpec", "AuxiliaryModelService", "StageExistsError", "StageNotFoundError",
]
