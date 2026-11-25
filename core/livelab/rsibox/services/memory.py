"""S1: Unified GPU Memory & State Service (pure Python stdlib).

Unifies inference RO/KV pools (`WeightStore`, Dynamo GMS, CUDA IPC), training
writable/CoW `S0` baseline snapshots (`weights`, `adam_m`, `adam_v`, `rng`), and
RL RO auxiliary weights + Actor<->Rollout handle handoff behind a thread-safe,
fail-closed lease and staleness-stamp boundary.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field, replace
from enum import StrEnum
import threading
from typing import Any, Sequence

from ..warm.types import StoreProbe, stamp_diff
from ..warm.weight_store import DiskStore, WeightStore

DEFAULT_S0_STATE_KEYS: tuple[str, ...] = ("weights", "adam_m", "adam_v", "rng")


class MemoryPoolKind(StrEnum):
    """Canonical GPU memory pool roles managed by `MemoryService` (S1)."""

    RO_WEIGHTS = "ro_weights"
    KV_POOL = "kv_pool"
    COW_S0_SNAPSHOT = "cow_s0_snapshot"
    RO_AUX_WEIGHTS = "ro_aux_weights"
    ACTOR_ROLLOUT_SHARED = "actor_rollout_shared"


READ_ONLY_POOLS: frozenset[MemoryPoolKind] = frozenset({MemoryPoolKind.RO_WEIGHTS, MemoryPoolKind.RO_AUX_WEIGHTS})


class MemoryServiceError(RuntimeError): """Base error raised by `MemoryService` on validation or lease errors."""
class StampMismatchError(MemoryServiceError): """Raised when a handle or store stamp fails `stamp_diff` validation."""
class PoolAccessError(MemoryServiceError): """Raised on RO violation, lock conflict, closed service, or store failure."""
class HandleNotFoundError(MemoryServiceError): """Raised when referencing an unknown or unregistered `handle_id`."""
class LeaseError(PoolAccessError): """Raised when referencing an unknown, released, or stale `SnapshotLease`."""


@dataclass(frozen=True)
class MemoryHandle:
    """Immutable descriptor for a registered GPU memory region or shared weight pool."""

    handle_id: str
    pool_kind: MemoryPoolKind
    engine: str
    role: str
    gpus: tuple[int, ...]
    size_bytes: int
    stamp: dict[str, Any] = field(default_factory=dict)
    ipc_token: str | None = None
    gms_handle: str | None = None
    read_only: bool = True
    incarnation: int = 1
    refcount: int = 0


@dataclass(frozen=True)
class SnapshotLease:
    """Fenced reader/writer lease bound to a `MemoryHandle` and snapshot epoch."""

    lease_id: str
    handle_id: str
    holder_id: str
    epoch: int
    writable: bool
    epoch_state_keys: tuple[str, ...] = ()


@dataclass
class _LockGroup:
    active_leases: dict[str, SnapshotLease] = field(default_factory=dict)
    writer_lease_id: str | None = None
    epoch: int = 0
    incarnation: int = 1


@dataclass
class _HandleEntry:
    handle: MemoryHandle
    lock_group: _LockGroup
    store: WeightStore | None = None
    owned_store: bool = False
    state_keys: tuple[str, ...] = ()


def _copy_stamp(stamp: dict[str, Any] | None) -> dict[str, Any]:
    if stamp is None:
        return {}
    if not isinstance(stamp, dict):
        raise TypeError(f"stamp must be a dict or None, not {type(stamp).__name__}")
    return copy.deepcopy(stamp)


def _normalize_gpus(gpus: Sequence[int]) -> tuple[int, ...]:
    if isinstance(gpus, (str, bytes, bytearray)):
        raise TypeError(f"gpus must be a sequence of ints, not {type(gpus).__name__}")
    for g in gpus:
        if isinstance(g, bool) or not isinstance(g, int) or g < 0:
            raise ValueError(f"GPU ordinal must be a non-negative int, got {g!r}")
    return tuple(gpus)


def _require_str(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _validate_int(value: int, field_name: str, *, min_value: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, not {type(value).__name__}")
    if value < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}, got {value}")
    return value


def _normalize_state_keys(kind: MemoryPoolKind, state_keys: Sequence[str] | None) -> tuple[str, ...]:
    if state_keys is None:
        return DEFAULT_S0_STATE_KEYS if kind == MemoryPoolKind.COW_S0_SNAPSHOT else ()
    if isinstance(state_keys, (str, bytes, bytearray)):
        raise TypeError("state_keys must be a sequence of strings")
    return tuple(_require_str(k, "state_key") for k in state_keys)


class MemoryService:
    """Thread-safe S1 coordinator for GPU memory handles, CoW S0 epochs, and leases."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handles: dict[str, _HandleEntry] = {}
        self._leases: dict[str, SnapshotLease] = {}
        self._handle_seq = 0
        self._lease_seq = 0
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise PoolAccessError("MemoryService is closed")

    def _resolve_pool_and_ro(self, pool_kind: MemoryPoolKind | str, read_only: bool | None) -> tuple[MemoryPoolKind, bool]:
        kind = MemoryPoolKind(pool_kind)
        if kind in READ_ONLY_POOLS:
            if read_only is False:
                raise PoolAccessError(f"pool {kind.value!r} is strictly read-only; read_only=False refused")
            return kind, True
        if kind == MemoryPoolKind.COW_S0_SNAPSHOT:
            return kind, (False if read_only is None else bool(read_only))
        return kind, (True if read_only is None else bool(read_only))

    def _verify_stamp(self, expected: dict[str, Any] | None, actual: dict[str, Any], context: str) -> None:
        if expected is None:
            return
        if not isinstance(expected, dict):
            raise TypeError(f"expected_stamp must be a dict or None, not {type(expected).__name__}")
        diff = stamp_diff(expected, actual)
        if diff:
            raise StampMismatchError(f"stamp mismatch for {context}: {', '.join(diff)}")

    def _get_entry(self, handle_id: str) -> _HandleEntry:
        entry = self._handles.get(handle_id)
        if entry is None:
            raise HandleNotFoundError(f"unknown handle_id {handle_id!r}")
        return entry

    def _snapshot(self, entry: _HandleEntry) -> MemoryHandle:
        return replace(entry.handle, stamp=_copy_stamp(entry.handle.stamp), incarnation=entry.lock_group.incarnation)

    def _add_handle(
        self, kind: MemoryPoolKind, engine: str, role: str, gpus: tuple[int, ...], size_bytes: int,
        stamp_dict: dict[str, Any], *, handle_id: str | None = None, ipc_token: str | None = None,
        gms_handle: str | None = None, read_only: bool = True, incarnation: int = 1,
        state_keys: tuple[str, ...] = (), store: WeightStore | None = None, owned_store: bool = False,
        lock_group: _LockGroup | None = None,
    ) -> MemoryHandle:
        inc = _validate_int(incarnation, "incarnation", min_value=1)
        if handle_id is not None:
            _require_str(handle_id, "handle_id")
            if handle_id in self._handles:
                raise PoolAccessError(f"handle {handle_id!r} is already registered")
            resolved_id = handle_id
        else:
            while True:
                self._handle_seq += 1
                resolved_id = f"{kind.value}:{engine}:{role}-{self._handle_seq}"
                if resolved_id not in self._handles:
                    break
        group = lock_group if lock_group is not None else _LockGroup(incarnation=inc)
        handle = MemoryHandle(
            resolved_id, kind, engine, role, gpus, size_bytes, stamp_dict,
            ipc_token, gms_handle, read_only, group.incarnation, 0,
        )
        entry = _HandleEntry(handle=handle, lock_group=group, store=store, owned_store=owned_store, state_keys=state_keys)
        self._handles[resolved_id] = entry
        return self._snapshot(entry)

    def register_region(
        self, pool_kind: MemoryPoolKind | str, engine: str, role: str, gpus: Sequence[int] = (),
        size_bytes: int = 0, stamp: dict[str, Any] | None = None, *, handle_id: str | None = None,
        expected_stamp: dict[str, Any] | None = None, ipc_token: str | None = None,
        gms_handle: str | None = None, read_only: bool | None = None, incarnation: int = 1,
        state_keys: Sequence[str] | None = None,
    ) -> MemoryHandle:
        """Register a GPU memory region after validating size, RO policy, and stamp."""
        with self._lock:
            self._require_open()
            kind, effective_ro = self._resolve_pool_and_ro(pool_kind, read_only)
            stamp_dict = _copy_stamp(stamp)
            self._verify_stamp(expected_stamp, stamp_dict, kind.value)
            return self._add_handle(
                kind, _require_str(engine, "engine"), _require_str(role, "role"),
                _normalize_gpus(gpus), _validate_int(size_bytes, "size_bytes", min_value=1),
                stamp_dict, handle_id=handle_id, ipc_token=ipc_token, gms_handle=gms_handle,
                read_only=effective_ro, incarnation=incarnation, state_keys=_normalize_state_keys(kind, state_keys),
            )

    def attach_store(
        self, pool_kind: MemoryPoolKind | str, store: WeightStore, *, engine: str = "generic",
        role: str = "weights", gpus: Sequence[int] = (), size_bytes: int = 0,
        stamp: dict[str, Any] | None = None, expected_stamp: dict[str, Any] | None = None,
        handle_id: str | None = None, ipc_token: str | None = None, gms_handle: str | None = None,
        read_only: bool | None = None, own_store: bool = True, ensure_ready: bool = False,
        timeout_s: float = 30.0, state_keys: Sequence[str] | None = None,
    ) -> MemoryHandle:
        """Attach a `WeightStore` (e.g. `DiskStore` / `DynamoGmsStore`) as a managed handle."""
        if not isinstance(store, WeightStore):
            raise TypeError(f"store must implement WeightStore, not {type(store).__name__}")
        with self._lock:
            self._require_open()
            kind, effective_ro = self._resolve_pool_and_ro(pool_kind, read_only)
            clean_engine, clean_role = _require_str(engine, "engine"), _require_str(role, "role")
            norm_gpus = _normalize_gpus(gpus)
            norm_size = _validate_int(size_bytes, "size_bytes", min_value=0 if (isinstance(store, DiskStore) or size_bytes == 0) else 1)
            norm_keys = _normalize_state_keys(kind, state_keys)
            in_stamp = _copy_stamp(stamp) if stamp is not None else None
            exp_stamp = _copy_stamp(expected_stamp) if expected_stamp is not None else None
            if handle_id is not None:
                _require_str(handle_id, "handle_id")
                if handle_id in self._handles:
                    raise PoolAccessError(f"handle {handle_id!r} is already registered")

        gpu_strs = tuple(str(g) for g in norm_gpus)
        store_expected = exp_stamp if exp_stamp is not None else in_stamp
        started_owned = False
        try:
            if ensure_ready:
                started_owned = bool(own_store)
                ready = store.ensure_ready(gpus=gpu_strs, timeout_s=timeout_s)
                if not ready.ok:
                    raise PoolAccessError(ready.error or f"store {store.name!r} ensure_ready failed")
            probe: StoreProbe = store.probe(gpus=gpu_strs, expected_stamp=store_expected)
            if not probe.ok:
                if probe.stale_keys:
                    raise StampMismatchError(f"store {store.name!r} stamp stale: {', '.join(probe.stale_keys)}")
                raise PoolAccessError(probe.error or f"store {store.name!r} probe failed")

            details = probe.details if isinstance(probe.details, dict) else {}
            raw_stamp = (
                details["stamp"] if isinstance(details.get("stamp"), dict)
                else (store.read_stamp() if callable(getattr(store, "read_stamp", None)) else None)  # type: ignore[attr-defined]
            )
            probe_stamp = _copy_stamp(raw_stamp) if isinstance(raw_stamp, dict) else None
            if probe_stamp is not None and in_stamp is not None:
                self._verify_stamp(in_stamp, probe_stamp, f"store {store.name!r}")
            eff_stamp = probe_stamp if probe_stamp is not None else (in_stamp if in_stamp is not None else {})
            self._verify_stamp(exp_stamp, eff_stamp, f"store {store.name!r}")

            detail_size = details.get("size_bytes")
            resolved_size = detail_size if (norm_size == 0 and isinstance(detail_size, int) and not isinstance(detail_size, bool) and detail_size > 0) else norm_size
            sock = probe.sockets[0] if probe.sockets else None
            with self._lock:
                self._require_open()
                return self._add_handle(
                    kind, clean_engine, clean_role, norm_gpus, resolved_size, eff_stamp, handle_id=handle_id,
                    ipc_token=ipc_token or (sock if sock and store.name != "dynamo_gms" else None),
                    gms_handle=gms_handle or (sock if sock and store.name == "dynamo_gms" else None),
                    read_only=effective_ro, state_keys=norm_keys, store=store, owned_store=bool(own_store),
                )
        except BaseException:
            if started_owned:
                with contextlib.suppress(Exception):
                    store.stop()
            raise

    def get_handle(self, handle_id: str) -> MemoryHandle:
        """Return an immutable snapshot of `handle_id`."""
        with self._lock:
            return self._snapshot(self._get_entry(handle_id))

    def list_handles(self, pool_kind: MemoryPoolKind | str | None = None, *, role: str | None = None) -> tuple[MemoryHandle, ...]:
        """Return immutable snapshots of matching registered handles."""
        kind = MemoryPoolKind(pool_kind) if pool_kind is not None else None
        with self._lock:
            return tuple(
                self._snapshot(e) for e in self._handles.values()
                if (kind is None or e.handle.pool_kind == kind) and (role is None or e.handle.role == role)
            )

    def acquire_lease(
        self, handle_id: str, holder_id: str, writable: bool = False,
        *, expected_stamp: dict[str, Any] | None = None,
    ) -> SnapshotLease:
        """Acquire a fenced reader or single-writer lease on `handle_id`."""
        clean_holder = _require_str(holder_id, "holder_id")
        with self._lock:
            self._require_open()
            entry = self._get_entry(handle_id)
            self._verify_stamp(expected_stamp, entry.handle.stamp, f"handle {handle_id!r}")
            if writable and (entry.handle.pool_kind in READ_ONLY_POOLS or entry.handle.read_only):
                raise PoolAccessError(f"refused writable lease on read-only handle {handle_id!r} ({entry.handle.pool_kind.value})")
            lg = entry.lock_group
            if lg.writer_lease_id is not None:
                raise PoolAccessError(f"handle {handle_id!r} already has active writer lease {lg.writer_lease_id!r}")
            if writable and lg.active_leases:
                raise PoolAccessError(f"cannot acquire writable lease on {handle_id!r} with {len(lg.active_leases)} active reader(s)")

            self._lease_seq += 1
            lease_id = f"lease-{self._lease_seq}"
            lease = SnapshotLease(lease_id, handle_id, clean_holder, lg.epoch, bool(writable), entry.state_keys)
            self._leases[lease_id] = lease
            lg.active_leases[lease_id] = lease
            if writable:
                lg.writer_lease_id = lease_id
            entry.handle = replace(entry.handle, refcount=entry.handle.refcount + 1)
            return lease

    def restore_s0(
        self, lease_id: str, expected_epoch: int | None = None, *, expected_stamp: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Restore baseline `S0` state under a valid writable CoW S0 lease and advance epoch."""
        if expected_epoch is not None:
            _validate_int(expected_epoch, "expected_epoch", min_value=0)
        with self._lock:
            self._require_open()
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseError(f"unknown or released lease_id {lease_id!r}")
            entry = self._handles.get(lease.handle_id)
            if entry is None:
                raise HandleNotFoundError(f"lease {lease_id!r} references missing handle {lease.handle_id!r}")
            if entry.handle.pool_kind != MemoryPoolKind.COW_S0_SNAPSHOT:
                raise PoolAccessError(f"restore_s0 requires {MemoryPoolKind.COW_S0_SNAPSHOT.value!r}, got {entry.handle.pool_kind.value!r}")
            lg = entry.lock_group
            if not lease.writable or lg.writer_lease_id != lease_id:
                raise PoolAccessError(f"restore_s0 requires active writable lease, got {lease_id!r}")
            self._verify_stamp(expected_stamp, entry.handle.stamp, f"handle {lease.handle_id!r}")
            if lease.epoch != lg.epoch or (expected_epoch is not None and lg.epoch != expected_epoch):
                raise LeaseError(f"stale S0 epoch for {lease.handle_id!r}: lease={lease.epoch}, current={lg.epoch}, expected={expected_epoch}")

            prev_epoch = lg.epoch
            lg.epoch += 1
            updated = replace(lease, epoch=lg.epoch)
            self._leases[lease_id] = updated
            lg.active_leases[lease_id] = updated
            return {
                "ok": True, "handle_id": lease.handle_id, "lease_id": lease_id, "previous_epoch": prev_epoch,
                "epoch": lg.epoch, "restored_keys": list(entry.state_keys),
                "incarnation": lg.incarnation, "stamp": _copy_stamp(entry.handle.stamp),
            }

    def release_lease(self, lease_id: str) -> bool:
        """Idempotently release `lease_id` and decrement the handle's refcount."""
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return False
            entry = self._handles.get(lease.handle_id)
            if entry is not None:
                lg = entry.lock_group
                lg.active_leases.pop(lease_id, None)
                if lg.writer_lease_id == lease_id:
                    lg.writer_lease_id = None
                entry.handle = replace(entry.handle, refcount=max(0, entry.handle.refcount - 1))
            return True

    def holder_died(self, holder_id: str) -> tuple[str, ...]:
        """Clean up all orphan leases held by `holder_id`."""
        clean_holder = _require_str(holder_id, "holder_id")
        with self._lock:
            orphans = [lid for lid, lease in self._leases.items() if lease.holder_id == clean_holder]
            for lid in orphans:
                self.release_lease(lid)
            return tuple(orphans)

    def share_handle(
        self, handle_id: str, target_role: str, *, target_engine: str | None = None,
        read_only: bool | None = None, bump_incarnation: bool = False,
    ) -> MemoryHandle:
        """Hand off a handle across roles (e.g. Actor <-> Rollout), preserving stamp & incarnation."""
        clean_role = _require_str(target_role, "target_role")
        with self._lock:
            self._require_open()
            entry = self._get_entry(handle_id)
            lg = entry.lock_group
            if lg.writer_lease_id is not None:
                raise PoolAccessError(f"cannot share handle {handle_id!r} while writer lease {lg.writer_lease_id!r} is active")
            src = entry.handle
            ro_forced = src.read_only or src.pool_kind in READ_ONLY_POOLS or src.pool_kind == MemoryPoolKind.COW_S0_SNAPSHOT
            if read_only is False and ro_forced:
                raise PoolAccessError(f"cannot share {src.pool_kind.value!r} handle {handle_id!r} with read_only=False")
            target_kind = MemoryPoolKind.ACTOR_ROLLOUT_SHARED if src.pool_kind == MemoryPoolKind.COW_S0_SNAPSHOT else src.pool_kind
            target_ro = True if (ro_forced or read_only is None) else bool(read_only)
            if bump_incarnation:
                lg.incarnation += 1
            return self._add_handle(
                target_kind, _require_str(target_engine, "target_engine") if target_engine is not None else src.engine,
                clean_role, src.gpus, src.size_bytes, _copy_stamp(src.stamp),
                ipc_token=src.ipc_token, gms_handle=src.gms_handle, read_only=target_ro,
                incarnation=lg.incarnation, state_keys=entry.state_keys,
                store=entry.store, owned_store=False, lock_group=lg,
            )

    def probe_all(self, *, expected_stamp: dict[str, Any] | None = None) -> dict[str, Any]:
        """Probe all registered handles and attached stores for liveness and stamp health."""
        with self._lock:
            if self._closed:
                return {"ok": False, "closed": True, "handles": {}, "active_leases": 0, "owned_stores": 0}
            snap = [(hid, self._snapshot(e), e.lock_group.epoch, e.store, e.owned_store) for hid, e in self._handles.items()]
            active_leases = len(self._leases)

        results: dict[str, dict[str, Any]] = {}
        all_ok, owned_ids = True, set()
        for hid, h, epoch, store, owned in snap:
            if owned and store is not None:
                owned_ids.add(id(store))
            stale = list(stamp_diff(expected_stamp, h.stamp)) if expected_stamp is not None else []
            err: str | None = None
            ok = not stale
            if store is not None:
                store_expected = expected_stamp if expected_stamp is not None else (h.stamp or None)
                probe = store.probe(gpus=tuple(str(g) for g in h.gpus), expected_stamp=store_expected)
                if not probe.ok:
                    ok = False
                    stale.extend(k for k in probe.stale_keys if k not in stale)
                    err = probe.error
            all_ok = all_ok and ok
            results[hid] = {
                "ok": ok, "pool_kind": h.pool_kind.value, "role": h.role, "refcount": h.refcount,
                "incarnation": h.incarnation, "epoch": epoch, "owned_store": owned, "stale_keys": stale, "error": err,
            }
        return {"ok": all_ok, "closed": False, "handles": results, "active_leases": active_leases, "owned_stores": len(owned_ids)}

    def teardown(self) -> dict[str, Any]:
        """Stop only stores owned by this `MemoryService`; never touch foreign handles."""
        with self._lock:
            if self._closed:
                return {"already_closed": True, "stopped_stores": [], "skipped_foreign_stores": [], "released_leases": 0, "errors": []}
            self._closed = True
            released_leases = len(self._leases)
            self._leases.clear()
            owned_by_id: dict[int, WeightStore] = {}
            foreign_by_id: dict[int, WeightStore] = {}
            for entry in self._handles.values():
                entry.lock_group.active_leases.clear()
                entry.lock_group.writer_lease_id = None
                entry.handle = replace(entry.handle, refcount=0)
                if entry.store is not None:
                    (owned_by_id if entry.owned_store else foreign_by_id).setdefault(id(entry.store), entry.store)
            self._handles.clear()
            foreign = [s for sid, s in foreign_by_id.items() if sid not in owned_by_id]

        stopped, errors = [], []
        for store in owned_by_id.values():
            try:
                store.stop()
                stopped.append(store.name)
            except Exception as exc:
                errors.append(f"{store.name}: {type(exc).__name__}: {exc}")
        return {
            "already_closed": False, "stopped_stores": stopped,
            "skipped_foreign_stores": [s.name for s in foreign], "released_leases": released_leases, "errors": errors,
        }
