"""Pluggable `WeightStore` protocol and registry (stdlib only).

Each `WeightStore` is the single owner of out-of-process persistent GPU weight memory:
- probing whether resident weights/sockets are alive before launching an engine,
- checking optional source/checkpoint/env stamps so edited loaders never read stale weights,
- producing a `LaunchDelta` (CLI args + env + conflict findings) for an engine launcher,
- starting and cleanly stopping any daemon processes it owns.

RSIBox does not hard-bind to any single store: `DiskStore` (`"none"`), engine-native IPC
caches (`"ipc_cache"`), and optional NVIDIA Dynamo GMS (`"dynamo_gms"`) all share this
protocol.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from .types import LaunchDelta, StoreProbe


@runtime_checkable
class WeightStore(Protocol):
    """Single-owner abstraction for persistent out-of-process GPU weight residency."""

    name: str

    def ensure_ready(
        self,
        *,
        gpus: Sequence[str] = (),
        timeout_s: float = 300.0,
    ) -> StoreProbe:
        """Start or verify the backing weight daemon so workers can attach without disk I/O."""
        ...

    def probe(
        self,
        *,
        gpus: Sequence[str] = (),
        expected_stamp: dict[str, Any] | None = None,
    ) -> StoreProbe:
        """Fast preflight check for liveness and optional staleness stamp match."""
        ...

    def launch_delta(
        self,
        *,
        engine: str,
        explicit_flags: set[str] | frozenset[str] = frozenset(),
        explicit_values: dict[str, Any] | None = None,
        gpus: Sequence[str] = (),
    ) -> LaunchDelta:
        """Return CLI/env overrides needed for `engine` to attach to this store."""
        ...

    def stop(self) -> None:
        """Stop only daemon processes started by this store instance (never external sidecars)."""
        ...


class DiskStore:
    """No-op `WeightStore`: engine loads weights normally from local/network storage."""

    name = "none"

    def ensure_ready(
        self,
        *,
        gpus: Sequence[str] = (),
        timeout_s: float = 300.0,
    ) -> StoreProbe:
        return StoreProbe(ok=True, backend=self.name)

    def probe(
        self,
        *,
        gpus: Sequence[str] = (),
        expected_stamp: dict[str, Any] | None = None,
    ) -> StoreProbe:
        return StoreProbe(ok=True, backend=self.name)

    def launch_delta(
        self,
        *,
        engine: str,
        explicit_flags: set[str] | frozenset[str] = frozenset(),
        explicit_values: dict[str, Any] | None = None,
        gpus: Sequence[str] = (),
    ) -> LaunchDelta:
        return LaunchDelta()

    def stop(self) -> None:
        return None


_StoreFactory = Callable[..., WeightStore]
_REGISTRY: dict[str, _StoreFactory] = {
    "none": lambda **_kw: DiskStore(),
}


def register_weight_store(name: str, factory: _StoreFactory) -> None:
    """Register a pluggable `WeightStore` factory under `name`."""
    _REGISTRY[name] = factory


def list_weight_stores() -> tuple[str, ...]:
    """Return sorted names of all registered `WeightStore` backends."""
    return tuple(sorted(_REGISTRY))


def get_weight_store(name: str, **kwargs: Any) -> WeightStore:
    """Instantiate a registered `WeightStore` backend by name."""
    if name not in _REGISTRY:
        known = ", ".join(list_weight_stores())
        raise KeyError(f"unknown weight store backend {name!r} (available: {known})")
    return _REGISTRY[name](**kwargs)
