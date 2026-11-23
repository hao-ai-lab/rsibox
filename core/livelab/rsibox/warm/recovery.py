"""Process-level recovery drivers and engine quiesce hooks (stdlib only).

Separates two responsibilities that were previously tangled inside `BaseLab.reset("respawn")`:
1. `QuiesceHooks`: the minimal 3-step engine lifecycle contract (`warmup`, `quiesce`, `resume`)
   that lets any RSIBox engine cooperate with process snapshots, shadow failover, or clean respawn.
2. `RecoveryDriver`: how a poisoned or crashed worker group is brought back (`DefaultRespawnDriver`,
   `StandbyShadowDriver`, or `DynamoK8sSnapshotDriver` in `snapshot_contract.py`).
"""
from __future__ import annotations

import contextlib
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


class DefaultRespawnDriver:
    """Default respawn path: stop the worker group, optionally verify `WeightStore`, respawn."""

    name = "default_respawn"

    def respawn(
        self,
        lab: Any,
        store: WeightStore | None = None,
        *,
        reason: str = "reset_respawn",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        with contextlib.suppress(Exception):
            lab._stop_engine()
        store_probe = None
        if store is not None and store.name != "none":
            store_probe = store.probe()
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
        replenish_fn: Callable[[Any, WeightStore | None], None] | None = None,
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
        t0 = time.perf_counter()
        with contextlib.suppress(Exception):
            lab._stop_engine()
        promoted = self._promote_fn(lab, store) or {}
        if self._replenish_fn is not None:
            self._replenish_fn(lab, store)
        ok = bool(promoted.get("ok", True)) and not lab.session.poisoned()
        return {
            "ok": ok,
            "reset": list(promoted.get("reset", ["shadow promoted", "session"])),
            "driver": self.name,
            "reason": reason,
            "cost_s": round(time.perf_counter() - t0, 6),
            **{k: v for k, v in promoted.items() if k not in ("ok", "reset", "cost_s")},
        }
