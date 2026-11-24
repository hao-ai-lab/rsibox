"""Process-level recovery drivers and engine quiesce hooks (stdlib only).

Separates two responsibilities that were previously tangled inside `BaseLab.reset("respawn")`:
1. `QuiesceHooks`: the minimal 3-step engine lifecycle contract (`warmup`, `quiesce`, `resume`)
   that lets any RSIBox engine cooperate with process snapshots, shadow failover, or clean respawn.
2. `RecoveryDriver`: how a poisoned or crashed worker group is brought back (`DefaultRespawnDriver`,
   `StandbyShadowDriver`, or `DynamoK8sSnapshotDriver` in `snapshot_contract.py`).
"""
from __future__ import annotations

import contextlib
import os
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
        promote_cost_s = round(time.perf_counter() - t0, 6)
        ok = bool(promoted.get("ok", True)) and not lab.session.poisoned()
        if ok and self._replenish_fn is not None:
            self._replenish_fn(lab, store)
        reserved = {"ok", "reset", "driver", "reason", "cost_s"}
        return {
            "ok": ok,
            "reset": list(promoted.get("reset", ["shadow promoted", "session"])) if ok else [],
            "driver": self.name,
            "reason": reason,
            "cost_s": promote_cost_s,
            **{k: v for k, v in promoted.items() if k not in reserved},
        }
