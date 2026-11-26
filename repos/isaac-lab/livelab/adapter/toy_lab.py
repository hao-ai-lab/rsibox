"""ToyIsaacLab: zero-dependency CPU conformance lab driving the real `IsaacLabLiveLab` adapter."""
from __future__ import annotations

from typing import Any

from adapter.lab import DEFAULT_MODULES, IsaacLabLiveLab


class ToyIsaacLab(IsaacLabLiveLab):
    """Pure-Python batched robotics simulation conformance lab (zero eager GPU/sim imports)."""

    name = "isaac-lab-toy"

    def __init__(self, receipts: str | None = None, **kwargs: Any) -> None:
        kw = {"rpc_timeout": 10.0, "check_timeout": 2.0, "startup_timeout": 15.0, **kwargs}
        super().__init__({"mode": "cpu", "num_envs": 64, "horizon_steps": 4, "seed": 42, "device": "cpu"}, DEFAULT_MODULES, receipts=receipts, **kw)
