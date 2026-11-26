"""ToySglangOmniLab: zero-dependency 4-stage CPU conformance lab driving `SglangOmniLab`."""
from __future__ import annotations
from typing import Any
from adapter.lab import DEFAULT_MODULES, SglangOmniLab

class ToySglangOmniLab(SglangOmniLab):
    """4-stage CPU conformance lab (`thinker`, `talker`, `dit`, `vocoder`) with zero GPU imports."""
    name = "sglang-omni-toy"
    def __init__(self, receipts: str | None = None, **kwargs: Any) -> None:
        kwargs = {"rpc_timeout": 5.0, "check_timeout": 2.0, "startup_timeout": 15.0, **kwargs}
        super().__init__({"mode": "cpu", "hidden_dim": 8, "dit_steps": 2, "max_new_tokens": 8, "image_size": 64, "seed": 0}, DEFAULT_MODULES, receipts=receipts, **kwargs)
