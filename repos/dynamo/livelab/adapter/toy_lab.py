"""ToyDynamoLab: CPU conformance lab for NVIDIA Dynamo (Frontend + Planner + Worker)."""
from __future__ import annotations
from typing import Any
from adapter.lab import DynamoLab

class ToyDynamoLab(DynamoLab):
    """Pure-Python CPU conformance lab running `DynamoLab` over toy worker subprocesses."""
    name, package = "dynamo-toy", "adapter"
    def __init__(self, receipts: str | None = None, **kwargs: Any) -> None:
        for k, v in (("rpc_timeout", 5.0), ("check_timeout", 3.0), ("startup_timeout", 15.0)): kwargs.setdefault(k, v)
        super().__init__({"mode": "toy", "num_workers": int(kwargs.pop("num_workers", 1)), "vocab_size": 128, "max_new_tokens": 6, "frontend_mode": "python"}, receipts=receipts, **kwargs)
