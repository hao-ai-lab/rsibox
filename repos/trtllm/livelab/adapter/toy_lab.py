"""CPU `ToyTrtllmLab` running `TrtllmLab` on spawned `ToyPyExecutor` subprocesses (11/11 conformance)."""
from __future__ import annotations

from typing import Any

from adapter.fixture.patches import TOY_EXECUTOR_MODULE, TOY_SAMPLER_MODULE, build_toy_patches
from adapter.lab import TrtllmLab

TOY_WORKLOAD: dict[str, Any] = {
    "prompts": [[5, 9, 2, 7, 1], [5, 9, 2, 7, 3], [11, 3]],
    "max_tokens": 6,
}


class ToyTrtllmLab(TrtllmLab):
    """Two-rank CPU conformance lab over spawned TensorRT-LLM-shaped worker processes."""

    name = "trtllm-toy"
    engine_mode = "toy"
    overlay_package = "adapter"
    default_tamper_module = TOY_SAMPLER_MODULE
    default_tamper_function = "greedy_search_sampling_batch"

    def __init__(self, ranks: int = 2, receipts: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("rpc_timeout", 5.0)
        kwargs.setdefault("check_timeout", 0.6)
        kwargs.setdefault("boot_timeout", 15.0)
        kwargs.setdefault("fresh_timeout", 30.0)
        kwargs.setdefault("quiesce_timeout", 10.0)
        super().__init__(
            model="toy-trtllm",
            tensor_parallel_size=ranks,
            modules=(TOY_SAMPLER_MODULE, TOY_EXECUTOR_MODULE),
            workload=dict(TOY_WORKLOAD),
            receipts=receipts,
            **kwargs,
        )

    def patches(self) -> dict[str, Any]:
        return build_toy_patches(self._base_source)
