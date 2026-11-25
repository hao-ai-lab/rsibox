"""ToyTransformersLab: `TransformersLab` on the CPU `toy_hf` engine (`adapter.fixture.toy_hf`)."""
from __future__ import annotations

from typing import Any

from adapter.fixture.patches import toy_patches
from adapter.lab import TransformersLab

FIXTURE_ROOT = "adapter.fixture.toy_hf"
LOGITS_MODULE = f"{FIXTURE_ROOT}.logits_process"
GENERATION_MODULE = f"{FIXTURE_ROOT}.generation"
MODELING_MODULE = f"{FIXTURE_ROOT}.modeling"
TOY_WORKLOAD: dict[str, Any] = {
    "input_ids": [[5, 9, 2, 7, 1], [5, 9, 2, 7, 3], [11, 3, 6]],
    "max_new_tokens": 6, "temperature": 0.0,
}


class ToyTransformersLab(TransformersLab):
    """CPU conformance lab running `TransformersLab` over the `toy_hf` generation package."""

    name, package = "transformers-toy", "adapter"

    def __init__(
        self, receipts: str | None = None, *, device_map: str | dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        defaults = (("rpc_timeout", 5.0), ("check_timeout", 1.5), ("startup_timeout", 15.0), ("quiesce_timeout", 10.0))
        for key, value in defaults:
            kwargs.setdefault(key, value)
        config: dict[str, Any] = {"mode": "toy", "vocab_size": 64, "seed": 0}
        if device_map is not None:
            config["device_map"] = device_map
        super().__init__(
            config,
            (LOGITS_MODULE, GENERATION_MODULE, MODELING_MODULE), receipts=receipts, **kwargs,
        )

    def _engine_version(self) -> str:
        return "toy-1"

    def default_workload(self) -> dict[str, Any]:
        return dict(TOY_WORKLOAD)

    def default_warmup_workload(self) -> dict[str, Any]:
        return {"input_ids": [[5, 9, 2]], "max_new_tokens": 2, "temperature": 0.0}

    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        if fault == "hang":
            kwargs.setdefault("hang_s", self.check_timeout + 1.0)
        if fault == "tamper":
            kwargs.setdefault("module", GENERATION_MODULE)
            kwargs.setdefault("qualname", "argmax_token")
        return super().inject(fault, **kwargs)

    def patches(self) -> dict[str, Any]:
        return toy_patches(self._base_source, LOGITS_MODULE, GENERATION_MODULE, MODELING_MODULE)
