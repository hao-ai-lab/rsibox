"""ToyTransformersLab: `TransformersLab` on the CPU `toy_hf` engine (`adapter.fixture.toy_hf`)."""
from __future__ import annotations

from typing import Any

from adapter.lab import TransformersLab
from rsibox.harness import edit_once

FIXTURE_ROOT = "adapter.fixture.toy_hf"
LOGITS_MODULE, GENERATION_MODULE, MODELING_MODULE = f"{FIXTURE_ROOT}.logits_process", f"{FIXTURE_ROOT}.generation", f"{FIXTURE_ROOT}.modeling"
TOY_WORKLOAD: dict[str, Any] = {"input_ids": [[5, 9, 2, 7, 1], [5, 9, 2, 7, 3], [11, 3, 6]], "max_new_tokens": 6, "temperature": 0.0}


class ToyTransformersLab(TransformersLab):
    """CPU conformance lab running `TransformersLab` over the `toy_hf` generation package."""

    name, package = "transformers-toy", "adapter"

    def __init__(self, receipts: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("rpc_timeout", 5.0)
        kwargs.setdefault("check_timeout", 1.5)
        kwargs.setdefault("startup_timeout", 15.0)
        super().__init__({"mode": "toy", "vocab_size": 64, "seed": 0}, (LOGITS_MODULE, GENERATION_MODULE, MODELING_MODULE), receipts=receipts, **kwargs)

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
        logits_src, gen_src, model_src = self._base_source(LOGITS_MODULE), self._base_source(GENERATION_MODULE), self._base_source(MODELING_MODULE)
        norm_v1 = edit_once(model_src, "(value * 7 + 1) % MODULUS", "(value * 11 + 1) % MODULUS")
        norm_v2 = edit_once(norm_v1, "(value * 11 + 1) % MODULUS", "(value * 13 + 1) % MODULUS")
        argmax_v3 = edit_once(gen_src, "    return best_index\n", "    return (best_index + 1) % len(scores)\n")
        multi_logits = edit_once(edit_once(logits_src, "if temperature < TEMPERATURE_EPS:", "if float(temperature) < TEMPERATURE_EPS:"), "((index + 1 + step_bias) % 7)", "((index + 2 + step_bias) % 7)")
        return {
            "noop": {"files": {GENERATION_MODULE: edit_once(gen_src, "    best_index = 0\n", "    best_index = int(0)\n")}},
            "visible": {"files": {MODELING_MODULE: norm_v1}},
            "chain": [{"files": {MODELING_MODULE: norm_v1}}, {"files": {MODELING_MODULE: norm_v2}}, {"files": {GENERATION_MODULE: argmax_v3}}],
            "multi": {"files": {LOGITS_MODULE: multi_logits, GENERATION_MODULE: argmax_v3}},
            "unreached": {"files": {MODELING_MODULE: edit_once(model_src, "    def _init_weights(self) -> None:\n        self.vocab_offset = 0", "    def _init_weights(self) -> None:\n        self.vocab_offset = 3")}},
            "refused": {
                "syntax": {"files": {GENERATION_MODULE: edit_once(gen_src, "    return best_index\n", "    return (\n")}},
                "closure": {"files": {LOGITS_MODULE: edit_once(logits_src, "return index < limit", "return index < limit + offset")}},
                "module_state": {"files": {LOGITS_MODULE: edit_once(logits_src, "TEMPERATURE_EPS = 1e-6", "TEMPERATURE_EPS = 1e-5")}},
                "empty": {"files": {GENERATION_MODULE: edit_once(gen_src, "def argmax_token(scores: list[float]) -> int:", "def argmax_token(scores: list[float]) -> int:  # greedy")}},
            },
        }
