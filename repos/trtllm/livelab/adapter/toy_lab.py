"""CPU `ToyTrtllmLab` running `TrtllmLab` on spawned `ToyPyExecutor` subprocesses (11/11 conformance)."""
from __future__ import annotations

from typing import Any
from adapter.lab import TrtllmLab
from rsibox import harness

TOY_SAMPLER, TOY_EXECUTOR = "adapter.fixture.sampler", "adapter.fixture.py_executor"
TOY_WORKLOAD: dict[str, Any] = {"prompts": [[5, 9, 2, 7, 1], [5, 9, 2, 7, 3], [11, 3]], "max_tokens": 6}

class ToyTrtllmLab(TrtllmLab):
    """Two-rank CPU conformance lab over spawned TensorRT-LLM-shaped worker processes."""
    name, engine_mode, overlay_package = "trtllm-toy", "toy", "adapter"
    default_tamper_module, default_tamper_function = TOY_SAMPLER, "greedy_search_sampling_batch"

    def __init__(self, ranks: int = 2, receipts: str | None = None, **kwargs: Any) -> None:
        for key, default in (("rpc_timeout", 5.0), ("check_timeout", 0.6), ("boot_timeout", 15.0), ("fresh_timeout", 30.0)):
            kwargs.setdefault(key, default)
        super().__init__(model="toy-trtllm", tensor_parallel_size=ranks, modules=(TOY_SAMPLER, TOY_EXECUTOR),
                         workload=dict(TOY_WORKLOAD), receipts=receipts, **kwargs)

    def patches(self) -> dict[str, Any]:
        sampler_src, executor_src = self._base_source(TOY_SAMPLER), self._base_source(TOY_EXECUTOR)
        vis_one = harness.edit_once(sampler_src, "    return best_index", "    return (best_index + 1) % len(logits)")
        vis_two = harness.edit_once(vis_one, "    return (best_index + 1) % len(logits)", "    return (best_index + 2) % len(logits)")
        exec_edit = harness.edit_once(executor_src, "    return (token_id + step_index) % sampler.VOCAB_SIZE", "    return (token_id + step_index + 1) % sampler.VOCAB_SIZE")
        multi_sampler = harness.edit_once(vis_one, "    seed_sum = sum(prompt_tokens) + step_index * 3 + startup_bias", "    seed_sum = sum(prompt_tokens) + step_index * 3 + startup_bias + 1")
        return {
            "noop": {"files": {TOY_SAMPLER: harness.edit_once(sampler_src, "    return best_index", "    chosen = best_index\n    return chosen")}},
            "visible": {"files": {TOY_SAMPLER: vis_one}},
            "chain": [{"files": {TOY_SAMPLER: vis_one}}, {"files": {TOY_SAMPLER: vis_two}}, {"files": {TOY_EXECUTOR: exec_edit}}],
            "multi": {"files": {TOY_SAMPLER: multi_sampler, TOY_EXECUTOR: exec_edit}},
            "unreached": {"files": {TOY_SAMPLER: harness.edit_once(sampler_src, "def initial_logit_bias() -> int:\n    return 3", "def initial_logit_bias() -> int:\n    return 5")}},
            "refused": {
                "syntax": {"files": {TOY_SAMPLER: harness.edit_once(sampler_src, "    return best_index", "    return (")}},
                "closure": {"files": {TOY_SAMPLER: harness.edit_once(sampler_src, "    return lambda score: score + offset", "    return lambda score: score + offset + extra")}},
                "module_state": {"files": {TOY_SAMPLER: harness.edit_once(sampler_src, "LOGIT_OFFSET, VOCAB_SIZE = 1, 32", "LOGIT_OFFSET, VOCAB_SIZE = 2, 32")}},
                "empty": {"files": {TOY_SAMPLER: harness.edit_once(sampler_src, "def greedy_search_sampling_batch(logits: list[int]) -> int:", "def greedy_search_sampling_batch(logits: list[int]) -> int:  # note")}},
            },
        }
