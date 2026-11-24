"""Conformance patch builders for real TensorRT-LLM (PyTorch backend) and the CPU toy fixture."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rsibox import harness

SAMPLER_MODULE = "tensorrt_llm._torch.pyexecutor.sampler"
PY_EXECUTOR_MODULE = "tensorrt_llm._torch.pyexecutor.py_executor"
MODEL_ENGINE_MODULE = "tensorrt_llm._torch.pyexecutor.model_engine"

TOY_SAMPLER_MODULE = "adapter.fixture.sampler"
TOY_EXECUTOR_MODULE = "adapter.fixture.py_executor"

_FAST_GREEDY_ANCHOR = (
    "            logits = self._apply_embedding_bias(logits, requests)\n"
    "            next_tokens = torch.argmax(logits, dim=-1)"
)
_LOAD_WEIGHTS_ANCHOR = (
    "                self._call_load_weights(model.load_weights, weights,\n"
    "                                        weight_mapper)"
)
_FORWARD_GENEXPR_ANCHOR = (
    "            gather_context_logits = any(\n"
    "                a.py_return_context_logits\n"
    "                for a in scheduled_requests.context_requests)"
)


def build_real_patches(read_base: Callable[[str], str]) -> dict[str, Any]:
    """Build real TensorRT-LLM `1.0.0rc6` conformance patches (`c0–c10`)."""
    sampler_src = read_base(SAMPLER_MODULE)
    executor_src = read_base(PY_EXECUTOR_MODULE)
    model_engine_src = read_base(MODEL_ENGINE_MODULE)

    noop_sampler = harness.edit_once(
        sampler_src,
        _FAST_GREEDY_ANCHOR,
        "            logits = self._apply_embedding_bias(logits, requests)\n"
        "            chosen_tokens = torch.argmax(logits, dim=-1)\n"
        "            next_tokens = chosen_tokens",
    )
    visible_one = harness.edit_once(
        sampler_src,
        _FAST_GREEDY_ANCHOR,
        "            logits = self._apply_embedding_bias(logits, requests)\n"
        "            next_tokens = (torch.argmax(logits, dim=-1) + 1) % logits.shape[-1]",
    )
    visible_two = harness.edit_once(
        visible_one,
        "            next_tokens = (torch.argmax(logits, dim=-1) + 1) % logits.shape[-1]",
        "            next_tokens = (torch.argmax(logits, dim=-1) + 2) % logits.shape[-1]",
    )
    chain_three_executor = harness.edit_once(
        executor_src,
        "                return self.sampler.sample_async(scheduled_batch, batch_outputs)",
        "                flipped_outputs = {**batch_outputs, 'logits': batch_outputs['logits'].flip(-1)}\n"
        "                return self.sampler.sample_async(scheduled_batch, flipped_outputs)",
    )
    multi_sampler = harness.edit_once(
        visible_one,
        '        vocab_size = model_outputs["logits"].shape[-1]',
        '        vocab_size = int(model_outputs["logits"].shape[-1])',
    )
    multi_executor = harness.edit_once(
        executor_src,
        "                return self.sampler.sample_async(scheduled_batch, batch_outputs)",
        "                sampled_state = self.sampler.sample_async(scheduled_batch, batch_outputs)\n"
        "                return sampled_state",
    )
    unreached_model_engine = harness.edit_once(
        model_engine_src,
        _LOAD_WEIGHTS_ANCHOR,
        _LOAD_WEIGHTS_ANCHOR
        + "\n                for param in model.parameters():\n"
        "                    param.data = param.data.neg().contiguous()\n"
        "                    break",
    )
    closure_refused = harness.edit_once(
        executor_src,
        _FORWARD_GENEXPR_ANCHOR,
        "            gather_context_logits = any(\n"
        "                a.py_return_context_logits and self.active\n"
        "                for a in scheduled_requests.context_requests)",
    )

    return {
        "noop": {"files": {SAMPLER_MODULE: noop_sampler}},
        "visible": {"files": {SAMPLER_MODULE: visible_one}},
        "chain": [
            {"files": {SAMPLER_MODULE: visible_one}},
            {"files": {SAMPLER_MODULE: visible_two}},
            {"files": {PY_EXECUTOR_MODULE: chain_three_executor}},
        ],
        "multi": {
            "files": {
                SAMPLER_MODULE: multi_sampler,
                PY_EXECUTOR_MODULE: multi_executor,
            }
        },
        "unreached": {"files": {MODEL_ENGINE_MODULE: unreached_model_engine}},
        "refused": {
            "syntax": {
                "files": {
                    SAMPLER_MODULE: harness.edit_once(
                        sampler_src,
                        _FAST_GREEDY_ANCHOR,
                        "            logits = self._apply_embedding_bias(logits, requests)\n"
                        "            next_tokens = torch.argmax(",
                    )
                }
            },
            "closure": {"files": {PY_EXECUTOR_MODULE: closure_refused}},
            "module_state": {
                "files": {
                    SAMPLER_MODULE: harness.edit_once(
                        sampler_src,
                        'GREEDY: Greedy = ("greedy", None)',
                        'GREEDY: Greedy = ("greedy", 0)',
                    )
                }
            },
            "empty": {
                "files": {
                    SAMPLER_MODULE: harness.edit_once(
                        sampler_src,
                        "def greedy_search_sampling_batch(logits):",
                        "def greedy_search_sampling_batch(logits):  # note",
                    )
                }
            },
        },
    }


def build_toy_patches(read_base: Callable[[str], str]) -> dict[str, Any]:
    """Build CPU fixture conformance patches (`c0–c10`)."""
    sampler_src = read_base(TOY_SAMPLER_MODULE)
    executor_src = read_base(TOY_EXECUTOR_MODULE)

    visible_one = harness.edit_once(
        sampler_src,
        "    return best_index",
        "    return (best_index + 1) % len(logits)",
    )
    visible_two = harness.edit_once(
        visible_one,
        "    return (best_index + 1) % len(logits)",
        "    return (best_index + 2) % len(logits)",
    )
    executor_edit = harness.edit_once(
        executor_src,
        "    return (token_id + step_index) % sampler.VOCAB_SIZE",
        "    return (token_id + step_index + 1) % sampler.VOCAB_SIZE",
    )
    multi_sampler = harness.edit_once(
        visible_one,
        "    seed_sum = sum(prompt_tokens) + step_index * 3 + startup_bias",
        "    seed_sum = sum(prompt_tokens) + step_index * 3 + startup_bias + 1",
    )

    return {
        "noop": {
            "files": {
                TOY_SAMPLER_MODULE: harness.edit_once(
                    sampler_src,
                    "    return best_index",
                    "    chosen = best_index\n    return chosen",
                )
            }
        },
        "visible": {"files": {TOY_SAMPLER_MODULE: visible_one}},
        "chain": [
            {"files": {TOY_SAMPLER_MODULE: visible_one}},
            {"files": {TOY_SAMPLER_MODULE: visible_two}},
            {"files": {TOY_EXECUTOR_MODULE: executor_edit}},
        ],
        "multi": {
            "files": {
                TOY_SAMPLER_MODULE: multi_sampler,
                TOY_EXECUTOR_MODULE: executor_edit,
            }
        },
        "unreached": {
            "files": {
                TOY_SAMPLER_MODULE: harness.edit_once(
                    sampler_src,
                    "def initial_logit_bias() -> int:\n    return 3",
                    "def initial_logit_bias() -> int:\n    return 5",
                )
            }
        },
        "refused": {
            "syntax": {
                "files": {
                    TOY_SAMPLER_MODULE: harness.edit_once(
                        sampler_src, "    return best_index", "    return ("
                    )
                }
            },
            "closure": {
                "files": {
                    TOY_SAMPLER_MODULE: harness.edit_once(
                        sampler_src,
                        "    return lambda score: score + offset",
                        "    return lambda score: score + offset + extra",
                    )
                }
            },
            "module_state": {
                "files": {
                    TOY_SAMPLER_MODULE: harness.edit_once(
                        sampler_src,
                        "LOGIT_OFFSET, VOCAB_SIZE = 1, 32",
                        "LOGIT_OFFSET, VOCAB_SIZE = 2, 32",
                    )
                }
            },
            "empty": {
                "files": {
                    TOY_SAMPLER_MODULE: harness.edit_once(
                        sampler_src,
                        "def greedy_search_sampling_batch(logits: list[int]) -> int:",
                        "def greedy_search_sampling_batch(logits: list[int]) -> int:  # note",
                    )
                }
            },
        },
    }
