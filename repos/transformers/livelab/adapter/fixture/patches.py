"""Conformance patch suites for `TransformersLab` (real) and `ToyTransformersLab` (CPU)."""
from __future__ import annotations

from typing import Any, Callable

from rsibox.harness import edit_once

DEFAULT_REAL_MODULES = (
    "transformers.generation.logits_process",
    "transformers.generation.utils",
    "transformers.modeling_utils",
)
ARGMAX_ANCHOR = "next_tokens = torch.argmax(next_token_scores, dim=-1)"
GEN_CONFIG_ANCHOR = "        generation_config_provided = generation_config is not None"
CLOSURE_ANCHOR = "            if token is None:\n                return token"
LOGGER_ANCHOR = "logger = logging.get_logger(__name__)"
LOGITS_RETURN_ANCHOR = (
    "            else:\n"
    "                scores = processor(input_ids, scores)\n\n"
    "        return scores\n"
)
TIE_WEIGHTS_ANCHOR = (
    "        if not recompute_mapping:\n"
    "            tied_keys = self.all_tied_weights_keys"
)


def real_patches(read_base: Callable[[str], str]) -> dict[str, Any]:
    """Named conformance patch suite (`c1`–`c10`) against real HuggingFace `transformers`."""
    logits_module, utils_module, modeling_module = DEFAULT_REAL_MODULES
    logits_source = read_base(logits_module)
    utils_source = read_base(utils_module)
    modeling_source = read_base(modeling_module)

    visible_v1 = edit_once(
        utils_source,
        ARGMAX_ANCHOR,
        "next_tokens = (torch.argmax(next_token_scores, dim=-1) + 1) % next_token_scores.shape[-1]",
    )
    visible_v2 = edit_once(
        utils_source,
        ARGMAX_ANCHOR,
        "next_tokens = (torch.argmax(next_token_scores, dim=-1) + 2) % next_token_scores.shape[-1]",
    )
    logits_flip = edit_once(
        logits_source,
        LOGITS_RETURN_ANCHOR,
        LOGITS_RETURN_ANCHOR.replace("return scores\n", "return scores.flip(-1)\n"),
    )
    multi_utils = edit_once(
        visible_v1,
        GEN_CONFIG_ANCHOR,
        "        generation_config_provided = bool(generation_config is not None)",
    )
    unreached_modeling = edit_once(
        modeling_source,
        TIE_WEIGHTS_ANCHOR,
        "        for param in self.parameters():\n"
        "            param.data = param.data.neg().contiguous()\n"
        "            break\n" + TIE_WEIGHTS_ANCHOR,
    )
    return {
        "noop": {"files": {utils_module: edit_once(utils_source, ARGMAX_ANCHOR, ARGMAX_ANCHOR + ".clone()")}},
        "visible": {"files": {utils_module: visible_v1}},
        "chain": [
            {"files": {utils_module: visible_v1}},
            {"files": {utils_module: visible_v2}},
            {"files": {logits_module: logits_flip}},
        ],
        "multi": {"files": {utils_module: multi_utils, logits_module: logits_flip}},
        "unreached": {"files": {modeling_module: unreached_modeling}},
        "refused": {
            "syntax": {"files": {utils_module: edit_once(utils_source, ARGMAX_ANCHOR, "next_tokens = (")}},
            "closure": {"files": {utils_module: edit_once(
                utils_source,
                CLOSURE_ANCHOR,
                "            if token is None and batch_size is None:\n                return token",
            )}},
            "module_state": {"files": {utils_module: edit_once(
                utils_source, LOGGER_ANCHOR, 'logger = logging.get_logger(__name__ + ".edited")'
            )}},
            "empty": {"files": {utils_module: edit_once(utils_source, ARGMAX_ANCHOR, ARGMAX_ANCHOR + "  # greedy")}},
        },
    }


def toy_patches(
    read_base: Callable[[str], str],
    logits_module: str,
    generation_module: str,
    modeling_module: str,
) -> dict[str, Any]:
    """Named conformance patch suite (`c1`–`c10`) against `adapter.fixture.toy_hf`."""
    logits_source = read_base(logits_module)
    generation_source = read_base(generation_module)
    modeling_source = read_base(modeling_module)

    norm_v1 = edit_once(modeling_source, "(value * 7 + 1) % MODULUS", "(value * 11 + 1) % MODULUS")
    norm_v2 = edit_once(norm_v1, "(value * 11 + 1) % MODULUS", "(value * 13 + 1) % MODULUS")
    argmax_v3 = edit_once(
        generation_source, "    return best_index\n", "    return (best_index + 1) % len(scores)\n"
    )
    multi_logits = edit_once(
        edit_once(logits_source, "if temperature < TEMPERATURE_EPS:", "if float(temperature) < TEMPERATURE_EPS:"),
        "((index + 1 + step_bias) % 7)",
        "((index + 2 + step_bias) % 7)",
    )
    return {
        "noop": {"files": {generation_module: edit_once(generation_source, "    best_index = 0\n", "    best_index = int(0)\n")}},
        "visible": {"files": {modeling_module: norm_v1}},
        "chain": [
            {"files": {modeling_module: norm_v1}},
            {"files": {modeling_module: norm_v2}},
            {"files": {generation_module: argmax_v3}},
        ],
        "multi": {"files": {logits_module: multi_logits, generation_module: argmax_v3}},
        "unreached": {"files": {modeling_module: edit_once(
            modeling_source,
            "    def _init_weights(self) -> None:\n        self.vocab_offset = 0",
            "    def _init_weights(self) -> None:\n        self.vocab_offset = 3",
        )}},
        "refused": {
            "syntax": {"files": {generation_module: edit_once(generation_source, "    return best_index\n", "    return (\n")}},
            "closure": {"files": {logits_module: edit_once(logits_source, "return index < limit", "return index < limit + offset")}},
            "module_state": {"files": {logits_module: edit_once(logits_source, "TEMPERATURE_EPS = 1e-6", "TEMPERATURE_EPS = 1e-5")}},
            "empty": {"files": {generation_module: edit_once(
                generation_source,
                "def argmax_token(scores: list[float]) -> int:",
                "def argmax_token(scores: list[float]) -> int:  # greedy",
            )}},
        },
    }
