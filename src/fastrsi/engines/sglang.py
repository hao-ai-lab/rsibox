"""SGLang. Host Python only: edits reach functions that run as ordinary Python."""

import os

from fastrsi.engine import Spec
from fastrsi.engines.container import Container

SAMPLER = "sglang/srt/layers/sampler.py"
GREEDY = "batch_next_token_ids = torch.argmax(logits, -1)"


class SGLang(Container):
    name = "sglang"
    image = "lmsysorg/sglang:latest"
    site = "/sgl-workspace/sglang/python"
    sources = (SAMPLER,)
    server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sglang_server.py")
    pins = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DO_NOT_TRACK": "1"}

    def spec(self):
        return Spec(
            noop=(SAMPLER, GREEDY, GREEDY + ".clone()"),
            visible=(SAMPLER, GREEDY,
                     "batch_next_token_ids = (torch.argmax(logits, -1) + 1) % logits.shape[-1]"),
            multiple=((SAMPLER, GREEDY, "batch_next_token_ids = torch.argmax(logits, -1) + 1"),
                      (SAMPLER, 'sanitize_nan_logits(logits, "sampler: next_token_logits")',
                       'sanitize_nan_logits(logits, "sampler: next token logits")')),
            refused={
                "signature": (SAMPLER, "self, logits: torch.Tensor, sampling_info: "
                              "SamplingBatchInfo\n    ) -> torch.Tensor:",
                              "self, logits: torch.Tensor, sampling_info: SamplingBatchInfo, "
                              "k=1\n    ) -> torch.Tensor:"),
                "new-definition": (SAMPLER, "class Sampler(",
                                   "def helper():\n    return 1\n\n\nclass Sampler("),
                "syntax": (SAMPLER, GREEDY, "batch_next_token_ids = ("),
            },
            chain=[(SAMPLER, GREEDY, f"batch_next_token_ids = (torch.argmax(logits, -1) + {n}) "
                    "% logits.shape[-1]") for n in range(1, 11)],
            crash=(SAMPLER, GREEDY, "__import__('os')._exit(3)"),
            hang=(SAMPLER, GREEDY, "__import__('time').sleep(3600)"),
            unreached=(SAMPLER, '"""Handle the full Ascend backend sampling path.',
                       '"""Handle the Ascend backend sampling path.'),
            witness_processes=1,
            exercise_timeout=60.0,
            exercise_args={"prompts": 8, "tokens": 32},
        )
