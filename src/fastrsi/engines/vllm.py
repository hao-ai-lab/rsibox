"""vLLM. Host Python only for now.

Edits reach functions that run as ordinary Python. A function inside a compiled graph is not
reached; the witness shows it did not run, so the edit never reads as proven.
"""

import os

from fastrsi.engine import Spec
from fastrsi.engines.container import Container

SAMPLER = "vllm/v1/worker/gpu/sample/sampler.py"
QWEN2 = "vllm/model_executor/models/qwen2.py"
RETURN = "return sampled, processed_logits"


class VLLM(Container):
    name = "vllm"
    image = "vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910"
    site = "/usr/local/lib/python3.12/dist-packages"
    sources = (SAMPLER, QWEN2)
    server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm_server.py")
    pins = {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE": "0", "VLLM_TRITON_FORCE_FIRST_CONFIG": "1",
            "VLLM_DISABLE_COMPILE_CACHE": "1", "VLLM_NO_USAGE_STATS": "1", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "DO_NOT_TRACK": "1"}

    def spec(self):
        return Spec(
            noop=(SAMPLER, RETURN, "return sampled.clone(), processed_logits"),
            visible=(SAMPLER, RETURN,
                     "return (sampled + 1) % processed_logits.shape[-1], processed_logits"),
            multiple=((SAMPLER, RETURN, "return sampled + 1, processed_logits"),
                      (SAMPLER, "(top_k is None and top_p is None)",
                       "(top_p is None and top_k is None)")),
            refused={
                "signature": (SAMPLER, "        use_flashinfer: bool,\n    ) -> tuple",
                              "        use_flashinfer: bool,\n        k: int = 1,\n    ) -> tuple"),
                "new-definition": (SAMPLER, "class Sampler:",
                                   "def helper():\n    return 1\n\n\nclass Sampler:"),
                "syntax": (SAMPLER, RETURN, "return ("),
            },
            chain=[(SAMPLER, RETURN, f"return (sampled + {n}) % processed_logits.shape[-1], "
                    "processed_logits") for n in range(1, 11)],
            crash=(SAMPLER, RETURN, "__import__('os')._exit(3)"),
            hang=(SAMPLER, RETURN, "__import__('time').sleep(3600)"),
            unreached=(QWEN2, "x = self.act_fn(gate_up)", "x = self.act_fn(gate_up) * 2"),
            witness_processes=1,
            exercise_timeout=60.0,
        )
