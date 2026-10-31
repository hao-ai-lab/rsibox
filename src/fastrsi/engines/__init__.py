"""Engine plugins."""

import importlib

ENGINES = {"toy": "fastrsi.engines.toy:Toy", "vllm": "fastrsi.engines.vllm:VLLM",
           "sglang": "fastrsi.engines.sglang:SGLang"}


def load(name):
    if name not in ENGINES:
        raise SystemExit(f"unknown engine {name}; available: {', '.join(sorted(ENGINES))}")
    module, attribute = ENGINES[name].split(":")
    return getattr(importlib.import_module(module), attribute)
