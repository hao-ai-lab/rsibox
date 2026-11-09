"""FastRSI live lab for Megatron-LM. The transaction kernel is the vLLM lab's
(`repos/vllm/livelab/kernel`, pure Python); it is put on sys.path here until it moves to the
framework (see ../README.md)."""
import os
import sys

_k = os.environ.get("FASTRSI_KERNEL") or os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "vllm", "livelab")
if os.path.isdir(os.path.join(_k, "kernel")) and _k not in sys.path:
    sys.path.insert(0, os.path.abspath(_k))
