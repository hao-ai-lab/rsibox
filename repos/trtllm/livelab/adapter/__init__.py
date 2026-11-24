"""NVIDIA TensorRT-LLM (PyTorch backend) adapter for the RSIBox transaction kernel."""
import os, sys

LIVELAB = os.path.abspath(
    os.environ.get("RSIBOX_LIVELAB")
    or os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab")
)
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)

from .snapshot_hooks import TrtllmQuiesceHooks  # noqa: E402

__all__ = ["LIVELAB", "TrtllmQuiesceHooks"]
