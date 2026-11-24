"""HuggingFace Transformers adapter for the RSIBox transaction kernel (`core/livelab`)."""
import os
import sys

LIVELAB = os.path.abspath(
    os.environ.get("RSIBOX_LIVELAB")
    or os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab")
)
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)

from .snapshot_hooks import TransformersQuiesceHooks  # noqa: E402

__all__ = ["LIVELAB", "TransformersQuiesceHooks"]
