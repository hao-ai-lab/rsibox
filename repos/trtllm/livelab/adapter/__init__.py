"""NVIDIA TensorRT-LLM (PyTorch backend) adapter for the RSIBox transaction kernel."""
import os, sys

LIVELAB = os.path.abspath(
    os.environ.get("RSIBOX_LIVELAB")
    or os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab")
)
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)

from .shadow import (  # noqa: E402
    DEFAULT_SHADOW_RELEASE_TAGS,
    ParkedTrtllmShadow,
    TrtllmShadowPromoter,
    park_shadow_lab,
)
from .snapshot_hooks import TrtllmQuiesceHooks  # noqa: E402

__all__ = [
    "DEFAULT_SHADOW_RELEASE_TAGS",
    "LIVELAB",
    "ParkedTrtllmShadow",
    "TrtllmQuiesceHooks",
    "TrtllmShadowPromoter",
    "park_shadow_lab",
]
