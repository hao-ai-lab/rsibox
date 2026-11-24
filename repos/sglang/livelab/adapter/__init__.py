"""SGLang adapter for the rsibox transaction kernel (`core/livelab/kernel`)."""
import os
import sys

LIVELAB = os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)

from .shadow import (  # noqa: E402
    ParkedSglangShadow,
    SglangShadowPromoter,
    make_shadow_recovery_driver,
    park_shadow_lab,
    stop_sglang_engine_targeted,
)
from .snapshot_hooks import SglangQuiesceHooks  # noqa: E402

__all__ = [
    "LIVELAB",
    "ParkedSglangShadow",
    "SglangQuiesceHooks",
    "SglangShadowPromoter",
    "make_shadow_recovery_driver",
    "park_shadow_lab",
    "stop_sglang_engine_targeted",
]
