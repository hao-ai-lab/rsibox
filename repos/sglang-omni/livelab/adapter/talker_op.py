"""Stage 2 (Talker): convert Thinker conditioning into prosody-modulated acoustic tokens."""
from __future__ import annotations
from typing import Any


def talker_prosody(val: float, gain: float) -> float:
    """Apply bounded prosody gain curve (CPU path)."""
    return val * gain / (1.0 + 0.1 * abs(val))


def talker_stage(thinker_cond: Any, cfg: Any) -> Any:
    """Convert Thinker conditioning into prosody-modulated acoustic tokens."""
    prosody_gain = 1.0
    norm_fn = lambda val: talker_prosody(val, prosody_gain)
    if hasattr(thinker_cond, "shape"):
        import torch  # type: ignore[import-not-found]
        x = thinker_cond.float() * prosody_gain
        return x / (1.0 + 0.1 * torch.abs(x))
    return [round(norm_fn(float(v)), 8) for v in thinker_cond]
