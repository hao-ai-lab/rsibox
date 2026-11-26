"""Stage 2 (Talker): convert Thinker conditioning into prosody-modulated acoustic tokens."""
from __future__ import annotations
from typing import Any


def talker_prosody(val: Any, gain: float) -> Any:
    """Apply bounded prosody gain curve."""
    mag = val.abs() if hasattr(val, "shape") else abs(val)
    return (val.float() if hasattr(val, "shape") else val) * gain / (1.0 + 0.1 * mag)


def talker_stage(thinker_cond: Any, cfg: Any) -> Any:
    """Convert Thinker conditioning into prosody-modulated acoustic tokens."""
    prosody_gain = 1.0
    norm_fn = lambda val: talker_prosody(val, prosody_gain)
    if hasattr(thinker_cond, "shape"):
        return talker_prosody(thinker_cond, prosody_gain)
    return [round(norm_fn(float(v)), 8) for v in thinker_cond]
