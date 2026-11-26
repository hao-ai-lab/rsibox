"""Stage 1 (Thinker): project frozen Vision+LM backbone features into semantic conditioning."""
from __future__ import annotations
from typing import Any


def thinker_stage(backbone: dict[str, Any], cfg: Any) -> Any:
    """Project Thinker backbone logits using configured startup stage scale."""
    logits = backbone["logits"]
    return logits.float() * cfg.stage_scale if hasattr(logits, "shape") else [round(float(v) * cfg.stage_scale, 8) for v in logits]
