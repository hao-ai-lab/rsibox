"""Stage 3 (DiT): iterative flow-matching denoising over Talker acoustic tokens."""
from __future__ import annotations
from typing import Any


def dit_stage(talker_tokens: Any, cfg: Any) -> Any:
    """Run flow-matching denoising steps over Talker acoustic tokens."""
    steps = max(1, int(cfg.dit_steps))
    if hasattr(talker_tokens, "shape"):
        latents = talker_tokens.clone()
        for step_idx in range(steps):
            alpha = 0.5 / (step_idx + 1)
            latents = latents * (1.0 - alpha) + alpha * talker_tokens.roll(shifts=1, dims=-1)
        return latents
    latents, n = list(talker_tokens), len(talker_tokens)
    for step_idx in range(steps):
        alpha = 0.5 / (step_idx + 1)
        latents = [round(latents[i] * (1.0 - alpha) + alpha * talker_tokens[(i - 1) % n], 8) for i in range(n)]
    return latents
