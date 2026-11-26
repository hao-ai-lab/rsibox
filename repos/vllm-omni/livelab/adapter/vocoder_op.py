"""Stage 4 (Vocoder): synthesize deterministic waveform summary from DiT latents."""
from __future__ import annotations
import math
from typing import Any


def vocoder_stage(dit_latents: Any, vocoder_proj: Any, cfg: Any) -> list[float]:
    """Synthesize deterministic waveform summary from denoised DiT latents."""
    if hasattr(dit_latents, "shape"):
        import torch  # type: ignore[import-not-found]
        with torch.no_grad():
            return [round(float(v), 6) for v in torch.tanh(dit_latents.float() @ vocoder_proj.float()).detach().cpu().tolist()]
    return [round(math.tanh(float(val) * float(vocoder_proj[idx % len(vocoder_proj)])), 6) for idx, val in enumerate(dit_latents)]
