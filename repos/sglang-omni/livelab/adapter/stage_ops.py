"""Shared startup config and lazy GPU backbone loader for SGLang-Omni."""
from __future__ import annotations
from typing import Any

INIT_STAGE_SCALE = 0.25


def init_stage_scale() -> float:
    """Startup-only stage gain (never called during pipeline execution; used by c10)."""
    return INIT_STAGE_SCALE


class OmniPipelineConfig:
    """Immutable stage config captured at worker startup (editing `__init__` is refused)."""

    def __init__(self, hidden_dim: int = 8, dit_steps: int = 2, max_new_tokens: int = 8, image_size: int = 64) -> None:
        self.hidden_dim, self.dit_steps = int(hidden_dim), int(dit_steps)
        self.max_new_tokens, self.image_size = int(max_new_tokens), int(image_size)
        self.stage_scale = init_stage_scale()


def build_gpu_stage_modules(cfg: OmniPipelineConfig, model_path: str, thinker_device: str, synth_device: str) -> tuple[Any, Any, Any]:
    """Lazy-load frozen Qwen3-VL Thinker backbone and frozen Vocoder projection on CUDA."""
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, device_map=thinker_device).eval()  # type: ignore[attr-defined]
    for param in model.parameters():
        param.requires_grad_(False)
    processor = transformers.AutoProcessor.from_pretrained(model_path)  # type: ignore[attr-defined]
    gen = torch.Generator(device="cpu").manual_seed(42)
    vocoder_proj = (torch.randn(cfg.hidden_dim, cfg.hidden_dim, generator=gen, dtype=torch.float32) * 0.25).to(synth_device)
    vocoder_proj.requires_grad_(False)
    return model, processor, vocoder_proj
