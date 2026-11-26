"""Shared startup config and lazy vLLM GPU backbone loader for vLLM-Omni."""
from __future__ import annotations
import os, sys
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


def build_gpu_stage_modules(cfg: OmniPipelineConfig, model_path: str, synth_device: str, *, seed: int = 0) -> tuple[Any, Any, Any]:
    """Lazy-load frozen vLLM Thinker engine and frozen Vocoder projection on CUDA."""
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(f"mode='real' requires existing model directory, got {model_path!r}")
    bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    if bin_dir and bin_dir not in (os.environ.get("PATH") or "").split(os.pathsep):
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]
    from vllm import LLM  # type: ignore[import-not-found]

    processor = transformers.AutoProcessor.from_pretrained(model_path)  # type: ignore[attr-defined]
    llm = LLM(model=model_path, enforce_eager=True, gpu_memory_utilization=0.15, max_model_len=512, seed=int(seed), enable_prefix_caching=False, disable_log_stats=True)
    gen = torch.Generator(device="cpu").manual_seed(42)
    vocoder_proj = (torch.randn(cfg.hidden_dim, cfg.hidden_dim, generator=gen, dtype=torch.float32) * 0.25).to(synth_device)
    vocoder_proj.requires_grad_(False)
    return llm, processor, vocoder_proj
