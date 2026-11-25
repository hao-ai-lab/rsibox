"""Hot-swappable SGLang rollout generation and weight-refit helpers for NeMo-RL."""
from __future__ import annotations

import math
from typing import Any


def generate_sglang_group(
    prompts: list[int],
    weights: list[float],
    refit_version: int,
    step_idx: int,
    group_size: int,
    kv_cache: dict[str, float],
    rng: Any,
) -> list[dict[str, Any]]:
    """Simulate SGLang grouped rollout generation conditioned on `weights` and `refit_version`."""
    w_mean = sum(weights) / max(len(weights), 1)
    groups: list[dict[str, Any]] = []
    for p in prompts:
        samples: list[dict[str, Any]] = []
        for g in range(group_size):
            cache_key = f"{p}:{refit_version}:{g}"
            if cache_key in kv_cache:
                logit = kv_cache[cache_key]
            else:
                jitter = (rng.random() - 0.5) * 1e-4
                logit = math.tanh(w_mean * (0.3 * (p + 1) + 0.1 * (g + 1)) + 0.05 * step_idx) + jitter
                kv_cache[cache_key] = logit
            old_lp = -math.log1p(math.exp(-logit))
            samples.append({
                "prompt": int(p),
                "sample_idx": int(g),
                "logit": round(logit, 8),
                "old_logprob": round(old_lp, 8),
            })
        groups.append({"prompt": int(p), "samples": samples})
    return groups


def apply_sglang_refit(
    rollout_state: dict[str, Any],
    new_weights: list[float],
    refit_version: int,
) -> dict[str, Any]:
    """Apply refit weights from AutoModel FSDP2 trainer to SGLang worker and flush KV cache."""
    rollout_state["weights"] = [float(w) for w in new_weights]
    rollout_state["refit_version"] = int(refit_version)
    rollout_state["kv_cache"].clear()
    return {
        "refit_version": int(refit_version),
        "weight_checksum": round(sum(rollout_state["weights"]), 8),
    }
