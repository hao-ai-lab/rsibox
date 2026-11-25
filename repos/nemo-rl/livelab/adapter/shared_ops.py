"""Shared GRPO prompt/reward scoring helpers executed across rollout, coordinator, and trainer."""
from __future__ import annotations

REWARD_WEIGHT_SCALE = 1.0


def score_prompt_reward(prompt_id: int, refit_version: int, signal: float) -> float:
    """Compute shaped reward scalar for a prompt trajectory under `refit_version`."""
    base = 0.25 * float((int(prompt_id) % 5) + 1)
    return round((base + 0.1 * float(signal) + 0.02 * float(refit_version)) * REWARD_WEIGHT_SCALE, 8)
