"""Hot-swappable AutoModel FSDP2 policy loss, clipping, and optimizer helpers for NeMo-RL."""
from __future__ import annotations

import math
from typing import Any

INIT_POLICY_SCALE = 0.25
GRPO_LOSS_SCALE = 1.0


def init_policy_scale() -> float:
    """Startup-only policy weight scale (called only at S0 init; used by c10 zero_witness)."""
    return INIT_POLICY_SCALE


def make_kl_schedule(base_kl: float, warmup_steps: int = 2):
    """Return a step-indexed KL penalty closure (used by c6 closure-change refusal)."""
    floor_kl, decay = base_kl * 0.1, 0.5  # noqa: F841

    def kl_at(step: int) -> float:
        return base_kl * (step + 1) / max(warmup_steps, 1) if step < warmup_steps else max(floor_kl, base_kl)

    return kl_at


DEFAULT_KL_SCHEDULE = make_kl_schedule(0.01)


class NemoRlConfig:
    """Startup-only GRPO/FSDP2 configuration (editing `__init__` is refused)."""

    def __init__(
        self,
        hidden: int = 8,
        group_size: int = 4,
        lr: float = 0.05,
        clip_eps: float = 0.2,
        kl_coef: float = 0.01,
    ) -> None:
        self.hidden = int(hidden)
        self.group_size = int(group_size)
        self.lr = float(lr)
        self.clip_eps = float(clip_eps)
        self.kl_coef = float(kl_coef)
        self.init_scale = init_policy_scale()


def compute_grpo_loss(
    policy_logprobs: list[float],
    old_logprobs: list[float],
    ref_logprobs: list[float],
    advantages: list[float],
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> tuple[float, list[float]]:
    """Compute clipped GRPO surrogate loss + KL regularization and per-sample dL/d_logprob."""
    n = max(len(policy_logprobs), 1)
    total_loss = 0.0
    grads: list[float] = []
    for lp, old_lp, ref_lp, adv in zip(policy_logprobs, old_logprobs, ref_logprobs, advantages):
        ratio = math.exp(max(-10.0, min(10.0, float(lp) - float(old_lp))))
        clipped = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
        surr1 = ratio * float(adv)
        surr2 = clipped * float(adv)
        pg_loss = -min(surr1, surr2)
        kl_diff = float(ref_lp) - float(lp)
        kl_pen = kl_coef * (math.exp(max(-10.0, min(10.0, kl_diff))) - kl_diff - 1.0)
        total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE
        # Exact subgradient wrt policy_logprobs
        use_unclipped = (surr1 <= surr2) if adv >= 0.0 else (surr1 <= surr2 and ratio == clipped)
        d_pg = (-ratio * float(adv) if use_unclipped else 0.0)
        d_kl = kl_coef * (1.0 - math.exp(max(-10.0, min(10.0, kl_diff))))
        grads.append((d_pg + d_kl) / n)
    return total_loss / n, grads


def clip_and_step_adamw(
    weights: list[float],
    weight_grads: list[float],
    optim_state: dict[str, Any],
    lr: float,
    max_norm: float = 1.0,
) -> float:
    """Clip policy gradients and apply one AdamW step, mutating `weights` and `optim_state`."""
    grad_scale = 1.0
    total_norm = math.sqrt(sum(g * g for g in weight_grads))
    coef = (max_norm / total_norm if total_norm > max_norm and total_norm > 0.0 else 1.0) * grad_scale
    step_num = int(optim_state.get("adam_step", 0)) + 1
    optim_state["adam_step"] = step_num
    m_buf: list[float] = optim_state.setdefault("exp_avg", [0.0] * len(weights))
    v_buf: list[float] = optim_state.setdefault("exp_avg_sq", [0.0] * len(weights))
    beta1, beta2, eps, wd = 0.9, 0.999, 1e-8, 0.01
    bc1 = 1.0 - (beta1 ** step_num)
    bc2 = 1.0 - (beta2 ** step_num)
    for i, (w, g_raw) in enumerate(zip(weights, weight_grads)):
        g = g_raw * coef
        m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g
        v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * (g * g)
        m_hat = m_buf[i] / bc1
        v_hat = v_buf[i] / bc2
        weights[i] = w * (1.0 - lr * wd) - lr * (m_hat / (math.sqrt(v_hat) + eps))
    return total_norm * grad_scale
