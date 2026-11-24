"""Hot-swappable gradient clipping and optimizer-step helpers for TorchTitan / FSDP2."""
from __future__ import annotations

import math
from typing import Any

GRAD_CLIP_NORM = 1.0


def make_lr_schedule(base_lr: float, warmup_steps: int = 2):
    """Return a step-indexed LR closure (used by c6 closure-change refusal check)."""
    floor_lr, decay = base_lr * 0.1, 0.5

    def lr_at(step: int) -> float:
        return base_lr * (step + 1) / max(warmup_steps, 1) if step < warmup_steps else max(floor_lr, base_lr)

    return lr_at


DEFAULT_LR_SCHEDULE = make_lr_schedule(0.01)


def clip_and_scale_grad(grads: Any, max_norm: float = GRAD_CLIP_NORM) -> float:
    """Compute global L2 grad norm and clip in-place (supports PyTorch params and CPU lists)."""
    grad_scale = 1.0
    if isinstance(grads, dict) and "params" in grads:
        import torch

        params = [p for p in grads["params"] if p.grad is not None]
        if not params:
            return 0.0
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if grad_scale != 1.0:
            for p in params:
                p.grad.mul_(grad_scale)
        return float((norm.item() if hasattr(norm, "item") else norm) * grad_scale)
    total_norm = math.sqrt(sum(val * val for mat in grads for row in mat for val in row))
    coef = (max_norm / total_norm if total_norm > max_norm and total_norm > 0.0 else 1.0) * grad_scale
    if coef != 1.0:
        for mat in grads:
            for row in mat:
                row[:] = [v * coef for v in row]
    return total_norm * grad_scale


def optimizer_step_scale(step_idx: int, base_lr: float) -> float:
    """Compute effective learning rate for `step_idx` before `optimizer.step()`."""
    return base_lr / (1.0 + 0.05 * step_idx)
