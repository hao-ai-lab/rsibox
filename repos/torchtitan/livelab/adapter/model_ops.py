"""Hot-swappable model forward and loss ops for TorchTitan / FSDP2 training loops."""
from __future__ import annotations

import math
from typing import Any

INIT_WEIGHT_SCALE = 0.25


def init_weight_scale() -> float:
    """Startup-only weight init scale (never called during training steps; used by c10)."""
    return INIT_WEIGHT_SCALE

class TitanStepConfig:
    """Immutable step config captured at worker startup (editing `__init__` is refused)."""

    def __init__(self, hidden: int = 16, ffn: int = 32, layers: int = 2, lr: float = 0.01) -> None:
        self.hidden, self.ffn, self.layers, self.lr, self.weight_scale = int(hidden), int(ffn), int(layers), float(lr), init_weight_scale()


def mlp_forward(x: Any, w1: Any, w2: Any) -> Any:
    """Two-layer SwiGLU-style MLP forward (supports PyTorch `nn.Linear` and pure-Python lists)."""
    if hasattr(x, "shape"):
        import torch.nn.functional as F

        hidden = F.silu(w1(x))
        return w2(hidden)
    out = []
    for row in x:
        h = [d / (1.0 + math.exp(-max(-20.0, min(20.0, d)))) for d in (sum(a * b for a, b in zip(row, col)) for col in w1)]
        out.append([sum(a * b for a, b in zip(h, col)) for col in w2])
    return out


def compute_loss(pred: Any, target: Any) -> Any:
    """Scalar MSE loss (hot-swappable across PyTorch CUDA/FSDP2 and CPU)."""
    if hasattr(pred, "shape"):
        diff = pred - target
        return (diff * diff).mean()
    diffs = [(p_val - t_val) for p_row, t_row in zip(pred, target) for p_val, t_val in zip(p_row, t_row)]
    return sum(d * d for d in diffs) / max(len(diffs), 1)


def build_torch_model(cfg: TitanStepConfig, device: str, use_fsdp2: bool) -> Any:
    """Build a multi-layer `nn.Module` MLP and optionally shard it with FSDP2 `fully_shard`."""
    import torch, torch.nn as nn

    class TitanBlock(nn.Module):
        def __init__(self, h: int, f: int) -> None:
            super().__init__()
            self.w1, self.w2 = nn.Linear(h, f, bias=False), nn.Linear(f, h, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + mlp_forward(x, self.w1, self.w2)

    class TitanMLPNet(nn.Module):
        def __init__(self, c: TitanStepConfig) -> None:
            super().__init__()
            self.blocks, self.head = nn.ModuleList([TitanBlock(c.hidden, c.ffn) for _ in range(c.layers)]), nn.Linear(c.hidden, c.hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    model = TitanMLPNet(cfg).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for idx, p in enumerate(model.parameters()):
            gen = torch.Generator(device="cpu").manual_seed(1000 + idx)
            p.copy_((torch.randn(p.shape, generator=gen, dtype=torch.float32) * (cfg.weight_scale / math.sqrt(max(p.shape[-1], 1)))).to(device=device))
    if use_fsdp2:
        from torch.distributed.fsdp import fully_shard
        for block in model.blocks:
            fully_shard(block)
        fully_shard(model)
    return model
