"""Real Triton `@triton.jit` kernels for `KernelLab` (imported lazily inside GPU workers)."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fused_affine_silu_kernel(x_ptr, bias_ptr, out_ptr, n_elements, alpha, beta, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gain = 1.0
    z = x * (alpha * gain) + b + beta
    y = z * tl.sigmoid(z)
    tl.store(out_ptr + offs, y, mask=mask)


def launch_triton_affine_silu(x, bias, out, *, alpha: float = 1.5, beta: float = 0.0, block_size: int = 128) -> None:
    """Launch `fused_affine_silu_kernel` with 1D grid over contiguous CUDA tensors."""
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)
    fused_affine_silu_kernel[grid](x, bias, out, n, float(alpha), float(beta), BLOCK=int(block_size))
