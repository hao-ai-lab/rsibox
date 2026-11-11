"""Model code: the toy lab declares this folder as captured in graphs."""
import torch
import triton
import cute


class Norm:
    def __init__(self, dim):
        self.dim = dim
        self.weight = [1.0] * dim

    def forward(self, x):
        return [v * w for v, w in zip(x, self.weight)]


@torch.compile
def fused(x):
    return x + 1


@triton.jit
def kernel(x_ptr, n):
    pass


@cute.kernel
def dsl_kernel(x):
    pass
