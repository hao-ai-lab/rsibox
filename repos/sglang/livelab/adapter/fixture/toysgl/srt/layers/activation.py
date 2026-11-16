"""Activation of the toy decoder layer (runs inside captured decode graphs)."""
from adapter.fixture.toysgl.srt.layers.layernorm import MODULUS


def silu_and_mul(h):
    return (h * h + 3) % MODULUS
