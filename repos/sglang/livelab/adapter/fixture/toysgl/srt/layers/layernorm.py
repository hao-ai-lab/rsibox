"""Norm of the toy decoder layer (runs inside captured decode graphs)."""
MODULUS = 4093          # hidden values live in [0, MODULUS): a graph is a table over them


def rms_norm(h):
    return (h * 7 + 1) % MODULUS
