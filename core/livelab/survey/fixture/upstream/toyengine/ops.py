"""Per-call ops. Nothing here is declared or decorated: the default rule tags them OP."""
SCALE = 2


def scale(x):
    return x * SCALE


def add(x, y):
    def inner(a):          # nested: counts with add
        return a + y
    return inner(x)


def build_table(n):
    return [scale(i) for i in range(n)]


def allocate_buffers(n):
    return [0] * n
