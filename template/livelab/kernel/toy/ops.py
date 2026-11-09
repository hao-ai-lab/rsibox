"""Toy ops."""
import functools

STEP_SCALE = 2


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return fn(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


def scale(x):
    return x * STEP_SCALE


@traced
def activate(x):
    return x + 1


def make_shift(k):
    bias = 0

    def shift(x):
        return x + k
    return shift
