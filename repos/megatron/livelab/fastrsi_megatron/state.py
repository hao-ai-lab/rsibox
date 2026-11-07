"""Training state the lab must put back so every check starts from the same point.

A training step changes the weights, the optimizer moments, the learning-rate schedule, the RNG
and the data position. A check that runs K steps after another check would start elsewhere, so
its loss and its timing would not compare. The lab therefore keeps one start state S and
restores it before every check:

- weights: every model parameter (bf16, in the DDP param buffers) and the optimizer's fp32
  master copies (`shard_fp32_from_float16_groups` and friends);
- optimizer state: every tensor in the inner optimizer's `state` (Adam moments, step);
- scalars in the optimizer's `param_groups` and the scheduler's `num_steps`;
- RNG: torch CPU, CUDA, and Megatron's tensor-parallel CUDA RNG tracker;
- data: the K x microbatches batches the check feeds, frozen once from the real iterator.

S must not depend on the code under test, or a fresh process started with an edit would start
from a different S than the live lab (conformance c3). Adam creates its moments at the first
optimizer step, so the lab takes S0 before any step, runs the warm-up steps (kernel selection,
lazy state), then puts S0 back and zeroes the state the warm-up created: moments and step
counters are zero in a fresh Adam. A state key it does not know is reported, never zeroed.

Copies live in pinned host memory: about 14 bytes per parameter per rank with bf16 weights,
fp32 master weights and two fp32 moments.
"""
import time

import torch

ZERO_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step")
FP32_GROUPS = ("shard_fp32_from_float16_groups", "shard_fp32_groups", "fp32_from_float16_groups",
               "fp32_from_fp32_groups")


def inner_optimizers(optimizer):
    """Megatron optimizer -> the Megatron optimizers it chains (one per param-group kind)."""
    return list(getattr(optimizer, "chained_optimizers", None) or [optimizer])


def tensors(model, optimizer):
    """{name: tensor} of every tensor a train step changes, deduplicated by storage."""
    out, seen = {}, set()

    def add(name, t):
        if not torch.is_tensor(t):
            return
        k = (t.data_ptr(), t.numel(), t.dtype)
        if k in seen or t.numel() == 0:
            return
        seen.add(k)
        out[name] = t
    for ci, chunk in enumerate(model):
        for n, p in chunk.named_parameters():
            add(f"model{ci}.{n}", p.data)
    for oi, opt in enumerate(inner_optimizers(optimizer)):
        for g in FP32_GROUPS:
            for gi, group in enumerate(getattr(opt, g, None) or []):
                for pi, t in enumerate(group):
                    add(f"opt{oi}.{g}.{gi}.{pi}", t)
        inner = getattr(opt, "optimizer", None)
        for si, (p, st) in enumerate(getattr(inner, "state", {}).items()):
            for k, v in st.items():
                add(f"opt{oi}.state.{si}.{k}", v)
    return out


def group_scalars(optimizer):
    out = []
    for opt in inner_optimizers(optimizer):
        inner = getattr(opt, "optimizer", None)
        out.append([{k: v for k, v in g.items() if k != "params" and not torch.is_tensor(v)}
                    for g in getattr(inner, "param_groups", [])])
    return out


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _tracker():
    try:
        from megatron.core import tensor_parallel
    except ImportError:
        return None
    return tensor_parallel.get_cuda_rng_tracker()


def rng_state():
    t = _tracker()
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "tracker": t.get_states() if t is not None else None}


def set_rng_state(s):
    torch.set_rng_state(s["cpu"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state(s["cuda"])
    if s["tracker"] is not None:
        _tracker().set_states(s["tracker"])


class Snapshot:
    """Host copies of the training state; `restore()` writes them back in place."""

    def __init__(self, model, optimizer, scheduler):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        t0 = time.perf_counter()
        _sync()
        live = tensors(model, optimizer)
        self.host = {}
        pin = torch.cuda.is_available()
        for n, t in live.items():
            h = torch.empty(t.shape, dtype=t.dtype, pin_memory=pin)
            h.copy_(t, non_blocking=pin)
            self.host[n] = h
        _sync()
        self.groups = group_scalars(optimizer)
        self.num_steps = getattr(scheduler, "num_steps", None)
        self.rng = rng_state()
        self.bytes = sum(h.numel() * h.element_size() for h in self.host.values())
        self.take_s = round(time.perf_counter() - t0, 4)

    def restore(self):
        """Write the copies back. Returns what was restored and what could not be."""
        t0 = time.perf_counter()
        live = tensors(self.model, self.optimizer)
        missing = sorted(set(self.host) - set(live))
        extra = sorted(set(live) - set(self.host))
        with torch.no_grad():
            for n, h in self.host.items():
                if n in live:
                    live[n].copy_(h, non_blocking=h.is_pinned())
        for opt, groups in zip(inner_optimizers(self.optimizer), self.groups):
            inner = getattr(opt, "optimizer", None)
            for g, s in zip(getattr(inner, "param_groups", []), groups):
                for k in [k for k in g if k != "params" and not torch.is_tensor(g[k])
                          and k not in s]:
                    del g[k]
                g.update(s)
        if self.num_steps is not None:
            self.scheduler.num_steps = self.num_steps
            self.scheduler.step(increment=0)
        set_rng_state(self.rng)
        _sync()
        return {"tensors": len(self.host) - len(missing), "missing": missing[:5],
                "extra": extra[:5], "n_missing": len(missing), "n_extra": len(extra),
                "seconds": round(time.perf_counter() - t0, 4)}


def zero_new_state(optimizer, before):
    """Zero the optimizer state tensors that are not in `before` (names) and are a fresh
    Adam's zeros; report the others. Returns (zeroed names, unknown names)."""
    zeroed, unknown = [], []
    for n, t in tensors([], optimizer).items():
        if n in before or ".state." not in n:
            continue
        if n.rsplit(".", 1)[1] in ZERO_KEYS:
            with torch.no_grad():
                t.zero_()
            zeroed.append(n)
        else:
            unknown.append(n)
    for opt in inner_optimizers(optimizer):
        inner = getattr(opt, "optimizer", None)
        for st in getattr(inner, "state", {}).values():
            if "step" in st and not torch.is_tensor(st["step"]):
                st["step"] = 0
    _sync()
    return zeroed, unknown


class FrozenBatches:
    """The first n items of the training iterator, kept to feed every check the same data.
    Ranks without an iterator (Megatron gives tensor-parallel ranks > 0 None) keep None."""

    def __init__(self, it, n):
        self.items = None if it is None else [next(it) for _ in range(n)]
        self.wrapper = None if it is None else type(it)

    def iterator(self):
        if self.items is None:
            return None
        base = iter(list(self.items))
        if self.wrapper is not None and self.wrapper.__name__ == "RerunDataIterator":
            return self.wrapper(base)
        return base
