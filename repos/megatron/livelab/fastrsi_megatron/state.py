"""Training state the lab must put back so every check starts from the same point.

A training step changes the weights, the optimizer moments, the learning-rate schedule, the RNG
and the data position. A check that runs K steps after another check would start elsewhere, so
its loss and its timing would not compare. The lab therefore keeps one start state S and
restores it before every check:

- weights: every model parameter (bf16, in the DDP param buffers) and the optimizer's fp32
  master copies (`shard_fp32_from_float16_groups` and friends);
- buffers: every registered buffer of the model chunks (normalization statistics and the like);
- optimizer state: every tensor in the inner optimizer's `state` (Adam moments, step);
- scalars in the optimizer's `param_groups` and the scheduler's `num_steps`;
- grad scaler: each inner optimizer's `grad_scaler.state_dict()` (fp16; bf16 runs have none);
- RNG: torch CPU, CUDA, Megatron's tensor-parallel CUDA RNG tracker, Python `random`, numpy;
- data: the K x microbatches batches the check feeds, frozen once from the real iterator.

Restore fails closed: if the live state has a tensor the snapshot does not (extra), lacks one it
has (missing), or one changed shape or dtype, or the optimizer's groups changed, it writes
nothing and returns ok False with the names. `digest()` hashes every piece above (sha256 of the
exact bytes), so a reset round trip can be shown bit for bit.

S must not depend on the code under test, or a fresh process started with an edit would start
from a different S than the live lab (conformance c3). Adam creates its moments at the first
optimizer step, so the lab takes S0 before any step, runs the warm-up steps (kernel selection,
lazy state), then puts S0 back and zeroes the state the warm-up created: moments and step
counters are zero in a fresh Adam. A state key it does not know is reported, never zeroed.

Quantized parameters (FP8 / MXFP8 weights, `fp8_param`) are copied as the plain tensors that hold
their bytes (data and scales), never through the quantized tensor itself, which would round
them again.

Copies live in pinned host memory: about 14 bytes per parameter per rank with bf16 weights,
fp32 master weights and two fp32 moments.
"""
import copy
import ctypes
import hashlib
import json
import random
import time

import torch

try:
    import numpy as np
except ImportError:                             # numpy is optional for the lab itself
    np = None

ZERO_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step")
FP32_GROUPS = ("shard_fp32_from_float16_groups", "shard_fp32_groups", "fp32_from_float16_groups",
               "fp32_from_fp32_groups")


def inner_optimizers(optimizer):
    """Megatron optimizer -> the Megatron optimizers it chains (one per param-group kind)."""
    return list(getattr(optimizer, "chained_optimizers", None) or [optimizer])


def raw_parts(t):
    """[(suffix, plain tensor)] that hold t's bytes. A quantized parameter (Transformer Engine's
    FP8 and MXFP8 tensors) is a tensor subclass that keeps its bytes in plain tensors among its
    attributes (`_rowwise_data`, `_rowwise_scale_inv`, `_columnwise_data`, ...). Copying through
    the subclass dequantizes on the way out and quantizes again on the way in, which does not
    give back the same bytes, so the plain tensors are copied instead. A plain tensor is itself."""
    if type(t) is torch.Tensor or not hasattr(t, "__dict__"):
        return [("", t)]
    parts = sorted((k, v) for k, v in vars(t).items() if type(v) is torch.Tensor)
    return [(f".{k}", v) for k, v in parts] or [("", t)]


def tensors(model, optimizer):
    """{name: plain tensor} of every tensor a train step changes, deduplicated by storage.
    Quantized parameters appear as their parts (`raw_parts`)."""
    out, seen = {}, set()

    def add(name, t):
        if not torch.is_tensor(t):
            return
        if type(t) is not torch.Tensor:
            parts = raw_parts(t)
            if parts[0][1] is not t:
                for suffix, part in parts:
                    add(name + suffix, part)
                return
        k = (t.data_ptr(), t.numel(), t.dtype)
        if k in seen or t.numel() == 0:
            return
        seen.add(k)
        out[name] = t
    for ci, chunk in enumerate(model):
        for n, p in chunk.named_parameters():
            add(f"model{ci}.{n}", p.data)
        for n, b in chunk.named_buffers():
            add(f"model{ci}.buffer.{n}", b)
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


def _tracker_states(t):
    """Copies of the tracker's states. Megatron's get_states() returns the live state objects and
    set_states() keeps the dict it is given, which fork() then updates in place: without copies a
    restore would bring back whatever the last check left in the snapshot's own dict."""
    out = {}
    for name, st in t.get_states().items():
        out[name] = st.get_state() if isinstance(st, torch.Generator) else st.clone()
    return out


def _set_tracker_states(t, saved):
    live = t.get_states()
    states = {}
    for name, st in saved.items():
        if isinstance(live.get(name), torch.Generator):    # graph-safe generators: set in place
            live[name].set_state(st.clone())
            states[name] = live[name]
        else:
            states[name] = st.clone()
    t.set_states(states)


def rng_state():
    t = _tracker()
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "tracker": _tracker_states(t) if t is not None and t.get_states() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state() if np is not None else None}


def set_rng_state(s):
    torch.set_rng_state(s["cpu"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state(s["cuda"])
    if s["tracker"] is not None:
        _set_tracker_states(_tracker(), s["tracker"])
    random.setstate(s["python"])
    if s.get("numpy") is not None:
        np.random.set_state(s["numpy"])


def _cpu(x):
    """A deep copy with tensors moved to host memory."""
    if torch.is_tensor(x):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    return copy.deepcopy(x)


def scaler_states(optimizer):
    """[grad_scaler.state_dict() copied to host, or None] per inner optimizer."""
    out = []
    for opt in inner_optimizers(optimizer):
        gs = getattr(opt, "grad_scaler", None)
        out.append(_cpu(gs.state_dict()) if gs is not None and hasattr(gs, "state_dict") else None)
    return out


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
        self.scalers = scaler_states(optimizer)
        self.rng = rng_state()
        self.bytes = sum(h.numel() * h.element_size() for h in self.host.values())
        self.take_s = round(time.perf_counter() - t0, 4)
        cats = {_category(n) for n in self.host}
        self.restored = [c for c in ("weights", "master_weights", "buffers", "optimizer")
                         if c in cats] + ["optimizer_scalars"] + \
            (["scheduler"] if self.num_steps is not None else []) + \
            (["grad_scaler"] if any(x is not None for x in self.scalers) else []) + \
            ["rng"]

    def compare(self, allow_extra=False):
        """What restore() would refuse: {missing, extra, mismatch, groups} (lists of names)."""
        live = tensors(self.model, self.optimizer)
        missing = sorted(set(self.host) - set(live))
        extra = sorted(set(live) - set(self.host))
        mismatch = sorted(n for n, h in self.host.items() if n in live and
                          (tuple(live[n].shape) != tuple(h.shape) or live[n].dtype != h.dtype))
        groups = []
        opts = inner_optimizers(self.optimizer)
        if len(opts) != len(self.groups):
            groups.append(f"{len(opts)} inner optimizers, snapshot has {len(self.groups)}")
        for i, (opt, gs) in enumerate(zip(opts, self.groups)):
            n = len(getattr(getattr(opt, "optimizer", None), "param_groups", []))
            if n != len(gs):
                groups.append(f"optimizer {i}: {n} param groups, snapshot has {len(gs)}")
            scaler = getattr(opt, "grad_scaler", None)
            if hasattr(scaler, "state_dict") != (self.scalers[i] is not None):
                groups.append(f"optimizer {i}: grad scaler presence changed")
        return live, {"missing": missing, "extra": [] if allow_extra else extra,
                      "mismatch": mismatch, "groups": groups, "n_extra_allowed":
                      len(extra) if allow_extra else 0}

    def restore(self, allow_extra=False):
        """Write the copies back, or nothing at all. allow_extra: state the snapshot does not
        have is left as it is (only for the start, where the warm-up creates Adam's state and
        zero_new_state handles it). Returns ok, what was restored, and what could not be."""
        t0 = time.perf_counter()
        live, bad = self.compare(allow_extra)
        out = {"tensors": 0, "missing": bad["missing"][:5], "extra": bad["extra"][:5],
               "mismatch": bad["mismatch"][:5], "groups": bad["groups"],
               "n_missing": len(bad["missing"]), "n_extra": len(bad["extra"])}
        if bad["missing"] or bad["extra"] or bad["mismatch"] or bad["groups"]:
            return {"ok": False, "restored": [], **out,
                    "error": "live state does not match the snapshot; nothing written",
                    "seconds": round(time.perf_counter() - t0, 4)}
        with torch.no_grad():
            for n, h in self.host.items():
                live[n].copy_(h, non_blocking=h.is_pinned())
        for opt, groups in zip(inner_optimizers(self.optimizer), self.groups):
            inner = getattr(opt, "optimizer", None)
            for g, s in zip(getattr(inner, "param_groups", []), groups):
                for k in [k for k in g if k != "params" and not torch.is_tensor(g[k])
                          and k not in s]:
                    del g[k]
                g.update(copy.deepcopy(s))
        if self.num_steps is not None:
            self.scheduler.num_steps = self.num_steps
            self.scheduler.step(increment=0)
        for opt, sd in zip(inner_optimizers(self.optimizer), self.scalers):
            if sd is not None:
                opt.grad_scaler.load_state_dict(_cpu(sd))
        set_rng_state(self.rng)
        _sync()
        out["tensors"] = len(self.host)
        return {"ok": True, "restored": list(self.restored), **out,
                "n_extra_allowed": bad["n_extra_allowed"],
                "seconds": round(time.perf_counter() - t0, 4)}


def _category(name):
    if name.startswith("model"):
        return "buffers" if ".buffer." in name else "weights"
    return "optimizer" if ".state." in name else "master_weights"


CHUNK = 256 << 20


BYTES_PER_READ = 1 << 30                        # one ctypes read; its size is a C int


def _feed(digest_state, host_bytes, bytes_per_read=BYTES_PER_READ):
    """Feed the bytes of a contiguous uint8 CPU tensor to a hash, without numpy when it is not
    installed (then in reads of at most `bytes_per_read` bytes)."""
    if np is not None:
        digest_state.update(memoryview(host_bytes.numpy()))
        return
    address, length = host_bytes.data_ptr(), host_bytes.numel()
    for offset in range(0, length, bytes_per_read):
        digest_state.update(ctypes.string_at(address + offset,
                                             min(bytes_per_read, length - offset)))


def _hash_tensor(h, t, buf):
    """Feed t's dtype, shape and exact bytes to h; GPU tensors through the pinned buffer."""
    h.update(f"{t.dtype}{tuple(t.shape)}".encode())
    u8 = t.detach().contiguous().reshape(-1).view(torch.uint8)
    if u8.device.type == "cpu":
        _feed(h, u8)
        return buf
    if u8.is_meta:
        raise ValueError("cannot hash a meta tensor without storage")
    if buf is None:
        buf = torch.empty(CHUNK, dtype=torch.uint8, pin_memory=True)
    for i in range(0, u8.numel(), CHUNK):
        m = min(CHUNK, u8.numel() - i)
        buf[:m].copy_(u8[i:i + m])
        _feed(h, buf[:m])
    return buf


def _hash_obj(h, x):
    if torch.is_tensor(x):
        _hash_tensor(h, x.cpu(), None)
    elif np is not None and isinstance(x, np.ndarray):
        h.update(f"{x.dtype}{x.shape}".encode())
        h.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            h.update(repr(k).encode())
            _hash_obj(h, x[k])
    elif isinstance(x, (list, tuple)):
        h.update(f"{type(x).__name__}{len(x)}".encode())
        for v in x:
            _hash_obj(h, v)
    else:
        h.update(repr(x).encode())


def digest(model, optimizer, scheduler):
    """sha256 per piece of the state a restore covers, of the exact bytes (live state, not the
    snapshot): weights, master_weights, buffers, optimizer (tensors), optimizer_scalars,
    scheduler, grad_scaler, rng; and `all` over them."""
    _sync()
    hs = {c: hashlib.sha256() for c in ("weights", "master_weights", "buffers", "optimizer")}
    buf = None
    for n, t in sorted(tensors(model, optimizer).items()):
        h = hs[_category(n)]
        h.update(n.encode())
        buf = _hash_tensor(h, t, buf)
    out = {c: h.hexdigest() for c, h in hs.items()}
    for name, obj in (("optimizer_scalars", group_scalars(optimizer)),
                      ("scheduler", getattr(scheduler, "num_steps", None)),
                      ("grad_scaler", scaler_states(optimizer)),
                      ("rng", rng_state())):
        h = hashlib.sha256()
        _hash_obj(h, obj)
        out[name] = h.hexdigest()
    out["all"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out


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
