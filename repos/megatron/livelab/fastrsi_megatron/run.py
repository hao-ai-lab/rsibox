"""Run a Megatron pretrain script unchanged, with startup phase timestamps and, optionally,
the live lab in its step gap.

    torchrun ... -m fastrsi_megatron.run <pretrain_gpt.py> <megatron args>

Wraps the module globals `pretrain()` calls in megatron.training.training
(initialize_megatron, setup_model_and_optimizer, build_train_valid_test_data_iterators,
train_step) and writes $FASTRSI_RUN/phases.rank<r>.json: unix times of each phase boundary
and, per train step outside the lab, its seconds, loss and grad norm. FASTRSI_T0 is the
launcher's start time (so process start-up is counted). FASTRSI_LAB=1 opens the lab at the
first train step (hook.py); FASTRSI_OVERLAY loads edited modules (overlay.py).

FASTRSI_STEP_DETAIL=K splits the first K train steps (synchronized, so a little slower): the
start of each microbatch's forward (after the previous microbatch's backward) and the optimizer
step, as seconds from the step's start, under "detail" of each step. The first step's extra cost
over a steady one then shows where it goes: the first microbatch (lazy communicators, kernel
choice, compilation), the others, or the optimizer (Adam state).
FASTRSI_EAGER_NCCL=1 passes device_id to torch.distributed.init_process_group, so NCCL
communicators are created at init (`init_done`) instead of inside the first step.
"""
import json
import os
import runpy
import sys
import time

T = {"proc_start": float(os.environ.get("FASTRSI_T0", "0")) or None,
     "driver_start": time.time()}
from . import overlay  # noqa: E402,F401  (before any megatron import)
import torch  # noqa: E402
T["torch_imported"] = time.time()
import megatron.training.training as tr  # noqa: E402
T["megatron_imported"] = time.time()
RUN = os.environ.get("FASTRSI_RUN", ".")
STEPS = []
_sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)


def _dump():
    with open(os.path.join(RUN, f"phases.rank{os.environ.get('RANK', '0')}.json"), "w") as f:
        json.dump({"t": T, "steps": STEPS}, f, indent=1)


def _wrap(name, before, after):
    fn = getattr(tr, name, None)
    if fn is None:                              # not in this Megatron version (e.g. kernel warm-up)
        return

    def w(*a, **k):
        T.setdefault(before, time.time())
        r = fn(*a, **k)
        T.setdefault(after, time.time())
        _dump()
        return r
    w.__wrapped__ = fn
    setattr(tr, name, w)


def _eager_nccl():
    import torch.distributed as dist
    raw = dist.init_process_group

    def init_process_group(*a, **k):
        if "device_id" not in k and torch.cuda.is_available():
            k["device_id"] = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        return raw(*a, **k)
    init_process_group.__wrapped__ = raw
    dist.init_process_group = init_process_group


def _detailed(a, k, raw):
    """Run one train step with sync stamps at each microbatch forward and the optimizer step."""
    d = {"mb": []}
    t0 = time.time()
    fwd = a[0]

    def forward_step(*fa, **fk):
        _sync()
        d["mb"].append(round(time.time() - t0, 4))
        return fwd(*fa, **fk)
    opt = a[3]
    step = opt.step

    def opt_step(*oa, **ok):
        _sync()
        d["opt_start"] = round(time.time() - t0, 4)
        r = step(*oa, **ok)
        _sync()
        d["opt_done"] = round(time.time() - t0, 4)
        return r
    opt.step = opt_step
    try:
        return raw(forward_step, *a[1:], **k), d
    finally:
        del opt.step


def main():
    if os.environ.get("FASTRSI_EAGER_NCCL") == "1":
        _eager_nccl()
    from . import zygote
    detail = int(os.environ.get("FASTRSI_STEP_DETAIL", "0") or 0)
    compile_warmup_steps = int(os.environ.get("FASTRSI_COMPILE_WARMUP_STEPS", "2") or 2)
    _wrap("initialize_megatron", "init_start", "init_done")
    _wrap("setup_model_and_optimizer", "model_opt_start", "model_opt_done")
    _wrap("build_train_valid_test_data_iterators", "data_start", "data_done")
    _wrap("get_model", "model_build_start", "model_build_done")
    _wrap("get_megatron_optimizer", "optimizer_start", "optimizer_done")
    _wrap("warmup_training_kernels", "kernel_warmup_start", "kernel_warmup_done")
    raw = tr.train_step

    def train_step(*a, **k):
        T.setdefault("first_step_start", time.time())
        _sync()
        t0 = time.time()
        d = None
        if len(STEPS) < detail and len(a) >= 4 and not k.get("forward_step_func"):
            r, d = _detailed(a, k, raw)
        else:
            r = raw(*a, **k)
        _sync()
        t1 = time.time()
        v = r[0].get("lm loss") if isinstance(r[0], dict) else None
        STEPS.append({"t0": t0, "s": round(t1 - t0, 4),
                      "loss": float(v) if v is not None else None,
                      "grad_norm": float(r[5]) if r[5] is not None else None})
        if d is not None:
            STEPS[-1]["detail"] = d
        if len(STEPS) == compile_warmup_steps:
            zygote.quiesce_compile_pools_after_warmup()
            T.setdefault("compile_pools_quiesced", time.time())
        T.setdefault("first_step_done", t1)
        _dump()
        return r
    train_step.__wrapped__ = raw
    tr.train_step = train_step
    if os.environ.get("FASTRSI_LAB") or os.environ.get("FASTRSI_FRESH"):
        from . import hook
        hook.install_hook(tr)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    T["pretrain_start"] = time.time()
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        T["exit"] = time.time()
        _dump()


if __name__ == "__main__":
    main()
