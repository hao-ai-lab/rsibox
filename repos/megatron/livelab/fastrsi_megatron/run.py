"""Run a Megatron pretrain script unchanged, with startup phase timestamps and, optionally,
the live lab in its step gap.

    torchrun ... -m fastrsi_megatron.run <pretrain_gpt.py> <megatron args>

Wraps the module globals `pretrain()` calls in megatron.training.training
(initialize_megatron, setup_model_and_optimizer, build_train_valid_test_data_iterators,
train_step) and writes $FASTRSI_RUN/phases.rank<r>.json: unix times of each phase boundary
and, per train step outside the lab, its seconds, loss and grad norm. FASTRSI_T0 is the
launcher's start time (so process start-up is counted). FASTRSI_LAB=1 opens the lab at the
first train step (hook.py); FASTRSI_OVERLAY loads edited modules (overlay.py).
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


def _dump():
    with open(os.path.join(RUN, f"phases.rank{os.environ.get('RANK', '0')}.json"), "w") as f:
        json.dump({"t": T, "steps": STEPS}, f, indent=1)


def _wrap(name, before, after):
    fn = getattr(tr, name)

    def w(*a, **k):
        T.setdefault(before, time.time())
        r = fn(*a, **k)
        T.setdefault(after, time.time())
        _dump()
        return r
    w.__wrapped__ = fn
    setattr(tr, name, w)


def main():
    _wrap("initialize_megatron", "init_start", "init_done")
    _wrap("setup_model_and_optimizer", "model_opt_start", "model_opt_done")
    _wrap("build_train_valid_test_data_iterators", "data_start", "data_done")
    _wrap("get_model", "model_build_start", "model_build_done")
    _wrap("get_megatron_optimizer", "optimizer_start", "optimizer_done")
    _wrap("warmup_training_kernels", "kernel_warmup_start", "kernel_warmup_done")
    raw = tr.train_step

    def train_step(*a, **k):
        T.setdefault("first_step_start", time.time())
        torch.cuda.synchronize()
        t0 = time.time()
        r = raw(*a, **k)
        torch.cuda.synchronize()
        t1 = time.time()
        v = r[0].get("lm loss") if isinstance(r[0], dict) else None
        STEPS.append({"t0": t0, "s": round(t1 - t0, 4),
                      "loss": float(v) if v is not None else None,
                      "grad_norm": float(r[5]) if r[5] is not None else None})
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
