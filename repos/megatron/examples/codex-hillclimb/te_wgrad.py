"""Replay iteration 12 on a live eager bf16 lab started without gradient accumulation fusion."""
import argparse
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "livelab"))
from fastrsi_megatron import client


def enable(model, journal, arm):
    import torch
    from transformer_engine.pytorch import LayerNormLinear, Linear

    targets = [(f"chunk{i}.{name}", module) for i, chunk in enumerate(model)
               for name, module in chunk.named_modules()
               if isinstance(module, (Linear, LayerNormLinear))]
    if not targets:
        raise ValueError("this workload has no TE linear modules")
    for name, module in targets:
        weight = module.weight
        if (module.fuse_wgrad_accumulation or weight.dtype != torch.bfloat16
                or weight.main_grad.dtype != torch.float32 or module.config.cuda_graph_impl != "none"):
            raise ValueError(f"{name} requires eager bf16, fp32 main_grad and fusion off")
        journal.set_attr(module, "fuse_wgrad_accumulation", True)
        hit = arm(name + ".fused_wgrad")

        def count_fused(param, hit=hit):
            if param.grad_added_to_main_grad:
                hit()

        handle = weight.register_post_accumulate_grad_hook(count_fused)
        journal.record_call(handle.remove, f"remove {name} gradient hook")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    def call(verb, **fields):
        reply = client.call(args.sock, {"verb": verb, **fields})
        print(json.dumps({"verb": verb, **{k: v for k, v in reply.items() if k != "receipt"}}),
              flush=True)
        if not reply.get("ok"):
            raise RuntimeError(f"{verb} failed")
        return reply

    if call("status")["state"] != "base":
        raise RuntimeError("start the replay at base")
    base = call("check", steps=args.steps)["output"]
    cell = inspect.getsource(enable) + "\nenable(model, journal, arm)"
    txn = call("install", patch={"files": {}, "repairs": [{"cell": cell, "declares": []}]})
    try:
        edited = call("check", steps=args.steps)
        if not edited["proven"]:
            raise RuntimeError("missing a fused backward witness")
        for k, limit in (("loss", 0.005), ("grad_norm", 0.05)):
            for a, b in zip(base[k], edited["output"][k], strict=True):
                if not abs(b - a) <= limit * abs(a):
                    raise RuntimeError(f"{k} outside gate: {a}, {b}")
    finally:
        call("uninstall", id=txn["id"])
    if call("status")["state"] != "base" or call("check", steps=args.steps)["output"] != base:
        raise RuntimeError("undo did not reproduce the deterministic base")


if __name__ == "__main__":
    main()
