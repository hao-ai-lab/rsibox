"""Run several request shapes in one warm H3 lab and report output, time and GPU memory per video.

    python3 repos/fastvideo/tools/h3_shapes.py --frames 124,243,124 [--gpus N] [--repeat K]
        [--reference FILE] [--out FILE]

The engine is the FastVideo on PYTHONPATH (the base, or a tree with `patches/h3-hillclimb`
applied); the workload and paths come from `adapter/h3.py` (`FASTH3_MODEL` and the other
variables it documents). Every check is one video of `kb/workload.md` with `num_frames` changed.
After each video the tool asks every worker for its CUDA memory (allocated, reserved) and,
when the engine has patch 20's graph cache, its shape key, graph count, captures, budget skips
and resets.

`--reference FILE`: JSON lines of an earlier run (`--out` of the base tree, say); each video's
output must equal the reference's output for the same frame count, and the tool exits 1 when one
does not. One JSON line per video on stdout (and `--out`); `"error"` holds a failed check.
"""
import argparse
import ast
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "livelab"))

PROBE = r"""
import json, torch
from fastvideo.models.dits import minimax_h3 as m
g = m._h3_graphs() if hasattr(m, "_h3_graphs") else None
gib = lambda b: round(b / 2**30, 2)
row = {"allocated_gib": gib(torch.cuda.memory_allocated()),
       "reserved_gib": gib(torch.cuda.memory_reserved())}
if g is not None:
    row["graphs"] = {"key": None if g.key is None else list(g.key[1]),
                     "held": sum(s[1] is not None for s in [*g.ff.values(), *g.projection.values()]),
                     "captures": g.captures, "skipped": getattr(g, "skipped", None),
                     "resets": g.resets}
out = json.dumps(row)
"""


def memory(lab):
    reply = lab.repl(PROBE)
    rows = {}
    for rank, r in reply["ranks"].items():
        rows[rank] = json.loads(ast.literal_eval(r["out"])) if r.get("ok") else {"error": r.get("error")}
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", default="124,243,124")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="videos per shape in a row")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from adapter.h3 import h3_config
    from adapter.lab import FastVideoLab
    ref = {}
    if a.reference:
        for line in open(a.reference):
            row = json.loads(line)
            if row.get("output"):
                ref.setdefault(row["frames"], row["output"])
    out = open(a.out, "a") if a.out else None
    t0 = time.perf_counter()
    lab = FastVideoLab(h3_config(a.gpus), check_timeout=1800.0)
    bad = 0
    try:
        print(json.dumps({"load_s": lab.costs.get("load_s"), "ranks": lab.ranks}), flush=True)
        for i, n in enumerate(int(x) for x in a.frames.split(",") for _ in range(a.repeat)):
            c = lab.check({"sampling": {"num_frames": n}})
            row = {"video": i, "frames": n, "ok": c["ok"], "seconds": c.get("seconds"),
                   "output": c.get("output"), "error": c.get("error"), "memory": memory(lab)}
            if c["ok"] and n in ref:
                row["equal_to_reference"] = c["output"] == ref[n]
                bad += not row["equal_to_reference"]
            bad += not c["ok"]
            print(json.dumps(row), flush=True)
            if out:
                out.write(json.dumps(row) + "\n")
                out.flush()
            if not c["ok"]:
                break
    finally:
        lab.close()
    print(json.dumps({"total_s": round(time.perf_counter() - t0, 1), "failed": bad}), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
