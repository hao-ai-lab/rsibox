"""Replay `patches/h3-hillclimb` live in one warm H3 lab and time the result against base.

    python3 repos/fastvideo/tools/h3_climb.py [--upto 19 | --only 15,16,17,20] [--gpus N] [--videos 6]
        [--rebuild] [--frames 243,124 --reference FILE] [--respawns 2] [--zygote] [--holder]
        [--out FILE]

The engine on PYTHONPATH must be the base FastVideo (no patch applied). Steps, one JSON line each:

1. `base`: `--videos` videos (the first is cold); output hash and seconds per video.
2. One line per patch, in file order up to `--upto`: `install({"diff": text})`, the live swap.
   When the swap refuses a patch (new module globals, a class, a constructor), the live edits
   so far and that patch are folded into one in-place `rebuild` (uninstall, then rebuild the
   touched modules from the live text plus the patch); later patches install on top of it.
   One video follows every patch (`seconds`, `equal_to_base`, `proven`) to measure per-patch
   speedup. Historically (earlier 2026-11-19 run), stacking patches without intermediate videos
   hit `CompilationError ... StopIteration` on lazily-built Triton kernels when later swaps
   shifted lines above an earlier builder, and undoing a cached Triton 3.5+ builder hit
   `closure_not_restored` (`undo_incomplete`). Both are now fixed in `core/livelab/kernel/`:
   `swap.py` pins per-version `<rsibox-live:...>` `linecache` keys on nested code objects, and
   `journal.py` evicts/restores `triton.runtime.jit._triton_jit_function_registry` entries and
   clears `defaultdict.default_factory = None` cycles on undo (verified on 1x GB200: 02-08
   stacked with no intermediate video ran bit-exact in 4.14 s vs 6.01 s base and undid 7/7 in
   9-11 ms RPC each). With `--rebuild`, all patches up to `--upto` go live in one in-place
   `rebuild` instead (the text of every touched module with the patches applied in order),
   which remains useful for applying constructor/global patches in one pass.
3. `patched`: `--videos` videos on the patched engine. Each output must equal base's (the
   patches are bit-exact; `equal_to_base`). `proven` is False when an installed edit never
   ran: on one GPU the sequence-parallel-only patches (09-14, 18, 19) are not reached.
   `--frames 243,124 --reference FILE`: then one video per frame count, compared with the
   output for that count in FILE (`tools/h3_shapes.py --out` of the base).
4. `respawn` x `--respawns`: `reset("respawn")` (with `--zygote` the workers fork from the
   preloaded server; with `--holder` they map the held weights) and one video: the engine is
   back at base files, so the output must equal base's.

Medians use the warm videos only. Exit 1 when an output differs, a patch fails to go live,
or a check fails.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "livelab"))
sys.path.insert(0, HERE)
from h3_shapes import memory  # noqa: E402  (per-rank CUDA memory and graph-cache counters)
PATCHES = os.path.join(os.path.dirname(HERE), "patches", "h3-hillclimb")

OUT = None


def say(**row):
    print(json.dumps(row, default=str), flush=True)
    if OUT:
        OUT.write(json.dumps(row, default=str) + "\n")
        OUT.flush()


def videos(lab, n):
    rows = [lab.check() for _ in range(n)]
    warm = [r["seconds"] for r in rows[1:] if r["ok"]] or [None]
    return rows, {"outputs": sorted({json.dumps(r.get("output"), sort_keys=True) for r in rows}),
                  "seconds": [r.get("seconds") for r in rows],
                  "warm_median_s": statistics.median(warm) if warm[0] is not None else None,
                  "ok": all(r["ok"] for r in rows),
                  "errors": [r.get("error") for r in rows if not r["ok"]]}


def go_live(lab, name, diff):
    """install, or fold every live edit and this patch into one rebuild."""
    t0 = time.time()
    r = lab.install({"diff": diff, "edit_time": t0})
    if r["outcome"] == "committed":
        return {"patch": name, "via": "R0 swap", "ok": True, "cost_s": r["cost_s"],
                "edit_to_run_s": r.get("edit_to_run_s")}
    refused = r["codes"]
    try:
        files = lab.files({"diff": diff})                 # the live text plus this patch
    except ValueError as e:
        return {"patch": name, "via": None, "ok": False, "refused": refused, "error": str(e)}
    live = dict(lab.live_text[-1])
    t1 = time.perf_counter()
    while lab.status().get("txns"):
        u = lab.uninstall(lab.status()["txns"][0])
        if not u["ok"]:
            return {"patch": name, "via": "rebuild", "ok": False, "refused": refused,
                    "error": f"uninstall before rebuild: {u.get('findings')}"}
    b = lab.rebuild({"files": {**live, **files}, "edit_time": t0})
    return {"patch": name, "via": "rebuild", "ok": b["ok"], "refused_by_swap": refused,
            "modules": sorted({**live, **files}), "cost_s": round(time.perf_counter() - t1, 4),
            "edit_to_run_s": b.get("edit_to_rebuilt_s"),
            "findings": [f.get("check") for f in b.get("findings", [])]}


def rebuild_all(lab, paths):
    """One in-place rebuild of every module the patches touch, the patches applied in order."""
    from adapter import diffs
    files = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            diff_text = fh.read()
        files.update(diffs.apply(diff_text, lambda m: files.get(m) or (
            lab._base(m) if lab._exists(m) else None)))
    t0 = time.time()
    t1 = time.perf_counter()
    b = lab.rebuild({"files": files, "edit_time": t0})
    return {"patch": [os.path.basename(p)[:2] for p in paths], "via": "rebuild", "ok": b["ok"],
            "outcome": b["outcome"], "modules": sorted(files),
            "cost_s": round(time.perf_counter() - t1, 4), "edit_to_run_s": b.get("edit_to_rebuilt_s"),
            "findings": [(f.get("check"), str(f.get("observed"))[-800:]) for f in b.get("findings", [])]}


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upto", type=int, default=19)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--respawns", type=int, default=0)
    ap.add_argument("--only", default="", help="patch numbers to take, instead of 01..--upto")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--frames", default="")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--zygote", action="store_true")
    ap.add_argument("--holder", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    OUT = open(a.out, "a") if a.out else None
    from adapter.h3 import h3_config
    from adapter.lab import FastVideoLab
    t0 = time.perf_counter()
    lab = FastVideoLab(h3_config(a.gpus), check_timeout=1800.0, zygote=a.zygote, holder=a.holder)
    bad = 0
    try:
        say(step="load", ranks=lab.ranks, costs=lab.costs)
        _, base = videos(lab, a.videos)
        say(step="base", **base)
        bad += not base["ok"] or len(base["outputs"]) != 1
        files = sorted(glob.glob(os.path.join(PATCHES, "[0-9][0-9]-*.patch")))
        only = {int(x) for x in a.only.split(",") if x}
        files = [p for p in files if (int(os.path.basename(p)[:2]) in only if only
                                      else int(os.path.basename(p)[:2]) <= a.upto)]
        if a.rebuild:
            row = rebuild_all(lab, files)
            say(step="rebuild", **row)
            bad += not row["ok"]
            files = []
        for path in files:
            with open(path, encoding="utf-8") as fh:
                diff_text = fh.read()
            row = go_live(lab, os.path.basename(path), diff_text)
            if row["ok"]:
                c = lab.check()
                row.update(seconds=c.get("seconds"), proven=c.get("proven"),
                           equal_to_base=json.dumps(c.get("output"), sort_keys=True) in base["outputs"],
                           error=c.get("error"), memory=memory(lab) if c["ok"] else None)
                row["ok"] = c["ok"] and row["equal_to_base"]
            say(step="live", **row)
            if not row["ok"]:
                bad += 1
                break
        rows, patched = videos(lab, a.videos)
        patched["equal_to_base"] = patched["outputs"] == base["outputs"]
        patched["proven"] = sorted({str(r.get("proven")) for r in rows})
        if base["warm_median_s"] and patched["warm_median_s"]:
            patched["speedup"] = round(base["warm_median_s"] / patched["warm_median_s"], 3)
        say(step="patched", status=lab.status().get("state"),
            memory=memory(lab) if patched["ok"] else None, **patched)
        bad += not patched["ok"] or not patched["equal_to_base"]
        ref = {}
        if a.reference:
            with open(a.reference, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("output"):
                        ref.setdefault(r["frames"], r["output"])
        for n in [int(x) for x in a.frames.split(",") if x]:
            c = lab.check({"sampling": {"num_frames": n}})
            same = c.get("output") == ref.get(n) if n in ref else None
            say(step="shape", frames=n, ok=c["ok"], seconds=c.get("seconds"), output=c.get("output"),
                equal_to_reference=same, memory=memory(lab) if c["ok"] else None,
                error=(c.get("error") or "")[-600:] or None)
            bad += not c["ok"] or same is False
        for i in range(a.respawns):
            r = lab.reset("respawn")
            rows, after = videos(lab, 1)
            say(step="respawn", i=i, ok=r["ok"], respawn_s=r["cost_s"], costs=lab.costs,
                first_video_s=after["seconds"][0], equal_to_base=after["outputs"] == base["outputs"])
            bad += not r["ok"] or after["outputs"] != base["outputs"]
    finally:
        lab.close()
        say(step="end", total_s=round(time.perf_counter() - t0, 1), failed=bad)
        if OUT:
            OUT.close()
            OUT = None
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
