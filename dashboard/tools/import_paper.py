#!/usr/bin/env python3
"""Copy the paper's figure data into dashboard/data/ (run once, on the machine that holds the paper sources).

The dashboard never reads outside dashboard/data/. This script is the only thing that does, and it writes:

  data/atoms/<id>.json      paper2 atoms, copied byte for byte (the paper's figure data)
  data/atoms-index.json     the index.json entries of the copied atoms (title, claim, caption, source)
  data/picks.json           which runs each paper figure draws (publish/runs-index.json, publish/index.json,
                            and the hidden lists from paper2.db): the user's own selection, used as-is
  data/fig1-ladder.json     Figure 1 rows, computed exactly as paper/figures/fig1_stacked.py does
  data/fig8-checks.json     Figure 8 values (publish/index.json p7-checks; counts screens, as the paper does)
  data/table2-ablation.json Table 2 inputs (paper-figs-20261024/data/ablations.json, group "DeepSeek v4-v5 (s6-s11)")
  data/decode-checks.json   every decode bench per race and case, from the raw decode ledgers (the decode atoms
                            carry no check dots)
  data/steps.json           commit and commit subject for each best-so-far step that could be matched to a ledger
                            row (by time); steps that could not be matched are left out, never guessed

Usage: python3 dashboard/tools/import_paper.py [--rsi DIR] [--paper DIR]
"""
import argparse
import json
import math
import os
import shutil
import sqlite3
import statistics as st
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

ATOMS = [
    "r-prefill-p8k", "r-prefill-p16k", "r-prefill-p32k", "r-prefill-p64k",
    "r-decode-d1x32k", "r-decode-d8x32k", "r-decode-d32x32k", "r-decode-d16x128k",
    "r-megatron", "r-t2av-seeds", "r-omni",
    "r-gqa-kernel", "r-mla-kernel", "r-moe-fwd-kernel", "r-moe-bwd-kernel", "r-nanogpt",
    "m-check-phases-3sys", "e1-race-check-dots-ordered", "f4-fidelity-systems", "m4-model-speed-replay",
]


def rows(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsi", default=os.path.expanduser("~/Projects/ambient/research/rsi-inference-design"))
    ap.add_argument("--paper", default=os.path.expanduser("~/Projects/rsi/paper"))
    a = ap.parse_args()
    P2, PF = f"{a.rsi}/paper2", f"{a.rsi}/paper-figs-20261024"
    os.makedirs(f"{DATA}/atoms", exist_ok=True)

    # 1. Atoms, verbatim.
    for aid in ATOMS:
        shutil.copyfile(f"{P2}/atoms/{aid}.json", f"{DATA}/atoms/{aid}.json")
    idx = {x["id"]: x for x in json.load(open(f"{P2}/atoms/index.json"))}
    # r-t2av-seeds has no index entry; keep what exists.
    dump("atoms-index.json", {aid: idx[aid] for aid in ATOMS if aid in idx})

    # 2. The paper's run selection.
    con = sqlite3.connect(f"{P2}/paper2.db")
    hidden = {i: json.loads(h or "[]") for i, h in con.execute("select id, hidden from picks")}
    runs_index = json.load(open(f"{P2}/publish/runs-index.json"))
    pub_index = json.load(open(f"{P2}/publish/index.json"))
    keep = {"main-vllm", "main-systems"}
    figs = [f for f in runs_index if f["id"] in keep] + [f for f in pub_index if f["id"] in ("p4-low-overhead", "p5-kernels-appendix")]
    dump("picks.json", {
        "figures": [{"id": f["id"], "title": f["title"], "panels": f["panels"], "caption": f.get("caption", "")} for f in figs],
        "hidden": {k: v for k, v in hidden.items() if k.startswith("runs:")},
        "rule": "Picked runs = the race atom's series minus hidden['runs:<race>']. Best run per side = the run named in "
                "panels[].best (highest mean speedup over the race, build_publish.py kept()).",
    })

    # 3. Figure 1 ladder (paper/figures/fig1_stacked.py, same arithmetic).
    bars = json.load(open(f"{a.paper}/data/nemotron-fig1-serve.9c1a829.json"))["bars"]

    def evaluation(b, swap=0.0):
        return {"Run workload": b["requests"] + b["verdict"] + b["post_other"] + swap, "Logprob check": b["drift"]}

    def restart(key):
        b = bars[key]["summary"]["median_breakdown_s"]
        return {"Kernel autotune": b["flashinfer_autotune"], "Weight load": b["load"], "CUDA graph capture": b["graph_capture"],
                "Engine init": b["import"] + b["config"] + b["compile"] + b["other"] + b["swap"], **evaluation(b)}

    stock, store = restart("warm_restart_stock"), restart("warm_restart_store")
    no_load = {k: v for k, v in store.items() if k != "Weight load"}
    no_capture = {k: v for k, v in no_load.items() if k != "CUDA graph capture"}
    live = bars["live_scheduler"]["summary"]["median_c0_breakdown_s"]
    quick = bars["live_scheduler_quick"]["summary"]["median_c0_breakdown_s"]
    live_eval = evaluation(live, live["swap"])
    r = lambda d: {k: round(v, 3) for k, v in d.items()}  # noqa: E731
    dump("fig1-ladder.json", {
        "model": "Nemotron 3.5 Super NVFP4, vllm serve, 1x GB200",
        "source": "paper/data/nemotron-fig1-serve.9c1a829.json via paper/figures/fig1_stacked.py",
        "a": [{"label": "Start w/ cache (baseline)", "seg": r(stock), "measured": True},
              {"label": "+ Reuse autotune config", "seg": r(store), "measured": True},
              {"label": "+ Retain model weight", "seg": r(no_load), "measured": False},
              {"label": "+ Retain CUDA graph", "seg": r(no_capture), "measured": False},
              {"label": "+ Retain live process", "seg": r(live_eval), "measured": True}],
        "b": [{"label": "Live swap, dev workload", "seg": r(live_eval), "measured": True},
              {"label": "+ Skip the logprob check", "seg": r({"Run workload": live_eval["Run workload"]}), "measured": True},
              {"label": "+ Test with a small batch", "seg": r({"Run workload": evaluation(quick, quick["swap"])["Run workload"]}),
               "measured": True}],
    })

    # 4. Figure 8 and Table 2.
    p7 = next(f for f in pub_index if f["id"] == "p7-checks")
    dump("fig8-checks.json", {"columns": p7["columns"], "rows": p7["points"], "caption": p7["caption"],
                              "source": "paper2/publish/index.json (p7-checks)"})
    ab = json.load(open(f"{PF}/data/ablations.json"))
    g = ab["by_group"]["DeepSeek v4-v5 (s6-s11)"]
    dump("table2-ablation.json", {"group": "DeepSeek v4-v5 (s6-s11)", "arms": g["arms"], "n_checks": g["n_checks"],
                                  "clock_h": g["clock_h"], "full": {"mean_check_s": g["mean_check_s"], "checks_per_h": g["checks_per_h"]},
                                  "without": g["without"], "source": "paper-figs-20261024/data/ablations.json by_group"})

    # 5. Decode check dots (all error-free 4-case benches per race), as speedup over the official start.
    DEC = ["d1x32k", "d8x32k", "d32x32k", "d16x128k"]
    dec = {}
    for seed in ("d1", "d2"):
        start = None
        for line in open(f"{PF}/raw/decode/score-dec{seed[1]}.out"):
            p = line.split()
            if len(p) == 10 and p[0] == "start":
                start = dict(zip(DEC, map(float, p[1:9:2])))
        R = rows(f"{PF}/raw/decode/ledger-{seed}.jsonl")
        t0 = min(x["t_req"] for x in R)
        for side in ("normal", "fastrsi"):
            ok = [x for x in R if x["agent"] == f"{seed}-{side}" and x.get("kind") not in ("gate-dummy", "warm-ab")
                  and not x.get("error") and x.get("verdict") != "FAIL" and x.get("t_end")
                  and all((x.get("ms") or {}).get(c) for c in DEC)]
            for c in DEC:
                dec.setdefault(c, {})[f"{side}:{seed}"] = [[round((x["t_end"] - t0) / 3600, 4), round(start[c] / x["ms"][c], 4)]
                                                           for x in ok if (x["t_end"] - t0) / 3600 <= 6]
    dump("decode-checks.json", {"note": "Every error-free 4-case bench of each decode race, speedup over the official start "
                                        "(score-dec*.out). Not in the decode atoms; added from paper-figs-20261024/raw/decode.",
                                "cases": dec})

    # 6. Commit subjects for best-so-far steps.
    subjects = {}
    arm = None
    for line in open(f"{PF}/raw/commit-subjects.txt"):
        if line.startswith("##ARM"):
            arm = line.split()[1]
            subjects[arm] = {}
        elif arm and line.strip():
            h, _, msg = line.strip().partition(" ")
            subjects[arm][h[:7]] = msg
    ideas = json.load(open(f"{PF}/data/deepseek_ideas.json"))["arms"]
    for k, v in ideas.items():
        subjects.setdefault(k, {}).update({c["h"][:7]: c["msg"] for c in v.get("commits", [])})

    def match(points, cands, tol=0.003):
        """cands: [(t_h, commit, extra)]; returns {step index: {...}} for steps with a candidate within tol hours."""
        out = {}
        for i, (t, _v) in enumerate(points[1:-1], start=1):
            best = min(cands, key=lambda c: abs(c[0] - t), default=None)
            if best and abs(best[0] - t) <= tol:
                out[i] = {"commit": best[1][:10], **best[2]}
        return out

    steps = {}
    # Prefill: ledger t_end relative to launch (build_atoms.ds_case).
    t0s = {l.split()[2]: datetime.fromisoformat(l.split()[0].replace("Z", "+00:00")).timestamp()
           for l in open(f"{PF}/raw/v41waves/launched.txt").read().splitlines() if l.strip()}
    ds_rows = [x for x in json.load(open(f"{PF}/data/deepseek_arms.json"))["rows"] if x["arm"] in t0s]
    names = {}
    for side, nm in (("normal", "Codex"), ("fastrsi", "LiveLab")):
        for i, x in enumerate([x for x in ds_rows if x["arm"].endswith(side)]):
            names[f"{nm} #{i + 1}"] = x["arm"]
    for case in ("p8k", "p16k", "p32k", "p64k"):
        atom = json.load(open(f"{DATA}/atoms/r-prefill-{case}.json"))
        for s in atom["series"]:
            armid = names.get(s["name"])
            if not armid:
                continue
            seed = armid.split("-")[0]
            led = next((rows(p) for p in (f"{PF}/raw/v41final/ledger-{seed}.jsonl", f"{PF}/data/ablation_src/deepseek/ledger-{seed}.jsonl")
                        if os.path.exists(p)), [])
            cands = [((x["t_end"] - t0s[armid]) / 3600, x["commit"], {"run": armid, "subject": subjects.get(armid, {}).get(x["commit"][:7])})
                     for x in led if x.get("agent") == armid and x.get("t_end") and (x.get("ms") or {}).get(case)]
            steps[f"r-prefill-{case}|{s['name']}"] = match(s["points"], cands, 1e-6)
    # Decode: a commit counts from max(bench end, gate time) (build_atoms.dec_curve), rounded to 3 decimals.
    for case in DEC:
        atom = json.load(open(f"{DATA}/atoms/r-decode-{case}.json"))
        for s in atom["series"]:
            side = s["arm"]
            seed = "d" + s["name"].rsplit(" ", 1)[1]
            R = rows(f"{PF}/raw/decode/ledger-{seed}.jsonl")
            t0 = min(x["t_req"] for x in R)
            gated = {gg["commit"]: gg["t"] for gg in rows(f"{PF}/raw/decode/gated-{seed}-{side}.jsonl")}
            cands = [(round(max((x["t_end"] - t0) / 3600, (gated[x["commit"]] - t0) / 3600), 3), x["commit"],
                      {"run": f"{seed}-{side}", "subject": subjects.get(f"{seed}-{side}", {}).get(x["commit"][:7])})
                     for x in R if x["agent"] == f"{seed}-{side}" and x.get("t_end") and x["commit"] in gated]
            steps[f"r-decode-{case}|{s['name']}"] = match(s["points"], cands, 1e-6)
    # Megatron: gate time relative to the first request (build_atoms.mega_curve); Codex/LiveLab #1..3 = seeds a..c.
    LAB = f"{a.rsi}/megatron-lab-20261028/data/race5"
    atom = json.load(open(f"{DATA}/atoms/r-megatron.json"))
    for s in atom["series"]:
        d, seed = ("n" if s["arm"] == "normal" else "f"), "abc"[int(s["name"].split("#")[1]) - 1]
        led = rows(f"{LAB}/{d}/ledger-5{seed}.jsonl")
        t0 = min(x["t_req"] for x in led)
        cands = [((gg["t"] - t0) / 3600, gg["commit"], {"run": f"5{seed}-{'normal' if d == 'n' else 'fastrsi'}"})
                 for gg in rows(f"{LAB}/{d}/gated-5{seed}.jsonl")]
        steps[f"r-megatron|{s['name']}"] = match(s["points"], cands, 1e-6)
    # Kernels and nanoGPT: check end times in the task files (fig_anytime.own_checks, speedrun_runs).
    for aid, tasks in (("r-gqa-kernel", ["kda10", "kda12", "kda13", "kda16", "kda19", "kda27"]),
                       ("r-moe-fwd-kernel", ["kda8", "kda11", "kda15", "kda18", "kda21", "kda23"]),
                       ("r-moe-bwd-kernel", ["kda9", "kda14", "kda17", "kda20", "kda22"]),
                       ("r-mla-kernel", ["kda24", "kda25"]), ("r-nanogpt", ["speedrun1", "speedrun2"])):
        atom = json.load(open(f"{DATA}/atoms/{aid}.json"))
        for s in atom["series"]:
            n = int(s["name"].split("race ")[1].split()[0])
            task = json.load(open(f"{PF}/data/{tasks[n - 1]}.json"))
            armd = task["arms"]["baseline" if s["arm"] == "normal" else "fastrsi"]
            if aid == "r-nanogpt":
                cands = [(x["t"] / 3600, x["commit"], {"run": tasks[n - 1]}) for x in armd["submissions"] if x.get("commit")]
            else:
                cands = [(c["t_end"] / 3600, c.get("commit") or "", {"run": tasks[n - 1], "check": c.get("id")})
                         for c in armd["checks"] if c.get("t_end") is not None and c.get("commit")]
            steps[f"{aid}|{s['name']}"] = match(s["points"], cands, 1e-6)
    for k in list(steps):
        for v in steps[k].values():
            if v.get("subject") is None:
                v.pop("subject", None)
    dump("steps.json", {"note": "Best-so-far steps matched to a ledger row by exact time; a step not listed was not "
                                "matched. subject: the commit's first line, from raw/commit-subjects.txt or "
                                "data/deepseek_ideas.json (git log of the run's worktree).",
                        "steps": {k: {str(i): v for i, v in m.items()} for k, m in steps.items()}})
    tot = sum(1 for m in steps.values() for _ in m)
    subj = sum(1 for m in steps.values() for v in m.values() if v.get("subject"))
    print(f"copied {len(ATOMS)} atoms; matched {tot} steps to a commit, {subj} with a subject")


def dump(name, obj):
    with open(f"{DATA}/{name}", "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
