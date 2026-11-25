#!/usr/bin/env python3
"""Turn the measured case tables (cases.csv, effect.csv, patches/) into site-facts/case-timing.json.
Usage: python3 make-case-timing.py <cases dir> [<cases dir> ...]"""
import csv, json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ENG = {"megatron-lm": "Megatron-LM", "vllm": "vLLM", "fastvideo": "FastVideo"}
# fine phases -> the few buckets the page shows
# Megatron-LM, FastVideo and vLLM phase names share these buckets; keep the names in step with PH in landing/src/site.js
BUCKETS = [("Failed live attempt, stop the old job", r"^(install|live check|uninstall|stop the old job|refusal|respawn / shut down)"),
           ("Process start-up and imports", r"^(fallback: |respawn / )?(process start|imports)"),
           ("Engine and model start-up", r"^(fallback: |respawn / )?(distributed init|between init|kernel warm-up|model build|optimizer|data iterators|weight load|CUDA graph|engine init|compile/JIT)"),
           ("Load the weights", r"(?!)"),   # the next three are filled from vLLM's own group column (VLLM_GROUP below), never by name
           ("Compile and JIT builds", r"(?!)"),
           ("CUDA graph capture", r"(?!)"),
           ("First step and warm-up steps", r"^(fallback: )?(lab: |step 1|warm-up)"),
           ("Install the edit live", r"^(install|rebuild or recapture|edit apply)"),
           ("Run the check", r"^(check \(|pre-check call|check workload)"),
           ("Other", r"")]
TITLES = {"te-layernorm-linear-gather-before-norm": "Gather the sequence before the layer norm, so backward needs no second all-gather",
          "lmhead-sp-allgather-before-cast": "Start the LM-head all-gather before the gradient cast",
          "residual-plain-add-without-bias-dropout": "Use a plain residual add when dropout is zero and there is no bias",
          "01-vsa-heads-first-tile": "Enable heads-first Triton tile scatter by default in H3 sparse attention",
          "02-qknorm-rope-16-heads": "Process multiple attention heads per Triton program in QK-norm and RoPE",
          "03-fused-gated-merge": "Fuse transpose and gated merge in H3 attention and skip zero padding",
          "08-compact-video-topk": "Pack video-only attention scores contiguously for faster CUDA top-k masking",
          "15-ffn-cuda-graph-replay": "Add CUDA graph signature checking and replay for H3 feed-forward layers"}
FV_SETUP = "All tests run back to back on one GB200 GPU with warm weight and kernel caches using a single-worker FastVideo MiniMax H3 engine (Oct 2026) that generates a 124-frame, 480x832 text-to-video clip in 8 diffusion steps, reporting the median of 3 runs."
VLLM_SETUP = "1 GB200 GPU, vLLM 0.31.0 serving Qwen3.6-27B (BF16, one GPU) with torch.compile and CUDA graphs, speculative decoding and structured outputs on. Warm on-disk caches. The check sends 4 greedy, 8 concurrent sampling, a JSON-schema and an invalid-grammar request. Median of 3 runs."
# vLLM group column -> bucket; groups with no entry go by phase name
VLLM_GROUP = {"imports": "Process start-up and imports", "weight load": "Load the weights", "warm-up": "First step and warm-up steps", "CUDA graph capture": "CUDA graph capture",
              "torch.compile/JIT": "Compile and JIT builds", "check": "Run the check"}
VLLM_ENGINE = r"^(fork_)?(engine_core_init_other|api_server_after_engine)$|^launcher_"
VLLM_FAILED = r"^(refusal|stop_live_server|zygote_refused_stale_so)$"
SETUP = {("megatron-lm", "upstream_pr"): "4 GB200 GPUs on one node. 4-layer MoE (8 experts, expert parallel 4), bf16, mock data. The check is 10 training steps after 4 warm-up steps. Median of 3 runs.",
         ("megatron-lm", "agent_edit"): "4 GB200 GPUs on one node. 8 Qwen3-14B-shaped layers, tensor parallel 2 with sequence parallel, data parallel 2, sequence 4096, bf16, mock data. The check is 10 training steps after 4 warm-up steps. Median of 3 runs.",
         ("fastvideo", "upstream_pr"): FV_SETUP, ("fastvideo", "agent_edit"): FV_SETUP}
def patch(p):
    files, add, dele = [], 0, 0
    for ln in open(p, encoding="utf-8").read().split("\n"):
        if ln.startswith("diff --git"): files.append({"path": ln.split(" b/")[-1], "lines": []})
        elif ln.startswith(("--- ", "+++ ", "index ")) or not files: continue
        else:
            files[-1]["lines"].append(ln); add += ln.startswith("+"); dele += ln.startswith("-")
    for f in files:
        while f["lines"] and f["lines"][-1] == "": f["lines"].pop()
    return {"additions": add, "deletions": dele, "changedFiles": len(files), "files": files}
def vllm_cases(d):
    """vLLM tables: per-phase rows come from the median run of each arm (so they add up to its total), not from per-phase medians."""
    det = json.load(open(os.path.join(d, "vllm-cases-detail.json")))["runs"]
    def arm(pr, name):
        rs = sorted([r for r in det if r["pr"] == pr and r["arm"] == name], key=lambda r: r["total_s"])
        if not rs: return None
        r = rs[len(rs) // 2]; tot = r["total_s"]; o = r["outcome"]; live = o.startswith("installed live")
        b = collections.OrderedDict((n, 0.0) for n, _ in BUCKETS)
        for ph, sec, grp, _, _ in r["rows"]:
            if ph.startswith("of_which"): continue   # not additive
            if grp == "install": n = "Install the edit live" if live else "Failed live attempt, stop the old job"
            elif re.search(VLLM_FAILED, ph): n = "Failed live attempt, stop the old job"
            elif grp in VLLM_GROUP: n = VLLM_GROUP[grp]
            elif re.search(VLLM_ENGINE, ph): n = "Engine and model start-up"
            else: n = "Other"
            b[n] += sec
        gap = tot - sum(b.values())
        if abs(gap) >= 0.01: b["Other"] += gap; print("  unattributed", pr, name, round(gap, 3))   # explicit remainder, shown inside Other
        assert abs(sum(b.values()) - tot) < 0.01, (pr, name)
        return {"total": round(tot, 2), "outcome": o, "phases": [[n, round(s, 2)] for n, s in b.items() if s >= 0.005]}
    res = []
    for pr in dict.fromkeys(r["pr"] for r in det):
        if not str(pr).isdigit(): continue   # drops the "base" pseudo-PR
        c = {"engine": "vLLM", "repo": "vllm-project/vllm", "source": "upstream_pr", "id": str(pr), "variant": "", "n": 3, "setup": VLLM_SETUP,
             "restart": arm(pr, "restart"), "rsibox": arm(pr, "rsibox"), "improved": arm(pr, "improved"), "effect": []}
        c["improved"].pop("phases"); res.append(c)
    return res
out = []
for d in sys.argv[1:]:
    if os.path.exists(os.path.join(d, "vllm-cases-detail.json")): out += vllm_cases(d); continue
    rows = list(csv.DictReader(open(os.path.join(d, "cases.csv"))))
    eff = collections.defaultdict(list)
    if os.path.exists(os.path.join(d, "effect.csv")):
        for x in csv.DictReader(open(os.path.join(d, "effect.csv"))): eff[x["pr"]].append({"method": x["method"], "base_step_s": float(x["base_step_s"]), "edit_step_s": float(x["edit_step_s"]), "ratio": float(x["edit_over_base"])})
    keys = collections.OrderedDict()
    for x in rows:
        if x["variant"] == "effect on the workload":   # FastVideo keeps the edit's effect in the same table
            eff[x["pr"]].append(x); continue
        keys.setdefault((x["engine"], x["repo"], x["source"], x["pr"], x["variant"]), collections.OrderedDict()).setdefault(x["arm"], []).append(x)
    for pr, v in list(eff.items()):
        if not v or "method" in v[0]: continue
        g = lambda k, v=v: [float(x["seconds"]) for x in v if x["phase"].startswith("effect / " + k)][0]
        b, e = g("warm check at base"), g("warm check with the edit")
        m = re.search(r"\((.*)\)", [x["phase"] for x in v if x["phase"].startswith("effect / warm check at base")][0]).group(1)
        eff[pr] = [{"method": m, "base_step_s": b, "edit_step_s": e, "ratio": round(e / b, 4), "percent": round(g("change in percent"), 2), "output": v[0]["outcome"]}]
    for (eng, repo, src, pr, var), arms in keys.items():
        def arm(name, arms=arms, pr=pr, var=var):
            v = arms.get(name)
            if not v: return None
            tot = [float(x["seconds"]) for x in v if x["phase"].startswith("total")][0]
            o = v[0]["outcome"]; fallback = "then" in o or "restarted warm" in o; live = o.startswith("installed live")
            b = collections.OrderedDict((n, 0.0) for n, _ in BUCKETS)
            for x in v:
                ph = x["phase"]
                if ph.startswith("total") or "/of which" in ph: continue
                if live and ph.startswith("warm-up / check excess"): b["Run the check"] += float(x["seconds"]); continue   # the check's own excess over the base check
                if ph.startswith("edit apply") and not live: b["Other"] += float(x["seconds"]); continue
                for n, rx in BUCKETS:
                    if n.startswith("Failed") and not fallback: continue
                    if n.startswith("Install") and fallback: continue
                    if re.search(rx, ph): b[n] += float(x["seconds"]); break
            assert abs(sum(b.values()) - tot) < 0.02, (pr, var, name, sum(b.values()), tot)
            ns = list(b)
            for i, n in enumerate(ns):   # a negative excess (overlap with a timed phase) folds into the bucket before it
                if b[n] < 0 and i: j = max(k for k in range(i) if b[ns[k]] > 0); b[ns[j]] += b[n]; b[n] = 0.0
            return {"total": round(tot, 2), "outcome": o, "phases": [[n, round(s, 2)] for n, s in b.items() if s >= 0.005]}
        cid = pr[:-6] if pr.endswith(".patch") else pr
        c = {"engine": ENG[eng], "repo": repo, "source": src, "id": cid, "variant": var, "n": 3, "setup": SETUP[(eng, src)],
             "restart": arm("restart"), "rsibox": arm("rsibox"), "improved": arm("rsibox, improved lab"), "effect": eff.get(pr, [])}
        if eng == "fastvideo": c["replay"], c["variant"] = var, ""   # how the edit was replayed, not a change to the check
        if c["improved"]: c["improved"].pop("phases")
        if src == "agent_edit":
            pf = [os.path.join(d, "patches", cid + s) for s in (".ported-to-pin.patch", ".patch")]
            c["diff"] = dict(patch([p for p in pf if os.path.exists(p)][0]), title=TITLES[cid])
        out.append(c)
json.dump({"_about": "Edit-to-check-result times per case, measured end to end; phases are sums of measured phase rows. Generated by make-case-timing.py.", "cases": out}, open(os.path.join(HERE, "case-timing.json"), "w"), indent=1)
for c in out: print(c["id"], c["variant"] or "-", c["restart"]["total"], c["rsibox"]["total"], c["rsibox"]["outcome"][:40], c["improved"] and c["improved"]["total"], [(e["method"][:12], e["ratio"]) for e in c["effect"]]); print("   R", c["restart"]["phases"]); print("   F", c["rsibox"]["phases"])
