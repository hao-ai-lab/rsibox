#!/usr/bin/env python3
"""Build dashboard.json from dashboard/data/ and check it against the numbers printed in the paper.

  python3 dashboard/build.py                          # writes dashboard/dashboard.json, prints the paper check
  python3 dashboard/build.py --standalone OUT.html    # also writes one self-contained HTML file
  python3 dashboard/build.py --table OUT.md           # also writes the paper check as a Markdown table

Standard library only. Reads nothing outside dashboard/data/.
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ARMS = {"normal": {"name": "Codex (restart per check)", "short": "Codex", "color": "#c8312b"},
        "fastrsi": {"name": "FastRSI (live checks)", "short": "FastRSI", "color": "#1f5fa8"}}

# (id, group, label, atom, hours, selection source, about)
WORKLOADS = [
    ("vllm-p8k", "vLLM prefill", "8K", "r-prefill-p8k", 6, "main-vllm", "DeepSeek-V4.1 Flash, 4x GB200 (EP4DP4); prefill of 8K new tokens after a 128K cached prefix"),
    ("vllm-p16k", "vLLM prefill", "16K", "r-prefill-p16k", 6, "main-vllm", "DeepSeek-V4.1 Flash, 4x GB200; prefill of 16K new tokens"),
    ("vllm-p32k", "vLLM prefill", "32K", "r-prefill-p32k", 6, "main-vllm", "DeepSeek-V4.1 Flash, 4x GB200; prefill of 32K new tokens"),
    ("vllm-p64k", "vLLM prefill", "64K", "r-prefill-p64k", 6, "main-vllm", "DeepSeek-V4.1 Flash, 4x GB200; prefill of 64K new tokens"),
    ("vllm-d1x32k", "vLLM decode", "1×32K", "r-decode-d1x32k", 6, "main-vllm", "DeepSeek-V4.1 Flash, 4x GB200; decode, 1 request per GPU at 32K context"),
    ("vllm-d8x32k", "vLLM decode", "8×32K", "r-decode-d8x32k", 6, "main-vllm", "DeepSeek-V4.1 Flash, 4x GB200; decode, 8 requests per GPU at 32K context"),
    ("vllm-d32x32k", "vLLM decode", "32×32K", "r-decode-d32x32k", 6, "main-vllm", "DeepSeek-V4.1 Flash, 4x GB200; decode, 32 requests per GPU at 32K context"),
    ("vllm-d16x128k", "vLLM decode", "16×128K", "r-decode-d16x128k", 6, "main-vllm", "DeepSeek-V4.1 Flash, 4x GB200; decode, 16 requests per GPU at 128K context"),
    ("megatron", "Training", "Megatron-LM", "r-megatron", 6, "main-systems", "Qwen3-30B-A3B MXFP8 cut to 24 layers, 4x GB200; tokens/s/GPU"),
    ("nanogpt", "Training", "nanoGPT speedrun", "r-nanogpt", 4, "p4-low-overhead", "modded-nanoGPT 124M speedrun, 4-hour budget; training-time speedup (Oct 29 draft, Fig. 12)"),
    ("t2va", "FastVideo", "T2VA", "r-t2av-seeds", 8, "main-systems", "MiniMax H3 text-to-audio-video, one DGX Spark (GB10); time per clip"),
    ("ref2va", "FastVideo", "Ref2VA", "r-omni", 8, "main-systems", "MiniMax H3 reference-to-audio-video, one DGX Spark; time per clip"),
    ("k-gqa", "Kernels", "GQA prefill", "r-gqa-kernel", 2, "p5-kernels-appendix", "FlashInfer-Bench GQA paged prefill, one B200, inside KDA; SOL score (Oct 29 draft, Fig. 13)"),
    ("k-mla", "Kernels", "MLA decode", "r-mla-kernel", 2, "p5-kernels-appendix", "MLA attention decode, one B200, inside KDA; SOL score (Oct 29 draft, Fig. 13)"),
    ("k-moe-fwd", "Kernels", "MoE forward", "r-moe-fwd-kernel", 2, "p5-kernels-appendix", "SOL-ExecBench L2/013, one B200, inside KDA; SOL score (Oct 29 draft, Fig. 13)"),
    ("k-moe-bwd", "Kernels", "MoE backward", "r-moe-bwd-kernel", 2, "p4-low-overhead", "SOL-ExecBench L2/025, one B200, inside KDA; SOL score (Oct 29 draft, Fig. 12)"),
]


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def atom(aid):
    return load(f"atoms/{aid}.json")


def half_up(x, nd):
    q = 10 ** nd
    return math.floor(x * q + 0.5) / q


def fmt(x, nd):
    return f"{half_up(x, nd):.{nd}f}"


def display(name):
    """Run names in the atoms say "LiveLab #5"; the project is FastRSI. The atom name stays in "name" for provenance."""
    return name.replace("LiveLab", "FastRSI")


def mean_over(pts, hours):
    xs = [0.0] + [min(p[0], hours) for p in pts] + [hours]
    ys = [pts[0][1]] + [p[1] for p in pts]
    return sum((xs[i + 1] - xs[i]) * ys[i] for i in range(len(ys))) / hours


def sooner(codex, live):
    """build_publish.sooner: Codex's final value, when each side first reaches it, and the printed label."""
    y = codex[-1][1]
    tc = next((p[0] for p in codex if p[1] >= y - 1e-9), None)
    tl = next((p[0] for p in live if p[1] >= y - 1e-9), None)
    label = f"{tc / tl:.1f}× sooner" if tl is not None and tl < tc else "tie"
    return {"target": y, "t_codex": tc, "t_livelab": tl, "label": label}


def build_workloads(picks, steps, dec_checks):
    figs = {f["id"]: f for f in picks["figures"]}
    out = []
    for wid, group, label, aid, hours, src, about in WORKLOADS:
        a = atom(aid)
        hidden = set(picks["hidden"].get(f"runs:{aid}", []))
        panel = next(p for p in figs[src]["panels"] if p["race"] == aid)  # first panel of this race = the drawn pair
        best = panel["best"]
        series = []
        for s in a["series"]:
            if s["name"] in hidden:
                continue
            checks = s.get("checks") or []
            check_src = "atom"
            if not checks and aid.startswith("r-decode-"):
                seed = "d" + s["name"].rsplit(" ", 1)[1]
                checks = dec_checks["cases"][aid.split("-")[-1]].get(f"{s['arm']}:{seed}", [])
                check_src = "decode-checks.json (raw ledger)"
            st_ = steps.get(f"{aid}|{s['name']}", {})
            pts = []
            for i, (t, v) in enumerate(s["points"]):
                info = st_.get(str(i))
                pts.append([round(t, 4), round(v, 5)] + ([info] if info else []))
            series.append({"name": s["name"], "label": display(s["name"]), "arm": s["arm"], "best": s["name"] in best, "style": s.get("style"),
                           "points": pts, "checks": [[round(t, 4), round(v, 5)] for t, v in checks],
                           "checks_source": check_src if checks else None,
                           "final": s["points"][-1][1], "mean": round(mean_over(s["points"], hours), 4)})
        b = {k: next(x for x in series if x["arm"] == k and x["best"]) for k in ARMS}
        so = sooner(b["normal"]["points"], b["fastrsi"]["points"])
        sol = a["y_label"] == "SOL score"
        out.append({"id": wid, "group": group, "label": label, "atom": aid, "hours": hours, "about": about,
                    "y_label": "SOL score (0-1)" if sol else "Speedup over the starting code (×)", "digits": 3 if sol else 2,
                    "unit": "" if sol else "×", "figure": figs[src]["title"], "selection": src,
                    "hidden_runs": sorted(display(h) for h in hidden), "series": series, "sooner": so,
                    "caption": load("atoms-index.json").get(aid, {}).get("caption", "")})
    return out


def verify(work, paper):
    rows = []
    W = {w["atom"]: w for w in work}

    def add(fig, src, item, pv, dv, ok, note=""):
        rows.append({"figure": fig, "source": src, "item": item, "paper": str(pv), "dashboard": str(dv), "match": ok, "note": note})

    for aid, pv in paper["races"].items():
        w = W[aid]
        b = {k: next(x for x in w["series"] if x["arm"] == k and x["best"]) for k in ARMS}
        nd = w["digits"]
        pdf = "fastrsi (11).pdf" if pv["pdf"] == "pdf11" else "fastrsi (3).pdf"
        names = f"{b['normal']['label']} vs {b['fastrsi']['label']}"
        for arm, key in (("normal", "codex"), ("fastrsi", "livelab")):
            d = fmt(b[arm]["final"], nd)
            add(f"{pv['fig']} ({pdf})", f"atoms/{aid}.json", f"{ARMS[arm]['short']} final ({b[arm]['label']})", pv[key], d, d == pv[key])
        if pv["sooner"]:
            d = w["sooner"]["label"]
            add(f"{pv['fig']} ({pdf})", f"atoms/{aid}.json", f"sooner label ({names})", pv["sooner"], d, d == pv["sooner"])
    for t in paper["race_text"]:
        w = W[t["atom"]]
        pdf = "fastrsi (11).pdf" if t["pdf"] == "pdf11" else "fastrsi (3).pdf"
        b = {k: next(x for x in w["series"] if x["arm"] == k and x["best"]) for k in ARMS}
        if t["check"] == "livelab_seconds":
            d = t["start_s"] / b["fastrsi"]["final"]
            add(f"{t['where']} text ({pdf})", f"atoms/{t['atom']}.json", "FastRSI final s/clip", t["value"], f"{d:.1f}",
                abs(d - t["value"]) < 0.05, f"Text: \"{t['text']}\". The figure's 1.95× is 60.1/30.8 s; 32.3 s would be 1.86×.")
        elif t["check"] == "reach_h":
            d = w["sooner"]["t_livelab"]
            add(f"{t['where']} text ({pdf})", f"atoms/{t['atom']}.json", "hours until FastRSI reaches Codex's final", t["value"],
                f"{d:.1f}", abs(d - t["value"]) < 0.05, f"Text: \"{t['text']}\".")
        elif t["check"] == "race2":
            r2 = {k: next(x for x in w["series"] if x["arm"] == k and "race 2" in x["name"]) for k in ARMS}
            for arm, key in (("normal", "codex"), ("fastrsi", "livelab")):
                d = fmt(r2[arm]["final"], 2)
                add(f"{t['where']} text ({pdf})", f"atoms/{t['atom']}.json", f"{ARMS[arm]['short']} race 2 final", t[key], d, d == t[key])
        elif t["check"] == "best_any":
            for arm, key in (("normal", "codex"), ("fastrsi", "livelab")):
                xs = [x for x in w["series"] if x["arm"] == arm]
                top = max(xs, key=lambda x: x["final"])
                d = fmt(top["final"], 3)
                add(f"{t['where']} text ({pdf})", f"atoms/{t['atom']}.json", f"{ARMS[arm]['short']} best of all races ({top['label']})",
                    t[key], d, d == t[key], f"Text: \"{t['text']}\".")
    for p in paper["fig13_pairs"]:
        w = W[p["atom"]]
        for arm, key in (("normal", "codex"), ("fastrsi", "livelab")):
            s = next(x for x in w["series"] if x["arm"] == arm and x["name"].endswith(f"race {p['race']}"))
            d = fmt(s["final"], 3)
            add("Fig. 13 (fastrsi (3).pdf)", f"atoms/{p['atom']}.json", f"{ARMS[arm]['short']} race {p['race']} final", p[key], d, d == p[key])
    return rows


def secs(t):
    return f"{t:.1f} s" if t < 30 else f"{half_up(t, 0):,.0f} s"


def bands(paper, rows):
    def add(fig, src, item, pv, dv, ok, note=""):
        rows.append({"figure": fig, "source": src, "item": item, "paper": str(pv), "dashboard": str(dv), "match": ok, "note": note})

    P11 = "fastrsi (11).pdf"
    # Figure 1 (both bands A and B).
    lad = load("fig1-ladder.json")
    for part in ("a", "b"):
        for r, pv in zip(lad[part], paper["fig1"][part]):
            d = secs(sum(r["seg"].values()))
            add(f"Fig. 1{part} ({P11})", "fig1-ladder.json", r["label"], pv, d, d == pv,
                "" if r["measured"] else "Not measured: the row removes the phase from the row above (fig1_stacked.py).")
    # Figure 3: vLLM panel from the Figure 1 data, FastVideo and Megatron from the atom.
    ph = atom("m-check-phases-3sys")
    stock, live = lad["a"][0]["seg"], lad["a"][-1]["seg"]
    a_v = [p for p in ph["panels"] if p["title"].startswith("vLLM")][0]
    av_tot = [sum(s[1] for s in r["segments"]) for r in a_v["rows"]]
    for (name, pv), got in zip(paper["fig3"]["panels"].items(), [None, *ph["panels"][1:]]):
        if got is None:
            tot = [sum(stock.values()), sum(live.values())]
            src = "fig1-ladder.json (rows 1 and 5)"
        else:
            tot = [sum(s[1] for s in r["segments"]) for r in got["rows"]]
            src = "atoms/m-check-phases-3sys.json"
        for lab, t, p_ in zip(("Restart", "Live"), tot, pv):
            d = secs(t)
            add(f"Fig. 3 ({P11})", src, f"{name} {lab}", p_, d, d == p_)
    add(f"Fig. 3 ({P11})", "atoms/m-check-phases-3sys.json", "vLLM panel in the atom (not drawn)", "228 s / 23.6 s",
        f"{secs(av_tot[0])} / {secs(av_tot[1])}", False,
        "The atom's vLLM panel is DeepSeek-V4.1 prefill on 4 GPUs; the paper's panel is the Fig. 1 Nemotron run on 1 GB200 "
        "(phases kernel autotune, weight load, CUDA graph, init, check). The dashboard draws the Fig. 1 data.")
    # Figure 7.
    e1 = atom("e1-race-check-dots-ordered")
    f7 = []
    for r, (name, pv) in zip(e1["rows"], paper["fig7"]["rows"].items()):
        mc, ml = st.median(r["values"][0]), st.median(r["values"][1])
        ratio = f"{half_up(mc / ml, 1):.1f}×"
        d = f"{half_up(mc, 0):.0f} / {half_up(ml, 0):.0f} / {ratio}"
        p_ = f"{pv[0]} / {pv[1]} / {pv[2]}"
        add(f"Fig. 7 ({P11})", "atoms/e1-race-check-dots-ordered.json", f"{name} median edit-to-result, Codex / FastRSI / ratio", p_, d, d == p_,
            "The atom's 'medians' field holds the FastRSI-arm mean (e.g. vLLM 58.7 s), not the median; the dashboard computes medians."
            if name == "vLLM" else "")
        f7.append({"label": name, "note": r.get("note"), "codex": r["values"][0], "livelab": r["values"][1],
                   "median": [mc, ml]})
    # Figure 8.
    f8 = load("fig8-checks.json")
    for r in f8["rows"]:
        pv = paper["fig8"]["rows"][r[0]]
        d = [r[1], r[2], half_up(r[4], 1), half_up(r[5], 1)]
        ok = [float(x) for x in d] == [float(x) for x in pv]
        add(f"Fig. 8 ({P11})", "fig8-checks.json", f"{r[0]} checks C/L, check-hours C/L", " / ".join(map(str, pv)),
            " / ".join(f"{x:g}" for x in d), ok)
    # Figure 9.
    f4 = atom("f4-fidelity-systems")
    names = {"vLLM": "vLLM (prefill)", "Megatron": "Megatron-LM", "FastVideo (H3)": "FastVideo-T2VA"}
    seen = set()
    for r in f4["rows"]:
        nm = names.get(r["label"], r["label"])
        seen.add(nm)
        vs = r["values"][0]
        d = f"{sum(abs(v) <= f4['band'] for v in vs)}/{len(vs)}"
        pv = paper["fig9"]["rows"].get(nm)
        add(f"Fig. 9 ({P11})", "atoms/f4-fidelity-systems.json", f"{nm}: checks within ±5%", pv, d, d == pv,
            f"median {st.median(vs):+.2f}%")
    for nm, pv in paper["fig9"]["rows"].items():
        if nm not in seen:
            add(f"Fig. 9 ({P11})", "atoms/f4-fidelity-systems.json", f"{nm}: checks within ±5%", pv, "absent", False,
                "Atom note: \"Omni has no same-commit pair on disk.\"")
    # Figure 10.
    m4 = atom("m4-model-speed-replay")
    for r in m4["rows"]:
        pv = paper["fig10"]["rows"][r["system"]][0 if r["arm"] == "Codex" else 1]
        d = f"{half_up(r['hours'][-1], 1):.1f} h"
        add(f"Fig. 10 ({P11})", "atoms/m4-model-speed-replay.json", f"{r['system']} {display(r['arm'])} at ∞ model speed", pv, d, d == pv,
            f"atom {r['hours'][-1]} h" + ("; the atom's own claim says 5.8 h" if d != pv else ""))
    # Table 2.
    t2 = load("table2-ablation.json")
    for lab, s, h, key in paper["table2"]["rows"]:
        g = t2["full"] if key == "full" else t2["without"][key]
        d = f"{half_up(g['mean_check_s'], 0):.0f} s, {half_up(g['checks_per_h'], 1):.1f}/h"
        p_ = f"{s} s, {h:.1f}/h"
        note = ""
        if key == "hotpatch":
            note = ("Table 2 prints 252 s / 13.7 per hr; the ablation data has 287.9 s / 11.05 per hr, and §5.5 text says "
                    "134 s to 288 s but 21.0 to 13.7 /hr. A draft comment says 252 s / 13.7 is the standalone restart "
                    "baseline, not the disable-install replay.")
        add(f"Table 2 ({P11})", "table2-ablation.json", lab, p_, d, d == p_, note)
    return {
        "ladder": lad,
        "phases": {"vllm": {"title": "vLLM (Nemotron 3.5 Super, 1 GB200)", "source": "fig1-ladder.json",
                            "rows": [{"label": "Restart", "seg": [[k, v] for k, v in stock.items()]},
                                     {"label": "Live", "seg": [[k, v] for k, v in live.items()]}]},
                   "others": [{"title": p["title"], "source": "atoms/m-check-phases-3sys.json",
                               "rows": [{"label": r["label"], "seg": [[s[0], s[1], s[2]] for s in r["segments"]]} for r in p["rows"]]}
                              for p in ph["panels"][1:]],
                   "caption": load("atoms-index.json")["m-check-phases-3sys"]["caption"]},
        "edit_to_result": f7, "checks": f8, "fidelity": f4, "replay": m4,
        "ablation": {"rows": [{"label": lab, "paper_s": s, "paper_h": h,
                               "s": (t2["full"] if key == "full" else t2["without"][key])["mean_check_s"],
                               "h": (t2["full"] if key == "full" else t2["without"][key])["checks_per_h"]}
                              for lab, s, h, key in paper["table2"]["rows"]],
                     "meta": {k: t2[k] for k in ("group", "arms", "n_checks", "clock_h", "source")}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "dashboard.json"))
    ap.add_argument("--standalone", help="also write one self-contained HTML file here")
    ap.add_argument("--table", help="also write the paper check as Markdown here")
    a = ap.parse_args()

    paper = load("paper_values.json")
    work = build_workloads(load("picks.json"), load("steps.json")["steps"], load("decode-checks.json"))
    rows = verify(work, paper)
    b = bands(paper, rows)
    doc = {"schema": "fastrsi-dashboard/v0", "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "paper": {"title": "LiveLab-MLSys: Accelerating Autonomous MLSys Optimization through Live Coding",
                     "pdf": "fastrsi (11).pdf", "older_pdf": "fastrsi (3).pdf (Oct 29) for kernels and nanoGPT"},
           "arms": ARMS, "workloads": work, "bands": b, "verification": rows}
    with open(a.out, "w") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    n_ok = sum(r["match"] for r in rows)
    print(f"wrote {a.out} ({os.path.getsize(a.out) // 1024} KB): {len(work)} workloads, paper check {n_ok}/{len(rows)} match")
    for r in rows:
        if not r["match"]:
            print(f"  MISMATCH {r['figure']} | {r['item']} | paper {r['paper']} | data {r['dashboard']}")
    if a.table:
        write_table(rows, a.table)
        print(f"wrote {a.table}")
    if a.standalone:
        standalone(doc, a.standalone)
        print(f"wrote {a.standalone}")


def write_table(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n_ok = sum(r["match"] for r in rows)
    lines = ["# Dashboard vs paper: every charted number", "",
             f"Generated by `python3 dashboard/build.py --table`. {n_ok} of {len(rows)} numbers match; "
             f"{len(rows) - n_ok} do not. Mismatches are reported, not fixed: the dashboard draws the data file.", "",
             "Paper = the value printed in the PDF (read by eye). Dashboard = the same quantity computed from the file in "
             "`dashboard/data/`, rounded the way the figure rounds it.", "",
             "| Figure | Data file | Item | Paper | Dashboard | Match | Note |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        cell = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
        lines.append(f"| {cell(r['figure'])} | `{cell(r['source'])}` | {cell(r['item'])} | {cell(r['paper'])} | "
                     f"{cell(r['dashboard'])} | {'yes' if r['match'] else '**no**'} | {cell(r['note'])} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def standalone(doc, path):
    html = open(os.path.join(HERE, "index.html")).read()
    css = open(os.path.join(HERE, "style.css")).read()
    js = open(os.path.join(HERE, "app.js")).read().replace("</script", "<\\/script")
    data = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="app.js"></script>',
                        f"<script>window.DASHBOARD = {data};</script>\n<script>\n{js}\n</script>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    sys.exit(main())
