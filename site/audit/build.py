#!/usr/bin/env python3
"""Build build/fastrsi-audit.html from src/ and the sourced facts in ../facts.
Usage: python3 build.py [path to the facts folder]"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FACTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "facts")
STORY = os.path.join(ROOT, "landing")
rd = lambda p: open(p, encoding="utf-8").read()
facts = {n: json.loads(rd(f"{FACTS}/{n}.json")) for n in ("engines", "prs", "coverage", "conversion", "faults")}
facts["extra"] = json.loads(rd(f"{HERE}/src/pr-extra.json"))

# Static survey (facts/coverage-scan): keep totals, the 10 largest modules and the history steps; no commit ids.
SEG = ("swap", "swap_recapture", "rebuild", "respawn")
def unit(x): return {"total": x["total"], "segs": [x["by_install"][k]["covered"] for k in SEG] + [x["restart"], x["not_covered"]],
                     "live": x["live_covered"], "firm": x["firm"], "uncertain": x["uncertain"]}
VERSION = {"vllm": "the pinned upstream commit (main at 2026-10-29)", "megatron": "megatron-core 0.20.0, scanned without the lab's local modifications", "fastvideo": "version 0.2.1"}
def survey(key, name):
    d, hist = (json.loads(rd(f"{FACTS}/coverage-scan/{key}{s}.json")) for s in ("", ".history"))
    assert d["upstream"]["version"].startswith({"vllm": "main at 2026-10-29", "megatron": "megatron-core 0.20.0", "fastvideo": "0.2.1"}[key])
    mods = sorted(d["modules"].items(), key=lambda kv: -kv[1]["lines"]["total"])[:10]
    steps, prev = [], {}
    for pt in hist["points"]:
        if not pt["changed"]: continue
        pr = re.search(r"\(#(\d+)\)\s*$", pt["subject"])
        steps.append({"idx": pt["index"], "date": pt["date"][:10], "gained": [k for k, on in pt["capabilities"].items() if on and not prev.get(k)],
                      "pr": int(pr.group(1)) if pr else None, "functions": pt["functions"]["live_covered"], "lines": pt["lines"]["live_covered"]})
        prev = pt["capabilities"]
    return {"engine": name, "repo": d["upstream"]["repo"], "version": VERSION[key], "scan_date": d["scan_date"], "capabilities": d["lab"]["capabilities"],
            "functions": unit(d["functions"]), "lines": unit(d["lines"]), "native": {"files": d["native"]["files"], "lines": d["native"]["lines"]},
            "modules_total": len(d["modules"]), "modules": [{"name": k, "functions": unit(m["functions"]), "lines": unit(m["lines"]), "native_files": m["native"]["files"]} for k, m in mods],
            "history": {"n": len(hist["points"]), "first": hist["points"][0]["date"][:10], "last": hist["points"][-1]["date"][:10], "steps": steps}}
facts["survey"] = [survey("vllm", "vLLM"), survey("megatron", "Megatron-LM"), survey("fastvideo", "FastVideo")]

# Keep only what the page shows. Dropped parts hold file-author remarks, mock-up figures or commit ids.
p = facts["prs"]
p["not_listed"] = {}
for u in p.pop("unsupported"):
    if u.get("pr"): p["not_listed"][u["engine"]] = p["not_listed"].get(u["engine"], 0) + 1
for k in ("agent_commit_replays_from_paper", "honest_summary"): p.pop(k)
p["meta"] = {"counts": p["meta"]["counts"]}
facts["engines"].pop("meta"); facts["engines"].pop("prs_135_141")
c = facts["coverage"]; c.pop("meta")
c.pop("template_for_code_coverage")
v = facts["conversion"]; v["meta"] = {k: v["meta"][k] for k in ("whole_repo", "shared_core", "caution")}
for e in v["engines"]:
    e["derived_from_git"].pop("how"); e["derived_from_git"]["merged_prs_titled_for_engine"].pop("how")
    e["derived_from_git"]["source"] = "git history of the main branch on 2026-11-10; GitHub's list of merged PRs"
    e["recorded_facts"] = [f for f in e["recorded_facts"] if not f["fact"].startswith("Hill-climb")]
f = facts["faults"]; f["meta"] = {k: f["meta"][k] for k in ("count_note", "caveat")}
for k in ("megatron_fault_injection", "kernel_fault_timing"): f["earlier_" + k] = f.pop("paper_" + k)
for e in v["engines"]:
    for k in ("wall_time", "agents", "iterations_or_rounds"): e.pop(k)
for e in facts["engines"]["engines"]:
    for k in ("cost_ladders", "headline"): e.pop(k)
# Sources that point at an earlier measurement run are labelled as that, without ids.
def label_sources(x):
    if isinstance(x, list): return [label_sources(i) for i in x]
    if isinstance(x, dict):
        x = {k: label_sources(i) for k, i in x.items()}
        if isinstance(x.get("source"), str):
            parts = [re.sub(r"^atom:[\w-]+$", "earlier measurement run", s.strip()) for s in x["source"].split(";")]
            x["source"] = "; ".join(dict.fromkeys(parts))
    return x
facts = label_sources(facts)

SCRUB = [
    (r"The paper text says 'six fault-injection tests' on Megatron-LM; the atom table has 7 rows because one is a control \(a new file that should install\)\.", "The table has 7 rows: six fault-injection tests on Megatron-LM and one control (a new file that should install)."),
    (r"The t2-faults and t3 tables come from the paper's LiveLab build \(2026-10-28\)\.", "The first two tables below come from an earlier build of the lab (2026-10-28)."),
    (r"This is the paper's LiveLab build for Megatron", "This is an earlier build of the lab for Megatron"),
    (r" \(date in the atom's source folder name\)", ""),
    (r"\batomic\b", "all-or-nothing"),
    (r"LiveLab", "FastRSI"),
    (r"megatron-core 0\.20 dev \(ae76c60e6\)", "megatron-core 0.20 dev"),
    (r"megatron-core 0\.20 dev 'commit ae76c60e6 plus local modifications'", "megatron-core 0.20 dev plus local modifications"),
    (r"FastVideo pin 376e445f", "a pinned FastVideo commit"),
    (r"hao-ai-lab/FastVideo 376e445f", "hao-ai-lab/FastVideo at a pinned commit"),
    (r"origin/main @ 3689728 \(2026-11-10\)", "the main branch on 2026-11-10"),
    (r"12 is my arithmetic from 'the 7 other in-place PRs were unchanged'; the repo states only the 7\.", "The repo states only the 7; 12 is 19 minus 7."),
    (r"The class column \(live / partial reset / restart\) is my mapping of the five mechanisms onto your three buckets\.", "The class column (live, partial reset, restart) is this page's grouping of the five mechanisms."),
    (r"Do not present 34/34 as a property of the public repo's Megatron lab\.", "34 of 34 is not a property of the public repo's Megatron lab."),
    (r"(counted|computed) by me", r"\1"),
    (r"me, from the matrix rows", "counted from the matrix rows"),
]
def scrub(x):
    if isinstance(x, str):
        for a, b in SCRUB: x = re.sub(a, b, x)
        return x
    if isinstance(x, list): return [scrub(i) for i in x]
    if isinstance(x, dict): return {k: scrub(i) for k, i in x.items()}
    return x
facts = scrub(facts)
blob = json.dumps(facts, ensure_ascii=False)

js = lambda s: s.replace("</script", "<\\/script")
css = lambda p: re.sub(r"/\*.*?\*/\s*", "", rd(p), flags=re.S)  # comments are not shipped
out = f"""<title>FastRSI Audit</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap">
<style>
{css(f"{STORY}/src/tokens.css")}
{css(f"{STORY}/dash/files/fr.css")}
{css(f"{STORY}/src/story.css")}
{css(f"{HERE}/src/audit.css")}
</style>
{rd(f"{HERE}/src/page.html")}
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>window.FACTS = {js(blob)};</script>
<script>{js(rd(f"{HERE}/src/audit.js"))}</script>
"""
# Public page: fail the build on private paths, names, commit ids, the old system name or first-person remarks.
bad = [m for pat in (r"/raid|/mnt/" r"lustre|/home/|/Users/|LiveLab|3689728|376e445f|ae76c60e6|41 ?%|(?i:paper|\bfig\.|\bfigures?\b|atom)|\b[Mm]y\b|\bby me\b|\bme,",)
       for m in re.findall(pat, out)]
hexes = [m for m in re.findall(r"\b[0-9a-f]{7,40}\b", blob) if re.search(r"\d", m) and re.search(r"[a-f]", m)]
if bad or hexes: sys.exit(f"refusing to build: private or disallowed text {sorted(set(bad + hexes))}")
# ASCII-only output, so the page reads the same whatever charset a host or viewer decodes it with.
# Non-ASCII becomes \uXXXX inside <script> (it only occurs in strings, templates and comments there),
# a CSS escape inside <style>, and a numeric character reference in markup.
def ascii_only(doc):
    def jsc(c):
        n = ord(c)
        if n < 0x10000: return "\\u%04x" % n
        n -= 0x10000
        return "\\u%04x\\u%04x" % (0xD800 + (n >> 10), 0xDC00 + (n & 0x3FF))
    esc = {"script": jsc, "style": lambda c: "\\%06x " % ord(c), "html": lambda c: "&#x%x;" % ord(c)}
    def sub(s, kind): return re.sub(r"[^\x00-\x7f]", lambda m: esc[kind](m.group(0)), s)
    parts, pos = [], 0
    for m in re.finditer(r"(<(script|style)\b[^>]*>)(.*?)(</\2\s*>)", doc, re.S | re.I):
        parts += [sub(doc[pos:m.start()], "html"), sub(m.group(1), "html"), sub(m.group(3), m.group(2).lower()), m.group(4)]
        pos = m.end()
    parts.append(sub(doc[pos:], "html"))
    out = "".join(parts)
    assert out.isascii()
    return out
out = ascii_only(out)
os.makedirs(f"{HERE}/build", exist_ok=True)
open(f"{HERE}/build/fastrsi-audit.html", "w", encoding="utf-8").write(out)
open(f"{HERE}/build/preview.html", "w", encoding="utf-8").write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}</style></head><body>' + out + '</body></html>')
print("built", len(out), "bytes ->", f"{HERE}/build/fastrsi-audit.html")
