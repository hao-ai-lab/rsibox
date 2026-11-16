#!/usr/bin/env python3
"""Build build/fastrsi-audit.html from src/ and the sourced facts in ../facts (already trimmed to what the page shows).
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
bad = [m for pat in (r"/raid|/mnt/" r"lustre|/home/|/Users/|LiveLab|3689728|376e445f|ae76c60e6|41 ?%|(?i:pa[p]er|\bfig\.|\bfigures?\b|atom)|\b[Mm]y\b|\bby me\b|\bme,",)
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
