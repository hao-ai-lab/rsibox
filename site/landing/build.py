#!/usr/bin/env python3
"""Build build/fastrsi-story.html: the home page (src/) plus the three results figures (dash/).
Usage: python3 build.py [path to the dashboard working copy with files/ and data/]"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DASH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dash")
rd = lambda p: open(p, encoding="utf-8").read()
idx = rd(f"{DASH}/files/index.html")
sec = lambda i: re.search(r'<section class="fr-sec" id="%s".*?</section>' % i, idx, re.S).group(0)
head = re.search(r'<header class="fr-head".*?</header>', idx, re.S).group(0)
title = re.search(r'<h1 class="fr-title">(.*?)</h1>', head, re.S).group(1)
lede = re.search(r'<p class="fr-lede">(.*?)</p>', head, re.S).group(1)
data = {f[:-4]: rd(f"{DASH}/data/{f}") for f in sorted(os.listdir(f"{DASH}/data")) if f.endswith(".csv") and ".bak" not in f}
js = lambda s: s.replace("</script", "<\\/script")
# What the offline toolsmith built: GPU-verified mechanisms with a measured before and after (facts/engines.json).
FACTS = os.path.join(HERE, "..", "facts", "engines.json")
built = []
# Replayed upstream edits per engine (prs.json beside engines.json). A speedup is computed only from two plain timings.
edits = {}
COPYF = os.path.join(os.path.dirname(FACTS), "case-copy.json")   # wording per case: labels and text
COPY = json.loads(rd(COPYF)) if os.path.exists(COPYF) else {}
COPY_BY = {str(c["pr"]): c for c in COPY.get("cases", [])}
TIMF = os.path.join(os.path.dirname(FACTS), "case-timing.json")
TIM = json.loads(rd(TIMF))["cases"] if os.path.exists(TIMF) else []
PRS = os.path.join(os.path.dirname(FACTS), "prs.json")
if os.path.exists(PRS):
    clean = lambda v: re.sub(r"\s*\(PR #\d+\)", "", str(v)).replace("`", "") if v is not None else ""
    one = lambda v: (lambda m: float(m.group(1)) if m else None)(re.match(r"^(\d+(?:\.\d+)?) s\b(?! eager)", str(v or "")))
    for e in json.loads(rd(PRS))["engines"]:
        rows = []
        for q in e["prs"]:
            run = (lambda m: float(m.group(1)) if m else None)(re.search(r"edit-to-run (\d+(?:\.\d+)?) s", str(q["install_time"]))) or (one(q["install_time"]) if q.get("installed_live") else None)
            rst = one(q.get("restart_time_replaced"))
            rows.append({"pr": q["pr"], "repo": q["upstream_repo"], "title": clean(q["title"]), "live": bool(q.get("installed_live")), "install": clean(q["install_time"]), "check": clean(q.get("check_time")), "restart": clean(q.get("restart_time_replaced")), "effect": clean(q.get("measured_effect")) if q.get("measured_effect") != "none recorded" else "", "x": round(rst / run, 1) if run and rst else None, "t_run": run, "t_rst": rst, "pair": ""})
            r, it = rows[-1], str(q.get("install_time") or "")
            # a refused edit that restarts from a warm copy: compare that with the plain restart, when the record gives both
            m = re.search(r"forked restart ([\d.]+) s eager / ([\d.]+) s graph \(was a plain-restart fallback: ([\d.]+) s / ([\d.]+) s\)", it)
            if m: r.update({"t_run": float(m.group(1)), "t_rst": float(m.group(3)), "x": round(float(m.group(3)) / float(m.group(1)), 1), "pair": "eager mode; with CUDA graphs %s s against %s s" % (m.group(2), m.group(4))})
            m = re.search(r"code restart: ([\d.]+) s; ([\d.]+) s from the zygote", it)
            if m: r.update({"t_run": float(m.group(2)), "t_rst": float(m.group(1)), "x": round(float(m.group(1)) / float(m.group(2)), 1)})
        for r in rows:   # the upstream diff (non-test files), fetched once from the public pull request
            f = os.path.join(os.path.dirname(FACTS), "pr-diffs", "%s__%d.json" % (r["repo"].replace("/", "__"), r["pr"]))
            if not os.path.exists(f): continue
            dj = json.loads(rd(f)); left = 110; files = []
            for fl in dj["files"]:
                if left <= 0: break
                files.append({"path": fl["path"], "lines": fl["lines"][:left], "cut": max(0, len(fl["lines"]) - left)}); left -= len(fl["lines"])
            r.update({"utitle": dj["title"].replace("`", ""), "add": dj["additions"], "del": dj["deletions"], "nfiles": dj["changedFiles"], "files": files, "more": len(dj["files"]) - len(files)})
        # measured edit-to-check-result times with phases (case-timing.json); they replace anything read from prs.json
        by = {}
        for t in TIM:
            if t["engine"] == e["engine"]: by.setdefault(t["id"], []).append(t)
        for cid, ts in by.items():
            isl = lambda o: o.startswith("installed live")   # Megatron-LM writes "installed live"; FastVideo adds how far it was proven
            t = sorted(ts, key=lambda t: not isl(t["fastrsi"]["outcome"]))[0]
            r = next((r for r in rows if str(r["pr"]) == cid), None)
            if r is None:   # one of our own agent edits: no upstream pull request, the diff comes with the timing
                dj = t["diff"]; r = {"pr": cid, "repo": t["repo"], "title": dj["title"], "utitle": dj["title"], "agent": True, "install": "", "add": dj["additions"], "del": dj["deletions"], "nfiles": dj["changedFiles"], "files": [{"path": f["path"], "lines": f["lines"], "cut": 0} for f in dj["files"]], "more": 0}; rows.append(r)
            fr, rs, notes = t["fastrsi"], t["restart"], []
            lab = lambda o: ("installed live, partly proven" if "partly proven" in o else "installed live") if isl(o) else "restarted from a warm copy" if re.search(r"warm restart|restarted warm", o) else "fell back to a restart"
            if t["variant"]:
                o = [x for x in ts if x is not t][0]
                notes.append("This check makes one %s. Without it the check never runs the edited code, and FastRSI restarts instead: %.1f s against %.1f s." % (t["variant"].replace(" before the check", " first"), o["fastrsi"]["total"], o["restart"]["total"]))
            if t.get("improved"):
                im = t["improved"]; notes.append("With a fix that is not merged yet: %.1f s (%.1f\u00d7), %s." % (im["total"], rs["total"] / im["total"], lab(im["outcome"])))
            r.update({"live": isl(fr["outcome"]), "tag": lab(fr["outcome"]), "t_rst": rs["total"], "t_run": fr["total"], "x": round(rs["total"] / fr["total"], 1), "ph": [rs["phases"], fr["phases"]], "setup": t["setup"], "notes": notes, "pair": "", "variant": t["variant"]})
        for r in rows: r["copy"] = [[b["label"], COPY_BY.get(str(r["pr"]), {}).get(b["key"], "")] for b in COPY.get("blocks", [])]
        for r in rows:
            if any(t for _, t in r["copy"]): r["notes"] = []   # the case text already says it
        # lead with edits that never restart from scratch: installed live, then restarted from a warm copy, then the rest
        warm = lambda r: "warm" in r.get("tag", "") or bool(re.search(r"forked restart|from the zygote", r["install"]))
        rows.sort(key=lambda r: (0 if r["live"] else 1 if warm(r) else 2, 1 if r.get("variant") else 0, -(r["x"] or 0)))
        edits[e["engine"]] = rows
if os.path.exists(FACTS):
    num = lambda v: (lambda m: float(m.group(0)) if m else None)(re.search(r"\d+(?:\.\d+)?", str(v)))
    for e in json.loads(rd(FACTS))["engines"]:
        ms = []
        for m in e["mechanisms"]:
            sv = m.get("measured_saving") or {}
            if m.get("status") != "merged_gpu_verified" or sv.get("before") is None or sv.get("after") is None or sv.get("unit") != "s": continue
            b, a = num(sv["before"]), num(sv["after"])
            if not b or a is None: continue
            ms.append({"name": re.sub(r"\s*\(`[^)]*\)|`", "", m["name"]).strip(), "what": m.get("what", ""), "before": str(sv["before"]), "after": str(sv["after"]), "b": b, "a": a, "setup": sv.get("setup", ""), "cost": m.get("startup_cost_removed", "")})
        built.append({"engine": e["engine"], "mechanisms": ms, "edits": edits.get(e["engine"], [])})
page = rd(f"{HERE}/src/page.html")
page = page.replace("{{CASE_SUB}}", COPY.get("subtitle", ""))
page = page.replace("{{TITLE}}", title).replace("{{LEDE}}", lede)
page = page.replace("{{RESULTS}}", "\n".join(sec(i) for i in ("sooner", "eval-time", "model-speed")))
out = f"""<title>FastRSI Home</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap">
<style>
{rd(f"{HERE}/src/tokens.css")}
{rd(f"{DASH}/files/fr.css")}
{rd(f"{HERE}/src/story.css")}
</style>
{page}
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>const RAW = {js(json.dumps(data))};
const BUILT = {js(json.dumps(built))};
{rd(f"{HERE}/src/shim.js")}</script>
""" + "".join(f"<script>{js(rd(f'{DASH}/files/{s}'))}</script>\n" for s in ("fr-core.js", "s1-race.js", "s2-ladder.js", "s3-speed.js")) + f"<script>{js(rd(f'{HERE}/src/story.js'))}</script>\n<script>{js(rd(f'{HERE}/src/site.js'))}</script>\n"
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
open(f"{HERE}/build/fastrsi-story.html", "w", encoding="utf-8").write(out)
print("built", len(out), "bytes")
open(f"{HERE}/build/preview.html", "w", encoding="utf-8").write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0;font:14px system-ui}[hidden]{display:none!important}</style></head><body>' + out + '</body></html>')
