#!/usr/bin/env python3
"""Build the FastRSI site into site/public/ (static files, no dependencies).

    python3 site/build.py            # writes site/public/index.html and site/public/audit/index.html
    python3 -m http.server 8760 --bind 127.0.0.1 --directory site/public

Each page is one self-contained HTML file. The only network requests are d3 from cdnjs and fonts from Google Fonts."""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "public")
PAGES = (("landing", "fastrsi-story.html", "index.html", "FastRSI",
          "FastRSI cuts the evaluation overhead of recursive self-improvement: an agent improving an ML system gets its answer in seconds, not minutes."),
         ("audit", "fastrsi-audit.html", os.path.join("audit", "index.html"), "FastRSI Audit",
          "The detail behind the FastRSI home page: replayed upstream pull requests, coverage, the interface and rules, and fault handling."))
for folder, built, dest, title, desc in PAGES:
    subprocess.run([sys.executable, os.path.join(HERE, folder, "build.py")], check=True)
    body = open(os.path.join(HERE, folder, "build", built), encoding="utf-8").read()
    body = re.sub(r"<title>.*?</title>\s*", "", body, count=1, flags=re.S)
    body = re.sub(r"<meta name=\"claude:[^>]*>\s*", "", body)
    # No </head> or <body> tags: the parser closes the head at the first body element, so the
    # page's own <link> and <style> stay in the head.
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
           f'<title>{title}</title><meta name="description" content="{desc}">'
           '<style>body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}</style>\n' + body)
    assert doc.isascii(), dest
    out = os.path.join(PUB, dest)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(doc)
    print("wrote", os.path.relpath(out, HERE), len(doc), "bytes")
