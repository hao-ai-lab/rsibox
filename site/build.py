#!/usr/bin/env python3
"""Build the rsibox site into site/public/ (static files, no dependencies).

    python3 site/build.py            # writes site/public/index.html
    python3 site/build.py --check    # validate build output without writing site/public/
    python3 -m http.server 8760 --bind 127.0.0.1 --directory site/public

The page is one self-contained HTML file. The only network requests are d3 from cdnjs and fonts from Google Fonts."""
import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "public")
PAGES = (
    (
        "landing",
        "rsibox-story.html",
        "index.html",
        "rsibox",
        "rsibox cuts the evaluation overhead of recursive self-improvement: an agent improving an ML system gets its answer in seconds, not minutes.",
    ),
)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build or verify the static rsibox site bundle.")
    ap.add_argument("--check", action="store_true", help="validate generated HTML without modifying site/public/")
    args = ap.parse_args(argv)

    if not args.check:
        shutil.rmtree(os.path.join(PUB, "audit"), ignore_errors=True)

    for folder, built, dest, title, desc in PAGES:
        subprocess.run([sys.executable, os.path.join(HERE, folder, "build.py")], check=True)
        with open(os.path.join(HERE, folder, "build", built), encoding="utf-8") as fh:
            body = fh.read()
        body = re.sub(r"<title>.*?</title>\s*", "", body, count=1, flags=re.S)
        body = re.sub(r"<meta name=\"claude:[^>]*>\s*", "", body)
        body = re.sub(r'<a href="[^"]*" data-tab="[a-z]+">[^<]*</a>', "", body)
        # No </head> or <body> tags: the parser closes the head at the first body element, so the
        # page's own <link> and <style> stay in the head.
        doc = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>{title}</title><meta name="description" content="{desc}">'
            '<style>body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}</style>\n'
            + body
        )
        assert doc.isascii(), dest
        assert len(doc) > 0, dest
        if args.check:
            print("checked", dest, len(doc), "bytes")
            continue
        out = os.path.join(PUB, dest)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(doc)
        print("wrote", os.path.relpath(out, HERE), len(doc), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
