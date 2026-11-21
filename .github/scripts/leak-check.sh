#!/bin/bash
# Fail if lines added between BASE and HEAD (or added filenames) look like environment details,
# secrets, or encoded/concatenated variants thereof.
# A pattern check, not a guarantee: it only sees the shapes listed below.
# Usage: leak-check.sh <base> <head>
set -euo pipefail
base=$1 head=$2

# Files that must name these patterns to describe the rule.
exclude=(':(exclude)AGENTS.md' ':(exclude).github/scripts/leak-check.sh'
         ':(exclude).github/workflows/pr-rules.yml')

# Diff from the merge base; with no shared history (an orphan branch), diff the trees directly.
if git merge-base "$base" "$head" >/dev/null 2>&1; then
  diff_range=("$base...$head")
else
  diff_range=("$base" "$head")
fi

added_files=$(git -c core.quotepath=false diff --diff-filter=AR --name-only "${diff_range[@]}" -- . "${exclude[@]}")
export FASTRSI_LEAK_ADDED_FILES="$added_files"

script_file=$(mktemp)
trap 'rm -f "$script_file"' EXIT
cat <<'PY' > "$script_file"
import base64
import os
import re
import sys
import urllib.parse

PATTERNS = [
    ("private IPv4 10/8",
     re.compile(r"(?:^|[^0-9.])10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?:[^0-9.]|$)", re.I)),
    ("private IPv4 172.16/12",
     re.compile(r"(?:^|[^0-9.])172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}(?:[^0-9.]|$)", re.I)),
    ("private IPv4 192.168/16",
     re.compile(r"(?:^|[^0-9.])192\.168\.[0-9]{1,3}\.[0-9]{1,3}(?:[^0-9.]|$)", re.I)),
    ("CGNAT/Tailscale IPv4 100.64/10",
     re.compile(r"(?:^|[^0-9.])100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}(?:[^0-9.]|$)", re.I)),
    ("private/link-local IPv6",
     re.compile(r"(?<![0-9a-fA-F:])(?:fe80:[0-9a-fA-F:]{2,}|f[cd][0-9a-fA-F]{2}:[0-9a-fA-F:]*:[0-9a-fA-F:]+)(?![0-9a-fA-F:])", re.I)),
    ("home or scratch path",
     re.compile(r"/(?:raid|workspace|scratch)/|/home/[a-z]|/Users/[A-Za-z]|/mnt/lustre|/root/[A-Za-z0-9._-]|/(?:gpfs|lustre|wekafs)/|[A-Za-z]:\\\\Users\\\\[A-Za-z]", re.I)),
    ("cluster access recipe",
     re.compile(r"kubeconfig|ProxyCommand|ProxyJump|tailscale|\.ts\.net|ts\.sock|kubectl\s+.*\b(?:exec|port-forward|cp)\b|ssh\s+.*(?:-J\b|-i\s)", re.I)),
    ("private key",
     re.compile(r"BEGIN\s+[A-Z0-9 ]*PRIVATE KEY", re.I)),
    ("token or credential",
     re.compile(
         r"(?<![A-Za-z0-9_-])(?:"
         r"(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}"
         r"|hf_[A-Za-z0-9]{30,}"
         r"|sk-[A-Za-z0-9_-]{20,}"
         r"|sk_(?:live|test)_[0-9A-Za-z]{24,}"
         r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
         r"|AIza[0-9A-Za-z_-]{35}"
         r"|ya29\.[0-9A-Za-z_-]{20,}"
         r"|xox[baprs]-[0-9A-Za-z-]{20,}"
         r"|glpat-[A-Za-z0-9_-]{20,}"
         r"|npm_[A-Za-z0-9]{30,}"
         r"|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
         r"|[A-Za-z][A-Za-z0-9+.-]{1,10}://[A-Za-z0-9._-]+:[^/@\s\"']{8,}@[A-Za-z0-9.-]+"
         r")"
     )),
]

SENSITIVE_FILE_RE = re.compile(
    r"(?:^|/)(?:id_(?:rsa|ed25519|ecdsa|dsa)|kubeconfig|\.env(?:\.[^/]+)?)$|\.(?:pem|key|p12|pfx|jks)$",
    re.I,
)
B64_RE = re.compile(r"[A-Za-z0-9+/_-]{12,}={0,2}")
HEX_RE = re.compile(r"\b(?:0[xX])?(?:[0-9a-fA-F]{2}){6,}\b")
QUOTE_JOIN_RE = re.compile(r"""(["'])\s*(?:\+\s*)?\1""")
MULTILINE_JOIN_RE = re.compile(r"""(["'])\s*(?:\+\s*)?(?:\\\s*)?\n\s*\1""")
BACKSLASH_LINE_RE = re.compile(r"\\\n\s*")


def _printable_ascii(text):
    return bool(text) and "\x00" not in text and (
        sum(32 <= ord(c) < 127 or c in "\n\r\t" for c in text) / len(text) >= 0.85
    )


def _decode_escapes(text):
    out = text.replace("\\/", "/").replace("\\\\", "\\")
    if "\\x" in out:
        out = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), out)
    if "\\u" in out:
        out = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), out)
    return out


def _variants(text):
    out = [text]
    unesc = _decode_escapes(text)
    pct = urllib.parse.unquote(unesc)
    for v in (unesc, pct):
        if v not in out:
            out.append(v)
    joined = QUOTE_JOIN_RE.sub("", pct)
    if joined not in out:
        out.append(joined)
    for src in dict.fromkeys((text, joined)):
        for m in B64_RE.finditer(src):
            raw = m.group(0).rstrip("=")
            pad = raw + "=" * ((4 - len(raw) % 4) % 4)
            for dec in (lambda p: base64.b64decode(p, validate=True), base64.urlsafe_b64decode):
                try:
                    s = dec(pad).decode("utf-8")
                except Exception:
                    continue
                if _printable_ascii(s) and s not in out:
                    out.append(s)
                    break
        for m in HEX_RE.finditer(src):
            hx = m.group(0)
            if hx[:2].lower() == "0x":
                hx = hx[2:]
            try:
                s = bytes.fromhex(hx).decode("utf-8")
            except Exception:
                continue
            if _printable_ascii(s) and s not in out:
                out.append(s)
    return out


def _check_text(file_path, line_no, display_text, text, hits):
    seen_names = set()
    for candidate in _variants(text):
        for name, rx in PATTERNS:
            if name not in seen_names and rx.search(candidate):
                seen_names.add(name)
                hits.setdefault(name, []).append(f"{file_path}:{line_no}: {display_text}"[:160])


hits = {}
for fname in os.environ.get("FASTRSI_LEAK_ADDED_FILES", "").splitlines():
    if fname and SENSITIVE_FILE_RE.search(fname) and not fname.endswith(".pub"):
        hits.setdefault("sensitive filename", []).append(f"{fname}:1: added sensitive file"[:160])

cur_file, line_no, added_count, remaining = "", 0, 0, 0
hunk_lines = []


def _flush_hunk():
    if len(hunk_lines) < 2:
        return
    raw_block = "\n".join(txt for _, txt in hunk_lines)
    joined = MULTILINE_JOIN_RE.sub("", BACKSLASH_LINE_RE.sub("", raw_block))
    if joined != raw_block:
        orig_set = {txt for _, txt in hunk_lines}
        for line in joined.splitlines():
            if line not in orig_set:
                _check_text(cur_file, hunk_lines[0][0], line, line, hits)


for raw_line in sys.stdin.read().splitlines():
    if remaining > 0 and not raw_line.startswith(("+", "-", " ", "\\")):
        _flush_hunk()
        hunk_lines = []
        remaining = 0
    if remaining == 0 and raw_line.startswith("+++ "):
        _flush_hunk()
        hunk_lines = []
        cur_file = raw_line[6:] if raw_line.startswith("+++ b/") else ""
    elif remaining == 0 and raw_line.startswith("@@"):
        _flush_hunk()
        hunk_lines = []
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", raw_line)
        line_no = int(m.group(1)) if m else 0
        remaining = (int(m.group(2)) if m.group(2) is not None else 1) if m else 0
    elif remaining > 0 and raw_line.startswith("+"):
        txt = raw_line[1:]
        added_count += 1
        _check_text(cur_file, line_no, txt, txt, hits)
        hunk_lines.append((line_no, txt))
        line_no += 1
        remaining -= 1
    elif remaining > 0 and raw_line.startswith("\\"):
        continue
    else:
        _flush_hunk()
        hunk_lines = []
_flush_hunk()

if hits:
    for name, entries in hits.items():
        print(f"LEAK? {name}:")
        for e in entries:
            print(f"  {e}")
    print("Move these to ~/fastrsi-lab/<engine>/ (or .lab/) and describe them generically. See AGENTS.md.")
    sys.exit(1)

print(f"OK: no leak pattern in {added_count} added lines. This is a pattern check, not a guarantee.")
PY

git -c core.quotepath=false diff -U0 --no-color "${diff_range[@]}" -- . "${exclude[@]}" | python3 "$script_file"
