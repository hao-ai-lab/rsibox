#!/usr/bin/env python3
"""Scan TLA+ specs and TLC configs for vacuous assumptions or disabled checks."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

FORMAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = [FORMAL_DIR, FORMAL_DIR.parent / "kernel" / "formal"]
CFG_KEYS = {
    "INIT", "NEXT", "SPECIFICATION", "CONSTANTS", "CONSTANT", "INVARIANT", "INVARIANTS",
    "PROPERTY", "PROPERTIES", "CHECK_DEADLOCK", "CONSTRAINT", "CONSTRAINTS",
    "ACTION_CONSTRAINT", "ACTION_CONSTRAINTS", "SYMMETRY", "VIEW", "ALIAS", "POSTCONDITION",
}


def strip_tla_comments(text: str) -> str:
    out, i, n, depth = [], 0, len(text), 0
    while i < n:
        two = text[i:i + 2]
        if depth == 0 and two == "\\*":
            nl = text.find("\n", i)
            i = n if nl == -1 else nl
        elif two == "(*" or (depth > 0 and two == "*)"):
            depth, i = depth + (1 if two == "(*" else -1), i + 2
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def _check_tla(path: Path) -> list[str]:
    cleaned = strip_tla_comments(path.read_text(encoding="utf-8"))
    checks = [
        (r"\bASSUME\s+FALSE\b", "vacuous ASSUME FALSE", re.IGNORECASE),
        (r"(?m)^\s*Spec\s*==\s*(?:TRUE|FALSE)\s*$", "disabled Spec definition", 0),
        (r"(?m)^\s*(?:Init|Next)\s*==\s*FALSE\s*$", "vacuous Init/Next == FALSE", 0),
        (r"(?m)^\s*Next\s*==\s*UNCHANGED\s+(?:\w+|<<[^>]*>>)\s*$", "stuttering-only Next == UNCHANGED", 0),
    ]
    return [f"{path}: {msg}" for pat, msg, flags in checks if re.search(pat, cleaned, flags)]


def _check_cfg(path: Path) -> list[str]:
    lines = [ln.strip() for ln in strip_tla_comments(path.read_text(encoding="utf-8")).splitlines() if ln.strip()]
    errs: list[str] = []
    if not (any(ln.startswith("SPECIFICATION ") for ln in lines)
            or (any(ln.startswith("INIT ") for ln in lines) and any(ln.startswith("NEXT ") for ln in lines))):
        errs.append(f"{path}: missing SPECIFICATION or INIT/NEXT")
    invs, in_block = [], False
    for ln in lines:
        head, rest = ln.split()[0], ln.split()[1:]
        if head in ("INVARIANT", "INVARIANTS"):
            in_block = head == "INVARIANTS"
            invs.extend(rest)
        elif head in CFG_KEYS:
            in_block = False
        elif in_block:
            invs.append(head)
    return errs + ([] if invs else [f"{path}: empty INVARIANT/INVARIANTS list"])


def check_assumptions(roots: list[Path]) -> list[str]:
    seen, errors = set(), []
    for root in roots:
        for path in sorted(root.rglob("*.tla")) + sorted(root.rglob("*.cfg")):
            if path.resolve() not in seen:
                seen.add(path.resolve())
                errors.extend(_check_tla(path) if path.suffix == ".tla" else _check_cfg(path))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    args = parser.parse_args(argv)
    roots = args.roots or DEFAULT_ROOTS
    errors = check_assumptions(roots)
    print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr) if errors else print(
        f"OK: assumption check passed across {len(roots)} root(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
