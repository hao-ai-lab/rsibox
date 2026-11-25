#!/usr/bin/env python3
"""Anti-cheat gate for Lean 4 proofs: strips nested comments and forbids axioms/cheats."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

FORMAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = FORMAL_DIR / "lean"

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sorry", re.compile(r"\bsorry\b")),
    ("admit", re.compile(r"\badmit\b")),
    ("axiom", re.compile(r"\baxiom\b")),
    ("opaque", re.compile(r"\bopaque\b")),
    ("partial def", re.compile(r"\bpartial\s+def\b")),
    ("unsafe", re.compile(r"\bunsafe\b")),
    ("native_decide", re.compile(r"\bnative_decide\b")),
    ("Lean.ofReduceBool", re.compile(r"\bLean\.ofReduceBool\b")),
    ("Lean.trustCompiler", re.compile(r"\bLean\.trustCompiler\b")),
    ("implemented_by", re.compile(r"\bimplemented_by\b")),
    ("@[extern", re.compile(r"@\[[^\]]*\bextern\b")),
    ("import Mathlib", re.compile(r"(?m)^\s*import\s+Mathlib\b")),
)

THEOREM_RE = re.compile(r"(?m)^\s*(?:(?:private|protected|noncomputable)\s+)*theorem\s+([A-Za-z_][A-Za-z0-9_']*)\b")
MATHLIB_RE = re.compile(r"\bmathlib\b", re.IGNORECASE)


def strip_lean_comments(text: str) -> str:
    """Strip Lean `--` line comments and arbitrarily nested `/- ... -/` block comments."""
    out: list[str] = []
    i, n, depth, in_string = 0, len(text), 0, False
    while i < n:
        ch = text[i]
        two = text[i : i + 2]
        if depth > 0:
            if two == "/-":
                depth += 1
                i += 2
            elif two == "-/":
                depth -= 1
                i += 2
            else:
                if ch == "\n":
                    out.append("\n")
                i += 1
        elif in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
            else:
                if ch == '"':
                    in_string = False
                i += 1
        elif two == "--":
            nl = text.find("\n", i)
            i = n if nl == -1 else nl
        elif two == "/-":
            depth += 1
            i += 2
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
            i += 1
    return "".join(out)


def _iter_lean_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*.lean"))
        if ".lake" not in path.relative_to(root).parts
    ]


def check_lean_axioms(root: Path) -> list[str]:
    """Return a list of anti-cheat/hygiene violations under `root` (empty when clean)."""
    if not root.is_dir():
        return [f"missing Lean project root: {root}"]

    errors: list[str] = []
    toolchain = root / "lean-toolchain"
    if not toolchain.is_file() or not toolchain.read_text(encoding="utf-8").strip():
        errors.append(f"missing or empty lean-toolchain in {root}")

    lakefile = root / "lakefile.toml"
    if not lakefile.is_file():
        errors.append(f"missing lakefile.toml in {root}")

    for cfg_name in ("lean-toolchain", "lakefile.toml", "lakefile.lean", "lake-manifest.json"):
        cfg_path = root / cfg_name
        if cfg_path.is_file() and MATHLIB_RE.search(cfg_path.read_text(encoding="utf-8")):
            errors.append(f"{cfg_path}: forbidden mathlib dependency")

    lean_files = _iter_lean_files(root)
    if not lean_files:
        return errors + [f"no .lean files found under {root}"]

    stripped_by_path: dict[Path, str] = {}
    for path in lean_files:
        cleaned = strip_lean_comments(path.read_text(encoding="utf-8"))
        stripped_by_path[path] = cleaned
        for label, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(cleaned):
                errors.append(f"{path}: forbidden construct '{label}'")

    lib_files = [
        path
        for path in lean_files
        if path.relative_to(root).as_posix() == "RsiboxFormal.lean"
        or path.relative_to(root).parts[:1] == ("RsiboxFormal",)
    ]
    declared_theorems: list[str] = []
    for path in lib_files:
        declared_theorems.extend(THEOREM_RE.findall(stripped_by_path[path]))

    if not declared_theorems:
        errors.append(f"{root}: no theorem declarations found in RsiboxFormal*.lean")

    verify_path = root / "VerifyAxioms.lean"
    if not verify_path.is_file():
        errors.append(f"missing VerifyAxioms.lean in {root}")
    else:
        verify_cleaned = stripped_by_path[verify_path]
        if not re.search(r"(?m)^\s*import\s+RsiboxFormal\b", verify_cleaned):
            errors.append(f"{verify_path}: must import RsiboxFormal")
        for thm in dict.fromkeys(declared_theorems):
            pat = rf"(?m)^\s*#print\s+axioms\s+(?:[A-Za-z_][A-Za-z0-9_'.]*\.)?{re.escape(thm)}\b"
            if not re.search(pat, verify_cleaned):
                errors.append(f"{verify_path}: missing '#print axioms' for theorem '{thm}'")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    errors = check_lean_axioms(args.root)
    print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr) if errors else print(
        f"OK: Lean axiom gate passed in {args.root}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
