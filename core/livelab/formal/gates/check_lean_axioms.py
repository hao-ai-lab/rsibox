#!/usr/bin/env python3
"""Anti-cheat gate for Lean 4 proofs and `#print axioms` output verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

FORMAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = FORMAL_DIR / "lean"

ALLOWED_AXIOMS = frozenset({"propext", "Quot.sound"})

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sorry", re.compile(r"\bsorry\b")),
    ("sorryAx", re.compile(r"\bsorryAx\b")),
    ("admit", re.compile(r"\badmit\b")),
    ("axiom", re.compile(r"\baxiom\b")),
    ("opaque", re.compile(r"\bopaque\b")),
    ("partial", re.compile(r"\bpartial\b")),
    ("unsafe", re.compile(r"\bunsafe\b")),
    ("native_decide", re.compile(r"\bnative_decide\b")),
    ("Lean.ofReduceBool", re.compile(r"\bLean\.ofReduceBool\b")),
    ("Lean.ofReduceNat", re.compile(r"\bLean\.ofReduceNat\b")),
    ("Lean.trustCompiler", re.compile(r"\bLean\.trustCompiler\b")),
    ("implemented_by", re.compile(r"\bimplemented_by\b")),
    ("@[extern", re.compile(r"@\[[^\]]*\bextern\b")),
)

IDENT_CHAR_RE = re.compile(r"[A-Za-z0-9_'»]")
CHAR_LITERAL_RE = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'\n])'")
IMPORT_RE = re.compile(r"(?m)^\s*import\s+([A-Za-z_][A-Za-z0-9_'.]*)")
THEOREM_RE = re.compile(
    r"(?m)^\s*(?:@\[[^\]]*\]\s*|(?:private|protected|noncomputable)\s+)*theorem\s+([A-Za-z_][A-Za-z0-9_']*)\b"
)
VERIFY_DECL_RE = re.compile(
    r"(?m)^\s*(?:@\[[^\]]*\]\s*|(?:private|protected|noncomputable)\s+)*(?:theorem|def|namespace)\b"
)
MATHLIB_RE = re.compile(r"\bmathlib\b", re.IGNORECASE)
TOML_REQUIRE_RE = re.compile(r"\[\[\s*require\s*\]\]|^\s*require\b", re.IGNORECASE | re.MULTILINE)
DIAGNOSTIC_RE = re.compile(r"(?im)(?:^|[\s:])(?:error|warning):")
NO_AXIOMS_RE = re.compile(r"^'RsiboxFormal\.([A-Za-z_][A-Za-z0-9_']*)'\s+does not depend on any axioms$")
DEPS_AXIOMS_RE = re.compile(r"^'RsiboxFormal\.([A-Za-z_][A-Za-z0-9_']*)'\s+depends on axioms:\s*\[([^\]]*)\]$")


def strip_lean_comments(text: str) -> str:
    """Strip `--` and nested `/- ... -/` comments while lexing char and string literals."""
    out: list[str] = []
    i, n, depth, in_string = 0, len(text), 0, False
    while i < n:
        ch = text[i]
        two = text[i : i + 2]
        if depth > 0:
            if two == "/-":
                depth, i = depth + 1, i + 2
            elif two == "-/":
                depth, i = depth - 1, i + 2
            else:
                if ch == "\n":
                    out.append("\n")
                i += 1
        elif in_string:
            if ch == "\\":
                out.append(text[i : i + 2])
                i += 2 if i + 1 < n else 1
            else:
                out.append(ch)
                if ch == '"':
                    in_string = False
                i += 1
        elif two == "--":
            nl = text.find("\n", i)
            i = n if nl == -1 else nl
        elif two == "/-":
            depth, i = depth + 1, i + 2
        elif ch == "'" and (i == 0 or not IDENT_CHAR_RE.match(text[i - 1])):
            m = CHAR_LITERAL_RE.match(text, i)
            if m:
                out.append(m.group(0))
                i = m.end()
            else:
                out.append(ch)
                i += 1
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
            i += 1
    return "".join(out)


def _iter_lean_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*.lean")) if ".lake" not in p.relative_to(root).parts]


def _is_allowed_import(module: str) -> bool:
    return module in ("Init", "RsiboxFormal") or module.startswith(("Init.", "RsiboxFormal."))


def _collect_theorems(root: Path) -> tuple[list[str], dict[Path, str], list[str]]:
    lean_files = _iter_lean_files(root)
    if not lean_files:
        return [], {}, [f"no .lean files found under {root}"]
    stripped: dict[Path, str] = {}
    theorems: list[str] = []
    for path in lean_files:
        cleaned = strip_lean_comments(path.read_text(encoding="utf-8"))
        stripped[path] = cleaned
        if path.relative_to(root).as_posix() != "VerifyAxioms.lean":
            theorems.extend(THEOREM_RE.findall(cleaned))
    return list(dict.fromkeys(theorems)), stripped, []


def check_lean_axioms(root: Path) -> list[str]:
    """Return anti-cheat, import-allowlist, and VerifyAxioms coverage violations under `root`."""
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
        if not cfg_path.is_file():
            continue
        cfg_text = cfg_path.read_text(encoding="utf-8")
        if MATHLIB_RE.search(cfg_text):
            errors.append(f"{cfg_path}: forbidden mathlib dependency")
        if cfg_name == "lakefile.toml" and TOML_REQUIRE_RE.search(cfg_text):
            errors.append(f"{cfg_path}: forbidden external require dependency")
        elif cfg_name == "lakefile.lean" and re.search(r"\brequire\b", cfg_text, re.IGNORECASE):
            errors.append(f"{cfg_path}: forbidden external require dependency")
        elif cfg_name == "lake-manifest.json":
            try:
                manifest = json.loads(cfg_text)
            except json.JSONDecodeError as exc:
                errors.append(f"{cfg_path}: invalid JSON ({exc})")
            else:
                if manifest.get("packages"):
                    errors.append(f"{cfg_path}: forbidden external packages in manifest")

    theorems, stripped, scan_errs = _collect_theorems(root)
    errors.extend(scan_errs)
    if not stripped:
        return errors

    for path, cleaned in stripped.items():
        for label, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(cleaned):
                errors.append(f"{path}: forbidden construct '{label}'")
        for mod in IMPORT_RE.findall(cleaned):
            if not _is_allowed_import(mod):
                errors.append(f"{path}: forbidden non-Init import '{mod}'")

    if not theorems:
        errors.append(f"{root}: no theorem declarations found in library .lean files")

    verify_path = root / "VerifyAxioms.lean"
    if not verify_path.is_file():
        errors.append(f"missing VerifyAxioms.lean in {root}")
    else:
        verify_cleaned = stripped[verify_path]
        if not re.search(r"(?m)^\s*import\s+RsiboxFormal\b", verify_cleaned):
            errors.append(f"{verify_path}: must import RsiboxFormal")
        if VERIFY_DECL_RE.search(verify_cleaned):
            errors.append(f"{verify_path}: must not declare theorem, def, or namespace")
        for thm in theorems:
            pat = rf"(?m)^\s*#print\s+axioms\s+RsiboxFormal\.{re.escape(thm)}\b"
            if not re.search(pat, verify_cleaned):
                errors.append(f"{verify_path}: missing '#print axioms RsiboxFormal.{thm}'")

    return errors


def verify_axioms_output(output_text: str, root: Path) -> list[str]:
    """Verify `lake env lean VerifyAxioms.lean` stdout/stderr against strict axiom allowlist."""
    errors = check_lean_axioms(root)
    theorems, _, _ = _collect_theorems(root)
    seen: set[str] = set()

    for lineno, raw in enumerate(output_text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if DIAGNOSTIC_RE.search(line):
            errors.append(f"line {lineno}: Lean diagnostic '{line}'")
            continue
        m_none = NO_AXIOMS_RE.match(line)
        if m_none:
            seen.add(m_none.group(1))
            continue
        m_deps = DEPS_AXIOMS_RE.match(line)
        if m_deps:
            thm = m_deps.group(1)
            seen.add(thm)
            axioms = [a.strip() for a in m_deps.group(2).split(",") if a.strip()]
            if not axioms:
                errors.append(f"line {lineno}: empty axiom bracket for 'RsiboxFormal.{thm}'")
            for ax in axioms:
                if ax not in ALLOWED_AXIOMS:
                    errors.append(f"line {lineno}: disallowed axiom '{ax}' for 'RsiboxFormal.{thm}'")

    if not seen:
        errors.append("no '#print axioms RsiboxFormal.<thm>' output lines found")
    for thm in theorems:
        if thm not in seen:
            errors.append(f"missing '#print axioms' output for 'RsiboxFormal.{thm}'")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--verify-axioms-output",
        type=Path,
        default=None,
        help="Path to captured `lake env lean VerifyAxioms.lean` output, or `-` for stdin",
    )
    args = parser.parse_args(argv)
    if args.verify_axioms_output is not None:
        raw = sys.stdin.read() if str(args.verify_axioms_output) == "-" else args.verify_axioms_output.read_text(encoding="utf-8")
        errors = verify_axioms_output(raw, args.root)
    else:
        errors = check_lean_axioms(args.root)
    print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr) if errors else print(
        f"OK: Lean axiom gate passed in {args.root}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
