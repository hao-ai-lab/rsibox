#!/usr/bin/env python3
"""Validate TLA+ spec-to-Python symbol anchors in anchors.tsv."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys

if __package__:
    from .assumptions_check import strip_tla_comments
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from assumptions_check import strip_tla_comments

FORMAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ANCHORS = FORMAL_DIR / "anchors.tsv"


def _safe_repo_path(rel: str, repo_root: Path) -> Path | None:
    if Path(rel).is_absolute():
        return None
    resolved = (repo_root / rel).resolve()
    return resolved if resolved.is_relative_to(repo_root.resolve()) else None


def _has_py_symbol(py_path: Path, symbol: str) -> bool:
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    if "." in symbol:
        cls_name, meth_name = symbol.split(".", 1)
        return any(
            isinstance(n, ast.ClassDef) and n.name == cls_name and any(
                isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == meth_name for m in n.body
            )
            for n in tree.body
        )
    return any(isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol for n in tree.body)


def _has_tla_symbol(tla_path: Path, symbol: str) -> bool:
    cleaned = strip_tla_comments(tla_path.read_text(encoding="utf-8"))
    return bool(re.search(rf"(?m)^\s*{re.escape(symbol)}(?:\([^)]*\))?\s*==", cleaned))


def check_anchors(anchors_path: Path, repo_root: Path) -> list[str]:
    if not anchors_path.is_file():
        return [f"missing anchors file: {anchors_path}"]
    errors, rows, header_checked = [], 0, False
    for lineno, raw in enumerate(anchors_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in raw.split("\t")]
        if not header_checked:
            header_checked = True
            if cols[:4] == ["spec_file", "spec_symbol", "code_file", "code_symbol"]:
                continue
        if len(cols) < 4:
            errors.append(f"{anchors_path}:{lineno}: expected >=4 tab-separated columns")
            continue
        rows += 1
        spec_rel, spec_sym, code_rel, code_sym = cols[:4]
        spec_path, code_path = _safe_repo_path(spec_rel, repo_root), _safe_repo_path(code_rel, repo_root)
        if spec_path is None or not spec_path.is_file() or not _has_tla_symbol(spec_path, spec_sym):
            errors.append(f"line {lineno}: missing TLA+ symbol {spec_sym} in {spec_rel}")
        if code_path is None or not code_path.is_file() or not _has_py_symbol(code_path, code_sym):
            errors.append(f"line {lineno}: missing Python symbol {code_sym} in {code_rel}")
    return errors if rows else errors + [f"no anchor rows in {anchors_path}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--repo", type=Path, default=FORMAL_DIR.parents[2])
    args = parser.parse_args(argv)
    errors = check_anchors(args.anchors, args.repo)
    print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr) if errors else print(
        f"OK: verified anchors in {args.anchors}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
