#!/usr/bin/env python3
"""Pre-flight check for baseline freshness, single engine boundary, and LOC cap."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, find_repo_root, run_cmd

CROSS_ENGINE_IMPORT_RE = re.compile(r"^\+\s*(?:from|import)\s+repos\.([a-z0-9_]+)\b")


def _is_test_path(path: str) -> bool:
    name = Path(path).name
    return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{path}"


def _changed_numstat(repo: Path, base: str) -> list[tuple[int, str]]:
    res = run_cmd(["git", "diff", "--numstat", f"{base}...HEAD"], cwd=repo, timeout=20)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or f"git diff failed against {base}")
    rows: list[tuple[int, str]] = []
    for line in res.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            rows.append((int(parts[0]) if parts[0].isdigit() else 0, parts[2]))
    return rows


def _cross_engine_hits(repo: Path, base: str) -> list[str]:
    res = run_cmd(["git", "diff", "-U0", f"{base}...HEAD"], cwd=repo, timeout=20)
    hits, cur_file = [], ""
    for line in res.stdout.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:]
        elif line.startswith("+") and not line.startswith("+++") and cur_file.endswith(".py"):
            m = CROSS_ENGINE_IMPORT_RE.match(line)
            if m:
                parts, target = Path(cur_file).parts, m.group(1)
                if parts[:1] == ("core",) or (len(parts) >= 2 and parts[0] == "repos" and parts[1] != target):
                    hits.append(f"{cur_file}: imports repos.{target}")
    return hits


def run_preflight(
    repo: Path, base: str, max_loc: int, allow_behind: bool, allow_multi_engine: bool
) -> tuple[bool, dict[str, object]]:
    """Run all preflight invariants and return (ok, report_dict)."""
    ancestor_ok = run_cmd(["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=repo, timeout=10).returncode == 0
    rows = _changed_numstat(repo, base)
    non_test_loc = sum(added for added, p in rows if not _is_test_path(p))
    engines = sorted({Path(p).parts[1] for _, p in rows if len(Path(p).parts) >= 2 and Path(p).parts[0] == "repos"})
    cross_imports = _cross_engine_hits(repo, base)
    dirty_files = [ln for ln in run_cmd(["git", "status", "--porcelain"], cwd=repo, timeout=10).stdout.splitlines() if ln.strip()]

    failures: list[str] = []
    if not ancestor_ok and not allow_behind:
        failures.append(f"HEAD is behind {base}; rebase onto {base}")
    if non_test_loc > max_loc:
        failures.append(f"non-test added LOC {non_test_loc} exceeds cap {max_loc}")
    if len(engines) > 1 and not allow_multi_engine:
        failures.append(f"PR touches multiple engines: {engines}")
    if cross_imports:
        failures.append(f"cross-engine imports found: {cross_imports}")

    return not failures, {
        "base": base,
        "ancestor_of_head": ancestor_ok,
        "non_test_added_loc": non_test_loc,
        "max_loc": max_loc,
        "engines_touched": engines,
        "cross_engine_imports": cross_imports,
        "dirty_count": len(dirty_files),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSIBox single-boundary preflight check")
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--max-loc", type=int, default=600)
    parser.add_argument("--allow-behind", action="store_true")
    parser.add_argument("--allow-multi-engine", action="store_true")
    args = parser.parse_args(argv)
    ok, details = run_preflight(find_repo_root(args.repo), args.base, args.max_loc, args.allow_behind, args.allow_multi_engine)
    return emit(ok, "preflight", **details)


if __name__ == "__main__":
    raise SystemExit(main())
