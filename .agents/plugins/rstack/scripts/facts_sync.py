#!/usr/bin/env python3
"""Post-batch verification wrapper for site/facts consistency and dashboard data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, find_repo_root, run_cmd


def run_facts_sync(repo: Path, skip_tests: bool = False) -> tuple[bool, list[dict[str, Any]]]:
    """Run verify_facts, build_site check, scripts unit tests, and matrix JSON parse."""
    checks: list[dict[str, Any]] = []
    verify_py = repo / "scripts" / "verify_facts.py"
    if verify_py.is_file():
        res = run_cmd([sys.executable, str(verify_py), "--strict"], cwd=repo, timeout=60)
        checks.append({"name": "verify_facts", "ok": res.returncode == 0, "detail": (res.stdout + res.stderr).strip()[:300]})

    build_py = repo / "scripts" / "build_site.py"
    if build_py.is_file():
        res = run_cmd([sys.executable, str(build_py), "--check"], cwd=repo, timeout=60)
        checks.append({"name": "build_site", "ok": res.returncode == 0, "detail": (res.stdout + res.stderr).strip()[:300]})

    scripts_dir = repo / "scripts"
    if not skip_tests and scripts_dir.is_dir() and any(scripts_dir.glob("test_*.py")):
        res = run_cmd(
            [sys.executable, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py", "-v"],
            cwd=repo,
            timeout=120,
        )
        checks.append({"name": "scripts_tests", "ok": res.returncode == 0, "detail": (res.stdout + res.stderr).strip()[-300:]})

    matrix_json = repo / "site" / "facts" / "delivery-matrix.json"
    if matrix_json.is_file():
        try:
            payload = json.loads(matrix_json.read_text(encoding="utf-8"))
            ok_matrix = isinstance(payload, dict) and bool(payload)
            checks.append({"name": "delivery_matrix_json", "ok": ok_matrix, "detail": f"keys={len(payload)}" if isinstance(payload, dict) else "not_dict"})
        except json.JSONDecodeError as exc:
            checks.append({"name": "delivery_matrix_json", "ok": False, "detail": str(exc)})

    return bool(checks) and all(bool(c["ok"]) for c in checks), checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSIBox site/facts verification gate")
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    repo = find_repo_root(args.repo)
    ok, checks = run_facts_sync(repo, skip_tests=args.skip_tests)
    return emit(ok, "facts_sync", checks=checks)


if __name__ == "__main__":
    raise SystemExit(main())
