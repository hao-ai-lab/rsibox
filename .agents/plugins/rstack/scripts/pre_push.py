#!/usr/bin/env python3
"""Pre-push deterministic blocker mirroring RSIBox CI gates locally."""
from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    AI_ATTR_RE,
    PR_TITLE_RE,
    REQUIRED_GPU_KEYS,
    REQUIRED_HEADINGS,
    REQUIRED_LOOP_KEYS,
    emit,
    find_repo_root,
    run_cmd,
    which,
)


def check_body_contract(body_text: str, require_gpu: bool) -> list[str]:
    """Validate PR body headings and mandatory Loop/GPU bullets."""
    errors = [f"missing heading: {h}" for h in REQUIRED_HEADINGS if h not in body_text]
    if require_gpu:
        for key in (*REQUIRED_LOOP_KEYS, *REQUIRED_GPU_KEYS):
            if not re.search(rf"-\s+\*\*{re.escape(key)}\*\*:\s+\S+", body_text):
                errors.append(f"missing or empty bullet: - **{key}**:")
    return errors


def _changed_files(repo: Path, base: str) -> list[str]:
    res = run_cmd(["git", "diff", "--name-only", f"{base}...HEAD"], cwd=repo, timeout=20)
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()] if res.returncode == 0 else []


def run_pre_push(args: argparse.Namespace, repo: Path) -> tuple[bool, list[dict[str, object]]]:
    """Execute pre-push checks in deterministic order."""
    checks: list[dict[str, object]] = []
    changed = _changed_files(repo, args.base)
    body_text = args.body_file.read_text(encoding="utf-8") if args.body_file else ""

    if args.title is not None:
        checks.append({"name": "pr_title", "ok": bool(PR_TITLE_RE.match(args.title)), "detail": args.title})

    log_out = run_cmd(["git", "log", "--format=%B", f"{args.base}..HEAD"], cwd=repo, timeout=20).stdout
    ai_hit = AI_ATTR_RE.search("\n".join([log_out, args.title or "", body_text]))
    checks.append({"name": "ai_attribution", "ok": ai_hit is None, "detail": ai_hit.group(0) if ai_hit else "clean"})

    leak_script = repo / ".github" / "scripts" / "leak-check.sh"
    if leak_script.exists():
        lres = run_cmd(["bash", str(leak_script), args.base, "HEAD"], cwd=repo, timeout=30)
        checks.append({"name": "leak_check", "ok": lres.returncode == 0, "detail": (lres.stdout + lres.stderr).strip()[:400]})

    if args.body_file is not None:
        touches_code = any(p.startswith(("core/livelab/", "repos/")) for p in changed)
        errs = check_body_contract(body_text, require_gpu=touches_code or args.require_gpu_fields)
        checks.append({"name": "pr_body", "ok": not errs, "detail": errs or "ok"})

    py_files = [p for p in changed if p.endswith(".py") and (repo / p).exists()]
    ruff_bin = which("ruff")
    if ruff_bin and py_files:
        rres = run_cmd([ruff_bin, "check", "--select", "F,B008,B023,S602", *py_files], cwd=repo, timeout=60)
        checks.append({"name": "ruff", "ok": rres.returncode == 0, "detail": (rres.stdout + rres.stderr).strip()[:400]})
    elif args.strict and py_files and not ruff_bin:
        checks.append({"name": "ruff", "ok": False, "detail": "ruff missing on PATH under --strict"})

    if not args.skip_tests:
        cmd = shlex.split(args.test_cmd) if args.test_cmd else [
            sys.executable, "-W", "error::ResourceWarning", "-m", "unittest", "discover", "-s", ".agents/plugins/rstack/scripts", "-v"
        ]
        tres = run_cmd(cmd, cwd=repo, timeout=180)
        checks.append({"name": "tests", "ok": tres.returncode == 0, "detail": (tres.stdout + tres.stderr).strip()[-400:]})

    return all(bool(c["ok"]) for c in checks), checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSIBox pre-push gate")
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--title", default=None)
    parser.add_argument("--body-file", type=Path, default=None)
    parser.add_argument("--require-gpu-fields", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--test-cmd", default=None)
    args = parser.parse_args(argv)
    ok, checks = run_pre_push(args, find_repo_root(args.repo))
    return emit(ok, "pre_push", base=args.base, checks=checks)


if __name__ == "__main__":
    raise SystemExit(main())
