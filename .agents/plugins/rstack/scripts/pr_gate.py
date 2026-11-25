#!/usr/bin/env python3
"""PR body contract validator and deterministic CI status rollup waiter."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, find_repo_root, run_cmd, which
from pre_push import check_body_contract

FAIL_CONCLUSIONS = {"FAILURE", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE"}


def evaluate_rollup(pr_json: dict[str, Any]) -> tuple[str, dict[str, list[str]]]:
    """Classify PR statusCheckRollup into ('green'|'pending'|'failed', counts).

    Treats conclusion in {SUCCESS, NEUTRAL, SKIPPED} as green even when
    GitHub GraphQL caches a stale status == 'IN_PROGRESS'.
    """
    passed: list[str] = []
    pending: list[str] = []
    failed: list[str] = []
    for item in pr_json.get("statusCheckRollup") or []:
        name = str(item.get("name") or item.get("context") or "unknown")
        conclusion = str(item.get("conclusion") or item.get("state") or "").upper()
        status = str(item.get("status") or "").upper()
        if conclusion in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
            passed.append(name)
        elif conclusion in FAIL_CONCLUSIONS:
            failed.append(f"{name}:{conclusion}")
        elif not conclusion and status in {"", "QUEUED", "IN_PROGRESS", "PENDING"}:
            pending.append(name)
        else:
            pending.append(f"{name}:{conclusion or status}")
    state = "failed" if failed else ("pending" if pending or pr_json.get("mergeStateStatus") != "CLEAN" else "green")
    return state, {"passed": passed, "pending": pending, "failed": failed}


def cmd_wait_ci(args: argparse.Namespace, repo: Path) -> int:
    """Wait deterministically until all PR CI checks pass and mergeStateStatus is CLEAN."""
    if not which("gh"):
        return emit(False, "pr_wait_ci", reason="gh_missing")
    deadline = time.monotonic() + args.timeout
    while True:
        res = run_cmd(
            ["gh", "pr", "view", str(args.pr), "--json", "state,isDraft,baseRefName,mergeStateStatus,statusCheckRollup"],
            cwd=repo,
            timeout=20,
        )
        if res.returncode != 0:
            return emit(False, "pr_wait_ci", pr=args.pr, reason=res.stderr.strip())
        pr_json = json.loads(res.stdout)
        state, counts = evaluate_rollup(pr_json)
        summary = {
            "pr": args.pr,
            "mergeStateStatus": pr_json.get("mergeStateStatus"),
            "state": state,
            "passed_count": len(counts["passed"]),
            "pending": counts["pending"],
            "failed": counts["failed"],
        }
        if state in {"green", "failed"}:
            return emit(state == "green", "pr_wait_ci", **summary)
        if time.monotonic() >= deadline:
            return emit(False, "pr_wait_ci", reason="timeout", **summary)
        time.sleep(args.poll)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSIBox PR body & CI gate")
    parser.add_argument("--repo", type=Path, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    vb = sub.add_parser("validate-body")
    vb.add_argument("--body-file", type=Path, required=True)
    vb.add_argument("--require-gpu-fields", action="store_true")
    wc = sub.add_parser("wait-ci")
    wc.add_argument("--pr", type=int, required=True)
    wc.add_argument("--timeout", type=int, default=600)
    wc.add_argument("--poll", type=int, default=15)
    args = parser.parse_args(argv)
    repo = find_repo_root(args.repo)
    if args.cmd == "validate-body":
        errors = check_body_contract(args.body_file.read_text(encoding="utf-8"), require_gpu=args.require_gpu_fields)
        return emit(not errors, "pr_validate_body", errors=errors)
    return cmd_wait_ci(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
