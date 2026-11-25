#!/usr/bin/env python3
"""GPU receipt validator (witness/shapes/cold-warm/leaks) and needs-gpu backlog linker."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, find_repo_root, load_json, run_cmd, which

PRIVATE_IP_RE = re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b")
_SENSITIVE_ROOTS = ("raid", "workspace", "scratch", "home", "Users")
SENSITIVE_PATH_RE = re.compile(rf"(?<![a-zA-Z0-9_]){chr(47)}(?:{'|'.join(_SENSITIVE_ROOTS)}){chr(47)}")


def validate_receipt(data: dict[str, Any], raw_text: str = "") -> list[str]:
    """Verify non-zero witness, replay coverage, >=2 shapes, cold/warm split, and leak hygiene."""
    errors: list[str] = []
    if not data.get("witness_nonzero"):
        errors.append("zero_witness or missing witness_nonzero verification")
    if not data.get("replay_verified"):
        errors.append("capture-only receipt; replay_verified must be true")
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or len({str(s) for s in shapes}) < 2:
        errors.append("must validate >= 2 distinct input shapes")
    for field in ("cold_s", "warm_s"):
        val = data.get(field)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0:
            errors.append(f"missing or invalid non-negative numeric timing: {field}")
    blob = raw_text or str(data)
    if PRIVATE_IP_RE.search(blob):
        errors.append("private IPv4 address leaked in receipt")
    if SENSITIVE_PATH_RE.search(blob):
        errors.append("internal filesystem path leaked in receipt")
    return errors


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a JSON GPU receipt file."""
    if not args.receipt.is_file():
        return emit(False, "gpu_receipt_validate", reason="file_not_found", path=str(args.receipt))
    raw_text = args.receipt.read_text(encoding="utf-8")
    data = load_json(args.receipt, default={})
    if not isinstance(data, dict):
        return emit(False, "gpu_receipt_validate", reason="invalid_json_object")
    errors = validate_receipt(data, raw_text=raw_text)
    return emit(not errors, "gpu_receipt_validate", path=str(args.receipt), errors=errors)


def cmd_backlog(args: argparse.Namespace, repo: Path) -> int:
    """Create or preview a status:needs-gpu issue under GPU Receipts Backlog."""
    body = "\n".join(
        [
            f"Tracked from PR #{args.pr}.",
            f"- Engine: `{args.engine}`",
            f"- Missing GPU verification: {args.summary}",
        ]
    )
    cmd = [
        "gh",
        "issue",
        "create",
        "--title",
        args.title,
        "--label",
        "status:needs-gpu",
        "--milestone",
        "GPU Receipts Backlog",
        "--body",
        body,
    ]
    if args.dry_run:
        return emit(True, "gpu_backlog", dry_run=True, cmd=cmd, body=body)
    if not which("gh"):
        return emit(False, "gpu_backlog", reason="gh_missing")
    res = run_cmd(cmd, cwd=repo, timeout=20)
    return emit(
        res.returncode == 0,
        "gpu_backlog",
        url=res.stdout.strip() if res.returncode == 0 else "",
        error=res.stderr.strip() if res.returncode != 0 else "",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSIBox GPU receipt validator & backlog linker")
    parser.add_argument("--repo", type=Path, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    va = sub.add_parser("validate")
    va.add_argument("--receipt", type=Path, required=True)
    bl = sub.add_parser("backlog")
    bl.add_argument("--title", required=True)
    bl.add_argument("--pr", type=int, required=True)
    bl.add_argument("--engine", default="core")
    bl.add_argument("--summary", default="Requires multi-GPU B200 receipt")
    bl.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "validate":
        return cmd_validate(args)
    return cmd_backlog(args, find_repo_root(args.repo))


if __name__ == "__main__":
    raise SystemExit(main())
