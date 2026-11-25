#!/usr/bin/env python3
"""Atomic issue claim board using mkdir leases and optional GitHub label sync."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, find_repo_root, load_json, run_cmd, save_json_atomic, which


def _lock_dir(repo: Path, issue: int) -> Path:
    return repo / ".lab" / "claims" / f"issue-{issue}.lock"


def _sync_github(repo: Path, issue: int, add: list[str], remove: list[str], note: str) -> dict[str, str]:
    if not which("gh"):
        return {"github": "skipped (gh not on PATH)"}
    cmd = ["gh", "issue", "edit", str(issue)]
    for lbl in add:
        cmd.extend(["--add-label", lbl])
    for lbl in remove:
        cmd.extend(["--remove-label", lbl])
    ed = run_cmd(cmd, cwd=repo, timeout=20)
    cm = run_cmd(["gh", "issue", "comment", str(issue), "--body", note], cwd=repo, timeout=20)
    return {
        "github_edit": "ok" if ed.returncode == 0 else ed.stderr.strip(),
        "github_comment": "ok" if cm.returncode == 0 else cm.stderr.strip(),
    }


def cmd_acquire(args: argparse.Namespace, repo: Path) -> int:
    """Acquire an atomic issue lease, failing on active foreign ownership."""
    lock, now = _lock_dir(repo, args.issue), int(time.time())
    lock.parent.mkdir(parents=True, exist_ok=True)
    lease_file = lock / "lease.json"
    try:
        os.mkdir(lock)
    except FileExistsError:
        existing = load_json(lease_file, default={}) or {}
        expired = int(existing.get("expires_at", 0)) <= now
        if existing.get("owner") != args.owner and not (expired and args.force_stale):
            return emit(False, "claim_acquire", issue=args.issue, reason="already_claimed", expired=expired, lease=existing)
    lease = {
        "issue": args.issue,
        "owner": args.owner,
        "scope": args.scope,
        "branch": args.branch,
        "pid": os.getpid(),
        "created_at": now,
        "expires_at": now + args.ttl,
    }
    save_json_atomic(lease_file, lease)
    gh_info = (
        _sync_github(
            repo, args.issue, ["status:claimed"], ["status:ready"],
            f"rstack claim acquired by `{args.owner}` (branch: `{args.branch}`, scope: `{args.scope}`).",
        )
        if args.github else {}
    )
    return emit(True, "claim_acquire", lease=lease, **gh_info)


def cmd_release(args: argparse.Namespace, repo: Path) -> int:
    """Release an existing issue lease cleanly."""
    lock = _lock_dir(repo, args.issue)
    if not lock.exists():
        return emit(True, "claim_release", issue=args.issue, status="not_claimed")
    existing = load_json(lock / "lease.json", default={}) or {}
    if args.owner and existing.get("owner") != args.owner and not args.force:
        return emit(False, "claim_release", issue=args.issue, reason="owner_mismatch", lease=existing)
    shutil.rmtree(lock)
    gh_info = (
        _sync_github(
            repo, args.issue, [] if args.done else ["status:ready"], ["status:claimed"],
            f"rstack claim released by `{args.owner or existing.get('owner', 'unknown')}`.",
        )
        if args.github else {}
    )
    return emit(True, "claim_release", issue=args.issue, status="released", **gh_info)


def cmd_status_or_gc(repo: Path, issue: int | None, prune: bool) -> int:
    """List current claims or prune expired claim leases."""
    now, cdir = int(time.time()), repo / ".lab" / "claims"
    items, removed = [], []
    if cdir.exists():
        for entry in sorted(cdir.iterdir()):
            if not (entry.is_dir() and entry.name.endswith(".lock")):
                continue
            lease = load_json(entry / "lease.json", default={}) or {}
            expired = int(lease.get("expires_at", 0)) <= now
            if prune and expired:
                shutil.rmtree(entry)
                if isinstance(lease.get("issue"), int):
                    removed.append(lease["issue"])
            elif not prune and (issue is None or lease.get("issue") == issue):
                lease["expired"] = expired
                items.append(lease)
    return emit(True, "claim_gc" if prune else "claim_status", **({"removed_issues": removed} if prune else {"claims": items}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSIBox atomic issue claim board")
    parser.add_argument("--repo", type=Path, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    acq = sub.add_parser("acquire")
    for fl, kw in [("--issue", {"type": int, "required": True}), ("--owner", {"required": True}), ("--scope", {"default": ""}), ("--branch", {"default": ""}), ("--ttl", {"type": int, "default": 14400}), ("--force-stale", {"action": "store_true"}), ("--github", {"action": "store_true"})]:
        acq.add_argument(fl, **kw)
    rel = sub.add_parser("release")
    for fl, kw in [("--issue", {"type": int, "required": True}), ("--owner", {"default": ""}), ("--force", {"action": "store_true"}), ("--done", {"action": "store_true"}), ("--github", {"action": "store_true"})]:
        rel.add_argument(fl, **kw)
    st = sub.add_parser("status")
    st.add_argument("--issue", type=int, default=None)
    sub.add_parser("gc")
    args = parser.parse_args(argv)
    repo = find_repo_root(args.repo)
    if args.cmd == "acquire":
        return cmd_acquire(args, repo)
    if args.cmd == "release":
        return cmd_release(args, repo)
    return cmd_status_or_gc(repo, getattr(args, "issue", None), prune=(args.cmd == "gc"))


if __name__ == "__main__":
    raise SystemExit(main())
