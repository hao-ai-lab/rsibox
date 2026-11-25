#!/usr/bin/env python3
"""DAG PR merge train planner, patch-id guard, and safe worktree garbage collector."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, find_repo_root, load_json, run_cmd, which

DEPENDS_RE = re.compile(r"depends\s+on\s+#(\d+)", re.IGNORECASE)


def plan_dag(prs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Build bottom-up topological merge order and classify ready vs blocked PRs."""
    by_num = {int(pr["number"]): pr for pr in prs}
    deps_map: dict[int, list[int]] = {}
    children: dict[int, list[int]] = {n: [] for n in by_num}
    indegree: dict[int, int] = {n: 0 for n in by_num}

    for num, pr in sorted(by_num.items()):
        found = [int(m) for m in DEPENDS_RE.findall(pr.get("body") or "")]
        deps = sorted({d for d in found if d in by_num and d != num})
        deps_map[num] = deps
        indegree[num] = len(deps)
        for dep in deps:
            children[dep].append(num)

    q = deque(sorted(n for n, deg in indegree.items() if deg == 0))
    order: list[int] = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in sorted(children[cur]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    cycle = sorted(set(by_num) - set(order))
    ready = sorted(
        n
        for n, pr in by_num.items()
        if not pr.get("isDraft") and pr.get("baseRefName") == "main" and not deps_map[n]
    )
    blocked = {str(n): deps_map[n] for n in sorted(by_num) if deps_map[n]}
    return not cycle, {"order": order, "ready": ready, "blocked": blocked, "cycle": cycle}


def cmd_plan(args: argparse.Namespace, repo: Path) -> int:
    """Emit topological merge order for open PRs."""
    if args.prs_json:
        prs = load_json(args.prs_json, default=[]) or []
    else:
        if not which("gh"):
            return emit(False, "merge_train_plan", reason="gh_missing")
        res = run_cmd(
            ["gh", "pr", "list", "--state", "open", "--json", "number,title,headRefName,baseRefName,isDraft,body,mergeStateStatus"],
            cwd=repo,
            timeout=20,
        )
        if res.returncode != 0:
            return emit(False, "merge_train_plan", reason=res.stderr.strip())
        prs = json.loads(res.stdout)
    ok, report = plan_dag(prs)
    return emit(ok, "merge_train_plan", **report)


def cmd_patch_id(args: argparse.Namespace, repo: Path) -> int:
    """Compute stable git patch-id for a commit range to check rebase equivalence."""
    diff_res = run_cmd(["git", "diff", f"{args.base}...{args.head}"], cwd=repo, timeout=30)
    if diff_res.returncode != 0:
        return emit(False, "merge_train_patch_id", reason=diff_res.stderr.strip())
    pid_res = subprocess.run(
        ["git", "patch-id", "--stable"],
        cwd=str(repo),
        input=diff_res.stdout,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if pid_res.returncode != 0:
        return emit(False, "merge_train_patch_id", reason=pid_res.stderr.strip())
    patch_id = pid_res.stdout.strip().split()[0] if pid_res.stdout.strip() else "empty"
    return emit(True, "merge_train_patch_id", base=args.base, head=args.head, patch_id=patch_id)


def cmd_worktree_gc(args: argparse.Namespace, repo: Path) -> int:
    """List or prune clean secondary worktrees that are detached or merged into base."""
    out = run_cmd(["git", "worktree", "list", "--porcelain"], cwd=repo, timeout=15)
    if out.returncode != 0:
        return emit(False, "merge_train_worktree_gc", reason=out.stderr.strip())
    candidates: list[str] = []
    removed: list[str] = []
    blocks = [b.strip().splitlines() for b in out.stdout.strip().split("\n\n") if b.strip()]
    for idx, lines in enumerate(blocks):
        info = dict(ln.partition(" ")[::2] for ln in lines)
        wt = Path(info.get("worktree", "")).resolve()
        if idx == 0 or wt == repo.resolve() or not wt.exists():
            continue
        dirty = [x for x in run_cmd(["git", "-C", str(wt), "status", "--porcelain"], timeout=10).stdout.splitlines() if x.strip()]
        if dirty:
            continue
        detached = "branch" not in info
        merged = run_cmd(["git", "-C", str(wt), "merge-base", "--is-ancestor", "HEAD", args.base], timeout=10).returncode == 0
        if detached or merged:
            candidates.append(str(wt))
            if args.apply and run_cmd(["git", "worktree", "remove", str(wt)], cwd=repo, timeout=20).returncode == 0:
                removed.append(str(wt))
    return emit(True, "merge_train_worktree_gc", base=args.base, apply=args.apply, candidates=candidates, removed=removed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSIBox DAG merge train and worktree GC")
    parser.add_argument("--repo", type=Path, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--prs-json", type=Path, default=None)
    pi = sub.add_parser("patch-id")
    pi.add_argument("--base", default="origin/main")
    pi.add_argument("--head", default="HEAD")
    gc = sub.add_parser("worktree-gc")
    gc.add_argument("--base", default="origin/main")
    gc.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    repo = find_repo_root(args.repo)
    if args.cmd == "plan":
        return cmd_plan(args, repo)
    if args.cmd == "patch-id":
        return cmd_patch_id(args, repo)
    return cmd_worktree_gc(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
