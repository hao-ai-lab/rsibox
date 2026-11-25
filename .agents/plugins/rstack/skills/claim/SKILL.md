---
name: claim
description: Acquire an atomic RSIBox issue lease with claim_board.py, create an isolated git worktree off origin/main, and run initial preflight checks.
---

# Claim & Worktree Setup

1. Run `python3 .agents/plugins/rstack/scripts/claim_board.py acquire --issue <N> --owner <agent> --branch <branch> --scope <paths> --github`. Abort if exit code != 0.
2. Create an isolated worktree: `git fetch origin main && git worktree add -b <branch> ../wt-<slug> origin/main`.
3. If depending on an unmerged PR `#M`, branch from that PR tip, open as Draft, and put `Depends on #<M>` on line 1 of the PR body.
4. Inspect `.lab/claims/` via `claim_board.py status` or `claim_board.py gc` to clear expired leases only.
