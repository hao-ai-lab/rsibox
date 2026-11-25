---
name: rsibox-merge-train
description: Execute bottom-up DAG PR merges onto main using pr_gate.py wait-ci, rebase dependent PRs with git patch-id verification, release claim leases, and prune worktrees.
---

# DAG Merge Train & Cleanup

1. Pick the lowest unblocked PR (`baseRefName == main`, non-Draft, `Depends on` satisfied).
2. Run `python3 .agents/plugins/rstack/scripts/pr_gate.py wait-ci --pr <N> --timeout 600` until `mergeStateStatus == CLEAN`.
3. Squash-merge (`gh pr merge <N> --squash --delete-branch`), release its lease (`claim_board.py release --issue <issue> --done --github`), and remove its temporary worktree.
4. For each dependent child PR: retarget to `main`, `git rebase origin/main`, and compare `git patch-id` before vs after rebase (re-run Gate 2 if code delta changed).
