---
name: train
description: Execute bottom-up DAG PR merges onto main using merge_train.py and pr_gate.py, verify rebase patch-ids, release claim leases, and prune worktrees.
---

# DAG Merge Train & Cleanup

1. Compute topological order and ready PRs: `python3 .agents/plugins/rstack/scripts/merge_train.py plan`.
2. Wait for clean CI on the lowest ready PR (`baseRefName == main`, non-Draft): `python3 .agents/plugins/rstack/scripts/pr_gate.py wait-ci --pr <N> --timeout 600`.
3. Squash-merge (`gh pr merge <N> --squash --delete-branch`), release its lease (`claim_board.py release --issue <issue> --done --github`), and prune clean merged worktrees (`merge_train.py worktree-gc --apply`).
4. For dependent child PRs: retarget to `main`, `git rebase origin/main`, and compare `python3 .agents/plugins/rstack/scripts/merge_train.py patch-id --base origin/main --head HEAD` (re-run Gate 2 if code delta changed).
