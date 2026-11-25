# SOP 4: DAG Merge Train & Post-Batch Site Sync

1. **Verify Order**: Start at bottom-most PR (`baseRefName == main`, non-Draft, `Depends on` resolved).
2. **Wait Clean CI**: Run `python3 .agents/plugins/rstack/scripts/pr_gate.py wait-ci --pr <N> --timeout 600`; confirm unresolved review threads == 0 and `GATE1_PASS` + `GATE2_PASS`.
3. **Squash & Advance Child PRs**: `gh pr merge <N> --squash --delete-branch`; rebase child PRs onto updated `origin/main` and inspect `git patch-id` (only re-run Gate 2 if code delta changed).
4. **Release & Sync**: Run `claim_board.py release --done --github`, prune temporary worktrees, and run `facts` (`verify_facts.py --strict` + `build_site.py --check`) once per merged batch.
