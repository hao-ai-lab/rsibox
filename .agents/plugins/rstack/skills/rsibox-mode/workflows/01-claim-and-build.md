# SOP 1: Claim & Single-Boundary Build

1. **Acquire Claim**: Invoke `rsibox-claim` (`claim_board.py acquire --issue <N> --owner <agent> --github`) and branch a clean worktree from `origin/main` (or parent branch if stacked `Depends on #<M>`).
2. **Build in Scope**: Invoke `rsibox-build`. Keep `non_test_added_loc <= 600` (`preflight.py`), zero cross-engine imports, and update `.tla` + conformance if touching state machines.
3. **Run Local Blockers**: Invoke `rsibox-precheck` (`pre_push.py --strict --title "<type>(<scope>): <summary>" --body-file <pr_body.md>`). Fix all failures before `git push`.
4. **Open PR**: Fill `.github/pull_request_template.md` (`## What and why`, `## Evidence`, `## Loop record` + 3 GPU bullets). Set Draft + `Depends on #<M>` if parent PR is unmerged.
