# rstack — In-Repo RSIBox Engineering Stack

`rstack` turns RSIBox workflow rules into deterministic Python stdlib gate scripts (`scripts/`), always-on guardrails (`rules/AGENTS.md`), specialized review roles (`agents/`), and composable workflow skills (`skills/`).

## Deterministic Gate CLIs (`scripts/`)
- `claim_board.py acquire|release|status|gc` — atomic `.lab/claims/issue-<N>.lock` leases + optional GitHub label sync.
- `preflight.py --base origin/main --max-loc 600` — checks baseline ancestry, non-test LOC cap, single-engine scope, and cross-engine imports.
- `pre_push.py --strict` — runs PR title check, AI attribution scan, `.github/scripts/leak-check.sh`, Ruff (`F,B008,B023,S602`), PR body contract, and `-W error::ResourceWarning` tests.
- `pr_gate.py validate-body|wait-ci` — verifies `.github/pull_request_template.md` and waits for `mergeStateStatus == CLEAN`.
- `trajectory_audit.py` — cross-checks agent claims against real `transcript.jsonl` commands, skips, and SHAs.

## Skills & SOP Entry Points
- `/rsibox-mode`: front-door router with the 12 RSIBox principles and SOP 1–4 dispatch.
- SOP 1 (Feature / Fix): `/rsibox-claim` -> `/rsibox-build` -> `/rsibox-precheck`.
- SOP 2 (GPU Receipt): `/rsibox-gpu-receipt`.
- SOP 3 (Review Triangle): `/rsibox-audit` -> `/rsibox-triage`.
- SOP 4 (Merge & Sync): `/rsibox-merge-train` -> `/rsibox-facts-sync`.
