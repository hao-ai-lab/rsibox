# rstack — In-Repo RSIBox Engineering Stack

`rstack` turns RSIBox workflow rules into deterministic Python stdlib gate scripts (`scripts/`), always-on guardrails (`rules/AGENTS.md`), specialized review roles (`agents/`), and composable workflow skills (`skills/`).

## Deterministic Gate CLIs (`scripts/`)
- `claim_board.py acquire|release|status|gc` — atomic `.lab/claims/issue-<N>.lock` leases + optional GitHub label sync.
- `preflight.py --base origin/main --max-loc 600` — checks baseline ancestry, non-test LOC cap, single-engine scope, and cross-engine imports.
- `pre_push.py --strict` — runs PR title check, AI attribution scan, `.github/scripts/leak-check.sh`, Ruff (`F,B008,B023,S602`), PR body contract, and `-W error::ResourceWarning` tests.
- `pr_gate.py validate-body|wait-ci` — verifies `.github/pull_request_template.md` and waits for `mergeStateStatus == CLEAN`.
- `merge_train.py plan|patch-id|worktree-gc` — computes bottom-up DAG merge order (`Depends on #N`), verifies rebase `patch-id` equivalence, and safely cleans merged/detached secondary worktrees.
- `gpu_receipt_close.py validate|backlog` — enforces multi-shape non-zero witness receipts with cold/warm separation and creates `status:needs-gpu` backlog issues.
- `facts_sync.py` — validates `verify_facts.py --strict`, `build_site.py --check`, `scripts/test_*.py`, and `site/facts/delivery-matrix.json`.
- `docs_lint.py` — checks top-level and core markdown docs for broken relative links, invalid heading anchors, and duplicate parenthetical artifacts.
- `trajectory_audit.py` — cross-checks agent claims against real `transcript.jsonl` commands, skips, and SHAs.

## Skills & SOP Entry Points
- `/rsibox`: front-door router with the 12 RSIBox principles and SOP 1–4 dispatch.
- `/polish`: plain-English public doc polish paired with `docs_lint.py` and `facts_sync.py`.
- `/plain`: restate technical docs, reviews, or metrics in plain human language with zero jargon.
- `/distill`: mine active session transcripts for durable lessons routed to `rstack` skills or gates.
- `/report`: inline visual change-report format (1-line file summary + verbatim `Before -> After` excerpts).
- `/formal`: run TLC verification, mutants/witnesses, spec pin/anchor/assumption gates, and Mermaid trace rendering.
- SOP 1 (Feature / Fix): `/claim` -> `/build` -> `/precheck`.
- SOP 2 (GPU Receipt): `/receipt`.
- SOP 3 (Review Triangle): `/audit` -> `/triage`.
- SOP 4 (Merge & Sync): `/train` -> `/facts`.
