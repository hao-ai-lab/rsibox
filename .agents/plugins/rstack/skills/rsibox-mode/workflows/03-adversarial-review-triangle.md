# SOP 3: Adversarial Review Triangle & Triage

1. **Delivery Receipt**: Worker submits commit SHA, exact commands executed, pass/skip counts, and `Not verified` list.
2. **Gate 1 (`rstack-gate1-auditor`)**: Run `trajectory_audit.py --transcript <path> --require-sha <sha> --claim-file <receipt>` plus `pre_push.py` hygiene and GPU honesty checks.
3. **Gate 2 (`rstack-gate2-adversary`)**: Inspect `git diff origin/main...HEAD` in an isolated read-only context using an independent model; output only evidence-backed `P0 / P1 / P2` findings.
4. **Triage (`rsibox-triage`)**: Fix every reproducing `P0/P1` with a regression test, log `P2` to GitHub issues, and stop after 3 rounds (`AGENTS.md` rule).
