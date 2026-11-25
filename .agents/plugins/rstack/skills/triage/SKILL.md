---
name: triage
description: Triage Gate 1/2 and PR review findings into P0/P1 (fix with regression test) or P2 (file tracking issue), enforcing the 3-round stop rule.
---

# Review Triage & Fix Loop

1. Classify every verified finding with file/line or test evidence: `P0` (critical corruption/loss/security), `P1` (functional/race/contract bug), or `P2` (low-severity debt).
2. `P0` and `P1` block merge: fix in the branch, add a regression test that fails without the fix, and reply `Fixed in <sha>`.
3. Reject false positives only with concrete counter-evidence (`Not a bug: <file:line or test output>`). Link valid `P2` items to GitHub issues.
4. Enforce the 3-round stop rule: after 3 review rounds with remaining disagreements, stop and escalate to the maintainer.
