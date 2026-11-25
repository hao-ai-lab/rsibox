---
name: audit
description: Run Gate 1 (trajectory_audit.py against transcript.jsonl and pre_push hygiene) and Gate 2 (independent adversarial code review) on a candidate PR or SHA.
---

# Adversarial Audit Triangle

1. Never accept a worker summary without Gate 1 and Gate 2 passing (`author != approver`).
2. Gate 1 (`rstack-gate1-auditor`): run `python3 .agents/plugins/rstack/scripts/trajectory_audit.py --transcript <transcript.jsonl> --require-sha <sha> --claim-file <receipt>` and `pre_push.py` body/hygiene checks.
3. Gate 2 (`rstack-gate2-adversary`): spawn an isolated read-only reviewer (preferably an independent model family) inspecting `git diff origin/main...HEAD`.
4. Require explicit `GATE1_PASS` and `GATE2_PASS` before entering `train`.
