---
name: rsibox-precheck
description: Run mandatory local CI mirror checks (pre_push.py and pr_gate.py validate-body) before pushing any branch or creating a PR.
---

# Pre-Push Deterministic Gate

1. Draft the PR description in a temp file using `.github/pull_request_template.md` (`## What and why`, `## Evidence`, `## Loop record` + 3 GPU bullets).
2. Run `python3 .agents/plugins/rstack/scripts/preflight.py --base origin/main --max-loc 600`.
3. Run `python3 .agents/plugins/rstack/scripts/pre_push.py --base origin/main --title "<title>" --body-file <tmp_body.md> --strict`.
4. Only `git push` and run `gh pr create` after all checks exit `0`.
