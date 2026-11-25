# rstack Worker

Implement one claimed issue on a fresh branch and worktree off `origin/main` (or explicit `Depends on #N` parent branch).

## Protocol
1. Run `python3 .agents/plugins/rstack/scripts/claim_board.py acquire --issue <N> --owner <id>` before editing.
2. Stay within a single boundary (`<= 600` non-test LOC, zero cross-engine imports); update `.tla` + conformance if touching pool/coordinator/swap/autoscaler state machines.
3. Verify locally with `python3 .agents/plugins/rstack/scripts/preflight.py` and `python3 .agents/plugins/rstack/scripts/pre_push.py --strict`.
4. Return a Delivery Receipt listing: branch/SHA, exact commands run, exact pass/skip counts, `Not verified` gaps, and any `tests` not executed. Never self-approve.
