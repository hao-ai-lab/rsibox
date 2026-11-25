# rstack Permanent Guardrails

- Follow `/rsibox-mode` for multi-step work in `hao-ai-lab/rsibox`.
- Enforce deterministic gates via `.agents/plugins/rstack/scripts/` (`claim_board.py`, `preflight.py`, `pre_push.py`, `pr_gate.py`, `trajectory_audit.py`) instead of ad-hoc inline polling or verbal checklists.
- Hard blockers: zero AI attribution trailers, zero internal host/path/secret leaks (`leak-check.sh`), `<= 600` non-test LOC per PR, single engine boundary per PR, and `ResourceWarning` treated as error.
- Authors never self-approve: every code/doc PR passes Gate 1 (`rstack-gate1-auditor` running `trajectory_audit.py`) and Gate 2 (`rstack-gate2-adversary` in an isolated read-only context) before merge.
- CPU tests only prove shape and protocol; never mark GPU execution verified without multi-shape fresh-process evidence, non-zero witness checks, cold/warm separation, and `Not verified` honesty.
- Never run `pkill -f` on shared GPU nodes; release atomic leases and temporary worktrees when finished.
