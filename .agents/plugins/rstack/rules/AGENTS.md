# rstack Permanent Guardrails

- Follow `/rsibox` for multi-step work in `hao-ai-lab/rsibox`.
- Follow `/report` when summarizing substantive doc/code edits or reviews: never hide changes behind bare file paths — print a 1-line description for each key file immediately followed by verbatim `Before -> After` excerpts inline on screen.
- Always refer to conformance checks as **the sandbox verification suite** (`rsibox conformance`); never hardcode a fixed case count in headings or rules (report fractions like `11/11` only when quoting a specific observed run).
- Enforce deterministic gates via `.agents/plugins/rstack/scripts/` (`claim_board.py`, `preflight.py`, `pre_push.py`, `pr_gate.py`, `trajectory_audit.py`) instead of ad-hoc inline polling or verbal checklists.
- Hard blockers: zero AI attribution trailers, zero internal host/path/secret leaks (`leak-check.sh`), `<= 600` non-test LOC per PR, single engine boundary per PR, and `ResourceWarning` treated as error.
- Authors never self-approve: every code/doc PR passes Gate 1 (`rstack-gate1-auditor` running `trajectory_audit.py`) and Gate 2 (`rstack-gate2-adversary` in an isolated read-only context) before merge.
- CPU tests only prove shape and protocol; never mark GPU execution verified without multi-shape fresh-process evidence, non-zero witness checks, cold/warm separation, and `Not verified` honesty.
- Never run `pkill -f` on shared GPU nodes; release atomic leases and temporary worktrees when finished.
