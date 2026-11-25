---
name: rsibox
description: Front-door engineering router for RSIBox (rstack). Dispatches tasks across SOP 1-4 playbooks, atomic issue leases, deterministic pre-push/CI gates, GPU witness honesty, and adversarial review triangles.
---

# RSIBox Engineering Mode (`rstack`)

Route every non-trivial RSIBox task into the matching SOP playbook and enforce exit-0 deterministic scripts under `.agents/plugins/rstack/scripts/`.

## Inline Principles Index
1. `one-boundary-600-loc`: One engine/module boundary and `<= 600` non-test LOC per PR (`preflight.py`).
2. `framework-artifact-loop-split`: Separate core framework, engine adapters, and experiment loops cleanly.
3. `gpu-oracle-honesty`: CPU tests only prove protocol shape; state `Not verified` gaps honestly.
4. `witness-before-victory`: GPU claims require real witness outputs (anti-zero) across `>= 2` shapes.
5. `cold-warm-separation`: Never merge JIT/compile cold startup with warm steady-state speedups.
6. `formal-tla-first`: Pool/coordinator/swap/autoscaler changes require `.tla` + conformance + TLC pass.
7. `deterministic-gate-over-prose`: Prefer blocking CLI exit codes over prose reminders (`pre_push.py`).
8. `zero-trust-trajectory`: Audit `transcript.jsonl` via `trajectory_audit.py` before accepting worker claims.
9. `no-slop-or-cryptic-abbrev`: Reject dead stubs, cryptic naming, or hidden fallbacks.
10. `zero-env-leak`: Zero internal hosts, paths, IPs, tokens, or AI trailers (`leak-check.sh`).
11. `main-base-patch-id-train`: Bottom-up squash merge onto `main` with `git patch-id` child verification.
12. `single-facts-truth`: Keep `delivery-matrix.json` and docs sync isolated in post-batch `docs(site):` PRs.

## SOP Routing Table
| Task Type | Playbook / Workflow | Skills & Gates |
|---|---|---|
| Feature / Bugfix / Engine | [01-claim-and-build.md](workflows/01-claim-and-build.md) | `claim` -> `build` -> `precheck` |
| GPU Benchmark / Receipt | [02-gpu-receipt-and-witness.md](workflows/02-gpu-receipt-and-witness.md) | `receipt` |
| PR Review / Triage | [03-adversarial-review-triangle.md](workflows/03-adversarial-review-triangle.md) | `audit` -> `triage` |
| Stacked Merge / Site Sync | [04-dag-merge-and-site-sync.md](workflows/04-dag-merge-and-site-sync.md) | `train` -> `facts` |

## Communication & Improvement Skills
- `/plain`: restate technical docs, reviews, or metrics in plain human language.
- `/distill`: mine active conversation lessons into `rstack` skill updates or deterministic gates.
- `/report`: format substantive code/doc changes with 1-line file summaries and verbatim `Before -> After` excerpts inline.
- `/polish`: keep public docs plain-English and aligned with `docs_lint.py` + `facts_sync.py`.
