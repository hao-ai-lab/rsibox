# Docs

Docs about how this repo and its process work. Nothing engine-specific lives here: an engine's
architecture goes in `repos/<engine>/docs/`, its measurements in `repos/<engine>/kb/`.

```
docs/
  README.md                       this index
  lessons.md                      framework lessons from building an RSIBox sandbox
  kernels-and-graphs.md           bytecode swap, Triton @jit rebuild, CUDA graphs, execution checks, and warm-cache restart taxes
  agent-teams-and-contention.md   multi-agent GPU contention, SandboxPool scaling, and team topology trade-offs
  process/
    developer_workflow.md         Boss-Subagent delegation SOP, model routing, PR bundling, and two-tier reporting
    local_and_self_hosted_ci.md   self-hosted runner routing (RSIBOX_CI_RUNNER) and local commit-status CI
  core/                           how the skill and the agent prompts work
  livelab/                        the sandbox interface, run logs, and verification suite
```

Rules:

- Make a folder only when its first doc arrives. No empty placeholders.
- Docs describe how things are now. Update them in place; no decision logs or changelogs.
- Add every new doc to the index below.

## Index

- [Lessons from building an RSIBox sandbox](lessons.md): framework lessons and rules for applying them.
- [Bytecode swap, kernel rebuild, and CUDA graph replay](kernels-and-graphs.md): why Python bytecode swaps stop at `@triton.jit` and `CUDAGraph` boundaries, how `core/livelab/kernel/jit.py` rebuilds kernels in place with zero-recompile undo, how the two-ledger execution check (`live` vs. `capture`) prevents false positives, what selective graph recapture does today in the vLLM sandbox, the target design (not implemented) for lazy recapture and for updating all ~50 shape buckets at once (`cuGraphExecKernelNodeSetParams`, `cuGraphExecUpdate`), and how warm-cache restart taxes break down across vLLM, Megatron-LM, and FastVideo.
- [Multi-agent GPU contention and agent-team topologies](agent-teams-and-contention.md): empirical NVIDIA B200 study (`110/110` SHA-256 hash-chained `SandboxPool` traces, `2,919` leases across 5 live engines including `TorchTitan` FSDP2) measuring shared-pool live hot-reload vs. warm-cache restarts (`43.7×–54.4×` clean; `399.2` edits/min at `N = 24` on `3× B200` with `3 s` think-time), Dynamo GMS Standby Shadow crash failover (`53.2×–53.6×` on vLLM, `34.51×–38.06×` on SGLang `52` decode CGs, `30.60×` faster multi-shadow Restart #2 post-PR #244/#247), sub-millisecond `SIGKILL` fast-fail (`0.502 ms` vLLM / `0.173 ms` TRT-LLM), rogue-agent circuit-breaking & `max_slots_per_agent=1` blast-radius containment (`12.40×` faster healthy team, `60.58×` lower p95 wait), priority/aging/elastic role-borrowing scheduling (`1.60×–1.65×` Triad / `2.60×` Phase-Shift), and `11.30×` faster full-stack 6-layer pipeline completion under an active rogue attack (`EXP-H1`).
- [Developer and multi-agent workflow SOP](process/developer_workflow.md): Boss–Subagent delegation architecture, mandatory `letitdo` trajectory audit triangle, model routing policy, pre-flight checklist, dependency-ordered PR bundling, P0/P1/P2 review triage, and two-tier empirical reporting standards.
- [Local and self-hosted CI](process/local_and_self_hosted_ci.md): how `.github/workflows/cpu-tests.yml` and `.github/workflows/pr-rules.yml` route between a self-hosted runner (`vars.RSIBOX_CI_RUNNER`) and `ubuntu-latest`, how persistent runner workspaces stay CPU-isolated and idempotent, and how local commit-status CI (`statuses/<sha>`) keeps pull requests verified when GitHub Actions dispatch is paused.
