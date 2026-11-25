# Docs

Docs about how this repo and its process work. Nothing engine-specific lives here: an engine's
architecture goes in `repos/<engine>/docs/`, its measurements in `repos/<engine>/kb/`.

```
docs/
  README.md                       this index
  quickstart.md                   30-second CPU and real-GPU live hot-reload walkthroughs
  architecture.html               offline interactive 4-layer architecture map and sequence walkthroughs
  concepts.md                     plain-English overview of rsibox, hot-reload layers, witness, and rollback
  engines.md                      supported ML engines, conformance status, and repository layout
  mechanisms.md                   restart boundaries, Dynamo GMS + Snapshot benchmarks, and CPU tests
  services.md                     shared control-plane services (S1-S3, S6 shipped; S4-S5 planned)
  lessons.md                      framework lessons from building an RSIBox sandbox
  kernels-and-graphs.md           bytecode swap, Triton @jit rebuild, CUDA graphs, execution checks, and warm-cache restart taxes
  agent-teams-and-contention.md   multi-agent GPU contention, SandboxPool scaling, and team topology trade-offs
  process/
    developer_workflow.md         Boss-Subagent delegation SOP, model routing, PR bundling, and two-tier reporting
    local_and_self_hosted_ci.md   self-hosted runner routing (RSIBOX_CI_RUNNER) and local commit-status CI
```

Rules:

- Make a folder only when its first doc arrives. No empty placeholders.
- Docs describe how things are now. Update them in place; no decision logs or changelogs.
- Add every new doc to the index below.

## Index

- [Quick Start](quickstart.md): copy-pasteable 30-second CPU (`sglang` toy) and real-GPU (`vllm --real`) walkthroughs covering `up`, `source`, `install`, `check`, `uninstall`, and `down`.
- [Interactive architecture explorer](architecture.html): offline single-file widget mapping the 4-layer runtime + cross-cutting plane, component ownership and boundaries, and 5 interactive sequence walkthroughs.
- [RSIBox concepts](concepts.md): plain-English overview of `rsibox` vs. `core/livelab/`, hot-reload layers, execution proof (`witness`), journaled rollback, restart boundaries, and the sandbox verification suite.
- [Supported engines & benchmarks](engines.md): summary table across all supported ML engines, headline speedups, and repository layout.
- [Restart boundaries, interface & tests](mechanisms.md): full restart-boundary table, NVIDIA Dynamo GMS & Snapshot integration benchmarks, and CPU test commands.
- [Shared control-plane services (`rsibox.services`)](services.md): unified S1 GPU memory (`MemoryService`), S2 shared zygote (`ZygoteService`), S3 compile/graph cache (`CompileCacheService`), S6 recovery (`RecoveryService`), and S4–S5 roadmap composed beneath `core/livelab/kernel/`.
- [Lessons from building an RSIBox sandbox](lessons.md): framework lessons and rules for applying them.
- [Bytecode swap, kernel rebuild, and CUDA graph replay](kernels-and-graphs.md): why Python bytecode swaps stop at `@triton.jit` and `CUDAGraph` boundaries, how `core/livelab/kernel/jit.py` rebuilds kernels in place with zero-recompile undo, how the two-ledger execution check (`live` vs. `capture`) prevents false positives, what selective graph recapture does today in the vLLM sandbox, the target design (not implemented) for lazy recapture and for updating all ~50 shape buckets at once (`cuGraphExecKernelNodeSetParams`, `cuGraphExecUpdate`), and how warm-cache restart taxes break down across vLLM, Megatron-LM, and FastVideo.
- [Multi-agent GPU contention and agent-team topologies](agent-teams-and-contention.md): empirical NVIDIA B200 study (`112/112` SHA-256 hash-chained `SandboxPool` traces, `3,226` leases across 5 live engines including `TorchTitan` FSDP2) evaluating shared-pool live hot-reload vs. restarts (`43.7×–54.4×` clean, `399.2` edits/min at `N = 24`), Dynamo GMS Standby Shadow crash failover & sub-millisecond `SIGKILL` fast-fail, circuit-breaker + `max_slots_per_agent=1` rogue-agent isolation, online slot healing (`heal_slot()`), zero-restart `3 → 2 → 3` elastic scaling (`drain_slot()` / `add_slot()`), priority/aging/elastic role-borrowing scheduling, and 6-layer full-stack resilience (`EXP-A`–`EXP-H4`).
- [Developer and multi-agent workflow SOP](process/developer_workflow.md): Boss–Subagent delegation architecture, mandatory `letitdo` trajectory audit triangle, model routing policy, pre-flight checklist, dependency-ordered PR bundling, P0/P1/P2 review triage, and two-tier empirical reporting standards.
- [Local and self-hosted CI](process/local_and_self_hosted_ci.md): how `.github/workflows/cpu-tests.yml` and `.github/workflows/pr-rules.yml` route between a self-hosted runner (`vars.RSIBOX_CI_RUNNER`) and `ubuntu-latest`, how persistent runner workspaces stay CPU-isolated and idempotent, and how local commit-status CI (`statuses/<sha>`) keeps pull requests verified when GitHub Actions dispatch is paused.
