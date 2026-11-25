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
- [Multi-agent GPU contention and agent-team topologies](agent-teams-and-contention.md): empirical NVIDIA B200 study (`EXP-A`–`EXP-D`, continuous multi-crash `EXP-E1`, 4-framework portability, Wave 5–6 systems hardening `EXP-F1`–`EXP-F5`, and Wave 8 think-time / `50/50` mixed restart-tier / rogue-agent circuit-breaker `EXP-G1`–`EXP-G3`; `100/100` SHA-256 hash-chained `SandboxPool` traces, `2,710` leases, scripted worker threads) measuring warm-cache restart contention vs. shared-pool live hot-reload (`43.7×–54.4×` clean, `19.2×` on 1 GPU vs. 3 GPUs restarting, `399.2` edits/min at `N = 24` on `3× B200` with `3 s` think-time, and `2.78×` speedup on a `50/50` mixed hot-swap + restart-tier workload via Standby Shadow), Dynamo GMS Standby Shadow crash failover (`53.2×–53.6×` on vLLM, `0.65–0.80 s` across 3 consecutive live `SIGKILL` crashes with `AsyncShadowReplenisher`, `34.51×` on SGLang with `52` decode CUDA graphs), sub-millisecond unassisted `SIGKILL` fast-fail (`0.502 ms` on vLLM, `0.173 ms` on TensorRT-LLM), per-agent circuit-breaker isolation (`3.08×` lower GPU recovery downtime under a rogue fault storm), and team-topology / priority / aging / elastic role-borrowing scheduling (`+53–61%` via `2 Eval : 1 Dev+Toolsmith`, `4.68×–8.65×` lower `N = 8` Eval wait on vLLM and `6.98×–7.21×` across SGLang/TRT-LLM/Transformers, `2.07×` lower Dev starvation via `priority_aging_s`, and `1.60×–1.65×` Triad / `2.60×` Phase-Shift speedup via work-conserving `fallback_roles`).
- [Developer and multi-agent workflow SOP](process/developer_workflow.md): Boss–Subagent delegation architecture, mandatory `letitdo` trajectory audit triangle, model routing policy, pre-flight checklist, dependency-ordered PR bundling, P0/P1/P2 review triage, and two-tier empirical reporting standards.
- [Local and self-hosted CI](process/local_and_self_hosted_ci.md): how `.github/workflows/cpu-tests.yml` and `.github/workflows/pr-rules.yml` route between a self-hosted runner (`vars.RSIBOX_CI_RUNNER`) and `ubuntu-latest`, how persistent runner workspaces stay CPU-isolated and idempotent, and how local commit-status CI (`statuses/<sha>`) keeps pull requests verified when GitHub Actions dispatch is paused.
