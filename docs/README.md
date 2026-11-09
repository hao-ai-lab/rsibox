# Docs

Docs about how this repo and its process work. Nothing engine-specific lives here: an engine's
architecture goes in `repos/<engine>/docs/`, its measurements in `repos/<engine>/kb/`.

```
docs/
  README.md                       this index
  lessons.md                      framework lessons from building a live lab
  kernels-and-graphs.md           bytecode swap, Triton @jit rebuild, CUDA graphs, launch witnesses, and warm-cache restart taxes
  process/
    developer_workflow.md         Boss-Subagent delegation SOP, model routing, PR bundling, and two-tier reporting
    local_and_self_hosted_ci.md   self-hosted runner routing (FASTRSI_CI_RUNNER) and local commit-status CI
  core/                           how the skill and the agent prompts work
  livelab/                        the interface, receipts, conformance
```

Rules:

- Make a folder only when its first doc arrives. No empty placeholders.
- Docs describe how things are now. Update them in place; no decision logs or changelogs.
- Add every new doc to the index below.

## Index

- [Lessons from building a live lab](lessons.md): framework lessons and rules for applying them.
- [Bytecode swap, kernel rebuild, and CUDA graph replay](kernels-and-graphs.md): why Python bytecode swaps stop at `@triton.jit` and `CUDAGraph` boundaries, how `core/livelab/kernel/jit.py` rebuilds kernels in place with zero-recompile undo, how the two-ledger witness (`live` vs. `capture`) prevents false positives, how three-level selective and lazy graph recapture avoids full-sweep stalls, how the whole-sweep update hierarchy (`cuGraphExecKernelNodeSetParams`, `cuGraphExecUpdate`, and lazy recapture) updates all ~50 shape buckets at once, and how warm-cache restart taxes break down across vLLM, Megatron-LM, and FastVideo.
- [Developer and multi-agent workflow SOP](process/developer_workflow.md): Boss–Subagent delegation architecture, mandatory `letitdo` trajectory audit triangle, model routing policy, pre-flight checklist, dependency-ordered PR bundling, P0/P1/P2 review triage, and two-tier empirical reporting standards.
- [Local and self-hosted CI](process/local_and_self_hosted_ci.md): how `.github/workflows/cpu-tests.yml` and `.github/workflows/pr-rules.yml` route between a self-hosted runner (`vars.FASTRSI_CI_RUNNER`) and `ubuntu-latest`, how persistent runner workspaces stay CPU-isolated and idempotent, and how local commit-status CI (`statuses/<sha>`) keeps pull requests verified when GitHub Actions dispatch is paused.
