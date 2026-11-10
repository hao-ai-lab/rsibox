# FastRSI

Hot-reload your inference and training engines. Make them faster while they keep running.

FastRSI provides autonomous coding agents and human engineers with a **live lab**: a running engine that accepts and tests code changes without requiring slow cold restarts. The system swaps code edits into live worker processes during step gaps, verifies with live execution witnesses that the new code ran, benchmarks performance, and can roll back or undo changes in microseconds to milliseconds. When an edit cannot be safely hot-swapped, the live lab clearly flags the exact constraint and routes to the cheapest restart boundary.

## Quickstart (30 seconds)

The shared live-lab transaction kernel (`core/livelab/`) uses the Python standard library only (`>= 3.10`, zero third-party dependencies):

```sh
# 1. Run all 11 golden conformance cases (c0-c10) on ToyLab in one command (~3s on CPU):
PYTHONPATH=core/livelab python3 -m conformance --lab kernel.toy_lab:ToyLab

# (Optional) Install core/livelab in editable mode in a fresh venv, so kernel, conformance and forksafe import from anywhere:
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
python3 -m conformance --lab kernel.toy_lab:ToyLab

# 2. Run the FastVideo ToyFastVideoLab conformance suite (~3s on CPU):
PYTHONPATH=core/livelab:repos/fastvideo/livelab python3 -m conformance --lab adapter.toy_lab:ToyFastVideoLab

# 3. Inspect the 3 recorded multi-GPU engine walkthroughs and receipts:
ls repos/vllm/examples/dsv41-prefill/          # vLLM: DeepSeek-V4.1-Flash prefill PRs (#59894, #62359, #62240)
ls repos/fastvideo/examples/replay-1882/       # FastVideo: MiniMax-H3 PR #1882 live swap, zero-witness catch, and undo
ls repos/megatron/examples/codex-hillclimb/    # Megatron-LM: 206-iteration live hill-climb (+5.1% to +8.5% tok/s/GPU)
```

### 4-Tier Live Mutation vs. Static Snapshot / Cold Restart

Static GPU checkpoint/restore tools capture a frozen process image and must invalidate or re-snapshot whenever Python bytecode, `@triton.jit` kernels, captured `CUDAGraph` descriptors, or parallelism flags change. FastRSI routes every edit to the lowest boundary that preserves sound execution witnesses:

| Boundary Tier | Mechanism | Edit-to-Run Latency | Undo / Restore Latency | What It Handles |
|---|---|---|---|---|
| **Tier 0 — In-Place Swap & Node Patch** | Python `__code__` swap (`swap.py`) + surgical `cuGraphExecKernelNodeSetParams` update | **0.13–1.0 ms** | **0.13–1.2 ms** (journaled inverse) | Function/method bodies, scheduler heuristics, sampler logic, and launch grid/scalar changes inside captured graphs |
| **Tier 1 — In-Process Kernel Re-JIT** | Transitive `@triton.jit` caller cascade (`jit.py`) + selective Dynamo frame reset | **50–500 ms** (`0.26–1.8 s` full autotune) | **0.56 ms** (cached binary pointer swap) | Triton kernel bodies, `@triton.heuristics`, and inline `@triton.jit` device helpers |
| **Tier 2 — Selective Graph & KV Replay** | Step-gap `CUDAGraph` bucket recapture (`recapture.py`) + tail KV block snapshot replay (`snapshot_replay.py`) | **0.37–1.4 s** (`< 5 ms` KV state restore) | **1.2–5.0 ms** (graph pool & tail block restore) | Structural edits inside graph-captured forward regions (`OP` / `GRAPH`) and deterministic 128K steady-state decode replay |
| **Tier 3 — Warm Zygote + IPC Weight Lease** | Pre-fork Zygote (`forksafe.py`) + zero-copy CUDA IPC `WeightHolder` + 2D `TP<->DP` / `TP<->EP` NVLink resharding (`reshard.py`) | **7.4–22.6 s** (`0.35–1.6 s` NVLink reshard) | **7.4–22.6 s** (`< 1 ms` standby failover) | Constructor/`__init__` state, module-level constants, KV cache layout flags, and live `TP <-> DP` / `MoE TP <-> EP` resharding |
| **Baseline — Cold Restart / Static Snapshot** | Full process respawn, safetensors disk load, JIT warmup, and full graph sweep | **200–650 s+** | **200–650 s+** | Required only when changing hardware topology or uncached base checkpoints |

## Repository layout

The repository is organized into four distinct layers:

```
core/livelab/            Shared engine-agnostic live-lab runtime:
  kernel/                Pure-Python transaction kernel (locate, swap, findings, witness,
                         journal, coordinator, session) with zero engine dependencies;
                         includes ToyLab for self-contained testing.
  forksafe.py            Pre-fork CUDA and driver safety guards for zygote server preloads.
  conformance/           11 golden conformance test cases (c0-c10) that verify any live lab:
                         repeatability, no-op guarantees, visibility, fresh process equivalence,
                         uninstall/undo, failure rollback, refusal, chains, crash/hang recovery,
                         soak stability, and unproven execution detection.
  INTERFACE.md           The live lab protocol contract (verbs, state, findings, receipts).

core/agents/             Agent prompts, protocols, and engineering discipline:
  livelab_overview.md    Live lab overview: interface verbs, boundaries, and function groups.
  developer.md           Developer agent prompt: edit, stage, verify, commit, and undo.
  toolsmith_offline.md   Offline toolsmith: mine PRs and tests to build engine knowledge.
  toolsmith_online.md    Online toolsmith: diagnose slow checks, account for state pollution,
                         and avoid over-conservative micro-edits.
  toolsmith_distill.md   Distillation toolsmith: promote proven mechanisms across 2+ engines.
  orchestrator.md        Orchestrator prompt: live streamer and human interface of the loop.

repos/<engine>/          Engine-specific live labs, knowledge bases, and PR replays:
  docs/                  Engine architecture: process tree, restart boundaries, and costs.
  kb/                    Measured facts and ground truth: costs, workloads, priors, lessons.
  livelab/               Engine-specific lab implementations (adapters, services, hooks).
  tools/                 Engine-specific probes and local tools.
  patches/               Engine patches with upstreaming ledgers.
  examples/<name>/       Reproducible walkthroughs replaying real PRs with receipts.

template/                Minimal onboarding scaffold for bootstrapping a new engine under repos/.

docs/                    Process documentation, rules, and framework lessons (docs/lessons.md).
```

## Engine live labs

FastRSI currently supports three primary engine live labs under `repos/`:

- **vLLM** (`repos/vllm/`): Live lab adapter (`adapter/`), step-gap coordinator, selective CUDA graph recapture (`recapture.py`), and pre-fork zygote server preload (`services/zygote.py`). Replays DeepSeek-V4.1 Flash prefill optimizations (`examples/dsv41-prefill/`).
- **FastVideo** (`repos/fastvideo/`): Live lab adapter (`adapter/`), worker supervisor, zygote respawn, and weights IPC holder for diffusion pipelines. Replays MiniMax-H3 live edits and PRs (`examples/pr-replay/`, `examples/h3-live-edit/`).
- **Megatron-LM** (`repos/megatron/`): Training live lab (`livelab/fastrsi_megatron/`), train-step quiesce hook, gloo synchronization rounds, and weights/optimizer state snapshots (`state.py`). Replays architecture hill-climbs on Qwen3-14B (`examples/codex-hillclimb/`).

## How it works

1. **Transaction kernel inside the step gap:** Code edits are staged, analyzed, and prepared outside the gap while requests run. During a quiescent step gap (between generation steps in inference, or between train steps in training), the coordinator validates the edit across all worker ranks. If accepted, it swaps code objects atomically and captures execution witnesses.
2. **Execution witness and proof:** The live lab tracks whether running steps hit the new bytecode. If an edit sits inside captured CUDA graphs or JIT-compiled kernels, the lab requires CUDA graph recapture or kernel rebuild before marking the change proven.
3. **Instant journaled undo:** Every transaction logs inverse operations in an append-only journal. Rolling back or undoing an edit restores exact bytecode, class attributes, and CUDA graph entries in microseconds to milliseconds with zero recompilation.
4. **Clean boundary routing:** Edits that change class hierarchies, constructor signatures, or distributed layout cannot be swapped in place. The kernel's findings reject unsafe swaps and route directly to the lowest appropriate restart boundary (component restart or warm engine restart via zygote fork).

## Verification and testing

Run CPU-only unittests and conformance checks locally:

```sh
# Shared kernel unittests
python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v

# Run the 11 conformance cases on the ToyLab
cd core/livelab && python3 -m conformance --lab kernel.toy_lab:ToyLab

# Pre-fork CUDA guards test suite
python3 -m unittest discover -s core/livelab -p 'test_forksafe.py' -v

# vLLM live lab adapter tests
python3 -m unittest discover -s repos/vllm/livelab/adapter -t repos/vllm/livelab -v

# FastVideo adapter and service tests
python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab -v
python3 -m unittest discover -s repos/fastvideo/livelab/services -t repos/fastvideo/livelab -v

# Megatron live lab tests
python3 -m unittest discover -s repos/megatron/livelab/fastrsi_megatron -t repos/megatron/livelab -v
```

## License

Apache-2.0
