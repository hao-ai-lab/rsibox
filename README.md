# FastRSI

Hot-reload your inference and training engines. Make them faster while they keep running.

FastRSI provides autonomous coding agents and human engineers with a **live lab**: a running engine that accepts and tests code changes without requiring slow cold restarts. The system swaps code edits into live worker processes during step gaps, verifies with live execution witnesses that the new code ran, benchmarks performance, and can roll back or undo changes in microseconds to milliseconds. When an edit cannot be safely hot-swapped, the live lab clearly flags the exact constraint and routes to the cheapest restart boundary.

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
