<p align="center">
  <img alt="rsibox" src="assets/rsibox-logo.svg" width="360">
</p>

<h3 align="center">Live Hot-Reload Sandbox &amp; Multi-Agent GPU Pool for ML Engines</h3>

<p align="center"><b>Keeping Weights Warm While AI Agents Optimize Production ML Engines</b></p>

<p align="center">
  <a href="#quick-start"><img alt="Quick Start" src="https://img.shields.io/badge/Quick%20Start-Guide-0f766e.svg"></a>
  <a href="core/livelab/formal/"><img alt="TLA+ Verified" src="https://img.shields.io/badge/Protocol-TLA%2B%20Verified-1d4ed8.svg"></a>
  <a href="pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776ab.svg"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache%202.0-green.svg"></a>
</p>

<p align="center">
  <a href="#who-should-use-rsibox"><b>Who Should Use RSIBox</b></a> ·
  <a href="#how-it-works"><b>How It Works</b></a> ·
  <a href="#supported-engines--results"><b>Engines &amp; Results</b></a> ·
  <a href="#quick-start"><b>Quick Start</b></a> ·
  <a href="#next-steps--road-map"><b>Roadmap &amp; Docs</b></a> ·
  <a href="https://github.com/hao-ai-lab"><b>Hao AI Lab</b></a>
</p>




RSIBox provides live hot-reload and multi-agent GPU sharing for RSI workloads on AI infra. 
Today, an agent optimizing a production-grade ML engine must sit through minutes of cold restart for even a one-line change. 
RSIBox eliminates this bottleneck while keeping shared GPU state safe and isolated. 
Agents hot-swap Python code, Triton/C++/CUDA kernels, and CUDA graphs directly into a warm engine, with live execution checks, instant rollback, and crash recovery in seconds.
On vLLM, this cuts edit turnaround from 228 s down to as low as ~100 ms across most of the codebase.

RSIBox works out of the box with existing serving (vLLM, SGLang), training (Megatron-LM), and RL (veRL, Miles) engines. You can deploy RSIBox locally or on Docker, gVisor (runsc), and Slurm, and drive it from any coding agent over an authenticated HTTP API. 
We also provide a set of agents to (toolsmiths) to transfer existing codebase into live-editable ones without intrusive code changes. 



## Quick Start

### Agent-Native Starting

```text
Install https://github.com/hao-ai-lab/rsibox.git 
and read core/agents/livelab_overview.md.

Whenever you edit / benchmark / profile an ML engine, 
use rsibox to hot-reload and verify changes in-place instead of restarting.
```

**Or install it as a reusable agent skill** ([`core/agents/livelab_overview.md`](core/agents/livelab_overview.md) already ships with `SKILL.md` frontmatter):

```bash
git clone https://github.com/hao-ai-lab/rsibox.git
mkdir -p ~/.claude/skills/rsibox 
ln -sf "$(pwd)/rsibox/core/agents/livelab_overview.md" ~/.claude/skills/rsibox/SKILL.md
```

### Manual

Requires Python 3.11+ and an active GPU engine environment (e.g., `vllm`, `sglang`, `megatron`, `verl`, or `miles`):

```bash
git clone https://github.com/hao-ai-lab/rsibox.git && cd rsibox
pip install -e .
```

**1. Start a warm vLLM engine** (pick your launcher):

```bash
# Local (process-native)
CUDA_VISIBLE_DEVICES=0 rsibox up --launcher local --lab vllm --real \
  --state /tmp/rsibox-vllm --env RSIBOX_VLLM_MODEL=Qwen/Qwen3-8B

# Docker (container isolation; add `--runtime runsc` for gVisor)
rsibox up --launcher docker --image <vllm-image> --gpus device=0 \
  --lab vllm --real --state /tmp/rsibox-vllm --env RSIBOX_VLLM_MODEL=Qwen/Qwen3-8B

# Standalone gVisor (`runsc` + `nvproxy` GPU passthrough)
CUDA_VISIBLE_DEVICES=0 rsibox up --launcher gvisor --lab vllm --real \
  --state /tmp/rsibox-vllm --env RSIBOX_VLLM_MODEL=Qwen/Qwen3-8B

# Slurm (`sbatch` job on shared filesystem)
rsibox up --launcher slurm --lab vllm --real --state /shared/rsibox-vllm \
  --sbatch-arg=--gres=gpu:1 --env RSIBOX_VLLM_MODEL=Qwen/Qwen3-8B
```

**2. Wait until warm, then hot-reload, verify, and roll back in place:**

```bash
export RSIBOX=/tmp/rsibox-vllm/box.json
rsibox wait

rsibox source vllm.v1.worker.gpu.sample.sampler > /tmp/sampler.py
# ... edit a function in /tmp/sampler.py ...
rsibox install --file vllm.v1.worker.gpu.sample.sampler=/tmp/sampler.py
rsibox check          # verifies the edited path executed + output parity
rsibox uninstall      # instant journaled rollback to base
rsibox down --state /tmp/rsibox-vllm
```

See [`core/livelab/rsibox/README.md`](core/livelab/rsibox/README.md) for launcher flags (`--mount`, `--dry-run`, `--setup`), or [`docs/quickstart.md`](docs/quickstart.md) for a 30-second CPU-only walkthrough.


## Update
- [2026/11/27] Released RSIBox v0.0.2 🎉
- [2026/11/24] Integrated [NVIDIA Dynamo Snapshot](https://github.com/ai-dynamo/snapshot) for fast warm recovery via `cuda-checkpoint`
- [2026/11/16] Added SGLang, veRL, and Miles live-editable adapters.
- [2026/11/10] Open-sourced RSIBox v0.0.1 🎉
- [2026/11/08] Added Megatron-LM and FastVideo support. 
- [2026/11/06] Added vLLM live hot-reload kernel and adapter. Add the RSIBox agents (toolsmiths) and knowledge base. 


## Who Should Use RSIBox?

- **Engineers iterating on production ML engines:** If you're repeatedly testing small scheduling, model, or kernel edits in a real system, RSIBox replaces multi-minute restarts with selective live updates (Python, Triton, CUDA graphs, and C++/CUDA) in milliseconds to seconds, with live execution checks so stale graphs or silent no-ops never fool your benchmarks.
- **Teams running autonomous coding agents:** RSIBox give agents a set of standard API to edit-and-verify safely (`install`, `check`, `reset`, `uninstall`). At failure, RSIBox can rollback or recovery from crash depending on what failed.
- **Multi-agent workflows sharing one GPU sandbox:** Run concurrent agents against warm GPUs without letting full restarts serialize the team or letting one bad edit crash or poison sibling workers.
- **Infra teams running battle-tested production forks:** Real production engines carry years of in-house logic and continuous upstream merges—you can't throw them away for a greenfield rewrite just to get fast iteration. RSIBox provides a non-intrusive entry point over your existing codebase, along with **Toolsmith** agents to onboard and continuously improve live-editability without invasive source changes.



## How It Works

- **Keep GPU weights and states warm while swapping only what changed:** RSIBox applies each edit at the lowest safe tier — runtime flags, Python functions, `@triton.jit` kernels, selective CUDA graphs, or `.cu` side-libraries — with pre-warmed fork server and NVIDIA Dynamo GMS & Snapshot fallback when an edit crosses a process boundary or crashes a worker.
- **Atomic commit with live execution checks and rollback:** Edits stage while the engine runs, commit atomically across all GPU ranks between steps, record per-rank call counts (`witness`) to prove the new code executed, and reverse via journaling (`~0.05 s` undo).
- **Formally verified protocols & sandbox verification suite:** Core swap, multi-rank coordinator, pool, standby shadow, and autoscaler protocols are specified in TLA+ ([`core/livelab/formal/`](core/livelab/formal/)) and validated against the growing **sandbox verification suite** (`rsibox conformance`, [`core/livelab/conformance/README.md`](core/livelab/conformance/README.md)).
- **Multi-agent GPU pool (`SandboxPool`):** Concurrent coding agents share warm GPUs via priority scheduling, anti-starvation aging, elastic role borrowing, per-agent fault isolation, and **`114/114`** verified tamper-evident traces (`43.7×–54.4×` faster than restarting; [study](docs/agent-teams-and-contention.md)).

Explore the interactive map in [`docs/architecture.html`](docs/architecture.html) or read the layer breakdown in [`docs/concepts.md`](docs/concepts.md):

```text
+---------------------------------------------------------------------------------------------------+
| Layer 1 · Control Plane               core/livelab/rsibox/                                        |
|   rsibox CLI & HTTP Box (cli/server) · SandboxPool & PoolAutoScaler · BaseLab / LabUnderTest      |
+---------------------------------------------------------------------------------------------------+
| Layer 2 · Hot-Reload Kernel (stdlib)  core/livelab/kernel/                                        |
|   Coordinator & Session (2PC) · swap & jit (R0 / Triton) · witness & Journal · findings/identity  |
+---------------------------------------------------------------------------------------------------+
| Layer 3 · Warm Recovery & Substrate   core/livelab/rsibox/warm/, forksafe.py                      |
|   forksafe & IPC · Zygote + CUDA IPC Holder · Dynamo GMS Standby Shadow & CRIU Snapshot           |
+---------------------------------------------------------------------------------------------------+
| Layer 4 · Thin Engine Adapters        repos/<engine>/livelab/ (10 engines)                        |
|   vLLM · SGLang · TRT-LLM · Transformers · Megatron-LM · FastVideo · veRL/miles/NeMo-RL/TorchTitan|
+===================================================================================================+
| Cross-Cutting Plane · core/livelab/{formal,conformance}/ & .agents/plugins/rstack/                |
|   TLA+ Protocol & Pool Models · Sandbox Verification Suite (c0-c10) · rstack PR & CI Gates        |
+---------------------------------------------------------------------------------------------------+
```

### Hot-Reload & Recovery Mechanism

RSIBox routes each edit to the cheapest tier for recovery (weighted by the time to recovery). It falls back to warm recovery only when an edit crosses a process boundary or faults. Here we demonstrate some example mechanisms and the time to edit supported in vLLM / Megatron:

| Hot-Reload / Recovery Tier | What Changes | Typical Latency |
| :--- | :--- | ---: |
| **Live runtime config** | Scheduler & batching flags (`max_num_seqs`, policy) in process | 5 ~ 20 ms |
| **Python & Triton hot-swap** | Python functions & `@triton.jit` kernels across all ranks (~0.05 s undo) | 50 ~ 200 ms |
| **Selective CUDA graph recapture** | Only CUDA graphs affected by an edited forward path | 0.5 ~ 5 s |
| **C++/CUDA (`.cu`) side-library reload** | Compiled native kernels rebound into `torch.ops._C` + selective recapture | 5 ~ 20 s [^1] |
| **Standby Shadow / Snapshot** | Parked GPU-weight-resident failover or NVIDIA Dynamo GMS / CRIU / `cuda-checkpoint` restore | 100 ms ~ 7 s |

[^1]: depending on kernel compilation time and the structure of shared library



### Safety & Correctness Guarantees

Every live edit is guarded against silent no-ops, stale CUDA graphs, and state drift:

| Guarantee | What It Enforces |
| :--- | :--- |
| **[Atomic step-gap swap](docs/mechanisms.md)** | Stages edits while the engine runs and commits across all ranks together between steps. |
| **[Execution check](core/livelab/INTERFACE.md)** | Counts calls on every rank so silent no-op edits (`zero_witness`) fail instead of passing. |
| **[Journaled rollback](docs/mechanisms.md)** | Logs every mutation's inverse; `uninstall` restores exact code objects and verifies base identity. |
| **[Explicit restart boundaries](docs/kernels-and-graphs.md)** | Refuses unsafe edits before touching state and names the cheapest warm tier (Triton, CUDA graph recapture, `.cu` reload, or pre-warmed fork server + CUDA IPC). |
| **[Sandbox verification suite](core/livelab/conformance/README.md)** | Verifies repeatability, parity with fresh restart, crash/hang fault handling, and repeated undo stability. |
| **[TLA+ formal verification](core/livelab/formal/)** | Model-checks swap lifecycle, 2PC multi-rank commit, `SandboxPool`, standby shadow, autoscaler, snapshot, and fork-safe lock order with trace conformance against the runtime. |

See [`docs/mechanisms.md`](docs/mechanisms.md) for all restart boundaries and CPU tests, and [`core/agents/`](core/agents/) for ready-to-use agent prompts.


## Supported Engines & Results

| Engine | Category | Supported Hot-Reload | Restart | `RSIBox` | Speedup |
| :--- | :--- | :--- | ---: | ---: | ---: |
| **`vLLM`** | Serving | Python, Triton, selective CUDA graphs, `.cu`, live reconfig, Dynamo GMS/Snapshot & Shadow | 228 s | 0.08 ~ 2.4 s | **up to 95×** |
| **`SGLang`** | Serving | Python, Triton, CUDA graphs, warm fork server, CUDA IPC, Dynamo GMS/Snapshot & Shadow | 66.4 s | 0.38 ~ 1.77 s | **up to 176×** |
| **`Megatron-LM`** | Training | Multi-GPU parallel live swaps, pre-warmed fork server | 57.5 ~ 266.7 s | 0.50 ~ 17.0 s | **up to 115×** |
| **`FastVideo`** | Video Gen | Sequence-parallel live swaps, FlashInfer preload, IPC weight holder | 206 s | 2.3 ~ 6.4 s | **up to 92×** |
| **`veRL`** | RL Post-Training | 21 Ray actors (eager & FSDP CUDA graphs) + coupled state reset | 298 s | 36.4 s | **8.2×** |
| **`miles`** | RL Post-Training | Ray rollout + trainer actors + coupled state reset | 82.4 s | 0.89 s | **up to 93×** |

On `Megatron-LM`, an autonomous agent ran an **82-iteration** live hill-climb with **zero restarts** and found **+5.1% tokens/s/GPU** ([walkthrough](repos/megatron/examples/codex-hillclimb/README.md)). See **[`docs/concepts.md`](docs/concepts.md)** for core concepts, **[`docs/mechanisms.md`](docs/mechanisms.md)** for restart boundaries and Dynamo benchmarks, and **[`docs/engines.md`](docs/engines.md)** for full engine tables and per-tier costs.

<!-- Some more features: 
- **Composes with NVIDIA Dynamo (`GMS` + `Snapshot`):** Keeps multi-GPU weights warm in VRAM for fast crash recovery (`9.3–15.4 s` on `151–491 GiB` frontier models, or `0.1–1.9 s` with Standby Shadow) while RSIBox applies live code edits in `~50–100 ms` ([details](docs/mechanisms.md)).
- **Multi-agent GPU sharing (`SandboxPool`):** Concurrent coding agents share warm GPUs with per-agent fault isolation—`44–54×` faster than restarting, with zero blast radius on sibling workers when an edit crashes ([study](docs/agent-teams-and-contention.md)). -->



<details>
<summary><b>Preview Support</b></summary>

| Engine | Category | Supported Hot-Reload | Restart | `RSIBox` | Speedup |
| :--- | :--- | :--- | ---: | ---: | ---: |
| **`TRT-LLM`** | Serving | PyTorch backend live Python swaps, KV-cache quiesce/resume | 12.4 ~ 26.7 s | 0.06 ~ 0.11 s | **up to 249×** |
| **`Transformers`** | Inference | Persistent worker multi-module swaps, pinned-CPU weight quiesce/resume | 7.75 s | 0.61 s | **12.8×** |
| **`TorchTitan`** | Training (FSDP2) | PyTorch FSDP2 (1 & 2 GPU NCCL), state & RNG reset | 2.69 ~ 5.52 s | 0.08 ~ 0.26 s | **up to 32.5×** |
| **`NeMo-RL`** | RL Post-Training | AutoModel FSDP2 + SGLang GRPO (1×/2× B200 Ray actors + coupled state reset) | 151.5 ~ 186.8 s | 19.0 ~ 34.7 s | **up to 9.8×** |
| **`Kernel` (Triton / CUDA)** | Kernel Dev | Standalone `@triton.jit` & `.cu` live-edit + region-granular GPU exclusivity | 2.96 ~ 4.64 s | 0.04 ~ 0.95 s | **up to 116×** |
| **`Isaac Lab`** | Robotics Sim | GPU physics/RL sim live swaps (`4096` envs) | 6.79 ~ 9.66 s | 0.30 ~ 0.40 s | **up to 24×** |
</details>




## Documentation Map
- **[`docs/concepts.md`](docs/concepts.md)** — Plain-English overview of `rsibox`, hot-reload tiers, live execution checks (`witness`), and journaled rollback.
- **[`docs/engines.md`](docs/engines.md)** — Full 10-engine summary, measured speedups, and repository layout.
- **[`docs/mechanisms.md`](docs/mechanisms.md)** — Restart-boundary table, NVIDIA Dynamo GMS & Snapshot alignment benchmarks, and CPU tests.
- **[`docs/kernels-and-graphs.md`](docs/kernels-and-graphs.md)** — Deep dive into Python bytecode swaps, Triton `@jit` rebuilds, and CUDA graph replay / `Tier-A` patching.
- **[`docs/agent-teams-and-contention.md`](docs/agent-teams-and-contention.md)** — Multi-agent `SandboxPool` contention, Standby Shadow failover, rogue isolation, and elastic scaling study.
- **[`core/livelab/conformance/README.md`](core/livelab/conformance/README.md)** — The sandbox verification suite (`rsibox conformance`) and CPU toy lab guide.
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** & **[`.agents/plugins/rstack/README.md`](.agents/plugins/rstack/README.md)** — Contributor guidelines and deterministic `rstack` workflow gates.

## Next Steps & Road map

- 👀 Robust integration of multi-tenant GPU sharing in modern serverless / microvm frameworks (preview `SandboxPool` for sandbox-level concurrency control)
- 🚧 Zero-recapture CUDA graph patching: Generalizing in-place kernel parameter updates inside live CUDA graphs without recapture.
- 🚧 Cluster & Kubernetes deployment: Native Kubernetes launcher (`--launcher k8s`) and multi-node warm recovery for training & RL.
- 🚧 Agent knowledge base & Toolsmith refactor: Reorganized per-engine knowledge bases and streamlined Toolsmith onboarding workflows.
- 🚧 Robotics & scientific domains: Deeper live-edit support for Isaac Lab and other non-LLM GPU simulation stacks.

👀 = in preview, 🚧 = in progress, 💭 = just an idea

<!-- - 👀 Standalone kernel platform (`KernelLab`): Live-editing and isolated benchmarking for `@triton.jit` and `.cu` kernels. -->
<!-- - 👀 Expanded engine coverage: Construct adapters for `TRT-LLM`, `Transformers`, `TorchTitan`, `NeMo-RL`, `vLLM-Omni`, `SGLang-Omni`, `NVIDIA Dynamo`, and `Isaac Lab`.  -->
<!-- :** Concurrent coding agents sharing warm GPUs with per-agent fault isolation and region-granular GPU exclusivity.  -->


## Contributors

<p>
  <a href="https://github.com/GindaChen"><img src="https://github.com/GindaChen.png?size=112" width="56" alt="Junda Chen" title="Junda Chen (@GindaChen)"></a>
</p>

## Citation

```bibtex
@misc{rsibox2026,
  title        = {{RSIBox}: A Live Hot-Reload Sandbox for {RSI} on {AI} Infrastructure},
  author       = {{Hao AI Lab}},
  year         = {2026},
  howpublished = {\url{https://github.com/hao-ai-lab/rsibox}}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
