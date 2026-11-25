<p align="center">
  <img alt="rsibox" src="assets/rsibox-logo.svg" width="360">
</p>

<p align="center">
  <a href="#use-with-your-coding-agent"><b>Agent Setup</b></a> ·
  <a href="#features"><b>Features</b></a> ·
  <a href="#supported-engines--results"><b>Engines &amp; Results</b></a> ·
  <a href="#cli-quick-start"><b>CLI Quick Start</b></a> ·
  <a href="docs/mechanisms.md"><b>Mechanisms</b></a> ·
  <a href="docs/agent-teams-and-contention.md"><b>Multi-Agent Pool</b></a> ·
  <a href="https://github.com/hao-ai-lab"><b>Hao AI Lab</b></a>
</p>

`rsibox` is a live hot-reload sandbox for ML engines. Instead of paying a multi-minute cold restart on every code edit to reload weights and recapture CUDA graphs, `rsibox` swaps Python, Triton kernels, CUDA graphs, and C++/CUDA directly into the warm running engine — with execution proof and instant bit-exact rollback.

## Use with Your Coding Agent

*Paste this prompt into Claude Code, Codex, Cursor, or Antigravity — zero manual setup:*

```text
Install https://github.com/hao-ai-lab/rsibox.git and read core/agents/livelab_overview.md.
Whenever we edit or benchmark an ML engine, use rsibox to hot-reload and verify changes in-place instead of restarting.
```

**Or install it as a reusable agent skill** ([`core/agents/livelab_overview.md`](core/agents/livelab_overview.md) already ships with `SKILL.md` frontmatter):

```bash
git clone https://github.com/hao-ai-lab/rsibox.git
mkdir -p ~/.claude/skills/rsibox && ln -sf "$(pwd)/rsibox/core/agents/livelab_overview.md" ~/.claude/skills/rsibox/SKILL.md
```

## Features

- **Python & Triton kernel hot-swap (`~80 ms`):** Atomically swap Python modules and Triton kernels across all multi-GPU ranks (`~0.08 s` install / `~0.05 s` rollback) with execution proof on every rank.
- **CUDA Graph selective recapture:** Recapture only affected CUDA graphs in-place when an edit touches graph-captured kernels, avoiding full cold restarts.
- **C++/CUDA (`.cu`) & live config reload:** Hot-reload compiled `.cu` kernels (`7.3–19.2 s` vs. `41–49 s`) and tune runtime flags (e.g., `max_num_seqs`) in-process in **`4.3 ms`**.
- **Multi-GPU & multi-launcher:** Supports TP / DP / PP / VPP / SP workers and Ray actors via local, Docker, gVisor, and Slurm launchers.
- **Pluggable fast recovery (NVIDIA Dynamo GMS & Snapshot):** Optional zero-dependency [`rsibox.warm`](core/livelab/rsibox/warm/README.md) hooks for persistent GPU weights, **`0.31–1.86 s`** vLLM Shadow failover (with off-critical-path `AsyncShadowReplenisher`), and **`2.73–6.96 s`** `0.6B` / GMS+CRIU + `cuda-checkpoint` restore (vs. `13.35–45.38 s` Plain CRIU on `14B–65.5 GB`).
- **Multi-agent shared-pool scheduling (`SandboxPool`):** Multiplex `N` concurrent agent workers over `C` warm GPUs with fenced leases, SHA-256 hash-chained `verify_trace` logs, and automatic Shadow failover — delivering **`43.7×–54.4×`** clean multi-agent speedup (`1` B200 at `N = 8` beats `3` B200s restarting at `N = 6` by **`19.2×`**) and sub-second crash recovery ([study](docs/agent-teams-and-contention.md)).

## Supported Engines & Results

| Engine | Category | Supported Hot-Reload | Restart (Before) | `rsibox` Live (After) | Speedup |
| :--- | :--- | :--- | ---: | ---: | ---: |
| **`vLLM`** | Serving | Python, Triton, V1/V2 CUDA graphs, `.cu`, live reconfig, Dynamo GMS/Snapshot & Shadow | `228 s` | **`2.4–23.6 s`** (`~0.08 s` swap) | **up to 95×** |
| **`SGLang`** | Serving | Python, Triton, CUDA graphs, fork-subprocs, CUDA IPC, Dynamo GMS/Snapshot & Shadow | `66.4 s` | **`0.38–1.77 s`** | **up to 176×** |
| **`TRT-LLM`** | Serving | PyTorch backend live Python swaps, KV-cache `TrtllmQuiesceHooks` (`23.6 GiB` freed) | `26.7 s` / `12.4 s` | **`0.107 s`** (`56.4 ms` install) | **up to 249×** |
| **`Transformers`** | Inference | Persistent worker multi-module swaps, pinned-CPU `TransformersQuiesceHooks` | `7.75 s` | **`0.61 s`** (`0.31 s` H2D resume) | **12.8×** |
| **`Megatron-LM`** | Training | Multi-GPU TP/DP/PP/VPP2 live swaps, pre-warmed fork server | `57.5 s` / `266.7 s` | **`0.50 s`** / **`8.0–17.0 s`** | **up to 115×** |
| **`FastVideo`** | Video Gen | SP1–SP4 live swaps, FlashInfer preload, IPC weight holder | `206 s` | **`2.3–6.4 s`** | **up to 92×** |
| **`veRL`** | RL Post-Training | 21 Ray actors (eager & FSDP CG) + coupled `S0` reset | `298 s` | **`36.4 s`** (`12.6 ms` replay) | **8.2×** |
| **`miles`** | RL Post-Training | Ray rollout + trainer actors + coupled `S0` reset (`38 ms` median commit) | `82.4 s` | **`0.89 s`** | **up to 93×** |

On `Megatron-LM`, an autonomous agent ran an **82-iteration** live hill-climb with **zero restarts** and found **+5.1% tokens/s/GPU** ([walkthrough](repos/megatron/examples/codex-hillclimb/README.md)). Full engine tables, per-tier costs, and walkthroughs live in **[`docs/engines.md`](docs/engines.md)**.

## CLI Quick Start

Requires Python 3.11+ (CPU-only, zero third-party dependencies for core and toy labs):

```bash
git clone https://github.com/hao-ai-lab/rsibox.git && cd rsibox
python3 -m venv .venv && . .venv/bin/activate && pip install -e .

# 1. Start a live sandbox and wait until warm
rsibox up --launcher local --lab toy --state /tmp/rsibox-toy
export RSIBOX=/tmp/rsibox-toy/box.json
rsibox wait

# 2. Fetch live source, edit 1 line, hot-reload, verify, and roll back
rsibox source toy.ops > /tmp/ops.py
sed -i.bak 's/return x + 1/return x + 100/' /tmp/ops.py
rsibox install --file toy.ops=/tmp/ops.py
rsibox check          # proves the edited function ran and shows the new output
rsibox uninstall      # instant journaled rollback to base
rsibox down --state /tmp/rsibox-toy
```

For real GPU engines (local, Docker, gVisor, or Slurm), pass `--real` and see [`core/livelab/rsibox/README.md`](core/livelab/rsibox/README.md) or `rsibox conformance <engine> --real -- --help`:

```bash
rsibox up --launcher local --lab vllm --real --state /tmp/rsibox-vllm \
  --env RSIBOX_VLLM_MODEL=Qwen/Qwen3-8B
```

<details>
<summary><b>How rsibox keeps live edits safe</b> (atomic step-gap swap, execution proof, rollback, restart boundaries)</summary>

| Guarantee | What It Enforces |
| :--- | :--- |
| **[Atomic step-gap swap](docs/mechanisms.md)** | Stages edits while the engine runs and commits across all ranks together between steps. |
| **[Execution check](core/livelab/INTERFACE.md)** | Counts calls on every rank so silent no-op edits (`zero_witness`) fail instead of passing. |
| **[Journaled rollback](docs/mechanisms.md)** | Logs every mutation's inverse; `uninstall` restores exact code objects and verifies base identity. |
| **[Explicit restart boundaries](docs/kernels-and-graphs.md)** | Refuses unsafe edits before touching state and names the cheapest warm tier (Triton, CUDA graph recapture, `.cu` reload, or pre-warmed fork server + CUDA IPC). |
| **[Sandbox verification suite](core/livelab/conformance/README.md)** | Verifies repeatability, parity with fresh restart, crash/hang fault handling, and repeated undo stability. |

See [`docs/mechanisms.md`](docs/mechanisms.md) for all restart boundaries and CPU tests, and [`core/agents/`](core/agents/) for ready-to-use agent prompts.

</details>

## Contributors

<p>
  <a href="https://github.com/GindaChen"><img src="https://github.com/GindaChen.png?size=112" width="56" alt="Junda Chen" title="Junda Chen (@GindaChen)"></a>
</p>

## Citation

```bibtex
@misc{rsibox2026,
  title        = {rsibox: Live Hot-Reload Sandbox for ML Engines},
  author       = {Hao AI Lab},
  year         = {2026},
  howpublished = {\url{https://github.com/hao-ai-lab/rsibox}}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
