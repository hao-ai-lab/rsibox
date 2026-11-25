<p align="center">
  <img alt="rsibox" src="assets/rsibox-logo.svg" width="360">
</p>

<p align="center">
  <a href="#use-with-your-coding-agent"><b>Agent Setup</b></a> ·
  <a href="#features"><b>Features</b></a> ·
  <a href="#supported-engines--results"><b>Engines &amp; Results</b></a> ·
  <a href="#cli-quick-start"><b>CLI Quick Start</b></a> ·
  <a href="docs/concepts.md"><b>Concepts</b></a> ·
  <a href="docs/mechanisms.md"><b>Mechanisms</b></a> ·
  <a href="docs/engines.md"><b>Engines</b></a> ·
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

- **Python & Triton hot-swap (`~80 ms`):** Atomically swap Python functions and Triton kernels across all GPU ranks (`~0.05 s` rollback) with execution proof on every rank.
- **Selective CUDA graph recapture:** Recapture only the CUDA graphs affected by an edit in place, avoiding full cold restarts.
- **C++/CUDA (`.cu`) & live config reload:** Hot-reload compiled `.cu` kernels (`7.3–19.2 s` vs. `41–49 s`) and tune runtime flags like `max_num_seqs` in-process in **`4.3 ms`**.
- **Multi-GPU & multi-launcher:** Supports tensor, data, pipeline, and sequence-parallel workers plus Ray actors via local, Docker, gVisor, and Slurm launchers.
- **Fast crash recovery ([`rsibox.warm`](core/livelab/rsibox/warm/README.md)):** Optional NVIDIA Dynamo GMS & Snapshot hooks keep weights resident in GPU memory for **`0.31–1.86 s`** standby shadow failover and **`2.73–6.96 s`** checkpoint restore.
- **Multi-agent GPU pool (`SandboxPool`):** Share warm GPUs across concurrent coding agents with priority scheduling, elastic role borrowing, fault isolation, and **`112/112`** verified tamper-evident traces (`43.7×–54.4×` faster than restarting; [study](docs/agent-teams-and-contention.md)).

## Supported Engines & Results

| Engine | Category | Supported Hot-Reload | Restart (Before) | `rsibox` Live (After) | Speedup |
| :--- | :--- | :--- | ---: | ---: | ---: |
| **`vLLM`** | Serving | Python, Triton, selective CUDA graphs, `.cu`, live reconfig, Dynamo GMS/Snapshot & Shadow | `228 s` | **`2.4–23.6 s`** (`~0.08 s` swap) | **up to 95×** |
| **`SGLang`** | Serving | Python, Triton, CUDA graphs, warm fork server, CUDA IPC, Dynamo GMS/Snapshot & Shadow | `66.4 s` | **`0.38–1.77 s`** | **up to 176×** |
| **`TRT-LLM`** | Serving | PyTorch backend live Python swaps, KV-cache quiesce/resume (`23.6 GiB` freed) | `26.7 s` / `12.4 s` | **`0.107 s`** (`56.4 ms` install) | **up to 249×** |
| **`Transformers`** | Inference | Persistent worker multi-module swaps, pinned-CPU weight quiesce/resume | `7.75 s` | **`0.61 s`** (`0.31 s` H2D resume) | **12.8×** |
| **`Megatron-LM`** | Training | Multi-GPU parallel live swaps, pre-warmed fork server | `57.5 s` / `266.7 s` | **`0.50 s`** / **`8.0–17.0 s`** | **up to 115×** |
| **`FastVideo`** | Video Gen | Sequence-parallel live swaps, FlashInfer preload, IPC weight holder | `206 s` | **`2.3–6.4 s`** | **up to 92×** |
| **`veRL`** | RL Post-Training | 21 Ray actors (eager & FSDP CUDA graphs) + coupled state reset | `298 s` | **`36.4 s`** (`12.6 ms` replay) | **8.2×** |
| **`miles`** | RL Post-Training | Ray rollout + trainer actors + coupled state reset (`38 ms` median commit) | `82.4 s` | **`0.89 s`** | **up to 93×** |
| **`TorchTitan`** | Training (FSDP2) | PyTorch FSDP2 (`1` & `2` GPU NCCL) + 2-rank CPU sandbox, state & RNG reset | `2.69 s` / `5.52 s` | **`0.083 s`** / **`0.255 s`** (`2`-rank cycle) | **32.5× / 21.6×** |
| **`NeMo-RL`** | RL Post-Training | AutoModel FSDP2 + SGLang GRPO (3-participant CPU verified; GPU pending) | `pending` | `8.5 ms` CPU install (`1.6 ms` undo) | **CPU verified (GPU pending)** |

On `Megatron-LM`, an autonomous agent ran an **82-iteration** live hill-climb with **zero restarts** and found **+5.1% tokens/s/GPU** ([walkthrough](repos/megatron/examples/codex-hillclimb/README.md)). See **[`docs/concepts.md`](docs/concepts.md)** for core concepts, **[`docs/mechanisms.md`](docs/mechanisms.md)** for restart boundaries and Dynamo benchmarks, and **[`docs/engines.md`](docs/engines.md)** for full engine tables and per-tier costs.

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
