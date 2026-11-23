<p align="center">
  <img alt="rsibox" src="assets/rsibox-logo.svg" width="360">
</p>

<p align="center">
  <a href="#quickstart"><b>Quickstart</b></a> ·
  <a href="#using-rsibox-with-coding-agents"><b>For Agents</b></a> ·
  <a href="docs/engines.md"><b>Engines &amp; Benchmarks</b></a> ·
  <a href="docs/mechanisms.md"><b>Mechanisms</b></a> ·
  <a href="CONTRIBUTING.md"><b>Contributing</b></a> ·
  <a href="https://github.com/hao-ai-lab"><b>Hao AI Lab</b></a>
</p>

`rsibox` (**FastRSI**) is a live hot-swap sandbox for ML engines. Instead of paying a multi-minute restart for every edit in **vLLM**, **SGLang**, **Megatron-LM**, **FastVideo**, **veRL**, or **miles**, coding agents and engineers swap edits into a warm running process, prove the new code executed, measure it, and undo it cleanly — or get an explicit refusal naming the cheapest restart boundary required.

## Quickstart

Requires Python 3.11+ (CPU-only, zero third-party dependencies for the core runtime and toy labs):

```sh
git clone https://github.com/hao-ai-lab/rsibox.git && cd rsibox
python3 -m venv .venv && . .venv/bin/activate && pip install -e .

# 1. Verify environment and run the 11-case conformance suite on the toy lab (~1.5 s)
fastrsi doctor
fastrsi conformance toy

# 2. Start a live rsibox sandbox, run a check, and shut it down
rsibox up --launcher local --lab toy --state /tmp/rsibox-toy
export RSIBOX=/tmp/rsibox-toy/box.json
rsibox status
rsibox check
rsibox down --state /tmp/rsibox-toy
```

## Using `rsibox` with Coding Agents

Point Claude Code, Codex, Cursor, or Antigravity at this repository:

```text
Read core/agents/livelab_overview.md and AGENTS.md, start an rsibox for <engine>,
and iterate on <target optimization> using install -> check -> uninstall without restarting the engine.
```

Agents drive any engine through one CLI (`RSIBOX` points to the box file; exit code `0` = ok, `2` = refused/aborted with the required restart boundary, `1` = error):

```sh
rsibox up --launcher local|docker|gvisor|slurm --lab <engine> [--real] --state <dir>
rsibox install --diff edit.patch        # atomic swap across all ranks (or refuse cleanly)
rsibox check                            # run probe; returns timing, output diff, and witness counts
rsibox uninstall                        # journaled revert to base with identity proof
rsibox reset --kind state|respawn       # restore caches/RNG, or respawn if poisoned
rsibox down --state <dir>
```

Ready-to-use agent prompts live in [`core/agents/`](core/agents/) (`livelab_overview.md`, `developer.md`, `orchestrator.md`, `toolsmith_offline.md`).

## How It Works

1. **Atomic step-gap swap:** Edits are staged while the engine runs and committed across all ranks in the gap between steps.
2. **Witness proof:** Every edited function counts calls on every rank, catching silent no-op edits (`zero_witness`) instead of reporting them as passing.
3. **Journaled undo:** Every mutation logs its inverse; `uninstall` restores exact code objects and verifies code identity against base.
4. **Explicit restart boundaries:** Edits that cannot be swapped in place are refused before mutating state, naming the cheapest warm boundary (Triton rebuild, selective CUDA graph recapture, C++/CUDA reload, or zygote + CUDA IPC).

## Documentation

| Document | Contents |
|---|---|
| **[Engines & Benchmarks](docs/engines.md)** | Supported engines (vLLM, SGLang, Megatron-LM, FastVideo, veRL, miles), GPU speedups, per-engine docs, and repository layout |
| **[Mechanisms & Conformance](docs/mechanisms.md)** | Restart boundaries (Python, Triton, CUDA graphs, C++/CUDA, zygote + IPC), 11 conformance guarantees, limits, and full CPU test suite |
| **[Lab Interface](core/livelab/INTERFACE.md)** | Formal `install` / `check` / `reset` / `uninstall` / `status` contract |
| **[Kernels & CUDA Graphs](docs/kernels-and-graphs.md)** | Design notes on live Triton/CUDA kernel swaps and selective graph recapture |
| **[Agent & Contributor Rules](AGENTS.md)** | Full workflow rules for coding agents and contributors (see also [`CONTRIBUTING.md`](CONTRIBUTING.md)) |

## Citation

```bibtex
@misc{rsibox2026,
  title        = {rsibox (FastRSI): Live Hot-Swap Sandbox for ML Engines},
  author       = {Hao AI Lab},
  year         = {2026},
  howpublished = {\url{https://github.com/hao-ai-lab/rsibox}}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
