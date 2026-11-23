<p align="center">
  <img alt="rsibox" src="assets/rsibox-logo.svg" width="360">
</p>

<p align="center">
  <a href="#quick-start"><b>Quick Start</b></a> ·
  <a href="docs/engines.md"><b>Engines &amp; Benchmarks</b></a> ·
  <a href="docs/mechanisms.md"><b>Mechanisms</b></a> ·
  <a href="CONTRIBUTING.md"><b>Contributing</b></a> ·
  <a href="https://github.com/hao-ai-lab"><b>Hao AI Lab</b></a>
</p>

`rsibox` is a live hot-reload sandbox for ML systems agents and engineers. Every edit to a serving or training engine normally pays a multi-minute restart to reload weights and rebuild CUDA graphs; `rsibox` hot-reloads Python, Triton, CUDA graphs, and C++/CUDA directly into the warm process so you iterate in seconds with execution proof and instant rollback.

## Quick Start

Requires Python 3.11+ (CPU-only, zero third-party dependencies for core and toy labs):

```bash
git clone https://github.com/hao-ai-lab/rsibox.git && cd rsibox
python3 -m venv .venv && . .venv/bin/activate && pip install -e .

# Run the live-swap verification check on the toy lab (~1.5 s)
rsibox conformance toy

# Start a live sandbox, run a check, and shut it down
rsibox up --launcher local --lab toy --state /tmp/rsibox-toy
RSIBOX=/tmp/rsibox-toy/box.json rsibox check
rsibox down --state /tmp/rsibox-toy
```

To use `rsibox` from Claude Code, Codex, Cursor, or Antigravity, point your agent at [`core/agents/livelab_overview.md`](core/agents/livelab_overview.md) and [`AGENTS.md`](AGENTS.md):

```text
Read core/agents/livelab_overview.md and AGENTS.md, start an rsibox for <engine>,
and iterate on <optimization> with rsibox install -> check -> uninstall without restarting.
```

## Hot-Reload Guarantees

| Guarantee | What It Enforces |
| :--- | :--- |
| **[Atomic step-gap swap](docs/mechanisms.md)** | Stages edits while the engine runs and commits across all ranks together between steps. |
| **[Witness execution proof](core/livelab/INTERFACE.md)** | Counts calls on every rank so silent no-op edits (`zero_witness`) fail instead of passing. |
| **[Journaled rollback](docs/mechanisms.md)** | Logs every mutation's inverse; `uninstall` restores exact code objects and verifies base identity. |
| **[Explicit restart boundaries](docs/kernels-and-graphs.md)** | Refuses unsafe edits before touching state and names the cheapest warm tier (Triton, CUDA graph recapture, `.cu` reload, or zygote + CUDA IPC). |
| **[Live-swap verification check](core/livelab/conformance/README.md)** | Verifies repeatability, parity with fresh restart, crash/hang poisoning, and repeated undo stability. |

See [`docs/mechanisms.md`](docs/mechanisms.md) for all restart boundaries and CPU tests, and [`core/agents/`](core/agents/) for ready-to-use agent prompts.

## Supported Engines & Highlights

- **Serving (**`vLLM`**,** `SGLang`**):** Live Python, Triton, V1/V2 CUDA graph recapture, and C++/CUDA `.cu` hot-reload — up to **95×** faster on vLLM (`228 s → 2.4–23.6 s`) and **176×** on SGLang (`66.4 s → 0.38–1.77 s`).
- **Training & Video (**`Megatron-LM`**,** `FastVideo`**):** Multi-GPU TP/DP/PP/VPP2 and SP4 live swaps + zygote/IPC weight holders — 82-iteration Megatron hill-climb without restarts and up to **92×** faster FastVideo checks.
- **RL Post-Training (**`veRL`**,** `miles`**):** 21-actor Ray live swaps with coupled `S0` reset — up to **93×** faster on miles (`82.4 s → 0.89 s`) and **8.2×** live / `12.6 ms` offline replay on veRL.

Full engine tables, hardware receipts, and walkthroughs live in **[`docs/engines.md`](docs/engines.md)**.

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
