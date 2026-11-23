<p align="center">
  <img alt="rsibox" src="assets/rsibox-logo.svg" width="360">
</p>

<p align="center">
  <a href="#quickstart"><b>Quickstart</b></a> ·
  <a href="#using-rsibox-with-coding-agents"><b>For Agents</b></a> ·
  <a href="#what-is-in-the-repo-today"><b>Engines</b></a> ·
  <a href="#how-it-works"><b>How It Works</b></a> ·
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

Point Claude Code, Codex, Cursor, or Antigravity at this repo with a prompt like:

```text
Read core/agents/livelab_overview.md and AGENTS.md, start an rsibox for <engine>,
and iterate on <target optimization> using install -> check -> uninstall without restarting the engine.
```

Agents drive any engine through the same CLI (`RSIBOX` points to the box file; exit code `0` = ok, `2` = refused/aborted with the required restart boundary, `1` = error):

```sh
rsibox up --launcher local|docker|gvisor|slurm --lab <engine> [--real] --state <dir>
rsibox install --diff edit.patch        # atomic swap across all ranks (or refuse cleanly)
rsibox check                            # run probe; returns timing, output diff, and witness counts
rsibox uninstall                        # journaled revert to base with identity proof
rsibox reset --kind state|respawn       # restore caches/RNG, or respawn if poisoned
rsibox down --state <dir>
```

Ready-to-use agent prompts live in [`core/agents/`](core/agents/): [`livelab_overview.md`](core/agents/livelab_overview.md) (quickstart for any agent), [`developer.md`](core/agents/developer.md), [`orchestrator.md`](core/agents/orchestrator.md), and [`toolsmith_offline.md`](core/agents/toolsmith_offline.md).

## How It Works

1. **Atomic step-gap swap:** Edits are staged while the engine runs and committed across all ranks in the gap between steps.
2. **Witness proof:** Every edited function counts calls on every rank, so silent no-op edits (`zero_witness`) are caught instead of reported as passing.
3. **Journaled undo:** Every mutation records its inverse; `uninstall` restores exact code objects and verifies code identity against base.
4. **Explicit restart boundaries:** Edits that cannot be swapped in place (e.g., constructors, layout changes) are refused before touching state, naming the cheapest warm boundary (Triton rebuild, selective CUDA graph recapture, C++/CUDA side-library reload, or zygote + CUDA IPC weight holder).

## What is in the repo today

| Engine | Conformance (11/11) | Headline Live vs. Restart | Docs & Costs |
|---|---|---|---|
| **vLLM** | CPU CI & 1–2× B200 (eager, V1/V2 CUDA graphs) | 228 s → **2.4–23.6 s** (up to 95×); C++/CUDA `.cu` reload 7.3–19.2 s vs. 41–49 s | [architecture](repos/vllm/docs/architecture.md) · [costs](repos/vllm/kb/costs.md) · [walkthrough](repos/vllm/examples/live-edit-qwen3/README.md) |
| **SGLang** | CPU CI & 1–2× B200 / GB200 (`--fork-subprocs`) | 66.4 s cold / 26.5 s warm → **0.38–1.77 s** live (up to 176×); zygote 10.6–19.5 s | [costs](repos/sglang/kb/costs.md) |
| **Megatron-LM** | CPU CI & 2–4× GB200 (TP/DP/PP/VPP2) | 57.5 s → **0.50 s** (2L TP2); 266.7 s → **8.0–17.0 s** (30B-A3B EP4); 82-iter hill-climb | [livelab](repos/megatron/livelab/README.md) · [costs](repos/megatron/kb/costs.md) · [hill-climb](repos/megatron/examples/codex-hillclimb/README.md) |
| **FastVideo** | CPU CI & 1–4× GB200 (SP1/SP2/SP4, zygote+holder) | 206 s cold / 40.6 s warm → **2.3–6.4 s** live (up to 92×); zygote+holder 14.2 s | [costs](repos/fastvideo/kb/costs.md) · [h3-conformance](repos/fastvideo/examples/h3-conformance/README.md) · [hill-climb](repos/fastvideo/patches/h3-hillclimb/README.md) |
| **veRL** | CPU CI & 4× GB200 (21 Ray actors, eager & FSDP CG) | 298 s cold / 180 s warm → **36.4 s** live + S0 reset (8.2×); offline `replay-adv` 12.6 ms | [livelab](repos/verl/livelab/README.md) · [costs](repos/verl/kb/costs.md) · [grpo-gsm8k](repos/verl/examples/grpo-gsm8k/README.md) |
| **miles** | CPU CI & 2× B200 / GB200 | 82.4 s cold / 53.5 s warm → **0.89 s** live + S0 reset (93×); 38 ms median commit | [costs](repos/miles/kb/costs.md) |
| **Toy lab** | CPU CI (2-rank stdlib reference) | ~1.5 s for all 11 conformance cases on a laptop | [conformance](core/livelab/conformance/README.md) |

---

<details>
<summary><b>Mechanisms by restart boundary</b> (Python, Triton, CUDA graphs, C++/CUDA, zygote + IPC)</summary>

| Boundary | Mechanism | Edits handled | Implementation |
|---|---|---|---|
| **In place, Python** | Swap function code on every rank with journaled undo | Function & method bodies | [`core/livelab/kernel/swap.py`](core/livelab/kernel/swap.py) |
| **In place, Triton** | Rebuild `@triton.jit` kernels & callers; restore binaries on undo | Triton kernel bodies | [`core/livelab/kernel/jit.py`](core/livelab/kernel/jit.py) |
| **In place, CUDA graphs** | Selective `R1` recapture of affected graphs in live workers | Edits inside graph-captured paths | [`repos/vllm/livelab/adapter/recapture.py`](repos/vllm/livelab/adapter/recapture.py) |
| **In place, C++/CUDA** | Compile edited `.cu` side-library, rebind `torch.ops._C` + `R1` recapture | Single-unit C++/CUDA kernels | [`repos/vllm/livelab/adapter/native.py`](repos/vllm/livelab/adapter/native.py) |
| **In process, rebuild** | Re-import edited module and rebuild live objects | Constructors, new fields, module constants | [`repos/fastvideo/livelab/services/rebuild.py`](repos/fastvideo/livelab/services/rebuild.py) |
| **Warm respawn (zygote)** | Fork from pre-imported server; weights served over CUDA IPC | Launch flags, startup code, refused edits | [`core/livelab/forksafe.py`](core/livelab/forksafe.py), `repos/*/livelab/services/{zygote,weights,holder}.py` |
| **In-memory reshard** | Reshard held weights between TP and EP over CUDA IPC | Expert-parallelism layout switches | [`repos/vllm/livelab/services/reshard_live.py`](repos/vllm/livelab/services/reshard_live.py) |

See [`docs/kernels-and-graphs.md`](docs/kernels-and-graphs.md) for the full design and each engine's `kb/costs.md` for measured GPU receipts.

</details>

<details>
<summary><b>Lab interface &amp; 11 conformance guarantees</b></summary>

Every engine adapter implements [`core/livelab/INTERFACE.md`](core/livelab/INTERFACE.md): `install(patch)`, `check(workload)`, `reset()`, `uninstall()`, and `status()` (`base`, `edited`, or `poisoned`).

The 11 conformance cases ([`core/livelab/conformance/README.md`](core/livelab/conformance/README.md)) verify:
- **c0–c2:** Repeatable workload; no-op edit stays bit-exact; visible edit changes output and fires witnesses on every rank.
- **c3:** Live-edited output (and a chain of 3 edits) matches a fresh process started with the edit.
- **c4, c7, c9:** Undo restores output and code identity to base, step-by-step through chains, and across 20 install/undo cycles.
- **c5–c6:** Mid-install faults roll back all ranks; unsupported or silent no-op edits are refused without mutating state.
- **c8:** Crashed, hung, or externally tampered workers mark the lab `poisoned` until respawned at base.
- **c10:** Startup-only edits whose new code never ran are reported not proven (`zero_witness`).

**Limits:** A live check proves the edited path ran and matched on the chosen workload, not equivalence on all inputs. Engines need a Python control layer reachable between steps, and one agent operates one warm process at a time.

</details>

<details>
<summary><b>Repository layout &amp; full CPU test suite</b></summary>

```
core/livelab/        shared stdlib runtime (kernel/, fastrsi/, rsibox/, conformance/, survey/, forksafe.py, INTERFACE.md)
core/agents/         prompts for coding agents (livelab_overview, developer, orchestrator, toolsmith)
repos/<engine>/      vllm, fastvideo, megatron, verl, sglang, miles (docs/, kb/, livelab/, examples/)
docs/                design notes on kernels, CUDA graphs, lessons learned, and CI
.github/             CPU workflows, automated review service, leak check, and Modal GPU CI runner
scripts/             fact and README consistency checks (verify_facts.py)
template/            scaffold for adding a new engine
```

Full CPU CI commands (`.github/workflows/cpu-tests.yml`):

```sh
PYTHONPATH=core/livelab python3 -m fastrsi doctor
python3 -m unittest discover -s core/livelab/kernel -t core/livelab
python3 -m unittest discover -s core/livelab/fastrsi -t core/livelab
python3 -m unittest discover -s core/livelab -p 'test_forksafe.py'
python3 -m unittest discover -s core/livelab/survey -t core/livelab
(cd core/livelab && python3 -m unittest discover -s conformance -t .)
python3 -m unittest discover -s template/livelab/adapter -t template/livelab
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/verify_facts.py --strict
python3 .github/modal-ci/test_modal_ci.py -v
```

</details>

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for local setup and PR guidelines, and [`AGENTS.md`](AGENTS.md) for the full contributor and coding-agent rules.

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
