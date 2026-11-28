# Quick Start

Get an `rsibox` live sandbox running, hot-reload a function between steps, run live execution checks, and roll back cleanly. Requires Python 3.11+.

## Install

```bash
git clone https://github.com/hao-ai-lab/rsibox.git && cd rsibox
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
```

*(Without installing the package, prefix commands with `PYTHONPATH=core/livelab python3 -m rsibox`.)*

---

## Option A: No GPU (30 seconds)

Run preflight diagnostics and the CPU sandbox verification suite, then drive a multi-rank `sglang` toy box end to end:

```bash
# 1. Verify core runtime and discovered engine adapters (no GPUs required)
rsibox doctor
rsibox conformance toy

# 2. Start a live sglang toy box and point client commands at its box file
rsibox up --launcher local --lab sglang --state /tmp/box1
export RSIBOX=/tmp/box1/box.json
rsibox wait

# 3. Fetch the live module, make a 1-line edit, and hot-reload it in place
rsibox source adapter.fixture.toysgl.srt.layers.sampler > /tmp/sampler.py
sed -i.bak 's/return best/return (best + 1) % len(row)/' /tmp/sampler.py
rsibox install --file adapter.fixture.toysgl.srt.layers.sampler=/tmp/sampler.py

# 4. Run the workload (verifies output + per-rank witness counts), then roll back and stop
rsibox check
rsibox uninstall
rsibox down --state /tmp/box1
```

---

## Option B: Real GPU (vLLM example)

With `vllm` installed in the active Python environment and a GPU assigned (`CUDA_VISIBLE_DEVICES=0`):

```bash
# 1. Launch a warm vLLM server inside a local box daemon
rsibox up --launcher local --lab vllm --real --state /tmp/box-gpu \
  --env RSIBOX_VLLM_MODEL=Qwen/Qwen3-8B
export RSIBOX=/tmp/box-gpu/box.json
rsibox wait

# 2. Pull the live sampler module, apply a 1-line edit, and install it
rsibox source vllm.v1.worker.gpu.sample.sampler > /tmp/vllm_sampler.py
# ... edit one function in /tmp/vllm_sampler.py (e.g., tweak logit/sampling logic) ...
rsibox install --file vllm.v1.worker.gpu.sample.sampler=/tmp/vllm_sampler.py

# 3. Verify output and witness proof on the warm GPU engine, then undo and tear down
rsibox check
rsibox uninstall
rsibox down --state /tmp/box-gpu
```

Run `rsibox conformance <engine> --real -- --help` to inspect engine-specific flags (for example `--tp`, `--gpus`, or model env vars). For Docker, gVisor, or Slurm launchers, see [`core/livelab/rsibox/README.md`](../core/livelab/rsibox/README.md).

---

## What Just Happened?

1. **`rsibox up` + `rsibox wait`:** Started the sandbox daemon and warmed up the engine once, writing connection details and an auth token to `$RSIBOX` (`box.json`).
2. **`rsibox source` + `rsibox install`:** Pulled the currently live Python module text and staged your replacement. In a brief step gap, every worker rank verified the base matched and swapped the modified code object atomically (`R0 swap`), or refused fail-closed if the edit touched a restart boundary.
3. **`rsibox check`:** Ran the fixed workload and returned `"proven": true` along with per-rank `"witness"` call counts proving the edited function actually executed on every rank.
4. **`rsibox uninstall` + `rsibox down`:** Replayed the inverse mutation journal in reverse order—restoring the exact original code objects without restarting—and shut down the daemon.

## Next Steps

- **[`docs/concepts.md`](concepts.md)** — Hot-reload layers, live execution checks (`witness`), journaled rollback, and restart boundaries.
- **[`docs/engines.md`](engines.md)** — Supported ML engines, measured speedups, and repository layout.
- **[`core/agents/livelab_overview.md`](../core/agents/livelab_overview.md)** — Coding-agent skill reference (installable as `~/.claude/skills/rsibox/SKILL.md`; see [`README.md`](../README.md#use-with-your-coding-agent)).
