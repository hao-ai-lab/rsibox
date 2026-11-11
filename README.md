# FastRSI

Test an edit to a running ML engine without restarting it.

A coding agent (or a person) optimizing vLLM, Megatron-LM or FastVideo pays a full engine restart
for every idea it tries. FastRSI installs the edit into the warm engine process, proves that the new
code actually ran, measures it, and undoes it. When an edit cannot be swapped in safely, it refuses
and says which restart boundary the edit needs.

## Why

A one-line vLLM scheduler edit on one GB200, warm on-disk caches. Each row keeps one more piece of
state instead of rebuilding it. Source: [`repos/vllm/kb/costs.md`](repos/vllm/kb/costs.md).

| Step removed | Check time | vs restart |
|---|---:|---:|
| Restart with caches (baseline) | 228 s | 1.0× |
| + reuse autotune config | 157 s | 1.5× |
| + keep model weights | 100 s (derived) | 2.3× |
| + keep CUDA graphs | 58 s (derived) | 3.9× |
| + keep the live process (hot swap) | 23.6 s | 9.6× |
| then skip the logprob check | 14.1 s | |
| then test with a small batch | 2.4 s | 95× end to end |

The two derived rows are computed by removing one phase from a measured run, not measured on their
own. 204 s of the 228 s restart is setup that the edit did not invalidate. The other engines:

- FastVideo (MiniMax H3, 1 GB200): a live swap reaches its first checked video in 6.7 s (0.72 s
  edit-to-run plus a 6.0 s video), against about 72 s for a warm restart and about 507 s cold.
  Source: [`repos/fastvideo/kb/costs.md`](repos/fastvideo/kb/costs.md).
- Megatron-LM (2-layer model, TP 2 on 2 GB200): install 0.05 s and a 3-step check 0.45 s, against
  57.5 s to restart to the same point. Source: [`repos/megatron/kb/costs.md`](repos/megatron/kb/costs.md).

## Quickstart

Python 3.11 or newer, no GPU, no third-party packages. From the repository root:

```sh
# The 11 conformance cases on the toy lab (about 1.5 s on a laptop)
PYTHONPATH=core/livelab python3 -m conformance --lab kernel.toy_lab:ToyLab
# Installed in a venv, so it runs from any directory
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
python3 -m conformance --lab kernel.toy_lab:ToyLab
# The same 11 cases on a FastVideo-shaped toy engine with real worker processes (about 5 s)
PYTHONPATH=core/livelab:repos/fastvideo/livelab python3 -m conformance --lab adapter.toy_lab:ToyFastVideoLab
```

Excerpt of the output of the first command:

```
case                  result  seconds  evidence
c2_visible            pass      0.004  output [11, 13, 15, 17, 21, 27] x2 ranks -> [12, 14, 16, 18, 22, 28] x2 ranks
c4_uninstall          pass      0.006  uninstall ['txn-42e962fd'] in 0.000939 s; identity 665aa2ee3a8d968d restored
c8_crash_and_hang     pass      1.035  crash: poisoned by ['participant_failed'], respawn 0.003573 s
c10_not_proven        pass      0.044  proven False, findings ['zero_witness', 'zero_witness']

11/11 as expected
```

The toy lab is two worker ranks stepping together. Each case drives it only through the lab
interface, so the same cases run against real engines. The toy cannot show GPU state, CUDA graphs or
real RPC latency. What passing shows ([`core/livelab/conformance/README.md`](core/livelab/conformance/README.md)):

- c0-c2: the workload repeats exactly; an edit that keeps behavior leaves output bit-exact; a visible
  edit changes output, and its witness fires on every rank.
- c3: the output after a live edit (and after a chain of 3) equals a fresh process started with it.
- c4, c7, c9: undo returns output and code identity to base, also one edit at a time through a chain,
  and after 20 install/undo cycles.
- c5: a fault halfway through an install rolls every rank back to base.
- c6: unsupported edits are refused with a reason and change nothing; an install that would silently
  change nothing is refused too, not reported as a success.
- c8: a crashed, hung or lost worker, or code changed behind the lab's back, marks the lab poisoned;
  it refuses installs until the worker group is respawned at base.
- c10: an edit whose new code never ran (startup-only code) is reported not proven (`zero_witness`),
  while a fresh process shows the edit does change the output.

## What you can do with it

Every engine's lab implements the same calls ([`core/livelab/INTERFACE.md`](core/livelab/INTERFACE.md), version 0, draft):

- `install(patch)`: stage and bind an edit on every rank at once, or refuse it with a reason before touching state.
- `check(workload)`: run a probe in the warm engine. Returns latency or throughput, output vs base,
  whether the edited code ran, and the check's own cost.
- `reset()`: drain in-flight work and restore caches, buffers, RNG and iterators, so the next check
  starts from the same state.
- `uninstall()`: revert to base in place, with proof that base holds again.
- `status()`: `base`, `edited` or `poisoned`.

## What is in the repo today

| Engine | CPU, in CI | Run on GPUs, with a written result | Not there yet |
|---|---|---|---|
| Toy lab | kernel unit tests; conformance 11/11 | not applicable | |
| FastVideo | adapter tests with real worker processes; conformance 11/11 on a FastVideo-shaped toy | conformance on MiniMax H3, 11/11 on 2 GPUs (10/11 on 1: c2 needs two ranks) ([h3-conformance](repos/fastvideo/examples/h3-conformance/README.md)); one live edit ([h3-live-edit](repos/fastvideo/examples/h3-live-edit/README.md)); upstream PR replays ([pr-replay](repos/fastvideo/examples/pr-replay/README.md), [replay-1882](repos/fastvideo/examples/replay-1882/README.md)) | edits inside compiled or graph-captured code; Triton kernels edited in place; 4 ranks |
| Megatron-LM | lab tests (overlay, mailbox, phase stamps, zygote; torch cases skip) | conformance 11/11 on a 2-layer model, TP 2 on 2 GB200 ([livelab](repos/megatron/livelab/README.md)); a Codex hill-climb ([codex-hillclimb](repos/megatron/examples/codex-hillclimb/README.md)); 8 upstream PRs ([upstream-replay](repos/megatron/examples/upstream-replay/README.md)) | pipeline parallelism; conformance beyond 2 ranks; CUDA graph edits |
| vLLM | adapter tests on in-process toy ranks; services tests against fake `vllm serve` processes | measurements only, no walkthrough: the ladder above, zygote fork and weight holder ([kb/costs.md](repos/vllm/kb/costs.md)), Triton rebuild and graph recapture ([docs/architecture.md](repos/vllm/docs/architecture.md)) | conformance on a real vLLM engine; the DeepSeek-V4.1-Flash prefill replay is a plan with the three upstream patches, no receipts yet ([dsv41-prefill](repos/vllm/examples/dsv41-prefill/README.md)); component restart; KV holder |

- **FastVideo replay-1882:** upstream PR #1882, reversed, was swapped into a warm H3 engine (install
  0.086 s), checked 3 times and undone (0.030 s). The witness showed 0 calls: this workload never
  reaches the edited code, so the replay reads "not proven" instead of passing.
- **Megatron codex-hillclimb:** 82 iterations on the Qwen3-14B architecture (TP 2, 2 GB200), no
  restart between edits. The 3 kept edits: +5.1% tokens/s/GPU (median of 4 paired rounds); +7.1% on
  the Qwen3-32B architecture at TP 4. One of them (iteration 12) ran in a private harness, not the lab.

## How it works

1. **Edits commit in the step gap.** An edit is staged and analyzed while the engine keeps running.
   In the gap between two generation or train steps, a coordinator commits it on all ranks or none.
2. **A witness proves the new code ran.** Each edited function counts its calls on every rank. Code
   inside captured CUDA graphs or compiled Triton kernels counts only after the graph is recaptured
   or the kernel rebuilt.
3. **Undo is journaled.** Every change logs its inverse. `uninstall` restores the exact code objects
   and checks code identity against base.
4. **Refusals name a restart boundary.** Edits that change constructors, add fields, touch module
   state or change the distributed layout are refused, with a finding that names the cheapest
   restart that can run them.

## Mechanisms by restart boundary

| Boundary | Mechanism | Edits it handles | File | Measured cost |
|---|---|---|---|---|
| In place, Python | swap function code on every rank, journaled undo | function and method bodies | `core/livelab/kernel/swap.py` | FastVideo install 0.06 s, uninstall 0.014 s ([costs](repos/fastvideo/kb/costs.md)); Megatron install 0.05 s, uninstall 0.017 s ([costs](repos/megatron/kb/costs.md)) |
| In place, Triton | rebuild `@triton.jit` kernels and their callers; undo restores the compiled binaries | Triton kernel bodies | `core/livelab/kernel/jit.py` | 0.07 s ([vLLM architecture](repos/vllm/docs/architecture.md)) |
| In place, CUDA graphs | recapture the affected graphs in the live worker | edits inside graph-captured code | `repos/vllm/livelab/adapter/recapture.py` | 3.0-5.6 s per region, 14.4 s all graphs, Qwen3-8B ([vLLM architecture](repos/vllm/docs/architecture.md)) |
| In process, module rebuild | re-import the edited module and rebuild the live objects | constructors, new fields, module constants | `repos/fastvideo/livelab/services/rebuild.py` | 0.19-1.27 s, 1 GPU ([costs](repos/fastvideo/kb/costs.md)) |
| New process, warm | fork from a server with imports done; weights served over CUDA IPC by a holder | launch flags, startup code, anything the in-place paths refuse | `core/livelab/forksafe.py`, `repos/*/livelab/services/zygote.py`, `weights.py`, `holder.py`, `repos/megatron/livelab/fastrsi_megatron/zygote.py` | vLLM 7.37 s to `/health`, Qwen3-8B ([costs](repos/vllm/kb/costs.md)); FastVideo respawn 4.78 s and 5.62 s, weights held ([costs](repos/fastvideo/kb/costs.md)); Megatron 15.0-16.5 s to first step, 14B ([costs](repos/megatron/kb/costs.md)) |
| New layout, same weights | reshard held weights between TP, DP and EP in memory | parallelism changes | `repos/vllm/livelab/services/reshard.py` | not yet measured here |
| Full restart | new process, warm disk caches | anything else | | 228 s in the vLLM ladder above |

## Repository layout

```
core/livelab/        shared runtime, standard library only
  kernel/            transaction kernel (locate, swap, witness, journal, coordinator) and the toy lab
  conformance/       the 11 cases every lab must pass
  INTERFACE.md       the lab interface
core/agents/         prompts for the developer, toolsmith and orchestrator agents
repos/<engine>/      vllm, fastvideo, megatron
  docs/, kb/         architecture and restart boundaries; measured costs, each with its source
  livelab/           the engine's lab, built on core/livelab
  examples/<name>/   walkthroughs of replayed changes
template/            scaffold for a new engine
```

## Tests

What CPU CI runs (`.github/workflows/cpu-tests.yml`), next to the two conformance runs of the
quickstart; torch, vLLM and Linux-only cases skip on a laptop:

```sh
python3 -m unittest discover -s core/livelab/kernel -t core/livelab
python3 -m unittest discover -s core/livelab -p 'test_forksafe.py'
(cd core/livelab && python3 -m unittest discover -s conformance -t .)
python3 -m unittest discover -s repos/vllm/livelab/adapter -t repos/vllm/livelab
python3 -m pytest -rs repos/vllm/livelab/services            # needs pytest
python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab
python3 -m unittest discover -s repos/fastvideo/livelab/services -t repos/fastvideo/livelab
python3 -m unittest discover -s repos/megatron/livelab/fastrsi_megatron -t repos/megatron/livelab
python3 -m unittest discover -s repos/megatron/tools
```

Rules for contributors, people and agents: [`AGENTS.md`](AGENTS.md).

## Limits

A live check shows that the edited path ran and that its output matched on the chosen workload; it
does not show the edit is equivalent on all inputs. An engine needs a Python control layer the lab
can reach between steps. One agent works on one warm process at a time.

## License

Apache-2.0 ([`LICENSE`](LICENSE)).
