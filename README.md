# FastRSI

Test an edit to a running ML engine without restarting it.

A coding agent (or a person) optimizing vLLM, Megatron-LM, FastVideo, veRL, SGLang or miles pays a full
engine restart for every idea it tries. FastRSI installs the edit into the warm engine process,
proves that the new code actually ran, measures it, and undoes it. When an edit cannot be swapped
in safely, it refuses and says which restart boundary the edit needs.

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
| then test with a small batch | 2.4 s | 95× end to end (4.8 s / 48× with the logprob check kept) |

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
# Check environment, kernel, and engine adapters without importing torch or CUDA
PYTHONPATH=core/livelab python3 -m fastrsi doctor
PYTHONPATH=core/livelab python3 -m fastrsi list

# The 11 conformance cases on the toy lab (about 1.5 s on a laptop)
PYTHONPATH=core/livelab python3 -m fastrsi conformance toy

# Installed in a venv, so `fastrsi` runs from any directory
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
python3 -m fastrsi conformance toy

# The same 11 cases on multi-process CPU engine toys (fastvideo, sglang, verl, miles)
PYTHONPATH=core/livelab python3 -m fastrsi conformance fastvideo
PYTHONPATH=core/livelab python3 -m fastrsi conformance sglang
PYTHONPATH=core/livelab python3 -m fastrsi conformance verl
PYTHONPATH=core/livelab python3 -m fastrsi conformance miles
```

Excerpt of the conformance output:

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
| Toy lab | kernel & `fastrsi` unit tests; conformance 11/11 | not applicable | |
| FastVideo | adapter & services tests with real worker processes; conformance 11/11 on `ToyFastVideoLab` | conformance on MiniMax H3, 11/11 on 1×, 2× (SP2), and 4× GB200 (SP4) via `fastrsi conformance fastvideo --real` ([h3-conformance](repos/fastvideo/examples/h3-conformance/README.md)); live 01–20 hill-climb (1.43–1.45× 1G, 1.51–1.52× 2G) with Patch 20 shared-pool single-shape graph cache (124→243→124→362→124, 5/5 bit-exact on 1×/2×/4×, 0 OOM) ([h3-hillclimb](repos/fastvideo/patches/h3-hillclimb/README.md)); zygote+holder respawn 3.05–3.53 s 1G / 6.23–7.27 s 2G ([kb/costs.md](repos/fastvideo/kb/costs.md)); upstream PR replays ([pr-replay](repos/fastvideo/examples/pr-replay/README.md), [replay-1882](repos/fastvideo/examples/replay-1882/README.md)) | edits inside `torch.compile` (VAE) or VAE NVFP4 CUDA graphs; Triton signature/decorator changes (top-level `@triton.jit` bodies rebuild in place); native `fastvideo-kernel` code |
| Megatron-LM | lab tests (overlay, mailbox, phase stamps, zygote; torch cases skip) | conformance 11/11 on 2× GB200 TP2, 4× GB200 TP2/DP2, and 4× GB200 TP4 via `fastrsi conformance megatron --real` ([livelab](repos/megatron/livelab/README.md)); 30/30 paired 8-layer Qwen3-14B dense runs (5.52–5.61 s live vs 41.83–42.27 s fresh, 0% diff), EP4 MoE 8-gate walkthrough, and 10-cycle warm-fork zygote soak ([kb/costs.md](repos/megatron/kb/costs.md)); Codex hill-climb with public `te_wgrad.py` repair ([codex-hillclimb](repos/megatron/examples/codex-hillclimb/README.md)); 8 upstream PRs ([upstream-replay](repos/megatron/examples/upstream-replay/README.md)) | pipeline parallelism > 1; `torch.compile` (`jit_fuser`) and CUDA graph edits; constructors/optimizer rebuild |
| vLLM | adapter tests on in-process toy ranks (incl. `native.py` unit checks); services tests against fake `vllm serve` processes; `live-edit-qwen3` unit checks | conformance 11/11 on 1× B200 eager, 2× B200 TP2, and 1× B200 V2 `FULL_DECODE_ONLY` CUDA graphs (including via `fastrsi conformance vllm --real`) ([adapter](repos/vllm/livelab/adapter/README.md)); C++/CUDA `NATIVE` hot-reload verified on B200 (`native.py` on PR `#49750`: 17.12 s commit + 0.83 s R1 recapture vs 49.47 s restart, 6.55 ms cascade undo, `max \|Δlogprob\| == 0.0`, 71/80 = 89% PR coverage) ([verify_native_pr49750.py](repos/vllm/examples/live-edit-native/verify_native_pr49750.py)); live TP2 ↔ EP2 holder-to-holder resharding (`reshard_live.py`: 1.0–2.8 s write, 0.36–1.18 s drain-aware stop, 49.4–52.8 s switch with zygote+kvplan reuse, `max \|Δlogprob\| == 0.0`) and live `--scheduling-policy` switching (0.218–0.374 ms core apply, HTTP 400 FCFS guard) ([kb/costs.md](repos/vllm/kb/costs.md)); Qwen3-8B walkthrough ([live-edit-qwen3](repos/vllm/examples/live-edit-qwen3/README.md)); Triton rebuild, constructor reinit (`#62872`), and V1 selective graph recapture ([docs/architecture.md](repos/vllm/docs/architecture.md)) | DeepSeek-V4.1-Flash prefill replay receipts ([dsv41-prefill](repos/vllm/examples/dsv41-prefill/README.md)); data parallel > 1; component restart; KV holder |
| veRL | adapter tests (incl. real local Ray cluster test when Ray is installed), CLI/probe tool tests; conformance 11/11 on `ToyVerlLab` via `fastrsi conformance verl` | conformance 11/11 on 4× GB200 Qwen3-0.6B GSM8K GRPO (`VerlConformanceLab`, 21 participants: driver, 8 reward, 8 `AgentLoopWorkerTQ`, 4 FSDP ranks; 6844 s, 113 receipts) and 8-actor `SingleTurnAgentLoop.run` live swap (`32×8` calls, 7.54 s install) ([livelab](repos/verl/livelab/README.md)); batch A 640-response live swap (117.2 s reward / 104.6 s advantage vs 259.8 s restart) & coupled `S0` restore (1.10–1.16 s) ([kb/costs.md](repos/verl/kb/costs.md)); GSM8K walkthrough and offline `rescore` (0.85 ms) / `replay_adv` (16–29 ms) ([grpo-gsm8k](repos/verl/examples/grpo-gsm8k/README.md)) | Megatron trainer path with gated `RayPPOTrainer`; multi-node jobs; in-place rebuild & Ray actor class layout changes; CUDA-graph rollout under `full_determinism=True` on vLLM 0.31 / torch 2.13 |
| SGLang | adapter & zygote tests; conformance 11/11 on `ToySglangLab` | conformance 11/11 on 2× GB200 TP2 Qwen3-0.6B (install median 0.091 s / undo 0.008 s, R1 decode graph recapture 0.839 s, `SglangZygote` 11.75 s; PR #163/#164); v0 Qwen3-8B conformance & overhead receipts plus static coverage survey (49.3% live / 66.1% in `srt`) ([kb/costs.md](repos/sglang/kb/costs.md)) | `dp_size > 1` and `--enable-dp-attention`; PD disaggregation; in-place rebuild (constructors); non-scheduler processes |
| miles | adapter tests (incl. Ray actor gate); conformance 11/11 on 3-process `ToyMilesLab` (~1.69 s) | 2× GB200 `miles` 0.2.0 multi-actor live swap (0.069 s, `EpochDataBuffer` `code_epoch` drain, coupled S0 weight sync; PR #162); 4× GB200 Qwen3-0.6B GRPO startup baseline (60–95 s cold) ([kb/costs.md](repos/miles/kb/costs.md)) | standalone GPU conformance launcher (`livelab.json` sets `real: null`); in-place rebuild; placement-group respawn |

- **FastVideo replay-1882:** upstream PR #1882, reversed, was swapped into a warm H3 engine (install
  0.086 s), checked 3 times and undone (0.030 s). The witness showed 0 calls: this workload never
  reaches the edited code, so the replay reads "not proven" instead of passing.
- **vLLM live-edit-qwen3:** three one-line edits on a running Qwen3-8B server (1 GB200). Checking
  each by restart took 63 s; live took 1.1 s (install 0.03 to 0.14 s, engine loop stalled about
  1 ms), with the same output as the restarted server. One edit changed nothing either way, and
  only the live check said why: the witness showed 0 calls.
- **Megatron codex-hillclimb:** 82 iterations on the Qwen3-14B architecture (TP 2, 2 GB200), no
  restart between edits. The 3 kept edits: +5.1% tokens/s/GPU (median of 4 paired rounds); +7.1% on
  the Qwen3-32B architecture at TP 4. Iteration 12 (TE fused weight-gradient accumulation)
  originally ran in a private harness and is now reproduced in the public lab as a journaled repair
  ([`te_wgrad.py`](repos/megatron/examples/codex-hillclimb/te_wgrad.py)).

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
| In place, C++/CUDA | compile edited `.cu` + versioned `reg.cu` under `flock`, rebind `torch.ops._C` + `R1` recapture, `lib._destroy()` undo | single-unit C++/CUDA kernels (`layernorm_kernels.cu`) | `repos/vllm/livelab/adapter/native.py` | 17.12 s commit (17.09 s `nvcc`) + 0.83 s R1 recapture vs 49.47 s restart; undo 6.55 ms (PR `#49750`, B200; [costs](repos/vllm/kb/costs.md)) |
| In process, module rebuild | re-import the edited module and rebuild the live objects | constructors, new fields, module constants | `repos/fastvideo/livelab/services/rebuild.py` | 0.19-1.27 s, 1 GPU ([costs](repos/fastvideo/kb/costs.md)) |
| New process, warm | fork from a server with imports done; weights served over CUDA IPC by a holder | launch flags, startup code, anything the in-place paths refuse | `core/livelab/forksafe.py`, `repos/*/livelab/services/zygote.py`, `weights.py`, `holder.py`, `repos/megatron/livelab/fastrsi_megatron/zygote.py` | vLLM 7.37 s to `/health`, Qwen3-8B ([costs](repos/vllm/kb/costs.md)); FastVideo respawn 3.05–3.53 s (1G) / 6.23–7.27 s (2G), weights held ([costs](repos/fastvideo/kb/costs.md)); Megatron 15.0–16.5 s to first step, 14B ([costs](repos/megatron/kb/costs.md)) |
| New layout, same weights | reshard held weights between TP and EP in memory over CUDA IPC | expert-parallelism layout switches | `repos/vllm/livelab/services/reshard_live.py` | 1.0–2.8 s write, 0.36–1.18 s drain-aware stop, 49.4–52.8 s switch with zygote+kvplan reuse (TP2 ↔ EP2, 2× B200; [costs](repos/vllm/kb/costs.md)) |
| Full restart | new process, warm disk caches | anything else | | 228 s in the vLLM ladder above |

## Repository layout

```
core/livelab/        shared runtime, standard library only
  kernel/            transaction kernel (locate, swap, witness, journal, jit, reinit, coordinator) and the toy lab
  fastrsi/           shared stdlib kit (BaseLab, diffs, ipc, proc, harness, unified CLI)
  conformance/       the 11 cases every lab must pass
  INTERFACE.md       the lab interface
core/agents/         prompts for the developer, toolsmith and orchestrator agents
repos/<engine>/      vllm, fastvideo, megatron, verl, sglang, miles
  docs/, kb/         architecture and restart boundaries; measured costs, each with its source
  livelab/           the engine's lab, built on core/livelab
  examples/<name>/   walkthroughs of replayed changes
scripts/             fact and README consistency checks (verify_facts.py)
template/            scaffold for a new engine
```

## Tests

What CPU CI runs (`.github/workflows/cpu-tests.yml`), next to the `fastrsi` doctor and conformance
runs of the quickstart; torch, vLLM and Linux-only cases skip on a laptop:

```sh
PYTHONPATH=core/livelab python3 -m fastrsi doctor
python3 -m unittest discover -s core/livelab/kernel -t core/livelab
python3 -m unittest discover -s core/livelab/fastrsi -t core/livelab
python3 -m unittest discover -s core/livelab -p 'test_forksafe.py'
(cd core/livelab && python3 -m unittest discover -s conformance -t .)
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/verify_facts.py
python3 -m unittest discover -s repos/vllm/livelab/adapter -t repos/vllm/livelab
python3 -m pytest -rs repos/vllm/livelab/services            # needs pytest
python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab
python3 -m unittest discover -s repos/fastvideo/livelab/services -t repos/fastvideo/livelab
python3 -m unittest discover -s repos/megatron/livelab/fastrsi_megatron -t repos/megatron/livelab
python3 -m unittest discover -s repos/megatron/tools
python3 -m unittest discover -s repos/verl/livelab/adapter -t repos/verl/livelab
python3 -m unittest discover -s repos/verl/tools -t repos/verl
python3 -m unittest discover -s repos/sglang/livelab/adapter -t repos/sglang/livelab
python3 -m unittest discover -s repos/sglang/livelab/services -t repos/sglang/livelab
python3 -m unittest discover -s repos/miles/livelab/adapter -t repos/miles/livelab
```

Rules for contributors, people and agents: [`AGENTS.md`](AGENTS.md).

## Limits

A live check shows that the edited path ran and that its output matched on the chosen workload; it
does not show the edit is equivalent on all inputs. An engine needs a Python control layer the lab
can reach between steps. One agent works on one warm process at a time.

## License

Apache-2.0 ([`LICENSE`](LICENSE)).
