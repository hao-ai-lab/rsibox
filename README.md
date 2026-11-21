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
| Toy lab | kernel, `forksafe`, `survey`, `fastrsi` & template unit tests; conformance 11/11 | not applicable | |
| FastVideo | adapter & services tests (`33` Linux GPU `services` tests w/ concurrent `holder` & pre-fork `torch_key`); conformance 11/11 on `ToyFastVideoLab` | conformance on MiniMax H3, 11/11 on 1×, 2× (SP2), and 4× GB200 (SP4) via `fastrsi conformance fastvideo --real` ([h3-conformance](repos/fastvideo/examples/h3-conformance/README.md)), plus `--zygote --holder` with inline `_HolderPickler` CPU IPC passing full **11/11** on both `2× SP2` (`856 s`, `108` receipts, `c8 208.13 s` [`6.0–9.3 s` respawns], `c9` `0.0579 s` / `0.0177 s`, `0` pipe errors) and `4× SP4` (`108` receipts, `c8 226.50 s` [`10.1–13.7 s` respawns], `c9` `0.1121 s` / `0.0329 s`, `0 MiB × 4`, plus earlier `5/5` & `4/4` fast-fail); live 01–20 hill-climb (1.43–1.45× 1G, 1.51–1.52× 2G) with Patch 20 shared-pool single-shape graph cache (124→243→124→362→124, 5/5 bit-exact on 1×/2×/4×, 0 OOM) ([h3-hillclimb](repos/fastvideo/patches/h3-hillclimb/README.md)); unified non-invasive `FastH3-QAD-v2` phase dissection (`0.000 s` `not_split`, `48.133 dB` PSNR bit-exact to fresh): `1× GB200` `206.23 s` cold -> `40.64 s` warm -> `17.61 s` zygote+holder -> **`14.22–14.31 s`** w/ concurrent `holder` + `audio_vae` `.detach()` IPC (`weight_load 1.17 -> 0.86–0.88 s`, `respawn 2.85–2.87 s`, pre-fork `torch_key 0.17–0.18 ms`, `TRITON_CACHE_AUTOTUNING=1` `0.000 s` bench) -> `6.40–6.41 s` live (`32.2×`) -> `2.25–2.30 s` `2`-fwd (`91.8×`), `2× SP2` `200.07 s` cold -> `45.98 s` warm -> `19.43 s` -> **`16.02–16.10 s`** (`respawn 6.16 s`) -> `4.76 s` live (`42.1×`) -> `1.87–1.97 s` `2`-fwd (`107.2×`), and `4× SP4` `48.87 s` warm -> **`21.14–21.87 s`** zygote+holder (`4/4` concurrent IPC `weight_load 2.37–2.46 s` vs `14.34 s` disk [`6.0×`], bit-exact audio) -> `3.35 s` live -> `1.66 s` `2`-fwd (`29.4×` vs warm) ([kb/costs.md](repos/fastvideo/kb/costs.md)); upstream PR replays ([pr-replay](repos/fastvideo/examples/pr-replay/README.md), [replay-1882](repos/fastvideo/examples/replay-1882/README.md)) | edits inside `torch.compile` (VAE) or VAE NVFP4 CUDA graphs; Triton signature/decorator changes (top-level `@triton.jit` bodies rebuild in place); native `fastvideo-kernel` code |
| Megatron-LM | lab tests (overlay, mailbox, phase stamps, zygote; torch cases skip) | conformance 11/11 on 2× GB200 TP2, 4× GB200 TP2/DP2, 4× GB200 TP4, 4× GB200 TP2/PP2, and 4× GB200 TP2/PP2/VPP2 (`8` layers, `96` calls/rank across all 4 ranks, `c9` install median `0.0468 s` / undo `0.0170 s`, `c8` respawns `30.04–31.05 s`) via `fastrsi conformance megatron --real` ([livelab](repos/megatron/livelab/README.md)); 30/30 paired 8-layer Qwen3-14B dense runs (5.52–5.61 s live vs 41.83–42.27 s fresh, 0% diff), EP4 MoE 8-gate walkthrough, Qwen3-30B-A3B 24L MXFP8 EP4 0-warmup 10-step (`45.52 s`), 3-step probe (`17.04 s` median, `15.6×` vs `266.7 s` restart; `probe-pairs6` `10/10` agreement), and 1-step probes (`8.02 s` median `ng`, `33.3×`; `9.63 s` median `g` with `c00-B` `13.84 s` router recapture), plus 10-cycle warm-fork zygote soak ([kb/costs.md](repos/megatron/kb/costs.md)); Codex hill-climb with public `te_wgrad.py` repair ([codex-hillclimb](repos/megatron/examples/codex-hillclimb/README.md)); 8 upstream PRs ([upstream-replay](repos/megatron/examples/upstream-replay/README.md)) | `torch.compile` (`jit_fuser`) and CUDA graph edits; constructors/optimizer rebuild |
| vLLM | adapter tests (`70/70` Linux GPU `adapter` incl. `_stage_csrc_overlay` & V2 `ModelCudaGraphManager`); services tests; `live-edit-qwen3` unit checks | conformance 11/11 on 1× B200 eager, 2× B200 TP2 V1, 1× B200 V2 `FULL_DECODE_ONLY` CUDA graphs, and 2× B200 TP2 V2 `FULL_DECODE_ONLY` CUDA graphs (`BATCH_INVARIANT=1`, `113` receipts, `c9` `0.0857 s` / `0.0440 s`, `0 MiB`) (including via `fastrsi conformance vllm --real`, plus post-refactor 3-case GPU smoke) ([adapter](repos/vllm/livelab/adapter/README.md)); selective `R1` + `recapture_extend` on V2 `ModelCudaGraphManager` (1× B200: `sizes=[1]` `0.299 s`, extend `[2, 4]` `0.360 s`, `2.93 ms` undo, `0.0 MiB` reserved growth across 5 cycles, `max \|Δlogprob\| = 0.0`); C++/CUDA `NATIVE` hot-reload verified on B200 with `_stage_csrc_overlay` (`1.62–1.90 ms` vs `12.40 ms` `copytree`, `6.5–7.7×`, `0` `csrc` mutation) across **3 distinct CUDA kernel families** on `TP=2` `FULL_DECODE_ONLY` (`max \|Δlogprob\| == 0.0` vs fresh restart): `layernorm_kernels.cu` `fused_add_rms_norm` PR `#49750` (`19.22 s` V1 TP1 / `13.62–13.81 s` TP2 / `14.13 s` V2 TP2 vs `46.97–49.47 s`, `7.0–9.7 ms` undo, `71/80 = 89%` PR coverage), `pos_encoding_kernels.cu` `rotary_embedding` (**`7.29 s`** vs `41.18 s`, `live [288,288]`, `cap [216,216]`, **`3.20 ms` undo**), and `activation_kernels.cu` `silu_and_mul` (**`10.36 s`** vs `46.29 s`, `live [288,288]`, `cap [216,216]`, **`5.26 ms` undo**) ([verify_native_pr49750.py](repos/vllm/examples/live-edit-native/verify_native_pr49750.py)); `134/134` `livelab/services` tests (`52/52` E2E + `72/72` `adapter+ex` after `-265` LOC consolidation); `weights.py` TP=2 + `--zygote --kvplan` (`43.58 -> 29.49 s`); live TP2 ↔ EP2 resharding (`1.0–2.8 s` write, `0.36–1.18 s` drain stop, `49.4–52.8 s` switch), live `--scheduling-policy` (`0.22–0.37 ms`), `--max-num-batched-tokens` (`0.035/0.022 ms`), and `core_hook.py` co-existence (`0.191–0.260 ms` + `79.2 ms` swap / `38.1 ms` undo) ([kb/costs.md](repos/vllm/kb/costs.md)); Qwen3-8B walkthrough ([live-edit-qwen3](repos/vllm/examples/live-edit-qwen3/README.md)); Triton rebuild, `#62872` constructor reinit, V1/V2 selective recapture ([docs/architecture.md](repos/vllm/docs/architecture.md)) | DeepSeek-V4.1-Flash prefill replay receipts ([dsv41-prefill](repos/vllm/examples/dsv41-prefill/README.md)); data parallel > 1; component restart; KV holder |
| veRL | adapter & PEP 578 negative-cache tests (`248/248` Linux `core/livelab`, `26/26` `verl`, `15/15` `miles`); conformance 11/11 on `ToyVerlLab` via `fastrsi conformance verl` | conformance **11/11** on 4× GB200 Qwen3-0.6B GSM8K GRPO (`VerlConformanceLab`, 21 participants: driver, 8 reward, 8 `AgentLoopWorkerTQ`, 4 FSDP ranks) in both **eager** (`6844 s` / `6879.8 s`, `113` receipts) and **FSDP CUDA-graph** mode (`enforce_eager=False`, `full_determinism=False`, `VLLM_BATCH_INVARIANT=1`, `FASTRSI_RL_SEED_PER_SAMPLE=1`: **`5,792.1 s`**, `113` receipts, `gen 4.18 s`, `c9 9.305 / 7.693 s`, `0 MiB × 4`) plus 8-actor `SingleTurnAgentLoop.run` live swap (`32×8` calls, 7.54 s install) ([livelab](repos/verl/livelab/README.md)); batch A 640-response live swap (117.2 s reward / 104.6 s advantage vs 259.8 s restart) & coupled `S0` restore (1.10–1.16 s), `fsdp_cg` cold-vs-warm phase dissection (`297.95 s` cold -> `180.32 s` warm [`89.16 s` vLLM JIT] -> `36.43 s` live + `S0` [`8.2×`] -> `12.57 ms` `replay-adv`), parallel `_fanout` (`8.606 -> 1.276 s` install, `6.149 -> 1.096 s` uninstall) + PEP 578 negative cache across 21 actors (`install_s` **`0.827 s` mean** [`10.41×` vs serial], **`uninstall_s 0.127 s`** [`48.42×` vs serial, `36.3 ms` undo], **`status_after_cache 0.032 s`** [`26.72×`], `status_timing 0.202 s` [`17.30×`, serial `fingerprint_s` sum `2.976 -> 0.168 s`], exact 3-cycle `adv_max`/`pg_loss < 4e-9`, `reward0..7` untouched), plus Batch B CUDA-graph `Megatron-Core` (`24.29–24.40 s` steady check, `4.18 s` gen, `c2_visible` pass; `deterministic_mode=True` & `++actor...attention_backend=unfused` bit-exact 4-rank determinism) & `replay-adv` (`17.0 ms`, `max_abs_diff 0.0`) ([kb/costs.md](repos/verl/kb/costs.md)); GSM8K walkthrough and offline `rescore` (0.85 ms) / `replay_adv` (16–29 ms) ([grpo-gsm8k](repos/verl/examples/grpo-gsm8k/README.md)) | multi-node jobs; in-place rebuild & Ray actor class layout changes; full `11/11` Megatron-Core determinism (`~1e-4` `pg_loss` variance) and `full_determinism=True` CUDA-graph Inductor conflict (use `VLLM_BATCH_INVARIANT=1` + `enforce_eager=False`) |
| SGLang | adapter & zygote tests (`25/25` container pytest incl. AST `ServerArgs` probe & `--fork-subprocs`); conformance 11/11 on `ToySglangLab` | conformance 11/11 via `fastrsi conformance sglang --real` on 2× GB200 TP2 Qwen3-0.6B (install median 0.091 s / undo 0.008 s, R1 decode graph recapture 0.839 s, `SglangZygote` 11.75 s; PR #163/#164), 1× B200 TP1 Qwen3-8B (`c9` 0.0589 s / 0.0107 s, `c8` 15.82–16.42 s, `0 MiB`; cold/warm/zygote/live ladder: **`66.39 s` cold** -> **`26.46 s` warm** -> **`19.49 s` `SglangZygote` `fork_clean`** -> **`1.77 s` live `R0` `16×64`** [`37.6×`, `59.5 ms` install, `0.785 s` `R1` recapture] -> **`0.377 s` `2×16` probe** [`176.3×`]; drain `0.076 ms` `retract` / `0.192 ms` `in_place`), and 2× B200 TP2 Qwen3-8B (`c9` 0.1056 s / 0.0182 s, `c8` 16.89–17.27 s, `SglangZygote` TP=2 `fork_clean` `17.598 s` [`4.41×`] vs `77.619 s`), plus `--fork-subprocs` paired (`TP=1` `8.29 -> 0.84 s`, `total_first_step_s 18.49 -> 10.59 s` / `10.55 s` refresh; `TP=2` `8.87 -> 0.85 s`, `20.06 -> 11.89 s` / `12.56 s`) AND **full `11/11` `SglangConformanceLab --fork-subprocs`** on B200 (`TP=1` `222.42 s` case sum vs `283.30 s` spawn [`1.27×`], **`c8 92.31 s` [`9.62–9.98 s` respawns vs `15.82–16.42 s`, `1.64×`]**; `TP=2` `270.79 s` vs `309.33 s` [`1.14×`], **`c8 97.83 s` [`10.91–11.92 s` respawns vs `16.89–17.27 s`, `1.48×`]**; `0` fallbacks across all `5/5` boots, `0 MiB` exit); v0 Qwen3-8B & static survey (49.3% / 66.1% in `srt`) ([kb/costs.md](repos/sglang/kb/costs.md)) | `dp_size > 1` and `--enable-dp-attention`; PD disaggregation; in-place rebuild (constructors); non-scheduler processes |
| miles | adapter tests (`15/15` Linux GPU `miles` + `33/33` `core/livelab` fanout/negcache); conformance 11/11 on 3-process `ToyMilesLab` (~1.69 s) | conformance **11/11** via first-class `fastrsi conformance miles --real` (`MilesConformanceLab`, `113` receipts, post-fanout & PEP 578 negative-cache **`c9 0.0536 s` install [`1.16×` vs `0.0622 s`] / `0.0129 s` undo [`1.31×` vs `0.0169 s`]**, median **`edit_to_commit_s 0.0384 s`** [`1.09×` vs `0.0419 s`] / **`undo_s 0.0046 s`** [`1.14×` vs `0.0053 s`], `gap_s 0.0083 s` [`1.26×`], `c8 47.07–47.81 s`, `c3` bit-exact, `c10 zero_witness`, `0 MiB × 2`, `6/6` real-GPU CLI launch) and earlier `RealSlime11Lab` **11/11** + bit-exact `parity_rm` `S0` undo (`88.76 s -> 0.697 s` [`127.3×`]); unified `parity_rm` dissection (**`82.38 s` cold** -> **`53.48 s` warm** -> **`0.890 s` live + `S0`** [`92.6×`] -> **`0.678 s` replay** [`121.5×`]); 2× GB200 `miles` 0.2.0 swap (`0.069 s`; PR #162); 4× GB200 startup baseline (60–95 s cold) ([kb/costs.md](repos/miles/kb/costs.md)) | in-place rebuild; placement-group respawn |

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
| In place, C++/CUDA | compile edited `.cu` + parallel `reg.cpp` under `flock`, rebind `torch.ops._C` + `R1` recapture, `lib._destroy()` undo | single-unit C++/CUDA kernels (`layernorm_kernels.cu`) | `repos/vllm/livelab/adapter/native.py` | 19.22 s TP1 serial / 13.62 s TP2 parallel total incl. check (`11.88 s` commit [`11.85 s` build, `0.0 s` cached peer] + `0.68 s` R1 recapture + `1.06 s` check; down from `20.26 s` serial) vs 49.47 s / 46.97 s restart; undo 6.55 / 8.94 ms (PR `#49750`, B200; [costs](repos/vllm/kb/costs.md)) |
| In process, module rebuild | re-import the edited module and rebuild the live objects | constructors, new fields, module constants | `repos/fastvideo/livelab/services/rebuild.py` | 0.19-1.27 s, 1 GPU ([costs](repos/fastvideo/kb/costs.md)) |
| New process, warm | fork from a server with imports done; weights served over CUDA IPC by a holder | launch flags, startup code, anything the in-place paths refuse | `core/livelab/forksafe.py`, `repos/{vllm,fastvideo,sglang}/livelab/services/zygote.py`, `repos/{vllm,sglang}/livelab/services/weights.py`, `repos/fastvideo/livelab/services/holder.py`, `repos/megatron/livelab/fastrsi_megatron/zygote.py` | vLLM 7.37 s to `/health`, Qwen3-8B ([costs](repos/vllm/kb/costs.md)); FastVideo respawn 3.05–3.51 s (1G) / 6.23–7.27 s (2G), weights held ([costs](repos/fastvideo/kb/costs.md)); Megatron 15.0–16.5 s to first step, 14B ([costs](repos/megatron/kb/costs.md)); SGLang 17.04 s clean / 19.69 s refresh vs 59.85 s fresh, Qwen3-8B ([costs](repos/sglang/kb/costs.md)) |
| New layout, same weights | reshard held weights between TP and EP in memory over CUDA IPC | expert-parallelism layout switches | `repos/vllm/livelab/services/reshard_live.py` | 1.0–2.8 s write, 0.36–1.18 s drain-aware stop, 49.4–52.8 s switch with zygote+kvplan reuse (TP2 ↔ EP2, 2× B200; [costs](repos/vllm/kb/costs.md)) |
| Full restart | new process, warm disk caches | anything else | | 228 s in the vLLM ladder above |

## Repository layout

```
core/livelab/        shared runtime, standard library only
  kernel/            transaction kernel (locate, swap, witness, journal, jit, reinit, coordinator) and the toy lab
  fastrsi/           shared stdlib kit (BaseLab, diffs, ipc, proc, harness, unified CLI)
  conformance/       the 11 cases every lab must pass
  survey/            static and coverage surveys over engine codebases
  forksafe.py        pre-fork thread/descriptor checks for zygote services
  INTERFACE.md       the lab interface
core/agents/         prompts for the developer, toolsmith and orchestrator agents
repos/<engine>/      vllm, fastvideo, megatron, verl, sglang, miles
  docs/, kb/         architecture and restart boundaries; measured costs, each with its source
  livelab/           the engine's lab, built on core/livelab
  examples/<name>/   walkthroughs of replayed changes
docs/                design notes on kernels, CUDA graphs, lessons learned, and CI
.github/             CPU workflows, automated review service, leak check, and Modal GPU CI runner
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
python3 -m unittest discover -s core/livelab/survey -t core/livelab
(cd core/livelab && python3 -m unittest discover -s conformance -t .)
python3 -m unittest discover -s template/livelab/adapter -t template/livelab
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/verify_facts.py --strict
python3 .github/modal-ci/test_modal_ci.py -v
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
