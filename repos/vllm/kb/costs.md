# vLLM costs

Prior measurements, gathered before any new run. Each table says where it came from.

## Restart vs live, one-line scheduler edit (1 GB200)

Source: our one-GB200 restart-vs-live measurement of a one-line scheduler edit. Warm on-disk
caches.

| Step removed | Check time | vs restart |
|---|---:|---:|
| Restart with caches (baseline) | 228 s | 1.0× |
| + reuse autotune config | 157 s | 1.5× |
| + keep model weights | 100 s | 2.3× |
| + keep CUDA graphs | 58 s | 3.9× |
| + keep the live process (hot swap) | 23.6 s | 9.6× |
| then skip the logprob check | 14.1 s | |
| then test with a small batch | 2.4 s | 95× end to end |

Reading: setup is 204 s of the 228 s. The ladder is the disaggregation opportunity: each row is
a piece of state that can be kept separately.

## DeepSeek-V4.1-Flash startup (TP4, 4× B200)

Source: our startup measurement, 2026-10-08, `vllm/vllm-openai:v0.31.0`,
`--max-model-len 16384`. Seconds.

| Phase | Cold | Warm |
|---|---:|---:|
| Weight load (TP0) | 178.6 | 124.9 |
| Profile + memory-profiling CUDA graph capture | 111.0 | 79.0 |
| JIT kernel warmup (190 keys) | 52.5 | 52.5 |
| FlashInfer autotune | 625.4 | 627.2 (retuned despite saved configs) |
| Final CUDA graph capture | 40 | 40 |
| **Time to /health** | **1110.2** | **1012.2** |

Reading: autotune is 62% of a warm start and its cache did not help. Engram tables are 2 × 23.6 GiB
pinned host memory per rank. No torch.compile for this model in this image.

## Restart per check vs live edit (Qwen3-8B, 1× GB200, 2026-11-10)

Source: our 2026-11-10 run of [`../examples/live-edit-qwen3`](../examples/live-edit-qwen3/README.md):
vLLM 0.31.0, TP 1, CUDA graphs on (defaults), `--no-enable-prefix-caching`, warm on-disk caches,
weights on a memory disk. Check: 8 requests, greedy, 32 tokens each (1.03 s). Each number is one
measurement from a run that preceded vLLM conformance verification.

| Edit | Restart, edit to result | Live, edit to result | Install (edit to commit) | Engine loop stalled | Undo | Witness calls |
|---|---:|---:|---:|---:|---:|---:|
| `Scheduler.schedule`, engine core | 62.98 s | 1.175 s | 0.141 s | 1.2 ms | 0.7 ms | 816 |
| `Sampler._sample_random`, worker | 63.37 s | 1.072 s | 0.030 s | 0.5 ms | 0.3 ms | 768 |
| `Sampler.greedy_sample` in `v1/sample/sampler.py`, worker | 63.08 s | 1.067 s | 0.030 s | 0.5 ms | 0.3 ms | 0 (`zero_witness`) |

- Launch to `/health`: 61.94–64.55 s over 6 starts (5 plain, 1 with the adapter: 63.03 s).
- Check time, median of 3: 1.035 s plain, 1.032 s with the adapter loaded (no cost detected).
- Live output equals the fresh process's output for all three edits.
- This 63 s restart is not the 228 s baseline of the ladder above (a different measurement,
  not reproduced here).

## Live edit and undo on one B200

Source: our 2026-11-19 Qwen3-8B walkthrough on vLLM 0.33.0, V2 runner, eager execution,
prefix caching disabled, 8 greedy 32-token prompts. The independent fresh and live
completions agreed for all three edits. The same B200 lab passed conformance c0–c10.

| Edit | Native edit to commit | First edited check | Native undo | Undo client round trip | Witness calls |
|---|---:|---:|---:|---:|---:|
| Scheduler | 0.163079 s | 5.320 s | 1.682 ms | 15.4 ms | 816 |
| Active sampler | 0.039252 s | 5.563 s | 1.270 ms | 10.4 ms | 768 |
| Legacy sampler | 0.030376 s | 6.102 s | 1.272 ms | 15.5 ms | 0 |

The legacy sampler produced `zero_witness`. All three undos restored the original
code fingerprint and completion text. The full walkthrough took 390.128 s including
six server launches, repeated checks and teardown.

## Restart after one edit (Qwen3-8B, warm caches otherwise)

Source: our per-edit restart measurement (pre-v0 internal tree), dev workload.

| Edit | Ready | vs warm 17.8 s |
|---|---:|---:|
| Triton kernel in the V2 runner | 17.9 s | +0.1 |
| Comment in `qwen3.py` | 31.0 s | +13.2 |
| Attention forward in `qwen3.py` | 41.7 s | +23.9 |
| Backend FLASHINFER → FLASH_ATTN | 60.7 s | +42.9 |
| Full cache loss (cold) | 150.3 s | |

Native CUDA incremental rebuild: 8.0–24.2 s per file (`ninja`, sm100).

## FastRSI v0 overhead (Qwen3-8B, 1 GPU, sampler edits)

Source: `legacy/v0` branch, `receipts/overhead-vllm.json`, `docs/overhead.md`.

| Cost | Value |
|---|---:|
| Workload, stock vs agent loaded | 0.2924 s vs 0.2923 s (not detected) |
| Install, median / max | 22 / 44 ms |
| Undo, median / max | 4 / 9 ms |
| Host growth | +0.37 MiB per install/undo cycle |
| Restart to base | 40 s |
| Witness, per call | 34 ns |
| Repeatability, 32 prompts batched | 1 distinct output in 60 runs |

## Zygote: edits after it started, refreshed in the fork (Qwen3-8B, 1× B200, 2026-11-06)

The zygote (`livelab/services/zygote.py`) imports vLLM and its libraries once and forks each
`vllm serve` from that state. An edit to a file it already imported used to make it refuse (or,
accepted, the server ran the old code silently). Now the fork drops the edited modules and their
importers before the server starts (`livelab/services/zygote_refresh.py`). Source: our 2026-11-06
run, Qwen3-8B TP1 eager, warm caches, forked engine core; launch to `/health`, one greedy
completion each (" Paris." in every case); markers printed by the edited code show what ran.

| Case | Modules refreshed | To `/health` (s) | Edited code ran |
|---|---:|---:|---|
| fresh `vllm serve` (no zygote) | – | 20.9 | – |
| fork, no edit | 0 | 10.9 | – |
| fork, attention backend edited, refresh off | 0 | 10.4 | **no** (old code, silently) |
| fork, attention backend edited (leaf) | 2 | 10.2 | yes (class body + `__init__`) |
| fork, `envs.py` + engine core edited (core) | 1,371 (all of vLLM) | 13.7, 12.4 | yes (API server + engine core) |

- The zygote imported 8,631 modules in 12.7 s and builds the import graph of its 1,371 vLLM
  modules at start (2.3 s, off the start path). Only imports that run at import time count;
  imports inside functions look the module up at call time and see the new one.
- vLLM's module graph has one large cycle through `vllm.config` and `vllm.platforms`: an edit to
  almost any module one of them imports reaches ~760 of ~1,000 modules. Above 100 modules the
  refresh re-imports all of vLLM instead (+1.5–2.8 s over a plain fork), since a partial
  re-import would register into package-wide tables (IR ops, CustomOp) twice. Third-party
  libraries, most of the import time, stay imported. Leaf modules (attention backends, model
  files, the GPU model runner: 1–35 modules) refresh in milliseconds.
- Re-imported torch ops: an identical schema is accepted and the new body replaces the old one;
  a changed schema is refused with the op's name. Opaque types re-register.
- Still a finding (restart the zygote): an edited extension `.so`, or an edited module outside
  `FASTRSI_ZYGOTE_REFRESH` (default `vllm`).

## Launcher CLI with state holders and zygote (`engine.py start`, Qwen3-8B TP1 eager, 1× GB200, 2026-11-07)

Source: our 2026-11-07 GPU receipt on 1× GB200, Qwen3-8B TP1, `--enforce-eager`, running through `livelab/services/engine.py` with `weights.py` (`vllm preload` daemon) and `kvplan.py` (`KVPlanWorker`). Each start prints one JSON line on stdout; greedy 32-token completion at temperature 0 (`" Paris. The capital of France is Paris..."`) is bit-identical across all started servers.

| Case | Launcher flags | `healthy_after_s` (s) | `load_model` (s) | `determine_available_memory` / `kvplan` | Notes |
|---|---|---:|---:|---|---|
| `a` | direct `start` (disk weights, `--load-format auto`) | 51.85 | 6.82 | `1.63 s` (warm profile run) | Baseline fresh-process start with warm disk caches |
| `b` | direct `start --weights` (`ipc_cache`) | 46.80 | 2.24 | profiled | `weights_probe`: `0.041 s` (`matched_gpus: true`, `fallback: false`) |
| `c0` | direct `start --kvplan` (first run, record) | 50.71 | 6.82 | `measured: 1` (`2.10 s`) | Records KV plan JSON for `load_format="auto"` |
| `c` | direct `start --kvplan` (second run, reuse) | 43.97 | 6.82 | `reused: 1` (`0.005 s`) | Skips step-4c profile run; same KV cache size |
| `d0` | `start --zygote S --fork --kvplan --weights` (first run, record) | 10.42 | 2.24 | `measured: 1` (`1.33 s`) | Separate plan key for `load_format="ipc_cache"`; `fork_check: {"spawn_forced": false}` |
| `d` | `start --zygote S --fork --kvplan --weights` (second run, reuse) | **7.37** | 2.24 | `reused: 1` (`0.006 s`) | `zygote_check`: `0.205 s`, `fork_check: {"spawn_forced": false}`, zero findings (`7.0×` vs. case `a`) |
| `e` | `start --weights` after stopping weight daemon | not started (`0.305 s`, exit 2) | – | – | Refuses immediately with `weights_daemon` finding naming the missing GPU socket instead of hanging `state_timeout_s` (`300 s`) |

## GB200 containers without IMEX channels (`#10`)

Source: our TP4 runs on GB200 with vLLM `4e0ad57` (`vllm preload` and `vllm serve`).

- When `/dev/nvidia-caps-imex-channels` is not exposed inside a GB200 container, CUDA refuses NVLink fabric memory handles and multicast objects with `CUDA_ERROR_NOT_PERMITTED` (error 800):
  ```
  Failed to initialize FlashInfer All Reduce workspace: CUDA error code=800(b'CUDA_ERROR_NOT_PERMITTED')
  FlashInfer mnnvl allreduce workspace unavailable (likely no NVSwitch multicast support); falling back ...
  ```
- FlashInfer's MNNVL all-reduce workspace fails to initialize and vLLM falls back to a slower all-reduce path. Any multi-GPU throughput or latency measurement recorded in such a container must state that IMEX was unavailable.
- Cross-process sharing built on NVLink fabric handles cannot run without IMEX channels, whereas intra-node POSIX file-descriptor CUDA IPC handles (`livelab/services/weights.py`) continue to work normally.

## Native compile: one translation unit (`layernorm_kernels.cu`, vLLM #49750's only `csrc/` file)

Source: our 2026-11-14 probe on one GB200 node (CPU only; nvcc 13.0, sm_100a only), the build
tree's own nvcc command for `csrc/libtorch_stable/layernorm_kernels.cu`, with the sources,
headers and venv on the lab's network file system, which was slow that day (other processes on
the node sat in NFS waits). n=3 each, seconds.

| Command | Runs | Median |
|---|---|---:|
| build tree's command | 115.4, 100.1, 114.5 | 114.5 |
| + `-split-compile=0` | 129.4, 81.6, 97.0 | 97.0 |
| + `-split-compile=8` | 68.3, 57.8, 74.4 | 68.3 |

Phases (`nvcc -time`, one run): host preprocessing 47.7, `cudafe++` 2.4, device preprocessing
44.8, `cicc` 6.9, `ptxas` 3.1, `fatbinary` 0.4, host compile 11.5. With `-split-compile=0`:
`cicc` 4.3, `ptxas` 0.8. `nvcc --version` takes 13 ms; `-E` alone 26.2 s.

Reading: the compute of the unit (cicc + ptxas + host compile, about 21 s) is close to the 21.4 s
incremental rebuild of the same file on local disk (our pre-repo prototype's notes) and consistent with the
28 s install of #49750 (that prototype's side library: whole-unit compile, link, rebind). The
rest here is the two preprocessing passes reading thousands of torch and CUDA headers over the
network file system. So: keep the build tree and headers on node-local disk; pass
`-split-compile` (it parallelizes cicc and ptxas across the unit's kernels); the unit, not the
edited kernel, stays the floor, since #49750 changes a kernel's signature and its host launcher.
The repo's `NATIVE` route is now implemented in `repos/vllm/livelab/adapter/native.py` and
verified on B200 below.

## Live C++/CUDA native hot-reload (PR #49750, 1× B200, 2026-11-19)

Source: our 2026-11-19 `verify_native_pr49750.py` run on 1 idle B200 GPU (vLLM 0.33.0, torch
2.14.0+cu130, nvcc 13.3, Qwen3-8B TP1, V1 `GPUModelRunner` with `FULL_DECODE_ONLY` CUDA graphs
`[1, 2, 4]`, `--kernel-config` routing RMSNorm through `vllm_c` / `torch.ops._C`, 8 greedy
prompts × 32 tokens, prefix caching disabled). Cold server boot took 31.80 s (live), 30.30 s
(fresh visible perturbation), and 30.80 s (fresh PR #49750).

| Case | nvcc build (s) | CDLL load (ms) | Gap + commit (ms) | R1 recapture (s) | Check (s) | Live vs fresh | Undo (ms) |
|---|---:|---:|---:|---:|---:|---|---:|
| Graph refusal (no `rebuild: ["R1"]`) | skipped (0.022 s prepare) | – | – | – | – | aborted (`captured_target`) | – |
| Visible `.cu` perturbation (`epsilon = 0.25f;`) | 17.10 | 0.88 | 2.30 | 0.473 (432 cap / 576 live calls) | 1.28 vs 1.57 fresh | exact (`max \|Δlogprob\| = 0.0`); total 18.87 s vs 48.97 s restart | 2.85 (1.96 worker) |
| Upstream PR #49750 (`pre-PR` -> PR #49750 `layernorm_kernels.cu`) | 17.09 | 1.17 | 2.17 | 0.826 (432 cap / 576 live calls) | 1.27 vs 1.58 fresh | exact (`max \|Δlogprob\| = 0.0`); total 19.22 s vs 49.47 s restart | 6.55 cascade (5.60 worker) |

Standalone `nvcc 13.3` compile/load/dispatch/undo probe on GPU 0 measured `16.83 s` build,
`0.68 ms` load, `0.028 ms` `lib._destroy()` unbind, and direct tensor verification confirmed the
pre-PR kernel rejects non-contiguous `residual` (`stride(-1) == 1, is_contiguous() == False`)
with `RuntimeError`, while the live-compiled PR #49750 kernel matches PyTorch FP32 reference (`max abs err = 0.015625`).

## Live TP2 <-> EP2 by holder reshard (Qwen3.5-35B-A3B BF16, 2× B200, 2026-11-19)

Source: our 2026-11-19 runs on 2 idle B200 GPUs (vLLM 0.33.0, torch 2.14), `--max-model-len 8192
--language-model-only --gpu-memory-utilization 0.30`, CUDA graphs on. Checkpoint 66.97 GiB,
already in the host page cache (1.5 TiB RAM), so a "disk" load reads RAM at memory speed here.
All rows are warm-cache starts (FlashInfer JIT, torch.compile) unless marked.

| Step | Seconds | Note |
|---|---:|---|
| Reshard write, TP2 -> EP2 (80 expert tensors 64.4 GB, 32.2 GB of it GPU to GPU; 493 dense tensors 4.9 GB copied rank-locally) | 1.0–2.8 | 1.0–1.7 in the first 3 runs, 2.4–2.8 in the 6 fills of the later end-to-end runs (both directions; cause not investigated); plus map 0.7–1.3 s, plan < 0.01 s |
| Reshard write, EP2 -> TP2 | 1.1 | 1 run |
| `reshard_live.py ensure` total (vLLM import + config 7.4, dummy EP2 group start 23.0, map, write) | 34.2 | runs while the old engine serves; off the downtime |
| `ensure` on the way back (TP2 group still resident: reused, nothing written) | 9.3–10.5 | |
| EP2 holder group from disk (page cache warm) | 31.0 | the alternative to the dummy start (23.0) |
| Stop engine (`stop_engine`), drain-aware (SIGTERM to the API server only) | 0.36–1.18 under load | 12 stops, never escalated to SIGKILL; port closed in 0.24–0.30. Before (SIGTERM to every process): 10.1 under load, see the stop table below |
| EP2 engine on the resharded group (ipc_cache), launch to /health | 73.7 | weight map 1.8; vLLM's init-engine span 43.1 (profile, KV alloc, warmup, incl. graph capture 19); the other ~29 s are process spawn, imports, distributed init and the API server |
| EP2 engine cold from disk, launch to /health | 82.5 | weight load 10.0 (32.35 GiB/rank) |
| Client-observed unavailability, `engine.py reconfigure` TP2 -> EP2 | 77.5 | 1-token requests every 50 ms throughout |
| Same, EP2 -> TP2 (reused group) | 82.7 | 10.1 s of it was the old stop under load; with the drain-aware stop see the table below |
| First engine in a fresh cache root (cold JIT), for scale | 697 | 654 s of it FlashInfer JIT builds |

Correctness (same runs): the resharded EP2 group equals a disk-loaded EP2 group byte for byte
(1308 tensors, buffers included), and the EP2 -> TP2 group equals the disk-loaded TP2 group
(1228 tensors). Greedy completions (4 prompts × 32 tokens, logprobs) of the live EP2 engine
equal a cold EP2 server's exactly, and TP2 after the round trip equals the original exactly.
References must run without concurrent traffic: run next to a 1-token client, the same engine
gave different greedy tokens (batch-dependent BF16 MoE numerics, not weights).

Reading: the reshard is not the bottleneck (about 2 s of GPU time, off the critical path). The
downtime is the engine start. With the checkpoint in page cache, resident weights alone save
about 9 s per switch over a cold restart (weight load 10.0 vs map 1.8), and read 0 bytes of
checkpoint where a disk load reads 67 GiB (that gap grows without the page cache, not measured
here). The zygote, the KV plan and the drain-aware stop take the next 20–25 s (below).

### Stop under client load (same model, TP2 ipc_cache, 3 clients with requests in flight)

Source: our 2026-11-19 stop probe, one engine start per row, on 2 idle B200.

| How the engine is stopped | Whole tree gone (s) | Port closed (s) | In-flight requests |
|---|---:|---:|---|
| SIGTERM to every process at once (the old `stop_engine`) | 10.10 (SIGKILL at the grace) | 10.1 | hung 10 s, then connection reset |
| SIGTERM to the API server only (`stop_engine` now) | 1.03 | 0.18 | answered at once (HTTP 500) |
| API server only, `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=5` | 5.59 | 5.29 | HTTP 500 |
| API server only, `--shutdown-timeout 5` (vLLM drain mode) | 5.56 | 5.31 | HTTP 500 |

Why: vLLM's API server shuts down in order (abort requests, stop the engine core and workers,
close the HTTP server). A SIGTERM that also reaches the engine core and workers kills them
before the aborts reach the API server, whose request handlers then wait forever; uvicorn waits
for them ("Waiting for connections to close") until the SIGKILL. GPU memory was back to the
holder's 34.2 GiB per GPU within 0.5 s of the leader-only SIGTERM. vLLM's graceful worker exit
costs the 5 s of the last two rows.

### Reshard switches with `--zygote --kvplan` vs plain (2× B200, 2026-11-19)

Source: our 2026-11-19 end-to-end runs through `engine.py reconfigure`, one per mode, same code
and flags as above, drain-aware stop in both. Per mode: TP2 start, then A TP2 -> EP2 (EP2 group
filled from TP2), B EP2 -> TP2 (original TP2 group reused), C TP2 -> EP2 (EP2 group filled
again), D EP2 -> TP2 (a dummy TP2 group filled from EP2). Clients: 1-token requests every 50 ms
and a 128-token streamer, throughout. Unavailability: first failed 1-token request to the end of
the first one that succeeds after it. Log offsets have 1 s resolution.

| Per switch (s) | plain | `--zygote --kvplan`, plan measured (A) | `--zygote --kvplan`, plan reused (B, C, D) |
|---|---:|---:|---:|
| `ensure` while the old engine serves: filled (vLLM import + config 7.1–7.7, dummy group start 23.0–24.5, map 0.9–1.3, write 2.4–2.8) / reused | 36.5–37.4 / 10.1 | 36.0 | 35.9–37.3 / 9.7 |
| Stop (drain-aware) | 0.73–1.12 | 0.41 | 0.36–0.41 |
| Launch to `vllm serve` args parsed (imports) | 5.2–5.9 | 0.2 | 0.2–0.5 |
| Launch to engine core init | 14.9–15.7 | 9.2 | 8.5–9.3 |
| Launch to weights mapped (map itself 1.6–2.0) | 25.9–27.7 | 21.2 | 19.5–21.2 |
| vLLM init-engine span: profile run + CUDA graph memory profile, KV alloc, warmup, graph capture | 40.7–44.4 | 41.2 | 21.2–22.6 |
| — of it CUDA graph capture | 16–17 | 17 | 17–18 |
| Launch to /health | 71.7–76.4 | 65.6 | 44.9–48.1 |
| First request after /health (CuTe DSL GDN prefill kernels JIT at the first prefill) | 0.04–0.06 | 0.03 | 4.3–4.7 |
| **Client-observed unavailability** | **72.7–77.1** | **65.9** | **49.4–52.8** |

Zygote: ready in 8.7 s (8.2 s of imports, 6,610 modules), off the switch path; its check before
each fork 0.18–0.24 s; `fork_check` found no spawn fallback. The KV plan's measured step takes
21.3–22.5 s (profile run 8 s, CUDA graph memory profile 11–12 s); a reused plan returns in
0.005 s, but kernels the profile run would have compiled (the GDN prefill kernels, CuTe DSL)
compile at the first live prefill: the 4.3–4.7 s first request. So the KV plan moves that
part rather than removing it.

Correctness, both modes: the filled EP2 group equals a disk-loaded EP2 group byte for byte
(1308/1308 tensors, buffers included), and the filled TP2 group equals a disk-loaded TP2 group
(1228/1228). Greedy completions (4 prompts × 32 tokens, logprobs): EP2 after A and after C
equal a cold EP2 server's exactly (max |Δlogprob| 0.0), TP2 after B and after D equal the
initial TP2's exactly, and the two modes agree exactly. The references need the clients
quiet: in a first run they started while a streamer request, slowed by the first-request
JIT, was still running, and greedy tokens differed (batch-dependent BF16 MoE numerics). After
the explicit stops, GPU memory went back to 0 MiB with no leftover process.

Where the remaining ~45 s go (plan reused): ~9 s from the forked API server to the engine core
(config, tokenizer, engine core start), ~11 s from there to mapped weights (worker fork,
distributed and all-reduce workspace init, model build), 17–18 s of CUDA graph capture, ~3–4 s
of API server start after the engine, and the 4.5 s first-request JIT. The reshard itself and
the stop are no longer on that list.

## Live scheduling policy

Source: our 2026-11-19 Qwen3-8B test on one B200, vLLM 0.33.0 in eager mode,
one running sequence, greedy decoding, prefix caching disabled. Each row is one
switch on the same API server and engine core. The core applied the change between
steps. The launcher timed the control request and acknowledgement. CLI wall time
includes starting the launcher and its preflight checks.

| Switch | Core apply | Launcher apply and acknowledgement | CLI wall |
|---|---:|---:|---:|
| FCFS to priority | 0.218 ms | 2.266 ms | 6.311 s |
| Priority to FCFS | 0.374 ms | 6.487 ms | 6.023 s |

Under FCFS a completion with priority −1 returned HTTP 400 both before and after the
round trip. Under priority scheduling the core reported one running and three waiting.
The waiting requests arrived with priorities 10, 5, −1. All returned HTTP 200 and
completed in the order −1, 5, 10. Under restored FCFS the three equal-priority
requests completed in arrival order. In each mode all four token sequences matched
those from a separately started server with the corresponding scheduling policy.

## Not measured yet

- DeepSeek-V4.1-Flash on 4× GB200 EP4DP4 (the target workload): restart breakdown and each
  probe's cost.
- Cost of `install` per group for the three example PRs.
