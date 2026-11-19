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
This repo has no native route yet (`mechanisms.md`).

## Not measured yet

- DeepSeek-V4.1-Flash on 4× GB200 EP4DP4 (the target workload): restart breakdown and each
  probe's cost.
- Cost of `install` per group for the three example PRs.
