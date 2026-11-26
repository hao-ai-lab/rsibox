# TensorRT-LLM (PyTorch backend) costs

Measured timings on `1x NVIDIA B200` (`TensorRT-LLM 1.0.0rc6`, `PyTorch 2.8.0a0+5228986c39.nv25.6`, `Python 3.12.3`, `Qwen3-0.6B`, `TP=1`, default PyTorch-backend `use_cuda_graph=True`, `KvCacheConfig(free_gpu_memory_fraction=0.15, enable_block_reuse=False)`, `disable_overlap_scheduler=True`, 2 prompts × 16 greedy tokens), recorded in our `2026-11-24` live GPU verification runs.

## Real-GPU `11/11` Conformance (`rsibox conformance trtllm --real -- --check-timeout 15`, `1x NVIDIA B200`, `Qwen3-0.6B`)

| Case | Result | Wall (s) | Evidence |
|---|---|---:|---|
| `c0_repeatable` | **pass** | `0.607` | 3 checks (`reset=['kv_cache_events', 'rng']`) gave identical token IDs |
| `c1_noop` | **pass** | `0.658` | `TorchSampler._process_requests:446` (`17` witness calls), output bit-exact |
| `c2_visible` | **pass** | `0.707` | `TorchSampler._process_requests:446` (`17` witness calls), output changed |
| `c3_fresh_equivalence` | **pass** | `41.227` | Both `visible` and `chain of 3` matched fresh `PackageOverlay` subprocesses bit-exact |
| `c4_uninstall` | **pass** | `0.727` | Undo in `0.00706 s`; exact code identity restored |
| `c5_failed_install` | **pass** | `1.074` | `fail_apply` & `fail_commit` rolled back cleanly; control committed 3 targets across 2 modules |
| `c6_refusal` | **pass** | `0.950` | `syntax` (`patch_invalid`), `closure` (`closure_changed`), `module_state` (`outside_changes`), `empty` (`empty_change`) |
| `c7_chain` | **pass** | `1.122` | 3 stacked edits (2 on `sampler.py`, 1 logit flip on `py_executor.py`) each changed output, undone 1-by-1 to base |
| `c8_crash_and_hang` | **pass** | `65.602` | `crash` (`12.41 s`), `hang` (`12.55 s`), `lost` (`12.04 s`), `tamper` (`11.54 s`) poisoned & respawned to base |
| `c9_soak` | **pass** | `3.701` | 20 cycles: install median **`0.0592 s`**, uninstall median **`0.0072 s`** |
| `c10_not_proven` | **pass** | `14.629` | Startup-only `PyTorchModelEngine._load_model` edit: `proven=False` (`zero_witness`), live == base, fresh != base |
| **Total** | **11/11 pass** | **131.0 s** | `rc=0`, host GPU cleanly returned to baseline (`0%` util) |

## Cold Boot vs. RSIBox Live Edit (`1x NVIDIA B200`, `Qwen3-0.6B`)

| Operation | Measured (mean ± std, N=3 unless noted) | Notes |
|---|---:|---|
| Cold boot (Cold-JIT, boot #1) | **26.6563 s** (N=1) | Includes FlashInfer `norm` JIT compile, weight load (`1.58 s`), KV pool alloc (`23.60 GiB`), warmup |
| Warm-JIT boot (boots #2–#3) | **12.4021 ± 0.1442 s** (N=2: `12.5040 s`, `12.3001 s`) | Warm FlashInfer JIT cache; fresh worker subprocess boot |
| Base `check()` (first call) | **0.2192 s** | 2 prompts × 16 greedy tokens (`fingerprint=ddbddc84ece4733a`) |
| No-op live edit (`TorchSampler._process_requests`) | `install 0.0602 s`, `uninstall 0.0057 s` | `proven=True` (`17` witness calls), bit-exact output |
| Live edit `install` (`TorchSampler._process_requests`, +1 mod V) | **0.0564 ± 0.0014 s** (`54.9–57.8 ms`) | Atomic `StepGapCoordinator` commit on `tensorrt_llm._torch.pyexecutor.sampler` |
| Live edit `check` | **0.0509 ± 0.0130 s** (`43.0–65.9 ms`) | `proven=True` (`17` witness calls each repeat), `output_changed=True` |
| **Live edit + check total** | **0.1072 ± 0.0118 s** (`99.4–120.8 ms`) | **`248.55×`** vs. Cold-JIT boot (`26.66 s`), **`115.64×`** vs. Warm-JIT boot (`12.40 s`) |
| Live `uninstall` (rollback to base) | **0.0062 ± 0.0005 s** (`5.8–6.7 ms`) | Bit-exact token rollback and exact fingerprint restoration (`3/3`) |
| Volatile `reset("state")` | **0.0011 ± 0.0001 s** (`1.0–1.2 ms`) | `kv_cache_manager.flush_iteration_events()` + RNG reseed |
| Fresh subprocess parity (`PackageOverlay` + `run_fresh_subprocess`) | **15.3054 s** (N=1) | `ok=True`, `matches_live_edited=True` (bit-exact vs. live hot-swap) |

## `TrtllmQuiesceHooks` + `DynamoK8sSnapshotDriver` (`1x NVIDIA B200`, `Qwen3-0.6B`)

| Stage / Metric | Measured | Notes |
|---|---:|---|
| `warmup_for_snapshot()` | **0.0167 s** | 1-prompt 2-token deterministic warmup + participant probe |
| `quiesce_for_snapshot()` (`kv_cache_manager.impl.release_pools()`) | **0.1875 s** | `ok=True`, `released=("kv_cache",)`, `token="trtllm:kv_cache"` |
| `resume_after_restore()` (`allocate_pools(False)` + pointer refresh) | **0.0074 s** (`0.0054 s` worker alloc) | `ok=True`, `participants_ok=True`, `trtllm-restore-ready` sentinel written |
| VRAM before quiesce | **46,283.94 MiB** | Model weights + `23.60 GiB` paged KV cache + runtime workspaces |
| VRAM while quiesced | **22,115.94 MiB** | **`24,168.0 MiB` (`23.60 GiB`) cleanly released** |
| VRAM after resume | **46,283.94 MiB** | Full paged KV cache pool restored |
| Post-restore base check | **Bit-exact (`True`)** | Identical token IDs to pre-snapshot base |
| Post-restore live edit + uninstall | `install 0.0569 s`, `proven=True` (`17` calls) | Matches pre-restore edited output bit-exact; bit-exact rollback |

## CPU conformance & unit suite (`ToyTrtllmLab`, 2 spawned ranks)

| Suite | Result | Timing |
|---|---|---|
| `rsibox conformance trtllm` (`c0–c10`) | **11/11 pass** | ~4.9 s total; `c9_soak` 20-cycle median `0.0100 s` install / `0.0048 s` undo |
| `rsibox doctor` | **9/9 labs ok (7/7 engines)** | `0` GPU import violations (`torch`/`tensorrt_llm` stay lazy) |
| `adapter/test_adapter.py` | **14/14 pass** | ~5.5 s (lazy imports, live swap, linecache undo, fresh parity, faults, file channel, snapshot hooks + pointer-shift & block-reuse guards) |

## Multi-GPU `TP=2` MPI + CUDA Graph `R1` Recapture (`2x NVIDIA B200`, `Qwen3-0.6B`)

Verified live on `2x NVIDIA B200` (`TensorRT-LLM 1.0.0rc6`, `Qwen3-0.6B`, `TP=2` `MPIPoolExecutor` hooked via pure-stdlib `adapter/mpi_hook.py`, `use_cuda_graph=True`, `2026-11-24`):

### Real-GPU `TP=2` `11/11` Conformance (`rsibox conformance trtllm --real -- --tp 2 --check-timeout 30 --rpc-timeout 30`)

| Case | Result | Wall (s) | Evidence |
|---|---|---:|---|
| `c0_repeatable` | **pass** | `0.940` | 3 checks (`reset=[kv_cache_events, rng]`) gave bit-exact token IDs across both ranks |
| `c1_noop` | **pass** | `1.048` | `TorchSampler._process_requests:446` (`[17, 17]` witness calls), output bit-exact |
| `c2_visible` | **pass** | `1.000` | `TorchSampler._process_requests:446` (`[17, 17]` witness calls), output changed |
| `c3_fresh_equivalence` | **pass** | `76.114` | Both `visible` and `chain of 3` matched fresh `TP=2` `PackageOverlay` MPI subprocesses bit-exact |
| `c4_uninstall` | **pass** | `1.111` | Undo in `0.0108 s`; exact code identity restored across both ranks |
| `c5_failed_install` | **pass** | `1.504` | `fail_apply` & `fail_commit` rolled back cleanly across both ranks; control committed 3 targets |
| `c6_refusal` | **pass** | `1.338` | `syntax`, `closure`, `module_state`, `empty` all refused |
| `c7_chain` | **pass** | `1.493` | 3 stacked edits across `sampler.py` & `py_executor.py`, undone 1-by-1 to base |
| `c8_crash_and_hang` | **pass** | `121.444` | `crash` (`22.35 s`), `hang` (`22.29 s`), `lost` (`22.53 s`), `tamper` (`22.66 s`) poisoned & respawned to base |
| `c9_soak` | **pass** | `4.672` | 20 cycles across 2 ranks: install median **`0.0676 s`**, uninstall median **`0.0134 s`** |
| `c10_not_proven` | **pass** | `26.368` | Startup-only `PyTorchModelEngine._load_model`: `proven=False` (`[zero_witness, zero_witness]`), live == base, fresh != base |
| **Total** | **11/11 pass** | **237.0 s** | `rc=0`, both GPUs cleanly returned to baseline (`0%` util) |

### Model-Layer Forward Hot-Swap Inside CUDA Graphs (`R1`) & `TP=2` KV Quiesce

| Operation (`2x NVIDIA B200`, `Qwen3-0.6B`, `TP=2`, `use_cuda_graph=True`) | Measured | Notes |
|---|---:|---|
| `TP=2` MPI cold boot | **36.93 s** | `workers=[rank0, rank1]`, deterministic `base1 == base2` |
| `Qwen3DecoderLayer.forward` live `install` (auto-`R1` recapture) | **0.2751 s** | Auto-triggered `_release_cuda_graphs()` on both ranks (`rank0`: `0.2441 s` / 4 graphs, `rank1`: `0.2439 s` / 4 graphs) |
| `Qwen3DecoderLayer.forward` live `check` (lazy CUDA graph recapture) | **0.3743 s** | `proven=True`, `output_changed=True`, **`224 / 224` witness calls on BOTH `rank0` and `rank1`** |
| `Qwen3DecoderLayer.forward` live `uninstall` (auto-`R1` invalidate) | **0.1491 s** | Both ranks invalidated & restored cleanly |
| `TrtllmQuiesceHooks` `quiesce_for_snapshot()` (`TP=2`) | **-46.67 GiB** (`-23.33 GiB`/GPU) | `rank0`: `47,256 -> 23,362 MiB`, `rank1`: `45,920 -> 22,026 MiB` |
| `TrtllmQuiesceHooks` `resume_after_restore()` (`TP=2`) | **0.0152 s** (`15.2 ms`) | `rank0`: `6.3 ms`, `rank1`: `7.1 ms`, `participants_ok=True`; post-restore `Qwen3DecoderLayer.forward` edit & bit-exact rollback passed |

## `TrtllmShadowPromoter` Scratch-KV Failover & Orphan PG Fix (`1×` & `2× MPI B200`, `#281`)

Measured on `1×` and `2× NVIDIA B200` (`TensorRT-LLM 1.0.0rc6`, `Qwen3-0.6B`, `short_single_1x16` & `batch_4x32`):

| Mode & Commit | Boot / Park (KV Freed) | `promote()` Breakdown | Orphan & VRAM Hygiene | Post-Promote Swap & Conformance |
|---|---|---|---|---|
| **`TP=1` (`02e09ea`)** | `28.05 s` primary / `22.33 s` shadow; `0.113 s` park (`-22,566 MiB`) | **`0.329 s`** (`0.314 s` stop + `0.012 s` resume; `0.315 s` dead-head skip) | `pointer_shift=False`, `0.0 MiB` on close | Bit-exact parity; `0.061 s` install (`16`/`49` witness, bit-exact fresh/undo); **`11/11`** real (`c9` `59.1 ms` / `7.2 ms`) |
| **`TP=2` MPI pre-fix (`02e09ea`)** | `40.05 s` primary / `35.94 s` shadow; `0.292 s` park (`-22,004 MiB`/rank) | **`0.287 s`** (`0.265 s` stop + `0.016 s` resume) | Temporary MPI child rank orphan (`<5 s`, `[57,293, 55,194] MiB`), `0.0 MiB` on close | Bit-exact parity; `0.116 s` install (`16`/`49` witness × 2 ranks); **`11/11`** real (`c9` `65.4 ms` / `10.9 ms`) |
| **`TP=2` MPI post-fix (PR `#315` / `d0cf50b7`, receipt run `e93bbfa`, PG sweep)** | `54.38 s` primary / `37.61 s` shadow; `0.299 s` park (`-22,004 MiB`/rank) | **`0.145 s`** (`0.14511 s`: `0.072 s` stop + `0.069 s` resume) | **`0` orphan PIDs / `0` orphan apps at `0 ms`** (`[35,287, 33,188] -> [26,604, 25,194] MiB`; `0.0 MiB` on close) | Bit-exact parity; `0.069 s` install (`16`/`49` witness × 2 ranks, `25.8–30.2 s` fresh parity, `0.015 s` undo); `28/28` unit tests + `doctor_ok` |
