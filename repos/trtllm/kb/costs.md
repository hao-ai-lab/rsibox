# TensorRT-LLM (PyTorch backend) costs

Measured timings on `1x NVIDIA B200` (`TensorRT-LLM 1.0.0rc6`, `PyTorch 2.8.0a0+5228986c39.nv25.6`, `Python 3.12.3`, `Qwen3-0.6B`, `TP=1`, `KvCacheConfig(free_gpu_memory_fraction=0.15, enable_block_reuse=False)`, `disable_overlap_scheduler=True`, 2 prompts × 16 greedy tokens), recorded in our `2026-11-24` live GPU verification run.

## Cold Boot vs. RSIBox Live Edit (`1x NVIDIA B200`, `Qwen3-0.6B`)

| Operation | Measured (mean ± std, N=3 unless noted) | Notes |
|---|---:|---|
| Cold boot (Cold-JIT, boot #1) | **26.6563 s** (N=1) | Includes FlashInfer `norm` JIT compile, weight load (`1.58 s`), KV pool alloc (`23.60 GiB`), warmup |
| Warm-JIT boot (boots #2–#3) | **12.4021 ± 0.1442 s** (`12.5040 s`, `12.3001 s`) | Warm FlashInfer JIT cache; fresh worker subprocess boot |
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
| `rsibox conformance trtllm` (`c0–c10`) | **11/11 pass** | ~3.0 s total; `c9_soak` 20-cycle median `0.0097 s` install / `0.0041 s` undo; `c8` respawns `0.12–0.14 s` |
| `rsibox doctor` | **9/9 labs ok (7/7 engines)** | `0` GPU import violations (`torch`/`tensorrt_llm` stay lazy) |
| `adapter/test_adapter.py` | **9/9 pass** | ~3.2 s (lazy imports, live swap, fresh parity, faults, file channel, snapshot hooks + fail-closed) |
