# vLLM Argument Reconfiguration Matrix

How every major `vllm serve` CLI argument is classified for runtime reconfiguration in `repos/vllm/livelab/services/vllm_args.py` and `engine.py reconfigure`, which internal vLLM V1 objects hold its state, its reconfiguration latency, and the engineering effort required to move slower flags into sub-second in-process hot-updates.

---

## 1. Reconfiguration Tiers Overview

| Tier ID | Latency | Process Restart? | Disk Weight Read? | Mechanism |
|---|---:|---|---|---|
| **`in_process_instant`** | **`< 100 ms`** | No (`same PID`) | No | Mutate Python config attributes (`SchedulerConfig`, `ModelConfig`, `KVCacheManager`) in place at the step gap and drain/reset prefix caches if needed. |
| **`in_process_graph`** | **`0.5 – 2.0 s`** | No (`same PID`) | No | Toggle dispatch between eager forward and `CUDAGraphWrapper`, or invoke `RecaptureCoordinator` to recapture CUDA graphs in the live worker. |
| **`warm_weights_ipc`** | **`7 – 20 s`** | Yes (worker restart, zygote fork) | No (`0 B` from disk) | Restart `vllm serve` from `zygote.py` while `weights.py` (`WeightHolder`) keeps post-processed GPU tensors resident over zero-copy CUDA IPC (`--load-format ipc_cache`) and `kvplan.py` skips the memory profile run. |
| **`live_reshard_or_cold`** | **`25 – 90 s`** (or `2 – 5 s` with NCCL reshard) | Yes (or live NCCL P2P reshard) | Reloads or reshards `WeightCacheKey` | Changes tensor sharding (`TP`/`PP`/`DP`/`EP`) or post-processed weight tensor layout (`dtype`, `quantization`, `kv_cache_dtype`, `moe_backend`). |

---

## 2. Full Argument-by-Argument Matrix

### Tier 1A — `in_process_instant` (`< 100 ms`, Zero Process Restart, Zero Graph Recapture)

| CLI Flag (`vllm serve`) | Parser Dest | Internal vLLM V1 Objects Mutated | Live Update Procedure | Target Latency | Effort for `< 0.1 s` In-Process |
|---|---|---|---|---:|---|
| `--max-num-seqs` | `max_num_seqs` | `Scheduler.max_num_running_reqs`, `SchedulerConfig.max_num_seqs`, `GPUModelRunner.input_batch` cap | Update scheduler admission cap at step boundary; if increased above initial `max_num_seqs`, grow `InputBatch` CPU/GPU metadata rows. | `< 50 ms` | **Low (`~40 LoC`)** — direct attribute update in `Scheduler` + `InputBatch` cap check. |
| `--max-num-batched-tokens` | `max_num_batched_tokens` | `Scheduler.max_num_scheduled_tokens`, `SchedulerConfig.max_num_batched_tokens` | Update token budget per scheduling step; if increased above initial `max_num_batched_tokens`, ensure `GPUModelRunner.input_ids` / `positions` scratch buffers are sized to `max(max_num_batched_tokens, max_cudagraph_capture_size)`. | `< 50 ms` | **Low (`~50 LoC`)** — scheduler integer field + scratch buffer slice check. |
| `--max-model-len` *(within allocated KV & RoPE capacity)* | `max_model_len` | `ModelConfig.max_model_len`, `SchedulerConfig.max_model_len`, `OpenAIServing*.max_model_len` | In vLLM V1, KV cache pre-allocates all blocks permitted by `gpu_memory_utilization` regardless of `max_model_len`. Lowering or raising `max_model_len` up to the allocated KV token capacity and RoPE `max_position_embeddings` only updates request validation and scheduler limits. | `< 50 ms` | **Low (`~60 LoC`)** — update `max_model_len` in `EngineCore` scheduler and API server request validator. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | `enable_prefix_caching` | `KVCacheManager.enable_caching`, `BlockPool.cached_block_hash_to_block` | Drain in-flight requests (`R1`), evict cached prefix hashes in `BlockPool`, and flip `enable_caching` boolean. | `< 50 ms` | **Low (`~30 LoC`)** — already compatible with `KVCacheManager.reset_prefix_cache()`. |
| `--scheduling-policy` | `scheduling_policy` | `Scheduler.policy` | Swap policy selector (`fcfs` vs `priority`) at step boundary. | `< 20 ms` | **Trivial (`~15 LoC`)** — attribute swap on `Scheduler`. |
| `--enable-chunked-prefill` / `--no-enable-chunked-prefill` | `enable_chunked_prefill` | `SchedulerConfig.enable_chunked_prefill`, `Scheduler.chunked_prefill_enabled` | Flip scheduler chunking flag at step boundary. | `< 20 ms` | **Trivial (`~20 LoC`)** — attribute swap on `Scheduler`. |
| `--async-scheduling` / `--no-async-scheduling` | `async_scheduling` | `EngineCoreProc` step loop dispatch | Toggle async output processing / step overlap flag between batches. | `< 50 ms` | **Low (`~40 LoC`)** — drain 1 step in `EngineCoreProc` before toggling. |

---

### Tier 1B — `in_process_graph` (`0.5 – 2.0 s`, Live Dispatch Toggle or CUDA Graph Recapture)

| CLI Flag (`vllm serve`) | Parser Dest | Internal vLLM V1 Objects Mutated | Live Update Procedure | Target Latency | Effort for `< 2 s` In-Process |
|---|---|---|---|---:|---|
| `--enforce-eager` / `--no-enforce-eager` | `enforce_eager` | `ModelConfig.enforce_eager`, `CompilationConfig.cudagraph_mode`, `CUDAGraphWrapper` dispatch | **On (`True`)**: bypass `CUDAGraphWrapper.__call__` and dispatch directly to the compiled/eager forward callable (`< 10 ms`). **Off (`False`)**: re-enable existing captured CUDA graphs (`< 10 ms`) or capture graphs via `RecaptureCoordinator` (`0.8 – 2.0 s`) if started in eager mode. | `10 ms` (if graphs captured) / `1.5 s` (first capture) | **Low (`~60 LoC`)** — branch in `CUDAGraphWrapper` / `GPUModelRunner._dummy_run` + `RecaptureCoordinator`. |
| `--cudagraph-capture-sizes` | `cudagraph_capture_sizes` | `CompilationConfig.cudagraph_capture_sizes`, `GPUModelRunner` graph pool | Release old graph batch descriptors not in the new set, capture missing batch sizes via `RecaptureCoordinator` on theexisting `graph_pool`. | `0.5 – 2.0 s` | **Low (`~80 LoC`)** — `RecaptureCoordinator` already recaptures `cudagraph_capture_sizes` in place. |
| `--max-cudagraph-capture-size` | `max_cudagraph_capture_size` | `CompilationConfig.max_cudagraph_capture_size`, `CompilationConfig.cudagraph_capture_sizes` | Recompute capture size list and run `RecaptureCoordinator` for any newly added batch sizes. | `0.5 – 2.0 s` | **Low (`~50 LoC`)** — delegates to `cudagraph_capture_sizes` recapture. |
| `-cc` / `--compilation-config` | `compilation_config` | `VllmConfig.compilation_config`, `TorchCompileWithNoGuardsWrapper`, `CUDAGraphWrapper` | Update `cudagraph_mode` (`NONE` / `PIECEWISE` / `FULL`), `pass_config`, or custom ops; invalidate compiled FX graph wrapper and recapture CUDA graphs via `RecaptureCoordinator` (hitting warm `TORCHINDUCTOR_CACHE_DIR`). | `1.0 – 2.5 s` | **Medium (`~150 LoC`)** — reset `TorchCompileWithNoGuardsWrapper` compiled fn + recapture graphs. |
| `-O` / `--optimization-level` | `optimization_level` | `VllmConfig.compilation_config` (`-O0` eager through `-O3` full compile + cudagraph) | Map `-O0..-O3` to `CompilationConfig` presets, toggle eager/compiled callable, and recapture graphs if transitioning into `-O2`/`-O3`. | `0.5 – 2.0 s` | **Medium (`~100 LoC`)** — reuses `compilation_config` and `enforce_eager` toggles. |
| `--speculative-config` / `--num-speculative-tokens` *(when draft model is un-weighted: ngram / eagle head already resident)* | `speculative_config` | `SpeculativeConfig`, `GPUModelRunner.drafter`, `InputBatch` bonus token slots | Adjust `num_speculative_tokens` loop count in `EagleProposer` / `NgramProposer` and recapture eagle CUDA graphs. | `0.8 – 2.0 s` | **Medium (`~180 LoC`)** — resize rejection-sampler bonus token buffer + recapture drafter graph. |

---

### Tier 2 — `warm_weights_ipc` (`7 – 20 s` Now via Zygote + `WeightHolder` IPC; Path to `< 1.5 s` In-Process)

These flags currently trigger a fast worker restart via `engine.py reconfigure` where `weights.py` (`WeightHolder`) keeps all model weights in GPU HBM (`0 B` disk read) and `zygote.py` skips Python imports. Below is how each one works today and how to promote it to `< 1.5 s` in-process:

| CLI Flag (`vllm serve`) | Parser Dest | Why It Currently Restarts Workers | Path to `< 1.5 s` In-Process Hot-Update | Effort to Make In-Process |
|---|---|---|---|---|
| `--gpu-memory-utilization` | `gpu_memory_utilization` | `GPUWorker.initialize_from_config` allocates a single contiguous `kv_cache_tensors` buffer per attention layer at startup sized to `gpu_memory_utilization`. | Add `GPUWorker.resize_kv_cache(new_utilization)`: drain in-flight requests, `del` old `kv_cache_tensors`, call `torch.cuda.empty_cache()`, allocate new KV tensors, rebind `layer.kv_cache` pointers, and re-initialize `KVCacheManager` (`~0.3 s` without touching weights or CUDA graphs when using `PIECEWISE` graphs, or `~1.5 s` with `FULL` graph recapture). | **Medium (`~200 LoC`)** — KV tensor rebind + `KVCacheManager` reset. |
| `--block-size` | `block_size` | Changes KV cache tensor shape `(num_blocks, block_size, num_kv_heads, head_size)` and attention backend metadata (`FlashAttentionMetadata` / `FlashInferMetadata`). | Reallocate `kv_cache_tensors` with new `block_size`, rebuild attention backend metadata builder, and recapture CUDA graphs. | **Medium (`~250 LoC`)** — KV tensor reshape + attention backend re-init + graph recapture. |
| `--swap-space` | `swap_space` | Pre-allocates pinned host DRAM swap blocks during worker initialization. | Lazily reallocate CPU pinned swap tensor pool in `CacheEngine` / `KVCacheManager` without touching GPU state. | **Low (`~80 LoC`)** — host-only buffer reallocation. |

---

### Tier 3 — `live_reshard_or_cold` (`WeightCacheKey` Changes: Sharding or Post-Processed Weight Layout)

These flags alter `WeightCacheKey` in `weights.py` because they change either (a) how tensors are partitioned across ranks (`tier2a_reshard`) or (b) the post-processed tensor bytes/dtypes created by `process_weights_after_loading` (`tier2b_weight_format`).

| CLI Flag (`vllm serve`) | Sub-tier | Internal Impact on `WeightCacheKey` | Current Handling in `engine.py reconfigure` | Path to Fast Reconfiguration (`2 – 5 s`) |
|---|---|---|---|---|
| `-tp` / `--tensor-parallel-size` | `tier2a_reshard` | Changes `QKVParallelLinear`, `RowParallelLinear`, `MergedColumnParallelLinear` shard slices per rank. | Refused before stopping running engine (`weights_reconfig_tier2`) unless `weights.py` is restarted (`--weights-accept-reconfig`). | **Live NCCL / IPC AllGather + Re-slice (`~2 – 4 s`)**: `WeightHolder` gathers shards across old TP ranks in GPU memory and re-slices for new TP world size without reading safetensors from disk. |
| `-pp` / `--pipeline-parallel-size` | `tier2a_reshard` | Changes which `LlamaDecoderLayer` / `Qwen3DecoderLayer` indices live on each GPU. | Refused before stopping running engine unless `--weights-accept-reconfig`. | **P2P GPU-to-GPU Layer Migration (`~1 – 3 s`)**: transfer non-overlapping decoder layer weights between GPUs over NVLink (`cudaMemcpyPeerAsync`). |
| `-dp` / `--data-parallel-size` | `tier2a_reshard` | Changes DP replica count and MoE expert communicator layout. | Refused before stopping running engine unless `--weights-accept-reconfig`. | **Zero-Copy Broadcast from Rank 0 (`~1 – 2 s`)**: broadcast resident GPU weights over NVLink from existing DP replica to new DP replica. |
| `--enable-expert-parallel` | `tier2a_reshard` | Switches `FusedMoE` between TP-sliced experts and EP-partitioned experts (`expert_map`). | Refused before stopping running engine unless `--weights-accept-reconfig`. | **Holder-to-holder expert transpose (`reshard_live.py`, implemented, not yet GPU-measured)**: a `--load-format dummy` holder group for the target flags is filled from the running group over CUDA IPC; dense weights are the same shard at TP2 and TP2+EP2 and are copied rank-locally, expert tensors are decoded from the kernel layout, transposed and re-encoded. On CUDA, vLLM 0.33's oracle tries the FlashInfer TRT-LLM BF16 kernel first (from its source; the B200 pick is not yet read from a log), whose stored expert bytes are a per-shape block permutation (TP2 and EP2 differ), so a slice of stored bytes is wrong (`test_reshard_live.py`, CPU, real vLLM conversion). |
| `--kv-cache-dtype` (`auto` / `fp8`) | `tier2b_weight_format` | Changes attention KV cache element size (`2 B` vs `1 B`) and `k_scale` / `v_scale` quantization factors. | Refused before stopping running engine unless `--weights-accept-reconfig`. | **Split KV Cache out of `WeightCacheKey` (`~0.5 – 1.5 s`)**: model weights (`qkv_proj`, `mlp`) are identical; only `kv_cache_tensors` and attention backend need reallocation + graph recapture! |
| `--moe-backend` / `--linear-backend` | `tier2b_weight_format` | `process_weights_after_loading` permutes/packs weight tensors into kernel-specific layouts (e.g. Marlin, DeepGEMM, FlashInfer CUTLASS). | Refused before stopping running engine unless `--weights-accept-reconfig`. | **Keep Pre-Pack Raw Weights in `WeightHolder` (`~1 – 2 s`)**: store canonical unpacked tensors in `WeightHolder` and run `process_weights_after_loading` on the GPU in the worker. |
| `--dtype` / `-q` / `--quantization` | `tier2b_weight_format` | Changes weight element type (`bf16` vs `fp16` vs `fp8` / `awq` / `gptq`). | Refused before stopping running engine unless `--weights-accept-reconfig`. | On-GPU cast (`bf16 <-> fp16` or dynamic `bf16 -> fp8` quantization in `~1 s`); different checkpoint quantization (`awq`/`gptq`) requires disk load. |
| `MODEL` / `--model` / `--revision` | `tier2b_weight_format` | Different model weights and architecture. | Refused before stopping running engine unless `--weights-accept-reconfig`. | Requires loading new checkpoint into `WeightHolder` (or dual-model HBM residency when free HBM permits). |
