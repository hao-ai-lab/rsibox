# TensorRT-LLM (PyTorch backend) live-lab architecture

How NVIDIA TensorRT-LLM's PyTorch backend (`tensorrt_llm.LLM(..., backend="pytorch")`) maps onto RSIBox's shared transaction kernel (`core/livelab/kernel`) and warm snapshot contract (`core/livelab/rsibox/warm`), across both `TP=1` single-process and `TP >= 2` multi-rank MPI deployments.

## Process tree

`TrtllmLab` supports both single-GPU (`tensor_parallel_size == 1`) and multi-GPU tensor-parallel (`tensor_parallel_size >= 2`) execution while keeping the host process pure stdlib (zero `torch` or `tensorrt_llm` imports):

- **`TP=1` single-process mode**: forces `TLLM_WORKER_USE_SINGLE_PROCESS=1` in the spawned RSIBox worker subprocess so `GenerationExecutor.create()` instantiates `GenerationExecutorWorker` in-process.
- **`TP >= 2` multi-rank MPI mode**: removes `TLLM_WORKER_USE_SINGLE_PROCESS` and launches a driver subprocess (`adapter.agent` in `"driver"` mode) that owns `tensorrt_llm.LLM(..., tensor_parallel_size=tp)` and spawns `tp` MPI workers via `MPIPoolExecutor`. A pure-stdlib zero-eager-import `sitecustomize.py` hook (`adapter/mpi_hook.py` via `sys.meta_path`) attaches an `rsibox.ipc.UnixJsonServer` daemon thread (`rank-0.sock`, `rank-1.sock`, ...) to each rank inside `PyExecutor.__init__`, pinned to that rank's CUDA device.

```text
Host process (TrtllmLab : rsibox.BaseLab, pure stdlib — zero torch/tensorrt_llm imports)
  ├── RecaptureCoordinator + kernel.session.Session (parallel fanout across all ranks)
  ├── TrtllmQuiesceHooks (quiesce_hooks() -> DynamoK8sSnapshotDriver across all ranks)
  ├── [TP=1]  _WorkerHandle[rank0] ── AF_UNIX / FileChannel ──▶ adapter.agent (GenerationExecutorWorker)
  └── [TP>=2] _WorkerHandle[driver] (generate / inject) + _RankClient[rank0..rankN-1]
        └── MPIPoolExecutor ranks (hooked via pure-stdlib sys.meta_path mpi_hook.py on PyExecutor.__init__)
              ├── kernel.coordinator.LocalParticipant("rank{i}")
              ├── PyTorchModelEngine (model forward + CUDA graph runner)
              ├── TorchSampler (_process_requests fast path, sample_async)
              └── KVCacheManager (C++ KVCacheManagerCpp pool manager)
```

## Priced restart boundaries

Costs below report **measured** timings (`1x` and `2x NVIDIA B200`, `TensorRT-LLM 1.0.0rc6`, `Qwen3-0.6B`, `use_cuda_graph=True`, `2026-11-24`):

| Restart boundary | Mechanism | Status | Cost | Scope |
|---|---|---|---|---|
| In-place Python code swap (`R0`) | `RecaptureCoordinator` parks all ranks in parallel, runs `validate` + `commit` + `linecache` sync on each `LocalParticipant`, and releases via `release_gap` | **measured** | `TP=1`: `56.4 ms` install / `6.2 ms` undo (`107.2 ms` edit+check); `TP=2`: `67.6 ms` install / `13.4 ms` undo (`c9` median) | `tensorrt_llm._torch.pyexecutor.*` (`sampler`, `py_executor`, `handle_logits`, `scheduler`) and eager Python code |
| CUDA Graph invalidation & lazy recapture (`R1`) | Auto-triggered when patching `CAPTURED_PREFIXES` (`tensorrt_llm._torch.models.*`, `modules.*`, `attention_backend.*`): calls `model_engine._release_cuda_graphs()` + `gc.collect()` + `torch.cuda.empty_cache()` in parallel across all ranks on commit & undo; `PyExecutor` lazily re-captures on the next decode step | **measured** | `TP=2`: `275.1 ms` install (`244 ms` graph release per rank), `374.3 ms` first check (`224/224` witness calls per rank), `149.1 ms` undo | Edits inside CUDA-graph-captured model forward passes (e.g., `Qwen3DecoderLayer.forward`) |
| Volatile state reset (`reset("state")`) | Rejects if `PyExecutor.active_requests` is non-empty, flushes KV iteration events (`kv_cache_manager.flush_iteration_events()`), and reseeds Python / PyTorch / CUDA RNGs | **measured** | `1.1 ms` (`TP=1`) | Between checks on the same warm worker(s) |
| Snapshot KV-pool quiesce / resume (`TrtllmQuiesceHooks`) | Rejects if busy/dirty/poisoned or `enable_block_reuse=True`; releases paged KV pools via `kv_cache_manager.impl.release_pools()` and restores via `allocate_pools(False)` + pointer/mapping refresh and fail-closed pointer-shift check | **measured** | `TP=1`: `187.5 ms` quiesce (`-23.60 GiB`), `7.4 ms` resume; `TP=2`: `-46.67 GiB` (`-23.33 GiB`/GPU), `15.2 ms` resume | `ai-dynamo/snapshot` (`ready-for-snapshot` -> `restore-complete` -> `trtllm-restore-ready`) across all ranks |
| Full worker respawn (`reset("respawn")`) | Stops worker/driver process tree via `rsibox.proc.stop_tree(..., include_pg=True)`, respawns fresh worker(s), and calls `session.recovered(coordinator)` | **measured** | `TP=1`: `12.40 s` warm-JIT / `26.66 s` cold-JIT; `TP=2`: `22.3–22.7 s` respawn (`36.9 s` cold) | Poisoned sessions (`crash`, `hang`, `lost`, `tamper`), constructors, module constants, or native C++ changes |

## Fresh-process parity

`TrtllmLab.fresh_check(patch, workload)` locates the top-level `tensorrt_llm` package directory via `importlib.util.find_spec("tensorrt_llm")` without importing it into the host, constructs a symlink overlay via `rsibox.PackageOverlay` with the patched `.py` modules written into the overlay, and runs `adapter.agent` in `"fresh"` mode via `rsibox.run_fresh_subprocess` (preserving `tensor_parallel_size`, so `TP=2` runs fresh `TP=2` MPI parity).

## Scope and current boundaries

- Verified live on `1x NVIDIA B200` (`TP=1`) and `2x NVIDIA B200` (`TP=2` MPI) inside TensorRT-LLM `1.0.0rc6` PyTorch backend with CUDA graphs enabled.
- `R1` recapture uses lazy release (`_release_cuda_graphs()`) from the RPC thread so `PyExecutor`'s own thread re-captures cleanly on the next decode step; eager warmup from the daemon RPC thread is avoided because `TllmGenFmhaKernel` requires the executor thread.
- Native TensorRT C++ engine plans (`backend="trt"`), TensorRT-LLM C++ bindings (`bindings.*.so`), constructors, and module-level constants are out of scope for in-place Python swaps and require respawn or rebuild.
