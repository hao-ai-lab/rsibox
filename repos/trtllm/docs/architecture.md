# TensorRT-LLM (PyTorch backend) live-lab architecture

How NVIDIA TensorRT-LLM's PyTorch backend (`tensorrt_llm.LLM(..., backend="pytorch")`) maps onto RSIBox's shared transaction kernel (`core/livelab/kernel`) and warm snapshot contract (`core/livelab/rsibox/warm`).

## Process tree

By default, `tensorrt_llm.LLM` creates a `GenerationExecutorProxy` that spawns an MPI worker subprocess via `mpi4py`, even when `tensor_parallel_size == 1`. Forcing `TLLM_WORKER_USE_SINGLE_PROCESS=1` in the spawned RSIBox worker process before importing `tensorrt_llm` instructs `GenerationExecutor.create()` to instantiate `GenerationExecutorWorker` in-process instead of spawning a nested MPI worker.

`TrtllmLab` uses this single-process worker boundary so each spawned RSIBox worker subprocess is the live PyTorch executor process, while the host process never imports `tensorrt_llm` or `torch`:

```text
Host process (TrtllmLab : rsibox.BaseLab, pure stdlib — zero torch/tensorrt_llm imports)
  ├── StepGapCoordinator + kernel.session.Session
  ├── TrtllmQuiesceHooks (quiesce_hooks() -> DynamoK8sSnapshotDriver)
  └── _WorkerHandle[rank0]  ── AF_UNIX (UnixJsonServer) or FileChannelServer ──▶
        Worker subprocess (python3 -m adapter.agent, start_new_session=True, ppid guard)
          ├── kernel.coordinator.LocalParticipant("rank0")
          ├── tensorrt_llm.LLM(backend="pytorch") [TLLM_WORKER_USE_SINGLE_PROCESS=1]
          │     └── GenerationExecutorWorker._executor
          │           └── PyExecutor (background worker_thread)
          │                 ├── PyTorchModelEngine (model forward + CUDA graph runner)
          │                 ├── TorchSampler (_process_requests fast path, sample_async)
          │                 └── KVCacheManager (C++ KVCacheManagerCpp pool manager)
          └── Step-gap RPC loop (park / validate / commit / undo / generate / quiesce_memory / resume_memory)
```

## Priced restart boundaries

Costs below separate **measured** timings (`1x NVIDIA B200`, `TensorRT-LLM 1.0.0rc6`, `Qwen3-0.6B`, `TP=1`, `2026-11-24`) from **anticipated** paths not yet wired in the adapter:

| Restart boundary | Mechanism | Status | Cost | Scope |
|---|---|---|---|---|
| In-place Python code swap | `StepGapCoordinator` parks workers via `park`, runs `validate` + `commit` + `linecache` sync on `LocalParticipant`, and releases via `release_gap` | **measured** | `56.4 ms` install, `6.2 ms` undo (`107.2 ms` edit+check) | `tensorrt_llm._torch.pyexecutor.*` (`sampler`, `py_executor`, `handle_logits`, `scheduler`) and eager model code |
| Volatile state reset (`reset("state")`) | Rejects if `PyExecutor.active_requests` is non-empty, flushes KV iteration events (`kv_cache_manager.flush_iteration_events()`), and reseeds Python / PyTorch / CUDA RNGs | **measured** | `1.1 ms` | Between checks on the same warm worker |
| Snapshot KV-pool quiesce / resume (`TrtllmQuiesceHooks`) | Rejects if busy/dirty/poisoned; releases paged KV pools via `kv_cache_manager.impl.release_pools()` (clearing pool pointers/mapping) and restores via `allocate_pools(enable_block_reuse)` + pointer/mapping refresh and CUDA-graph pointer-shift check | **measured** | `187.5 ms` quiesce (`-24,168 MiB`), `7.4 ms` resume | `ai-dynamo/snapshot` (`ready-for-snapshot` -> `restore-complete` -> `trtllm-restore-ready`) |
| Selective decode CUDA graph recapture | Release and recapture only affected batch sizes inside `PyTorchModelEngine._cuda_graphs` in the step gap | **anticipated** | ~`0.5–1.5 s` (not yet measured) | Edits inside CUDA-graph-captured decode forward passes |
| Full worker respawn (`reset("respawn")`) | Stops worker process tree via `rsibox.proc.stop_tree(..., include_pg=True)`, respawns a fresh worker, and calls `session.recovered(coordinator)` | **measured** | `12.40 s` warm-JIT / `26.66 s` cold-JIT | Poisoned sessions (`crash`, `hang`, `lost`, `tamper`), constructors, module constants, or native C++ changes |

## Fresh-process parity

`TrtllmLab.fresh_check(patch, workload)` locates the top-level `tensorrt_llm` package directory via `importlib.util.find_spec("tensorrt_llm")` without importing it into the host, constructs a symlink overlay via `rsibox.PackageOverlay` with the patched `.py` modules written into the overlay, and runs `adapter.agent` in `"fresh"` mode via `rsibox.run_fresh_subprocess`.

## Scope and current boundaries

- Verified live on `1x NVIDIA B200` (`TP=1`) inside TensorRT-LLM `1.0.0rc6` PyTorch backend (`TLLM_WORKER_USE_SINGLE_PROCESS=1`).
- `TrtllmLab` enforces `tensor_parallel_size == 1` at construction; multi-GPU MPI (`tensor_parallel_size > 1`) spawns `GenerationExecutorProxy` MPI workers rather than in-process `GenerationExecutorWorker` and is left for a follow-up PR.
- Native TensorRT C++ engine plans (`backend="trt"`), TensorRT-LLM C++ bindings (`bindings.*.so`), constructors, and module-level constants are out of scope for in-place Python swaps and require respawn or rebuild.
