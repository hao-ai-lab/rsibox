# TensorRT-LLM (PyTorch backend) live-lab architecture

How NVIDIA TensorRT-LLM's PyTorch backend (`tensorrt_llm.LLM(..., backend="pytorch")`) maps onto RSIBox's shared transaction kernel (`core/livelab/kernel`) and warm snapshot contract (`core/livelab/rsibox/warm`).

## Process tree

By default, `tensorrt_llm.LLM` creates a `GenerationExecutorProxy` that spawns an MPI worker subprocess via `mpi4py`, even when `tensor_parallel_size == 1`. Setting `TLLM_WORKER_USE_SINGLE_PROCESS=1` before importing `tensorrt_llm` instructs `GenerationExecutor.create()` to instantiate `GenerationExecutorWorker` in-process instead of spawning a nested MPI worker.

`TrtllmLab` exploits this clean single-process boundary so each RSIBox worker subprocess IS the live PyTorch executor process:

```text
Host process (TrtllmLab : rsibox.BaseLab)
  ├── StepGapCoordinator + kernel.session.Session
  ├── TrtllmQuiesceHooks (quiesce_hooks() -> DynamoK8sSnapshotDriver)
  └── _WorkerHandle[rank0..rankN]  ── AF_UNIX (UnixJsonServer) or FileChannelServer ──▶
        Worker subprocess (python3 -m adapter.agent)
          ├── kernel.coordinator.LocalParticipant("rank{i}")
          ├── tensorrt_llm.LLM(backend="pytorch") [TLLM_WORKER_USE_SINGLE_PROCESS=1]
          │     └── GenerationExecutorWorker._executor
          │           └── PyExecutor (background worker_thread)
          │                 ├── PyTorchModelEngine (model forward + CUDA graph runner)
          │                 ├── TorchSampler (_process_requests fast path, sample_async)
          │                 └── KVCacheManager (C++ KVCacheManagerCpp pool manager)
          └── Step-gap RPC loop (park / validate / commit / undo / generate / quiesce_memory / resume_memory)
```

## Restart boundaries and transaction gap

| Boundary | Mechanism | Scope |
|---|---|---|
| Live Python edit (`R0 swap`) | `StepGapCoordinator` parks all workers via `park`, runs `validate` + `commit` + `linecache` sync on each `LocalParticipant`, and releases via `release_gap` | `tensorrt_llm._torch.pyexecutor.*` (e.g., `sampler`, `py_executor`, `handle_logits`, `scheduler`) and eager model modules |
| Volatile state reset (`reset("state")`) | Rejects if `PyExecutor.active_requests` is non-empty, flushes KV iteration events (`kv_cache_manager.flush_iteration_events()`), and reseeds Python / PyTorch / CUDA RNGs | Between checks on the same warm worker |
| Snapshot quiesce / restore (`TrtllmQuiesceHooks`) | Rejects if busy/dirty/poisoned; releases paged KV pools via `kv_cache_manager.impl.release_pools()`, optional `model_engine._release_cuda_graphs()`, and optional `tensorrt_llm._torch.virtual_memory` tags; restores via `allocate_pools(False)` + refreshing `kv_cache_pool_pointers` and `kv_cache_pool_mapping` | `ai-dynamo/snapshot` (`ready-for-snapshot` -> `restore-complete` -> `trtllm-restore-ready`) |
| Full worker respawn (`reset("respawn")`) | Terminates all worker subprocesses, cleans up IPC endpoints, respawns fresh workers, and calls `session.recovered(coordinator)` | Poisoned sessions (`crash`, `hang`, `lost`, `tamper`) or unsupported startup/C++ changes |

## Fresh-process parity

`TrtllmLab.fresh_check(patch, workload)` builds a clean symlink overlay of the `tensorrt_llm` package (or `adapter` for `ToyTrtllmLab`) via `rsibox.PackageOverlay` with the patched `.py` modules written into the overlay, then runs `adapter.agent` in `"fresh"` mode via `rsibox.run_fresh_subprocess` so live hot-swaps are verified bit-exact against a fresh process restart.

## Scope and current boundaries

- Verified live on `1x NVIDIA B200` (`TP=1`) inside TensorRT-LLM `1.0.0rc6` PyTorch backend (`TLLM_WORKER_USE_SINGLE_PROCESS=1`).
- Multi-GPU MPI (`tensor_parallel_size > 1`) spawns `GenerationExecutorProxy` MPI workers rather than in-process `GenerationExecutorWorker`; extending live swaps across MPI ranks is left for a follow-up PR.
- Native TensorRT C++ engine plans (`backend="trt"`), TensorRT-LLM C++ bindings (`bindings.*.so`), and constructors/module-level constants are out of scope for `R0` Python swaps and require respawn or rebuild.
