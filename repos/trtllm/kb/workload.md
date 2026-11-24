# Workload F(x) and its correctness gate

The deterministic workload for the TensorRT-LLM (PyTorch backend) live lab.

## Setup

| Setting | Value |
|---|---|
| Model | `Qwen3-0.6B`, real BF16 weights |
| Backend | TensorRT-LLM `1.0.0rc6` PyTorch backend (`backend="pytorch"`, `TLLM_WORKER_USE_SINGLE_PROCESS=1`) |
| Parallelism | `tensor_parallel_size=1` (`1x NVIDIA B200`) |
| Executor config | `max_batch_size=4`, `max_num_tokens=2048`, `disable_overlap_scheduler=True`, `KvCacheConfig(free_gpu_memory_fraction=0.15, enable_block_reuse=False)` |
| Prompts | `["The capital of France is", "In computer science, a hash table is a data structure that"]` |
| Sampling | `SamplingParams(max_tokens=16, temperature=0.0)` |
| Reset before each check | `reset("state")`: verify `active_requests` is empty, call `kv_cache_manager.flush_iteration_events()`, and reseed `random` / `torch` / `torch.cuda` to `0` |

## Function F(x)

Wall seconds for `engine.generate(prompts, sampling_params=params)` returning exact per-prompt `token_ids` lists (`{"rank0": [[...], [...]]}`), paired with kernel execution witness counts (`proven=True`) on every swapped target.

## Correctness gate

- **Repeatable base:** repeated `check()` calls with `reset("state")` between them and across install/uninstall cycles must return identical `token_ids` and restore the exact base code fingerprint (`c0`, `c4`).
- **Execution witness:** every live edit must report `proven=True` with non-zero witness counts on every participating rank (`17` calls on `TorchSampler._process_requests` for 1 prefill + 16 decode steps on the 2-prompt greedy batch).
- **Fresh-process parity:** a visible edit counts only when a fresh TensorRT-LLM subprocess started from a `PackageOverlay` of the edited files produces bit-exact `token_ids` (`c3`).
- **Post-snapshot parity:** after `TrtllmQuiesceHooks` releases and re-allocates KV cache pools through `DynamoK8sSnapshotDriver`, base generation must remain bit-exact and subsequent live swaps + rollbacks must remain `proven=True` and bit-exact.
