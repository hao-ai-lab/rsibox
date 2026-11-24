# Lessons Learned

1. **Snapshot and restore FSDP2 `DTensor` shards via `.to_local()` and exact post-init RNG states.**
   Parameters wrapped by `torch.distributed.fsdp.fully_shard` are `DTensor` instances whose
   sharded local storage is accessed via `param.to_local()`. Cloning `param.to_local().detach().clone()`
   at startup, capturing `random.getstate()`, `torch.get_rng_state()`, and
   `torch.cuda.get_rng_state_all()` after model initialization, and restoring all of them alongside
   `optimizer.state.clear()` on `reset("state")` yields `0.0` `max_abs_loss_diff` and bit-exact
   repeatability even for stochastic hot-swapped ops.

2. **Fail closed on unsupported multi-rank FSDP2 fresh-process topologies.**
   Running `torch.distributed.init_process_group(world_size > 1)` sequentially in a single fresh
   subprocess deadlocks waiting for peer ranks. Validating `mode == "fsdp2" and world_size > 1`
   at both `TorchTitanLab.__init__` and `run_fresh` raises an explicit `ValueError` until
   multi-process `torchrun` fresh-check orchestration is added.

3. **Use `FIRST_EXCEPTION` fan-out so a crashed rank never stalls healthy siblings.**
   Awaiting worker futures sequentially inside `with ThreadPoolExecutor(...)` blocks until
   `check_timeout` if rank 1 crashes while rank 0 is still running. Waiting with
   `concurrent.futures.wait(..., return_when=FIRST_EXCEPTION)` and calling
   `pool.shutdown(wait=False, cancel_futures=True)` fails fast on any rank error.

4. **Deterministic CUDA GEMM flags are required for bit-exact `c3_fresh_equivalence`.**
   Disabling TF32 (`allow_tf32 = False`), enabling `torch.use_deterministic_algorithms(True,
   warn_only=True)`, and setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` in both warm worker and fresh
   `PackageOverlay` subprocess environments ensures live FSDP2 training steps match fresh restarts
   to 8 decimal places across 3-step chains.

5. **Scale initial weights by `1 / sqrt(fan_in)` and keep `torch` imports strictly lazy.**
   Xavier-scaled initialization keeps 4-layer residual MLP losses near `~1.0` for sensitive edit
   detection, while deferring `import torch` into worker runtime methods keeps `rsibox doctor` and
   `rsibox conformance torchtitan --real -- --help` CPU-only with zero eager GPU imports.
