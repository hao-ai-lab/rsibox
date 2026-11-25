# Lessons Learned

1. **Snapshot and restore FSDP2 `DTensor` shards via `.to_local()` and per-device CUDA RNG states.**
   Parameters wrapped by `torch.distributed.fsdp.fully_shard` are `DTensor` instances whose
   sharded local storage is accessed via `param.to_local()`. Cloning `param.to_local().detach().clone()`
   at startup, capturing `random.getstate()`, `torch.get_rng_state()`, and per-device
   `torch.cuda.get_rng_state(self.device)` (never `get_rng_state_all()` / `manual_seed_all()`,
   which open stray CUDA contexts on sibling GPUs in multi-rank runs), and restoring all of them
   alongside `optimizer.state.clear()` on `reset("state")` yields `0.0` `max_abs_loss_diff` and
   bit-exact repeatability across both 1-rank and 2-rank NCCL runs.

2. **Spawn parallel per-rank `PackageOverlay` children for multi-rank `fresh_check` with a shorter inner timeout.**
   Running `torch.distributed.init_process_group(world_size > 1)` sequentially in one process
   deadlocks waiting for peer ranks. `_run_fresh_parallel` spawns `world_size` isolated rank
   subprocesses inheriting `PackageOverlay`'s `PYTHONPATH`, arms `PR_SET_PDEATHSIG(SIGTERM)` +
   immediate `parent_pid` check on those short-lived rank children, enforces an inner deadline
   15 s shorter than the outer `run_fresh_subprocess` timeout (so `_stop_fresh_children` always
   runs before an outer `SIGKILL`), and sends parallel `SIGTERM` across all sibling process groups
   as soon as any rank exits non-zero.

3. **Never arm Linux `PR_SET_PDEATHSIG` on long-lived workers that may be spawned from transient threads.**
   Linux binds `PR_SET_PDEATHSIG` to the spawning OS thread rather than the parent process, so a
   `SandboxPool` or `ThreadPoolExecutor` thread exiting after spawning/respawning `TorchTitanLab`
   would immediately send `SIGTERM` to healthy workers. Long-lived `serve_worker` processes instead
   check `os.getppid() != parent_pid` at entry (before NCCL init) and in their 1 s idle poll loop.

4. **Use `FIRST_EXCEPTION` fan-out so a crashed rank never stalls healthy siblings.**
   Awaiting worker futures sequentially inside `with ThreadPoolExecutor(...)` blocks until
   `check_timeout` if rank 1 crashes while rank 0 is still running. Waiting with
   `concurrent.futures.wait(..., return_when=FIRST_EXCEPTION)` and calling
   `pool.shutdown(wait=False, cancel_futures=True)` fails fast on any rank error.

5. **Deterministic CUDA GEMM flags are required for bit-exact `c3_fresh_equivalence`.**
   Disabling TF32 (`allow_tf32 = False`), enabling `torch.use_deterministic_algorithms(True,
   warn_only=True)`, and setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` in both warm worker and fresh
   `PackageOverlay` subprocess environments ensures live FSDP2 training steps match fresh restarts
   to 8 decimal places across 3-step chains.

6. **Scale initial weights by `1 / sqrt(fan_in)` and keep `torch` imports strictly lazy.**
   Xavier-scaled initialization keeps 4-layer residual MLP losses near `~1.0` for sensitive edit
   detection, while deferring `import torch` into worker runtime methods keeps `rsibox doctor` and
   `rsibox conformance torchtitan --real -- --help` CPU-only with zero eager GPU imports.
