# TorchTitan / FSDP2 Live-Lab Architecture

## Scope & Process Tree

`TorchTitanLab` hosts isolated worker subprocesses over per-rank `AF_UNIX` JSON channels
(`rsibox.ipc.UnixJsonServer`). Today it supports:

- **Real GPU FSDP2 (`mode="fsdp2"`, `world_size >= 1`)**: Real CUDA PyTorch `nn.Module`
  (`TitanMLPNet`) sharded with `torch.distributed.fsdp.fully_shard` over an `nccl` process group
  plus `AdamW`, verified at both `world_size=1` (`1× B200`) and `world_size=2` (`2× B200` over real
  NCCL collectives with parallel multi-rank `PackageOverlay` `fresh_check`). Invalid `world_size`
  (`<= 0`, non-int) or `world_size > torch.cuda.device_count()` fails closed immediately. The
  adapter does not yet import upstream `pytorch/torchtitan`.
- **Eager PyTorch (`mode="cuda"` or `mode="torch_cpu"`)**: Same `TitanMLPNet` training loop without
  `fully_shard` or `torch.distributed` (parallel multi-rank `fresh_check` is also used for
  `mode="cuda"` when `world_size > 1`).
- **Multi-rank CPU toy (`ToyTorchTitanLab`, `mode="cpu"`, `world_size=2`)**: Pure-Python
  fake-distributed 2-worker topology exercising multi-rank atomic commits, per-rank witnesses,
  and crash/hang fail-fast fan-out in zero-dependency CPU CI.

```text
Host process (TorchTitanLab, BaseLab + RecaptureCoordinator + pluggable weight_store/recovery_driver)
  ├── AF_UNIX socket (rsitt0) ──► Rank 0 worker (python -m adapter.worker <spec.json>, own session)
  │     ├── Backend: nccl + fully_shard FSDP2 DTensor (world_size >= 1) | eager PyTorch | CPU toy
  │     ├── TitanMLPNet + AdamW + post-init S0 weight & per-device RNG snapshot
  │     ├── kernel.coordinator.LocalParticipant("rank0")
  │     └── Step-gap barrier loop (commits/resets served strictly between training steps)
  └── AF_UNIX socket (rsittN) ──► Rank N worker (FSDP2 NCCL / CUDA / CPU toy; per-rank shard & witness)
```

## Step-Gap Barrier and `S0` Deterministic State Reset

Each rank worker runs a single-threaded queue-mode RPC loop (`UnixJsonServer(sock_path,
handler=None)`) around `StepGapTrainer.run_steps(...)`:

1. **Step-gap atomicity and fail-fast fan-out.** Training steps (`forward -> loss -> backward ->
   grad clip -> optimizer.step`) run to completion before any `prepare`, `validate`, `commit`,
   `abort`, `undo`, or `reset` RPC is served. In multi-rank fan-out (`_run_workload`), the host
   waits with `FIRST_EXCEPTION`, marks the channel dirty on any rank failure, and cancels sibling
   futures without stalling until `check_timeout`.
2. **Post-init FSDP2 `DTensor` & exact RNG `S0` snapshot/restore.** At the end of `_init_backend`
   (after model/weight and batch initialization), `StepGapTrainer` clones every parameter's local
   shard (`p.to_local().detach().clone()` for FSDP2 `DTensor`, or `p.detach().clone()` for plain
   tensors) and captures the exact post-init RNG states (`random.getstate()`,
   `torch.get_rng_state()`, and per-device `torch.cuda.get_rng_state(self.device)` on CUDA so rank
   0 never opens a stray CUDA context on rank 1's GPU). On `reset("state")` —
   and automatically after every commit (`post_commit_reset="state"`) and undo
   (`undo_resets=["state"]`) — every rank restores those exact `S0` weights in-place under
   `torch.no_grad()`, clears gradients and `AdamW` optimizer state, restores base LR and step
   counter, restores the exact saved Python/CPU/per-device CUDA RNG states (always reporting `"rng"`
   in `reset`), and flushes CUDA allocator cache (`torch.cuda.empty_cache()`).
3. **Zero eager GPU imports in the host.** Importing `adapter`, `adapter.lab`, `adapter.model_ops`,
   `adapter.optim_ops`, or `adapter.worker` imports only Python stdlib + `core/livelab`. `torch`
   and `torch.distributed.fsdp.fully_shard` load lazily inside worker runtime methods.
4. **Orphan, parallel-fresh, and shutdown safety.** Each worker runs in its own session
   (`start_new_session=True`), writes stderr to a managed temp file, checks
   `os.getppid() != parent_pid` at entry (before `StepGapTrainer` / NCCL init) and in its 1 s idle
   loop (without arming Linux `PR_SET_PDEATHSIG` on long-lived workers so transient pool threads
   cannot accidentally kill healthy workers on exit), waits briefly after a `shutdown` RPC before
   escalating to `stop_session`/`stop_tree`, and calls `torch.distributed.destroy_process_group()`
   cleanly on exit. Multi-rank `fresh_check` (`_run_fresh_parallel`) spawns per-rank
   `PackageOverlay` children armed with `PR_SET_PDEATHSIG(SIGTERM)` + immediate `parent_pid` check
   (safe because the parent thread blocks until all children exit), enforces an inner timeout 15 s
   below the outer subprocess timeout, and sends parallel `SIGTERM` across all sibling process
   groups on any rank failure or timeout.

## Restart Boundaries and Edit Routing

| Path | Mechanism | Scope | Cost (1× / 2× NVIDIA B200 FSDP2) |
|---|---|---|---|
| **Live step-gap swap** | `LocalParticipant` atomic `__code__` swap + per-rank AST witness + `S0` reset | Forward ops (`mlp_forward`), loss functions (`compute_loss`), gradient clipping (`clip_and_scale_grad`), LR/optimizer step helpers (`optimizer_step_scale`) | `world_size=1`: `0.059 s` install, `0.024 s` check (`0.083 s` total), `0.048 s` undo (`32.5×`); `world_size=2` NCCL: `0.255 s` median full cycle (`21.63×`; component medians: `0.101 s` install, `0.054 s` check, `0.099 s` undo) |
| **Refused at prepare** | Fail-closed static checks before touching live state | Syntax errors (`patch_invalid`), closure freevar changes (`closure_changed`), module-level constant edits (`outside_changes`), empty/comment-only diffs (`empty_change`), `__init__` constructor edits (`constructor_edit`) | Derived/estimated: `~0.01–0.05 s` per refusal (`c6_refusal` ran all 4 classes + base checks in `1.241 s` on 1× B200, `10.334 s` incl. boot on 2× B200) |
| **Cold respawn** | `reset("respawn")` tearing down workers and re-initializing NCCL + FSDP2 (or via pluggable `recovery_driver` / `weight_store` with `tp == ranks`) | Worker crash, hang, lost rank, external code tamper, or distributed topology / constructor changes | `world_size=1`: `2.689 ± 0.044 s` cold boot + 1st check; `world_size=2` NCCL: `4.830 s` boot / `5.515 s` boot + 1st check |

## Conformance Summary

- **CPU 2-rank (`ToyTorchTitanLab`)**: **11/11** conformance (`c0–c10`) in pure Python stdlib
  (no `torch` dependency), validating 2-rank commit/rollback, witness counting on both ranks,
  stochastic RNG repeatability, and all fault/refusal paths.
- **Real GPU FSDP2 (`TorchTitanLab`, 1× NVIDIA B200, `world_size=1`)**: **11/11** conformance
  (`c0–c10`) via `rsibox conformance torchtitan --real -- --mode fsdp2`, exercising real CUDA
  tensors, `nccl` process group, `fully_shard` `DTensor` parameters, `AdamW`, bit-exact `S0`
  rollback (`max_abs_loss_diff == 0.0`), and clean `0 MiB` VRAM release on exit.
- **Real Multi-GPU FSDP2 (`TorchTitanLab`, 2× NVIDIA B200, `world_size=2`, NCCL)**: **11/11**
  conformance (`c0–c10`, `239.00 s` `c0–c10` sum / `248.34 s` suite wall) across 2 NCCL ranks
  (rank 0 and rank 1) with real NCCL `fully_shard` collectives, parallel 2-rank `PackageOverlay`
  `fresh_check` parity (`c3_fresh_equivalence` `34.42 s`), multi-rank crash/hang/lost recovery
  (`c8_crash_and_hang` `104.95 s`), and bit-exact rollback (`max_abs_loss_diff == 0.0`).
