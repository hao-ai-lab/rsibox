# TorchTitan / FSDP2 Live-Lab Architecture

## Scope & Process Tree

`TorchTitanLab` hosts isolated worker subprocesses over per-rank `AF_UNIX` JSON channels
(`rsibox.ipc.UnixJsonServer`). Today it supports:

- **Real GPU FSDP2 (`mode="fsdp2"`, `world_size=1`)**: Real CUDA PyTorch `nn.Module`
  (`TitanMLPNet`) sharded with `torch.distributed.fsdp.fully_shard` over an `nccl` process group
  plus `AdamW`. Setting `mode="fsdp2"` with `world_size > 1` fails closed with `ValueError`
  (multi-GPU FSDP2 `fresh_check` and multi-rank collective orchestration are not yet wired).
  The adapter does not yet import upstream `pytorch/torchtitan`.
- **Single-process eager PyTorch (`mode="cuda"` or `mode="torch_cpu"`)**: Same `TitanMLPNet`
  training loop without `fully_shard` or `torch.distributed`.
- **Multi-rank CPU toy (`ToyTorchTitanLab`, `mode="cpu"`, `world_size=2`)**: Pure-Python
  fake-distributed 2-worker topology exercising multi-rank atomic commits, per-rank witnesses,
  and crash/hang fail-fast fan-out in zero-dependency CPU CI.

```text
Host process (TorchTitanLab, BaseLab + RecaptureCoordinator)
  ├── AF_UNIX socket (rsitt0) ──► Rank 0 worker (python -m adapter.worker <spec.json>, own session)
  │     ├── Backend: nccl + fully_shard FSDP2 DTensor (world_size=1) | eager PyTorch | CPU toy
  │     ├── TitanMLPNet + AdamW + post-init S0 weight & exact RNG snapshot
  │     ├── kernel.coordinator.LocalParticipant("rank0")
  │     └── Step-gap barrier loop (commits/resets served strictly between training steps)
  └── AF_UNIX socket (rsittN) ──► Rank N worker (CPU toy multi-rank; per-rank data shard & witness)
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
   `torch.get_rng_state()`, and `torch.cuda.get_rng_state_all()` on CUDA). On `reset("state")` —
   and automatically after every commit (`post_commit_reset="state"`) and undo
   (`undo_resets=["state"]`) — every rank restores those exact `S0` weights in-place under
   `torch.no_grad()`, clears gradients and `AdamW` optimizer state, restores base LR and step
   counter, restores the exact saved Python/CPU/CUDA RNG states (always reporting `"rng"` in
   `reset`), and flushes CUDA allocator cache (`torch.cuda.empty_cache()`).
3. **Zero eager GPU imports in the host.** Importing `adapter`, `adapter.lab`, `adapter.model_ops`,
   `adapter.optim_ops`, or `adapter.worker` imports only Python stdlib + `core/livelab`. `torch`
   and `torch.distributed.fsdp.fully_shard` load lazily inside worker runtime methods.
4. **Orphan and shutdown safety.** Each worker runs in its own session (`start_new_session=True`),
   writes stderr to a managed temp file, checks `os.getppid() != parent_pid` in its 1 s idle loop
   so orphaned idle workers exit automatically, waits briefly after a `shutdown` RPC before
   escalating to `stop_session`/`stop_tree`, and calls `torch.distributed.destroy_process_group()`
   cleanly on exit.

## Restart Boundaries and Edit Routing

| Path | Mechanism | Scope | Cost (1× NVIDIA B200, FSDP2 `world_size=1`) |
|---|---|---|---|
| **Live step-gap swap** | `LocalParticipant` atomic `__code__` swap + per-rank AST witness + `S0` reset | Forward ops (`mlp_forward`), loss functions (`compute_loss`), gradient clipping (`clip_and_scale_grad`), LR/optimizer step helpers (`optimizer_step_scale`) | Measured: `0.059 s` install, `0.024 s` check (`0.083 s` total), `0.048 s` undo |
| **Refused at prepare** | Fail-closed static checks before touching live state | Syntax errors (`patch_invalid`), closure freevar changes (`closure_changed`), module-level constant edits (`outside_changes`), empty/comment-only diffs (`empty_change`), `__init__` constructor edits (`constructor_edit`) | Derived/estimated: `~0.01–0.05 s` per refusal (`c6_refusal` ran all 4 classes + base checks in `1.241 s` total) |
| **Cold respawn** | `reset("respawn")` tearing down worker and re-initializing NCCL + FSDP2 | Worker crash, hang, lost rank, external code tamper, or distributed topology / constructor changes | Measured: `2.689 ± 0.044 s` cold boot + 1st check (`2.58–2.80 s` `c8` respawns) |

## Conformance Summary

- **CPU 2-rank (`ToyTorchTitanLab`)**: **11/11** conformance (`c0–c10`) in pure Python stdlib
  (no `torch` dependency), validating 2-rank commit/rollback, witness counting on both ranks,
  stochastic RNG repeatability, and all fault/refusal paths.
- **Real GPU FSDP2 (`TorchTitanLab`, 1× NVIDIA B200, `world_size=1`)**: **11/11** conformance
  (`c0–c10`) via `rsibox conformance torchtitan --real -- --mode fsdp2`, exercising real CUDA
  tensors, `nccl` process group, `fully_shard` `DTensor` parameters, `AdamW`, bit-exact `S0`
  rollback (`max_abs_loss_diff == 0.0`), and clean `0 MiB` VRAM release on exit.
