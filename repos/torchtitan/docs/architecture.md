# TorchTitan / FSDP2 Live-Lab Architecture

## Process Tree

PyTorch Native Distributed Training launches one worker process per rank (mirroring `torchrun`
`RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `MASTER_PORT` topology), each owning its local FSDP2
parameter shards, optimizer moments, and CUDA stream. `TorchTitanLab` manages the rank group over
per-rank `AF_UNIX` JSON channels (`rsibox.ipc.UnixJsonServer`):

```text
Host process (TorchTitanLab, BaseLab + RecaptureCoordinator)
  ├── AF_UNIX socket (rsitt0) ──► Rank 0 worker (python -m adapter.worker <spec.json>, own session)
  │     ├── torch.distributed process group (nccl on CUDA / gloo on torch_cpu / fake-dist on CPU)
  │     ├── TitanMLPNet wrapped with torch.distributed.fsdp.fully_shard (FSDP2 DTensor shards)
  │     ├── AdamW optimizer + deterministic S0 local-tensor snapshot
  │     ├── kernel.coordinator.LocalParticipant("rank0")
  │     └── Step-gap barrier loop (commits/resets served strictly between training steps)
  └── AF_UNIX socket (rsittN) ──► Rank N worker (same layout, per-rank data shard & witness)
```

## Step-Gap Barrier and `S0` Deterministic State Reset

Each rank worker runs a single-threaded queue-mode RPC loop (`UnixJsonServer(sock_path,
handler=None)`) around `StepGapTrainer.run_steps(...)`:

1. **Step-gap atomicity across ranks.** Training steps (`forward -> loss -> backward -> grad clip
   -> optimizer.step`) execute cleanly to completion before any `prepare`, `validate`, `commit`,
   `abort`, `undo`, or `reset` RPC is processed. `RecaptureCoordinator` commits across every rank
   in one coordinated gap so no training step ever mixes code versions across ranks.
2. **FSDP2 `DTensor` `S0` snapshot & restore.** At worker startup, `StepGapTrainer` clones each
   parameter's local shard (`p.to_local().detach().clone()` for FSDP2 `DTensor`, or
   `p.detach().clone()` for plain tensors). On `reset("state")` — and automatically after every
   commit (`post_commit_reset="state"`) and undo (`undo_resets=["state"]`) — every rank restores
   its `S0` local weight shards in-place under `torch.no_grad()`, zeroes gradients, clears
   `AdamW` optimizer state buffers, restores base LR and step counter, re-seeds Python/PyTorch
   CPU/CUDA RNG, and flushes CUDA allocator cache (`torch.cuda.empty_cache()`).
3. **Zero eager GPU imports in the host.** Importing `adapter`, `adapter.lab`, `adapter.model_ops`,
   `adapter.optim_ops`, or `adapter.worker` imports only Python stdlib + `core/livelab`. `torch`
   and `torch.distributed.fsdp.fully_shard` load lazily only when a PyTorch/CUDA worker starts.
4. **Orphan and pipe cleanup safety.** Each worker runs in its own session (`start_new_session=True`),
   writes stderr to a managed temp file, exits automatically if its parent PID disappears, and tears
   down `torch.distributed.destroy_process_group()` cleanly on shutdown.

## Restart Boundaries and Edit Routing

| Path | Mechanism | Scope | Measured Cost (1× NVIDIA B200, FSDP2) |
|---|---|---|---|
| **Live step-gap swap** | `LocalParticipant` atomic `__code__` swap + per-rank AST witness + `S0` reset | Forward ops (`mlp_forward`), loss functions (`compute_loss`), gradient clipping (`clip_and_scale_grad`), LR/optimizer step helpers (`optimizer_step_scale`) | `0.054 s` install, `0.020 s` check (`0.075 s` total), `0.046 s` undo |
| **Refused at prepare** | Fail-closed static checks before touching live state | Syntax errors (`patch_invalid`), closure freevar changes (`closure_changed`), module-level constant edits (`outside_changes`), empty/comment-only diffs (`empty_change`), `__init__` constructor edits (`constructor_edit`) | `~0.01–0.05 s` refusal, base state & fingerprint untouched |
| **Cold respawn** | `reset("respawn")` tearing down worker group and re-initializing NCCL + FSDP2 | Worker crash, hang, lost rank, external code tamper, or distributed topology / constructor changes | `2.650 ± 0.044 s` cold boot + 1st check (`2.65–2.78 s` `c8` respawn) |

## Conformance Summary

- **CPU 2-rank (`ToyTorchTitanLab`)**: **11/11** conformance (`c0–c10`) in pure Python stdlib
  (no `torch` dependency), validating multi-rank commit/rollback, witness counting on both ranks,
  and all fault/refusal paths.
- **Real GPU FSDP2 (`TorchTitanLab`, 1× NVIDIA B200)**: **11/11** conformance (`c0–c10`) via
  `rsibox conformance torchtitan --real -- --mode fsdp2`, exercising real CUDA tensors,
  `nccl` process group, `fully_shard` `DTensor` parameters, `AdamW`, bit-exact `S0` rollback
  (`max_abs_loss_diff == 0.0`), and clean `0 MiB` VRAM release on exit.
