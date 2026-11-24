# PyTorch Native Distributed Training (`TorchTitan` / FSDP2)

The `torchtitan` live-lab adapter brings RSIBox step-gap hot-reload, execution proof, and bit-exact
`S0` rollback to PyTorch Native FSDP2 training loops (`torch.distributed.fsdp.fully_shard` +
`AdamW`).

**Current scope:** Real-GPU verification exercises PyTorch `fully_shard` (FSDP2) + `nccl` at
`world_size=1` over a self-contained `TitanMLPNet` training loop (the adapter does not yet import
upstream `pytorch/torchtitan` or support multi-GPU `world_size > 1` FSDP2, which fails closed with
`ValueError`). Multi-rank orchestration (`world_size=2`) is exercised in pure-Python CPU toy mode
(`ToyTorchTitanLab`).

| Folder | Holds |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Process tree, FSDP2 `DTensor` sharded parameters, step-gap barrier, and restart boundaries |
| [kb/costs.md](kb/costs.md) | Measured CPU 2-rank and 1× NVIDIA B200 FSDP2 cold-boot, live-swap, and rollback costs |
| [kb/workload.md](kb/workload.md) | Deterministic `TitanMLPNet` FSDP2 workload F(x) and correctness gate |
| [kb/lessons.md](kb/lessons.md) | Engine-specific FSDP2 `DTensor`, saved-RNG, and step-gap lessons |
| [examples/live-step-swap/README.md](examples/live-step-swap/README.md) | Runnable walkthrough of a live training-op swap and bit-exact rollback |
| `livelab/adapter/` | `TorchTitanLab`, `ToyTorchTitanLab`, `StepGapTrainer` worker, hot-swappable ops, and tests |

## Onboarding LOC Scorecard

| Layer / File | Owner | Non-Test Python LOC | Test Python LOC | Reused `core/livelab/` Primitives |
|---|---|---:|---:|---|
| `livelab/adapter/__init__.py` | `repos/torchtitan` | `9` | `0` | `RSIBOX_LIVELAB` path bootstrap |
| `livelab/adapter/model_ops.py` | `repos/torchtitan` | `76` | `0` | Pure-Python + PyTorch FSDP2 `fully_shard` forward/loss ops |
| `livelab/adapter/optim_ops.py` | `repos/torchtitan` | `47` | `0` | Hot-swappable grad-clip and LR-scale step helpers |
| `livelab/adapter/worker.py` | `repos/torchtitan` | `253` | `0` | `LocalParticipant`, `UnixJsonServer`, `inject_participant_fault`, `S0` reset |
| `livelab/adapter/lab.py` | `repos/torchtitan` | `212` | `0` | `BaseLab`, `RecaptureCoordinator`, `CallParticipant`, `PackageOverlay` |
| `livelab/adapter/test_torchtitan_lab.py` | `repos/torchtitan` | `0` | `243` | Full `c0–c10` conformance + FSDP2/training-loop/RNG golden tests |
| **Adapter boundary total** | **`repos/torchtitan/livelab/adapter/`** | **`597`** | **`243`** | **`0` cross-`repos/` imports; `<= 600` non-test LOC budget** |
| Shared runtime (reused, 0 new LOC) | `core/livelab/` | `0` | `0` | `kernel/*`, `rsibox/*`, `conformance/*` |
