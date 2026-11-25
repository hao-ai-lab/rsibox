# PyTorch Native Distributed Training (`TorchTitan` / FSDP2)

The `torchtitan` live-lab adapter brings RSIBox step-gap hot-reload, execution proof, and bit-exact
`S0` rollback to PyTorch Native FSDP2 training loops (`torch.distributed.fsdp.fully_shard` +
`AdamW`).

**Current scope:** Real-GPU verification exercises PyTorch `fully_shard` (FSDP2) + `nccl` at both
`world_size=1` (`1× NVIDIA B200`) and `world_size=2` (`2× NVIDIA B200` over real NCCL collectives
with parallel 2-rank `fresh_check` parity) over a self-contained `TitanMLPNet` training loop (the
adapter does not yet import upstream `pytorch/torchtitan`). Multi-rank orchestration
(`world_size=2`) is also exercised in pure-Python CPU toy mode (`ToyTorchTitanLab`).

| Folder | Holds |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Process tree, FSDP2 `DTensor` sharded parameters, step-gap barrier, and restart boundaries |
| [kb/costs.md](kb/costs.md) | Measured CPU 2-rank, 1× B200 (`world_size=1`), and 2× B200 (`world_size=2` NCCL) FSDP2 costs |
| [kb/workload.md](kb/workload.md) | Deterministic `TitanMLPNet` FSDP2 workload F(x) and correctness gate |
| [kb/lessons.md](kb/lessons.md) | Engine-specific FSDP2 `DTensor`, per-device RNG, parallel `fresh_check`, and step-gap lessons |
| [examples/live-step-swap/README.md](examples/live-step-swap/README.md) | Runnable walkthrough of a live training-op swap and bit-exact rollback |
| `livelab/adapter/` | `TorchTitanLab`, `ToyTorchTitanLab`, `StepGapTrainer` worker, hot-swappable ops, and tests |

## Adapter LOC Scorecard

| Layer / File | Owner | Non-Test Python LOC | Test Python LOC | Reused `core/livelab/` Primitives |
|---|---|---:|---:|---|
| `livelab/adapter/__init__.py` | `repos/torchtitan` | `9` | `0` | `RSIBOX_LIVELAB` path bootstrap |
| `livelab/adapter/model_ops.py` | `repos/torchtitan` | `76` | `0` | Pure-Python + PyTorch FSDP2 `fully_shard` forward/loss ops |
| `livelab/adapter/optim_ops.py` | `repos/torchtitan` | `47` | `0` | Hot-swappable grad-clip and LR-scale step helpers |
| `livelab/adapter/worker.py` | `repos/torchtitan` | `425` | `0` | `LocalParticipant`, `UnixJsonServer`, `inject_participant_fault`, parallel `fresh_check`, `S0` reset |
| `livelab/adapter/lab.py` | `repos/torchtitan` | `234` | `0` | `BaseLab`, `RecaptureCoordinator`, `CallParticipant`, `PackageOverlay`, `weight_store`/`recovery_driver` |
| `livelab/adapter/test_torchtitan_lab.py` | `repos/torchtitan` | `0` | `400` | Full `c0–c10` conformance + 1-rank/2-rank FSDP2, parallel fresh, and lifecycle golden tests |
| **Adapter boundary total** | **`repos/torchtitan/livelab/adapter/`** | **`791`** | **`400`** | **`0` cross-`repos/` imports** |
| Shared runtime (reused, 0 new LOC) | `core/livelab/` | `0` | `0` | `kernel/*`, `rsibox/*`, `conformance/*` |
