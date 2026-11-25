# NVIDIA NeMo-RL (`NVIDIA-NeMo/RL` — AutoModel FSDP2 + SGLang GRPO)

The `nemo-rl` live-lab adapter brings RSIBox step-gap hot-reload, AST execution proof, and
coupled `S0` rollback to `NVIDIA-NeMo/RL` GRPO post-training (`AutoModel` FSDP2 policy trainer +
colocated `SGLang` generation workers + GRPO coordinator).

**Current status:**
- **CPU 3-participant conformance (`ToyNemoRlLab`):** **11/11 PASS** (`c0–c10`, `9/9` golden unit
  tests with `-W error::ResourceWarning`), exercising a 3-process GRPO loop (`coordinator0` +
  `trainer0` + `rollout0`) over `AF_UNIX` JSON channels and `core/livelab/ray_participants.py`.
- **Real GPU (`NemoRlLab --real`):** Adapter scaffolding and Ray participant definitions
  (`NemoRlRayParticipant`, `serve_nemo_rl_actor`, `discover_nemo_rl_actors`, `snapshot_automodel_s0`,
  `restore_automodel_s0`, `coupled_nemo_rl_reset`) are implemented with fail-closed lazy dependency
  guards (`require_nemo_rl_deps()`); **1×/2× NVIDIA B200 NeMo-RL GPU benchmarks and `--real`
  conformance are `pending hardware/env staging`**.

| Path | Holds |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 3-participant GRPO topology, Ray actor reach (`ray_participants.py`), step-gap barrier, and coupled `S0` reset |
| [kb/costs.md](kb/costs.md) | Measured CPU `11/11` conformance latencies; GPU section marked `pending hardware/env staging` |
| `livelab/livelab.json` | Unified CLI entrypoints (`ToyNemoRlLab` default, `NemoRlLab` for `--real`) |
| `livelab/adapter/` | `NemoRlLab`, `ToyNemoRlLab`, 3-role GRPO worker, `NemoRlRayParticipant`, hot-swappable ops, and tests |
