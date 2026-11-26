# NVIDIA NeMo-RL (`NVIDIA-NeMo/RL` — AutoModel FSDP2 + SGLang GRPO)

The `nemo-rl` live-lab adapter brings RSIBox step-gap hot-reload, AST execution proof, and
coupled `S0` rollback to `NVIDIA-NeMo/RL` GRPO post-training (`AutoModel` FSDP2 policy trainer +
colocated `SGLang` generation workers + GRPO coordinator).

**Current status:**
- **CPU 3-participant conformance (`ToyNemoRlLab`):** **11/11 PASS** (`c0–c10`, `14/14` unit tests and `11/11` conformance
  tests with `-W error::ResourceWarning`), exercising a 3-process GRPO loop (`coordinator0` +
  `trainer0` + `rollout0`) over `AF_UNIX` JSON channels and `core/livelab/ray_participants.py`.
- **Real GPU (`NemoRlLab --real`):** Implemented in [ray_driver.py](livelab/adapter/ray_driver.py) over real `NVIDIA-NeMo/RL` (`AutomodelPolicyWorker` FSDP2 + `SGLangGeneration` + `SGLangColocatedWeightSynchronizer` + `ClippedPGLossFn` over Ray) and verified on `1×` (`GPU 6`) and `2×` (`GPU 6,7`) NVIDIA B200 with `gemma-3-1b-it` (multi-shape cold/warm/S0/live-vs-fresh/uninstall parity + `c8_crash_and_hang` after final edit; `c0–c7, c9–c10` passed immediately prior, full single-run `11/11` rerun interrupted by external contention on assigned GPUs).

| Path | Holds |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 3-participant GRPO topology, Ray actor reach (`ray_participants.py`), step-gap barrier, and coupled `S0` reset |
| [kb/costs.md](kb/costs.md) | Measured CPU `11/11` conformance and `1×/2×` NVIDIA B200 GPU latencies |
| `livelab/livelab.json` | Unified CLI entrypoints (`ToyNemoRlLab` default, `NemoRlLab` for `--real`) |
| `livelab/adapter/` | `NemoRlLab`, `ToyNemoRlLab`, `ray_driver.py`, 3-role GRPO worker, `NemoRlRayParticipant`, hot-swappable ops, and tests |
