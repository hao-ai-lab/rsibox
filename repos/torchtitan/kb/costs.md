# Measured Costs

CPU and GPU timings are reported separately below. All GPU numbers were measured in our
2026-11-24 run on **1× NVIDIA B200** (180 GiB HBM3e), Python 3.12.11, PyTorch 2.14.0+cu130,
`torch.distributed` `nccl` backend with FSDP2 (`torch.distributed.fsdp.fully_shard`), `float32`
deterministic mode (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled), 3 repeats per measurement
(mean ± std), using the 4-layer `TitanMLPNet` (`hidden=256`, `ffn=512`, `batch_size=8`, 3 steps)
defined in [workload.md](workload.md).

## Real GPU: Cold Boot vs. Warm `TorchTitanLab` Live Swap (1× NVIDIA B200, FSDP2)

| Path | Breakdown | Measured Time (3× mean ± std) | Speedup vs. Cold |
|---|---|---|---|
| **Cold boot + 1st check** | `2.282 ± 0.046 s` worker import + NCCL + FSDP2 init + `0.368 ± 0.002 s` 1st check | **`2.650 ± 0.044 s`** | `1.0×` |
| **Fresh `PackageOverlay` parity** | Spawned isolated subprocess + FSDP2 init + 3 steps | `3.264 s` (bit-exact match) | — |
| **Warm steady base check** | `S0` restore + 3 FSDP2 training steps | `0.082 ± 0.035 s` | — |
| **Live edit (install + check)** | `0.0544 ± 0.0005 s` install + `0.0204 ± 0.0004 s` edited check | **`0.075 s`** | **`35.4×`** |
| **Live undo (bit-exact rollback)** | Journaled code undo + `S0` weights/AdamW/CUDA cache reset | **`0.0458 ± 0.0009 s`** | `max_abs_loss_diff = 0.0` |

## Real GPU Conformance (`c0–c10`, 1× NVIDIA B200, FSDP2)

Measured in our 2026-11-24 B200 run via
`rsibox conformance torchtitan --real -- --mode fsdp2 --world-size 1 --hidden 256 --ffn 512 --layers 4 --steps 3 --check-timeout 15.0`:

| Case | Result | Time (s) | Key Measurement |
|---|---:|---:|---|
| `c0_repeatable` | pass | `0.996` | 3 checks identical after `['weights_s0', 'optimizer_state', 'cuda_cache', 'step_counter']` resets |
| `c1_noop` | pass | `0.960` | Witness `compute_loss: 3`, bit-exact output |
| `c2_visible` | pass | `0.936` | Loss `1.02023852 -> 2.04047704`, witness `compute_loss: 3` |
| `c3_fresh_equivalence` | pass | `11.496` | Visible & 3-edit chain bit-exact vs fresh `PackageOverlay` FSDP2 subprocess |
| `c4_uninstall` | pass | `0.982` | Uninstall in `0.0445 s`, exact base code fingerprint restored |
| `c5_failed_install` | pass | `1.098` | `fail_apply` & `fail_commit` rolled back cleanly; 3-target/2-module control committed |
| `c6_refusal` | pass | `1.097` | All 4 refusal classes (`patch_invalid`, `closure_changed`, `outside_changes`, `empty_change`) |
| `c7_chain` | pass | `1.294` | 3 stacked edits undone step-by-step back to exact base losses & grad norms |
| `c8_crash_and_hang` | pass | `28.452` | Poisoned on `crash`/`hang`/`lost`/`tamper`; respawns `2.65–2.78 s` |
| `c9_soak` | pass | `3.874` | 20 cycles: install median **`0.0555 s`**, uninstall median **`0.0468 s`** |
| `c10_not_proven` | pass | `4.317` | `proven=False` (`zero_witness`) on startup-only `init_weight_scale`; fresh diverges |
| **Total** | **11/11** | **55.50 s** | **0 MiB VRAM leaked on exit** |

## CPU 2-Rank Conformance (`ToyTorchTitanLab`, Pure Python Stdlib)

Measured in our 2026-11-24 CPU run via `rsibox conformance torchtitan` (2 worker processes, no
`torch` or GPU required):

- **Result**: **11/11 PASS** (`c0–c10`), total `~6.3 s`
- **Per-rank witness**: `[2, 2]` across `rank0` and `rank1` for every swapped training op
- **`c9_soak` (20 cycles)**: install median `0.0155 s`, uninstall median `0.0052 s`
- **`c8_crash_and_hang` respawns**: `0.177–0.319 s` across `crash`, `hang`, `lost`, and `tamper`
