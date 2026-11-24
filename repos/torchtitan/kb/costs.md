# Measured Costs

CPU and GPU timings are reported separately below. All GPU numbers were measured in our
2026-11-24 run on **1× NVIDIA B200** (180 GiB HBM3e), Python 3.12.11, PyTorch 2.14.0+cu130,
`torch.distributed` `nccl` backend with FSDP2 (`torch.distributed.fsdp.fully_shard`,
`world_size=1`), `float32` deterministic mode (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled),
using the self-contained 4-layer synthetic `TitanMLPNet` (`hidden=256`, `ffn=512`, `layers=4`,
`batch_size=8`, `steps=3`) defined in [workload.md](workload.md) (upstream `pytorch/torchtitan`
is not yet imported).

## Real GPU: Cold Boot vs. Warm `TorchTitanLab` Live Swap (1× NVIDIA B200, FSDP2 `world_size=1`)

Measured over 3 repeats (mean ± std) at `batch_size=8`, `steps=3`:

| Path | Breakdown | Measured Time (3× mean ± std) | Speedup vs. Cold |
|---|---|---|---|
| **Cold boot + 1st check** | `2.304 ± 0.047 s` worker import + NCCL + FSDP2 init + `0.384 ± 0.011 s` 1st check | **`2.689 ± 0.044 s`** | `1.0×` |
| **Fresh `PackageOverlay` parity** | Spawned isolated subprocess + FSDP2 init + 3 steps | `3.221 s` (bit-exact match) | — |
| **Warm steady base check** | `S0` restore + 3 FSDP2 training steps | `0.078 ± 0.025 s` | — |
| **Live edit (install + check)** | `0.0593 ± 0.0080 s` install + `0.0235 ± 0.0051 s` edited check | **`0.083 s`** | **`32.5×`** |
| **Live undo (bit-exact rollback)** | Journaled code undo + `S0` weights/AdamW/RNG/CUDA cache reset | **`0.0482 ± 0.0010 s`** | `max_abs_loss_diff = 0.0` |

## Real GPU Conformance (`c0–c10`, 1× NVIDIA B200, FSDP2 `world_size=1`)

Measured in our 2026-11-24 B200 run via
`rsibox conformance torchtitan --real -- --mode fsdp2 --world-size 1 --hidden 256 --ffn 512 --layers 4 --steps 3 --check-timeout 15.0`
(default `--batch-size 8`, identical to the 3× benchmark above):

| Case | Result | Time (s) | Key Measurement |
|---|---:|---:|---|
| `c0_repeatable` | pass | `1.051` | 3 checks identical after `['weights_s0', 'optimizer_state', 'rng', 'cuda_cache', 'step_counter']` resets |
| `c1_noop` | pass | `1.102` | Witness `adapter.model_ops:compute_loss:35` = `3`, bit-exact output |
| `c2_visible` | pass | `1.175` | Loss `1.08151245 -> 2.1630249`, witness `adapter.model_ops:compute_loss:35` = `3` |
| `c3_fresh_equivalence` | pass | `11.573` | Visible & 3-edit chain bit-exact vs fresh `PackageOverlay` FSDP2 subprocess |
| `c4_uninstall` | pass | `1.196` | Uninstall in `0.0442 s`, exact base code fingerprint restored |
| `c5_failed_install` | pass | `1.447` | `fail_apply` & `fail_commit` rolled back cleanly; 3-target/2-module control committed |
| `c6_refusal` | pass | `1.241` | All 4 refusal classes (`patch_invalid`, `closure_changed`, `outside_changes`, `empty_change`) |
| `c7_chain` | pass | `1.618` | 3 stacked edits undone step-by-step back to exact base losses & grad norms |
| `c8_crash_and_hang` | pass | `28.242` | Poisoned on `crash`/`hang`/`lost`/`tamper`; respawns `2.58–2.80 s` |
| `c9_soak` | pass | `3.887` | 20 cycles: install median **`0.0541 s`**, uninstall median **`0.0458 s`** |
| `c10_not_proven` | pass | `4.595` | `proven=False` (`zero_witness`) on startup-only `init_weight_scale`; fresh diverges |
| **Total** | **11/11** | **57.13 s** | **0 MiB VRAM leaked on exit** |

## CPU 2-Rank Conformance (`ToyTorchTitanLab`, Pure Python Stdlib)

Measured in our 2026-11-24 CPU run via `rsibox conformance torchtitan` (2 worker processes, no
`torch` or GPU required):

- **Result**: **11/11 PASS** (`c0–c10`), total `4.66 s`
- **Per-rank witness**: `[2, 2]` across `rank0` and `rank1` for `adapter.model_ops:compute_loss:35`
- **`c9_soak` (20 cycles)**: install median `0.0142 s`, uninstall median `0.0045 s`
- **`c8_crash_and_hang` respawns**: `0.097–0.320 s` across `crash`, `hang`, `lost`, and `tamper`
