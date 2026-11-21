# Costs (`miles` / `slime`)

Baseline startup and step-phase costs for `miles` (`radixark/miles` `0.2.0`, `THUDM/slime`
architecture) and measured live-lab adapter latencies. Every number names its measurement source
and hardware tier.

## 1. Live-Lab Adapter & Conformance Overheads (Measured on CPU, 3 Spawned Actor Processes)

Source: `core/livelab/conformance` (`c0`–`c10`) against `adapter.toy_lab:ToyMilesLab` (1 spawned
`RolloutManager` process + 2 spawned `MegatronTrainRayActor` processes over pipes, Python 3.12):

| Operation | Median Latency | Source Case | What Runs |
|---|---:|---|---|
| `install` (`R0 swap`, 1–3 functions across actors) | `0.0029 s` (`2.9 ms`) | `c9_soak` (`N = 20` cycles) | `build_plan` + `unloaded` check + Phase 1 stage + Phase 2 `StepGapCoordinator` commit + fingerprint verify |
| `uninstall` (exact code object & globals restore) | `0.0007 s` (`0.7 ms`) | `c9_soak` (`N = 20` cycles) | Reverse journal undo + `code_epoch` advance + base fingerprint check |
| `check` (coupled `S0` reset + rollout + 2-rank train + witness proof) | `~0.016 s` | `c0_repeatable` | `CoupledWeightReset.restore_and_sync` + `RolloutManager.generate_batch` + `EpochDataBuffer` + 2-rank `train_step` + `prove` |
| `reset("respawn")` (3 spawned actor processes) | `0.093–0.109 s` | `c8_crash_and_hang` | Process teardown + fresh spawn + `#actor` view prime + `Session.recovered` |
| Full `11/11` conformance suite (`c0`–`c10`) | `~1.69 s` | `python3 -m conformance` | All 11 positive, refusal, rollback, soak, and fault-recovery cases |

## 2. Full RL Startup vs. Live Swap Opportunity (`4x GB200`, `aarch64`, CUDA 13.0)

Source: our baseline `Qwen3-0.6B` GSM8K GRPO smoke runs on `4x GB200` (`aarch64`, CUDA 13.0,
Megatron-LM trainer + SGLang rollout server + `sgl-router`, shared Ray worker allocation).
Because `radixark/miles` shares the `THUDM/slime` Ray placement-group, SGLang server launch, and
Megatron initialization pipeline (`train.py:25-69`), cold restarts pay every phase below before the
first training step completes:

| Startup / Loop Phase | Typical Wall Time (`Qwen3-0.6B`, `4x GB200`) | Avoided by `R0` Live Swap? | Notes |
|---|---:|---|---|
| Ray placement group + `InfoActor` GPU bundle probe | `3–6 s` | Yes | `miles/ray/placement_group.py:_create_placement_group` |
| SGLang server launch + CUDA graph capture (`sgl-router` ready) | `28–45 s` | Yes | `miles/backends/sglang_utils/sglang_engine.py` |
| `MegatronTrainRayActor.init` (`import megatron`, distributed groups, model & Adam build) | `25–40 s` | Yes | `miles/backends/megatron_utils/actor.py:131-308` |
| Initial `update_weights` (trainer $\to$ SGLang P2P/NCCL sync) | `0.8–2.0 s` | Yes (reused on `S0` reset) | `train.py:57` |
| **Cold start to first GRPO step ready** | **`~60–95 s`** (small model) / **`120–210 s`** (`8B–32B`) | **Yes** | Paid on every candidate edit without a live lab |
| Steady synchronous GRPO step (`rollout` + `train` + `update_weights`) | `~2.5–8.0 s` | Workload step | Dominated by SGLang rollout generation length |

## 3. Paired 2× B200 Receipt (`slime` / `MilesStepGate` + `miles 0.2.0` `PR #162`)

Source: our 2026-11-19 paired live-swap run on `2× B200` (`Qwen3-0.6B` GSM8K GRPO, `1` SGLang rollout + `1` Megatron trainer via `MilesStepGate`, `slime` `miles`-lineage trainer stack; `fastrsi_megatron.state.Snapshot` `S0` over `3,576,299,520 B` / `3.33 GiB`).

| Measurement | Wall Time | Breakdown / Notes |
|---|---:|---|
| Fresh cold restart $\to$ first GRPO step verdict | **`83.87 s`** (`83.8696 s`) | `import_ray 10.5145 s` + `pg 5.8218 s` + `rollout_boot 0.0404 s` + `trainer_boot 57.0017 s` + `init_weight_sync 1.9714 s` (`boot_to_ready 75.3498 s`) + `first_step 8.5198 s` (`gen 1.7481 s` + `train 6.6442 s` + `sync 0.1275 s`); `subproc_wall_s 87.3501 s` |
| Coupled `S0` capture (`Snapshot`: weights, master weights, optimizer scalars, scheduler, RNG) | `1.8499 s` | `3,576,299,520 B` (`3.33 GiB`) pinned host copy (`take_s 1.8434 s`) |
| Live `get_grpo_returns` swap (`install`) | **`0.0570 s`** (`57.0 ms`) | Witnessed `trainer0`: `slime.utils.ppo_utils:get_grpo_returns:201: 1` |
| Coupled `S0` restore + rollout `update_weights` | `0.2027 s` | Restores `Snapshot` + zeros post-`S0` Adam state + syncs to SGLang |
| Live full GRPO step (`install` + `S0` restore + rollout `0.3357 s` + train/sync `0.56 s`) | **`1.1555 s`** (`0.9528 s` w/o restore) | **`72.6x`** faster than cold restart (`83.8696 / 1.1555`) |
| Recorded-rollout trainer replay (`install` + `S0` restore + train/sync `0.4043 s`, skip rollout) | **`0.6641 s`** | **`126.3x`** faster than cold restart (`83.8696 / 0.6641`) |
| `uninstall` + post-undo `S0` replay bit-exact check | **`0.0159 s`** (`15.9 ms`) | Base `0b23fd915b3ecbfc` (`13217973.159927`, unchanged on 0-reward base batch) $\to$ edited `b90017963cbcc302` (`13217979.694672`) $\to$ restored `0b23fd915b3ecbfc` (`bit_exact_undo_on_same_rollout: true`) |
| Separate `miles 0.2.0` multi-actor live swap (`2× GB200`, `PR #162`) | **`0.069 s`** (`69.22 ms`) | Multi-actor Ray swap + `EpochDataBuffer` `code_epoch` drain + coupled `S0` weight sync |

## 4. Non-Zero-Gradient `parity_rm` Receipt & `RealSlimeMilesLab` Conformance (`2× B200`, `2026-11-19`)

Source: `2× B200` `Qwen3-0.6B` GSM8K GRPO (`1` SGLang rollout + `1` Megatron trainer via `MilesStepGate` + `fastrsi_megatron.state.Snapshot` `S0`) using `parity_rm` (non-zero reward variance so optimizer steps change weights on every base and edited check).

- **`slime` `weights_backuper` finding:** `MegatronTrainRayActor` defaults to `enable_weights_backuper=True`, backing up updated weights at the end of `train_actor()` and calling `self._switch_model("actor")` at the start of the next step (and reading `weights_backuper.get("actor")` inside `update_weights()`). Coupled `S0` restore must therefore refresh `actor.weights_backuper.backup("actor")` right after `MState.zero_new_state(...)` so subsequent replay steps train from `S0` rather than stale post-step weights.

| Measurement | Wall Time | Breakdown / Notes |
|---|---:|---|
| Fresh cold restart $\to$ first non-zero-grad step verdict | **`88.76 s`** (`88.7571 s`) | `import_ray 11.789 s` + `pg 6.2286 s` + `rollout_boot 0.0430 s` + `trainer_boot 59.1861 s` + `init_weight_sync 2.0441 s` (`boot_to_ready 79.2909 s`) + `first_step 9.4662 s` (`gen 1.8618 s` + `train 7.4802 s` + `sync 0.1243 s`) |
| `Snapshot S0` capture | `1.5504 s` | `3,576,299,520 B` (`3.33 GiB`) pinned host copy (`take_s 1.5470 s`) |
| Base weight step shift from `S0` (`weights_changed_from_s0: true`) | `8.2871 s` (`1.7628 s` gen + `6.5243 s` train/sync) | `w_s0 0b23fd915b3ecbfc` (`13217973.159927`) $\to$ `w_base_after 0d9e38119a154108` (`13217978.506348`) |
| Live `get_grpo_returns` swap + `S0`-restored trainer replay | **`0.6972 s`** (`127.3×` vs fresh) | `0.0629 s` (`62.9 ms`) install (`trainer0` witness `1`) + `0.6343 s` `S0`-restored train + weight sync (`w_edit fcb55d6cec4f9268`, `13217978.401443`) |
| `uninstall` + post-undo same-rollout bit-exact check | **`0.0159 s`** (`15.9 ms`) | `w_post_undo 0d9e38119a154108` (`13217978.506348`), **`bit_exact_undo_on_same_rollout: true`** |
| `RealSlimeMilesLab` conformance (`c0, c1, c2, c4, c5, c6, c7, c9`) | **`8/8 pass`** | `c4` undo `0.014808 s`; `c7` 3-edit chain & unwind; `c9` 20-cycle soak `0.0616 s` install / `0.0166 s` undo; `0 MiB × 2` leftover |

## 5. Full `11/11` `RealSlime11Lab` Conformance (`2× B200`, `2026-11-19`)

Source: `2× B200` `Qwen3-0.6B` GSM8K GRPO (`parity_rm.parity_rm`, `1` SGLang rollout + `1` Megatron trainer via `MilesStepGate` + `MState` `S0` + `weights_backuper.backup("actor")`) with Ray `2.54.0` C++ OTel/metrics disabled (`RAY_enable_open_telemetry=0`, `RAY_enable_metrics_collection=0`, `_system_config`):

- **Baseline fresh & `S0` full step:** fresh cold subprocess `82.1291 s` (`73.7571 s` boot + `8.372 s` step 1, `subproc_wall_s 86.079 s`, output `d203e3461c786d22`); live `S0`-restored full GRPO step `2.7481 s` (`0.3512 s` generate + `2.3969 s` train/sync, `w_s0 0b23fd915b3ecbfc` $\to$ `w_after abed0e1d07924c50`, `weights_changed_from_s0: true`).
- **Full `11/11` conformance (`113` receipts [`43` txn / `38` proof / `32` undo], `median edit_to_commit_s 0.0417 s` / `undo_s 0.0052 s`, `0 MiB × 2` teardown):**
  - `c0_repeatable` (`2.824 s`, digest `da874eca217a0229`), `c1_noop` (`1.748 s`, witness `[1]`), `c2_visible` (`1.762 s`, `da874eca217a0229 -> c6aaa489530a5116`, witness `[1]`).
  - `c3_fresh_equivalence` (`339.647 s`, bit-exact `live == fresh` on both single visible `c6aaa489530a5116` and 3-edit chain `ee45dec3e1bf1c1a`).
  - `c4_uninstall` (`4.855 s`, undo `0.017711 s`), `c5_failed_install` (`4.047 s`), `c6_refusal` (`4.512 s`), `c7_chain` (`5.839 s`).
  - `c8_crash_and_hang` (`238.428 s`, respawns: crash `50.141 s`, hang `50.659 s`, lost `50.193 s`, tamper `50.091 s`).
  - `c9_soak` (`29.854 s`, `20` cycles: install median `0.0638 s`, undo median `0.0172 s`).
  - `c10_not_proven` (`111.592 s`, `proven False` with `['zero_witness']` on startup-only `setup_model_and_optimizer` `config.lr = config.lr * 5.0` while fresh subprocess diverged cleanly to `e2b8a46d3a18e787`).

## 6. First-Class CLI `fastrsi conformance miles --real` (`MilesConformanceLab`, `2× B200`, `2026-11-19`)

Source: `python3 -m fastrsi conformance miles --real` (`adapter.conformance_lab:MilesConformanceLab`, `livelab.json` `"real"` enabled, `6/6` real-GPU engines `cli_launch=yes`) inside `slimerl/slime:latest` on `2× B200` (`Qwen3-0.6B`, `parity_rm.parity_rm`, `check_timeout=20.0`, `rpc_timeout=60.0`):
- **Result:** **`11/11` PASS**, `113` receipts (`43 txn`, `38 proof`, `32 undo`), `0 MiB × 8` teardown on `b200` (`median edit_to_commit_s 0.0419 s` / `median undo_s 0.0053 s`):
  - `c0_repeatable` (`13.26 s`, `4872311a7182bf7b`), `c1_noop` (`8.062 s`, witness `[1]`), `c2_visible` (`7.973 s`, `4872311a7182bf7b -> 6a673f84cb0ddfc8`, witness `[1]`).
  - `c3_fresh_equivalence` (`321.051 s`, bit-exact `live == fresh` on visible `6a673f84cb0ddfc8` & 3-chain `893da88d23d7f9d3`).
  - `c4_uninstall` (`9.322 s`, undo `0.013525 s`), `c5_failed_install` (`10.263 s`), `c6_refusal` (`10.465 s`), `c7_chain` (`12.096 s`).
  - `c8_crash_and_hang` (`238.569 s`, respawns: crash `47.415 s`, hang `47.279 s`, lost `47.307 s`, tamper `47.415 s`).
  - `c9_soak` (`36.169 s`, `20` cycles: install median `0.0622 s`, undo median `0.0169 s`).
  - `c10_not_proven` (`110.138 s`, `proven False` with `['zero_witness']`, base `4872311a7182bf7b` vs fresh `3112e92c2f60e8d1`).

## 7. Unified Cold-vs-Warm Restart-Overhead Dissection & Live vs. Replay Ladder (`2× B200`, `2026-11-20`)

Source: paired `2× NVIDIA B200` (GPUs `4, 5`, `0 MiB` exit) end-to-end instrumentation across `fresh_cold` (empty compile caches), `fresh_warm`, `live_full` (`MilesStepGate` + `Snapshot S0` `3.58 GB`), and `live_replay` (recorded `rollout0.pt`) on `Qwen3-0.6B` GSM8K GRPO (`parity_rm.parity_rm`, `1` SGLang rollout + `1` Megatron trainer, `243` spawned processes, `not_split = 0.000 s`).

- **Parallel actor overlap (zero double-counting):** At `create_training_models`, Ray spawns `MegatronTrainRayActor` and `RolloutManager -> SGLangEngine` concurrently. `MegatronTrainRayActor` initializes in **`12.17 s` cold (`11.86 s` warm)** (`dist_init 0.31 s`, `setup_model_opt 0.54 s`, HF `load_ckpt 0.46 s`), finishing `~44.8 s` cold (`~21.6 s` warm) **before** `SGLangEngine` (`56.99 s` cold / `33.43 s` warm), so the SGLang boot dominates the critical path.
- **Cold compile tax inside `fresh_cold` (`25.924 s` `jit_build`):** Clearing Inductor/Triton/FlashInfer/DeepGEMM caches inflates SGLang `Capture cuda graph bs [1, 2, 4, 8, 12, 16]` from **`0.70 s` warm to `21.96 s` cold** (`+21.260 s`) and Step-1 first-step JIT/warmup from **`2.385 s` warm (`3.072 s` Step 1 vs `0.687 s` steady) to `7.049 s` cold (`7.884 s` Step 1 vs `0.834 s` steady)** (`+4.664 s` across SGLang step-0 `+0.50 s` and Megatron/TE/cuDNN `+4.17 s`).

| Arm / Rung | Total (`s`) | Speedup vs Cold | Critical-Path Phase Breakdown (`s`) |
|---|---:|---:|---|
| 1. `fresh_cold` (empty compile caches) | **`82.385 s`** | `1.0×` | `import_init 9.556` + `ray_init 22.210` + `spawn_import 11.207` + `weight_load 0.264` + `jit_build 25.924` + `cuda_graph_capture 0.700` + `engine_init 9.304` + `first_step 2.385` + `run_workload 0.834` (`boot_to_ready 74.501 s`, `step1 7.884 s`, `subproc_wall_s 89.360 s`) |
| 2. `fresh_warm` (warm compile caches) | **`53.480 s`** | `1.5×` | `import_init 9.524` + `ray_init 21.423` + `spawn_import 10.751` + `weight_load 0.432` + `cuda_graph_capture 0.700` + `engine_init 7.578` + `first_step 2.385` + `run_workload 0.687` (`boot_to_ready 50.408 s`, `step1 3.072 s`, `subproc_wall_s 60.510 s`) |
| 3. Derived peel `import_init` | `43.956 s` | `1.9×` | Row 2 minus `import_init` (`9.524 s`) |
| 4. Derived peel `ray_init + spawn_import` | `11.782 s` | `7.0×` | Row 3 minus `ray_init` (`21.423 s`) and `spawn_import` (`10.751 s`) |
| 5. Derived peel `weight_load + cuda_graph_capture` | `10.650 s` | `7.7×` | Row 4 minus `weight_load` (`0.432 s`) and `cuda_graph_capture` (`0.700 s`) |
| 6. Derived peel `engine_init` | `3.072 s` | `26.8×` | Row 5 minus `engine_init` (`7.578 s`; leaves Step-1 warmup + step) |
| 7. Live `R0` swap + `S0` restore + full GRPO step | **`0.890 s`** | **`92.6×`** | `live_install 0.0529` (`get_grpo_returns`, witness `1`) + `restore_state 0.2204` (`0.0967 s` trainer + `0.1237 s` sync) + `run_workload 0.6168` (`0.1966 gen + 0.2962 train + 0.1240 sync`; `undo 0.0157 s` bit-exact `w_post_undo == 2372e7ae7cac01a7`) |
| 8. Live `R0` swap + `S0` + recorded-rollout replay | **`0.678 s`** | **`121.5×`** | `live_install 0.0529` + `restore_state 0.2240` (`0.0969 + 0.1271`) + `run_workload 0.4013` (`0.2730 train + 0.1284 sync`, skips SGLang generation) |

## 8. Post-fanout & PEP 578 negative-cache full `11/11` `MilesConformanceLab` (`2× B200`, `2026-11-20`)

Re-ran container tests (`repos/miles/livelab` **`15 passed, 11 subtests`**; `core/livelab` `test_ray_participants` + `test_coordinator` **`33 passed`**; `0` FastRSI `ResourceWarning`s) and full `python3 -m fastrsi conformance miles --real` inside `slimerl/slime:latest` on `2× B200` after `core/livelab` parallel `_fanout` + PEP 578 `file_namespace` negative cache (`0 MiB × 2` exit):
- **Conformance:** **`11/11` PASS**, `113` receipts (`43` txn / `38` proof / `32` undo; total case sum `791.114 s`: `c0 13.344 s`, `c1 8.015 s`, `c2 8.022 s`, `c3 333.664 s` bit-exact, `c4 9.448 s`, `c5 11.225 s`, `c6 10.120 s`, `c7 12.183 s`, `c8 238.547 s` [`47.07–47.81 s` respawns], `c9 36.029 s`, `c10 110.517 s`).
- **`c9_soak` & coordinator speedups vs prior serial baseline:** `c9` install median **`0.0622 -> 0.0536 s`** (**`1.160×`**), `c9` undo median **`0.0169 -> 0.0129 s`** (**`1.310×`**), `edit_to_commit_s` median (`37` committed txns) **`0.041859 -> 0.038396 s`** (**`1.090×`**), `gap_s` median **`0.010459 -> 0.008311 s`** (**`1.258×`**), and `undo_s` median (`32` undos) **`0.005291 -> 0.004624 s`** (**`1.144×`**).

## 9. AST `compile()` negative cache & two-stage `2× B200` ladder (`2026-11-21`)

- **AST `compile()` negative cache (`Qwen3-0.6B` GSM8K GRPO + `parity_rm`):** **`37.15×`** warm `status()` (`0.875 -> 0.0236 s`), `32` AST compiles with `0` wipes, **`11/11`** conformance, **`15/15`** container `pytest`.
- **Two-stage `2× B200` ladder:** Cold **`102.58 s`** -> Warm **`72.27 s`** -> Stage 1 Live `MilesStepGate` + `S0` + `320`-trajectory (`40×8×512`) GRPO step **`14.07 s`** (**`7.3×`**: `0.0495 s` install + `0.424 s` `S0` + `13.60 s` step) -> Stage 2 `64`-trajectory (`8×8×512`) recorded-rollout Megatron replay + weight sync **`1.89 s`** (**`54.3×`**: `0.0511 s` install + `0.226 s` `S0` + `1.61 s` train/sync; bit-exact `S0`/undo/fresh).
