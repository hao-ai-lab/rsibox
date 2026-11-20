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
