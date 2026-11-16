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
