# Workload F(x) and Correctness Gate (`verl`)

Fixed target workload and verification contract for `repos/verl/`.

## 1. Reference GPU Workload (`4x GB200`, Colocated Synchronous GRPO)

- **Model**: `Qwen/Qwen3-0.6B` (dense causal LM), `bf16` actor/ref parameters with `fp32` Megatron
  master weights and distributed Adam.
- **Trainer**: `verl.trainer.ppo.v1.trainer_sync:PPOTrainerSync` (`trainer.v1.trainer_mode=fastrsi`),
  Megatron-Core backend (`TP=1, PP=1, CP=1, DP=4`), `ppo_mini_batch_size=64`.
- **Rollout**: 4 colocated vLLM replicas (`TP=1` each), `rollout.n=10` group responses per prompt,
  `train_batch_size=64` prompts -> **640 responses per RL step**.
- **Dataset & Reward**: GSM8K (`train.parquet`), custom rule-based reward function loaded via
  `reward.custom_reward_function.path` across 8 `RewardLoopWorker` Ray actors.

## 2. Observable Output & Correctness Gate

1. **Performance-only edits (`--set` micro-batch / dynamic-batch overrides, fused kernels)**:
   - Measure steady-state `timing_s/step` (excluding gate idle wait: use `verdict_s - install_s`
     or steps 2+ of a multi-step request).
   - **Correctness gate**: With `reset("state")` restoring coupled `S0` policy weights + Adam
     moments + rollout weights, per-step `actor/pg_loss`, `critic/score/mean`, and `actor/grad_norm`
     must stay within **0.5% relative** (loss/score) and **5% relative** (grad norm) of baseline
     on identical prompt batches.
2. **Algorithmic / reward / rollout-request edits (`core_algos.py`, custom reward `.py`, `agent_loop_tq.py`)**:
   - **Witness gate (mandatory)**: Every edited non-constructor target must record `witness > 0`
     on the actors that execute its role (`reward` for reward scripts, `driver` for `core_algos`,
     `rollout` `#actor` view for `AgentLoopWorkerTQ`, `trainer` for `losses.py` / `engine_workers.py`).
     Zero-witness commits fail closed (`proven=False`).
   - **Offline pre-screen**:
     - Reward edits: `python3 -m tools.rescore --rollouts <gate>/rollouts/<step>.jsonl --reward <file.py>`
       (`0.5 ms`) verifies exact score deltas on 640 real responses before advancing trainer weights.
     - Advantage edits: `python3 -m tools.replay_adv --step <gate>/steps/<step>` (`12 ms`) verifies
       exact advantage tensor deltas (`max_abs_diff`, `rows_changed`, min/max range).
3. **CPU Conformance Workload (`ToyVerlLab`)**:
   - 4 spawned actors (`rollout0`, `rollout1`, `trainer0`, `reward0`), 2 synchronous RL steps over
     4 prompts (`default_workload = {"prompts": [1, 2, 3, 4], "steps": 2}`).
   - Bit-exact comparison of per-step `tokens`, `scores`, `loss`, and post-step `weights` across
     `c0_repeatable` through `c10_not_proven`.
