# Walkthrough: live edits on a real verl GRPO job (4x GB200)

What one gated `Qwen3-0.6B` GSM8K GRPO job shows when edited live: a reward edit, an advantage edit, a
`--set` knob, a coupled `S0` reset, and the offline replays. Numbers are from our 2026-11-19 run on 4x GB200
(verl `cd58f550`, FSDP trainer, 4 colocated vLLM 0.31 replicas in eager mode, 8 reward actors, 640 responses per
step; `../../kb/costs.md` section 0 has the full table). Paths below are relative to `repos/verl`; `$GATE` is
the gate directory the job and the client share.

## 1. Start the job

```sh
python3 -m tools.cli train --gate $GATE -- \
  algorithm.adv_estimator=grpo data.train_files=gsm8k/train.parquet data.train_batch_size=64 \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B actor_rollout_ref.rollout.n=10 \
  actor_rollout_ref.rollout.full_determinism=True actor_rollout_ref.rollout.enforce_eager=True \
  reward.custom_reward_function.path=examples/grpo-gsm8k/reward_gsm8k.py \
  trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.total_training_steps=500 ...
```

`full_determinism` needs `enforce_eager` on this stack (vLLM's compile autotune refuses deterministic mode), and
`RSIBOX_RL_SEED_PER_SAMPLE=1` in the environment keeps GRPO groups diverse (`../../kb/lessons.md`).
122 s after launch `$GATE/gate.json` appears:

```json
{"pid": 2823611, "snapshot_files": 361, "snapshot_secs": 1.08, "warm_processes": 13,
 "s0": [{"ok": true, "digest": {"weights": "93f4ed14c2e6661c", "optimizer": "834c330340151767"}, "tensors": 31}, ...],
 "s0_capture_s": 0.239}
```

## 2. A step with nothing edited, then a coupled `S0` reset

```sh
python3 -m tools.cli request --gate $GATE --verb check
{"ok": true, "verdict_s": 112.621, "install_s": 0.0, "processes": 13, "metrics": {"critic/score/mean": 0.1875, ...}}
python3 -m tools.cli request --gate $GATE --verb reset
{"ok": true, "verdict_s": 1.097, "processes": 13}
```

The reset restores weights, Adam state, scheduler, RNG and dataloader position on the 4 trainer ranks and pushes the
weights to the 4 vLLM replicas. With per-sample seeds the next step reproduces the previous one bit-for-bit: five
checks in a row on the conformance job gave `critic/score/mean 0.12891`, `critic/advantages/max 2.47487`,
`actor/pg_loss 0.00841` each time.

## 3. Edit the reward on disk and ask for a live verdict

Change `return SCORE if answer == str(ground_truth) else FORMAT_SCORE` to `return 2.0 if ...` in
`examples/grpo-gsm8k/reward_gsm8k.py`, then:

```sh
python3 -m tools.cli request --gate $GATE --steps 1 examples/grpo-gsm8k/reward_gsm8k.py
{"ok": true, "outcome": "committed", "verdict_s": 117.163, "install_s": 5.771, "processes": 13, "proven": true,
 "witness": {"reward0": {"rsibox_file_60c85056948a:compute_score:15": 65}, "reward1": {...: 87}, ..., "reward7": {...}},
 "metrics": {"critic/score/mean": 0.4688, ...}}
```

The edit reached all 8 reward actors (640 witnessed calls) in 5.8 s; the verdict is one RL step later. verl loads
the reward file without a `sys.modules` entry, so the swap goes through `ray_participants.file_namespace`.

## 4. Edit the advantage estimator in `core_algos.py`

Multiply the normalized GRPO advantage by 0.5 in `compute_grpo_outcome_advantage`, then:

```sh
python3 -m tools.cli request --gate $GATE --steps 1 <verl>/verl/trainer/ppo/core_algos.py
{"ok": true, "outcome": "committed", "verdict_s": 104.622, "install_s": 0.575, "processes": 13, "proven": true,
 "witness": {"driver": {"verl.trainer.ppo.core_algos:compute_grpo_outcome_advantage:267": 1}, "trainer0": {...: 0}, ...}}
python3 -m tools.cli request --gate $GATE --verb uninstall
{"ok": true, "verdict_s": 0.058, "processes": 13}
```

Only the driver runs the estimator; the trainers also import `core_algos` and get the swap, their zero witness is
not a finding because the role that runs it fired.

## 5. A live performance knob

```sh
python3 -m tools.cli request --gate $GATE --verb check --set actor.ppo_micro_batch_size_per_gpu=8
{"ok": true, "verdict_s": 107.251, "set": {"actor.ppo_micro_batch_size_per_gpu": [16, 8]}, ...}
python3 -m tools.cli request --gate $GATE --verb check --set actor.ppo_micro_batch_size_per_gpu=7
{"ok": false, "verdict_s": 0.0, "error": "actor.ppo_micro_batch_size_per_gpu=7 does not divide the per-GPU batch of 160"}
```

## 6. Offline replays on what the job recorded

```sh
python3 -m tools.rescore --rollouts $GATE/rollouts/4.jsonl --reward examples/grpo-gsm8k/reward_gsm8k.py
{"responses": 640, "changed": 150, "mean_before": 0.4688, "mean_after": 0.2344, "score_ms": 0.834, "examples": [...]}
python3 -m tools.replay_adv --step $GATE/steps/5
{"rows": 640, "equal": true, "max_abs_diff": 0.0, "rows_changed": 0, ..., "compute_ms": 29.012}
```

Step 4 was scored by the live x2 reward, so re-scoring it with the base reward changes 150 of 640 responses in
0.8 ms. `replay_adv` recomputes the recorded step's advantages with the current `core_algos` in 16-29 ms.

## 7. Conformance on the same hardware

```sh
python3 -m tools.cli conformance --gate $GATE --reward examples/grpo-gsm8k/reward_gsm8k.py --launch '<the train command above>'
```

Runs `c0`-`c10` from `core/livelab/conformance` against the live job (`../../livelab/README.md`); the pass table
of our run is in `../../kb/costs.md`.
