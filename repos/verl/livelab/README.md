# verl Live Lab (`repos/verl/livelab/`)

Step-gap live-edit adapter for **verl** synchronous RL (`PPOTrainerSync`), built on
`core/livelab/kernel/` and `core/livelab/ray_participants.py`.

## Architecture & Contract

- **Quiesce point**: `GateMixin.on_step_begin()` inside `PPOTrainer.fit()`. Between step `t` and
  step `t+1`, rollout generation, reward scoring, reference/old log-probs, actor update, and
  `checkpoint_manager.update_weights()` are all complete; no Ray RPC or NCCL collective is in
  flight.
- **Participants**: `TaskRunner` (`driver`) + every warm Ray actor (`RewardLoopWorker`,
  `AgentLoopWorkerTQ`, `WorkerDict`) wrapped in `RayActorParticipant` over `__ray_call__`.
  - `@ray.remote` class copies (`AgentLoopWorkerTQ`) are unwrapped via `actor_view` (`#actor`).
  - `sys.modules`-less custom reward `.py` files are resolved via `file_namespace`.
- **Coupled `S0` state reset**: `state.py` captures trainer parameters, optimizer moments,
  scheduler step, and RNG at `on_init_end()`, and on `reset("state")` restores `S0` across all
  trainer ranks and immediately pushes `S0` weights to rollout replicas (`sleep_replicas` +
  `update_weights`) so `c0`, `c3`, and `c4` start from identical policy weights.
- **Live `--set` performance settings**: `settings.py` validates micro-batch divisibility and
  updates `micro_batch_size_per_gpu`, `infer_micro_batch_size_per_gpu`, `use_dynamic_bsz`, and
  `max_token_len_per_gpu` across `WorkerDict` actors in place.

## Running Conformance (`11/11` on CPU)

```sh
cd core/livelab
PYTHONPATH=../../repos/verl/livelab \
  python3 -m conformance --lab adapter.toy_lab:ToyVerlLab
```

## Running Golden Adapter Unit Tests

```sh
PYTHONPATH=repos/verl/livelab \
  python3 -m unittest discover -s repos/verl/livelab/adapter -t repos/verl/livelab -v
```

## Running on a Real `verl` GRPO Job

```sh
# 1. Start verl with the fastrsi trainer mode and setup hook
PYTHONPATH=repos/verl/livelab:repos/verl \
  python3 -m tools.cli train --gate /tmp/verl-gate --trace /tmp/verl-trace.jsonl -- \
  data.train_files=gsm8k/train.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  reward.custom_reward_function.path=rewards/gsm8k.py

# 2. Edit a reward, core_algos, loss, or agent-loop file and request a live step verdict
PYTHONPATH=repos/verl/livelab:repos/verl \
  python3 -m tools.cli request --gate /tmp/verl-gate --steps 1 rewards/gsm8k.py

# 3. Restore coupled S0 weights or undo the last live edit
PYTHONPATH=repos/verl/livelab:repos/verl \
  python3 -m tools.cli request --gate /tmp/verl-gate --verb reset
PYTHONPATH=repos/verl/livelab:repos/verl \
  python3 -m tools.cli request --gate /tmp/verl-gate --verb uninstall

# 4. Run sub-millisecond / millisecond offline probes on recorded rollouts & advantages
PYTHONPATH=repos/verl python3 -m tools.rescore \
  --rollouts /tmp/verl-gate/rollouts/1.jsonl --reward rewards/gsm8k.py
PYTHONPATH=repos/verl python3 -m tools.replay_adv \
  --step /tmp/verl-gate/steps/1
```
