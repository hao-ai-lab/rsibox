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

## Participants and the gate protocol

The gate (`adapter/gate.py`) serves one request per step gap from `$FASTRSI_RL_GATE`: `ready.json`
names the next request number `n`, the client writes `req-<n>.json` and reads `resp-<n>.json`
(`tools.cli request` does this). Participants are `driver` (the `TaskRunner` process) plus every
warm Ray actor: `reward<i>` (`RewardLoopWorker`), `rollout<i>` (`AgentLoopWorkerTQ`, discovered at
the first step gap because `fit()` creates them) and `trainer<i>` (`WorkerDict`, one per GPU).

| Verb | Request fields | Does |
|---|---|---|
| `install` | `files` (paths under `$FASTRSI_RL_ROOTS`, read from disk) or `sources` (`{module or path: text}`), `steps`, `set` | one transaction on every participant that loaded the module, then `steps` RL steps with a witness verdict |
| `check` | `steps` | runs steps and reports metrics, the rollout dump path, and the witness of every live edit |
| `reset` | | coupled `S0`: trainer weights, optimizer, scheduler, RNG, **dataloader position**, then rollout weights (`sleep_replicas` + `update_weights`) |
| `status` | | `base` / `edited` / `poisoned`, the transaction stack, and the fingerprint of the tracked modules (`$FASTRSI_RL_TRACK`; default `core_algos` plus the custom reward file) |
| `uninstall` | `id` | undoes that transaction and every later one (default: the newest) |
| `inject` | `fault`, `participant`, `kwargs` | conformance faults only: `fail_apply`, `fail_commit`, `lost`, `tamper` in any participant; `crash`, `hang` at the driver's step boundary |
| `stop` | | ends the job cleanly (`stopped.json`) |

`FASTRSI_RL_SEED_PER_SAMPLE=1` makes checks reproducible: verl's `rollout.full_determinism` gives every
request on a replica the same vLLM seed, so a group's samples collapse and the replica a request lands
on decides its text; `adapter/determinism.py` seeds each request from `(sample_index, session_id)`
instead, inside every agent-loop actor.

## Running Conformance

On CPU, against the spawned multi-actor toy (`11/11`):

```sh
cd core/livelab
PYTHONPATH=../../repos/verl/livelab \
  python3 -m conformance --lab adapter.toy_lab:ToyVerlLab
```

On GPUs, against a real gated verl job (`adapter/conformance_lab.py`, a client of the gate that also
owns the job's lifetime, so `respawn` and `fresh_check` really restart it). `--launch` is a shell
command that starts the job on `$GATE` with the reward file `$REWARD` and exits when the job exits; a
fresh process with a patched module gets a `PYTHONPATH` overlay first. The job must run with
`rollout.full_determinism=True` and `FASTRSI_RL_SEED_PER_SAMPLE=1`, and its reward file must be
`examples/grpo-gsm8k/reward_gsm8k.py` (the harness edits its anchors):

```sh
cd repos/verl
python3 -m tools.cli conformance --gate /tmp/verl-gate --reward examples/grpo-gsm8k/reward_gsm8k.py \
  --launch 'python3 -m tools.cli train --gate $GATE -- ... reward.custom_reward_function.path=$REWARD' \
  --receipts receipts.jsonl --json
```

A check is one RL step after a coupled `S0` reset; its output is `critic/score/mean`,
`critic/advantages/{max,min}`, `actor/pg_loss` and a digest of the dumped rollouts (prompt, response,
score, ground truth), so `c0` really asks the same prompts to be sampled, scored and trained the same
way twice. `fresh_check` stops the live job, boots a fresh one from the edited files, reads its first
step, then relaunches the live job and re-installs the live edits.

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
  reward.custom_reward_function.path=examples/grpo-gsm8k/reward_gsm8k.py

# 2. Edit a reward, core_algos, loss, or agent-loop file and request a live step verdict
PYTHONPATH=repos/verl/livelab:repos/verl \
  python3 -m tools.cli request --gate /tmp/verl-gate --steps 1 examples/grpo-gsm8k/reward_gsm8k.py
# a step with no edit, a live knob, or the session state
PYTHONPATH=repos/verl/livelab:repos/verl python3 -m tools.cli request --gate /tmp/verl-gate --verb check
PYTHONPATH=repos/verl/livelab:repos/verl python3 -m tools.cli request --gate /tmp/verl-gate --verb check --set actor.ppo_micro_batch_size_per_gpu=8
PYTHONPATH=repos/verl/livelab:repos/verl python3 -m tools.cli request --gate /tmp/verl-gate --verb status

# 3. Restore coupled S0 weights or undo the last live edit
PYTHONPATH=repos/verl/livelab:repos/verl \
  python3 -m tools.cli request --gate /tmp/verl-gate --verb reset
PYTHONPATH=repos/verl/livelab:repos/verl \
  python3 -m tools.cli request --gate /tmp/verl-gate --verb uninstall

# 4. Run sub-millisecond / millisecond offline probes on recorded rollouts & advantages
PYTHONPATH=repos/verl python3 -m tools.rescore \
  --rollouts /tmp/verl-gate/rollouts/1.jsonl --reward examples/grpo-gsm8k/reward_gsm8k.py
PYTHONPATH=repos/verl python3 -m tools.replay_adv \
  --step /tmp/verl-gate/steps/1
```
