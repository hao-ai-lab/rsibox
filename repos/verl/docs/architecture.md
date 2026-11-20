# verl Architecture: Ray Process Tree, Step-Gap Quiescence, and Coupled `S0` Reset

This document maps a synchronous `verl` RL post-training job (`verl.trainer.ppo.v1.trainer_sync:PPOTrainerSync`
at `volcengine/verl` commit `975f063b`, `0.10.0.dev`) onto the FastRSI live-lab primitives:
which processes exist, what state each holds, where the global quiescence barrier sits, and
what invalidates a live swap vs. a coupled `S0` state reset vs. a full Ray job restart.
All timings cite `../kb/costs.md`.

## 1. Process Tree (`PPOTrainerSync`, 4x GB200 Colocated GRPO)

```
main_ppo driver                 Hydra config parse, ray.init(runtime_env=setup_hook)
  |  spawns remote actor
  v
TaskRunnerV1 (1 CPU actor)      PPOTrainerFastRSISync.fit(), dataloader, core_algos advantage,
  |                             CheckpointEngineManager, TransferQueue coordinator, Gate host
  +-- WorkerDict x4 (1/GPU)     Colocated ActorRolloutRefWorker per rank (NCCL + Megatron/FSDP):
  |     +-- actor               TrainingWorker (Megatron-Core or FSDP model, fp32 master weights,
  |     |                       Adam moments, LR scheduler)
  |     +-- ref                 Reference policy engine (frozen weights, log-prob forward)
  |     +-- rollout             Colocated vLLM / SGLang replica (sleeps during train, wakes on
  |                             CheckpointEngineManager.update_weights)
  +-- RewardLoopWorker x8       CPU Ray actors scoring responses; custom reward .py loaded via
  |                             importlib.util.spec_from_file_location (NO sys.modules entry!)
  +-- AgentLoopWorkerTQ xN      @ray.remote CPU actors assembling prompts, calling LLMServerClient,
                                dispatching streaming rewards, and writing to TransferQueue
```

In the standard 4x GB200 `Qwen3-0.6B` GRPO configuration (`TP1/PP1/DP4`, 8 reward workers,
colocated rollout), `1 + 8 + N + 4` warm Ray processes (`13` before lazy agent-loop spawn,
`13+` once `fit()` initializes `AgentLoopManager`) participate in every step.

## 2. Startup Init Chain & Priced Restart Boundaries

From `VERL_STARTUP_TRACE` on 4x GB200 (`Qwen3-0.6B`, Megatron-Core trainer + 4 vLLM replicas,
warm compile caches on parallel storage vs. shared NFS; `../kb/costs.md`):

| Phase | Parallel FS (warm) | Shared NFS (warm) | State Built |
|---|---:|---:|---|
| 1. Interpreter + imports | 9.1-13.5 s | 20.2 s | Python, PyTorch, Ray, Hydra |
| 2. `ray.init` + `TaskRunner` | 12.5-13.8 s | 20.6 s | Raylet, GCS, runtime_env setup hook |
| 3. Tokenizer + dataloader | 23.9-24.3 s | 251.4 s | Tokenizes 7.5k GSM8K prompts for overlong filter |
| 4. Trainer `WorkerDict` init | 28.1-29.1 s | 178.2 s | NCCL groups, Megatron/TE build, HF weight load (actor + ref) |
| 5. Rollout engine boot (x4) | 57.4-68.5 s | 242.4 s | vLLM `torch.compile`, FlashInfer/Triton caches, CUDA graphs |
| 6. First weight sync + managers | 2.4-2.9 s | 5.6 s | `sleep_replicas` -> `update_weights`, `RewardLoopManager` |
| 7. First step extras | ~25.0 s | ~275.3 s | Lazy `AgentLoopWorkerTQ` spawn + first-step kernel warmup |
| 8. Steady RL step | 25.1-26.8 s | 25.1 s | `gen` (6.0 s) + `old_log_prob` (4.9 s) + `ref` (6.0 s) + `adv` (0.3 s) + `update_actor` (8.1 s) + `update_weights` (1.4 s) |
| **Launch -> first step verdict** | **187-204 s** | **1,022 s** | **Full cold/warm job restart floor** |

## 3. Two Ray-Specific Indirections Solved by `core/livelab/ray_participants.py`

1. **`@ray.remote` Class Copy Split (`actor_view`)**:
   `@ray.remote` on `AgentLoopWorkerTQ` (`verl/trainer/ppo/v1/agent_loop_tq.py:55`) replaces the
   class in its module with a Ray `ActorClass` handle. When Ray spawns the actor worker,
   cloudpickle deserializes a **by-value copy** of the class (`type(self).__ray_actor_class__`)
   whose methods hold a fresh `__globals__` dict distinct from `sys.modules[module].__dict__`.
   A naive `LocalParticipant` swaps the dead class inside `ActorClass.__ray_metadata__` and
   reports `committed`, while the actor keeps executing old bytecode!
   `ray_participants.actor_view(host)` exposes the live copy's `__globals__` under a synthetic
   `<module>#actor` plan entry and filters `zero_witness` on the dead module-level class body.
2. **`sys.modules`-less Custom Reward Scripts (`file_namespace`)**:
   `verl/utils/import_utils.py:load_extern_object` loads `reward.custom_reward_function.path`
   via `spec_from_file_location` + `exec_module` **without** inserting `module_name` into
   `sys.modules`, retaining only the `compute_score` function reference. `file_namespace(path)`
   scans live functions (`locate.all_objects()`) by `co_filename`, wraps the function's live
   `__globals__` in a `Namespace` stand-in, and fails closed if the file was loaded into
   multiple disjoint globals dicts.

## 4. Global Step-Gap Synchronization in `PPOTrainer.fit()`

In `PPOTrainerSync`, `fit()` (`verl/trainer/ppo/v1/trainer_base.py:446-539`) is strictly
synchronous across steps:
- Before `on_step_begin()` runs at step `t+1`, step `t`'s rollout generation, reward scoring,
  log-prob evaluation, actor optimizer step, `CheckpointEngineManager.update_weights()`, and
  `tq.kv_clear()` have all finished.
- No rollout request, reward RPC, or NCCL collective is in flight.
- `GateMixin.on_step_begin()` (`livelab/adapter/gate.py`) runs `StepGapCoordinator` Phase 1
  (`prepare`) and Phase 2 (`validate` -> `commit` or `abort`/`undo`) across `TaskRunner` +
  every warm Ray actor handle via `handle.__ray_call__.remote(ray_participants.serve, ...)`.
- Because every actor processes `__ray_call__` sequentially on its worker loop while idle at
  the step boundary, **no step ever observes mixed code versions across actors**.

## 5. Coupled Trainer + Rollout Weight Snapshot (`S0`) & Reset Rules

In RL post-training, policy weights mutate on every step (`optimizer.step()` on `WorkerDict`
followed by `checkpoint_manager.update_weights()` into the rollout engines). Restoring only
trainer weights leaves the rollout engine generating trajectories from post-step policy weights,
breaking repeatability (`c0`), fresh-process equivalence (`c3`), and uninstall state parity.

`livelab/adapter/state.py` enforces **coupled `S0` restoration** on `reset("state")`:
1. At `on_init_end()` (after checkpoint load and initial weight sync), each `WorkerDict` rank
   snapshots its trainable parameters, buffers, optimizer moments/scalars, LR scheduler step,
   and RNG state (`worker_capture_s0`).
2. On `reset("state")`, every `WorkerDict` rank pre-validates parameter names, shapes, and
   dtypes (failing closed without mutating tensors on any layout drift), copies `S0` back in
   place, restores optimizer/scheduler/RNG state (`worker_restore_s0`), and `TaskRunner`
   immediately runs `checkpoint_manager.sleep_replicas()` + `checkpoint_manager.update_weights()`
   (`sync_rollout_weights`) so rollout KV caches are discarded and rollout weights match `S0`
   bit-for-bit.

## 6. Three-Tier Verdict Ladder

Costs below are batch B of `../kb/costs.md` (Megatron-Core, CUDA graphs, 2026-10-21). Batch A (FSDP, eager vLLM,
2026-11-19) measured the same ladder as 0.85 ms / 16-29 ms / 104.6-117.2 s / 259.8 s.

| Edit Target | Fastest Sound Tier | Typical Cost | Mechanism |
|---|---|---:|---|
| Custom reward function (`compute_score`) | Offline `tools/rescore.py` | **0.5 ms** | Re-scores recorded `<gate>/rollouts/<step>.jsonl` in a fresh CPU process |
| Advantage / KL shaping (`core_algos.py`) | Offline `tools/replay_adv.py` | **12 ms** | Replays recorded `<gate>/steps/<step>/adv.pt` on CPU |
| Reward, advantage, loss, agent-loop, or `--set` micro-batch | Live step-gap (`adapter/gate.py`) | **28-39 s** | Atomic swap across 13+ Ray processes + 1 full GRPO step with witness check |
| Constructor state, parallel topology (`TP/PP/DP`), rollout CUDA graphs, C++ | Full job restart | **187-204 s** | Refused cleanly at `prepare`; requires Ray job relaunch |
