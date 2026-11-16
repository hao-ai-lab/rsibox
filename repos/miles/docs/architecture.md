# `miles` (`THUDM/slime` Fork) Process Architecture and Live-Lab Boundaries

Upstream reference: `radixark/miles` `0.2.0` (`7767e158bcb743b47a7db0b0674abebacc02e82a`), an
agentic and asynchronous RL post-training framework forked from `THUDM/slime`.

## 1. Lineage: `THUDM/slime` to `radixark/miles`

`miles` preserves `slime`'s core three-plane Ray architecture—decoupled SGLang rollout servers, a
Ray rollout/data-buffer orchestrator, and Megatron-LM / FSDP training actors joined by in-memory
weight transfer—while expanding cell-level fault tolerance and fully asynchronous generation:

| Plane | `THUDM/slime` Origin | `radixark/miles` (`0.2.0`) | Live-Lab Reach (`repos/miles/livelab/adapter/`) |
|---|---|---|---|
| **Orchestrator driver** | `train.py` Ray driver | `train.py` (sync), `train_async.py` (overlapped / `--fully-async`), `train_multi_policy.py` | Drives `MilesStepGate` at RL step boundaries |
| **Rollout & buffer plane** | `RolloutManager` + `DataBuffer` | `RolloutExecutor` (`miles/ray/rollout/rollout_executor.py`), `FullyAsyncRolloutFn` (`miles/rollout/fully_async_rollout.py`), `DefaultDataBuffer` (`miles/rollout/fully_async_data_buffer.py`), `InferenceController` / `RolloutServer` (`miles/ray/rollout/rollout_server.py`) | Reached via `MilesActorParticipant` (`ray_participants.py`) + `EpochDataBuffer` |
| **Inference engines** | SGLang servers behind `sgl-router` | `ServerCell` launching `sglang.launch_server` (`miles/backends/sglang_utils/sglang_engine.py`) | Quiesced via `pause_generation` + `flush_cache`; synced via `update_weights` |
| **Trainer plane** | `RayTrainGroup` (`_actor_handlers`) + `MegatronTrainRayActor` | `TrainerController` (`miles/ray/train/group.py`) managing `TrainerCell`s of `@ray.remote` `MegatronTrainRayActor` (`miles/backends/megatron_utils/actor.py`) or `FSDPTrainRayActor` (`miles/backends/fsdp_utils/actor.py`) | Reached via `MilesActorParticipant` (`actor_view` unwraps `@ray.remote` class copies) |
| **Weight transfer** | P2P / NCCL broadcast | `WeightUpdater` (`miles/backends/training_utils/weight_update/protocols/{p2p,broadcast,cuda_ipc}.py`) + `TensorBackuper` / `InMemoryCheckpointManager` | Coupled with trainer `S0` restore in `CoupledWeightReset` |

`discover_actors()` in `adapter/gate.py` inspects both `miles` (`TrainerController._cells`,
`RolloutExecutor`) and `THUDM/slime` (`RayTrainGroup._actor_handlers`, `RolloutManager`) handle
layouts so the same adapter attaches to either runtime without code duplication.

## 2. Why Derived-Engine Onboarding Stays Under 600 LOC

Ray-based RL frameworks share two structural obstacles that break a naive module swap:

1. **`@ray.remote` deserialized class copies:** `@ray.remote` replaces `MegatronTrainRayActor` and
   rollout worker classes in `sys.modules` with an `ActorClass` handle, while each actor process
   executes a cloudpickled by-value class copy bound to its own `__globals__` dict. Swapping the
   module's class commits cleanly and changes nothing at runtime.
2. **`sys.modules`-less custom functions:** Reward shaping, dynamic sampling filters
   (`--dynamic-sampling-filter-path`), and custom rollout files loaded from disk may execute
   outside standard package imports.

Rather than copying ~1,000 lines of Ray introspection from another RL engine, `repos/miles/`
imports `core/livelab/ray_participants.py` (`actor_view`, `file_namespace`, `file_module`, `serve`,
`RayActorParticipant`, `unloaded`) directly. `MilesActorParticipant` adds a 22-line wrapper
(`serve_actor`) that primes `#actor` aliases prior to `Session` base fingerprinting and scopes
multi-role fingerprints to the modules each actor process actually loaded.

## 3. Dual-Mode Step-Gap Quiescence (`sync` vs. `async`)

### Mode A: Synchronous / Colocated Step Gap (`train.py`, `train_async.py --colocate`)

In synchronous RL (`train.py:112-176`), each `rollout_id` iteration runs:
`prepare_rollout` $\to$ `rollout_executor.get(rollout_id)` $\to$ optional offload $\to$
`actor_model.train(rollout_id, rollout_data_pack)` $\to$ `update_weights(...)`.
Between `update_weights` of step $t$ and `prepare_rollout` of step $t+1$, every rollout and trainer
Ray actor sits idle between remote method calls. `MilesStepGate(mode="sync")` parks the orchestrator
at this boundary while `StepGapCoordinator` runs Phase 2 (`validate` + `commit`) across all
`MilesActorParticipant`s via `__ray_call__`.

### Mode B: Asynchronous Rollout Quiescence (`train_async.py --fully-async`)

In `--fully-async` mode (`miles/rollout/fully_async_rollout.py`), `FullyAsyncRolloutFn._worker_loop`
continuously submits prompt groups and pushes finished `DataBufferInput` batches into
`DefaultDataBuffer` (`miles/rollout/fully_async_data_buffer.py`). Two hazards arise if an edit
commits without quiescing the rollout pipeline:

- **Mid-trajectory skew:** an in-flight multi-turn rollout or reward evaluation can observe pre-edit
  logic on turn 1 and post-edit logic on turn 2.
- **Stale-code buffer leakage:** `DefaultDataBuffer` only evicts groups exceeding
  `--max-weight-staleness` (`weight_version`), not groups generated under an older code version.

`MilesStepGate(mode="async")` + `EpochDataBuffer` (`adapter/state.py`) close both gaps:

1. `_quiesce_async()` pauses buffer admission (`EpochDataBuffer.pause()`), invokes
   `pause_generation` + `flush_cache` on `InferenceController` / `RolloutServer`
   (`miles/ray/rollout/rollout_server.py:166-176`), executes Phase 2 across all actors, and calls
   `continue_generation()` + `resume()` in `finally:`.
2. Every committed `install` or `uninstall` calls `EpochDataBuffer.advance_epoch()`, bumping
   `code_epoch`. Subsequent `get_groups()` / `drain_stale()` calls evict any buffered group whose
   stamped `epoch < code_epoch` (routing evicted groups to `unused_handler` with reason
   `"stale_code_epoch"`).

## 4. Coupled `S0` Weight-Sync Reset (`adapter/state.py`)

During conformance (`c0_repeatable`, `c3_fresh_equivalence`, `c4_uninstall`) and paired A/B
candidate evaluation, each RL check mutates trainer weights (`MegatronTrainRayActor`) and publishes
them to rollout cells (`update_weights`). Restoring trainer `S0` state alone
(`InMemoryCheckpointManager` / `TensorBackuper`) leaves SGLang rollout cells serving post-step
weights. `CoupledWeightReset.restore_and_sync()` enforces a 4-part atomic reset at the step gap:

1. Drain uncommitted and stale-epoch groups from `EpochDataBuffer` (`drain_stale(clear_all=True)`).
2. Restore `S0` weights, optimizer step counters, and RNG state on all trainer actors.
3. Broadcast restored `S0` weights to rollout actors via `update_weights`.
4. Clear rollout KV / Radix caches (`flush_cache`).

## 5. Priced Restart Ladder

| Tier | Boundary | What stays warm | Typical cost |
|---|---|---|---|
| **R0** | `MilesStepGate` live swap (`OP`, `@triton.jit`, custom `.py`) | Ray cluster, placement groups, NCCL/Gloo groups, Megatron weights, SGLang servers | `< 0.05 s` (CPU); `< 1 s` (Ray RPC) |
| **R2 / R3** | `EpochDataBuffer.drain_stale` + `flush_cache` | All processes and weights; flushes stale-epoch trajectories and KV cache | `< 0.1 s` |
| **S0 Reset** | `CoupledWeightReset.restore_and_sync` | In-memory `S0` trainer snapshot + P2P/NCCL `update_weights` to SGLang | `~0.3–2.5 s` (model-size dependent) |
| **Respawn** | `MilesLab.reset("respawn")` (actor pool restart) | Ray cluster & OS page cache; rebuilds actors and reloads checkpoints | `~25–60 s` |
| **Cold start** | Full `ray start` + `train.py` placement group + Megatron + SGLang init | Nothing | `~90–210 s` |
