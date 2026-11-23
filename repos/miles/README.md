# `miles` (`radixark/miles`, `THUDM/slime` Fork)

Live-lab adapter, static coverage rules, and conformance suite for `miles` (`radixark/miles`
`0.2.0` @ `7767e158bcb743b47a7db0b0674abebacc02e82a`), a Ray-based RL post-training framework
forked from `THUDM/slime` that couples SGLang rollout servers with Megatron-LM / FSDP training
actors across synchronous (`train.py`) and fully asynchronous (`train_async.py --fully-async`)
loops.

## Highlights

- **Derived-engine zero-copy Ray reuse (`<= 600` non-test LOC):** Reuses
  `core/livelab/ray_participants.py` (`RayActorParticipant`, `actor_view`, `file_namespace`,
  `file_module`, `unloaded`) directly rather than duplicating Ray actor unwrapping.
- **Dual-mode step-gap quiescence (`adapter/gate.py`):** Coordinates Phase-2 commits across
  `RolloutExecutor` / `RolloutManager` and `MegatronTrainRayActor` / `FSDPTrainRayActor` ranks at
  synchronous RL step barriers (`mode="sync"`) or after pausing async rollout generation and
  flushing caches (`mode="async"`).
- **`code_epoch` rollout buffer drain & coupled `S0` weight reset (`adapter/state.py`):** Tags
  buffered trajectory groups with `code_epoch` so pre-edit async rollouts never leak into post-edit
  training steps, and couples trainer `S0` weight restoration with an immediate `update_weights`
  broadcast + `flush_cache` to rollout servers.
- **`11/11` conformance (`c0`–`c10`):** Passes all 11 conformance cases across Python `3.11`–`3.14`
  on spawned rollout and trainer actor processes (`ToyMilesLab`), plus real `@ray.remote` actor
  live-swap verification (`test_adapter.py`).
- **Static coverage (`kb/survey_rules.json`):** Out of `6,422` Python functions (`111,721` lines)
  in `radixark/miles`, `3,886` functions (`60.5%`, including `47/47` `@triton.jit` kernels) have a
  live install path (`swap` + `jit_rebuild`) and `622` (`9.7%`) have a warm respawn path.

## Onboarding LOC Scorecard

| Layer / File | Owner | Non-Test Python LOC | Test Python LOC | Reused `core/livelab/` Primitives |
|---|---|---:|---:|---|
| `livelab/adapter/__init__.py` | `repos/miles` | `10` | `0` | `RSIBOX_LIVELAB` path bootstrap |
| `livelab/adapter/gate.py` | `repos/miles` | `116` | `0` | `StepGapCoordinator`, `RayActorParticipant`, `actor_view`, `file_namespace`, `unloaded` |
| `livelab/adapter/state.py` | `repos/miles` | `76` | `0` | `EpochDataBuffer` (`code_epoch` drain) + `CoupledWeightReset` (`S0` + rollout sync) |
| `livelab/adapter/lab.py` | `repos/miles` | `135` | `0` | `Session`, `ReceiptWriter`, `LabUnderTest` (`MilesLab`) |
| `livelab/adapter/toy_lab.py` + `toymiles/*` | `repos/miles` | `260` | `0` | Spawned multi-actor CPU stand-in (`ToyMilesLab`, `11/11` conformance) |
| `livelab/adapter/test_adapter.py` | `repos/miles` | `0` | `234` | 8 golden unit tests (incl. real Ray 2.x cluster test) |
| **Adapter boundary total** | **`repos/miles/livelab/adapter/`** | **`597`** (`337` prod + `260` toy) | **`234`** | **`0` cross-`repos/` imports; `0` duplicated kernel files** |
| Shared runtime (reused, 0 new LOC) | `core/livelab/` | `0` (shared) | `0` (shared) | `kernel/*`, `ray_participants.py`, `forksafe.py`, `conformance/*`, `survey/*` |

## Layout

- `docs/architecture.md`: `THUDM/slime` $\to$ `radixark/miles` process topology, dual-mode step-gap
  quiescence, `code_epoch` drain, and coupled `S0` weight-sync reset.
- `kb/{costs.md, workload.md, lessons.md, survey_rules.json}`: measured latencies, workload and
  witness gates, RL-specific live-swap lessons, and static survey declarations.
- `livelab/`: `MilesStepGate`, `EpochDataBuffer`, `CoupledWeightReset`, `MilesLab`, `ToyMilesLab`,
  and golden unit tests.
