# RSIBox Shared Services (`rsibox.services`)

`core/livelab/rsibox/services/` houses pure-stdlib, engine-agnostic control-plane services that sit beneath [`core/livelab/kernel/`](../core/livelab/INTERFACE.md) and compose across all ten supported engines (`vLLM`, `SGLang`, `Megatron-LM`, `FastVideo`, `veRL`, `miles`, `Transformers`, `TensorRT-LLM`, `TorchTitan`, and `NeMo-RL`).

Instead of each `repos/<engine>/livelab/services/` adapter reinventing weight residency, fork servers, artifact caches, or crash recovery, shared services standardize ownership, lease fencing, and staleness-stamp contracts while keeping `core/livelab/` free of engine or torch imports.

## Architecture Layering

```text
core/livelab/kernel/          (two-phase txns, code swap, JIT rebuild, witness, journal)
        │
        ▼
core/livelab/rsibox/services/ (S1 Memory [shipped] · S2 Zygote [shipped] · S3 Cache · S4 Aux · S5 Buffers · S6 Recovery)
        │
        ▼
core/livelab/rsibox/warm/     (WeightStore, DynamoGmsStore, QuiesceHooks, RecoveryDriver, stamp_diff)
        │
        ▼
repos/<engine>/livelab/       (10 engine adapters implementing the 5-verb INTERFACE.md contract)
```

## Service Map (`S1`–`S6`)

| Service | Scope | Status | Primary Contract & Mechanism |
|---|---|---|---|
| **S1 Memory** | Unified GPU memory & state ownership (`ro_weights`, `kv_pool`, `cow_s0_snapshot`, `ro_aux_weights`, `actor_rollout_shared`) | **Shipped** ([`memory.py`](../core/livelab/rsibox/services/memory.py)) | `MemoryService`, `MemoryHandle`, `SnapshotLease`; `WeightStore` / Dynamo GMS / CUDA IPC + `stamp_diff` |
| **S2 Zygote** | Pre-imported Python/CUDA fork server & warm worker pool | **Shipped** ([`zygote.py`](../core/livelab/rsibox/services/zygote.py)) | `ZygoteService`, `ZygoteProfile`, `SpawnedChild`, `profile_for_engine`; `forksafe` guards + `AF_UNIX` `SCM_RIGHTS` |
| **S3 Cache** | Triton, Inductor, C++ side-library & CUDA-graph artifact store | Planned | Content-addressed artifact reuse across respawns with `witness` and rollback pinning |
| **S4 Auxiliary** | Out-of-band reward, reference, and judge model pools | Planned | Dedicated RO placement and RPC endpoints decoupled from actor/trainer loops |
| **S5 Buffers** | Pre-allocated activation, KV scratch, and step replay buffers | Planned | Quiesce-aware sizing and deterministic zeroing on `reset("state")` |
| **S6 Recovery** | Coordinated fault detection, shadow failover, and snapshot restore | Planned | Unifies `RecoveryDriver`, `StandbyShadowDriver`, and CRIU/`cuda-checkpoint` |

## S1: Unified GPU Memory & State Service (Shipped)

[`core/livelab/rsibox/services/memory.py`](../core/livelab/rsibox/services/memory.py) unifies three engine-family memory lifecycles behind `MemoryService`:

1. **Inference RO & KV pools (`ro_weights`, `kv_pool`):** Attaches pluggable [`WeightStore`](../core/livelab/rsibox/warm/weight_store.py) backends (`DiskStore`, NVIDIA Dynamo GMS VMM, or CUDA IPC caches) via `attach_store()` or registers explicit regions via `register_region()`. Read-only pools refuse `read_only=False` and `writable=True` leases fail-closed, and every stamp is validated with [`stamp_diff`](../core/livelab/rsibox/warm/types.py) so stale loaders or drifted checkpoints never attach silently.
2. **Training baseline snapshots (`cow_s0_snapshot`):** Coordinates single-writer / multi-reader `SnapshotLease` locks over `S0` baseline keys (`weights`, `adam_m`, `adam_v`, `rng`). `restore_s0()` verifies an active writable lease, checks the expected epoch and stamp, and increments the epoch monotonically so stale retries fail closed.
3. **RL auxiliary & Actor↔Rollout handoff (`ro_aux_weights`, `actor_rollout_shared`):** Protects reference/reward weights as read-only (`ro_aux_weights`) and transfers handles across roles with `share_handle()` while preserving the `stamp` and `incarnation` fence (and blocking handoff while an exclusive writer lease is active).

### Safety & Ownership Invariants

- **Single-writer / multi-reader locking:** Multiple readers share `ro_weights`, `ro_aux_weights`, and `cow_s0_snapshot` concurrently; writers require exclusive access (zero active readers and zero active writers), and `ro_weights` / `ro_aux_weights` reject writable leases unconditionally.
- **Stamp staleness guard:** Every `register_region()`, `attach_store()`, `acquire_lease(..., expected_stamp=...)`, and `restore_s0(..., expected_stamp=...)` validates stamps via `stamp_diff()` across `files`, `weights`, `env`, and `extra`, failing closed on any drift.
- **Incarnation fence on role handoff:** `share_handle()` blocks while an exclusive writer lease is active and preserves both `stamp` and `incarnation` (or bumps `incarnation` explicitly on epoch promotion) across `actor` ↔ `rollout` transitions.
- **Orphan lease cleanup:** When a participant dies, `holder_died(holder_id)` releases all leases held by that worker, decrements handle refcounts idempotently, and clears stale writer locks so healthy workers resume without a cluster restart.
- **Owner-scoped teardown:** `teardown()` calls `store.stop()` only on `WeightStore` instances registered with `own_store=True`; foreign stores and shared external handles (`own_store=False`) are never stopped.

## S2: Shared Zygote & Pre-Fork Service (Shipped)

[`core/livelab/rsibox/services/zygote.py`](../core/livelab/rsibox/services/zygote.py) standardizes the warm pre-fork daemon across all 10 engines on top of [`core/livelab/forksafe.py`](../core/livelab/forksafe.py):

- **Fail-closed poison & guard arming:** `arm_guards()` checks `forksafe.fork_poisoned()` and `forksafe.poison_reasons()` before calling `forksafe.install_all_guards(...)`, optional module preloading, and `forksafe.freeze_gc_before_fork()`.
- **`AF_UNIX` + `SCM_RIGHTS` fork protocol:** `spawn_child()` passes file descriptors via `socket.send_fds` / `socket.recv_fds`, forks the worker, runs `forksafe.unpatch_child()` + `forksafe.apply_child_runtime_affinity()`, invokes the worker handler, and tracks/reaps PIDs with zero leaked sockets or zombies.

### Public API Summary (`rsibox.services.zygote`)

| Symbol | Purpose |
|---|---|
| `ZygoteProfile` | Frozen engine profile (`engine`, `preload_modules`, `env_defaults`, guard/GC flags) |
| `profile_for_engine` | Resolves any of the 10 RSIBox engines or aliases to its `ZygoteProfile` |
| `SpawnedChild` | Frozen child metadata (`pid`, `role`, `engine`, `incarnation`, `child_guard_report`, `affinity_report`) |
| `ZygoteService` | Thread-safe context-managed coordinator (`arm_guards`, `start`, `spawn_child`, `wait_child`, `reap_finished`, `status`, `teardown`, `close`) |

## Planned Services (`S3`–`S6`)

- **S3 (Cache):** Shares compiled `.so` side libraries, `@triton.jit` binaries, and selective CUDA-graph recipes across worker restarts without invalidating journaled undo.
- **S4 (Auxiliary):** Manages non-updated reward/reference/judge pools in RL setups (`veRL`, `miles`, `NeMo-RL`) using `S1` `ro_aux_weights` handles.
- **S5 (Buffers):** Tracks volatile step buffers (`kv_pool`, replay/rollout buffers) that must drain and reset on `reset("state")` or quiesce during snapshot capture.
- **S6 (Recovery):** Composes [`RecoveryDriver`](../core/livelab/rsibox/warm/recovery.py) (`DefaultRespawnDriver`, `StandbyShadowDriver`, `DynamoK8sSnapshotDriver`) with `S1` handle rebinding after crash or poison events.

## Public API Summary (`rsibox.services.memory`)

| Symbol | Purpose |
|---|---|
| `MemoryPoolKind` | Enum of pool kinds (`ro_weights`, `kv_pool`, `cow_s0_snapshot`, `ro_aux_weights`, `actor_rollout_shared`) |
| `MemoryHandle` | Frozen descriptor (`handle_id`, `pool_kind`, `engine`, `role`, `gpus`, `size_bytes`, `stamp`, `ipc_token`, `gms_handle`, `read_only`, `incarnation`, `refcount`) |
| `SnapshotLease` | Frozen reader/writer lease (`lease_id`, `handle_id`, `holder_id`, `epoch`, `writable`, `epoch_state_keys`) |
| `MemoryService` | Thread-safe coordinator (`attach_store`, `register_region`, `acquire_lease`, `restore_s0`, `release_lease`, `holder_died`, `share_handle`, `probe_all`, `teardown`) |

## How the 10 Engines Compose `S1`–`S6`

| Workload Family | Engines | Services Composed Beneath `core/livelab/kernel/` |
|---|---|---|
| **LLM / Diffusion Inference** | `vLLM`, `SGLang`, `TensorRT-LLM`, `FastVideo`, `Transformers` | `S1` (`ro_weights`, `kv_pool`) + `S2` zygote / `S6` shadow recovery + `S3` kernel/graph cache |
| **Pretraining / Fine-tuning** | `Megatron-LM`, `TorchTitan` | `S1` (`cow_s0_snapshot` `S0` reset) + `S3` compile cache + `S5` step/optimizer buffers |
| **Reinforcement Learning** | `veRL`, `miles`, `NeMo-RL` | `S1` (`ro_aux_weights`, `actor_rollout_shared`, `cow_s0_snapshot`) + `S4` aux pools + `S6` recovery |

See [`docs/mechanisms.md`](mechanisms.md) for the restart-boundary latency ladder and [`docs/engines.md`](engines.md) for per-engine verification summaries.
