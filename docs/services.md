# RSIBox Shared Services (`rsibox.services`)

`core/livelab/rsibox/services/` houses pure-stdlib, engine-agnostic control-plane services that sit beneath [`core/livelab/kernel/`](../core/livelab/INTERFACE.md) and compose across all ten supported engines (`vLLM`, `SGLang`, `Megatron-LM`, `FastVideo`, `veRL`, `miles`, `Transformers`, `TensorRT-LLM`, `TorchTitan`, and `NeMo-RL`).

Instead of each `repos/<engine>/livelab/services/` adapter reinventing weight residency, fork servers, artifact caches, or crash recovery, shared services standardize ownership, lease fencing, and staleness-stamp contracts while keeping `core/livelab/` free of engine or torch imports.

## Architecture Layering

```text
core/livelab/kernel/          (two-phase txns, code swap, JIT rebuild, witness, journal)
        │
        ▼
core/livelab/rsibox/services/ (S1 Memory [shipped] · S2 Zygote [shipped] · S3 Cache [shipped] · S4 Aux [shipped] · S5 Buffers [shipped] · S6 Recovery [shipped])
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
| **S3 Cache** | Triton, Inductor, FlashInfer & CUDA-graph artifact store | **Shipped** ([`cache.py`](../core/livelab/rsibox/services/cache.py)) | `CompileCacheService`, `EnvFingerprint`, `CacheEntry`; per-entry `flock` + atomic `os.replace` + SHA-256 verification |
| **S4 Auxiliary** | Shared frozen VAE, text encoder, reward, reference, and judge stage pool | **Shipped** ([`auxiliary.py`](../core/livelab/rsibox/services/auxiliary.py)) | `AuxiliaryModelService`, `AuxStageKind`, `AuxStageSpec`, `AuxInvocationReceipt`; S1 `ro_aux_weights` RO leases + LRU cache |
| **S5 Buffers** | Indexed mmap prefetch & `code_version`-fenced RL replay buffers | **Shipped** ([`buffers.py`](../core/livelab/rsibox/services/buffers.py)) | `IndexedPrefetchService`, `VersionedReplayBufferService`, `PrefetchItem`, `VersionedEntry`, `BufferResetReceipt` |
| **S6 Recovery** | Coordinated fault detection, shadow failover, and snapshot restore | **Shipped** ([`recovery_service.py`](../core/livelab/rsibox/services/recovery_service.py)) | `RecoveryService`, `RecoveryTier`, `ServiceHealth`, `RecoveryReceipt`; `RecoveryDriver` + `QuiesceHooks` + `SandboxPool` |

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

## S3: Shared Compile & Graph Cache Service (Shipped)

[`core/livelab/rsibox/services/cache.py`](../core/livelab/rsibox/services/cache.py) provides a fail-closed, cross-process and cross-container artifact cache behind `CompileCacheService`:

- **Environment fingerprint binding:** `EnvFingerprint` hashes `gpu_arch`, `cuda_version`, `triton_version`, `torch_version`, `driver_version`, `engine`, and `extra` into a canonical SHA-256 `digest()`. Mismatches raise `FingerprintMismatchError` under `strict=True` or treat the entry as a cache miss (`None`) under `strict=False`.
- **Per-entry `flock` + atomic replace:** `put()` writes payload and sidecar JSON manifest to temporary files and publishes them via `os.replace` under an exclusive per-entry `fcntl.flock`; `get()` verifies manifest integrity, payload length, and SHA-256 checksum under a shared `flock`, raising `CacheCorruptError` (`strict=True`) or evicting (`strict=False`) on tampering.

### Public API Summary (`rsibox.services.cache`)

| Symbol | Purpose |
|---|---|
| `CacheArtifactKind` | Artifact category enum (`triton_kernel`, `torch_inductor`, `flashinfer`, `cuda_graph_plan`, `custom`) |
| `EnvFingerprint` | Frozen GPU/CUDA/Triton/Torch ABI fingerprint with `to_dict()`, `from_dict()`, and `digest()` |
| `CacheEntry` | Frozen entry descriptor (`namespace`, `key`, `kind`, `fingerprint_digest`, `payload_path`, `size_bytes`, `sha256`, `created_at`, `metadata`) |
| `CompileCacheService` | Thread-safe, context-managed disk cache (`put`, `get`, `invalidate`, `list_entries`, `stats`, `teardown`, `close`) |

## S4: Shared Auxiliary & Frozen Stage Pool (Shipped)

[`core/livelab/rsibox/services/auxiliary.py`](../core/livelab/rsibox/services/auxiliary.py) manages non-updated auxiliary models (`vae`, `text_encoder`, `reward_model`, `reference_model`, `judge`, `custom`) shared across concurrent inference and RL workers behind `AuxiliaryModelService`:

- **Read-only S1 weight fencing:** `register_stage()` rejects writable handles and non-RO pools fail-closed (`read_only=True` in `ro_aux_weights` or `ro_weights`), while `invoke()` acquires a per-call RO `SnapshotLease` with `expected_stamp` validation and releases it in `finally:` even when `infer_fn` raises.
- **Lock-free inference & bounded LRU caching:** Internal locks are released across blocking `infer_fn(payload, context)` calls, results are cached per `(stage_id, version, input_digest)` in a bounded `OrderedDict` LRU cache, and `bump_version()` / `unregister_stage()` atomically invalidate cached entries and run stage `close_fn()` hooks outside locks.

### Public API Summary (`rsibox.services.auxiliary`)

| Symbol | Purpose |
|---|---|
| `AuxStageKind` | Enum of frozen stage categories (`vae`, `text_encoder`, `reward_model`, `reference_model`, `judge`, `custom`) |
| `AuxStageSpec` | Frozen stage descriptor (`stage_id`, `kind`, `engine`, `model_id`, `version`, `handle_id`, `gpus`, `max_cache_entries`) |
| `AuxInvocationReceipt` | Frozen invocation receipt (`stage_id`, `version`, `caller_id`, `input_digest`, `cache_hit`, `cost_s`, `lease_id`, `output`) |
| `AuxiliaryModelService` | Thread-safe context-managed stage pool (`register_stage`, `invoke`, `bump_version`, `unregister_stage`, `list_stages`, `get_stage`, `stats`, `teardown`, `close`) |

## S5: Indexed Prefetch & Versioned RL Replay Buffers (Shipped)

[`core/livelab/rsibox/services/buffers.py`](../core/livelab/rsibox/services/buffers.py) provides two thread-safe, context-managed buffer services with zero leaked file descriptors or `mmap` handles:

- **Indexed dataset prefetch (`IndexedPrefetchService`):** Builds atomic headered binary offset + SHA-256-prefix indexes (`RSIIDX01`), maps non-empty datasets read-only via `mmap.ACCESS_READ` (supporting empty 0-record files cleanly), verifies slice lengths/checksums fail-closed (`IndexCorruptError`), and checkpoints/restores sequential cursors atomically.
- **Versioned RL replay (`VersionedReplayBufferService`):** Bounded `deque` rollout buffer tagging every entry with `code_version`, filtering `sample()` to the active version unless `allow_stale=True`, enforcing `reject_stale_push` / `drain_on_version_mismatch`, and returning `BufferResetReceipt` on `advance_code_version()` and `reset_state()`.

### Public API Summary (`rsibox.services.buffers`)

| Symbol | Purpose |
|---|---|
| `PrefetchItem` | Frozen dataset record (`index`, `offset`, `length`, `payload`) |
| `VersionedEntry` | Frozen rollout entry (`seq`, `code_version`, `payload`, `metadata`) |
| `BufferResetReceipt` | Frozen reset/promotion receipt (`drained`, `retained`, `active_code_version`, `epoch`) |
| `IndexedPrefetchService` | Thread-safe mmap reader (`build_index`, `open`, `get`, `seek`, `next_batch`, `checkpoint_cursor`, `restore_cursor`, `close`) |
| `VersionedReplayBufferService` | Thread-safe versioned replay buffer (`push`, `sample`, `advance_code_version`, `reset_state`, `stats`, `close`) |

## S6: Unified Recovery & Health Service (Shipped)

[`core/livelab/rsibox/services/recovery_service.py`](../core/livelab/rsibox/services/recovery_service.py) unifies process recovery, snapshot quiesce handshakes, and pool/autoscaler health aggregation behind `RecoveryService`:

- **Single-flight per-endpoint recovery:** `recover()` fences concurrent recovery on the same `endpoint_id` (`RecoveryInProgressError`) while allowing distinct endpoints to recover in parallel and releasing internal locks across blocking driver I/O.
- **Deterministic tier escalation & quarantine:** Primary failures (`StandbyShadowDriver` or `DynamoK8sSnapshotDriver`) escalate cleanly to `fallback_driver` without losing `last_error`, increment `incarnation`, and transition repeatedly failing endpoints to `quarantined` (or raise `EscalationFailedError` if all tiers fail).
- **Pool aggregation & owner-scoped teardown:** `attach_pool()` aggregates `SandboxPool.slot_states()`, `SandboxPool.metrics()`, and `PoolAutoScaler.status()` in `health_snapshot()` / `probe_health()`, and `teardown()` stops owned pools, autoscalers, drivers, and stores outside locks with fail-closed post-close guards.

### Public API Summary (`rsibox.services.recovery_service`)

| Symbol | Purpose |
|---|---|
| `RecoveryTier` | Enum of recovery tiers (`cold_respawn`, `standby_shadow`, `snapshot_restore`) |
| `ServiceHealthStatus` | Lifecycle status enum (`healthy`, `degraded`, `recovering`, `quarantined`, `closed`) |
| `ServiceHealth` | Frozen endpoint health snapshot (`status`, `incarnation`, `consecutive_failures`, `last_error`, `recovery_count`) |
| `RecoveryReceipt` | Frozen outcome record (`tier_attempted`, `tier_succeeded`, `ok`, `fallback_used`, `cost_s`, `incarnation`, `driver_receipt`) |
| `RecoveryService` | Thread-safe coordinator (`register_endpoint`, `attach_pool`, `record_failure`, `recover`, `quiesce_and_checkpoint`, `get_health`, `health_snapshot`, `probe_health`, `teardown`, `close`) |

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
