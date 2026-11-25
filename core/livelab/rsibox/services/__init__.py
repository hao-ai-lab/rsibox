"""Composable shared infrastructure services (`rsibox.services`, stdlib only).

Provides engine-agnostic control-plane services composed beneath `core/livelab/kernel/`:
- `memory` (`S1`): Unified GPU Memory & State Service (`MemoryPoolKind`, `MemoryHandle`,
  `SnapshotLease`, `MemoryService`) unifying RO/KV weight pools, CoW `S0` snapshots, and
  Actor<->Rollout handle sharing.
- `zygote` (`S2`): Shared Zygote & Pre-Fork Daemon Service (`ZygoteProfile`,
  `SpawnedChild`, `ZygoteState`, `ZygoteService`, `profile_for_engine`, `list_profiles`)
  unifying pre-fork `forksafe` guards and `AF_UNIX` + `SCM_RIGHTS` worker spawning.
- `cache` (`S3`): Shared Compile & Graph Cache Service (`CacheArtifactKind`, `EnvFingerprint`,
  `CacheEntry`, `CompileCacheService`) providing fail-closed cross-process artifact reuse.
- `auxiliary` (`S4`): Shared Auxiliary & Frozen Stage Pool (`AuxiliaryModelService`,
  `AuxStageKind`, `AuxStageSpec`, `AuxInvocationReceipt`) managing frozen VAE, text
  encoder, reward, reference, and judge stages with S1 RO weight leases and LRU caching.
- `buffers` (`S5`): Indexed Prefetch & Versioned RL Replay Buffers (`IndexedPrefetchService`,
  `VersionedReplayBufferService`, `PrefetchItem`, `VersionedEntry`, `BufferResetReceipt`).
- `recovery_service` (`S6`): Unified Recovery & Health Service (`RecoveryService`,
  `RecoveryTier`, `ServiceHealthStatus`, `ServiceHealth`, `RecoveryReceipt`) unifying
  `RecoveryDriver` escalation, `QuiesceHooks` checkpoints, and `SandboxPool` health.
"""
from __future__ import annotations

from . import auxiliary, buffers, cache, memory, recovery_service, zygote
from .auxiliary import (
    AuxClosedError,
    AuxiliaryModelService,
    AuxInvocationError,
    AuxInvocationReceipt,
    AuxServiceError,
    AuxStageKind,
    AuxStageSpec,
    StageExistsError,
    StageNotFoundError,
)
from .buffers import (
    BufferClosedError,
    BufferResetReceipt,
    BufferServiceError,
    CursorOutOfRangeError,
    EmptyBufferError,
    IndexCorruptError,
    IndexedPrefetchService,
    PrefetchItem,
    StaleVersionError,
    VersionedEntry,
    VersionedReplayBufferService,
)
from .cache import (
    CacheArtifactKind,
    CacheClosedError,
    CacheCorruptError,
    CacheEntry,
    CompileCacheError,
    CompileCacheService,
    EnvFingerprint,
    FingerprintMismatchError,
)
from .memory import (
    DEFAULT_S0_STATE_KEYS,
    READ_ONLY_POOLS,
    HandleNotFoundError,
    LeaseError,
    MemoryHandle,
    MemoryPoolKind,
    MemoryService,
    MemoryServiceError,
    PoolAccessError,
    SnapshotLease,
    StampMismatchError,
)
from .recovery_service import (
    EndpointNotFoundError,
    EscalationFailedError,
    RecoveryClosedError,
    RecoveryInProgressError,
    RecoveryReceipt,
    RecoveryService,
    RecoveryServiceError,
    RecoveryTier,
    ServiceHealth,
    ServiceHealthStatus,
)
from .zygote import (
    ForkPoisonedError,
    SpawnedChild,
    SpawnError,
    ZygoteClosedError,
    ZygoteError,
    ZygoteProfile,
    ZygoteService,
    ZygoteState,
    list_profiles,
    profile_for_engine,
)

__all__ = [
    "AuxClosedError",
    "AuxInvocationError",
    "AuxInvocationReceipt",
    "AuxServiceError",
    "AuxStageKind",
    "AuxStageSpec",
    "AuxiliaryModelService",
    "BufferClosedError",
    "BufferResetReceipt",
    "BufferServiceError",
    "CacheArtifactKind",
    "CacheClosedError",
    "CacheCorruptError",
    "CacheEntry",
    "CompileCacheError",
    "CompileCacheService",
    "CursorOutOfRangeError",
    "DEFAULT_S0_STATE_KEYS",
    "EmptyBufferError",
    "EndpointNotFoundError",
    "EnvFingerprint",
    "EscalationFailedError",
    "FingerprintMismatchError",
    "ForkPoisonedError",
    "HandleNotFoundError",
    "IndexCorruptError",
    "IndexedPrefetchService",
    "LeaseError",
    "MemoryHandle",
    "MemoryPoolKind",
    "MemoryService",
    "MemoryServiceError",
    "PoolAccessError",
    "PrefetchItem",
    "READ_ONLY_POOLS",
    "RecoveryClosedError",
    "RecoveryInProgressError",
    "RecoveryReceipt",
    "RecoveryService",
    "RecoveryServiceError",
    "RecoveryTier",
    "ServiceHealth",
    "ServiceHealthStatus",
    "SnapshotLease",
    "SpawnError",
    "SpawnedChild",
    "StageExistsError",
    "StageNotFoundError",
    "StaleVersionError",
    "StampMismatchError",
    "VersionedEntry",
    "VersionedReplayBufferService",
    "ZygoteClosedError",
    "ZygoteError",
    "ZygoteProfile",
    "ZygoteService",
    "ZygoteState",
    "auxiliary",
    "buffers",
    "cache",
    "list_profiles",
    "memory",
    "profile_for_engine",
    "recovery_service",
    "zygote",
]
