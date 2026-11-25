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
- `recovery_service` (`S6`): Unified Recovery & Health Service (`RecoveryService`,
  `RecoveryTier`, `ServiceHealthStatus`, `ServiceHealth`, `RecoveryReceipt`) unifying
  `RecoveryDriver` escalation, `QuiesceHooks` checkpoints, and `SandboxPool` health.
"""
from __future__ import annotations

from . import cache, memory, recovery_service, zygote
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
    "CacheArtifactKind",
    "CacheClosedError",
    "CacheCorruptError",
    "CacheEntry",
    "CompileCacheError",
    "CompileCacheService",
    "DEFAULT_S0_STATE_KEYS",
    "EndpointNotFoundError",
    "EnvFingerprint",
    "EscalationFailedError",
    "FingerprintMismatchError",
    "ForkPoisonedError",
    "HandleNotFoundError",
    "LeaseError",
    "MemoryHandle",
    "MemoryPoolKind",
    "MemoryService",
    "MemoryServiceError",
    "PoolAccessError",
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
    "StampMismatchError",
    "ZygoteClosedError",
    "ZygoteError",
    "ZygoteProfile",
    "ZygoteService",
    "ZygoteState",
    "cache",
    "list_profiles",
    "memory",
    "profile_for_engine",
    "recovery_service",
    "zygote",
]
