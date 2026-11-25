"""Composable shared infrastructure services (`rsibox.services`, stdlib only).

Provides engine-agnostic control-plane services composed beneath `core/livelab/kernel/`:
- `memory` (`S1`): Unified GPU Memory & State Service (`MemoryPoolKind`, `MemoryHandle`,
  `SnapshotLease`, `MemoryService`) unifying RO/KV weight pools, CoW `S0` snapshots, and
  Actor<->Rollout handle sharing.
"""
from __future__ import annotations

from . import memory
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

__all__ = [
    "DEFAULT_S0_STATE_KEYS",
    "HandleNotFoundError",
    "LeaseError",
    "MemoryHandle",
    "MemoryPoolKind",
    "MemoryService",
    "MemoryServiceError",
    "PoolAccessError",
    "READ_ONLY_POOLS",
    "SnapshotLease",
    "StampMismatchError",
    "memory",
]
