"""Pluggable warm-state foundations (`WeightStore`) and snapshot/recovery drivers.

Engine-agnostic, pure Python stdlib kit inside `rsibox.warm`:
- `WeightStore` (`DiskStore`, `DynamoGmsStore`, registry): persistent GPU weight residency.
- `QuiesceHooks` & `RecoveryDriver` (`DefaultRespawnDriver`, `StandbyShadowDriver`,
  `DynamoK8sSnapshotDriver`): process-level recovery and `ai-dynamo/snapshot` interoperability.
"""
from . import dynamo_gms, recovery, snapshot_contract, types, weight_store
from .dynamo_gms import (
    DEFAULT_MIN_NOFILE,
    ENV_GMS_DEVICE_UUID_MAP,
    GMS_SGLANG_LOAD_FORMAT,
    GMS_TAGS,
    GMS_VLLM_LOAD_FORMAT,
    GMS_VLLM_WORKER_CLS,
    RECOMMENDED_NOFILE,
    DynamoGmsStore,
    build_device_uuid_map,
    install_visible_devices_uuid_shim,
    nofile_limit_finding,
    uninstall_visible_devices_uuid_shim,
)
from .recovery import (
    AsyncShadowReplenisher,
    DefaultRespawnDriver,
    QuiesceHooks,
    RecoveryDriver,
    StandbyShadowDriver,
)
from .snapshot_contract import (
    ENV_SNAPSHOT_CONTROL_DIR,
    SENTINEL_READY,
    SENTINEL_RESTORE_COMPLETE,
    DynamoK8sSnapshotDriver,
    SnapshotLifecycleResult,
    run_snapshot_lifecycle,
)
from .types import LaunchDelta, Mutability, QuiesceReceipt, StoreProbe, stamp_diff
from .weight_store import (
    DiskStore,
    WeightStore,
    get_weight_store,
    list_weight_stores,
    register_weight_store,
)

__all__ = [
    "AsyncShadowReplenisher",
    "DEFAULT_MIN_NOFILE",
    "DefaultRespawnDriver",
    "DiskStore",
    "DynamoGmsStore",
    "DynamoK8sSnapshotDriver",
    "ENV_GMS_DEVICE_UUID_MAP",
    "ENV_SNAPSHOT_CONTROL_DIR",
    "GMS_SGLANG_LOAD_FORMAT",
    "GMS_TAGS",
    "GMS_VLLM_LOAD_FORMAT",
    "GMS_VLLM_WORKER_CLS",
    "LaunchDelta",
    "Mutability",
    "QuiesceHooks",
    "QuiesceReceipt",
    "RECOMMENDED_NOFILE",
    "RecoveryDriver",
    "SENTINEL_READY",
    "SENTINEL_RESTORE_COMPLETE",
    "SnapshotLifecycleResult",
    "StandbyShadowDriver",
    "StoreProbe",
    "WeightStore",
    "build_device_uuid_map",
    "dynamo_gms",
    "get_weight_store",
    "install_visible_devices_uuid_shim",
    "list_weight_stores",
    "nofile_limit_finding",
    "recovery",
    "register_weight_store",
    "run_snapshot_lifecycle",
    "snapshot_contract",
    "stamp_diff",
    "types",
    "uninstall_visible_devices_uuid_shim",
    "weight_store",
]
