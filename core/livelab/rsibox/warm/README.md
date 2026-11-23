# `rsibox.warm`: Pluggable Warm Foundations & Snapshot Contracts

Pure Python stdlib (zero `torch`, engine, or Kubernetes imports) runtime layer separating
**sub-second live code editing** (`rsibox.lab` / `kernel`) from **process-level weight
residency and crash/snapshot recovery**:

```
  [ Agent / Conformance / RL Verifier Pool ]  <-- INTERFACE v0 five verbs only
                      │
                   BaseLab
           ├── Live-edit plane (0.05–2.4 s): Session + RecaptureCoordinator
           ├── QuiesceHooks: warmup / quiesce / resume
           └── rsibox.warm
                ├── WeightStore    (none | ipc_cache | dynamo_gms)
                └── RecoveryDriver (default_respawn | standby_shadow | dynamo_k8s_snapshot)
```

## Components

| Module | Owns |
|---|---|
| `types.py` | `LaunchDelta`, `StoreProbe`, `QuiesceReceipt`, `Mutability`, `stamp_diff`. |
| `weight_store.py` | `WeightStore` protocol, `DiskStore` (`"none"`), and backend registry. |
| `dynamo_gms.py` | Optional `DynamoGmsStore` (`"dynamo_gms"`) for NVIDIA Dynamo's `gpu_memory_service` (CUDA VMM `gms_<uuid>_{weights,kv_cache}.sock`, `AF_UNIX` cap, stamp staleness guard, `LaunchDelta` for vLLM/SGLang/FastVideo). |
| `recovery.py` | `QuiesceHooks`, `RecoveryDriver`, `DefaultRespawnDriver`, `StandbyShadowDriver`. |
| `snapshot_contract.py` | `DynamoK8sSnapshotDriver` and `run_snapshot_lifecycle()` implementing `ai-dynamo/snapshot`'s 3-sentinel `$SNAPSHOT_CONTROL_DIR` Workload Contract (`ready-for-snapshot` -> `restore-complete` -> `<engine>-restore-ready`) for any RSIBox engine with `QuiesceHooks`. |

## Design invariants

1. **Zero hard coupling to Dynamo or Kubernetes.** RSIBox keeps its built-in lightweight warm
   paths across all 6 engines; Dynamo GMS and `ai-dynamo/snapshot` are opt-in backends.
2. **Single owner per state.** `WeightStore` owns resident GPU weights and staleness stamps;
   `RecoveryDriver` owns respawn/failover/snapshot handshakes; `engine.py` checks `LaunchDelta`
   conflicts via structured `findings` before launching anything.
3. **Composable worker hooks in vLLM.** Dynamo's `GMSWorker` uses `--worker-cls`, while
   RSIBox's live-edit `KernelWorkerExtension` uses `--worker-extension-cls`, so both run
   together with zero upstream vLLM changes.
