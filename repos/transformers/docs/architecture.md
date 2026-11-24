# HuggingFace Transformers Live-Lab Architecture

## Process Tree

Unlike multi-process serving engines (vLLM, SGLang) or distributed RL/training frameworks,
HuggingFace `transformers` runs model loading, tokenization, and autoregressive generation inside a
single Python process. `TransformersLab` isolates that stateful runtime in one persistent warm
worker subprocess managed over an `AF_UNIX` newline-delimited JSON channel (`rsibox.ipc.UnixJsonServer`):

```text
Host process (TransformersLab, BaseLab + RecaptureCoordinator)
  └── AF_UNIX JSON socket (rsibox.ipc.short_sock_path("rsitf"))
        └── Worker subprocess (python -m adapter.agent <spec.json>, own session)
              ├── AutoTokenizer + AutoModelForCausalLM (resident on target CUDA/CPU device)
              ├── kernel.coordinator.LocalParticipant("worker0")
              └── Main-thread queue-mode step-gap loop (one RPC or generate step at a time)
```

## Why a Queue-Mode Main-Thread Step-Gap Loop

The worker runs `rsibox.ipc.UnixJsonServer(sock_path, handler=None)` and drains requests serially on
its main thread via `server.next(timeout=1.0)`:

1. **Guaranteed step-gap atomicity.** `prepare`, `validate`, `commit`, `abort`, `undo`, `reset`,
   `quiesce`, and `resume` never race with an in-flight `model.generate()` call or active CUDA
   kernels.
2. **Automatic cache/RNG hygiene across edits.** `TransformersLab` uses `RecaptureCoordinator` with
   `post_commit_reset="state"` and `undo_resets=["state"]`, flushing KV/CUDA allocator caches and
   resetting Python/PyTorch RNG seeds after every commit and undo.
3. **Zero eager GPU imports in the host.** Importing `adapter`, `adapter.lab`, `adapter.toy_lab`,
   or `adapter.snapshot_hooks` never imports `torch` or `transformers`; both load lazily inside the
   worker subprocess.
4. **Automatic model-architecture tracking.** On startup handshake (`ping`), the worker reports
   `type(model).__module__` (for example `transformers.models.qwen3.modeling_qwen3`), which the host
   appends to the session's fingerprinted module set alongside `transformers.generation.logits_process`,
   `transformers.generation.utils`, and `transformers.modeling_utils`.
5. **Orphan & pipe-deadlock safety.** Worker stderr is routed to a managed temp log file (never an
   unread pipe), the worker runs in its own session (`start_new_session=True`, stopped via
   `rsibox.proc.stop_session` / `stop_tree`), and the worker exits cleanly if its parent PID changes.

## Restart Boundaries and Edit Routing

Timings below are from our 2026-11-24 B200 run on `Qwen3-8B` (`bfloat16`); see
[../kb/costs.md](../kb/costs.md) for full 3-repeat tables across both `Qwen3-0.6B` and `Qwen3-8B`.

| Path | Mechanism | Scope | Measured Cost (B200, Qwen3-8B) |
|---|---|---|---|
| **Live in-process edit** | `LocalParticipant` atomic `__code__` swap + AST witness + `reset("state")` | Python functions/methods in `transformers.generation.*`, `transformers.modeling_utils`, and `transformers.models.*` | `0.119–0.270 s` install, `0.083–0.091 s` undo (`0.605 s` install+check) |
| **Snapshot quiesce/resume** | `TransformersQuiesceHooks` + `DynamoK8sSnapshotDriver` (CPU pinned offload + DMA restore) | Container/pod checkpoint-restore (`ai-dynamo/snapshot` sentinel contract) | `8.76–15.64 s` quiesce (frees `15,740–15,778 MiB`), `0.310 ± 0.011 s` H2D resume |
| **Cold respawn** | `reset("respawn")` via `DefaultRespawnDriver` or configured `RecoveryDriver` | Process crash, hang, lost socket, external tamper, or startup-flag change | `7.746 ± 1.536 s` cold worker boot + first check (`9.508 s` cold page cache) |

## Snapshot Quiesce/Resume Lifecycle (`TransformersQuiesceHooks`)

`TransformersQuiesceHooks` implements `rsibox.warm.QuiesceHooks` (`mutability = "readonly_weights"`)
so `TransformersLab` plugs directly into `rsibox.warm.DynamoK8sSnapshotDriver`:

1. **`warmup_for_snapshot()`**: Holds `lab.lock`, verifies worker liveness and non-poisoned status
   (including live fingerprint verification against external tamper), runs one deterministic warmup
   generation step, and probes the worker control channel.
2. **`quiesce_for_snapshot()`**: Holds `lab.lock`, verifies non-poisoned status, clears cached
   state, moves model parameters and buffers to CPU memory (pinning contiguous unaliased tensors
   without breaking weight tying such as shared `embed_tokens`/`lm_head` parameters), runs
   `gc.collect()` + `torch.cuda.synchronize(device)` + `torch.cuda.empty_cache()` +
   `torch.cuda.ipc_collect()`, marks the worker quiesced (refusing `generate`/`prepare`/`commit`
   until resumed), and verifies the worker channel is still alive. Fails closed with best-effort
   rollback if post-offload operations or probing fail, never writing `ready-for-snapshot` on error.
3. **`resume_after_restore(receipt)`**: Rejects failed receipts immediately, holds `lab.lock`,
   restores parameters and buffers onto the exact target CUDA device with `non_blocking=True` +
   `torch.cuda.synchronize(device)`, verifies worker channel `ping`, and checks that `lab.status()`
   is non-poisoned with a valid code fingerprint before `transformers-restore-ready` is written.

## Fault Containment and Conformance Verification

- **CPU conformance (`ToyTransformersLab`)**: Passes all 11/11 RSIBox conformance cases (`c0–c10`),
  including `fail_apply`/`fail_commit` atomic rollback (`c5`), all 4 refusal classes (`c6`),
  `crash`/`hang`/`lost`/`tamper` poison + respawn recovery (`c8`), 20-cycle soak (`c9`), and
  unreached startup-only edit detection (`c10`).
- **Real GPU verification (`TransformersLab`)**: Verified on 1× NVIDIA B200 (`Qwen3-0.6B` and
  `Qwen3-8B`, our 2026-11-24 B200 run) across 3 real `transformers` modules (`generation.utils`,
  `generation.logits_process`, `models.qwen3.modeling_qwen3`), fresh `PackageOverlay` parity, and
  `DynamoK8sSnapshotDriver` quiesce/resume + post-resume live edits.
