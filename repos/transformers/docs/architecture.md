# HuggingFace Transformers Live-Lab Architecture

## Process Tree

Unlike multi-process serving engines (vLLM, SGLang) or distributed RL/training frameworks,
HuggingFace `transformers` runs model loading, tokenization, and autoregressive generation inside a
single Python process. `TransformersLab` isolates that stateful runtime in one persistent warm
worker subprocess managed over an `AF_UNIX` newline-delimited JSON channel (`rsibox.ipc.UnixJsonServer`):

```text
Host process (TransformersLab, BaseLab + RecaptureCoordinator)
  └── AF_UNIX JSON socket (rsibox.ipc.short_sock_path("rsitf"))
        └── Worker subprocess (python -m adapter.agent <spec.json>)
              ├── AutoTokenizer + AutoModelForCausalLM (resident on target CUDA/CPU device)
              ├── kernel.coordinator.LocalParticipant("worker0")
              └── Main-thread queue-mode step-gap loop (one RPC or generate step at a time)
```

## Why a Queue-Mode Main-Thread Step-Gap Loop

The worker runs `rsibox.ipc.UnixJsonServer(sock_path, handler=None)` and drains requests serially on
its main thread via `server.next()`:

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
   appends to the session's fingerprinted module set alongside `transformers.generation.logits_process`
   and `transformers.generation.utils`.

## Restart Boundaries and Edit Routing

| Tier | Mechanism | Scope | Typical Cost (B200, Qwen3-8B) |
|---|---|---|---|
| **T0 (R0 swap)** | `LocalParticipant` atomic `__code__` swap + AST witness + `reset("state")` | Python functions/methods in `transformers.generation.*` and `transformers.models.*` | `0.12–0.27 s` install, `0.08–0.09 s` undo |
| **T1 (Snapshot resume)** | `TransformersQuiesceHooks` + `DynamoK8sSnapshotDriver` (CPU pinned offload + DMA restore) | Container/pod checkpoint-restore (`ai-dynamo/snapshot` sentinel contract) | `0.31 s` H2D weight restore + channel/fingerprint verification |
| **T2 (Respawn)** | `reset("respawn")` via `DefaultRespawnDriver` or configured `RecoveryDriver` | Process crash, hang, lost socket, external tamper, or startup-flag change | `6.7–9.5 s` cold worker boot + first check |

## Snapshot Quiesce/Resume Lifecycle (`TransformersQuiesceHooks`)

`TransformersQuiesceHooks` implements `rsibox.warm.QuiesceHooks` (`mutability = "readonly_weights"`)
so `TransformersLab` plugs directly into `rsibox.warm.DynamoK8sSnapshotDriver`:

1. **`warmup_for_snapshot()`**: Runs one deterministic warmup generation step so lazy CUDA contexts
   and cuBLAS workspaces are initialized, then probes the worker control channel.
2. **`quiesce_for_snapshot()`**: Clears cached state, moves model parameters and buffers to CPU
   memory (pinning contiguous unaliased tensors without breaking weight tying such as shared
   `embed_tokens`/`lm_head` parameters), runs `gc.collect()` + `torch.cuda.synchronize()` +
   `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`, and verifies the worker channel is still
   alive before returning `QuiesceReceipt(ok=True, ...)`. Fails closed without writing
   `ready-for-snapshot` on any error.
3. **`resume_after_restore(receipt)`**: Transfers model weights back onto the target CUDA device
   with `non_blocking=True` + `torch.cuda.synchronize()`, verifies worker channel `ping`, and checks
   that `lab.status()` is non-poisoned with a valid code fingerprint before `transformers-restore-ready`
   is written.

## Fault Containment

`TransformersLab` supports the full conformance fault surface (`c0–c10`):

- **`fail_apply` / `fail_commit`**: Delegated to `rsibox.harness.inject_participant_fault`; the
  coordinator rolls back atomically to base code identity and base output.
- **`crash` / `hang` / `lost` / `tamper`**: Mark the control channel dirty and poison the session;
  subsequent `install()` calls fail closed with `session_poisoned` until `reset("respawn")` starts a
  clean worker.
- **`fresh_check(patch, workload)`**: Builds a symlink `PackageOverlay` over `transformers` (or
  `adapter` in toy mode) and runs `adapter.fixture.fresh` in an isolated subprocess for bit-exact
  live-vs-fresh equivalence checks (`c3` and `c10`).
