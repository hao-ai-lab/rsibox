# RSIBox Concepts

`rsibox` is a live hot-reload sandbox for ML engines. Instead of paying a multi-minute cold restart on every edit to reload model weights and re-capture CUDA graphs, it swaps changes directly into the running engine, proves the new code ran, and rolls back bit-for-bit on undo.

## Product vs. Runtime: `rsibox` and `core/livelab`

- **`rsibox` (CLI & HTTP box):** The user-facing package, CLI (`rsibox up`, `install`, `check`, `uninstall`), and local HTTP daemon that coding agents and developers drive.
- **`core/livelab/` (internal runtime):** The engine-agnostic standard-library runtime inside this repository. Every engine adapter under `repos/<engine>/livelab/` implements the shared five-verb contract in [`core/livelab/INTERFACE.md`](../core/livelab/INTERFACE.md): `install`, `check`, `reset`, `uninstall`, and `status`.

## 4-Layer Architecture & Design Rationale

| Layer | What It Owns | Code Path | Key Boundary |
|---|---|---|---|
| **1. Control Plane** | CLI/HTTP Box, `SandboxPool` scheduling & circuit breaker, `PoolAutoScaler`, `BaseLab` | `core/livelab/rsibox/` | Routes agent calls to warm slots via `LabUnderTest` |
| **2. Hot-Reload Kernel** | 2PC `Coordinator` & `Session`, `swap`/`jit`, `witness` & `Journal`, `findings`/`identity` | `core/livelab/kernel/` | Pure stdlib; step-gap in-place swap & exact LIFO undo |
| **3. Warm Recovery & Substrate** | `forksafe`, JSON IPC, Zygote + CUDA IPC holder, Dynamo GMS Shadow & CRIU hooks | `core/livelab/rsibox/warm/`, `core/livelab/forksafe.py` | Process-boundary failover while keeping GPU weights resident |
| **4. Thin Engine Adapters** | Engine-specific graph/worker hooks for 10 serving, training, video, and RL engines | `repos/<engine>/livelab/` | Implements `LabUnderTest` (plus optional `QuiesceHooks`); zero cross-engine imports |
| **Cross-Cutting Plane** | TLC-checked TLA+ specs, `c0-c10` verification suite, and `rstack` PR/CI gates | `core/livelab/{formal,conformance}/`, `.agents/plugins/rstack/` | Bounds protocol safety, adapter conformance, and merge hygiene |

Rather than asking production engine teams to rewrite schedulers, model loaders, or CUDA-graph pipelines from scratch, RSIBox wraps all 10 engines behind an external stdlib runtime (`LabUnderTest`) paired with per-rank execution proof (`witness`) and a tiered warm-recovery ladder—letting teams adopt live sub-second iteration and multi-agent GPU sharing with a thin, isolated adapter. Explore each component and step-by-step sequence flow interactively in [`docs/architecture.html`](architecture.html).

## Hot-Reload Layers (Cheapest First)

RSIBox applies each edit at the lowest layer that can safely update what the hardware actually runs:

1. **Live config & Python bytecode (`~4–90 ms`):** Update runtime scheduler flags in process or atomically swap Python function code objects across all GPU ranks between steps.
2. **Triton kernels (`@triton.jit`):** Rebuild modified Triton kernels and any callers that inline them, while keeping the prior binaries cached for instant undo.
3. **Selective CUDA graph recapture (`~0.5–3.7 s`):** Re-capture only the CUDA graphs touched by an edit instead of restarting the worker process.
4. **C++/CUDA (`.cu`) side-library reload (`~7–19 s`):** Compile edited native kernels into a versioned side library, rebind PyTorch ops, and recapture affected graphs.
5. **Pre-warmed fork server (`~5–20 s`):** Respawn from a pre-imported zygote server when an edit changes launch flags or startup code, keeping weights in GPU memory when paired with a CUDA IPC weight holder.
6. **Dynamo GMS & Snapshot fallback (`~0.3–7 s`):** When a worker crashes or is poisoned, promote a parked NVIDIA Dynamo GPU Memory Service (GMS) standby shadow or restore a quiesced CRIU + `cuda-checkpoint` snapshot so weights stay resident in GPU memory.

## How a Live Edit Flows

Every live edit runs as an atomic two-phase transaction across all worker ranks:

1. **Stage (while the engine runs):** Parse the edited module, verify the live base matches, and check for unsupported changes without interrupting active inference or training steps.
2. **Commit (in a step gap):** Pause briefly between steps so every rank swaps together—or aborts together if any rank hits a conflict.
3. **Verify & undo:** Run `rsibox check` to confirm output and execution proof, then keep the change or call `rsibox uninstall` to revert cleanly.

## Execution Proof (`witness`)

Swapping a function in memory does not prove the workload ever executed it. Every live swap inserts a lightweight call counter (`witness`) on every rank so `rsibox check` distinguishes real execution (`proven: True`) from unreachable or startup-only code (`proven: False` with `zero_witness`).

## Journaled Rollback

Before touching live state, each rank records the exact inverse of every mutation in an ordered journal. Calling `rsibox uninstall` replays the journal in reverse order—restoring original Python code objects, cached Triton binaries, and CUDA graphs without recompiling or restarting.

## Restart Boundaries

Long-lived state (constructors, module-level constants, changed closures, or launch flags) cannot be patched safely in place. RSIBox checks every patch before mutating state, refuses unsupported edits fail-closed, and names the cheapest restart boundary that handles them. See [`docs/mechanisms.md`](mechanisms.md) and [`docs/kernels-and-graphs.md`](kernels-and-graphs.md).

## Sandbox Verification Suite

Every engine adapter must pass the sandbox verification suite ([`core/livelab/conformance/README.md`](../core/livelab/conformance/README.md)) before its measurements count:

- **Repeatability (`c0`):** Repeated checks after state resets produce identical outputs.
- **Execution & fresh parity (`c1`–`c3`, `c10`):** Edits execute on every rank, match a fresh process started from the edited files, and report `zero_witness` when code never runs.
- **Safe refusal & rollback (`c4`–`c7`, `c9`):** Unsupported or mid-fault edits refuse or roll back cleanly, and repeated install/undo cycles restore exact base code identity.
- **Fault detection & recovery (`c8`):** Crashed, hung, lost, or tampered workers mark the sandbox `poisoned` and block edits until respawned.

## Multi-Agent GPU Pool

When multiple coding agents share a small GPU budget, `SandboxPool` multiplexes concurrent workers across warm slots with priority scheduling, elastic role borrowing, per-agent fault isolation, and hash-chained trace logs (`112/112` verified traces; see [`docs/agent-teams-and-contention.md`](agent-teams-and-contention.md)).

## Where to Go Next

- [`docs/engines.md`](engines.md): supported ML engines, measured speedups, and repository layout.
- [`docs/mechanisms.md`](mechanisms.md): full restart-boundary table, Dynamo GMS + Snapshot benchmarks, and CPU test commands.
- [`docs/kernels-and-graphs.md`](kernels-and-graphs.md): deep dive into bytecode swaps, Triton kernel rebuilds, and CUDA graph replay.

