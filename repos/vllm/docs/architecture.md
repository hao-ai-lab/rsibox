# vLLM architecture: process tree, restart boundaries, costs

This page is the map a reviewer needs for any vLLM live-lab PR: what runs where, which box a
change invalidates, and what restarting that box costs. Numbers come from `../kb/costs.md`
(source named there) unless marked **est.** (estimated, not measured).

## 1. Process tree (vLLM V1)

```
API server                      HTTP, tokenization, streaming; one process
  |  starts
  v
engine core                     scheduler, KV block manager, structured output, busy loop
  |  starts (executor)
  v
executor                        lives inside the engine core process
  |  starts (worker group)
  v
worker group                    one process per rank (TP/PP/EP)
   +-- worker rank 0 -- model, sampler, attention backends, KV tensors, CUDA graphs
   +-- worker rank 1
   +-- ...
```

Each arrow is "starts and owns". A parent's restart kills its children: restarting the API
server restarts the engine core and everything below it, and restarting the engine core kills
the workers. A worker dying takes the engine core down with it (the executor reports the
failure and the engine exits).

Native ops are a side library loaded into each worker, not a process. Under data parallelism
the tree repeats per DP rank behind one API server.

## 2. Init chains

Worker init, per rank, in order:

| Step | What it does |
|---|---|
| `init_device` | pick the GPU, set up NCCL and parallel groups, custom all-reduce IPC, EP buffers |
| `load_model` | build the model object, load weights |
| `determine_available_memory` | profile run (with memory-profiling graph capture), leaves the KV budget |
| `initialize_from_config` | allocate KV tensors from the KV cache spec, build attention metadata builders |
| `compile_or_warm_up_model` | torch.compile or kernel JIT, autotune, final CUDA graph capture |

Engine-core init, in order: executor (starts the workers, runs the chain above), KV cache init
(asks workers for the available memory, makes the spec, tells workers to allocate), structured
output backend, scheduler (sized from the KV block count), then the busy loop.

Measured shape of these steps:

- Qwen3-8B, warm caches, one engine: 17.8 s to ready, 150.3 s with every cache lost (our
  per-edit restart measurement, `kb/costs.md`). With the zygote (section 6) a forked
  `vllm serve` reaches `/health` in 10.9 s vs 20.9 s fresh (our 2026-11-06 run).
- DeepSeek-V4.1-Flash, TP4, 4x B200, warm: weight load 124.9 s, profile + memory-profiling
  capture 79.0 s, JIT warmup 52.5 s, FlashInfer autotune 627.2 s (retuned despite saved
  configs), final graph capture 40 s; 1012.2 s to `/health` (our 2026-10-08 startup
  measurement).
- In a one-line scheduler edit check on one GB200, setup is 204 s of the 228 s restart
  (our one-GB200 restart-vs-live measurement, `kb/costs.md`).
- CUDA graph capture on Qwen3-8B TP1: 4.5-9 s per size set per rank. A full rebuild (Dynamo
  re-trace 4.6, capture 6.0, warmups 1.9, split/reset 1.3, Inductor 0.2) is 14.4 s. A
  per-region rebuild, when the edit sits under an opaque op (Triton kernel, attention
  launcher), is 3.0-5.6 s; a traced Python edit still pays Dynamo, 12-14.8 s (our 2026-11-05
  CUDA graph recapture measurement on B200, warm cache).

## 3. Atomic units

Some groups of state cannot be rebuilt one piece at a time. The restart unit is the whole group.

- **Worker group under TP/EP.** NCCL communicators, custom all-reduce IPC handles and EP
  buffers (DeepEP) span all ranks. Restarting one rank leaves the others holding handles to a
  dead peer. The unit is the rank group; its cost includes communicator init.
- **Model -> KV spec -> KV tensors -> CUDA graphs.** A chain where each link is derived from
  the one before. A model edit that changes layer types or head layout changes the KV spec,
  which changes the KV tensors, which invalidates every captured graph (graphs hold the tensor
  addresses). Graphs are not cacheable across processes: a new process recaptures.
- **Scheduler bookkeeping vs worker KV memory.** The engine core's block table says which
  blocks are free or cached; the workers' KV memory holds what is in them. They must agree.
  Dropping one without the other serves wrong or missing prefix-cache hits, so a restart of
  either side keeps both or drops both (preempt the running requests, then re-prefill).
- **Persistent worker state.** Under async scheduling, speculative decoding or a persistent
  batch, workers are not stateless between steps. Restarting them costs a preemption and a
  re-prefill, not only a process start.

## 4. Tiers and the restart rule

| Tier | When | Mechanism | Target edit-to-run |
|---|---|---|---|
| **Live code swap** | function or method body that adds no fields, classes, outliving closures, constructor changes or module state; outside captured graphs (or with in-place graph recapture / JIT rebuild) | kernel: locate, swap, journal, witness, two-phase across ranks | < 1 s |
| **Component restart** | anything else inside one box: constructors, new fields, classes, config read by that box, model code, graph or compile content without in-place rebuild | stop that box's process, start it on the new code against state holders that outlive it; no in-place repair | 2-10 s (**est.**) |
| **Warm engine restart** | distributed setup, parallelism, weights format, anything spanning boxes | zygote fork plus state holders | 20-40 s (**est.**) |

**Rule: restart the lowest box that contains the invalidated state.** A body edit is a live code swap. A new
field on the model triggers a worker-group component restart. A scheduler constructor change restarts the engine core.
Parallelism or weight-format changes touch every holder, requiring a warm engine restart.

The kernel's detectors (new fields, captured targets, closures that outlive the call, removed
definitions, edits outside the patched function, Triton JIT) are the classifier: their action is
"route to component restart", not "repair in place". Triton `@jit` in-place rebuild stays in the live code swap path because it is a pure
artifact with exact undo (0.07 s vs a 61.5 s restart).

**Propagation.** After restarting a box, recheck only what is downstream of it, top to bottom,
and stop when the value crossing the interface is unchanged. Example: a worker restart that
re-derives the same KV spec stops there; the engine core and its block table stay. A changed
spec continues to the engine core (re-size the scheduler) and on to KV tensors and graphs.

**State holders that outlive restarts.**

| Holder | State | Status |
|---|---|---|
| weights | cuMem handles mapped by workers; preload daemon; host Engram tables | exists (three engines shared one weight copy in our 2026-11-05 run) |
| KV memory | KV pool allocated by a holder, mapped by workers; prefix-cache contents kept or dropped explicitly | not built |
| compile caches | vLLM compile cache and Triton cache, content-addressed by code hash | exist on disk; CUDA graphs are not among them |
| warm processes | zygote with imports done | built (section 6) |

Request state is not held: the engine core drains or preempts before its own restart.

## 5. Cost per boundary

Measured means a number from a run of ours; **est.** means derived and unverified. Component restart is not
built, so every component-restart row is an estimate.

| Boundary | Cost | Status | Source |
|---|---:|---|---|
| Live code swap (one function) | < 1 s; install 22 ms median, undo 4 ms (v0) | measured | v0 overhead receipt, `kb/costs.md` |
| Triton `@jit` rebuild | 0.07 s | measured | 2026-11-05 kernel run |
| Per-region CUDA graph recapture, op edit | 3.0-5.6 s | measured | 2026-11-05 CUDA graph recapture, Qwen3-8B TP1 |
| Full CUDA graph recapture, in place | 14.4 s | measured | same |
| Fork a server from the zygote | 10.9 s to `/health` | measured | 2026-11-06 zygote run |
| Zygote with a core edit (re-import all of vLLM) | 12.4-13.7 s | measured | same |
| Full restart, warm caches, Qwen3-8B | 17.8-60.7 s depending on the edit | measured | per-edit restart measurement |
| Full restart, cold, Qwen3-8B | 150.3 s | measured | same |
| Full restart, DeepSeek-V4.1-Flash TP4 | 1012 s warm | measured | 2026-10-08 startup measurement |
| Worker component restart, TP1, holders and cached graphs | 2-10 s | est. | needs the spike below |
| Worker-group component restart, TP4 | not estimated | est. | adds communicator init and graph recapture per rank |
| Engine-core component restart, workers and KV kept | about 3 s | est. | needs the spike |
| Warm engine restart (zygote + holders) | 20-40 s | est. | zygote fork 10.9-35 s measured, plus holder attach |

Open measurement (the spike that decides component restart): a worker restart on Qwen3-8B with weights from
the daemon, KV from a holder and cached graphs, at TP1 and TP4, including preempt and
re-prefill; then an engine-core restart with workers and KV kept. If the worker restart lands
at or under 10 s it replaces the in-place state-repair mechanisms. For model-Python edits under
torch.compile the bar is the in-place ~14 s, capture included. Until then, the expected
edit-to-run of about 2.6 s (a 5 s component restart estimate and a replay mix of 69% live code swap) is provisional.

## 6. How the live lab maps onto this

| Piece | Where | State |
|---|---|---|
| Kernel | `template/livelab/kernel/` (shared by every engine's lab) | locate, swap, journal, witness, findings, two-phase coordinator, session |
| Adapter | engine-core host plus worker extension | replies per rank; in review, not on main yet |
| Services | `livelab/services/` | `zygote.py`, `zygote_refresh.py`: imports vLLM once, forks servers, refreshes edited modules before the fork |
| Interface | `template/livelab/INTERFACE-v1.md` | draft with conformance cases; on main |

Not built:

- Component restart for worker group, engine core and API server.
- The KV holder, and any holder-driven engine-core restart with KV kept.
- A classifier that maps an edit to live code swap, component restart, or warm engine restart (the kernel's findings are its input).
- Reaching every rank at DP > 1: the adapter host does not start under DP engine cores yet.
- Graph rebuild under load: a recompile while serving can empty worker request state and
  kill the engine; the idle rebuild works. Component restart must preempt first.
