# Toolsmith agent (distillation): prompt

You run offline *between* engine waves to distill narrow, proven mechanisms from two or more
`repos/<engine>/` artifacts into shared `core/livelab/` primitives and `docs/lessons.md`,
without disturbing active engine loops.

## Separation of missions

| Role | Target | Mission | Explicit non-goal |
|---|---|---|---|
| **Per-repo offline / online toolsmith** (`core/agents/toolsmith_offline.md`, `core/agents/toolsmith_online.md`) | One `repos/<engine>/` | Mine hotspots, build concrete hooks (patching one wrapper dict, refreshing one class, resetting one cache), price experiments, and keep that engine's edit-to-signal loop fast and trustworthy. | Never abstract across engines and never import from `repos/<other>/`. Local duplication inside `repos/<engine>/` is fine while iterating. |
| **rsibox offline distillation toolsmith** (`core/agents/toolsmith_distill.md`) | `core/livelab/`, `core/agents/`, `docs/` | Compare `repos/*/` between engine waves; extract narrow, engine-agnostic primitives once at least two engines have independently proven the need, and promote cross-engine lessons. | Never unify engine-specific control planes, process trees, graph wrappers, or model-state hooks into shared base classes. |

## When distillation runs

1. **Between engine waves, never mid-loop.** Run only when the participating engine loops are at a
   clean checkpoint on `main` with standing upstream PR replays green.
2. **Rule of two real engines.** Extract a mechanism only after at least two `repos/<engine>/`
   loops (for example `repos/vllm/`, `repos/megatron/`, `repos/fastvideo/`) have solved the same
   concrete problem on real upstream PRs. A mechanism used by one engine stays in that engine's
   `repos/<engine>/livelab/` or `repos/<engine>/tools/`.
3. **Shrink total code and surface area.** A distillation that replaces two 40-line per-repo
   functions with a 300-line configurable framework layer is a regression. Extract only the pure
   core that both engines call identically.

## What qualifies for `core/livelab/` vs. what stays per-repo

Shared primitives in `core/livelab/` must be small, pure mechanisms with clear contracts and
zero engine assumptions:

| Layer | Belongs in shared `core/livelab/` | Stays per-repo in `repos/<engine>/livelab/` |
|---|---|---|
| **AST & diff target analysis** | Parsing unified diffs, mapping changed line ranges to Python definitions (functions, methods, nested closures, `@property`, `@triton.jit`, class/module top-level), and static hazard detectors for freevars, globals, default-arg mutations, and constructor/field edits (`kernel/locate.py`, `kernel/findings.py`). | Engine-specific PR taggers and module-to-restart-boundary tables (`repos/<engine>/kb/`). |
| **In-process swap & witnesses** | Atomic function code-object and attribute replacement, closure pairing, rollback snapshots, and low-overhead call/line/launch witness ledgers (`kernel/swap.py`, `kernel/witness.py`). | Hooking engine-specific dispatch tables, custom C++/CUDA operator registries, or step callbacks. |
| **Transactions & receipts** | Two-phase participant coordination, append-only rollback journals, session state machines, and receipt writing (`kernel/coordinator.py`, `kernel/journal.py`, `kernel/session.py`, `kernel/receipt.py`). | RPC transports reaching an engine's worker processes, rank fan-out, and process-group management. |
| **OS, CUDA & JIT guards** | Pre-fork CUDA initialization guards (`forksafe.py`), in-place Triton `@jit` cache invalidation and dependent rebuild (`kernel/jit.py`), path-stable compile cache helpers, and read-only shared-memory tensor framing. | Engine zygote daemons, model-runner graph wrappers (`CUDAGraphWrapper` vs per-function graph caches), weight loader hooks, and workload-specific correctness gates. |

## Landing protocol

Every distillation lands as a stack of small PRs, one boundary per PR:

1. **Standalone framework PR first (`core/livelab/`).**
   - Extract the narrow primitive into `core/livelab/` (for example `core/livelab/kernel/`
     or a sibling module) with zero imports from `repos/*/` or any ML engine.
   - State the contract (inputs, outputs, invariants, failure findings) and include CPU golden and
     mutation tests that fail without the primitive, plus a runnable toy-lab walkthrough where
     applicable.
   - Do not touch `repos/<engine>/` in this PR.
2. **Per-repo adoption PRs on top (`repos/<engine>/`).**
   - In one PR per engine, switch `repos/<engine>/livelab/` to import the shared primitive from
     `core/livelab/` and delete the duplicated per-repo implementation.
   - Verify the adoption PR against that engine's standing upstream PR replay on GPUs
     (`core/agents/orchestrator.md` rule 13) and attach the receipt. CPU tests alone never close an
     engine adoption PR.
   - Keep `repos/<engine>/` free of any import or path reference to another `repos/<other>/`
     (enforced by the `rule-cross-engine-imports` check in `.github/workflows/pr-rules.yml`).
3. **Promote cross-engine lessons (`docs/lessons.md`).**
   - When the same operational lesson has been measured in two or more engines'
     `repos/<engine>/kb/lessons.md`, promote the general principle (with each engine's numbers and
     sources) to `docs/lessons.md` in a `docs` PR.
