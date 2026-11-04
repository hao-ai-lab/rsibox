# Live lab interface, version 0 (draft)

Every `repos/<engine>/livelab/` implements this. How is up to the engine; what is fixed here.
Changing this file is a breaking change: bump the version and say so in the PR.

## The four calls

| Call | Does | Returns |
|---|---|---|
| `install(patch)` | Stages and binds a change on every worker rank at once, refreshing only what it invalidates. Refuses unsupported edits before touching state. | install id, the mechanism used per function, or a refusal with its reason |
| `check(workload)` | Runs a probe inside the warm engine | latency/throughput, output correctness vs base, **whether the patched path ran**, optional profiler trace, and the check's own cost in seconds |
| `reset()` | Drains in-flight work and restores volatile state (caches, buffers, RNG, iterators) so the next check starts from the same state | which state was reset |
| `uninstall()` | Reverts to base in place | proof that base invariants hold again |

Plus `status()`: `base`, `edited` or `poisoned`.

## How `install` refreshes (by function group)

| Group | Mechanism |
|---|---|
| OP | Swap the Python function or object in place |
| GRAPH | Swap, then selectively re-trace or recapture the affected compiled graphs |
| STATE / RESOURCE | Swap, then rebuild the pre-allocated buffers, runners or tables it touches |
| NATIVE | Compile into a versioned shared library, rebind the op, recapture |
| RANKS | Any of the above, committed on all ranks or none |
| (restart) | Startup logic, launch flags, environment variables: refused, needs a restart |

## Process layout

A small coordinator runs outside the ML framework and hosts the engine as a worker process group.
Recoverable failures roll back in place with `uninstall()` + `reset()`. Unrecoverable ones
(sticky CUDA errors, hangs, failed resets) make the coordinator kill and respawn only the worker
group, from warm host and GPU caches. A standby worker shortens that respawn.

## Around the calls

- **Cost hook:** every call reports its seconds. The harness raises an alarm when a check is
  slower than the knowledge base's threshold or a call fell back to a restart.
- **Full evaluation F(x):** promising edits run the unmodified full benchmark asynchronously,
  on separate GPUs when available. A regression there reverts the edit.

## Receipt

Written by the shared receipt writer only, never by engine code or agents. Carries this
interface version, engine, commit, patch hash, groups and mechanisms, the witness result,
measurements, and the cost of each call.

<!-- TODO: exact schema; the shared receipt writer and status state machine each lab imports. -->
