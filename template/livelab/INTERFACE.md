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

<!-- TODO: exact schema; the shared receipt writer and status state machine each lab imports.
     A schema is proposed below for v1; this file stays version 0 until it is adopted. -->

## Proposed for v1: receipt schema (not part of version 0)

> **Proposal.** Not binding for version 0 labs. Adopting it is the v1 bump. The writer is
> `kernel/receipt.py` (`ReceiptWriter`), the only code that writes receipts: one JSON object
> per line, appended under an exclusive lock, never rewritten. One record per transaction.

| Field | Type | Meaning |
|---|---|---|
| `schema` | str | `fastrsi-receipt/proposed-v1` |
| `id` | str | transaction id |
| `interface_version` | str | the INTERFACE version the lab implements |
| `engine` | `{name, version, commit}` | the engine under change |
| `base_hash`, `patch_hash` | str | hash of the base sources, and of the patched sources plus repair cells |
| `plan` | `{targets, predicted_resources, recipes, certainty, accept}` | what the planner predicted, and the accepted check names |
| `actual_recipes` | [str] | recipes that ran (empty unless committed) |
| `participants` | `{name: {prepare, commit, abort, undo}}` | each call's `{"ok": bool, ...}` reply |
| `proofs` | `{participants: {name: {hashes, witness}}, hashes_agree}` | code hashes and witness counts after commit |
| `findings` | [finding + `participant`] | every finding of every phase, in the findings schema |
| `decisions` | `{accepted: [check], blocking: [check]}` | how the accept list was applied |
| `outcome` | `committed` \| `aborted` \| `rolled_back` | all switched, none switched, or switched and undone |
| `timeline` | `{edit, prepare_start, vote, committed, finished}` | measured unix seconds |
| `edit_to_run_s` | float \| null | `committed - edit`; null if nothing committed |
| `edit_to_finish_s` | float | `finished - edit` |
| `baseline_restart_s` | `{seconds, source}` \| null | the warm restart this transaction replaces |
| `undo` | `{status, participants}` | `not_needed`, `undone` or `partial`, with each undo reply |
| `writer` | `{pid, written_at}` | added by the writer |

Findings schema: `{check, observed, expected, why, actions: [{action, cost}], risk_if_ignored,
lessons: [...], overridable}`, plus optional `subject` and `accepted`.
