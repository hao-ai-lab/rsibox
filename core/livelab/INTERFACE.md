# Live lab interface, version 0 (draft)

Every `repos/<engine>/livelab/` implements this. How is up to the engine; what is fixed here.
Changing this file is a breaking change: bump the version and say so in the PR.

## The five verbs

Every verb returns a plain dict with `cost_s` (seconds) and never raises (`conformance/lab.py`):

| Call | Does | Returns |
|---|---|---|
| `install(patch)` | Stages and binds a change (`files`, optional `accept`, `undo_resets`) on every worker rank at once, refreshing only what it invalidates. Refuses unsupported edits (`c6_refusal`: `patch_invalid`, `closure_changed`, `outside_changes`, `empty_change`) and poisoned labs (`session_poisoned`) before touching state. | `{ok, id, outcome, codes, mechanisms, findings, cost_s}` (`outcome`: `committed \| aborted \| rolled_back \| refused \| in-doubt`) |
| `check(workload=None)` | Runs a probe inside the warm engine | `{ok, output, proven, witness, findings, cost_s}` (+ optional `abstained`); `output` is bit-exact JSON, `proven` is `True \| False \| None` (`False` with `zero_witness` if any involved edit missed execution; `None` at base) |
| `reset(kind="state")` | `kind="state"`: drains in-flight work and restores volatile state (caches, buffers, RNG, iterators) so the next check starts from the same state. `kind="respawn"`: restarts the worker group from warm caches and clears poison (`c8_crash_and_hang`). | `{ok, reset, cost_s}` where `reset` is a non-empty list naming what state was cleared |
| `uninstall(id=None)` | Reverts `id` (default newest) and any later edits in place, newest first | `{ok, undone, state, cost_s}` (`state`: `base \| edited \| poisoned`) |
| `status()` | Reports live lab state and code identity digest per participant | `{ok, state, txns, fingerprint, poison}` (`state`: `base \| edited \| poisoned`) |

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
> `core/livelab/kernel/receipt.py` (`ReceiptWriter`): one JSON object per line, appended under an exclusive
> lock, never rewritten. It was written by an agent and **needs a human owner** before its
> receipts count (see "Trust" in AGENTS.md).

A transaction leaves linked records that share its `id`, appended to one file: `txn` when it
runs, `proof` after the workload ran, `undo` whenever it is undone. Every record has
`schema` (`fastrsi-receipt/proposed-v1`), `kind`, `id`, and `writer: {pid, written_at}` added
by the writer.

`txn`: the transaction. Phase 1 (prepare, the vote) runs while the engine steps; phase 2
(validate, commit or abort, rollback of a failed commit) runs in one gap between steps. A
caller that times out gets `in-doubt` and asks the host's `status(id)` (prepared, committed,
aborted, undone, in-doubt); the record is written once the outcome is known.

| Field | Type | Meaning |
|---|---|---|
| `interface_version` | str | the INTERFACE version the lab implements |
| `engine` | `{name, version, commit}` | the engine under change |
| `base_hash`, `patch_hash` | str | hash of the base sources, and of the patched sources plus repair cells |
| `plan` | `{targets, predicted_resources, recipes, certainty, accept, undo_resets}` | what the planner predicted, the accepted check names, the resets an undo must run |
| `actual_recipes` | [str] | recipes that ran (empty unless committed) |
| `participants` | `{name: {prepare, validate, commit, abort, undo, proof}}` | each call's `{"ok": bool, ...}` reply |
| `commit_hashes` | `{participants: {name: {key: [hash]}}, agree}` | code hashes right after commit, and whether all participants agree |
| `findings` | [finding + `participant`] | every finding of the transaction, in the findings schema |
| `decisions` | `{accepted: [check], blocking: [check]}` | how the accept list was applied |
| `outcome` | `committed` \| `aborted` \| `rolled_back` \| `refused` | all switched, none switched, switched and undone, or refused while poisoned |
| `timeline` | `{edit, prepare_start, prepared, gap_start, vote, committed, gap_end, txn_end}` | measured unix seconds |
| `edit_to_run_s` | null | null here: nothing has run at commit; the `proof` record has the first execution |
| `edit_to_commit_s` | float \| null | `committed - edit` (monotonic clock); null if nothing committed |
| `timeline_mono` | `{stamp: seconds since edit}` | the same stamps on the monotonic clock; every `_s` span uses it |
| `prepare_s` | float | `prepared - prepare_start`: phase 1, while the engine steps |
| `gap_s` | float \| null | `gap_end - gap_start`: phase 2, how long the engine's loop stalled |
| `baseline_restart_s` | `{seconds, source}` \| null | the warm restart this transaction replaces |
| `rollback` | `{status, participants, missed_deadline?}` | undo of a failed commit: `not_needed`, `undone` or `partial` |

`proof`: what the workload showed after the transaction.

| Field | Type | Meaning |
|---|---|---|
| `witness` | `{name: {key: calls}}` | calls of each swapped target per participant |
| `findings` | [finding + `participant`] | `zero_witness`, `running_frame`, ... |
| `golden` | any | the caller's golden-check results, recorded as given |
| `timeline` | `{edit, proved}` | measured unix seconds |
| `edit_to_finish_s` | float \| null | `proved - edit` (monotonic) |
| `edit_to_run_s` | float \| null | first live witness hit, the latest over participants and ranks, minus edit (monotonic, same host); null if some participant shows none |
| `first_run_s` | `{participant[/rank]: float \| null}` | each one's first hit minus edit |
| `launches` | `{participant: {key: {kernel: n}}}` | Triton: launches per kernel that runs the edited code |

`undo`: a later undo of a committed transaction.

| Field | Type | Meaning |
|---|---|---|
| `participants` | `{name: {undo, reset: {kind: reply}}}` | code undo, then each declared data reset |
| `resets` | [str] | the plan's `undo_resets` (drain request state, clear cached data, snapshot revert) |
| `status` | `undone` \| `partial` | `partial` if any undo or reset failed |
| `findings` | [finding + `participant`] | e.g. `reset_failed`, `undo_incomplete` |
| `timeline` | `{queued, code_undone, undo_end}` | measured unix seconds |
| `undo_s` | float | `undo_end - queued` |

Findings schema: `{check, observed, expected, why, actions: [{action, cost}], risk_if_ignored,
lessons: [...], overridable}`, plus optional `subject` and `accepted`.
