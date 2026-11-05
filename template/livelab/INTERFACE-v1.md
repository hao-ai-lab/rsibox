# Live lab interface, version 1 (draft)

**Draft, not adopted.** `INTERFACE.md` (version 0) stays in force. Adopting this file replaces
it, which is a breaking change: its own PR, titled `feat(livelab)!: ...`.

Every `repos/<engine>/livelab/` implements this. How is up to the engine; what is fixed here.
The engine-agnostic parts already exist in `kernel/` (CPU-tested) and `conformance/`.

## What changed from version 0, and why

`§` references are to the loop design notes (2026-11), kept outside the repo; toy numbers are
in `kernel/README.md`.

| v0 | v1 | Why (measured) |
|---|---|---|
| `install` binds "on every rank at once", no protocol | Every verb is a **transaction**: prepare, then validate and commit (or abort) on every participant inside **one step gap** | Committing ranks one after another gave at least one step with mixed versions in 20 of 20 toy runs; one step gap gave none in 20 of 20 (`kernel/README.md`). |
| Install pauses the engine for the whole install | **Prepare outside the gap** (locate, compile, stage, vote) while steps run; only validate and commit stall the loop | In the gap, prepare was 97% of the gap on the toy. Split, the first live run (vLLM, 8B dense model, one GPU) stalled 0.6 ms for a 0.84 s prepare (design notes §4b, items 7 and 12). |
| Locate by heap scan | Locate from the module namespace; scan the heap only for closures and suspended frames | A full scan holds the GIL: one live step took 221 ms against a 20 ms median. Without it, the largest gap between stream chunks during prepare fell from 249 to 37 ms; on the toy, step-stall p99 fell from 17.5 to 2.8 ms (§4b item 11, `kernel/README.md`). |
| `status()`: base, edited, poisoned (undefined) | Defined by **code fingerprints** and faults; plus `status(id)` for in-doubt callers | A caller that times out does not know the outcome; it must ask, never assume failure (§4b item 8). |
| Function groups pick one mechanism | **Recipes R0-R10** name what is rebuilt, with measured costs; groups stay a planner's tags | Planned mechanisms were too pessimistic: in the first 4 replayed PRs, every actual recipe was cheaper than predicted, mostly swap plus repair, R0+R10 (§4b item 13). |
| "Refuses unsupported edits" | **Findings** with an accept list; only physical limits cannot be accepted | AGENTS.md "Checks advise; agents decide". |
| One receipt, schema TODO | **Linked receipts** `txn`, `proof`, `undo` sharing a transaction id | Proof needs the workload, which runs after the commit; undo happens whenever it happens (§4b item 2). |
| Undo "proves base invariants" | Undo is exact for code; data needs the plan's **undo resets** (R2, R3, R8) | Undo cannot un-write requests or cache entries the new code produced (§4b item 3). |
| (not covered) | Deleting a definition is an edit (`removed_definition`); an edit that changes no code is refused (`empty_change`) | A deletion-only PR changed nothing live, so a forgotten caller passed live and failed after restart (§4b item 10). |
| Cost per call | Two headline metrics: **edit-to-run** and **edit-to-finish**, against a warm restart | First live run: edit-to-run 0.88 s, edit-to-finish 6.4 s, warm restart 88 s (§4b item 12). |

## Participants

| Participant | Quiesce point | Reached by |
|---|---|---|
| Worker ranks | between two steps | the engine core, by a collective call issued from its loop |
| Engine core | inside its busy loop, between steps | its input-queue channel |
| API server | between requests to the edited route | its own channel (not in the kernel yet) |

The **host** (coordinator) of a worker + engine-core transaction lives in the engine core's
loop: a collective call issued there completes on every worker before the next step, so one
gap holds the whole phase 2. Every participant call returns `{"ok": bool, ...}` and never
raises; a crash or a malformed reply counts as not-ok.

## Transactions

```
phase 1, steps go on:   prepare on every participant (stage beside the old state, check, vote)
phase 2, one step gap:  validate on every participant (staged code still current)
                        -> commit on all, or abort on all
                        -> a commit that fails undoes the committed ones in the same gap
```

Participant calls: `prepare(plan)`, `validate(id)`, `commit(id)`, `abort(id)`, `undo(id)`,
`status(id)`, `proof(id)`, `witness_check(id)`, `reset(kind)`, `fingerprint(modules)`.
Signatures and plan format: `kernel/coordinator.py`.

- Outcomes: `committed` (all switched), `aborted` (none switched), `rolled_back` (switched and
  undone in the same gap).
- Host states for `status(id)`: `prepared`, `committed`, `aborted`, `undone`, `in-doubt`
  (phase 2 started, not finished). A phase 2 not yet started when the caller gave up is
  cancelled.
- Chained edits: a participant stages a new plan against the text it last installed.

## Verbs

| Verb | Transaction | Returns |
|---|---|---|
| `install(patch)` | plan the patch, then one transaction | `id`, `outcome`, `codes` (blocking checks), `mechanisms` (recipes run), findings, seconds |
| `check(workload)` | none; runs the workload in the warm engine, then proves every live transaction | `output`, `proven` (true, false, or null when nothing is live), witness counts per participant, findings, seconds |
| `reset(kind)` | `state`: drain in-flight work (R2) and clear cached data (R3). `respawn`: restart the worker group from warm caches | what was reset, seconds |
| `uninstall(id)` | undo `id` and every later transaction, newest first, each followed by its plan's undo resets | ids undone, state, seconds |
| `status()` / `status(id)` | none | `base`, `edited` or `poisoned`; live ids; code fingerprint; poison findings / one transaction's host state |

A patch is `{"files": {module: full patched text}, "accept": [check], "undo_resets": [...]}`.
The lab needs no planner: each participant's prepare finds the changed definitions (every
def, lambda and generator expression whose AST differs, plus added and removed definitions)
from its one parse of the running text and the patched text. A patch may change several
modules at once: one module's patch can import a name another module's patch adds. The
modules are staged together and nothing is visible before the commit, which applies them all
or none.

The lab records a code **fingerprint** per participant (content hash and object identity of
every function its engine modules reach) at start and after every commit, and checks it after
every abort, rollback, undo and `status()`. The lab is **poisoned** by: a fingerprint that
differs (code changed outside the journal), a crash, a hang, a lost participant, a failed undo
or rollback, an unresolved in-doubt transaction, a participant that missed the phase-2
deadline. A poisoned lab refuses `install` with `session_poisoned` until `reset("respawn")`;
`uninstall` stays allowed (it backs out before the respawn) and leaves the lab poisoned.

## Recipes (mechanism vocabulary)

Costs from our vLLM runs (design notes §2; per-engine numbers belong in `repos/<engine>/kb/costs.md`).

| Recipe | Rebuilds | Keeps | Cost |
|---|---|---|---|
| R0 swap | code objects | everything else | 0.03-0.04 s per worker; µs in the engine core |
| R1 swap + recapture | code, CUDA graphs, compiled kernels | model, weights, KV | 4.5-7.8 s |
| R2 preempt all | request state (re-prefill) | prefix cache, objects | 0.1 ms + re-prefill |
| R3 cache reset | prefix-cache contents | objects, KV memory | 1-2 ms + full re-prefill |
| R4 core rebuild | engine-core objects | KV manager, requests, workers | 8-35 ms |
| R5 soft restart | model object, graphs | processes, comm groups, weights, KV memory | 1.4-2.2 s eager |
| R6 weight reload | processed weights | process | 5 s by IPC mapping; ~100 s from disk |
| R7 native reload | op library registration | process | build 8-24 s per file |
| R8 snapshot revert | prefix-cache contents, from tiers | everything | 6 ms + reload |
| R9 warm restart | everything below config | caches, weight daemon | 31-84 s |
| R10 repair cell | what the cell declares it sets | everything else | ms per call |

Recipes compose (R0+R10 for a new field, R0+R1 for a forward edit in graph mode). The kernel
implements R0 and R10, and R2, R3, R8 as reset hooks; the others are named here and not yet
behind the kernel. A change that no recipe below R9 covers (startup logic, launch flags,
environment variables) is refused with its reason: it needs a restart.

## Findings and accept

Every check returns findings, never an exception:

`{check, observed, expected, why, actions: [{action, cost}], risk_if_ignored, lessons: [...],
overridable}`, plus optional `subject` and `accepted`.

The caller's `accept` list (check names, accept records, or `"all"`) decides: a transaction proceeds only if
every finding is accepted, and only overridable findings can be. Physical limits are not
overridable: `patch_invalid`, `no_live_function`, `not_in_patch`, `closure_changed`,
`changed_since_prepare`, `empty_change`, `repair_failed`, `participant_failed`, `reset_failed`,
`undo_incomplete`, `tampered`, `session_poisoned`. Catalog: `kernel/findings.py`.
A check that could not run as designed says so as a finding too: a prepare whose analysis
fell back from its child process to the engine's own process reports `analyze_fallback`
(overridable) with the reason and the time it held the engine's GIL.

An accept record `{check, by, reason, scope}` says who accepted a finding and why; `decide`
returns the records it used (`accepts`) so the transaction record can carry them. Scope `txn` covers
this transaction; `standing` also names `participant` and `code_hash` and covers a finding only while
both still match, so it expires when the checked code changes. Stage findings about a staged
definition carry its new `code_hash`; participants add `participant`. A check accepted on every run is a
calibration bug in the check, not a standing accept. Writing accepts into receipts waits for the
receipt writer's human owner.

## Receipts

Written by one shared writer, never by engine code or agents. A transaction leaves linked
records that share its `id`, appended as JSON lines: `txn` (plan, every participant's reply,
outcome, rollback, findings and decisions, timeline, edit-to-run), `proof` (witness counts,
findings, the caller's golden results, edit-to-finish), `undo` (code undo, then each reset;
`undone` or `partial`). Fields: the "Proposed for v1" section of `INTERFACE.md`, and
`kernel/receipt.py`.

The writer, `kernel/receipt.py`, was written by an agent and **needs a human owner** before
its receipts count (AGENTS.md, Trust). It has no record kind for a conformance run yet, so the
conformance runner prints its verdict and does not write it.

## Metrics

| Metric | Measured as |
|---|---|
| `edit_to_run_s` | commit time minus edit time: how soon the engine runs the edit |
| `edit_to_finish_s` | proof time minus edit time: how soon the edit is proven on the workload |
| `baseline_restart_s` | `{seconds, source}` of the warm restart (R9) the transaction replaces |

Report both against the baseline. Also in every `txn` record: `prepare_s` (phase 1, steps go
on) and `gap_s` (phase 2, how long the loop stalled); `undo_s` in every `undo` record.

## Conformance

A lab conforms when it passes `conformance/` cases c0-c10, driven only through these verbs
plus the harness in `conformance/lab.py`. The toy lab passes all 11 on CPU.

## Not settled in this draft

- The API-server participant (request-scoped commit) is not in the kernel.
- The host's transaction table is in memory: a host crash loses `status(id)`.
- Live: one transaction on one GPU so far; more than one rank (TP > 1) is not verified.
- Who owns the receipt writer, and whether a conformance run gets its own record kind.
