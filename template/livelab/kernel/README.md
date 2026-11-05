# Transaction kernel

The small, engine-agnostic core that every live-change mechanism shares: locate the live code,
swap it, journal every change so undo is exact, report findings instead of raising, count
calls of the new code, run two-phase transactions over participants, and write linked
receipts. Pure Python, standard library only; it imports no engine.

| File | What it does |
|---|---|
| `locate.py` | (module, qualname, first line) -> running code objects, the function objects that hold them (gc heap scan), and every holder (module and class dicts, closure cells, `__wrapped__`, bound methods, attributes). One code identity: file, `co_qualname`, `co_firstlineno` + `content_hash`. |
| `swap.py` | `stage` compiles the patched module, adds the witness, pairs old and new code for each target and its enclosing functions, evaluates changed defaults, runs imports in a scratch namespace, runs the checks. Nothing is visible. `apply` makes it visible through the journal. |
| `journal.py` | One ordered undo log per participant. Entries `code`, `attr`, `added`, `object`, each with its inverse. `mark()` / `undo_to(mark)`. `record_attr` and `repair(...)` journal hand-written repair cells (recipe R10). |
| `findings.py` | One findings schema, the catalog of checks, and `decide(findings, accept)`: the one accept mechanism. |
| `witness.py` | Counters keyed `module:qualname:firstline`; `check` reports `zero_witness` or `running_frame`. |
| `receipt.py` | **Proposal, needs a human owner** (AGENTS.md, Trust). The one receipt writer: linked `txn` / `proof` / `undo` records sharing a transaction id, appended as JSON lines under a lock. Schema in the docstring and in `../INTERFACE.md` (proposed for v1). |
| `coordinator.py` | Two-phase transactions: prepare (stage and vote), commit, abort, undo of committed participants when a commit fails; `prove` after the workload; `undo` then the plan's data resets (`undo_resets`: R2, R3, ...). `StepGapCoordinator` runs a whole transaction in one gap between two steps of the loop owner; `Coordinator` calls participants one by one. `LocalParticipant` runs the kernel in-process; `CallParticipant` is the shape an engine implements over its RPC. |
| `toy/` | A toy engine: ops with a decorator and a closure factory; an `Engine` built in `__init__` with a saved bound-method callback, a cache, and a `step` that records the code version it ran; a single-engine busy loop; and a `Cluster` (the engine core) that steps several ranks together and runs queued work only between steps. `load_copy` gives each participant its own module copy. |

## Walkthrough: one transaction in one step gap

Three ranks, each with its own copy of the toy engine, step together in a `Cluster` loop
thread. One transaction swaps `Engine.step` (its version tag goes from `base` to `v1`) on all
of them inside one gap between steps; after the workload the coordinator proves it, then
undoes it in another gap. Run from `template/livelab`:

```python
import itertools, os, sys, tempfile, threading, time
from kernel import toy, coordinator as C, receipt as R, locate as L

ENG = toy.source("engine")
engines, parts = {}, []
for rank in ("rank0", "rank1", "rank2"):
    toy.load_copy("w_" + rank)                          # each rank: its own module copy
    engines[rank] = sys.modules[f"w_{rank}.engine"].Engine()
    parts.append(C.LocalParticipant(rank, alias={"toy": "w_" + rank}))
cluster = sys.modules["w_rank0.engine"].Cluster(engines)   # the loop owner (engine core)
line = next(i for i, s in enumerate(ENG.splitlines(), 1) if "def step" in s)
plan = {"modules": {"toy.engine": {
    "source": ENG.replace('self.ran.append("base")', 'self.ran.append("v1")'),
    "base_hash": L.text_hash(ENG), "targets": [["Engine.step", line]]}}}
path = os.path.join(tempfile.mkdtemp(), "receipts.jsonl")
co = C.StepGapCoordinator(parts, cluster.at_gap, writer=R.ReceiptWriter(path))

stop = threading.Event()
loop = threading.Thread(target=cluster.run, args=(stop,))
loop.start()                                            # steps run continuously from here
time.sleep(0.02)
res = co.run(plan, edit_time=time.time())               # queued to the loop, runs in one gap
time.sleep(0.02)
proof = co.prove(res["txn"])
undo = co.undo(res["txn"])
time.sleep(0.02)
stop.set()
loop.join()

print("outcome:       ", res["outcome"], "edit_to_run_s", res["receipt"]["edit_to_run_s"],
      "prepare_s", res["receipt"]["prepare_s"], "gap_s", res["receipt"]["gap_s"])
tl = res["receipt"]["timeline"]
print("in the gap:     validate", round(tl["vote"] - tl["gap_start"], 6),
      "s, commit", round(tl["committed"] - tl["vote"], 6), "s; longest stall between steps",
      round(cluster.longest_stall(), 6))
print("proof witness: ", {r: list(w.values()) for r, w in proof["receipt"]["witness"].items()},
      "edit_to_finish_s", proof["receipt"]["edit_to_finish_s"])
print("undo:          ", undo["receipt"]["status"], "undo_s", undo["receipt"]["undo_s"])
print("versions per step, collapsed:", [k for k, _ in itertools.groupby(cluster.log)])
print("steps with mixed versions:   ", [t for t in cluster.log if len(set(t)) > 1])
print("receipts:      ", [(r["kind"], r["id"] == res["txn"]) for r in R.read(path)])
```

Real output:

```
outcome:        committed edit_to_run_s 0.024824 prepare_s 0.024473 gap_s 0.000103
in the gap:     validate 3e-05 s, commit 7.3e-05 s; longest stall between steps 0.008111
proof witness:  {'rank0': [40], 'rank1': [40], 'rank2': [40]} edit_to_finish_s 0.049065
undo:           undone undo_s 0.00054
versions per step, collapsed: [('base', 'base', 'base'), ('v1', 'v1', 'v1'), ('base', 'base', 'base')]
steps with mixed versions:    []
receipts:       [('txn', True), ('proof', True), ('undo', True)]
```

Prepare (locate, heap scan, compile, stage) runs while the loop steps; the loop stops only for
validate and commit (0.1 ms here). But the longest stall between steps was 8 ms, not the gap:
the heap scan is one C call that holds the GIL, so a prepare in another thread of the SAME
process still stalls the loop. On this machine, with 5 million extra objects, a 0.15 s scan
stalled the loop thread 53 ms. Phase 1 is free only in another process, or once locate stops
scanning the whole heap.

So locate no longer scans by default. It walks the module's own namespace (functions,
classes, static and class methods, properties, decorator wrappers' originals), which finds
top-level and method targets in microseconds; swapping `__code__` on those objects still
reaches every holder. It scans the heap only for nested targets (closures made at run time)
or targets the namespace does not hold, and looks for suspended frames only when a target
is a generator or coroutine (no other frame can be suspended off a thread's stack). A scan
refreezes the heap right after its one `get_objects` call and then filters in 2 ms slices.
On the toy, with 2.0 M objects (one `get_objects` takes 22 ms) and a median step of 0.56 ms:

| Locate | prepare | step stall p99 | max |
|---|---|---|---|
| before: two heap scans, unsliced | 0.30 s | 17.5 ms | 21.4 ms |
| two heap scans, sliced | 0.32 s | 12.7 ms | 15.0 ms |
| namespace, no scan (method target) | 0.008 s | 2.8 ms | 6.1 ms |
| nested target: one sliced scan | 0.16 s | 17.2 ms | 18.5 ms |

Slicing barely helps: what remains is the single `get_objects` call. Nested targets still pay
it once.

With the one-by-one `Coordinator` instead (each rank switching at its own boundary), the same
commit and undo gave at least one step with mixed versions in 20 of 20 runs on CPU; the
step-gap coordinator gave none in 20 of 20.

## Golden tests

`python3 -m unittest -v` from `template/livelab` (CPU, about 1.5 s; one test builds a 2.5 M-object heap):

```
test_accept_list ... ok
test_closure_change_is_a_finding_not_an_exception ... ok
test_code_identity_and_hash ... ok
test_commit_failure_undoes_the_committed ... ok
test_gap_stays_small_while_prepare_is_slow ... ok
test_linked_receipts_and_measured_times ... ok
test_one_by_one_commit_mixes_versions ... ok
test_prepare_never_holds_the_loop_longer_than_the_budget ... ok
test_removed_definition_is_a_finding_and_undo_restores_it ... ok
test_repair_cell_sets_missing_field_and_undo_removes_it ... ok
test_repair_that_cannot_be_undone_is_a_finding ... ok
test_running_frame_never_runs_the_new_code ... ok
test_step_gap_commit_and_undo_never_mix_versions ... ok
test_step_gap_failed_commit_is_undone_in_the_same_gap ... ok
test_swap_reaches_every_holder ... ok
test_timed_out_caller_resolves_the_outcome_with_status ... ok
test_undo_restores_exact_base ... ok
test_undo_runs_the_declared_data_reset ... ok
test_validate_catches_a_change_between_prepare_and_gap ... ok
test_vote_no_changes_nobody ... ok
----------------------------------------------------------------------
Ran 20 tests in 1.49s

OK
```

Each test's docstring says what it proves and what it does not.

## Rules the kernel keeps

- **Nothing raises to the caller.** Every call returns `{"ok": bool, ..., "findings": [...]}`.
  A participant that raises or replies with something else counts as not-ok.
- **Checks advise.** A check returns a finding; the caller's accept list decides. Physical
  limits (`closure_changed`, `no_live_function`, ...) are not overridable.
- **Prepare changes nothing visible.** New code, defaults, imports and added definitions are
  built beside the old state. Only `apply` (inside `commit`) switches.
- **Ranks that step together switch in one gap.** Phase 1 (prepare) runs while the loop
  steps; `StepGapCoordinator` queues phase 2 (validate, commit or abort) to the loop owner.
  Validate re-checks that every function still runs the code object prepare staged against;
  a change in between (`changed_since_prepare`) aborts everyone. A failed commit is undone in
  the same gap, so no step runs the new code anywhere.
- **Never assume a timed-out transaction failed.** The caller gets `in-doubt`; the host's
  `status(txn)` gives the real state (prepared, committed, aborted, undone, in-doubt). A phase
  2 that had not started when the caller gave up is cancelled.
- **Deletions are edits too.** A definition the edit deletes (module function, method,
  class) but the live module still defines is a `removed_definition` finding; accepted, it is
  deleted through the journal, and undo puts the same object back. Copies made by
  `from x import f` elsewhere still hold it (not covered).
- **Undo is exact for code, not for data.** The journal restores the same code and attribute
  objects and removes every name it added. Data the new code wrote (requests, cache entries)
  needs the reset the plan declares in `undo_resets`; a missing hook is a `reset_failed`
  finding.
- **The heap scan unfreezes the gc.** Engines call `gc.freeze()` after startup, and
  `gc.get_objects()` and `gc.get_referrers()` both skip frozen objects.

## Not covered yet

- Nested targets (closures) still need one heap scan, whose `get_objects` call holds the GIL
  in one piece. Copies of a function object made with `types.FunctionType(code, ...)` are not
  found by the namespace path.
- The host's transaction table lives in memory: a host crash loses `status`.
- A second edit on top of a committed one, when the edit moved the target's first line: the
  running code then has the patch's line, and the target is not found (`no_live_function`).
- Functions defined with `exec` into foreign globals are not found (the scan matches by
  `__globals__`, so module copies stay apart).
- Real RPC: the toy tests reach ranks through `CallParticipant` over a 1 ms in-process call;
  the vLLM side (collective_rpc from the engine-core busy loop) is the next step.
