# Transaction kernel

The small, engine-agnostic core that every live-change mechanism shares: locate the live code,
swap it, journal every change so undo is exact, report findings instead of raising, count
calls of the new code, run two-phase transactions over participants, and write linked
receipts, and keep the session state (base, edited, poisoned) that `status()` reports.
Pure Python, standard library only; it imports no engine.

| File | What it does |
|---|---|
| `locate.py` | (module, qualname, first line) -> running code objects, the function objects that hold them (gc heap scan), and every holder (module and class dicts, closure cells, `__wrapped__`, bound methods, attributes). One code identity: file, `co_qualname`, `co_firstlineno` + `content_hash`. |
| `swap.py` | `changed_targets(base, new)`: the definitions whose AST changed (the smallest planner). `stage(module, source)` with the targets omitted finds that same list from its own parse, so prepare needs no planner pass. `stage` parses the patched module and compiles only the units that hold the targets (`compile_units`), adds the witness, pairs old and new code for each target and its enclosing functions, evaluates changed defaults, runs imports in a scratch namespace, runs the checks. Nothing is visible. `apply` makes it visible through the journal. `analyze` is stage's text half (parse, compile, compare); `ProcessAnalyzer` runs it in a child process so prepare does not hold the engine's GIL. `stage(..., trees=(base, patch))` takes trees the caller already parsed (its own planner) instead of parsing again. New defaults and added methods are evaluated in a scratch class body that reads the live class's names (`_LiveBody`).. A later swap of a module whose earlier swaps changed its line numbers keys its targets at the installed text's lines; `stage(line_maps=)` walks the committed swaps' line maps back to the lines live code carries (the participant keeps them per transaction), and falls back to the one live code with the qualname only as an advisory `line_aliased` finding. |
| `journal.py` | One ordered undo log per participant. Entries `code`, `attr`, `added`, `object`, `removed`, `call`, `closures`, each with its inverse. `mark()` / `undo_to(mark)`. `record_call(fn, label, after_code=True)` puts the inverse at the start of the transaction, so undo runs it once the code is back (a cell that released graphs journals their recapture this way). `set_item(..., absent=stub)` gives undo a value to put back instead of deleting (the witness global). `record_attr` and `repair(...)` journal hand-written repair cells (recipe R10; a plan's cell gets the participant's journal as `journal`); `record_attr` restores through `setattr` / `delattr` what is not a plain `__dict__` entry (descriptors, `nn.Module` submodules, `__doc__`). `record_call(undo_fn, label)` journals an in-place re-init by the function that undoes it. `record_closures` (called by `apply`) lets undo take back closures the new code made at run time; undo scans the heap only for the new nested codes that gained a reference since apply, so a swap that made no lasting closure undoes without the pass. |
| `jit.py` | JIT targets. A Triton `@jit` kernel runs what Triton compiled from the source text, never the function's bytecode, so a code swap alone was a silent no-op on the GPU. `stage` finds the JIT objects holding a target (`jit_rebuild`, advisory), `apply` reruns their own `__init__` on the swapped function reading the patched text, resets the module's JIT objects that call it, journals a snapshot of each (undo restores the old compiled kernels) and counts launches (pre-run hook) as the witness. Duck-typed: no Triton import. |
| `coordinator.py` (additions) | `participant_missing`: a group participant's `expected` ranks that sent no reply block the transaction (at DP>1 a client returned DP rank 0's replies only). `at_base()`: pending transactions, journal length, code digests of the touched modules (base vs live), added names and scope attributes left behind: a warm engine pool checks it between jobs. |
| `fields.py` | New fields. `analyze` finds attributes that changed or added methods read on `self`, that the edited `__init__` sets and no base method or class body does, and compiles the new `__init__`'s statement for each (`new_fields`, blocking). The participant walks its scope (engine core, worker) breadth first for the live instances, resolves the statement's `__init__` parameters from each instance or its nearest owner, and with `"backfill": true` sets the fields at commit, journaled. No heap scan (vLLM freezes the gc heap). |
| `findings.py` | One findings schema, the catalog of checks, and `decide(findings, accept)`: the one accept mechanism. |
| `witness.py` | Counters keyed `module:qualname:firstline`; `check` reports `zero_witness` or `running_frame`. Undo never deletes the hook global: it binds the no-op `stub` (compiled code may still call it). `arm(key)` gives a repair cell a counter; a proof with no key at all is `no_witness_possible`. Calls made while the engine captures graphs (`Witness(capturing=fn)`, or inside `witness.capture()` during a recapture) are counted apart (`capture_counts`): a replay runs no Python, so a target that ran only during capture is a `zero_witness` routed to R1 that asks for a device-side probe, and proofs say "ran during capture N, ran live M". The hook the swap binds is a plain function marked for Dynamo (`assume_constant_result`): a re-trace calls it once instead of tracing the counter. Running frames: stage sees steps in flight (prepare runs while the engine steps), so a frame on a thread stack there is an advisory `running_frame` (reported, never blocks); `validate` checks the thread stacks again in the gap, and only a frame live at the quiesce point votes no (through the plan's accept). A suspended generator or coroutine still blocks at stage. |
| `receipt.py` | **Proposal, needs a human owner** (AGENTS.md, Trust). The one receipt writer: linked `txn` / `proof` / `undo` records sharing a transaction id, appended as JSON lines under a lock. Schema in the docstring and in `../INTERFACE.md` (proposed for v1). |
| `coordinator.py` | Two-phase transactions: prepare (stage and vote), commit, abort, undo of committed participants when a commit fails, a phase-2 deadline (`Lane`, `deadline_missed`, host poison); `prove` after the workload (witness counts per rank, live and during capture); a post-commit rebuild phase (`rebuilds={"R1": fn}`: the engine recaptures or recompiles after the commit, outside the gap under its own deadline, with undo and a rebuild back if it fails, and again after undo; `rebuild` and `recapture_s` in the receipts); `undo` then the plan's data resets (`undo_resets`: R2, R3, ...). Callers with no host use `swap.stage_and_apply` (stage, decide, validate at the apply point, apply): the frame check that blocks is validate's. `StepGapCoordinator` runs a whole transaction in one gap between two steps of the loop owner; `Coordinator` calls participants one by one. `LocalParticipant` runs the kernel in-process (an engine adds its own static checks with `checks=[fn(module, items) -> findings]`, e.g. vLLM's `captured_target` for model code in graph mode); `CallParticipant` is the shape an engine implements over its RPC. `GroupParticipant` / `merge_ranks`: several ranks reached by one call, with per-rank replies in `ranks`. |
| `session.py` | The host's state machine for one lab: a stack of committed transactions, the code fingerprint each participant should show (content hash and code object id of every function the modules' namespaces reach), checked after every abort, rollback, undo and `status()`. Crash, hang, lost participant, a missed phase-2 deadline, failed undo or code changed outside the journal poison it; a poisoned session refuses install (undo stays allowed) until the lab respawns its workers. |
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

Compile units. Prepare also holds the GIL to parse and compile: `compile` is one C call. On
vLLM 0.33's `scheduler.py` (3,783 lines), staging a one-line edit of `Scheduler.schedule`
on CPU (11 runs, medians) went from 56.4 ms of parse+compile (4 parses, 41.5 ms; 2 whole-module
compiles, 15.4 ms) to 26.9 ms (2 parses, 23.0 ms; 2 compiles of the `Scheduler.schedule`
unit, 4.1 ms); the whole stage from 114 to 78 ms. `compile_units` compiles the statement
that holds each target inside empty shells of its classes, with the module's `__future__` and
import statements (CPython 3.12 compiles `mod.attr()` differently when `mod` is a module
import). Each unit's code objects equal the whole-module compile for all 43,457 definitions in
vLLM 0.33 (content hash). The longest single GIL hold is now one parse (11-13 ms here);
the outside-changes comparison (two `ast.dump` calls of about 9 ms) is next.

Analysis off the process. Those parses and dumps still stretched engine steps: at TP4 (8.6 ms
steps) the longest step during a core prepare was 3.4-3.95x the median, per-definition compile
or not. A helper thread does not help (same GIL). So `stage` is split: `analyze` is its text
half (parse both texts, match targets, add the witness, compile the units, find added,
removed and imported names, compare the blanked trees); it needs only the two texts, the
targets and which targets are live, and returns plain data and code objects. Import, locate,
the base check against running code, exec of imports, added definitions and defaults, old/new
pairing and the running-frame check stay in the engine process. `LocalParticipant(analyzer=
swap.ProcessAnalyzer())` runs `analyze` in one long-lived child of the same interpreter (same
`-O` level; requests and replies are `marshal` data, code objects included); a plan module with
`"analyze": "inline"` runs it in-process (for a before/after). The child loads the kernel
from its directory under a private package name, so it works whatever name the parent loaded
the kernel as (the vLLM adapter's `fastrsi_kernel` from a directory named `kernel` killed the
old child on import, and every prepare analyzed in-process without a word). If the child fails,
`stage` analyzes in-process and reports an `analyze_fallback` finding (overridable) with the
reason, the child's last stderr line, and the time the analysis held this process's GIL; the
reason is also in `timings["analyze_fallback"]`.

The child boots when the participant is created, not in the first prepare: a `LocalParticipant`
given an analyzer calls its `start()` (idempotent: a live child is kept), and the child writes a
ready byte once the kernel is imported (`wait_ready(s)` waits for it, `state()` reports it
without blocking). Before, the first prepare after engine start paid the spawn: 1.5-2.2 s on an
engine host, against 0.12 s for later prepares (our NVL runs); on CPU with the toy engine, after
a 2 s idle, the first prepare took 32.7 ms against 5.0 ms for the second, and with the warm start
5.4 ms against 5.1 ms (5 runs, medians; our 2026-11-05 CPU run). A child that dies is respawned
lazily, by the next prepare, whose timings then say `analyze_spawned` (and
`analyze_boot_wait_s`, the time it waited for the boot); if the respawn fails, the prepare
reports `analyze_fallback` as above. The same edit of
`Scheduler.schedule` on CPU, prepared by a core `LocalParticipant` in a background thread while
the main thread ticks every 0.5 ms (9 runs each; our 2026-11-05 CPU run):

| Analysis | longest GIL hold, median (max) | prepare wall, median |
|---|---|---|
| before (in-process, previous kernel) | 20.7 ms (219.7 ms, a gc pass over the new ASTs) | 83 ms |
| in-process (`"analyze": "inline"`) | 20.4 ms (221.3 ms) | 77 ms |
| child process (`ProcessAnalyzer`) | 1.4 ms (1.8 ms) | 82 ms |

Same staged code (new hash `73d9f00e04cc058d`) and the same findings (none) in all three;
`test_process_analysis_stages_the_same_code_and_findings` checks code objects (`==`: bytecode,
constants, names, line tables) and findings for 14 kinds of edit.

Targets found by the analysis. A plan module may omit `targets` (or say `"auto"`): the
analysis then diffs the running text against the patch from the two trees it parses anyway
and takes every def, lambda and generator expression whose AST differs as a target (added and
removed definitions are found as before), the same list as `changed_targets`. Most
definitions are settled by comparing their source bytes (equal text, equal AST); only those
whose text differs pay an `ast.dump` compare. The live side gives the analysis every live
code key of the module (no target list yet), and re-analyzes with the explicit list in the
rare case where the heap scan then finds a live closure the namespace did not
(`timings["analyze_again"]`). Prepare of `vllm/v1/core/sched/scheduler.py` (vLLM 0.33,
3,783 lines) on CPU, 15 runs each, median wall time of planning plus `prepare` (our
2026-11-05 CPU run):

| Edit | before: `changed_targets` + prepare | explicit targets (new planner) | targets omitted |
|---|---|---|---|
| 1 function, in-process analysis | 126.8 ms | 114.5 ms | 76.2 ms |
| 1 function, `ProcessAnalyzer` | 136.2 ms | 123.7 ms | 84.9 ms |
| 3 functions, in-process analysis | 127.6 ms | 114.7 ms | 75.8 ms |
| 3 functions, `ProcessAnalyzer` | 135.7 ms | 124.1 ms | 87.1 ms |

The planner alone took 51.6 ms before and 38.8 ms with the text shortcut; omitting the targets
removes it, and finding them inside the analysis costs about 5 ms. What remains of the 76 ms is
the analysis (two parses 21 ms, the outside-changes compare 27 ms, unit compiles 8 ms).
Across 54 generated edits of 9 vLLM 0.33 modules (one body, three bodies, a lambda, a generator
expression, a comment, an added function, a removed method), the targets equal an
`ast.dump`-only reference and the staged findings, items and code objects equal those of
explicit targets, in-process and in the child.

Trees the caller already parsed. A caller with its own planner (one that plans module
constants or dataclass fields itself) has parsed both texts already; `stage(module, source,
trees=(base tree, patch tree))` analyzes those instead of parsing again, and gives them back
unchanged (the analysis logs every edit it makes to them, the witness call and the blanked
bodies of the outside-changes compare, and takes it back before returning). The trees must be
`ast.parse` of exactly the text the module runs (`base_source`, or its file) and `source`;
nothing checks that. `changed_targets(base, new, trees=...)` takes them too, so a caller that
wants only some of the changed definitions picks them and passes them as explicit targets,
still with no parse. The trees are used when the analysis runs in this process; a
`ProcessAnalyzer` child parses its own copy (an AST cannot cross the pipe, and that parse does
not hold this process's GIL). Stage of a one-line edit of `Scheduler.schedule` in vLLM 0.33's
`scheduler.py` (3,783 lines) on CPU, in-process analysis, targets omitted, 21 runs each (our
2026-11-05 CPU run; the caller's own parse of both texts took 21.9 ms and is not counted):

| Stage | wall, median (min-max) | parse inside stage |
|---|---|---|
| before (previous kernel, parses both texts again) | 76.4 ms (72.9-83.4) | 19.0 ms |
| this kernel, no trees | 74.1 ms (73.7-76.2) | 18.2 ms |
| this kernel, `trees=` the caller's | 55.0 ms (54.2-56.4) | 0 |
| this kernel, `ProcessAnalyzer` (trees not used) | 79.9 ms (77.8-112.6) | 20.9 ms, in the child |

The staged code is the same in all four (new hash `73d9f00e04cc058d`, no findings). Across the
54 generated edits of 9 vLLM 0.33 modules used above, stage with the caller's trees (twice
with the same trees) stages the same findings, items and code objects as with explicit
targets, and the trees come back unchanged (`ast.dump` with positions).

Class bodies. New defaults of a method (`def f(self, x=CONST)`) and added methods (their
decorators and defaults) are evaluated as in the live class body: the scratch class's
metaclass is a `_LiveBody` whose namespace is a `ChainMap` that reads the live class's names,
then the patch's new imports, then (as any class body) the module globals; writes go to a
fresh dict. So `CONST` from the class body, not a module-level `CONST`, and a decorator defined
in the class work, and the live class is not touched before commit. Every name of the live
class is visible, also those defined below the method in the file. A decorator that changes
a class-level object in place does so at stage, as it would at import.

Several modules in one transaction. A participant stages a plan's modules so that each sees
the names the others' patches add (`overlay`, from `swap.staged_names`): module B's
`from .a import new_helper`, where A's patch adds `new_helper`, takes A's staged object, and B's
added definitions and top-level defaults can use it at definition time. A module that fails
is staged again after the others, so the plan's module order does not matter. Nothing is put
in `sys.modules` or any module dict before the commit, which applies every module through
one journal mark (all or none; undo removes them together), as vLLM #51899 needs (it adds a
helper in one file and imports it in another).

With the one-by-one `Coordinator` instead (each rank switching at its own boundary), the same
commit and undo gave at least one step with mixed versions in 20 of 20 runs on CPU; the
step-gap coordinator gave none in 20 of 20.

## Walkthrough: a commit that hangs

The gap is a global barrier: without a deadline, a rank that hangs in commit stalls every
step for as long as it hangs. Same toy cluster, rank1's RPC hangs 0.6 s in commit; once with
no deadline, once with `phase2_deadline_s=0.2` (validate and commit by 0.1 s, rollback by
0.2 s). Run from `template/livelab`:

```python
import sys, threading, time
from kernel import toy, coordinator as C, locate as L

ENG = toy.source("engine")
line = next(i for i, s in enumerate(ENG.splitlines(), 1) if "def step" in s)
plan = {"modules": {"toy.engine": {
    "source": ENG.replace('self.ran.append("base")', 'self.ran.append("v1")'),
    "base_hash": L.text_hash(ENG), "targets": [["Engine.step", line]]}}}


def lab(tag, deadline):
    engines, parts = {}, []
    for rank in ("rank0", "rank1", "rank2"):
        toy.load_copy(f"{tag}_{rank}")
        engines[rank] = sys.modules[f"{tag}_{rank}.engine"].Engine()
        lp = C.LocalParticipant(rank, alias={"toy": f"{tag}_{rank}"})

        def rpc(method, payload, lp=lp, rank=rank):        # rank1 hangs 0.6 s in commit
            time.sleep(0.6 if (rank, method) == ("rank1", "commit") else 0.001)
            return getattr(lp, method)(payload)
        parts.append(C.CallParticipant(rank, rpc))
    cluster = sys.modules[f"{tag}_rank0.engine"].Cluster(engines)
    co = C.StepGapCoordinator(parts, cluster.at_gap, phase2_deadline_s=deadline)
    stop = threading.Event()
    loop = threading.Thread(target=cluster.run, args=(stop,))
    loop.start()
    time.sleep(0.02)
    res = co.run(dict(plan))
    time.sleep(0.8)                                         # the hung commit returns
    stop.set()
    loop.join()
    st = co.status(res["txn"])
    print(f"deadline {deadline}: outcome {res['outcome']}, state {st['state']}, poisoned "
          f"{st['poisoned']}, missed {st['missed']}, gap_s {res['receipt']['gap_s']}, "
          f"longest stall {cluster.longest_stall():.3f} s, mixed steps "
          f"{sum(1 for t in cluster.log if len(set(t)) > 1)}")
    if st["poisoned"]:
        print("  next run:", co.run(dict(plan))["outcome"])


lab("nodl", None)
lab("dl", 0.2)
```

Real output:

```
deadline None: outcome committed, state committed, poisoned False, missed [], gap_s 0.605508, longest stall 0.610 s, mixed steps 0
deadline 0.2: outcome rolled_back, state aborted, poisoned True, missed ['rank1'], gap_s 0.103268, longest stall 0.108 s, mixed steps 0
  next run: refused
```

With the deadline the loop stalled 0.11 s instead of 0.61 s; rank0's commit was undone in
the gap; rank1's late commit, when it arrived, saw its deadline had passed and applied
nothing; the host is poisoned, so the next transaction is refused (undo is still allowed).

## Golden tests

`python3 -m unittest kernel.test_kernel -v` from `template/livelab` (CPU, about 5 s; one
test builds a 2.5 M-object heap, one waits out a 0.6 s hang, seven use an analyzer child; one
imports torch on CPU and is skipped without it). `python3 -m unittest` also runs the conformance cases on
the toy lab (`../conformance/`, about 1.5 s more).

```
test_a_partial_undo_stays_partial_when_undone_again ... ok
test_a_step_in_flight_at_prepare_is_advisory_and_the_gap_decides ... ok
test_accept_list ... ok
test_accept_record_with_a_subject_covers_only_that_subject ... ok
test_auto_target_only_the_heap_finds_live ... ok
test_auto_targets_stage_like_explicit_ones ... ok
test_chained_edit_stages_against_the_installed_text ... ok
test_changed_targets_match_the_ast_reference ... ok
test_child_runs_when_the_kernel_has_another_package_name ... ok
test_added_method_uses_a_decorator_defined_in_the_class ... ok
test_closure_change_is_a_finding_not_an_exception ... ok
test_closure_made_after_the_swap_runs_the_old_code_after_undo ... ok
test_closure_with_no_base_version_is_a_finding ... ok
test_code_identity_and_hash ... ok
test_commit_failure_undoes_the_committed ... ok
test_edit_that_changes_no_code_is_refused ... ok
test_engine_checks_add_findings_at_prepare ... ok
test_editing_a_deleted_definition_is_a_finding_not_a_crash ... ok
test_fallback_to_in_process_is_a_finding ... ok
test_gap_stays_small_while_prepare_is_slow ... ok
test_group_proof_reports_ranks_agree ... ok
test_group_witness_is_kept_per_rank_and_uneven_is_a_finding ... ok
test_half_committed_group_is_undone ... ok
test_hung_commit_ends_the_gap_at_the_deadline ... ok
test_linked_receipts_and_measured_times ... ok
test_module_imports_a_name_another_module_of_the_txn_adds ... ok
test_name_and_txn_record_still_accept ... ok
test_new_defaults_see_the_live_class_body ... ok
test_one_by_one_commit_mixes_versions ... ok
test_prepare_never_holds_the_loop_longer_than_the_budget ... ok
test_record_call_runs_in_reverse_order_and_a_failure_is_a_finding ... ok
test_process_analysis_keeps_the_gil_free ... ok
test_process_analysis_stages_the_same_code_and_findings ... ok
test_removed_definition_is_a_finding_and_undo_restores_it ... ok
test_repair_cell_journals_through_record_attr_and_record_call ... ok
test_repair_cell_sets_missing_field_and_undo_removes_it ... ok
test_repair_that_cannot_be_undone_is_a_finding ... ok
test_rank_split_names_the_ranks_and_whether_one_function_differs ... ok
test_ranks_that_disagree_after_commit_are_a_finding ... ok
test_running_frame_never_runs_the_new_code ... ok
test_second_genexpr_edit_on_top_is_not_a_base_mismatch ... ok
test_stage_compiles_only_the_changed_definitions ... ok
test_stage_findings_carry_the_checked_code_hash ... ok
test_stage_with_the_callers_trees_parses_nothing_and_stages_the_same ... ok
test_standing_accept_expires_when_code_changes ... ok
test_step_gap_commit_and_undo_never_mix_versions ... ok
test_step_gap_failed_commit_is_undone_in_the_same_gap ... ok
test_swap_a_generator_expression ... ok
test_swap_reaches_every_holder ... ok
test_timed_out_caller_resolves_the_outcome_with_status ... ok
test_undo_after_a_deadline_rollback_leaves_every_rank_undone ... ok
test_undo_of_an_older_txn_reports_the_later_ones ... ok
test_undo_restores_descriptor_side_dict_and_doc ... ok
test_undo_restores_exact_base ... ok
test_undo_restores_parameter_data_and_module_submodules ... ok
test_undo_runs_the_declared_data_reset ... ok
test_undo_skips_the_heap_pass_when_no_closure_outlives_the_call ... ok
test_validate_catches_a_change_between_prepare_and_gap ... ok
test_vote_no_changes_nobody ... ok
----------------------------------------------------------------------
Ran 59 tests in 7.7s

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
- **Phase 2 has a deadline.** `phase2_deadline_s` (default 5 s) bounds the gap: validate and
  commit by half of it, the rollback by all of it. A participant that misses it is failed
  (`deadline_missed`), the others are undone in the gap, and the host is poisoned (its late
  call may still run): `run` is refused, undo is allowed, `recovered()` after a respawn. A
  participant must not switch after the `deadline_at` it was given.
- **Ranks are kept apart.** A group participant (one call reaches several ranks) replies with
  every rank's reply in `ranks`. Proofs keep witness counts per rank (`workers/rank0`, ...);
  ranks whose counts differ by more than 1% of the largest (at least 1 call) are
  `witness_uneven` (ranks step together and are read between the same two steps, so code
  that runs on every rank counts the same). The txn receipt keeps each rank's code hashes and
  `commit_hashes.ranks_agree` per group; `agree` is false if any comparison differs. Ranks
  that disagree are also a `ranks_disagree` finding in the receipt naming the split ("ranks 1
  vs 0,2: <functions>"), blocking in `decisions`; not overridable when one function has
  different hashes on different ranks (overridable when only some ranks hold a function).
  The outcome stays what happened (committed): undo or respawn.
- **Never assume a timed-out transaction failed.** The caller gets `in-doubt`; the host's
  `status(txn)` gives the real state (prepared, committed, aborted, undone, in-doubt). A phase
  2 that had not started when the caller gave up is cancelled. Once undo ran on a participant,
  its state is `undone` whether it had committed or not (a rank whose late commit applied
  nothing too), so every rank of an undone transaction says the same.
- **An edit that changes nothing is refused.** A plan whose targets are all unchanged (a
  comment, or a module constant whose `outside_changes` was accepted), with nothing added,
  deleted or repaired, is an `empty_change` finding, not a commit that reports success.
- **Chained edits stage against what is live.** Each participant remembers the text every
  committed transaction installed and the code objects it swapped in; a later plan for the
  same module is staged against that text, and those code objects are not `base_mismatch`.
  Undo of a transaction forgets it and every later one (the journal undoes them too).
- **Deletions are edits too.** A definition the edit deletes (module function, method,
  class) but the live module still defines is a `removed_definition` finding; accepted, it is
  deleted through the journal, and undo puts the same object back. Copies made by
  `from x import f` elsewhere still hold it (not covered).
- **Undo is exact for code, not for data.** The journal restores the same code and attribute
  objects and removes every name it added. Data the new code wrote (requests, cache entries)
  needs the reset the plan declares in `undo_resets`; a missing hook is a `reset_failed`
  finding. An in-place re-init (a Triton kernel rebuilt) journals its own inverse with
  `record_call(undo_fn, label)`; it runs at its place in the reverse order, and if it raises,
  the undo reports `undo_incomplete` with the label and goes on.
- **Attributes outside `__dict__` are journaled through `getattr` / `setattr`.**
  `record_attr` reads the instance dict only when that is exact (default `__setattr__` and
  `__delattr__`, no data descriptor of the type owns the name). A function's `__doc__`, a
  property, torch's `Parameter.data`, an `nn.Module` submodule (kept in `_modules`) or a slot
  is recorded with `getattr`, and undo puts it back with `setattr`, or `delattr` when it was
  absent.
- **Closures the new code made come back too.** A swapped function's new code makes function
  objects at run time (a closure it returns, a callback it registers), which no journal entry
  holds. `apply` journals, first, the pairs (new nested code, old nested code of the same
  qualname and ordinal) of every code it installs; at undo, last, and only when there are
  pairs, one `gc.get_referrers` pass over the new nested codes (frozen objects included) finds
  the functions that run one and gives each its old code (`closures_restored` in the undo
  result). A function made from a nested definition the edit added, or one whose free
  variables changed, has no base code: a `closure_not_restored` finding with the count and the
  qualname, and the undo is not ok: the transaction's state is "partial", and undoing it again
  repeats the finding instead of answering "undone". Generator expressions are left out (their function never
  outlives the call). Cost: in a CPU process that imported vLLM 0.33's model runner (0.86 M
  gc objects), the pass took 57.6 ms median (one `get_objects` plus a Python loop: 84.1 ms;
  our 2026-11-05 CPU run); it holds the GIL in one piece, inside the undo's gap. So it runs only
  for the new nested codes whose reference count grew since apply (a live function holds one):
  on our 2026-11-05 GPU run (1x B200, Qwen3-8B, TP1) undo of a `Scheduler.schedule` swap went
  from 0.12 s (the pass, 0 closures restored) to under 1 ms on the engine core, and undos of
  swaps that add a transient lambda (worker and core) stalled streams 1.13-1.21x the median
  step. When a closure is alive the pass still runs in the gap (0.12-0.15 s on that engine core).
- **The heap scan unfreezes the gc.** Engines call `gc.freeze()` after startup, and
  `gc.get_objects()` and `gc.get_referrers()` both skip frozen objects.

## Not covered yet

- A participant stuck inside `apply` at its deadline (not before it) can still switch late;
  the undo the host queued behind it then undoes it, but steps in between run its new code.
  The session is poisoned either way.
- A unit compile does not check that the rest of the patched module compiles: a
  compile-time error (not a syntax error) outside the targets shows up only as
  `outside_changes`, or at the next restart if that finding was accepted.
- Nested targets (closures) still need one heap scan, whose `get_objects` call holds the GIL
  in one piece. Copies of a function object made with `types.FunctionType(code, ...)` are not
  found by the namespace path.
- The host's transaction table lives in memory: a host crash loses `status`.
- The analyzer child is this interpreter, started with the same `-O` level; equivalence is
  tested for that case only. A child that dies or does not answer within `timeout_s` is
  killed and the prepare analyzes in-process, holding the GIL as before; the
  `analyze_fallback` finding blocks the vote until accepted, and each later prepare tries a
  new child first.
- Names a patch adds that another module of the same transaction uses are found only through
  `from m import name` at module level (and in added definitions, top-level defaults, and,
  through `_LiveBody`, method defaults; that last case is not tested). A class-level
  statement of an existing class that needs them is evaluated against the live module and
  fails; two patches that import from each other are not staged.
- The closure undo finds closures by their code object. A closure copied with
  `types.FunctionType(code, ...)` is found too; one whose code was replaced again after the
  swap (another tool, not the journal) is not. A closure's defaults, evaluated by the new code
  when it was made, stay. The `get_referrers` pass is not gated: it runs even when the new
  code never ran.
- `trees=` is trusted: trees parsed from another text than the one the module runs give
  wrong code, with no finding.
- A chained edit of a function the earlier edit did not touch, when the earlier edit moved
  lines above it: the function's running code keeps the file's line, the text the
  participant stages against has another, and the target is not found (`no_live_function`).
- The installed text is what the plan asked for. When `outside_changes` was accepted, the
  live module is neither that text nor the file, and the next stage compares against the plan.
- Functions defined with `exec` into foreign globals are not found (the scan matches by
  `__globals__`, so module copies stay apart).
- Real RPC: the toy tests reach ranks through `CallParticipant` over a 1 ms in-process call;
  the vLLM side (collective_rpc from the engine-core busy loop) is the next step.
