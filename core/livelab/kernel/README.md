# Transaction kernel

The live-edit transaction kernel, built bottom up: find the code a running engine executes,
record every change to it so undo is exact, then swap, witness and coordinate. Pure Python,
standard library only; it imports no engine. This folder holds the bottom layer so far:
findings, locate, journal, witness, the R0 swap (a function-body swap), two-phase
transactions (participants and their host), the session, and a toy lab that passes the
template's conformance cases c0-c10.

## Contract

- **Locate** finds the running function by qualname across live objects.
  `locate(module, qualname, firstlineno)` resolves a target to the code objects the module's
  live functions run and the function objects that hold them, found by a scan of the gc heap
  (every function whose globals are the module's dict). It also names every holder: module
  dicts (including `from x import f` copies), class dicts, closure cells of decorator
  wrappers, `__wrapped__`, bound methods saved as callbacks, and instance attributes.
  Identity is (file, `co_qualname`, `co_firstlineno`); for a decorated function the first
  line is the first decorator's. `content_hash(code)` hashes content only (bytecode, names,
  variables, argument counts, flags, constants, nested codes), so a moved function keeps its
  hash and an edited one does not. `module_functions(mod)` is the namespace walk that needs
  no heap scan.
- **Journal** records a code change and undoes it exactly. Each participant keeps one
  ordered undo log. Every change goes through a method that records the inverse first:
  `set_code` (`__code__`, `__defaults__`, `__kwdefaults__`), `set_attr` / `record_attr`,
  `set_item` / `record_item`, `delete`, `record_call` (an in-place re-init only a function
  can undo), `record_closures`, and `repair(...)` for hand-written repair cells.
  `mark()` returns the log position; `undo_to(mark)` replays inverses newest first, back to
  the same code and attribute objects, with added names removed.
- **Change analysis** (`swap.py`, the text half of the swap) reads only the two module
  texts. `changed_targets(base, new)` lists every definition (function, method, nested
  function, lambda, generator expression) whose AST differs, with the functions that enclose
  it (they make its closures). `analyze(base, new, targets, filename, key_module, live)`
  resolves the targets (auto: that list; explicit: by qualname and ordinal, so a patch may
  move lines), adds the witness call to each new body, compiles the patched module, and
  finds added, removed and imported names. It blanks target bodies, added definitions,
  imports and docstrings in both trees and compares them: any other difference is
  `outside_changes`, an edit a code swap would not apply. A docstring edit alone is not a
  target. It touches no live object and returns plain data and code objects; trees a caller
  passes (`trees=`) come back unchanged.
- **Stage and apply** (`swap.py`, the live half) install the edit into the running module.
  `stage(module, source, targets="auto")` runs the analysis, locates the running code
  (module namespace first, the heap only for nested targets), checks it against the base
  text, builds new imports, functions, methods and defaults in a scratch namespace (a
  method's body reads the live class's names; the live class is not touched), and pairs each
  running code object of a target and of its enclosing functions with its new version, so
  closures made later run the new code. Nothing is visible yet, and nothing of the edit has
  run: definitions the edit adds are built at apply, so their decorators and class bodies run
  only for a transaction that commits. Findings: `no_live_function`,
  `not_in_patch`, `base_mismatch`, `outside_changes`, `removed_definition`, `closure_changed`
  (free variables differ; not overridable), `defaults_not_evaluated`, `running_frame`
  (advisory at stage: a step in flight), `patch_invalid`, `new_fields`, `constructor_edit`, `line_aliased`,
  `jit_rebuild` (advisory) and `jit_target` (below). `validate(staged)` re-checks at
  the apply point: the same code objects are installed, and no thread stack holds a frame of
  the old code. `apply(staged, journal, witness)` makes it visible, every change through the
  journal (closure pairs first, so undo also takes back closures the new code made).
  `stage_and_apply` is stage, decide, validate, apply in one call for a host with no two
  phases.
- **Triton kernels** (`jit.py`). A target held by a JIT object (`@triton.jit`, autotune,
  heuristics) runs what the JIT compiled from its text, never the swapped bytecode. The
  analysis cuts the kernel's new text from the clean patch text (no witness call in it) and
  notes whether the parameters or decorators changed. `stage` then asks `jit.plan`: a Triton
  `JITFunction` (duck-typed: `jit.NEED`) whose parameters and decorators are unchanged is
  rebuilt in place (advisory `jit_rebuild`); anything else is a blocking `jit_target`. At
  apply the kernel gets its new `_src`, `raw_src` and first line, and the kernel and every
  JIT kernel of any loaded module that calls it, at any depth, lose their memoised `hash`,
  `used_global_vals` and compiled kernels (`device_caches`); autotuner caches over them are
  cleared. All of it is snapshotted and journaled first: undo puts back the old compiled
  kernels and tuning choices, with no recompile. The bytecode swap of the kernel's `.fn`
  still runs, so fingerprints, chained edits and `base_mismatch` work as for any function.
  Callers are found by object identity (what the caller's text names resolves to the edited
  JIT object), never by bare name, so a same-named kernel in another module is not a caller,
  and a helper imported under an alias is still found. An added kernel's counter also goes
  into the other added kernels that call it.
  The witness counter goes into `pre_run_hooks` of the kernel and of every caller (a helper
  inlined into a caller runs only the caller's hooks), each kernel with its own launch counter
  (`witness_check` replies `launches`: {key: {module.qualname: launches}}); a hook that fired
  on a kernel with no binary compiled after the edit proves nothing. So a kernel that never launches, or
  replays only inside a graph captured before the edit, reads 0. A `@triton.jit` definition
  the edit adds is built while linecache holds the patch text, so `inspect.getsourcelines`
  reads its lines; it is counted through the hooks of the rebuilt kernels that name it.
  Imports no `triton`: CPU CI uses the stand-in runtime in `test_jit.py`.
- **Detectors** in `stage`. *New fields:* a changed or added method that reads `self.X`,
  where only the edited `__init__` sets X (no old method sets it, no class body defines it),
  gives a blocking `new_fields` finding with `"route": "component_restart"`: instances built before the swap
  lack X, so the new code would raise AttributeError at its first call. The fix is a rebuild
  of the objects or a restart (component restart); nothing here sets fields on live objects. *Constructors:*
  a target named `__init__`, `__new__` or `__post_init__` gives a blocking, overridable
  `constructor_edit` finding with `"route": "component_restart"`: instances built before the swap keep what
  the old constructor set (vLLM #61323 edits `RMSNorm.__init__`; the model's norms were built
  at start-up, so the commit changed nothing they hold and the witness read 0). The kernel
  does not search the heap for instances, so every constructor edit blocks; accept it when the
  class is built per request or per step. *Line maps:* a
  second edit of a module whose first edit shifted its lines passes `base_source` (the
  installed text), `own_codes` and `line_maps` (each committed swap's `staged.line_map`,
  newest first); a definition no swap replaced still carries its original lines, and the maps
  lead from the installed text's line to it (`timings["line_map_aliases"]`). Without a map
  that leads there, the one live function with the target's qualname is taken, with an
  advisory `line_aliased` finding.
- **Witness** proves that swapped code ran, per participant. The swap puts
  `__fastrsi_hit__("<module>:<qualname>:<line>")` at the top of each new body (`inject`,
  after the docstring; for a lambda or generator expression, in the expression: a generator
  expression counts once per item). The module global is bound to the participant's
  `Witness.hook`; undo binds the no-op `stub` instead of deleting it, since code compiled or
  captured while the edit was installed may still call it. `Witness.check(keys, old_codes)`
  turns counts into findings, so a swap that never ran is never a pass: zero live calls is
  `running_frame` when a frame of the old code is still running (on a thread stack, or a
  suspended generator or coroutine), else `zero_witness` (not reached). Calls made while the
  engine captures a graph (`Witness(capturing=fn)`, `with witness.capture():`) count apart:
  they prove capture, not a run, so zero live calls with capture calls is a `zero_witness`
  routed to CUDA graph recapture that asks for a device-side probe. With torch loaded, the hook is marked
  `assume_constant_result`, so Dynamo calls it while tracing and never traces it.
- **Participant** (`coordinator.py`, part 1) is one side of a two-phase transaction: a worker
  rank, the engine core or the API server. Every call returns an ok-dict and never raises:
  `prepare(plan)` stages every module of the plan (nothing visible; the reply's `ok` is the
  vote, after the plan's `accept`), `validate(ref)` re-checks the staged identities,
  `commit(ref)` applies at the participant's quiesce point (refusing a commit that starts
  after `deadline_at`), `abort`, `undo` (back to the transaction's journal mark, saying which
  later transactions it undid too), `status`, `proof` (code hashes and witness counts),
  `witness_check`, `reset`, `fingerprint`, `at_base`. `LocalParticipant` runs the kernel in
  this process; `CallParticipant` reaches one in another process through `call(method,
  payload)`. Chained edits: a later plan for a module is staged against the text the last
  committed transaction installed, with that participant's own code and line maps. A plan
  that would change nothing is `empty_change`. `at_base` tells names its transactions and
  cells added to a module (journaled) from modules the engine imported lazily into it since
  base (vLLM's LazyLoader binds `llguidance.hf` at first use, #62872): the first stay
  `extra_names` and make it false, the second are listed in `lazy_imports` and do not. A
  base name that is gone is listed in `missing_names` and makes it false too.
- **Host** (`coordinator.py`, part 2) runs one transaction over every participant.
  `Coordinator.run(plan)`: phase 1, prepare on all while steps go on (a no vote, or a
  participant missing from a group's reply, aborts all); phase 2, validate on all, then
  commit on all or abort on all, and a commit that fails undoes every participant that
  committed (`rolled_back`). `StepGapCoordinator` runs phase 2 inside one gap between two
  steps of the loop owner, so when every participant answers within `phase2_deadline_s`,
  ranks that step together never run a step on different versions. When one misses it (a
  commit or undo that started but has not returned), the gap still ends at the deadline:
  the host is poisoned, and until the lab respawns the participants, steps may run on mixed
  versions (the late commit can land after steps resumed, before its queued undo runs;
  `test_late_commit_poisons_and_steps_may_mix`). A group participant (several ranks behind one call) merges its ranks' replies
  (`merge_ranks`); one that declares `expected` ranks and lacks some is a blocking, not
  overridable `participant_missing`. With `phase2_deadline_s`, a participant that misses
  the deadline is treated as failed and poisons the host (only undo, until `recovered()`).
  `prove` and `undo` close the transaction. Each call appends a linked receipt (`txn`,
  `proof`, `undo`) through `receipt.ReceiptWriter`.
- **Session** (`session.py`) strings transactions together for a live lab: base, edited or
  poisoned. It keeps the stack of committed transactions (`uninstall` undoes the newest, or
  down to a given one) and the code fingerprint every participant should show, checked
  after every abort, rollback and undo and on every `status()`. A fingerprint that differs
  (`tampered`), a failed undo, a transaction left in doubt, a missed phase-2 deadline or a
  fault the lab saw poisons it: `install` is refused (`session_poisoned`), `uninstall` and
  `recovered()` are allowed.
- **Toy lab** (`toy_lab.py`, `ToyLab`) puts the toy engine behind the interface verbs
  (`install`, `check`, `reset`, `uninstall`, `status`) and the conformance harness
  (`patches`, `fresh_check`, `inject`): two ranks step together in a loop thread, the host
  is a `StepGapCoordinator` inside a `Session`, ranks are reached through `CallParticipant`.
  CI runs `python3 -m conformance --lab kernel.toy_lab:ToyLab` from `core/livelab`, the
  folder that holds `kernel/`.
- **Findings** are the one result schema. Every call returns `{"ok": bool, ...,
  "findings": [...]}`. A finding says what was checked, what was observed against what was
  expected, why it matters, the actions and their cost, and the risk of ignoring it.
  `decide(findings, accept)` is the one accept mechanism (a check name, or a record of who
  accepted it and why; a standing record expires when the checked code changes).

| File | What it owns |
|---|---|
| `findings.py` | The findings schema, the catalog of checks, `decide`. No imports. |
| `locate.py` | Code identity and `content_hash`; the heap and namespace scans; `locate`, `holders`. Imports only `findings`. |
| `journal.py` | `Journal`: typed entries and their inverses, marks, `undo_to`, repair cells. Imports `findings` and `locate`. |
| `swap.py` | The change analysis (`analyze`, `changed_targets`: text only) and the live half (`stage`, `validate`, `apply`, `stage_and_apply`, `closure_pairs`, the scratch namespace). Imports `findings`, `jit`, `journal`, `locate`, `witness`. |
| `jit.py` | The Triton rebuild: `text_of` (text half), `plan` (find the JIT object, its callers in every loaded module by identity, its autotuners), `apply` (rebuild, reset, launch hooks, all journaled), `arm_added`, `seeded_lines`. Standard library only; imports nothing of the kernel. |
| `coordinator.py` | Participants (`Participant`, `CallParticipant`, `LocalParticipant`) and the host (`Coordinator`, `StepGapCoordinator`, `GroupParticipant`, `merge_ranks`, `expect_ranks`, `Lane`). Imports `findings`, `journal`, `locate`, `receipt`, `swap`, `witness`. |
| `session.py` | `Session`: the transaction stack, expected fingerprints, poison. Imports `coordinator`, `findings`, `locate`. |
| `toy_lab.py` | `ToyLab`: the conformance lab on the toy engine (2 ranks, step-gap host, session). |
| `receipt.py` | The receipt writer: linked `txn` / `proof` / `undo` records in an append-only JSON-lines file, validated against the proposed schema. A proposal that needs a human owner (see its docstring). |
| `witness.py` | `Witness` (live and capture counts, `check`), `inject`, `stub`, `trace_safe`, and the running-frame scans `stack_codes` / `live_codes`. Imports `findings` and `locate`. |
| `toy/` | A toy engine for the tests: ops with a decorator and a closure factory, an `Engine` that saves a bound-method callback, a busy loop, and `load_copy`, which gives each test its own module copy. |

## Invariants

- **Nothing raises to the caller.** `locate` of a module that does not import is a
  `patch_invalid` finding; a qualname with no live function is `no_live_function`. An inverse
  that fails during `undo_to` is an `undo_incomplete` finding, and the other inverses still
  run.
- **Checks advise.** A check returns a finding; the caller's accept list decides. Physical
  limits (`no_live_function`, ...) are not overridable.
- **Undo is exact for code, not for data.** The journal restores the same code and attribute
  objects and removes every name it added. Data the new code wrote (a token it appended, a
  cache entry) is not the journal's: a later PR adds declared resets.
- **Attributes outside `__dict__` go through `getattr` / `setattr`.** `record_attr` reads the
  instance dict only when that is exact (default `__setattr__` and `__delattr__`, and no data
  descriptor of the type owns the name). A function's `__doc__`, a property, torch's
  `Parameter.data`, an `nn.Module` submodule (kept in `_modules`) or a slot is recorded with
  `getattr`, and undo puts it back with `setattr`, or `delattr` when it was absent.
- **The heap scan unfreezes the gc.** Engines call `gc.freeze()` after startup, and
  `gc.get_objects()` and `gc.get_referrers()` both skip frozen objects. The scan yields the
  GIL in slices (`SLICE_S`); the one `get_objects` call cannot be sliced.
- **The witness's frame check stays cheap.** `check` walks thread stacks, and looks for
  suspended frames only when an old code can be suspended (generator, coroutine), from that
  code's referrers. `gc.get_referrers` skips frozen objects, so a generator suspended since
  before `gc.freeze()` is not seen: the price of not scanning the whole heap.

## Golden tests

From the repo root (CPU, a few seconds; the torch cases are skipped without torch; CI runs the same command in `.github/workflows/cpu-tests.yml`):

```
python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
```

Each test's docstring says what it proves and what it does not.

| Test | Pain point |
|---|---|
| `test_locate_finds_every_holder_of_a_running_function` | a probe that misses a `from x import f` copy, a decorator's original or a saved callback, on a frozen heap |
| `test_a_target_that_does_not_resolve_is_a_finding_not_an_exception` | a lookup that crashes the caller, or reports success with nothing found |
| `test_code_identity_and_hash` | a hash that changes when a function only moves, or stays when it is edited |
| `test_standing_accept_expires_when_code_changes` and two more `AcceptRecordTest` cases | a standing accept that keeps covering a check after the checked code changed |
| `test_set_code_reaches_every_holder_and_undo_restores_exact_base` | an undo that leaves one holder on the new code |
| `test_undo_restores_descriptor_side_dict_and_doc` | a repair of `__doc__`, a property or a custom `__setattr__` name that undo left in place |
| `test_undo_restores_parameter_data_and_module_submodules` | the same for torch `Parameter.data` and `nn.Module` submodules |
| `test_record_call_runs_in_reverse_order_and_a_failure_is_a_finding` | an in-place re-init with no undo entry |
| `test_calls_are_counted_per_participant_and_zero_calls_is_not_reached` | a swap that never ran on one participant reported as a pass |
| `test_a_running_loop_and_a_suspended_generator_are_running_frame` | a loop entered before the swap, or a suspended generator on no thread's stack, read as "not reached yet" when it can never run the new code |
| `test_capture_calls_are_counted_apart_from_live_calls` | 13280 calls per rank after a CUDA-graph recapture, all made while capturing: proof of capture, not of a run |
| `test_the_counter_keeps_the_docstring_and_counts_lambdas_and_genexprs` | a counter that breaks a docstring, or miscounts code with no statement body |
| `test_the_witness_is_a_constant_for_dynamo` | a recompile that failed in Dynamo on a traced counter |
| `test_a_changed_body_is_a_target_and_its_new_code_counts_its_calls` | an analysis that misses a closure's maker, or compiles a body that never says it ran |
| `test_comment_and_docstring_edits_are_no_target_and_a_constant_edit_is_outside` | a docstring swap reported as a false zero_witness; a constant edit a code swap silently does not apply |
| `test_targets_that_do_not_resolve_are_reported_and_new_names_are_found` | a target that does not resolve crashing the analysis; an added helper or import read as an outside change |
| `test_the_callers_trees_come_back_unchanged` | a planner that plans its next edit on a tree still holding the last edit's witness calls |
| `test_swap_reaches_every_holder` | a swap that misses a `from x import f` copy, a decorator's original, a saved callback or a closure made after the swap |
| `test_undo_restores_exact_base` | an undo that leaves an added method or import behind, or deletes the witness global that compiled code still calls |
| `test_closure_change_is_a_finding_not_an_exception` | a new free variable that crashes the swap, or is accepted and breaks the closure |
| `test_an_edit_a_code_swap_cannot_install_is_a_finding` | a module-constant edit that "succeeds" while the engine runs what it ran before |
| `test_a_loop_frame_live_at_apply_blocks_through_the_helper` | a loop that never leaves the old code, swapped with no blocking finding |
| `test_a_field_only_the_new_init_sets_blocks_and_routes_to_t1` | a swap that applied with no finding, then killed the engine with AttributeError at the next step (vLLM #63819's shape) |
| `test_a_class_default_or_an_old_setter_is_not_a_new_field` | a false new_fields block when old instances do have the attribute |
| `test_a_constructor_edit_blocks_and_routes_to_t1` | an `__init__` (or `__post_init__`) edit that commits with no finding while the objects built at start-up keep the old state (vLLM #61323) |
| `test_an_edit_to_another_method_is_not_a_constructor_edit` | a constructor detector keyed on the class that would block every layer edit |
| `test_edit_again_after_a_line_shift_follows_the_line_map` | a second edit refused with no_live_function + outside_changes after the first shifted the module's lines |
| `test_without_a_line_map_the_one_live_qualname_is_an_advisory_alias` | the same with no line map: the one live definition is taken, and said |
| `test_leaf_autotuned_kernel_edit_chained_edit_and_undo` | a Triton kernel edit that commits while every launch runs the old compiled kernel; an undo that recompiles or re-tunes |
| `test_helper_edit_resets_caller_in_another_module` | a helper edit whose caller in another module keeps (or reloads from the disk cache) the old binary, or fails on a stale global the new helper no longer reads |
| `test_zero_launch_and_capture_only_are_not_proven` | a kernel edit read as proven while a graph captured before it replays the old kernel |
| `test_signature_or_decorator_change_is_refused` | a rebuild that launches with the parameter binding or tuning configs Triton built for the old text |
| `test_added_jit_helper_is_built_from_the_patch_text_and_removed_on_undo` | `@triton.jit` on an added helper reading the file's lines (or raising OSError) instead of the patch's |
| `test_real_kernel_edit_and_undo` | the same rebuild on real Triton and a GPU (skipped without them) |
| `test_two_ranks_switch_together_and_undo_to_the_same_code_objects` | a rank that runs the edit before commit, or keeps it after undo |
| `test_edit_that_changes_no_code_is_refused` | a transaction that "commits" while nothing changed |
| `test_running_frame_never_runs_the_new_code` | a loop frame that never runs the new code reported as a pass |
| `test_a_new_field_votes_no_and_routes_to_t1` | a participant that votes yes on an edit that would raise AttributeError |
| `test_a_module_the_engine_imports_lazily_after_base_is_not_an_extra_name` | a clean undo reported as not at base because the engine imported a module lazily meanwhile (vLLM #62872) |
| `test_three_edit_chain_explicit` / `_auto` | a second transaction refused after the first shifted lines; an undo that leaves the wrong text as the base |
| `test_step_gap_commit_and_undo_never_mix_versions` | ranks that run one step on different versions (all-or-nothing in one gap) |
| `test_vote_no_changes_nobody` | one rank on another build that votes no while the others switch |
| `test_a_dp_client_that_drops_replies_blocks_the_transaction` | a commit on 4 ranks whose witness, receipt and undo covered 2 (a DP client that returned DP group 0's replies only) |
| `test_commit_failure_undoes_the_committed` | a rank lost at commit leaving the others on the new code |
| `test_half_committed_group_is_undone` | a group whose ranks committed in part |
| `test_step_gap_failed_commit_is_undone_in_the_same_gap` | a step that ran the edit on some ranks before the rollback |
| `test_late_commit_poisons_and_steps_may_mix` | a commit that misses the deadline read as a clean rollback: shows the host poisoned, the next run refused, and the mixed steps until the queued undo |
| `test_receipt_written_only_after_append_succeeds` | a failed receipt append reported as written (and never retried) |
| `test_a_rank_missing_from_proof_is_in_the_receipt` | a rank missing from the post-commit proof dropped from the receipt, which then says the ranks agree |
| `test_code_changed_outside_the_journal_poisons_the_session` | a lab that says "edited" while it runs code no transaction installed |
| `test_hung_commit_ends_the_gap_at_the_deadline` | a rank that hangs in commit and stalls every step |
| `test_install_after_an_unjournaled_edit_is_refused` | an install that adopts a hand edit made since the last status() as the expected code |
| `test_no_fingerprint_never_verifies` | a participant with no fingerprint passing verify because none was expected either |

The conformance cases c0-c10 (`core/livelab/conformance/cases.py`) run against the toy
lab in CI; all 11 pass (no XFAIL).

## Not here yet

The prototype swap (fastrsi#12) had more than a plain live code swap needs; these parts come back in later PRs:

- **New-field backfill** (`_backfill_code`, `fields.py`, plan `"backfill": true`) and
  `rebuilds=` (a recipe that rebuilds the objects resolves the finding). Until then
  `new_fields` blocks and routes to component restart.
- **Per-block witness** (`_inject_hunks`: a key per changed branch, #62206's shape).
- **Compile units** (compile only the statements holding the changed definitions; a whole
  3,800-line module compile stalls a step) and the **analyzer child process**
  (`ProcessAnalyzer`, `analyze_fallback`), which keep the GIL free during prepare.
- **Transaction-level parts**: `overlay=` and `staged_names` (a module importing a name
  another module of the same transaction adds), `docstring_only`, and the participant's
  analyzer.
- **Post-commit rebuilds** (CUDA graph recapture, warm weight reload) and their routes: the receipt's
  `rebuild` and `recapture_s` stay None.

These come in later PRs, each importing and testing on top of this series.

## Audit fixes (2026-11-10)

| Row | Rule now | Test (`test_audit_fixes.py`) |
|---|---|---|
| 8 | A commit is decided under a lock before it leaves the quiesce point: it stands only while the caller still waits and the deadline holds; else the job rolls itself back. An undo waits for an in-flight commit (or says `in_flight`, which poisons the host). | `CommitUndoRaceTest` |
| 9 | Triton callers by identity; a launch counter per kernel; a hook with no new binary proves nothing. | `TritonWitnessIdentityTest` |
| 10 | A nested function whose defaults change is a blocking `nested_signature` (live closures keep their defaults). | `NestedSignatureTest` |
| 18 | Undo cascades by commit order; the host refuses unknown and repeated undo and marks cascaded transactions undone; an undo whose quiesce gave up before it started is cancelled, one that started records its own state. | `UndoBookkeepingTest` |
| 19 | A phase-2 deadline is never re-armed: participants whose turn comes after it are not asked to validate and are aborted without a wait. | `DeadlineTest` |
| 20 | Spans on `time.monotonic` (`Timeline`); `edit_to_run_s` is in the proof: the first live witness hit on every participant minus the edit. | `TimingTest` |
| 21 | Restart witness is the per-rank minimum; a capture probe that raises counts the call as not live. | `WitnessAcrossRanksTest` |
| 23 | Prepare runs no decorator of an added definition (built at apply). | `StageSideEffectTest` |
