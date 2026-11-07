# Transaction kernel

The live-edit transaction kernel, built bottom up: find the code a running engine executes,
record every change to it so undo is exact, then swap, witness and coordinate. Pure Python,
standard library only; it imports no engine. This folder holds the bottom layer so far:
findings, locate, journal, witness, and the R0 swap (a function-body swap).

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
  closures made later run the new code. Nothing is visible yet. Findings: `no_live_function`,
  `not_in_patch`, `base_mismatch`, `outside_changes`, `removed_definition`, `closure_changed`
  (free variables differ; not overridable), `defaults_not_evaluated`, `running_frame`
  (advisory at stage: a step in flight), `patch_invalid`, `new_fields`, `line_aliased`. `validate(staged)` re-checks at
  the apply point: the same code objects are installed, and no thread stack holds a frame of
  the old code. `apply(staged, journal, witness)` makes it visible, every change through the
  journal (closure pairs first, so undo also takes back closures the new code made).
  `stage_and_apply` is stage, decide, validate, apply in one call for a host with no two
  phases.
- **Detectors** in `stage`. *New fields:* a changed or added method that reads `self.X`,
  where only the edited `__init__` sets X (no old method sets it, no class body defines it),
  gives a blocking `new_fields` finding with `"route": "T1"`: instances built before the swap
  lack X, so the new code would raise AttributeError at its first call. The fix is a rebuild
  of the objects or a restart (T1); nothing here sets fields on live objects. *Line maps:* a
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
  routed to R1 that asks for a device-side probe. With torch loaded, the hook is marked
  `assume_constant_result`, so Dynamo calls it while tracing and never traces it.
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
| `swap.py` | The change analysis (`analyze`, `changed_targets`: text only) and the live half (`stage`, `validate`, `apply`, `stage_and_apply`, `closure_pairs`, the scratch namespace). Imports `findings`, `journal`, `locate`, `witness`. |
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
python3 -m unittest discover -s repos/vllm/livelab/kernel -t repos/vllm/livelab -v
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
| `test_edit_again_after_a_line_shift_follows_the_line_map` | a second edit refused with no_live_function + outside_changes after the first shifted the module's lines |
| `test_without_a_line_map_the_one_live_qualname_is_an_advisory_alias` | the same with no line map: the one live definition is taken, and said |

## Not here yet

The prototype swap (fastrsi#12) had more than R0 needs; these parts come back in later PRs:

- **New-field backfill** (`_backfill_code`, `fields.py`, plan `"backfill": true`) and
  `rebuilds=` (a recipe that rebuilds the objects resolves the finding). Until then
  `new_fields` blocks and routes to T1.
- **Triton `@jit` targets** (`jit_rebuild`, rebuild in place, launch counters) with `jit.py`.
  Until then such a target is swapped as bytecode the kernel compiler never runs, and its
  witness reads zero (`zero_witness`).
- **Per-block witness** (`_inject_hunks`: a key per changed branch, #62206's shape).
- **Compile units** (compile only the statements holding the changed definitions; a whole
  3,800-line module compile stalls a step) and the **analyzer child process**
  (`ProcessAnalyzer`, `analyze_fallback`), which keep the GIL free during prepare.
- **Transaction-level parts**, with the coordinator: `overlay=` and `staged_names` (a module
  importing a name another module of the same transaction adds), `docstring_only`, the
  `empty_change` verdict, `Witness.arm` (counters for repair cells), and the
  `record_closures` cases for partial undo.
- **Coordinator** (two-phase transactions over participants, step-gap commits, deadlines,
  proofs) and **session** (the host's state machine and poisoning), and the receipt writer.

These come in later PRs of the kernel series, each importing and testing on top of this one.
