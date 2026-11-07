# Transaction kernel

The live-edit transaction kernel, built bottom up: find the code a running engine executes,
record every change to it so undo is exact, then swap, witness and coordinate. Pure Python,
standard library only; it imports no engine. This folder holds the bottom layer so far:
findings, locate, journal and witness.

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

## Not here yet

- **Swap** (stage the patched module, pair old and new code, `apply` through the journal),
  with its line maps and analyzer child. The journal's `record_closures` entry is driven by
  `apply`, so taking back closures the new code made at run time is tested there. The swap
  also adds the per-block witness of changed branches, and drives `Witness.arm` (counters for
  repair cells). **jit** (Triton `@jit` targets),
  **fields** (fields only the new `__init__` sets).
- **Coordinator** (two-phase transactions over participants, step-gap commits, deadlines,
  proofs) and **session** (the host's state machine and poisoning), and the receipt writer.

These come in later PRs of the kernel series, each importing and testing on top of this one.
