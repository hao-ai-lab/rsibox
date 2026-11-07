# Transaction kernel

The live-edit transaction kernel, built bottom up: find the code a running engine executes,
record every change to it so undo is exact, then swap, witness and coordinate. Pure Python,
standard library only; it imports no engine. This folder holds the bottom layer so far:
findings and locate.

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
- **Findings** are the one result schema. Every call returns `{"ok": bool, ...,
  "findings": [...]}`. A finding says what was checked, what was observed against what was
  expected, why it matters, the actions and their cost, and the risk of ignoring it.
  `decide(findings, accept)` is the one accept mechanism (a check name, or a record of who
  accepted it and why; a standing record expires when the checked code changes).

| File | What it owns |
|---|---|
| `findings.py` | The findings schema, the catalog of checks, `decide`. No imports. |
| `locate.py` | Code identity and `content_hash`; the heap and namespace scans; `locate`, `holders`. Imports only `findings`. |
| `toy/` | A toy engine for the tests: ops with a decorator and a closure factory, an `Engine` that saves a bound-method callback, a busy loop, and `load_copy`, which gives each test its own module copy. |

## Invariants

- **Nothing raises to the caller.** `locate` of a module that does not import is a
  `patch_invalid` finding; a qualname with no live function is `no_live_function`.
- **Checks advise.** A check returns a finding; the caller's accept list decides. Physical
  limits (`no_live_function`, ...) are not overridable.
- **The heap scan unfreezes the gc.** Engines call `gc.freeze()` after startup, and
  `gc.get_objects()` and `gc.get_referrers()` both skip frozen objects. The scan yields the
  GIL in slices (`SLICE_S`); the one `get_objects` call cannot be sliced.

## Golden tests

From the repo root (CPU, under 1 s; CI runs the same command in `.github/workflows/cpu-tests.yml`):

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

## Not here yet

- **Journal** (record every change with its inverse; exact undo).
- **Swap** (stage the patched module, pair old and new code, `apply` through the journal),
  **witness** (counters of the new code, running frames), **jit** (Triton `@jit` targets),
  **fields** (fields only the new `__init__` sets).
- **Coordinator** (two-phase transactions over participants, step-gap commits, deadlines,
  proofs) and **session** (the host's state machine and poisoning), and the receipt writer.

These come in later PRs of the kernel series, each importing and testing on top of this one.
