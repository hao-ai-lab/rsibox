# Transaction kernel

The small, engine-agnostic core that every live-change mechanism shares: locate the live code,
swap it, journal every change so undo is exact, report findings instead of raising, count
calls of the new code, run two-phase transactions over participants, and write one receipt
per transaction. Pure Python, standard library only; it imports no engine.

| File | What it does |
|---|---|
| `locate.py` | (module, qualname, first line) -> running code objects, the function objects that hold them (gc heap scan), and every holder (module and class dicts, closure cells, `__wrapped__`, bound methods, attributes). One code identity: file, `co_qualname`, `co_firstlineno` + `content_hash`. |
| `swap.py` | `stage` compiles the patched module, adds the witness, pairs old and new code for each target and its enclosing functions, evaluates changed defaults, runs imports in a scratch namespace, runs the checks. Nothing is visible. `apply` makes it visible through the journal. |
| `journal.py` | One ordered undo log per participant. Entries `code`, `attr`, `added`, `object`, each with its inverse. `mark()` / `undo_to(mark)`. `record_attr` and `repair(...)` journal hand-written repair cells (recipe R10). |
| `findings.py` | One findings schema, the catalog of checks, and `decide(findings, accept)`: the one accept mechanism. |
| `witness.py` | Counters keyed `module:qualname:firstline`; `check` reports `zero_witness` or `running_frame`. |
| `receipt.py` | The one receipt writer: append-only JSON lines under a lock. Schema in the docstring and in `../INTERFACE.md` (proposed for v1). |
| `coordinator.py` | Two-phase transactions: prepare (stage and vote), commit at each participant's quiesce point, abort, undo of committed participants when a commit fails. `LocalParticipant` runs the kernel in-process; `CallParticipant` is the shape an engine implements over its RPC. |
| `toy/` | A toy engine: ops with a decorator and a closure factory, an `Engine` built in `__init__` with a saved bound-method callback, and a busy loop with a step-boundary queue. `load_copy` gives each participant its own module copy. |

## Walkthrough

Three participants, each with its own copy of the toy engine; one transaction swaps
`Engine.step` on all of them, then the coordinator undoes it. Run from `template/livelab`:

```python
import os, sys, tempfile, time
from kernel import toy, coordinator as C, receipt as R, locate as L

ENG = toy.source("engine")
parts, engines = [], []
for name in ("rank0", "rank1", "core"):
    toy.load_copy("w_" + name)                       # each participant: its own module copy
    e = sys.modules[f"w_{name}.engine"].Engine()
    engines.append(e)
    parts.append(C.LocalParticipant(name, alias={"toy": "w_" + name}))
line = next(i for i, s in enumerate(ENG.splitlines(), 1) if "def step" in s)
plan = {"modules": {"toy.engine": {
    "source": ENG.replace("return y", "return y * 100", 1),
    "base_hash": L.text_hash(ENG), "targets": [["Engine.step", line]]}}}
co = C.Coordinator(parts, R.ReceiptWriter(os.path.join(tempfile.mkdtemp(), "r.jsonl")))

print("base:     ", [e.step(3) for e in engines])
res = co.run(plan, edit_time=time.time())
print("outcome:  ", res["outcome"], [e.step(3) for e in engines])
rec = res["receipt"]
print("receipt:  ", {k: rec[k] for k in ("edit_to_run_s", "actual_recipes", "undo")})
print("proofs:   ", rec["proofs"]["participants"]["core"], rec["proofs"]["hashes_agree"])
print("undo:     ", co.undo(res["txn"])["ok"], [e.step(3) for e in engines])
```

Real output:

```
base:      [17, 17, 17]
outcome:   committed [1700, 1700, 1700]
receipt:   {'edit_to_run_s': 0.013389, 'actual_recipes': ['R0 swap'], 'undo': {'status': 'not_needed', 'participants': {}}}
proofs:    {'ok': True, 'findings': [], 'hashes': {'toy.engine:Engine.step:21': ['7d6d770bcd9637da']}, 'witness': {'toy.engine:Engine.step:21': 0}} True
undo:      True [17, 17, 17]
```

The witness count in the receipt is 0 because the proof is taken right after commit, before
any step. `LocalParticipant.witness_check(txn)` after the workload is what proves the new code
ran; it returns `zero_witness` or `running_frame` findings, never a silent pass.

## Golden tests

`python3 -m unittest -v` from `template/livelab` (CPU, about 0.2 s):

```
test_accept_list ... ok
test_closure_change_is_a_finding_not_an_exception ... ok
test_code_identity_and_hash ... ok
test_commit_failure_undoes_the_committed ... ok
test_receipt_fields_and_measured_times ... ok
test_repair_cell_sets_missing_field_and_undo_removes_it ... ok
test_repair_that_cannot_be_undone_is_a_finding ... ok
test_running_frame_never_runs_the_new_code ... ok
test_swap_reaches_every_holder ... ok
test_undo_restores_exact_base ... ok
test_vote_no_changes_nobody ... ok
----------------------------------------------------------------------
Ran 11 tests in 0.191s

OK
```

Each test's docstring says what it proves and what it does not.

## Rules the kernel keeps

- **Nothing raises to the caller.** Every call returns `{"ok": bool, ..., "findings": [...]}`.
  A participant that raises or replies with something else counts as not-ok.
- **Checks advise.** A check returns a finding; the caller's accept list decides. Physical
  limits (`closure_changed`, `no_live_function`, ...) are not overridable.
- **Prepare changes nothing visible.** New code, defaults, imports and added definitions are
  built beside the old state. Only `apply` (inside `commit`, at the quiesce point) switches.
- **Undo is exact.** The journal restores the same code and attribute objects and removes
  every name it added, including the witness hook and its counters.
- **The heap scan unfreezes the gc.** Engines call `gc.freeze()` after startup, and
  `gc.get_objects()` and `gc.get_referrers()` both skip frozen objects.

## Not covered yet

- A second edit on top of a committed one, when the edit moved the target's first line: the
  running code then has the patch's line, and the target is not found (`no_live_function`).
- Functions defined with `exec` into foreign globals are not found (the scan matches by
  `__globals__`, so module copies stay apart).
- Real RPC: `CallParticipant` is the interface; the vLLM side (collective_rpc, the engine-core
  channel) is the next step.
