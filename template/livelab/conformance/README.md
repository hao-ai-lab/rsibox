# Conformance

A lab's receipts count only after it passes every case.

Each case drives a lab only through `LabUnderTest` (`lab.py`): the five interface verbs
(`install`, `check`, `reset`, `uninstall`, `status`) plus a harness the lab provides for
conformance only (`patches`: named edits; `fresh_check`: the workload in a new process
started from the edited files; `inject`: faults). So the same cases run against the toy lab
on CPU now and against an engine adapter later.

| Case | Passes only if |
|---|---|
| `c0_repeatable` | The workload gives the same output when nothing changes (3 checks, one after `reset`) |
| `c1_noop` | A change that keeps behavior installs, runs (witness), and leaves output bit-exact |
| `c2_visible` | A visible edit changes output, and the witness fires on every participant |
| `c3_fresh_equivalence` | The live edit's output (one edit, and a chain of 3) equals a fresh process started with that edit |
| `c4_uninstall` | Undo returns output and code identity (hash and object of every function) to base |
| `c5_failed_install` | A fault mid-install (one participant after 2 of 3 swaps; another's commit) leaves base, proven, across 3 functions in 2 modules |
| `c6_refusal` | Unsupported edits are refused with the right code, and nothing changes: syntax `patch_invalid`, closure `closure_changed`, module constant `outside_changes`, no code change `empty_change` |
| `c7_chain` | 3 stacked edits (two on the same function) each run; undoing them one by one returns each earlier output exactly, ending at base |
| `c8_crash_and_hang` | Crash, hang, lost participant and tampered code each poison the session; a poisoned lab refuses install; respawn returns to base |
| `c9_soak` | Output returns to base after 20 cycles of install and undo |
| `c10_not_proven` | An edit that never runs as Python (startup-only code) installs but its check reads `proven: False` with `zero_witness`, while a fresh process shows the edit matters |

An install that changes nothing must fail, not report success: `c6_refusal` (`empty`).

## Run

From `template/livelab`:

```sh
python3 -m conformance --lab module:Class        # the pass table
python3 -m conformance --lab module:Class --only c7_chain --receipts receipts.jsonl --json
python3 -m unittest discover -s conformance -t . -v
```

`--lab` takes `module:Class` (a `LabUnderTest`). `--receipts` keeps the `txn` / `proof` / `undo`
receipts the cases' transactions write (kernel/receipt.py, a proposal that needs a human
owner). The run's own verdict is printed, not written: the receipt proposal has no record
kind for a conformance run, and adding one extends the receipt writer, which needs its human
owner first (AGENTS.md, Trust).

Real output on the toy lab (CPU, about 1.5 s; c8 waits out a 0.5 s check timeout):

```
case                  result  seconds  evidence
c0_repeatable         pass      0.003  3 checks (one after reset ['tokens', 'cache']) gave [11, 13, 15, 17, 21, 27] x2 ranks
c1_noop               pass      0.008  installed by ['R0 swap']; calls {'toy.ops:scale:16': [8, 8]}; output unchanged
c2_visible            pass      0.006  output [11, 13, 15, 17, 21, 27] x2 ranks -> [12, 14, 16, 18, 22, 28] x2 ranks
                                       calls per participant {'toy.ops:activate:20': [8, 8]}
c3_fresh_equivalence  pass      0.140  visible: live == fresh == [12, 14, 16, 18, 22, 28] x2 ranks
                                       chain of 3: live == fresh == [130, 150, 170, 190, 230, 290] x2 ranks
c4_uninstall          pass      0.007  uninstall ['txn-b19e2bf9'] in 0.000799 s; identity 88618bf0b31ab13d restored
c5_failed_install     pass      0.064  fail_apply: rolled_back, codes ['participant_failed'], base proven
                                       fail_commit: rolled_back, codes ['participant_failed'], base proven
                                       control: no fault, committed 3 targets, output changed
c6_refusal            pass      0.018  syntax: ['patch_invalid']
                                       closure: ['closure_changed']
                                       module_state: ['empty_change', 'outside_changes']
                                       empty: ['empty_change']
c7_chain              pass      0.031  [11, 13, 15, ...] -> [12, 14, 16, ...] -> [13, 15, 17, ...] -> [130, 150, 170, ...] -> undone one by one to base
c8_crash_and_hang     pass      1.031  crash: poisoned by ['participant_failed'], respawn 0.004713 s
                                       hang: poisoned by ['participant_failed'], respawn 0.50553 s
                                       lost: poisoned by ['participant_failed'], respawn 0.005368 s
                                       tamper: poisoned by ['tampered'], respawn 0.005252 s
c9_soak               pass      0.102  20 cycles; install median 0.0035 s, uninstall median 0.0008 s
c10_not_proven        pass      0.057  proven False, findings ['zero_witness', 'zero_witness']
                                       live [11, 13, 15, 17, 21, 27] x2 ranks (base: True); fresh [12, 14, 16, 18, 22, 28] x2 ranks

11/11 as expected
```

## The toy lab

The toy lab is not in `template/`: it drives the vLLM reference kernel, so it lives with it in
`repos/vllm/livelab/kernel/toy_lab.py`. This folder tests its own cases against small fake
labs (`test_cases.py`). The output above is from the toy lab.

The toy lab: two worker ranks, each its own copy of the kernel's toy engine, step together in
a `Cluster` loop thread; a `StepGapCoordinator` runs phase 2 of every transaction in one gap
between steps; a kernel `Session` keeps base / edited / poisoned. Ranks are reached through
`CallParticipant`, as an engine reaches workers over RPC. The planner is
`swap.changed_targets` (definitions whose AST changed against the live text). The workload
runs in the loop, between steps. `fresh_check` runs it in a new Python process from a copy of
the toy files with the patch written in.

What the toy cannot show: GPU state, CUDA graphs (c10 uses startup-only code instead), real
RPC latency, a process that really dies (c8's crash, hang and lost participant are injected
at the RPC and step boundary; respawn rebuilds the module copies in the same process).

## Kernel pieces the cases rely on

See `repos/vllm/livelab/kernel/README.md`:

- `empty_change` (c6): an edit that changes no code is refused, not committed as a success.
- Chained edits (c7, c3): participants stage against the text they installed, so a second
  edit of the same function is not a `base_mismatch`.
- `fingerprint` and `session.py` (c4, c5, c8): code identity per participant, and the
  base / edited / poisoned state machine `status()` reports.

## Adding a case

A case is a function in `cases.py` that takes a lab, calls only `LabUnderTest` methods, and
raises `CaseFailed` with what it saw. Its docstring says what passing proves and what it does
not. Show that it fails when the property it checks is broken. A case a lab cannot pass yet
goes in `XFAIL` with the reason; it is reported `xfail`, never `pass`.
