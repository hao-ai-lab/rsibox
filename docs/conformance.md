# Conformance

The same cases run on every engine. A plugin that fails is not listed.

| Case | Passes only if |
|---|---|
| `c1_noop` | A code change that keeps behavior leaves output bit-exact |
| `c2_visible` | A visible edit changes output, and the witness fires |
| `c3_fresh_equivalence` | The live edit's output equals a fresh process started with that edit |
| `c4_uninstall` | Undo returns output and code identity to base |
| `c5_failed_install` | A fault in prepare or commit leaves base, proven, across several functions |
| `c6_refusal` | Unsupported edits are refused with the right code, and nothing changes |
| `c7_chain` | Output returns exactly to base after a chain of edits |
| `c8_crash_and_hang` | Crash, hang, a process lost mid-commit and tampered code each poison the session, and restart restores base |
| `c9_soak` | Output returns to base after repeated install and undo. Growth is recorded |
| `c10_not_proven` | An edit that never runs as Python does not read as proven |

## The suite is tested too

Each of these deliberate defects makes at least one case fail. Rerun them after changing the
session or the agent.

| Defect | Caught by |
|---|---|
| Commit swaps nothing | `c1`, `c2` and others |
| Undo restores nothing | `c1`, `c2` and others |
| Agent does not roll back a partial commit | `c5` |
| Witness is never armed | `c2` and others |
| Host skips undo after a partial commit | `c5` |
| Identity check always passes | `c8` |
| Classifier accepts a signature change | `c6` |
| A hang never poisons | `c8` |
| Prepare skips a process | `c2`, `c3` |

## What it does not show

The cases use small edits and a small deterministic workload on a single GPU. Passing shows the
mechanism behaves, not that any edit is safe, and not that a live result equals a fresh one for
your edit. Run the fresh comparison before relying on a result.
