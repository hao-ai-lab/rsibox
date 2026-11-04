# Conformance

A lab's receipts count only after it passes every case. Carried over from `legacy/v0`
(`src/fastrsi/conformance.py`, `docs/conformance.md`); to be ported.

| Case | Passes only if |
|---|---|
| `c0_repeatable` | The workload gives the same output when nothing changes |
| `c1_noop` | A change that keeps behavior leaves output bit-exact |
| `c2_visible` | A visible edit changes output, and the witness fires |
| `c3_fresh_equivalence` | The live edit's output equals a fresh process started with that edit |
| `c4_uninstall` | Undo returns output and code identity to base |
| `c5_failed_install` | A fault mid-install leaves base, proven, across several functions |
| `c6_refusal` | Unsupported edits are refused with the right code, and nothing changes |
| `c7_chain` | Output returns exactly to base after a chain of edits |
| `c8_crash_and_hang` | Crash, hang, lost process and tampered code each poison the session |
| `c9_soak` | Output returns to base after repeated install and undo |
| `c10_not_proven` | An edit that never runs as Python does not read as proven |

To add: an install that changes nothing must fail, not report success.
