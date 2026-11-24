# Bounded Formal Protocol Specifications (`core/livelab/kernel/formal/`)

This directory holds four self-contained TLA+ specifications describing the
bounded transaction-protocol state machine implemented by
`core/livelab/kernel/{coordinator.py, session.py}`:

| Specification | Scope | Primary Safety Invariants |
|---|---|---|
| `Transaction.tla` | Coordinator + participant 2PC (`prepare -> validate -> commit / abort / rollback -> recovery_required -> respawn`) | `NoSplitCommit` (`no_split_commit`), `RollbackFreeze` (`rollback_freeze`) |
| `Participant.tla` | Per-participant staging, deadline cut-off, journal mark depth, and partial-undo poisoning | `PartialImpliesPoisoned`, `NoCommitAfterDeadline` |
| `SequentialTx.tla` | Session transaction stack, LIFO uninstall, duplicate-ID refusal, and base restoration | `EmptyStackRestoresBase`, `NoDuplicateInStack` |
| `Incarnation.tla` | Coordinator incarnation epochs across respawns and stale/delayed reply filtering | `SingleActiveIncarnation` (`single_active_incarnation`), `StaleReplyRejection` (`stale_reply_rejection`) |

## Native CPU Verification (Zero Java / TLC Dependency)

`core/livelab/kernel/test_formal_protocol.py` runs natively in CPU CI using only
the Python standard library (`unittest`):

1. **Bounded BFS State-Space Verifier**: Exhaustively explores the finite state
   spaces corresponding to all four specifications, verifying `rollback_freeze`,
   `no_split_commit`, `single_active_incarnation`, and `stale_reply_rejection`
   at every reachable state, plus four intentional bug-mutant assertions
   confirming that breaking any single guard produces a counterexample trace.
2. **Kernel Trace-Replay & Adversarial Fault Injection**: Exercises the real
   `Coordinator`, `StepGapCoordinator`, `LocalParticipant`, `CallParticipant`,
   `Session`, and `ToyLab` under commit drops, rollback/undo failures,
   unconfirmed aborts, partial-rank timeouts, duplicate/stale txn and
   incarnation replies, non-mutating `plan()` checks, and SHA-256 `identity()`
   determinism.

Run from the repository root:

```bash
python3 -m unittest core/livelab/kernel/test_formal_protocol.py -v
```

## Honest Scope & Limitations

These specifications verify **bounded protocol state transitions and safety
invariants** for small finite instantiations (e.g., 2-3 participants, bounded
transactions and incarnations). They do **not** prove:

- numerical equivalence or kernel correctness of arbitrary Python, Triton, or
  CUDA code installed by a patch;
- OS thread scheduling, CPython GIL/GC internals, or GPU hardware memory models;
- unbounded liveness or network-partition recovery beyond fail-closed freeze
  and respawn.
