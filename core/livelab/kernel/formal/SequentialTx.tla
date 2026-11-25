---------------------------- MODULE SequentialTx ----------------------------
EXTENDS Naturals, Sequences, FiniteSets

(*
 * Bounded sequential transaction stack for Session (install / uninstall / respawn).
 * Models LIFO stack ordering, duplicate txn ID refusal, rollback-failure freeze,
 * and exact base-state restoration when the stack drains cleanly.
 *)

CONSTANTS TxnIds, Bug

VARIABLES
    stack,             \* Sequence of committed txn IDs (oldest first)
    seen,              \* Subset of TxnIds ever started in this incarnation
    at_base,           \* BOOLEAN: all participants match base fingerprint
    recovery_required  \* BOOLEAN: latched on failed uninstall/rollback

vars == <<stack, seen, at_base, recovery_required>>

TypeOK ==
    /\ seen \subseteq TxnIds
    /\ at_base \in BOOLEAN
    /\ recovery_required \in BOOLEAN

Init ==
    /\ stack = <<>>
    /\ seen = {}
    /\ at_base = TRUE
    /\ recovery_required = FALSE

InstallCommit(t) ==
    /\ ~recovery_required
    /\ t \in TxnIds
    /\ Len(stack) < Cardinality(TxnIds)
    /\ (Bug = "AllowDuplicateTxn" \/ t \notin seen)
    /\ stack' = Append(stack, t)
    /\ seen' = seen \cup {t}
    /\ at_base' = FALSE
    /\ UNCHANGED recovery_required

InstallAbort(t) ==
    /\ ~recovery_required
    /\ t \in TxnIds
    /\ t \notin seen
    /\ seen' = seen \cup {t}
    /\ UNCHANGED <<stack, at_base, recovery_required>>

UninstallClean ==
    /\ ~recovery_required
    /\ Len(stack) > 0
    /\ stack' = SubSeq(stack, 1, Len(stack) - 1)
    /\ at_base' = (Len(stack) = 1)
    /\ UNCHANGED <<seen, recovery_required>>

UninstallFail ==
    /\ ~recovery_required
    /\ Len(stack) > 0
    /\ recovery_required' = TRUE
    /\ UNCHANGED <<stack, seen, at_base>>

Respawn ==
    /\ stack' = <<>>
    /\ seen' = {}
    /\ at_base' = TRUE
    /\ recovery_required' = FALSE

Next ==
    \/ (\E t \in TxnIds : InstallCommit(t))
    \/ (\E t \in TxnIds : InstallAbort(t))
    \/ UninstallClean
    \/ UninstallFail
    \/ Respawn

\* Safety invariants
EmptyStackRestoresBase ==
    (Len(stack) = 0 /\ ~recovery_required) => at_base

NoDuplicateInStack ==
    \A i, j \in 1..Len(stack) : (i # j) => (stack[i] # stack[j])

\* Reachability witness invariants (expected to fail on non-mutant model)
NeverTwoStacked == Len(stack) < 2
NeverRecovery == ~recovery_required

Spec == Init /\ [][Next]_vars
=============================================================================
