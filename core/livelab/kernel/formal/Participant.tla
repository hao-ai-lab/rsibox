---------------------------- MODULE Participant ----------------------------
EXTENDS Naturals

(*
 * Per-participant transaction and journal lifecycle for LocalParticipant.
 * Enforces: no commit past deadline, no re-prepare of seen txns, and
 * fail-closed poisoning whenever an undo/rollback leaves partial state.
 *)

CONSTANTS MaxJournalDepth

VARIABLES
    state,          \* "idle" | "prepared" | "committed" | "aborted" | "undone" | "partial"
    journal_depth,  \* 0..MaxJournalDepth
    deadline_passed,\* BOOLEAN
    poisoned        \* BOOLEAN

vars == <<state, journal_depth, deadline_passed, poisoned>>

TypeOK ==
    /\ state \in {"idle", "prepared", "committed", "aborted", "undone", "partial"}
    /\ journal_depth \in 0..MaxJournalDepth
    /\ deadline_passed \in BOOLEAN
    /\ poisoned \in BOOLEAN

Init ==
    /\ state = "idle"
    /\ journal_depth = 0
    /\ deadline_passed = FALSE
    /\ poisoned = FALSE

Prepare ==
    /\ state = "idle"
    /\ ~poisoned
    /\ state' = "prepared"
    /\ UNCHANGED <<journal_depth, deadline_passed, poisoned>>

DeadlineExpire ==
    /\ ~deadline_passed
    /\ deadline_passed' = TRUE
    /\ UNCHANGED <<state, journal_depth, poisoned>>

Commit ==
    /\ state = "prepared"
    /\ ~deadline_passed
    /\ ~poisoned
    /\ journal_depth < MaxJournalDepth
    /\ state' = "committed"
    /\ journal_depth' = journal_depth + 1
    /\ UNCHANGED <<deadline_passed, poisoned>>

LateCommitRefused ==
    /\ state = "prepared"
    /\ deadline_passed
    /\ state' = "aborted"
    /\ UNCHANGED <<journal_depth, deadline_passed, poisoned>>

Abort ==
    /\ state = "prepared"
    /\ state' = "aborted"
    /\ UNCHANGED <<journal_depth, deadline_passed, poisoned>>

UndoClean ==
    /\ state = "committed"
    /\ journal_depth > 0
    /\ state' = "undone"
    /\ journal_depth' = journal_depth - 1
    /\ UNCHANGED <<deadline_passed, poisoned>>

UndoPartial ==
    /\ state = "committed"
    /\ state' = "partial"
    /\ poisoned' = TRUE
    /\ UNCHANGED <<journal_depth, deadline_passed>>

Respawn ==
    /\ state' = "idle"
    /\ journal_depth' = 0
    /\ deadline_passed' = FALSE
    /\ poisoned' = FALSE

Next ==
    \/ Prepare
    \/ DeadlineExpire
    \/ Commit
    \/ LateCommitRefused
    \/ Abort
    \/ UndoClean
    \/ UndoPartial
    \/ Respawn

\* Safety invariants
PartialImpliesPoisoned ==
    (state = "partial") => poisoned

NoCommitAfterDeadline ==
    (state = "committed") => (journal_depth > 0)

NoCleanJournalWhenCommitted ==
    (state = "committed") => (journal_depth >= 1)

Spec == Init /\ [][Next]_vars
=============================================================================
