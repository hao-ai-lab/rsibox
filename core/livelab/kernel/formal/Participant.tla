---------------------------- MODULE Participant ----------------------------
EXTENDS Naturals

(*
 * Per-participant transaction and journal lifecycle for LocalParticipant.
 * Enforces: no commit past deadline, multi-txn journal depth up to MaxJournalDepth,
 * and fail-closed poisoning whenever an undo/rollback leaves partial state.
 *)

CONSTANTS MaxJournalDepth

VARIABLES
    state,          \* "idle" | "prepared" | "committed" | "aborted" | "undone" | "partial"
    journal_depth,  \* 0..MaxJournalDepth
    deadline_passed,\* BOOLEAN
    committed_late, \* BOOLEAN: TRUE if a commit ever executed after deadline_passed
    poisoned        \* BOOLEAN

vars == <<state, journal_depth, deadline_passed, committed_late, poisoned>>

TypeOK ==
    /\ state \in {"idle", "prepared", "committed", "aborted", "undone", "partial"}
    /\ journal_depth \in 0..MaxJournalDepth
    /\ deadline_passed \in BOOLEAN
    /\ committed_late \in BOOLEAN
    /\ poisoned \in BOOLEAN

Init ==
    /\ state = "idle"
    /\ journal_depth = 0
    /\ deadline_passed = FALSE
    /\ committed_late = FALSE
    /\ poisoned = FALSE

Prepare ==
    /\ state \in {"idle", "committed", "aborted", "undone"}
    /\ ~poisoned
    /\ journal_depth < MaxJournalDepth
    /\ state' = "prepared"
    /\ deadline_passed' = FALSE
    /\ UNCHANGED <<journal_depth, committed_late, poisoned>>

DeadlineExpire ==
    /\ state = "prepared"
    /\ ~deadline_passed
    /\ deadline_passed' = TRUE
    /\ UNCHANGED <<state, journal_depth, committed_late, poisoned>>

Commit ==
    /\ state = "prepared"
    /\ ~deadline_passed
    /\ ~poisoned
    /\ journal_depth < MaxJournalDepth
    /\ state' = "committed"
    /\ journal_depth' = journal_depth + 1
    /\ committed_late' = (committed_late \/ deadline_passed)
    /\ UNCHANGED <<deadline_passed, poisoned>>

LateCommitRefused ==
    /\ state = "prepared"
    /\ deadline_passed
    /\ state' = "aborted"
    /\ UNCHANGED <<journal_depth, deadline_passed, committed_late, poisoned>>

Abort ==
    /\ state = "prepared"
    /\ state' = "aborted"
    /\ UNCHANGED <<journal_depth, deadline_passed, committed_late, poisoned>>

UndoClean ==
    /\ state \in {"committed", "undone"}
    /\ journal_depth > 0
    /\ state' = IF journal_depth = 1 THEN "idle" ELSE "undone"
    /\ journal_depth' = journal_depth - 1
    /\ UNCHANGED <<deadline_passed, committed_late, poisoned>>

UndoPartial ==
    /\ state \in {"committed", "undone"}
    /\ journal_depth > 0
    /\ state' = "partial"
    /\ poisoned' = TRUE
    /\ UNCHANGED <<journal_depth, deadline_passed, committed_late>>

Respawn ==
    /\ state' = "idle"
    /\ journal_depth' = 0
    /\ deadline_passed' = FALSE
    /\ committed_late' = FALSE
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
    ~committed_late

CommittedHasJournal ==
    (state = "committed") => (journal_depth >= 1)

Spec == Init /\ [][Next]_vars
=============================================================================
