---------------------------- MODULE Transaction ----------------------------
EXTENDS Naturals, FiniteSets

(*
 * Bounded 2PC transaction protocol for RSIBox core/livelab/kernel/coordinator.py.
 * Models prepare -> validate -> commit / abort / rollback and fail-closed
 * recovery_required freeze until clean Respawn.
 *)

CONSTANTS Participants

VARIABLES
    coord_state,       \* "idle" | "preparing" | "validating" | "committing" | "committed" | "aborted" | "rolling_back" | "rolled_back" | "recovery_required"
    p_state,           \* [Participants -> {"idle", "prepared", "committed", "aborted", "undone", "partial", "dead"}]
    recovery_required, \* BOOLEAN: latched on rollback/undo failure or unconfirmed abort
    poisoned           \* BOOLEAN

vars == <<coord_state, p_state, recovery_required, poisoned>>

TerminalStates == {"idle", "committed", "aborted", "rolled_back", "recovery_required"}

TypeOK ==
    /\ coord_state \in {"idle", "preparing", "validating", "committing",
                        "committed", "aborted", "rolling_back", "rolled_back",
                        "recovery_required"}
    /\ p_state \in [Participants -> {"idle", "prepared", "committed",
                                     "aborted", "undone", "partial", "dead"}]
    /\ recovery_required \in BOOLEAN
    /\ poisoned \in BOOLEAN

Init ==
    /\ coord_state = "idle"
    /\ p_state = [p \in Participants |-> "idle"]
    /\ recovery_required = FALSE
    /\ poisoned = FALSE

StartPrepare ==
    /\ coord_state \in TerminalStates
    /\ ~recovery_required
    /\ ~poisoned
    /\ coord_state' = "preparing"
    /\ p_state' = [p \in Participants |-> "idle"]
    /\ UNCHANGED <<recovery_required, poisoned>>

ParticipantPrepare(p) ==
    /\ coord_state = "preparing"
    /\ p_state[p] = "idle"
    /\ p_state' = [p_state EXCEPT ![p] = "prepared"]
    /\ UNCHANGED <<coord_state, recovery_required, poisoned>>

PrepareAllYes ==
    /\ coord_state = "preparing"
    /\ \A p \in Participants : p_state[p] = "prepared"
    /\ coord_state' = "validating"
    /\ UNCHANGED <<p_state, recovery_required, poisoned>>

PrepareAbortClean ==
    /\ coord_state = "preparing"
    /\ coord_state' = "aborted"
    /\ p_state' = [p \in Participants |-> "aborted"]
    /\ UNCHANGED <<recovery_required, poisoned>>

AbortUnconfirmed(p) ==
    /\ coord_state \in {"preparing", "validating", "rolling_back"}
    /\ p_state' = [p_state EXCEPT ![p] = "dead"]
    /\ recovery_required' = TRUE
    /\ poisoned' = TRUE
    /\ coord_state' = "recovery_required"

ValidatePass ==
    /\ coord_state = "validating"
    /\ \A p \in Participants : p_state[p] = "prepared"
    /\ coord_state' = "committing"
    /\ UNCHANGED <<p_state, recovery_required, poisoned>>

ParticipantCommit(p) ==
    /\ coord_state = "committing"
    /\ p_state[p] = "prepared"
    /\ p_state' = [p_state EXCEPT ![p] = "committed"]
    /\ UNCHANGED <<coord_state, recovery_required, poisoned>>

FinalizeCommit ==
    /\ coord_state = "committing"
    /\ \A p \in Participants : p_state[p] = "committed"
    /\ coord_state' = "committed"
    /\ UNCHANGED <<p_state, recovery_required, poisoned>>

CommitFailStartRollback(p) ==
    /\ coord_state = "committing"
    /\ p_state[p] = "prepared"
    /\ coord_state' = "rolling_back"
    /\ p_state' = [p_state EXCEPT ![p] = "aborted"]
    /\ UNCHANGED <<recovery_required, poisoned>>

RollbackParticipantClean(p) ==
    /\ coord_state = "rolling_back"
    /\ p_state[p] = "committed"
    /\ p_state' = [p_state EXCEPT ![p] = "undone"]
    /\ UNCHANGED <<coord_state, recovery_required, poisoned>>

RollbackParticipantFail(p) ==
    /\ coord_state = "rolling_back"
    /\ p_state[p] \in {"committed", "prepared"}
    /\ p_state' = [p_state EXCEPT ![p] = "partial"]
    /\ recovery_required' = TRUE
    /\ poisoned' = TRUE
    /\ coord_state' = "recovery_required"

FinalizeRollbackClean ==
    /\ coord_state = "rolling_back"
    /\ \A p \in Participants : p_state[p] \in {"undone", "aborted", "idle"}
    /\ coord_state' = "rolled_back"
    /\ UNCHANGED <<recovery_required, poisoned>>

Respawn ==
    /\ coord_state' = "idle"
    /\ p_state' = [p \in Participants |-> "idle"]
    /\ recovery_required' = FALSE
    /\ poisoned' = FALSE

Next ==
    \/ StartPrepare
    \/ (\E p \in Participants : ParticipantPrepare(p))
    \/ PrepareAllYes
    \/ PrepareAbortClean
    \/ (\E p \in Participants : AbortUnconfirmed(p))
    \/ ValidatePass
    \/ (\E p \in Participants : ParticipantCommit(p))
    \/ FinalizeCommit
    \/ (\E p \in Participants : CommitFailStartRollback(p))
    \/ (\E p \in Participants : RollbackParticipantClean(p))
    \/ (\E p \in Participants : RollbackParticipantFail(p))
    \/ FinalizeRollbackClean
    \/ Respawn

\* Safety invariants
NoSplitCommit ==
    (coord_state = "committed") => (\A p \in Participants : p_state[p] = "committed")

RollbackFreeze ==
    recovery_required => (coord_state = "recovery_required" /\ poisoned)

Spec == Init /\ [][Next]_vars
=============================================================================
