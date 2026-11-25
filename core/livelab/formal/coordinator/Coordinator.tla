---- MODULE Coordinator ----
EXTENDS Naturals

CONSTANTS None, Committed, Undone, InDoubt, Partial, Superseded

VARIABLES t1, inc, undo_inflight, undo_inc, poisoned, rec_req, superseded_seen

vars == << t1, inc, undo_inflight, undo_inc, poisoned, rec_req, superseded_seen >>

Init ==
    /\ t1 = None
    /\ inc = 1
    /\ undo_inflight = FALSE
    /\ undo_inc = 0
    /\ poisoned = FALSE
    /\ rec_req = FALSE
    /\ superseded_seen = FALSE

CommitT1 ==
    /\ ~poisoned /\ ~rec_req /\ ~undo_inflight
    /\ t1 = None
    /\ t1' = Committed
    /\ UNCHANGED << inc, undo_inflight, undo_inc, poisoned, rec_req, superseded_seen >>

UndoStart ==
    /\ ~rec_req /\ ~undo_inflight
    \* C2: Only committed (or aborted-with-pending-rollback) enters undo; never Prepared/InDoubt.
    /\ t1 = Committed
    /\ undo_inflight' = TRUE
    /\ undo_inc' = inc
    /\ UNCHANGED << t1, inc, poisoned, rec_req, superseded_seen >>

UndoFinishOk ==
    /\ undo_inflight
    /\ undo_inflight' = FALSE
    /\ undo_inc' = 0
    \* C3/C9: Drop late updates when incarnation advanced via recovered().
    /\ IF undo_inc = inc
       THEN t1' = Undone
       ELSE UNCHANGED t1
    /\ UNCHANGED << inc, poisoned, rec_req, superseded_seen >>

UndoFailPartial ==
    /\ undo_inflight
    /\ undo_inflight' = FALSE
    /\ undo_inc' = 0
    /\ IF undo_inc = inc
       THEN /\ t1' = Partial
            /\ poisoned' = TRUE
            /\ rec_req' = TRUE
       ELSE UNCHANGED << t1, poisoned, rec_req >>
    /\ UNCHANGED << inc, superseded_seen >>

UndoFailInDoubt ==
    /\ undo_inflight
    /\ undo_inflight' = FALSE
    /\ undo_inc' = 0
    \* C5: In-doubt undo latches recovery_required just like partial undo.
    /\ IF undo_inc = inc
       THEN /\ t1' = InDoubt
            /\ poisoned' = TRUE
            /\ rec_req' = TRUE
       ELSE UNCHANGED << t1, poisoned, rec_req >>
    /\ UNCHANGED << inc, superseded_seen >>

RecoverRespawn ==
    /\ (poisoned \/ rec_req)
    /\ inc = 1
    /\ inc' = 2
    /\ poisoned' = FALSE
    /\ rec_req' = FALSE
    /\ t1' = IF t1 \in { Committed, Partial, InDoubt } THEN Superseded ELSE t1
    /\ superseded_seen' = (superseded_seen \/ t1 \in { Committed, Partial, InDoubt })
    /\ UNCHANGED << undo_inflight, undo_inc >>

Done ==
    /\ ~undo_inflight
    /\ t1 \in { Undone, Superseded }
    /\ UNCHANGED vars

Next ==
    \/ CommitT1
    \/ UndoStart
    \/ UndoFinishOk
    \/ UndoFailPartial
    \/ UndoFailInDoubt
    \/ RecoverRespawn
    \/ Done

InDoubtUndoLatchesRecovery ==
    (t1 = InDoubt) => rec_req

NoUndoAfterInDoubt ==
    ~(undo_inflight /\ t1 = InDoubt)

SupersededNeverOverwritten ==
    superseded_seen => t1 = Superseded

====
