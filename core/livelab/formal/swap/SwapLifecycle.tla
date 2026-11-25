---- MODULE SwapLifecycle ----
EXTENDS Naturals

CONSTANTS Base, Staged, Validated, Applied, RolledBack, Undone, Poisoned

VARIABLES phase, live_code, staged_dirty, witness_dirty, hit_stub_leaked, journal_poisoned

vars == << phase, live_code, staged_dirty, witness_dirty, hit_stub_leaked, journal_poisoned >>

Init ==
    /\ phase = Base
    /\ live_code = "base"
    /\ staged_dirty = FALSE
    /\ witness_dirty = FALSE
    /\ hit_stub_leaked = FALSE
    /\ journal_poisoned = FALSE

Stage ==
    /\ phase = Base
    /\ ~journal_poisoned
    /\ phase' = Staged
    /\ staged_dirty' = FALSE
    /\ witness_dirty' = FALSE
    /\ hit_stub_leaked' = FALSE
    /\ UNCHANGED << live_code, journal_poisoned >>

ConcurrentMutate ==
    /\ phase = Staged
    /\ live_code = "base"
    /\ live_code' = "mutated"
    /\ UNCHANGED << phase, staged_dirty, witness_dirty, hit_stub_leaked, journal_poisoned >>

ValidateOk ==
    /\ phase = Staged
    /\ live_code = "base"
    /\ phase' = Validated
    /\ UNCHANGED << live_code, staged_dirty, witness_dirty, hit_stub_leaked, journal_poisoned >>

ValidateReject ==
    /\ phase = Staged
    /\ live_code = "mutated"
    /\ phase' = Base
    /\ UNCHANGED << live_code, staged_dirty, witness_dirty, hit_stub_leaked, journal_poisoned >>

ApplyOk ==
    /\ phase = Validated
    /\ ~journal_poisoned
    /\ phase' = Applied
    /\ live_code' = "swapped"
    /\ staged_dirty' = FALSE
    /\ witness_dirty' = TRUE
    /\ hit_stub_leaked' = FALSE
    /\ UNCHANGED journal_poisoned

ApplyFailMidway ==
    /\ phase = Validated
    /\ ~journal_poisoned
    \* S1 + S2: Failed apply() restores witness snapshots, leaves Staged intact, and rolls back W.HIT.
    /\ phase' = RolledBack
    /\ staged_dirty' = FALSE
    /\ witness_dirty' = FALSE
    /\ hit_stub_leaked' = FALSE
    /\ UNCHANGED << live_code, journal_poisoned >>

UndoOk ==
    /\ phase = Applied
    /\ phase' = Undone
    /\ live_code' = "base"
    /\ witness_dirty' = FALSE
    /\ hit_stub_leaked' = FALSE
    /\ UNCHANGED << staged_dirty, journal_poisoned >>

UndoPartialPoison ==
    /\ phase = Applied
    /\ phase' = Poisoned
    /\ journal_poisoned' = TRUE
    /\ UNCHANGED << live_code, staged_dirty, witness_dirty, hit_stub_leaked >>

Done ==
    /\ (phase \in { RolledBack, Undone, Poisoned } \/ (phase = Base /\ live_code = "mutated"))
    /\ UNCHANGED vars

Next ==
    \/ Stage
    \/ ConcurrentMutate
    \/ ValidateOk
    \/ ValidateReject
    \/ ApplyOk
    \/ ApplyFailMidway
    \/ UndoOk
    \/ UndoPartialPoison
    \/ Done

FailedApplyRestoresAll ==
    (phase = RolledBack) => (~witness_dirty /\ ~staged_dirty /\ ~hit_stub_leaked)

NeverRolledBack ==
    phase # RolledBack

====
