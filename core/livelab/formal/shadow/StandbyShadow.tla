---- MODULE StandbyShadow ----
EXTENDS Naturals

CONSTANTS Active, Crashed, MaxShadows

VARIABLES primary, parked, dead_head, in_flight, pending_replenish,
          last_failed, waiting, closed, orphans

vars == << primary, parked, dead_head, in_flight, pending_replenish,
           last_failed, waiting, closed, orphans >>

Init ==
    /\ primary = Active
    /\ parked = MaxShadows
    /\ dead_head = FALSE
    /\ in_flight = FALSE
    /\ pending_replenish = FALSE
    /\ last_failed = FALSE
    /\ waiting = FALSE
    /\ closed = FALSE
    /\ orphans = 0

PrimaryCrash ==
    /\ ~closed /\ ~waiting
    /\ primary = Active
    /\ primary' = Crashed
    /\ UNCHANGED << parked, dead_head, in_flight, pending_replenish,
                    last_failed, waiting, closed, orphans >>

ShadowDies ==
    /\ ~closed
    /\ parked > 0
    /\ parked' = parked - 1
    /\ dead_head' = TRUE
    /\ UNCHANGED << primary, in_flight, pending_replenish,
                    last_failed, waiting, closed, orphans >>

FastPromote ==
    /\ ~closed /\ ~waiting
    /\ primary = Crashed
    /\ parked > 0
    /\ primary' = Active
    /\ parked' = parked - 1
    /\ dead_head' = FALSE
    /\ in_flight' = TRUE
    /\ pending_replenish' = IF in_flight THEN TRUE ELSE FALSE
    /\ UNCHANGED << last_failed, waiting, closed, orphans >>

WaitForRestock ==
    /\ ~closed /\ ~waiting
    /\ primary = Crashed
    /\ parked = 0
    /\ in_flight
    /\ waiting' = TRUE
    /\ UNCHANGED << primary, parked, dead_head, in_flight, pending_replenish,
                    last_failed, closed, orphans >>

EarlyWakePromote ==
    /\ ~closed /\ waiting
    /\ parked > 0
    /\ primary' = Active
    /\ parked' = parked - 1
    /\ dead_head' = FALSE
    /\ waiting' = FALSE
    /\ in_flight' = TRUE
    /\ pending_replenish' = IF in_flight THEN TRUE ELSE FALSE
    /\ UNCHANGED << last_failed, closed, orphans >>

WaitTimeoutFallback ==
    /\ ~closed /\ waiting
    /\ parked = 0
    /\ primary' = Active
    /\ dead_head' = FALSE
    /\ waiting' = FALSE
    /\ UNCHANGED << parked, in_flight, pending_replenish, last_failed, closed, orphans >>

IdleEmptyFallbackOrNudge ==
    /\ ~closed /\ ~waiting
    /\ primary = Crashed
    /\ parked = 0 /\ ~in_flight
    /\ dead_head' = FALSE
    /\ in_flight' = TRUE
    /\ pending_replenish' = FALSE
    /\ primary' \in { Active, Crashed }
    /\ UNCHANGED << parked, last_failed, waiting, closed, orphans >>

WorkerFinishOk ==
    /\ in_flight
    /\ last_failed' = FALSE
    /\ IF closed
       THEN /\ in_flight' = FALSE
            /\ pending_replenish' = FALSE
            /\ waiting' = FALSE
            /\ UNCHANGED << primary, parked, dead_head, closed, orphans >>
       ELSE /\ parked' = IF parked < MaxShadows THEN parked + 1 ELSE parked
            /\ IF pending_replenish
               THEN /\ in_flight' = TRUE
                    /\ pending_replenish' = FALSE
               ELSE /\ in_flight' = FALSE
                    /\ pending_replenish' = FALSE
            /\ UNCHANGED << primary, dead_head, waiting, closed, orphans >>

WorkerFinishFail ==
    /\ in_flight
    /\ last_failed' = TRUE
    /\ IF closed
       THEN /\ in_flight' = FALSE
            /\ pending_replenish' = FALSE
            /\ waiting' = FALSE
            /\ UNCHANGED << primary, parked, dead_head, closed, orphans >>
       ELSE /\ IF pending_replenish
               THEN /\ in_flight' = TRUE
                    /\ pending_replenish' = FALSE
               ELSE /\ in_flight' = FALSE
                    /\ pending_replenish' = FALSE
            /\ UNCHANGED << primary, parked, dead_head, waiting, closed, orphans >>

Close ==
    /\ ~closed
    /\ closed' = TRUE
    /\ parked' = 0
    /\ dead_head' = FALSE
    /\ pending_replenish' = FALSE
    /\ waiting' = FALSE
    /\ UNCHANGED << primary, in_flight, last_failed, orphans >>

Done ==
    /\ closed /\ ~in_flight
    /\ UNCHANGED vars

Next ==
    \/ PrimaryCrash
    \/ ShadowDies
    \/ FastPromote
    \/ WaitForRestock
    \/ EarlyWakePromote
    \/ WaitTimeoutFallback
    \/ IdleEmptyFallbackOrNudge
    \/ WorkerFinishOk
    \/ WorkerFinishFail
    \/ Close
    \/ Done

BoundedParkedShadows ==
    parked \in 0..MaxShadows

SingleFlightReplenisher ==
    pending_replenish => in_flight

NeverWaitWhenClosed ==
    closed => ~waiting

ZeroOrphanOnClose ==
    (closed /\ ~in_flight) => (parked = 0 /\ ~dead_head /\ ~pending_replenish /\ orphans = 0)

ZeroOrphanLeak ==
    orphans = 0

NeverClosed ==
    ~closed

====
