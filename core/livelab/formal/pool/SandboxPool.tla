---- MODULE SandboxPool ----
EXTENDS Naturals

CONSTANTS Idle, Leased, Quarantined, Healing, DrainingLeased, DrainingHealing, Removed,
          RepNone, RepCallerOwned, RepAttached

VARIABLES slot, rep_lab, active_call, deferred_close, lab_closed, closed

vars == << slot, rep_lab, active_call, deferred_close, lab_closed, closed >>

Init ==
    /\ slot = Idle
    /\ rep_lab = RepNone
    /\ active_call = FALSE
    /\ deferred_close = FALSE
    /\ lab_closed = FALSE
    /\ closed = FALSE

Acquire ==
    /\ ~closed /\ ~lab_closed
    /\ slot = Idle
    /\ slot' = Leased
    /\ UNCHANGED << rep_lab, active_call, deferred_close, lab_closed, closed >>

CallStart ==
    /\ slot \in { Leased, DrainingLeased }
    /\ ~active_call /\ ~lab_closed
    /\ active_call' = TRUE
    /\ UNCHANGED << slot, rep_lab, deferred_close, lab_closed, closed >>

CallExit ==
    /\ active_call
    /\ active_call' = FALSE
    /\ IF slot \notin { Leased, DrainingLeased } /\ deferred_close
       THEN /\ lab_closed' = TRUE
            /\ deferred_close' = FALSE
       ELSE UNCHANGED << lab_closed, deferred_close >>
    /\ UNCHANGED << slot, rep_lab, closed >>

ReleaseClean ==
    /\ slot \in { Leased, DrainingLeased }
    /\ ~active_call
    /\ IF slot = DrainingLeased
       THEN /\ slot' = Removed
            /\ lab_closed' = IF deferred_close THEN TRUE ELSE lab_closed
            /\ deferred_close' = FALSE
       ELSE /\ slot' = Idle
            /\ IF closed /\ deferred_close
               THEN /\ lab_closed' = TRUE
                    /\ deferred_close' = FALSE
               ELSE UNCHANGED << lab_closed, deferred_close >>
    /\ UNCHANGED << rep_lab, active_call, closed >>

ReleaseQuarantine ==
    /\ slot \in { Leased, DrainingLeased }
    /\ IF slot = DrainingLeased
       THEN /\ slot' = Removed
            /\ IF ~active_call /\ deferred_close
               THEN /\ lab_closed' = TRUE
                    /\ deferred_close' = FALSE
               ELSE UNCHANGED << lab_closed, deferred_close >>
       ELSE /\ slot' = Quarantined
            /\ IF ~active_call /\ closed /\ deferred_close
               THEN /\ lab_closed' = TRUE
                    /\ deferred_close' = FALSE
               ELSE UNCHANGED << lab_closed, deferred_close >>
    /\ UNCHANGED << rep_lab, active_call, closed >>

HealStartWithReplacement ==
    /\ ~closed /\ ~active_call /\ ~lab_closed
    /\ slot = Quarantined
    /\ slot' = Healing
    /\ rep_lab' = RepCallerOwned
    /\ UNCHANGED << active_call, deferred_close, lab_closed, closed >>

HealFinishOk ==
    /\ slot = Healing
    /\ ~closed
    /\ slot' = Idle
    /\ rep_lab' = RepAttached
    /\ UNCHANGED << active_call, deferred_close, lab_closed, closed >>

HealAbortedByDrainOrClose ==
    /\ slot \in { Healing, DrainingHealing }
    /\ (closed \/ slot = DrainingHealing)
    \* Caller retains ownership of unattached replacement lab (PR #256 contract).
    /\ IF slot = DrainingHealing
       THEN /\ slot' = Removed
            /\ IF deferred_close /\ ~active_call
               THEN /\ lab_closed' = TRUE
                    /\ deferred_close' = FALSE
               ELSE UNCHANGED << lab_closed, deferred_close >>
       ELSE /\ slot' = Quarantined
            /\ IF closed /\ deferred_close /\ ~active_call
               THEN /\ lab_closed' = TRUE
                    /\ deferred_close' = FALSE
               ELSE UNCHANGED << lab_closed, deferred_close >>
    /\ UNCHANGED << rep_lab, active_call, closed >>

DrainStartCloseLab ==
    /\ ~closed
    /\ slot \in { Idle, Quarantined, Leased, Healing }
    /\ CASE slot = Idle ->
            /\ slot' = Removed
            /\ IF active_call
               THEN /\ deferred_close' = TRUE
                    /\ UNCHANGED lab_closed
               ELSE /\ lab_closed' = TRUE
                    /\ UNCHANGED deferred_close
         [] slot = Quarantined ->
            /\ slot' = Removed
            /\ IF active_call
               THEN /\ deferred_close' = TRUE
                    /\ UNCHANGED lab_closed
               ELSE /\ lab_closed' = TRUE
                    /\ UNCHANGED deferred_close
         [] slot = Leased ->
            /\ slot' = DrainingLeased
            /\ deferred_close' = TRUE
            /\ UNCHANGED lab_closed
         [] slot = Healing ->
            /\ slot' = DrainingHealing
            /\ deferred_close' = TRUE
            /\ UNCHANGED lab_closed
    /\ UNCHANGED << rep_lab, active_call, closed >>

DrainTimeoutRevert ==
    /\ ~closed
    /\ slot \in { DrainingLeased, DrainingHealing }
    /\ slot' = IF slot = DrainingLeased THEN Leased ELSE Healing
    /\ deferred_close' = FALSE
    /\ UNCHANGED << rep_lab, active_call, lab_closed, closed >>

ClosePoolWithCloseLabs ==
    /\ ~closed
    /\ closed' = TRUE
    /\ IF active_call \/ slot \in { Leased, DrainingLeased, Healing, DrainingHealing }
       THEN /\ deferred_close' = TRUE
            /\ UNCHANGED lab_closed
       ELSE /\ lab_closed' = TRUE
            /\ deferred_close' = FALSE
    /\ UNCHANGED << slot, rep_lab, active_call >>

Done ==
    /\ (closed \/ slot = Removed)
    /\ ~active_call /\ ~deferred_close
    /\ slot \notin { Leased, DrainingLeased, Healing, DrainingHealing }
    /\ UNCHANGED vars

Next ==
    \/ Acquire
    \/ CallStart
    \/ CallExit
    \/ ReleaseClean
    \/ ReleaseQuarantine
    \/ HealStartWithReplacement
    \/ HealFinishOk
    \/ HealAbortedByDrainOrClose
    \/ DrainStartCloseLab
    \/ DrainTimeoutRevert
    \/ ClosePoolWithCloseLabs
    \/ Done

NeverCloseLabWhileCallActive ==
    active_call => ~lab_closed

NeverHealWhileCallActive ==
    ~(active_call /\ slot = Healing)

DeferredCloseCompletes ==
    (deferred_close /\ ~active_call /\ slot \notin { Leased, DrainingLeased, Healing, DrainingHealing }) => lab_closed

====
