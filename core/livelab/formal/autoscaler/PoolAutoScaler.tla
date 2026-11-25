---- MODULE PoolAutoScaler ----
EXTENDS Naturals

CONSTANTS Healthy, Quarantined, Undrained, Removed,
          ActNone, ActHeal, ActHealFailed, ActReplaceFailed,
          ActReplace, ActScaleUp, ActScaleDown, ActNoop,
          MinSlots, MaxSlots, MaxHealAttempts

VARIABLES focal, extra_healthy, heal_attempts, backoff, exhausted,
          replace_pending, queue_busy, pool_closed, last_action

vars == << focal, extra_healthy, heal_attempts, backoff, exhausted,
           replace_pending, queue_busy, pool_closed, last_action >>

HealthyCount ==
    (IF focal = Healthy THEN 1 ELSE 0) + extra_healthy

ActiveCount ==
    (IF focal \in { Healthy, Quarantined, Undrained } THEN 1 ELSE 0) + extra_healthy

Priority1Eligible ==
    /\ focal \in { Quarantined, Undrained }
    /\ ~backoff
    /\ (~exhausted \/ replace_pending)

Init ==
    /\ focal = Healthy
    /\ extra_healthy = 0
    /\ heal_attempts = 0
    /\ backoff = FALSE
    /\ exhausted = FALSE
    /\ replace_pending = FALSE
    /\ queue_busy = FALSE
    /\ pool_closed = FALSE
    /\ last_action = ActNone

EnvFaultClean ==
    /\ ~pool_closed /\ focal = Healthy
    /\ focal' = Quarantined
    /\ last_action' = ActNone
    /\ UNCHANGED << extra_healthy, heal_attempts, backoff, exhausted,
                    replace_pending, queue_busy, pool_closed >>

EnvFaultUndrained ==
    /\ ~pool_closed /\ focal = Healthy
    /\ focal' = Undrained
    /\ last_action' = ActNone
    /\ UNCHANGED << extra_healthy, heal_attempts, backoff, exhausted,
                    replace_pending, queue_busy, pool_closed >>

EnvCallDrains ==
    /\ ~pool_closed /\ focal = Undrained
    /\ focal' = Quarantined
    /\ last_action' = ActNone
    /\ UNCHANGED << extra_healthy, heal_attempts, backoff, exhausted,
                    replace_pending, queue_busy, pool_closed >>

EnvQueueToggle ==
    /\ ~pool_closed
    /\ queue_busy' = ~queue_busy
    /\ last_action' = ActNone
    /\ UNCHANGED << focal, extra_healthy, heal_attempts, backoff, exhausted,
                    replace_pending, pool_closed >>

EnvBackoffExpires ==
    /\ ~pool_closed /\ backoff
    /\ backoff' = FALSE
    /\ last_action' = ActNone
    /\ UNCHANGED << focal, extra_healthy, heal_attempts, exhausted,
                    replace_pending, queue_busy, pool_closed >>

EnvClosePool ==
    /\ ~pool_closed
    /\ pool_closed' = TRUE
    /\ last_action' = ActNone
    /\ UNCHANGED << focal, extra_healthy, heal_attempts, backoff, exhausted,
                    replace_pending, queue_busy >>

ReconcilePoolClosed ==
    /\ pool_closed
    /\ last_action' = ActNoop
    /\ UNCHANGED << focal, extra_healthy, heal_attempts, backoff, exhausted,
                    replace_pending, queue_busy, pool_closed >>

HealUndrainedBackoff ==
    /\ ~pool_closed /\ Priority1Eligible
    /\ ~replace_pending /\ focal = Undrained
    /\ backoff' = TRUE
    /\ last_action' = ActNoop
    /\ UNCHANGED << focal, extra_healthy, heal_attempts, exhausted,
                    replace_pending, queue_busy, pool_closed >>

HealSucceed ==
    /\ ~pool_closed /\ Priority1Eligible
    /\ ~replace_pending /\ focal = Quarantined
    /\ focal' = Healthy
    /\ heal_attempts' = 0
    /\ backoff' = FALSE
    /\ exhausted' = FALSE
    /\ replace_pending' = FALSE
    /\ last_action' = ActHeal
    /\ UNCHANGED << extra_healthy, queue_busy, pool_closed >>

HealFailRetry ==
    /\ ~pool_closed /\ Priority1Eligible
    /\ ~replace_pending /\ focal = Quarantined
    /\ heal_attempts + 1 < MaxHealAttempts
    /\ heal_attempts' = heal_attempts + 1
    /\ backoff' = TRUE
    /\ last_action' = ActHealFailed
    /\ UNCHANGED << focal, extra_healthy, exhausted, replace_pending,
                    queue_busy, pool_closed >>

HealFailExhaustAndDrainFail ==
    /\ ~pool_closed /\ Priority1Eligible
    /\ ~replace_pending /\ focal = Quarantined
    /\ heal_attempts + 1 = MaxHealAttempts
    /\ heal_attempts' = MaxHealAttempts
    /\ exhausted' = TRUE
    /\ replace_pending' = TRUE
    /\ backoff' = TRUE
    /\ last_action' = ActReplaceFailed
    /\ UNCHANGED << focal, extra_healthy, queue_busy, pool_closed >>

HealFailExhaustAndReplaceOk ==
    /\ ~pool_closed /\ Priority1Eligible
    /\ ~replace_pending /\ focal = Quarantined
    /\ heal_attempts + 1 = MaxHealAttempts
    /\ focal' = Removed
    /\ extra_healthy' = extra_healthy + 1
    /\ heal_attempts' = 0
    /\ backoff' = FALSE
    /\ exhausted' = FALSE
    /\ replace_pending' = FALSE
    /\ last_action' = ActReplace
    /\ UNCHANGED << queue_busy, pool_closed >>

ReplaceRetryFail ==
    /\ ~pool_closed /\ Priority1Eligible
    /\ replace_pending
    /\ backoff' = TRUE
    /\ last_action' = ActReplaceFailed
    /\ UNCHANGED << focal, extra_healthy, heal_attempts, exhausted,
                    replace_pending, queue_busy, pool_closed >>

ReplaceRetryOk ==
    /\ ~pool_closed /\ Priority1Eligible
    /\ replace_pending
    /\ focal' = Removed
    /\ extra_healthy' = extra_healthy + 1
    /\ heal_attempts' = 0
    /\ backoff' = FALSE
    /\ exhausted' = FALSE
    /\ replace_pending' = FALSE
    /\ last_action' = ActReplace
    /\ UNCHANGED << queue_busy, pool_closed >>

ScaleUp ==
    /\ ~pool_closed /\ ~Priority1Eligible
    /\ ActiveCount < MaxSlots
    /\ (HealthyCount < MinSlots \/ queue_busy)
    /\ extra_healthy' = extra_healthy + 1
    /\ last_action' = ActScaleUp
    /\ UNCHANGED << focal, heal_attempts, backoff, exhausted,
                    replace_pending, queue_busy, pool_closed >>

ScaleDown ==
    /\ ~pool_closed /\ ~Priority1Eligible
    /\ ~(ActiveCount < MaxSlots /\ (HealthyCount < MinSlots \/ queue_busy))
    /\ HealthyCount > MinSlots /\ ~queue_busy
    /\ IF extra_healthy > 0
       THEN /\ extra_healthy' = extra_healthy - 1
            /\ UNCHANGED << focal, heal_attempts, backoff, exhausted, replace_pending >>
       ELSE /\ focal = Healthy
            /\ focal' = Removed
            /\ heal_attempts' = 0
            /\ backoff' = FALSE
            /\ exhausted' = FALSE
            /\ replace_pending' = FALSE
            /\ UNCHANGED extra_healthy
    /\ last_action' = ActScaleDown
    /\ UNCHANGED << queue_busy, pool_closed >>

ReconcileNoop ==
    /\ ~pool_closed /\ ~Priority1Eligible
    /\ ~(ActiveCount < MaxSlots /\ (HealthyCount < MinSlots \/ queue_busy))
    /\ ~(HealthyCount > MinSlots /\ ~queue_busy)
    /\ last_action' = ActNoop
    /\ UNCHANGED << focal, extra_healthy, heal_attempts, backoff, exhausted,
                    replace_pending, queue_busy, pool_closed >>

Done ==
    /\ pool_closed /\ last_action = ActNoop
    /\ UNCHANGED vars

Next ==
    \/ EnvFaultClean
    \/ EnvFaultUndrained
    \/ EnvCallDrains
    \/ EnvQueueToggle
    \/ EnvBackoffExpires
    \/ EnvClosePool
    \/ ReconcilePoolClosed
    \/ HealUndrainedBackoff
    \/ HealSucceed
    \/ HealFailRetry
    \/ HealFailExhaustAndDrainFail
    \/ HealFailExhaustAndReplaceOk
    \/ ReplaceRetryFail
    \/ ReplaceRetryOk
    \/ ScaleUp
    \/ ScaleDown
    \/ ReconcileNoop
    \/ Done

BoundedSlots ==
    /\ ActiveCount \in 0..MaxSlots
    /\ HealthyCount \in 0..MaxSlots

NeverScaleDownBelowMin ==
    (last_action = ActScaleDown) => (HealthyCount >= MinSlots)

UndrainedNeverExhaustsDirectly ==
    (focal = Undrained /\ ~replace_pending) => ~exhausted

ExhaustedImpliesMaxAttempts ==
    /\ replace_pending => exhausted
    /\ exhausted => (heal_attempts = MaxHealAttempts)

RemovedClearsTracking ==
    (focal = Removed) => (heal_attempts = 0 /\ ~backoff /\ ~exhausted /\ ~replace_pending)

ClosedYieldsNoop ==
    (pool_closed /\ last_action # ActNone) => (last_action = ActNoop)

NeverScaleDown ==
    last_action # ActScaleDown

====
