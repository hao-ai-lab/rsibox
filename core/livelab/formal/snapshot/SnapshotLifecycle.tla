---------------------------- MODULE SnapshotLifecycle ----------------------------
EXTENDS Naturals

(*
 * Bounded ai-dynamo/snapshot workload-contract lifecycle for RSIBox:
 * core/livelab/rsibox/warm/snapshot_contract.py (run_snapshot_lifecycle &
 * DynamoK8sSnapshotDriver).
 * Models Init -> Warmup -> Quiescing -> ReadyForSnapshot -> RestoreComplete ->
 * Resuming -> Ready (+ Failed branches) and fail-closed sentinel ordering.
 *)

CONSTANTS Bug

VARIABLES
    phase,                     \* "Init" | "Warmup" | "Quiescing" | "ReadyForSnapshot" | "RestoreComplete" | "Resuming" | "Ready" | "Failed"
    ready_sentinel,            \* BOOLEAN: ready-for-snapshot sentinel present
    restore_complete_sentinel, \* BOOLEAN: restore-complete sentinel present
    resume_ready_sentinel,     \* BOOLEAN: <engine>-restore-ready sentinel present
    memory_quiesced            \* BOOLEAN: GPU memory quiesced and parked

vars == <<phase, ready_sentinel, restore_complete_sentinel, resume_ready_sentinel, memory_quiesced>>

Phases == {"Init", "Warmup", "Quiescing", "ReadyForSnapshot",
           "RestoreComplete", "Resuming", "Ready", "Failed"}

TypeOK ==
    /\ phase \in Phases
    /\ ready_sentinel \in BOOLEAN
    /\ restore_complete_sentinel \in BOOLEAN
    /\ resume_ready_sentinel \in BOOLEAN
    /\ memory_quiesced \in BOOLEAN

Init ==
    /\ phase = "Init"
    /\ ready_sentinel = FALSE
    /\ restore_complete_sentinel = FALSE
    /\ resume_ready_sentinel = FALSE
    /\ memory_quiesced = FALSE

StartWarmup ==
    /\ phase = "Init"
    /\ phase' = "Warmup"
    /\ ready_sentinel' = FALSE
    /\ restore_complete_sentinel' = FALSE
    /\ resume_ready_sentinel' = FALSE
    /\ UNCHANGED <<memory_quiesced>>

WarmupOk ==
    /\ phase = "Warmup"
    /\ phase' = "Quiescing"
    /\ IF Bug = "WriteReadyBeforeQuiesce"
       THEN ready_sentinel' = TRUE
       ELSE UNCHANGED ready_sentinel
    /\ UNCHANGED <<restore_complete_sentinel, resume_ready_sentinel, memory_quiesced>>

WarmupFail ==
    /\ phase = "Warmup"
    /\ phase' = "Failed"
    /\ UNCHANGED <<ready_sentinel, restore_complete_sentinel, resume_ready_sentinel, memory_quiesced>>

QuiesceOk ==
    /\ phase = "Quiescing"
    /\ phase' = "ReadyForSnapshot"
    /\ memory_quiesced' = TRUE
    /\ ready_sentinel' = TRUE
    /\ UNCHANGED <<restore_complete_sentinel, resume_ready_sentinel>>

QuiesceFail ==
    /\ phase = "Quiescing"
    /\ phase' = "Failed"
    /\ memory_quiesced' = FALSE
    /\ ready_sentinel' = FALSE
    /\ UNCHANGED <<restore_complete_sentinel, resume_ready_sentinel>>

AgentRestoreComplete ==
    /\ phase = "ReadyForSnapshot"
    /\ ready_sentinel
    /\ phase' = "RestoreComplete"
    /\ restore_complete_sentinel' = TRUE
    /\ UNCHANGED <<ready_sentinel, resume_ready_sentinel, memory_quiesced>>

WaitRestoreTimeout ==
    /\ phase = "ReadyForSnapshot"
    /\ phase' = "Failed"
    /\ ready_sentinel' = FALSE
    /\ UNCHANGED <<restore_complete_sentinel, resume_ready_sentinel, memory_quiesced>>

BeginResume ==
    /\ (phase = "RestoreComplete" \/ (Bug = "WriteResumeBeforeRestore" /\ phase = "ReadyForSnapshot"))
    /\ phase' = "Resuming"
    /\ IF Bug = "WriteResumeBeforeRestore"
       THEN resume_ready_sentinel' = TRUE
       ELSE UNCHANGED resume_ready_sentinel
    /\ UNCHANGED <<ready_sentinel, restore_complete_sentinel, memory_quiesced>>

ResumeOk ==
    /\ phase = "Resuming"
    /\ restore_complete_sentinel
    /\ phase' = "Ready"
    /\ resume_ready_sentinel' = TRUE
    /\ UNCHANGED <<ready_sentinel, restore_complete_sentinel, memory_quiesced>>

ResumeFail ==
    /\ phase = "Resuming"
    /\ phase' = "Failed"
    /\ resume_ready_sentinel' = FALSE
    /\ UNCHANGED <<ready_sentinel, restore_complete_sentinel, memory_quiesced>>

Done ==
    /\ phase \in {"Ready", "Failed"}
    /\ UNCHANGED vars

Next ==
    \/ StartWarmup
    \/ WarmupOk
    \/ WarmupFail
    \/ QuiesceOk
    \/ QuiesceFail
    \/ AgentRestoreComplete
    \/ WaitRestoreTimeout
    \/ BeginResume
    \/ ResumeOk
    \/ ResumeFail
    \/ Done

\* Safety invariants
ReadySentinelRequiresQuiesce ==
    ready_sentinel => memory_quiesced

ResumeReadyRequiresRestore ==
    resume_ready_sentinel => restore_complete_sentinel

FailedNoResumeReady ==
    (phase = "Failed") => ~resume_ready_sentinel

\* Reachability witness invariant (expected to fail on non-mutant model)
NeverReady ==
    phase # "Ready"

Spec == Init /\ [][Next]_vars
=============================================================================
