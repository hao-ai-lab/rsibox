---------------------------- MODULE ForksafeLockOrder ----------------------------
EXTENDS Naturals

(*
 * Bounded pre-fork guard ordering & poison-refusal protocol for RSIBox:
 * core/livelab/forksafe.py (install_all_guards, unpatch_child) &
 * core/livelab/rsibox/services/zygote.py (ZygoteService).
 * Models guard installation, Triton pin/unpin around module preloading,
 * fail-closed CUDA driver poison refusal, and child unpatching after fork.
 *)

CONSTANTS Bug, MaxChildren

VARIABLES
    cuda_initialized, \* BOOLEAN: CUDA driver/runtime initialized in parent
    guards_armed,     \* BOOLEAN: forksafe CUDA/cuInit guards armed
    triton_pinned,    \* BOOLEAN: Triton CudaDriver.get_current_target pinned
    preloaded,        \* BOOLEAN: engine modules preloaded
    forked_children,  \* 0..MaxChildren: number of forked children
    child_unpatched,  \* BOOLEAN: latest forked child completed unpatch_child()
    closed            \* BOOLEAN: ZygoteService closed / torn down

vars == <<cuda_initialized, guards_armed, triton_pinned, preloaded,
          forked_children, child_unpatched, closed>>

TypeOK ==
    /\ cuda_initialized \in BOOLEAN
    /\ guards_armed \in BOOLEAN
    /\ triton_pinned \in BOOLEAN
    /\ preloaded \in BOOLEAN
    /\ forked_children \in 0..MaxChildren
    /\ child_unpatched \in BOOLEAN
    /\ closed \in BOOLEAN

Init ==
    /\ cuda_initialized = FALSE
    /\ guards_armed = FALSE
    /\ triton_pinned = FALSE
    /\ preloaded = FALSE
    /\ forked_children = 0
    /\ child_unpatched = FALSE
    /\ closed = FALSE

CudaInitPoison ==
    /\ ~closed
    /\ forked_children = 0
    /\ ~cuda_initialized
    /\ cuda_initialized' = TRUE
    /\ UNCHANGED <<guards_armed, triton_pinned, preloaded, forked_children, child_unpatched, closed>>

ArmGuards ==
    /\ ~closed
    /\ ~guards_armed
    /\ ~cuda_initialized
    /\ guards_armed' = TRUE
    /\ triton_pinned' = TRUE
    /\ UNCHANGED <<cuda_initialized, preloaded, forked_children, child_unpatched, closed>>

PreloadModules ==
    /\ ~closed
    /\ ~preloaded
    /\ ~cuda_initialized
    /\ (Bug = "PreloadBeforeGuards" \/ guards_armed)
    /\ preloaded' = TRUE
    /\ triton_pinned' = FALSE
    /\ UNCHANGED <<cuda_initialized, guards_armed, forked_children, child_unpatched, closed>>

ForkChild ==
    /\ ~closed
    /\ guards_armed
    /\ ~triton_pinned
    /\ preloaded
    /\ (Bug = "ForkWhileCudaInit" \/ ~cuda_initialized)
    /\ forked_children < MaxChildren
    /\ (forked_children = 0 \/ child_unpatched)
    /\ forked_children' = forked_children + 1
    /\ child_unpatched' = FALSE
    /\ UNCHANGED <<cuda_initialized, guards_armed, triton_pinned, preloaded, closed>>

ChildUnpatch ==
    /\ ~closed
    /\ forked_children > 0
    /\ ~child_unpatched
    /\ child_unpatched' = TRUE
    /\ UNCHANGED <<cuda_initialized, guards_armed, triton_pinned, preloaded, forked_children, closed>>

Close ==
    /\ ~closed
    /\ (forked_children = 0 \/ child_unpatched)
    /\ closed' = TRUE
    /\ guards_armed' = FALSE
    /\ triton_pinned' = FALSE
    /\ UNCHANGED <<cuda_initialized, preloaded, forked_children, child_unpatched>>

Done ==
    /\ closed
    /\ UNCHANGED vars

Next ==
    \/ CudaInitPoison
    \/ ArmGuards
    \/ PreloadModules
    \/ ForkChild
    \/ ChildUnpatch
    \/ Close
    \/ Done

\* Safety invariants
PreloadAfterGuards ==
    (preloaded /\ ~closed) => guards_armed

ForkRequiresCleanDriver ==
    (forked_children > 0) => ~cuda_initialized

ReadyChildUnpatched ==
    (child_unpatched /\ ~closed) => (forked_children > 0 /\ guards_armed /\ ~triton_pinned)

\* Reachability witness invariant (expected to fail on non-mutant model)
NeverChildReady ==
    ~(forked_children > 0 /\ child_unpatched)

Spec == Init /\ [][Next]_vars
=============================================================================
