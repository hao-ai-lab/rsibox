namespace RsiboxFormal

/-!
Unbounded algebraic safety proofs for RSIBox protocol invariants (Lean 4.16.0 core `Init` only).
-/

/-- Increment the coordinator incarnation epoch on respawn. -/
def respawn (cur : Nat) : Nat := cur + 1

/-- Iterate `k` coordinator respawns starting from `cur`. -/
def respawnK (cur k : Nat) : Nat :=
  match k with
  | 0 => cur
  | n + 1 => respawn (respawnK cur n)

/-- 1. Any delayed message with incarnation `< cur` is rejected at `cur` and after `respawn cur`. -/
theorem incarnation_stale_rejected (cur msg : Nat) (h : msg < cur) :
    msg ≠ cur ∧ msg ≠ respawn cur := by
  unfold respawn
  constructor <;> omega

/-- 2. Iterating `k` respawns produces `cur + k ≥ cur`, strictly increasing when `k > 0`. -/
theorem incarnation_monotone_after_k (cur k : Nat) :
    respawnK cur k ≥ cur ∧ (k > 0 → cur < respawnK cur k) := by
  have h_eq : ∀ n : Nat, respawnK cur n = cur + n := by
    intro n
    induction n with
    | zero => simp [respawnK]
    | succ m ih =>
      simp only [respawnK, respawn, ih]
      omega
  rw [h_eq k]
  constructor
  · omega
  · intro _
    omega

/-- All-or-nothing commit predicate over an arbitrary-length participant vote list. -/
def allCommitted (votes : List Bool) : Bool :=
  match votes with
  | [] => true
  | v :: vs => v && allCommitted vs

/-- 3. Any `false` vote prevents a split/partial commit across an unbounded participant list. -/
theorem two_phase_no_split_commit (votes : List Bool) (h : false ∈ votes) :
    allCommitted votes = false := by
  induction votes with
  | nil => simp at h
  | cons v vs ih =>
    simp only [List.mem_cons] at h
    cases h with
    | inl heq =>
      rw [← heq]
      simp [allCommitted]
    | inr hmem =>
      have hvs := ih hmem
      cases v <;> simp [allCommitted, hvs]

/-- Phase-2 terminal outcome for `RecaptureCoordinator`. -/
inductive Phase2Outcome where
  | committed
  | cleanRollback
  | recoveryLatched
  deriving DecidableEq, Repr

/-- Phase-2 state recording terminal outcome and fail-closed recovery latch. -/
structure Phase2State where
  outcome : Phase2Outcome
  recoveryLatched : Bool
  deriving DecidableEq, Repr

/-- Deterministic Phase-2 transition from apply and undo status booleans. -/
def runPhase2 (applyOk undoOk : Bool) : Phase2State :=
  if applyOk then
    { outcome := .committed, recoveryLatched := false }
  else if undoOk then
    { outcome := .cleanRollback, recoveryLatched := false }
  else
    { outcome := .recoveryLatched, recoveryLatched := true }

/-- 4. Phase-2 transition is total and fail-closed: clean apply commits without latching,
    failed apply + clean undo rolls back cleanly, and failed apply + incomplete undo latches recovery. -/
theorem rollback_or_recovery_latched (applyOk undoOk : Bool) :
    (applyOk = true →
      (runPhase2 applyOk undoOk).outcome = .committed ∧
      (runPhase2 applyOk undoOk).recoveryLatched = false) ∧
    (applyOk = false ∧ undoOk = true →
      (runPhase2 applyOk undoOk).outcome = .cleanRollback ∧
      (runPhase2 applyOk undoOk).recoveryLatched = false) ∧
    (applyOk = false ∧ undoOk = false →
      (runPhase2 applyOk undoOk).recoveryLatched = true ∧
      (runPhase2 applyOk undoOk).outcome = .recoveryLatched) := by
  cases applyOk <;> cases undoOk <;> simp [runPhase2]

/-- Sandbox pool slot count accounting across all lifecycle buckets. -/
structure PoolState where
  cap : Nat
  active : Nat
  standby : Nat
  healing : Nat
  closed : Nat
  deriving DecidableEq, Repr

/-- Capacity conservation invariant across unbounded pool sizes. -/
def poolConserved (s : PoolState) : Prop :=
  s.active + s.standby + s.healing + s.closed = s.cap

def poolAcquire (s : PoolState) : PoolState :=
  if s.standby ≥ 1 then { s with active := s.active + 1, standby := s.standby - 1 } else s

def poolRelease (s : PoolState) : PoolState :=
  if s.active ≥ 1 then { s with active := s.active - 1, standby := s.standby + 1 } else s

def poolHealStart (s : PoolState) : PoolState :=
  if s.standby ≥ 1 then { s with standby := s.standby - 1, healing := s.healing + 1 } else s

def poolHealFinish (s : PoolState) : PoolState :=
  if s.healing ≥ 1 then { s with healing := s.healing - 1, standby := s.standby + 1 } else s

def poolCloseSlot (s : PoolState) : PoolState :=
  if s.standby ≥ 1 then { s with standby := s.standby - 1, closed := s.closed + 1 } else s

/-- 5. Every pool lifecycle transition preserves total slot capacity conservation. -/
theorem sandbox_pool_capacity_conservation
    (cap active standby healing closed : Nat)
    (h : active + standby + healing + closed = cap) :
    let s : PoolState := ⟨cap, active, standby, healing, closed⟩
    poolConserved (poolAcquire s) ∧
    poolConserved (poolRelease s) ∧
    poolConserved (poolHealStart s) ∧
    poolConserved (poolHealFinish s) ∧
    poolConserved (poolCloseSlot s) := by
  intro s
  unfold poolConserved poolAcquire poolRelease poolHealStart poolHealFinish poolCloseSlot
  dsimp only [s]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · by_cases hg : standby ≥ 1 <;> simp [hg] <;> omega
  · by_cases hg : active ≥ 1 <;> simp [hg] <;> omega
  · by_cases hg : standby ≥ 1 <;> simp [hg] <;> omega
  · by_cases hg : healing ≥ 1 <;> simp [hg] <;> omega
  · by_cases hg : standby ≥ 1 <;> simp [hg] <;> omega

/-- Standby shadow promotion state. -/
structure ShadowState where
  promoting : Bool
  standby : Nat
  deriving DecidableEq, Repr

/-- Single-flight promotion guard: requires `¬promoting` and at least one standby. -/
def canPromote (s : ShadowState) : Bool :=
  (!s.promoting) && decide (s.standby ≥ 1)

/-- Start a shadow promotion if eligible. -/
def startPromote (s : ShadowState) : ShadowState :=
  if canPromote s then { promoting := true, standby := s.standby - 1 } else s

/-- 6. Shadow promotion is single-flight: sets `promoting = true`, decrements `standby`,
    and forces any concurrent second promote to fail closed. -/
theorem standby_shadow_single_flight (s : ShadowState)
    (hReady : s.promoting = false ∧ s.standby ≥ 1) :
    let s' := startPromote s
    s'.promoting = true ∧
    s'.standby = s.standby - 1 ∧
    canPromote s' = false ∧
    s'.standby < s.standby := by
  rcases s with ⟨promoting, standby⟩
  rcases hReady with ⟨hProm, hStandby⟩
  dsimp only at hProm hStandby
  subst hProm
  have hCan : canPromote ⟨false, standby⟩ = true := by
    simp [canPromote, hStandby]
  intro s'
  have hs' : s' = { promoting := true, standby := standby - 1 } := by
    dsimp only [s']
    simp [startPromote, hCan]
  rw [hs']
  refine ⟨rfl, rfl, ?_, ?_⟩
  · simp [canPromote]
  · dsimp only
    omega

end RsiboxFormal
