# Formal Verification (`core/livelab/formal/`)

Two complementary layers verify RSIBox protocol state machines and algebraic invariants:

1. **Stage A — Bounded TLA+ / TLC State Machines**
   - Specs live under `core/livelab/formal/{pool,coordinator,swap,shadow,autoscaler}/` and `core/livelab/kernel/formal/`.
   - Anti-cheat gates:
     - `gates/check_spec_pins.py`: verifies `SPEC_PINS.sha256` matches every `.tla` file on disk.
     - `gates/check_anchors.py`: verifies every spec-to-Python anchor in `anchors.tsv`.
     - `gates/assumptions_check.py`: rejects vacuous `ASSUME FALSE`, disabled `Spec`, stuttering-only `Next`, and empty invariants.
   - Runner: `python3 core/livelab/formal/run_formal.py --jar .tla-cache/tla2tools.jar --bootstrap`

2. **Stage B — Unbounded Lean 4 Algebraic Proofs (`lean/`)**
   - Pure Lean 4.16.0 (`Init` only, zero Mathlib) project in [lean/lakefile.toml](lean/lakefile.toml) proving six unbounded invariants in [lean/RsiboxFormal/Invariants.lean](lean/RsiboxFormal/Invariants.lean):
     1. `incarnation_stale_rejected`
     2. `incarnation_monotone_after_k`
     3. `two_phase_no_split_commit`
     4. `rollback_or_recovery_latched`
     5. `sandbox_pool_capacity_conservation`
     6. `standby_shadow_single_flight`
   - Anti-cheat axiom gate: `python3 core/livelab/formal/gates/check_lean_axioms.py --root core/livelab/formal/lean`
   - CI builds `RsiboxFormal` and checks `lake env lean VerifyAxioms.lean` in [formal-tla.yml](../../../.github/workflows/formal-tla.yml).
