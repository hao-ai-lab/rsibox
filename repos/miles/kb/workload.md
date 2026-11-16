# Workload and Verification Gate (`miles`)

## 1. Target Workload $F(x)$

For deterministic live-edit evaluation and conformance (`c0`–`c10`), `miles` runs a fixed-seed
(`seed = 0`, greedy / temperature-0 or fixed-RNG sampling) 1-step GRPO iteration starting from a
coupled initial state `S0`:

1. **Coupled initial state (`S0`):** Trainer ranks (`MegatronTrainRayActor` / `FSDPTrainRayActor`)
   restore their `S0` weights, optimizer state, and step RNG, broadcast `S0` weights to all rollout
   engines via `update_weights`, flush SGLang KV / Radix caches (`flush_cache`), and drain any
   buffered trajectories from earlier `code_epoch`s (`EpochDataBuffer.drain_stale(clear_all=True)`).
2. **Rollout & reward evaluation:** `RolloutExecutor` / `RolloutManager` generates trajectories on
   a fixed prompt batch and evaluates rewards (including any custom `.py` reward or filter function
   tracked via `ray_participants.file_namespace`).
3. **Policy update:** Trainer ranks compute group-normalized advantages (`miles/backends/training_utils/loss/objective.py`)
   and execute one training step, returning per-rank step metrics and updated parameter digests.

## 2. Correctness and Witness Gates

Every candidate edit must satisfy three checks before its measurement counts:

- **Repeatability (`c0_repeatable`):** Three consecutive `check()` calls at base (including an
  explicit `reset("state")`) produce identical per-rank outputs, confirming zero state or weight
  drift across checks.
- **Active witness (`c1`, `c2`, `c10`):** `Coordinator.prove()` confirms `witness > 0` on every
  participating actor copy (`<module>#actor` for `@ray.remote` methods, `<module>` for shared
  helpers, `fastrsi_file_<hash>` for `sys.modules`-less reward files). Startup-only edits that
  never execute during `check()` report `proven = False` (`zero_witness`).
- **Fresh-process equivalence & exact undo (`c3`, `c4`, `c7`):** Live output matches a fresh
  interpreter started from the edited files, and `uninstall()` restores base `Session` code
  fingerprints and base output bit-for-bit.
