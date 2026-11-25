# NVIDIA NeMo-RL Live-Lab Architecture

## Scope & Three-Participant GRPO Topology

`NemoRlLab` targets `NVIDIA-NeMo/RL` GRPO post-training where an AutoModel FSDP2 policy trainer
(`fully_shard` + `AdamW`) and colocated `SGLang` rollout generation actors are orchestrated by a
GRPO training driver with weight refits between steps:

```text
Host / Driver (NemoRlLab, BaseLab + RecaptureCoordinator + pluggable weight_store/recovery_driver)
  ├── coordinator0  (GRPO batch/advantage coordinator; loads adapter.shared_ops)
  ├── trainer0      (AutoModel FSDP2 policy trainer; loads adapter.grpo_ops + adapter.shared_ops)
  │     └── Owns: policy weights S0, AdamW moments (exp_avg/exp_avg_sq/adam_step), RNG, step_counter
  └── rollout0      (Colocated SGLang rollout worker; loads adapter.rollout_ops + adapter.shared_ops)
        └── Owns: rollout weights S0, refit_version, KV/Radix cache, RNG, step_counter
```

Two runtime modes share the same `NemoRlLab` (`BaseLab`) contract:

1. **CPU 3-participant toy (`ToyNemoRlLab`, `mode="cpu"`):** Spawns 3 isolated worker subprocesses
   (`coordinator0`, `trainer0`, `rollout0`) over `AF_UNIX` JSON channels (`rsibox.ipc.UnixJsonServer`).
   Each worker wraps its actor in a `@_fake_remote` / `by_value` class copy and serves
   `core/livelab/ray_participants.py` (`serve`, `actor_view`, `file_namespace`, `unloaded`),
   running a full GRPO episode (`rollout0.generate -> coordinator0.prepare_batch ->
   trainer0.train_step -> rollout0.refit_weights`) per `check()`.
2. **Real colocated Ray (`NemoRlLab`, `mode="ray"`):** Reuses `core/livelab/ray_participants.py`
   via `NemoRlRayParticipant` and `serve_nemo_rl_actor` over `handle.__ray_call__.remote(...)` for
   NeMo-RL AutoModel FSDP2 policy workers and SGLang generation workers, plus `snapshot_automodel_s0` /
   `restore_automodel_s0` / `coupled_nemo_rl_reset` (`DTensor.to_local()` + `AdamW` + per-device
   CUDA RNG + `refit_policy_generation` + `flush_cache`). When `nemo_rl`, `ray`, `torch`, or
   `sglang` are absent, `require_nemo_rl_deps()` fails closed cleanly with an actionable error.

## Step-Gap Barrier, Abstain Semantics, and Coupled `S0` Reset

1. **Role-scoped module loading & fail-closed `unloaded` check:** `build_nemo_rl_plan()` calls
   `ray_participants.unloaded(participants, plan)` before `Session.install()` so editing a module
   or `.py` file loaded by zero participants refuses with `unloaded_module` instead of committing
   an all-abstaining no-op. When an edit targets a single role (e.g., `adapter.grpo_ops` on
   `trainer0` or `adapter.rollout_ops` on `rollout0`), uninvolved participants abstain cleanly in
   Phase 1 (`abstain=True`) and `clean_nemo_rl_witness()` filters dead `@ray.remote` keys and `{}`.
2. **Coupled trainer + rollout `S0` reset:** `reset("state")` (and `post_commit_reset="state"` /
   `undo_resets=["state"]`) restores `trainer0` `S0` weights, `AdamW` state (`exp_avg`,
   `exp_avg_sq`, `adam_step`), `step_counter`, and RNG **together with** `rollout0` `S0` weights,
   `refit_version = 0`, `kv_cache` clear, and RNG, plus `coordinator0` `step_counter` and RNG.
3. **Zero eager GPU/Ray imports in the host:** Importing `adapter`, `adapter.lab`,
   `adapter.worker`, `adapter.grpo_ops`, `adapter.rollout_ops`, or `adapter.shared_ops` uses only
   Python stdlib + `core/livelab`. `rsibox doctor --engine nemo-rl` verifies zero eager GPU/Ray
   imports.

## Restart Boundaries

| Path | Mechanism | Scope | CPU Status | Real GPU Status |
|---|---|---|---|---|
| **Live step-gap swap** | `LocalParticipant` / `ray_participants.serve` `__code__` swap + AST witness + coupled `S0` reset | `adapter.grpo_ops`, `adapter.rollout_ops`, `adapter.shared_ops`, `@ray.remote` methods (`#actor`), `sys.modules`-less `.py` files (`file_namespace`) | Verified (`c9` install `0.0085 s`, undo `0.0016 s`) | `pending hardware/env staging` |
| **Refused at prepare** | Fail-closed static checks before mutating state | `patch_invalid`, `closure_changed`, `outside_changes`, `empty_change`, `constructor_edit`, `unloaded_module` | Verified (`c6` + golden tests) | `pending hardware/env staging` |
| **Cold respawn** | `reset("respawn")` (or pluggable `recovery_driver` / `weight_store`) | Worker crash, hang, lost participant, external tamper, or constructor/topology edits | Verified (`c8` `0.114–0.220 s`) | `pending hardware/env staging` |
