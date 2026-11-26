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
2. **Real colocated Ray (`NemoRlLab`, `mode="ray"`):** Implemented in [ray_driver.py](../livelab/adapter/ray_driver.py) over `RayVirtualCluster`, `SGLangGeneration`, `Policy` (`AutomodelPolicyWorker` FSDP2), `SGLangColocatedWeightSynchronizer`, and `ClippedPGLossFn`, hot-swapping upstream `nemo_rl` modules (`clipped_pg`, `loss_functions`, `automodel.train`, `automodel.setup`) via `NemoRlRayParticipant` (`__ray_call__`) plus `snapshot_automodel_s0`/`restore_automodel_s0` and `invalidate_kv_cache()`. When deps are absent, `require_nemo_rl_deps()` fails closed.

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
| **Live step-gap swap** | `LocalParticipant` / `ray_participants.serve` `__code__` swap + AST witness + coupled `S0` reset | CPU `adapter.*`; GPU `nemo_rl.algorithms.loss.*`, `nemo_rl.models.automodel.train` | Verified (`c9` install `0.0073 s`, undo `0.0015 s`) | Verified `1×/2× B200` (`install 4.6–4.8 s`, bit-exact `S0` + `fresh_parity`) |
| **Refused at prepare** | Fail-closed static checks before mutating state | `patch_invalid`, `closure_changed`, `outside_changes`, `empty_change`, `constructor_edit`, `unloaded_module` | Verified (`c6` + golden tests) | Verified (`c6` on `2× B200`, pre-Gate 2 P1 fix) |
| **Cold respawn** | `reset("respawn")` + `ray.kill`/`ray.shutdown` cleanup | Worker crash, hang, lost participant, external tamper, or startup-only edits | Verified (`c8` `0.120–0.268 s`) | Verified (`c8` on `2× B200` `731.472 s`, respawns `92.0–124.9 s`) |
