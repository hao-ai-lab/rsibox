# Measured Costs (`NVIDIA-NeMo/RL`)

## 1. CPU 3-Participant Conformance (`ToyNemoRlLab`, Pure Python Stdlib)

Measured locally on CPU via `PYTHONPATH=core/livelab python3 -m rsibox conformance nemo-rl`
(3 spawned worker subprocesses over `AF_UNIX` JSON channels: `coordinator0` + `trainer0` +
`rollout0`, Python 3.14.6, zero `torch`/`ray`/`nemo_rl`/`sglang` dependencies):

| Case | Result | Time (s) | Key Measurement |
|---|---:|---:|---|
| `c0_repeatable` | pass | `0.039` | 3 checks identical after coupled `S0` resets (`weights_s0`, `optimizer_state`, `rollout_weights`, `rollout_refit_version`, `kv_cache`, `rng`, `step_counter`) |
| `c1_noop` | pass | `0.048` | Witness `adapter.grpo_ops:compute_grpo_loss:48` = `[2]`, bit-exact output |
| `c2_visible` | pass | `0.059` | GRPO loss `[0.0111483, -0.01088132] -> [0.02229659, -0.02176264]`, witness `[2]` |
| `c3_fresh_equivalence` | pass | `0.292` | Visible & 3-edit chain bit-exact vs fresh `PackageOverlay` subprocess |
| `c4_uninstall` | pass | `0.050` | Uninstall in `0.001515 s`, exact base code fingerprint restored |
| `c5_failed_install` | pass | `0.069` | `fail_apply` & `fail_commit` rolled back cleanly; 3-target / 2-module control committed |
| `c6_refusal` | pass | `0.067` | All 4 refusal classes (`patch_invalid`, `closure_changed`, `outside_changes`, `empty_change`) |
| `c7_chain` | pass | `0.077` | 3 stacked edits (`grpo_ops` + `rollout_ops`) undone step-by-step back to base |
| `c8_crash_and_hang` | pass | `2.835` | Poisoned on `crash`/`hang`/`lost`/`tamper`; respawns `0.120–0.268 s` (`0.195 s` / `0.268 s` / `0.196 s` / `0.120 s`) |
| `c9_soak` | pass | `0.287` | 20 cycles: install median **`0.0073 s`**, uninstall median **`0.0015 s`** |
| `c10_not_proven` | pass | `0.119` | `proven=False` (`zero_witness`) on startup-only `init_policy_scale`; fresh diverges |
| **Total (`c0–c10`)** | **11/11** | **3.942 s** | **Sum of `c0–c10` case times; 14/14 unit tests pass (`-W error::ResourceWarning`)** |

## 2. Real GPU (`1×` / `2× NVIDIA B200`, Colocated AutoModel FSDP2 + SGLang GRPO)

Measured inside `rsib/slime_grpo:b200-bw` on `NVIDIA B200` (`gemma-3-1b-it`, `bf16`, deterministic `sdpa`, SGLang `tp_size = num_trainers`):

| Shape / Suite | GPUs | Cold (`boot + check`) | Warm `check` | Reset / Install / Fresh | Notes (`witness_nonzero=true`, `S0` bit-exact) |
|---|---:|---:|---:|---:|---|
| `g2_s2_mnt8_gpu1` (`bs=4, steps=2, mnt=8`) | `1× B200` | `161.934 s` (`121.165 + 40.768`) | `20.170 s` | `4.331 / 4.825 / 228.752 s` | Post-final-edit; `trainer0` witness `4`, `fresh_parity=true`, `66411 MiB` |
| `g2_s2_mnt8_gpu2` (`bs=4, steps=2, mnt=8`) | `2× B200` | `186.756 s` (`155.594 + 31.161`) | `18.985 s` | `4.856 / 4.627 / 252.776 s` | Post-final-edit; `trainer0/1` witness `[2, 2]`, `fresh_parity=true`, `68589 MiB/GPU` |
| `g4_s3_mnt16_gpu2` (`bs=8, steps=3, mnt=16`) | `2× B200` | `151.530 s` (`104.373 + 47.157`) | `34.664 s` | `3.641 / 4.721 / 251.258 s` | Post-final-edit; `trainer0/1` witness `[6, 6]`, `fresh_parity=true`, `68589 MiB/GPU` |
| `c8_crash_and_hang` (`rsibox conformance --real`) | `2× B200` | — | — | Respawns `91.960–124.851 s` | Post-final-edit `1/1 pass` (`731.472 s`: `crash`, `hang`, `lost`, `tamper`) |
| `c0–c7, c9–c10` (`rsibox conformance --real`) | `2× B200` | — | — | `c9` install `3.840 s`, undo `3.784 s` | `10/11 pass` immediately prior to Gate 2 P1 `fresh_check`/`inject(hang)` fix |
