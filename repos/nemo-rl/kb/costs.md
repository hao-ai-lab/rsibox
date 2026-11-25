# Measured Costs (`NVIDIA-NeMo/RL`)

## 1. CPU 3-Participant Conformance (`ToyNemoRlLab`, Pure Python Stdlib)

Measured locally on CPU via `PYTHONPATH=core/livelab python3 -m rsibox conformance nemo-rl`
(3 spawned worker subprocesses over `AF_UNIX` JSON channels: `coordinator0` + `trainer0` +
`rollout0`, Python 3.14.6, zero `torch`/`ray`/`nemo_rl`/`sglang` dependencies):

| Case | Result | Time (s) | Key Measurement |
|---|---:|---:|---|
| `c0_repeatable` | pass | `0.054` | 3 checks identical after coupled `S0` resets (`weights_s0`, `optimizer_state`, `rollout_weights`, `rollout_refit_version`, `kv_cache`, `rng`, `step_counter`) |
| `c1_noop` | pass | `0.055` | Witness `adapter.grpo_ops:compute_grpo_loss:48` = `[2]`, bit-exact output |
| `c2_visible` | pass | `0.045` | GRPO loss `[0.0111483, -0.01088132] -> [0.02229659, -0.02176264]`, witness `[2]` |
| `c3_fresh_equivalence` | pass | `0.283` | Visible & 3-edit chain bit-exact vs fresh `PackageOverlay` subprocess |
| `c4_uninstall` | pass | `0.047` | Uninstall in `0.001709 s`, exact base code fingerprint restored |
| `c5_failed_install` | pass | `0.067` | `fail_apply` & `fail_commit` rolled back cleanly; 3-target / 2-module control committed |
| `c6_refusal` | pass | `0.064` | All 4 refusal classes (`patch_invalid`, `closure_changed`, `outside_changes`, `empty_change`) |
| `c7_chain` | pass | `0.072` | 3 stacked edits (`grpo_ops` + `rollout_ops`) undone step-by-step back to base |
| `c8_crash_and_hang` | pass | `2.708` | Poisoned on `crash`/`hang`/`lost`/`tamper`; respawns `0.114–0.220 s` (`0.161 s` / `0.220 s` / `0.164 s` / `0.114 s`) |
| `c9_soak` | pass | `0.340` | 20 cycles: install median **`0.0085 s`**, uninstall median **`0.0016 s`** |
| `c10_not_proven` | pass | `0.113` | `proven=False` (`zero_witness`) on startup-only `init_policy_scale`; fresh diverges |
| **Total (`c0–c10`)** | **11/11** | **3.848 s** | **Sum of `c0–c10` case times; 9/9 unit tests pass in `6.500 s` (`-W error::ResourceWarning`)** |

## 2. Real GPU (`1×` / `2× NVIDIA B200`, Colocated AutoModel FSDP2 + SGLang GRPO)

- **Status:** **`pending hardware/env staging`** (zero fabricated GPU numbers; real `1×/2× B200`
  `NemoRlLab --real` conformance and cold-vs-live GRPO latency ladder will be recorded here once
  the `NVIDIA-NeMo/RL` + `ray` + `sglang` GPU environment is staged on B200).
