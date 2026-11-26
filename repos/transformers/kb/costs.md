# Measured Costs

All GPU numbers below were measured in our 2026-11-24 B200 run on 1× NVIDIA B200 (180 GiB HBM3e),
Python 3.12.11, PyTorch 2.14.0+cu130, Transformers 5.20.2 (`bfloat16`), 3 repeats per measurement
(mean ± std), using the fixed 2-prompt × 16-token greedy workload in [workload.md](workload.md).

## Cold Boot vs. Warm `TransformersLab` Live Edit

| Model | Cold Boot + 1st Check (3× mean ± std) | Cold Run 1 (Cold Page Cache) | Warm Steady Base Check (3×) | Live Edit (`_sample`: install + check) | Speedup vs. Cold Mean | Fresh Subprocess Parity Check |
|---|---|---|---|---|---|---|
| **Qwen3-0.6B** | `4.718 ± 0.039 s` (`3.966 s` load + `0.752 s` check) | `4.760 s` | `0.231 ± 0.002 s` | `0.521 s` (`0.281 s` install + `0.240 s` check) | **9.05×** | `6.301 s` (exact match) |
| **Qwen3-8B** | `7.746 ± 1.536 s` (`6.925 s` load + `0.820 s` check) | `9.508 s` (`8.681 s` load) | `0.304 ± 0.001 s` | `0.605 s` (`0.270 s` install + `0.335 s` check) | **12.81×** (`15.7×` vs. run 1) | `8.283 s` (exact match) |

## Per-Module Live Edit Breakdown (3× Repeats, `proven=True`, Bit-Exact Rollback)

| Model | Target Module & Function | Witness Count | Install (s) | Check (s) | Uninstall (s) | Full Edit+Check+Undo (s) |
|---|---|---|---|---|---|---|
| **Qwen3-0.6B** | `transformers.generation.utils` (`GenerationMixin._sample`) | `1` | `0.281 ± 0.017` | `0.240 ± 0.012` | `0.086 ± 0.008` | `0.608 ± 0.027` |
| **Qwen3-0.6B** | `transformers.generation.logits_process` (`LogitsProcessorList.__call__`) | `16` | `0.195 ± 0.036` | `0.233 ± 0.001` | `0.097 ± 0.008` | `0.526 ± 0.042` |
| **Qwen3-0.6B** | `transformers.models.qwen3.modeling_qwen3` (`Qwen3RMSNorm.forward`) | `1,808` | `0.113 ± 0.011` | `0.257 ± 0.022` | `0.072 ± 0.006` | `0.444 ± 0.039` |
| **Qwen3-8B** | `transformers.generation.utils` (`GenerationMixin._sample`) | `1` | `0.270 ± 0.038` | `0.335 ± 0.051` | `0.083 ± 0.016` | `0.689 ± 0.104` |
| **Qwen3-8B** | `transformers.generation.logits_process` (`LogitsProcessorList.__call__`) | `16` | `0.215 ± 0.043` | `0.317 ± 0.013` | `0.084 ± 0.006` | `0.617 ± 0.050` |
| **Qwen3-8B** | `transformers.models.qwen3.modeling_qwen3` (`Qwen3RMSNorm.forward`) | `2,320` | `0.119 ± 0.025` | `0.331 ± 0.006` | `0.091 ± 0.005` | `0.541 ± 0.028` |

## Snapshot Quiesce & Resume (`DynamoK8sSnapshotDriver` + `TransformersQuiesceHooks`)

Measured in our 2026-11-24 B200 run over 3 consecutive `checkpoint_and_wait()` cycles with
`offload_to_cpu=True, pin_memory=True`:

| Model | Warmup (s) | Quiesce Run 1 (Cold Pinning) | Quiesce Steady (Runs 2–3) | Resume (3× mean ± std) | VRAM Freed During Quiesce | Post-Resume Bit-Exact & Live Edit |
|---|---|---|---|---|---|---|
| **Qwen3-0.6B** | `0.040 ± 0.021` | `1.415 s` | `0.413 s`, `0.422 s` | **`0.032 ± 0.004 s`** | `1,164–1,166 MiB` | Yes (`0.270 s` install, `0.231 s` check) |
| **Qwen3-8B** | `0.048 ± 0.019` | `15.644 s` | `8.758 s`, `8.976 s` | **`0.310 ± 0.011 s`** | `15,740–15,778 MiB` | Yes (`0.287 s` install, `0.307 s` check) |

## Conformance (`c0–c10`) Receipts

Measured in our 2026-11-24 B200 run:

- **Real GPU (`TransformersLab`, 1× NVIDIA B200, `Qwen3-0.6B` `bfloat16` via `rsibox conformance transformers --real`)**: **11/11 PASS** (`c0–c10`), `c9_soak` (20 cycles) install median `0.2727 s` / uninstall median `0.0843 s`, `c8_crash_and_hang` respawns `4.26–4.50 s` across `crash`/`hang`/`lost`/`tamper`, `c3_fresh_equivalence` visible & 3-edit chain bit-exact (`22.820 s`), `c10_not_proven` `zero_witness` with fresh divergence via `PreTrainedModel.tie_weights` (`7.954 s`), clean `0 MiB` exit.
- **CPU (`ToyTransformersLab` via `rsibox conformance transformers`)**: **11/11 PASS** (`c0–c10`), total ~3.7 s on CPU (`c9_soak` install median `~0.007 s` / uninstall median `~0.002 s`, `c8` respawn `~0.05–0.17 s`).

## Multi-GPU `device_map="auto"` Sharded Quiesce, Live Edit & `SIGKILL` Fast-Fail (`2× NVIDIA B200`, `#280`)

Measured on `2× NVIDIA B200` (`gemma-3-1b-it`, `bfloat16`, `device_map="auto"`, `max_memory={"0": "1100MiB", "1": "1100MiB"}` placing `11` layers on `cuda:0` and `19` on `cuda:1`; `2` prompts, `max_new_tokens` `8` and `16`):

| Operation / Stage | Measured | Notes |
|---|---:|---|
| Cold boot + 1st check | **21.496 s** | `5.223 s` boot + `16.273 s` 1st check |
| Warm steady base check | **0.1656 s** / **0.8923 s** | Shape A (`bs1`, `8` tok) / Shape B (`bs3`, `16` tok) |
| `TransformersQuiesceHooks` quiesce (`offload_to_cpu=True, pin_memory=True`) | **2.685 s** (`0.173 s` warmup) | Freed **`1,050 MiB` / `892 MiB`** (`1,894 -> 844` & `1,734 -> 842 MiB`); generate blocked while quiesced |
| `TransformersQuiesceHooks` resume | **0.160 s** (`0.153 s` reported) | Restored both GPUs (`1,900` / `1,750 MiB`) with `post_resume_parity=True` |
| Post-resume live edit (`GenerationMixin._sample`) | **`0.293 s`** install + **`0.193 s`** check | Proven (`1` / `2` witness), bit-exact vs. fresh (`22.479 s` / `14.203 s`), `0.116 s` bit-exact undo |
| Unassisted `SIGKILL` fast-fail | **0.153 s** (`0.1533 s`) | Immediate `ConnectionError` + `dirty=True`, clean respawn & post-respawn parity |
