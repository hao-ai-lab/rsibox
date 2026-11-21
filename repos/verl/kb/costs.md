# Costs (`verl` GRPO on 4x GB200)

Two measured batches, both on one node of **4x NVIDIA GB200 (`aarch64`, SM100, CUDA 13.0)** with `Qwen3-0.6B`
GRPO on GSM8K (`PPOTrainerSync`, 4 colocated vLLM TP1 replicas, 8 `RewardLoopWorker` actors, 640 responses per
step: 64 prompts x `rollout.n=10` or 128 x 5):

- **Batch A (current, 2026-11-19):** `volcengine/verl` `cd58f550` (`0.10.0.dev`), FSDP trainer (`sdpa`, gradient
  checkpointing), vLLM 0.31 with `rollout.full_determinism=True` and therefore `enforce_eager=True` (its
  `torch.compile` autotune refuses deterministic mode), torch 2.13, Ray 2.59. Receipts: the gate's
  `resp-<n>.json` files of our 2026-11-19 run (`verdict_s`, `install_s`, `witness`), `gate.json` (`s0_capture_s`),
  the launcher's timestamps, and `tools.rescore` / `tools.replay_adv` output.
- **Batch B (2026-10-21):** `975f063b`, Megatron-Core `TP1/PP1/DP4` with Transformer Engine and flash-attn, CUDA
  graphs on. Receipts: `VERL_STARTUP_TRACE` phase stamps and the gate of that lab generation.

The two differ in trainer backend and in eager vs. CUDA-graph rollout, so a step costs 102-117 s in A and 25-27 s in
B; read ratios within one batch only. The site's facts (`site/facts/engines.json`) quote batch A.

## 0. Batch A: One Unified Receipt Set (2026-11-19, FSDP, eager vLLM)

| Measurement | Value | Source |
|---|---:|---|
| Launch -> gate ready (imports, `ray.init`, dataloader, FSDP actor+ref init, 4x vLLM boot, first weight sync) | **121.9 s** | `gate.json:started` minus launch stamp |
| of which 4x vLLM eager boot | 60.7 s | `RLPROBE` stamps 58.4 -> 119.1 s |
| First RL step (agent-loop spawn, 640 responses, first update) | 137.9 s | `resp-1.json` `verdict_s` |
| **Restart -> first step verdict** | **259.8 s** | the two above |
| Steady RL step (steps 2-8, eager FSDP) | 100.3-116.7 s | `verdict_s` of `check` requests (gen 58-66 s, old_log_prob 7.6-9.7 s, ref 5.9-8.5 s, adv 0.25 s, update_actor 23-27 s, update_weights 4.2-4.9 s) |
| Live reward swap (`compute_score` on disk -> 13 processes, witnessed 640 calls on 8 reward actors) | install **5.8 s**, verdict **117.2 s** | `resp-6.json` |
| Live advantage swap (`core_algos.compute_grpo_outcome_advantage`, witnessed on the driver) | install **0.58 s**, verdict **104.6 s** | `resp-9.json` |
| Live `--set actor.ppo_micro_batch_size_per_gpu` 16 -> 8 -> 16 | applied in the gap, verdict 100.3-107.3 s | `resp-11/12.json` (`set: [16, 8]`, `[8, 16]`); `=7` refused: does not divide the per-GPU batch of 160 |
| Coupled `S0` capture (4 ranks, 31 FSDP flat parameters each, optimizer, RNG, dataloader) | 0.24 s | `gate.json:s0_capture_s` |
| **Coupled `S0` restore** (4 ranks + `sleep_replicas` + `update_weights`) | **1.10-1.16 s** | `resp-2/4.json` |
| `uninstall` (undo on 13 processes) | 0.056-0.058 s | `resp-7/10.json` |
| `status` (fingerprint on 13 processes) | 0.031-0.041 s | `resp-0/14.json` |
| Inline-source install (`core_algos` edit) / `uninstall` on 21 participants, before / after the fingerprint fix | 21-26 s / **6.8-7.0 s** and 24-27 s / **5.7-6.0 s** | `install_s`, `cost_s` of the 2026-11-19 conformance jobs |
| `status` with 21 participants, before / after the fingerprint fix (`core/livelab`: realpath memo, one round) | 23-25 s / **2.5-5.6 s** | `verdict_s` of the 2026-11-19 conformance jobs |
| One fingerprint round over 21 participants, before the fix: driver 0.77 s, reward 0.21-0.25 s each, agent-loop 0.49-0.53 s each, trainer 0.72-0.87 s each | 9.8 s | `fingerprint_s` in a `status` reply (`timing: true`) |
| Offline `tools.rescore` on 640 recorded responses | **0.83-0.86 ms** | `score_ms` |
| Offline `tools.replay_adv` on 640 recorded trajectories | **16-29 ms** | `compute_ms` (estimator only; import and `torch.load` are outside) |

With the per-sample seeds (`FASTRSI_RL_SEED_PER_SAMPLE=1`, 32 prompts x 8 samples), five checks after coupled `S0`
resets gave the same `critic/score/mean 0.12890625`, `critic/advantages/max 2.4748666`, `actor/pg_loss 0.0084084`
and the same rollout digest every time; a live `x2.0` advantage edit doubled `advantages/max` to `4.9497333` and
`pg_loss` to `0.0168168` on the very same rollouts; a reward `x2` doubled `score/mean` to `0.2578125`. The job had
21 warm participants (driver, 8 reward, 8 agent-loop, 4 trainer ranks); boot to gate ready 126.7-127 s.

So in batch A a live verdict costs one step (plus 0.6-5.8 s of install): **2.2-2.5x** cheaper than a restart to
the first verdict, and the offline probes are 5-6 orders of magnitude cheaper than either. The restart floor
here is eager vLLM; with CUDA graphs (batch B) the step is 4x shorter and the ratio grows to 6-7x.

### 0.1 Batch A: the eleven conformance cases on the live job

`VerlConformanceLab` (c0-c10, `core/livelab/conformance/`) against one gated job with 21 participants (driver,
8 reward, 8 agent-loop, 4 trainer ranks), `train_batch_size=32`, `rollout.n=8` (256 responses per step, a step of
~45 s), per-sample seeds on. Source: the pass table and `receipts.jsonl` of our 2026-11-19 run (11/11).

| Case | Seconds | What the time is made of |
|---|---:|---|
| c0_repeatable | 188 | 3 steps and 2 coupled `S0` resets; all three gave the same metrics and rollout digest |
| c1_noop, c2_visible | 127, 119 | 2 steps each plus one inline `core_algos` install (6.9-8.1 s) and uninstall (5.9-6.5 s) |
| c3_fresh_equivalence | 1072 | 5 fresh boots of the job from edited files (about 2 min each) plus their first steps; live == fresh for the visible edit and for a chain of 3 |
| c4_uninstall | 209 | uninstall 5.96 s; the rest is steps |
| c5_failed_install, c6_refusal | 255, 297 | fault injections and refused installs; each refusal is cheap (< 1 s), the base checks between them are steps |
| c7_chain | 421 | 3 stacked installs, 4 steps |
| c8_crash_and_hang | 1295 | 4 poisonings, each followed by a respawn (relaunch) of 112-162 s and a base step |
| c9_soak | 2349 | 20 install/uninstall cycles with a step each; install median 8.09 s, uninstall median 6.52 s |
| c10_not_proven | 511 | one unreached edit (`proven False`, 8 `zero_witness` findings), one fresh boot |
| **Total** | **6844** | 1 h 54 min; 12 job launches in all |

Only the driver computes GRPO advantages, so an edit of `compute_grpo_outcome_advantage` is witnessed with 1 call
on the driver; the agent-loop actors and trainer ranks import `core_algos` (the swap reaches them, 0 calls) and the
reward actors do not (they abstain). The reward edit of c3's chain is witnessed with 256 calls on the 8 reward actors.
An inline edit of `SingleTurnAgentLoop.run` (the agent-loop module, loaded only by the 8 agent-loop actors) installs
in 7.5 s and is witnessed with 32 calls on each of the 8 actors (one per sample), with the step's metrics and
rollouts unchanged by the no-op (our 2026-11-19 probe on the same job configuration).

## 1. Batch B: Verdict Cost Ladder (2026-10-21, Megatron-Core, CUDA graphs)

| Verdict Path | Total Cost | Breakdown | Speedup vs. Fast-Storage Restart |
|---|---:|---|---:|
| **Offline reward probe (`tools/rescore.py`)** | **0.5 ms** | CPU re-score of 640 recorded rollouts (`<step>.jsonl`) | **~374,000x** |
| **Offline advantage probe (`tools/replay_adv.py`)** | **12 ms** | CPU recompute of 640 recorded trajectories (`adv.pt`) | **~15,500x** |
| **Live step-gap swap (advantage edit, fast FS)** | **28.0 s** | `2.6 s` install across Ray actors + `25.4 s` RL step | **6.7x** |
| **Live step-gap swap (reward edit, fast FS)** | **30.9 s** | `5.3 s` install across 13 processes + `25.6 s` RL step | **6.1x** |
| **Live step-gap swap (NFS-backed venv)** | **36.2-39.1 s** | `9.7-12.5 s` install + `25.2-26.6 s` RL step | **26-28x** (vs. `1,022 s` NFS restart) |
| **Fresh launch, warm caches (fast parallel FS)** | **187-204 s** | `137 s` init + `25 s` first-step extras + `25 s` step | **1.0x** (baseline) |
| **Fresh launch, cold compile caches (fast FS)** | **370 s** | Dominated by `235.6 s` cold vLLM CUDA-graph/FlashInfer JIT | 0.5x |
| **Fresh launch, warm caches (shared NFS venv)** | **1,022 s** | Python imports + 7.5k prompt tokenization across ~20 procs | 0.18x |

With FSDP (`sdpa`, no sequence packing) on the same 4x GB200 node: steady step is `52.8 s`
(`update_actor` `26.2 s`), live verdict is `57-66 s`, warm-cache restart is `184-201 s`, and
cold-cache restart is `362 s`.

## 2. Batch B: Startup Phase Anatomy (`VERL_STARTUP_TRACE`, Seconds)

| Phase | Parallel FS (cold) | Parallel FS (warm, 3 runs) | Node-Local Copy | Shared NFS (warm) | Shared NFS (cold) |
|---|---:|---:|---:|---:|---:|
| Interpreter + imports | 41.6 | 9.1-13.5 | 6.4 | 20.2 | 41.7 |
| `ray.init` + `TaskRunner` spawn | 18.6 | 12.5-13.8 | 7.7 | 20.6 | 27.6 |
| Tokenizer + dataloader (7.5k prompts) | 34.6 | 23.9-24.3 | 20.1 | 251.4 | 90.4 |
| Trainer init (Megatron/TE, actor + ref) | 44.8 | 28.1-29.1 | 22.0 | 178.2 | 141.1 |
| Rollout engine boot (4x vLLM replicas) | 235.6 | 57.4-68.5 | 104.6* | 242.4 | 343.5 |
| First weight sync + managers | 2.5-3.0 | 2.4-2.9 | 1.4 | 5.6 | 5.9 |
| First training step (incl. lazy spawn) | 51.2-55.1 | 50.0-53.9 | 45.3 | 300.4 | 462.2 |
| **Launch -> first step done** | **369.6** | **187-204.2** | **207.5*** | **1,022.2** | **1,112.8** |
| Steady training step (steps 2+) | 26.8 | 24.5-26.8 | 24.5 | 25.1 | 24.9 |

*\*Copying the venv to a new node-local path invalidated path-keyed `torch.compile` and Triton
caches once (`57.4 s -> 104.6 s`); with path-stable caches, node-local startup is ~`160 s`.*

## 3. Batch B: Steady GRPO Step Breakdown (Megatron-Core, Mean of Steps 2-4)

| `gen` (rollout + reward) | `old_log_prob` | `ref` log-prob | `adv` (`core_algos`) | `update_actor` | `update_weights` | Total Step |
|---:|---:|---:|---:|---:|---:|---:|
| 6.0 s | 4.9 s | 6.0 s | 0.3 s | 8.1 s | 1.4 s | **26.8 s** |

## 4. Static Coverage Survey (`fastrsi-survey/1` on `volcengine/verl@975f063b`)

Command:
```sh
cd core/livelab && python3 -m survey scan \
  --rules ../../repos/verl/kb/survey_rules.json \
  --source <verl-upstream-tree> --repo ../.. --summary
```

| Metric | Count / Share | Notes |
|---|---:|---|
| **Total Python functions** (excl. tests) | **3,571** | `83,083` non-comment lines; `0` native C++/CUDA files |
| **Live-covered functions (`swap` + `jit_rebuild`)** | **2,564 (71.8%)** | `53,212` lines (`64.0%`): `2,553 / 2,731` `swap` + `11 / 11` `@triton.jit` |
| **Uncovered `rebuild` (`STATE` + `RESOURCE`)** | **744 (20.8%)** | `15,320` lines (constructors, config builders, checkpoint loaders) |
| **Uncovered `respawn` (`RANKS` / module-level)** | **54 (1.5%)** | `11,236` lines (Ray placement groups, distributed init, module globals) |
| **Restart-only (`main_*`, launch flags)** | **31 (0.9%)** | `609` lines |
| **Firm (syntax-proved) vs. heuristic** | `276` (`7.7%`) firm | `3,295` (`92.3%`) heuristic (`default_op`, `path`, `name_*`) |

### 0.2 Batch A re-verification after adapter refactor (4x GB200, 2026-11-19)

Post-refactor `VerlConformanceLab` (`fastrsi conformance verl --real`, 21 participants: driver, 8 reward, 8 agent-loop, 4 FSDP trainer ranks, `train_batch_size=32`, `rollout.n=8`, per-sample seeds) re-ran `11/11` cases in **6879.773 s** total case time with **113** receipts (`43` txn, `38` proof, `32` undo; median `edit_to_commit_s` **1.216 s**, median `undo_s` **44.77 ms**, `c9_soak` install median **8.1093 s**, uninstall median **6.6840 s**, `0` `ResourceWarning`s, immediate `@atexit` `0 MiB × 4`).

### 0.3 Batch B CUDA-graph re-verification (`Megatron-Core` and `FSDP`, `4× GB200`, `2026-11-19`)

Running the same `Qwen3-0.6B` GSM8K `640`-response (`TBS=64, N=10`, `21` warm participants) recipe with `rollout.enforce_eager=False` and `rollout.full_determinism=False` confirms CUDA-graph rollout speedups on the current `verl` stack:

| Backend (`enforce_eager=False`, `full_determinism=False`) | Boot to ready / Step 1 | Steady live check (`verdict_s`) | Breakdown & Conformance notes |
|---|---:|---:|---|
| **`Megatron-Core` (`mcore`)** | `287.489 s` (cold CUDA-graph/Inductor compile) / `35.828 s` step 1 | **`24.293–24.397 s`** (`n=4` checks) | `gen 4.18 s`, `old_log_prob 4.21–4.37 s`, `ref 5.83–6.03 s`, `adv 0.23–0.25 s`, `update_actor 7.73–7.86 s`, `update_weights 1.36–1.40 s`; `install` `9.71–9.81 s` (`34.1 s` total live check, `9.5×` vs `323.3 s` cold start), `uninstall 8.68 s`, `S0` capture `1.194 s` (`226` tensors/rank), reset `2.26–2.50 s` (`1.035 s` initial); **`c2_visible` pass** (`adv_max 2.8460 -> 5.6921` exact 2×, `pg_loss 0.022 -> 0.0433`, witness `[1]`), `c4_uninstall` restored exact `adv_max/adv_min` with non-deterministic CUDA-graph rollout digest variance |
| **`FSDP + CUDA graphs` (`fsdp_cg`)** | `49.591 s` step 1 | **`44.939 s`** (`gen 4.173 s`) | `update_actor 22.995 s`, `install 8.351 s`, `reset 1.103 s`; **`c2_visible` pass** (`adv_max 2.8460 -> 5.6921`); hit transient `torch.save` `inline_container.cc` write failure on `req-12`; `0 MiB × 4` teardown |

### 0.4 Wave 10 Lustre atomic-write fix & MCore determinism audit (`4× GB200`, `2026-11-19`)

- **`fsdp_cg` 5-cycle Lustre fix + offline `replay-adv` verification:** After replacing direct `torch.save` with `_atomic_torch_save` (`io.BytesIO()` + sibling `.tmp` + `os.replace` in `adapter/gate.py`), the `fsdp_cg` multi-cycle driver (`boot_s 133.066 s`, `reset 0.687–1.187 s`) completed all `5` live steps without Lustre `ZipFileWriter` errors: base `adv_max 2.846041`, `c1` 2× advantage edit `adv_max 5.692081 -> 2.846041` on undo, `c2` reward edit `score_mean 0.1500 -> 0.30625` (`adv_max 2.846045 -> 2.846041` on undo), writing all `5` `steps/{1..5}/adv.pt` artifacts. Offline `python3 -m tools.cli replay-adv --step steps/5` exited `rc=0` with `rows: 640`, `equal: true`, `max_abs_diff: 0.0`, `compute_ms: 17.0`.
- **`mcore_conf` determinism audit (`VerlConformanceLab --only c0_repeatable,c2_visible,c4_uninstall,c5_failed_install,c6_refusal`, `deterministic_mode=True`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `0 MiB × 4` exit):**
  - Fixing tensor `"step"` zeroing + `opt.reload_model_params()` in `adapter/state.py` made all 4 trainer rank digests bit-identical across every `reset` (`weights: 78de8177237190d3` across all ranks and `optimizer: ['90dd82166744e5ed', '8eb47d6dac7f65e8', 'abcb0fe9a1d43152', '792128859a6d8659']`), while `GateMixin._flush()` rollout backfill eliminated step-1 `rollouts: None` and produced an identical `'42169d45c2a3678c'` rollout digest across all checks.
  - Across every base check in `c0`, `c2`, `c4`, `c5`, and `c6`, `score_mean` (`0.12890625`), `adv_max` (`2.4748666`, doubled to `4.9497333` in `c2_visible` which **passed** with `[1]` driver witness), and `adv_min` (`-1.6201816`, doubled to `-3.2403631`) matched bit-for-bit; however, `verl`'s Megatron-Core backward pass still exhibits `~1e-4` stochastic variance in `actor/pg_loss` (`0.00833`, `0.008347`, `0.008397`, `0.008443`, `0.008449`, `0.008374`, `0.008346`). We intentionally keep strict `ROUND = {"pg_loss": 6}` in `adapter/conformance_lab.py` for FSDP conformance rather than relaxing tolerance (`1/5` pass on `mcore_conf`: `c2_visible` pass; `c0, c4, c5, c6` fail solely on `actor/pg_loss` at the 4th–5th decimal place).

### 0.5 Full `fsdp_cg` `11/11` Conformance (`enforce_eager=False`) & `mcore_det_v2` `unfused` Probe (`4× GB200`)

- **Full `11/11` `FSDP + CUDA graphs` (`enforce_eager=False`, `full_determinism=False`, `VLLM_BATCH_INVARIANT=1`, `FASTRSI_RL_SEED_PER_SAMPLE=1`, `21` participants, `TBS=32`, `N=8`, `0 MiB × 4` exit):**
  - Passed **`11/11`** in **`5,792.1 s`** total wall (`5,704.0 s` 11-case sum; down from `6,844–6,880 s` eager, with steady `gen 4.18 s` and `verdict_s ~26.5 s`), writing **`113` receipts** (`43` txn, `38` proof, `32` undo; median `edit_to_commit_s 1.2231 s`, median `undo_s 0.045 s`):
  - `c0_repeatable` (`136.780 s`, bit-exact `score_mean 0.109375`, `adv_max 2.4748666286468506`, `pg_loss 0.016291`, rollout `0b1dafd8ad4dbad6`), `c1_noop` (`89.634 s`), `c2_visible` (`82.243 s`, `pg_loss 0.016291 -> 0.032583`, witness `[1]`).
  - `c3_fresh_equivalence` (`1,102.893 s`, bit-exact `live == fresh` on visible & 3-chain), `c4_uninstall` (`151.478 s`, `7.227662 s` uninstall), `c5_failed_install` (`193.763 s`), `c6_refusal` (`234.141 s`), `c7_chain` (`300.341 s`).
  - `c8_crash_and_hang` (`1,280.220 s`, respawns `193.209 / 148.702 / 134.234 / 130.630 s`), `c9_soak` (`1,608.076 s`, 20 cycles: install median `9.3050 s`, uninstall median `7.6934 s`), `c10_not_proven` (`524.432 s`, `8 × zero_witness` vs fresh divergence).
- **`mcore_det_v2` (`++actor...attention_backend=unfused`, `+...deterministic_mode=True`, `use_remove_padding=True`, against consolidated `repo`, `0 MiB × 4` exit):**
  - Passed `c2_visible` (`119.298 s`, `adv_max 2.4748666 -> 4.9497333` exact `2×`, witness `[1]`, identical 4-rank `S0` digests `78de8177237190d3` and rollouts `'42169d45c2a3678c'`).
  - `c0_repeatable` (`193.885 s`) and `c4_uninstall` (`161.965 s`) still differed at the 4th–5th decimal in `actor/pg_loss` (`0.008483`, `0.008458`, `0.008522`, `0.008432`, `0.008532`) and `actor/entropy` (`0.46614–0.46620`, while `ref.megatron.override_transformer_config.attention_backend` remained `'flash'`), so strict `ROUND = {"pg_loss": 6}` stays unchanged for FSDP conformance.

### 0.6 Unified cold-vs-warm `fsdp_cg` restart-overhead dissection (`4× GB200`, `2026-11-20`)

Source: paired `4× GB200` (`Qwen3-0.6B` GSM8K GRPO, `train_batch_size=32`, `rollout.n=8`, `256` responses/step, `enforce_eager=False`, `full_determinism=False`, `VLLM_BATCH_INVARIANT=1`, `FASTRSI_RL_SEED_PER_SAMPLE=1`, `21` gate participants, `0 MiB × 4` teardown) phase dissection across `fresh_cold` (empty compile caches, `253` subprocesses), `fresh_warm` (`251` subprocesses), live `core_algos` swap + coupled `S0` reset, and offline `tools.cli replay-adv` (`not_split = 0.000 s`).

- **Pre-Step-1 lazy actor spawn (`24.426 s` cold / `23.868 s` warm):** Between `trainer_fit_begin` (`244.636 s` cold / `128.004 s` warm) and `on_step_begin` (`269.065 s` / `151.872 s`), `8 AgentLoopWorkerTQ` actors spawn, import `verl`/`torch`, and load `RLHFDataset`, plus `GateMixin._ensure_session()` synchronizes the 21 participants; `verl`'s `marked_timer("step")` wraps this window inside `timing_s/step` (`53.26 s` cold / `52.30 s` warm). Inside boot, `gate.on_init_end()` (`S0` capture + file snapshot) accounts for `1.169 s` cold (`0.576 s` warm), so cold restart excluding gate `S0` init is `296.780 s` (`297.949 s` raw wall).
- **Cold compile tax inside 4× vLLM `EngineCore` (`89.161 s` `jit_build`):** `profile_run` `torch.compile` drops `50.668 -> 11.945 s` (`-38.724 s`), piecewise `capture_model_1` drops `42.912 -> 4.150 s` (`-38.762 s`), and `compile_or_warm_up_model` drops `17.865 -> 6.190 s` (`-11.675 s`), while warm `FULL_AND_PIECEWISE` graph capture remains `7.708 s` (`4.150 s` pass 1 + `3.558 s` pass 2). Step 1 still pays `3.321 s` (`3.186 s` warm) of Triton `_topk_log_softmax_kernel` JIT/warmup over steady `25.566 s` (`25.265 s` warm) steps.

| Arm / Rung | Total (`s`) | Speedup vs Cold | Critical-Path Phase Breakdown (`s`) |
|---|---:|---:|---|
| 1. `fresh_cold` (empty compile caches) | **`297.949 s`** | `1.0×` | `import_init 32.939` + `dataloader_init 8.191` + `trainer_init 23.818` (`worker_spawn 1.010`, `fsdp_init 19.147`) + `weight_load 2.374` + `jit_build 89.161` + `cuda_graph_capture 7.708` + `engine_init 80.446` + `first_step 27.747` (`24.426` pre-step1 + `3.321` Triton JIT) + `run_workload 25.566` |
| 2. `fresh_warm` (warm compile caches) | **`180.324 s`** | `1.7×` | `import_init 32.028` + `dataloader_init 7.849` + `trainer_init 23.479` + `weight_load 4.537` + `cuda_graph_capture 7.708` + `engine_init 52.404` + `first_step 27.054` + `run_workload 25.265` |
| 3. Derived peel `import_init + dataloader_init + trainer_init` | `116.968 s` | `2.5×` | Row 2 minus `import_init` (`32.028 s`), `dataloader_init` (`7.849 s`), and `trainer_init` (`23.479 s`) |
| 4. Derived peel `weight_load + cuda_graph_capture` | `104.723 s` | `2.8×` | Row 3 minus vLLM `weight_load` (`4.537 s`) and `cuda_graph_capture` (`7.708 s`) |
| 5. Derived peel vLLM `engine_init` | `52.319 s` | `5.7×` | Row 4 minus `engine_init` (`52.404 s`) |
| 6. Live `core_algos` swap + coupled `S0` reset | **`36.428 s`** | **`8.2×`** | `live_install 8.606` (21 participants, witness `driver:1`; `uninstall 6.149 s`, `undo 0.0444 s` bit-exact) + `restore_state 1.192` + `run_workload 26.630` (`35.236 s` sans reset; `adv_max 2.474867 -> 4.949733` exact `2×`, `pg_loss 0.016291 -> 0.032583`) |
| 7. Offline `tools.cli replay-adv` (`256` rows) | **`0.0126 s`** (`12.57 ms`) | **`23,647×`** | Estimator-only recompute on recorded `steps/2/adv.pt` (`max_abs_diff 0.0`) |

### 0.7 Parallel multi-actor `_fanout` on 21-participant `fsdp_cg` (`4× GB200`, `2026-11-20`)

With `core/livelab` parallel `_fanout` across `prepare`, `validate`, `abort`, `proof`, `witness_check`, `at_base`, `Session._fingerprint()`, and `ray_participants.unloaded()` (while keeping `commit` and `undo` serial) verified on the same `4× GB200` `Qwen3-0.6B` `fsdp_cg` recipe (`21` participants: `driver`, `8 reward`, `8 rollout`, `4 trainer`; `boot_to_ready_s 168.568 s`, `s0_capture_s 0.305 s`, warm vLLM graph captures `[3.0..4.0] s`, `0 MiB × 4` teardown):
- **`install_s` (`core_algos` `2×` advantage, 3 cycles):** **`1.245 s` / `1.308 s` / `1.274 s`** (mean **`1.276 s`**, **`6.75×` faster** than serial `8.606 s`; `prepare_s 0.189–0.194 s`, `gap_s 0.046–0.052 s`, `edit_to_commit_s 0.238–0.241 s`). Step-only (`verdict_s - install_s`) held at `26.434 / 27.706 / 27.317 s` (mean `27.152 s`, `r_base 26.559 s`), total `install` verdict `27.679 / 29.014 / 28.591 s`.
- **`uninstall_s` (3 cycles):** **`1.109 s` / `1.087 s` / `1.092 s`** (mean **`1.096 s`**, **`5.61×` faster** than serial `6.149 s`; coordinator `undo_s 0.0473–0.0487 s`).
- **`status` (`extra=None`):** **`0.855 s`** (`status_parallel`) and **`0.558 s`** (`status_final`), **`3.48×–5.33×` faster** than the same job's serial `fingerprint_s` sum of **`2.976 s`** (`max` single actor `0.277 s` on `trainer1`: `driver 0.214 s`, `reward0..7 0.062 s`, `rollout0..7 1.741 s`, `trainer0..3 0.959 s`).
- **Parity & CPU/Ray regression suites:** `reset` `cost_s 1.221–1.223 s` (`1.269 s` initial), `adv_max 2.474867 -> 4.949733 -> 2.474867` exact across all 3 cycles, `pg_loss 0.016291 -> 0.032583 -> 0.016291` (`< 4e-9`), `witness={'driver': {'verl.trainer.ppo.core_algos:compute_grpo_outcome_advantage:267': 1}}`. Linux Python 3.12 `pytest` (`-W error::ResourceWarning`): `repos/verl/livelab` **`26 passed`**, `repos/miles/livelab` **`15 passed`** (`11` subtests).

### 0.8 PEP 578 negative cache on `file_namespace` scans (`4× GB200`, `2026-11-20`)

On `veRL` jobs whose `tracked_modules` includes `custom_reward_function.py`, `13` of `21` processes (`driver`, `rollout0..7`, `trainer0..3`) never load the reward file and previously ran full `gc.get_objects()` heap scans (`205–277 ms` each) on every `status()`, `install()`, `uninstall()`, and `RP.unloaded()` call. Adding a negative cache in `core/livelab/ray_participants.py` invalidated via a PEP 578 `sys.addaudithook` (`compile`, `exec`, `code.__new__`, `marshal.loads`) on the same `4× GB200` `fsdp_cg` workload (`21` participants, `0 MiB × 4` teardown):
- **`status` & per-actor `fingerprint_s`:** `status_parallel 0.475 s`, `status_timing 0.202 s` (**`17.30×` faster** than parallel-only `3.495 s`; serial per-actor `fingerprint_s` sum **`2.976 s -> 0.168 s`** [`17.71×`], max actor **`0.277 s -> 0.012 s`** [`23.08×`]: `driver 0.007 s` [`30.57×`], `reward0..7 0.044 s`, `rollout0..7 0.073 s` [`23.85×`], `trainer0..3 0.044 s` [`21.80×`]), **`status_after_cache 0.032 s`** (**`26.72×` faster** than `0.855 s`), `status_final 0.057 s` (`9.79×`).
- **`install_s` & `uninstall_s` (`3` live `core_algos` cycles):** `install_s` **`0.892 / 0.777 / 0.811 s`** (mean **`0.827 s`**, **`1.54×`** vs `1.276 s`, **`10.41×`** vs `8.606 s` serial; `prepare_s 0.189–0.203 s`, `gap_s 0.045–0.048 s`, `edit_to_commit_s 0.237–0.250 s`); **`uninstall_s` `0.127 / 0.127 / 0.127 s`** (**`8.63×`** vs `1.096 s`, **`48.42×`** vs `6.149 s` serial; coordinator `undo_s 0.0361–0.0366 s`).
- **Correctness & Linux CPU suites:** `adv_max 2.474867 -> 4.949733 -> 2.474867`, `pg_loss` exact `< 4e-9`, `rewards/mean=0.109375`, `reward0..7` abstained accurately, `s0_digests` unchanged; Linux Python 3.12 (`-W error::ResourceWarning`): `core/livelab 248 passed, 1 skipped` (`162/162` kernel `unittest`), `repos/verl/livelab 26 passed`, `repos/miles/livelab 15 passed`.

### 0.9 AST `compile(ast.AST)` negative-cache preservation & two-stage online replay ladder (`4× GB200`, `2026-11-21`)

Preserving `_ABSENT_FILES` across AST `compile(ast.AST)` calls in `core/livelab/ray_participants.py` (`507/507` AST compiles with `0` cache wipes and `0` drift on `adv_max`) and running the two-stage online replay ladder on the same `4× GB200` `Qwen3-0.6B` GSM8K `fsdp_cg` workload (`21` participants, `0 MiB × 4` teardown):
- **21-participant swap & post-commit overhead:** `install_s` **`0.440 / 0.324 / 0.448 s`** (mean **`0.404 s`**, **`2.05×`** vs prior `0.827 s`, **`21.30×`** vs `8.606 s` serial); post-commit overhead **`0.109 s`** (**`5.33×`** smaller vs `0.584 s`).
- **Two-stage online landing ladder:** Cold restart **`297.95 s`** -> Stage 1 Full `256`-trajectory GRPO step **`27.46 s`** (**`10.9×`**: `0.404 s` swap + `1.22 s` `S0` + `25.84 s` step) -> Stage 2 Online `TransferQueue` `8`-trajectory FSDP mini-batch replay through the `21`-participant gate **`2.70 s`** (**`110.3×`**: `0.338 s` swap + `1.396 s` coupled `S0` + `0.967 s` FSDP replay/update/proof; `weights_changed_from_s0=true`, `0.0` drift on `adv_max` and `pg_loss`), plus **`2.54 s`** (**`117.3×`**) for online `256`-trajectory advantage replay.
