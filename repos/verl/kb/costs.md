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
