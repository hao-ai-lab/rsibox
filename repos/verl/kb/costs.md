# Costs (`verl` GRPO on 4x GB200)

Measured on one node of **4x NVIDIA GB200 (`aarch64`, SM100, CUDA 13.0)** running `volcengine/verl`
(`975f063b`, `0.10.0.dev`) with `Qwen3-0.6B` GRPO on GSM8K (`PPOTrainerSync`, Megatron-Core
`TP1/PP1/DP4` with Transformer Engine + flash-attn, 4 colocated vLLM rollout replicas, 8
`RewardLoopWorker` actors; 640 responses per step).

Sources:
- Startup phase breakdowns: `VERL_STARTUP_TRACE` (`verl/utils/startup_trace.py`) from our
  2026-10-21 4x GB200 runs on fast parallel storage and on shared NFS.
- Live install + step verdicts and offline probes: gate receipts (`install_s`, `verdict_s`,
  `score_ms`, `compute_ms`) on the same 4x GB200 node.

## 1. Verdict Cost Ladder: Offline Probes vs. Live Step-Gap vs. Restart

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

## 2. Startup Phase Anatomy (`VERL_STARTUP_TRACE`, Seconds)

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

## 3. Steady GRPO Step Breakdown (Megatron-Core, Mean of Steps 2-4)

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
