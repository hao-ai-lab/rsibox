# vLLM costs

Measured costs. Each table says where it came from.

## Restart vs live, one-line scheduler edit (1 GB200)

Source: LiveLab-MLSys paper draft, Figure 1. Warm on-disk caches.

| Step removed | Check time | vs restart |
|---|---:|---:|
| Restart with caches (baseline) | 228 s | 1.0× |
| + reuse autotune config | 157 s | 1.5× |
| + keep model weights | 100 s | 2.3× |
| + keep CUDA graphs | 58 s | 3.9× |
| + keep the live process (hot swap) | 23.6 s | 9.6× |
| then skip the logprob check | 14.1 s | |
| then test with a small batch | 2.4 s | 95× end to end |

Reading: setup is 204 s of the 228 s. The ladder is the disaggregation opportunity: each row is
a piece of state that can be kept separately.

## DeepSeek-V4.1-Flash startup (TP4, 4× B200)

Source: our startup measurement, 2026-10-08, `vllm/vllm-openai:v0.31.0`,
`--max-model-len 16384`. Seconds.

| Phase | Cold | Warm |
|---|---:|---:|
| Weight load (TP0) | 178.6 | 124.9 |
| Profile + memory-profiling CUDA graph capture | 111.0 | 79.0 |
| JIT kernel warmup (190 keys) | 52.5 | 52.5 |
| FlashInfer autotune | 625.4 | 627.2 (retuned despite saved configs) |
| Final CUDA graph capture | 40 | 40 |
| **Time to /health** | **1110.2** | **1012.2** |

Reading: autotune is 62% of a warm start and its cache did not help. Engram tables are 2 × 23.6 GiB
pinned host memory per rank. No torch.compile for this model in this image.

## Restart after one edit (Qwen3-8B, warm caches otherwise)

Source: our per-edit restart measurement (pre-v0 internal tree), dev workload.

| Edit | Ready | vs warm 17.8 s |
|---|---:|---:|
| Triton kernel in the V2 runner | 17.9 s | +0.1 |
| Comment in `qwen3.py` | 31.0 s | +13.2 |
| Attention forward in `qwen3.py` | 41.7 s | +23.9 |
| Backend FLASHINFER → FLASH_ATTN | 60.7 s | +42.9 |
| Full cache loss (cold) | 150.3 s | |

Native CUDA incremental rebuild: 8.0–24.2 s per file (`ninja`, sm100).

## FastRSI v0 overhead (Qwen3-8B, 1 GPU, sampler edits)

Source: `legacy/v0` branch, `receipts/overhead-vllm.json`, `docs/overhead.md`.

| Cost | Value |
|---|---:|
| Workload, stock vs agent loaded | 0.2924 s vs 0.2923 s (not detected) |
| Install, median / max | 22 / 44 ms |
| Undo, median / max | 4 / 9 ms |
| Host growth | +0.37 MiB per install/undo cycle |
| Restart to base | 40 s |
| Witness, per call | 34 ns |
| Repeatability, 32 prompts batched | 1 distinct output in 60 runs |

## Zygote: edits after it started, refreshed in the fork (Qwen3-8B, 1× B200, 2026-11-06)

The zygote (`livelab/services/zygote.py`) imports vLLM and its libraries once and forks each
`vllm serve` from that state. Before the server starts, the fork drops the edited
modules and their importers, so the server imports their new source (`livelab/services/zygote_refresh.py`). Source: our 2026-11-06
run, Qwen3-8B TP1 eager, warm caches, forked engine core; launch to `/health`, one greedy
completion each (" Paris." in every case); markers printed by the edited code show what ran.

| Case | Modules refreshed | To `/health` (s) | Edited code ran |
|---|---:|---:|---|
| fresh `vllm serve` (no zygote) | – | 20.9 | – |
| fork, no edit | 0 | 10.9 | – |
| fork, attention backend edited, refresh off | 0 | 10.4 | **no** (old code, silently) |
| fork, attention backend edited (leaf) | 2 | 10.2 | yes (class body + `__init__`) |
| fork, `envs.py` + engine core edited (core) | 1,371 (all of vLLM) | 13.7, 12.4 | yes (API server + engine core) |

- The zygote imported 8,631 modules in 12.7 s and builds the import graph of its 1,371 vLLM
  modules at start (2.3 s, off the start path). Only imports that run at import time count;
  imports inside functions look the module up at call time and see the new one.
- vLLM's module graph has one large cycle through `vllm.config` and `vllm.platforms`: an edit to
  almost any module one of them imports reaches ~760 of ~1,000 modules. Above 100 modules the
  refresh re-imports all of vLLM instead (+1.5–2.8 s over a plain fork), since a partial
  re-import would register into package-wide tables (IR ops, CustomOp) twice. Third-party
  libraries, most of the import time, stay imported. Leaf modules (attention backends, model
  files, the GPU model runner: 1–35 modules) refresh in milliseconds.
- Re-imported torch ops: an identical schema is accepted and the new body replaces the old one;
  a changed schema is refused with the op's name. Opaque types re-register.
- Still a finding (restart the zygote): an edited extension `.so`, or an edited module outside
  `FASTRSI_ZYGOTE_REFRESH` (default `vllm`).

## Edit-to-run by restart boundary (GB200)

Source: our 2026-11-05 runs. Graphs and compilation, not Python, decide edit-to-run.

| Boundary | Edit-to-run |
|---|---|
| function swap | 0.2–2 s |
| Triton kernel rebuild | 1–5 s |
| CUDA graph recapture | 4.5–10 s |
| torch.compile recompile | 16–37 s |
| model module rebuild in place | 4–10 s |
| forked restart from a warm process | 16–35 s |
| warm restart | 25–166 s |

## Not measured yet

- DeepSeek-V4.1-Flash on 4× GB200 EP4DP4 (the target workload): restart breakdown and each
  probe's cost.
- Cost of `install` per group for the three example PRs.
