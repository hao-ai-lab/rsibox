# SGLang costs

Prior measurements gathered before new runs, plus the static-survey baseline. Every number names
its source and setup; anything we have not timed yet is listed under [Not measured yet](#not-measured-yet).

## FastRSI v0 overhead and conformance (Qwen3-8B, 1 GPU, sampler edits)

Source: `legacy/v0` branch, `receipts/overhead-sglang.json`, `receipts/conformance-sglang.json`,
`receipts/control-repeatability.json`, and `docs/overhead.md` (model and workload reconstructed
at `0ab2ef8`: `lmsysorg/sglang@sha256:b1259f3ea327...`, `sglang.Engine`, `random_seed=0`,
`max_running_requests=1`, 8 prompts x 32 greedy tokens = 256 tokens; GPU model not recorded in
the v0 receipt).

| Cost | Value | Source |
|---|---:|---|
| Fresh engine startup, stock median (min-max, n=3) | 79.05 s (78.86-79.23 s) | `overhead-sglang.json` |
| Fresh engine startup, agent loaded (min-max, n=3) | 79.42 s (78.67-79.44 s) | same |
| Workload (8 x 32 greedy tokens), stock vs agent median (n=45) | 1.0349 s vs 1.0354 s (+0.06%, within run-to-run spread) | same |
| Live code swap install, median / p90 / max (n=40) | 79.1 / 91.9 / 740.2 ms | same |
| Live code swap undo, median / p90 / max (n=40) | 16.0 / 19.2 / 25.6 ms | same |
| `reset` (`engine.flush_cache()`), single call | 2.26 ms | same |
| Host RSS, stock vs agent | 6,076-6,095 MiB vs 6,080-6,089 MiB | same |
| Host RSS growth across 40 install/undo cycles | +48.9 MiB (+1.22 MiB/cycle) | same |
| GPU memory, stock vs agent (and after 40 cycles) | 58,927 MiB vs 58,927 MiB (+0 MiB) | same |
| Fresh-process equivalence check (`c3_fresh_equivalence`) | 86.09 s | `conformance-sglang.json` |
| Crash detect / crash recover (`c8_crash_and_hang`) | 6.32 s / 59.10 s | same |
| Hang detect (60 s timeout) / hang recover | 60.01 s / 60.29 s | same |
| Repeatability, `max_running_requests=1`, 8 x 32 tokens | 1 distinct digest in 30 runs (`c0_repeatable`, 31.10 s) | same |
| Repeatability, batched 32 x 64 tokens, stock (reset between runs) | 6 distinct digests in 60 runs | `control-repeatability.json` |
| Repeatability, batched 32 x 64 tokens, agent (reset / swap+undo) | 5 / 5 distinct digests in 60 runs | same |

Reading: a fresh engine start (~79 s) or crash recovery (~65 s) is roughly three orders of
magnitude slower than a sampler swap + undo + `flush_cache` (~97 ms). Batched greedy decoding
varied even with no agent loaded, which is why the workload gates at `max_running_requests=1`
(`workload.md`).

## Static coverage survey (`fastrsi-survey/1`)

Source: `core/livelab/survey` (ruleset `1`) run against the `sgl-project/sglang` Python tree at
snapshot `cf9845f8e` (`0.5.10.post2.dev438`), excluding test files (`kb/survey_rules.json`,
version `sglang-1`). Command (from `core/livelab` in the survey worktree):

```sh
python3 -m survey scan --rules repos/sglang/kb/survey_rules.json \
    --source <sglang>/python --repo . --date 2026-11-16 \
    --out coverage_scan_sglang.json --summary
```

Totals across `<sglang>/python/sglang` (19,769 Python functions, 499,480 non-comment Python
lines; 96 native C++/CUDA files with 20,521 lines; 1 unparsed file of 55 lines,
`sglang/srt/models/qwen2_classification.py`, leading U+FEFF BOM):

| Install class | Functions | Covered (with adapter + recapture + zygote) | Python lines | Covered lines |
|---|---:|---:|---:|---:|
| `swap` (`OP`) | 10,084 | 4,582 | 186,286 | 83,661 |
| `swap_recapture` (`GRAPH` + `@triton.jit`) | 5,185 | 5,169 | 130,110 | 129,732 |
| `rebuild` (`STATE` + `RESOURCE`) | 3,946 | 0 | 105,590 | 0 |
| `respawn` (`RANKS` + CuteDSL/TileLang + `module_level`) | 290 | 290 | 66,238 | 47,336 |
| `restart` (CLI, `ServerArgs`, `environ`, plugins) | 264 | - | 11,256 | - |
| **Total Python** | **19,769** | **9,751 live (49.3%) + 290 respawn (1.5%)** | **499,480** | **213,393 live (42.7%) + 47,336 respawn (9.5%)** |
| Native C++/CUDA (counted apart) | 96 files | 0 (restart) | 20,521 | 0 |

Note on capability gating: `survey` marks a capability covered only when the evidence files in
`survey_rules.json` exist in `--repo`. Before `adapter/{lab,agent,recapture}.py` land on disk
(with only `services/zygote.py` present), live covered is `0 / 19,769` and respawn covered is
`290 / 19,769` (`47,336` lines); once those three adapter files are present, the same rules file
reports the `9,751` live-covered functions (`49.3%`) shown above.

Where the 9,464 uncovered non-restart functions sit once the adapter is present:

- **Out-of-scope subpackages (`sglang/multimodal_gen/*`, `sglang/lang/*`, benchmarks):** 4,612
  functions live in `multimodal_gen` (diffusion/video generation, a separate runtime from `srt`).
  Within `sglang/srt` alone (14,036 functions), **9,272 functions (66.1%)** are live-covered
  (`4,117` `swap` + `5,155` `swap_recapture`) and `280` (`2.0%`) are respawn-covered.
- **Constructors and start-time builders (`3,946` `rebuild` functions; `2,868` in `srt`, of
  which `1,817` are `__init__`/`__post_init__`):** no in-place `rebuild` capability is wired yet.
- **Unreached non-scheduler processes in `srt` (`1,452` `swap` functions):** `entrypoints/*`
  (`400`), `debug_utils/*` (`448`), `function_call/*` (`181`), `TokenizerManager` /
  `DetokenizerManager` / `DataParallelController` (`179`), and multimodal processors (`167`) run
  outside the scheduler process where the participant lives.
- **Firm vs heuristic share:** `2,780` functions (`14.1%`) are syntax-proved (`@triton.jit`
  `236`, `@torch.compile` `40`, CuteDSL/TileLang `17`, constructors `2,487`); `16,989` (`85.9%`)
  rest on path, name, or `default_op` heuristics.

## Not measured yet

- Zygote fork time to ready vs fresh `sglang.Engine` start, and module-refresh cost on SGLang's
  import graph.
- Overlap-scheduler drain latency (`pause_generation(mode="retract")` + `continue_generation`)
  at the step gap.

## 1x B200 phase-split 1-change ladder (`Qwen3-8B`, TP=1, `2026-11-19`)

Source: nightly `sglang` `0.0.0.dev1+g302b78ccf` on `1x NVIDIA B200` (`Qwen3-8B`, TP=1, decode CUDA graphs `[1, 2, 4]`), visible edit on `Sampler.forward` (`greedy + 1 mod V`) + selective `R1` recapture on `Qwen3DecoderLayer.forward` (`sizes=[1, 2]`):

| Arm / Phase | Value | Notes |
|---|---:|---|
| Fresh container start + `16x64` check (`fresh_base`) | **66.47 s** | `19.96 s` imports + `33.15 s` engine init (`1.83 s` weights, `18.37 s` cold graph capture/warmup, `12.95 s` spawn/KV/init) + `13.37 s` check |
| Warm-cache fresh restart + `16x64` check (`fresh_edited`) | **25.10 s** | `6.92 s` imports + `15.92 s` engine init (`1.77 s` weights, `2.88 s` decode graphs, `11.27 s` spawn/KV/init) + `2.26 s` check; `proc_wall_s = 27.08 s` |
| Live `R0` swap + `16x64` check (`SglangLab`) | **1.97 s** (`12.8x` vs warm / `33.8x` vs cold) | `0.0600 s` (`60.0 ms`) install + `1.908 s` check (`witness = 65` calls, digest bit-exact to `fresh_edited`), `0.0109 s` (`10.9 ms`) undo (bit-exact to base) |
| Live `R0` swap + targeted `2x16` probe | **0.427 s** (`58.8x` vs warm / `155.6x` vs cold) | `0.0600 s` install + `0.367 s` probe (`proven = True`, witness `81` calls cumulative) |
| Live `R1` selective decode graph recapture (`Qwen3DecoderLayer.forward`, `sizes=[1, 2]`) | **0.839 s** edit-to-rebuilt | `0.045 s` commit + `0.794 s` R1 recapture (`216` capture witness calls); post-recapture probe `0.095 s` (`proven = True`) |

## Real-GPU `SglangConformanceLab` `11/11` receipts (`1x B200 TP=1` and `2x B200 TP=2`, `2026-11-19`)

Source: `fastrsi conformance sglang --real` on `Qwen3-8B` inside nightly `sglang` (`lmsysorg/sglang:nightly-dev-cu13-20261030-302b78cc`, decode CUDA graphs `[1, 2, 4]`, `--mem-fraction 0.45 --max-bs 4 --check-timeout 25`), after fixing `adapter/fixture/faults.py` (`local_participant.name` instead of removed `.tp_rank`) and `adapter/fixture/harness.py` (`SGLANG_RETURN_ORIGINAL_LOGPROB` fallback for `module_state` refusal):

| Configuration | Result | `c9_soak` (20 cycles install / undo) | `c8_crash_and_hang` respawns | `c3_fresh_equivalence` | GPU Cleanup |
|---|---|---:|---|---|---|
| `1x B200` (`TP=1`) | **11/11 pass** (`rc=0`) | **`0.0589 s` / `0.0107 s`** (`7.212 s` total) | crash `16.419 s`, hang `16.249 s`, lost `15.821 s`, tamper `16.059 s` (`117.933 s` case) | `112.782 s` (visible + 3-chain bit-exact to fresh) | `0 MiB` |
| `2x B200` (`TP=2`) | **11/11 pass** (`rc=0`) | **`0.1056 s` / `0.0182 s`** (`8.362 s` total) | crash `17.215 s`, hang `16.895 s`, lost `17.271 s`, tamper `17.167 s` (`120.666 s` case) | `134.339 s` (`[16, 16]` witness calls across ranks) | `0 MiB x 2` |
