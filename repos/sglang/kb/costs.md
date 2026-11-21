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

## Zygote fork and overlap-scheduler drain (`Qwen3-8B`, TP=1, 1× B200, `2026-11-19`)

Source: `repos/sglang/livelab/services/zygote.py` + `sglang` `0.0.0.dev1+g302b78ccf` on `1× B200` (`Qwen3-8B`, TP=1, `cuda_graph_max_bs_decode=4`, `disable_prefill_cuda_graph=True`, `0 MiB` exit):

| Path / Operation | Measured | Notes |
|---|---:|---|
| Zygote preload (`torch`, `sglang`, model/scheduler modules) | `8.813 s` | One-time startup off the fork path (`zygote_ping ok: true`) |
| Fresh `sglang.Engine` + probe | `59.855 s` (`ready 52.572 s`: `10.262 s` import + `42.310 s` init; `5.497 s` probe) | Output digest `2301fe01c380e740` |
| Fork clean (`fork_clean`, `refreshed: []`) + probe | **`17.041 s`** (**`3.51×`** vs fresh) | `16.148 s` fork+init + `0.709 s` probe; digest `2301fe01c380e740` |
| Fork after editing `entrypoints/engine.py` (`fork_refresh`) + probe | **`19.693 s`** (**`3.04×`** vs fresh) | `18.742 s` (`16.592 s` init, `refreshed: [".../entrypoints/engine.py"]`) + `0.697 s` probe; digest `2301fe01c380e740` |
| Idle step-gap drain (`5×` `pause_generation(mode="retract")` + `continue`) | `pause [0.050, 0.012, 0.008, 0.006, 0.006] ms`; `continue [0.017, 0.013, 0.006, 0.006, 0.006] ms` | Control-plane drain latency when queue is idle |
| Active overlap drain (`6` in-flight requests, `all_completed: true`, `mode="retract"`) | **`0.076 ms`** (`pause 0.059 ms` + `continue 0.017 ms`; `total_batch_s 8.969 s`) | All `6` requests resumed and completed after `ContinueGenerationReqInput()` |
| Active overlap drain (`6` in-flight requests, `all_completed: true`, `mode="in_place"`) | **`0.192 ms`** (`pause 0.161 ms` + `continue 0.031 ms`; `total_batch_s 2.904 s`) | `3.09×` shorter total batch wall than `retract` on the active 6-request batch |

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

### `SglangZygote` `TP=2` & `> 1 s` post-fork subphase breakdown (`2× B200`, `Qwen3-8B`, `2026-11-19`)

Source: `repos/sglang/livelab/services/zygote.py` on `2× B200` (`Qwen3-8B`, `TP=2`, `0 MiB × 2` exit):

| Path | Total (to probe done) | Subphases & `> 1 s` Breakdown |
|---|---:|---|
| Zygote preload | `9.4145 s` | One-time `torch` + `sglang` import off the restart path |
| Fresh `TP=2` `sglang.Engine` + probe | `77.619 s` (`80.2911 s` unhooked) | `11.0621 s` import + `59.2256 s` init (`scheduler_init_end 41.5965–41.9262 s`: `load_model 1.994 s`, `init_cuda_graphs 22.2229–22.2275 s`) + `5.0808 s` probe; digest `2301fe01c380e740` |
| `fork_clean` (`refreshed: []`) | **`17.5977 s`** (**`4.41×`**; `17.9878 s` [`4.46×`] unhooked) | `16.9349 s` enter-to-done + `0.4328 s` probe (`digest: 2301fe01c380e740`); post-fork `> 1 s` subphases inside `16.9349 s`: **`spawn_plus_scheduler_import_s 7.2304–7.3105 s`** (`mp.spawn` re-imports `torch` + `sglang` in each scheduler child), **`load_model 1.8905 s`**, **`tp_worker_init_end 3.3957–3.4188 s`**, **`init_cuda_graphs 2.8316–2.8330 s`**, remaining KV pool/warmup `~3.8 s` (`scheduler_init_end 8.5297–8.9220 s`, `dist_init 0.0017–0.0242 s`) |
| `fork_refresh` (`entrypoints/engine.py`) | **`20.1091 s`** (**`3.86×`**; `20.444 s` [`3.93×`] unhooked) | `19.1006 s` enter-to-done + `0.8662 s` probe (`spawn_plus_scheduler_import_s 7.2491–7.2748 s`, `load_model 1.8876–1.8897 s`, `init_cuda_graphs 2.8648–2.8652 s`, `scheduler_init_end 8.6544–8.9796 s`) |

## Unified cold-vs-warm-vs-zygote restart-overhead dissection (`Qwen3-8B`, TP=1, `1× B200`, `2026-11-20`)

Source: paired `1× NVIDIA B200` (GPU `0`, `0 MiB` exit) dissection across `fresh_cold` (empty TorchInductor/Triton/FlashInfer/DeepGEMM caches), `fresh_warm`, `SglangZygote` `fork_clean` (`9.8325 s` one-time preload off path), and `SglangLab` live swaps (`Sampler.forward` `R0`, plus `Qwen3DecoderLayer.forward` `R1` decode-graph recapture) on `Qwen3-8B` TP=1 (`decode` CUDA graphs `[1, 2, 4]`, `prefill` graphs disabled, full check `16 × 64` greedy tokens, targeted probe `2 × 16` greedy tokens; base digest `ac0b97366ea1bd86`, edited digest `2b798b1998ad257f` bit-exact between fresh and live, `not_split = 0.000 s`):

| Arm / Rung | Total (`s`) | Speedup vs Cold | Critical-Path Phase Breakdown (`s`) |
|---|---:|---:|---|
| 1. `fresh_cold` (empty compile caches) | **`66.387 s`** | `1.0×` | `import_init 20.011` + `spawn_import 8.023` (`mp.spawn` `0.736` + child re-import `7.286`) + `weight_load 2.070` + `jit_build 25.153` (`14.423` graph compile + `10.730` first-batch JIT) + `cuda_graph_capture 3.116` + `engine_init 5.203` + `first_step 0.627` + `run_workload 2.185` (`check 2: 1.877 s`; `proc_wall_s 72.736 s`) |
| 2. `fresh_warm` (warm compile caches) | **`26.456 s`** | `2.5×` | `import_init 7.292` + `spawn_import 8.021` + `weight_load 1.985` + `cuda_graph_capture 3.116` + `engine_init 3.446` + `first_step 0.627` + `run_workload 1.970` (`check 2: 2.216 s`; `proc_wall_s 32.753 s`) |
| 3. `SglangZygote` `fork_clean` (in-worker) | **`19.487 s`** | `3.4×` | `import_init 0.000` (cut by `9.83 s` preload) + `spawn_import 8.609` (`1.209` fork + `7.401` child re-import) + `weight_load 1.930` + `cuda_graph_capture 2.936` + `engine_init 3.528` + `first_step 0.426` + `run_workload 2.057` (`21.765 s` external wall incl. `2.278 s` fork RPC/teardown) |
| 4. Derived peel `weight_load` | `17.556 s` | `3.8×` | Row 3 minus `weight_load` (`1.930 s`) |
| 5. Derived peel `spawn_import` (keep scheduler) | `8.947 s` | `7.4×` | Row 4 minus `spawn_import` (`8.609 s`) |
| 6. Derived peel `cuda_graph_capture` | `6.011 s` | `11.0×` | Row 5 minus `cuda_graph_capture` (`2.936 s`) |
| 7. Live `R0` step-gap swap + `16 × 64` check | **`1.766 s`** | **`37.6×`** | `live_install 0.0595` (`prepare 0.0478`, `gap 0.0043`) + `run_workload 1.7061` (check 2: `2.0442 / 2.1037 s`; `undo 0.0111 s` bit-exact to `ac0b97366ea1bd86`; `R1` graphed layer selective recapture adds `0.7848 s` edit-to-rebuilt [`0.7448 s` recapture]) |
| 8. Live `R0` swap + `2 × 16` targeted probe | **`0.377 s`** | **`176.3×`** | `live_install 0.0595` + `run_workload 0.3171` (`probe witness 146` calls) |

## `SglangZygote --fork-subprocs` paired comparison (`Qwen3-8B`, `TP=1` and `TP=2`, `B200`, `2026-11-20`)

Source: paired `1×` and `2× NVIDIA B200` runs (`lmsysorg/sglang:nightly-dev-cu13-20261030-302b78cc`, `Qwen3-8B`, warm caches, decode CUDA graphs `[1, 2, 4]`, `disable_prefill_cuda_graph=True`, `random_seed=0`, `0 MiB` exit across all GPUs) comparing `fresh_warm`, `SglangZygote` default (`spawn`), and `SglangZygote --fork-subprocs` (`SUBPROC_PRELOAD` of `scheduler` + `detokenizer_manager`). Container `pytest` (`-W error::ResourceWarning`): **`20 passed in 13.79 s`**.

Two guards make `--fork-subprocs` safe across `TP=1` and `TP=2` refreshes:
- **In-child NUMA binding (`SGLANG_NUMA_BIND_V2=0`):** SGLang's NUMA v2 path exec-wraps `multiprocessing.spawn.get_executable()` and asserts `get_start_method() == "spawn"`. Disabling v2 when `fork_subprocs` is active skips the parent `numactl` exec wrapper and triggers `scheduler.py`'s built-in in-process `numa_bind_to_node(numa_node)` inside each forked scheduler child.
- **Selective AST refresh + `torch.ops` fallback (`plan_refresh`):** Preloading `scheduler` registers `torch.ops.sglang.outplace_all_reduce` in PyTorch's C++ dispatcher against `parallel_state._groups`. Dropping all modules unconditionally splits `_groups` across old and new imports and raises `AssertionError: Group tp:0 is not found.` on `TP=2` graph capture. Using `hooks.import_graph` + `dependents` (`WHOLE_REFRESH_ABOVE=100`) limits `entrypoints/engine.py` refreshes to **`2` modules** (`enter -> import 0.010 s` vs `2.14 s` whole-package drop, `0` fallbacks), while edits reaching `parallel_state`, `custom_op`, or `scope="all"` emit `fork_subprocs_fallback` (`reason: "refresh_rebinds_torch_ops"`) and spawn cleanly.

| Topology | Arm | Preload (s) | Enter -> Import (s) | `spawn_plus_import_s` (max, s) | `engine_init_s` (s) | `total_first_step_s` (s) | External Wall (s) | Speedup vs Zygote `spawn` / `fresh_warm` |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `TP=1` (`1× B200`) | `fresh_warm` | — | 6.7826 | 7.8497 | 15.8067 | **24.8494** | 29.4031 | `0.74×` / `1.0×` |
| `TP=1` (`1× B200`) | `SglangZygote` `spawn` (`fork_clean`) | 10.6098 | 0.0000 | 8.2922 | 16.2158 | **18.4883** | 21.5262 | `1.0×` / `1.34×` |
| `TP=1` (`1× B200`) | `SglangZygote --fork-subprocs` (`fork_clean`) | 11.2106 | 0.0000 | **0.8408** (`9.86×`) | **8.3491** | **10.5882** | **13.5809** | **`1.75×` / `2.35×`** |
| `TP=1` (`1× B200`) | `SglangZygote --fork-subprocs` (`fork_refresh` `engine.py`) | 11.2106 | 0.0102 | **0.7099** | **8.3654** | **10.5521** | **13.6812** | **`1.75×` / `2.35×`** |
| `TP=2` (`2× B200`) | `fresh_warm` | — | 6.9715 | 8.3414 | 16.8610 | **26.3547** | 31.6186 | `0.76×` / `1.0×` |
| `TP=2` (`2× B200`) | `SglangZygote` `spawn` (`fork_clean`) | 10.8082 | 0.0000 | 8.8680 | 17.5237 | **20.0605** | 23.0338 | `1.0×` / `1.31×` |
| `TP=2` (`2× B200`) | `SglangZygote --fork-subprocs` (`fork_clean`) | 11.2097 | 0.0000 | **0.8518** (`10.41×`) | **9.4288** | **11.8862** | **15.0708** | **`1.69×` / `2.22×`** |
| `TP=2` (`2× B200`) | `SglangZygote --fork-subprocs` (`fork_refresh` `engine.py`) | 11.2097 | 0.0095 | **0.8060** | **10.1825** | **12.5577** | **15.9895** | **`1.60×` / `2.10×`** |

All `--fork-subprocs` runs recorded `fallbacks: []`, `numa_bind_v2: "0"`, `cuda_init_at_sched_begin: false`, `scheduler_forked_without_reimport: true`, and bit-exact initial `16×64` (`ac0b97366ea1bd86` `TP=1`, `8073e9c4c63ce0dd` `TP=2`), repeated `2×16` (`2301fe01c380e740`), and `1×64` (`45c000390a7e3ad1`) output digests (on `TP=2` `fork_clean`, the repeated `16×64` batch varied while `fork_refresh` repeated bit-exactly).

## Full 11/11 `SglangConformanceLab --fork-subprocs` (`TP=1` and `TP=2`, `Qwen3-8B`, `B200`, `2026-11-20`)

Wired opt-in `fork_subprocs` / `--fork-subprocs` into `SglangLab` and `SglangConformanceLab`, with static AST inspection of `ServerArgs` in `adapter/fixture/harness.py` plus pre-config `SglangLab._preload_fork_subprocs()` guards (avoiding an unguarded `ServerArgs` CUDA init in `sglang_env_config()` that otherwise forced `spawn` fallback across all `5/5` boots, `c8 = 120.319 s`). Container `fastrsi doctor` **`8/8` (`0` violations)** and container `pytest repos/sglang/livelab`: **`25 passed in 16.20 s`** (`0` `ResourceWarning`s), `0 MiB × 8` exit:
- **`TP=1 --fork-subprocs` (`1× B200`, `11/11` PASS, `108` receipts, `fork_subprocs_fallback: []` on `5/5` boots & `5/5` scheduler ranks):** total case sum **`222.417 s`** (`258.701 s` CLI wall vs `283.300 s` `spawn`, **`1.27×`**): `c0 1.632 s`, `c1 0.324 s`, `c2 0.284 s`, `c3 88.231 s` (vs `112.782 s`, `1.28×`), `c4 0.550 s`, `c5 0.777 s`, `c6 0.730 s`, `c7 1.163 s`, **`c8_crash_and_hang 92.308 s`** (vs `117.933 s`, `1.28×`; respawns **`9.79 / 9.98 / 9.86 / 9.62 s`**, sum `39.259 s` vs `64.548 s` `spawn`, **`1.64×`**), `c9 7.332 s` (`0.0620 / 0.0111 s`), `c10 29.086 s`.
- **`TP=2 --fork-subprocs` (`2× B200`, `11/11` PASS, `108` receipts, `fork_subprocs_fallback: []` on `5/5` boots & `10/10` scheduler ranks):** total case sum **`270.789 s`** (`309.316 s` CLI wall vs `309.327 s` `spawn`, **`1.14×`**): `c0 0.515 s`, `c1 0.274 s`, `c2 0.272 s`, `c3 132.334 s`, `c4 0.896 s`, `c5 0.629 s`, `c6 0.605 s`, `c7 0.913 s`, **`c8_crash_and_hang 97.829 s`** (vs `120.666 s`, `1.23×`; respawns **`11.92 / 11.54 / 11.84 / 10.91 s`**, sum `46.218 s` vs `68.547 s` `spawn`, **`1.48×`**), `c9 6.125 s` (`0.0630 / 0.0157 s`), `c10 30.397 s`.

## Native CUDA IPC weight-cache (`services/weights.py`) & two-stage `TP=1` ladder (`B200`, `2026-11-21`)

- **Native CUDA IPC weight-cache (`services/weights.py` + `SglangLab --weight-cache-dir`, unprefixed UUID normalization, out-of-place `c10` fresh control):** standalone `start`/`probe` `rc=0` without `--uuids`; `TP=2` (`2× B200`, `Qwen3-8B`) **`11/11`** conformance with `--fork-subprocs --weight-cache-dir` (`IpcModelLoader` clean forked loads **`0.077–0.082 s`**, **`310.42 s`** case sum, `c8` respawns **`47.86 s`**, post-`c10` standalone `c0` preserved base digest `dd7819950a7ed411`); container `pytest repos/sglang/livelab`: **`36 passed`**.
- **Two-stage `TP=1` `1× B200` ladder (`Qwen3-8B`):** Cold **`96.35 s`** (`128×512` decode) -> Warm **`57.31 s`** -> `SglangZygote` **`40.25 s`** -> Stage 1 Live `Sampler.forward` swap + `128×512` full check **`10.34 s`** (**`9.3×`**: `0.0663 s` install + `10.28 s` check, bit-exact) -> Stage 2 `8×192` targeted probe **`1.07 s`** (**`89.7×`**: `0.0663 s` install + `1.01 s` probe, bit-exact); `R1` selective `[1,2,4,8]` recapture **`1.20 s`** (`1.16 s` recapture).
