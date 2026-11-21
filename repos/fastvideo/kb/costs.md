# FastVideo costs: where the time goes

Measured on one GB200 tray (4× GB200, aarch64, CUDA 13.0, torch 2.12, FastVideo pin
`376e445f`), workload of `workload.md`. Weights on a shared network filesystem. Sources are
named per row.

## Cost ladder: a cold start to a warm video

Our 2026-11-07 ladder runs, one process per run, timing wraps around the real calls
(`VideoGenerator.from_config`, `generate`, stage `__call__`, component loads).

| Phase | 1 GPU | 2 GPUs | 4 GPUs | Keep warm by |
|---|---|---|---|---|
| Imports (numpy, torch, fastvideo) | 93.5 s cold (torch 11.0, fastvideo 81.9) | 21.8 s warm page cache | 25.4 s warm | a resident process |
| Model load (`from_config`) | 195.6 s cold page cache | 38.2 s warm, 2 workers | 40.4 s warm, 4 workers | a resident process; page cache |
| ↳ text encoder (NVFP4 Qwen3-VL) | 91.1 s | 4.3 s | 5.6 s | |
| ↳ DiT (NVFP4) | 76.3 s | 2.5 s | 3.2 s | |
| ↳ VAE / audio VAE | 21.9 s / 3.6 s | 5.0 s / 0.5 s | 5.6 s / 0.6 s | |
| First video (FlashInfer NVFP4 JIT build, VAE compile, Triton JIT) | 217.9 s cold JIT | 18.9 s warm JIT | 16.2 s warm JIT | JIT caches on disk; resident process |
| ↳ conditioning stage (FlashInfer `fp4_quantization`, `fp4_gemm_cutlass` JIT) | 174.4 s | 2.4-2.7 s | 2.1-2.8 s | |
| ↳ denoising stage | 30.4 s | 10.0-10.3 s | 8.7-9.4 s | |
| ↳ video decode (VAE compile) | 12.6 s | 5.0 s | 2.0 s | |
| Warm video (2nd, 3rd) | **6.13 s, 6.09 s** | **4.38 s, 4.38 s** | **3.12 s** median of 5 (3.06-3.44) | |
| ↳ denoise (8 DiT forwards) | 5.50 s (0.69 s per forward) | 3.86 s (0.48 s per forward) | 2.50 s (0.31 s per forward) | |
| ↳ conditioning / video decode / audio decode | 0.23 / 0.17 / 0.12 s | 0.23 / 0.17 / 0.07 s | 0.23 / 0.18 / 0.10 s | |
| Shutdown | 0.47 s | 3.4 s | 6.3 s | |

- **Repeatable:** the three videos of the 1-GPU run gave bit-identical frames and audio
  (frames hash `d0e55e892690e260`, audio `29b690c8ac7ae3be`), the first one included. The 2-GPU
  run (sequence parallel) gave the same hashes: the golden is the same on 1 and 2 GPUs.
- **2 GPUs:** 1.39x on the warm video (6.09 to 4.38 s); denoise scales 1.42x, the rest
  (conditioning, decode, driver) does not. The 2-GPU column ran right after the 1-GPU one on a
  warm page cache, so its imports and load are warm numbers, not a cold start.
- **4 GPUs:** 1.95x over 1 GPU on the warm video (6.09 to 3.12 s); denoise 2.2x. Warm videos
  vary more (3.06-3.44 s over 5 videos): measure 5 or more before comparing.
- **4 GPUs give a different video.** At sequence-parallel size 4 the frames hash is
  `0134242de77c3f4c` and the audio `8674315c6648fb91` (PSNR 17.7 dB against the 1-GPU video), the
  same in every 4-GPU run, while 1 and 2 GPUs give identical bits. Not a lab effect (no edit
  installed). Unexplained; a candidate upstream report. The 4-GPU gate therefore uses its own
  golden.
- **A restart costs about 500 s cold** (imports + load + first video) against a 6.1 s video:
  98.8% of a restart-based check is setup. With warm page and JIT caches the earlier GB200 run
  (2026-10-20, `kb/priors.md`) took 65 s imports and 53 s for the first video.

## Warm restart (same node, page and JIT caches warm)

Our 2026-11-07 live-lab starts (1 GPU, multiprocess executor), from the lab's own timers.

| Phase | Seconds |
|---|---|
| Lab process start to ready (imports + load) | 58.5 |
| ↳ model load (`from_config`) | 32.0 (68.6 on an earlier start the same night) |
| First video in the process (VAE compile, Triton JIT; FlashInfer JIT cached) | 13.9 |
| **Warm restart to a first measured video** | **about 72 s**, 12x a 6 s video |

## Live lab

Our 2026-11-07 runs on the 1-GPU H3 lab (`livelab/`), from the lab's replies and receipts.

| Call | Cost | What it covers |
|---|---|---|
| install (live swap of one stage method, 1 rank) | 0.06 s in the lab | stage, vote, commit in the request gap, fingerprint |
| edit-to-run (receipt `edit_to_run_s`) | 0.72 s | from the edit's timestamp on the agent's machine to commit, incl. the agent's network hop |
| check (one warm video) | 6.00-6.02 s | the workload; same frames and audio hashes as the in-process executor run |
| uninstall | 0.014 s in the lab | undo + fingerprint |
| mailbox round trip, agent to lab | 0.12-0.15 s | file mailbox on a shared filesystem |

The multiprocess executor at N = 1 gave the same output hashes as FastVideo's in-process
executor, and 6.00 s per video against 6.09 s: the frames crossing the process boundary cost
nothing measurable at this size.

**Edit-to-run:** 0.72 s live against about 72 s for a warm restart and about 500 s for a cold one.

## Why a restart costs what it costs (1 GPU, phase by phase)

Our 2026-11-07 phase-timed restarts on one GB200 of a 4-GPU tray: one fresh process per run,
timers wrapped around the real calls in the driver and in the spawned worker (interpreter start,
imports, `init_device`, distributed init, each component load, each stage, every Triton compile
and autotune bench, every FlashInfer JIT load and autotune lookup, every Dynamo frame compile,
plus Dynamo's own `compile_times`). Page cache warm. "Warm JIT" = every JIT cache at the path it
was built at. Workload of `workload.md`, all outputs bit-exact to the 1-GPU golden.

| Phase | Cold | Warm | Avoidable? | How | Action that avoids it |
|---|---|---|---|---|---|
| Driver interpreter start | 0.5 s | 0.5 s | in a resident host | keep the lab host | lab-host actions (respawns, in-place rebuild, live swap) |
| Driver imports (torch 1.5-2.4 s, fastvideo 14.8-35.6 s) | 93.5 s | 17.4-38.4 s | yes | resident lab host; zygote | zygote respawn, in-place rebuild |
| Worker spawn: interpreter 0.5 s + torch 1.6-2.1 s + fastvideo re-import 14.8-17.3 s | in load | 17-20 s | yes | fork from a preloaded server (`services/zygote.py`) | zygote respawn |
| CUDA context + device init (`init_device` minus loads) | | 0.2-0.3 s (micro: context 0.28-0.34 s) | no, per process | | |
| Distributed init (N = 1) | | 0.02 s | | | |
| Text encoder load (NVFP4 Qwen3-VL) | 91.1 s | 2.3-2.7 s | yes | keep the process; map by CUDA IPC from a holder | in-place rebuild; zygote respawn with weights holder |
| Tokenizer + processor | | 0.5-0.85 s | yes | keep the process | in-place rebuild |
| VAE load (incl. NVFP4 decoder apply) | 21.9 s | 5.0-5.2 s | yes | keep the process; IPC | in-place rebuild |
| Audio VAE | 3.6 s | 0.55-0.6 s | yes | keep the process; IPC | in-place rebuild |
| DiT load (NVFP4 checkpoint) | 76.3 s | 2.5-3.0 s | yes | keep the process; IPC: 1,838 tensors (36.9 GiB) mapped in 0.12 s | in-place rebuild; zygote respawn with weights holder |
| Stage creation, post-init, NCCL warm-up | | < 0.1 s | | | |
| **Load total (`from_config`)** | 195.6 s | 30.1-35.3 s (alone) | | | |
| FlashInfer NVFP4 JIT (`fp4_quantization`, `fp4_gemm_cutlass`) | 158.7 s (a warm cache copied to a new path rebuilds) | 0.09-1.05 s (ninja no-op + load) | yes | one cache root at a fixed path (`services/jitcache.py`) | warm restart with path-stable caches |
| Inductor `torch_key()` (hashes every torch source file) | 2.55 s | 0.8-2.6 s | yes | preset from disk (`jitcache.preset_torch_key`) | warm restart with path-stable caches |
| Dynamo: 5 frames (VAE + DiT `@torch.compile` helper), Inductor FX cache hit | | 1.1-1.2 s besides `torch_key` (bytecode tracing 0.72-1.02 s) | no, per process | keep the process | in-place rebuild |
| Triton kernel loads (23 kernels from the disk cache) | | 0.39-1.26 s | no, per process | keep the process | in-place rebuild |
| Triton autotune (VSA `_attn_fwd_sparse`, 12 configs) | | 1.94-2.64 s | yes (12 benches gone with the cache on; warm videos no slower in an A/B with and without the cache) | `TRITON_CACHE_AUTOTUNING=1` | warm restart with path-stable caches |
| FlashInfer autotune lookups (3,150 per video) | | 0.17 s (also on warm videos) | | | |
| cuBLAS / cuDNN handles, allocator growth | | 0.04 s / 0.09-0.14 s / 0.003 s per 40 GiB (micro) | no | | |
| **First video** | 217.9 s cold JIT; 174.0 s after a cache move | 13.95 s alone (warm 6.03-6.13 s): +1.0 s conditioning, +5.55 s denoise, +1.25 s decode | | | |
| Shutdown | | 2.2-2.6 s | | | |

- **The cold first video is FlashInfer's NVFP4 JIT, and it is a path problem.** Copying a warm
  cache root byte for byte to a new path rebuilt both modules (72.6 s + 86.1 s): FlashInfer runs
  ninja on every load and the ninja command lines hold the absolute path. Triton's and Inductor's
  caches hit after the same copy (content-keyed). A fresh process at the same path paid 0.09-1.05 s.
- **The warm first video's 7.8 s excess** (13.95 s against 6.1 s, a run alone on the tray): Dynamo
  3.8 s in total (of which `torch_key` 2.55 s, bytecode tracing 0.79 s), Triton autotune 1.94 s,
  Triton kernel loads 0.44 s, FlashInfer loads 0.09 s; about 1.5 s is not attributed (first calls of
  the text encoder and the CUDA modules they load).
- **The warm load's 31 s is half imports:** the spawned worker imports torch and FastVideo again
  (16-19 s); the component loads take 11-12 s, the VAE (5.1 s) the largest.
- **Noise:** from 13:04 PST the tray's other GPUs ran H3 checks too, and our warm videos read
  9.0-15.0 s instead of 6.0-6.1 s, with or without the Triton autotune cache (A/B: without the cache
  10.6-10.8 s, with it 11.4-15.0 s), so the cache is not the cause; which shared resource is,
  is not isolated (`lessons.md`). The phase rows above come from runs alone or from
  per-process work (imports, loads, JIT) that the noise moves less.

## Restart ladder: request to first video

Our 2026-11-07 runs (`livelab/services/`, lab verbs), the workload of `workload.md`, every
output bit-exact to the golden unless the edit changes it. "Overhead" is what the action costs
before the request can run; "first video" adds the video and whatever per-process work it
redoes. From 13:04 PST the tray's other GPUs ran H3 checks and a warm video read 9-15 s instead
of 6.0-6.1 s alone, so each row also has a **quiet** estimate: the measured overhead plus the
measured per-process first-video work plus a 6.1 s video.

| Action | What survives | Overhead | Request to first video, measured | Quiet (estimated) | Source |
|---|---|---|---|---|---|
| Live swap (`install`) | everything | install 0.06 s, no restart (measured) | 6.7 s (0.72 s edit-to-run + 6.0 s check) | 6.7 s | live lab table above |
| In-place rebuild (`rebuild`) | process, CUDA context, weights, JIT and compile state | 0.19-1.27 s at 1 GPU, 0.31-1.16 s at 2 ranks (measured) | 12.6-16.4 s (first video 10.9-15.3 s; the next warm video 11.5-15.0 s) | **6.4-7.5 s** (+0.1 s Dynamo recompile; +2.8-3.2 s to re-capture per-layer CUDA graphs when the edited module keeps them) | four 1-GPU rebuild runs, one at 2 ranks, one with the kept H3 patches (2026-11-07) |
| Zygote respawn with weights holder (`FastVideoLab(holder=True)`) | lab host, imports (forked from the zygote), DiT, text encoder and VAE on the GPU (holder) | 4.78 s and 5.62 s at 1 GPU with the VAE held (measured); 9.98-12.46 s at 1 GPU and 13.56-14.56 s at 2 ranks with DiT and text encoder only (measured) | 18.9-22.0 s (first video 14.1-16.4 s) | about 15 s (5.2 s + 2.2 s JIT + 1.5 s first calls + 6.1 s) | holder runs at 1 GPU (with and without the VAE held) and 2 ranks (2026-11-07) |
| Zygote respawn (`FastVideoLab(zygote=True)`) | lab host, imports | 14.05-15.33 s at 1 GPU; 18.30-18.33 s at 2 ranks (measured, paired) | 28.5-34.8 s (first video 14.2-20.1 s) | about 24 s | one zygote run and two paired runs against plain respawn, 1 and 2 GPUs (2026-11-07) |
| Plain respawn (`reset("respawn")`, spawned workers) | lab host | 43.0-45.0 s (measured, paired) | 58.1-62.1 s (first video 15.1-17.1 s) | about 47 s | plain respawn run (2026-11-07) |
| Warm restart with path-stable caches (fresh process; FlashInfer, Triton, Inductor caches at a fixed path, autotune cache, `torch_key` preset) | page cache, JIT caches | 51.4 s to ready (18.1 s imports + 33.3 s load, measured) | 66.4 s (first video 15.1 s; its per-process JIT 2.2 s instead of 6.3 s) | about 58 s | fresh-process runs with path-stable caches (2026-11-07) |
| Restart after the JIT cache root moved | page cache | 58-60 s to ready (measured) | moved, unfixed: 223.4 s (first video 174.0 s; FlashInfer rebuild 153-159 s), also through a re-pointed symlink (first video 170.0 s); staged back at its real path: first video 37.9 s on the cold copy (FlashInfer 11.8 s, no ninja step re-ran), 17.1 s on the next start (FlashInfer 0.60 s) | about 66 s once staged | moved-cache and staged-back runs (2026-11-07) |
| Cold restart | nothing | 289 s to ready (93.5 + 195.6, measured earlier) | about 507 s (first video 217.9 s) | | ladder above |

**Re-measured on 2026-11-19** (one GB200 tray, newer environment: torch 2.12, CUDA 13; weights
on a node-local memory filesystem, so loads are 20-26 s instead of 60-150 s; other H3 runs on
the tray's other GPUs; `tools/h3_climb.py`, every video bit-exact to base):

| Action | 1 GPU | 2 ranks | First video after it |
|---|---|---|---|
| Live swap of one hill-climb patch from its file (`install({"diff": ...})`) | 0.02-0.13 s | 0.05-0.26 s | a warm video (a patch that captures CUDA graphs: +0.5-1.4 s once) |
| In-place rebuild of `minimax_h3.py` with patches 15, 16, 17, 20 | 1.12 s | | 6.09 s |
| Zygote respawn with weights holder | 3.05-3.51 s (six respawns) | 6.23-7.27 s | 12.3-13.7 s |
| Zygote respawn | 13.8-14.5 s | | 11.3-11.6 s |
| Plain respawn (`reset("respawn")`) | 21.2-23.7 s | 29.7 s | 10.9-11.4 s |
| Respawn after a poisoning fault (conformance c8) | 22.2-53.5 s | 29.7-59.2 s (plain) / 6.08-9.65 s (`--zygote --holder` inline CPU IPC + fast-fail: crash 9.65 s, hang 9.21 s, lost 6.08 s, tamper 6.52 s; `24/24` holder hits, 0 `Broken pipe`) | |

- **`SP2 --zygote --holder` conformance recovery and soak (2026-11-19, 2× GB200):**
  - Initial `5/5` run (`fastrsi conformance fastvideo --real --only c0_repeatable,c2_visible,c4_uninstall,c8_crash_and_hang,c9_soak -- --gpus 2 --zygote --holder`, `66` receipts, `0 MiB` on GPUs 0–3): `c0` `25.732 s` (holder deposit `10.57–10.99 s` text encoder / `0.07–0.09 s` VAE / `1.15–1.45 s` DiT), `c2` `9.632 s`, `c4` `18.786 s` (undo `0.016595 s`), `c9` `197.869 s` (20 cycles; install median `0.0589 s`, uninstall `0.0175 s`). Across all four `c8` respawns (`1362.848 s` total case wall), every rank re-mapped all three held modules via holder IPC (`text_encoder` `0.145–0.358 s`, `vae` `0.032–0.063 s`, `transformer` `0.185–0.738 s`), with `lost` (`6.235 s`) and `tamper` (`5.727 s`) fast, while `crash` (`38.178 s` respawn) and `hang` (`37.991 s` respawn) waited ~`600 s` each on the default NCCL watchdog for the SP2 collective.
  - With asymmetric `SP >= 2` fault-beacon detection + parallel dirty `kill_tree(..., include_pg=False)` fast-fail (`adapter/agent.py` + `adapter/lab.py`) and `_HolderPickler` inline CPU storage serialization (`services/holder.py`, bypassing `torch_shm_manager` so killing a depositor never poisons subsequent `get`s with `RuntimeError: Broken pipe`), `4/4` (`c0, c2, c4, c8`, `0 MiB × 4`) ran cleanly: `c0` `~25 s`, `c2` `9.451 s`, `c4` `18.798 s` (undo `0.016544 s`), **`c8_crash_and_hang` `214.018 s` (`6.37×` vs `1362.848 s`)**: `crash` respawn `9.647 s`, `hang` `9.208 s`, `lost` `6.081 s`, `tamper` `6.516 s`. All **`24/24`** respawn fetches (`4` respawns × `2` ranks × `3` components) hit holder IPC cleanly (`text_encoder` `0.146–0.333 s`, `vae` `0.033–0.036 s`, `transformer` `0.256–0.393 s`) with **0 `Broken pipe` / `holder_error` lines**.

- **In-place rebuild** covers the edits the live swap refuses (a constructor, a new field, a
  module constant) on the H3 stages and the H3 DiT: a stage constructor edit (4 modules
  re-imported), a DiT block constructor edit that adds a field (merged into 50 live blocks), a
  DiT edit that changes `norm1.eps` (output changed, `9eadd39e7674caf0`), and a revert
  (bit-exact again). The first video after a rebuild costs what the next warm video costs
  (paired: 10.9 vs 13.3, 12.4 vs 12.4, 11.0 vs 11.8, 15.3 vs 15.0 s); the only JIT
  work is one Dynamo recompile of the DiT's `@torch.compile` helper (0.08-0.09 s, Inductor cache
  hit). Cost, ranks in parallel and each file's imports parsed once per process: DiT edits
  0.19-0.64 s (re-import 0.04-0.17 s, meta builds of the old and new DiT 0.13-0.47 s), a stage
  edit 1.16-1.27 s (the process's first import graph: 1.1 s), stages < 1 ms.
- **State the old code held is released** (`rebuild.release_held`). A module that keeps
  per-layer CUDA graphs on its functions would otherwise keep them alive next to the new
  code's captures, and the next video runs out of GPU memory (`lessons.md`). Each of 4 rebuilds of `minimax_h3.py` on the kept tree
  released 100 CUDA graphs and 62.3 GiB of static buffers, reserved memory stayed at
  158.7-158.9 GiB, every video bit-exact (with the kept H3 patches).
- **Zygote respawn, where its 14 s go** (1 GPU, paired run, forked worker timers): component
  loads 10.9 s (VAE 5.0 s, DiT 2.45 s, text encoder 2.35 s, audio VAE 0.56 s, tokenizer and
  processor 0.5 s, schedulers 0.2 s), CUDA context, device and distributed init 0.17 s, the rest
  (old group shutdown, fork, ready handshake) about 2.6 s. Weights the holder serves (DiT and
  text encoder) are 4.8 s of it.
- **Weights holder:** the respawned worker received the text encoder in 0.19-0.21 s and the DiT
  in 0.48-0.76 s (module objects, CUDA tensors mapped by IPC) instead of 2.3 s and 2.1-2.5 s; the
  forked worker's whole init fell from 11.5-12.5 s to 7.8 s, the VAE (5.0 s) now its largest
  part. After an edit of `component_loader.py` both entries were refused (`holder_stale`, naming
  the file), loaded from disk and deposited again (respawn 17.65 s, including the re-import of
  the whole package the edit reached), and served again at the next respawn. At 2 ranks each
  rank's entries were served (text encoder 0.18-0.36 s, DiT 0.26-0.48 s): respawn 13.56-14.56 s
  against 18.30-18.33 s without the holder (paired runs, distributed init 1.9-2.2 s each time),
  every first video bit-exact. With the VAE held too, the VAE came in
  0.05-0.09 s instead of 5.0-5.3 s, the forked worker's init took 2.25-2.7 s, and the respawn
  4.78 s and 5.62 s. The audio VAE holds a non-leaf tensor that requires grad, which torch will
  not share across processes: its deposit fails and the worker loads it (0.9 s).
- **Zygote:** the server imports 357 modules once at lab start (16-21 s; 112 s once on a busy
  filesystem). It must not touch the CUDA runtime: FastVideo's `torch.cuda.is_available()` and
  importing `flashinfer.jit` both do (fixed with an NVML-based check and Triton's target pinned
  from NVML; `fork_poisoned()` checks it with a probe fork).
- **Warm restart with path-stable caches alone does not make a restart cheap:** it takes the
  first video's JIT from 6.3 s to 2.2 s, but a fresh process still pays 17-38 s of imports and
  30-35 s of loads.
- **FlashInfer's cache is keyed on the absolute path.** `build.ninja` names every output by its
  absolute path under `FLASHINFER_WORKSPACE_BASE` (`flashinfer/jit/core.py: write_ninja`), and
  ninja's log records outputs and command hashes by those paths, so a copy at another path
  rebuilds (153-159 s). The path is the *resolved* one (`flashinfer/jit/cpp_ext.py`,
  `(output_dir / obj_name).resolve()`): a fixed symlink re-pointed at the copy rebuilt too
  (155.2 s, first video 170.0 s). The fix keeps the real path: build the root at a
  fixed path and stage shipped copies back at it (`jitcache.stage_root`, mtimes kept). Receipt:
  built cold at a path (FlashInfer 153.2 s, first video 169.0 s), moved away, copied
  back to the same path: no ninja step re-ran (logs unchanged), FlashInfer 11.8 s on the first
  start from the cold copy and 0.60 s on the next; bit-exact. Keying FlashInfer on content would
  need its ninja files written with relative paths (an upstream change), not done.

## Shortened 2-forward (`num_inference_steps=3`) H3 screen (`1× GB200`, `2026-11-19`)

Measured separately on `1× GB200` (`FastVideoLab(zygote=True, holder=True)` on `FastH3-QAD-v2` NVFP4; kept strictly distinct from the `GB10` `t2va` race numbers):

| Path | Setup / Scope | Total Wall | Notes / Witness |
|---|---|---:|---|
| `fresh_full` | Fresh worker + `9` sigma (`8` DiT forwards, `124` frames) | **`44.295 s`** | Warm disk caches; output `d0e55e892690e260 / ee49674663eae6e2` |
| `warm_full_base` | Warm zygote+holder lab first full check | `12.893 s` | Matches `fresh_full` bit-for-bit |
| `live_full` | Live `MiniMaxH3VideoDecodingStage.forward` swap + full check | **`6.398 s`** (`6.9×` vs fresh) | `0.0382 s` install + `6.360 s` check, `0.0110 s` undo, witness `1`, `cd32f50b70d9479f`, bit-exact undo verified |
| `live_probe_2fwd` | `num_inference_steps=3` (`dmd_denoising_steps=[999, 874]` via `repl`: `2` DiT forwards vs `8`) | **`2.356 s`** (`18.8×` vs fresh) | `base_probe 2.177 s`, `0.0329 s` install + `2.323 s` check, `0.0106 s` undo, witness `1`, `474d5ebd470a6063 -> b2a414f50233cb60`, **`undo_exact: true`**, `0 MiB × 4` |

Multi-GPU SP (`SP=2` and `SP=4`, `4× GB200`, `FastVideoLab(zygote=True, holder=True)` on `FastH3` NVFP4, `base_steps=9`, `0 MiB × 4` teardown):

| Topology | `load_s` (zygote) | `fresh_full` (`subproc_wall_s`) | `warm_full_base` | `live_full` (install + check, undo) | `live_probe_2fwd` (`num_inference_steps=3`, install + check, undo) |
|---|---:|---:|---:|---:|---:|
| **`SP=2` (`2× GB200`)** | `45.924 s` | **`49.904 s`** (`24.747 s` import/ready + `10.531 s` first video) | `16.267 s` | **`4.771 s`** (**`10.5×`** vs fresh; `0.079 s + 4.692 s`, undo `17.5 ms`) | **`1.859 s`** (**`26.8×`** vs fresh; `0.055 s + 1.804 s` vs `base_probe 1.767 s`, undo `16.4 ms`, `undo_exact: true`, `full_restored_after_probe: true`) |
| **`SP=4` (`4× GB200`)** | `47.211 s` | **`56.042 s`** (`29.628 s` import/ready + `11.871 s` first video) | `12.554 s` | **`3.578 s`** (**`15.7×`** vs fresh; `0.139 s + 3.439 s`, undo `32.2 ms`) | **`1.670 s`** (**`33.6×`** vs fresh; `0.108 s + 1.562 s` vs `base_probe 1.504 s`, undo `31.6 ms`, `undo_exact: true`, `full_restored_after_probe: true`) |

Full `11/11` `H3Lab` conformance under `--zygote --holder` with inline `_HolderPickler` CPU IPC (`4× GB200`, `108` receipts per run, `0` `Broken pipe`, `0` `holder_error`, `0 MiB × 4` teardown):
- **`SP=2 --zygote --holder` (`11/11` pass, `856 s` total):** `c0 25.789 s`, `c1 9.418 s`, `c2 9.440 s`, `c3 160.040 s` (bit-exact visible `cd32f50b70d9479f` & 3-chain `23bd4a78af68a546`), `c4 18.679 s` (`17.55 ms` undo), `c5 18.964 s`, `c6 23.915 s`, `c7 37.602 s`, `c8_crash_and_hang 208.127 s` (respawns `9.001 / 9.323 / 5.990 / 6.553 s`), `c9_soak 197.727 s` (`0.0579 s` install / `0.0177 s` undo), `c10 57.449 s`.
- **`SP=4 --zygote --holder` (`11/11` pass + earlier `5/5` `[404 s]`):** `c0 24.391 s`, `c1 6.806 s`, `c2 6.814 s`, `c3 177.869 s` (bit-exact visible `94849c72941908c3` & 3-chain `837c08243df1ce30`), `c4 13.423 s` (`31.99 ms` undo), `c5 13.907 s`, `c6 18.281 s`, `c7 27.210 s`, `c8_crash_and_hang 226.499 s` (respawns `13.023 / 13.668 / 10.077 / 10.195 s`), `c9_soak 144.204 s` (`0.1121 s` install / `0.0329 s` undo), `c10 61.549 s`.

## Non-invasive `FastH3-QAD-v2` phase-dissection waterfall (`SP=1` and `SP=2`, `4× GB200`, `2026-11-20`)

Source: paired non-invasive `sitecustomize.py` + `FastVideoLab(zygote=True, holder=True)` phase dissection on `FastH3-QAD-v2` NVFP4 (`480×832`, `124` frames, `9` sigma / `8` DiT forwards full check; `num_inference_steps=3` / `2` DiT forwards shortened screen; `0.000 s` `not_split` across every row, critical-path max-rank accounting with zero parallel double-counting on `SP=2`, `0 MiB × 4` cleanup; base output `d0e55e892690e260 / ee49674663eae6e2` identical on `SP=1` and `SP=2`, live edit `MiniMaxH3VideoDecodingStage.forward:109` PSNR `48.133 dB` matching `fresh_warm_edited` bit-for-bit, bit-exact `undo_exact: true`). Off-path warm-up: `zygote_serve` `9.431 s` (`SP=1`) / `10.522 s` (`SP=2`) for `357` modules, `holder_start` `1.254 s` / `1.354 s`, initial lab boot `32.651 s` / `39.217 s`.

| Topology | Arm | Total (`s`) | Speedup vs Cold | Critical-Path Phase Breakdown (`s`) |
|---|---|---:|---:|---|
| **`1× GB200` (`SP=1`)** | 1. `fresh_cold` | **`206.226 s`** | `1.0×` | `import_init 9.082` + `spawn_import 8.782` + `weight_load 12.232` + `jit_build 162.915` (`144.95` FlashInfer + `7.37` Triton VSA + `10.60` Inductor/Dynamo) + `cuda_graph_capture 0.026` + `engine_init 1.121` + `first_step 6.039` + `run_workload 6.030` (`208.52 s` equiv single-video wall incl. shutdown) |
| **`1× GB200` (`SP=1`)** | 2. `fresh_warm` | **`40.642 s`** | `5.1×` | `import_init 8.210` + `spawn_import 8.408` + `weight_load 11.092` + `cuda_graph_capture 0.026` + `engine_init 0.829` + `first_step 6.041` + `run_workload 6.036` (`42.75 s` equiv wall) |
| **`1× GB200` (`SP=1`)** | 3. `zygote + holder` respawn #1 (`#2`) | **`17.606 s`** (`17.415 s`) | `11.7×` | `spawn_import 0.240` (`0.242`) + `weight_load 1.172` (`1.130`: `text_encoder 0.166`, `vae 0.037`, `transformer 0.270` IPC + `audio_vae 0.699` disk) + `cuda_graph_capture 0.008` + `engine_init 1.725` (`1.711` incl. `0.776 s` shutdown) + `first_step 8.037` (`7.906`) + `run_workload 6.423` (`6.418`); `reset("respawn")` alone: **`3.105 s`** / **`3.052 s`** |
| **`1× GB200` (`SP=1`)** | 4. `live_r0_full` | **`6.404 s`** | **`32.2×`** | `live_install 0.0284` (`prepare 20.1 ms`, `gap 0.78 ms`, `undo 7.42 ms`, witness `1`) + `run_workload 6.375` (`cd32f50b70d9479f`, `48.133 dB` PSNR, bit-exact to `fresh_warm_edited`) |
| **`1× GB200` (`SP=1`)** | 5. `live_r0_screen` (`2-fwd`) | **`2.246 s`** | **`91.8×`** | `live_install 0.0264` (`prepare 18.8 ms`, `gap 0.85 ms`, `undo 6.96 ms`) + `run_workload 2.220` (`474d5ebd470a6063 -> b2a414f50233cb60`) |
| **`2× GB200` (`SP=2`)** | 1. `fresh_cold` | **`200.067 s`** | `1.0×` | `import_init 10.931` + `spawn_import 9.159` + `weight_load 11.536` + `jit_build 154.450` (`141.70 s` `FileLock`-serialized FlashInfer + `7.28 s` Triton VSA + `5.47 s` Inductor/Dynamo) + `cuda_graph_capture 0.026` + `engine_init 3.757` + `first_step 5.850` + `run_workload 4.358` (`203.53 s` equiv wall) |
| **`2× GB200` (`SP=2`)** | 2. `fresh_warm` | **`45.979 s`** | `4.4×` | `import_init 10.593` + `spawn_import 9.073` + `weight_load 12.042` + `cuda_graph_capture 0.026` + `engine_init 4.044` + `first_step 5.851` + `run_workload 4.351` (`49.04 s` equiv wall; `49.96 s` first `SP=2` after `SP=1`) |
| **`2× GB200` (`SP=2`)** | 3. `zygote + holder` respawn #1 (`#2`) | **`19.426 s`** (`18.483 s`) | `10.3×` | `spawn_import 0.045` (`0.044`) + `weight_load 1.403` (`1.342` bottleneck rank) + `cuda_graph_capture 0.008` + `engine_init 5.811` (`4.852`) + `first_step 7.255` (`7.313`) + `run_workload 4.904` (`4.925`); `reset("respawn")` alone: **`6.598 s`** / **`5.531 s`** |
| **`2× GB200` (`SP=2`)** | 4. `live_r0_full` | **`4.755 s`** | **`42.1×`** | `live_install 0.0546` (`prepare 38.0 ms`, `gap 1.87 ms`, `undo 13.2 ms`, witness `[1, 1]`) + `run_workload 4.701` (`48.133 dB` PSNR, bit-exact to `fresh_warm_edited`) |
| **`2× GB200` (`SP=2`)** | 5. `live_r0_screen` (`2-fwd`) | **`1.867 s`** | **`107.2×`** | `live_install 0.0513` (`prepare 36.9 ms`, `gap 1.07 ms`, `undo 13.1 ms`, witness `[1, 1]`) + `run_workload 1.815` (`474d5ebd470a6063 -> b2a414f50233cb60`) |

## Concurrent holder, non-leaf `audio_vae` IPC, pre-fork `torch_key`, and Triton autotune cache (`SP=1/2/4`, `4× GB200`, `2026-11-20`)

Source: Linux GPU `pytest repos/fastvideo/livelab/services/` (**`33 passed, 1 skipped`**, `0` `ResourceWarning`s) plus `FastH3-QAD-v2` NVFP4 (`not_split = 0.0000 s`, `0 MiB × 4` teardown) across `SP=1`, `SP=2`, and `SP=4` after enabling concurrent threaded IPC in `services/holder.py`, non-leaf `.detach()` serialization for `audio_vae`, pre-fork `torch_key()` in `services/zygote.py`, and `TRITON_CACHE_AUTOTUNING=1` in `adapter/h3.py`:
- **Pre-fork `torch_key` + Triton autotune elimination:** `zygote._serve` computed `torch_key` cleanly before fork (`0.280 s` `SP=1`, `0.303 s` `SP=2`, `0.332 s` `SP=4`; `poison=[]`, `driver_initialized=False`, `torch_cuda_initialized=False`), cutting in-worker `inductor_torch_key_end` (`56` calls) from `1.422 s` / `1.251 s` to **`0.17–0.18 ms`** (`inductor_compile_fx 1.64 s -> 0.34–0.48 s`), while `TRITON_CACHE_AUTOTUNING=1` reduced `triton_autotune_bench` from `12` calls (`1.851 s` `SP=1` / `1.692 s` `SP=2`) to **`0` calls (`0.000 s`)**.
- **`audio_vae` CUDA IPC & concurrent multi-rank `holder.serve()`:** `audio_vae` logged **`"served": true`** on every rank of `SP=1`, `SP=2`, and `SP=4` across both respawns while preserving bit-exact audio (`ee49674663eae6e2` `SP=1/2`, `69356c2116d3c58a` `SP=4`) and video (`d0e55e892690e260` `SP=1/2`, `0134242de77c3f4c` `SP=4`). `SP=1` `weight_load` fell from `1.172 / 1.130 s` to **`0.884 / 0.862 s`** (`-24.5%`); `SP=2` overlapped `2/2` concurrent `get` calls (`sum 2.682 / 2.285 s -> wall 1.900 / 1.685 s`); `SP=4` overlapped `4/4` concurrent `get` calls (`sum 7.847 / 7.836 s -> wall 2.847 / 2.975 s`, bottleneck `weight_load 2.373 / 2.461 s` vs `14.339 s` disk, **`6.0×`**).
- **Updated `zygote + holder` respawn -> 1st video (`#1 / #2`) and live speeds (`not_split = 0.0000 s`):**
  - **`SP=1` (`fresh_warm 46.309 s`):** respawn wall **`2.867 / 2.849 s`**, `first_step` **`4.936 / 4.802 s`** (`1.63×` vs `8.037 s`), total to 1st video **`14.309 / 14.222 s`** (**`1.23×`** vs prior `17.606 / 17.415 s`, **`14.4×`** vs `206.226 s` cold), `live_r0_full 6.415 s` (`48.133 dB` PSNR), `2-fwd 2.300 s`.
  - **`SP=2` (`fresh_warm 41.015 s`):** respawn wall **`6.165 / 6.163 s`**, `first_step` **`4.332 / 4.412 s`** (`1.68×` vs `7.255 s`), total to 1st video **`16.019 / 16.095 s`** (**`1.21×`** vs prior `19.426 / 18.483 s`, **`12.5×`** vs `200.067 s` cold), `live_r0_full 4.765 s` (`48.133 dB` PSNR), `2-fwd 1.971 s`.
  - **`SP=4` (`fresh_warm 48.874 s`):** respawn wall **`11.122 / 10.365 s`**, `weight_load 2.373 / 2.461 s`, `first_step 4.764 / 4.765 s`, total to 1st video **`21.873 / 21.136 s`**, `live_r0_full 3.348 s` (`48.133 dB` PSNR), `2-fwd 1.663 s` (`29.4×` vs warm).

## Zygote `OPTIONAL_PRELOAD` (FlashInfer + NVFP4 VAE decoder) under `forksafe` guards (`4× GB200`, `2026-11-21`)

Preloading FlashInfer and the NVFP4 VAE decoder via `OPTIONAL_PRELOAD` under `forksafe` guards on `FastH3-QAD-v2` NVFP4 (`4× GB200`):
- **`conditioning_stage` speedups:** **`2.32×`** on `SP=1` (`2.583 -> 1.111 s`), **`2.48×`** on `SP=2` (`2.676 -> 1.079 s`), and **`2.03×`** on `SP=4` (`2.299 -> 1.130 s`).
- **`first_step` & live screen:** `first_step` cut **`1.38× / 1.42–1.51× / 1.29–1.33×`** (**`3.355 s`** `SP=1`, **`3.209 s`** `SP=2`, **`3.847 s`** `SP=4`); `live_r0_screen` **`2.30 / 1.86 / 1.53 s`** (`48.133 dB` PSNR bit-exact, `poison=[]`).
- **Linux GPU `pytest`:** unfiltered combined `repos/fastvideo/livelab` **`70 passed, 1 skipped`** (`repos/fastvideo/livelab/services/` standalone **`39 passed, 1 skipped`**).
