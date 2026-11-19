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
| Respawn after a poisoning fault (conformance c8) | 22.2-53.5 s | 29.7-59.2 s | |

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
