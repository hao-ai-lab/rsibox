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
| install (R0 swap of one stage method, 1 rank) | 0.06 s in the lab | stage, vote, commit in the request gap, fingerprint |
| edit-to-run (receipt `edit_to_run_s`) | 0.72 s | from the edit's timestamp on the agent's machine to commit, incl. the agent's network hop |
| check (one warm video) | 6.00-6.02 s | the workload; same frames and audio hashes as the in-process executor run |
| uninstall | 0.014 s in the lab | undo + fingerprint |
| mailbox round trip, agent to lab | 0.12-0.15 s | file mailbox on a shared filesystem |

The multiprocess executor at N = 1 gave the same output hashes as FastVideo's in-process
executor, and 6.00 s per video against 6.09 s: the frames crossing the process boundary cost
nothing measurable at this size.

**Edit-to-run:** 0.72 s live against about 72 s for a warm restart and about 500 s for a cold one.
