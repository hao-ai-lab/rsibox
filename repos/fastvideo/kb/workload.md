# FastVideo target workload

## F(x): one FastH3 text-to-video+audio generation

- **Model:** FastH3-QAD-v2 (MiniMax H3 distilled to 8 DiT forwards). NVFP4 DiT linears, BF16
  video sparse attention (`VIDEO_SPARSE_ATTN_H3`, sparsity 0.8, tile 64), NVFP4 Qwen3-VL text
  encoder, lightweight H3 VAE with the NVFP4 fused decoder, audio VAE. Fused H3 helpers on
  (`FASTVIDEO_MINIMAX_H3_FUSIONS=all`). FP4 attention off (`h3_vsa_fp4.enabled: false`): the
  pinned `fastvideo-kernel` lacks the FP4 VSA tile API it calls (an earlier GB200 run, 2026-10-20).
- **Config:** FastVideo's `examples/inference/basic/basic_fasth3_spark_qad_v2_nvfp4.yaml` at the
  pin, with FP4 attention off, `num_gpus = sp_size = N` (sequence parallel), `tp_size = 1`, the
  multiprocess executor (also at N = 1, see `livelab/README.md`), no offload, no lazy module load,
  DiT eager (`compile.enabled: false`), VAE compile on.
- **Request:** prompt "A quiet pottery studio with a potter finishing a bowl at the wheel.",
  seed 1234, 480x832, 124 frames at 24 fps, 9 sigma points (8 DiT forwards), guidance 1.0.
- **Parallelism:** N = 1, 2, 4 GPUs of one GB200 tray. Hill-climb on N = 1 first.
- **Frozen:** checkpoint, quantization, config above, the request, and the timing code
  (`livelab/adapter/lab.py: check`). An edit may change engine code only.

## Metric

- **Time per video:** wall seconds of `VideoGenerator.generate` for the request, warm (after
  the first video in the process), measured by the lab's `check` around the call. Includes text
  encoding, denoising, video and audio decode, and the uint8 frames crossing from worker to
  driver. Score: speedup over the base commit, same N, same lab session or a fresh process.
- **Time per denoise step:** the denoise stage seconds divided by 8 (stage logging), reported
  next to it. Not the score.

## Correctness gate

- **Repeatable base:** the base must give bit-identical frames and audio across runs in one
  process and across processes (conformance c0 and c3). Measured: see `costs.md`.
- **Golden per GPU count:** 1 and 2 GPUs share one golden (bit-identical); 4 GPUs produce a
  different, repeatable video (`costs.md`), so it has its own golden.
- **Exact edits:** an edit meant to keep numerics (scheduling, copies, buffers, Python
  overheads) must keep the video and audio hashes identical to base.
- **Numerics-changing edits** (kernel or fusion changes, dtype or reduction order): video PSNR
  against the base golden frames of at least **40 dB**, and the audio hash may change only if
  the edit touches the shared DiT path. Below 40 dB the edit is rejected whatever its speed.
  The threshold is a starting choice, not a measured quality bound; a lower one needs an
  owner's decision.
- **Fresh confirmation:** a kept edit counts only after a fresh process started from the edited
  files gives the same output hashes as the live lab (the lab's `fresh_check`).
