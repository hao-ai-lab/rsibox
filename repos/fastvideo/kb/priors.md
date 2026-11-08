# FastVideo priors: MiniMax H3 / FastH3 inference

The sources the knowledge base builds on: the pinned FastVideo tree, earlier runs, and the real
H3 performance PRs the lab replays. The architecture read from the pin is in
`../docs/architecture.md`; measured costs are in `costs.md`.

- **Workload:** FastH3-QAD-v2 checkpoint. NVFP4 DiT linears, BF16 video sparse attention
  (VSA_H3), NVFP4 Qwen3-VL text encoder, lightweight H3 VAE, audio VAE. 480x832, 124 frames,
  9 sigma points = 8 DiT forwards. 1, 2 or 4 GPUs (sequence parallel).
- **Pin:** FastVideo `376e445f0e8d192c2d9a5f28010a6b3a7012667e` ("FastH3 OmniRef inference on
  consumer devices (#1980)", committed 2026-10-19). All `file:line` below are at this pin
  unless marked otherwise. Link form:
  `https://github.com/hao-ai-lab/FastVideo/blob/376e445f0e8d192c2d9a5f28010a6b3a7012667e/<path>#L<n>`.
- **Older pins:** an earlier source survey (pin `02a027c`) and an earlier CPU pilot (pin
  `d5253fd`) studied the Wan path and some shared code. A fact from them is marked "older pin"
  and was re-checked against `376e445f` where the text says so.
- **Earlier run:** one GB200 run (2026-10-20), 1 GPU, BF16 VSA: imports 65 s, first video
  53 s end to end (includes lazy compile/JIT), second video 6.1 s, denoise stage 5.5 s.
  `costs.md` has the phase split from our own runs.

## Real H3 performance PRs to replay

Source: `gh pr view`/`gh pr list` on `hao-ai-lab/FastVideo`, 2026-11-07. "In pin" means the
change is in the `376e445f` tree (merged before 2026-10-19 and grep-checked); in-pin PRs are
replayed by revert then reapply, the rest by applying forward. Function group and live-ness are
read from the file list and diff, not run.

| PR | Title (short) | Merged | Files (main) | Group | Live or restart | In pin |
|---|---|---|---|---|---|---|
| #1828 | uint8 frames from the decode worker | 09-25 | `minimax_h3_decoding.py`, `video_generator.py` | OP | Live (decode stage and parent conversion) | Yes |
| #1858 | VSA-H3 tile buffer out of autograd | 09-25 | `video_sparse_attn_h3.py` | OP, STATE (holder) | Live; drop the holder | Yes |
| #1882 | Reuse cached VAE offload for H3 conditioning | 09-25 | `minimax_h3_latent_preparation.py` | OP | Live (uses `pinned_offload`, `:206`) | Yes |
| #1735 | Opt-in Sol-Engine fusions (modulate, qknorm_rope, swiglu) | 08-21 | `minimax_h3.py`, `minimax_h3_fusions/*.py` | OP, GRAPH (Triton), STATE (flag read at init) | Wrapper live; enabling needs block rebuild | Yes |
| #1742 | Opt-in packed-varlen FA4 | 08-23 | `flash_attn.py`, `minimax_h3.py` | STATE (backend choice) | Restart or rebuild attention | Yes |
| #1744 | Parallelize VAE over sequence ranks | 08-23 | `minimax_h3_parallel.py`, `minimax_h3_video.py`, decode/latent stages | RANKS | All ranks together | Yes |
| #1793 | Sequential start with GPU-direct DiT load | 08-31 | `fsdp_load.py`, `minimax_h3_pipeline.py`, `fastvideo_args.py` | RESOURCE, STATE | Restart | Yes |
| #1807 | Harden Ulysses agreement, bounded H3 transfers | 10-07 | `ulysses_all_to_all.cu`, `comm_ops.py`, `ulysses_a2a.py` | NATIVE, RANKS | Side library plus all ranks, else restart | Yes |
| #1900 | Fuse VSA64 tile layout for SM100 inference | 10-07 | `triton_kernels/vsa_tile_layout.py`, `video_sparse_attn.py` | GRAPH (Triton) | Live (re-JIT) | Yes |
| #1930 | Exact Triton RoPE/AdaLN/residual/SwiGLU, heads-first tiles, VAE tile parallelism | 10-07 | `minimax_h3_fusions/exact.py`, `minimax_h3.py`, `video_sparse_attn_h3*.py`, `minimax_h3_video.py`, `memo.py` | OP, GRAPH, RANKS (VAE tile split) | Mixed: kernels live; flags read at init need rebuild | Yes |
| #2040 | Decode eight VAE spatial tiles per call by default | 11-02 | `minimax_h3_video.py`, `minimax_h3_parallel.py`, `envs.py` | OP, config default | Live code; default change is an env/config restart | No (pin has `FASTVIDEO_H3_VAE_TILE_BATCH` default 1, `envs.py:711`) |
| #2041 | Fuse VAE decoder QK RMSNorm and partial RoPE | 11-02 | `minimax_h3_qk_norm_rope.py` (new), `minimax_h3_video.py` | OP, GRAPH (Triton) | Live (re-JIT); enabling flag is an env var | No |
| #2042 | Opt-in faster DiT fused-op tier (`mk`) | 11-01 | `minimax_h3_fusions/mk/*`, `minimax_h3.py`, `attention/layer.py` | OP, GRAPH (Triton), STATE | Kernels live; tier choice at init | No |
| #2046 | Faster FP4 attention padding, uint8 VAE path, cuDNN FP4 GEMMs | 11-02 | `flash_attn.py`, `cudnn_sdpa.py`, `nvfp4_config.py`, decoding stage | OP, RESOURCE (GEMM backend) | Mostly live; backend env var needs restart | No |
| #2052 | Fused Q/K preconditioning + NVFP4 quantize for FA4 (bit-identical) | 11-03 | `flash_attn.py`, `attention/utils/nvfp4_fused_qk_quant.py` | OP, GRAPH (Triton) | Live (re-JIT) | No |

Notes on the table:

- Merge dates are 2026; PRs merged after the pin date (2026-10-19) are absent by construction.
  "In pin" for the others was checked by grepping for a mark of the change (for example
  `_tile_heads_first`, `FASTVIDEO_DISABLE_VSA64_FUSED_LAYOUT`, `pinned_offload.load`,
  `minimax_h3_fusions/exact.py`, `comm_ops.py`); a PR's whole diff was not compared.
- Several PRs touch `minimax_h3.py` and `flash_attn.py`: replay them one at a time on a
  clean tree, since later PRs depend on earlier ones.
- PR bodies carry the authors' own end-to-end numbers on other setups (for example #2042 reports
  DiT 16.54 s to 15.70 s at 768x1344, 6 forwards, 1 GB200). They are not our workload and are
  not priors for it.
- Replay candidates with the most DiT-path weight for the 8-forward workload: #1735, #1930,
  #2042 (fusions), #1900 and #1930 (VSA layout and tiles), #2052 and #2046 (FP4 attention and
  GEMMs). Decode-side: #1828, #2040, #2041.

## Not verified

- Effective `lazy_module_load` and `h3_sequential_load` on the GB200 nodes.
- PR "in pin" status beyond the marker greps above.
