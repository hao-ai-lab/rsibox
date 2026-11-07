# MiniMax H3 hill-climb: eight bit-exact edits to FastVideo

Edits to FastVideo `376e445f0e8d192c2d9a5f28010a6b3a7012667e` that developer agents found and
the live lab kept, our 2026-11-07 overnight run on one GB200 GPU (workload: `kb/workload.md`).
Apply in order with `git apply` from a FastVideo checkout at that commit; together they are the
set the fresh-process numbers below measure. None is proposed upstream yet.

## Result (fresh processes, no live lab)

Our 2026-11-07 07:20 PST runs: a new FastVideo process per run, FastVideo's own executor, 1 cold
+ 5 warm videos per run, base and patched runs alternating twice. Every video of every run had
frames and audio bit-identical to the golden of its GPU count.

| GPUs | Base, median of 5 (two runs) | Patches 01-08 (two runs) | Speedup |
|---|---|---|---|
| 1 | 6.09 / 6.16 s | 4.29 / 4.24 s | **1.44x** |
| 2 (sequence parallel) | 4.40 / 4.78 s (second run noisy: 4.37-5.12) | 3.56 / 3.55 s | **1.24x** against 4.40 |
| 4 (sequence parallel) | 3.16 / 3.18 s | 3.16 / 3.03 s | within noise |

Patch 04 and the attention paths of 06-08 are single-rank only by design (they fall back when
the sequence-parallel size is above 1), which is why 2 and 4 GPUs gain less.

## How they were found and kept

- A developer agent edits a copy of the FastVideo source and installs each edit into the warm
  H3 engine through the live lab (`livelab/`): R0 swap, no restart. A harness, not the agent,
  measures and decides: keep only if frames and audio stay bit-identical to base (sha256),
  every changed function ran on the new transaction (witness), and the video got at least 1%
  faster than a baseline measured on the same engine just before.
- Agents: Claude subagents (patches 01-05, 7 iterations, 5 kept) and Codex (06-08, 166
  iterations, 107 measured, 4 kept by the harness, 3 in this set). Codex ran propose-only: no
  shell, read-only, given the source files and a profile, returning whole new definitions the
  harness installed.
- Antigravity reviewed every kept edit for correctness beyond this one request; a Claude
  review is in the last column.

## Ledger

Seconds per video are the lab's warm medians on the iteration's engine instance (the baseline
is the one the harness judged against). Edit-to-run: from the agent's edit to the commit of the
live install, as the lab's receipt records it.

| # | Edit | Agent | s/video in the lab | Edit-to-run | Antigravity | Review notes and upstream needs |
|---|---|---|---|---|---|---|
| 01 | VSA-H3 `tile()`: take FastVideo's existing heads-first Triton scatter by default (it was behind `FASTVIDEO_H3_VSA_HEADS_FIRST_TILE`), replacing an `index_put` and three transpose copies | Claude | 6.031 -> 5.112 | 0.96 s | correct | Upstream: flip the default instead of deleting the flag check; setting the variable to 0 no longer disables the route |
| 02 | qknorm + partial RoPE: one program per 16 heads (was one per head row), reduction tree in the same order as `tl.sum` | Claude | 5.112 -> 4.691 | 0.72 s | correct | Built lazily in a cached function (the live lab's witness needs a Python call); upstream can be a plain top-level `@triton.jit` |
| 03 | Fused attention-output copy + gated `addcmul_` in one Triton pass; skip the zero-width `F.pad` | Claude | 4.691 -> 4.520 | 5.69 s, 3.67 s (queued behind another lab call; 0.12 s, 0.05 s in the lab) | **correct for this workload only** | Antigravity: the fused path drops the autograd path when only `gate_compress` needs grad (training); the `triton` import is unguarded; no `torch.cuda.device` guard. Fix all three before upstreaming |
| 04 | One rank: scatter q/k/v/gate straight into the tile buffer, no stacked `torch.cat` copy | Claude | 4.520 -> 4.381 | 1.04 s | correct | Single-rank only by design |
| 05 | Vectorized bool-mask -> (q2k_idx, q2k_num) kernel; calls `block_sparse_attn_from_indices` directly | Claude | 4.381 -> 4.329 | 1.09 s | correct | Better upstreamed into `fastvideo-kernel`'s `map_to_index` |
| 06 | Packed-output gated merge + untile in one kernel on the single-rank tile-64 path | Codex | 4.386 -> 4.199 (baseline measured right after a lab restart; earlier A/B baseline on the same set 4.276) | 1.15 s | correct | Claude review: exact and well guarded, but it re-implements `MiniMaxH3VSAImpl.forward`'s path (pooling, scores, mask, attention, compression) in a new function, so a later change to `forward` would silently diverge; it adds module globals (`_h3_triton`, `_h3_tl`) at run time; the agent's whole-definition rewrite dropped the docstring and comments of `DistributedAttention_VSA.forward`. Not upstreamable as is |
| 07 | The fused merge reads the original packed gate, no gate tiling | Codex | 4.248 -> 4.182 | 1.91 s | correct | Extends 06; same upstream caveats |
| 08 | Compact video-only score rows for single-block top-k; no prefix-query selection or output sorting; scatter region-local indices | Codex | 4.199 -> 4.153 | 0.98 s | correct | Guarded to batch 1, SP 1, 256-2048 tiles; gives a bit-exact video here; it skips sorting the selected indices, so add a test that compares its mask with `torch.topk`'s on tied scores before upstreaming (not checked) |

At 1 GPU the fresh-process result of 01-08 (4.24-4.29 s) is no better than an earlier
fresh-process run of 01-05 (4.26 s, our 2026-11-07 01:24 run): the lab-measured gains of 06-08
(1-2% each) are within fresh-process noise at 1 GPU. At 2 GPUs the full set beat 01-05 (3.55 vs
3.61-3.90 s).

## Not in this set

- **Kept by the harness, gain not established:** worker B's iteration 38 (8-row SwiGLU tiles,
  Codex, Antigravity "correct"): judged against a baseline taken right after a lab restart
  (4.354 s, where that lab's baselines were otherwise 4.17-4.24 s), so its gain is likely under
  0.5%. Two earlier keeps of this kind (iterations 9 and 31) were re-measured by A/B and reverted
  (+0.84% and about +0.5%).
- **Tried, exact, under 1% or slower:** about 100 Codex and Claude edits: SwiGLU tilings, register
  caps, persistent grids, modulation row batching, stream overlap of independent attention
  kernels, Python dispatch shortcuts.
- **Failed:** CUDA-graph capture of the DiT stack (crashed the next video; the lab respawned);
  a top-k mask kernel whose undo poisoned the lab (a kernel gap: undo of an edit that added a
  cached Triton-kernel builder is refused with `closure_not_restored`).
