# MiniMax H3 hill-climb: bit-exact edits to FastVideo (1 and 2 GPUs)

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

## Patches 09-19: the 2-GPU climb (sequence parallel 2)

A second climb the next morning (2026-11-07, 08:00-12:00 PST) on 2 GPUs, on top of 01-08, Codex
only (propose-only, rotating two accounts): 58 iterations, 48 measured, 11 kept. Same gate.
Apply 01-19 in order; together they are the set measured below.

Fresh processes, our 2026-11-07 12:03-12:38 PST runs, base and patched alternating twice,
1 cold + 5 warm videos each, every video bit-identical to the golden:

| GPUs | Base | Patches 01-19 | Speedup |
|---|---|---|---|
| 2 (sequence parallel) | 4.41 / 4.42 s | 2.93 / 2.88 s | **1.52x** |
| 1 | 6.10 / 6.12 s | 4.20 / 4.23 s | **1.45x** (the same as 01-08: 09-19 neither help nor hurt at 1 GPU) |

| # | Edit | s/video in the lab | Edit-to-run (in the lab) | Antigravity | Review notes and upstream needs |
|---|---|---|---|---|---|
| 09 | SP=2: pack q/k/v/gate for the all-to-all and scatter received rows straight into the tile buffer (one cached Triton helper) | 3.650 -> 3.568 | 1.83 s (0.27 s) | correct | Specialised to SP=2; generalise or guard by world size (it falls back otherwise) |
| 10 | Fold the gated merge and untile into the return all-to-all's packing | 3.639 -> 3.414 | 1.52 s (0.32 s) | correct | Same |
| 11 | Exchange only the remote half of q/k/v/gate heads; tile local rows in place | 3.615 -> 3.446 | 1.45 s (0.28 s) | correct | Replaces the collective's self-copy; keep a test at SP=4 before upstreaming |
| 12 | Write local gated heads straight into the final output; return only remote heads | 3.568 -> 3.475 | 1.22 s (0.29 s) | correct | Same |
| 13 | Generic Ulysses packing and local placement, exchanging only remote data | 3.032 -> 2.989 | 0.61 s (0.03 s) | correct | Generalises 09-12 |
| 14 | Patch 08's compact video-only top-k extended to SP=2 | 3.194 -> 2.999 | 1.30 s (0.25 s) | correct | Patch 08's tie test applies |
| 15 | Replay the unchanged NVFP4 feed-forward per layer from a CUDA graph (signature: weight pointers and versions, shapes, stream, autocast; eager fallback on any change or error) | 3.010 -> 2.957 | 0.91 s (0.20 s) | **correct for this workload only** | Antigravity: FastVideo's hook manager (activation tracing, shape tracking) wraps `forward` outside PyTorch's hook lists, so tracing breaks the capture or silently drops records on replayed steps; `FASTVIDEO_NVFP4_MM_BACKEND` / `_ROW_SPLIT` are read per call but not in the signature, so changing them replays a stale graph. Claude review: exact here; per-layer graph pools cost tens of GB per GPU; alternating resolutions re-capture every time. Needs an opt-in flag, a memory budget, and both fixes |
| 16 | Feed the FFN graph's output straight into the gated residual (no clone) | 3.009 -> 2.940 | 0.90 s (0.20 s) | correct | Claude review: safe only while every consumer is done with the output before that layer's next replay; true for the denoise loop as written, not enforced |
| 17 | Replay the NVFP4 QKV and gate projections plus fused qk-norm/RoPE from per-layer CUDA graphs | 3.005 -> 2.959 | 1.10 s (0.22 s) | correct | Same caveats as 15 |
| 18 | Fused strided q/k/v/gate remote packing and local tiling; scatter only remote rows | 2.958 -> 2.902 | 1.89 s (0.26 s) | correct | |
| 19 | Contiguous row/head partitions for the SP=2 gated return | 2.931 -> 2.901 | 1.25 s (0.26 s) | pending | |

Lab seconds per video drift between iterations (the baseline is the median of the recent
baselines on the same engine); the fresh-process table is the result.

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
