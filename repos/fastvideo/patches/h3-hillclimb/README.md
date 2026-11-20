# MiniMax H3 hill-climb: bit-exact edits to FastVideo (1 and 2 GPUs)

Edits to FastVideo `376e445f0e8d192c2d9a5f28010a6b3a7012667e` that developer agents found and
the live lab kept, our 2026-11-07 overnight run on one GB200 GPU (workload: `kb/workload.md`).
Apply in order with `git apply` from a FastVideo checkout at that commit; together they are the
set the fresh-process numbers below measure. None is proposed upstream yet. Patch 20 is not an
agent's edit: it fixes the out-of-memory failure of 15-17 on a second request length (below);
apply it with 15-17 whenever requests vary in length.

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
| 02 | qknorm + partial RoPE: one program per 16 heads (was one per head row), reduction tree in the same order as `tl.sum` | Claude | 5.112 -> 4.691 | 0.72 s | correct | Built lazily in a cached function originally; top-level `@triton.jit` kernels now also get live launch-witness proof via `core/livelab/kernel/jit.py` `R1` (verified on `_qknorm_partial_rope_kernel` on 1× GB200: `800` witnesses, `0.342 ms` zero-recompile undo) |
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

Re-measured on 2026-11-19 in a newer environment (torch 2.12, CUDA 13; base video hash the same,
audio hash not: `kb/lessons.md`), same protocol with `tools/h3_shapes.py`, trees alternating
twice, other H3 runs on the tray's other GPUs:

| GPUs | Base | 01-19 | 01-20 | Speedup |
|---|---|---|---|---|
| 1 | 6.07 / 6.06, 6.13 / 6.05 s | 4.20 / 4.16, 4.27 / 4.22 s | 4.22 / 4.22 s (and 4.23 / 4.24 s in a pair with 01-19 at 4.22 / 4.19 s) | **1.44-1.45x**, 20 within 1% of 19 |
| 2 (sequence parallel) | 4.51 / 4.54 s | 3.00 / 3.01 s | 2.99 / 3.00 s | **1.51x** |

Live, in one warm base engine (`tools/h3_climb.py`: each patch installed from its file by the
live swap, one video after each, every video bit-identical to base): 1 GPU 6.04 -> 4.22 s
(**1.43x**) with 01-20, 6.03 -> 4.19 s (1.44x) with 01-19; 2 GPUs 4.55 -> 2.99 s (**1.52x**)
with 01-20. Each install took 0.02-0.26 s. Stacking patches 02-08 live with no intermediate
video between installs also runs bit-exact (1 GB200: 4.14 s vs 6.01 s base, 0 `StopIteration`,
per-version nested `linecache` keys in `core/livelab/kernel/swap.py`) and undoes all 7/7
transactions cleanly in 9-11 ms RPC each (0 `closure_not_restored`, restored video 6.09 s
bit-exact; `core/livelab/kernel/journal.py` evicts Triton 3.5+ `_triton_jit_function_registry`
entries and breaks `defaultdict` cycles on undo).

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
| 19 | Contiguous row/head partitions for the SP=2 gated return | 2.931 -> 2.901 | 1.25 s (0.26 s) | correct | |

## Patch 20: the FFN and QKV graphs of 15-17 for any request shape

Patches 15-17 ran out of GPU memory on a second video length in the same process (our
2026-11-07 run; again on 2026-11-19: 124 then 243 frames on 1 GPU, 176.8 GiB allocated, 67.0 GiB
in graph pools; 124 then 362 frames on 2 GPUs, 174.1 GiB). Two causes, both measured:

- **Old graphs stay alive while new ones capture.** Each of the 100 graphs (FFN and QKV of 50
  blocks) has its own private pool and its own static buffers. On a new length, layer 0 replaced
  only its own graph, and only after capturing the new one; layers 1-49 still held theirs.
- **One length's graphs alone are large.** At 124 frames they hold 62.3 GiB on 1 GPU (14.9 GiB
  FFN, 47.4 GiB QKV, measured), mostly static input and output buffers, so at 243 frames a full
  set is about 120 GiB beside the 30 GiB model.

Also, `FASTVIDEO_NVFP4_MM_BACKEND` and `FASTVIDEO_NVFP4_MM_ROW_SPLIT` are read on every GEMM but
were not in the graph signature, so changing them replayed a stale graph.

Patch 20 keeps every graph in one cache for one request shape at a time:

- a call with a new shape, or a new NVFP4 GEMM setting, resets **every** graph first (their
  memory is back before anything new is captured);
- all graphs share one pool, one side stream and one input buffer (they replay one at a time,
  in capture order);
- a graph is captured only while free memory stays above a headroom (64x the layer input,
  about twice the eager working set measured at 124 and 243 frames;
  `FASTVIDEO_H3_GRAPH_HEADROOM_GIB` sets it), and the layers past it run eagerly;
- `FASTVIDEO_H3_DIT_CUDA_GRAPHS=0` turns the graphs off;
- it is plain functions (state on a function attribute), so a running engine takes it as a
  live edit, and on its first call it releases the caches 15-17 kept, which a live engine still
  holds.

Our 2026-11-19 runs (GB200, one warm process each, `tools/h3_shapes.py`; every video
bit-identical to the base engine's for the same length):

| GPUs | Frames in order | 01-19: GPU memory per rank, allocated / reserved | 01-20: allocated / reserved | 01-20 graphs held (skipped by the budget) |
|---|---|---|---|---|
| 1 | 124, 243, 124, 243 | 92.6 / 155.8 GiB, then out of memory at 243 | 77.6 / 81.1, 122.1 / 134.9, 77.4 / 84.9, 122.1 / 135.3 GiB | 100, 100, 100, 100 |
| 1 | 124, 362, 124 | out of memory at 362 | 77.6 / 81.1, 150.0 / 167.0, 77.4 / 85.0 GiB | 100, 87 (13), 100 |
| 2 | 124, 243, 124 | 61.4 / 94.6, 90.8 / 154.8, 61.4 / 102.1 GiB (fits) | 53.9 / 57.0, 76.2 / 84.2, 53.8 / 59.1 GiB | 100 each |
| 2 | 124, 362, 124 | out of memory at 362 (174.1 GiB) | 53.9 / 57.0, 98.5 / 109.8, 53.8 / 59.7 GiB | 100 each |

With a headroom of 110 GiB forced, 1 GPU kept 92 graphs at 124 frames and 41 at 243, all videos
bit-exact. The first video after a length change pays the capture again (124 frames after 243:
5.3-5.5 s against 4.2 s warm). The GPU tests of the cache itself are `test_20_graph_cache.py`
(5 cases, about 1 s; they fail on 01-19).

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
