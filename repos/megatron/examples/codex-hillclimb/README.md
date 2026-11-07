# Codex hill-climb on a live Megatron training job

A developer agent (Codex) proposed one performance edit at a time to Megatron-LM. The live lab
installed each edit into a running training job, measured it against the base in the same job,
and kept or undid it by the workload's gate (`../../kb/workload.md`). No restart between edits.
Our 2026-11-07 run: 82 iterations on the Qwen3-14B architecture, TP 2 on 2 GB200, 00:55-06:23 PST.

## Result

**The kept set is +5.1% tokens/s/GPU** (median of 4 paired rounds, kept vs base in the same job:
1.052, 1.047, 1.070, 1.050; kept 7,337-7,872 vs base 6,973-7,360), loss within 2.2e-3 and grad
norm within 3.5e-3 of the base at every step (gate: 5e-3 and 5e-2). Three edits:

| iter | edit | by | alone (independent 4-pair re-measure) | review | upstream |
|---|---|---|---|---|---|
| 11 | cast the LM head's fp32 output gradient after the sequence-parallel all-gather is launched, so the two overlap (`tensor_parallel/layers.py`) | Codex (gpt-6-astra) | +1.24% (1.018, 1.012, 1.003, 1.013) | Antigravity: correct, no risks found; by hand: equivalent | yes: `../../patches/lmhead-sp-allgather-before-cast.patch` |
| 61 | gather the sequence shards before Transformer Engine's fused LayerNorm+linear, so backward reuses the full-sequence norm output instead of all-gathering it again (`extensions/transformer_engine.py`) | Codex (gpt-6.1-sol) | +2.97% (1.036, 1.036, 1.023, 1.024); an earlier sample: median +6.3% with one pair at 0.983 | Antigravity: correct for this config; breaks with activation recompute, `return_layernorm_output`, fp8, CUDA graphs; TP times the norm activations | not as is: `../../patches/te-layernorm-linear-gather-before-norm.patch` needs those guards |
| 12 | Transformer Engine's fused weight-gradient accumulation on the 160 live TE linear modules (a repair cell: an attribute change, not code) | the orchestrator, by hand; Codex proposed the same at iteration 65 | +5.0% (3 pairs: 1.026, 1.056, 1.050) | Antigravity: correct (fp32 `main_grad` assumed; not with CUDA graphs) | it is Megatron's `--gradient-accumulation-fusion`; private, see below |

An earlier confirmation of iterations 11 + 12 alone gave a median of +2.68% (1.023, 1.046, 1.017,
1.031). One-at-a-time gains do not add: single A/B samples of 8 steps on a shared node carry a few
percent of noise even when paired (two 4-pair samples of iteration 57 gave +2.1% and -3.0%). Report
the confirmation.

Iteration 12 is not a code edit. It sets `fuse_wgrad_accumulation` on the live TE modules through
the journal, so weight gradients accumulate into `main_grad` inside the GEMM instead of a separate
`add_` (about 10% of CUDA time before). It ran only in a private harness and is **not** part of the
lab: changing long-lived state in place is what the vLLM lab moved to restart tiers. Proof that it
ran: after a check, all 160 TE weights on both ranks carried `grad_added_to_main_grad` (set only by
TE's fused path). The supported way is the flag `--gradient-accumulation-fusion`, a restart (T1:
rebuild the model from the in-memory start state with the flag on), which on this aarch64 build
also needs the LM head's non-TE path to work without apex.

## The kept set on the Qwen3-32B architecture (TP 4, 4 GB200)

Same lab, same 8-step checks and lr, Qwen3-32B architecture at TP 4 (our 2026-11-07 07:38-07:54 run):
the three kept edits installed cleanly (iterations 11 + 61 in 0.32 s, the iteration-12 cell in
0.04 s, all four ranks). Kept set vs base, 4 paired rounds: **median +7.1% tokens/s/GPU** (1.089,
1.069, 1.055, 1.074; kept 2,475-2,521 vs base 2,306-2,389).

The loss gate is tight at this size: two identical base checks differed by 4.5e-3 in loss (step 8:
15.128 vs 15.059) and 1.4e-2 in grad norm. Two rounds were inside the gate (loss 2.0e-3 and 2.6e-3),
two just outside (5.2e-3 and 5.9e-3); grad norm within 0.7-2.0% (gate 5%). Read as: the kept set
changes loss by about as much as the base changes against itself here; a tighter gate for 32B needs
deterministic kernels or more steps. Lab start state 106.8 GiB per rank, restore 0.79 s.

## The kept set on two nodes (Qwen3-32B architecture, TP 4 x DP 2, 8 GB200)

Our 2026-11-07 08:04-08:37 run, one supervised job for both nodes. The lab ran on all 8 ranks
across the two nodes; the kept set installed in one step gap on every rank (code 1.39 s, cell
0.10 s). Base vs base: loss 6.6e-4, grad norm 2.5e-3. Kept set vs base, 4 paired rounds: **median
+7.3% tokens/s/GPU** (1.108, 1.075, 1.071, 1.064), every round inside the gate (loss at most 3.2e-3,
grad norm at most 7.8e-3). Outside load lowered both arms during the A/B (base 1,209-1,238 against
1,667-1,687 an hour before); the pairing absorbs it.

Plain 2-node baseline (20 steps, lr 3e-4, global batch 32): 11.57 s per step, about 1,416
tokens/s/GPU, against 7.48 s and 2,190 on one node. The cross-node link is not the limit (a 512 MiB
all-reduce over the cross-node data-parallel pair reached about 533 GB/s of bus bandwidth); NCCL
collectives were 74% of CUDA time, all-gathers 1.02 ms each against 0.22 ms on the 14B single-node
run. Sweeping NCCL protocol, algorithm and DDP bucket size is a restart per setting, not a live edit.

### Final 2-node result (11:02 PST)

The kept set after the 2-node session: iterations 11, 61, 12 (cell) and 119, the plain residual add found
on 14B. Kept set vs base, 4 paired rounds in the same 2-node job (default DDP bucket): **median +8.5%
tokens/s/GPU** (1.097, 1.097, 1.072, 1.067; kept 1,490-1,527 vs base 1,389-1,407), every round inside the
gate (loss at most 1.7e-3, grad norm at most 3.9e-3). Iteration 119 alone on 2 nodes: 4-pair re-measure
median +6.0% (1.062, 1.059, 1.051, 1.064).

A plain sweep made `--ddp-bucket-size 4e8` look like a gain; in the lab it changed the loss with no edits
at all (step 6: 47.5 against 14.9), so it is not one (`../../kb/costs.md`).

2-node Codex iterations (Qwen3-32B architecture, TP 4 x DP 2; iteration 1 ran on an earlier lab whose
harness lacked the crash record; 2 is the 14B transfer; 13 was refused by the known-crasher filter):

| iter | idea (agent's words) | by | edit | install s | paired ratios (cand/base) | gain % (median) | max rel loss diff | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | Dispatch equal-split CUDA sequence-parallel collectives directly through their process group to reduce Python frontend overhead while preserving synch | Codex | diff tensor_parallel.mappings | 0.123435 |  |  |  | check failed |
| 2 | transfer of 14B iteration 118/119 (bias-free zero-dropout residual -> plain add) to the 32B 2-node run | Codex | diff transformer.transformer_layer | 0.297953 | 1.046, 1.063 | 5.45 | 2.02e-03 | keep |
| 3 | none | Codex |   |  |  |  |  | skip |
| 4 | none | Codex |   |  |  |  |  | skip |
| 5 | Route the loss head through the existing Blackwell fused linear cross-entropy kernel while preserving module hooks, testing whether avoiding materiali | Codex | cell  | 0.088967 |  |  |  | refused |
| 6 | re-measure of iter 2: transfer of 14B iteration 118/119 (bias-free zero-dropout residual -> plain add) to the 32B 2-node run | Codex | diff transformer.transformer_layer | 0.416002 | 1.062, 1.059, 1.051, 1.064 | 6.03 | 2.86e-03 | keep |
| 7 | none | Codex |   |  |  |  |  | skip |
| 8 | Copy reduced gradients back into separate FP32 accumulation buffers only for locally owned optimizer shards, avoiding conversion and writes for the ot | Codex | diff distributed.param_and_grad_buffer | 0.18609 | 0.990, 0.998 | -0.59 | 5.55e-04 | revert (no paired gain >= 1%) |
| 9 | none | Codex |   |  |  |  |  | skip |
| 10 | none | Codex |   |  |  |  |  | skip |
| 11 | Fuse DP=2 gradient prescaling into eligible FP32-to-BF16 communication-buffer copies to eliminate a separate full-buffer scaling pass, retaining the o | Codex | diff distributed.param_and_grad_buffer | 0.225403 | 0.994, 1.013 | 0.36 | 1.88e-03 | revert (no paired gain >= 1%) |
| 12 | none | Codex |   |  |  |  |  | skip |
| 13 | Test TP=4 all-reduce followed by a local shard copy to let NCCL select potentially faster all-reduce algorithms for the profiled sequence-parallel red | Codex | diff tensor_parallel.mappings |  |  |  |  | skip (touches a known crasher: climb2n/1: megatron/core/tensor_parallel/mappings.py ['_reduce_scatter_along_first_dim']) |
| 14 | Skip redundant communication-buffer zeroing when every parameter has separate accumulation storage, preserving padding zeros and batching the remainin | Codex | diff distributed.param_and_grad_buffer | 0.179554 | 0.984, 1.009 | -0.37 | 4.28e-04 | revert (no paired gain >= 1%) |
| 15 | none | Codex |   |  |  |  |  | skip |
| 16 | none | Codex |   |  |  |  |  | skip |
| 17 | Batch zeroing of separate accumulation gradients with foreach to reduce kernel launches while preserving all gradient-buffer initialization. | Codex | diff distributed.param_and_grad_buffer | 0.169146 | 0.998, 1.010 | 0.37 | 5.98e-04 | revert (no paired gain >= 1%) |
| 18 | Test skipping masked row-zero accumulation in embedding backward while preserving genuine row-zero gradients through a broadcast reduction, reducing r | Codex | diff tensor_parallel.layers | 0.14968 | 0.998, 1.005 | 0.17 | 4.93e-04 | revert (no paired gain >= 1%) |


## Where the time is (profile of 2 steps, rank 0, after iterations 11 and 12)

Sequence-parallel all-gather 21.5% and reduce-scatter 14.7% of CUDA time (NCCL ring, LL protocol,
about 218 us each), Transformer Engine GEMMs about 40%. Iteration 61 removes one of the all-gathers.
The rest is TP communication overlap (`--tp-comm-overlap`): a launch flag, so a restart.

## Loop

```
harness (any host)                          training job (2 GB200, TP 2)
  Codex round 1: profile, history, index      rank 0: lab host, mailbox on a shared file system
    of hot files -> asks for source ranges    every rank: one participant, gloo rounds
  Codex round 2: one diff (and/or a cell)
  apply to a mirror of the tree
  install ------------------------------->   R0 swap on every rank in the step gap   median 0.10 s
  check(8) candidate / uninstall / check(8) base, twice (4 times for a re-measure)
  keep: install again, commit the mirror; else revert the mirror
```

- Decision: the gate (loss within 0.5%, grad norm within 5% of the base at every step), every
  changed function ran (witness > 0 on every rank), every pair faster and the **median** gain at
  least 1% (it was the mean until iteration 27: one pair under outside load read 1.23, the other
  1.016; re-measured with 4 pairs the edit failed the gate).
- Codex worked without tools: its shell sandbox could not start on our host, and running it
  unsandboxed was not allowed. Round 1 sees an index of the hot files and names line ranges; round 2
  gets them and answers with a diff. The harness recounts hunk headers (Codex miscounted them).
- Costs per iteration (medians): Codex 51 s, evaluation (2 pairs of 8-step checks) 196 s, install
  0.10 s. A restart of the same job to its first step is about 60 s plus 21 s of lab setup.
- Three candidates failed at run time (a Triton kernel, a missing collective handle) and poisoned the
  lab; respawn and re-install of the kept edits took 224, 183 and 131 s, then the loop went on.

Outcomes over 82 iterations (re-measures included): 3 kept (11, 61, 12); 36 below the gain rule; 4
failed the gate; 3 not proven (a changed function never ran); 5 refused at install (`closure_changed`,
a cell's repair failing); 3 failed in the check (the lab was poisoned and respawned itself); 20 with
no usable diff (Codex proposed nothing, a repeat, or a diff that did not apply); 3 rejected by a
harness bug and re-run; 1 invalid row (issued while the lab was poisoned); and 3 kept on two pairs
then reverted by an independent 4-pair re-measure (27, 69, 77). From iteration 40 on, most proposals
were "none" or repeats: the search space a live function swap reaches in this configuration was
close to exhausted.

## Iterations

Ideas are the agent's own words. "by: orchestrator (private)" marks the one edit not written by Codex.

| iter | idea (agent's words) | by | edit | install s | paired ratios (cand/base) | gain % (median) | max rel loss diff | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | Compute cross-entropy maxima with value-only amax before fp32 conversion to eliminate argmax work and reduce reduction bandwidth for bf16 logits. | Codex | diff fusions.fused_cross_entropy, tensor_parallel.cross_entropy | 0.18904 |  |  | 9.79e-04 | revert (not proven: edited code did not run) |
| 2 | Bypass identity parameter and gradient collectives for singleton data-parallel groups to remove redundant communication dispatch and buffer copies whi | Codex |   |  |  |  |  | skip |
| 3 | Bypass identity parameter and gradient collectives for singleton distributed-optimizer groups to avoid communication overhead while preserving gradien | Codex |   |  |  |  |  | skip |
| 4 | Launch input-gradient reduce-scatter before waiting for the weight-gradient input all-gather and preparing its tensors, allowing communication to over | Codex | diff tensor_parallel.layers | 0.097695 | 1.013, 0.991 | 0.20 | 1.49e-03 | revert (no paired gain >= 1%) |
| 5 | Batch FP32 master-to-model parameter copies with torch._foreach_copy_ to reduce per-parameter CUDA launches during the optimizer step. | Codex | diff optimizer.distrib_optimizer | 0.439686 |  |  |  | refused |
| 6 | Retry value-only amax solely in the active unfused cross-entropy helper, avoiding unused argmax computation and reducing read bandwidth when logits ar | Codex | diff tensor_parallel.cross_entropy | 0.075649 | 1.006, 0.989 | -0.25 | 1.44e-03 | revert (no paired gain >= 1%) |
| 7 | Launch the sequence-parallel input all-gather before converting the LM-head output gradient to bf16 so communication can overlap the conversion. | Codex | diff tensor_parallel.layers | 0.104212 | 0.997, 1.020 | 0.88 | 1.60e-03 | revert (no paired gain >= 1%) |
| 8 | Retain the linear forward’s gathered activation in owned storage for backward reuse, eliminating a redundant sequence-parallel all-gather and its CPU  | Codex | diff tensor_parallel.layers | 0.098584 | 0.990, 1.015 | 0.27 | 1.02e-03 | revert (no paired gain >= 1%) |
| 9 | none | Codex |   |  |  |  |  | skip |
| 10 | Defer cross-entropy normalization without label smoothing to backward and combine it with upstream gradient scaling, eliminating one full-vocabulary m | Codex | diff tensor_parallel.cross_entropy | 0.047022 | 1.006, 0.990 | -0.19 | 7.48e-04 | revert (no paired gain >= 1%) |
| 11 | re-measure of iter 7: Launch the sequence-parallel input all-gather before converting the LM-head output gradient to bf16 so communication can overlap | Codex | diff tensor_parallel.layers | 0.108214 | 1.018, 1.012, 1.003, 1.013 | 1.24 | 7.89e-04 | keep |
| 12 | repair-cell: TE fuse_wgrad_accumulation on every live TE linear (orchestrator probe, private) | orchestrator (private) | cell  | 0.039525 | 1.026, 1.056, 1.050 | 4.95 | 1.59e-03 | keep |
| 13 | none | Codex |   |  |  |  |  | skip |
| 14 | Batch required optimizer gradient conversions to FP32 with foreach copy to reduce per-parameter cast launches while preserving existing FP32 gradient  | Codex | diff optimizer.distrib_optimizer | 0.676926 |  |  | 6.22e-04 | revert (not proven: edited code did not run) (harness bug: a kept cell's proof was counted against the candidate; re-run as 18) |
| 15 | Route the active cross-entropy adapter through the available TE fused implementation to reduce kernel launches and full-vocabulary memory passes while | Codex | diff tensor_parallel.cross_entropy | 0.105927 |  |  | 2.89e-04 | revert (not proven: edited code did not run) (harness bug: a kept cell's proof was counted against the candidate; re-run as 19) |
| 16 | Move sequence-parallel gathering before TE normalization and retain full normalized activations to eliminate backward activation all-gathers, trading  | Codex | diff extensions.transformer_engine | 0.229301 |  |  | 1.20e-03 | revert (not proven: edited code did not run) (harness bug: a kept cell's proof was counted against the candidate; re-run as 20) |
| 17 | none | Codex |   |  |  |  |  | skip |
| 18 | re-run (proof bug) of iter 14: Batch required optimizer gradient conversions to FP32 with foreach copy to reduce per-parameter cast launches while pre | Codex | diff optimizer.distrib_optimizer | 0.517841 | 1.000, 1.015 | 0.80 | 1.09e-03 | revert (no paired gain >= 1%) |
| 19 | re-run (proof bug) of iter 15: Route the active cross-entropy adapter through the available TE fused implementation to reduce kernel launches and full | Codex | diff tensor_parallel.cross_entropy | 0.068059 | 1.003, 1.000 | 0.13 | 1.18e-03 | revert (no paired gain >= 1%) |
| 20 | re-run (proof bug) of iter 16: Move sequence-parallel gathering before TE normalization and retain full normalized activations to eliminate backward a | Codex | diff extensions.transformer_engine | 0.242254 | 1.000, 1.010 | 0.51 | 9.35e-04 | revert (no paired gain >= 1%) |
| 21 | Route training loss directly from the active GPT postprocess path through TE fused parallel cross-entropy, preserving FP32 inputs while reducing vocab | Codex | diff models.gpt.gpt_model | 0.082749 | 0.989, 1.011 | -0.01 | 2.94e-03 | revert (no paired gain >= 1%) |
| 22 | Enable TE’s gathered normalization output and discard its extra public return value so backward can reuse gathered activations, targeting fewer sequen | Codex | diff extensions.transformer_engine | 0.229082 | 1.003, 0.979 | -0.91 | 8.48e-04 | revert (no paired gain >= 1%) |
| 23 | Batch master-to-model parameter copies with foreach to reduce kernel launches, keeping batching lists inside the existing helper so its closure remain | Codex | diff optimizer.distrib_optimizer | 0.524583 | 0.984, 0.984 | -1.61 | 1.47e-03 | revert (no paired gain >= 1%) |
| 24 | Dispatch supported full-head RoPE directly to TE’s fused implementation to replace unfused elementwise kernels when fusion is disabled and reduce disp | Codex | diff models.common.embeddings.rope_utils | 0.078563 | 0.982, 1.008 | -0.51 | 6.58e-04 | revert (no paired gain >= 1%) |
| 25 | Move TE row-linear sequence collectives into Megatron’s autograd mapping and dispatch directly through the resolved process group to reduce Python com | Codex | diff extensions.transformer_engine, tensor_parallel.mappings | 0.259844 | 0.968, 0.982 | -2.48 | 7.03e-04 | revert (no paired gain >= 1%) |
| 26 | Coalesce densely packed gradient casts into one FP32 conversion per backing storage to reduce allocation and kernel-launch overhead while preserving p | Codex |   |  |  |  |  | skip |
| 27 | Route unclamped bias-free MLP activations through TE’s fused SwiGLU forward and backward to target the activation kernel and dispatch overhead. | Codex | cell  | 0.033009 | 1.230, 1.016 | 12.32 | 3.05e-03 | kept on the mean of 2 pairs, then reverted: re-measured with 4 pairs as 35, failed the gate |
| 28 | none | Codex |   |  |  |  |  | skip |
| 29 | Coalesce adjacent gradient casts into bounded FP32 allocations to reduce conversion kernel launches while preserving each optimizer parameter’s gradie | Codex | diff optimizer.distrib_optimizer | 1.888524 |  |  |  | refused |
| 30 | Combine cross-entropy’s predicted-logit and exponential-sum reductions into one TP collective to reduce launch latency per microbatch. | Codex | diff tensor_parallel.cross_entropy | 0.081658 |  |  | 6.42e-03 | revert (gate) |
| 31 | Launch LM-head dgrad reduce-scatter before waiting for the input all-gather and preparing wgrad operands, allowing communication to overlap operand pr | Codex | diff tensor_parallel.layers | 0.103508 |  |  | 6.89e-03 | revert (gate) |
| 32 | Bypass identity gradient collectives for a single data-parallel rank to remove collective dispatch and any associated buffer copies while preserving s | Codex | diff distributed.param_and_grad_buffer | 0.144129 |  |  | 5.61e-03 | revert (gate) |
| 33 | Fuse cross-entropy’s logit shift, exponentiation and partial sums into one tiled kernel to reduce full-vocabulary memory passes while preserving the e | Codex | diff tensor_parallel.cross_entropy | 0.049113 |  |  |  | check failed |
| 34 | re-measure (4 pairs, median) of iter 27: Codex repair cell routing MLP SwiGLU to TE fused op | Codex | cell  | 5.3e-05 |  |  |  | invalid: issued while the lab was poisoned by iteration 33; superseded by the next re-measure |
| 35 | re-measure (4 pairs, median) of iter 27: Codex repair cell routing MLP SwiGLU to TE fused op | Codex | cell  | 0.032886 |  |  | 5.16e-03 | revert (gate) |
| 36 | Retain the LM head’s forward-gathered activation in owned storage to eliminate its redundant backward all-gather and associated CPU dispatch overhead. | Codex | diff tensor_parallel.layers | 0.34209 | 1.004, 1.004 | 0.42 | 2.59e-03 | revert (no paired gain >= 1%) |
| 37 | Scale exactly adjacent FP32 gradient views with one dense multiplication to reduce multi-tensor clipping overhead, retaining the existing path for oth | Codex | diff optimizer.clip_grads | 0.051119 | 1.059, 0.976 | 1.71 | 1.13e-03 | revert (no paired gain >= 1%) |
| 38 | Remove unit-scale multiplications and the empty-tail concatenation from unfused full-head RoPE to eliminate redundant kernel launches and an output co | Codex | diff models.common.embeddings.rope_utils | 0.071676 |  |  | 3.33e-03 | revert (not proven: edited code did not run) |
| 39 | none | Codex |   |  |  |  |  | skip |
| 40 | none | Codex |   |  |  |  |  | skip |
| 41 | Replace embedding boolean-index assignments with broadcast masked fills and remove the redundant input clone to reduce masking overhead in forward and | Codex | diff tensor_parallel.layers | 0.097092 | 0.991, 1.005 | -0.23 | 8.72e-04 | revert (no paired gain >= 1%) |
| 42 | none | Codex |   |  |  |  |  | skip |
| 43 | Batch compatible main_grad additions during unsynchronized microbatches into bounded foreach operations to reduce GPU kernel launches while preserving | Codex | diff pipeline_parallel.schedules | 0.145318 | 0.990, 0.966 | -2.23 | 7.09e-04 | revert (no paired gain >= 1%) |
| 44 | none | Codex |   |  |  |  |  | skip |
| 45 | none | Codex |   |  |  |  |  | skip |
| 46 | re-measure of iter 37: Scale exactly adjacent FP32 gradient views with one dense multiplication to reduce multi-tensor clipping overhead, retaining th | Codex | diff optimizer.clip_grads | 0.060707 | 1.008, 1.033, 1.004, 0.998 | 0.57 | 6.82e-04 | revert (no paired gain >= 1%) |
| 47 | none | Codex |   |  |  |  |  | skip |
| 48 | Batch gradient casts into FP32 optimizer gradients with foreach copy to reduce kernel launches while retaining separate allocations and existing FP32  | Codex | diff optimizer.distrib_optimizer | 0.473484 |  |  |  | refused |
| 49 | Defer unsmoothed cross-entropy normalization to backward and combine it with upstream gradient scaling to eliminate one full-vocabulary memory pass. | Codex | diff tensor_parallel.cross_entropy | 0.045799 | 1.002, 0.993 | -0.26 | 1.48e-03 | revert (no paired gain >= 1%) |
| 50 | none | Codex |   |  |  |  |  | skip |
| 51 | Batch master-weight copies into model parameter buffers with foreach copy to reduce conversion kernel launches and Python dispatch overhead during the | Codex | diff optimizer.distrib_optimizer | 0.503893 | 0.991, 1.021 | 0.63 | 3.10e-04 | revert (no paired gain >= 1%) |
| 52 | Use a values-only maximum before FP32 conversion to avoid computing unused argmax indices and reduce reduction bandwidth when logits arrive in bf16. | Codex | diff tensor_parallel.cross_entropy | 0.041662 | 0.999, 0.979 | -1.09 | 1.54e-03 | revert (no paired gain >= 1%) |
| 53 | none | Codex |   |  |  |  |  | skip |
| 54 | none | Codex |   |  |  |  |  | skip |
| 55 | Fuse deferred cross-entropy normalization, target subtraction and upstream scaling into one in-place Triton pass, removing a full-vocabulary memory pa | Codex | diff tensor_parallel.cross_entropy | 0.056459 |  |  |  | check failed |
| 56 | Enable TE’s gathered LayerNorm output retention while preserving Megatron’s return contract, targeting redundant backward sequence-parallel all-gather | Codex | diff extensions.transformer_engine | 0.22219 | 0.976, 0.989 | -1.76 | 1.66e-03 | revert (no paired gain >= 1%) |
| 57 | Compute gradient norms over exactly adjacent FP32 gradient spans with native reductions to reduce multi-tensor chunk-launch overhead without including | Codex | diff optimizer.clip_grads | 0.058577 | 0.983, 1.047 | 1.49 | 2.10e-03 | revert (no paired gain >= 1%) |
| 58 | Route unsmoothed CUDA cross-entropy through TE’s fused implementation with an FP32 workspace to reduce full-vocabulary memory passes and kernel launch | Codex | diff tensor_parallel.cross_entropy | 0.044455 | 0.983, 0.945 | -3.62 | 6.43e-04 | revert (no paired gain >= 1%) |
| 59 | re-measure of iter 57: Compute gradient norms over exactly adjacent FP32 gradient spans with native reductions to reduce multi-tensor chunk-launch ove | Codex | diff optimizer.clip_grads | 0.108387 | 0.998, 1.044, 1.011, 1.031 | 2.12 | 7.79e-04 | revert (no paired gain >= 1%) |
| 60 | re-measure of iter 57: Compute gradient norms over exactly adjacent FP32 gradient spans with native reductions to reduce multi-tensor chunk-launch ove | Codex | diff optimizer.clip_grads | 0.061687 | 0.969, 0.961, 1.029, 0.971 | -3.00 | 2.63e-03 | revert (no paired gain >= 1%) |
| 61 | Gather sequence shards before fused normalization and use the gather’s reduce-scatter backward to eliminate TE’s repeated activation all-gather for we | Codex | diff extensions.transformer_engine | 0.261896 | 1.012, 1.058 | 3.52 | 5.69e-04 | kept on 2 pairs; re-measured as 67 (one pair below 1: reverted) and 71 (kept) |
| 62 | none | Codex |   |  |  |  |  | skip |
| 63 | Use native fused local softmax and recover partition normalization from its maximum probability to reduce full-vocabulary memory passes in unsmoothed  | Codex | diff tensor_parallel.cross_entropy | 0.078351 | 0.990, 0.998 | -0.61 | 1.44e-03 | revert (no paired gain >= 1%) |
| 64 | Dispatch equal-split CUDA sequence-parallel collectives directly through the resolved ProcessGroup to reduce Python validation and dispatch overhead w | Codex |   |  |  |  |  | skip |
| 65 | Enable TE’s native weight-gradient GEMM accumulation into existing FP32 main_grad buffers to eliminate separate weight-gradient materialization and ad | Codex | cell  | 0.045207 |  |  |  | refused |
| 66 | Reuse version-checked FP32 sine/cosine tables across Q/K and layers with FlashAttention’s rotary kernel to eliminate repeated trigonometric computatio | Codex | diff models.common.embeddings.rope_utils | 0.098607 | 1.012, 1.006 | 0.88 | 3.54e-03 | revert (no paired gain >= 1%) |
| 67 | re-measure of iter 61: Gather sequence shards before fused normalization and use the gather’s reduce-scatter backward to eliminate TE’s repeated activ | Codex | diff extensions.transformer_engine | 0.265042 | 1.064, 0.983, 1.065, 1.062 | 6.30 | 7.79e-04 | revert (no paired gain >= 1%) |
| 68 | Dispatch equal-split CUDA sequence-parallel collectives directly through the resolved ProcessGroup to reduce Python wrapper overhead while preserving  | Codex | diff tensor_parallel.mappings | 0.090693 |  |  |  | check failed |
| 69 | Coalesce exactly adjacent non-FP32 gradient shards into single FP32 casts to reduce conversion launches and allocations while preserving the existing  | Codex | diff optimizer.distrib_optimizer | 0.482962 | 1.031, 1.007 | 1.90 | 8.65e-04 | kept on 2 pairs, reverted: independent 4-pair re-measure (79) failed the rule |
| 70 | Use fused local softmax and recover partition sums through indexed maximum probabilities, avoiding a second vocabulary reduction and separate shift an | Codex | diff tensor_parallel.cross_entropy | 0.271749 | 1.003, 0.951 | -2.33 | 7.21e-04 | revert (no paired gain >= 1%) |
| 71 | re-measure of iter 61: Gather sequence shards before fused normalization and use the gather’s reduce-scatter backward to eliminate TE’s repeated activ | Codex | diff extensions.transformer_engine | 0.288423 | 1.036, 1.036, 1.023, 1.024 | 2.97 | 1.38e-03 | keep |
| 72 | Route equal-split CUDA sequence-parallel collectives through functional dispatcher operators with explicit waits to reduce Python dispatch overhead wi | Codex | diff tensor_parallel.mappings | 0.066902 | 0.982, 0.947 | -3.57 | 1.13e-03 | revert (no paired gain >= 1%) |
| 73 | Batch small, same-dtype main_grad additions during unsynchronized microbatches with foreach to reduce kernel launches while preserving each parameter’ | Codex | diff distributed.distributed_data_parallel | 0.38617 |  |  | 1.93e-03 | revert (not proven: edited code did not run) |
| 74 | none | Codex |   |  |  |  |  | skip |
| 75 | Retain the LM head’s gathered forward activations in an owned buffer to eliminate its backward all-gather and associated CPU dispatch and wait overhea | Codex | diff tensor_parallel.layers | 0.095886 | 1.058, 0.978 | 1.84 | 1.42e-03 | revert (no paired gain >= 1%) |
| 76 | Patch already registered DDP hooks to batch small BF16 gradient additions through a packed FP32 cast and foreach accumulation, reducing kernel launche | Codex | cell  | 0.04353 |  |  |  | refused |
| 77 | Coalesce the cross-entropy target-logit and exponential-sum reductions into one collective to remove one collective launch per microbatch. | Codex | diff tensor_parallel.cross_entropy | 0.045776 | 1.021, 1.009 | 1.50 | 9.32e-04 | kept on 2 pairs, reverted: independent 4-pair re-measure (82) failed the rule |
| 78 | Launch the LM head’s gradient reduce-scatter before waiting for activation gathering and preparing weight-gradient inputs, allowing communication to o | Codex | diff tensor_parallel.layers | 0.097022 | 1.002, 0.983 | -0.78 | 1.26e-03 | revert (no paired gain >= 1%) |
| 79 | re-measure of iter 69: Coalesce exactly adjacent non-FP32 gradient shards into single FP32 casts to reduce conversion launches and allocations while p | Codex | diff optimizer.distrib_optimizer | 0.541683 | 1.035, 1.010, 0.985, 1.005 | 0.78 | 9.50e-04 | revert (no paired gain >= 1%) |
| 80 | none | Codex |   |  |  |  |  | skip |
| 81 | Batch contiguous master-to-model parameter copies by device and dtype with foreach to reduce cast-and-copy kernel launches during the optimizer step. | Codex | diff optimizer.distrib_optimizer | 0.606684 | 0.996, 0.966 | -1.93 | 2.40e-03 | revert (no paired gain >= 1%) |
| 82 | re-measure of iter 77: Coalesce the cross-entropy target-logit and exponential-sum reductions into one collective to remove one collective launch per  | Codex | diff tensor_parallel.cross_entropy | 0.044197 | 0.992, 1.047, 0.981, 1.005 | -0.16 | 1.89e-03 | revert (no paired gain >= 1%) |
