# Patches to Megatron-LM

Edits to Megatron-LM itself that the loop kept, one file each, against megatron-core 0.20 dev
(commit ae76c60e6 plus local modifications). Each row says what, why, how it was checked and
whether it is a candidate for upstream.

| Patch | What | Why | Checked | Upstream? |
|---|---|---|---|---|
| `lmhead-sp-allgather-before-cast.patch` | In `LinearWithGradAccumulationAndAsyncCommunication.backward`, cast `grad_output` to the input dtype after the sequence-parallel input all-gather is launched (and before buffering on the deferred-wgrad path), not first | With sequence parallel and FP32 logits, the LM head's `grad_output` arrives in fp32 (`seq x mb x vocab/TP`; 4096 x 1 x 75,968 x 4 B = 1.2 GB read for Qwen3 at TP 2); casting it first delays the all-gather. Launching the all-gather first lets it overlap the cast. Hidden layers that run Transformer Engine do not reach this code. | Live in the 14B hill-climb lab (`../examples/codex-hillclimb/`, iteration 11, Codex gpt-6-astra): proven on both ranks, loss within 7.9e-4 of base over 8 steps, paired A/B over 4 pairs, median +1.24% tokens/s/GPU (ratios 1.018, 1.012, 1.003, 1.013). Reviewed: Antigravity "correct, no risks found"; by hand: between the old and new cast points `grad_output` is read only on the deferred-wgrad path, which now casts too. | Candidate. Not checked: PP > 1, wgrad deferral on, fp8. |
| `te-layernorm-linear-gather-before-norm.patch` | In `TELayerNormColumnParallelLinear.forward`, when training with sequence parallel (and no fp8 params, TP comm overlap, delayed wgrad or GTP remat), gather the sequence shards first and run TE's fused LayerNorm+linear as a plain column-parallel layer for that call | TE with sequence parallel keeps the norm output sharded and all-gathers it again in backward for the weight gradient; gathering before the norm keeps the full-sequence norm output, and the gather's backward does the reduce-scatter TE would do | Live in the 14B hill-climb (iteration 61, Codex gpt-6.1-sol): proven on both ranks (10,240 forward calls each); independent 4-pair re-measure median +2.97% (1.036, 1.036, 1.023, 1.024), an earlier one +6.3% with one pair at 0.983; loss within 1.4e-3. Antigravity: correct here; breaks under activation recompute (the no-grad first forward and the recompute take different paths), with `return_layernorm_output` (callers expect a sharded norm output), fp8 autocast and CUDA graphs; saves TP times the norm activations. It also mutates module attributes during forward. | Not as is: needs guards for the cases above, or the same idea inside TE. |
| `residual-plain-add-without-bias-dropout.patch` | In `TransformerLayer`, when hidden dropout is 0, the attention / MLP output has no bias and the residual has the same dtype, add the residual directly (`residual + x`) instead of calling the fused bias-dropout-add | With no bias and no dropout, bias-dropout-add is a plain addition, but the fused path still dispatches a TorchScript graph with a dropout node twice per layer, forward and backward | Codex (gpt-6.1-sol), 14B hill-climb iterations 118/119: independent 4-pair re-measure median +5.0% (1.057, 1.047, 1.054, 1.045), loss within 1.8e-3. Transferred unchanged to the Qwen3-32B architecture on 2 nodes (TP 4 x DP 2): 4-pair re-measure median +6.0% (1.062, 1.059, 1.051, 1.064), loss within 2.9e-3. Antigravity: correct; a custom layer spec that overrides `self_attn_bda` / `mlp_bda` or the exec handler would be bypassed; loss moves slightly (eager add vs the fused graph). | Candidate. Needs: the same fast path only for the default BDA modules (check the spec), a note on the numerics. |

When replaying the LM-head patch, set `--output-logit-dtype fp32` in both arms. In our
2026-11-19 two-GB200 run on Megatron-LM `db087049a`, the default bf16 job reached the
edited backward function but the cast already had bf16 input. With FP32 logits, a line probe
observed the all-gather launch before each of 32 FP32-to-bf16 conversions per rank, and the
wait afterwards. Across eight deterministic steps, the edited job matched both the baseline
and the fresh edited process in loss and gradient norm.

On the same pin, the residual patch needs a context adjustment around the residual
accumulator. The [pin-compatible patch](residual-plain-add-without-bias-dropout.ported-to-pin.patch) preserves both guarded additions. On four GB200s
(TP2/DP2), all three edits ran on every rank and passed the per-step loss and gradient-norm
gates over ten steps on eight Qwen3-14B-shaped layers. See [measured costs](../kb/costs.md).
