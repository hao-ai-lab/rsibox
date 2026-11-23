# Workload F(x) and its correctness gate

The target the hill-climb optimizes, fixed before any edit. Developers never change it, the gate
or the timing code (`core/agents/developer.md`).

## Model

Qwen3-14B architecture, dense: 40 layers, hidden 5120, FFN 17408 (SwiGLU), 40 query heads and 8
KV groups of head dim 128, QK RMSNorm, RoPE base 1e6, vocab 151936, untied embeddings, no
biases. About 14.8 B parameters. Random init (`--init-method-std 0.02`, seed 1234), bf16 with
fp32 master weights.

Why not the Qwen3.8-27B checkpoint: it is a hybrid (three linear-attention layers for every
full-attention layer, full attention with head dim 256). On GB200 (SM100) its full-attention
backward found no backend in Megatron (flash-attn: "SM100 backward with head_dim=256 does not
support local attention"; TE fused attention: "No dot product attention backend is available"),
seen in our 2026-11-05 RL bring-up. A dense model with head dim 128 runs every path today.
Why random weights: the hill-climb measures training speed at fixed numerics, which needs the
same code paths and shapes, not a pretrained start (`core/agents/developer.md`, "No weights yet").

A Qwen3-32B-architecture run (64 layers, FFN 25600, 64 heads) is the 2-node target; see
"Two nodes" below.

## Run

| Setting | Value |
|---|---|
| Parallelism | TP 2 with sequence parallel, PP 1, DP 1 (2 GPUs); distributed optimizer |
| Batch | micro-batch 1, global batch 16, sequence 4096: 65,536 tokens per step |
| Optimizer | TE FusedAdam, lr 3e-5 cosine to 3e-6, warm-up 2 steps, betas 0.9/0.95, wd 0.1, clip 1.0 |
| Data | wikitext-2 (document level, train split), Qwen3 tokenizer, EOD appended: about 2.5 M tokens |
| Kernels | Transformer Engine; no gradient-accumulation fusion (no apex) |

## Probe and full evaluation

Both run inside the live lab (`livelab/rsibox_megatron/hook.py`), from the same start state S:
initial weights, Adam moments and step counters at zero, scheduler at step 0, fixed RNG, and
the same frozen batches. So two checks of the same code start from the same point.

- **Check:** `check(steps=8)`: 8 train steps.
- **Decision (paired A/B):** candidate check, uninstall, base check, reinstall, candidate check,
  uninstall, base check. The live lab swaps an edit in or out in about 0.05 s, so each candidate
  is compared with the base under the same outside load.

Metric: **tokens/s/GPU** = 65,536 / mean of the 3 fastest step times among steps 2..8 / 2 GPUs.
Step time is measured around Megatron's `train_step` with a CUDA synchronize on both sides, so it
includes forward, backward, gradient reduction and the optimizer step. Why the fastest steps:
on a node shared with other jobs, step times of the same code varied from 4.34 to 5.38 s within
one check, and the median of three identical base checks spread 6% (our 2026-11-07 base runs);
the fastest steps spread 4%, and pairing removes most of the rest. An edit is kept when both
pairs are faster and the mean gain is at least 1%.

## Correctness gate

The edited run must keep the loss behavior of the base run on the same probe:

- per-step `lm loss` within a relative 0.5% of the base, at every step, and
- per-step grad norm within a relative 5% of the base, at every step.

Why lr 3e-5: at 3e-4 the loss from random init spiked (12.9 to 24 at step 3) and three identical
base checks differed by up to 4% in loss by step 8, because nondeterministic bf16 reductions grow
through the spikes. The tolerance comes from the base against itself at 3e-5 (see `costs.md`,
"Repeatability"). An edit outside
the tolerance is reverted whatever its speed. An edit whose loss is bit-exact with the base is
reported as such.

What the gate does not prove: equality over a long run, or convergence. It sees 8 steps from
random init.

## Two nodes

Qwen3-32B architecture on 2 x 4 GB200, TP 4 within a node, DP 2 across nodes, distributed
optimizer, overlapped gradient reduce and parameter gather, global batch 32, sequence 4096. Same
data, gate and metric (tokens/s/GPU over 8 GPUs).

Measured (our 2026-11-07 runs): one node (TP 4, global batch 16) 7.48 s per step, about 2,190
tokens/s/GPU, 141 GB per GPU; two nodes 11.57 s per step, about 1,416 tokens/s/GPU, 94 GB per GPU,
137 s from process start to the first step done. The estimate made from the one-node run (7.6-8.2 s,
2,000-2,150) assumed the data-parallel traffic would hide behind compute; it did not.
