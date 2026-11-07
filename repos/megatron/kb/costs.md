# Costs

Measured on 4× GB200 (aarch64), torch 2.13 + CUDA 13, Transformer Engine 2.18, megatron-core
0.20 dev, code and Python environment on a network file system. Source: our 2026-11-07 runs
(startup phases from `fastrsi_megatron/run.py`, lab calls from the lab's own `cost_s`). Every
row names its run.

## Restart ladder: from process start to the first train step

`run.py` stamps each phase boundary. "Tiny" is 2 layers, hidden 512 (TP 2, 2 GPUs): it shows the
fixed costs. "14B" is the workload model (`workload.md`, TP 2, 2 GPUs).

| Phase | Tiny (s) | 14B (s) | What it is |
|---|---|---|---|
| Interpreter, torchrun spawn | 2.6 | 2.8 | process start to our driver |
| `import torch` | 1.4-2.2 | 1.4 | |
| `import megatron.training.training` | 16-31 | 31 | Megatron, TE and their imports, from the network file system |
| Arg parsing, config | 0.7-1.4 | 1.0 | |
| `initialize_megatron` | 1.0-1.2 | 1.0 | distributed init, groups, seeds, dataset helpers |
| `warmup_training_kernels` and the rest before the model | 3-4 | | fused kernels compiled at static shapes |
| Model and optimizer build | 0.06 | 10.7 * | weights init, DDP buffers, distributed optimizer |
| Data iterators | 0.4-0.7 | 0.5 | dataset indexes (cached after the first run) |
| First train step | 5.9 | 11.4 | NCCL communicators, Adam state, TE kernel choice |
| **Process start to first step done** | **57.5** | **59.8** | |
| Steady train step | 0.147 | 4.44 | median of steps 3-20 |

\* The 14B row before the finer wrappers: 10.7 s from `initialize_megatron` done to
`setup_model_and_optimizer` start plus 0.25 s inside it; the split is in the hill-climb run below.

The import of Megatron is the largest fixed cost (a third to a half of every restart), and the
first step pays 6-11 s of lazy setup. Both stay warm in a live lab.

## Live lab calls

From the lab smoke (tiny, TP 2, 2026-11-07) and the conformance run.

| Call | Seconds | Notes |
|---|---|---|
| Lab setup at the first step: freeze batches | 0.12 | 3 x 8 microbatches |
| Snapshot S0 to pinned host memory | 0.04 | 1.06 GiB per rank (tiny) |
| Warm-up steps (2) | 5.9 | first step 5.75 s, second 0.16 s |
| Restore S0, zero Adam state, snapshot S | 0.05 | 38 state tensors zeroed |
| `install` (R0 swap of `MLP.forward`, 2 ranks) | 0.05 | edit-to-run |
| `check(steps=3)` | 0.45 | of which restore of S 0.0065 s |
| `uninstall` | 0.017 | |
| Restart to the same point (tiny) | 57.5 | the warm restart a live edit replaces |

## Repeatability

Tiny model with `--deterministic-mode`: three checks of the base gave bit-identical loss
(`12.064836502075195, 12.067537307739258, 11.609490394592285`), before and after an install
and uninstall of a visible edit.
