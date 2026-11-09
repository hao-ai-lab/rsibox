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
| `warmup_training_kernels` | 2.9-3.5 | 5.7 | |
| Model and optimizer build | 0.06 | 0.3 | weights init on GPU, DDP buffers, distributed optimizer |
| Data iterators | 0.4-0.7 | 0.5 | dataset indexes (cached after the first run) |
| First train step | 5.9 | 11.4 | NCCL communicators, Adam state, TE kernel choice |
| **Process start to first step done** | **57.5** | **59.8** | |
| Steady train step | 0.147 | 4.44 | median of steps 3-20 |

The 14B split is from the hill-climb lab's start (2026-11-07): from `initialize_megatron` done,
2.6 s of setup, 5.7 s of kernel warm-up, then 0.2 s model build and 0.05 s optimizer build (weights
are initialized on the GPU; there is no checkpoint to load).

The import of Megatron is the largest fixed cost (a third to a half of every restart), and the
first step pays 6-11 s of lazy setup. Both stay warm in a live lab.

Qwen3-32B architecture, TP 4 on 4 GB200 (our 2026-11-07 run of the staged 2-node script on one
node): 94.7 s from process start to the first step done (Megatron import 54 s, kernel warm-up 6.8 s,
model and optimizer build 0.4 s, first step 15.2 s); steady step 7.48 s for 65,536 tokens; 141 GB
peak allocated per GPU.

Two nodes, Qwen3-32B architecture, TP 4 x DP 2 (2 x 4 GB200, our 2026-11-07 run): 137 s from process
start to the first step done (Megatron import 85 s, first step 21.4 s); steady step 11.57 s for
131,072 tokens; 94 GB per GPU. In the lab: start state 61 GiB per rank, restore 0.28 s, setup 31 s,
install of the kept set on all 8 ranks 1.39 s (code) + 0.10 s (cell).

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
| Respawn after a crash, hang or lost rank (tiny, conformance c8) | 37.6-51.6 | kill the job's process tree, start again |

At 14B (Qwen3-14B architecture, TP 2, hill-climb lab, 2026-11-07):

| Call | Seconds | Notes |
|---|---|---|
| Lab setup at the first step | 21 | freeze 0.7, S0 2.5, 2 warm-up steps 13.9 (9.5 + 4.4), restore + zero + snapshot S 3.8 |
| Start state S in pinned host memory | | 96.3 GiB per rank (bf16 weights, fp32 masters, two fp32 moments), 646 state tensors zeroed |
| Restore S before a check | 0.25-0.27 | |
| `check(steps=8)` | 43-44 | 8 train steps of about 4.4 s, plus restore |
| Lab at Qwen3-32B architecture, TP 4 on 4 GB200 | | start state S 106.8 GiB per rank, restore 0.79 s, setup 31 s (2 warm-up steps 14.8 + 7.7 s), 8-step check about 65 s |
| A client request through the NFS mailbox from another host | 5-6 wall | copy of the request, a remote Python start and the mailbox round trip (26-34 ms of it) |

## Repeatability

32B architecture, TP 4, lr 3e-5, two identical 8-step checks from S (2026-11-07): loss differed by
up to 4.5e-3 (step 8) and grad norm by 1.4e-2; tokens/s/GPU 2,398-2,409.

14B, no deterministic mode, three identical 8-step checks from S (2026-11-07): at lr 3e-4 the loss
spiked and differed by up to 4% at step 8; at lr 3e-5 it differed by at most 0.15% in loss and about
0.1% in grad norm. Step times of the same code ranged 4.34-5.38 s within one check on the shared
node; tokens/s/GPU from the 3 fastest of steps 2..8 was 7,324-7,463 (2% spread at lr 3e-5, 4% at 3e-4).

Tiny model with `--deterministic-mode`: three checks of the base gave bit-identical loss
(`12.064836502075195, 12.067537307739258, 11.609490394592285`), before and after an install
and uninstall of a visible edit.

## Probing a restart-tier setting: NCCL sweep vs a collective microbenchmark

Two nodes, Qwen3-32B architecture, TP 4 x DP 2 (our 2026-11-07 runs). Full restarts, 10 steps each,
against a repeat of the default (8.85 s per step): `NCCL_PROTO=Simple` 8.75 s, NVLS 8.81 s, NVLSTree
9.24 s, NVLS + cuMem 11.27 s (+27%), `--ddp-bucket-size 4e8` 8.53 s (-3.6%). Cost: 17.4 min for six
points (about 160-175 s each).

The same NCCL settings in a collective microbenchmark (one fresh 8-rank process per setting: the TP
all-gather of a 10.5 MB sequence-parallel shard, the cross-node DP reduce-scatter of 80 and 800 MB,
and both at once on two streams): every setting within about 3% on every collective (TP all-gather
135-139 us; DP reduce-scatter 80 MB 177-184 us, 800 MB 1005-1016 us). Cost: 71 s for all five,
including scheduling.

What the probe got right: the algorithm and protocol do not matter here, and larger messages move more
bytes per second (800 MB at 2.2x the rate of 80 MB), the direction of the bucket-size gain. What it
missed: the cuMem regression (+4% in the probe, +27% in training). Inside training the same TP
all-gather took 1.02 ms (profile), 7.5x the isolated 135 us: the cost is interference in the training
schedule, not the NCCL settings. Use the probe to screen, then confirm the chosen setting with one
restart.

The bucket-size "win" is not one. In the 2-node lab (lr 3e-5, kept set or none), `--ddp-bucket-size 4e8`
ran at 1,972-2,127 tokens/s/GPU against 1,500-2,096 for the default, but its loss left the default's
curve from step 3 (14.82 vs 14.72; step 6 47.5 vs 14.9), identically with no edits, with the code edits,
and with code edits plus the fused weight-gradient cell; each setting repeated itself to 1e-3. With DP 2
the bucket size should not change the gradients. Not a valid speedup; cause not established (suspect:
overlapped parameter all-gather with one 400M-element bucket).

