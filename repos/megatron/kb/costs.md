# Costs

Measured on 4× GB200 (aarch64), torch 2.13 + CUDA 13, Transformer Engine 2.18, megatron-core
0.20 dev, code and Python environment on a network file system. Source: our 2026-11-07 runs
(startup phases from `fastrsi_megatron/run.py`, lab calls from the lab's own `cost_s`). Every
row names its run.

## Live and fresh checks on eight Qwen3-14B-shaped layers

Our 2026-11-19 NeMo 26.08.01 runs on four GB200s used TP 2 with sequence parallelism and
DP 2, 4,096 tokens per sequence, global batch 16, BF16 and FP32 output logits. Each of five
repetitions paired a base check and a live install with a separate torchrun of the same
edited source. Both paths used four warm-up steps per new process group and ten steps per
check. The client timed install through check completion on the live path and
process launch through check completion on the fresh path. Source preparation and teardown
were outside the timers. Filesystem and compilation caches were shared across repetitions.

| Edit | Live install and check (s) | Fresh launch and check (s) | Fresh launch to ready (s) |
|---|---:|---:|---:|
| `lmhead-sp-allgather-before-cast` | 5.52 [5.52, 5.52] | 42.27 [42.22, 42.43] | 36.82 [36.73, 36.96] |
| `residual-plain-add-without-bias-dropout` | 5.55 [5.55, 5.56] | 42.15 [41.75, 42.23] | 36.71 [36.32, 36.81] |
| `te-layernorm-linear-gather-before-norm` | 5.61 [5.60, 5.61] | 41.83 [41.79, 41.92] | 36.41 [36.35, 36.49] |

Values are the median [25th, 75th percentile] of five per-edit client measurements.
Fresh and live edited checks agreed on every reported loss and gradient norm. Relative to
base, the TE edit changed loss by at most 0.02221% and gradient norm by at most 0.35367%.
The other two edits matched base on both metrics at every step. The live
path amortizes model setup and warm-up across checks. These timings measure the wait for
ten-step feedback on the eight-layer model.

### Ten warm forks and the EP 4 walkthrough

In a separate two-layer TP 2 soak in the same environment, one zygote served ten
consecutive pairs of ranks with a different executed source overlay each time. Client
fork times ranged from 0.064 to 0.095 s.
The server held 25 open file descriptors after each pair exited. Both rank exit
codes were zero throughout, and GPU memory returned to zero after every cycle. These measurements come from
the same day's fork events, training phase timestamps, process snapshots and GPU queries.

On a four-layer BF16 MoE with eight experts, top-2 routing and EP 4, installing
the grouped MLP forward edit took 0.119 s round-trip and a four-step edited check
took 0.385 s. Each rank recorded 32 calls to the edited forward. Step-one
loss and the restored state digest matched base in the eight-gate EP 4 walkthrough. Its transaction,
proof and undo records come from the same day.

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

## Restarts: why one costs 60-110 s, and what each kind of restart keeps

Our 2026-11-07 runs on 4 GB200 (one node), 3 train steps, request (or process start) to the first train step done, from
`phases.rank0.json`. "NFS" = Python environment and Megatron on the network file system; "local" = both copied to the
node's RAM disk (`tools/stage_local.sh`); "zygote" = ranks forked from a process that has them imported
(`livelab/fastrsi_megatron/zygote.py`).

| Phase | 14B NFS | 14B local | 14B zygote | 32B NFS | 32B local | 32B zygote | Avoidable? How |
|---|---|---|---|---|---|---|---|
| Launcher to our driver (torchrun, interpreter) | 3.3 | 1.5 | 0.1 | 3.5 | 1.5 | 0.1 | yes: flag restart from the zygote |
| `import torch` | 1.5 | 1.0 | 0 | 2.1 | 1.7 | 0 | yes: kept imported in the zygote |
| `import megatron.training.training` (8,300 modules) | 31.8 | 10.4 | 0 | 52.4 | 10.0 | 0 | mostly: node-local copy (environment restart), kept imported (zygote) |
| Args, `initialize_megatron` | 2.0 | 2.0 | 2.2-2.8 | 3.4 | 3.7 | 3.8-4.5 | no (process groups, seeds); eager NCCL makes it worse |
| Init to kernel warm-up (lazy torch.compile imports, compile pool) | 4.5 | 1.3 | 1.6-1.7 | 24.7 | 1.3 | 1.6 | yes: node-local copy |
| `warmup_training_kernels` (torch.compile, cache hit) | 4.0 | 2.8 | 2.2-2.8 | 7.3 | 3.2 | 2.5-3.1 | partly: per process; +6-9 s on a flag's first compile |
| Model and optimizer build, data | 0.7 | 0.7 | 0.6-0.7 | 1.2 | 0.9 | 0.9-1.0 | no (random init on GPU, 0.3-0.4 s) |
| First train step (steady step) | 10.2 (4.5) | 9.8 (4.4) | 9.4-11.4 (4.6) | 15.6 (6.8) | 15.3 (6.9) | 14.9-16.6 (6.9) | per process: see below |
| **Total** | **58.0** | **29.3** | **17.1-19.6** | **110.1** | **37.6** | **24.5-27.5** | |

Sources: runs t1-14b-b, t1-14b-c, t1-14b-d (14B) and t1-32b, t1-32b-b (32B); zygote ranges over 6 (14B) and 4 (32B)
restarts with the same flags or another NCCL setting. With torch._dynamo and torch._inductor preloaded in the zygote (now
its default) and TORCHINDUCTOR_COMPILE_THREADS=1, the warm-up takes 0.6-1.4 s (14B) / 0.65-0.84 s (32B, 2.9 s on a new
flag's cached compile) and the total is 15.0-16.5 s (14B, 3 runs, t1-14b-e) / 22.8-25.1 s (32B, 4 runs, t1-32b-c). A flag that needs a new torch.compile
(`--cross-entropy-loss-fusion`, first time) adds 6-9 s of warm-up: 23.7 s (14B), 30.8 s (32B). One 14B zygote restart took
40.1 s (first step 17 s, its first microbatch 11.5 s), not reproduced in 8 more. The zygote itself is ready 12-25 s after
its start, once per job; staging the 23 GB environment on a node takes 161-247 s (two runs, 32 copies in
parallel) and Megatron-LM 4.4-7.4 s, once per node. A plain `torchrun` from the staged environment (`stage_local.sh`'s
`env.sh`): 29.7 s to the first step at 14B (our run t2-tool).

`torchrun` from a staged environment needs care: the venv's `torchrun` script names the original python in its shebang,
and its ranks import from the network file system again (54.0 s at 14B). `stage_local.sh` repoints the shebangs.

Import from a network file system, cold vs warm: `python -X importtime` of the same import, 37.7 s self time cold (the
transformers package alone 16.6 s), 19.3 s warm; wall 20.5-23.0 s warm; 11.6-12.4 s from local RAM disk. Most of the
8,300 modules are not used by training (transformers, openai, fla, flashinfer, wandb, fastapi ...).

### The first train step

`FASTRSI_STEP_DETAIL=2` (synchronized split, `run.py`): the first step's extra time over a steady one is in the first
microbatch (14B 3.8-5.7 s vs 0.28-0.33 s; 32B 5.1-7.0 s vs 0.42-0.48 s), the optimizer step (+0.5 s / +1.3-1.4 s: Adam state
allocation, first grad-norm reduction) and the step's tail (+0.5 s / +1.0-1.3 s). A Python profile of rank 0's first step
(14B, zygote; profiled step 11.3 s vs 5.4 s) attributes the excess to: NCCL communicator creation on first use (`all_reduce`
1.1 s, `broadcast` 0.5 s, coalesced 0.2 s), cuDNN fused-attention plans (`fused_attn_fwd` + `bwd` 1.96 s vs 0.12 s),
the first RMSNorm calls (1.72 s vs 0.23 s, caching-allocator growth lands here), and torch.compile'd functions the warm-up
does not cover (1.6 s more in dynamo frames, 0.8 s of it waiting on compile workers). All of it is per process: a fork
cannot inherit it (no CUDA in the zygote). Creating communicators at init (`FASTRSI_EAGER_NCCL=1`, `device_id`) moves
rather than removes it: init +3.0 s (14B) / +9-10 s (32B), first step -0.6 s / -2 to -4 s, net slower.

What of the first step a cache on disk can keep (our 2026-11-07 runs, Qwen3-14B architecture TP 2 unless noted; single-layer
numbers from a one-GPU probe of one attention, RMSNorm and linear layer at the same shapes, first call vs later calls):

| Part | Persists across processes? | Cold | Warm | Cut by environment/cache, or only by a live process | Source |
|---|---|---|---|---|---|
| torch.compile (inductor FX graphs, Triton kernels) | yes, cache directories | warm-up 1.38 s (6 FX misses) | warm-up 0.55-0.76 s (8 hits); 0 compiles in step 1 either way | cache; network vs local directory makes no difference when warm (16.6-17.5 vs 16.6-17.0 s) | runs fs-cache, fs-paired cold1/cold2 |
| cuDNN fused attention (TE) | plans: no (in-process dict, no serialization in TE 2.18); their JIT-compiled kernels: yes, CUDA JIT cache | first call 1.26-1.42 s | 0.10-0.25 s (later calls 0.006 s) | cache: CUDA_CACHE_PATH; the 0.1-0.25 s left only by a live process | single-layer probe |
| TE RMSNorm | kernels: yes, CUDA JIT cache | 0.85-1.14 s | 0.14-0.19 s | cache | single-layer probe |
| cuBLASLt heuristics | no | 0.045 s per shape | same | live process only; negligible (+0.08 s per 14B step); CUDA_MODULE_LOADING=EAGER makes it 0.78 s | single-layer probe, profile |
| NCCL communicators | no (process state) | about 1.7 s of the first step (profile: all_reduce 1.1, broadcast 0.5, coalesced 0.2) | | live process only. NCCL_RUNTIME_CONNECT=0: 18.1 s vs 15.7-16.3 s to the first step (worse); device_id eager init: net +2.4 s (14B), +5-7 s (32B); NCCL topology/graph files: dump works, reusing the graph file crashed the job | runs fs-nccl, t1-14b-d, t1-32b-b |
| CUDA caching allocator | no | 414 cudaMalloc in step 1, 84 in step 2 | | live process only. expandable_segments: 601/361 cudaMalloc, 18.3 s to the first step, optimizer step 1.11 vs 0.63 s (worse) | run fs-nccl |
| Adam state | no | optimizer step +0.5 s (14B) / +1.3-1.4 s (32B) | | live process only | step split |

Paired receipt for the caches that persist, flag restart from the zygote, 14B TP 2, four alternating pairs from two
zygotes alive at once (run fs-paired): caches as before (inductor and Triton on the network file system, the CUDA JIT cache
in its default place under the home directory, also on the network file system) 15.69, 17.27, 17.99, 17.18 s (mean 17.03)
to the first step done; all three caches on node-local disk 16.63, 15.47, 15.76, 15.07 s (mean 15.73). The difference is
in the first microbatch: 3.96-5.71 s before, 3.35-3.64 s after. Empty local caches: 18.24 s, then 15.51 s.
Qwen3-32B architecture TP 4, three alternating pairs (run fs-paired-32b): 23.27, 25.33, 25.00 s (mean 24.53) before,
22.58, 23.34, 22.57 s (mean 22.83) after; first microbatch 5.94-6.56 s before, 4.26-4.58 s after.
So of the 5-6.5 s a first step costs over a steady one (14B), a disk cache keeps about 1.3 s on average (1.7 s at 32B; the JIT
cache's place) and up to 2.3 s on a bad run; the remaining 4-5 s (communicators, allocator, Adam state, cuDNN plans, the step's tail)
only a live process keeps. `tools/stage_local.sh` now puts all three caches on local disk.

What only a live process keeps (the live lab's in-place edit): communicators, cuDNN plans, compiled graphs, allocator pools, Adam state.

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

For the two-layer, hidden-512 deterministic lab on two GB200s (TP2, sequence length 128,
global batch 4, eight-step checks), our 2026-11-19 NeMo 26.08 measurements were:

| Phase | Fresh torchrun | Request to a preloaded zygote |
|---|---:|---:|
| Launch to lab ready | 31.01 s | 10.42 s |
| Freeze batches | 0.045 s | 0.035 s |
| Take S0 | 0.007 s | 0.006 s |
| Two warmup steps | 3.821 s | 3.267 s |
| Restore S0, zero Adam state, take S | 0.006 s | 0.006 s |

The warm request includes 0.058 s to fork. Starting the parent separately took 21.62 s.
The two jobs returned identical loss and gradient norm over all eight steps. These are
single startup measurements with node-local engine sources and warm disk caches.

The four-GB200 dense replay on 2026-11-19 used eight Qwen3-14B-shaped layers, TP2/DP2,
sequence length 4096, global batch 16 and ten steps per check on mock data.

| Edit | Base check | Edited check |
|---|---:|---:|
| LM-head gather before cast, FP32 logits | 4.501 s | 4.503 s |
| Residual plain add | 4.532 s | 4.505 s |
| TE gather before norm | 5.019 s | 5.430 s |

Each row is one pair. All four ranks witnessed each edit and undo restored the base
fingerprint. Across the three edits, the maximum per-step relative differences were 0.2525%
in loss and 1.9140% in gradient norm.

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

## CPU socket placement on a GB200 node

Each node has two CPU sockets. GPUs 0 and 1 are attached to socket 0 (cores 0-71) and GPUs 2 and 3 to
socket 1 (cores 72-143), all four on NVLink (NV18). Our training processes were not bound to cores, so
any of the 144 could run them. Source: our 2026-11-07 hill-climb lab, 14B TP 2, two jobs on one node:
the base on GPUs 0,1 and flag-restart candidates on GPUs 2,3.

| Job, GPUs | CPU binding | tokens/s/GPU (fastest 3 of 8 steps) |
|---|---|---|
| base, 0,1 | none | 6,694-6,801 (7,134-7,537 earlier the same day) |
| candidate, 2,3, same code | none | 4,857-4,862 (0.71-0.73 of the base) |
| candidate, 2,3, same code | `taskset` to socket 1 from start | 7,553-7,569 |
| base, 0,1, bound to socket 0 after it started | `taskset -ap` on the running ranks | 6,725-6,740 (candidate / base 1.12) |

Clocks were at maximum on all four GPUs with no throttle reasons, and the other processes on the GPUs
did no work. The unbound candidate also took 6.5 s, against 2.6 s, to snapshot its start state to
pinned host memory. Binding a job's CPUs to its GPUs' socket from the start made it 11-56% faster.
Rebinding a running job did not recover the same speed, because memory it had already allocated
stayed where it was. Two jobs on the two GPU pairs of one node therefore differ by a fixed factor:
the hill-climb measures that factor with a no-op restart and divides it out of every cross-pair A/B.

## Flag restarts in the hill-climb: a fresh candidate job next to the running base

The harness offers launch-flag changes (a restart) next to live swaps (no restart). The base lab stays
on GPUs 0,1. For each flag-restart candidate, a job on GPUs 2,3 starts with the flags from the same seed and start state.
Checks then alternate between the candidate job and the base, and the ratio is divided by the measured
GPU-pair factor (above). Source: our 2026-11-07 hill-climb lab, 14B TP 2.

| Step | Seconds | Notes |
|---|---|---|
| Request to lab ready, torchrun restart | 73.1-88.8 | about 45 s to the first step, plus 25 s of lab setup (warm-up 15, restore + zero + snapshot 5-8, snapshot 2.6-6.5) |
| Paired checks, 2 pairs of 8 steps | about 175 | the same as for a live-swap candidate |
| Exit of the candidate job | 2-15 | |
| A no-op restart to measure the GPU-pair factor | about 300 | before the first flag restart, then every 6 |

## Upstream PR replay

Six merged upstream PRs as live swaps (`examples/upstream-replay/`): 218 s in all, including two 4-step
base checks. Installs took 0.07-0.60 s, refusals included. A code restart for a refused PR: restart 79-89 s, then
one 8-step check and exit, about 3.5-7.5 min in all. The placement and compile check of a fuzzy-applied
diff takes about 1 s and needs no GPU.

## Pipeline parallelism (`TP2 × PP2`, 4× GB200, 2026-11-19)

Source: our 2026-11-19 NeMo 26.04.01 conformance run on four GB200s (4 layers, hidden 512, 8
attention heads, `--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2
--sequence-parallel --global-batch-size 8 --deterministic-mode`, VPP1, 3-step checks). Pre-run
unit tests (`test_pieces.py`, incl. `PipelineAndVppTest`) passed 13/13 with torch (`-W error::ResourceWarning`).

| Case | Result | Seconds | Notes |
|---|---|---:|---|
| `c0_repeatable` | pass | 1.054 | 3 identical base checks after reset |
| `c1_noop` | pass | 0.786 | `MLP.forward` calls `[48, 48, 48, 48]` across all 4 ranks |
| `c2_visible` | pass | 0.768 | Loss changed `[9.0970, 9.0850, 9.0339] -> [9.1006, 9.0772, 9.0361]`, `[48, 48, 48, 48]` calls |
| `c3_fresh_equivalence` | pass | 87.315 | Live == fresh (visible and 3-edit chain) |
| `c4_uninstall` | pass | 1.456 | Undo `0.016704 s`; identity restored |
| `c5_failed_install` / `c6_refusal` / `c7_chain` | pass | 1.865 / 2.761 / 3.109 | Clean rollback, refusals, 3-edit chain & unwind |
| `c8_crash_and_hang` | pass | 297.264 | Crash / hang / lost / tamper respawns in `28.528–29.042 s` |
| `c9_soak` | pass | 15.813 | 20 cycles; install median `0.0468 s`, undo median `0.0168 s` |
| `c10_not_proven` | pass | 29.012 | `proven: false` (`zero_witness` × 4 ranks) vs altered fresh |

11/11 conformance passed (`108` receipts, GPUs returned to `0 MiB`).

## Virtual pipeline parallelism (`TP2 × PP2 × VPP2`, 4× GB200, 2026-11-19)

Source: our 2026-11-19 NeMo 26.04.01 conformance run on four GB200s (8 layers, hidden 512, 8
attention heads, `--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2
--num-layers-per-virtual-pipeline-stage 2 --sequence-parallel --global-batch-size 8
--deterministic-mode`, VPP2 interleaved chunks, 3-step checks via `fastrsi conformance megatron --real`).

| Case | Result | Seconds | Notes |
|---|---|---:|---|
| `c0_repeatable` | pass | 1.811 | 3 identical base checks after reset |
| `c1_noop` | pass | 1.296 | `MLP.forward` calls `[96, 96, 96, 96]` across all 4 ranks |
| `c2_visible` | pass | 1.285 | Loss changed `[9.1179, 9.0996, 9.0368] -> [9.1245, 9.0891, 9.0389]`, `[96, 96, 96, 96]` calls |
| `c3_fresh_equivalence` | pass | 94.707 | Live == fresh (visible and 3-edit chain) |
| `c4_uninstall` | pass | 2.521 | Undo `0.017231 s`; identity restored |
| `c5_failed_install` / `c6_refusal` / `c7_chain` | pass | 2.905 / 4.013 / 5.220 | Clean rollback, refusals, 3-edit chain & unwind |
| `c8_crash_and_hang` | pass | 306.980 | Crash / hang / lost / tamper respawns in `30.044–31.046 s` |
| `c9_soak` | pass | 26.507 | 20 cycles; install median `0.0468 s`, undo median `0.0170 s` |
| `c10_not_proven` | pass | 31.717 | `proven: false` (`zero_witness` × 4 ranks) vs altered fresh |

11/11 conformance passed (`108` receipts, all 4 GPUs returned to `0 MiB`), with every rank recording all `96` calls across both virtual chunks.

## Zero-warm-up and 3-step short-probe ladder (`Qwen3-30B-A3B` 24L MXFP8 `EP4`, `4× GB200`)

Source: per-check runs on the race recipe (`Qwen3-30B-A3B` cut to 24 layers, MXFP8 `EP4`, `4× GB200`; `266.67 s` restart baseline and `65.52 s` 4-warm-up live baseline):

| Run | Warm-up steps | Timed steps | Wall median (min–max, n=12) | Phase split (`pickup+install` / `restore` / `timed` / `pre_loop+other`) | Graphs / Verdict agreement |
|---|---:|---:|---:|---|---|
| `w-keep-ng-w0` | `0` | `10` | **`45.52 s`** (`45.32–45.72 s`) | `2.75 s` (`0.15 + 2.60`) / `0.27 s` (`0.21 s` copy) / `41.79 s` / `0.71 s` (`0.62 s` pre-loop) | `reused`, `62,851 tok/s/GPU` (`5.9×` vs restart) |
| `w9-keep-ng-w0-s3` (`2026-11-19`) | `0` | `3` | **`17.04 s`** (`16.65–17.25 s`) | `2.64 s` (`0.14 + 2.50`) / `0.33 s` (`0.26 s` copy) / `12.97 s` (`4.28 + 4.34 + 4.34`) / `1.10 s` (`0.79 s` pre-loop) | `reused`, `12/12` ok, `0 MiB × 4` (`15.6×` vs restart) |
| `w10-keep-ng-w0-s1` (`2026-11-19`) | `0` | `1` | **`8.02 s`** (`8.02–8.42 s`, iqr `[8.02, 8.22]`) | `2.71 s` (`0.19 + 2.52`) / `0.30 s` (`0.23 s` copy) / `1` step / `0.69 s` (`0.68 s` pre-loop + `0.01 s` tail) | `reused`, `12/12` ok (`0` failed), `0 MiB × 4` (**`33.3×`** vs restart) |
| `w10-keep-g-w0-s1` (`2026-11-19`, `--edit g`) | `0` | `1` | **`9.63 s`** (`9.43–13.84 s`, iqr `[9.63, 10.83]`, `n=8`) | `2.63 s` (`0.08 + 2.55`) / `0.29 s` (`0.23 s` copy) / `1` step / `1.81 s` (`0.65 s` pre-loop + `1.16 s` tail) | `c00-B 13.84 s` recaptured (`megatron.core.transformer.moe.router` ran during capture); `c01–c07` `9.43–10.83 s`, `8/8` ok, `0 MiB × 4` |

In `probe-pairs6` (`10` candidate/incumbent pairs on the same setup: `5` regressions and `5` speedups), `k=1`, `k=3`, and `k=5` timed steps all agree **`10/10`** with the full `10`-step verdict at `0.5%`, `1%`, and `2%` thresholds.

