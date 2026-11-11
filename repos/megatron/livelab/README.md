# Megatron-LM live lab

A running Megatron-LM training job that takes code edits between two train steps and proves
they ran, with the framework's transaction kernel (`core/livelab/kernel`, pure Python)
unchanged. Interface: `core/livelab/INTERFACE-v1.md` (draft).

## Process layout

```
torchrun (one process per GPU, sessions of their own)
 └─ rank r: python -m fastrsi_megatron.run pretrain_gpt.py <megatron args>
     run.py      wraps pretrain()'s phases with timestamps; train_step -> hook
     hook.py     at the first train step, every rank enters the lab:
                   freeze batches -> snapshot S0 -> warm-up steps -> restore S0, zero Adam -> S
                 rank 0: kernel host (StepGapCoordinator + Session), client server
                 ranks: one participant ("ranks"), each kernel call one gloo round
     gap.py      gloo rounds (rank 0 broadcasts, all handle, rank 0 gathers);
                 Unix socket and NFS mailbox servers on rank 0
     state.py    snapshot / restore of weights, fp32 masters, Adam state, scheduler, RNG;
                 frozen batches
client.py        one JSON request per call (socket or mailbox)
zygote.py        restart by fork: torch, TE, Megatron imported once, no CUDA; ranks forked per request
overlay.py       fresh processes with edited modules, same file names and lines
conformance_lab.py   LabUnderTest for core/livelab/conformance
```

## Contract

- **Quiesce point:** the gap before a train step. While the lab serves, no step runs, so phase 1
  and phase 2 of a transaction both happen in the gap and no step sees two code versions.
  (Unlike vLLM, prepare does not overlap steps: in lab mode the steps are the checks.)
- **Reach:** every rank, in rank order, through one gloo group created by the lab. A rank that
  does not answer is `participant_missing` (blocking). Gloo, so the rounds never touch the
  training streams or NCCL communicators.
- **Check:** `check(steps=K)` restores S on every rank, then runs K real Megatron `train_step`s
  on the frozen batches. Output: per-step loss and grad norm (rank 0). Also per-rank step
  seconds and, for each live transaction, the witness per rank.
- **Start state S:** independent of the code under test: weights as initialized, Adam moments
  and step counters zero, scheduler at its start, RNG fixed, batches frozen. So a fresh process
  started with an edit reaches the same S (conformance c3). State keys the lab does not know
  how to reset are a finding at start (`optimizer_state_unknown`), never zeroed silently.
- **Poison:** a round is point-to-point messages between rank 0 and each rank; rank 0 waits for
  each with a deadline, never in a collective. It waits at most `FASTRSI_ROUND_TIMEOUT_S` (default
  900 s) for its sends, and at most that long after its own part for the replies. A rank that has
  not answered by then (it exited, was killed, hangs, or is slower than rank 0 by more than the
  timeout) or a send or receive that fails fails the round and poisons the session
  (`participant_failed`); no round runs after it. Rank 0 keeps the failed round's buffers, so a
  reply that comes late lands there harmlessly. What a rank other than 0 does next depends on the
  transport: it stays blocked in its send or receive on the long-timeout group (seen with gloo's
  macOS transport) until the harness respawns the job, or its send or receive fails (seen on
  Linux) and it exits with code 19 and a one-line reason on stderr, never 0. A rank that answers
  before a slower rank 0 just waits. Not bounded: rank 0's own part (a hang in rank 0's own train
  step). The client's timeout or a closed connection may poison it on the client's side first, and
  `reset("respawn")` restarts the job (the harness does it).
- **Faults reach the rank they name:** `inject` takes a rank number or a participant name
  (`rank1`); a target that names no rank is refused, never skipped on every rank.
- **Clients cannot break the lab.** A request that is not a JSON object with a string `verb`
  gets an error reply. A client that times out or disconnects before its reply costs only that
  reply: rank 0 logs `client gone before the reply` and serves the next request; the other ranks
  never see it.
- **Reset fails closed.** S covers weights, fp32 masters, buffers, Adam state, param-group
  scalars, scheduler, grad scaler, and torch CPU/CUDA, TP-tracker, Python and numpy RNG. If the
  live state has a tensor S lacks, lacks one S has, or one changed shape or dtype, `reset` (and
  `check`) answer ok false and write nothing. `digest` returns each rank's sha256 of every piece
  (numpy not needed), so a round trip is shown bit for bit; a digest that fails on a rank returns
  no hash. `resume` puts S back (`state: "keep"` skips that) and feeds
  the job's next steps the frozen batches first: the job continues as if the lab had not run.
- **Respawn kills by PID tree.** torchrun starts workers in sessions of their own: a kill of the
  launcher's process group leaves them holding GPU memory.

## What install covers here

A patch may edit any module the job has imported; each one it names joins the session's code
fingerprint (status, tamper checks). A module the job has not imported is refused
(`patch_invalid`).

The kernel's live code swap (function bodies, every rank) and repair cells. Not yet: edits of
`jit_fuser` (torch.compile) functions, CUDA graph regions, constructors (which require a model
rebuild), optimizer construction, TE C++ code.

## Kernel location

The kernel and the pre-fork guards are imported from `core/livelab` (`kernel/`,
`forksafe.py`), which `fastrsi_megatron/__init__.py` puts on `sys.path`; `FASTRSI_LIVELAB` (or
`FASTRSI_KERNEL`) names another folder.

## Conformance

`conformance_lab.py` is the `LabUnderTest`: it starts the job with `FASTRSI_LAB=1` in a session
of its own, drives it over the socket, and builds its patches from the running tree's
`megatron/core/transformer/mlp.py` and `attention.py`. Faults are real: crash and lost make a
rank's process exit, hang makes a rank sleep inside a train step, tamper rebinds a function's
code outside the journal.

```sh
FASTRSI_MG_CMD='<a command that runs torchrun -m fastrsi_megatron.run pretrain_gpt.py ...>' \
FASTRSI_MG_ROOT=<Megatron-LM tree> FASTRSI_CHECK_STEPS=3 FASTRSI_CHECK_TIMEOUT=60 \
python3 -m conformance --lab fastrsi_megatron.conformance_lab:MegatronLab   # from core/livelab
```

Our 2026-11-07 run: 2 layers, hidden 512, TP 2 on 2 GB200, `--deterministic-mode`, 3-step
checks, 11/11 as expected:

```
case                  result  seconds  evidence
c0_repeatable         pass      1.406  3 checks (one after reset ['weights', 'optimizer', 'scheduler', 'rng', 'data']) gave {'loss': [12.064836502075195, 12.067537307739258, 11.609490394592285], ...}
c1_noop               pass      1.043  installed by ['R0 swap']; calls {'megatron.core.transformer.mlp:MLP.forward:259': [48, 48]}; output unchanged
c2_visible            pass      1.038  loss [12.0648, 12.0675, 11.6095] -> [12.0546, 12.0663, 11.5643]; calls per participant [48, 48]
c3_fresh_equivalence  pass    124.224  visible: live == fresh; chain of 3: live == fresh (bit-exact loss and grad norm)
c4_uninstall          pass      1.937  uninstall ['txn-3728055d'] in 0.026158 s; identity 2138091df388705c restored
c5_failed_install     pass      2.547  fail_apply: rolled_back, codes ['participant_failed'], base proven
                                       fail_commit: rolled_back, codes ['participant_failed'], base proven
                                       control: no fault, committed 3 targets, output changed
c6_refusal            pass      3.481  syntax: ['patch_invalid']; closure: ['closure_changed']
                                       module_state: ['empty_change', 'outside_changes']; empty: ['empty_change']
c7_chain              pass      5.757  3 stacked edits each visible, undone one by one to base
c8_crash_and_hang     pass    243.804  crash: poisoned by ['participant_failed'], respawn 46.029 s
                                       hang: poisoned by ['participant_failed'], respawn 51.576 s
                                       lost: poisoned by ['participant_failed'], respawn 37.613 s
                                       tamper: poisoned by ['tampered'], respawn 45.019 s
c9_soak               pass     27.711  20 cycles; install median 0.0716 s, uninstall median 0.0320 s
c10_not_proven        pass     48.798  MLP.__init__ edit: proven False, findings ['zero_witness', 'zero_witness'];
                                       live loss stays at base, a fresh process differs
```

What it does not show: more than 2 ranks, PP > 1, edits of `jit_fuser` functions or CUDA graphs,
the 14B model (see the hill-climb), a conformance receipt (the receipt writer has no record kind
for it yet).

## Restart by fork: the zygote

`zygote.py` keeps one process per node with torch, Transformer Engine, Megatron and the training
script's imports loaded, and CUDA not initialized. A restart forks the ranks from it instead of
`torchrun` + exec, so a new process skips 10-50 s of imports and pays only Megatron's setup and
the first step. Use it when a candidate differs in launch flags or NCCL settings; code edits go
through the live lab (or a fresh `torchrun` with an overlay, which the zygote refuses).

```sh
python -m fastrsi_megatron.zygote serve --sock S --script pretrain_gpt.py      # once
FASTRSI_T0=$(date +%s.%N) python -m fastrsi_megatron.zygote start --sock S --nproc 4 --port P \
    --log-dir D --cwd D --env FASTRSI_RUN=D [--env K=V ...] -- pretrain_gpt.py <megatron args>
```

Contract: each rank gets torchrun's variables (OMP_NUM_THREADS=1 too, applied to torch's pool)
and the request's, a session of its own, `D/rank<r>.log`, optionally a CPU list (`--cpus`), and
torch.cuda's own functions back; it runs `fastrsi_megatron.run` (phases, lab hook) with FASTRSI_T0 = the request
time. One rank exiting non-zero stops the others by PID tree. No rank or child of a rank outlives
its job: the zygote is child subreaper and reaps everything after each job, each rank dies with
the zygote (PR_SET_PDEATHSIG), a `fastrsi-guard` process per rank kills the rank's process group
when the rank dies, and a client that disconnects stops its job (`on_client_gone: "keep"` lets it
run). A malformed request gets an `error` event; the zygote serves on. Findings refuse a request unless
it accepts them: `stale` (an imported source file changed), `env` (a variable a library may
read at import, such as NVTE_*, differs from the zygote's), `overlay` (FASTRSI_OVERLAY replaces a
module the zygote has imported).

Keeping CUDA uninitialized: Megatron's import reaches three libraries that ask about the GPU at
import. quack (through TE) calls `get_device_capability`: answered from NVML by the framework's
`forksafe.py` (`core/livelab`). tvm_ffi calls `is_available`: answered by NVML with
PYTORCH_NVML_BASED_CUDA_CHECK=1. flashinfer's gdn kernels read
`get_device_properties(0).multi_processor_count` at module level: answered with the values CUDA
reported in a one-off helper process (`forksafe.install_properties`, 1.5-1.9 s at zygote
start). Without these the zygote stops and names the module.

Receipt (our 2026-11-07 runs, 4x GB200, environment copied to node-local RAM disk; from the request to the first train
step done):

| | torchrun, env on NFS | torchrun, env local | zygote | zygote + dynamo/inductor preloaded, TORCHINDUCTOR_COMPILE_THREADS=1 |
|---|---|---|---|---|
| Qwen3-14B arch, TP 2 | 58.0 s | 29.3 s | 17.1-19.6 s (6 runs) | 15.0-16.5 s (3) |
| Qwen3-32B arch, TP 4 | 110.1 s | 37.6 s | 24.5-27.5 s (4) | 22.8-25.1 s (4) |

The zygote runs include another NCCL setting (NCCL_ALGO=Ring) and a cached `--cross-entropy-loss-fusion`; that flag's
first torch.compile adds 6-9 s (23.7 s, 30.8 s). One 14B restart took 40.1 s (a 17 s first step), not reproduced in 12
more. The zygote is ready 12-25 s after its start (64 native threads, one Python thread at fork); a fork takes 0.1-0.3 s.
Same results as `torchrun` (no deterministic mode): loss of steps 1-2 equal to 4 decimals (14B 12.9215, 12.9391; 32B
12.8841, 12.9135); step 3 within the spread of `torchrun` repeats (14B: zygote 14.1019-14.1043, torchrun 14.1019-14.1059;
32B: 19.0409-19.0575 vs 19.0453-19.0471); steady step 4.6 s (14B), 6.7-7.2 s (32B) either way.
Steady state is slower in forked ranks than in torchrun ranks: 14B, same CPU binding, steps 3-12,
torchrun 4.40-4.46 s per step (14 runs) against 4.50-4.93 s for forked ranks (22 runs), +2% early in
the day and +8-11% later on the same node while torchrun stayed put. Bisected without finding the
cause: it stays when the zygote imported nothing before the fork, and does not depend on the
OpenMP pool, the CUDA-query patches, garbage collection (same collections, gc.freeze no help),
TORCH_NCCL_ASYNC_ERROR_HANDLING or where the rendezvous store runs. A forked rank has 31 more
native threads than a torchrun rank (176 vs 144). Do not compare throughput between a forked and a
torchrun start; use torchrun for throughput and the zygote for fast loss and correctness checks.
What remains per restart is Megatron's own: init 2.2-4.5 s, warm-up 0.6-3 s, build 0.7-1 s, and the first step (9.4-11.4 s
at 14B, 14-17.4 s at 32B: lazy communicators, cuDNN attention plans, allocator growth; see `kb/costs.md`).
