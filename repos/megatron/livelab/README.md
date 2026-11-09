# Megatron-LM live lab

A running Megatron-LM training job that takes code edits between two train steps and proves
they ran, with the vLLM lab's transaction kernel (`repos/vllm/livelab/kernel`, pure Python)
unchanged. Interface: `template/livelab/INTERFACE-v1.md` (draft).

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
overlay.py       fresh processes with edited modules, same file names and lines
conformance_lab.py   LabUnderTest for template/livelab/conformance
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
- **Poison:** the lab cannot answer for a hung or dead rank; the client's timeout or a closed
  connection poisons it, and `reset("respawn")` restarts the job (the harness does it).
- **Respawn kills by PID tree.** torchrun starts workers in sessions of their own: a kill of the
  launcher's process group leaves them holding GPU memory.

## What install covers here

The kernel's R0 swap (function bodies, every rank) and R10 repair cells. Not yet: edits of
`jit_fuser` (torch.compile) functions, CUDA graph regions, constructors (they need a model
rebuild: R5 in Megatron terms), optimizer construction, TE C++ code.

## Kernel location

The kernel is imported from `repos/vllm/livelab/kernel` (put on `sys.path` by
`fastrsi_megatron/__init__.py`, or `FASTRSI_KERNEL`). Megatron is the second engine that needs
it; moving it to the framework is its own PR.

## Conformance

`conformance_lab.py` is the `LabUnderTest`: it starts the job with `FASTRSI_LAB=1` in a session
of its own, drives it over the socket, and builds its patches from the running tree's
`megatron/core/transformer/mlp.py` and `attention.py`. Faults are real: crash and lost make a
rank's process exit, hang makes a rank sleep inside a train step, tamper rebinds a function's
code outside the journal.

```sh
FASTRSI_MG_CMD='<a command that runs torchrun -m fastrsi_megatron.run pretrain_gpt.py ...>' \
FASTRSI_MG_ROOT=<Megatron-LM tree> FASTRSI_CHECK_STEPS=3 FASTRSI_CHECK_TIMEOUT=60 \
python3 -m conformance --lab fastrsi_megatron.conformance_lab:MegatronLab   # from template/livelab
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
