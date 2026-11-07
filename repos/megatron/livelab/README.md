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
