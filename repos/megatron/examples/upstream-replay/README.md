# Replaying real upstream Megatron-LM PRs in the live lab

A standing check, like the one we keep for vLLM: take recent merged Megatron-LM PRs that touch code
this workload runs, apply each one to the running training job, and record what the lab did with it.
Each PR goes to the cheapest action that can run it: first a live swap of function bodies into the
running job: the lab replaces the changed function bodies in place on every rank (install
0.07-0.60 s, no restart, measured). When the lab refuses the swap, a
code restart: a fresh job with the PR's modules loaded at import (79-89 s from request to lab ready,
measured).

Workload: Qwen3-14B architecture, TP 2 with sequence parallel, 2 GB200, lr 3e-5, 8 microbatches of
4096 tokens per rank, from the lab's start state (`kb/workload.md`). Base: the Megatron-LM tree the
lab runs (`megatron-core` 0.20 dev, upstream commit ae76c60e6). Source: our 2026-11-07 runs.

## How a PR is replayed

1. Take the PR's merged diff, leaving out tests, docs and examples. Apply it to a copy of the lab's tree.
2. **Placement check (no GPU, about 1 s).** A diff from a newer tree needs fuzz to apply. Check that each
   upstream hunk landed in the function upstream put it in and added the same lines. Then compile every
   changed module.
3. **Live swap.** Send the changed modules' full text to the running job as one install. The kernel compares
   old and new text and either commits a swap of the changed function bodies or refuses with codes.
4. If committed, run one 4-step check from the start state. Compare loss with the base, and read the
   per-function witness: did each changed function run?
5. If refused, do a **code restart**: start a fresh candidate job with the changed modules as an import overlay
   (`FASTRSI_OVERLAY`), run one 8-step check from the same seed and start state, and compare loss with
   the base. A fresh process has no swap to witness, so a code restart shows "no regression", not "proven".

## Records

| PR | Change | Live swap | Witness | Code restart | Result |
|---|---|---|---|---|---|
| NVIDIA/Megatron-LM#8579 | `math.isclose` for gradient-scaling asserts (DDP) | refused in 0.07 s: `constructor_edit` (the asserts are in `DistributedDataParallel.__init__`) | | restart to lab ready 88.8 s; loss within 1.0e-3, grad norm 2.6e-3 of base | runs, no regression (code restart) |
| NVIDIA/Megatron-LM#8532 | fp32 token-scaling factors for low-precision grad buffers | committed in 0.18 s, 3 modules | 0 calls to all 3 `scale_gradients` | | installed, not reached: bf16 buffers here never call it; loss within 4.4e-5 |
| NVIDIA/Megatron-LM#8844 | cache `get_start_time_from_progress_log` | committed in 0.37 s | 0 calls (start-up only) | | installed, not reached; loss within 1.5e-4 |
| NVIDIA/Megatron-LM#7575 | partial activation offload fix | refused in 0.60 s: `constructor_edit`, `new_fields`, `removed_definition` | | restart 78.8 s; loss within 1.5e-3, grad norm 3.9e-3 | runs, no regression (code restart); one hunk left out: its file (an experimental attention variant) is not in this tree |
| NVIDIA/Megatron-LM#8524 | step-wise LR multipliers | refused in 0.16 s: `constructor_edit`, `new_fields`, `outside_changes`, `patch_invalid` | | not run: fuzz put a hunk in the wrong place (`pretrain` signature, a non-default parameter after defaults); the compile check caught it in 0.1 s | needs a manual rebase |
| NVIDIA/Megatron-LM#8246 | RMSNorm gamma weight-decay multiplier | not attempted: the fuzzy apply misplaced a hunk (next column) | | not run: fuzz put the new argument inside another `add_argument` call (syntax error, caught in 0.1 s) | needs a manual rebase |
| NVIDIA/Megatron-LM#7158 | per-group optimizer step counts | | | | does not apply (drift) |
| NVIDIA/Megatron-LM#8424 | residual contracts refactor | | | | does not apply (drift) |

Base vs base, same 4 steps: loss equal to 3 decimals (`12.948, 12.928, 14.255, 15.599`); 7,134 and
7,537 tokens/s/GPU.

Code restart from the zygote (https://github.com/hao-ai-lab/fastrsi/pull/88): the forked ranks re-import
only `megatron`, while torch and Transformer Engine stay loaded. #7575 was ready 40.9 s after the
request, against 78.8 s for the plain restart. It reached the step start at 19.0 s, of which the
re-import took 1.4 s, and its loss stayed within 1.6e-3 of the base. #8579 took 148.7 s: it was the
first fork after the zygote started (40.4 s), and its kernel warm-up took 65.8 s instead of 4-5 s,
not explained yet.

Costs: the live-swap pass over six PRs took 218 s in all (two base checks, six installs, two 4-step checks),
against 450 s predicted. Each code-restart PR took 3.5-7.5 min (restart 79-89 s, an 8-step check, exit). The
#7575 record also includes 22 minutes of waiting for the restart GPUs.

## What the replay says

- **Half of the real PRs cannot be swapped live.** Three of six change constructors, add fields or
  remove definitions. The swap refuses them in under a second with the reason, and they need a restart.
  This is why the restart tier has to be cheap, not only the swap.
- **A third of them are not on this workload's path.** #8532 and #8844 install cleanly, but their
  functions never run here (bf16 gradient buffers, start-up-only code). The zero witness says so;
  without it, an installed edit would look like a successful replay.
- **Fuzzy patching lies quietly.** Two PRs applied "successfully" with fuzz 3 but landed hunks in the
  wrong function. Compiling caught both in 0.1 s each. A misplacement that still compiles would not be
  caught by compiling, so the placement check (same enclosing function, same added lines) runs first.

Not shown: a witness in the code restart (that the PR's code ran in the fresh job); the two PRs that need a rebase; any
speed effect (none of these PRs is a speedup on this workload); more than one check per PR.
