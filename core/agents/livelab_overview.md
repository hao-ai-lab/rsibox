---
name: livelab
description: Use when changing, testing or profiling code in an ML engine (vLLM, SGLang, FastVideo, Megatron, verl, ...) that has a rsibox live lab under repos/<engine>/livelab. Hot-reload changes into the running engine instead of restarting it.
---

# Live lab

> For the complete codebase directory map, live-sandbox probing commands, and fail-closed refusal vs. bug triage guide, see [`.claude/skills/rsibox/SKILL.md`](../../.claude/skills/rsibox/SKILL.md).

A live lab is a warm, running engine you can change without restarting it. Restarting spends
most of a check on setup (77–90% in our vLLM, Megatron and FastVideo measurements); a live
lab keeps that state resident.

## Calls & CLI

Every lab exposes `install(patch)`, `check(workload)`, `reset()`, `uninstall()`, and `status()`
(`core/livelab/INTERFACE.md`). Drive any lab over HTTP via `rsibox` (`core/livelab/rsibox/README.md`):

```sh
rsibox up --launcher local --lab <engine> [--real] --state /tmp/box1
export RSIBOX=/tmp/box1/box.json
rsibox wait                                 # wait until warm; prints boot_s
rsibox source <module> > mod.py             # fetch live source text
# ... edit mod.py ...
rsibox install --file <module>=mod.py       # atomic swap across ranks (exit 2 if refused)
rsibox check                                # output, proven, witness counts, seconds, cost_s
rsibox fresh-check --file <module>=mod.py   # fresh subprocess parity check (optional --workload)
rsibox uninstall                            # journaled revert to base
rsibox down --state /tmp/box1
```

Engine specifics: `repos/<engine>/docs/architecture.md` and/or `repos/<engine>/livelab/README.md` (not every engine has both — e.g. `transformers` uses [`repos/transformers/docs/architecture.md`](../../repos/transformers/docs/architecture.md)), plus `rsibox conformance <engine> --real -- --help` for GPU flags.

## Before every experiment

1. Read `repos/<engine>/kb/`: probes, their cost, reset rules, unsupported paths.
2. Pick the cheapest probe that still runs the code you changed.
3. Promising result: send it to the full evaluation F(x) asynchronously and keep working.

## Function groups

| Group | What it is | Install refreshes |
|---|---|---|
| OP | Body of a per-call Python function | Swap in place |
| GRAPH | Code inside a CUDA graph or torch.compile region | Swap, recapture affected graphs |
| STATE | Objects built once at start (buffers, runners, batch builders) | Rebuild them in place |
| RESOURCE | Load-time memory layout and allocation | Allocate new, copy, switch, free |
| RANKS | Anything all GPUs must agree on | All ranks or none |
| NATIVE | CUDA/C++ kernel source | Versioned shared library, rebind, recapture |

Startup logic, launch flags and environment variables need a restart.

## Rules

- On `poisoned`, or when the cost hook says a call restarted, do not keep measuring on that lab.
- A check counts only if it says the patched path ran (`proven=True`, non-zero `witness`). Note that `proven=True` proves execution only, **not** numerical equivalence — also verify `output` / `output_digest` / `exact_match` against the base.
- When `check()` returns both `seconds` and `cost_s`, use `seconds` (workload-only) for performance comparisons and `cost_s` (total verb wall time) for overhead tracking.
- Across `reset("respawn")`, compare `status()["identity"]["modules"]` (or `box["modules_digest"]`), which is process-invariant, rather than `status()["fingerprint"]` (which embeds process-local `id(code_obj)`).
- A result counts only with the lab's receipt. Never write or edit one.
- Slow check, failed install or an uncovered module: call `toolsmith_online` (or report the blocker in standalone `rsibox` sandbox runs with no toolsmith peer).
