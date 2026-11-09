---
name: livelab
description: Use when changing, testing or profiling code in an ML engine (vLLM, SGLang, FastVideo, Megatron, verl, ...) that has a FastRSI live lab under repos/<engine>/livelab. Hot-reload changes into the running engine instead of restarting it.
---

# Live lab

A live lab is a warm, running engine you can change without restarting it. Restarting spends
most of a check on setup (77–90% in our vLLM, Megatron and FastVideo measurements); a live
lab keeps that state resident.

## Calls

`install(patch)`, `check(workload)`, `reset()`, `uninstall()`, `status()`.
Contract: `core/livelab/INTERFACE.md`. Engine specifics: `repos/<engine>/livelab/README.md`.

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
- A check counts only if it says the patched path ran.
- A result counts only with the lab's receipt. Never write or edit one.
- Slow check, failed install or an uncovered module: call `toolsmith_online`.
