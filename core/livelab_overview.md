---
name: livelab
description: Use when changing, testing or profiling code in an ML engine (vLLM, SGLang, FastVideo, Megatron, verl, ...) that has a FastRSI live lab under repos/<engine>/livelab. Hot-reload changes into the running engine instead of restarting it.
---

# Live lab

A live lab is a running engine you can change without restarting it. Every lab has the same
verbs. How each verb works depends on the engine: read `repos/<engine>/livelab/README.md`.

## Verbs

`start`, `stop`, `install`, `undo`, `status`, `witness`, `capture`, `profile`, `receipt`.
The contract is `template/livelab/INTERFACE.md`.

## Function groups

Every changed function belongs to one group. The group decides how it reloads and how to check it.

| Group | What it is | Reload | Check |
|---|---|---|---|
| OP | Body of a per-call Python function | Swap the body | Rerun on captured inputs |
| GRAPH | Code inside a CUDA graph or torch.compile region | Swap, recapture affected graphs | Graph output equals eager |
| STATE | Objects built once at start | Rebuild in place, keep weights and KV | PR tests plus one real step |
| RESOURCE | Load-time memory layout and allocation | Allocate new, copy, switch, free | Lookups return the same values |
| RANKS | Anything all GPUs must agree on | All ranks or none | All ranks report the same version |
| NATIVE | CUDA/C++ kernel source | Refused: restart | Kernel microbenchmark |

## Rules

- `status` is `base`, `edited` or `poisoned`. On `poisoned`, restart. Do not keep measuring.
- A result counts only with a receipt from the lab. Never write or edit a receipt yourself.
- Building tools: hand off to `toolsmith_offline` (no engine) or `toolsmith_online` (live engine).

<!-- TODO: expand once template/livelab/INTERFACE.md is settled. -->
