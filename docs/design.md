# Design

One small core, one contract, one conformance suite. A plugin ships only if it passes.

## Three parts

| Part | Where it runs | Job |
|---|---|---|
| Agent, `agent.py` | Inside every engine process | Compile, swap code, witness, report identity. Standard library only |
| Session, `core.py` | On the host | State machine, two-phase install, receipts, budgets, hang detection |
| Plugin, `engines/` | On the host | Answer engine questions: start, exercise, reset, stats |

## States

```
base --install--> edited --undo--> base
any doubt -----> poisoned --restart--> base
```

A process is poisoned when it dies, hangs, changes its code behind the session, disagrees with
its peers, or exceeds a budget. Poisoned is final until a restart.

## Installing an edit

1. The classifier reads source text and refuses anything a code swap cannot do: new definitions,
   signatures, decorators on changed functions, and code outside a function body.
2. Each process with the file compiles both versions with its own interpreter and checks that its
   live code matches the base source. This rejects stale sources and wrapped functions.
3. Every process prepares, or none commits.
4. Commit assigns `function.__code__`. A failure rolls back and the host proves base.
5. The host reads back a hash of the live code in every process.

Installed does not mean proven. `sys.monitoring` records whether each new function ran, and by
default disables itself after the first hit. An edit inside a compiled graph installs but never
fires, so it reads as not proven.

## Decisions

| Decision | Why |
|---|---|
| Agent boots through `sitecustomize` | Reaches every worker with no engine change and no per-engine RPC |
| One Unix socket per process, in a private temp directory | No pickle, no open port. Socket paths are limited to about 104 bytes, so workdirs cannot hold them |
| Agent thread blocks on `accept` | An idle agent runs no Python |
| Code identity from marshal v2 | Other versions embed refcount-dependent flags |
| Refusal is a normal outcome | An agent needs a reason, not an error |
| Live results are signals | A fresh process stays the only final answer |

## Limits

- Python functions only. No native code, graphs or kernels yet.
- A swap applies to the next call. An in-flight call finishes on old code, so install between
  requests.
- One edit at a time. Undo before the next.
- Functions held only through wrappers are refused rather than patched.
- Edits run unsandboxed in the engine.
