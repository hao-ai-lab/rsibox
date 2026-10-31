# FastRSI

Edit a running ML engine without restarting it. FastRSI installs the edit into the live processes, proves the new code
ran, undoes it on request, and tells you when only a restart can be trusted.

FastRSI is a development tool. Edits run unsandboxed inside the engine. Do not use it in production.

## Try it without a GPU

```bash
uv pip install fastrsi
fastrsi demo
```

```
installed  edit-1   route=function  processes=3/3
witness    edit-1   ran in 3/3
output     changed
undone     edit-1   base identity and output restored
refused    edit-2   signature change: restart needed
```

## Use it on an engine

```bash
fastrsi start vllm -- --model Qwen/Qwen3-8B
fastrsi install fix.patch
fastrsi check -- python my_check.py
fastrsi undo
fastrsi status
```

`install` exits 0 when the edit is live and proven, 2 when it is refused, and 1 when it fails. A failed install leaves
the engine on base. Every command writes a receipt to `.fastrsi/receipts/`.

`status` reports one of three states:

| State | Meaning |
|---|---|
| `base` | Original code, proven |
| `edited` | Your edit is live and witnessed |
| `poisoned` | FastRSI no longer trusts this process. Restart it |

A live result is a signal, not a final answer. Before you rely on one, run the same edit in a fresh process:

```bash
fastrsi check --fresh -- python my_check.py
```

## Supported engines

| Engine | Works live | Needs a restart |
|---|---|---|
| Toy | Functions | Signatures, new modules |
| vLLM, TP<=2 | To be filled from conformance receipts | To be filled |
| verl | To be filled | To be filled |
| Kernels | To be filled | To be filled |

An engine is listed only if it passes the [conformance suite](docs/conformance.md).

## What FastRSI costs

FastRSI is not free. The [overhead book](docs/overhead.md) reports its idle cost, memory, install time, growth, drift
against a cold process, and time to detect failure, per engine, generated from receipts you can rerun:

```bash
fastrsi overhead vllm
```

This page makes no speedup claims.

## Known limits

- State can drift in a long-lived process. Long edit chains can read differently from a cold run.
- Hangs and bad memory are detected by timeouts, so detection can take minutes.
- Native crashes can leave a process unusable. FastRSI marks it `poisoned`.

## Add an engine

Copy `fastrsi/engines/toy.py`, implement the contract, then run:

```bash
fastrsi conformance my_engine
```

## License

Apache-2.0
