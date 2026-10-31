# FastRSI

Edit a running ML engine without restarting it. FastRSI installs the edit into the live processes,
proves the new code ran, undoes it on request, and tells you when only a restart can be trusted.

FastRSI is a development tool. Edits run unsandboxed inside the engine. Do not use it in
production.

This is an early, private version. It handles Python functions only.

## Try it without a GPU

```bash
uv sync
uv run fastrsi demo
```

```
installed  a1b2c3  processes=3
witness    ran in 2 of 3 processes
output     changed
undone     base identity proven, output restored
refused    forward: signature changed: restart needed
```

## Use it on an engine

```bash
export FASTRSI_MODEL=/path/to/model
fastrsi start vllm -o gpu=2
fastrsi exercise
fastrsi install fix.patch
fastrsi status
fastrsi fresh fix.patch
fastrsi undo
fastrsi stop
```

Patch paths are relative to the engine's site directory, with `a/` and `b/` prefixes.

`install` exits 0 when the edit is live, 2 when it is refused, and 1 when it fails. A failed
install leaves the engine on base. Every command writes a receipt to `.fastrsi/session/receipts/`.

`status` reports one of three states, and whether the edit has actually run:

| State | Meaning |
|---|---|
| `base` | Original code, proven by hash in every process |
| `edited` | Your edit is live. `proven` says whether every edited function has run |
| `poisoned` | FastRSI no longer trusts this process. Run `fastrsi restart` |

A live result is a signal, not a final answer. `fastrsi fresh` starts a separate engine with the
edit and compares outputs. It needs a second GPU.

## Supported engines

| Engine | Conformance | Reaches | Needs a restart |
|---|---|---|---|
| Toy | 11 of 11, in CI | Function bodies | Signatures, new definitions, module code |
| vLLM | 11 of 11, [receipt](receipts/conformance-vllm.json) | Python that runs per request, such as the sampler | Same, plus anything in a compiled graph |
| SGLang, one request at a time | 11 of 11, [receipt](receipts/conformance-sglang.json) | Python that runs per request, such as the sampler | Same |

An edit inside a compiled graph installs, but `status` reports it as not proven.

## What FastRSI costs

The [overhead book](docs/overhead.md) reports idle cost, memory, install time, growth and failure
detection per engine. Rerun it with:

```bash
fastrsi overhead vllm -o gpu=2
python3 tools/overhead_doc.py
```

This page makes no speedup claims.

## Known limits

- Python functions only: no native code, compiled graphs or kernels.
- One edit at a time, installed between requests.
- Hangs are found by timeout, so detection takes as long as the timeout you set.
- Exact comparison needs a repeatable workload. `c0_repeatable` checks this first.

## More

- [Design](docs/design.md)
- [Conformance](docs/conformance.md)
- [Add an engine](docs/adding-an-engine.md)

## License

Apache-2.0
