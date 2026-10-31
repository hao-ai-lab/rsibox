# Add an engine

A plugin answers engine questions. The session owns transactions, identity, witnesses, receipts
and the state machine, so a plugin never touches them.

Start from `src/fastrsi/engines/toy.py`. For a containerized engine, subclass `Container` as
`src/fastrsi/engines/sglang.py` does; it is about 45 lines plus a small server.

| You provide | Purpose |
|---|---|
| `start(agent, overlay)` | Launch to ready. With `agent=True`, pass `agent_env(...)` to the process |
| `stop()`, `alive()` | Lifecycle |
| `root()` | Directory holding the base source that edit paths are relative to |
| `exercise(timeout)` | A deterministic workload returning `digest` and `seconds` |
| `reset()` | Return engine state to a known point, such as flushing caches |
| `stats()` | Measured costs, such as `rss_kib` and `gpu_mib` |
| `fresh(edit)` | A separate engine started with the edit, for comparison |
| `spec()` | The edits the conformance suite uses |

## How the agent reaches your engine

`agent_env` puts a small `sitecustomize` on `PYTHONPATH`. Every Python process that inherits the
environment starts one daemon thread on a private Unix socket. The engine needs no change, and
worker processes need no registration. If your engine clears the environment for its workers,
the agents will not start there, and installs will report that no process has the file.

## Choosing edits for `spec()`

Pick a function that runs as ordinary Python on every request. Ask the engine what it loaded
rather than guessing: engines often have several files with the same name, and the `modules`
agent operation lists what each process really imported.

| Field | Must be |
|---|---|
| `noop` | Changes the code, not the output |
| `visible` | Changes the output |
| `multiple` | At least two functions in one file, so rollback has something to undo |
| `refused` | Edits that must be refused, keyed by refusal code |
| `chain` | Several distinct visible edits |
| `crash`, `hang` | Kill the process, or block it forever, when run |
| `unreached` | Installs, but this workload never runs it as Python |
| `witness_processes` | How many processes must run the visible edit |

## Pass the suite

```bash
fastrsi conformance my_engine
```

An engine is listed as supported only with a passing receipt.
