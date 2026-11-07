# FastRSI

Hot-reload your inference engine. Make it faster while it keeps running.

FastRSI gives a coding agent a **live lab**: a running engine it can change without a restart.
The agent swaps a code change into the live processes, proves the new code ran, measures it, and
undoes it on request. When a change cannot be swapped safely, the lab says "restart needed".

> **Status:** restarting development around a skill-first design. Nothing here runs yet.
> The previous version (Python package, vLLM and SGLang plugins, conformance receipts) is on the
> [`legacy/v0`](https://github.com/GindaChen/fastrsi/tree/legacy/v0) branch.

## Layout

```
core/                    what agents read
  livelab_overview.md    skill: the live lab, its calls, function groups
  developer.md           developer agent: edit, install, check, keep or revert
  toolsmith_offline.md   toolsmith, once per repo: mine PRs and tests, build and verify the lab
  toolsmith_online.md    toolsmith, on call: polish slow checks, extend to new modules
  orchestrator.md        orchestrator agent: run the loop with programs and agents, schedule by GPU
template/livelab/        blueprint for a new engine: the fixed interface (install, check, reset,
                         uninstall), a stub and conformance cases
docs/                    how this repo and its process work (see docs/README.md)
repos/<engine>/          everything about one engine:
  docs/                  architecture: process tree, restart boundaries and their costs
  kb/                    measured facts: costs, workload, lessons
  livelab/               all code, one folder per boundary (kernel/, adapter/, services/, ...)
  tools/ patches/        proven probes; our edits to the engine, with an upstreaming ledger
  examples/<name>/       one walkthrough per replayed change, with its receipt
```

## How it fits together

1. **Offline, once per repo:** `toolsmith_offline` mines the repo's PRs and tests, bootstraps
   `repos/<engine>/livelab/` from the template, verifies it against fresh processes, and fills
   `repos/<engine>/kb/`.
2. **Online:** the developer agent loads the skill, then edits, `install`s, `check`s, and keeps
   (`reset`) or reverts (`uninstall`) inside the warm engine. Promising edits run the full
   benchmark asynchronously.
3. **On call:** a cost hook flags slow checks and fallback restarts; `toolsmith_online` then
   polishes or extends the lab and saves the tool for the next agent.

## First example

`repos/vllm/examples/dsv41-prefill/`: the three vLLM PRs behind "+35.6–43.9% new-token prefill
throughput" in vLLM's DeepSeek-V4.1-Flash post, hot-reloaded one at a time.

## License

Apache-2.0
