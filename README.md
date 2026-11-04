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
core/                    what every agent reads
  livelab_overview.md    skill: a live lab exists, its fixed verbs, function groups
  toolsmith_offline.md   subagent prompt: build tools with no engine running
  toolsmith_online.md    subagent prompt: build tools against the running engine
template/livelab/        the fixed interface, a stub, and the conformance cases
repos/<engine>/          one live lab per engine, plus kept tools and real examples
docs/                    design docs and decisions (see docs/README.md)
```

## How it fits together

1. An agent loads `core/livelab_overview.md` and learns the lab's verbs.
2. For a new engine, `toolsmith_online` bootstraps `repos/<engine>/livelab/` from
   `template/livelab/` and iterates until it passes conformance.
3. For a change, `toolsmith_offline` tags the patch by function group and builds checks;
   `toolsmith_online` reloads it, checks it, and measures it in the live lab.
4. Tools that prove useful are kept in `repos/<engine>/tools/` for the next agent.

## First example

`repos/vllm/examples/dsv41-prefill/`: the three vLLM PRs behind "+35.6–43.9% new-token prefill
throughput" in vLLM's DeepSeek-V4.1-Flash post, hot-reloaded one at a time.

## License

Apache-2.0
