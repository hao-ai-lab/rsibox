# Engines

One folder per engine, each with the same structure:

```
repos/<engine>/
  docs/        architecture: process tree, restart boundaries, costs
  kb/          knowledge base: costs, workload, mechanisms, reset rules, raw measurements
  livelab/     the engine's live lab, built on template/livelab/
  tools/       proven probes and tools
  examples/    real upstream PRs, replayed in the lab, with receipts
```

Everything about one engine lives in its folder. Code under `repos/<engine>/` never imports from another `repos/<other>/`.

| Engine | Knowledge base | Lab | Replay examples |
|---|---|---|---|
| vllm | Costs, workload, mechanisms, lessons | Adapter (`adapter/`), step-gap coordinator, zygote preload, CUDA graph recapture | `examples/dsv41-prefill` (DeepSeek-V4.1 Flash prefill PRs) |
| fastvideo | Costs, workload, prices, lessons | Adapter (`adapter/`), worker supervisor, zygote respawn, weights holder | `examples/pr-replay` (FastVideo #2041 MiniMax-H3), `examples/h3-live-edit` |
| megatron | Costs, workload, priors, lessons | `fastrsi_megatron` (train-step gap hook, gloo rounds, state snapshots) | `examples/codex-hillclimb` (Qwen3-14B), `examples/upstream-replay` |
| sglang | Planned | Planned | |

Planned: verl, slime, miles, vllm-omni, sglang-omni.
