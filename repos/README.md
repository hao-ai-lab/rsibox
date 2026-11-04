# Engines

One folder per engine, each with the same four parts:

```
repos/<engine>/
  kb/          knowledge base: costs, workload, mechanisms, reset rules, raw measurements
  livelab/     the engine's live lab, bootstrapped from template/livelab/
  tools/       tools agents built and we kept, each listed in tools/README.md
  examples/    real upstream PRs, replayed in the lab, with receipts
```

Everything about one engine lives here, including its docs and measurements.

| Engine | Knowledge base | Lab | First example |
|---|---|---|---|
| vllm | prior measurements in | not started (v0 plugin on `legacy/v0`) | `examples/dsv41-prefill` (patches ready) |
| sglang | | not started (v0 plugin on `legacy/v0`) | |
| fastvideo | | not started | MiniMax-H3 VAE fusion (FastVideo #2041) |

Planned: megatron, verl, slime, miles, vllm-omni, sglang-omni.
