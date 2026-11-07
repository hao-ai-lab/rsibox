# Engines

One folder per engine, each with the same layout:

```
repos/<engine>/
  docs/        architecture: process tree, restart boundaries and their costs
  kb/          knowledge base: costs, workload, mechanisms, reset rules, raw measurements
  livelab/     the engine's live lab, one folder per boundary, following template/livelab/
  tools/       tools agents built and we kept, each listed in tools/README.md
  patches/     our edits to the engine itself, with a ledger for upstreaming
  examples/    real upstream PRs, replayed in the lab, with receipts
```

Everything about one engine lives here, including its docs and measurements.

| Engine | Knowledge base | Lab | First example |
|---|---|---|---|
| vllm | `kb/`: costs, workload, mechanisms | `livelab/`: kernel, adapter, services | `examples/dsv41-prefill` (patches, no receipts yet) |

Planned: sglang, fastvideo (first example: MiniMax-H3 VAE fusion, FastVideo #2041), megatron,
verl, slime, miles, vllm-omni, sglang-omni.
