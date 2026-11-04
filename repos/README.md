# Engines

One folder per engine. Each has the same three parts:

```
repos/<engine>/
  livelab/     the engine's live lab, bootstrapped from template/livelab/
  tools/       tools agents built and we kept, each listed in tools/README.md
  examples/    real upstream PRs, replayed in the lab, with receipts
```

| Engine | Lab | First example |
|---|---|---|
| vllm | not started (v0 plugin on `legacy/v0`) | `examples/dsv41-prefill` |
| sglang | not started (v0 plugin on `legacy/v0`) | |
| fastvideo | not started | MiniMax-H3 VAE fusion (FastVideo #2041) |

Planned: megatron, verl, slime, miles, vllm-omni, sglang-omni.
