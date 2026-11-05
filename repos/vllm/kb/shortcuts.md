# vLLM development shortcuts

Ways to get an experiment running before everything is ready. Each one skips something; the
last column says what you can no longer conclude. Numbers are from our runs on DeepSeek-V4.1-Flash
(4× GB200, TP4) unless marked.

| Shortcut | How | Saves | You can no longer conclude |
|---|---|---|---|
| Random weights | `--load-format dummy` (seeded: `initialize_dummy_weights`, seed 1234) | Downloading and loading weights (476 GB; preload took about 10.5 min) | Anything about answers. Outputs are noise |
| Resident weights | `vllm preload` once, then `--load-format ipc_cache` with `fallback: false` | Weight load on every engine start | Nothing, if fallback is off (the engine refuses to start otherwise) |
| Warm kernel caches | One persistent cache root (`livelab/startup/caches.py`), same config | FlashInfer autotune (about 415 s cold), JIT compiles | Nothing, once a second start adds no cache files |
| Skip graph-memory profiling | `livelab/startup/graphmem.py` (`FASTRSI_GRAPHMEM=1`) | About 24 s per start | KV cache size may be off if an edit changes graph memory without a new salt |
| Smaller context | `--max-model-len` small | KV cache setup; but changes the config hash, so autotune caches miss | Results at long context |
| No CUDA graphs | `--enforce-eager` | Graph capture (about 42 s) and its memory profiling | Speed (eager is slower); GRAPH-group edits are not exercised |
| Skip JIT warmup | `kernel_config.enable_jit_warmup` off | About 26 s | First requests stall while kernels compile, unless caches are warm |

## Correctness without real weights

Random-weight outputs are not answers, but an edit can still be checked against the base on
the same engine: same prompts, compare logits (not the top token) within a tolerance. This
needs a repeatable base first. On our first runs base vs base again matched 0 of 5 top tokens,
from run-to-run numerics, not the weights (they are seeded). Try `VLLM_BATCH_INVARIANT=1` and
confirm base vs base again before trusting any comparison. Not yet done.
