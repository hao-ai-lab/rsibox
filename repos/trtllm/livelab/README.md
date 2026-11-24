# `repos/trtllm/livelab/`

RSIBox adapter for NVIDIA TensorRT-LLM (PyTorch backend).

## Files

- `livelab.json` — manifest registering `ToyTrtllmLab` (`toy`) and `TrtllmLab` (`real`).
- `adapter/__init__.py` — lazy-import entry putting `core/livelab` on `sys.path` and exporting `TrtllmQuiesceHooks`.
- `adapter/agent.py` — spawned worker process running `LocalParticipant` over `rsibox.ipc` (`UnixJsonServer` or `FileChannelServer`) around `tensorrt_llm.LLM` (or `ToyPyExecutor`), plus `--fresh` subprocess execution for `c3_fresh_equivalence` / `c10_not_proven`.
- `adapter/lab.py` — `TrtllmLab(rsibox.BaseLab)` with `StepGapCoordinator`, live atomic Python hot-swaps, `PackageOverlay` fresh checks, fault injection, and `quiesce_hooks()`.
- `adapter/snapshot_hooks.py` — `TrtllmQuiesceHooks` implementing `rsibox.warm.QuiesceHooks` via `kv_cache_manager.impl.release_pools()` / `allocate_pools(False)` (and optional CUDA graph / `virtual_memory` tags).
- `adapter/toy_lab.py` + `adapter/fixture/` — real & toy patch builders (`patches.py`) and two-rank CPU fixture (`sampler.py`, `py_executor.py`) passing `11/11` conformance cases (`c0–c10`).
- `adapter/test_adapter.py` — CPU golden unit tests.
