# vLLM live lab

Not started. Bootstrap from `core/livelab/`. The v0 plugin (`src/fastrsi/engines/vllm.py`,
11/11 conformance on Qwen3-8B, sampler edits only) is on `legacy/v0` and is the starting point.

- The transaction kernel is the framework's (`core/livelab/kernel/`, engine-agnostic,
  standard library only); importing `adapter` puts `core/livelab` on `sys.path`
  (`FASTRSI_LIVELAB` overrides the folder).
- `adapter/`: the kernel inside a live vLLM engine: the engine-core host and the worker extension.
