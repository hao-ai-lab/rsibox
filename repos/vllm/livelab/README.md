# vLLM live lab

The live lab for vLLM, following `template/livelab/`. One folder per boundary:

- `kernel/`: the transaction kernel (engine-agnostic, standard library only).
- `adapter/`: the kernel inside a live vLLM engine: the engine-core host and the worker extension.
- `services/`: processes and state kept warm outside the engine: the launcher, the zygote and
  its module refresh, and the state holders for weights and the KV plan.

How they map onto vLLM's processes and restart boundaries: `../docs/architecture.md`.
