# vLLM: which edits install live

Source: our survey of vLLM performance PRs. 959 performance PRs found; 80 inspected by hand.

| Mechanism | Share of 80 | Group | Example |
|---|---:|---|---|
| Swap Python methods or objects in place | 20% | OP, STATE | Scheduler hot patch + drain request/prefix queues (#57951): 66.3 s → 0.6 s |
| Swap code inside compiled graphs, recapture graphs, re-JIT kernels | 51% | GRAPH | GDN spec outputs into caller buffer (#57140): 70.6 s → 2.4 s |
| Swap code and rebuild pre-allocated buffers or runners | 9% | STATE, RESOURCE | |
| Compile C++/CUDA into a versioned shared library, recapture | 9% | NATIVE | fused add RMSNorm (#49750): installed in 28 s by a pre-repo prototype (paper draft); this repo has no NATIVE route yet |
| Needs a restart (startup logic, environment variables) | 11% | | |

## Reset rules (known so far)

- Scheduler and KV manager: swap methods, then drain request queues and clear prefix caches.
- Compiled paths: torch.compile inlines functions, so swapping the Python is not enough;
  recapture the affected CUDA graphs.
- C++/CUDA extensions cannot be unloaded: build a versioned side library and rebind op pointers.
  Not built in this repo: a `csrc/` edit gets a rebuilt extension and a restart. The side library's
  cost is the nvcc compile of the whole edited translation unit (no unit smaller than the file);
  see [costs.md](costs.md), "Native compile".
- Functions decorated with `@eager_break_during_capture` (for example `sparse_attn_indexer`):
  swap `fn.__wrapped__.__code__`, not the wrapper. Breakable-graph replay calls the inner
  function, so no recapture is needed, except for batches captured in `CUDAGraphMode.FULL`.
  Source: `vllm/compilation/breakable_cudagraph.py` at `4e0ad57`. Not yet run.

## Unsupported

Startup logic, launch flags, environment variables.
