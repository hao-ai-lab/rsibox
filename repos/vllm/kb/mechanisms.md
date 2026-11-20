# vLLM: which edits install live

Source: our survey of vLLM performance PRs (`2026-07-24` to `2026-10-24`: 487 performance PRs found, 80 sampled and inspected by hand; **71/80 = 89%** install in place in this repo: `63/80` via Python/graph/resource swaps plus `8/80 = 10%` C++/CUDA `NATIVE` edits via [`repos/vllm/livelab/adapter/native.py`](../livelab/adapter/native.py), leaving `9/80 = 11%` requiring a restart).

| Mechanism | Share of 80 | Group | Example |
|---|---:|---|---|
| Swap Python methods or objects in place | 20% | OP, STATE | Scheduler hot patch + drain request/prefix queues (#57951): 66.3 s → 0.6 s |
| Swap code inside compiled graphs, recapture graphs, re-JIT kernels | 51% | GRAPH | GDN spec outputs into caller buffer (#57140): 70.6 s → 2.4 s |
| Swap code and rebuild pre-allocated buffers or runners | 8% | STATE, RESOURCE | |
| Compile C++/CUDA into a versioned shared library, recapture | 10% | NATIVE | fused add RMSNorm (#49750): live 19.22 s total incl. check (17.12 s commit [17.09 s nvcc build, 1.17 ms load] + 0.83 s R1 recapture + 1.27 s check) vs 49.47 s restart, undo 6.55 ms (`native.py`) |
| Needs a restart (Rust rebuild, weights, startup logic, launch flags) | 11% | | |

## Reset rules (known so far)

- Scheduler and KV manager: swap methods, then drain request queues and clear prefix caches.
- Compiled paths: torch.compile inlines functions, so swapping the Python is not enough;
  recapture the affected CUDA graphs.
- C++/CUDA extensions cannot be `dlclose`-unloaded safely (`STABLE_TORCH_LIBRARY` / PyTorch
  dispatcher state): [`native.py`](../livelab/adapter/native.py) builds edited `.cu` files + a
  generated `_Cv<key>` `reg.cu` into node-local `/tmp/fastrsi-native-<uid>/<key>/libfastrsi_<key>.so`
  under `fcntl.flock`, loads via `ctypes.CDLL(..., RTLD_NOW | RTLD_LOCAL)`, binds the op on
  `torch.library.Library(ns, "IMPL")` inside the commit gap (paired with `R1` recapture when
  CUDA graphs are active), and undoes in milliseconds via `lib._destroy()`. The cost floor
  remains the `nvcc` compile of the edited translation unit; see [costs.md](costs.md).
- Functions decorated with `@eager_break_during_capture` (for example `sparse_attn_indexer`):
  swap `fn.__wrapped__.__code__`, not the wrapper. Breakable-graph replay calls the inner
  function, so no recapture is needed, except for batches captured in `CUDAGraphMode.FULL`.
  Source: `vllm/compilation/breakable_cudagraph.py` at `4e0ad57`. Not yet run.

## Unsupported

Startup logic, launch flags, environment variables.
