# SGLang live-lab adapter

Owns the five live-lab verbs (`install`, `check`, `reset`, `uninstall`, `status`) on a running
`sglang.Engine` (and on the CPU `toysgl` fixture).

## Contract and invariants

- **Lockstep TP step-gap quiescence (`lab.py`, `agent.py`):** `SglangLab.at_gap` sends an
  `RpcReqInput("rsibox_gap")` on `engine.send_to_rpc`. TP rank 0 receives it in
  `Scheduler.recv_requests` and broadcasts it over `tp_cpu_group` to every TP rank in the same
  scheduler iteration, before `get_next_batch_to_run`. Each rank drains any pending overlap
  `result_queue` via `pause_generation(mode="retract")` + `continue_generation`, parks on its
  main loop thread while Phase 2 runs over per-rank `AF_UNIX` control connections, and exits
  through `handle_rpc_request`'s collective barrier.
- **Selective `CudaGraphRunner` recapture (`recapture.py`):** Edits to `GRAPH`-reachable modules
  (`*.srt.models.*` and `*.srt.layers.*`, excluding eager `sampler` and `logits_processor`)
  automatically run `R1 recapture` across all TP ranks concurrently in the commit gap under
  `witness.capture()`. Unchosen batch sizes are removed from `capture_bs` / `max_bs` so they run
  eagerly on the new code until `recapture_extend`. If recapture or `recapture_extend` fails on
  any single TP rank, every rank rolls back in the same gap (`rebuild_failed`), restoring the
  prior decode graphs without re-capturing.
- **Deterministic `reset("state")` (`agent.py`):** Drains overlap state, asserts
  `scheduler.flush_cache() is True` (clearing `RadixCache`, `req_to_token_pool`, and
  `token_to_kv_pool_allocator`), zeroes hybrid Mamba/GDN recurrent state buffers (guarding
  upstream issue #50139), and resets `random` and `torch` RNG seeds to `random_seed=0`. Both
  `install` (post-commit) and `uninstall` (`R3` undo reset) flush cached KV/recurrent state in
  the step gap so subsequent checks never read pre-edit or post-edit prefix cache entries.
- **Dynamo GMS + non-pauseable decode CUDA graphs (`lab.py`, `agent.py`, `snapshot_hooks.py`):**
  Passing `allow_nonpauseable_cuda_graphs=True` on a Dynamo GMS `SglangLab` enables decode CUDA
  graphs (`cuda_graph_backend_decode="full"`, `cuda_graph_backend_prefill="disabled"` by default)
  by delegating `GMSMemorySaverImpl.cuda_graph` to standard `torch.cuda.graph` in both parent and
  spawned scheduler workers. It fails closed whenever any caller attempts `pause("cuda_graph")` or
  passes `"cuda_graph"` in `SglangQuiesceHooks(release_tags=...)`, keeping CRIU + `cuda-checkpoint`
  snapshot quiesce strictly scoped to `("weights", "kv_cache")`.
- **Zero-copy Dynamo GMS RO meta-loader shim (`agent.py`, `lab.py`):**
  For all Dynamo GMS `SglangLab` runs (independent of `allow_nonpauseable_cuda_graphs`),
  `sync_sglang_gms_globals` patches `GMSModelLoader._create_meta_model` so RO workers build their
  module skeleton directly under `torch.device("meta")` via `_initialize_model` instead of
  SGLang's `DummyModelLoader` (which re-enters `torch.device("cuda")` and leaks a full duplicate
  weight copy into PyTorch's caching allocator). Teardown (`_restore_weight_cache_env`) restores
  the original `_create_meta_model`.
- **Scope limits:** `dp_size=1` and `pp_size=1` without `--enable-dp-attention` or PD
  disaggregation (`docs/architecture.md`).
