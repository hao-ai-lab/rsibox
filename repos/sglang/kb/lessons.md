# SGLang lessons

What misled us (or can corrupt a run) on SGLang and the rule that prevents it. Edit rows in place
when a rule changes; this is not a log. Framework lessons are in `docs/lessons.md`. Bug numbers
below are SGLang upstream PRs and issues from our 2026-11 SGLang audit (TLC/Z3 checks and GPU
repros on GB200 at upstream commit `d800cb859e85`) and code paths in `<sglang>/python/sglang/srt`
at snapshot `cf9845f8e`.

## Quiescence and state reset

| Pitfall | Rule | Evidence |
|---|---|---|
| `event_loop_overlap` (the default scheduler loop unless `--disable-overlap-schedule` is passed) keeps the previous batch's GPU work in `result_queue` at the top of the next iteration; `pause_generation(mode="in_place")` leaves `running_batch`, `last_batch` and `result_queue` untouched. | Drain overlap state via the non-in-place pause path (`mode="retract"`, which pops and processes `result_queue`, clears `last_batch`/`cur_batch`, and retracts `running_batch` to `waiting_queue`) before committing an edit or calling `flush_cache`. | `managers/scheduler.py: Scheduler.event_loop_overlap`, `pause_generation`; `managers/io_struct.py: PauseGenerationReqInput`. |
| `flush_cache` returns `False` when `is_fully_idle()` is false (for example under an `in_place` pause); on a hybrid Mamba/GDN model, a refused non-streaming session turn keeps its `MambaPool` state slot, later runs on that unreset slot, and a paused flush exits the scheduler. | Call `flush_cache` only after draining to full idle, assert its boolean return value is `True`, and close any open sessions before `reset("state")`. | Upstream issue #50139: 1x GB200, Qwen3.5-4B (`d800cb859e85`), wrong output with no error in 2/3 runs (4/4 at `page_size=1`), scheduler exit after pause + flush in 3/3 runs. |
| `MambaPool.clear()` resets `free_slots` to `1..size` without zeroing `mamba_cache.conv` or `mamba_cache.temporal` tensors (only slot 0 is reserved). | Do not assume `flush_cache()` zeroes recurrent/conv state buffers on GPU; an edit to hybrid state handling must initialize every allocated slot before reading it. | `mem_cache/memory_pool.py: MambaPool.clear`, `HybridReqToTokenPool.clear`. |

## Determinism and repeatability

| Pitfall | Rule | Evidence |
|---|---|---|
| Batched greedy decoding (`temperature=0.0`, 32 prompts submitted together) produced multiple distinct outputs across identical runs even with no live-lab agent loaded. | Fix the repeatability and conformance workload at `max_running_requests=1` (or single-request sequential dispatch) with `random_seed=0`. | `legacy/v0` `receipts/control-repeatability.json`: 6 distinct digests in 60 stock runs batched vs 1 distinct digest in 30 runs at `max_running_requests=1` (`conformance-sglang.json`). |
| Even with `--enable-deterministic-inference`, Mamba-1 `dt_proj` selects its matmul kernel by `a.is_contiguous()`, changing fp16/bf16 logprobs between batch-of-1 and batch-of-4 while greedy token ids stay equal. | Gate correctness on exact `output_ids` digests rather than raw logprobs when batch shapes can vary; in v0, `--enable-deterministic-inference` also hit a FlashInfer workspace overflow on Qwen3-8B. | Upstream PR #50148 / issue #50138 (1x GB200, `mamba-130m-hf` at `d800cb859e85`: 15/15 decode positions differed in logprobs, 0/15 in greedy tokens); `legacy/v0` `docs/overhead.md`. |
| `ServerArgs.__post_init__` picks a random `random_seed` when `--random-seed` is omitted, and `Scheduler.__init__` seeds `random`, `numpy` and `torch` once at startup. | Always pass `random_seed=0` at launch and re-run `set_random_seed(self.random_seed)` on every scheduler rank inside `reset("state")`. | `server_args.py`, `managers/scheduler.py: Scheduler.__init__`, `utils/common.py: set_random_seed`. |

## Edits a plain swap does not reach

| Pitfall | Rule | Evidence |
|---|---|---|
| Decode forward passes in `sglang/srt/models/*` and `sglang/srt/layers/*` (except `sampler.py`, which runs in `ModelRunner.sample` after graph replay) are recorded inside `CudaGraphRunner.graphs[bs]`. | Route `GRAPH` edits through `CudaGraphRunner` recapture across all TP ranks in lockstep, and roll back the swap on every rank if recapture raises on any rank. | `model_executor/cuda_graph_runner.py: CudaGraphRunner.capture`; `managers/tp_worker.py: TpModelWorker.forward_batch_generation`. |
| Zero-RoPE MLA KV buffer writes lacked a skip-index guard on padded CUDA graph rows, overwriting reserved slot 0 and producing NaN logits. | Any edit or recapture touching padded decode graphs must preserve slot-0 dummy/padding guards in KV write paths. | Upstream PR #47013 (`GraphPad.tla` TLC trace, verified on upstream `main`). |

## Multi-rank, DP, memory saver, and PD disaggregation hazards

| Pitfall | Rule | Evidence |
|---|---|---|
| Under `dp_size > 1`, `DataParallelController.send_control_message` fans control messages out over ZMQ without a cross-DP barrier, and a `release_memory_occupation` refused on one busy DP rank still releases GPU memory on idle ranks while returning HTTP 400. | Keep the v1 adapter scoped to `dp_size=1` (single TP group synced via `recv_requests` `broadcast_pyobj` on `tp_cpu_group`), and never call `release_memory_occupation` unless every DP rank is verified idle first. | `managers/data_parallel_controller.py: send_control_message`; 487-run TLC trace at DP=2 (`AdminPlane.tla`). |
| Calling `resume_memory_occupation` without a matching prior `release_memory_occupation` raises an uncaught `KeyError` in `offload_tags.remove(tag)` and terminates the scheduler. | Track `offload_tags` state before calling `resume_memory_occupation`, or avoid memory-saver offload calls in the adapter reset path. | `managers/scheduler_update_weights_mixin.py: resume_memory_occupation`; upstream PR #24123 (1-GPU repro). |
| Weight update checks non-paused state before acquiring the writer lock, so a concurrent `in_place` or `retract` pause can freeze readers and deadlock; under DP attention, unordered control vs request channels can drop an early abort (`19/200` lost) or diverge collectives under prefill delayer (`3/3` aborts). | Serialize live-lab pause/swap transactions strictly against `update_weights_*`, and refuse DP-attention (`--enable-dp-attention`) until control-channel ordering is synchronized. | TLC liveness trace (`M1_Live`); upstream PR #50147 / issue #50137 and PR #50146 / issue #50136 (2x GB200, Qwen3-0.6B at `d800cb859e85`). |
| PD disaggregation reuses bootstrap rooms without a generation/epoch tag, causing cross-request RDMA KV overwrites, use-after-free on poisoned aborts, and stale `TransferInfo` eviction. | Keep PD disaggregation (`--disaggregation-mode`) out of scope for the live lab until room generation tagging lands upstream. | Upstream PRs #48889 (68/68 GB200), #48960 (65/65 GB200), #48979 (GB200 repro), #48982 (5/5 GB200), #48883 (1/1 GB200). |
