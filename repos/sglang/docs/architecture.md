# SGLang architecture: process tree, restart boundaries, costs

This page is the map a reviewer needs for any SGLang live-lab PR: what runs where, which box a
change invalidates, what restarting that box costs, and where the live lab hooks in. Numbers
come from `../kb/costs.md` (source named there); anything we have not timed is marked **not
measured**.

Source read: the `sgl-project/sglang` snapshot at `cf9845f8e` (`0.5.10.post2.dev438`). Below,
`<srt>` is `<sglang>/python/sglang/srt`, and a citation `file: function` names upstream code at
that snapshot. Bug numbers are SGLang upstream PRs and issues from our 2026-11 SGLang bug audit
(TLC/Z3 model checks with GPU repros); those repros ran at a newer upstream commit, `d800cb859e85`.

## 1. Process tree (`sglang.Engine` or `launch_server`, PD disaggregation off)

```
main process                   Engine / HTTP server, TokenizerManager (tokenize, stream, pause)
  |  mp.Process per rank       entrypoints/engine.py: _launch_scheduler_processes
  v
scheduler, one per TP x PP rank (DP = 1)   or   DataParallelController -> one TP group per DP rank
  |  in-process                Scheduler: event loop, waiting queue, batches, RadixCache
  v
TpModelWorker                  managers/tp_worker.py
  |  in-process
  v
ModelRunner                    model, sampler, attention backend, pools, graphs
   +-- CudaGraphRunner         decode graphs per batch size (model_executor/cuda_graph_runner.py)
   +-- req_to_token pool       ReqToTokenPool, or HybridReqToTokenPool + MambaPool (Mamba/GDN state)
   +-- KV pool + allocator     token_to_kv_pool_allocator
main process
  |  mp.Process
  v
DetokenizerManager             detokenize, send text back to the TokenizerManager
```

- Requests travel over ZMQ: TokenizerManager -> scheduler TP rank 0 (`scheduler_input_ipc_name`);
  `Engine.collective_rpc` uses a second socket (`rpc_ipc_name`) into the same rank
  (`managers/scheduler.py: Scheduler.init_ipc_channels`). Only attention-TP rank 0 of PP stage 0
  owns these sockets; the other ranks get requests from it.
- Each arrow is "starts and owns". A scheduler that raises sends SIGQUIT to its parent
  (`managers/scheduler.py: run_scheduler_process`), and the main process watches scheduler and
  detokenizer liveness and SIGQUITs on a crash (`utils/watchdog.py: SubprocessWatchdog`); on Linux
  each scheduler also dies with its parent (`utils/common.py: kill_itself_when_parent_died`).
  So any rank failing takes the whole server down. Upstream has no component restart.
- Plugins registered under the `sglang.srt.plugins` entry point run once at the start of every
  scheduler process, before the `Scheduler` is built (`plugins/__init__.py: load_plugins`, called
  from `run_scheduler_process`). That is the sanctioned way to put code into every rank.

## 2. Init chain, per scheduler rank

`ModelRunner.initialize` (`model_executor/model_runner.py`), in order:

| Step | What it does |
|---|---|
| `init_torch_distributed` | default process group (NCCL on CUDA), TP/PP groups, custom all-reduce |
| `create_sampler`, `load_model` | build the model, load weights |
| `init_memory_pool` | profile, then size and allocate the req_to_token, KV and (hybrid models) Mamba pools |
| `init_attention_backend`, `kernel_warmup` | attention backend, kernel JIT and warmup |
| `init_device_graphs` | build a new `CudaGraphRunner`, whose constructor captures every decode batch size |
| `init_piecewise_cuda_graphs` | optional piecewise graphs (prefill) |

Then the `Scheduler` builds its tree cache, seeds every library with `random_seed`
(`managers/scheduler.py: Scheduler.__init__`, `utils/common.py: set_random_seed`; a random seed is
drawn when the flag is unset, `server_args.py`), reports ready over a pipe and enters its loop.

Measured shape: a fresh Qwen3-8B engine start took 79.0 s median in the v0 prototype
(`kb/costs.md`). Per-phase times for this snapshot: **not measured**.

## 3. Atomic units

- **TP (and PP) rank group.** The default process group, TP groups, custom all-reduce IPC
  handles and symmetric memory span all ranks. One rank cannot restart alone; the unit is the
  server.
- **Model -> pools -> graphs.** The KV and Mamba pool sizes come from profiling the loaded model;
  captured graphs hold pool and buffer addresses. A model edit that changes layer types or
  head layout changes the pools and invalidates every graph. Graphs are not cacheable across
  processes.
- **Scheduler bookkeeping vs device memory.** The tree cache, `req_to_token_pool` and the KV
  allocator say which slots hold what; the device tensors hold the data. `flush_cache` resets
  them together, and only when the scheduler is fully idle (`managers/scheduler.py:
  Scheduler.flush_cache`, `is_fully_idle`).

## 4. Restart boundaries

| Boundary | What survives | Cost | Status |
|---|---|---|---|
| Live code swap at the step gap (one function) | everything | install 79 ms median, 740 ms max; undo 16 ms (v0, sampler edits) | measured (v0) |
| Swap + decode graph recapture, in place | weights, pools, caches | not measured | upstream already recaptures in place: `CudaGraphRunner.recapture_if_needed` calls `capture()` |
| `reset("state")` = `flush_cache` + reseed | everything but cache contents | flush 2.3 ms (v0) | measured (v0) |
| Weight reload in place | process, pools, graphs | not measured | upstream `update_weights_from_disk/tensor/distributed` |
| Zygote fork of a new server | imported modules | not measured | `livelab/services/zygote.py` |
| Full restart, warm caches | disk caches | 79.0 s (Qwen3-8B, v0) | measured (v0) |
| Crash recovery (detect + restart) | disk caches | 6.3 s detect + 59.1 s recover (v0) | measured (v0) |

## 5. How the live lab maps onto this

- **Step gap.** The lab's control message enters through the same `recv_requests` call that
  carries requests. TP rank 0 drains its sockets, then broadcasts the whole list over
  `tp_cpu_group` (`managers/scheduler.py: Scheduler.recv_requests`), and every rank dispatches it
  in the same loop iteration, before `get_next_batch_to_run`. So all TP ranks handle it between
  the same two batches, in lockstep. An `RpcReqInput` runs a scheduler method by name on every
  rank and then calls `torch.distributed.barrier()` on the default group before rank 0 replies
  (`Scheduler.handle_rpc_request`). This holds for PP = 1: with PP > 1 later stages receive by
  point-to-point from the stage before, one iteration later.
- **Overlap scheduler.** The default loop is `event_loop_overlap` (`--disable-overlap-schedule`
  selects `event_loop_normal`; `dispatch_event_loop`). It launches batch k before processing the
  result of batch k-1, so at the top of an iteration one batch is still in flight in
  `result_queue`. Before a commit the lab drains it through the non-in-place pause path:
  `pause_generation` with mode `retract` processes the queued result, merges `last_batch`, clears
  `last_batch` and `cur_batch`, and retracts running requests to the waiting queue
  (`Scheduler.pause_generation`). Mode `in_place` leaves all of that untouched and is not a
  quiescent point. `continue_generation` resumes.
- **GRAPH edits.** Code in `models/` and `layers/` runs inside captured decode graphs. After the
  swap the lab recaptures on every rank (`CudaGraphRunner.capture`, or a new runner via
  `ModelRunner.init_device_graphs`); capture runs the forward, whose TP collectives need all
  ranks together. Sampling runs after replay, outside the graph (`tp_worker.py:
  forward_batch_generation` -> `ModelRunner.sample`), so a sampler edit needs no recapture.
- **`reset("state")`** = `flush_cache` (tree cache, `req_to_token_pool` including the Mamba
  pool's free slots and mapping, KV allocator, grammar, draft pool, then `empty_cache`) plus
  `set_random_seed(random_seed)` on every rank.

Lab pieces (`livelab/`): `adapter/lab.py` (host, `StepGapCoordinator`), `adapter/agent.py`
(scheduler-side participant), `adapter/recapture.py` (graph recapture), `services/zygote.py`.

## 6. Failure risks

| Risk | What happens | Lab rule | Source |
|---|---|---|---|
| Overlap not quiescent | A swap at the loop top runs new code on the result of a batch launched with old code | drain with `pause_generation(retract)` before commit | `Scheduler.event_loop_overlap`, `pause_generation` |
| Paused flush | `flush_cache` refuses while requests sit in `running_batch` (in-place pause); on a hybrid model a refused session turn keeps its Mamba/GDN slot, later runs on it unreset, and a paused flush ends the scheduler | reset only after a retract drain, and check `flush_cache` returned true | `PauseGenerationReqInput` docstring; issue #50139 (wrong output 2/3 runs, scheduler exit 3/3, 1x GB200, Qwen3.5-4B) |
| Mamba pool reset is bookkeeping only | `MambaPool.clear` resets free slots; state tensors are not zeroed | a reset proves nothing for state-reading code unless every slot is rewritten before use | `mem_cache/memory_pool.py: MambaPool.clear` |
| DP > 1 memory release | a refused release on a busy rank still releases memory on idle ranks and returns HTTP 400 | lab does not reach DP > 1 | audit: TLC trace (487 runs, DP = 2), no repro, no upstream number |
| Release asserts idle; resume without release | `release_memory_occupation` asserts `is_fully_idle`; `resume_memory_occupation` without a release raises `KeyError` in `offload_tags.remove` and ends the scheduler | lab never calls them | `scheduler_update_weights_mixin.py`; audit: upstream PR #24123 |
| Weight update vs pause | weight update checks the pause state before taking the writer lock; a concurrent pause deadlocks | do not overlap a lab pause with a weight update | audit: TLC liveness trace, no repro |
| PD disaggregation room reuse | bootstrap rooms reopen without a generation tag: cross-request KV overwrite, freed pages still written by RDMA | lab does not run PD disaggregation | upstream PRs #48889, #48960, #48979, #48982, #48883 (GB200 repros) |
| DP-attention abort race | an abort that reaches its rank before its request is dropped; the request runs on (19/200 lost) | lab does not reach DP attention | upstream PR #50147 |
| DP-attention prefill delayer | DP ranks defer prefill locally and enter different collectives; server aborts | same | upstream PR #50146 |
| Mamba-1 deterministic mode | `dt_proj` picks its matmul kernel by `is_contiguous()`, so logprobs differ alone vs batched (greedy tokens equal) | gate on token ids, not logprobs | upstream PR #50148 |
| Batched greedy not repeatable | the same batched greedy workload gave 6 distinct outputs in 60 runs with no agent | workload runs one request at a time (`kb/workload.md`) | v0 control-repeatability receipt |

Not built or not reached: DP > 1 and DP attention (control messages go per DP replica via the
`DataParallelController`, with no cross-replica gap), PP > 1, PD disaggregation, piecewise
graphs and `--enable-torch-compile` recapture, the TokenizerManager and DetokenizerManager
processes (no participant), and in-place rebuild of objects built at start.
