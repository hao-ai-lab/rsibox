# Megatron-LM: what is known before we measure

Draft. Read from the source, not run. No timings here: ours go in `costs.md`.
Source tree: megatron-core 0.20.0 dev, commit `ae76c60e6` plus local modifications.
Env: torch 2.13 (cu130), Transformer Engine 2.18.0, NCCL 2.30.7, Apex not installed. 4× GB200, aarch64.
Paths are under `megatron/`. "(unverified)" marks what the source did not confirm.
An older tree (core 0.13.0rc0) has the same optimizer fallback order but no `kernel_warmup.py`
and no `training/determinism.py`: do not assume either exists on other versions.

## 1. Startup phases

Call chain: `pretrain_gpt.py:__main__` parses args (`parse_and_validate_args`), builds the run config,
calls `initialize_runtime_services` (tokenizer, loggers, `disable_jit_fuser` if asked), then
`training/training.py:pretrain` → `initialize_megatron` → `set_jit_fusion_options` →
`warmup_training_kernels` → `setup_model_and_optimizer` → `build_train_valid_test_data_iterators` → `train`.

| # | Phase | Entry point | What it builds | Expensive part | Keep warm across an edit? |
|---|---|---|---|---|---|
| 1 | Imports | top of `pretrain_gpt.py` | torch, TE, megatron modules; `jit_fuser` decorators run at import | import time, TE shared libs (unverified) | Yes: same process |
| 2 | Arg parse | `training/arguments.py:parse_and_validate_args` | `args` namespace, run config | cheap (unverified) | No: restart (see §2) |
| 3 | Distributed init | `training/initialize.py:_initialize_distributed` | `init_process_group` (NCCL), then `core/parallel_state.py:initialize_model_parallel` makes TP/PP/CP/EP/DP groups, plus gloo groups (`use_gloo_process_groups` defaults to True) | group creation, rendezvous | Yes: keep process and groups |
| 4 | RNG init | `initialize.py:_set_random_seed` → `tensor_parallel/random.py:model_parallel_cuda_manual_seed` | python/numpy/torch seeds, CUDA RNG tracker states | cheap | Yes, but re-seeding changes init |
| 5 | Dataset helper build | `initialize.py:_compile_dependencies` (rank 0, `training=True`) | `core/datasets/helpers.py` C++ index helpers (a prebuilt `.so` is in the tree) | `compile_helpers()` if the `.so` is missing | Yes: cached `.so` |
| 6 | TP comm overlap | `initialize.py:_initialize_tp_communicators` (only `--tp-comm-overlap`) | TE userbuffers | allocation, bootstrap | Only with the process |
| 7 | JIT fusion options | `initialize.py:set_jit_fusion_options` → `_warmup_jit_function` | On torch ≥ 2.2: only the dynamo cache limit (env `DYNAMO_CACHE_SIZE_LIMIT`) and a warmup of the fused bias/swiglu functions. nvFuser flags are set only on old torch. | compile of the warmed fusions | Yes: compile cache |
| 8 | Kernel warmup | `training/kernel_warmup.py:warmup_training_kernels` | Runs fused cross-entropy (native impl only), bias-swiglu/gelu, bias-dropout-add at the configured static shapes, inside `torch.random.fork_rng`. Then shuts down inductor compile workers, `gc.collect`, `empty_cache` | compile/autotune of those kernels | Yes: compile caches on disk (unverified); in-memory compiled code stays with the process |
| 9 | Model build | `training.py:setup_model_and_optimizer` → builder `build_distributed_models` (config path, `training/models/gpt.py`) or `get_model` (provider path) | `GPTModel` from `TransformerConfig` + layer spec; `.cuda()`; `Float16Module`; `DistributedDataParallel` → `_ParamAndGradBuffer`s | weight init, buffer allocation, `broadcast_params` | Process and comm groups yes; model itself rebuilt for constructor edits |
| 10 | Optimizer build | `core/optimizer/__init__.py:get_megatron_optimizer`; scheduler: `training.py:get_optimizer_param_scheduler` | Inner Adam, `DistributedOptimizer` shards and fp32 masters, `OptimizerParamScheduler` | fp32 master copy per shard | Rebuildable from state |
| 11 | Checkpoint load | `training/checkpointing.py:load_checkpoint`, called inside `setup_model_and_optimizer` after the optimizer exists | model, optimizer, scheduler, RNG, `consumed_train_samples`; dist_checkpointing for `torch_dist` | disk read, resharding | A staged copy in memory skips it (R6 analogue) |
| 12 | Data iterators | `training.py:build_train_valid_test_data_iterators` → `training/datasets/data_samplers.py:build_pretraining_data_loader` | `GPTDataset` indexes (document, sample, shuffle) built by rank 0 or loaded from a `.npy` cache (`core/datasets/gpt_dataset.py:_build_document_sample_shuffle_indices`), a `DataLoader`, a `RerunDataIterator` | index build on first use; cached after | Yes: cache on disk; the iterator is cheap to rebuild at a known sample count |
| 13 | `train()` preamble | `training.py:train` | num-microbatches, `FullCudaGraphWrapper` (`--cuda-graph-impl full_iteration`), `OptimizerCudaGraphWrapper` (`--optimizer-cuda-graph`), `TECudaGraphHelper` (`transformer_engine`), weight-hash check | wrapper setup is cheap; capture is later | Graphs: no, they are recaptured |
| 14 | First step(s) | `training.py:train_step` | See below | See below | Mostly lost on process exit |
| 15 | Steady state | `train()` loop | | | |

What the first step pays (all lazy, by design or by library):

- **NCCL communicators.** Not created by `init_process_group` here: no `device_id` is passed, and
  `parallel_state.py` says "PyTorch is performing lazy initialization of the communicator group"
  and forces it only for SHARP and dynamic-CP groups. Each group builds its communicator at its first
  collective. Exception: `--inprocess-restart` forces one all-reduce (`inprocess_restart.py:maybe_force_nccl_backend_init`).
- **Optimizer state.** Created at the first `optimizer.step()`. See §3.
- **TE kernels and library heuristics.** cuBLAS/cuDNN algorithm choice, TE fused-attention backend choice,
  FP8 amax history if FP8 is on (all unverified: TE internals not read).
- **`jit_fuser` code.** `core/jit.py` sets `jit_fuser = torch.compile` for torch ≥ 2.2 (`torch.jit.script`
  before that, a no-op after `disable_jit_fuser`). 98 uses under `core/`. Functions not covered by the
  warmup in phase 7/8 compile on first call. Shape changes recompile.
- **CUDA graphs.** `transformer_engine` impl: `TECudaGraphHelper.create_cudagraphs()` runs at
  `iteration - start_iteration == cuda_graph_warmup_steps` (`training.py:train`). `local` impl:
  `core/transformer/cuda_graphs.py:CudaGraphManager` per layer. `full_iteration`: the whole forward-backward
  except the optimizer step.
- **Rerun state machine.** On by default: `rerun_mode` default is `validate_results`
  (`training/config/resilience_config.py`). It wraps each iteration in `should_run_forward_backward`.

## 2. Restart classes

Mapped to the groups in `core/livelab_overview.md` and recipes R0–R10 in `template/livelab/INTERFACE-v1.md`.
The "where" column says what holds a reference to the code.

| Edit | Group | Recipe | Notes |
|---|---|---|---|
| Body of a per-call function: layer `forward`, `loss_func`, `forward_step`, `get_batch`, a schedule helper | OP | R0 | `train()` binds `forward_backward_func = get_forward_backward_func(...)` once (`training.py:6488`), so rebinding the module attribute does nothing: swap the function's `__code__`. `train_step` is looked up each iteration. Swap between iterations. |
| Same, but inside a `jit_fuser` / `torch.compile` function | GRAPH | R1 | The decorator holds the compiled wrapper. Swapping Python is not enough: reset the compiled function or its dynamo cache (unverified for this torch). Functions warmed in phases 7–8 recompile on next call. |
| Inside a CUDA graph (`full_iteration`, `local`, `transformer_engine` impls) | GRAPH | R1 | Delete and recapture: `TECudaGraphHelper.delete_cuda_graphs()` / `create_cudagraphs()` exist. Capture needs the same buffers and RNG registration (see §3). |
| Module constructor, `TransformerConfig` field, layer spec, new parameter | STATE, RESOURCE | R5 | Rebuild the model through the builder. Keep process and comm groups. DDP buffers and optimizer shards must be rebuilt after it, since both point at the old parameters. Weights come back from a snapshot (R6). |
| Optimizer construction, param groups, hyperparameters baked at build (`get_megatron_optimizer`, `_get_param_groups`) | STATE | R5 | Rebuild the optimizer from checkpointed state. Hyperparameters read each step from `param_group` (lr, wd) can be changed in place by `OptimizerParamScheduler`. |
| `DistributedOptimizer` sharding, bucket size, `ddp_config` | RESOURCE | R5 | Changes the shard layout in the flat buffer: rebuild DDP and optimizer; saved state must be resharded (`dp_reshardable` / `fully_reshardable` formats). |
| Arg parsing, defaults, validation | | R9 | `args` is parsed before `pretrain`; `get_args()` is a global read in many places. Some fields can be mutated live (unverified which are safe). |
| TP / PP / CP / EP / VPP sizes, `order`, process-group creation | RANKS | R9 | Groups come from `initialize_model_parallel`. Changing sizes changes every rank's model, buffers and data shard. Full restart. |
| Env vars (`NCCL_*`, `CUBLAS_WORKSPACE_CONFIG`, `NVTE_*`), `--deterministic-mode` | | R9 | Read at library init or first use. Restart. |
| Python-only dataset code (`GPTDataset`, sampler) | OP, STATE | R0 + rebuild iterator | Rebuild the iterator at the saved `consumed_train_samples`. Index cache keys on config hash (unverified). |
| C++/CUDA: `core/datasets/helpers.cpp`, TE, Apex | NATIVE | R7 | A loaded `.so` cannot be unloaded. Build a versioned side library and rebind; TE kernels also need graph recapture. Whether TE can be rebound is unverified. |

## 3. Where state lives

| State | Where | Snapshot | Restore |
|---|---|---|---|
| bf16/fp16 params | With `--use-distributed-optimizer`: a separate flat `param_data` per `_ParamAndGradBuffer`, and every `param.data` is rebound to a view of it (`core/distributed/param_and_grad_buffer.py:_ParamAndGradBuffer.__init__`). Without it: normal parameter storage, only grads are in the buffer. | Copy `param_data`, or `model.state_dict()` | Copy back in place; `param.data` views stay valid. A new buffer breaks every view and every CUDA graph input. |
| Main grads | `param.main_grad`, a view of `grad_data` in the same buffer | Not needed between steps: `zero_grad_buffer()` runs at the start of each `train_step` | Zeroed each step |
| fp32 master params | `DistributedOptimizer.shard_fp32_from_float16_groups` (clone of the local shard, built in `_build_model_and_main_param_groups`); fp32 params use `shard_fp32_groups`. Model-side lists: `model_float16_groups`, `model_fp32_groups`, `shard_float16_groups`. The inner optimizer's `param_groups` hold the fp32 shards. Without DistOpt: `Float16OptimizerWithFloat16Params.fp32_from_float16_groups`. | `save_parameter_state` / `sharded_state_dict`, or clone the shard lists | `load_parameter_state`, or `_copy_model_params_to_main_params()` (`reload_model_params`) to rebuild masters from bf16 |
| Adam moments (`exp_avg`, `exp_avg_sq`) | Inner optimizer `.state[p]` of the fp32 shards | Same as above | See below |
| Adam step count | TE `FusedAdam`: `param_group["step"]` (a python int, or a tensor if `capturable`), not in `.state`. torch `AdamW`: `state[p]["step"]`. | `DistributedOptimizer.state_dict()` extracts it | Restored by `load_state_dict` |
| LR / WD schedule | `OptimizerParamScheduler.num_steps` (counted in samples; `step(increment=…)` in `train_step`) | `state_dict()` | `load_state_dict`; `checkpointing.py` can reset `num_steps = consumed_train_samples` |
| RNG | python, numpy, torch CPU, torch CUDA, and `tensor_parallel.get_cuda_rng_tracker()` states (named: default/data-parallel, model-parallel, expert-parallel). Seeds: `seed + 100*pp_rank` (+ `10*dp_rank` if `--data-parallel-random-init`); model-parallel tracker `seed + 2718 + tp_rank`. | `checkpointing.py:get_rng_state` (python, numpy, torch, cuda, tracker); `get_cuda_rng_tracker().get_states()` | `set_states`. TE's tracker (`te_rng_tracker`) keeps states inside TE: `get_all_rng_states()`. With CUDA graphs the tracker must be graph-safe (`use_cudagraphable_rng`). |
| Data position | `args.consumed_train_samples`; samplers take `consumed_samples` at construction (`MegatronPretrainingSampler`, `MegatronPretrainingRandomSampler`); `RerunDataIterator` wraps the iterator and keeps `saved_microbatches`. DataLoader worker prefetch holds batches already read. | The integer is enough for `single`; `cyclic` re-derives the epoch from it | Rebuild loader and iterator at the integer. In-flight prefetch is lost (replayed from the integer). |
| Process groups | `core/parallel_state.py` module globals (`_TENSOR_MODEL_PARALLEL_GROUP`, `_DATA_PARALLEL_GROUP`, …) | Not snapshotable | Keep the process, or restart |
| NCCL communicators | Lazy, inside each group's backend | | Rebuilt on first collective after any group re-creation |
| Rerun state machine | `get_rerun_state_machine()`; `state_dict()`; replay buffers in `RerunDataIterator` | `state_dict(data_iterator, ckpt_format)` | `load_state_dict`. Set `current_iteration` after a restore (`train()` does this). |
| TE workspace, userbuffers | TE-internal, process-global | | Lives with the process (unverified) |
| CUDA graphs | Graph objects and their static input buffers (`CudaGraphManager`, `TECudaGraphHelper`, `FullCudaGraphWrapper.cuda_graph`) | | Delete and recapture |
| Timers, args | `get_timers()`, `get_args()` globals | `vars(args)` | Assign back |

### Is Adam state lazy?

Yes, in this build, unless something loads a checkpoint first.

- Optimizer class: `core/optimizer/__init__.py` tries `transformer_engine.pytorch.optimizers.FusedAdam`,
  then `apex.optimizers.FusedAdam`, then `torch.optim.AdamW` with a warning. TE is installed here and
  Apex is not, so **TE `FusedAdam` is used** (flag `USING_PYTORCH_OPTIMIZER = False`). Only a build with neither
  TE nor Apex gets torch `AdamW`.
- State creation: TE `FusedAdam.step` does `if len(state) == 0: self.initialize_state(p, …)` per parameter
  (`transformer_engine/pytorch/optimizers/fused_adam.py`). So `exp_avg` and `exp_avg_sq` appear at step 1.
  The step counter is created in `param_group["step"]` at the same time.
- Megatron's `init_state_fn` (`get_megatron_optimizer`) fills zeros for the same keys. It is called from
  `Float16OptimizerWithFloat16Params.state_dict(is_loading=True)` and `sharded_state_dict(is_loading=True)`,
  `FP32Optimizer`, and the offloader path, not at construction.
- `DistributedOptimizer.load_state_dict` allocates `torch.empty` placeholders when `len(optimizer.state) == 0`;
  `load_parameter_state` then overwrites them. So a checkpoint resume has state before step 1; a cold start does not.
  `_init_optimizer_states_with_dummy_values` (a zero-grad dummy `step()`) exists only for Megatron-FSDP loading.
- Consequence for a live lab: a snapshot taken before step 1 has no moments. Restore must either skip them or
  call `init_state_fn`. A snapshot taken after step 1 must also carry `param_group["step"]`.
  Peak memory grows at step 1 by the two moment tensors per shard (inferred, unverified).

### Where `optimizer.step()` lands

`train_step` → `optimizer.step()` → `prepare_grads` (copy model grads to fp32 main grads, unscale, check finite)
→ clip → `step_with_ready_grads` → inner `step()` → copy main params to model params → param all-gather
(`DistributedOptimizer.step_with_ready_grads`). Then `opt_param_scheduler.step` if the update succeeded.
With `--optimizer-cuda-graph`, `train()` replaces `optimizer.step` by an `OptimizerCudaGraphWrapper`
(`capturable` Adam), so a swap of the step code needs recapture.

## 4. Determinism

| Item | Fact | Source |
|---|---|---|
| `--deterministic-mode` | Requires `cross_entropy_loss_fusion=False`, `tp_comm_overlap=False`, and MoE aux-loss fusion off. Sets, if unset, `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `NCCL_EP_HT_EM_AG_SCAN_MODE=1`. Calls `torch.use_deterministic_algorithms(True)` and then clears `fill_uninitialized_memory` (cost noted in the comment: about 15% TFLOP/s on large configs). Pins Megatron's own Triton autotuners. | `training/determinism.py`, `training/arguments.py` |
| Env must be set before first cuBLAS/TE call | The helper says so; it runs at arg parsing | `determinism.py` docstring |
| Launcher values that break it | `NVTE_ALLOW_NONDETERMINISTIC_ALGO` other than `0`; `CUBLAS_WORKSPACE_CONFIG` other than `:4096:8` or `:16:8`; `NCCL_ALGO` tokens outside the accepted set | `determinism.py` |
| Flash attention | Not rejected: TE FlashAttention is described as deterministic on supported configs | `determinism.py` comment |
| Known nondeterministic | TE fused MoE aux-loss kernel (differs run to run; unfused is bit-identical). Atomic-add paths: NCCL EP count exchange without scan mode. Reduction order in NCCL `Tree`/`NVLS` vs `Ring` (NCCL behavior, unverified here). | `determinism.py` comments |
| Batch-invariant mode | `--batch-invariant-mode` swaps kernels and collectives so results do not depend on batch composition | `initialize.py` (`enable_batch_invariant_mode`) |
| Seed | `seed + 100*pp_rank` (+ `10*dp_rank` only with `--data-parallel-random-init`); `--deterministic-mode` does not change this | `initialize.py:_set_random_seed` |
| Rerun machine | Default `validate_results` re-runs an iteration to compare results; `report_determinism_stats` mode reports | `core/rerun_state_machine.py` |
| Not covered by the source | Dropout with CUDA graphs and a non-graph-safe tracker; DP-size changes (reduction order changes with the DP group); resuming mid-epoch with prefetch (data order is deterministic from `consumed_samples` for `single`) | (unverified) |

For a live-edit lab: judge a patched step by loss against a reference run with the same seed and
`--deterministic-mode`. Without it, expect step-to-step drift that is not a regression (unverified in our setup).

## 5. Facts that matter most for a live lab

1. Adam state is lazy (TE `FusedAdam`, no Apex): a pre-step snapshot has none.
2. NCCL communicators are lazy: a "warm" process has them only after the first collective on each group.
3. `train()` caches `forward_backward_func` and the `optimizer.step` wrapper in locals: swap `__code__`, not module attributes.
4. With DistOpt, `param.data` is a view into `param_data`; a rebuilt buffer invalidates views and graphs.
5. Load happens inside `setup_model_and_optimizer`, after the optimizer exists and before the data iterators.
6. The rerun state machine is on by default and sits around every iteration.
