# Sources for `priors.md`

All read directly; none run yet. Paths are relative to the Megatron-LM root.
Source tree: megatron-core 0.20.0 dev, commit `ae76c60e6` plus local modifications.
Comparison tree: megatron-core 0.13.0rc0.

| Topic | File : function |
|---|---|
| Entry and arg parse | `pretrain_gpt.py:__main__`, `megatron/training/global_vars.py:initialize_runtime_services` |
| Startup order, first-step hooks, CUDA graph capture | `megatron/training/training.py:pretrain`, `setup_model_and_optimizer`, `get_model`, `train_step`, `train` |
| Distributed init, RNG seeding, JIT warmup | `megatron/training/initialize.py:initialize_megatron`, `_initialize_distributed`, `_set_random_seed`, `set_jit_fusion_options`, `_compile_dependencies`, `_initialize_tp_communicators` |
| Kernel warmup | `megatron/training/kernel_warmup.py:warmup_training_kernels` |
| Process groups | `megatron/core/parallel_state.py:initialize_model_parallel`, `create_group` |
| In-process restart NCCL force | `megatron/training/inprocess_restart.py:maybe_force_nccl_backend_init` |
| `jit_fuser` | `megatron/core/jit.py` |
| Param/grad buffers, `param.data` remap, `main_grad` | `megatron/core/distributed/param_and_grad_buffer.py:_ParamAndGradBuffer` |
| Optimizer class choice, `init_state_fn` | `megatron/core/optimizer/__init__.py` (top-level imports, `get_megatron_optimizer`) |
| DistributedOptimizer groups, state load/save, step | `megatron/core/optimizer/distrib_optimizer.py:_build_model_and_main_param_groups`, `state_dict`, `load_state_dict`, `_init_optimizer_states_with_dummy_values`, `_copy_model_params_to_main_params`, `step_with_ready_grads` |
| Other optimizer wrappers | `megatron/core/optimizer/optimizer.py:Float16OptimizerWithFloat16Params`, `FP32Optimizer` |
| TE Adam lazy state | `transformer_engine/pytorch/optimizers/fused_adam.py:FusedAdam.step`, `initialize_state` (TE 2.18.0) |
| LR schedule | `megatron/core/optimizer_param_scheduler.py:OptimizerParamScheduler` |
| RNG tracker | `megatron/core/tensor_parallel/random.py:model_parallel_cuda_manual_seed`, `CudaRNGStatesTracker`, `get_all_rng_states` |
| Checkpoint contents and load | `megatron/training/checkpointing.py:get_rng_state`, `load_checkpoint` |
| Data samplers and iterators | `megatron/training/datasets/data_samplers.py`, `megatron/training/training.py:build_train_valid_test_data_iterators` |
| Dataset index cache | `megatron/core/datasets/gpt_dataset.py:_build_document_sample_shuffle_indices` |
| Rerun state machine | `megatron/core/rerun_state_machine.py:RerunStateMachine`, `RerunDataIterator`; default mode in `megatron/training/config/resilience_config.py` |
| CUDA graph impls | `megatron/core/transformer/transformer_config.py:cuda_graph_impl`, `megatron/core/transformer/cuda_graphs.py:CudaGraphManager`, `TECudaGraphHelper` |
| Determinism | `megatron/training/determinism.py`, `megatron/training/arguments.py` (deterministic-mode hook) |
| Lab vocabulary | `core/livelab_overview.md`, `template/livelab/INTERFACE-v1.md`, `repos/vllm/kb/mechanisms.md` |

No external documents or published numbers are cited.
