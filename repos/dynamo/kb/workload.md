# NVIDIA Dynamo Workload & Not-Covered Boundaries

## Verified Workload Shapes
1. **Shape A (`kv_aware_bs1_max8`)**: `prompts=["The capital of France is"]`, `max_new_tokens=8`, `policy="kv_aware"`, `1` call/check to `adapter.planner_ops:shape_decode_params`, deterministic greedy decode via `model.generate(..., do_sample=False)`.
2. **Shape B (`round_robin_bs2_max6`)**: `prompts=["Water boils at", "2 + 2 equals"]`, `max_new_tokens=6`, `policy="round_robin"`, `2` calls/check standalone across `[worker0, worker1]` (`3` cumulative when run immediately after Shape A before `uninstall`) (dispatched onto GPU 0 and GPU 1 over `AF_UNIX` JSON RPC).

## Known Scope & Constraints
- Constructors (`PlannerConfig.__init__`), module-level routing constants, and closures capturing non-module state are refused at prepare time (`c6_refusal`, `test_refuses_constructor_edit_and_rejecting_weight_store`).
- Weight sharing across sibling workers uses `gpu_memory_service` (`RW_OR_RO` for primary workers, `RO` across `fresh_check` subprocesses, each GPU device isolated in its own process because GMS client `_tag_states` binds one device per process for `tag="weights"`).
- Real-GPU tests exercise `rust_http_proxy` + Python planner + `transformers` GMS workers rather than upstream full `ai-dynamo` Rust NATS/etcd runtime; CUDA graph recapture and upstream compiled Rust router extensions require a restart.
