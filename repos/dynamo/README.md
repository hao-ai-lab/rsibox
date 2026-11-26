# NVIDIA Dynamo Live-Lab Adapter (`repos/dynamo/`)

RSIBox adapter for **NVIDIA Dynamo** composing an isolated HTTP frontend proxy (`rust_http_proxy` compiled via `rustc`, with Python `ThreadingHTTPServer` fallback), an in-process Python planner (`adapter.planner_ops`), and **GPU Memory Service (`gpu_memory_service`)** weight-resident GPU workers (`adapter.worker`).

- **Live swap**: `adapter.planner_ops` and `adapter.worker` functions swap in ~20 ms on 2x B200 (`0.160 s` swap + check vs `14.643 s` cold boot, `91.4x`), verified bit-identical against isolated `PackageOverlay` fresh subprocesses across two request shapes (`kv_aware` bs=1, `round_robin` bs=2).
- **Frontend micro-restart**: `DynamoLab.restart_frontend()` recycles only the frontend HTTP process (`77.7 ms` for `rust_http_proxy`) while GMS-committed weights (`2,059,403,264` B per B200, `97` allocations/GPU) and worker PIDs remain alive with `0` weight reloads.
- **CPU & GPU conformance**: `11/11` `rsibox conformance dynamo` on CPU (`ToyDynamoLab`) and `11/11` `rsibox conformance dynamo --real` on 2x NVIDIA B200 (`CUDA_VISIBLE_DEVICES=0,1`, `gemma-3-1b-it`, `0 MiB` exit); `python3 -m rsibox doctor --engine dynamo` checks adapter and `gpu_memory_service` availability.

See [kb/costs.md](kb/costs.md) and [kb/workload.md](kb/workload.md).
