# NVIDIA Dynamo Measured Costs & Receipts

## Hardware & Environment
- **GPU**: 2x NVIDIA B200 (`CUDA_VISIBLE_DEVICES=0,1`, driver `615.71.09`), Python `3.12.11`, PyTorch `2.14.0+cu130`, Transformers `5.20.2`, `gpu_memory_service 0.9.0`, `gemma-3-1b-it` (`bfloat16`, `2,059,403,264` bytes committed per GPU across `97` GMS allocations).
- **Frontend binary tested**: zero-dependency `rust_http_proxy` binary (`dynamo_frontend_rs`) compiled via `rustc 1.92.0` (`-O`) fronting Python `PlannerConfig` + `adapter.planner_ops` + 2 GMS worker processes (upstream compiled Rust `ai-dynamo` runtime container was not required and Python fallback `ThreadingHTTPServer` is tested in CPU mode).

## Real 2x NVIDIA B200 Measurements (`num_workers=2`, `-W error::ResourceWarning`)
- **Cold boot + 1st check**: `14.643 s` (`13.602 s` GMS daemon + weight commit + Rust frontend startup; `1.041 s` 1st check).
- **Warm live swap + check (shape A `kv_aware` bs=1)**: `0.160 s` total (`19.7 ms` install + `140.4 ms` check = **91.4x** vs cold), `12.9 ms` bit-exact undo; shape B (`round_robin` bs=2) check `213.4 ms`; `witness_nonzero=true`, `replay_verified=true` (`fresh_check` RO GMS attach bit-identical on both shapes).
- **Frontend micro-restart & remapped GMS**: `77.7 ms` (`old_pid != new_pid`), `worker_pids_unchanged=true`, `weight_reload_count=0`, both GMS sockets remain `alive=true` and `committed=true`, post-restart outputs bit-exact on both shapes; `CUDA_VISIBLE_DEVICES=2,3` verified physical GPU UUID socket routing (`RSIBOX_GMS_DEVICE_UUID_MAP`) and `fresh_check` parity.
- **`rsibox conformance dynamo --real`**: `11/11` passed (`c0` `5.336 s`, `c1` `5.432 s`, `c2` `5.164 s`, `c3` `33.770 s`, `c4` `5.815 s`, `c5` `5.852 s`, `c6` `6.046 s`, `c7` `6.688 s`, `c8` `138.422 s` [respawns `15.37–16.95 s`], `c9` `16.957 s` [median `16.3 ms` install / `7.4 ms` undo], `c10` `14.586 s`), exit VRAM `0 MiB` on GPUs 0–3.

## CPU Verification (`ToyDynamoLab`)
- **`rsibox conformance dynamo`**: `11/11` passed (`c9` `20` cycles: `5.4 ms` install / `1.8 ms` undo median), `-W error::ResourceWarning`.
- **Unit & golden tests (`test_dynamo_lab.py`)**: `9/9` passed (`37.447 s`), including shutdown-error worker reaping, `FrontendDriver.start()` failure cleanup & `mode="rust"` fail-closed without `rustc`, `RSIBOX_GMS_DEVICE_UUID_MAP` population/fail-closed guard, `num_workers=3` `c5`/`c8`, constructor/store refusals, and zero eager `torch`/`gpu_memory_service` imports.
