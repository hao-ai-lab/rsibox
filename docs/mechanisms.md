# Restart Boundaries, Interface & Tests

## Mechanisms by restart boundary

When an edit cannot be swapped in place, `rsibox` refuses before mutating state and names the cheapest restart boundary that can run it:

| Boundary | Mechanism | Edits handled | Implementation |
|---|---|---|---|
| **In place, Python** | Swap function code on every rank with journaled undo | Function & method bodies | [`core/livelab/kernel/swap.py`](../core/livelab/kernel/swap.py) |
| **In place, Triton** | Rebuild `@triton.jit` kernels & callers; restore binaries on undo | Triton kernel bodies | [`core/livelab/kernel/jit.py`](../core/livelab/kernel/jit.py) |
| **In place, CUDA graphs** | Selective `R1` recapture of affected graphs in live workers | Edits inside graph-captured paths | [`repos/vllm/livelab/adapter/recapture.py`](../repos/vllm/livelab/adapter/recapture.py) |
| **In place, C++/CUDA** | Compile edited `.cu` side-library, rebind `torch.ops._C` + `R1` recapture | Single-unit C++/CUDA kernels | [`repos/vllm/livelab/adapter/native.py`](../repos/vllm/livelab/adapter/native.py) |
| **In process, rebuild** | Re-import edited module and rebuild live objects | Constructors, new fields, module constants | [`repos/fastvideo/livelab/services/rebuild.py`](../repos/fastvideo/livelab/services/rebuild.py) |
| **Warm respawn (fork server)** | Fork from pre-imported server (`zygote`); weights served over CUDA IPC | Launch flags, startup code, refused edits | [`core/livelab/forksafe.py`](../core/livelab/forksafe.py), `repos/*/livelab/services/{zygote,weights,holder}.py` |
| **In-memory reshard** | Reshard held weights between TP and EP over CUDA IPC | Expert-parallelism layout switches | [`repos/vllm/livelab/services/reshard_live.py`](../repos/vllm/livelab/services/reshard_live.py) |
| **Dynamo GMS (weight-resident respawn & Shadow promotion)** | Keep weights resident in CUDA VMM (`gpu_memory_service`); zero-disk RO respawn or Scratch-KV Shadow promotion on crash/poison | Poisoned ranks, sticky CUDA faults, process crashes, startup/flag restarts | [`core/livelab/rsibox/warm/dynamo_gms.py`](../core/livelab/rsibox/warm/dynamo_gms.py), [`repos/vllm/livelab/services/snapshot_replay.py`](../repos/vllm/livelab/services/snapshot_replay.py) |
| **Dynamo Snapshot (CRIU + `cuda-checkpoint` + GMS)** | Freeze quiesced workers with `cuda-checkpoint` + CRIU while weights remain in GMS VMM; restore CUDA graphs and re-attach live-edit channels | Clean process-boundary restore without weight reload or graph recapture | [`core/livelab/rsibox/warm/snapshot_contract.py`](../core/livelab/rsibox/warm/snapshot_contract.py), [`repos/vllm/livelab/services/snapshot_replay.py`](../repos/vllm/livelab/services/snapshot_replay.py) |

See [`docs/kernels-and-graphs.md`](kernels-and-graphs.md) for the deeper kernel and CUDA graph design.

## Composing RSIBox with NVIDIA Dynamo (GMS + Snapshot)

RSIBox and NVIDIA Dynamo (GPU Memory Service + Snapshot) address orthogonal boundaries and compose into a single latency ladder:

- **In-process edits (RSIBox):** Runtime config changes (`in_process_live`), Python operator swaps (`R0`), and selective CUDA-graph recapture (`R1`) run inside the live engine in milliseconds to a few seconds without restarting workers.
- **Process-boundary recovery (Dynamo GMS + Snapshot):** When an edit crashes a worker, leaves a sticky CUDA fault, or crosses a process boundary, RSIBox promotes a parked Dynamo GMS Scratch-KV shadow (`0.31–1.86 s`) or restores a quiesced `Dynamo GMS + Snapshot` CRIU checkpoint (`3.62–6.96 s`). Once recovered, RSIBox re-attaches its kernel channels so subsequent `R0` live edits immediately resume at `~76–92 ms`.

### Verified NVIDIA B200 Multi-Model Benchmarks

Measured on NVIDIA B200 GPUs (183,359 MiB HBM3e per GPU; driver `595.71.05`, vLLM `0.33.0`, NVIDIA Dynamo GMS `0.9.0`, CRIU `4.2.1`, `cuda-checkpoint` `595.71.05`) across 8 vLLM model configurations (6 TP1 models from `1.5 GB` to `65.5 GB` and 2 TP4 models at `151.5 GiB` and `491.1 GiB`, all with `FULL_AND_PIECEWISE` CUDA graphs, `torch.compile`, and FlashInfer autotune enabled) plus SGLang (`gemma-3-1b-it`, TP1, CUDA graphs disabled). Values are `mean ± std` where `n >= 2` (`n=3` for live edits, Shadow promotion, and CRIU restores; `n=2–3` for Warm-JIT and GMS RO restarts; `n=1` for Cold-JIT and `gemma-4-26B-A4B-it` Warm-JIT/GMS RO).

| Model (Tier / TP / CUDA Graphs) | Cold-JIT Disk | Warm-JIT Disk | GMS RO Restart | Plain CRIU Restore | GMS + CRIU Restore | GMS Shadow Fallback | RSIBox R1 Recapture | RSIBox R0 Live Edit | Tier-1 Live Reconfig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` (1.5 GB, TP1, 166) | 155.60 s | 35.07 ± 0.35 s | 33.83 ± 0.71 s | 4.12 ± 0.13 s | 3.62 ± 0.15 s | 0.316 ± 0.009 s | 0.99 ± 0.05 s | 84.7 ± 0.3 ms | 3.70 ± 2.39 ms |
| `google/gemma-3-1b-it` (1.9 GB, TP1, 166) | 258.22 s | 45.12 ± 5.69 s | 37.96 ± 0.18 s | 3.93 ± 0.03 s | 3.70 ± 0.10 s | 0.312 ± 0.010 s | 0.62 ± 0.07 s | 79.4 ± 5.2 ms | 5.05 ± 2.47 ms |
| `Qwen/Qwen3.8-27B-FP8` (29 GB, TP1, 166) | 258.15 s | 84.97 ± 8.87 s | 73.17 ± 1.45 s | 15.23 ± 0.24 s | 6.96 ± 0.08 s | 1.034 ± 0.073 s | 0.98 ± 0.03 s | 82.7 ± 8.0 ms | 4.38 ± 0.02 ms |
| `google/gemma-4-26B-A4B-it` (49 GB MoE, TP1, 166) | 757.40 s | 87.95 s | 72.42 s | 16.22 ± 0.82 s | 6.92 ± 0.23 s | 0.807 ± 0.105 s | 10.52 ± 0.45 s (all 166) | 77.6 ± 5.8 ms | 4.40 ± 2.07 ms |
| `Qwen/Qwen3.8-27B` BF16 (52 GB, TP1, 86) | 219.03 s | 74.29 ± 0.17 s | 63.77 ± 1.56 s | 20.20 ± 0.52 s | 5.96 ± 0.08 s | 0.975 ± 0.015 s | 1.01 ± 0.17 s | 80.9 ± 3.0 ms | 4.36 ± 2.09 ms |
| `Qwen3-32.8B-Dummy`\* (65.5 GB, TP1, 86) | 183.61 s | 49.76 ± 1.38 s | 38.42 ± 1.05 s | 19.65 ± 0.68 s | 5.16 ± 0.40 s | 0.817 ± 0.094 s | 0.91 ± 0.14 s | 76.0 ± 2.8 ms | 5.15 ± 2.33 ms |
| `Qwen/Qwen3-Next-80B-A3B` (151.5 GiB, TP4, 30) | 779.09 s | 91.72 ± 10.42 s | 60.89 ± 1.31 s | n/a (driver <615) | n/a (driver <615) | 1.861 ± 0.146 s | -- | 90.6 ± 4.9 ms | -- |
| `nvidia/DeepSeek-V4.1-Flash-NVFP4` (491.1 GiB, TP4, 30) | 1,840.35 s | 296.39 ± 91.28 s | blocked (GMS 0.9.0†) | n/a (driver <615) | n/a (driver <615) | blocked (GMS 0.9.0†) | 3.69 ± 0.68 s | 92.0 ± 6.0 ms | 5.12 ± 2.35 ms |

SGLang (`google/gemma-3-1b-it`, TP1, decode and prefill CUDA graphs disabled): Cold-JIT disk `37.99 s`, Warm-JIT disk `15.82 ± 1.68 s`, GMS RO restart `15.33 ± 0.33 s` (`0.308 s` GMS import), quiesce `0.844 ± 0.023 s`, resume `0.300 ± 0.021 s`, post-restore RSIBox `R0` live edit `68.0 ± 0.4 ms` (uninstall `9.8 ± 0.2 ms`, 100% proven, bit-exact rollback).

### Snapshot Volume & Quiesce Scissors (`Plain CRIU` vs `Dynamo GMS + CRIU`)

Keeping weights resident in Dynamo GMS VMM removes weight serialization from both `quiesce()` and `criu dump`, so checkpoint volume and quiesce latency diverge sharply as model size grows:

| Model Tier (TP1) | Plain CRIU Ckpt | GMS + CRIU Ckpt | Volume Reduction | Plain Quiesce | GMS Quiesce | Quiesce Speedup | Restore Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` (1.5 GB) | 8.85 GiB | 6.45 GiB | 1.37× (-27.1%) | 1.290 s | 0.064 s | 20.0× | 1.14× |
| `google/gemma-3-1b-it` (1.9 GB) | 10.28 GiB | 7.51 GiB | 1.37× (-27.0%) | 1.509 s | 0.079 s | 19.2× | 1.06× |
| `Qwen/Qwen3.8-27B-FP8` (29 GB) | 56.78 GiB | 12.49 GiB | 4.54× (-78.0%) | 15.396 s | 0.101 s | 153.1× | 2.19× |
| `google/gemma-4-26B-A4B-it` (49 GB) | 64.95 GiB | 13.31 GiB | 4.88× (-79.5%) | 19.664 s | 0.098 s | 201.6× | 2.34× |
| `Qwen/Qwen3.8-27B` BF16 (52 GB) | 88.80 GiB | 10.62 GiB | 8.36× (-88.0%) | 26.866 s | 0.193 s | 139.3× | 3.39× |
| `Qwen3-32.8B-Dummy`\* (65.5 GB) | 76.24 GiB | 7.88 GiB | 9.67× (-89.7%) | 22.696 s | 0.070 s | 323.3× | 3.81× |

### Scope and Empirical Notes

- **Correctness:** All runs recording `logprobs=5` (`Qwen3-0.6B`, `Qwen3.8-27B-FP8`, `Qwen3.8-27B` BF16, `Qwen3-32.8B-Dummy`, `Qwen3-Next-80B-A3B`, and `DeepSeek-V4.1-Flash-NVFP4`) verified `max_abs_logprob_diff == 0.0` across restarts, restores, Shadow promotions, and live-edit rollbacks; `gemma-3-1b-it`, `gemma-4-26B-A4B-it`, and SGLang verified 100% bit-exact output digests. All GMS runs recorded `0` handshake recv errors.
- **GMS `0.9.0` RW seed pattern:** Under vLLM `0.33.0` with `FULL_AND_PIECEWISE` CUDA graphs, first-writer GMS RW mode hits `CUDA_ERROR_ILLEGAL_ADDRESS` during FULL graph capture after deferred `publish_pending_gms_write()`. Seeding weights once via an eager RW pre-loader (`--enforce-eager`) and attaching serving workers in GMS RO mode (`--gms-read-only`) resolves this cleanly with full CUDA graphs enabled.
- **Determinism and `R1` recapture scope:** On `gemma-4-26B-A4B-it`, default FlashInfer CUTLASS unquantized MoE was non-deterministic across greedy calls (`VLLM_BATCH_INVARIANT=0`); switching to `--moe-backend triton` restored bit-exact determinism while retaining all 166 CUDA graphs. `R1` on `gemma-4-26B-A4B-it` recaptured all 166 graphs (`10.52 s`), whereas the other 6 `R1`-measured models selectively recaptured sizes `[1, 2, 4]` (`0.62–3.69 s`).
- **\*Dummy weights (`65.5 GB` tier):** `Qwen3-32.8B-Dummy` uses deterministic dummy BF16 weights (`Qwen3ForCausalLM`, 32.76B params, `65.52 GB` tensor bytes, 17 safetensors shards) matching the `65.3 GB` single-GPU tier; memory footprint, disk I/O, CUDA graph capture, GMS VMM registration, and CRIU checkpoint size match real weights of the same geometry, while generated tokens are synthetic (deterministic with `max_abs_logprob_diff == 0.0`).
- **†Upstream GMS `0.9.0` and multi-GPU NCCL CRIU limits:** On `nvidia/DeepSeek-V4.1-Flash-NVFP4` (TP4), 4-GPU GMS RW seeding committed `488.61 GiB` (`163.99 s`, with `--engram-config '{"cpu_offload":false}'`), but GMS RO attachment failed inside upstream `gpu_memory_service 0.9.0` `_create_meta_model` due to a shape mismatch on post-processed `wo_a.weight` (`(2048, 4096)` vs `(2, 1024, 4096)`). Pure BF16 `Qwen/Qwen3-Next-80B-A3B-Instruct` (`151.5 GiB`, TP4) verified 4-GPU GMS RO restart (`60.89 s`, `7.83 s` weight import) and 4-GPU Shadow promotion (`1.861 s`) on the same 4× NVIDIA B200 GPUs. Multi-GPU NCCL CRIU checkpoint/restore requires NVIDIA driver `615+` with the `cuInterpose` shim (unavailable on host driver `595.71.05`).
- **SGLang CUDA graphs:** SGLang Matrix C ran with `cuda_graph_backend_decode=disabled` and `cuda_graph_backend_prefill=disabled`; its absolute boot and resume times are not directly comparable to CUDA-graph-enabled vLLM tiers.

## Sandbox interface & verification suite

Every engine adapter implements [`core/livelab/INTERFACE.md`](../core/livelab/INTERFACE.md): `install(patch)`, `check(workload)`, `reset()`, `uninstall()`, and `status()` (`base`, `edited`, or `poisoned`).

The sandbox verification suite ([`core/livelab/conformance/README.md`](../core/livelab/conformance/README.md)) verifies:

- **c0–c2:** Repeatable workload; no-op edit stays bit-exact; visible edit changes output and triggers execution checks (`witness`) on every rank.
- **c3:** Live-edited output (and a chain of 3 edits) matches a fresh process started with the edit.
- **c4, c7, c9:** Undo restores output and code identity to base, step-by-step through chains, and across a 20-cycle repeated install/undo stress test.
- **c5–c6:** Mid-install faults roll back all ranks; unsupported or silent no-op edits are refused without mutating state.
- **c8:** Crashed, hung, or externally tampered workers mark the sandbox faulted (`poisoned`) until respawned at base.
- **c10:** Startup-only edits whose new code never ran are reported not proven (`zero_witness`).

**Limits:** A live check proves the edited path ran and matched on the chosen workload, not equivalence on all inputs. Engines need a Python control layer reachable between steps, and one agent operates one warm process at a time.

## Full CPU test suite

What CPU CI runs (`.github/workflows/cpu-tests.yml`); torch, vLLM, and Linux-only cases skip automatically on a laptop:

```sh
PYTHONPATH=core/livelab python3 -m rsibox doctor
python3 -m unittest discover -s core/livelab/kernel -t core/livelab
python3 -m unittest discover -s core/livelab/rsibox -t core/livelab
python3 -m unittest discover -s core/livelab -p 'test_forksafe.py'
python3 -m unittest discover -s core/livelab/survey -t core/livelab
(cd core/livelab && python3 -m unittest discover -s conformance -t .)
python3 -m unittest discover -s template/livelab/adapter -t template/livelab
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/verify_facts.py --strict
python3 .github/modal-ci/test_modal_ci.py -v
```
