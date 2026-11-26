# Supported Engines & Benchmarks

Each engine adapter under `repos/<engine>/` implements the shared [`core/livelab/INTERFACE.md`](../core/livelab/INTERFACE.md) contract and is validated against the sandbox verification suite ([`core/livelab/conformance/README.md`](../core/livelab/conformance/README.md)).

## What is in the repo today

| Engine | Verification Suite | Headline Live vs. Restart | Docs & Costs |
|---|---|---|---|
| **vLLM** | CPU CI & 1–2× B200 (eager, V1/V2 CUDA graphs) | 228 s → **2.4–23.6 s** (up to 95×); C++/CUDA `.cu` reload 7.3–19.2 s vs. 41–49 s | [architecture](../repos/vllm/docs/architecture.md) · [costs](../repos/vllm/kb/costs.md) · [walkthrough](../repos/vllm/examples/live-edit-qwen3/README.md) |
| **SGLang** | CPU CI & 1–2× B200 / GB200 (`--fork-subprocs`) | 66.4 s cold / 26.5 s warm → **0.38–1.77 s** live (up to 176×); fork server 10.6–19.5 s | [costs](../repos/sglang/kb/costs.md) |
| **Megatron-LM** | CPU CI & 2–4× GB200 (TP/DP/PP/VPP2) | 57.5 s → **0.50 s** (2L TP2); 266.7 s → **8.0–17.0 s** (30B-A3B EP4); 82-iter hill-climb | [adapter](../repos/megatron/livelab/README.md) · [costs](../repos/megatron/kb/costs.md) · [hill-climb](../repos/megatron/examples/codex-hillclimb/README.md) |
| **FastVideo** | CPU CI & 1–4× GB200 (SP1/SP2/SP4, fork server + weight holder) | 206 s cold / 40.6 s warm → **2.3–6.4 s** live (up to 92×); fork server + holder 14.2 s | [costs](../repos/fastvideo/kb/costs.md) · [h3-verification](../repos/fastvideo/examples/h3-conformance/README.md) · [hill-climb](../repos/fastvideo/patches/h3-hillclimb/README.md) |
| **veRL** | CPU CI & 4× GB200 (21 Ray actors, eager & FSDP CG) | 298 s cold / 180 s warm → **36.4 s** live + S0 reset (8.2×); offline `replay-adv` 12.6 ms | [adapter](../repos/verl/livelab/README.md) · [costs](../repos/verl/kb/costs.md) · [grpo-gsm8k](../repos/verl/examples/grpo-gsm8k/README.md) |
| **miles** | CPU CI & 2× B200 / GB200 | 82.4 s cold / 53.5 s warm → **0.89 s** live + S0 reset (93×); 38 ms median commit | [costs](../repos/miles/kb/costs.md) |
| **Transformers** | CPU CI (`11/11`) & `1×` B200 (`Qwen3-0.6B` `11/11`; `Qwen3-8B` live swap & quiesce/resume); `2×` B200 `device_map='auto'` (`gemma-3-1b-it`) live edit, quiesce/resume & `SIGKILL` verified | `1×`: 7.75 s cold → **0.61 s** live (12.8×), H2D resume **0.31 s** (`15.4 GiB` freed); `2×` `device_map='auto'`: `-1,050 / -892 MiB` quiesce (`0.160 s` resume), `0.153 s` `SIGKILL` fast-fail | [architecture](../repos/transformers/docs/architecture.md) · [costs](../repos/transformers/kb/costs.md) |
| **TRT-LLM** | CPU CI (`ToyTrtllmLab` 11/11) & 1×/2× B200 (`TP=1` & `TP=2` MPI `11/11` conformance + `R1` CUDA-graph model-layer swap + `TrtllmQuiesceHooks`) | `TP=1`: 26.7 s cold / 12.4 s warm → **0.107 s** live (249× / 116×); `TP=2`: **0.275 s** `Qwen3DecoderLayer.forward` `R1` install (`224/224` witness/rank); KV quiesce frees **46.7 GiB** (`15.2 ms` resume) | [architecture](../repos/trtllm/docs/architecture.md) · [costs](../repos/trtllm/kb/costs.md) |
| **TorchTitan** | CPU CI (2-rank toy) & 1×/2× B200 (FSDP2 `fully_shard` `world_size=1` & `world_size=2` NCCL, synthetic `TitanMLPNet`) | `1×`: 2.69 s cold → **0.083 s** live + S0 reset (32.5×); `2×` NCCL: 5.515 s cold → **0.255 s** full cycle (**21.63×**, `max_abs_loss_diff=0.0`) | [architecture](../repos/torchtitan/docs/architecture.md) · [costs](../repos/torchtitan/kb/costs.md) · [walkthrough](../repos/torchtitan/examples/live-step-swap/README.md) |
| **NeMo-RL** | CPU CI (`ToyNemoRlLab` 11/11) & `1×/2×` B200 (`NemoRlLab --real`, `gemma-3-1b-it` AutoModel FSDP2 + SGLang GRPO: post-fix 3-shape parity + `c8` `731.5 s`; pre-fix `c0–c7, c9–c10` pass, follow-up `#325`) | `2×` B200: `186.8 s` cold → **19.0 s** warm (`g2_s2`, **9.8×**), `151.5 s` → **34.7 s** warm (`g4_s3`, **4.4×**); `1×` B200: `161.9 s` → **20.2 s** warm (**8.0×**); `4.6–4.8 s` install (`1×/2×`), `S0` bit-exact | [architecture](../repos/nemo-rl/docs/architecture.md) · [costs](../repos/nemo-rl/kb/costs.md) |
| **Isaac Lab** | CPU CI (`ToyIsaacLab`, 11/11) & 1× B200 (`Isaac-Cartpole-Direct`, `newton_mjwarp`, `4096` envs, `11/11` conformance) | `4096×10`: 9.66 s → **0.397 s** (**24.3×**); `1024×20`: 6.79 s → **0.302 s** (**22.5×**, `max_abs_diff=0.0`) | [readme](../repos/isaac-lab/README.md) · [costs](../repos/isaac-lab/kb/costs.md) |
| **SGLang-Omni** | CPU CI (4-stage `ToySglangOmniLab` 11/11) & 2× B200 (`Qwen3-VL-4B-Instruct` Thinker + CUDA Talker/DiT/Vocoder in `juc049/sglang-omni-qwen3:deps`, standalone `sglang_omni` pkg absent) | 25.20 s cold → **0.0813 s** warm / **0.0874 s** `talker` swap (**0 B** frozen VRAM delta, 2 shapes `max_abs_diff=0.0`) | [README](../repos/sglang-omni/README.md) · [costs](../repos/sglang-omni/kb/costs.md) |
| **Dynamo** | CPU CI (`ToyDynamoLab` 11/11) & 2× B200 (`gpu_memory_service 0.9.0`, `gemma-3-1b-it`, `rust_http_proxy`, `rsibox conformance dynamo --real` 11/11) | 14.64 s cold → **0.160 s** live (**91.4×**, `19.7 ms` install); Rust frontend micro-restart **77.7 ms** (`0` weight reloads) | [readme](../repos/dynamo/README.md) · [costs](../repos/dynamo/kb/costs.md) · [workload](../repos/dynamo/kb/workload.md) |
| **Kernel** | CPU CI (`ToyKernelLab` 11/11) & 1× B200 (`KernelLab`: Triton 3.8.0 `@triton.jit` + CUDA `.cu` `nvcc 13.3` `CDLL`, `rsibox conformance kernel --real` 11/11 & `--backend cuda` 5/5) | Triton: 4.64 s → **0.040 s** (**116.5×**, `0.005 s` undo); `.cu`: 2.96 s → **0.947 s** (`917.4 ms` `nvcc`, `0.004 s` undo check) | [readme](../repos/kernel/README.md) · [costs](../repos/kernel/kb/costs.md) |
| **Toy sandbox** | CPU CI (2-rank stdlib reference) | ~1.5 s for the full sandbox verification suite on a laptop | [verification suite](../core/livelab/conformance/README.md) |

## Repository layout

```
core/livelab/        shared stdlib runtime (kernel/, rsibox/, conformance/, survey/, forksafe.py, INTERFACE.md)
core/agents/         prompts for coding agents (livelab_overview, developer, orchestrator, toolsmith)
repos/<engine>/      vllm, fastvideo, megatron, verl, sglang, miles, transformers, trtllm, torchtitan, nemo-rl, isaac-lab, sglang-omni, dynamo, kernel (docs/, kb/, livelab/, examples/)
docs/                design notes on engines, restart mechanisms, kernels/CUDA graphs, and CI
.github/             CPU workflows, automated review service, leak check, and Modal GPU CI runner
scripts/             fact and consistency checks (verify_facts.py)
template/            scaffold for adding a new engine
```
