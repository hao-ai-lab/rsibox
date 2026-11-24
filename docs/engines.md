# Supported Engines & Benchmarks

Each engine adapter under `repos/<engine>/` implements the shared [`core/livelab/INTERFACE.md`](../core/livelab/INTERFACE.md) contract and is validated against the 11 sandbox verification checks ([`core/livelab/conformance/README.md`](../core/livelab/conformance/README.md)).

## What is in the repo today

| Engine | Verification Suite (11/11) | Headline Live vs. Restart | Docs & Costs |
|---|---|---|---|
| **vLLM** | CPU CI & 1–2× B200 (eager, V1/V2 CUDA graphs) | 228 s → **2.4–23.6 s** (up to 95×); C++/CUDA `.cu` reload 7.3–19.2 s vs. 41–49 s | [architecture](../repos/vllm/docs/architecture.md) · [costs](../repos/vllm/kb/costs.md) · [walkthrough](../repos/vllm/examples/live-edit-qwen3/README.md) |
| **SGLang** | CPU CI & 1–2× B200 / GB200 (`--fork-subprocs`) | 66.4 s cold / 26.5 s warm → **0.38–1.77 s** live (up to 176×); fork server 10.6–19.5 s | [costs](../repos/sglang/kb/costs.md) |
| **Megatron-LM** | CPU CI & 2–4× GB200 (TP/DP/PP/VPP2) | 57.5 s → **0.50 s** (2L TP2); 266.7 s → **8.0–17.0 s** (30B-A3B EP4); 82-iter hill-climb | [adapter](../repos/megatron/livelab/README.md) · [costs](../repos/megatron/kb/costs.md) · [hill-climb](../repos/megatron/examples/codex-hillclimb/README.md) |
| **FastVideo** | CPU CI & 1–4× GB200 (SP1/SP2/SP4, fork server + weight holder) | 206 s cold / 40.6 s warm → **2.3–6.4 s** live (up to 92×); fork server + holder 14.2 s | [costs](../repos/fastvideo/kb/costs.md) · [h3-verification](../repos/fastvideo/examples/h3-conformance/README.md) · [hill-climb](../repos/fastvideo/patches/h3-hillclimb/README.md) |
| **veRL** | CPU CI & 4× GB200 (21 Ray actors, eager & FSDP CG) | 298 s cold / 180 s warm → **36.4 s** live + S0 reset (8.2×); offline `replay-adv` 12.6 ms | [adapter](../repos/verl/livelab/README.md) · [costs](../repos/verl/kb/costs.md) · [grpo-gsm8k](../repos/verl/examples/grpo-gsm8k/README.md) |
| **miles** | CPU CI & 2× B200 / GB200 | 82.4 s cold / 53.5 s warm → **0.89 s** live + S0 reset (93×); 38 ms median commit | [costs](../repos/miles/kb/costs.md) |
| **Transformers** | CPU CI & 1× B200 (`Qwen3-0.6B` / `Qwen3-8B`) | 7.75 s cold → **0.61 s** live (12.8×); H2D hook resume **0.31 s** (15.4 GiB freed) | [architecture](../repos/transformers/docs/architecture.md) · [costs](../repos/transformers/kb/costs.md) |
| **TRT-LLM** | CPU CI (`ToyTrtllmLab` 11/11) & 1×/2× B200 (`TP=1` & `TP=2` MPI `11/11` conformance + `R1` CUDA-graph model-layer swap + `TrtllmQuiesceHooks`) | `TP=1`: 26.7 s cold / 12.4 s warm → **0.107 s** live (249× / 116×); `TP=2`: **0.275 s** `Qwen3DecoderLayer.forward` `R1` install (`224/224` witness/rank); KV quiesce frees **46.7 GiB** (`15.2 ms` resume) | [architecture](../repos/trtllm/docs/architecture.md) · [costs](../repos/trtllm/kb/costs.md) |
| **TorchTitan** | CPU CI (2-rank toy) & 1× B200 (FSDP2 `fully_shard` `world_size=1`, synthetic `TitanMLPNet`) | 2.69 s cold → **0.083 s** live + S0 reset (32.5×); 54.1 ms median install, 45.8 ms undo | [architecture](../repos/torchtitan/docs/architecture.md) · [costs](../repos/torchtitan/kb/costs.md) · [walkthrough](../repos/torchtitan/examples/live-step-swap/README.md) |
| **Toy sandbox** | CPU CI (2-rank stdlib reference) | ~1.5 s for all 11 verification cases on a laptop | [verification suite](../core/livelab/conformance/README.md) |

## Repository layout

```
core/livelab/        shared stdlib runtime (kernel/, rsibox/, conformance/, survey/, forksafe.py, INTERFACE.md)
core/agents/         prompts for coding agents (livelab_overview, developer, orchestrator, toolsmith)
repos/<engine>/      vllm, fastvideo, megatron, verl, sglang, miles, transformers, trtllm, torchtitan (docs/, kb/, livelab/, examples/)
docs/                design notes on engines, restart mechanisms, kernels/CUDA graphs, and CI
.github/             CPU workflows, automated review service, leak check, and Modal GPU CI runner
scripts/             fact and consistency checks (verify_facts.py)
template/            scaffold for adding a new engine
```
