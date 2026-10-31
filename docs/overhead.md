# Overhead book

What FastRSI itself costs, generated from `receipts/` by `tools/overhead_doc.py`. One model, Qwen3-8B, on one GPU per engine. Measured on a shared host.

| Cost | vLLM | SGLang |
|---|---:|---:|
| Workload, stock median | 0.2924 s | 1.0349 s |
| Workload, agent loaded | 0.2923 s (-0.05%) | 1.0354 s (+0.06%) |
| Host memory, stock | 3,905-3,913 MiB | 6,076-6,095 MiB |
| Host memory, agent | 3,886-3,920 MiB | 6,080-6,089 MiB |
| GPU memory, stock vs agent | [54892] vs [54892] MiB | [58927] vs [58927] MiB |
| Install, median / max | 22 / 44 ms | 79 / 740 ms |
| Undo, median / max | 4 / 9 ms | 16 / 26 ms |
| Host growth | +37 MiB over 100 cycles (+0.37 MiB/cycle) | +49 MiB over 40 cycles (+1.22 MiB/cycle) |
| GPU growth over cycles | +0 MiB | +0 MiB |
| Base read, late vs early cycles | +0.01% | -0.11% |
| Hang detection | 60 s (the configured timeout) | 60 s (the configured timeout) |
| Restart to base | 40 s | 59 s |

## How to read this

- Stock and agent arms alternate in ABBA order, with a full engine start each. The idle difference is within run-to-run noise on both engines, so read it as not detected, not as zero.
- SGLang runs one request at a time here, so its figures do not describe batched serving. See below.
- Host growth is real and repeatable, but its cause is not isolated. It stays small next to a process of several GiB, and the install budget forces a restart first.
- Hang detection equals the timeout you choose. A shorter one finds hangs sooner and risks calling a slow request a hang.
- Only sampler edits were measured. Other routes will cost differently.

## Witness cost per call

The workloads above cannot resolve this, so it was timed directly.

| Mode | ns per call |
|---|---:|
| Off | 36 |
| First hit only, the default | 34 |
| Every call | 137 |

## Repeatability

Exact comparison only means something if base is repeatable. Each row is 60 identical runs, with a cache reset between them.

| Engine | Distinct outputs |
|---|---:|
| sglang-stock, 32 prompts batched | 6 |
| sglang-agent, 32 prompts batched | 5 |
| vllm-agent, 32 prompts batched | 1 |

SGLang varies with no agent present, so FastRSI does not cause it. Its plugin runs one request at a time to get a repeatable base. SGLang's deterministic inference mode failed to start here with a FlashInfer workspace overflow.
