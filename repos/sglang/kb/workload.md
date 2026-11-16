# Workload F(x) and its correctness gate

The target workload for the SGLang live lab, fixed before any edit.

## Setup

| Setting | Value |
|---|---|
| Model | Qwen3-0.6B, real weights |
| Parallelism | TP 1, PP 1, DP 1 (1 GPU); PD disaggregation off |
| Engine | `sglang.Engine`, `random_seed=0`, `max_running_requests=1` |
| Prompts | Fixed deterministic list of prompts, frozen across runs |
| Sampling | `temperature=0.0`, fixed `max_new_tokens`, `ignore_eos=True` |
| Reset before each check | `reset("state")`: drain via `pause_generation(retract)` if overlap is on, `flush_cache()` (must return true), reseed `random_seed=0` |

Why `max_running_requests=1`: in our v0 repeatability check (`costs.md`), 32 prompts submitted
as one batch at `temperature=0.0` produced 6 distinct digests across 60 runs with no agent
loaded, whereas running one request at a time gave 1 distinct digest across 30 runs.

## Function F(x)

Wall seconds around `engine.generate(prompt=prompts, sampling_params=params)` for the fixed
prompt list, starting from the reset state above, plus a 16-hex SHA-256 digest of the JSON list
of returned token-id sequences (`output_ids`).

Score: step-time speedup over the starting commit on the same GPU in the same lab session,
confirmed on a fresh process before an edit is kept.

## Correctness gate

- **Repeatable base:** repeated `check()` calls with `reset("state")` between them, and across
  install/undo cycles, must produce the exact same token-id digest (`conformance` cases `c0`
  and `c4`).
- **Bit-exact token ids:** an edit that preserves numerics (scheduling, Python overheads, buffer
  reuse, caches) must produce the exact same `output_ids` digest as the base commit on every
  prompt.
- **Why token ids and not logprobs:** even under `--enable-deterministic-inference`, kernel
  dispatch that branches on batch shape or contiguity can move fp16/bf16 logprobs between
  single-request and batched calls while greedy token ids stay identical (upstream PR #50148 on
  Mamba-1 `dt_proj`: 15/15 decode positions differed in logprobs, 0/15 in greedy tokens).
- **Fresh confirmation:** a candidate edit counts only if a fresh SGLang process started from
  the edited files produces the same digest as the live swap (`conformance` case `c3`).
