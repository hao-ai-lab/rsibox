# Codex Native RSIBox Hill-Climb on `TransformersLab` (`gemma-3-1b-it`)

We ran an end-to-end 5-iteration hill-climb experiment (`#289`) where OpenAI Codex (`codex-cli 0.171.0`, `gpt-6.2-astra`, `reasoning effort: high`) operated strictly inside an isolated temporary sandbox using only the native `rsibox` CLI/HTTP client (`rsibox`, `rsibox.client.Client`) and the repository agent/KB docs (`core/agents/*.md`, `core/livelab/INTERFACE.md`, `core/livelab/rsibox/README.md`, `repos/transformers/kb/*`, `repos/transformers/docs/architecture.md`), with zero hand-holding hints.

The target sandbox was a live `TransformersLab` daemon serving `gemma-3-1b-it` (`bfloat16`, PyTorch `2.14.0+cu130`, Transformers `5.20.2`) on `1x NVIDIA B200` (`CUDA_VISIBLE_DEVICES=3`), bridged to the local client via an SSH localhost port forward (`RSIBOX=./box.json`).

## Result Summary

- **Kept 1 of 5 edits:** [`rmsnorm-addcmul.patch`](rmsnorm-addcmul.patch) replaces `output * (1.0 + self.weight.float())` in `Gemma3RMSNorm.forward` (`transformers.models.gemma3.modeling_gemma3`) with `torch.addcmul(output, output, self.weight)`.
- **Warm speedup (2 alternating fresh-worker pairs, 1 warmup + 3 checks/shape):**
  - **Shape 1 (`2 prompts × 32 tokens`, `temperature=0.0`):** `0.562045 s -> 0.515439 s` median (**`1.090x`**, `-8.29%`; pair ratios `1.099x`, `1.085x`), bit-exact digest `d25f052c37d9308a`.
  - **Shape 2 (`1 prompt × 16 tokens`, `temperature=0.0`):** `0.256656 s -> 0.229571 s` median (**`1.118x`**, `-10.55%`; pair ratios `1.117x`, `1.120x`), bit-exact digest `7887461c74a8bcc4`.
- **Fresh-process `PackageOverlay` parity (`TransformersLab.fresh_check`):** Verified bit-exact token parity on both shapes (`bs2_tok32`: `8.9862 s` base vs `8.3616 s` candidate, digest `d25f052c37d9308a`; `bs1_tok16`: `7.5029 s` base vs `7.7387 s` candidate, digest `7887461c74a8bcc4`).
- **Cold vs. live loop overhead:** Cold worker boot took `5.772 s` (`13.068442 s` first check; `18.840442 s` cold total), whereas live `install` took `0.159–0.286 s`, `uninstall` `0.111–0.126 s`, and warm checks `~0.50–0.55 s` (`239.891 s` total Codex box span across 30 shell commands; `2,254,048` input / `11,791` output tokens).

## Baseline Noise

Across 8 warm base checks on Shape 1 (`2 prompts × 32 tokens`, `temperature=0.0`), runtimes were `0.547595`, `0.550717`, `0.548962`, `0.550817`, `0.561626`, `0.555045`, `0.548535`, and `0.548820 s` (all digest `d25f052c37d9308a`): median **`0.549840 s`**, min–max spread **`0.014031 s` (`2.55%`)**.

## Five-Iteration Hill Climb

| iter | hypothesis | module / target | install_s | check_s | base_s | speedup | witness (per check) | output_match | decision |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | Use `nn.functional.rms_norm` | `modeling_gemma3` · `Gemma3RMSNorm.forward` | 0.159245 | 0.469807 | 0.549840 | 1.170x | 5,024 | No (token 17 diverged) | Revert (`0.111323 s`) |
| 2 | Fuse Q/K rotary `cat` + `split` | `modeling_gemma3` · `apply_rotary_pos_emb` | 0.162312 | 0.540464 | 0.544998 | 1.008x | 832 | Exact | Revert (inside 2.55% noise) |
| 3 | Replace `*(1+w)` with `torch.addcmul` | `modeling_gemma3` · `Gemma3RMSNorm.forward` | 0.159947 | 0.499922 | 0.547336 | 1.095x | 5,024 | Exact (2 shapes) | **Keep** (`txn-6a399520`) |
| 4 | Cache packed Q/K/V weights on `DynamicCache` | `modeling_gemma3` · `Gemma3Attention.forward` | 0.173618 | 0.525501 | 0.502053 | 0.955x | 832 | No (token 17 diverged) | Revert (`0.122058 s`) |
| 5 | Skip FP32 logits copy on greedy argmax | `generation.utils` · `GenerationMixin._sample` | 0.285624 | 0.502930 | 0.505257 | 1.005x | 1 | Exact | Revert (inside 2.55% noise) |

Notes:
- Iterations 1 and 4 both failed parity on generated token 17 of the first prompt (`12763` vs `167612`, digest `2f1ccaf88224f03f`) and were reverted immediately.
- Iteration 3's first post-install check took `2.579783 s` (included in its 5-check median `0.499922 s`; steady checks 2–5 were `0.497716–0.503687 s`).

## Native `rsibox` Usage & Toolsmith Friction Findings

### What Worked on the First Try

- Codex read `core/agents/` (`livelab_overview.md`, `developer.md`, `toolsmith_online.md`), `core/livelab/INTERFACE.md`, `core/livelab/rsibox/README.md`, `client.py`, and `repos/transformers/kb/` before touching the box.
- **Zero install refusals, zero HTTP failures, zero poisoned states, and zero failed rollbacks.** Codex successfully used `rsibox status`, `rsibox source`, and `rsibox.client.Client.from_env()` with `base` module hashes (`"base": {module: hash}`) on its first attempt.

### Exact Mistakes & Toolsmith / API Friction Points

1. **`cost_s` vs. `seconds` timing ambiguity:** `core/livelab/INTERFACE.md` documents `cost_s` on every verb, whereas `BaseLab.check()` also returns `seconds` (workload execution time without probe/proof overhead). Codex initially logged `cost_s` (`13.708 ms` spread) before inspecting `core/livelab/rsibox/lab.py` and switching to `seconds` (`14.031 ms` spread).
2. **`status()["fingerprint"]` changes across `reset("respawn")`:** In its confirmation script, Codex asserted `status()["fingerprint"] == "5ebfed89e412fd06"` after `box.reset("respawn")`, hit an `AssertionError` (`0bccd99b5831ec5c`), and had to inspect `core/livelab/kernel/coordinator.py` and `identity.py` to discover that `LocalParticipant.fingerprint()` pairs code hashes with process-local `id(code_obj)`. Switching the assertion to `status()["identity"]["modules"]` + `state == "base"` + `txns == []` resolved it. Also `identity.metadata.engine.commit` reported `null`.
3. **`fresh_check` is missing from the HTTP box API and CLI:** `TransformersLab` implements `fresh_check(patch, workload)`, but `Box`, `Client`, and `rsibox` CLI expose only the five core verbs (`install`, `check`, `reset`, `uninstall`, `status`), and `rsibox promote` expects external shell commands. Codex therefore could not run an on-disk `PackageOverlay` fresh check via `RSIBOX` and used `reset("respawn")` + live `install` instead (we verified `TransformersLab.fresh_check` separately after the run).
4. **Cumulative witness counters & entry-only `proven` semantics:** Witness counts accumulated monotonically across `check()` calls (`5,024 -> 10,048 -> ...`) and persisted across `reset("state")`, forcing client-side differencing. Entry-only witnesses cannot prove a guarded fast path inside a function executed (iteration 5), and `proven=True` also appeared on token-diverging runs (iterations 1 and 4) because proof receipts record `golden.output_hash` without comparing against a registered base golden server-side.
5. **No phase breakdown, profiler, or recompile telemetry:** `check()` returned scalar `seconds` without prefill/decode split or `torch._dynamo` recompile counts, and the `2.580 s` spike on iteration 3's first post-install check emitted no cost-hook alarm or phase explanation.
6. **Unreachable `toolsmith_online` and missing/stale KB links:** `core/agents/developer.md` instructs calling `toolsmith_online` on slow checks or uncovered paths, but no callable Toolsmith endpoint exists inside the `rsibox` sandbox. `repos/transformers/kb/` lacked a slow-check threshold and a 32-token Gemma baseline entry, and `repos/transformers/livelab/README.md` (referenced in `core/agents/livelab_overview.md`) does not exist.
