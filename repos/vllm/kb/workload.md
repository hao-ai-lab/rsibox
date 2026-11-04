# vLLM target workload

Source: LiveLab-MLSys paper draft, Appendix A.

- **Model:** DeepSeek-V4.1-Flash, real weights.
- **Hardware:** one GB200 tray, 4 GPUs, 4 data-parallel replicas with expert parallelism (EP4DP4).
- **Benchmark:** the test cases of `vllm bench forward`.
  - Prefill: time to process 8K, 16K, 32K and 64K new tokens after a cached 128K-token prefix.
  - Decode: time for one new token per request at 1, 8 and 32 requests per GPU (32K context),
    and 16 requests per GPU (128K context).
- **Score:** each case independently, as step-time speedup over the starting commit.
- **Correctness gate:** decode outputs identical to the starting commit; prefill outputs within
  relative RMSE 0.10 of the reference.
- **Frozen:** weights, quantization, launch flags, test cases and timing code.

The vLLM blog's prefill numbers for the three example PRs use 4K–64K new tokens after a 128K
prefix on DEP2 GB300, so they are a target, not a like-for-like comparison.
