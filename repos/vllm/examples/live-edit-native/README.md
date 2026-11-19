# Live C++/CUDA Hot-Reload on vLLM PR `#49750` (`NATIVE`)

End-to-end B200 GPU testification for [`repos/vllm/livelab/adapter/native.py`](../../livelab/adapter/native.py)
using upstream vLLM PR `#49750` (`csrc/libtorch_stable/layernorm_kernels.cu`: non-contiguous
`residual` layout support in `fused_add_rms_norm`).

## What [`verify_native_pr49750.py`](verify_native_pr49750.py) Proves

1. **Direct GPU non-contiguous check:** builds both the pre-PR and PR `#49750` kernels into
   versioned `.so` libraries on node-local `/tmp`; confirms the pre-PR kernel rejects a
   non-contiguous `residual` (`stride(-1) == 1, is_contiguous() == False`) with `RuntimeError`,
   while PR `#49750` accepts it (`max abs err 0.015625` vs FP32 reference).
2. **CUDA-graph safety refusal:** submitting a `native` plan against a `FULL_DECODE_ONLY`
   server without `"rebuild": ["R1"]` aborts cleanly in `0.022 s` (`captured_target`, routing to
   `R1`) before invoking `nvcc`.
3. **Visible `.cu` perturbation (`epsilon = 0.25f;`) + `R1` recapture:** compiles in `17.10 s`,
   loads in `0.88 ms`, commits in a `2.30 ms` gap, recaptures `[4, 2, 1]` CUDA graphs in
   `0.473 s`, records `432` capture calls and `576` live witness calls, visibly alters greedy
   output, matches a fresh server running the same `.cu` edit bit-for-bit
   (`max |Δlogprob| = 0.0`; `18.87 s` total vs `48.97 s` restart), and restores exact base output
   on `undo` in `2.85 ms` (`1.96 ms` worker unbind).
4. **Upstream PR `#49750` (`pre-PR` -> PR `#49750`):** compiles in `17.09 s`, loads in
   `1.17 ms`, commits in `2.17 ms`, recaptures graphs in `0.826 s` (`432` capture / `576` live
   witness calls), matches fresh restart bit-for-bit (`max |Δlogprob| = 0.0`; `19.22 s` total
   vs `49.47 s` restart), and restores exact base on `cascade_undo` in `6.55 ms` (`5.60 ms`
   worker unbind).

## Usage

```sh
CUDA_VISIBLE_DEVICES=0 python3 -u verify_native_pr49750.py \
  --model <qwen3-8b-path> \
  --work <scratch-dir> \
  --csrc-root <path-to-vllm-csrc> \
  --vllm-git <path-to-vllm.git> \
  --port 29495 \
  --out summary.json
```
