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
5. **Multi-rank `TP=2` (`--tp 2` on `2× B200`):** node-local `flock` compiles once per `.so`
   cache key across TP ranks (`0.0 s` on the cached peer), `576` live / `432` capture witness calls
   per rank, `max |Δlogprob| = 0.0`, and `0 MiB` residual GPU memory:
   - Serial `nvcc`: `18.24–18.27 s` build (`visible_perturbation` `19.95 s` total vs `53.10 s`
     restart; PR `#49750` `0.916 s` `R1`, `20.26 s` total vs `55.28 s` restart, `9.17 ms` cascade undo).
   - Parallel `native.py` (`g++` `reg.cpp` + `-split-compile=16 --threads=16` + `g++ -shared`):
     `11.72–11.85 s` build (`visible_perturbation` `11.716 s` build, `0.480 s` `R1`, `13.28 s`
     total vs `49.42 s` restart; PR `#49750` `11.854 s` build, `0.676 s` `R1`, `12.558 s`
     edit-to-rebuilt, `13.62 s` total vs `46.97 s` restart, `8.94 ms` cascade undo).

6. **V2 model runner (`--v2` -> `ModelCudaGraphManager`):**
   - `TP=1` selective `R1`: refuses without `R1`, selectively recaptures `[1]` in `0.299 s` (running `[4, 2]` eager concurrently), extends `[2, 4]` via `recapture_extend` in `0.360 s`, proves `1728` live / `432` capture witness calls, restores base in `2.93 ms` (`undo`), keeps `0.0 MiB` reserved growth across 5 consecutive recapture+undo cycles (`0.381–0.491 s` recapture, `2.56–2.81 ms` undo), and matches fresh V2 (`30.892 s`) with `max |Δlogprob| = 0.0`.
   - `TP=2` (`2× B200`) PR `#49750` on V2: visible perturbation `13.73 s` total incl. check (`12.659 s` edit-to-rebuilt [`12.129 s` build + `0.511 s` R1] + `1.070 s` check vs `48.878 s` fresh) and PR `#49750` `14.13 s` total incl. check (`13.063 s` edit-to-rebuilt [`12.224 s` build + `0.818 s` R1] + `1.070 s` check, `9.71 ms` cascade undo vs `47.667 s` fresh), `576` live / `432` capture witness calls per rank, `max |Δlogprob| = 0.0`, `0 MiB` residual GPU memory.

## Usage

```sh
CUDA_VISIBLE_DEVICES=0 python3 -u verify_native_pr49750.py \
  --model <qwen3-8b-path> \
  --work <scratch-dir> \
  --csrc-root <path-to-vllm-csrc> \
  --vllm-git <path-to-vllm.git> \
  --tp 1 [--v2] \
  --port 29495 \
  --out summary.json
```
