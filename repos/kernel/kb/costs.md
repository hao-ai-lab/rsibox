# Measured Costs (Standalone Kernel Engine: Triton, CUDA `.cu`, & CPU Toy)

## Real GPU (`1× NVIDIA B200`, `CUDA_VISIBLE_DEVICES=1`, `torch 2.14.0+cu130`, `triton 3.8.0`, `nvcc 13.3` `sm_100`)

| Backend | Shape(s) | Cold Boot + 1st Check | Live Install + Check | Speedup | Undo | Fresh Parity |
|---|---|---:|---:|---:|---:|---|
| **Triton `@triton.jit`** | `4096×128`, `16384×256` | `4.6373 s` (`4.6323 s` + `0.0050 s`) | **`0.0398 s`** (`0.0336 s` + `0.0061 s`) | **`116.5×`** | `0.0050 s` | `max_abs_diff=0.0` base; exact vs `fresh_check` (`0.1776 / 0.1471 ms` kernel) |
| **CUDA `.cu` (`nvcc` + `CDLL`)** | `4096×128`, `16384×256` | `2.9640 s` (`2.9600 s` + `0.0040 s`) | **`0.9467 s`** (`0.0234 s` + `917.442 ms` `nvcc` + launch) | **`3.13×`** | `0.0050 s` (`0.0037 s` check, `0 ms` recompile) | `max_abs_diff=0.0` base; exact vs `fresh_check` (`0.0684 / 0.0593 ms` kernel) |

- **B200 Conformance (`rsibox conformance kernel --real`):** **11/11 PASS** in `59.334 s` (`c0` `0.540 s`, `c1` `0.612 s`, `c2` `0.562 s`, `c3` `8.251 s`, `c4` `0.668 s`, `c5` `0.689 s`, `c6` `0.576 s`, `c7` `0.645 s`, `c8` `42.513 s`, `c9` `1.222 s` [install median `0.0204 s`, uninstall median `0.0029 s`], `c10` `3.056 s`), explicit `--backend cuda` conformance **5/5 PASS** (`c0,c2,c3,c4,c5`), plus real `CUDA_ERROR_MISALIGNED_ADDRESS` sticky CUDA fault (`poisoned` -> `session_poisoned` refusal -> `respawn` `2.874 s`, `0 MiB` exit).

## CPU Toy Conformance (`ToyKernelLab`, `numel=256, block_size=128`)

- **11/11 PASS** (`c9_soak` install median `0.0067 s`, uninstall median `0.0009 s`; `6/6` unit tests pass with `-W error::ResourceWarning`).
