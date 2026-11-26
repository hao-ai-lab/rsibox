# Standalone Kernel Engine (`kernel` Triton `@triton.jit` + CUDA `.cu`)

RSIBox live-lab adapter for KernelBench-style single Triton `@triton.jit` and CUDA `.cu` operators without booting an LLM serving or training framework, featuring sub-50ms Triton JIT rebuilds (`core/livelab/kernel/jit.py`), persistent-context `.cu` incremental `nvcc` + `ctypes.CDLL` reload, deterministic `S0` multi-shape reference parity, and optional `KernelService` (`critical_region`) composition.

| Path | Holds |
|---|---|
| [kb/costs.md](kb/costs.md) | Measured CPU (`ToyKernelLab`) and NVIDIA B200 Triton / CUDA `.cu` timings |
| `livelab/` | `livelab.json` and `adapter/{lab,kernel_ops,triton_kernels,worker,test_kernel_lab}.py` |
