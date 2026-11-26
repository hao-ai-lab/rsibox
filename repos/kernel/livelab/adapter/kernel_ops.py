"""Hot-swappable operators, CUDA `.cu` builder/loader, and reference math for KernelLab."""
from __future__ import annotations

import ctypes, fcntl, glob, hashlib, math, os, shutil, subprocess, tempfile, threading, time
from typing import Any

INIT_KERNEL_GAIN, KERNEL_SCALE = 1.0, 1.0
_CU_LIBS: dict[str, ctypes.CDLL] = {}
_CU_LOCK = threading.Lock()


def init_kernel_gain() -> float:
    """Startup-only gain factor (called only at S0 init; used by c10 zero_witness)."""
    return INIT_KERNEL_GAIN


def make_scale_schedule(base_weight: float, decay: float = 0.95):
    """Return a step-indexed schedule closure (used by c6 closure-change refusal)."""
    floor_w = base_weight * 0.1  # noqa: F841
    return lambda step: max(floor_w, base_weight) if step < 2 else base_weight


DEFAULT_SCALE_SCHEDULE = make_scale_schedule(1.0)


class KernelConfig:
    """Immutable kernel launch configuration captured at S0 startup."""

    def __init__(self, numel: int = 256, block_size: int = 128, seed: int = 42, backend: str = "cpu") -> None:
        self.numel, self.block_size, self.seed, self.backend, self.gain = int(numel), int(block_size), int(seed), str(backend), init_kernel_gain()


def cuda_kernel_source() -> str:
    """Return hot-swappable CUDA `.cu` source for the fused affine-SiLU operator."""
    cu_gain = 1.0
    return (
        '#include <cuda_runtime.h>\n#include <math.h>\nextern "C" __global__ void fused_affine_silu_cu(const float* x,const float* b,float* o,int n,float a,float beta){\n'
        f"  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){{float z=x[i]*(a*{cu_gain:.8f}f)+b[i]+beta; o[i]=z/(1.0f+expf(-z));}}\n}}\n"
        'extern "C" int launch_fused_affine_silu(const float* x,const float* b,float* o,int n,float a,float beta,int blk,void* s){\n'
        "  fused_affine_silu_cu<<<(n+blk-1)/blk,blk,0,(cudaStream_t)s>>>(x,b,o,n,a,beta); return (int)cudaGetLastError();\n}\n"
    )


def find_nvcc() -> str:
    """Locate `nvcc` without requiring it on default PATH."""
    ch = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    cands = [os.environ.get("RSIBOX_NVCC"), shutil.which("nvcc"), os.path.join(ch, "bin", "nvcc") if ch else None, "/usr/local/cuda/bin/nvcc", *sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True)]
    for path in cands:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError("nvcc not found (set RSIBOX_NVCC or CUDA_HOME)")


def compile_and_load_cuda(source: str, *, build_root: str | None = None, arch: str | None = None, device: str | int | None = None, use_cache: bool = True) -> dict[str, Any]:
    """Compile `.cu` into a versioned `.so` under `flock` and load via `RTLD_NOW | RTLD_LOCAL`."""
    if not arch:
        import torch  # type: ignore[import-not-found]
        dev = torch.cuda.current_device() if device is None else device
        arch = "".join(str(v) for v in torch.cuda.get_device_capability(dev))
    sm = arch.removeprefix("sm_").removeprefix("compute_")
    key = hashlib.sha256(f"sm_{sm}:{source}".encode("utf-8")).hexdigest()[:12]
    bdir = os.path.join(build_root or os.environ.get("RSIBOX_KERNEL_BUILD_ROOT") or os.path.join(tempfile.gettempdir(), f"rsibox-kernel-{os.getuid()}"), key)
    os.makedirs(bdir, exist_ok=True)
    so_path, lock_path, cached, t0 = os.path.join(bdir, f"libkernel_{key}.so"), os.path.join(bdir, "build.lock"), False, time.perf_counter()
    with open(lock_path, "w", encoding="utf-8") as lfh:
        fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
        if use_cache and os.path.isfile(so_path):
            cached = True
        else:
            nvcc = find_nvcc()
            cu_tmp, so_tmp = os.path.join(bdir, f"k_{os.getpid()}.cu"), os.path.join(bdir, f"libkernel_{key}.so.tmp.{os.getpid()}")
            try:
                with open(cu_tmp, "w", encoding="utf-8") as cfh: cfh.write(source)
                proc = subprocess.run([nvcc, "-O2", "-std=c++17", "-Xcompiler=-fPIC", f"-gencode=arch=compute_{sm},code=sm_{sm}", "-shared", cu_tmp, "-o", so_tmp], capture_output=True, text=True, timeout=120)
                if proc.returncode != 0: raise RuntimeError(f"nvcc failed: {(proc.stderr or proc.stdout).strip()[-1000:]}")
                os.replace(so_tmp, so_path)
            finally:
                for pth in (cu_tmp, so_tmp):
                    if os.path.exists(pth): os.unlink(pth)
    with _CU_LOCK:
        if key not in _CU_LIBS:
            lib = ctypes.CDLL(so_path, mode=getattr(os, "RTLD_NOW", ctypes.DEFAULT_MODE) | getattr(os, "RTLD_LOCAL", 0))
            lib.launch_fused_affine_silu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
            lib.launch_fused_affine_silu.restype = ctypes.c_int
            _CU_LIBS[key] = lib
    return {"key": key, "so": so_path, "lib": _CU_LIBS[key], "cached": cached, "compile_ms": round((time.perf_counter() - t0) * 1000.0, 3)}


def cuda_launch_params(alpha: float = 1.5, beta: float = 0.0) -> tuple[float, float]:
    """Return `(alpha, beta)` launch scalars for CUDA `.cu` execution."""
    scale = 1.0
    return float(alpha) * scale, float(beta)


def prepare_cpu_bias(bias: list[float], scale: float = 1.0) -> list[float]:
    """Scale bias vector for CPU toy kernels (used by multi-function conformance swaps)."""
    bias_gain = 1.0
    return [b * scale * bias_gain for b in bias]


def run_cpu_kernel(xs: list[float], bias: list[float], alpha: float = 1.5, beta: float = 0.0) -> list[float]:
    """Pure-Python fused affine-SiLU kernel mirroring the GPU Triton/CUDA operators."""
    kernel_gain = 1.0 * KERNEL_SCALE
    return [(z / (1.0 + math.exp(-z)) if z >= -60.0 else 0.0) for x, b in zip(xs, bias) for z in (x * (alpha * kernel_gain) + b + beta,)]
