"""Forksafe: keep CUDA uninitialized in a process that only asks what GPU this is, so the
servers the zygote forks can fork their engine core and workers too.

vLLM forks the engine core and the workers when VLLM_WORKER_MULTIPROC_METHOD=fork, so they
inherit the parent's imports. It falls back to spawn ("We must use the `spawn` multiprocessing
start method ... Reasons: CUDA is initialized") as soon as anything in the API server has
initialized CUDA. Spawned processes import vLLM again: on a shared filesystem that cost 33 s
and 121 s in two of our TP2 starts (engine core plus workers). In our TP2 runs a warm forked
start reached /health in 22-24 s, against 64-157 s when vLLM fell back to spawn.

In the API server CUDA gets initialized only to read the compute capability. At TP>1 the
config imports FlashInfer (for the all-reduce fusion size), and FlashInfer calls
torch.cuda.get_device_capability at import (flashinfer/jit/env.py through CompilationContext,
and flashinfer/kda_kernels). install() makes that function, and torch.cuda.current_device,
answer from NVML while CUDA is not initialized:
  get_device_capability(d)  the capability every visible GPU shares
  current_device()          0 (what CUDA reports right after it initializes)
Once CUDA is initialized, the original functions run. When the visible GPUs do not share one
capability, or NVML fails, the original runs too (and initializes CUDA, as without this);
nothing is guessed. Every answer and every fallback is counted in COUNTS.

Contract:
- install(cuda) patches the given torch.cuda module once; it returns what it did.
- An answer from NVML equals what CUDA would return for these devices: one capability shared
  by every visible GPU, for device None, 0..n-1, "cuda", "cuda:i" or a cuda torch.device,
  where n is the number of visible GPUs.
- Anything else (an index outside 0..n-1, another device type, differing capabilities, no
  NVML, CUDA already initialized) calls the original function, which raises or initializes
  CUDA exactly as without this module.
The zygote installs it before its imports (zygote.py), so every server it forks inherits it.
"""
import functools, os

COUNTS = {"get_device_capability": 0, "current_device": 0, "fallback": 0}


def _nvml():
    try:
        from vllm.third_party import pynvml
        return pynvml
    except Exception:
        try:
            import pynvml
            return pynvml
        except Exception:
            return None


def nvml_capability(visible=None, nvml=None):
    """The compute capability (major, minor) every visible GPU shares, or None."""
    r = nvml_visible(visible, nvml)
    return r[0] if r else None


def nvml_visible(visible=None, nvml=None):
    """(the compute capability every visible GPU shares, the number of visible GPUs), or
    None."""
    nvml = nvml or _nvml()
    if nvml is None:
        return None
    try:
        nvml.nvmlInit()
    except Exception:
        return None
    try:
        if visible is None:
            idx = list(range(nvml.nvmlDeviceGetCount()))
        else:
            ids = [x.strip() for x in visible.split(",") if x.strip()]
            if not ids or not all(x.isdigit() for x in ids):
                return None                       # UUIDs or MIG: leave it to CUDA
            idx = [int(x) for x in ids]
        caps = {tuple(nvml.nvmlDeviceGetCudaComputeCapability(nvml.nvmlDeviceGetHandleByIndex(i)))
                for i in idx}
        return (caps.pop(), len(idx)) if len(caps) == 1 else None
    except Exception:
        return None
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def _index(device):
    """Device index for a cuda device argument, None for the current device, -1 otherwise."""
    if device is None or isinstance(device, int):
        return device
    if isinstance(device, str):
        if device == "cuda":
            return None
        return int(device.split(":")[1]) if device.startswith("cuda:") else -1
    if getattr(device, "type", None) == "cuda":
        return device.index
    return -1


def install(cuda, capability=None):
    """Patch cuda.get_device_capability and cuda.current_device (see the module doc).
    `capability` returns (the shared capability, the number of visible GPUs) or None; by
    default NVML for the visible GPUs, read at the first call (in the forked server, with its
    environment)."""
    if not all(hasattr(cuda, f) for f in ("get_device_capability", "current_device",
                                          "is_initialized")):
        return {"installed": False, "why": "this torch.cuda lacks the functions to patch"}
    if getattr(cuda.get_device_capability, "_fastrsi_forksafe", False):
        return {"installed": False, "why": "already installed"}
    cap_of = capability or functools.lru_cache(maxsize=1)(
        lambda: nvml_visible(os.environ.get("CUDA_VISIBLE_DEVICES")))
    orig_cap, orig_cur = cuda.get_device_capability, cuda.current_device

    @functools.wraps(orig_cap)
    def get_device_capability(device=None):
        if not cuda.is_initialized():
            i, known = _index(device), cap_of()
            if known is not None and (i is None or 0 <= i < known[1]):
                COUNTS["get_device_capability"] += 1
                return known[0]
            COUNTS["fallback"] += 1
        return orig_cap(device)

    @functools.wraps(orig_cur)
    def current_device():
        if not cuda.is_initialized():
            if cap_of() is not None:
                COUNTS["current_device"] += 1
                return 0
            COUNTS["fallback"] += 1
        return orig_cur()

    get_device_capability._fastrsi_forksafe = True
    cuda.get_device_capability = get_device_capability
    cuda.current_device = current_device
    return {"installed": True}
