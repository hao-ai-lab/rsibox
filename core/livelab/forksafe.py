"""Forksafe: keep CUDA uninitialized in a process that only asks what GPU this is, so the
processes it forks (servers, engine cores, workers, training ranks) can still use the GPU.

An engine zygote imports the heavy modules once and forks every later process from that warm
state. A child forked from a process that initialized CUDA cannot use CUDA, so the zygote must
import without touching it. Several imports touch it only to ask about the GPU:

- the compute capability: `torch.cuda.get_device_capability` (FlashInfer at import, imported by
  vLLM's API server at TP>1; quack through Transformer Engine under Megatron), and
  `torch.cuda.current_device`;
- the device properties: `torch.cuda.get_device_properties(i)` read at module level
  (FlashInfer's gdn kernels read `multi_processor_count`);
- the Triton target: `CudaDriver.get_current_target`, asked at module level by kernels that
  build autotune configs at import (`fastvideo_kernel`).

Each guard answers that question without CUDA while CUDA is not initialized, and runs the
original once it is:
  install(cuda)                    get_device_capability(d) answers the capability every
                                   visible GPU shares (NVML), current_device() answers 0 (what
                                   CUDA reports right after it initializes)
  install_properties(cuda, probe)  get_device_properties(i) answers the values CUDA reported
                                   for device i in a short helper process
  pin_triton_target(capability)    Triton's current target answers the GPU's capability
                                   (NVML); returns restore()
  unpatch(cuda, clear_env)         in a forked child: puts back the torch.cuda functions above
When the visible GPUs do not share one capability, or NVML fails, the original runs (and
initializes CUDA, as without this); nothing is guessed. Every answer and every fallback of
install() is counted in COUNTS.

Contract:
- install(cuda) patches the given torch.cuda module once; it returns what it did.
- An answer from NVML equals what CUDA would return for these devices: one capability shared
  by every visible GPU, for device None, 0..n-1, "cuda", "cuda:i" or a cuda torch.device,
  where n is the number of visible GPUs.
- Anything else (an index outside 0..n-1, another device type, differing capabilities, no
  NVML, CUDA already initialized) calls the original function, which raises or initializes
  CUDA exactly as without this module.
- install_properties answers only for the devices the probe reported, before CUDA is
  initialized; any other call runs the original.
- unpatch puts back only functions this module patched, and returns their names.
A zygote installs the guards before its imports, so every process it forks inherits them.
"""
import functools, json, os, subprocess, sys, time

COUNTS = {"get_device_capability": 0, "current_device": 0, "fallback": 0}
PATCHED = ("get_device_capability", "current_device", "get_device_properties")


def _nvml():
    try:
        import pynvml
        return pynvml
    except Exception:
        try:
            from vllm.third_party import pynvml
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
    default NVML for the visible GPUs, read at the first call (in the forked process, with its
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
    current_device._fastrsi_forksafe = True
    cuda.get_device_capability = get_device_capability
    cuda.current_device = current_device
    return {"installed": True}


PROPS_CODE = """import json, torch
out = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    out.append({k: getattr(p, k) for k in dir(p) if not k.startswith("_")
                and isinstance(getattr(p, k), (int, float, str, bool))})
print(json.dumps(out))
"""


class Properties:
    """Stand-in for torch.cuda's device properties, with the values CUDA reported."""
    def __init__(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        return f"Properties({self.__dict__})"


def _props_index(device):
    """0 for None/"cuda", i for i, "cuda:i" or a cuda torch.device (index None -> 0), None
    for anything else."""
    if device is None or device == "cuda":
        return 0
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        return int(device.split(":")[1]) if device.startswith("cuda:") else None
    if getattr(device, "type", None) == "cuda":
        return device.index or 0
    return None


def install_properties(cuda, probe=None):
    """Make cuda.get_device_properties(i) answer, while CUDA is not initialized here, with the
    values CUDA itself reported for device i in a short helper process (probe: a callable
    returning the list of dicts; default a subprocess running PROPS_CODE with this
    environment). Some imports read a value at module level (flashinfer's gdn kernels:
    get_device_properties(0).multi_processor_count) and would otherwise initialize CUDA.
    Indices outside the probed devices, or calls after CUDA is initialized, run the original."""
    orig = cuda.get_device_properties
    if getattr(orig, "_fastrsi_props", False):
        return {"installed": False, "why": "already installed"}
    try:
        t = time.time()
        props = probe() if probe else json.loads(subprocess.run(
            [sys.executable, "-c", PROPS_CODE], capture_output=True, text=True, check=True,
            timeout=300).stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"installed": False, "why": repr(e)[:300]}

    def get_device_properties(device=None):
        if not cuda.is_initialized():
            i = _props_index(device)
            if i is not None and 0 <= i < len(props):
                return Properties(props[i])
        return orig(device)
    get_device_properties._fastrsi_props = True
    get_device_properties.__wrapped__ = orig
    cuda.get_device_properties = get_device_properties
    return {"installed": True, "devices": len(props), "probe_s": round(time.time() - t, 2)}


def unpatch(cuda=None, clear_env=False):
    """In a forked child: put back the torch.cuda functions this module patched (install's
    get_device_capability and current_device, install_properties' get_device_properties).
    The child initializes CUDA itself, so it needs none of them, and the wrappers cost a check
    on every call. `cuda` defaults to the loaded torch.cuda; `clear_env` also drops
    PYTORCH_NVML_BASED_CUDA_CHECK (pass it when the zygote set the variable). Returns the names
    put back."""
    if cuda is None:
        t = sys.modules.get("torch")
        cuda = t.cuda if t is not None else None
    out = []
    if cuda is not None:
        for name in PATCHED:
            f = getattr(cuda, name, None)
            if f is not None and hasattr(f, "__wrapped__") and (
                    getattr(f, "_fastrsi_forksafe", False) or getattr(f, "_fastrsi_props", False)
                    or getattr(f, "__module__", "") == __name__):
                setattr(cuda, name, f.__wrapped__)
                out.append(name)
    if clear_env:
        os.environ.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    return out


def triton_capability(capability=None, nvml=None):
    """The (major, minor) Triton should target: `capability` if given, else FASTRSI_ZYGOTE_CC
    ("10.0"), else NVML: the capability every visible GPU shares, else the first visible GPU's
    (physical GPU 0's when they are named by UUID). Raises RuntimeError when NVML cannot answer."""
    if capability is not None:
        major, minor = capability
        return int(major), int(minor)
    cc = os.environ.get("FASTRSI_ZYGOTE_CC")
    if cc:
        major, minor = cc.split(".")
        return int(major), int(minor)
    nvml = nvml or _nvml()
    cap = nvml_capability(os.environ.get("CUDA_VISIBLE_DEVICES"), nvml)
    if cap is not None:
        return tuple(cap)
    if nvml is None:
        raise RuntimeError("no NVML (pynvml) to read the GPU's capability; set FASTRSI_ZYGOTE_CC")
    # mixed GPUs: the first visible one is CUDA device 0 in the child; UUIDs fall back to GPU 0
    vis = os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    first = [int(x.strip()) for x in vis.split(",") if x.strip().isdigit()]
    nvml.nvmlInit()
    try:
        return tuple(nvml.nvmlDeviceGetCudaComputeCapability(
            nvml.nvmlDeviceGetHandleByIndex(first[0] if first else 0)))
    finally:
        nvml.nvmlShutdown()


def pin_triton_target(capability=None, nvml=None):
    """Answer Triton's "current GPU target" from `capability` (see triton_capability) instead of
    the CUDA driver, which a module-level query would initialize. Call it before the imports
    and the returned restore() after them. Without Triton nothing is patched and restore()
    does nothing."""
    try:
        from triton.backends.compiler import GPUTarget
        from triton.backends.nvidia.driver import CudaDriver
    except Exception:
        return lambda: None
    major, minor = triton_capability(capability, nvml)
    had = "get_current_target" in CudaDriver.__dict__
    orig = CudaDriver.__dict__.get("get_current_target")
    CudaDriver.get_current_target = lambda self: GPUTarget("cuda", major * 10 + minor, 32)

    def restore():
        if had:
            CudaDriver.get_current_target = orig
        else:
            del CudaDriver.get_current_target
    return restore
