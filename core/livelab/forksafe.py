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
import contextlib, ctypes, functools, gc, json, os, subprocess, sys, time

COUNTS = {"get_device_capability": 0, "current_device": 0, "fallback": 0}
PATCHED = ("get_device_capability", "current_device", "get_device_properties")
LIBCUDA = "libcuda.so.1"
_CUINIT = {}
_NONE_AFFINITY = {"cpus": [], "numa_node": None, "source": "none"}


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


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
        with contextlib.suppress(Exception):
            nvml.nvmlShutdown()


def _index(device):
    """Device index for a cuda device argument (0 for None or unindexed 'cuda'), or None."""
    if device is None or device == "cuda":
        return 0
    if isinstance(device, int) and not isinstance(device, bool):
        return device
    if isinstance(device, str):
        if not device.startswith("cuda:") or device.count(":") != 1:
            return None
        suffix = device.split(":", 1)[1]
        if suffix.isdigit() or (suffix[:1] in "+-" and suffix[1:].isdigit()):
            return int(suffix)
        return None
    if getattr(device, "type", None) == "cuda":
        return 0 if device.index is None else device.index
    return None


def install(cuda, capability=None):
    """Patch cuda.get_device_capability and cuda.current_device (see the module doc).
    `capability` returns (the shared capability, the number of visible GPUs) or None; by
    default NVML for the visible GPUs, read at the first call (in the forked process, with its
    environment)."""
    if not all(hasattr(cuda, attr_name) for attr_name in ("get_device_capability", "current_device",
                                                          "is_initialized")):
        return {"installed": False, "why": "this torch.cuda lacks the functions to patch"}
    if getattr(cuda.get_device_capability, "_fastrsi_forksafe", False):
        return {"installed": False, "why": "already installed"}

    @functools.lru_cache(maxsize=8)
    def cached_capability(process_id, visible_devices_env, has_visible_env):
        return nvml_visible(visible_devices_env if has_visible_env else None)

    cap_of = capability or (
        lambda: cached_capability(
            os.getpid(),
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CUDA_VISIBLE_DEVICES" in os.environ,
        )
    )
    orig_cap, orig_cur = cuda.get_device_capability, cuda.current_device

    @functools.wraps(orig_cap)
    def get_device_capability(device=None):
        if not cuda.is_initialized():
            device_index, known = _index(device), cap_of()
            if known is not None and device_index is not None and 0 <= device_index < known[1]:
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


def _probed_index(device_index, probed_visible):
    """The probe's index for CUDA device `device_index` of this process."""
    if device_index is None or os.environ.get("CUDA_VISIBLE_DEVICES") == probed_visible:
        return device_index
    physical = _resolve_physical_gpu_index(device_index)
    if not isinstance(physical, int):
        return None
    if probed_visible is None:
        return physical
    probed = [token.strip() for token in probed_visible.split(",") if token.strip()]
    return probed.index(str(physical)) if str(physical) in probed else None


def install_properties(cuda, probe=None):
    """Make cuda.get_device_properties(i) answer, while CUDA is not initialized here, with the
    values CUDA itself reported for device i in a short helper process."""
    orig = cuda.get_device_properties
    if getattr(orig, "_fastrsi_props", False):
        return {"installed": False, "why": "already installed"}
    probed_visible = os.environ.get("CUDA_VISIBLE_DEVICES")    # what the probe saw
    try:
        t = time.time()
        props = probe() if probe else json.loads(subprocess.run(
            [sys.executable, "-c", PROPS_CODE], capture_output=True, text=True, check=True,
            timeout=300).stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"installed": False, "why": repr(e)[:300]}

    def get_device_properties(device=None):
        if not cuda.is_initialized():
            i = _probed_index(_index(device), probed_visible)
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


def _parse_capability(text):
    """(major, minor) from FASTRSI_ZYGOTE_CC: "10.0", "100" or "sm_100" (an arch suffix such
    as "sm_90a" is dropped)."""
    value = text.strip().lower()
    value = value[3:] if value.startswith("sm_") else value
    value = value.rstrip("af")
    if value.count(".") == 1 and all(part.isdigit() for part in value.split(".")):
        major, minor = value.split(".")
    elif value.isdigit() and len(value) >= 2:
        major, minor = value[:-1], value[-1]
    else:
        raise ValueError(f"FASTRSI_ZYGOTE_CC={text!r}: expected a capability such as "
                         "10.0, 100 or sm_100")
    return int(major), int(minor)


def triton_capability(capability=None, nvml=None):
    """The (major, minor) Triton should target: `capability` if given, else FASTRSI_ZYGOTE_CC
    ("10.0", "100", "sm_100"), else NVML: the capability every visible GPU shares, else the
    first visible GPU's (found by UUID when CUDA_VISIBLE_DEVICES names GPUs so). Raises
    RuntimeError when CUDA_VISIBLE_DEVICES hides every GPU or NVML cannot answer."""
    if capability is not None:
        major, minor = capability
        return int(major), int(minor)
    cc = os.environ.get("FASTRSI_ZYGOTE_CC")
    if cc:
        return _parse_capability(cc)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and not vis.strip(", "):
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty: this process sees no GPU to "
                           "target; set FASTRSI_ZYGOTE_CC")
    nvml = nvml or _nvml()
    cap = nvml_capability(vis, nvml)
    if cap is not None:
        return tuple(cap)
    if nvml is None:
        raise RuntimeError("no NVML (pynvml) to read the GPU's capability; set FASTRSI_ZYGOTE_CC")
    # mixed GPUs, or GPUs named by UUID: the first visible one is CUDA device 0 in the child
    first = _resolve_physical_gpu_index(0)
    if first is None:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={vis!r} names no GPU; set FASTRSI_ZYGOTE_CC")
    nvml.nvmlInit()
    try:
        return tuple(nvml.nvmlDeviceGetCudaComputeCapability(_nvml_handle(nvml, first)))
    except Exception as exc:
        raise RuntimeError(f"NVML could not read the capability of GPU {first!r}: "
                           f"{type(exc).__name__}: {exc}; set FASTRSI_ZYGOTE_CC") from exc
    finally:
        nvml.nvmlShutdown()


def pin_triton_target(capability=None, nvml=None):
    """Answer Triton's "current GPU target" from `capability` (see triton_capability) instead of
    the CUDA driver, which a module-level query would initialize. Call it before the imports
    and the returned restore() after them. Without Triton nothing is patched and restore()
    does nothing. With CUDA_VISIBLE_DEVICES empty (every GPU hidden) and no capability or
    FASTRSI_ZYGOTE_CC given, nothing is patched either: the returned restore() has a
    `skipped` reason, also printed to stderr (triton_capability itself refuses that case)."""
    try:
        from triton.backends.compiler import GPUTarget
        from triton.backends.nvidia.driver import CudaDriver
    except Exception:
        return lambda: None
    if capability is None and not os.environ.get("FASTRSI_ZYGOTE_CC") and \
            os.environ.get("CUDA_VISIBLE_DEVICES") is not None and \
            not os.environ["CUDA_VISIBLE_DEVICES"].strip(", "):
        # every GPU hidden (a CPU-only process): no target to pin, and nothing to protect
        # (CUDA has no device to initialize); the process goes on unpinned
        def skipped():
            return None
        skipped.skipped = "CUDA_VISIBLE_DEVICES is empty: Triton's target is not pinned"
        print(f"forksafe: {skipped.skipped}", file=sys.stderr)
        return skipped
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


def driver_initialized(libcuda: str = LIBCUDA) -> bool:
    """Whether this process has initialized the CUDA driver (cuInit).

    `cuDeviceGetCount` does not initialize the driver and returns
    `CUDA_ERROR_NOT_INITIALIZED` (3) until something has called `cuInit`. Returns False
    when `libcuda` is absent on this host; raises if `libcuda` loads but lacks
    `cuDeviceGetCount` so callers can fail closed.
    """
    try:
        driver = ctypes.CDLL(libcuda)
    except OSError:
        return False
    count = ctypes.c_int()
    return driver.cuDeviceGetCount(ctypes.byref(count)) != 3


def cuda_initialized(libcuda: str = LIBCUDA) -> bool:
    """True when either torch's CUDA runtime or the CUDA driver is initialized."""
    torch_mod = sys.modules.get("torch")
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_initialized", None)) and cuda.is_initialized():
        return True
    return driver_initialized(libcuda)


def fork_poisoned() -> bool:
    """Would a child forked now be unable to use CUDA (torch bad-fork flag)?

    Forks a short probe child to read `torch._C._cuda_isInBadFork()` without affecting
    this process. Returns False when `os.fork` or torch is unavailable.
    """
    if not hasattr(os, "fork"):
        return False
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            torch_mod = sys.modules.get("torch")
            if torch_mod is None:
                import torch as torch_mod
            bad_fork = getattr(getattr(torch_mod, "_C", None), "_cuda_isInBadFork", None)
            bad = bool(bad_fork()) if callable(bad_fork) else False
        except Exception:
            bad = False
        try:
            os.write(write_fd, b"1" if bad else b"0")
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    try:
        out = os.read(read_fd, 1)
    finally:
        os.close(read_fd)
        os.waitpid(pid, 0)
    return out == b"1"


def poison_reasons(libcuda: str = LIBCUDA) -> list[str]:
    """Fail-closed list of reasons a child forked from this process cannot use CUDA.

    Empty list means no CUDA driver or torch runtime poisoning was detected.
    """
    reasons: list[str] = []
    try:
        if driver_initialized(libcuda):
            reasons.append("CUDA driver initialized in the zygote (cuInit)")
    except Exception as exc:
        reasons.append(f"CUDA driver state unreadable ({type(exc).__name__}: {exc})")
    try:
        if fork_poisoned():
            reasons.append("torch CUDA runtime initialized in the zygote (bad fork)")
    except Exception as exc:
        reasons.append(f"bad-fork probe failed ({type(exc).__name__}: {exc})")
    return reasons


def cute_dsl_arch(capability=None, nvml=None) -> str | None:
    """CuTe DSL arch tag (e.g. 'sm_100a', 'sm_90a', 'sm_80') without touching CUDA, or None."""
    try:
        major, minor = triton_capability(capability, nvml)
    except Exception:
        return None
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def pin_cute_dsl_arch(capability=None, nvml=None) -> str | None:
    """Set `CUTE_DSL_ARCH` when unset so CuTe DSL (flashinfer -> cutlass) does not call
    `cuInit` at import time to probe the GPU arch. Returns the active value or None."""
    existing = os.environ.get("CUTE_DSL_ARCH")
    if existing:
        return existing
    arch = cute_dsl_arch(capability, nvml)
    if arch:
        os.environ["CUTE_DSL_ARCH"] = arch
    return arch


def guard_cuda_python(capability=None, nvml=None) -> dict:
    """Stub `cuda.bindings.driver.cuInit` to answer `CUDA_ERROR_NOT_INITIALIZED` pre-fork
    and pin `CUTE_DSL_ARCH` if unset. Call `unguard_cuda_python()` in the forked child."""
    arch = pin_cute_dsl_arch(capability, nvml)
    try:
        import cuda.bindings.driver as drv
    except ImportError:
        return {"installed": False, "why": "no cuda-python", "cute_dsl_arch": arch}
    if "fn" not in _CUINIT:
        _CUINIT["fn"], _CUINIT["mod"] = drv.cuInit, drv
        drv.cuInit = lambda flags=0: (drv.CUresult.CUDA_ERROR_NOT_INITIALIZED,)
    return {"installed": True, "cute_dsl_arch": arch}


def unguard_cuda_python() -> bool:
    """Restore `cuda.bindings.driver.cuInit` in a forked child. True when restored."""
    if "fn" in _CUINIT:
        _CUINIT["mod"].cuInit = _CUINIT.pop("fn")
        _CUINIT.pop("mod")
        return True
    return False


def unpatch_child(cuda=None, clear_env: bool = False) -> dict:
    """In a forked worker child: restore `cuda-python` `cuInit` and all patched `torch.cuda`
    functions before real GPU work."""
    return {
        "cuda_python": unguard_cuda_python(),
        "torch_cuda": unpatch(cuda, clear_env=clear_env),
    }


def _resolve_guard_capability(capability):
    """Normalize `capability` for `install_all_guards` into `(cap_pair_or_None, cap_fn_or_None)`."""
    if capability is None:
        return None, None
    is_fn = callable(capability)
    val = capability() if is_fn else capability
    if val is None:
        return None, capability
    if isinstance(val, (tuple, list)) and len(val) == 2 and isinstance(val[0], (tuple, list)):
        pair, count = (int(val[0][0]), int(val[0][1])), int(val[1])
        return pair, (capability if is_fn else (lambda _p=pair, _c=count: (_p, _c)))
    pair = (int(val[0]), int(val[1]))
    return pair, (lambda _p=pair: (_p, 1))


def install_all_guards(
    cuda=None, *, capability=None, nvml=None, properties: bool = True, probe=None,
    triton: bool = True, cuda_python: bool = True, cute_dsl: bool = True, import_torch: bool = False,
):
    """Install all pre-fork CUDA guards in the safe order before heavy ML imports."""
    nvml_env_set = "PYTORCH_NVML_BASED_CUDA_CHECK" not in os.environ
    if nvml_env_set:
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    report = {"nvml_env_set": nvml_env_set}
    resolved_cap, cap_fn = _resolve_guard_capability(capability)
    if cute_dsl:
        report["cute_dsl_arch"] = pin_cute_dsl_arch(resolved_cap, nvml)
    if cuda_python:
        report["cuda_python"] = guard_cuda_python(resolved_cap, nvml)
    if cuda is None:
        if import_torch and "torch" not in sys.modules:
            with contextlib.suppress(ImportError):
                __import__("torch")
        cuda = getattr(sys.modules.get("torch"), "cuda", None)
    if cuda is not None:
        report["torch_cuda"] = install(cuda, capability=cap_fn)
        if properties:
            report["properties"] = install_properties(cuda, probe=probe)
    restore = lambda: None
    if triton:
        try:
            restore = pin_triton_target(capability=resolved_cap, nvml=nvml)
            report["triton"] = {"installed": not hasattr(restore, "skipped"),
                                **({"skipped": restore.skipped} if hasattr(restore, "skipped") else {})}
        except RuntimeError as exc:
            report["triton"] = {"installed": False, "why": str(exc)}
    return report, restore


def parse_cpulist(cpulist_text: str) -> list[int]:
    """Parse Linux sysfs cpulist format (e.g. '0-3,8,10-11') into a sorted list of CPU IDs."""
    cpus: set[int] = set()
    for token in (cpulist_text or "").strip().split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) == 2 and bounds[0].strip().isdigit() and bounds[1].strip().isdigit():
                start_cpu = int(bounds[0].strip())
                end_cpu = int(bounds[1].strip())
                if start_cpu <= end_cpu:
                    cpus.update(range(start_cpu, end_cpu + 1))
        elif part.isdigit():
            cpus.add(int(part))
    return sorted(cpus)


def _resolve_physical_gpu_index(device_index: int) -> int | str | None:
    """The physical GPU that CUDA device `device_index` of this process is: its NVML index,
    the UUID string ("GPU-...", "MIG-...") when CUDA_VISIBLE_DEVICES names GPUs so (Slurm
    does), or None when CUDA_VISIBLE_DEVICES does not list that many GPUs."""
    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_raw is None:
        return device_index
    tokens = [token.strip() for token in visible_raw.split(",") if token.strip()]
    if not 0 <= device_index < len(tokens):
        return None
    token = tokens[device_index]
    return int(token) if token.isdigit() else token


def _nvml_handle(nvml_module, physical):
    """NVML's handle for a physical GPU given as an index or a UUID string."""
    if isinstance(physical, str):
        return nvml_module.nvmlDeviceGetHandleByUUID(physical)
    return nvml_module.nvmlDeviceGetHandleByIndex(physical)


def _read_sysfs_pci_node(device_dir: str, source_label: str) -> dict | None:
    cpus = parse_cpulist(_read_text(os.path.join(device_dir, "local_cpulist")))
    raw_numa = _read_text(os.path.join(device_dir, "numa_node")).strip()
    numa_node = (
        int(raw_numa)
        if (raw_numa.isdigit() or (raw_numa[:1] in "+-" and raw_numa[1:].isdigit())) and int(raw_numa) >= 0
        else None
    )
    if not cpus and numa_node is None:
        return None
    return {"cpus": cpus, "numa_node": numa_node, "source": source_label}


def detect_gpu_cpu_affinity(device_index: int, sysfs_root: str = "/sys/bus/pci/devices") -> dict:
    """Detect the CPU affinity list and NUMA node for `device_index` without initializing CUDA."""
    if device_index < 0 or (physical_index := _resolve_physical_gpu_index(device_index)) is None:
        return dict(_NONE_AFFINITY)
    if sysfs_root == "/sys/bus/pci/devices" and (nvml_module := _nvml()) is not None:
        with contextlib.suppress(Exception):
            nvml_module.nvmlInit()
            try:
                handle, sysfs_info = _nvml_handle(nvml_module, physical_index), None
                if hasattr(nvml_module, "nvmlDeviceGetPciInfo") and os.path.isdir(sysfs_root):
                    raw_bus = getattr(nvml_module.nvmlDeviceGetPciInfo(handle), "busId", b"")
                    bus_text = (raw_bus.decode("utf-8", errors="ignore") if isinstance(raw_bus, bytes) else str(raw_bus)).rstrip("\x00").strip().lower()
                    for bus_id in (bus_text, *([bus_text[4:]] if bus_text.startswith("00000000:") else [])):
                        cand = os.path.join(sysfs_root, bus_id)
                        if os.path.isdir(cand):
                            sysfs_info = _read_sysfs_pci_node(cand, "nvml_pci_sysfs")
                            if sysfs_info and sysfs_info["cpus"]:
                                return sysfs_info
                if hasattr(nvml_module, "nvmlDeviceGetCpuAffinity"):
                    mask_words = max(1, ((os.cpu_count() or 64) + 63) // 64)
                    masks = nvml_module.nvmlDeviceGetCpuAffinity(handle, mask_words)
                    nvml_cpus = sorted({w * 64 + b for w, m in enumerate(masks) for b in range(64) if int(m) & (1 << b)})
                    if nvml_cpus:
                        return {"cpus": nvml_cpus, "numa_node": sysfs_info["numa_node"] if sysfs_info else None, "source": "nvml"}
            finally:
                with contextlib.suppress(Exception):
                    nvml_module.nvmlShutdown()
    if isinstance(physical_index, str):  # a UUID: only NVML maps it to a PCI device
        return dict(_NONE_AFFINITY)
    if sysfs_root and os.path.isdir(sysfs_root):
        all_candidates, nvidia_candidates = [], []
        try:
            entry_names = sorted(os.listdir(sysfs_root))
        except OSError:
            entry_names = []
        for entry_name in entry_names:
            entry_dir = os.path.join(sysfs_root, entry_name)
            if not (os.path.isdir(entry_dir) and (os.path.isfile(os.path.join(entry_dir, "local_cpulist"))
                                                  or os.path.isfile(os.path.join(entry_dir, "numa_node")))):
                continue
            all_candidates.append(entry_dir)
            vendor_val = _read_text(os.path.join(entry_dir, "vendor")).strip().lower()
            class_val = _read_text(os.path.join(entry_dir, "class")).strip().lower()
            if vendor_val == "0x10de" and (not class_val or class_val.startswith("0x03")):
                nvidia_candidates.append(entry_dir)
        selected_dirs = nvidia_candidates or all_candidates
        target_index = physical_index if 0 <= physical_index < len(selected_dirs) else (
            device_index if 0 <= device_index < len(selected_dirs) else -1
        )
        if target_index >= 0 and (res := _read_sysfs_pci_node(selected_dirs[target_index], "sysfs")) is not None:
            return res
    return dict(_NONE_AFFINITY)


def bind_rank_numa_and_cpus(
    local_rank: int,
    explicit_cpus: list[int] | None = None,
    sysfs_root: str = "/sys/bus/pci/devices",
) -> dict:
    """Bind the calling process to `explicit_cpus` or to the CPUs local to `local_rank`'s GPU,
    within the CPUs it may run on. As Linux does, a set partly outside the process's cpuset
    binds to the part inside it (this can widen a narrower pin the process inherited); a set
    wholly outside it (EINVAL) falls back to the CPUs of the current mask among them. Returns {bound, bound_cpus (what it
    is bound to now; empty when not bound), numa_node, source, why (None when bound)}."""
    if explicit_cpus is not None:
        target_cpus = sorted(set(int(cpu_id) for cpu_id in explicit_cpus))
        numa_node = None
        source = "explicit"
    else:
        affinity = detect_gpu_cpu_affinity(local_rank, sysfs_root=sysfs_root)
        target_cpus = list(affinity["cpus"])
        numa_node = affinity["numa_node"]
        source = affinity["source"]
    bound, why = False, None
    if not target_cpus:
        why = "no CPUs found for this GPU"
    elif not hasattr(os, "sched_setaffinity"):
        why = "this platform has no sched_setaffinity"
    else:
        try:
            os.sched_setaffinity(0, target_cpus)
            bound = True
        except OSError as exc:
            # a Slurm or container cpuset excludes every one of them (EINVAL): try the
            # GPU's CPUs within the current mask
            usable = sorted(set(target_cpus) & os.sched_getaffinity(0))
            why = f"sched_setaffinity({target_cpus[:16]}): {exc}"
            if usable:
                try:
                    os.sched_setaffinity(0, usable)
                    bound, why = True, None
                except OSError as retry_exc:
                    why += f"; within the current mask: {retry_exc}"
        if bound:                        # what the kernel bound (the cpuset may cut it)
            target_cpus = sorted(os.sched_getaffinity(0))
    return {"bound": bound, "bound_cpus": target_cpus if bound else [], "numa_node": numa_node,
            "source": source, "why": why}


def clamp_thread_pools(num_threads: int = 1, clamp_compile_threads: bool = False) -> dict:
    """Clamp OpenMP, OpenBLAS, MKL, and active torch intra-op thread pools to `num_threads`."""
    thread_count = max(1, int(num_threads))
    thread_str = str(thread_count)
    env_updates = {
        "OMP_NUM_THREADS": thread_str,
        "OPENBLAS_NUM_THREADS": thread_str,
        "MKL_NUM_THREADS": thread_str,
    }
    if clamp_compile_threads:
        env_updates["TORCHINDUCTOR_COMPILE_THREADS"] = thread_str
    os.environ.update(env_updates)
    torch_module = sys.modules.get("torch")
    torch_threads_set = False
    if torch_module is not None and hasattr(torch_module, "set_num_threads"):
        try:
            torch_module.set_num_threads(thread_count)
            torch_threads_set = True
        except Exception:
            pass
    return {
        "num_threads": thread_count,
        "env": env_updates,
        "torch_threads_set": torch_threads_set,
    }


def apply_child_runtime_affinity(env: dict[str, str] | None = None) -> dict:
    """Clamp default thread pools and bind GPU NUMA/CPUs for a freshly forked zygote child."""
    src = env or {}
    if not src.get("OMP_NUM_THREADS") and not os.environ.get("OMP_NUM_THREADS"):
        clamp_thread_pools(num_threads=1)
    rank_str = (src.get("LOCAL_RANK") or os.environ.get("LOCAL_RANK") or "").strip()
    vis_raw = src.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    visible = [
        tok for raw in vis_raw.split(",")
        if (tok := raw.strip()) and tok.lower() not in ("nodevfiles", "-1")
    ]
    if rank_str.isdigit():
        return bind_rank_numa_and_cpus(int(rank_str))
    if len(visible) == 1:
        return bind_rank_numa_and_cpus(0)
    return {"bound_cpus": [], "numa_node": None}


def freeze_gc_before_fork() -> dict:
    """Run a full generation-2 collection and freeze the heap before forking worker children."""
    collected = gc.collect(2)
    if hasattr(gc, "freeze"):
        gc.freeze()
    frozen_count = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else 0
    return {"collected": collected, "frozen_count": frozen_count}


def _build_finding(check_name: str, observed: str, expected: str) -> dict:
    try:
        from kernel import findings as kernel_findings
    except ImportError:
        from .kernel import findings as kernel_findings
    return kernel_findings.finding(check_name, observed, expected)


def audit_system_environment(
    local_rank: int = 0,
    local_world_size: int = 1,
    bound_cpus: list[int] | None = None,
    gc_frozen: bool = False,
    sysfs_root: str = "/sys/bus/pci/devices",
    thp_enabled_path: str = "/sys/kernel/mm/transparent_hugepage/enabled",
    imex_path: str = "/dev/nvidia-caps-imex-channels",
    require_imex: bool = False,
) -> list[dict]:
    """Non-blocking diagnostic returning findings for NUMA, threads, THP/GC, and IMEX."""
    findings_list: list[dict] = []
    if bound_cpus:
        affinity = detect_gpu_cpu_affinity(local_rank, sysfs_root=sysfs_root)
        gpu_cpus = affinity["cpus"]
        if gpu_cpus and set(bound_cpus).isdisjoint(gpu_cpus):
            findings_list.append(_build_finding(
                "numa_mismatch",
                f"bound_cpus={list(bound_cpus)} disjoint from gpu_cpus={gpu_cpus}",
                f"subset of gpu_cpus={gpu_cpus}",
            ))
    omp_raw = os.environ.get("OMP_NUM_THREADS", "1").strip()
    omp_threads = int(omp_raw) if omp_raw.isdigit() else 1
    cpu_total = os.cpu_count() or 1
    world_size = max(1, int(local_world_size))
    total_threads = omp_threads * world_size
    if total_threads > cpu_total:
        findings_list.append(_build_finding(
            "thread_oversubscription",
            f"{total_threads} threads ({omp_threads} OMP_NUM_THREADS * {world_size} ranks)",
            f"<= {cpu_total} logical CPUs",
        ))
    if not gc_frozen and thp_enabled_path and "[always]" in _read_text(thp_enabled_path):
        findings_list.append(_build_finding(
            "thp_unfrozen_gc",
            "transparent_hugepage/enabled has [always] with gc_frozen=False",
            "gc.freeze() called before fork or transparent_hugepage set to madvise",
        ))
    if require_imex and not os.path.exists(imex_path):
        findings_list.append(_build_finding(
            "imex_channel_missing",
            f"missing {imex_path}",
            f"existing IMEX channel path {imex_path}",
        ))
    return findings_list
