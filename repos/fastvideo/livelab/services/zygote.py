"""Zygote for FastVideo workers: multiprocessing's forkserver, preloaded, with module refresh.

FastVideo starts each worker with `multiprocessing` (`FASTVIDEO_WORKER_MULTIPROC_METHOD`; it
refuses `fork` and uses `spawn`). A spawned worker imports torch and FastVideo again: 16-18 s
of a 31 s load on one GB200 (kb/costs.md). The forkserver start method is a zygote that
Python already has: one server process imports the preload list once, without CUDA, and forks
every worker from it. A worker group respawned by the lab host then starts from warm imports.

A module edited after the server imported it would be stale in every forked worker. As in the
vLLM lab's zygote refresh, the forked child drops the edited modules and every module that
imports them from `sys.modules` before it builds anything, so the worker imports them again
from the new source. The server snapshots the size and mtime of every preloaded source file;
the child compares.

    enable()            # in the lab host, before the first VideoGenerator is built
    ...                 # every later worker group is forked from the preloaded server

The server must not touch the CUDA runtime (a forked child of such a process cannot use CUDA).
At the pin two imports do: FastVideo calls `torch.cuda.is_available()` (the runtime, unless
`PYTORCH_NVML_BASED_CUDA_CHECK=1`, which `enable` sets), and `fastvideo_kernel` asks Triton for
the GPU target at module level, which the server answers from NVML while it imports
(`forksafe.pin_triton_target`, RSIBOX_ZYGOTE_CC="10.0" to set it). CuTe DSL (flashinfer ->
cutlass) calls the driver's cuInit itself to find the arch unless CUTE_DSL_ARCH is set, which
the server sets from the same capability (`pin_cute_dsl_arch`). After the preload the server
checks both ways a fork is spoiled (`poison_reasons`: the driver's cuDeviceGetCount, torch's
bad-fork flag in a probe fork), also when the preload raised; when one is, or a probe cannot
answer on a host with libcuda, every forked worker refuses to start.

Not refreshed (a finding in the worker log; restart the lab host): an edited extension `.so`,
and modules outside `PREFIXES`.
"""
import importlib
import multiprocessing
import os
import sys
import time

import forksafe
from rsibox import hooks

PREFIXES = ("fastvideo",)
PRELOAD = ["torch", "fastvideo", "fastvideo.worker.multiproc_executor", "fastvideo.worker.gpu_worker",
           "fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline", "fastvideo.models.dits.minimax_h3"]
_SNAPSHOT = {}


def _sources(prefixes=PREFIXES, modules=None):
    modules = sys.modules if modules is None else modules
    out = {}
    for name, m in list(modules.items()):
        f = getattr(m, "__file__", None)
        if f and name.split(".")[0] in prefixes:
            try:
                st = os.stat(f)
            except OSError:
                continue
            out[name] = (f, st.st_size, st.st_mtime_ns)
    return out


def snapshot(prefixes=PREFIXES):
    _SNAPSHOT.clear()
    _SNAPSHOT.update(_sources(prefixes))
    return len(_SNAPSHOT)


def refresh_stale(prefixes=PREFIXES, modules=None):
    """In a forked child, before it builds anything: drop modules whose source changed since
    the snapshot, and their importers. Returns {"stale", "dropped", "blocked", "seconds"}."""
    from services import rebuild as RB
    t0 = time.perf_counter()
    modules = sys.modules if modules is None else modules
    now = _sources(prefixes, modules)
    stale = sorted(n for n, v in _SNAPSHOT.items() if now.get(n, v) != v)
    blocked = [n for n in stale if not now[n][0].endswith(".py")]
    stale = [n for n in stale if n not in blocked]
    dropped = []
    if stale:
        dropped = RB.dependents(set(stale), RB.import_graph(prefixes, modules))
        if dropped & set(prefixes):
            # the edit reached the package root: drop the whole package so registries start clean
            dropped |= {n for n in modules if n.split(".")[0] in prefixes}
        dropped = sorted(dropped)
        stale_set = set(stale)
        hooks.purge_modules(stale, modules, unlink_pyc=True)
        rest = [n for n in dropped if n not in stale_set]
        if rest:
            hooks.purge_modules(rest, modules, unlink_pyc=False)
    return {"stale": stale, "dropped": dropped, "blocked": blocked,
            "seconds": round(time.perf_counter() - t0, 4)}


def _install_worker_hook(poison=None, clear_nvml_env=False):
    """Runs in the server: each forked child refreshes before it unpickles its process object."""
    import multiprocessing.spawn as SP
    orig = SP.prepare

    def prepare(data):
        if poison:
            raise RuntimeError(f"[rsibox-zygote] forked from a server whose forks cannot use CUDA: "
                               f"{poison}; restart the lab host")
        forksafe.unpatch_child(None, clear_env=clear_nvml_env)
        forksafe.apply_child_runtime_affinity()
        rep = refresh_stale()
        if rep["stale"] or rep["blocked"]:
            print(f"[rsibox-zygote] pid {os.getpid()} refreshed stale={rep['stale']} "
                  f"dropped={len(rep['dropped'])} blocked={rep['blocked']} in {rep['seconds']} s",
                  file=sys.stderr, flush=True)
        if os.environ.get("RSIBOX_HOLDER_SOCK"):
            from services import holder
            holder.install_loader_hook(os.environ["RSIBOX_HOLDER_SOCK"])
        return orig(data)
    prepare._rsibox = True
    SP.prepare = prepare


fork_poisoned = forksafe.fork_poisoned
LIBCUDA = forksafe.LIBCUDA


def driver_initialized():
    """Has this process initialized the CUDA driver (cuInit)? Delegates to `forksafe` using
    module-level `LIBCUDA` so unit tests can inject a stub library path."""
    return forksafe.driver_initialized(libcuda=LIBCUDA)


def poison_reasons():
    """Why a child forked now could not use CUDA (empty: it can). Calls module-level
    `driver_initialized` / `fork_poisoned` so callers and tests stay fail-closed."""
    out = []
    try:
        if driver_initialized():
            out.append("CUDA driver initialized in the zygote (cuInit)")
    except Exception as e:
        out.append(f"CUDA driver state unreadable ({type(e).__name__}: {e})")
    try:
        if fork_poisoned():
            out.append("torch CUDA runtime initialized in the zygote (bad fork)")
    except Exception as e:
        out.append(f"bad-fork probe failed ({type(e).__name__}: {e})")
    return out


pin_cute_dsl_arch = forksafe.pin_cute_dsl_arch


def _preset_torch_key():
    """Pre-populate Inductor's `torch_key()` in the zygote so forked workers inherit it in RAM
    without touching CUDA."""
    if "torch" not in sys.modules or os.environ.get("RSIBOX_ZYGOTE_PRESET_TORCH_KEY", "1") == "0":
        return None
    if os.environ.get("RSIBOX_TORCH_KEY_CACHE"):
        from services import jitcache
        return jitcache.preset_torch_key()
    t0 = time.perf_counter()
    try:
        from torch._inductor import codecache
        codecache.torch_key()
        return {"hit": False, "in_memory": True, "seconds": round(time.perf_counter() - t0, 4)}
    except Exception as e:
        return {"hit": False, "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.perf_counter() - t0, 4)}


OPTIONAL_PRELOAD = (
    "flashinfer",
    "flashinfer.fp4_quantization",
    "flashinfer.gemm",
    "fastvideo.models.vaes.minimax_h3_nvfp4_decoder",
)


def _optional_preload_modules(default_enabled=True):
    raw = os.environ.get("RSIBOX_ZYGOTE_OPTIONAL_PRELOAD")
    if raw is None:
        return list(OPTIONAL_PRELOAD) if default_enabled else []
    token = raw.strip().lower()
    if token in ("1", "true", "yes"):
        return list(OPTIONAL_PRELOAD)
    if token in ("", "0", "false", "no", "off"):
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _serve():
    """In the forkserver: import the preload list without CUDA, snapshot, hook the workers.
    The poison check and the worker hook run however the preload ends (a preload that called
    cuInit and then raised poisons the forks as much as one that imported)."""
    t0 = time.perf_counter()
    error = None
    optional_preloaded = []
    clear_nvml_env = False
    preload_modules = [m for m in os.environ.get("RSIBOX_ZYGOTE_PRELOAD", "").split(",") if m]
    loaded_cuda = getattr(sys.modules.get("torch"), "cuda", None)
    has_torch_cuda = (
        loaded_cuda is not None
        and hasattr(loaded_cuda, "get_device_properties")
        and hasattr(loaded_cuda, "get_device_capability")
    )
    # Default OPTIONAL_PRELOAD drags in flashinfer -> torch -> cuInit; only enable it
    # automatically when torch is part of the preload or already loaded with CUDA.
    optional_modules = _optional_preload_modules(
        default_enabled=("torch" in preload_modules or has_torch_cuda)
    )
    use_all_guards = (
        (loaded_cuda is None and ("torch" in preload_modules or bool(optional_modules)))
        or has_torch_cuda
    )
    try:
        pin_cute_dsl_arch()
        if use_all_guards:
            guard_report, restore = forksafe.install_all_guards(
                properties=True, import_torch=(loaded_cuda is None)
            )
            clear_nvml_env = bool((guard_report or {}).get("nvml_env_set"))
        else:
            restore = forksafe.pin_triton_target()
        try:
            for m in preload_modules:
                importlib.import_module(m)
            for m in optional_modules:
                try:
                    importlib.import_module(m)
                    optional_preloaded.append(m)
                except Exception as opt_exc:
                    print(f"[rsibox-zygote] optional preload skipped {m}: "
                          f"{type(opt_exc).__name__}: {opt_exc}",
                          file=sys.stderr, flush=True)
        finally:
            restore()
    except Exception as e:          # workers still start (cold imports) unless poisoned below
        error = f"{type(e).__name__}: {e}"
        print(f"[rsibox-zygote] preload failed: {error}", file=sys.stderr, flush=True)
    torch_key = _preset_torch_key()
    poison = poison_reasons()
    if poison:
        print(f"[rsibox-zygote] {poison}: forked workers cannot use CUDA; they refuse to start",
              file=sys.stderr, flush=True)
    try:
        snapshot()
    finally:
        _install_worker_hook(poison, clear_nvml_env=clear_nvml_env)
    if os.environ.get("RSIBOX_ZYGOTE_GC_FREEZE", "1") == "1":
        forksafe.freeze_gc_before_fork()
    print(f"[rsibox-zygote] server {os.getpid()} preloaded {len(_SNAPSHOT)} modules in "
          f"{time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)
    return {"poison": poison, "preload_error": error, "torch_key": torch_key,
            "optional_preloaded": optional_preloaded}


def enable(preload=None):
    """Make FastVideo fork its workers from a preloaded forkserver (call in the lab host before
    building the first generator). Returns the preload list."""
    preload = [m for m in (preload or PRELOAD) if m != "services.zygote"]
    preload += [m for m in os.environ.get("RSIBOX_ZYGOTE_EXTRA", "").split(",") if m and m not in preload]
    os.environ["FASTVIDEO_WORKER_MULTIPROC_METHOD"] = "forkserver"
    # FastVideo calls torch.cuda.is_available() at import; through the CUDA runtime that poisons
    # every later fork, through NVML it does not
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    os.environ["RSIBOX_ZYGOTE_PRELOAD"] = ",".join(preload)
    try:
        from fastvideo import envs
        envs.FASTVIDEO_WORKER_MULTIPROC_METHOD.set("forkserver")
    except Exception:
        pass
    multiprocessing.set_forkserver_preload(["services.zygote"])
    return preload


def in_forkserver():
    """Is this process multiprocessing's forkserver (started as `python -c ...forkserver...`)?"""
    return bool(sys.argv) and sys.argv[0] == "-c" and "multiprocessing.forkserver" in sys.modules


# the forkserver imports this module as its preload: it imports the rest itself (above)
if in_forkserver() and os.environ.get("RSIBOX_ZYGOTE_PRELOAD"):
    try:
        _serve()
    except Exception as e:                       # a broken zygote must not stop the server
        print(f"[rsibox-zygote] not serving: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        import multiprocessing.spawn as _spawn
        if not getattr(_spawn.prepare, "_rsibox", False):   # the poison check runs regardless
            _install_worker_hook(poison_reasons())
