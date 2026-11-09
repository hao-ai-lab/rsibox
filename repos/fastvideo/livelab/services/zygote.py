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
(`forksafe.pin_triton_target`, FASTRSI_ZYGOTE_CC="10.0" to set it). `fork_poisoned()` checks
the result with a probe fork.

Not refreshed (a finding in the worker log; restart the lab host): an edited extension `.so`,
and modules outside `PREFIXES`.
"""
import importlib
import multiprocessing
import os
import sys
import time

# the framework's pre-fork guards (core/livelab/forksafe.py; FASTRSI_LIVELAB overrides the
# folder): the forkserver imports this module first, before the adapter package
_LIVELAB = os.path.abspath(os.environ.get("FASTRSI_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if _LIVELAB not in sys.path:
    sys.path.insert(0, _LIVELAB)
import forksafe  # noqa: E402

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
            # the edit reached the package root (FastVideo's __init__ imports the entry points,
            # which import the loader): a partial re-import would register FastVideo's presets
            # again into the registry module it kept ("Duplicate preset registration"). Drop the
            # whole package, the state a fresh process starts from (~15 s of imports).
            dropped |= {n for n in modules if n.split(".")[0] in prefixes}
        dropped = sorted(dropped)
        for n in stale:
            f = now[n][0]
            try:
                os.unlink(importlib.util.cache_from_source(f))
            except OSError:
                pass
        for n in dropped:
            modules.pop(n, None)
            RB.unbind(modules, n)
        importlib.invalidate_caches()
    return {"stale": stale, "dropped": dropped, "blocked": blocked,
            "seconds": round(time.perf_counter() - t0, 4)}


def _install_worker_hook():
    """Runs in the server: each forked child refreshes before it unpickles its process object.

    The child's process object (target and kwargs: FastVideo's args, pipes) is unpickled by
    `multiprocessing.spawn._main` right after `prepare()`. Refreshing inside the target would
    be too late: the kwargs would already be instances of the old classes, which the
    re-imported code does not recognize. So the hook wraps `spawn.prepare`: refresh, then (with
    a holder) hook the loader the child will use, then unpickle."""
    import multiprocessing.spawn as SP
    orig = SP.prepare

    def prepare(data):
        rep = refresh_stale()
        if rep["stale"] or rep["blocked"]:
            print(f"[fastrsi-zygote] pid {os.getpid()} refreshed stale={rep['stale']} "
                  f"dropped={len(rep['dropped'])} blocked={rep['blocked']} in {rep['seconds']} s",
                  file=sys.stderr, flush=True)
        if os.environ.get("FASTRSI_HOLDER_SOCK"):
            from services import holder
            holder.install_loader_hook(os.environ["FASTRSI_HOLDER_SOCK"])
        return orig(data)
    SP.prepare = prepare


def fork_poisoned():
    """Would a child forked now be unable to use CUDA? torch marks a child "in bad fork" when the
    parent touched the CUDA runtime at all, also through `torch.cuda.is_available()` or
    `device_count()` unless PYTORCH_NVML_BASED_CUDA_CHECK=1. Forks a probe child to ask."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            import torch
            bad = bool(torch._C._cuda_isInBadFork()) if hasattr(torch._C, "_cuda_isInBadFork") else False
        except Exception:
            bad = False
        os.write(w, b"1" if bad else b"0")
        os._exit(0)
    os.close(w)
    out = os.read(r, 1)
    os.close(r)
    os.waitpid(pid, 0)
    return out == b"1"


def _serve():
    """In the forkserver: import the preload list without CUDA, snapshot, hook the workers."""
    t0 = time.perf_counter()
    restore = forksafe.pin_triton_target()
    try:
        for m in filter(None, os.environ.get("FASTRSI_ZYGOTE_PRELOAD", "").split(",")):
            importlib.import_module(m)
    finally:
        restore()
    if fork_poisoned():
        print("[fastrsi-zygote] CUDA was initialized while preloading: forked workers cannot use it",
              file=sys.stderr, flush=True)
    snapshot()
    _install_worker_hook()
    print(f"[fastrsi-zygote] server {os.getpid()} preloaded {len(_SNAPSHOT)} modules in "
          f"{time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)


def enable(preload=None):
    """Make FastVideo fork its workers from a preloaded forkserver (call in the lab host before
    building the first generator). Returns the preload list."""
    preload = [m for m in (preload or PRELOAD) if m != "services.zygote"]
    preload += [m for m in os.environ.get("FASTRSI_ZYGOTE_EXTRA", "").split(",") if m and m not in preload]
    os.environ["FASTVIDEO_WORKER_MULTIPROC_METHOD"] = "forkserver"
    # FastVideo calls torch.cuda.is_available() at import; through the CUDA runtime that poisons
    # every later fork, through NVML it does not
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    os.environ["FASTRSI_ZYGOTE_PRELOAD"] = ",".join(preload)
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
if in_forkserver() and os.environ.get("FASTRSI_ZYGOTE_PRELOAD"):
    try:
        _serve()
    except Exception as e:                       # a broken zygote must not stop the server
        print(f"[fastrsi-zygote] not serving: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
