"""Zygote: one process per node that has torch, Transformer Engine and Megatron imported, without
CUDA, and starts the ranks of a training job by fork instead of torchrun + exec. A restart then
skips the Python imports (16-85 s from a network file system in our runs) and pays only
Megatron's own setup and the first step.

    python -m rsibox_megatron.zygote serve --sock S [--preload M1,M2] [--script pretrain_gpt.py]
    python -m rsibox_megatron.zygote start --sock S --nproc N [--port P] [--log-dir D]
        [--env K=V ...] [--accept stale,env] [--refresh megatron] [--timeout SEC] -- pretrain_gpt.py <megatron args>
    python -m rsibox_megatron.zygote stop --sock S

`start` sends one request and waits until every rank has exited; it prints one JSON line per
event (`forked`, then `exited` with each rank's exit code) and returns the job's exit code.
Each rank runs `rsibox_megatron.run` (phase stamps, lab hook) exactly as under torchrun, with
RSIBOX_T0 set to the request time, so phases.rank<r>.json counts from the request.

Contract:
- The zygote never initializes CUDA or the CUDA driver. If an import does, `serve` exits and
  names the module: no forked rank could use the GPU.
- A rank gets the variables torchrun sets (RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE,
  GROUP_RANK, MASTER_ADDR, MASTER_PORT, TORCHELASTIC_RUN_ID, and OMP_NUM_THREADS=1 for more than
  one rank unless set) plus the request's `env`, its own session (setsid) and its stdout/stderr
  in <log-dir>/rank<r>.log; with `cpus` (one CPU list per local rank) it is pinned to its list
  right after the fork (a fork otherwise keeps the zygote's affinity). OMP_NUM_THREADS is applied to torch's thread pool in the rank
  (torch.set_num_threads), since torch read the variable when the zygote imported it.
- When a rank exits non-zero, the zygote terminates the other ranks' process trees by PID and
  their process groups (each rank leads its own), as torchrun does, never a group it shares.
- Supervision (no rank or child of a rank outlives its job): the zygote is the child subreaper of
  everything it forks and reaps all of it before it serves the next request. Each rank has
  PR_SET_PDEATHSIG=SIGKILL, so it dies with the zygote, and a guard process (forked from the rank
  before CUDA, named rsibox-guard) that kills the rank's process group when the rank dies,
  however it dies. A client that disconnects while its job runs stops the job (SIGTERM, then
  SIGKILL after 10 s), as torchrun's workers stop with torchrun; `on_client_gone: "keep"` lets it
  run to its end instead. A malformed request (not JSON, not an object, missing fields) gets an
  `error` event; a send to a gone client is logged on stderr, never fatal.
- Findings (not gates; a request lists the ones it accepts in `accept`, otherwise no fork):
  "stale": a source file of a module the zygote imported changed since (the ranks would run
  the old code; restart the zygote, or edit through the live lab); "env": a request variable
  with a prefix that a preloaded library may read at import (NVTE_, TORCH_, TORCHINDUCTOR_,
  TRITON_, LD_, PYTHON...) differs from the zygote's start value. NCCL_*, TORCH_NCCL_*,
  CUDA_VISIBLE_DEVICES and the rank variables are read after the fork and never count;
  "overlay": the request's RSIBOX_OVERLAY replaces modules the zygote has imported already
  (overlay.py works only before the first import), so the edit would not run.
- Code restart: `start --refresh megatron` makes each forked rank drop the named packages, the
  script's own modules and our modules that hold references to them (run, hook, gap, state,
  client) from sys.modules, install the overlay finder, and import them again, so an overlaid
  edit runs while torch, Transformer Engine and the rest stay loaded. Stale and overlay findings
  for refreshed modules do not count. Cost (our 2026-11-07 run, Qwen3-14B architecture, TP 2): the
  re-import took 1.4 s, and the lab was ready 40.9 s after the request, against 78.8 s for torchrun.
"""
import argparse, ast, contextlib, ctypes, gc, importlib, json, os, select, signal, socket, sys, time, traceback

# the framework's pre-fork CUDA guards (core/livelab/forksafe.py, on sys.path through the
# package __init__); PROPS_CODE and Properties are re-exported for callers of this module
import forksafe
from forksafe import PROPS_CODE, Properties, freeze_gc_before_fork  # noqa: F401
from rsibox import hooks, ipc, proc

START_ENV = dict(os.environ)
# Variables a preloaded library may read at import (in the zygote, before the fork) ...
EARLY_ENV_PREFIXES = ("NVTE_", "PYTORCH_", "TORCH_", "TORCHINDUCTOR_", "TRITON_", "LD_",
                      "PYTHON", "MEGATRON_")
# ... except these, read at process-group or communicator creation, or applied in set_threads() after the fork.
LATE_ENV_PREFIXES = ("TORCH_NCCL_", "TORCH_DISTRIBUTED_", "TORCHINDUCTOR_COMPILE_THREADS")
# torch._dynamo and torch._inductor are imported lazily at the first torch.compile, in Megatron's
# kernel warm-up: preloading them takes 1.5-2 s off each restart. During preload and warmup,
# TorchInductor uses a multi-threaded compile pool (default_compile_threads) for fast compilation,
# then quiesce_compile_pools_after_warmup() shuts down the compile pools and clamps ATen to 1
# thread before steady-state GPU training steps.
DEFAULT_PRELOAD = ["torch", "megatron.training.training", "torch._dynamo",
                   "torch._inductor.compile_fx", "torch._inductor.codecache"]
PRELOAD_COMPILE_THREADS = 8


PR_SET_PDEATHSIG, PR_SET_NAME, PR_SET_CHILD_SUBREAPER = 1, 15, 36
CLIENT_GONE_GRACE_S = 10.0


def _prctl(option, arg):
    """prctl(2) on Linux; False elsewhere (or when libc is not found)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if isinstance(arg, bytes):
            return libc.prctl(option, ctypes.c_char_p(arg), 0, 0, 0) == 0
        return libc.prctl(option, ctypes.c_ulong(arg), 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def log(event, **kw):
    """One JSON line on stderr: supervision events that no client may be left to receive."""
    try:
        print(json.dumps({"event": event, "t": round(time.time(), 3), **kw}, default=str),
              file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def install_forksafe():
    """Keep the imports from initializing CUDA via `forksafe.install_all_guards`."""
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    try:
        import torch
    except ImportError as e:
        return {"installed": False, "why": repr(e)}
    report, _ = forksafe.install_all_guards(torch.cuda, triton=False, cute_dsl=True, cuda_python=True)
    if os.environ.get("CUTE_DSL_ARCH"):
        START_ENV.setdefault("CUTE_DSL_ARCH", os.environ["CUTE_DSL_ARCH"])
    out = dict(report.get("torch_cuda", {"installed": False}))
    out["properties"] = report.get("properties", {})
    out["cuda_python"] = report.get("cuda_python", {})
    return out


def guard_cuda_python(cuda=None):
    """Stub `cuda.bindings.driver.cuInit` and pin `CUTE_DSL_ARCH` via `forksafe.guard_cuda_python`."""
    cap = None
    if cuda is not None and "CUTE_DSL_ARCH" not in os.environ:
        try:
            cap = cuda.get_device_capability(0)
        except Exception:
            cap = None
    out = forksafe.guard_cuda_python(capability=cap)
    if os.environ.get("CUTE_DSL_ARCH"):
        START_ENV.setdefault("CUTE_DSL_ARCH", os.environ["CUTE_DSL_ARCH"])
    return out


unguard_cuda_python = forksafe.unguard_cuda_python
install_properties = forksafe.install_properties
driver_initialized = forksafe.driver_initialized
cuda_initialized = forksafe.cuda_initialized


def script_imports(script):
    """Top-level modules a script imports (its `import` statements at module level and inside
    `if` blocks), in source order; relative imports are skipped."""
    with open(script) as f:
        tree = ast.parse(f.read(), script)
    out, todo = [], list(tree.body)
    while todo:
        n = todo.pop(0)
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out.append(n.module)
        elif isinstance(n, (ast.If, ast.Try)):
            todo[:0] = n.body + getattr(n, "orelse", []) + [
                s for h in getattr(n, "handlers", []) for s in h.body]
    return list(dict.fromkeys(out))


def sources():
    """(mtime, size) of each imported module's source; the zygote's own file (`__main__`) is
    left out: the ranks run the code of their own modules, not this one's."""
    out = {}
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    for n, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if f == main:                           # also under another name (__mp_main__)
            continue
        if f and f.endswith(".py") and f not in out:
            try:
                st = os.stat(f)
                out[f] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass
    return out


def preload(mods, script=None, check=cuda_initialized):
    """Import `mods` and the script's imports; returns {module: seconds} or raises SystemExit
    naming the first module whose import initialized CUDA. Resets ATen threads to 1 and
    quiesces any Inductor compile pools after preloading so forked ranks start clean."""
    if script:
        sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
        mods = list(mods) + script_imports(script)
    took = {}
    for mod_name in dict.fromkeys(mods):
        start_ts = time.time()
        try:
            importlib.import_module(mod_name)
        except ImportError as exc:
            took[mod_name] = f"skipped: {exc}"
            continue
        took[mod_name] = round(time.time() - start_ts, 3)
        if check():
            raise SystemExit(
                f"zygote: importing {mod_name} initialized CUDA; a forked rank could not "
                "use the GPU"
            )
    quiesce_compile_pools_after_warmup()
    return took


# our own modules that hold references to megatron's: re-imported with it
REFRESH_OWN = ("run", "hook", "gap", "state", "client")
# modules a code restart never drops even when they sit next to the script: this lab, the live-lab
# kernel it hosts (`from kernel import ...` in hook.py), and forksafe, whose patches the rank removes
KEEP_LOADED = ("rsibox_megatron", "kernel", "forksafe")


def refreshed(req, name):
    """Is module `name` re-imported in the ranks of this request (a code restart)?"""
    if name.startswith("rsibox_megatron.") and name.rsplit(".", 1)[1] in REFRESH_OWN:
        return bool(req.get("refresh"))
    return any(name == p or name.startswith(p + ".") for p in req.get("refresh", []))


def findings(req, src):
    out = []
    mod_of = {getattr(m, "__file__", None): n for n, m in list(sys.modules.items())}
    stale = [f for f, v in src.items() if _stat(f) != v and not refreshed(req, mod_of.get(f, ""))]
    if stale:
        out.append({"kind": "stale", "files": stale[:20], "n": len(stale),
                    "why": "the ranks would run the code imported when the zygote started",
                    "do": "restart the zygote (warm restart), apply the edit in the live lab, or "
                          "start with --refresh <package> to re-import it in the ranks"})
    env = [k for k, v in req.get("env", {}).items()
           if START_ENV.get(k) != v and k.startswith(EARLY_ENV_PREFIXES)
           and not k.startswith(LATE_ENV_PREFIXES)]
    if env:
        out.append({"kind": "env", "keys": sorted(env),
                    "why": "a library may have read these at import, in the zygote",
                    "do": "accept if they are read later, or start the zygote with them"})
    ov = req.get("env", {}).get("RSIBOX_OVERLAY")
    if ov and os.path.isdir(ov):
        pre = sorted(f[:-3] for f in os.listdir(ov) if f.endswith(".py") and f[:-3] in sys.modules
                     and not refreshed(req, f[:-3]))
        if pre:
            out.append({"kind": "overlay", "modules": pre,
                        "why": "the zygote imported these before the overlay could replace them, "
                               "so the ranks would run the original code",
                        "do": "start with --refresh <top-level package> (the ranks re-import it from the "
                          "overlay), start fresh (torchrun), or use the live lab"})
    acc = set(req.get("accept", []))
    return [f for f in out if f["kind"] not in acc], out


def _stat(f):
    try:
        st = os.stat(f)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def default_compile_threads(local_world_size, cpu_count=None):
    """Return per-rank TorchInductor compile thread count during warmup: fair share of host
    CPUs (clamped to [2, 8]) so cold/refresh torch.compile stays fast without oversubscribing
    all ranks on the node."""
    total_cpus = int(cpu_count or os.cpu_count() or 8)
    world_procs = max(1, int(local_world_size))
    return max(2, min(8, total_cpus // max(1, 2 * world_procs)))


def clamp_pre_fork_threads(environ=os.environ):
    """Configure OpenMP, BLAS, and TorchInductor thread pools before preloading modules in the
    zygote. OpenMP and BLAS intra-op pools default to 1 thread so forked ranks do not inherit
    32-thread CPU pools, while TORCHINDUCTOR_COMPILE_THREADS defaults to PRELOAD_COMPILE_THREADS
    (8) during preload/warmup and is quiesced via quiesce_compile_pools_after_warmup() before
    fork and before steady-state training."""
    for env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environ.setdefault(env_key, "1")
        START_ENV.setdefault(env_key, environ[env_key])
    environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(PRELOAD_COMPILE_THREADS))
    START_ENV.setdefault(
        "TORCHINDUCTOR_COMPILE_THREADS", environ["TORCHINDUCTOR_COMPILE_THREADS"]
    )


def quiesce_compile_pools_after_warmup():
    """Reset torch's intra-op thread pool to 1 and shut down TorchInductor AsyncCompile pools."""
    torch_mod = sys.modules.get("torch")
    applied_torch_threads = 1 if (torch_mod is not None and hasattr(torch_mod, "set_num_threads")) else None
    if applied_torch_threads:
        torch_mod.set_num_threads(1)
    inductor_config = sys.modules.get("torch._inductor.config")
    if inductor_config is not None and hasattr(inductor_config, "compile_threads"):
        inductor_config.compile_threads = 1
    pools_shutdown = []
    if (ac := sys.modules.get("torch._inductor.async_compile")) is not None:
        targets = [ac] + ([cls] if (cls := getattr(ac, "AsyncCompile", None)) is not None else [])
        for holder in targets:
            for attr_name in ("_pool", "_process_pool", "_compile_threads_pool"):
                if (pool_obj := getattr(holder, attr_name, None)) is not None and hasattr(pool_obj, "shutdown"):
                    with contextlib.suppress(Exception):
                        pool_obj.shutdown(wait=False)
                        pools_shutdown.append(f"{getattr(holder, '__name__', 'async_compile')}.{attr_name}")
                    with contextlib.suppress(Exception):
                        setattr(holder, attr_name, None)
    return {
        "torch_threads": applied_torch_threads,
        "compile_threads": getattr(inductor_config, "compile_threads", None),
        "pools_shutdown": pools_shutdown,
    }


def rank_env(request_data, local_rank):
    num_procs = int(request_data["nproc"])
    node_rank = int(request_data.get("node_rank", 0))
    env_vars = dict(request_data.get("env", {}))
    env_vars.update(
        RANK=str(node_rank * num_procs + local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(num_procs * int(request_data.get("nnodes", 1))),
        LOCAL_WORLD_SIZE=str(num_procs),
        GROUP_RANK=str(node_rank),
        MASTER_ADDR=request_data.get("master_addr", "127.0.0.1"),
        MASTER_PORT=str(request_data["port"]),
        TORCHELASTIC_RUN_ID=str(request_data.get("run_id", "zygote")),
        RSIBOX_T0=repr(request_data.get("t0", time.time())),
    )
    if num_procs > 1 and "OMP_NUM_THREADS" not in env_vars and "OMP_NUM_THREADS" not in os.environ:
        env_vars["OMP_NUM_THREADS"] = "1"  # torchrun's default for more than one rank
    if "TORCHINDUCTOR_COMPILE_THREADS" not in env_vars:
        env_vars["TORCHINDUCTOR_COMPILE_THREADS"] = str(default_compile_threads(num_procs))
    return env_vars


def cpu_list(spec):
    """'0-3,8,10-11' -> {0, 1, 2, 3, 8, 10, 11}."""
    return set(forksafe.parse_cpulist(spec))


def unpatch(cuda=None):
    """In a forked rank: put back the torch.cuda functions the zygote patched for its imports
    (forksafe's get_device_capability, current_device and device-properties stand-in), and
    drop PYTORCH_NVML_BASED_CUDA_CHECK if the zygote set it. The rank initializes CUDA itself,
    so it needs neither, and the wrappers cost a check on every call. Returns the names put back."""
    return forksafe.unpatch(cuda, clear_env="PYTORCH_NVML_BASED_CUDA_CHECK" not in START_ENV)


def bind(req, local_rank):
    """Pin the rank to its CPU list (`cpus`: one 'a-b,c' string per local rank, or auto-detected
    GPU NUMA affinity when RSIBOX_ZYGOTE_AUTO_NUMA=1) before CUDA and its threads start. A fork
    keeps the zygote's affinity; without a list nothing changes, as under torchrun. Returns the
    CPUs set, or None."""
    lists = req.get("cpus") or []
    if local_rank < len(lists) and lists[local_rank]:
        cpus = sorted(cpu_list(lists[local_rank]))
        forksafe.bind_rank_numa_and_cpus(local_rank, explicit_cpus=cpus)
        return cpus
    if os.environ.get("RSIBOX_ZYGOTE_AUTO_NUMA") == "1":
        res = forksafe.bind_rank_numa_and_cpus(local_rank)
        return res.get("bound_cpus") or None
    return None


def set_threads(environ=os.environ):
    """Apply OMP_NUM_THREADS and TORCHINDUCTOR_COMPILE_THREADS in a forked rank."""
    applied = None
    if (torch_mod := sys.modules.get("torch")) is not None and (omp := environ.get("OMP_NUM_THREADS", "")).isdigit() and int(omp) > 0:
        applied = int(omp)
        torch_mod.set_num_threads(applied)
    if (cfg := sys.modules.get("torch._inductor.config")) is not None and (ct := environ.get("TORCHINDUCTOR_COMPILE_THREADS", "")).isdigit() and int(ct) > 0:
        cfg.compile_threads = int(ct)
    return applied


def _guard(rank_pid):
    """In a process forked from the rank: wait until the rank is gone (PR_SET_PDEATHSIG delivers
    SIGUSR2), then kill the rank's process group (its data-loader, compile and other children
    that did not leave the group). A SIGTERM (the zygote's own stop of the tree, which signals
    the group too) just ends the guard. Never returns."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        why = []
        signal.signal(signal.SIGTERM, lambda *_: why.append("term"))
        signal.signal(signal.SIGUSR2, lambda *_: why.append("rank_gone"))
        _prctl(PR_SET_NAME, b"rsibox-guard")
        _prctl(PR_SET_PDEATHSIG, signal.SIGUSR2)
        fd = os.open(os.devnull, os.O_RDWR)
        for i in (0, 1, 2):
            os.dup2(fd, i)
        os.closerange(3, 65536)
        if os.getppid() != rank_pid:            # the rank died before prctl took effect
            why.append("rank_gone")
        while not why:
            signal.pause()
        if "term" not in why:
            os.killpg(rank_pid, signal.SIGKILL)    # the guard is in the group: last act
    finally:
        os._exit(0)


def _supervised(zygote_pid):
    """In the forked rank, right after setsid: die with the zygote, and start the guard."""
    _prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() != zygote_pid:               # the zygote died before prctl took effect
        os._exit(1)
    rank = os.getpid()
    if os.fork() == 0:
        _guard(rank)


def drop_modules(req, here=None):
    """Code restart: forget the refreshed modules (and those under `here`, the script's folder)
    and unbind each from its parent package so the next import reloads from overlay/source."""
    drop = [name for name, module in list(sys.modules.items()) if refreshed(req, name)
            or (here and not name.startswith(KEEP_LOADED)
                and os.path.abspath(getattr(module, "__file__", None) or "/").startswith(here))]
    return hooks.purge_modules(drop)


def _child(req, local_rank, lsock, conn, zygote_pid=None):
    """In the forked rank: never returns."""
    code = 1
    try:
        lsock.close()
        conn.close()
        os.setsid()
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD):
            signal.signal(s, signal.SIG_DFL)
        if zygote_pid is not None:
            _supervised(zygote_pid)
        os.environ.update(rank_env(req, local_rank))
        unguard_cuda_python()                   # the rank initializes CUDA itself
        if os.environ.get("RSIBOX_ZYGOTE_UNPATCH", "1") != "0":
            unpatch()
        bind(req, local_rank)
        set_threads()
        d = req.get("log_dir") or "."
        os.makedirs(d, exist_ok=True)
        fd = os.open(os.path.join(d, f"rank{local_rank}.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        if fd > 2:
            os.close(fd)
        if req.get("cwd"):
            os.chdir(req["cwd"])
        sys.argv = ["rsibox_megatron.run"] + list(req["argv"])
        if req.get("refresh"):
            # code restart: forget the refreshed packages (and the script's own modules next to it), so the
            # ranks import them again, from the overlay where it has them; torch, TE and the rest stay loaded
            here = os.path.dirname(os.path.abspath(req["argv"][0])) + os.sep if req.get("argv") else None
            drop = drop_modules(req, here)
            from rsibox_megatron import overlay
            overlay.install()
            print(json.dumps({"zygote_refresh": {"dropped": len(drop), "overlay": os.environ.get("RSIBOX_OVERLAY")}}),
                  flush=True)
        if req.get("entry", "run") == "run":
            from rsibox_megatron import run
            run.main()
        else:                                   # the script alone (tests, other launchers)
            import runpy
            sys.argv = sys.argv[1:]
            runpy.run_path(sys.argv[0], run_name="__main__")
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(code)


stat_ppid = proc.stat_ppid


def _read_nspid(status_path):
    """Return `(nspid_list, ppid)` from Linux `/proc/<pid>/status`, or `(None, None)`."""
    try:
        with open(status_path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None, None
    nspid, ppid = None, None
    for line in text.splitlines():
        if line.startswith("NSpid:"):
            parts = line.split()[1:]
            if parts and all(part.isdigit() for part in parts):
                nspid = [int(part) for part in parts]
        elif line.startswith("PPid:"):
            parts = line.split()[1:]
            if parts and parts[0].isdigit():
                ppid = int(parts[0])
    return nspid, ppid


def _is_procfs(proc_root):
    """True when `proc_root` looks like a real procfs mount rather than a minimal test dir."""
    if os.path.islink(os.path.join(proc_root, "self")):
        return True
    return any(os.path.exists(os.path.join(proc_root, name))
               for name in ("stat", "uptime", "version"))


def _translated_ppids(proc_root, my_pid):
    """Translate `/proc` from an ancestor PID namespace into `{local_pid: local_ppid}` via
    `NSpid` and `ns/pid`; return `{}` if `/proc` is from an incompatible PID namespace."""
    self_nspid, _ = _read_nspid(os.path.join(proc_root, "self", "status"))
    if not self_nspid or len(self_nspid) < 2 or self_nspid[-1] != my_pid:
        return {}
    ns_idx = len(self_nspid) - 1
    try:
        self_ns = os.readlink(os.path.join(proc_root, "self", "ns", "pid"))
    except OSError:
        self_ns = None
    outer_to_local, outer_ppids, deeper, missing_ns = {}, {}, [], []
    if self_ns is None:
        outer_to_local[self_nspid[0]] = my_pid
        outer_ppids[self_nspid[0]] = 0
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        outer_pid = int(entry)
        nspid, ppid = _read_nspid(os.path.join(proc_root, entry, "status"))
        if nspid is None or ppid is None or len(nspid) <= ns_idx:
            continue
        if self_ns is not None:
            try:
                entry_ns = os.readlink(os.path.join(proc_root, entry, "ns", "pid"))
            except FileNotFoundError:
                if len(nspid) == len(self_nspid):
                    missing_ns.append((outer_pid, nspid[ns_idx], ppid))
                elif len(nspid) > len(self_nspid):
                    deeper.append((outer_pid, nspid[ns_idx], ppid))
                continue
            except OSError:
                continue
            if entry_ns == self_ns:
                outer_to_local[outer_pid] = nspid[ns_idx]
                outer_ppids[outer_pid] = ppid
            elif len(nspid) > len(self_nspid):
                deeper.append((outer_pid, nspid[ns_idx], ppid))
        elif outer_pid != self_nspid[0]:
            missing_ns.append((outer_pid, nspid[ns_idx], ppid))
    # Include children whose parent is in our namespace if `ns/pid` vanished or they unshared.
    candidates = missing_ns + deeper
    changed = True
    while changed and candidates:
        changed, rest = False, []
        for outer_pid, local_pid, ppid in candidates:
            if ppid in outer_to_local:
                outer_to_local[outer_pid] = local_pid
                outer_ppids[outer_pid] = ppid
                changed = True
            else:
                rest.append((outer_pid, local_pid, ppid))
        candidates = rest
    return {outer_to_local[opid]: outer_to_local.get(oppid, 0)
            for opid, oppid in outer_ppids.items()}


def _ppids(proc_root="/proc"):
    """Return `{pid: ppid}` in our PID namespace (translating ancestor `/proc` via `NSpid`)."""
    if not proc_root or not os.path.isdir(proc_root):
        return proc.ppid_map(proc_root=proc_root)
    self_path = os.path.join(proc_root, "self")
    if os.path.islink(self_path):
        try:
            self_target = os.readlink(self_path)
        except OSError:
            return {}
        if not self_target.isdigit():
            return {}
        my_pid = os.getpid()
        if int(self_target) == my_pid:
            self_nspid, _ = _read_nspid(os.path.join(proc_root, "self", "status"))
            if not self_nspid or len(self_nspid) <= 1:
                return proc.ppid_map(proc_root=proc_root)
        return _translated_ppids(proc_root, my_pid)
    if _is_procfs(proc_root):
        return {}
    return proc.ppid_map(proc_root=proc_root)


def _tree(pid, proc_root="/proc"):
    kids = {}
    for child, parent in _ppids(proc_root=proc_root).items():
        kids.setdefault(parent, []).append(child)
    found, todo, seen = [pid], [pid], {pid}
    while todo:
        current = todo.pop(0)
        for child in sorted(kids.get(current, ())):
            if child not in seen:
                seen.add(child)
                found.append(child)
                todo.append(child)
    return found


def kill_tree(pid, sig=signal.SIGTERM, proc_root="/proc"):
    """Signal pid's process tree (children first) and its process group."""
    targets = list(reversed(_tree(pid, proc_root=proc_root)))
    signaled = []
    for target in targets:
        try:
            os.kill(target, sig)
            signaled.append(target)
        except (ProcessLookupError, PermissionError):
            pass
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pid, sig)
    return signaled


def _send(conn, obj):
    """Best effort: False (and a log line) when the client is gone."""
    try:
        conn.sendall((json.dumps(obj, default=str) + "\n").encode())
        return True
    except OSError as e:
        log("client_gone", on=obj.get("event"), error=repr(e))
        return False


def _client_gone(conn):
    """Has the client closed its end? (It sends nothing after its request line.)"""
    try:
        r, _, _ = select.select([conn], [], [], 0)
        if not r:
            return False
        return conn.recv(4096) == b""
    except (OSError, ValueError):
        return True


def _children(parent_pid=None, proc_root="/proc"):
    """Our direct children (ranks, guards, orphans the subreaper took in), by parent pid."""
    target = os.getpid() if parent_pid is None else parent_pid
    return sorted(child for child, parent in _ppids(proc_root=proc_root).items() if parent == target)


def _reap_unusable_proc(timeout, keep_set, proc_root, known):
    """Kill and reap known rank PIDs when `/proc` is from an untranslatable PID namespace."""
    reaped = []
    pending = []
    for pid in known:
        if pid in keep_set or pid in pending:
            continue
        kill_tree(pid, signal.SIGKILL, proc_root=proc_root)
        pending.append(pid)
    deadline = time.monotonic() + timeout
    while pending:
        still = []
        for pid in pending:
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if done:
                reaped.append(done)
            else:
                still.append(pid)
        pending = still
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    if not keep_set:
        while True:
            try:
                done, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not done:
                break
            reaped.append(done)
    return reaped


def reap_all(timeout=10.0, keep=(), proc_root="/proc", known=()):
    """Kill and reap non-kept direct children (ranks, guards, adopted orphans)."""
    keep_set = set(keep)
    me = os.getpid()
    ppids = _ppids(proc_root=proc_root)
    if proc_root and os.path.isdir(proc_root) and me not in ppids:
        return _reap_unusable_proc(timeout, keep_set, proc_root, known)
    reaped, killed = [], set()
    deadline = time.monotonic() + timeout
    left = sorted(child for child, parent in ppids.items() if parent == me and child not in keep_set)
    while left:
        for child in left:
            if child not in killed:
                kill_tree(child, signal.SIGKILL, proc_root=proc_root)
                killed.add(child)
            try:
                pid, _ = os.waitpid(child, os.WNOHANG)
            except ChildProcessError:
                continue
            if pid:
                reaped.append(pid)
        left = [child for child in _children(me, proc_root=proc_root) if child not in keep_set]
        if not left or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    if left:
        log("reap_timeout", left=left)
    return reaped


def validate(req):
    """The fields handle() needs, before any fork; raises ValueError naming the first bad one."""
    if not isinstance(req, dict):
        raise ValueError(f"request must be a JSON object, got {type(req).__name__}")
    try:
        n = int(req["nproc"])
        int(req["port"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"nproc and port must be integers: {e!r}") from None
    if n < 1:
        raise ValueError("nproc must be at least 1")
    if not isinstance(req.get("argv"), list) or not req["argv"]:
        raise ValueError("argv must be a non-empty list")
    if not isinstance(req.get("env", {}), dict):
        raise ValueError("env must be an object")
    if req.get("on_client_gone", "kill") not in ("kill", "keep"):
        raise ValueError("on_client_gone must be kill or keep")


def handle(req, lsock, conn, src):
    validate(req)
    block, seen = findings(req, src)
    if block:
        _send(conn, {"event": "refused", "findings": block})
        return
    me, pids, rc = os.getpid(), [], {}
    # Pre-job helpers survive; the zygote starts no helpers of its own during a job, so any
    # other child appearing under it is a job rank, guard, or adopted job helper/orphan.
    before = set(_children())
    t0 = req.get("t0", time.time())
    try:
        for r in range(int(req["nproc"])):
            pid = os.fork()
            if pid == 0:
                _child(req, r, lsock, conn, me)
            pids.append(pid)
        connected = _send(conn, {"event": "forked", "pids": pids, "findings": seen,
                                 "fork_s": round(time.time() - t0, 3), "mono": time.monotonic()})
        policy = req.get("on_client_gone", "kill")
        left, stopping = set(pids), False
        deadline = time.time() + float(req.get("timeout", 0) or 1e12)
        while left:
            if connected and _client_gone(conn):
                connected = False
                log("client_gone", on="running", pids=sorted(left), policy=policy)
            if not connected and policy == "kill" and not stopping:
                stopping = True                 # SIGTERM now, SIGKILL after the grace
                for p in left:
                    kill_tree(p)
                deadline = min(deadline, time.time() + CLIENT_GONE_GRACE_S)
            for pid in tuple(left):
                done, st = os.waitpid(pid, os.WNOHANG)
                if not done:
                    continue
                left.remove(pid)
                code = os.waitstatus_to_exitcode(st)
                rc[pids.index(pid)] = code
                if code:                        # torchrun's policy: one rank fails, stop all
                    for peer in left:
                        kill_tree(peer)
                    deadline = min(deadline, time.time() + CLIENT_GONE_GRACE_S)
            if left and time.time() > deadline:
                for pid in left:
                    kill_tree(pid, signal.SIGKILL)
                deadline = float("inf")
            if left:
                time.sleep(0.05)
    finally:
        unreaped = [pid for idx, pid in enumerate(pids) if idx not in rc]
        reaped = reap_all(keep=before, known=unreaped)  # ranks left by exception, guards, orphans
        if pids:
            log("job_reaped", ranks=pids, reaped=len(reaped))
    _send(conn, {"event": "exited", "rc": [rc.get(i) for i in range(len(pids))],
                 "s": round(time.time() - t0, 3)})


def serve(sock, mods, script=None, forksafe=True):
    start_time = time.time()
    subreaper = _prctl(PR_SET_CHILD_SUBREAPER, 1)    # orphans of a job come back here to be reaped
    clamp_pre_fork_threads()
    forksafe_status = install_forksafe() if forksafe else {"installed": False, "why": "off"}
    took = preload(mods, script)
    src = sources()
    gc_frozen = 0
    if os.environ.get("RSIBOX_ZYGOTE_GC_FREEZE", "1") != "0":
        # Freeze preloaded modules (~900K objects) in the permanent GC generation so cyclic GC
        # traversals in forked ranks do not dirty shared Copy-on-Write pages.
        gc_info = freeze_gc_before_fork()
        gc_frozen = gc_info.get("frozen_count", gc.get_freeze_count())
    if os.path.exists(sock):
        os.unlink(sock)
    listen_sock = socket.socket(socket.AF_UNIX)
    listen_sock.bind(sock)
    listen_sock.listen(4)
    old_sigterm = None
    if os.getpid() == 1 and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        # Linux PID 1 ignores SIG_DFL SIGTERM; exit via SystemExit so handle() reaps ranks first.
        old_sigterm = signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    import threading
    threads = {"native": len(os.listdir("/proc/self/task")) if os.path.isdir("/proc/self/task")
               else None, "python": [thread_obj.name for thread_obj in threading.enumerate()]}
    print(json.dumps({"event": "ready", "pid": os.getpid(),
                      "import_s": round(time.time() - start_time, 3),
                      "threads": threads, "gc_frozen": gc_frozen, "subreaper": subreaper,
                      "modules": len(sys.modules), "forksafe": forksafe_status, "took": took}),
          flush=True)
    try:
        while True:
            conn, _ = listen_sock.accept()
            with conn:
                try:                            # a bad request or a gone client must not kill the zygote
                    conn.settimeout(30)         # for the request line only
                    with conn.makefile() as reader:
                        line = reader.readline()
                    conn.settimeout(None)
                    if not line:
                        continue
                    request_data = json.loads(line)
                    if not isinstance(request_data, dict):
                        raise ValueError(f"request must be a JSON object, got {type(request_data).__name__}")
                    if request_data.get("op") == "stop":
                        _send(conn, {"event": "stopped"})
                        break
                    if request_data.get("op") == "ping":
                        _send(conn, {"event": "pong", "pid": os.getpid()})
                        continue
                    handle(request_data, listen_sock, conn, src)
                except Exception as exc:
                    log("bad_request", error=repr(exc))
                    _send(conn, {"event": "error", "error": repr(exc)})
    finally:
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        listen_sock.close()
        with contextlib.suppress(OSError):
            os.unlink(sock)


def request(sock, req, out=sys.stdout):
    """Send one request; yield each reply event."""
    return ipc.unix_json_stream(sock, req)


def main(argv=None):
    p = argparse.ArgumentParser(prog="rsibox_megatron.zygote")
    p.add_argument("cmd", choices=["serve", "start", "stop", "ping"])
    p.add_argument("--sock", required=True)
    p.add_argument("--preload", default=",".join(DEFAULT_PRELOAD))
    p.add_argument("--script")
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--port", type=int, default=29600)
    p.add_argument("--log-dir")
    p.add_argument("--cwd")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--accept", default="")
    p.add_argument("--entry", default="run", choices=["run", "script"])
    p.add_argument("--timeout", type=float, default=0)
    p.add_argument("--refresh", default="", help="code restart: comma-separated top-level packages the forked "
                   "ranks drop and re-import (e.g. megatron), so RSIBOX_OVERLAY edits to them run")
    p.add_argument("--cpus", default="", help="per local rank CPU lists, ';'-separated: '0-71;72-143'")
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    a = p.parse_args(argv[:len(argv) - len(rest) - (1 if rest or "--" in argv else 0)])
    if a.cmd == "serve":
        serve(a.sock, [m for m in a.preload.split(",") if m], a.script,
              os.environ.get("RSIBOX_ZYGOTE_FORKSAFE", "1") != "0")
        return 0
    if a.cmd in ("stop", "ping"):
        for ev in request(a.sock, {"op": a.cmd}):
            print(json.dumps(ev), flush=True)
        return 0
    req = {"argv": rest, "nproc": a.nproc, "port": a.port, "log_dir": a.log_dir, "cwd": a.cwd,
           "env": dict(kv.split("=", 1) for kv in a.env), "entry": a.entry,
           "accept": [x for x in a.accept.split(",") if x], "timeout": a.timeout,
           "refresh": [x for x in a.refresh.split(",") if x],
           "cpus": [x for x in a.cpus.split(";")] if a.cpus else [],
           "t0": float(os.environ.get("RSIBOX_T0") or time.time())}
    rc = 1
    for ev in request(a.sock, req):
        print(json.dumps(ev), flush=True)
        if ev["event"] == "exited":
            rc = next((c for c in ev["rc"] if c), 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
