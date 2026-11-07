"""Zygote: one process per node that has torch, Transformer Engine and Megatron imported, without
CUDA, and starts the ranks of a training job by fork instead of torchrun + exec. A restart then
skips the Python imports (16-85 s from a network file system in our runs) and pays only
Megatron's own setup and the first step.

    python -m fastrsi_megatron.zygote serve --sock S [--preload M1,M2] [--script pretrain_gpt.py]
    python -m fastrsi_megatron.zygote start --sock S --nproc N [--port P] [--log-dir D]
        [--env K=V ...] [--accept stale,env] [--timeout SEC] -- pretrain_gpt.py <megatron args>
    python -m fastrsi_megatron.zygote stop --sock S

`start` sends one request and waits until every rank has exited; it prints one JSON line per
event (`forked`, then `exited` with each rank's exit code) and returns the job's exit code.
Each rank runs `fastrsi_megatron.run` (phase stamps, lab hook) exactly as under torchrun, with
FASTRSI_T0 set to the request time, so phases.rank<r>.json counts from the request.

Contract:
- The zygote never initializes CUDA or the CUDA driver. If an import does, `serve` exits and
  names the module: no forked rank could use the GPU.
- A rank gets the variables torchrun sets (RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE,
  GROUP_RANK, MASTER_ADDR, MASTER_PORT, TORCHELASTIC_RUN_ID, and OMP_NUM_THREADS=1 for more than
  one rank unless set) plus the request's `env`, its own session (setsid) and its stdout/stderr
  in <log-dir>/rank<r>.log; with `cpus` (one CPU list per local rank) it is pinned to its list
  right after the fork (a fork otherwise keeps the zygote's affinity). OMP_NUM_THREADS is applied to torch's thread pool in the rank
  (torch.set_num_threads), since torch read the variable when the zygote imported it.
- When a rank exits non-zero, the zygote terminates the other ranks' process trees by PID (as
  torchrun does), never by a process group it shares.
- Findings (not gates; a request lists the ones it accepts in `accept`, otherwise no fork):
  "stale": a source file of a module the zygote imported changed since (the ranks would run
  the old code; restart the zygote, or edit through the live lab); "env": a request variable
  with a prefix that a preloaded library may read at import (NVTE_, TORCH_, TORCHINDUCTOR_,
  TRITON_, LD_, PYTHON...) differs from the zygote's start value. NCCL_*, TORCH_NCCL_*,
  CUDA_VISIBLE_DEVICES and the rank variables are read after the fork and never count;
  "overlay": the request's FASTRSI_OVERLAY replaces modules the zygote has imported already
  (overlay.py works only before the first import), so the edit would not run.
"""
import argparse, ast, ctypes, importlib, json, os, signal, socket, sys, time, traceback

START_ENV = dict(os.environ)
# Variables a preloaded library may read at import (in the zygote, before the fork) ...
EARLY_ENV_PREFIXES = ("NVTE_", "PYTORCH_", "TORCH_", "TORCHINDUCTOR_", "TRITON_", "LD_",
                      "PYTHON", "MEGATRON_")
# ... except these, read at process-group or communicator creation, after the fork.
LATE_ENV_PREFIXES = ("TORCH_NCCL_", "TORCH_DISTRIBUTED_")
# torch._dynamo and torch._inductor are imported lazily at the first torch.compile, in Megatron's
# kernel warm-up: preloading them takes 1.5-2 s off each restart (with
# TORCHINDUCTOR_COMPILE_THREADS=1 in the zygote's environment, our 14B runs: warm-up 0.6-1.4 s
# instead of 2.2-2.8 s).
DEFAULT_PRELOAD = ["torch", "megatron.training.training", "torch._dynamo",
                   "torch._inductor.compile_fx", "torch._inductor.codecache"]


def install_forksafe():
    """Keep the imports from initializing CUDA. Some imports only ask about the GPU (quack, via
    Transformer Engine, calls torch.cuda.get_device_capability at import, which initializes
    CUDA; tvm_ffi calls torch.cuda.is_available, which initializes the driver): answer those
    from NVML, with the vLLM lab's forksafe (FASTRSI_SERVICES, default its folder in this
    repo) and torch's NVML-based availability check. Returns what was done."""
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    d = os.environ.get("FASTRSI_SERVICES") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vllm", "livelab", "services")
    if os.path.isfile(os.path.join(d, "forksafe.py")):
        sys.path.insert(0, os.path.abspath(d))
    try:
        import forksafe
        import torch
    except ImportError as e:
        return {"installed": False, "why": repr(e)}
    out = forksafe.install(torch.cuda)
    out["properties"] = install_properties(torch.cuda)
    return out


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


def install_properties(cuda, probe=None):
    """Make cuda.get_device_properties(i) answer, while CUDA is not initialized here, with the
    values CUDA itself reported for device i in a short helper process (probe: a callable
    returning the list of dicts; default a subprocess running PROPS_CODE with this
    environment). Some imports read a value at module level (flashinfer's gdn kernels:
    get_device_properties(0).multi_processor_count) and would otherwise initialize CUDA.
    Indices outside the probed devices, or calls after CUDA is initialized, run the original."""
    import subprocess
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
            i = _device_index(device)
            if i is not None and 0 <= i < len(props):
                return Properties(props[i])
        return orig(device)
    get_device_properties._fastrsi_props = True
    get_device_properties.__wrapped__ = orig
    cuda.get_device_properties = get_device_properties
    return {"installed": True, "devices": len(props), "probe_s": round(time.time() - t, 2)}


def _device_index(device):
    """0 for None/"cuda", i for i, "cuda:i" or a cuda torch.device (index None -> 0)."""
    if device is None or device == "cuda":
        return 0
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        return int(device.split(":")[1]) if device.startswith("cuda:") else None
    if getattr(device, "type", None) == "cuda":
        return device.index or 0
    return None


def driver_initialized():
    """Whether this process has initialized the CUDA driver: cuDeviceGetCount fails with
    CUDA_ERROR_NOT_INITIALIZED (3) until something called cuInit."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    n = ctypes.c_int()
    return cu.cuDeviceGetCount(ctypes.byref(n)) != 3


def cuda_initialized():
    t = sys.modules.get("torch")
    if t is not None and t.cuda.is_initialized():
        return True
    return driver_initialized()


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
    naming the first module whose import initialized CUDA."""
    if script:
        sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
        mods = list(mods) + script_imports(script)
    took = {}
    for m in dict.fromkeys(mods):
        t = time.time()
        try:
            importlib.import_module(m)
        except ImportError as e:
            took[m] = f"skipped: {e}"
            continue
        took[m] = round(time.time() - t, 3)
        if check():
            raise SystemExit(f"zygote: importing {m} initialized CUDA; a forked rank could not "
                             "use the GPU")
    return took


def findings(req, src):
    out = []
    stale = [f for f, v in src.items() if _stat(f) != v]
    if stale:
        out.append({"kind": "stale", "files": stale[:20], "n": len(stale),
                    "why": "the ranks would run the code imported when the zygote started",
                    "do": "restart the zygote (warm restart), or apply the edit in the live lab"})
    env = [k for k, v in req.get("env", {}).items()
           if START_ENV.get(k) != v and k.startswith(EARLY_ENV_PREFIXES)
           and not k.startswith(LATE_ENV_PREFIXES)]
    if env:
        out.append({"kind": "env", "keys": sorted(env),
                    "why": "a library may have read these at import, in the zygote",
                    "do": "accept if they are read later, or start the zygote with them"})
    ov = req.get("env", {}).get("FASTRSI_OVERLAY")
    if ov and os.path.isdir(ov):
        pre = sorted(f[:-3] for f in os.listdir(ov) if f.endswith(".py") and f[:-3] in sys.modules)
        if pre:
            out.append({"kind": "overlay", "modules": pre,
                        "why": "the zygote imported these before the overlay could replace them, "
                               "so the ranks would run the original code",
                        "do": "start fresh (torchrun) for code edits, or use the live lab"})
    acc = set(req.get("accept", []))
    return [f for f in out if f["kind"] not in acc], out


def _stat(f):
    try:
        st = os.stat(f)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def rank_env(req, local_rank):
    n = int(req["nproc"])
    node = int(req.get("node_rank", 0))
    env = dict(req.get("env", {}))
    env.update(RANK=str(node * n + local_rank), LOCAL_RANK=str(local_rank),
               WORLD_SIZE=str(n * int(req.get("nnodes", 1))), LOCAL_WORLD_SIZE=str(n),
               GROUP_RANK=str(node), MASTER_ADDR=req.get("master_addr", "127.0.0.1"),
               MASTER_PORT=str(req["port"]), TORCHELASTIC_RUN_ID=str(req.get("run_id", "zygote")),
               FASTRSI_T0=repr(req.get("t0", time.time())))
    if n > 1 and "OMP_NUM_THREADS" not in env and "OMP_NUM_THREADS" not in os.environ:
        env["OMP_NUM_THREADS"] = "1"            # torchrun's default for more than one rank
    return env


def cpu_list(spec):
    """'0-3,8,10-11' -> {0, 1, 2, 3, 8, 10, 11}."""
    out = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def unpatch(cuda=None):
    """In a forked rank: put back the torch.cuda functions the zygote patched for its imports
    (forksafe's get_device_capability and current_device, the device-properties stand-in), and
    drop PYTORCH_NVML_BASED_CUDA_CHECK if the zygote set it. The rank initializes CUDA itself,
    so it needs neither, and the wrappers cost a check on every call. Returns the names put back."""
    if cuda is None:
        t = sys.modules.get("torch")
        cuda = t.cuda if t is not None else None
    out = []
    if cuda is not None:
        for name in ("get_device_capability", "current_device", "get_device_properties"):
            f = getattr(cuda, name, None)
            if f is not None and hasattr(f, "__wrapped__") and (
                    getattr(f, "_fastrsi_forksafe", False) or getattr(f, "_fastrsi_props", False)
                    or getattr(f, "__module__", "") == "forksafe"):
                setattr(cuda, name, f.__wrapped__)
                out.append(name)
    if "PYTORCH_NVML_BASED_CUDA_CHECK" not in START_ENV:
        os.environ.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    return out


def bind(req, local_rank):
    """Pin the rank to its CPU list (`cpus`: one 'a-b,c' string per local rank) before CUDA and
    its threads start. A fork keeps the zygote's affinity; without a list nothing changes,
    as under torchrun. Returns the CPUs set, or None."""
    lists = req.get("cpus") or []
    if local_rank < len(lists) and lists[local_rank]:
        cpus = cpu_list(lists[local_rank])
        os.sched_setaffinity(0, cpus)
        return sorted(cpus)
    return None


def set_threads(env=os.environ):
    """Apply OMP_NUM_THREADS to torch's intra-op pool in a forked rank. OpenMP and torch read
    it when they load, in the zygote, so setting the variable after the fork alone does
    nothing: without this a rank uses every core, as the zygote does."""
    t = sys.modules.get("torch")
    n = env.get("OMP_NUM_THREADS", "")
    if t is not None and n.isdigit() and int(n) > 0:
        t.set_num_threads(int(n))
        return int(n)
    return None


def _child(req, local_rank, lsock, conn):
    """In the forked rank: never returns."""
    code = 1
    try:
        lsock.close()
        conn.close()
        os.setsid()
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD):
            signal.signal(s, signal.SIG_DFL)
        os.environ.update(rank_env(req, local_rank))
        if os.environ.get("FASTRSI_ZYGOTE_UNPATCH", "1") != "0":
            unpatch()
        bind(req, local_rank)
        set_threads()
        d = req.get("log_dir") or "."
        os.makedirs(d, exist_ok=True)
        fd = os.open(os.path.join(d, f"rank{local_rank}.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        if req.get("cwd"):
            os.chdir(req["cwd"])
        sys.argv = ["fastrsi_megatron.run"] + list(req["argv"])
        if req.get("entry", "run") == "run":
            from fastrsi_megatron import run
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


def _tree(pid):
    """pid and its descendants, from /proc."""
    kids = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open(f"/proc/{p}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(p))
            except (OSError, IndexError, ValueError):
                pass
    out, todo = [], [pid]
    while todo:
        q = todo.pop()
        out.append(q)
        todo += kids.get(q, [])
    return out


def kill_tree(pid, sig=signal.SIGTERM):
    for p in reversed(_tree(pid)):
        try:
            os.kill(p, sig)
        except ProcessLookupError:
            pass


def _send(conn, obj):
    conn.sendall((json.dumps(obj) + "\n").encode())


def handle(req, lsock, conn, src):
    block, seen = findings(req, src)
    if block:
        _send(conn, {"event": "refused", "findings": block})
        return
    pids = []
    for r in range(int(req["nproc"])):
        pid = os.fork()
        if pid == 0:
            _child(req, r, lsock, conn)
        pids.append(pid)
    _send(conn, {"event": "forked", "pids": pids, "findings": seen,
                 "fork_s": round(time.time() - req.get("t0", time.time()), 3)})
    rc, left = {}, set(pids)
    deadline = time.time() + float(req.get("timeout", 0) or 1e12)
    while left:
        pid, st = os.waitpid(-1, os.WNOHANG)
        if pid == 0:
            if time.time() > deadline:
                for p in left:
                    kill_tree(p, signal.SIGKILL)
                deadline = 1e18
            time.sleep(0.05)
            continue
        if pid not in left:
            continue
        left.discard(pid)
        rc[pids.index(pid)] = os.waitstatus_to_exitcode(st)
        if rc[pids.index(pid)] != 0:            # torchrun's policy: one rank fails, stop all
            for p in left:
                kill_tree(p)
    _send(conn, {"event": "exited", "rc": [rc[i] for i in range(len(pids))],
                 "s": round(time.time() - req.get("t0", time.time()), 3)})


def serve(sock, mods, script=None, forksafe=True):
    t = time.time()
    fs = install_forksafe() if forksafe else {"installed": False, "why": "off"}
    took = preload(mods, script)
    src = sources()
    if os.path.exists(sock):
        os.unlink(sock)
    ls = socket.socket(socket.AF_UNIX)
    ls.bind(sock)
    ls.listen(4)
    import threading
    threads = {"native": len(os.listdir("/proc/self/task")) if os.path.isdir("/proc/self/task")
               else None, "python": [x.name for x in threading.enumerate()]}
    print(json.dumps({"event": "ready", "pid": os.getpid(), "import_s": round(time.time() - t, 3),
                      "threads": threads,
                      "modules": len(sys.modules), "forksafe": fs, "took": took}),
          flush=True)
    while True:
        conn, _ = ls.accept()
        with conn:
            line = conn.makefile().readline()
            if not line:
                continue
            req = json.loads(line)
            if req.get("op") == "stop":
                _send(conn, {"event": "stopped"})
                break
            if req.get("op") == "ping":
                _send(conn, {"event": "pong", "pid": os.getpid()})
                continue
            try:
                handle(req, ls, conn, src)
            except Exception as e:              # a bad request must not kill the zygote
                _send(conn, {"event": "error", "error": repr(e)})
    ls.close()
    os.unlink(sock)


def request(sock, req, out=sys.stdout):
    """Send one request; yield each reply event."""
    c = socket.socket(socket.AF_UNIX)
    c.connect(sock)
    c.sendall((json.dumps(req) + "\n").encode())
    for line in c.makefile():
        ev = json.loads(line)
        yield ev
        if ev["event"] in ("exited", "refused", "error", "stopped", "pong"):
            break
    c.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="fastrsi_megatron.zygote")
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
    p.add_argument("--cpus", default="", help="per local rank CPU lists, ';'-separated: '0-71;72-143'")
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    a = p.parse_args(argv[:len(argv) - len(rest) - (1 if rest or "--" in argv else 0)])
    if a.cmd == "serve":
        serve(a.sock, [m for m in a.preload.split(",") if m], a.script,
              os.environ.get("FASTRSI_ZYGOTE_FORKSAFE", "1") != "0")
        return 0
    if a.cmd in ("stop", "ping"):
        for ev in request(a.sock, {"op": a.cmd}):
            print(json.dumps(ev), flush=True)
        return 0
    req = {"argv": rest, "nproc": a.nproc, "port": a.port, "log_dir": a.log_dir, "cwd": a.cwd,
           "env": dict(kv.split("=", 1) for kv in a.env), "entry": a.entry,
           "accept": [x for x in a.accept.split(",") if x], "timeout": a.timeout,
           "cpus": [x for x in a.cpus.split(";")] if a.cpus else [],
           "t0": float(os.environ.get("FASTRSI_T0") or time.time())}
    rc = 1
    for ev in request(a.sock, req):
        print(json.dumps(ev), flush=True)
        if ev["event"] == "exited":
            rc = next((c for c in ev["rc"] if c), 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
