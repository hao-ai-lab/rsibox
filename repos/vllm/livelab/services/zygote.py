"""Zygote: a process that has vLLM and its heavy dependencies imported and starts `vllm serve`
by fork instead of exec. The API server then skips its Python imports, and with
VLLM_WORKER_MULTIPROC_METHOD=fork the engine core (forked by the API server) and the workers
(forked by the engine core) skip theirs too.

Usage:
  python3 zygote.py serve --sock S [--modules FILE]   # started by `engine.py zygote`, which
                                                      # gives it the engine's environment
  python3 zygote.py ping --sock S
Client: `engine.py start --zygote S` (request() below) gets the pid of a new `vllm serve`;
parse() below reads a `vllm serve` argv with vLLM's own parser (vllm_args.py), which the zygote
has imported already: ~0.2 s instead of ~5 s of imports in the launcher.

Checks (findings, not gates: each says what it saw, why it matters and what to do, and the
caller can accept it with `accept`; nothing passes silently):
- "env": the request's environment differs from the zygote's start environment (vLLM and its
  libraries read many variables at import time, so the forked server would run with the
  zygote's values), except the keys in VOLATILE;
- "stale": a source file of an imported module changed since the zygote imported it. By default
  the fork refreshes it (zygote_refresh.py): the child drops the edited modules and every module
  that imports them before the server starts, so it imports them again from the new source. The
  reply lists them under "refreshed". Only edits a refresh cannot apply (a changed extension
  `.so`, a module outside the refresh packages) stay a finding.
A finding without acceptance means no fork: the reply carries the finding and the suggested
actions. The zygote itself stops if its imports initialized CUDA or the CUDA driver, since no
forked child could then use the GPU; it names the module, which the modules file can exclude.

Before its imports the zygote installs the framework's forksafe.py (core/livelab;
FASTRSI_ZYGOTE_FORKSAFE=0 turns it off): torch.cuda.get_device_capability and current_device
answer from NVML while CUDA is not initialized. Every forked server inherits it, so a library
that asks for the GPU's capability at import (FlashInfer, imported by the API server at TP>1)
no longer initializes CUDA there, and vLLM keeps forking the engine core and the workers
instead of spawning them.
"""
import argparse, importlib, json, os, signal, socket, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corelab  # noqa: F401,E402  (core/livelab on sys.path: forksafe, fastrsi)
import forksafe, vllm_args, zygote_refresh  # noqa: E402
from fastrsi import ipc, proc  # noqa: E402

# Top-level packages whose edited modules a fork refreshes instead of refusing.
REFRESH_PREFIXES = tuple(p for p in os.environ.get("FASTRSI_ZYGOTE_REFRESH", "vllm").split(",")
                         if p)
# A refresh that reaches more modules than this re-imports the whole package (see plan()).
REFRESH_ALL_ABOVE = int(os.environ.get("FASTRSI_ZYGOTE_REFRESH_ALL_ABOVE", "100"))

# The environment the zygote was started with, before its imports. Libraries set variables
# while importing (torch, tilelang, vLLM's env_override ...); a request is compared with this
# snapshot, and the child keeps the post-import environment, as a fresh process would have.
START_ENV = dict(os.environ)

# Keys that may differ between the zygote's and the request's environment: shell bookkeeping,
# the cluster harness's per-task ids, and the profiler's per-run output settings (re-read
# after fork).
VOLATILE = {"_", "PWD", "OLDPWD", "SHLVL", "FASTRSI_PROF_DIR", "FASTRSI_PROF_T0",
            "FASTRSI_LIVE_RECONFIG_DIR"}   # engine.py: one live-agent folder per launch
VOLATILE_PREFIXES = ("RAY_",)

# What `vllm serve` imports before it parses its arguments (vllm/entrypoints/cli/main.py), and
# the engine core and worker modules it would import later.
BASE_MODULES = [
    "vllm.entrypoints.cli.main", "vllm.entrypoints.cli.benchmark.main",
    "vllm.entrypoints.cli.collect_env", "vllm.entrypoints.cli.download_kernels",
    "vllm.entrypoints.cli.launch", "vllm.entrypoints.cli.openai", "vllm.entrypoints.cli.preload",
    "vllm.entrypoints.cli.run_batch", "vllm.entrypoints.cli.serve", "vllm.entrypoints.cli.snapshot",
    "vllm.entrypoints.openai.api_server", "vllm.v1.engine.async_llm", "vllm.v1.engine.core",
    "vllm.v1.executor.multiproc_executor", "vllm.v1.worker.gpu_worker",
]


# Whether this process has initialized the CUDA driver (cuInit): a forked child of a process
# that did cannot use CUDA (forksafe: cuDeviceGetCount answers NOT_INITIALIZED until then).
driver_initialized = forksafe.driver_initialized


def _volatile(k):
    return k in VOLATILE or k.startswith(VOLATILE_PREFIXES)


def _sources():
    """{path: (mtime_ns, size)} of every imported module's file."""
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and f not in out:
            try:
                st = os.stat(f)
                out[f] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass
    return out


def _changed(snapshot):
    """Source files whose (mtime, size) differ from the snapshot. Stats run in threads: on a
    shared filesystem one stat costs ~0.1-0.2 ms and the snapshot holds ~8,000 files."""
    from concurrent.futures import ThreadPoolExecutor

    def stat(item):
        f, sig = item
        try:
            st = os.stat(f)
            return None if (st.st_mtime_ns, st.st_size) == sig else f
        except OSError:
            return f
    with ThreadPoolExecutor(32) as pool:
        return [f for f in pool.map(stat, snapshot.items(), chunksize=64) if f]


def _env_diff(env):
    mine = {k: v for k, v in START_ENV.items() if not _volatile(k)}
    theirs = {k: v for k, v in env.items() if not _volatile(k)}
    return sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))


def _child(req, conns, refresh=()):
    """In the forked child: become `vllm serve` with the request's argv, cwd, log and volatile
    env keys, after dropping the `refresh` modules so the server imports them again from their
    new source. With the `env` finding accepted, the keys where the request differs from the
    zygote's start environment take the request's value (or are removed); the other keys keep
    the zygote's post-import values. Risk of accepting: a key read at import time
    (CUDA_VISIBLE_DEVICES, HF_*, VLLM_*) shows the request's value in os.environ while the
    modules the zygote imported already acted on its start value."""
    code = 1
    try:
        for c in conns:
            c.close()
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)   # multiprocessing must reap its children
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        os.setsid()
        fd = os.open(req["log"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0); os.close(null)
        os.chdir(req.get("cwd") or "/")
        for env_key in [env_key for env_key in os.environ if _volatile(env_key) and env_key not in req["env"]]:
            del os.environ[env_key]
        os.environ.update({env_key: env_val for env_key, env_val in req["env"].items() if _volatile(env_key)})
        if "env" in req.get("accept", []):
            for env_key in _env_diff(req["env"]):
                if env_key in req["env"]:
                    os.environ[env_key] = req["env"][env_key]
                else:
                    os.environ.pop(env_key, None)
        if not req.get("env", {}).get("OMP_NUM_THREADS") and not os.environ.get("OMP_NUM_THREADS"):
            forksafe.clamp_thread_pools(num_threads=1)
        local_rank_str = (
            req.get("env", {}).get("LOCAL_RANK")
            or os.environ.get("LOCAL_RANK")
            or ""
        ).strip()
        visible_gpu_ids = [
            token.strip()
            for token in (
                req.get("env", {}).get("CUDA_VISIBLE_DEVICES")
                or os.environ.get("CUDA_VISIBLE_DEVICES")
                or ""
            ).split(",")
            if token.strip().isdigit()
        ]
        if local_rank_str.isdigit():
            forksafe.bind_rank_numa_and_cpus(int(local_rank_str))
        elif len(visible_gpu_ids) == 1:
            forksafe.bind_rank_numa_and_cpus(0)
        sys.argv = list(req["argv"])
        if refresh:
            r = zygote_refresh.refresh(refresh)
            print(json.dumps({"event": "zygote_refresh", **r, "names": list(refresh)[:50]}),
                  flush=True)
        # Only a refreshed fork re-registers ops; each skipped define, replaced implementation
        # and re-registered opaque type is a JSON line in the server log.
        with zygote_refresh.guard_for(refresh, zygote_refresh.print_event):
            from vllm.entrypoints.cli.main import main
            main()
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
    finally:
        try:
            sys.stdout.flush(); sys.stderr.flush()
        finally:
            os._exit(code)


def _refresh_plan(sources, graph, refresh_on=True):
    """The refresh plan for the source files that changed since the snapshot, or None."""
    stale = _changed(sources)
    fix = zygote_refresh.plan(stale, graph, whole_above=REFRESH_ALL_ABOVE) if stale else None
    if fix and not refresh_on:
        fix = {"modules": [], "changed": [], "blocked": [
            {"file": f, "why": "the request turned refresh off"} for f in stale]}
    return fix


def _stale_finding(fix, what="the forked server"):
    """The "stale" finding for changed files a refresh cannot apply, or None."""
    if not (fix and fix["blocked"]):
        return None
    return {"check": "stale", "observed": f"{len(fix['blocked'])} changed files a refresh "
            f"cannot apply: {fix['blocked'][:5]}",
            "expected": "edited files the fork can refresh (Python source under "
                        f"{','.join(REFRESH_PREFIXES) or 'no package'})",
            "why": f"{what} would run the code the zygote imported",
            "actions": ["restart the zygote (~25 s of imports, off the start path)",
                        "start without the zygote", "accept 'stale'"],
            "risk": "an edit is silently not applied"}


def _parse_in_child(argv, refresh, conns, cwd=None):
    """vllm_args.parse(argv) in a forked child, after the same refresh a fork would apply: an
    edit to vLLM's argument code is read, and the zygote's own state (modules, heap) stays as
    it was. The child sends its reply through a pipe and exits."""
    r, w = os.pipe()
    sys.stdout.flush(); sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        out = {"ok": False, "error": "parse child failed", "zygote_error": True}
        try:
            os.close(r)
            for c in conns:
                c.close()
            if cwd:
                os.chdir(cwd)
            if refresh:
                zygote_refresh.refresh(refresh)
            with zygote_refresh.guard_for(refresh):
                out = vllm_args.parse(argv)
        except BaseException as e:            # not vLLM judging the argv: the refresh or
            out = {"ok": False, "error": repr(e), "zygote_error": True}   # the import failed
        finally:
            try:
                data = json.dumps(out).encode()
                while data:
                    data = data[os.write(w, data):]
            finally:
                os._exit(0)
    os.close(w)
    with os.fdopen(r, "rb") as f:
        data = f.read()
    return json.loads(data) if data else {"ok": False, "error": "parse child died",
                                          "zygote_error": True}


def serve(sock_path, modules_file):
    err = sock_path_error(sock_path)
    if err:                      # fail before minutes of imports, not after them
        raise SystemExit(err)
    t0 = time.time()
    fs = {"installed": False, "why": "FASTRSI_ZYGOTE_FORKSAFE=0"}
    if os.environ.get("FASTRSI_ZYGOTE_FORKSAFE", "1") == "1":
        import torch
        fs = forksafe.install(torch.cuda)
    mods = list(BASE_MODULES)
    skip = set()
    if modules_file:
        for line in open(modules_file):
            m = line.strip()
            if m.startswith("-"):           # "-name": never pre-import (it initializes CUDA)
                skip.add(m[1:])
            elif m and not m.startswith("#"):
                mods.append(m)
    failed = []
    for m in mods:
        if m in sys.modules or m in skip:
            continue
        try:
            importlib.import_module(m)
        except BaseException as e:      # some optional modules fail to import; vLLM skips them too
            failed.append(f"{m}: {type(e).__name__}")
        if driver_initialized():
            raise SystemExit(f"zygote: importing {m} initialized the CUDA driver; a forked "
                             f"server could not use the GPU. Add '-{m}' to the modules file.")
    import torch
    if torch.cuda.is_initialized() or driver_initialized():
        raise SystemExit("zygote: imports initialized CUDA; a forked server could not use the GPU")
    sources = _sources()
    tg = time.time()
    graph = zygote_refresh.import_graph(REFRESH_PREFIXES) if REFRESH_PREFIXES else {}
    graph_s = round(time.time() - tg, 2)
    # Move everything imported so far out of the garbage collector's reach: a forked child
    # would otherwise scan (and so write to, copying every page) the whole inherited heap on
    # each full collection. Prefork servers do the same (Python docs, gc.freeze).
    if os.environ.get("FASTRSI_ZYGOTE_GC_FREEZE", "1") == "1":
        forksafe.freeze_gc_before_fork()
    print(json.dumps({"event": "zygote_ready", "pid": os.getpid(), "import_seconds":
                      round(time.time() - t0, 2), "modules": len(sys.modules),
                      "source_files": len(sources), "graph_modules": len(graph),
                      "graph_seconds": graph_s, "failed_imports": failed[:50],
                      "forksafe": {**fs, "counts": dict(forksafe.COUNTS)}}), flush=True)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)   # children are reaped automatically
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); os.chmod(sock_path, 0o600); srv.listen(4)
    while True:
        conn, _ = srv.accept()
        try:
            req = json.loads(conn.makefile().readline())
            if req.get("cmd") == "ping":
                reply = {"ok": True, "pid": os.getpid()}
            elif req.get("cmd") == "parse":
                t = time.time()
                fix = _refresh_plan(sources, graph)
                stale = _stale_finding(fix, "the parse")
                if stale and "stale" not in req.get("accept", []):
                    # Same refusal as a fork: vLLM's argument code may be among the files the
                    # child cannot refresh, so its parse would read the pre-edit flags.
                    reply = {"ok": False, "findings": [stale]}
                else:
                    reply = _parse_in_child(req["argv"], fix["modules"] if fix else [],
                                            [conn, srv], req.get("cwd"))
                reply["zygote_seconds"] = round(time.time() - t, 3)
                if fix and fix["changed"]:
                    reply["refreshed"] = {"changed": fix["changed"],
                                          "modules": len(fix["modules"])}
            else:
                t = time.time()
                findings = []
                bad_env = _env_diff(req["env"])
                if bad_env:
                    findings.append({
                        "check": "env", "observed": bad_env[:20],
                        "expected": "same environment as the zygote's start (except VOLATILE)",
                        "why": "variables read at import time keep the zygote's values in the "
                               "forked server", "actions": [
                            "restart the zygote with this environment (~25 s of imports, off "
                            "the start path)", "start without the zygote (+~21 s)",
                            "accept 'env' if these variables are not read at import time "
                            "(the server then gets the request's values)"],
                        "risk": "what the zygote imported keeps the zygote's values of these "
                                "variables"})
                fix = _refresh_plan(sources, graph, req.get("refresh", True))
                if _stale_finding(fix):
                    findings.append(_stale_finding(fix))
                refresh = fix["modules"] if fix else []
                check_s = round(time.time() - t, 3)
                accept = req.get("accept", [])
                open_findings = [f for f in findings if f["check"] not in accept]
                accepted = [f for f in findings if f["check"] in accept]
                if req.get("cmd") == "check":
                    # The same checks as a fork, nothing forked: a caller about to stop the
                    # running engine asks first, so a refusal does not leave it with none.
                    reply = {"ok": not open_findings, "findings": open_findings,
                             "check_seconds": check_s, "accepted_findings": accepted}
                elif open_findings:
                    reply = {"ok": False, "findings": open_findings, "check_seconds": check_s}
                else:
                    sys.stdout.flush(); sys.stderr.flush()
                    pid = os.fork()
                    if pid == 0:
                        _child(req, [conn, srv], refresh)
                    reply = {"ok": True, "pid": pid, "check_seconds": check_s,
                             "accepted_findings": accepted}
                if fix and fix["changed"]:
                    reply["refreshed"] = {"changed": fix["changed"], "modules": len(refresh),
                                          "scope": fix["scope"]}
                print(json.dumps({"event": "request", **reply}), flush=True)
            conn.sendall((json.dumps(reply) + "\n").encode())
        except Exception as e:
            try:
                conn.sendall((json.dumps({"ok": False, "error": repr(e), "zygote_error": True})
                              + "\n").encode())
            except OSError:
                pass
        finally:
            conn.close()


def _call(sock_path, msg, timeout=60):
    """One JSON line to the zygote, its JSON reply back (OSError when it does not answer)."""
    return ipc.unix_json_call(sock_path, msg, timeout=timeout)


def request(sock_path, argv, env, log, cwd=None, accept=()):
    """Ask the zygote for a new `vllm serve` process. Returns the reply: {"ok": True, "pid"} or
    {"ok": False, "findings": [...]} (accept=("env", "stale") overrides those findings)."""
    return _call(sock_path, {"cmd": "fork", "argv": argv, "env": env, "log": log,
                             "cwd": cwd or os.getcwd(), "accept": list(accept)})


def parse(sock_path, argv, accept=(), timeout=60, cwd=None):
    """Read a `vllm serve` argv with the zygote's vLLM parser: vllm_args.parse()'s reply, plus
    "zygote_seconds" (and "refreshed" when edited modules were re-imported for the parse). Edits
    a refresh cannot apply refuse it like a fork: {"ok": False, "findings": [stale]}, unless
    accept has "stale". A failure that is not vLLM's verdict on the argv (the refresh or the
    parser's import raised, the child died) is {"ok": False, "error", "zygote_error": True}."""
    return _call(sock_path, {"cmd": "parse", "argv": list(argv), "accept": list(accept),
                             "cwd": cwd or os.getcwd()},
                 timeout=timeout)


def check(sock_path, env, accept=()):
    """Would the zygote fork a server with this environment now? Same checks as `request`
    (environment, stale sources), nothing forked: {"ok", "findings", "accepted_findings"}."""
    return _call(sock_path, {"cmd": "check", "env": env, "accept": list(accept)})


# Why a Unix socket cannot be bound at this path (sun_path: 107 bytes on Linux), or None.
MAX_SOCK_PATH = ipc.MAX_SOCK_PATH
sock_path_error = ipc.sock_path_error


def _exited(pid):
    """True when pid is gone or a zombie (a dead child its parent has not reaped yet)."""
    return not proc.pid_alive(pid)


def wait_ready(sock_path, pid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _exited(pid):
            return {"ready": False, "error": "zygote exited"}
        try:
            if _call(sock_path, {"cmd": "ping"}, timeout=5).get("ok"):
                return {"ready": True, "seconds": round(time.time() - t0, 2)}
        except OSError:
            pass
        time.sleep(0.5)
    return {"ready": False, "error": "timeout"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["serve", "ping"])
    p.add_argument("--sock", required=True)
    p.add_argument("--modules", default="")
    a = p.parse_args()
    if a.cmd == "serve":
        serve(a.sock, a.modules)
    else:
        print(json.dumps(_call(a.sock, {"cmd": "ping"})))
