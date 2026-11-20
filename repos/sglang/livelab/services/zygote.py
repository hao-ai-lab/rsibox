"""Zygote for SGLang: pre-imports torch and SGLang without CUDA, listens on a short AF_UNIX
socket, and starts each server (or callable) by fork instead of exec. SGLang spawns its
schedulers (`engine.py`), so only the server inherits; see README.md for guards (a)-(d)."""
import argparse, contextlib, importlib, importlib.util, json, linecache, os, runpy
import signal, socket, sys, time, tokenize, traceback

LIVELAB = os.path.abspath(os.environ.get("FASTRSI_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
import forksafe  # noqa: E402
from fastrsi import ipc, proc  # noqa: E402

DEFAULT_PRELOAD = ("torch", "sglang.srt.server_args", "sglang.srt.entrypoints.engine",
                   "sglang.srt.entrypoints.http_server", "sglang.srt.managers.tokenizer_manager")
MAX_SOCK_PATH = min(ipc.MAX_SOCK_PATH, 106)   # <107 bytes, fits sun_path + NUL

def sock_path_error(sock_path):
    """(d) Reject an AF_UNIX socket path exceeding MAX_SOCK_PATH (<107 bytes) pre-import."""
    if err := ipc.sock_path_error(sock_path): return err
    length = len(os.fsencode(os.path.abspath(sock_path)))
    return (f"socket path too long for AF_UNIX ({length} bytes > {MAX_SOCK_PATH}): {sock_path}"
            if length > MAX_SOCK_PATH else None)

def install_guards(modules):
    """Install pre-import guards (a) and (b) via `forksafe.install_all_guards`."""
    report, restore = forksafe.install_all_guards(
        properties=os.environ.get("FASTRSI_ZYGOTE_PROPS", "1") == "1",
        import_torch="torch" in modules,
    )
    report["forksafe"] = report.get("torch_cuda", {"installed": False})
    report["cuinit_stub"] = bool(report.get("cuda_python", {}).get("installed"))
    return report, restore

def preload(modules):
    """Import each module; SystemExit naming the first one that initialized CUDA."""
    took = {}
    for name in modules:
        start = time.time()
        try:
            importlib.import_module(name)
        except ImportError as exc:
            took[name] = f"skipped: {exc!r}"[:200]; continue
        took[name] = round(time.time() - start, 3)
        if forksafe.cuda_initialized():
            raise SystemExit(f"zygote: importing {name} initialized CUDA; fork cannot use GPU")
    return took

def _stat(path):
    with contextlib.suppress(OSError):
        return (lambda info: (info.st_mtime_ns, info.st_size))(os.stat(path))

def refresh_child(changed, prefixes):
    """(c) Drop edited packages/.pyc and evict stale linecache lines (including mtime=None)."""
    dropped = [n for n in list(sys.modules) if changed
               and any(n == p or n.startswith(p + ".") for p in prefixes)]
    for name in dropped:
        module_file = getattr(sys.modules.pop(name, None), "__file__", None)
        parent, _, attr = name.rpartition(".")
        getattr(sys.modules.get(parent), "__dict__", {}).pop(attr, None)
        with contextlib.suppress(OSError, ValueError, NotImplementedError):
            os.unlink(importlib.util.cache_from_source(module_file or ""))
    if dropped: importlib.invalidate_caches()
    stale_text = []
    for path, entry in list(linecache.cache.items()):
        if len(entry) == 4 and os.path.isfile(path):
            with contextlib.suppress(OSError, SyntaxError, UnicodeDecodeError):
                with tokenize.open(path) as handle:
                    if handle.read() == "".join(entry[2]):
                        continue
            linecache.cache.pop(path, None); stale_text.append(path)
    linecache.checkcache()
    return {"dropped": len(dropped), "stale_linecache": stale_text}

def children():
    """(d) Direct child PIDs by ppid, never process group: children call setsid (Megatron M-H17)."""
    return proc.direct_children()

def reap_exited(exits):
    with contextlib.suppress(ChildProcessError):
        while (reaped := os.waitpid(-1, os.WNOHANG))[0]:
            exits[str(reaped[0])] = os.waitstatus_to_exitcode(reaped[1])
    return exits

def kill_children(grace=5.0):
    pids, exits = children(), {}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            proc.kill_tree(pid, sig, include_pg=True)
        deadline = time.time() + grace
        while time.time() < deadline and any(str(pid) not in exits for pid in pids):
            time.sleep(0.02); reap_exited(exits)
    return {str(pid): exits.get(str(pid)) for pid in pids}

def child_main(req, changed, prefixes, sockets, clear_nvml_env=False):
    code = 1
    try:
        for sock in sockets:
            sock.close()
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_DFL); signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        if req.get("log"):
            log_fd = os.open(req["log"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(log_fd, 1); os.dup2(log_fd, 2); os.close(log_fd)
        os.chdir(req.get("cwd") or os.getcwd())
        unpatched = forksafe.unpatch_child(None, clear_env=clear_nvml_env)["torch_cuda"]
        for key in req.get("unset", ()):
            os.environ.pop(key, None)
        # (b) apply full request env including CUDA_VISIBLE_DEVICES (vLLM zygote lesson)
        os.environ.update({str(key): str(value) for key, value in req.get("env", {}).items()})
        report = refresh_child(changed, prefixes)
        print(json.dumps({"event": "zygote_child", "pid": os.getpid(), "unpatched": unpatched,
                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          **report}), flush=True)
        entry, argv = req.get("entry") or {"module": "sglang.launch_server"}, req.get("argv", [])
        if "callable" in entry:
            module_name, _, function_name = entry["callable"].partition(":")
            sys.argv = [module_name] + list(argv)
            getattr(importlib.import_module(module_name), function_name)(*entry.get("args", []), **entry.get("kwargs", {}))
        else:
            sys.argv = [entry["module"]] + list(argv)
            runpy.run_module(entry["module"], run_name="__main__", alter_sys=True)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc()
    finally:
        with contextlib.suppress(BaseException):
            sys.stdout.flush(); sys.stderr.flush()
        os._exit(code)

def fork_request(req, snap, prefixes, sockets, clear_nvml_env=False):
    in_scope = lambda n: any(n == p or n.startswith(p + ".") for p in prefixes)
    changed = sorted(path for path, (_, sig) in snap.items() if _stat(path) != sig)
    blocked = [p for p in changed if not (p.endswith(".py") and in_scope(snap[p][0]))]
    if blocked and "stale" not in req.get("accept", []):
        return {"ok": False, "findings": [{"check": "stale", "files": blocked[:20]}]}
    sys.stdout.flush(); sys.stderr.flush()
    refreshed = [p for p in changed if p not in blocked]
    if (pid := os.fork()) == 0: child_main(req, refreshed, prefixes, sockets, clear_nvml_env=clear_nvml_env)
    return {"ok": True, "pid": pid, "refreshed": refreshed}

def serve(sock_path, modules=DEFAULT_PRELOAD, prefixes=("sglang",)):
    if error := sock_path_error(sock_path): raise SystemExit(error)
    start = time.time()
    guards, restore_triton = install_guards(modules)
    clear_nvml_env = bool(guards.get("nvml_env_set"))
    try:
        took = preload(modules)
    finally:
        restore_triton()
    snap = {module.__file__: (name, _stat(module.__file__)) for name, module
            in list(sys.modules.items()) if getattr(module, "__file__", None)}
    if os.environ.get("FASTRSI_ZYGOTE_GC_FREEZE", "1") == "1": forksafe.freeze_gc_before_fork()
    with contextlib.suppress(FileNotFoundError): os.unlink(sock_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path); os.chmod(sock_path, 0o600); listener.listen(16); listener.settimeout(0.5)
    print(json.dumps({"event": "zygote_ready", "pid": os.getpid(), "took": took, "guards": guards,
                      "seconds": round(time.time() - start, 3)}), flush=True)
    exits, cmd = {}, None
    while cmd != "stop":
        reap_exited(exits)
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            continue
        with conn:
            cmd = None
            try:
                conn.settimeout(30)
                with conn.makefile() as reader:
                    req = json.loads(reader.readline() or "null"); reap_exited(exits)
                cmd = req.get("cmd", "fork") if isinstance(req, dict) else None
                if cmd == "ping":
                    reply = {"ok": True, "pid": os.getpid(), "children": children(), "exits": exits}
                elif cmd in ("kill", "stop"):
                    exits.update(killed := kill_children()); reply = {"ok": True, "killed": killed}
                elif cmd == "fork":
                    reply = fork_request(req, snap, prefixes, [listener, conn], clear_nvml_env=clear_nvml_env)
                else:
                    reply = {"ok": False, "error": f"bad request: {req!r}"[:300]}
            except Exception as exc:
                reply = {"ok": False, "error": repr(exc)}
            print(json.dumps({"event": "request", "cmd": cmd, **reply}), flush=True)
            conn.sendall((json.dumps(reply) + "\n").encode())
    listener.close()
    with contextlib.suppress(FileNotFoundError): os.unlink(sock_path)

def call(sock_path, message, timeout=60):
    return ipc.unix_json_call(sock_path, message, timeout=timeout)

def request(sock_path, argv=(), env=None, log="", entry=None, cwd=None, accept=(), unset=()):
    return call(sock_path, {"cmd": "fork", "argv": list(argv), "env": dict(env or {}), "log": log,
                "entry": entry, "cwd": cwd or os.getcwd(), "accept": list(accept), "unset": list(unset)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="zygote.py")
    for flag, kw in (("cmd", {"choices": ("serve", "fork", "ping", "kill", "stop")}),
                     ("--sock", {"required": True}), ("--preload", {"default": ",".join(DEFAULT_PRELOAD)}),
                     ("--refresh", {"default": "sglang"}), ("--log", {"default": ""}),
                     ("--env", {"action": "append", "default": []}), ("argv", {"nargs": "*"})):
        parser.add_argument(flag, **kw)
    args = parser.parse_args()
    if args.cmd == "serve":
        serve(args.sock, [n for n in args.preload.split(",") if n], tuple(n for n in args.refresh.split(",") if n))
    else:
        print(json.dumps(call(args.sock, {"cmd": args.cmd, "argv": args.argv, "log": args.log,
              "cwd": os.getcwd(), "env": dict(kv.split("=", 1) for kv in args.env)})))
