"""Zygote for SGLang: pre-imports torch and SGLang without CUDA, listens on a short AF_UNIX
socket, and starts each server (or callable) by fork instead of exec. SGLang spawns its
schedulers (`engine.py`), so only the server inherits; see README.md for guards (a)-(d)."""
import argparse, ast, contextlib, importlib, json, linecache, multiprocessing, os, runpy
import signal, socket, sys, threading, time, tokenize, traceback

LIVELAB = os.path.abspath(os.environ.get("FASTRSI_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
import forksafe  # noqa: E402
from fastrsi import hooks, ipc, op_guard, proc  # noqa: E402

DEFAULT_PRELOAD = ("torch", "sglang.srt.server_args", "sglang.srt.entrypoints.engine",
                   "sglang.srt.entrypoints.http_server", "sglang.srt.managers.tokenizer_manager")
SUBPROC_PRELOAD = ("sglang.srt.managers.scheduler", "sglang.srt.managers.detokenizer_manager")
WHOLE_REFRESH_ABOVE = 100
TORCH_OP_STATE_MODULES = frozenset((
    "sglang.srt.distributed.parallel_state",
    "sglang.srt.utils.custom_op",
    "sglang.srt.utils.common",
    "sglang.srt.batch_invariant_ops.batch_invariant_ops",
))
_CUSTOM_OP_IMPORT_MODS = frozenset(("sglang.srt.utils.custom_op", "torch.library"))
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

def plan_refresh(changed, prefixes, graph=None, whole_above=WHOLE_REFRESH_ABOVE):
    """Return `(modules_to_drop, scope)` where `scope` is 'none', 'dependents', or 'all'."""
    if not changed:
        return [], "none"
    in_scope = lambda n: any(n == p or n.startswith(p + ".") for p in prefixes)
    changed_set = set(changed)
    changed_real = {os.path.realpath(path) for path in changed_set}
    changed_names = [
        name for name, mod in list(sys.modules.items())
        if in_scope(name) and (mod_file := getattr(mod, "__file__", None)) is not None
        and (mod_file in changed_set or os.path.realpath(mod_file) in changed_real)
    ]
    if graph and changed_names:
        deps = {n for n in hooks.dependents(changed_names, graph) if in_scope(n)} | set(changed_names)
        if len(deps) <= whole_above:
            return sorted(deps), "dependents"
    return sorted(n for n in list(sys.modules) if in_scope(n)), "all"


def refresh_child(changed, prefixes, *, graph=None, whole_above=WHOLE_REFRESH_ABOVE, plan=None):
    """(c) Drop edited modules (or AST dependents) + .pyc and evict stale linecache lines."""
    to_drop, scope = (
        plan_refresh(changed, prefixes, graph=graph, whole_above=whole_above)
        if plan is None else plan
    )
    dropped = hooks.purge_modules(to_drop, unlink_pyc=True)
    stale_text = []
    for path, entry in list(linecache.cache.items()):
        if len(entry) == 4 and os.path.isfile(path):
            with contextlib.suppress(OSError, SyntaxError, UnicodeDecodeError):
                with tokenize.open(path) as handle:
                    if handle.read() == "".join(entry[2]):
                        continue
            linecache.cache.pop(path, None); stale_text.append(path)
    linecache.checkcache()
    return {"dropped": len(dropped), "scope": scope, "dropped_modules": dropped[:20],
            "stale_linecache": stale_text}

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

def _source_registers_torch_ops(source, filename="<string>"):
    """True when Python source imports or invokes custom torch op registration APIs."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return any(tok in source for tok in ("sglang.srt.utils.custom_op", "torch.library", "direct_register_custom_op"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _CUSTOM_OP_IMPORT_MODS or alias.name.startswith("torch.library."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in _CUSTOM_OP_IMPORT_MODS or mod.startswith("torch.library."):
                return True
            names = {alias.name for alias in node.names}
            if (mod == "torch" and "library" in names) or (mod == "sglang.srt.utils" and "custom_op" in names):
                return True
            if "direct_register_custom_op" in names:
                return True
            if mod.endswith(".custom_op") and ({"direct_register_custom_op", "CustomOp"} & names):
                return True
        elif isinstance(node, ast.Name) and node.id == "direct_register_custom_op":
            return True
        elif isinstance(node, ast.Attribute):
            if node.attr == "direct_register_custom_op":
                return True
            if node.attr == "library" and isinstance(node.value, ast.Name) and node.value.id == "torch":
                return True
    return False


def _drops_rebind_torch_ops(to_drop, refresh_scope, graph=None):
    """True when refreshing `to_drop` may rebind torch custom ops or core distributed/op state."""
    if refresh_scope == "all":
        return True
    for name in to_drop:
        if name in TORCH_OP_STATE_MODULES:
            return True
        if graph and (graph.get(name, set()) & _CUSTOM_OP_IMPORT_MODS):
            return True
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "direct_register_custom_op", None) is not None:
            return True
        mod_file = getattr(mod, "__file__", None) if mod is not None else None
        if mod_file and mod_file.endswith(".py"):
            with contextlib.suppress(OSError, UnicodeDecodeError):
                with tokenize.open(mod_file) as handle:
                    if _source_registers_torch_ops(handle.read(), filename=mod_file):
                        return True
    return False


def enable_fork_subprocs(clear_nvml_env=False):
    """Redirect multiprocessing 'spawn' requests to 'fork' while CUDA stays uninitialized,
    falling back to 'spawn' with a JSON event if CUDA initializes early, and unpatching
    forksafe guards inside each forked child via os.register_at_fork."""
    orig_get_context = multiprocessing.get_context
    orig_set_start_method = multiprocessing.set_start_method
    default_ctx = getattr(getattr(multiprocessing, "context", None), "_default_context", None)
    had_default_get = default_ctx is not None and "get_context" in default_ctx.__dict__
    orig_default_get = default_ctx.__dict__.get("get_context") if had_default_get else None
    had_default_set = default_ctx is not None and "set_start_method" in default_ctx.__dict__
    orig_default_set = default_ctx.__dict__.get("set_start_method") if had_default_set else None
    torch_mp = sys.modules.get("torch.multiprocessing")
    orig_torch_get = getattr(torch_mp, "get_context", None) if torch_mp is not None else None
    orig_torch_set = getattr(torch_mp, "set_start_method", None) if torch_mp is not None else None
    state = {"active": True, "rewrote_default": False}
    tls = threading.local()

    def _emit_fallback(api, requested):
        print(json.dumps({
            "event": "fork_subprocs_fallback", "api": api, "requested": requested,
            "effective": "spawn", "reason": "cuda_initialized", "pid": os.getpid(),
        }), flush=True)

    def _call_orig_set(target, force):
        depth = getattr(tls, "in_set", 0)
        tls.in_set = depth + 1
        try:
            return orig_set_start_method(target, force=force)
        finally:
            tls.in_set = depth

    def _wrapped_set_start_method(method, force=False):
        if method == "spawn":
            if not forksafe.cuda_initialized():
                res = _call_orig_set("fork", force)
                state["rewrote_default"] = True
                return res
            _emit_fallback("set_start_method", method)
            res = _call_orig_set(method, force)
            state["rewrote_default"] = False
            return res
        res = _call_orig_set(method, force)
        state["rewrote_default"] = False
        return res

    def _wrapped_get_context(method=None):
        if getattr(tls, "in_set", 0):
            return orig_get_context(method)
        if method == "spawn":
            if not forksafe.cuda_initialized():
                return orig_get_context("fork")
            _emit_fallback("get_context", method)
            return orig_get_context("spawn")
        if method is None and state["rewrote_default"] and forksafe.cuda_initialized():
            _emit_fallback("get_context", None)
            return orig_get_context("spawn")
        return orig_get_context(method)

    def restore():
        state["active"] = False
        multiprocessing.get_context = orig_get_context
        multiprocessing.set_start_method = orig_set_start_method
        if default_ctx is not None:
            if default_ctx.__dict__.get("get_context") is _wrapped_get_context:
                if had_default_get:
                    default_ctx.get_context = orig_default_get
                else:
                    delattr(default_ctx, "get_context")
            if default_ctx.__dict__.get("set_start_method") is _wrapped_set_start_method:
                if had_default_set:
                    default_ctx.set_start_method = orig_default_set
                else:
                    delattr(default_ctx, "set_start_method")
        cur_torch_mp = sys.modules.get("torch.multiprocessing")
        if cur_torch_mp is not None:
            if getattr(cur_torch_mp, "get_context", None) is _wrapped_get_context:
                cur_torch_mp.get_context = orig_torch_get or orig_get_context
            if getattr(cur_torch_mp, "set_start_method", None) is _wrapped_set_start_method:
                cur_torch_mp.set_start_method = orig_torch_set or orig_set_start_method

    def _after_in_child():
        if not state["active"]:
            return
        restore()
        forksafe.unpatch_child(None, clear_env=clear_nvml_env)

    multiprocessing.get_context = _wrapped_get_context
    multiprocessing.set_start_method = _wrapped_set_start_method
    if default_ctx is not None:
        default_ctx.get_context = _wrapped_get_context
        default_ctx.set_start_method = _wrapped_set_start_method
    if torch_mp is not None:
        if orig_torch_get is not None:
            torch_mp.get_context = _wrapped_get_context
        if orig_torch_set is not None:
            torch_mp.set_start_method = _wrapped_set_start_method
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_in_child)
    return restore

def child_main(req, changed, prefixes, sockets, clear_nvml_env=False, graph=None):
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
        to_drop, refresh_scope = plan_refresh(changed, prefixes, graph=graph)
        fork_subprocs_requested = bool(req.get("fork_subprocs"))
        refresh_rebinds_ops = _drops_rebind_torch_ops(to_drop, refresh_scope, graph=graph)
        fork_subprocs = fork_subprocs_requested and not refresh_rebinds_ops
        if fork_subprocs:
            unpatched = []
            enable_fork_subprocs(clear_nvml_env=clear_nvml_env)
        else:
            unpatched = forksafe.unpatch_child(None, clear_env=clear_nvml_env)["torch_cuda"]
        for key in req.get("unset", ()):
            os.environ.pop(key, None)
        # (b) apply full request env including CUDA_VISIBLE_DEVICES (vLLM zygote lesson)
        os.environ.update({str(key): str(value) for key, value in req.get("env", {}).items()})
        if fork_subprocs:
            # SGLang's NUMA bind v2 exec-wraps `multiprocessing.spawn.get_executable()` and
            # asserts `get_start_method() == "spawn"`; disabling v2 falls back cleanly to
            # SGLang's in-child `numa_bind_to_node()` inside each forked scheduler process.
            os.environ["SGLANG_NUMA_BIND_V2"] = "0"
        report = refresh_child(changed, prefixes, graph=graph, plan=(to_drop, refresh_scope))
        if fork_subprocs_requested and not fork_subprocs:
            print(json.dumps({
                "event": "fork_subprocs_fallback", "api": "refresh_child",
                "requested": "fork", "effective": "spawn",
                "reason": "refresh_rebinds_torch_ops",
                "scope": refresh_scope, "dropped": report["dropped"], "pid": os.getpid(),
            }), flush=True)
        print(json.dumps({"event": "zygote_child", "pid": os.getpid(), "unpatched": unpatched,
                          "fork_subprocs": fork_subprocs,
                          "numa_bind_v2": os.environ.get("SGLANG_NUMA_BIND_V2"),
                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          **report}), flush=True)
        entry, argv = req.get("entry") or {"module": "sglang.launch_server"}, req.get("argv", [])
        guard_ctx = (
            op_guard.op_redefine_guard(lambda ev: print(json.dumps(ev), flush=True))
            if report["dropped"] else contextlib.nullcontext()
        )
        with guard_ctx:
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

def fork_request(req, snap, prefixes, sockets, clear_nvml_env=False, default_fork_subprocs=False, graph=None):
    if default_fork_subprocs and "fork_subprocs" not in req:
        req = {**req, "fork_subprocs": True}
    in_scope = lambda n: any(n == p or n.startswith(p + ".") for p in prefixes)
    changed = sorted(path for path, (_, sig) in snap.items() if _stat(path) != sig)
    blocked = [p for p in changed if not (p.endswith(".py") and in_scope(snap[p][0]))]
    if blocked and "stale" not in req.get("accept", []):
        return {"ok": False, "findings": [{"check": "stale", "files": blocked[:20]}]}
    sys.stdout.flush(); sys.stderr.flush()
    refreshed = [p for p in changed if p not in blocked]
    if (pid := os.fork()) == 0:
        child_main(req, refreshed, prefixes, sockets, clear_nvml_env=clear_nvml_env, graph=graph)
    return {"ok": True, "pid": pid, "refreshed": refreshed}

def serve(sock_path, modules=None, prefixes=("sglang",), fork_subprocs=False):
    if error := sock_path_error(sock_path): raise SystemExit(error)
    if modules is None:
        modules = DEFAULT_PRELOAD + SUBPROC_PRELOAD if fork_subprocs else DEFAULT_PRELOAD
    start = time.time()
    guards, restore_triton = install_guards(modules)
    clear_nvml_env = bool(guards.get("nvml_env_set"))
    try:
        took = preload(modules)
    finally:
        restore_triton()
    snap = {module.__file__: (name, _stat(module.__file__)) for name, module
            in list(sys.modules.items()) if getattr(module, "__file__", None)}
    roots = tuple(sorted({p.split(".")[0] for p in prefixes if p}))
    graph = hooks.import_graph(roots) if roots else {}
    if os.environ.get("FASTRSI_ZYGOTE_GC_FREEZE", "1") == "1": forksafe.freeze_gc_before_fork()
    with contextlib.suppress(FileNotFoundError): os.unlink(sock_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path); os.chmod(sock_path, 0o600); listener.listen(16); listener.settimeout(0.5)
    print(json.dumps({"event": "zygote_ready", "pid": os.getpid(), "took": took, "guards": guards,
                      "fork_subprocs": bool(fork_subprocs),
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
                    reply = fork_request(
                        req, snap, prefixes, [listener, conn],
                        clear_nvml_env=clear_nvml_env, default_fork_subprocs=fork_subprocs,
                        graph=graph,
                    )
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

def request(sock_path, argv=(), env=None, log="", entry=None, cwd=None, accept=(), unset=(), fork_subprocs=None):
    payload = {"cmd": "fork", "argv": list(argv), "env": dict(env or {}), "log": log,
               "entry": entry, "cwd": cwd or os.getcwd(), "accept": list(accept), "unset": list(unset)}
    if fork_subprocs is not None:
        payload["fork_subprocs"] = bool(fork_subprocs)
    return call(sock_path, payload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="zygote.py")
    for flag, kw in (("cmd", {"choices": ("serve", "fork", "ping", "kill", "stop")}),
                     ("--sock", {"required": True}), ("--preload", {"default": None}),
                     ("--refresh", {"default": "sglang"}), ("--log", {"default": ""}),
                     ("--fork-subprocs", {"action": "store_true", "default": False}),
                     ("--env", {"action": "append", "default": []}), ("argv", {"nargs": "*"})):
        parser.add_argument(flag, **kw)
    args = parser.parse_args()
    if args.cmd == "serve":
        modules = None if args.preload is None else [n for n in args.preload.split(",") if n]
        serve(args.sock, modules, tuple(n for n in args.refresh.split(",") if n),
              fork_subprocs=args.fork_subprocs)
    else:
        payload = {"cmd": args.cmd, "argv": args.argv, "log": args.log,
                   "cwd": os.getcwd(), "env": dict(kv.split("=", 1) for kv in args.env)}
        if args.fork_subprocs:
            payload["fork_subprocs"] = True
        print(json.dumps(call(args.sock, payload)))
