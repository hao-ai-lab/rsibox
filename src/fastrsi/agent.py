"""The in-process agent. Standard library only: this file loads inside the engine.

One daemon thread blocks on a private Unix socket, so an idle agent runs no Python.
The host speaks one JSON request per connection.
"""

import atexit
import hashlib
import json
import marshal
import os
import resource
import socket
import sys
import threading
import traceback
import types

TOOL = 4
PY_START = sys.monitoring.events.PY_START


def identity(code):
    """Hash of a code object. Marshal v2 omits refcount-dependent flags."""
    blob = marshal.dumps(code.replace(co_filename=""), 2)
    return hashlib.sha256(blob).hexdigest()[:16]


def code_map(code):
    found = {}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            found[const.co_qualname] = const
            found.update(code_map(const))
    return found


def resolve(module, qualname):
    obj = module
    for part in qualname.split("."):
        namespace = vars(obj) if hasattr(obj, "__dict__") else {}
        if part not in namespace:
            return None
        obj = namespace[part]
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        elif isinstance(obj, property):
            obj = obj.fget
    return obj if isinstance(obj, types.FunctionType) else None


def find_module(path):
    suffix = os.sep + path.lstrip(os.sep)
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if isinstance(name, str) and name.endswith(suffix):
            return module
    return None


class Agent:
    def __init__(self, directory):
        self.directory = directory
        self.pending = {}
        self.live = []
        self.hits = {}
        self.count_every_call = False
        self.monitoring = False
        self.path = os.path.join(directory, f"{os.getpid()}.sock")

    def serve(self):
        server = socket.socket(socket.AF_UNIX)
        if os.path.exists(self.path):
            os.unlink(self.path)
        server.bind(self.path)
        os.chmod(self.path, 0o600)
        server.listen(16)
        atexit.register(self.close)
        while True:
            connection, _ = server.accept()
            with connection:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = connection.recv(1 << 20)
                    if not chunk:
                        break
                    data += chunk
                if data:
                    connection.sendall(json.dumps(self.handle(json.loads(data))).encode() + b"\n")

    def close(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def handle(self, request):
        try:
            return {"ok": True, **getattr(self, "op_" + request["op"])(request)}
        except Exception as error:
            return {"ok": False, "error": repr(error), "trace": traceback.format_exc(limit=4)}

    def op_ping(self, request):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"pid": os.getpid(), "argv": sys.argv[:3], "maxrss": usage.ru_maxrss,
                "live": len(self.live), "threads": threading.active_count()}

    def op_present(self, request):
        return {"present": [p for p in request["paths"] if find_module(p)]}

    def op_modules(self, request):
        return {"modules": sorted(
            getattr(module, "__file__", None) for module in list(sys.modules.values())
            if request["pattern"] in (getattr(module, "__file__", None) or ""))}

    def op_prepare(self, request):
        if request.get("fault") == "prepare":
            raise RuntimeError("injected fault in prepare")
        plan = []
        absent = []
        for path, spec in request["files"].items():
            module = find_module(path)
            if module is None:
                absent.append(path)
                continue
            base = code_map(compile(spec["base"], module.__file__, "exec", dont_inherit=True))
            new = code_map(compile(spec["new"], module.__file__, "exec", dont_inherit=True))
            for qualname in spec["functions"]:
                function = resolve(module, qualname)
                if function is None:
                    raise ValueError(f"{qualname}: not a plain function in this process")
                if identity(function.__code__) != identity(base[qualname]):
                    raise ValueError(f"{qualname}: live code does not match the base source")
                if function.__code__.co_freevars != new[qualname].co_freevars:
                    raise ValueError(f"{qualname}: closure variables changed")
                plan.append((path, qualname, function, function.__code__, new[qualname]))
        self.pending[request["txn"]] = plan
        self.count_every_call = request.get("witness") == "count"
        return {"absent": absent, "functions": {
            f"{path}:{qualname}": {"base": identity(old), "new": identity(new)}
            for path, qualname, _, old, new in plan}}

    def op_abort(self, request):
        self.pending.pop(request["txn"], None)
        return {}

    def op_commit(self, request):
        plan = self.pending.pop(request["txn"])
        fault = request.get("fault")
        if fault == "exit":
            os._exit(9)
        done = []
        try:
            for index, entry in enumerate(plan):
                if fault == "commit" and index == len(plan) - 1:
                    raise RuntimeError("injected fault in commit")
                entry[2].__code__ = entry[4]
                done.append(entry)
        except Exception:
            for entry in done:
                entry[2].__code__ = entry[3]
            raise
        self.live = plan
        self.arm()
        return {"swapped": len(plan)}

    def op_undo(self, request):
        self.disarm()
        for entry in self.live:
            entry[2].__code__ = entry[3]
        self.live = []
        return {}

    def op_identity(self, request):
        return {"functions": {
            f"{path}:{qualname}": identity(resolve(find_module(path), qualname).__code__)
            for path, qualname in request["functions"] if find_module(path)}}

    def op_witness(self, request):
        return {"hits": {name: hits for name, hits in self.hits.values()}}

    def on_start(self, code, offset):
        entry = self.hits.get(id(code))
        if entry is None:
            return sys.monitoring.DISABLE
        entry[1] += 1
        return None if self.count_every_call else sys.monitoring.DISABLE

    def arm(self):
        if not self.monitoring:
            sys.monitoring.use_tool_id(TOOL, "fastrsi")
            sys.monitoring.register_callback(TOOL, PY_START, self.on_start)
            self.monitoring = True
        self.hits = {}
        for path, qualname, _, _, new in self.live:
            self.hits[id(new)] = [f"{path}:{qualname}", 0]
            sys.monitoring.set_local_events(TOOL, new, PY_START)
        sys.monitoring.restart_events()

    def disarm(self):
        for _, _, _, _, new in self.live:
            sys.monitoring.set_local_events(TOOL, new, 0)
        self.hits = {}


_agent = None


def _run():
    try:
        _agent.serve()
    except OSError as error:
        print(f"fastrsi: agent failed in pid {os.getpid()}: {error}", file=sys.stderr, flush=True)


def _serve():
    threading.Thread(target=_run, name="fastrsi-agent", daemon=True).start()


def _after_fork():
    _agent.path = os.path.join(_agent.directory, f"{os.getpid()}.sock")
    _serve()


def boot(directory):
    global _agent
    if _agent is None:
        _agent = Agent(directory)
        os.register_at_fork(after_in_child=_after_fork)
        _serve()
