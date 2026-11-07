"""Weights holder: loaded components that outlive the worker processes, for the zygote
respawn with weights holder.

A respawned worker forked from the zygote (`zygote.py`) still loads its components from disk:
11-12 s of a 14 s respawn on one GB200 (`kb/costs.md`). The holder is a long-lived process that
owns a copy of the loaded components (the DiT, the text encoder and the VAE by default) on the GPU. The
first worker loads them from disk and deposits them (`put`); a respawned worker asks for them
(`get`) and receives the module objects with their CUDA tensors mapped by CUDA IPC
(`torch.multiprocessing`): no file read, no host-to-device copy, no new GPU memory.

Never stale. Each entry carries a stamp: the content hash of every source file that decides
what the loaded tensors are (the files of the classes in the component's module tree, FastVideo's
loader and quantization code), the size and mtime of the checkpoint files, and the load-time
environment. `get` recomputes it from the files as they are now; any difference is a refusal
(`holder_stale`, naming the files), the worker loads from disk, and its deposit replaces the
entry. An edit of the loader, the quantization or a model class is never served from the holder.

What the worker receives is the module object: its classes are looked up by name in the
worker's (possibly refreshed) modules and its attributes restored, as a class swap does: method
edits apply, constructor edits do not (the stamp refuses those, since they live in a stamped
file). CPU tensors in the tree travel through shared memory (a copy, not pinned).

Transport: `multiprocessing.connection` over a unix socket (no network port), ForkingPickler.
"""
import hashlib
import json
import os
import sys
import time

DEFAULT_COMPONENTS = ("transformer", "text_encoder", "vae")   # audio_vae: a non-leaf grad tensor, not shareable
STAMP_PREFIXES = ("fastvideo.models.loader", "fastvideo.layers.quantization")
STAMP_ENV = ("FASTVIDEO_H3_ADALN_TABLE", "FASTVIDEO_H3_VAE_NVFP4_DECODER", "FASTVIDEO_ATTENTION_BACKEND",
             "FASTVIDEO_MINIMAX_H3_FUSIONS", "FASTVIDEO_H3_VSA_FP4", "FASTVIDEO_FA4")


def finding(check, observed, expected):
    return {"check": check, "observed": observed, "expected": expected}


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def stamp_files(module=None, modules=None, prefixes=STAMP_PREFIXES):
    """Source files whose content decides the tensors of a component: the modules of every class
    in its module tree (and their bases), plus every loaded module under `prefixes`."""
    modules = sys.modules if modules is None else modules
    names = set()
    if module is not None:
        objs = list(module.modules()) if callable(getattr(module, "modules", None)) else [module]
        for o in objs:
            for c in type(o).__mro__:
                if c.__module__ and not c.__module__.startswith(("torch", "builtins")):
                    names.add(c.__module__)
    names.update(n for n in list(modules) if n.startswith(prefixes))
    files = {}
    for n in sorted(names):
        f = getattr(modules.get(n), "__file__", None)
        if f and f.endswith(".py"):
            files[n] = f
    return files


def stamp(files, weights_dir=None, extra=None):
    """{"files": {module: content hash}, "weights": {file: [size, mtime]}, "env": {...}, "extra"}"""
    out = {"files": {}, "weights": {}, "env": {k: os.environ.get(k) for k in STAMP_ENV},
           "extra": extra}
    for n, f in sorted(files.items()):
        try:
            out["files"][n] = _file_hash(f)
        except OSError:
            out["files"][n] = None
    if weights_dir and os.path.isdir(weights_dir):
        for name in sorted(os.listdir(weights_dir)):
            p = os.path.join(weights_dir, name)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out["weights"][name] = [st.st_size, st.st_mtime_ns]
    return out


def stamp_diff(old, new):
    """What differs between two stamps (empty: the same)."""
    diff = []
    for part in ("files", "weights", "env"):
        a, b = old.get(part) or {}, new.get(part) or {}
        diff += [f"{part}:{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if old.get("extra") != new.get("extra"):
        diff.append("extra")
    return diff


# -- the holder process --------------------------------------------------------------------
def _own(module):
    """A copy of a received module (its tensors IPC-mapped from the sender) whose every tensor,
    nested ones included (lists, dicts, helper objects), this process owns: the sender may
    exit, and the copy can be sent on (torch refuses to forward a received CUDA tensor)."""
    import copy
    import torch
    owned = copy.deepcopy(module)       # the memo keeps tied tensors tied
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return owned


def _ipc():
    """CUDA tensors travel as IPC handles; CPU tensors through named shared memory (no fd
    passing between processes that do not share a parent's resource sharer)."""
    import torch.multiprocessing as tmp
    tmp.set_sharing_strategy("file_system")


def serve(sock_path, ready_path=None):
    """Run the holder: one request per connection: {"op": "put"|"get"|"status"|"stop", ...}."""
    _ipc()
    from multiprocessing.connection import Listener
    entries = {}
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    with Listener(sock_path, family="AF_UNIX") as ls:
        if ready_path:
            with open(ready_path, "w") as f:
                json.dump({"pid": os.getpid(), "sock": sock_path}, f)
        while True:
            with ls.accept() as conn:
                try:
                    if _handle(conn, entries) == "stop":
                        return
                except Exception as e:                # one bad request never stops the holder
                    try:
                        conn.send({"ok": False, "findings": [finding(
                            "holder_error", f"{type(e).__name__}: {e}", "the request served")]})
                    except Exception:
                        pass


def _handle(conn, entries):
    """One request: put, get (stamp-checked), files, status or stop."""
    req = conn.recv()
    op, key = req.get("op"), req.get("key")
    t0 = time.perf_counter()
    if op == "put":
        entries[key] = {"stamp": req["stamp"], "module": _own(req["module"]),
                        "files": req.get("files")}
        conn.send({"ok": True, "op": op, "key": key, "seconds": round(time.perf_counter() - t0, 4)})
    elif op == "get":
        e = entries.get(key)
        if e is None:
            conn.send({"ok": False, "findings": [finding("holder_empty", f"no entry {key}",
                                                          "a deposit before the first get")]})
        else:
            diff = stamp_diff(e["stamp"], req["stamp"])
            if diff:
                del entries[key]          # never serve it again
                conn.send({"ok": False, "findings": [finding(
                    "holder_stale", f"{key}: changed since the deposit: {diff[:8]}",
                    "the loader, quantization, model classes, weights and env unchanged")]})
            else:
                conn.send({"ok": True, "module": e["module"], "files": e["files"]})
                conn.recv()               # the worker mapped the tensors
    elif op == "files":
        e = entries.get(key)
        conn.send({"ok": e is not None, "files": e["files"] if e else None})
    elif op == "status":
        conn.send({"ok": True, "keys": sorted(map(str, entries)), "pid": os.getpid()})
    elif op == "stop":
        conn.send({"ok": True})
        return "stop"
    else:
        conn.send({"ok": False, "error": f"unknown op {op}"})


def _call(sock_path, msg, timeout=120.0):
    _ipc()
    from multiprocessing.connection import Client
    c = Client(sock_path, family="AF_UNIX")
    try:
        c.send(msg)
        if not c.poll(timeout):
            raise TimeoutError(f"holder: no reply to {msg.get('op')} in {timeout} s")
        r = c.recv()
        if msg.get("op") == "get" and r.get("ok"):
            c.send("mapped")
        return r
    finally:
        c.close()


def start(sock_path, ctx=None):
    """Start the holder as a spawned process of the caller (the lab host). Returns the process."""
    import multiprocessing
    ctx = ctx or multiprocessing.get_context("spawn")
    ready = sock_path + ".ready"
    if os.path.exists(ready):
        os.unlink(ready)
    p = ctx.Process(target=serve, args=(sock_path, ready), daemon=True, name="fastrsi-holder")
    p.start()
    t0 = time.time()
    while not os.path.exists(ready):
        if not p.is_alive() or time.time() - t0 > 300:
            raise RuntimeError("holder did not start")
        time.sleep(0.05)
    return p


# -- the worker side -----------------------------------------------------------------------
def install_loader_hook(sock_path, components=DEFAULT_COMPONENTS, report=None):
    """In a worker, before it loads: `PipelineComponentLoader.load_module` asks the holder for
    each component in `components` and loads from disk (then deposits) when refused."""
    from fastvideo.models.loader import component_loader as CL
    env = os.environ.get("FASTRSI_HOLDER_COMPONENTS")
    components = tuple(c for c in env.split(",") if c) if env else tuple(components)
    orig = CL.PipelineComponentLoader.__dict__["load_module"]
    orig = orig.__func__ if isinstance(orig, staticmethod) else orig
    report = report if report is not None else []

    def load_module(module_name, component_model_path, transformers_or_diffusers, fastvideo_args):
        if module_name not in components:
            return orig(module_name, component_model_path, transformers_or_diffusers, fastvideo_args)
        rank = int(os.environ.get("RANK", "0"))
        key = f"rank{rank}:{module_name}"
        t0 = time.perf_counter()
        ev = {"key": key}
        try:
            prev = _call(sock_path, {"op": "files", "key": key})
            if prev.get("ok"):
                st = stamp(prev["files"], component_model_path)
                r = _call(sock_path, {"op": "get", "key": key, "stamp": st})
                if r.get("ok"):
                    ev.update(served=True, seconds=round(time.perf_counter() - t0, 4))
                    report.append(ev)
                    _log(ev)
                    return r["module"]
                ev["findings"] = r.get("findings")
        except Exception as e:
            ev["findings"] = [finding("holder_unreachable", f"{type(e).__name__}: {e}", "a holder")]
        module = orig(module_name, component_model_path, transformers_or_diffusers, fastvideo_args)
        ev["load_s"] = round(time.perf_counter() - t0, 4)
        try:
            files = stamp_files(module)
            t1 = time.perf_counter()
            pr = _call(sock_path, {"op": "put", "key": key, "module": module, "files": files,
                                   "stamp": stamp(files, component_model_path)}, timeout=600)
            ev.update(deposited=pr.get("ok"), deposit_s=round(time.perf_counter() - t1, 4))
        except Exception as e:
            ev["deposit_error"] = f"{type(e).__name__}: {e}"
        ev.update(served=False)
        report.append(ev)
        _log(ev)
        return module
    CL.PipelineComponentLoader.load_module = staticmethod(load_module)
    return report


def _log(ev):
    print(f"[fastrsi-holder] pid {os.getpid()} {json.dumps(ev, default=str)[:1500]}", file=sys.stderr, flush=True)
