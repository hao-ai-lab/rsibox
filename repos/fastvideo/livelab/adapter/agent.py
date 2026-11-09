"""Worker side of the FastVideo live lab: one kernel participant per worker process.

FastVideo reaches its workers with `executor.collective_rpc(method, args=...)`. A callable
`method` runs as `method(worker, *args)`: the multiprocess executor pickles it (by reference,
so this module must be importable in the worker) to each spawned worker's busy loop, which
takes it between two requests. `rpc` below is that callable. It owns the one
`LocalParticipant` of this process and answers the kernel's participant calls (prepare,
validate, commit, abort, undo, proof, witness_check, reset, fingerprint, at_base, status),
plus `ping`, `inject` (conformance faults) and `repl` (run a cell with the worker in scope).

Never raises for a participant call: the kernel's replies are ok-dicts. `lost` makes every
call raise ConnectionError until the process is respawned, as a dead worker would.
"""
import importlib.util
import linecache
import os
import time
import traceback

_STATE = {"participant": None, "faults": {}, "hooks": []}


def rank_of(worker):
    for attr in ("rank", "local_rank"):
        v = getattr(worker, attr, None)
        if isinstance(v, int):
            return v
    return 0


def _resets(worker):
    def cuda_cache():
        try:
            import torch
        except ImportError:
            return "no torch"
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return "cuda cache emptied"
    return {"R3": cuda_cache}


def participant(worker):
    p = _STATE["participant"]
    if p is None:
        from kernel import coordinator as C
        r = rank_of(worker)
        scope = {"worker": worker, "pipeline": getattr(worker, "pipeline", None)}
        p = _STATE["participant"] = C.LocalParticipant(f"rank{r}", scope=scope,
                                                       resets=_resets(worker))
    return p


def _transformer(worker):
    pipe = getattr(worker, "pipeline", None)
    mods = getattr(pipe, "modules", {}) or {}
    return mods.get("transformer")


def inject(worker, fault, **kw):
    """Conformance faults, injected in this worker. Returns an ok-dict."""
    p = participant(worker)
    if fault == "fail_apply":
        j, after, n = p.journal, kw.get("after", 1), [0]
        orig = j.set_code

        def set_code(*a, **k):
            n[0] += 1
            if n[0] > after:
                del j.set_code                      # one shot
                raise RuntimeError(f"{p.name}: injected fault after {after} swap(s)")
            return orig(*a, **k)
        j.set_code = set_code
    elif fault == "fail_commit":
        q = p.quiesce

        def failing(fn):
            p.quiesce = q
            raise RuntimeError(f"{p.name}: injected commit failure")
        p.quiesce = failing
    elif fault in ("crash", "hang"):
        t = _transformer(worker)
        if t is None:
            return {"ok": False, "error": "no transformer module to hook"}
        hang_s = float(kw.get("hang_s", 30.0))

        def hook(module, args):
            for h in list(_STATE["hooks"]):
                h.remove()
            _STATE["hooks"].clear()
            if fault == "crash":
                raise RuntimeError(f"{p.name}: injected crash in the DiT forward")
            time.sleep(hang_s)
        _STATE["hooks"].append(t.register_forward_pre_hook(hook))
    elif fault == "lost":
        _STATE["faults"]["lost"] = True
    elif fault == "tamper":
        import sys
        mod = sys.modules[kw.get("module", "fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_decoding")]
        fn = getattr(mod, kw.get("function", "_layout"))
        # a REPL edit, not journaled: same behavior, another code object
        fn.__code__ = fn.__code__.replace(co_name=fn.__code__.co_name)
    else:
        return {"ok": False, "error": f"unknown fault {fault}"}
    return {"ok": True, "fault": fault, "participant": p.name, "pid": os.getpid()}


def repl(worker, code):
    """Run a cell in this worker (the notebook verb). `out` in the cell's scope is returned."""
    scope = {"worker": worker, "pipeline": getattr(worker, "pipeline", None),
             "participant": participant(worker), "out": None}
    try:
        exec(compile(code, "<fastrsi-repl>", "exec"), scope)
        return {"ok": True, "out": repr(scope.get("out"))[:20000]}
    except Exception:
        return {"ok": False, "error": traceback.format_exc()[-4000:]}


def _file_of(module):
    spec = importlib.util.find_spec(module)
    return spec.origin if spec else None


def _show_lines(module, text):
    """Make `linecache` (what inspect.getsource, Triton's @jit and Dynamo read) hold `text` for
    the module's file; None: drop the entry, so the file on disk is read again."""
    f = _file_of(module)
    if f is None:
        return
    if text is None:
        linecache.cache.pop(f, None)
    else:
        linecache.cache[f] = (len(text), None, text.splitlines(True), f)


def _sync_lines(p, modules):
    """After a transaction step, each module's lines are the text that is live now."""
    for m in modules:
        _show_lines(m, p._live_source(m))


def _plan_modules(plan):
    return list((plan or {}).get("modules", {})) if isinstance(plan, dict) else []


def rpc(worker, method, payload=None):
    """The collective_rpc target. `worker` is the FastVideo worker of this process."""
    if _STATE["faults"].get("lost"):
        raise ConnectionError(f"rank{rank_of(worker)}: process lost (injected)")
    if method == "ping":
        return {"ok": True, "rank": rank_of(worker), "pid": os.getpid()}
    if method == "inject":
        return inject(worker, **(payload or {}))
    if method == "repl":
        return repl(worker, payload)
    p = participant(worker)
    if method == "prepare":
        # stage compiles the new module text: decorators that read source at definition time
        # (Triton's @jit, torch.compile) must see the new text, not the file on disk
        mods = _plan_modules(payload)
        for m, spec in (payload or {}).get("modules", {}).items():
            _show_lines(m, spec.get("source"))
        r = p.prepare(payload)
        _STATE.setdefault("plans", {})[r.get("txn")] = mods
        if not r.get("ok"):
            _sync_lines(p, mods)
        return r
    r = getattr(p, method)(payload)
    if method in ("commit", "abort", "undo"):
        # an undo also takes back later transactions: resync every module ever planned
        _sync_lines(p, {m for ms in _STATE.get("plans", {}).values() for m in ms})
    return r
