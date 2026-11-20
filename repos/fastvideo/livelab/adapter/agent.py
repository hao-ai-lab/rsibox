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
import copy
import linecache
import os
import sys
import time
import traceback

# the kernel lives in the framework's live-lab folder; the package __init__ adds it too, this
# covers a worker that imports this module by path
_LIVELAB = os.path.abspath(os.environ.get("FASTRSI_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if _LIVELAB not in sys.path:
    sys.path.insert(0, _LIVELAB)

from fastrsi import harness, ipc  # noqa: E402

# rebuilt: {module: text} an in-place rebuild made the module's base (what it runs, not its file)
_STATE = {"participant": None, "faults": {}, "hooks": [], "rebuilt": {}, "fault_path": None}

if os.environ.get("FASTRSI_TORCH_KEY_CACHE"):
    # warm restart with path-stable caches: the worker imports this module at the session's first call, before the first
    # compile; preset Inductor's torch_key from disk (services/jitcache.py)
    from services import jitcache as _jitcache
    _jitcache.preset_torch_key()


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

        class Participant(C.LocalParticipant):
            def _live_source(self, module):
                # no committed transaction: a rebuilt module runs its rebuilt text, not its file
                text = super()._live_source(module)
                return text if text is not None else _STATE["rebuilt"].get(module)
        r = rank_of(worker)
        scope = {"worker": worker, "pipeline": getattr(worker, "pipeline", None)}
        p = _STATE["participant"] = Participant(f"rank{r}", scope=scope, resets=_resets(worker))
    return p


def _transformer(worker):
    pipe = getattr(worker, "pipeline", None)
    mods = getattr(pipe, "modules", {}) or {}
    return mods.get("transformer")


def inject(worker, fault, **kw):
    """Conformance faults, injected in this worker. Returns an ok-dict."""
    p = participant(worker)
    if fault in ("fail_apply", "fail_commit", "tamper"):
        if fault == "tamper":
            kw = {
                "module": "fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_decoding",
                "function": "_layout",
                **kw,
            }
        res = harness.inject_participant_fault(p, fault, **kw)
        return {**res, "pid": os.getpid()} if res.get("ok") else res
    if fault in ("crash", "hang"):
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


_file_of = harness.module_file


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


def _ctor_args(module):
    """Constructor arguments of a loaded component, when the module keeps them: FastVideo's
    DiTs (`BaseDiT`) keep `config` and `hf_config`, their only arguments."""
    if hasattr(module, "config") and hasattr(module, "hf_config"):
        return (module.config, module.hf_config)
    return None


def _reset_stages(pipe):
    """Drop the pipeline's stages (FastVideo `ComposedPipelineBase` keeps three views)."""
    for st in list(getattr(pipe, "_stages", None) or []):
        name = getattr(st, "_pipeline_stage_name", None)
        if name and name in pipe.__dict__:
            delattr(pipe, name)
    if hasattr(pipe, "_stages"):
        pipe._stages = []
        pipe._stage_name_mapping = {}


def _restore_modules(old):
    """Put the refreshed modules back; a module rebuilt before shows its rebuilt text again."""
    from services import rebuild as RB
    RB.restore(old)
    for name in old:
        if name in _STATE["rebuilt"]:
            _show_lines(name, _STATE["rebuilt"][name])


def _snapshot(obj):
    """obj's attributes, each with a copy of its contents when it is a dict, list or set (a
    stage build may change those in place). None: obj has no attributes to keep."""
    if not hasattr(obj, "__dict__"):
        return None
    return {name: (value, copy.copy(value) if isinstance(value, (dict, list, set)) else None)
            for name, value in vars(obj).items()}


def _put_back_attributes(obj, snapshot):
    """obj's attributes as `_snapshot` saw them, containers' contents included (one level:
    a change inside an object the pipeline holds is not taken back)."""
    vars(obj).clear()
    for name, (value, contents) in snapshot.items():
        if isinstance(value, list):
            value[:] = contents
        elif contents is not None:
            value.clear()
            value.update(contents)
        vars(obj)[name] = value


def _roll_back(undo, pipe, saved_pipe, old):
    """Take back a rebuild that failed after it began to change the worker: the class swaps
    and field merges (newest first), the pipeline's attributes and their dict/list/set
    contents (its stages, its component table), the modules. True when every step ran (not a
    check of the whole state); False: the worker is in an unknown state."""
    clean = True
    for step in reversed(undo):
        try:
            step()
        except Exception:
            clean = False
    try:
        if saved_pipe is not None:
            _put_back_attributes(pipe, saved_pipe)
        _restore_modules(old)
    except Exception:
        clean = False
    return clean


def rebuild(worker, payload):
    """In-place rebuild in this warm worker: re-import the edited modules (and their importers) from
    the new text, give the live objects the new classes, merge constructor edits into the
    loaded torch components (their tensors stay), and build the pipeline's stages again. The
    CUDA context, weights, JIT modules and compile caches stay. On a finding nothing changes:
    a failure after the first change (a class swap refused, the stage build raising) takes
    every change back. `clean` in a failed reply: True when every rollback step ran (the
    pipeline's attributes are restored one level deep).
    payload: {"files": {module: text}, "prefixes": [packages the import graph covers]}."""
    from services import rebuild as RB
    t0 = time.perf_counter()
    files = (payload or {}).get("files") or {}
    pipe = getattr(worker, "pipeline", None)
    prefixes = tuple((payload or {}).get("prefixes") or sorted({m.split(".")[0] for m in files}))
    # importers rebuilt before are re-imported from their rebuilt text, not reverted to disk
    r = RB.refresh(files, prefixes, kept=_STATE["rebuilt"])
    out = {"rank": rank_of(worker), "pid": os.getpid(), "refreshed": r["refreshed"],
           "refresh_s": r["seconds"], "findings": list(r["findings"])}
    if not r["ok"]:                             # refresh put the old modules back itself
        return {"ok": False, **out, "clean": True, "cost_s": round(time.perf_counter() - t0, 4)}
    cmap = RB.class_map(r["old"])
    out["classes"] = sorted(f"{c.__module__}.{c.__qualname__}" for c in cmap)
    merged, plans = {}, []
    try:
        try:
            import torch
            Module = torch.nn.Module
        except ImportError:                     # the CPU toy engine
            torch, Module = None, ()
        mods = dict(getattr(pipe, "modules", {}) or {})
        for name, m in mods.items():
            if not isinstance(m, Module):
                if type(m) in cmap:
                    plans.append((name, m, None))
                continue
            if not any(type(s) in cmap for s in m.modules()):
                continue
            args = _ctor_args(m)
            if args is None:
                plans.append((name, m, None))
                continue
            tm = time.perf_counter()
            with torch.device("meta"):
                base = type(m)(*args)
                new = cmap.get(type(m), type(m))(*args)
            mr = RB.merge_fields(m, base, new, dry_run=True)
            mr["meta_build_s"] = round(time.perf_counter() - tm, 4)
            merged[name] = mr
            if not mr["ok"]:
                out["findings"] += mr["findings"]
            plans.append((name, m, (base, new)))
    except Exception:
        out["findings"].append(RB.finding("rebuild_failed", traceback.format_exc()[-2000:],
                                          "the components rebuild"))
    if out["findings"]:
        _restore_modules(r["old"])
        return {"ok": False, **out, "merged": merged, "clean": True,
                "cost_s": round(time.perf_counter() - t0, 4)}
    # every check passed: apply (class swaps and field merges), then the stages; a failure
    # from here on takes every change back
    undo, failed, stages, sw, ts = [], [], [], {"swapped": 0}, time.perf_counter()
    saved_pipe = _snapshot(pipe)
    try:
        for name, m, bn in plans:
            if bn is None:
                objs = list(m.modules()) if callable(getattr(m, "modules", None)) else [m]
                merged[name] = RB.swap_classes(objs, cmap, undo=undo)
            else:
                merged[name] = {**RB.merge_fields(m, *bn, undo=undo),
                                "meta_build_s": merged[name]["meta_build_s"]}
            failed += merged[name]["findings"]
        sw = RB.swap_classes([worker, pipe], cmap, undo=undo)
        failed += sw["findings"]
        ts = time.perf_counter()
        if not failed and pipe is not None and hasattr(pipe, "create_pipeline_stages"):
            _reset_stages(pipe)
            args = getattr(pipe, "fastvideo_args", None)
            pipe.create_pipeline_stages(*([args] if args is not None else []))
            stages = [type(s).__name__ for s in (getattr(pipe, "_stages", None) or [])] or \
                sorted(k for k, v in vars(pipe).items() if hasattr(v, "forward"))
    except Exception:
        failed.append(RB.finding("rebuild_failed", traceback.format_exc()[-2000:],
                                 "the components and the stages rebuild"))
    if failed:
        out["findings"] += failed
        clean = _roll_back(undo, pipe, saved_pipe, r["old"])
        return {"ok": False, **out, "merged": merged, "clean": clean,
                "cost_s": round(time.perf_counter() - t0, 4)}
    out.update(merged=merged, swapped=sw["swapped"], stages=stages,
               stages_s=round(time.perf_counter() - ts, 4))
    # caches the old code kept on its functions or globals (CUDA graphs, static buffers) would
    # stay alive next to the new code's own: release them before the first new call (after the
    # stage build: a release cannot be taken back)
    held = RB.release_held(r["old"])
    out["held_released"] = [{k: h[k] for k in ("where", "cuda_bytes", "graphs")} for h in held["released"]]
    out["held_findings"] = held["findings"]
    out["cuda_bytes_released"] = held["cuda_bytes_released"]
    _STATE["rebuilt"].update(files)         # what a later live swap of these modules starts from
    _STATE["participant"] = None            # a new base: the kernel journal starts over
    return {"ok": True, **out, "cost_s": round(time.perf_counter() - t0, 4)}


def arm_fault_beacon(worker, path):
    """Wrap the worker's forward entrypoint once so an uncaught exception writes an atomic
    fault beacon for the driver (fast-failing SP>=2 asymmetric crashes instead of blocking on
    peer ranks' NCCL watchdog)."""
    _STATE["fault_path"] = path
    target = worker if callable(getattr(worker, "execute_forward", None)) else getattr(worker, "pipeline", None)
    attr = "execute_forward" if target is worker else "forward"
    orig = getattr(target, attr, None) if target is not None else None
    if not callable(orig) or getattr(orig, "_fastrsi_fault_guarded", False):
        return {"ok": True, "armed": callable(orig)}

    def guarded(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except BaseException as exc:
            fault_path = _STATE.get("fault_path")
            if fault_path:
                try:
                    ipc.atomic_write_json(fault_path, {
                        "rank": rank_of(worker),
                        "pid": os.getpid(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "t": time.time(),
                    })
                except OSError:
                    pass
            raise

    guarded._fastrsi_fault_guarded = True
    setattr(target, attr, guarded)
    return {"ok": True, "armed": True}


def rpc(worker, method, payload=None):
    """The collective_rpc target. `worker` is the FastVideo worker of this process."""
    if _STATE["faults"].get("lost"):
        raise ConnectionError(f"rank{rank_of(worker)}: process lost (injected)")
    if method == "ping":
        return {"ok": True, "rank": rank_of(worker), "pid": os.getpid()}
    if method == "arm_fault_beacon":
        return arm_fault_beacon(worker, payload)
    if method == "inject":
        return inject(worker, **(payload or {}))
    if method == "repl":
        return repl(worker, payload)
    if method == "rebuild":
        try:
            return rebuild(worker, payload)
        except Exception:
            return {"ok": False, "findings": [{"check": "rebuild_failed",
                                               "observed": traceback.format_exc()[-3000:],
                                               "expected": "the rebuild runs"}]}
    p = participant(worker)
    # `core/livelab/kernel/swap.py` manages linecache (including per-version nested keys for
    # lazy `@triton.jit` builders) during commit/abort/undo without clobbering linecache at prepare.
    return getattr(p, method)(payload)
