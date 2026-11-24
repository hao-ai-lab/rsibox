"""Scheduler-side agent: one kernel `LocalParticipant` per TP rank, parked via `rsibox_gap`."""
import json, linecache, os, queue, random, sys, threading
from multiprocessing.connection import Client
from adapter import recapture as R1
from rsibox import harness
from kernel import coordinator as C, findings as F

GAP_METHODS = frozenset({"park", "validate", "commit", "undo", "recapture",
                         "recapture_extend", "recapture_extend_drop", "reset", "release_gap"})
_STATE = {"participant": None, "recapture": None, "faults": {}, "hooks": [], "plans": {}, "gap_queue": queue.Queue()}

def _origin(name):
    return getattr(sys.modules.get(name), "__file__", None) or harness.module_file(name)

def _sync_lines(local_p, names):
    for name in names:
        path, src = _origin(name), local_p._live_source(name)
        if path:
            if src is not None: linecache.cache[path] = (len(src), None, src.splitlines(True), path)
            else:
                linecache.cache.pop(path, None)

def _rank(scheduler):
    for obj in (
        scheduler,
        getattr(scheduler, "tp_worker", None),
        getattr(getattr(scheduler, "tp_worker", None), "model_runner", None),
    ):
        if obj is not None and hasattr(obj, "tp_rank"):
            return int(obj.tp_rank)
    for mod_name in ("sglang.srt.runtime_context", "sglang.srt.distributed"):
        mod = sys.modules.get(mod_name)
        for getter_name, attr in (("get_parallel", "tp_rank"), ("get_tensor_model_parallel_rank", None)):
            getter = getattr(mod, getter_name, None)
            if callable(getter):
                try:
                    val = getter()
                    return int(getattr(val, attr) if attr else val)
                except Exception:
                    pass
    return 0

def _is_capturing(scheduler):
    runner, torch_mod = scheduler.tp_worker.model_runner, sys.modules.get("torch")
    if torch_mod is not None and getattr(torch_mod.cuda, "is_initialized", lambda: False)() and torch_mod.cuda.is_current_stream_capturing(): return True
    graph_runner = getattr(runner, "graph_runner", None) or getattr(runner, "decode_cuda_graph_runner", None)
    getter = getattr(sys.modules.get(type(graph_runner).__module__), "get_is_capture_mode", None)
    return bool(callable(getter) and getter())

def _drain_overlap(scheduler):
    if getattr(scheduler, "enable_overlap", False) and getattr(scheduler, "last_batch", None):
        io_name = type(scheduler).__module__.rsplit(".", 1)[0] + ".io_struct"
        io_mod = sys.modules.get(io_name) or __import__(io_name, fromlist=["*"])
        scheduler.pause_generation(io_mod.PauseGenerationReqInput(mode="retract"))
        scheduler.continue_generation(io_mod.ContinueGenerationReqInput())

def _reset_state(scheduler):
    _drain_overlap(scheduler)
    if not scheduler.flush_cache(): raise RuntimeError("flush_cache refused: pending requests remain on scheduler")
    runner = scheduler.tp_worker.model_runner
    pool = (getattr(scheduler, "mamba_pool", None) or getattr(runner, "mamba_pool", None)
            or getattr(getattr(scheduler, "req_to_token_pool", None), "mamba_pool", None))
    if isinstance(getattr(pool, "states", None), list): pool.states[:] = [0] * len(pool.states)
    for attr in ("conv_state", "ssm_state", "temporal_state"):
        buf = getattr(pool, attr, None) if pool is not None else None
        for item in (buf if isinstance(buf, (list, tuple)) else (buf,)):
            if hasattr(item, "zero_"): item.zero_()
    seed = int(getattr(runner, "random_seed", 0) or 0)
    random.seed(seed)
    if hasattr(getattr(runner, "rng", None), "seed"): runner.rng.seed(seed)
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        torch_mod.manual_seed(seed)
        if getattr(torch_mod.cuda, "is_initialized", lambda: False)():
            torch_mod.cuda.manual_seed_all(seed)
    return ["radix_cache", "kv_pool", "mamba_pool", "rng"]

def participant(scheduler):
    if _STATE["participant"] is None:
        runner = scheduler.tp_worker.model_runner
        scope = {"scheduler": scheduler, "worker": scheduler.tp_worker, "model_runner": runner}
        resets = {"state": lambda: _reset_state(scheduler), "R3": lambda: _reset_state(scheduler)}
        _STATE["participant"] = C.LocalParticipant(f"rank{_rank(scheduler)}", scope=scope, resets=resets,
                                                   capturing=lambda: _is_capturing(scheduler))
        _STATE["recapture"] = R1.Recapture()
    return _STATE["participant"]

def rpc(scheduler, method, payload=None):
    local_p, info, rank_id = participant(scheduler), payload or {}, _rank(scheduler)
    runner = scheduler.tp_worker.model_runner
    graph_runner = getattr(runner, "graph_runner", None) or getattr(runner, "decode_cuda_graph_runner", None)
    if method == "ping": return {"ok": True, "rank": rank_id, "pid": os.getpid()}
    if method == "inject":
        from adapter.fixture import faults
        return faults.inject(scheduler, local_p, _STATE, **info)
    if method == "graphs":
        gmap = getattr(graph_runner, "graphs", None) if hasattr(graph_runner, "graphs") else getattr(getattr(graph_runner, "backend", None), "_graphs", None)
        return {"ok": True, "rank": rank_id, "graphs": len(gmap or {}), "capture_bs": sorted(getattr(graph_runner, "capture_bs", ()) or ())}
    if method in ("recapture", "recapture_extend"):
        if _STATE["faults"].pop("fail_recapture", False): return F.not_ok(f"rank{rank_id}: injected failure", [F.finding(
                "rebuild_failed", "injected OOM during graph capture", "every size captured")], rank=rank_id)
        reply = (R1.recapture(_STATE["recapture"], graph_runner, info["txn"], local_p.journal, local_p.witness, sizes=info.get("sizes"))
                 if method == "recapture" else
                 _STATE["recapture"].extend(graph_runner, info["txn"], local_p.journal, local_p.witness,
                                            sizes=info.get("sizes"), extend_id=info.get("extend_id")))
        return {**reply, "rank": rank_id}
    if method == "recapture_extend_drop": return {**_STATE["recapture"].drop_extend(info["txn"], local_p.journal, info.get("extend_id")), "rank": rank_id}
    if method == "prepare":
        specs = info.get("modules", {})
        for name, spec in specs.items():
            path, src = _origin(name), spec.get("source")
            if path and src is not None: linecache.cache[path] = (len(src), None, src.splitlines(True), path)
        reply = local_p.prepare(payload)
        _STATE["plans"][reply.get("txn")] = set(specs)
        if not reply.get("ok"): _sync_lines(local_p, specs)
        return reply
    reply = getattr(local_p, method)(payload)
    if method in ("commit", "abort", "undo"):
        _sync_lines(local_p, {m for names in _STATE["plans"].values() for m in names})
    return reply

def rsibox_gap(scheduler, gap_id=""):
    _drain_overlap(scheduler)
    while True:
        method, payload, box, done = _STATE["gap_queue"].get()
        try:
            box["reply"] = ({"ok": True, "gap_id": gap_id, "rank": _rank(scheduler)}
                            if method in ("park", "release_gap") else rpc(scheduler, method, payload))
        except Exception as exc:
            box["reply"] = F.not_ok(f"{method}: {type(exc).__name__}: {exc}")
        done.set()
        if method == "release_gap": return

def _control_loop(scheduler, conn):
    try:
        while True:
            req = conn.recv()
            method, payload = req["method"], req.get("payload")
            if _STATE["faults"].get("lost"):
                conn.close()
                return
            if method in GAP_METHODS:
                box, done = {}, threading.Event()
                _STATE["gap_queue"].put((method, payload, box, done))
                done.wait()
                reply = box["reply"]
            else:
                try:
                    reply = rpc(scheduler, method, payload)
                except Exception as exc:
                    reply = F.not_ok(f"{method}: {type(exc).__name__}: {exc}")
            conn.send({"id": req["id"], "reply": reply})
    except (EOFError, OSError):
        return

def sync_sglang_gms_globals(server_args):
    """Sync `gpu_memory_service.integrations.sglang` lock-mode globals from `server_args`.

    SGLang >=0.5.21 `declare_resolution()` records overrides in `_resolved_overrides`
    without mutating `server_args.model_loader_extra_config`, and spawned scheduler
    children unpickle `server_args` without re-running `setup_gms()`. Calling this in
    both the parent (after `setup_gms`) and `run_scheduler_entry` guarantees
    `gms_read_only` and `gms_ro_connect_timeout_ms` are honored in every process,
    and installs the bare-metal `CUDA_VISIBLE_DEVICES` UUID shim when configured.
    """
    if os.environ.get("RSIBOX_GMS_DEVICE_UUID_MAP"):
        try:
            from rsibox.warm.dynamo_gms import install_visible_devices_uuid_shim
            install_visible_devices_uuid_shim()
        except Exception:
            pass
    else:
        try:
            from rsibox.warm.dynamo_gms import uninstall_visible_devices_uuid_shim
            uninstall_visible_devices_uuid_shim()
        except Exception:
            pass
    if server_args is None:
        return
    gms_sglang = sys.modules.get("gpu_memory_service.integrations.sglang")
    if gms_sglang is None:
        return
    raw_extra = None
    try:
        from sglang.srt.arg_groups.overrides import resolution_result
        raw_extra = resolution_result(server_args, "model_loader_extra_config")
    except Exception:
        for _src, declared in reversed(getattr(server_args, "_resolved_overrides", None) or ()):
            if isinstance(declared, dict) and "model_loader_extra_config" in declared:
                raw_extra = declared["model_loader_extra_config"]
                break
    if raw_extra is None:
        raw_extra = getattr(server_args, "model_loader_extra_config", None)
    if isinstance(raw_extra, str):
        raw_str = raw_extra.strip()
        extra = json.loads(raw_str) if raw_str else {}
    elif isinstance(raw_extra, dict):
        extra = dict(raw_extra)
    else:
        extra = {}
    if not isinstance(extra, dict):
        return
    from gpu_memory_service.integrations.common.utils import (
        get_gms_lock_mode,
        get_gms_ro_connect_timeout_ms,
    )
    gms_sglang._gms_lock_mode = get_gms_lock_mode(extra)
    gms_sglang._gms_ro_connect_timeout_ms = get_gms_ro_connect_timeout_ms(extra)
    gms_sglang._gms_initialized = True


def run_scheduler_entry(sock_path, scheduler_module_name, real_entry, *args, **kwargs):
    server_args = args[0] if args else kwargs.get("server_args")
    if server_args is not None:
        sync_sglang_gms_globals(server_args)
    scheduler_cls = __import__(scheduler_module_name, fromlist=["Scheduler"]).Scheduler
    if not getattr(scheduler_cls, "_rsibox_hooked", False):
        orig_init = scheduler_cls.__init__
        def hooked_init(self, *a, **k):
            orig_init(self, *a, **k)
            participant(self)
            conn = Client(sock_path, "AF_UNIX")
            conn.send({"rank": int(_rank(self)), "pid": os.getpid()})
            threading.Thread(target=_control_loop, args=(self, conn), daemon=True).start()

        scheduler_cls.__init__, scheduler_cls.rsibox_gap, scheduler_cls._rsibox_hooked = hooked_init, rsibox_gap, True
    try:
        return real_entry(*args, **kwargs)
    except SystemExit:
        raise
    except BaseException as exc:
        print(f"rsibox scheduler exited on {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        os._exit(1)
