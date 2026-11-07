"""KV plan holder: reuse the KV-cache size measured on an earlier start with the same config,
and skip the profile run that measured it.

At start-up each worker runs Worker.determine_available_memory (startup step 4c): one dummy
forward pass with the largest batch (the "profile run") to measure peak activation memory, then
the KV-cache size from it. This holder stores the result (and the attributes the method sets,
which later steps read) in a plan file after a measured start. The next start with the same
ident returns it without profiling; the KV cache then has exactly the same size.

How it is installed: engine.py passes `--worker-cls kvplan.KVPlanWorker` (vLLM's own hook for a
custom worker class) with this folder on PYTHONPATH and FASTRSI_KVPLAN_DIR set. KVPlanWorker is
vLLM's GPU Worker with determine_available_memory wrapped; importing this module does not import
vLLM or torch until KVPlanWorker is asked for.

Contract:
- Plan file: <FASTRSI_KVPLAN_DIR>/<key>.json, key = hash of ident(): vLLM version, config hash,
  load format (the config hash leaves it out, but resident weights change what is free), rank,
  world size, GPU name and total memory, requested memory, FASTRSI_KVPLAN_SALT.
- Reuse needs the same key and at least as much free GPU memory at start as when measured.
  Otherwise the worker logs a finding and profiles as usual. A miss next to plans of other
  configs names the fields that differ (finding "kvplan_mismatch"). FASTRSI_KVPLAN_FORCE=1
  reuses despite less free memory.
- With TP > 1 every rank takes the same branch (all-reduce MIN of "can reuse"): the profile run
  has collectives, and one rank skipping it would hang the others.
- Every outcome is one log line, `fastrsi kvplan: {json}` or `fastrsi kvplan finding: {json}`,
  with the seconds the step took. engine.py counts them (kvplan_report).

What moves: kernels the profile run would JIT-compile are compiled at their first use instead
(warmup, or the first request in eager mode).
Risk: an edit that changes activation memory but not the config reuses a stale size. Set
FASTRSI_KVPLAN_SALT (for example to the patch hash) when an edit touches the forward pass.
"""
import functools, hashlib, json, logging, os, time
from pathlib import Path

ATTRS = ("available_kv_cache_memory_bytes", "total_consumed", "peak_activation_memory",
         "cudagraph_memory_estimate")
TAG = "fastrsi kvplan"
log = logging.getLogger("vllm.fastrsi.kvplan")   # under vllm.*: vLLM's handler prints INFO


def _say(event, finding=False, **kw):
    line = json.dumps({"event": event, **kw}, sort_keys=True)
    (log.warning if finding else log.info)("%s%s: %s", TAG, " finding" if finding else "", line)


def gpu(worker):
    """(name, total bytes) of the worker's GPU."""
    import torch
    p = torch.cuda.get_device_properties(worker.device)
    return p.name, int(p.total_memory)


def ident(worker):
    import vllm
    cfg = worker.vllm_config
    name, total = gpu(worker)
    return {"vllm": vllm.__version__, "config": cfg.compute_hash(),
            "load_format": str(cfg.load_config.load_format),
            "rank": worker.rank, "world_size": worker.parallel_config.world_size,
            "gpu": name, "gpu_total": total, "requested": int(worker.requested_memory),
            "salt": os.environ.get("FASTRSI_KVPLAN_SALT", "")}


def key(idn):
    return hashlib.sha256(json.dumps(idn, sort_keys=True).encode()).hexdigest()[:20]


def save(plan_dir, idn, rec):
    """Write the plan atomically (concurrent ranks and starts never see half a file)."""
    path = Path(plan_dir) / f"{key(idn)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"ident": idn, **rec}, indent=1))
    tmp.replace(path)
    return path


def load(plan_dir, idn):
    path = Path(plan_dir) / f"{key(idn)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def nearest(plan_dir, idn):
    """Among plans of the same rank, the one whose ident differs in the fewest fields:
    {"plan": file, "differs": {field: [plan, now]}}, or None if there is none."""
    best = None
    for p in sorted(Path(plan_dir).glob("*.json")) if Path(plan_dir).is_dir() else []:
        try:
            other = json.loads(p.read_text())["ident"]
        except (OSError, ValueError, KeyError):
            continue
        if other.get("rank") != idn["rank"]:
            continue
        diff = {k: [other.get(k), v] for k, v in idn.items() if other.get(k) != v}
        if best is None or len(diff) < len(best["differs"]):
            best = {"plan": p.name, "differs": diff}
    return best


def all_agree(ok):
    """True only if every rank says True (all-reduce MIN over the world group); `ok` itself
    when torch.distributed is not initialized or the world has one rank."""
    try:
        import torch
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return ok
        from vllm.distributed.parallel_state import get_world_group
        group = get_world_group().cpu_group
    except Exception:  # noqa: BLE001  (no torch, or no vLLM world group: one process)
        return ok
    if dist.get_world_size(group) <= 1:
        return ok
    t = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def workspace_bytes():
    """Bytes per slot of vLLM's WorkspaceManager, or None. The profile run grows the workspace
    to the largest batch's needs and graph mode locks it after capture, so a start that skips
    the profile run grows it to the recorded size itself."""
    try:
        import vllm.v1.worker.workspace as ws
        m = ws._manager
        if m is None:
            return None
        return [int(t.numel() * t.element_size()) if t is not None else 0
                for t in m._current_workspaces]
    except Exception:  # noqa: BLE001
        return None


def grow_workspace(sizes):
    """Grow each workspace lane to the recorded size; return the bytes per slot now."""
    try:
        import vllm.v1.worker.workspace as ws
    except Exception:  # noqa: BLE001
        return None
    m = ws._manager
    if m is None:
        return None
    for wid, n in enumerate(sizes or []):
        if n and wid < getattr(m, "_num_lanes", 1):
            with ws.use_workspace_lane(wid):
                m._ensure_workspace_size(n)
    return workspace_bytes()


def wrap(orig):
    """determine_available_memory that reuses or records the plan around `orig`."""
    @functools.wraps(orig)
    def determine_available_memory(self, *a, **k):
        t0 = time.perf_counter()
        plan_dir = os.environ.get("FASTRSI_KVPLAN_DIR")
        if self.cache_config.kv_cache_memory_bytes or not plan_dir:
            why = "kv_cache_memory_bytes given" if plan_dir else "FASTRSI_KVPLAN_DIR unset"
            r = orig(self, *a, **k)
            _say("profiled", finding=not plan_dir, why=why,
                 seconds=round(time.perf_counter() - t0, 3))
            return r
        idn = ident(self)
        free = int(self.init_snapshot.free_memory)
        rec = load(plan_dir, idn)
        local = rec is not None and (free >= rec["init_free"]
                                     or os.environ.get("FASTRSI_KVPLAN_FORCE") == "1")
        agreed = all_agree(local)
        if rec is not None and agreed:
            for name in ATTRS:
                if rec["attrs"].get(name) is not None:
                    setattr(self, name, rec["attrs"][name])
            ws = grow_workspace(rec.get("workspace_bytes"))
            _say("reused", key=key(idn), rank=idn["rank"], bytes=rec["result"],
                 free_then=rec["init_free"], free_now=free, workspace=ws,
                 seconds=round(time.perf_counter() - t0, 3))
            return rec["result"]
        if rec is None:
            near = nearest(plan_dir, idn)
            if near is not None:
                _say("kvplan_mismatch", finding=True, key=key(idn), rank=idn["rank"], **near,
                     why="no plan for this config; profiling (+ the profile run's seconds)",
                     actions=["accept: the plan is recorded for this config now",
                              "align the differing fields to reuse the other plan"])
        elif not local:
            _say("kvplan_free_memory", finding=True, key=key(idn), rank=idn["rank"],
                 free_now=free, free_then=rec["init_free"],
                 why="less free GPU memory than when the plan was measured; profiling",
                 actions=["FASTRSI_KVPLAN_FORCE=1 reuses it (risk: OOM at the first large "
                          "batch)", "free the GPU memory held by other processes"])
        else:
            _say("kvplan_rank_disagrees", finding=True, key=key(idn), rank=idn["rank"],
                 why="another rank cannot reuse its plan; every rank profiles so the profile "
                     "run's collectives match")
        result = orig(self, *a, **k)
        attrs = {n: (int(getattr(self, n)) if getattr(self, n, None) is not None else None)
                 for n in ATTRS}
        path = save(plan_dir, idn, {"result": int(result), "init_free": free, "attrs": attrs,
                                    "workspace_bytes": workspace_bytes() or []})
        _say("measured", key=key(idn), rank=idn["rank"], bytes=int(result), plan=path.name,
             seconds=round(time.perf_counter() - t0, 3))
        return result

    return determine_available_memory


def __getattr__(name):
    """KVPlanWorker, built on first access (so `import kvplan` needs neither torch nor vLLM)."""
    if name != "KVPlanWorker":
        raise AttributeError(name)
    from vllm.v1.worker.gpu_worker import Worker
    cls = type("KVPlanWorker", (Worker,), {
        "determine_available_memory": wrap(Worker.determine_available_memory),
        "__module__": __name__})
    globals()[name] = cls
    return cls
