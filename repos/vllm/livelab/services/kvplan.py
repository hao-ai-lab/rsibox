"""KV plan holder: reuse the KV-cache size measured on an earlier start with the same config,
and skip the profile run that measured it.

At start-up each worker runs Worker.determine_available_memory (startup step 4c): one dummy
forward pass with the largest batch (the "profile run") to measure peak activation memory, then
the KV-cache size from it. This holder stores the result (and the attributes the method sets,
which later steps read) in a plan file after a measured start. The next start with the same
ident returns it without profiling; the KV cache then has exactly the same size.

How it is installed: engine.py passes `--worker-cls kvplan.KVPlanWorker` (vLLM's own hook for a
custom worker class) with this folder on PYTHONPATH and RSIBOX_KVPLAN_DIR set. KVPlanWorker is
vLLM's GPU Worker with determine_available_memory wrapped; importing this module does not import
vLLM or torch until KVPlanWorker is asked for.

Contract:
- Plan file: <RSIBOX_KVPLAN_DIR>/<key>.json, key = hash of ident(): vLLM version, config hash,
  load format (the config hash leaves it out, but resident weights change what is free), rank,
  world size, GPU name and total memory, requested memory, RSIBOX_KVPLAN_SALT.
- Reuse needs the same key and free GPU memory at start that holds this engine's budget:
  at least as much as when measured, or at least `requested_memory` (vLLM's budget for the
  engine's own allocations: gpu_memory_utilization x total, minus weights a holder process
  keeps; it is in the key, and the plan's KV size was measured inside it). The second bound
  matters with resident weights: a holder group of another parallel config (the way back of a
  TP <-> EP reshard) takes memory outside the budget, and the engine fits all the same.
  Otherwise the worker logs a finding and profiles as usual. A miss next to plans of other
  configs names the fields that differ (finding "kvplan_mismatch"). RSIBOX_KVPLAN_FORCE=1
  reuses despite less free memory.
- A plan holds a positive KV size only. A measurement of zero or less (vLLM then fails this
  start with "No available memory for the cache blocks") is not recorded, and a plan file that
  holds one is not reused: each is a finding ("kvplan_not_recorded", "kvplan_unusable"), so one
  start that ran while another process held GPU memory cannot fail every later start.
- With TP > 1 every rank takes the same branch (all-reduce MIN of "can reuse"): the profile run
  has collectives, and one rank skipping it would hang the others. When torch.distributed is
  initialized but the ranks cannot agree (no vLLM world group), every rank profiles (finding
  "kvplan_no_agreement"): no rank decides alone.
- Every outcome is one log line, `rsibox kvplan: {json}` (with the seconds the step took) or
  `rsibox kvplan finding: {json}`. engine.py counts them (kvplan_report).

What moves: kernels the profile run would JIT-compile are compiled at their first use instead
(warmup, or the first request in eager mode).
Risk: an edit that changes activation memory but not the config reuses a stale size. Set
RSIBOX_KVPLAN_SALT (for example to the patch hash) when an edit touches the forward pass.
"""
import functools, hashlib, json, logging, os, sys, time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import corelab  # noqa: F401,E402
from rsibox import ipc  # noqa: E402

ATTRS = ("available_kv_cache_memory_bytes", "total_consumed", "peak_activation_memory",
         "cudagraph_memory_estimate")
TAG = "rsibox kvplan"
log = logging.getLogger("vllm.rsibox.kvplan")   # under vllm.*: vLLM's handler prints INFO


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
            "salt": os.environ.get("RSIBOX_KVPLAN_SALT", "")}


def key(idn):
    return hashlib.sha256(json.dumps(idn, sort_keys=True).encode()).hexdigest()[:20]


def save(plan_dir, idn, rec):
    """Write the plan atomically (concurrent ranks and starts never see half a file)."""
    path = Path(plan_dir) / f"{key(idn)}.json"
    ipc.atomic_write_json(path, {"ident": idn, **rec})
    return path


def load(plan_dir, idn):
    return ipc.read_json(Path(plan_dir) / f"{key(idn)}.json")


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
    when torch.distributed is not initialized or the world has one rank. None when
    torch.distributed is initialized but vLLM's world group cannot be had: other ranks may
    exist, and a rank-local answer could split them between reuse and the profile run."""
    try:
        import torch
        import torch.distributed as dist
    except Exception:  # noqa: BLE001  (no torch: one process)
        return ok
    if not dist.is_available() or not dist.is_initialized():
        return ok
    try:
        from vllm.distributed.parallel_state import get_world_group
        group = get_world_group().cpu_group
    except Exception as e:  # noqa: BLE001
        _say("kvplan_no_agreement", finding=True, error=repr(e)[:300],
             why="torch.distributed is up but vLLM's world group is not; this rank profiles "
                 "rather than decide alone (every rank gets the same error and profiles too)")
        return None
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


def fits(free, rec, requested):
    """Can a start with `free` bytes free reuse plan `rec`? Its KV size was measured inside the
    engine's budget `requested` (part of the key), so free memory that holds the budget is
    enough, even when it is less than at the measurement (another holder group resident)."""
    return (free >= rec["init_free"] or free >= requested
            or os.environ.get("RSIBOX_KVPLAN_FORCE") == "1")


def wrap(orig):
    """determine_available_memory that reuses or records the plan around `orig`."""
    @functools.wraps(orig)
    def determine_available_memory(self, *a, **k):
        t0 = time.perf_counter()
        plan_dir = os.environ.get("RSIBOX_KVPLAN_DIR")
        if self.cache_config.kv_cache_memory_bytes or not plan_dir:
            why = "kv_cache_memory_bytes given" if plan_dir else "RSIBOX_KVPLAN_DIR unset"
            r = orig(self, *a, **k)
            _say("profiled", finding=not plan_dir, why=why,
                 seconds=round(time.perf_counter() - t0, 3))
            return r
        idn = ident(self)
        free = int(self.init_snapshot.free_memory)
        rec = load(plan_dir, idn)
        unusable = rec is not None and int(rec.get("result", 0)) <= 0
        if unusable:
            _say("kvplan_unusable", finding=True, key=key(idn), rank=idn["rank"],
                 bytes=rec.get("result"), free_then=rec.get("init_free"),
                 why="the plan holds no positive KV size (measured while GPU memory was held "
                     "elsewhere?); profiling and replacing it",
                 actions=["accept: this start records a new plan if its measurement is "
                          "positive", "free the GPU memory held by other processes"])
            rec = None
        local = rec is not None and fits(free, rec, int(self.requested_memory))
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
        if rec is None and not unusable:
            near = nearest(plan_dir, idn)
            if near is not None:
                _say("kvplan_mismatch", finding=True, key=key(idn), rank=idn["rank"], **near,
                     why="no plan for this config; profiling (+ the profile run's seconds)",
                     actions=["accept: the plan is recorded for this config now",
                              "align the differing fields to reuse the other plan"])
        elif rec is None or agreed is None:
            pass                # kvplan_unusable, or all_agree's kvplan_no_agreement, is logged
        elif not local:
            _say("kvplan_free_memory", finding=True, key=key(idn), rank=idn["rank"],
                 free_now=free, free_then=rec["init_free"],
                 requested=int(self.requested_memory),
                 why="less free GPU memory than when the plan was measured, and less than "
                     "the engine's budget (requested_memory); profiling",
                 actions=["RSIBOX_KVPLAN_FORCE=1 reuses it (risk: OOM at the first large "
                          "batch)", "free the GPU memory held by other processes"])
        elif rec is not None:
            _say("kvplan_rank_disagrees", finding=True, key=key(idn), rank=idn["rank"],
                 why="another rank cannot reuse its plan; every rank profiles so the profile "
                     "run's collectives match")
        result = orig(self, *a, **k)
        if int(result) <= 0:
            _say("kvplan_not_recorded", finding=True, key=key(idn), rank=idn["rank"],
                 bytes=int(result), free_now=free,
                 why="the profile run left no memory for the KV cache; a plan with this size "
                     "would make every later start with this config fail",
                 actions=["free the GPU memory held by other processes, then start again",
                          "raise --gpu-memory-utilization"],
                 seconds=round(time.perf_counter() - t0, 3))
            return result
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
