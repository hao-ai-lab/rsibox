"""R1 on a vLLM worker: make captured and compiled model code run the swapped Python.

The kernel runs R1 as the post-commit phase of a transaction (and again after its undo, or
after a failed R1 to rebuild from the base code). How depends on the engine mode (`how`):

- `recapture` (cudagraph mode on, compile mode NONE: pure CUDA graphs, breakable or full):
  release every captured graph and capture them again. The capture runs the Python model code,
  so the graphs now hold the swapped code.
- `recompile` (torch.compile on, e.g. VLLM_COMPILE): a recapture alone replays the old compiled
  program (TP4 Qwen3-8B: witness 0 and the output unchanged after it). Release every graph,
  reset Dynamo and the compile wrapper, run a profile-size dummy run (Dynamo re-traces the
  swapped code), then `compile_or_warm_up_model` (Inductor compile and capture). The compile
  cache is bypassed: vLLM keys it on the files on disk, and a live edit is not on disk.
- `none` (eager): nothing is captured or compiled.

Both release the piecewise graphs held by vLLM's graph wrappers as well as the graph managers'
graphs: a recompile builds new wrappers, and the old ones kept their graphs alive (+44-92 MiB per
GPU per recompile before this). Recapture keeps the engine's graph pool alive with an anchor
graph and captures on the engine's capture stream, as the live lab's recapture does (a capture
into a pool whose last graph was dropped asserted in PyTorch; a new stream per capture leaked a
per-stream workspace).

V2 model runner only (`vllm.v1.worker.gpu`); another runner gets an error, not a guess.
Call on every rank between two engine steps (the host's rebuild job, a collective_rpc).
"""
import gc
import os
import sys
import time

WRAPPERS = (("vllm.compilation.cuda_graph", "CUDAGraphWrapper"),
            ("vllm.compilation.breakable_cudagraph", "BreakableCUDAGraphWrapper"))
_POOL_ANCHORS = {}       # str(pool) -> CUDAGraph that holds one use of the pool


def how(worker):
    """"recompile", "recapture" or "none" for this worker's vLLM config."""
    cfg = getattr(worker, "vllm_config", None)
    cc = getattr(cfg, "compilation_config", None)
    if cc is None or getattr(getattr(cfg, "model_config", None), "enforce_eager", False):
        return "none"
    cg = getattr(getattr(cc, "cudagraph_mode", None), "name", None) or "NONE"
    mode = getattr(getattr(cc, "mode", None), "name", None) or "NONE"
    if mode != "NONE":
        return "recompile"
    return "recapture" if cg != "NONE" else "none"


def _managers(runner):
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    mgr = getattr(runner, "cudagraph_manager", None)
    if mgr is None:
        raise RuntimeError("this model runner has no cudagraph_manager (R1 supports the V2 "
                           "model runner, vllm.v1.worker.gpu)")
    state = getattr(runner, "model_state", None)
    return [mgr] + [v for v in vars(state or object()).values()
                    if isinstance(v, CudaGraphManager)]


def release_all_graphs(managers):
    """Release every captured graph: the managers' (FULL and breakable) and the piecewise
    graphs inside vLLM's graph wrappers. {"managers": n, "piecewise": n} graphs released."""
    out = {"managers": 0, "piecewise": 0}
    for m in managers:
        if m is not None:
            out["managers"] += len(getattr(m, "graphs", None) or {})
            m.release_graphs()
    for modname, clsname in WRAPPERS:
        mod = sys.modules.get(modname)
        cls = getattr(mod, clsname, None) if mod is not None else None
        for w in list(getattr(cls, "_all_instances", ()) or ()):
            out["piecewise"] += len(getattr(w, "concrete_cudagraph_entries", None)
                                    or getattr(w, "entries", None) or {})
            clear = getattr(w, "clear_graphs", None)
            if clear is not None:
                clear()
    return out


def _keep_graph_pool(managers):
    """Hold one use of the engine's graph pool with an empty anchor graph, so dropping every
    graph does not free the pool that the next capture uses."""
    import torch
    from vllm.platforms import current_platform
    pool = type(current_platform)._global_graph_pool
    pools = {str(m.pool) for m in managers if getattr(m, "pool", None) is not None}
    note = None if pool is not None and not pools - {str(pool)} else \
        f"graph managers use pools {sorted(pools)}, the global pool is {pool}"
    if pool is None:
        return {"pool": None, "anchored_now": False, "note": note}
    key = str(pool)
    anchored_now = key not in _POOL_ANCHORS
    if anchored_now:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g.capture_begin(pool=pool, capture_error_mode="thread_local")
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        _POOL_ANCHORS[key] = g
    return {"pool": key, "anchored_now": anchored_now, "note": note}


def _keep_capture_stream(mgr):
    """Capture on the stream the engine captured on (installed once, process-wide)."""
    import torch
    import vllm.v1.worker.gpu.cudagraph_utils as cu
    cur = cu.graph_capture
    if getattr(cur, "_fastrsi_stream", None) is not None:
        return {"stream": hex(cur._fastrsi_stream.cuda_stream), "installed_now": False}
    keep = getattr(mgr, "capture_stream", None) or torch.cuda.Stream()

    def graph_capture(device, stream=None, _orig=cur, _keep=keep):
        return _orig(device, stream if stream is not None else _keep)
    graph_capture._fastrsi_stream = keep
    cu.graph_capture = graph_capture
    return {"stream": hex(keep.cuda_stream), "installed_now": True}


def _mib():
    import torch
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 2**20, 1)


def recapture(worker):
    runner = worker.model_runner
    managers = _managers(runner)
    out = {"how": "recapture", "used_mib_before": _mib()}
    t0 = time.perf_counter()
    if managers[0].pool is not None:
        out["pool"] = _keep_graph_pool(managers)
        out["capture_stream"] = _keep_capture_stream(managers[0])
    out["released"] = release_all_graphs(managers)
    runner.capture_model()
    out["graphs"] = sum(len(getattr(m, "graphs", None) or {}) for m in managers)
    out["seconds"] = round(time.perf_counter() - t0, 3)
    out["used_mib_after"] = _mib()
    return out


def recompile(worker, disable_cache=True):
    import torch
    import vllm.envs as envs
    from vllm.compilation.wrapper import reset_compile_wrapper
    from vllm.config import set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu.eplb_utils import preserve_serving_state
    runner = worker.model_runner
    managers = _managers(runner)
    out = {"how": "recompile", "used_mib_before": _mib()}
    t0 = time.perf_counter()
    out["released"] = release_all_graphs(managers)
    torch.compiler.reset()
    with set_current_vllm_config(worker.vllm_config):
        reset_compile_wrapper(runner.get_model())
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    platform_cls = type(current_platform)        # a fresh graph pool, as elastic EP does
    if getattr(platform_cls, "_global_graph_pool", None) is not None:
        pool = current_platform.graph_pool_handle()
        platform_cls._global_graph_pool = pool
        for m in managers:
            if getattr(m, "pool", None) is not None:
                m.pool = pool
        for modname, clsname in WRAPPERS:
            cls = getattr(sys.modules.get(modname), clsname, None)
            for w in list(getattr(cls, "_all_instances", ()) or ()):
                w.graph_pool = pool
    out["release_reset_s"] = round(time.perf_counter() - t0, 3)
    old_env = os.environ.get("VLLM_DISABLE_COMPILE_CACHE")
    if disable_cache:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    clear = getattr(envs.__getattr__, "cache_clear", None)
    if clear:
        clear()
    try:
        with preserve_serving_state(runner):
            t1 = time.perf_counter()
            runner._dummy_run(runner.max_num_tokens, is_profile=True, skip_eplb=True)
            out["trace_s"] = round(time.perf_counter() - t1, 3)
            t2 = time.perf_counter()
            worker.compile_or_warm_up_model()
            out["compile_and_capture_s"] = round(time.perf_counter() - t2, 3)
    finally:
        if old_env is None:
            os.environ.pop("VLLM_DISABLE_COMPILE_CACHE", None)
        else:
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = old_env
        if clear:
            clear()
    out["graphs"] = sum(len(getattr(m, "graphs", None) or {}) for m in managers)
    out["seconds"] = round(time.perf_counter() - t0, 3)
    out["used_mib_after"] = _mib()
    return out


def run(worker):
    """R1 on this worker, by engine mode. Returns a JSON-able dict with `how`; raises on
    failure (the side turns it into a not-ok reply)."""
    h = how(worker)
    if h == "recompile":
        return recompile(worker)
    if h == "recapture":
        return recapture(worker)
    return {"how": "none", "note": "eager engine: nothing captured or compiled"}
