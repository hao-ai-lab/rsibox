"""Tier-A surgical CUDA-graph kernel-node pointer swapping (`cuGraphExecKernelNodeSetParams`).

Scans captured graphs first, compiles only matched cached `@triton.jit` specializations preserving
full backend options, and patches matching `CU_GRAPH_NODE_TYPE_KERNEL` nodes in place (0 dummy
forwards, TP=1, `keep_graph=True`). Fails closed on rollback/undo errors and falls back to R1 on
unmet preconditions. Pure Python at import time (no eager CUDA imports).
"""
from __future__ import annotations

import contextlib
import dataclasses
import sys
import time
import weakref
from typing import Any

from kernel import findings as F
from kernel import jit as JIT
from kernel import witness as W

from .graph_guard import GraphReplayGuard, graph_key

ENTRY_ATTRS = ("concrete_cudagraph_entries", "entries")
_LIVE_NODE_STATE: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = weakref.WeakKeyDictionary()
_FALLBACK_NODE_STATE: dict[tuple[int, int], dict[int, tuple[Any, ...]]] = {}
_KEEP_GRAPH_HOOK = None
_FALLBACK_OPT_FIELDS = (
    "num_warps", "num_stages", "num_ctas", "warp_size", "maxnreg", "ptx_version", "ptx_options",
    "ir_override", "enable_fp_fusion", "enable_reflect_ftz", "launch_cooperative_grid", "launch_pdl",
    "supported_fp8_dtypes", "deprecated_fp8_dot_operand_dtypes", "default_dot_input_precision",
    "allowed_dot_input_precisions", "max_num_imprecise_acc_default", "extern_libs", "debug",
    "backend_name", "sanitize_overflow", "arch", "instrumentation_mode",
)
_SAFE_LEAF_TYPES = (
    "CU_GRAPH_NODE_TYPE_MEMCPY", "CU_GRAPH_NODE_TYPE_MEMSET", "CU_GRAPH_NODE_TYPE_HOST",
    "CU_GRAPH_NODE_TYPE_EMPTY", "CU_GRAPH_NODE_TYPE_WAIT_EVENT", "CU_GRAPH_NODE_TYPE_EVENT_RECORD",
    "CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL", "CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT",
    "CU_GRAPH_NODE_TYPE_MEM_ALLOC", "CU_GRAPH_NODE_TYPE_MEM_FREE", "CU_GRAPH_NODE_TYPE_BATCH_MEM_OP",
)


def _safe_handle(obj: Any, method: str) -> int:
    fn = getattr(obj, method, None)
    if not callable(fn): return 0
    try: return int(fn() or 0)
    except Exception: return 0

def _opt_int(obj: Any, attr: str, default: int = 0) -> int:
    return default if (val := getattr(obj, attr, default)) is None else int(val)

def _freeze(val: Any) -> Any:
    return tuple(_freeze(x) for x in val) if isinstance(val, list) else val

def _cuda_device_ctx(device: Any):
    cuda = getattr(sys.modules.get("torch"), "cuda", None)
    dev_ctx, is_init = getattr(cuda, "device", None), getattr(cuda, "is_initialized", None)
    if device is None or not callable(dev_ctx) or (callable(is_init) and not is_init()):
        return contextlib.nullcontext()
    try: return dev_ctx(device)
    except Exception: return contextlib.nullcontext()

def _distributed_world_size() -> int:
    dist = sys.modules.get("torch.distributed")
    if dist is not None and callable(getattr(dist, "is_initialized", None)):
        with contextlib.suppress(Exception):
            return int(dist.get_world_size() or 1) if dist.is_initialized() else 1
    return 1

def _graphs_mod(torch_mod: Any) -> Any:
    return (getattr(getattr(torch_mod, "cuda", None), "graphs", None)
            if torch_mod is not None else sys.modules.get("torch.cuda.graphs"))


def _patch_cudagraph_cls(mod: Any) -> bool:
    cls = getattr(mod, "CUDAGraph", None)
    if cls is None or getattr(cls, "_rsibox_keep_graph_patched", False):
        return bool(cls is not None)
    orig_new, orig_init, orig_end = cls.__new__, cls.__init__, cls.capture_end
    orig_new_desc = cls.__dict__.get("__new__")

    def patched_new(target_cls, *args, **kwargs):
        auto = not args and "keep_graph" not in kwargs
        if auto:
            try: inst = orig_new(target_cls, True)
            except TypeError: inst, auto = orig_new(target_cls), False
        else:
            inst = orig_new(target_cls, *args, **kwargs)
        inst._rsibox_auto_keep_graph = auto
        return inst

    def patched_init(self, *args, **kwargs):
        if getattr(self, "_rsibox_auto_keep_graph", False) and not args and "keep_graph" not in kwargs:
            try: return orig_init(self, True)
            except TypeError: return orig_init(self)
        return orig_init(self, *args, **kwargs)

    def patched_end(self):
        res = orig_end(self)
        if (getattr(self, "_rsibox_auto_keep_graph", False) and not _safe_handle(self, "raw_cuda_graph_exec")
                and callable(getattr(self, "instantiate", None)) and _safe_handle(self, "raw_cuda_graph")):
            self.instantiate()
        return res

    cls.__new__, cls.capture_end = staticmethod(patched_new), patched_end
    if orig_init is not object.__init__:
        cls.__init__ = patched_init
    cls._rsibox_orig, cls._rsibox_keep_graph_patched = (orig_new_desc, orig_init, orig_end), True
    return True


def enable_keep_graph(torch_mod: Any = None) -> bool:
    """Default `torch.cuda.CUDAGraph()` to `keep_graph=True` with eager `instantiate()`."""
    global _KEEP_GRAPH_HOOK
    if (graphs_mod := _graphs_mod(torch_mod)) is not None:
        return _patch_cudagraph_cls(graphs_mod)
    if _KEEP_GRAPH_HOOK is None:
        with contextlib.suppress(Exception):
            from rsibox import hooks
            hook = hooks.AfterImportHook("torch.cuda.graphs", _patch_cudagraph_cls, once=True)
            sys.meta_path.insert(0, hook)
            _KEEP_GRAPH_HOOK = hook
    return False


def disable_keep_graph(torch_mod: Any = None) -> bool:
    """Restore original `torch.cuda.CUDAGraph` hooks and remove any pending import hook."""
    global _KEEP_GRAPH_HOOK
    removed = False
    if _KEEP_GRAPH_HOOK is not None:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(_KEEP_GRAPH_HOOK); removed = True
        _KEEP_GRAPH_HOOK = None
    cls = getattr(_graphs_mod(torch_mod), "CUDAGraph", None)
    if cls is None or not getattr(cls, "_rsibox_keep_graph_patched", False):
        return removed
    orig_new_desc, orig_init, orig_end = cls._rsibox_orig
    if orig_new_desc is not None: cls.__new__ = orig_new_desc
    else:
        with contextlib.suppress(AttributeError): delattr(cls, "__new__")
    cls.capture_end = orig_end
    if orig_init is not object.__init__: cls.__init__ = orig_init
    for attr in ("_rsibox_orig", "_rsibox_keep_graph_patched"):
        with contextlib.suppress(AttributeError): delattr(cls, attr)
    return True


class CudaDriverError(RuntimeError):
    """Raised when a CUDA driver graph API call fails or a graph handle is stale/destroyed."""


class CudaBindingsDriver:
    """Thin wrapper over `cuda.bindings.driver` with SWIG-safe int snapshotting & node allowlist."""

    def __init__(self, drv_mod: Any):
        self.drv = drv_mod
        nt = drv_mod.CUgraphNodeType
        self._leaf_types = {getattr(nt, n) for n in _SAFE_LEAF_TYPES if hasattr(nt, n)}

    def _check(self, status: Any, op: str) -> None:
        if status != self.drv.CUresult.CUDA_SUCCESS: raise CudaDriverError(f"{op} failed with {status!r}")

    def get_graph_nodes(self, h_graph_int: int) -> list[tuple[Any, int, str]]:
        drv, h_graph = self.drv, self.drv.CUgraph(int(h_graph_int))
        res, _, num_nodes = drv.cuGraphGetNodes(h_graph, 0); self._check(res, "cuGraphGetNodes(count)")
        if not num_nodes: return []
        res, nodes, _ = drv.cuGraphGetNodes(h_graph, int(num_nodes)); self._check(res, "cuGraphGetNodes(list)")
        nt, out = drv.CUgraphNodeType, []
        for nd in nodes:
            res, ntype = drv.cuGraphNodeGetType(nd); self._check(res, "cuGraphNodeGetType")
            kind = ("kernel" if ntype == nt.CU_GRAPH_NODE_TYPE_KERNEL else
                    "child_graph" if ntype == nt.CU_GRAPH_NODE_TYPE_GRAPH else
                    "leaf" if ntype in self._leaf_types else f"unsupported:{getattr(ntype, 'name', ntype)}")
            out.append((nd, int(nd), kind))
        return out

    def get_kernel_node_params(self, node: Any):
        res, p = self.drv.cuGraphKernelNodeGetParams(node); self._check(res, "cuGraphKernelNodeGetParams")
        return (p, int(p.func), (int(p.blockDimX), int(p.blockDimY), int(p.blockDimZ)),
                (int(p.gridDimX), int(p.gridDimY), int(p.gridDimZ)), int(p.sharedMemBytes))

    def set_exec_kernel_node_params(self, h_exec_int: int, node: Any, params: Any,
                                    func_int: int, block, grid, shm: int) -> None:
        drv = self.drv
        params.func = drv.CUfunction(int(func_int))
        params.blockDimX, params.blockDimY, params.blockDimZ = int(block[0]), int(block[1]), int(block[2])
        params.gridDimX, params.gridDimY, params.gridDimZ = int(grid[0]), int(grid[1]), int(grid[2])
        params.sharedMemBytes = int(shm)
        res = drv.cuGraphExecKernelNodeSetParams(drv.CUgraphExec(int(h_exec_int)), node, params)
        self._check(res[0] if isinstance(res, tuple) else res, "cuGraphExecKernelNodeSetParams")


def default_driver() -> CudaBindingsDriver | None:
    try:
        from cuda.bindings import driver as drv
        return CudaBindingsDriver(drv)
    except Exception: return None

def _node_state_map(graph_obj: Any, h_exec_int: int) -> dict[int, tuple[Any, ...]]:
    try:
        rec = _LIVE_NODE_STATE.get(graph_obj)
        if rec is None or rec["exec"] != h_exec_int:
            rec = _LIVE_NODE_STATE[graph_obj] = {"exec": h_exec_int, "nodes": {}}
        return rec["nodes"]
    except TypeError:
        return _FALLBACK_NODE_STATE.setdefault((id(graph_obj), h_exec_int), {})


def snapshot_jit_specs(lp: Any, txn: str, *, is_native: bool = False) -> dict[str, Any]:
    """Capture pre-commit Triton `device_caches` from `lp._staged[txn]` before `jit.apply` clears them."""
    def _no(reason: str, wkeys: list[str] | None = None) -> dict[str, Any]:
        return {"eligible": False, "reason": reason, "specs": [], "witness_keys": wkeys or []}

    if is_native: return _no("native_edit_requires_r1")
    if not (staged_tuple := getattr(lp, "_staged", {}).get(txn)): return _no("no_staged_transaction")
    staged, cells, _ = staged_tuple
    if cells: return _no("repair_cell_present")
    if not staged: return _no("empty_staged_modules")

    specs, witness_keys, seen_specs, jit_count = [], [], {}, 0
    for st in staged:
        jit_keys = {wk for _pl, wk in (getattr(st, "jits", None) or ())}
        if non_jit := sorted(set((getattr(st, "items", None) or {}).keys()) - jit_keys):
            return _no(f"non_jit_target_present:{non_jit[0]}")
        for ad in (getattr(st, "pending_adds", None) or ()):
            if not (ad.get("jit") or ad.get("is_jit")):
                return _no(f"non_jit_addition_present:{ad.get('key', '?')}")
        for pl, wk in (getattr(st, "jits", None) or ()):
            jit_count += 1
            if wk not in witness_keys: witness_keys.append(wk)
            for jf in pl.get("affected") or ():
                k_label = JIT.label(jf)
                for dev, entry in list((getattr(jf, "device_caches", None) or {}).items()):
                    if not isinstance(entry, (tuple, list)) or not entry or not isinstance(entry[0], dict):
                        continue
                    e2, e3 = (entry[2] if len(entry) >= 3 else None), (entry[3] if len(entry) >= 4 else None)
                    backend = e3 if hasattr(e3, "parse_options") else (e2 if hasattr(e2, "parse_options") else None)
                    target = None if e2 is backend else e2
                    for ckey, old_ck in list(entry[0].items()):
                        if not (old_func := _opt_int(old_ck, "function", 0)): continue
                        sig = (id(jf), dev, ckey)
                        if sig in seen_specs:
                            if wk not in seen_specs[sig]["witness_keys"]: seen_specs[sig]["witness_keys"].append(wk)
                            continue
                        rec = {"jf": jf, "witness_keys": [wk], "kernel_label": k_label, "device": dev,
                               "cache_key": ckey, "old_ck": old_ck, "old_func": old_func,
                               "target": target, "backend": backend, "cache_shell": tuple(entry[1:])}
                        seen_specs[sig] = rec; specs.append(rec)
    if not jit_count: return _no("no_jit_targets")
    if not specs: return _no("no_cached_jit_specializations", witness_keys)
    return {"eligible": True, "reason": None, "specs": specs, "witness_keys": witness_keys}


def _extract_compile_options(old_meta: Any, backend: Any) -> dict[str, Any]:
    raw = old_meta._asdict() if callable(getattr(old_meta, "_asdict", None)) else dict(getattr(old_meta, "__dict__", {}))
    allowed: Any = _FALLBACK_OPT_FIELDS
    if backend is not None and callable(getattr(backend, "parse_options", None)):
        with contextlib.suppress(Exception):
            if dataclasses.is_dataclass(opt_obj := backend.parse_options({})):
                allowed = {f.name for f in dataclasses.fields(opt_obj)}
    return {k: _freeze(v) for k in allowed if (v := raw.get(k, getattr(old_meta, k, None))) is not None}


def _check_abi_compatible(old_ck: Any, new_ck: Any) -> str | None:
    if not _opt_int(new_ck, "function", 0): return "new_kernel_function_null"
    om, nm = getattr(old_ck, "metadata", None), getattr(new_ck, "metadata", None)
    if om is None or nm is None: return "missing_kernel_metadata"
    checks = (
        (_opt_int(om, "warp_size", 32) != _opt_int(nm, "warp_size", 32), "warp_size_changed"),
        (_opt_int(om, "num_ctas", 1) != 1 or _opt_int(nm, "num_ctas", 1) != 1, "cluster_num_ctas_unsupported"),
        (bool(getattr(om, "launch_cooperative_grid", False)) or bool(getattr(nm, "launch_cooperative_grid", False)), "cooperative_grid_unsupported"),
        (bool(getattr(om, "launch_pdl", False)) != bool(getattr(nm, "launch_pdl", False)), "launch_pdl_changed"),
        (_opt_int(om, "global_scratch_size", 0) != 0 or _opt_int(nm, "global_scratch_size", 0) != 0, "global_scratch_required"),
        (_opt_int(om, "profile_scratch_size", 0) != 0 or _opt_int(nm, "profile_scratch_size", 0) != 0, "profile_scratch_required"),
        (_opt_int(om, "tmem_size", 0) != _opt_int(nm, "tmem_size", 0), "tmem_size_changed"),
        (list(getattr(om, "tensordesc_meta", ()) or ()) != list(getattr(nm, "tensordesc_meta", ()) or ()), "tensordesc_meta_changed"),
    )
    return next((msg for failed, msg in checks if failed), None)


def compile_replacements(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompile only matched cached Triton specializations without mutating `device_caches`."""
    t0, staged, replacements = time.perf_counter(), [], {}
    for spec in specs:
        jf, old_ck = spec["jf"], spec["old_ck"]
        old_src, old_meta = getattr(old_ck, "src", None), getattr(old_ck, "metadata", None)
        if old_src is None or old_meta is None:
            return {"ok": False, "reason": "missing_old_compiled_kernel_src", "compile_s": round(time.perf_counter() - t0, 6)}
        target = spec.get("target") or getattr(old_meta, "target", None)
        opts = _extract_compile_options(old_meta, spec.get("backend"))
        try:
            new_ck = jf.compile(jf.ASTSource(jf, old_src.signature, old_src.constants, old_src.attrs),
                                target=target, options=opts)
            if callable(getattr(new_ck, "_init_handles", None)):
                with _cuda_device_ctx(spec.get("device")): new_ck._init_handles()
        except Exception as exc:
            return {"ok": False, "reason": f"compile_failed:{type(exc).__name__}:{exc}", "compile_s": round(time.perf_counter() - t0, 6)}
        if (abi_err := _check_abi_compatible(old_ck, new_ck)) is not None:
            return {"ok": False, "reason": f"abi_mismatch:{abi_err}", "compile_s": round(time.perf_counter() - t0, 6)}
        staged.append((spec, new_ck))
        nm = new_ck.metadata
        w_keys = tuple(spec["witness_keys"]) if "witness_keys" in spec else (spec["witness_key"],)
        replacements[spec["old_func"]] = {
            "old_ck": old_ck, "new_ck": new_ck, "new_func": _opt_int(new_ck, "function", 0),
            "block_x": _opt_int(nm, "warp_size", 32) * _opt_int(nm, "num_warps", 1),
            "shm": _opt_int(nm, "shared", 0), "witness_keys": w_keys, "kernel_label": spec["kernel_label"],
        }
    return {"ok": True, "replacements": replacements, "staged_compiled": staged,
            "compiled_specs": len(staged), "compile_s": round(time.perf_counter() - t0, 6)}


def _is_cudagraph(obj: Any) -> bool:
    return callable(getattr(obj, "raw_cuda_graph", None)) and callable(getattr(obj, "raw_cuda_graph_exec", None))

def extract_cudagraphs(entry: Any) -> list[Any]:
    if entry is None: return []
    if _is_cudagraph(entry): return [entry]
    for attr in ("cudagraph", "graph"):
        if _is_cudagraph(cg := getattr(entry, attr, None)): return [cg]
    for container in (getattr(entry, "capture", None), entry):
        if container is None: continue
        found = [cand for attr in ("graphs", "_graphs", "segments", "_segments")
                 if isinstance(seq := getattr(container, attr, None), (list, tuple))
                 for item in seq for cand in (next((c for c in (item, getattr(item, "cudagraph", None),
                                                                getattr(item, "__self__", None)) if _is_cudagraph(c)), None),)
                 if cand is not None]
        if found: return found
    return []

def _mode_name(mode: Any) -> str: return getattr(mode, "name", None) or str(mode)
def _wrapper_attr(w: Any) -> str | None: return next((a for a in ENTRY_ATTRS if isinstance(getattr(w, a, None), dict)), None)
def _wrapper_mode(w: Any) -> Any: return getattr(w, "runtime_mode", None) or "PIECEWISE"


def patch_graphs(
    engine: Any, txn: str, journal: Any, witness: Any, guard: GraphReplayGuard,
    jit_snap: dict[str, Any] | None, *, modes: Any = None, sizes: Any = None,
    pre_epoch: int | None = None, world_size: int | None = None, driver: Any = None,
) -> dict[str, Any]:
    """Attempt Tier-A `cuGraphExecKernelNodeSetParams` across captured graphs; fall back on unmet preconditions."""
    def _fb(reason: str, **extra: Any) -> dict[str, Any]:
        return {"ok": False, "fallback": True, "reason": reason, **extra}

    t_total = time.perf_counter()
    if max(int(world_size or 1), _distributed_world_size()) > 1:
        return _fb("multi_rank_world_size_unsupported")
    if sizes is not None and len(sizes) == 0: return _fb("explicit_empty_sizes")
    if not jit_snap or not jit_snap.get("eligible"):
        return _fb((jit_snap or {}).get("reason") or "no_jit_snapshot")
    if (drv := driver if driver is not None else default_driver()) is None:
        return _fb("cuda_driver_unavailable")
    selected = [m for m in engine.keys if engine.keys[m] and (modes is None or _mode_name(m) in modes)]
    if not selected: return _fb("no_selected_graph_modes")
    selected_names = [_mode_name(m) for m in selected]
    wrappers = [(w, a) for w in engine.wrappers()
                if (a := _wrapper_attr(w)) and _mode_name(_wrapper_mode(w)) in set(selected_names)]
    if not wrappers: return _fb("no_graph_wrappers")

    graph_items, all_keys = [], []
    for w, attr in wrappers:
        w_mode = _mode_name(_wrapper_mode(w))
        for desc, entry in getattr(w, attr).items():
            g_key = graph_key(getattr(desc, "cg_mode", None) or w_mode, desc)
            all_keys.append(g_key)
            if not (cgs := extract_cudagraphs(entry)):
                return _fb(f"unsupported_graph_entry:{type(entry).__name__}")
            for cg in cgs:
                h_graph, h_exec = _safe_handle(cg, "raw_cuda_graph"), _safe_handle(cg, "raw_cuda_graph_exec")
                if not h_graph: return _fb("keep_graph_unavailable")
                if not h_exec: return _fb("graph_not_instantiated")
                graph_items.append((g_key, cg, h_graph, h_exec))
    if not graph_items: return _fb("no_captured_graphs")
    if pre_epoch is not None and any(not guard.is_fresh_at(k, pre_epoch) for k in all_keys):
        return _fb("stale_graph_from_prior_edit")
    guard_modes = sorted(set(selected_names) | {k[0] for k in all_keys if isinstance(k, tuple) and k and isinstance(k[0], str)})

    specs_by_func, matched_nodes, needed_funcs = {s["old_func"]: s for s in jit_snap["specs"]}, [], set()
    try:
        for g_key, cg, h_graph, h_exec in graph_items:
            nmap = _node_state_map(cg, h_exec)
            for nd, nd_int, kind in drv.get_graph_nodes(h_graph):
                if kind == "child_graph": return _fb("nested_child_graph_unsupported")
                if kind == "leaf": continue
                if kind != "kernel": return _fb(f"unsupported_graph_node:{kind}")
                raw_params, b_func, b_blk, b_grid, b_shm = drv.get_kernel_node_params(nd)
                had_prev = nd_int in nmap
                cur_state = nmap[nd_int] if had_prev else (b_func, b_blk, b_grid, b_shm)
                if cur_state[0] in specs_by_func:
                    needed_funcs.add(cur_state[0])
                    matched_nodes.append((g_key, cg, h_exec, nd, nd_int, raw_params, cur_state, had_prev))
    except Exception as exc:
        return _fb(f"driver_scan_error:{type(exc).__name__}:{exc}")
    if not matched_nodes: return _fb("no_matching_kernel_nodes")

    comp = compile_replacements([s for s in jit_snap["specs"] if s["old_func"] in needed_funcs])
    if not comp["ok"]: return _fb(comp["reason"], compile_s=comp["compile_s"])
    replacements, mem0 = comp["replacements"], engine.memory()
    engine.sync()
    t_patch, patched_records, patched_graph_keys, witness_hits = time.perf_counter(), [], {}, []

    def _restore_nodes() -> None:
        first_err = None
        for cg, h_exec, nd, nd_int, raw_params, prev_state, had_prev in reversed(patched_records):
            if (live_exec := _safe_handle(cg, "raw_cuda_graph_exec")) != h_exec:
                first_err = first_err or CudaDriverError(f"CUgraphExec handle changed/destroyed ({h_exec} -> {live_exec})")
                continue
            try: drv.set_exec_kernel_node_params(h_exec, nd, raw_params, *prev_state)
            except Exception as err: first_err = first_err or err; continue
            nmap = _node_state_map(cg, h_exec)
            if had_prev: nmap[nd_int] = prev_state
            else: nmap.pop(nd_int, None)
        try: engine.sync()
        except Exception as err: first_err = first_err or err
        if first_err is not None: raise first_err

    try:
        for g_key, cg, h_exec, nd, nd_int, raw_params, cur_state, had_prev in matched_nodes:
            rep = replacements[cur_state[0]]
            new_state = (rep["new_func"], (rep["block_x"], cur_state[1][1], cur_state[1][2]), cur_state[2], rep["shm"])
            drv.set_exec_kernel_node_params(h_exec, nd, raw_params, *new_state)
            _node_state_map(cg, h_exec)[nd_int] = new_state
            patched_records.append((cg, h_exec, nd, nd_int, raw_params, cur_state, had_prev))
            for wk in rep["witness_keys"]:
                lk = W.launch_key(wk, rep["kernel_label"])
                witness_hits.extend((wk, lk))
                g_list = patched_graph_keys.setdefault(g_key, [])
                for k_item in (wk, lk):
                    if k_item not in g_list: g_list.append(k_item)
        engine.sync()
    except Exception as exc:
        try: _restore_nodes()
        except Exception as rb_err:
            guard.invalidate_modes(guard_modes)
            guard.bump_pool_epoch(f"tier_a_rollback_failed:{txn}", modes=guard_modes)
            return {"ok": False, "fallback": False, "reason": f"driver_rollback_error:{type(rb_err).__name__}:{rb_err}"}
        return _fb(f"driver_patch_error:{type(exc).__name__}:{exc}")

    patch_s = round(time.perf_counter() - t_patch, 6)
    for spec, new_ck in comp["staged_compiled"]:
        spec["jf"].device_caches.setdefault(spec["device"], ({}, *spec["cache_shell"]))[0][spec["cache_key"]] = new_ck
    with witness.capture():
        for k_item in witness_hits: witness.hit(k_item)
    pre_pool_epochs, pre_stamps, pre_bindings = guard.snapshot_modes_with_witness(guard_modes)
    for m in selected:
        for desc in engine.keys[m]: guard.stamp(graph_key(m, desc))
    guard.stamp_many(all_keys)
    for g_key, wk_list in patched_graph_keys.items():
        guard.bind_replay_witness(g_key, witness, tuple(wk_list))
    keep_alive = [(r["old_ck"], r["new_ck"]) for r in replacements.values()]

    def undo_patch() -> None:
        _ = keep_alive
        sync_err = None
        try: engine.sync()
        except Exception as err: sync_err = err
        try:
            _restore_nodes()
            if sync_err is not None: raise sync_err
        except Exception:
            guard.invalidate_modes(guard_modes)
            guard.bump_pool_epoch(f"tier_a_undo_failed:{txn}", modes=guard_modes)
            raise
        guard.restore_modes_with_witness(pre_pool_epochs, pre_stamps, pre_bindings)

    journal.record_call(undo_patch, f"Tier-A pointer swap of {txn}")
    return F.ok(tier="tier_a_pointer_swap", patched_nodes=len(patched_records),
                patched_graphs=len(patched_graph_keys), graphs=len(patched_graph_keys),
                total_graphs=len(graph_items), compiled_specs=comp["compiled_specs"],
                compile_s=comp["compile_s"], patch_s=patch_s, seconds=round(time.perf_counter() - t_total, 6),
                modes=selected_names, wrappers=len(wrappers), captured=[], eager={},
                memory={"before": mem0, "after": engine.memory()})
