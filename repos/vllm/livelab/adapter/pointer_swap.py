"""Tier-A surgical in-place CUDA graph kernel-node pointer swapping for Triton edits.

Locates matching `CUfunction` nodes in captured `CUgraph` templates, recompiles affected
specializations via Triton's compiler, verifies ABI compatibility, and atomically updates
exec (`cuGraphExecKernelNodeSetParams`) then template (`cuGraphKernelNodeSetParams`) nodes.
"""
import contextlib, sys, threading, time
from typing import Any
from kernel import jit as kernel_jit, witness as kernel_witness
from .graph_guard import graph_key

_KEEP_GRAPH_LOCK, _PINNED_KERNELS_LOCK = threading.Lock(), threading.Lock()
_KEEP_GRAPH_STATE: dict[str, Any] = {"enabled": False, "graph_cls": None, "orig_new": None, "orig_init": None, "orig_capture_end": None, "had_own_new": False, "had_own_init": False, "had_own_capture_end": False}
_PINNED_KERNELS: dict[tuple[int, int], list[Any]] = {}
_SHARED_MEM_CARVE_LIMIT = 228 * 1024
_SAFE_NON_KERNEL_NODE_NAMES = (
    "CU_GRAPH_NODE_TYPE_MEMCPY", "CU_GRAPH_NODE_TYPE_MEMSET", "CU_GRAPH_NODE_TYPE_EMPTY", "CU_GRAPH_NODE_TYPE_EVENT_RECORD",
    "CU_GRAPH_NODE_TYPE_WAIT_EVENT", "CU_GRAPH_NODE_TYPE_MEM_ALLOC", "CU_GRAPH_NODE_TYPE_MEM_FREE", "CU_GRAPH_NODE_TYPE_HOST",
    "CU_GRAPH_NODE_TYPE_BATCH_MEM_OP", "CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL", "CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT",
)

class _DriverPatchError(RuntimeError): pass
class _FallbackSignal(Exception): pass

def _fail(reason: str, **extra) -> dict[str, Any]: return {"applied": False, "fallback_reason": reason, **extra}

def _safe_int(raw_value: Any) -> int:
    try: return int(raw_value) if raw_value is not None else 0
    except (TypeError, ValueError): return 0

def _safe_handle(graph_obj: Any, attr_name: str) -> int:
    getter = getattr(graph_obj, attr_name, None)
    if not callable(getter): return 0
    try: return _safe_int(getter())
    except RuntimeError: return 0

def _freeze_option_value(value: Any) -> Any:
    if isinstance(value, list): return tuple(_freeze_option_value(item) for item in value)
    if isinstance(value, dict): return {dict_key: _freeze_option_value(dict_val) for dict_key, dict_val in value.items()}
    return value

def _is_cuda_graph(candidate: Any) -> bool:
    return candidate is not None and callable(getattr(candidate, "raw_cuda_graph", None)) and callable(getattr(candidate, "raw_cuda_graph_exec", None))

def extract_graphs(entry_obj: Any) -> list[Any]:
    """Extract all `CUDAGraph` objects from a V2 graph, V1 `CUDAGraphEntry`, or breakable entry."""
    if entry_obj is None: return []
    if _is_cuda_graph(entry_obj): return [entry_obj]
    capture_obj = getattr(entry_obj, "capture", None)
    candidates = [getattr(entry_obj, "cudagraph", None)] + [
        item for segment in ((getattr(capture_obj, "segments", None) or ()) if capture_obj is not None else ())
        for item in (getattr(segment, "__self__", None), segment)
    ]
    return list({id(graph_obj): graph_obj for graph_obj in candidates if _is_cuda_graph(graph_obj)}.values())

def pin_graph_kernels(handle_pairs, kernels: list[Any]) -> None:
    """Pin compiled kernel objects against GC while their `CUfunction`s live in graph execs."""
    with _PINNED_KERNELS_LOCK:
        for handle_pair in handle_pairs:
            bucket = _PINNED_KERNELS.setdefault(handle_pair, [])
            seen_ids = {id(item) for item in bucket}
            bucket.extend(kernel_obj for kernel_obj in kernels if id(kernel_obj) not in seen_ids)

def release_graph_kernels(entry_obj: Any) -> int:
    """Release pinned `CUmodule` holders associated with a graph or wrapper entry being reset."""
    graphs = extract_graphs(entry_obj)
    raw_graphs = {_safe_handle(graph_obj, "raw_cuda_graph") for graph_obj in graphs} - {0}
    raw_execs = {_safe_handle(graph_obj, "raw_cuda_graph_exec") for graph_obj in graphs} - {0}
    if not raw_graphs and not raw_execs: return 0
    with _PINNED_KERNELS_LOCK:
        drop_keys = [pair for pair in _PINNED_KERNELS if pair[0] in raw_graphs or pair[1] in raw_execs]
        for pair in drop_keys: _PINNED_KERNELS.pop(pair, None)
        return len(drop_keys)

def _restore_keep_graph_unlocked() -> None:
    if (graph_cls := _KEEP_GRAPH_STATE["graph_cls"]) is not None:
        for attr_name, own_key, orig_key in (("__new__", "had_own_new", "orig_new"), ("__init__", "had_own_init", "orig_init"), ("capture_end", "had_own_capture_end", "orig_capture_end")):
            if _KEEP_GRAPH_STATE[orig_key] is None: continue
            if _KEEP_GRAPH_STATE[own_key]: setattr(graph_cls, attr_name, _KEEP_GRAPH_STATE[orig_key])
            elif attr_name in graph_cls.__dict__: delattr(graph_cls, attr_name)
    _KEEP_GRAPH_STATE.update(enabled=False, graph_cls=None, orig_new=None, orig_init=None, orig_capture_end=None, had_own_new=False, had_own_init=False, had_own_capture_end=False)

def enable_keep_graph(cuda_graph_cls: Any = None) -> bool:
    """Idempotently default `torch.cuda.CUDAGraph(keep_graph=True)` and auto-instantiate."""
    if cuda_graph_cls is None:
        cuda_graph_cls = getattr(getattr(sys.modules.get("torch"), "cuda", None), "CUDAGraph", None)
    if cuda_graph_cls is None or any(not hasattr(cuda_graph_cls, attr) for attr in ("raw_cuda_graph", "instantiate", "capture_end")):
        return False
    with _KEEP_GRAPH_LOCK:
        if _KEEP_GRAPH_STATE["enabled"] and _KEEP_GRAPH_STATE["graph_cls"] is cuda_graph_cls: return True
        if _KEEP_GRAPH_STATE["enabled"]: _restore_keep_graph_unlocked()
        had_own_new, had_own_init, had_own_capture_end = "__new__" in cuda_graph_cls.__dict__, "__init__" in cuda_graph_cls.__dict__, "capture_end" in cuda_graph_cls.__dict__
        orig_new, orig_capture_end = cuda_graph_cls.__new__, cuda_graph_cls.capture_end
        orig_init = None if cuda_graph_cls.__init__ is object.__init__ else cuda_graph_cls.__init__
        def patched_new(cls, *args, **kwargs):
            auto_enabled = not args and "keep_graph" not in kwargs
            instance = orig_new(cls, *args, **({**kwargs, "keep_graph": True} if auto_enabled else kwargs))
            if auto_enabled:
                with contextlib.suppress(AttributeError): instance._rsibox_auto_keep_graph = True
            return instance
        def patched_init(self, *args, **kwargs):
            auto_enabled = getattr(self, "_rsibox_auto_keep_graph", False) and not args and "keep_graph" not in kwargs
            return orig_init(self, *args, **({**kwargs, "keep_graph": True} if auto_enabled else kwargs))
        def patched_capture_end(self, *args, **kwargs):
            end_result = orig_capture_end(self, *args, **kwargs)
            if getattr(self, "_rsibox_auto_keep_graph", False):
                has_exec = getattr(self, "_has_graph_exec", None)
                if has_exec is None: has_exec = _safe_handle(self, "raw_cuda_graph_exec") != 0
                if _safe_handle(self, "raw_cuda_graph") != 0 and not has_exec: self.instantiate()
            return end_result
        patched_new.__wrapped__, patched_capture_end.__wrapped__ = orig_new, orig_capture_end
        cuda_graph_cls.__new__, cuda_graph_cls.capture_end = patched_new, patched_capture_end
        if orig_init is not None:
            patched_init.__wrapped__ = orig_init
            cuda_graph_cls.__init__ = patched_init
        _KEEP_GRAPH_STATE.update(enabled=True, graph_cls=cuda_graph_cls, orig_new=orig_new, orig_init=orig_init, orig_capture_end=orig_capture_end, had_own_new=had_own_new, had_own_init=had_own_init, had_own_capture_end=had_own_capture_end)
        return True

def disable_keep_graph() -> bool:
    with _KEEP_GRAPH_LOCK:
        if not _KEEP_GRAPH_STATE["enabled"] or _KEEP_GRAPH_STATE["graph_cls"] is None: return False
        _restore_keep_graph_unlocked()
        return True

def is_keep_graph_enabled() -> bool:
    with _KEEP_GRAPH_LOCK: return bool(_KEEP_GRAPH_STATE["enabled"])

def snapshot_triton_commit(staged_list: list[Any], cells: list[Any]) -> dict[str, Any]:
    """Snapshot loaded Triton specializations before `JIT.apply` clears `device_caches`."""
    all_jits = [(plan_data, witness_key) for item in staged_list for plan_data, witness_key in item.jits]
    jit_func_ids = {id(plan_data["jf"].fn) for plan_data, _ in all_jits}
    if ineligible := next((reason for cond, reason in (
        (bool(cells), "has_repair_cells"), (not staged_list, "no_staged_modules"),
        (any(item.adds or item.pending_adds or item.removes or item.imports or getattr(item, "jit_adds", ()) for item in staged_list), "has_structural_python_edits"),
        (not all_jits, "not_triton_jit_edit"),
        (any(id(swap_entry[0]) not in jit_func_ids for item in staged_list for swap_entry in item.swaps), "mixed_non_triton_python_edit"),
    ) if cond), None): return {"eligible": False, "reason": ineligible}
    entries_by_func, affected_names, jit_summaries = {}, set(), []
    for plan_data, witness_key in all_jits:
        jit_summaries.append((kernel_jit.label(plan_data["jf"]), witness_key))
        for affected_jit in plan_data["affected"]:
            kernel_label = kernel_jit.label(affected_jit)
            if python_name := getattr(getattr(affected_jit, "fn", None), "__name__", None): affected_names.add(str(python_name))
            witness_pair = (witness_key, kernel_witness.launch_key(witness_key, kernel_label))
            for device_id, cache_entry in list((getattr(affected_jit, "device_caches", None) or {}).items()):
                if not cache_entry or not isinstance(cache_entry[0], dict): continue
                cache_tuple = tuple(cache_entry[2:])
                for cache_key, compiled_kernel in list(cache_entry[0].items()):
                    if meta_name := getattr(getattr(compiled_kernel, "metadata", None), "name", None): affected_names.add(str(meta_name))
                    if (old_func_int := _safe_int(getattr(compiled_kernel, "function", None))) == 0: continue
                    if (existing := entries_by_func.get(old_func_int)) is None:
                        entries_by_func[old_func_int] = {
                            "jit_fn": affected_jit, "kernel_label": kernel_label, "device_id": device_id, "cache_key": cache_key,
                            "old_kernel": compiled_kernel, "old_func_int": old_func_int, "cache_tuple": cache_tuple, "witness_pairs": [witness_pair],
                        }
                    elif witness_pair not in existing["witness_pairs"]: existing["witness_pairs"].append(witness_pair)
    return {"eligible": True, "entries_by_func": entries_by_func, "affected_kernel_names": sorted(affected_names), "jits": jit_summaries}

def check_abi_compatible(old_kernel: Any, new_kernel: Any) -> tuple[bool, str | None]:
    """Fail-closed ABI and launch-config compatibility check between old and new binaries."""
    old_src, new_src = getattr(old_kernel, "src", None), getattr(new_kernel, "src", None)
    old_meta, new_meta = getattr(old_kernel, "metadata", None), getattr(new_kernel, "metadata", None)
    if early_err := next((reason for cond, reason in (
        (_safe_int(getattr(new_kernel, "function", None)) == 0, "new_kernel_function_null"),
        (old_src is None or new_src is None, "missing_src"), (old_meta is None or new_meta is None, "missing_metadata"),
        (old_meta is not None and new_meta is not None and getattr(new_meta, "shared", None) is None, "missing_shared_metadata"),
    ) if cond), None): return False, early_err
    for src_attr, reason in (("signature", "signature_mismatch"), ("constants", "constants_mismatch"), ("attrs", "attrs_mismatch")):
        if getattr(old_src, src_attr, None) != getattr(new_src, src_attr, None): return False, reason
    old_gscratch, new_gscratch = _safe_int(getattr(old_meta, "global_scratch_size", 0)), _safe_int(getattr(new_meta, "global_scratch_size", 0))
    old_shared, new_shared = _safe_int(getattr(old_meta, "shared", 0)), int(new_meta.shared)
    block_dim_x = (_safe_int(getattr(new_meta, "warp_size", 32)) or 32) * (_safe_int(getattr(new_meta, "num_warps", 4)) or 4)
    max_threads = getattr(new_kernel, "n_max_threads", None)
    if meta_err := next((reason for cond, reason in (
        (new_gscratch != old_gscratch, f"global_scratch_size_changed:{old_gscratch}->{new_gscratch}"),
        (new_gscratch > 0 and _safe_int(getattr(new_meta, "global_scratch_align", 1)) > _safe_int(getattr(old_meta, "global_scratch_align", 1)), "global_scratch_align_grew"),
        (_safe_int(getattr(old_meta, "profile_scratch_size", 0)) != 0 or _safe_int(getattr(new_meta, "profile_scratch_size", 0)) != 0, "profile_scratch_nonzero"),
        ((_safe_int(getattr(old_meta, "num_ctas", 1)) or 1) != 1 or (_safe_int(getattr(new_meta, "num_ctas", 1)) or 1) != 1, "multi_cta_unsupported"),
        ((old_shared <= _SHARED_MEM_CARVE_LIMIT) != (new_shared <= _SHARED_MEM_CARVE_LIMIT), f"shared_mem_tier_changed:{old_shared}->{new_shared}"),
        (block_dim_x <= 0, f"invalid_block_dim_x:{block_dim_x}"),
        (max_threads is not None and block_dim_x > int(max_threads), f"block_dim_exceeds_max_threads:{block_dim_x}>{max_threads}"),
    ) if cond), None): return False, meta_err
    for attr_name, caster, default_val, reason in (
        ("launch_cooperative_grid", bool, False, "launch_cooperative_grid_changed"), ("launch_pdl", bool, False, "launch_pdl_changed"),
        ("tensordesc_meta", lambda raw: tuple(raw or ()), (), "tensordesc_meta_changed"), ("tmem_size", _safe_int, 0, "tmem_size_changed"),
        ("instrumentation_mode", lambda raw: str(raw or ""), "", "instrumentation_mode_changed"),
        ("cluster_dims", lambda raw: tuple(raw or ()), (), "cluster_dims_changed"), ("maxnreg", lambda raw: raw, None, "maxnreg_changed"),
    ):
        if caster(getattr(old_meta, attr_name, default_val)) != caster(getattr(new_meta, attr_name, default_val)): return False, reason
    return True, None

def _insert_device_cache(record: dict[str, Any], new_kernel: Any) -> tuple[Any, Any, Any, bool, Any, bool]:
    jit_fn, device_id, cache_key = record["jit_fn"], record["device_id"], record["cache_key"]
    device_caches, created_slot = jit_fn.device_caches, device_id not in jit_fn.device_caches
    cache_entry = device_caches[device_id] if (not created_slot or getattr(device_caches, "default_factory", None) is not None) else device_caches.setdefault(device_id, ({}, {}) + tuple(record.get("cache_tuple") or ()))
    had_prev, prev_val = cache_key in cache_entry[0], cache_entry[0].get(cache_key)
    cache_entry[0][cache_key] = new_kernel
    return (jit_fn, device_id, cache_key, had_prev, prev_val, created_slot)

def _restore_cache_entries(saved_entries: list[tuple[Any, Any, Any, bool, Any, bool]]) -> None:
    for jit_fn, device_id, cache_key, had_prev, prev_val, created_slot in reversed(saved_entries):
        device_caches = getattr(jit_fn, "device_caches", {})
        if (cache_entry := device_caches.get(device_id)) and isinstance(cache_entry[0], dict):
            if had_prev: cache_entry[0][cache_key] = prev_val
            else: cache_entry[0].pop(cache_key, None)
            if created_slot and not cache_entry[0]: device_caches.pop(device_id, None)

def recompile_specialization(record: dict[str, Any], *, insert_cache: bool = True) -> Any:
    """Compile one matched Triton specialization directly from its ASTSource and metadata."""
    from .recapture import cuda_device_context
    jit_fn, old_kernel = record["jit_fn"], record["old_kernel"]
    old_src, old_meta = old_kernel.src, old_kernel.metadata
    ast_source_cls = getattr(jit_fn, "ASTSource", None)
    if ast_source_cls is None: from triton.compiler import ASTSource as ast_source_cls
    compile_fn = getattr(jit_fn, "compile", None)
    if compile_fn is None: from triton.compiler import compile as compile_fn
    ast_source = ast_source_cls(jit_fn, old_src.signature, constexprs=old_src.constants, attrs=old_src.attrs)
    raw_options = old_meta._asdict() if hasattr(old_meta, "_asdict") else dict(vars(old_meta))
    with cuda_device_context(record["device_id"]):
        new_kernel = compile_fn(ast_source, target=getattr(old_meta, "target", None), options=_freeze_option_value(raw_options))
        if callable(getattr(new_kernel, "_init_handles", None)): new_kernel._init_handles()
    if insert_cache: _insert_device_cache(record, new_kernel)
    return new_kernel

def _lora_compatible(first_desc: Any, second_desc: Any) -> bool:
    return all(getattr(first_desc, attr) == getattr(second_desc, attr) for attr in ("num_active_loras", "has_lora") if hasattr(first_desc, attr) and hasattr(second_desc, attr))

def _partition_wrapper_entries(selected, wrappers, chosen, left):
    chosen_by_mode = {getattr(mode, "name", None) or str(mode): list(chosen[mode]) for mode in selected}
    left_by_mode = {getattr(mode, "name", None) or str(mode): list(left[mode]) for mode in selected}
    mode_by_name = {getattr(mode, "name", None) or str(mode): mode for mode in selected}
    graph_infos, info_by_handles, left_entry_refs = [], {}, []
    chosen_guard_keys = {graph_key(mode, desc) for mode in selected for desc in chosen[mode]}
    for wrapper, attr_name in wrappers:
        wrapper_mode = getattr(getattr(wrapper, "runtime_mode", None), "name", None) or str(getattr(wrapper, "runtime_mode", None) or "PIECEWISE")
        mode_chosen, mode_left = chosen_by_mode.get(wrapper_mode, ()), left_by_mode.get(wrapper_mode, ())
        chosen_set, left_set, mode_obj, entry_dict = set(mode_chosen), set(mode_left), mode_by_name.get(wrapper_mode, wrapper_mode), getattr(wrapper, attr_name)
        for entry_desc, entry_obj in entry_dict.items():
            entry_size = getattr(entry_desc, "num_tokens", entry_desc)
            if entry_desc not in chosen_set and (entry_desc in left_set or any(getattr(left_desc, "num_tokens", left_desc) == entry_size and _lora_compatible(entry_desc, left_desc) for left_desc in mode_left)):
                left_entry_refs.append((entry_dict, entry_desc))
                continue
            if entry_desc in chosen_set:
                entry_keys = [graph_key(getattr(entry_desc, "cg_mode", None) or wrapper_mode, entry_desc), graph_key(mode_obj, entry_desc)]
            else:
                matches = [candidate for candidate in mode_chosen if getattr(candidate, "num_tokens", candidate) == entry_size and _lora_compatible(entry_desc, candidate)]
                if not matches: continue
                entry_keys = [graph_key(getattr(entry_desc, "cg_mode", None) or wrapper_mode, entry_desc)] + [graph_key(mode_obj, candidate) for candidate in matches]
            deduped_keys = list(dict.fromkeys(entry_keys))
            chosen_guard_keys.update(deduped_keys)
            for graph_obj in extract_graphs(entry_obj):
                raw_graph, raw_exec = _safe_handle(graph_obj, "raw_cuda_graph"), _safe_handle(graph_obj, "raw_cuda_graph_exec")
                if raw_graph == 0 or raw_exec == 0: return None, None, None, "keep_graph_unavailable"
                if (existing_info := info_by_handles.get((raw_graph, raw_exec))) is None:
                    graph_info = {"raw_graph_handle": raw_graph, "raw_exec_handle": raw_exec, "guard_keys": list(deduped_keys)}
                    info_by_handles[(raw_graph, raw_exec)] = graph_info
                    graph_infos.append(graph_info)
                else: existing_info["guard_keys"].extend(key for key in deduped_keys if key not in existing_info["guard_keys"])
    return (graph_infos, chosen_guard_keys, left_entry_refs, None) if graph_infos else (None, None, None, "no_captured_graphs")

def _restore_node_params(driver: Any, patched: dict[str, Any], *, strict: bool = False) -> bool:
    cuda_success, node, exec_handle = driver.CUresult.CUDA_SUCCESS, patched["node"], patched["exec_handle"]
    get_status, fresh_params = driver.cuGraphKernelNodeGetParams(node)
    if get_status != cuda_success:
        if strict: raise _DriverPatchError(f"undo cuGraphKernelNodeGetParams failed: {get_status}")
        return False
    fresh_params.func = driver.CUfunction(patched["old_func_int"])
    fresh_params.blockDimX, fresh_params.blockDimY, fresh_params.blockDimZ = patched["old_block"]
    fresh_params.gridDimX, fresh_params.gridDimY, fresh_params.gridDimZ = patched["old_grid"]
    fresh_params.sharedMemBytes = patched["old_shared_mem"]
    steps = (
        ("cuGraphExecKernelNodeSetParams", lambda: driver.cuGraphExecKernelNodeSetParams(exec_handle, node, fresh_params)),
        ("cuGraphKernelNodeSetParams", lambda: driver.cuGraphKernelNodeSetParams(node, fresh_params)),
    )
    for call_name, call_fn in steps[0 if exec_handle is not None else 1:]:
        result = call_fn()
        if (status := result[0] if isinstance(result, tuple) else result) != cuda_success:
            if strict: raise _DriverPatchError(f"undo {call_name} failed: {status}")
            return False
    return True

def _scan_matching_nodes(driver: Any, graph_infos: list[dict[str, Any]], entries_by_func: dict[int, dict[str, Any]], affected_names: set[str]):
    cuda_success, kernel_type = driver.CUresult.CUDA_SUCCESS, driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
    child_graph_type = getattr(driver.CUgraphNodeType, "CU_GRAPH_NODE_TYPE_GRAPH", None)
    safe_types = {val for name in _SAFE_NON_KERNEL_NODE_NAMES if (val := getattr(driver.CUgraphNodeType, name, None)) is not None}
    get_func_name, matched_nodes = getattr(driver, "cuFuncGetName", None), []
    def checked(result, label: str):
        if result[0] != cuda_success: raise _FallbackSignal(f"{label}:{result[0]}")
        return result[1:]
    try:
        for graph_info in graph_infos:
            graph_handle, exec_handle = driver.CUgraph(graph_info["raw_graph_handle"]), driver.CUgraphExec(graph_info["raw_exec_handle"])
            graph_info["exec_handle"] = exec_handle
            _, num_nodes = checked(driver.cuGraphGetNodes(graph_handle, 0), "cuGraphGetNodes_count")
            if not num_nodes: continue
            nodes, _ = checked(driver.cuGraphGetNodes(graph_handle, num_nodes), "cuGraphGetNodes_list")
            for node in nodes:
                (node_type,) = checked(driver.cuGraphNodeGetType(node), "cuGraphNodeGetType")
                if node_type == child_graph_type: raise _FallbackSignal("unsupported_child_graph_node")
                if node_type != kernel_type:
                    if node_type not in safe_types: raise _FallbackSignal(f"unsupported_node_type:{node_type}")
                    continue
                (params,) = checked(driver.cuGraphKernelNodeGetParams(node), "cuGraphKernelNodeGetParams")
                if (old_func_int := int(params.func)) not in entries_by_func:
                    if affected_names and callable(get_func_name):
                        name_status, raw_name = get_func_name(params.func)
                        if name_status == cuda_success and raw_name and (raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, (bytes, bytearray)) else str(raw_name)) in affected_names:
                            raise _FallbackSignal("unmatched_affected_node")
                    continue
                old_block = (int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ))
                old_meta = getattr(entries_by_func[old_func_int]["old_kernel"], "metadata", None)
                meta_warp, meta_warps = getattr(old_meta, "warp_size", None), getattr(old_meta, "num_warps", None)
                if node_err := next((reason for cond, reason in (
                    (_safe_int(getattr(params, "extra", 0)) != 0, "unsupported_kernel_params_extra"),
                    (old_block[1:] != (1, 1), f"unsupported_non_1d_block_dim:{old_block}"),
                    (meta_warp is not None and meta_warps is not None and old_block[0] != int(meta_warp) * int(meta_warps), f"block_dim_metadata_mismatch:{old_block[0]}"),
                ) if cond), None): raise _FallbackSignal(node_err)
                matched_nodes.append({
                    "graph_info": graph_info, "exec_handle": exec_handle, "node": node, "params": params, "old_func_int": old_func_int,
                    "old_shared_mem": int(params.sharedMemBytes), "old_block": old_block, "old_grid": (int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ)),
                })
    except _FallbackSignal as signal: return None, str(signal)
    return (matched_nodes, None) if matched_nodes else (None, "no_matching_kernel_nodes")

def _apply_node_patches(driver: Any, matched_nodes: list[dict[str, Any]], compiled_by_old_func: dict[int, Any]):
    cuda_success, patched_records, revert_ok = driver.CUresult.CUDA_SUCCESS, [], True
    try:
        for match in matched_nodes:
            new_kernel = compiled_by_old_func[match["old_func_int"]]
            new_meta, params = new_kernel.metadata, match["params"]
            params.func = driver.CUfunction(int(new_kernel.function))
            params.blockDimX = (_safe_int(getattr(new_meta, "warp_size", 32)) or 32) * (_safe_int(getattr(new_meta, "num_warps", 4)) or 4)
            params.sharedMemBytes = int(new_meta.shared)
            exec_res = driver.cuGraphExecKernelNodeSetParams(match["exec_handle"], match["node"], params)
            if (exec_status := exec_res[0] if isinstance(exec_res, tuple) else exec_res) != cuda_success:
                raise _DriverPatchError(f"cuGraphExecKernelNodeSetParams:{exec_status}")
            tmpl_res = driver.cuGraphKernelNodeSetParams(match["node"], params)
            if (tmpl_status := tmpl_res[0] if isinstance(tmpl_res, tuple) else tmpl_res) != cuda_success:
                get_status, orig_params = driver.cuGraphKernelNodeGetParams(match["node"])
                revert_res = driver.cuGraphExecKernelNodeSetParams(match["exec_handle"], match["node"], orig_params) if get_status == cuda_success else None
                if get_status != cuda_success or (revert_res[0] if isinstance(revert_res, tuple) else revert_res) != cuda_success: revert_ok = False
                raise _DriverPatchError(f"cuGraphKernelNodeSetParams:{tmpl_status}")
            patched_records.append(match)
    except Exception as error:
        for patched in reversed(patched_records):
            if not _restore_node_params(driver, patched, strict=False): revert_ok = False
        return None, f"driver_patch_failed: {error}", not revert_ok
    return patched_records, None, False

def _rollback_patch_state(driver: Any, engine: Any, guard: Any, rollback_ctx: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try: engine.sync()
    except Exception as sync_error: errors.append(f"pre_undo_sync: {sync_error}")
    finally:
        for patched in rollback_ctx["undo_nodes"]:
            try: _restore_node_params(driver, patched, strict=True)
            except Exception as node_error: errors.append(str(node_error))
        for entry_dict, entry_desc, saved_entry in reversed(rollback_ctx["saved_left_entries"]): entry_dict[entry_desc] = saved_entry
        for mode, left_descs in rollback_ctx["removed_left"].items(): engine.keys[mode].update(left_descs)
        _restore_cache_entries(rollback_ctx["saved_cache_entries"])
        if rollback_ctx["registered_witness_keys"] and hasattr(guard, "clear_replay_witnesses"):
            guard.clear_replay_witnesses(rollback_ctx["registered_witness_keys"], owner=rollback_ctx["txn"])
        try: engine.sync()
        except Exception as sync_error: errors.append(f"post_undo_sync: {sync_error}")
        if selected_names := rollback_ctx["selected_names"]:
            if not errors:
                guard.restore_modes(rollback_ctx["pre_pool_epochs"], rollback_ctx["pre_stamps"])
                if hasattr(guard, "restore_replay_witnesses"): guard.restore_replay_witnesses(rollback_ctx["pre_replay_witnesses"], modes=selected_names)
            else:
                guard.invalidate_modes(selected_names)
                guard.bump_pool_epoch(f"poison:{rollback_ctx['txn']}", modes=selected_names)
    return errors

def try_pointer_swap(
    engine: Any, txn: str, journal: Any, witness: Any, guard: Any, selected: list[Any],
    wrappers: list[tuple[Any, str]], chosen: dict[Any, list[Any]], left: dict[Any, list[Any]],
    jit_snapshot: dict[str, Any] | None, *, driver: Any = None, compiler: Any = None,
) -> dict[str, Any]:
    """Attempt Tier-A surgical kernel-node pointer swapping; fail closed to R1 on any mismatch."""
    total_start = time.perf_counter()
    if pre_err := next((reason for cond, reason in (
        (not jit_snapshot, "no_triton_snapshot"),
        (bool(jit_snapshot) and not jit_snapshot.get("eligible"), (jit_snapshot or {}).get("reason") or "ineligible_snapshot"),
        (bool(jit_snapshot) and not (jit_snapshot.get("entries_by_func") or {}), "no_loaded_triton_kernels"),
    ) if cond), None): return _fail(pre_err)
    entries_by_func = jit_snapshot["entries_by_func"]
    if driver is None:
        try:
            from cuda.bindings import driver as cuda_driver
            driver = cuda_driver
        except ImportError: return _fail("cuda_driver_unavailable")
    graph_infos, chosen_guard_keys, left_entry_refs, collect_err = _partition_wrapper_entries(selected, wrappers, chosen, left)
    if collect_err is not None: return _fail(collect_err)
    matched_nodes, scan_err = _scan_matching_nodes(driver, graph_infos, entries_by_func, set(jit_snapshot.get("affected_kernel_names") or ()))
    if scan_err is not None: return _fail(scan_err)

    unique_old_funcs = list(dict.fromkeys(match["old_func_int"] for match in matched_nodes))
    compiled_by_old_func, saved_cache_entries, compile_start = {}, [], time.perf_counter()
    for old_func_int in unique_old_funcs:
        record = entries_by_func[old_func_int]
        try:
            new_kernel = compiler(record) if compiler is not None else recompile_specialization(record, insert_cache=False)
            saved_cache_entries.append(_insert_device_cache(record, new_kernel))
        except Exception as error:
            _restore_cache_entries(saved_cache_entries)
            return _fail(f"compile_failed: {type(error).__name__}: {error}")
        abi_ok, abi_reason = check_abi_compatible(record["old_kernel"], new_kernel)
        if not abi_ok:
            _restore_cache_entries(saved_cache_entries)
            return _fail(f"abi_mismatch: {abi_reason}")
        compiled_by_old_func[old_func_int] = new_kernel
    compile_ms = round((time.perf_counter() - compile_start) * 1000.0, 3)

    selected_names = [getattr(mode, "name", None) or str(mode) for mode in selected]
    pre_pool_epochs, pre_stamps = guard.snapshot_modes(selected_names)
    pre_replay_witnesses = guard.snapshot_replay_witnesses(selected_names) if hasattr(guard, "snapshot_replay_witnesses") else {}
    patch_start = time.perf_counter()
    engine.sync()
    patched_records, patch_err, poisoned = _apply_node_patches(driver, matched_nodes, compiled_by_old_func)
    if patch_err is not None:
        _restore_cache_entries(saved_cache_entries)
        engine.sync()
        return _fail(patch_err, poisoned=poisoned)
    engine.sync()
    patch_ms = round((time.perf_counter() - patch_start) * 1000.0, 3)

    rollback_ctx: dict[str, Any] = {
        "txn": txn, "undo_nodes": list(reversed(patched_records)), "saved_left_entries": [], "removed_left": {}, "saved_cache_entries": saved_cache_entries,
        "registered_witness_keys": [], "selected_names": selected_names, "pre_pool_epochs": pre_pool_epochs, "pre_stamps": pre_stamps, "pre_replay_witnesses": pre_replay_witnesses,
    }
    try:
        rollback_ctx["saved_left_entries"] = [(entry_dict, entry_desc, entry_dict.pop(entry_desc)) for entry_dict, entry_desc in left_entry_refs if entry_desc in entry_dict]
        rollback_ctx["removed_left"] = {mode: list(descs) for mode, descs in left.items() if descs}
        for mode, left_descs in rollback_ctx["removed_left"].items():
            for left_desc in left_descs: engine.keys[mode].discard(left_desc)
        witness_keys_by_guard_key, patched_graph_handles = {}, set()
        with witness.capture():
            for match in patched_records:
                graph_info = match["graph_info"]
                patched_graph_handles.add((graph_info["raw_graph_handle"], graph_info["raw_exec_handle"]))
                for witness_key, launch_witness_key in entries_by_func[match["old_func_int"]]["witness_pairs"]:
                    witness.hit(witness_key)
                    witness.hit(launch_witness_key)
                    for guard_entry_key in graph_info["guard_keys"]:
                        witness_keys_by_guard_key.setdefault(guard_entry_key, []).extend((witness_key, launch_witness_key))
        for guard_entry_key in chosen_guard_keys: guard.stamp(guard_entry_key)
        for guard_entry_key, key_list in witness_keys_by_guard_key.items():
            guard.register_replay_witness(guard_entry_key, witness, key_list, owner=txn)
            rollback_ctx["registered_witness_keys"].append(guard_entry_key)
        retained_kernels = [entries_by_func[func_id]["old_kernel"] for func_id in unique_old_funcs] + list(compiled_by_old_func.values())
        pin_graph_kernels(patched_graph_handles, retained_kernels)
        undo_lock, undo_done = threading.Lock(), False
        def undo_pointer_swap() -> None:
            nonlocal undo_done
            with undo_lock:
                if undo_done: return
                undo_done = True
            if errors := _rollback_patch_state(driver, engine, guard, rollback_ctx): raise _DriverPatchError("; ".join(errors))
        undo_pointer_swap.retained_kernels = retained_kernels
        journal.record_call(undo_pointer_swap, f"Tier-A pointer swap of {txn}")
    except Exception as error:
        return _fail(f"post_patch_failed: {type(error).__name__}: {error}", poisoned=bool(_rollback_patch_state(driver, engine, guard, rollback_ctx)))
    return {"applied": True, "restore": undo_pointer_swap, "report": {
        "tier": "A", "patched_nodes": len(patched_records), "patched_graphs": len(patched_graph_handles), "compiled_specializations": len(compiled_by_old_func),
        "patch_ms": patch_ms, "compile_ms": compile_ms, "seconds": round(time.perf_counter() - total_start, 6), "graphs": len(patched_graph_handles), "captured": [],
    }}
