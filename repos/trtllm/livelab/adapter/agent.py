"""Worker-side RSIBox agent for TensorRT-LLM (PyTorch backend) and the CPU toy fixture."""
from __future__ import annotations

import gc, importlib, json, linecache, os, random, sys, time
from typing import Any
from kernel import coordinator, findings
from rsibox import harness, ipc


def _cuda_module() -> Any:
    torch_mod = sys.modules.get("torch")
    return torch_mod.cuda if (torch_mod is not None and getattr(torch_mod.cuda, "is_initialized", lambda: False)()) else None

def _cuda_vram_mib() -> float | None:
    cuda_mod = _cuda_module()
    if cuda_mod is None:
        return None
    free_bytes, total_bytes = cuda_mod.mem_get_info()
    return round((total_bytes - free_bytes) / (1024 ** 2), 2)

def _sync_linecache(participant: coordinator.LocalParticipant, module_names: list[str] | set[str]) -> None:
    for module_name in module_names:
        file_path = getattr(sys.modules.get(module_name), "__file__", None) or harness.module_file(module_name)
        live_source = participant._live_source(module_name)
        if file_path and live_source is not None:
            linecache.cache[file_path] = (len(live_source), None, live_source.splitlines(True), file_path)
        elif file_path:
            linecache.cache.pop(file_path, None)

def _build_engine(mode: str, engine_config: dict[str, Any]) -> Any:
    if mode == "toy":
        from adapter.fixture.py_executor import ToyPyExecutor
        return ToyPyExecutor()
    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    kv_config = KvCacheConfig(free_gpu_memory_fraction=float(engine_config.get("free_gpu_memory_fraction", 0.15)),
                              enable_block_reuse=bool(engine_config.get("enable_block_reuse", False)))
    return LLM(model=engine_config["model"], backend="pytorch",
               tensor_parallel_size=int(engine_config.get("tensor_parallel_size", 1)), kv_cache_config=kv_config,
               max_batch_size=int(engine_config.get("max_batch_size", 4)), max_num_tokens=int(engine_config.get("max_num_tokens", 2048)),
               disable_overlap_scheduler=bool(engine_config.get("disable_overlap_scheduler", True)), **(engine_config.get("extra_kwargs") or {}))

def _run_engine_generate(mode: str, engine: Any, workload: dict[str, Any]) -> list[list[int]]:
    if mode == "toy":
        return engine.generate(workload.get("prompts") or [[2, 5, 1], [4, 7, 3]], max_tokens=int(workload.get("max_tokens", 6)))
    from tensorrt_llm import SamplingParams
    prompts = list(workload.get("prompts") or ["The capital of France is"])
    params = SamplingParams(max_tokens=int(workload.get("max_tokens", 16)), temperature=float(workload.get("temperature", 0.0)))
    return [list(item.outputs[0].token_ids) for item in engine.generate(prompts, sampling_params=params)]

def _reset_worker_state(mode: str, engine: Any, seed: int) -> list[str]:
    py_executor = getattr(getattr(engine, "_executor", None), "engine", None) or engine
    if getattr(py_executor, "active_requests", None):
        raise RuntimeError("cannot reset state while requests are active")
    if mode == "toy":
        py_executor.history.clear()
        py_executor.kv_cache_manager.impl.entries.clear()
        reset_items = ["kv_cache", "history", "rng"]
    else:
        kv_manager = getattr(py_executor, "kv_cache_manager", None)
        if kv_manager is not None and hasattr(kv_manager, "flush_iteration_events"):
            kv_manager.flush_iteration_events()
        reset_items = ["kv_cache_events", "rng"]
    random.seed(seed)
    torch_mod, cuda_mod = sys.modules.get("torch"), _cuda_module()
    if torch_mod is not None:
        torch_mod.manual_seed(seed)
    if cuda_mod is not None:
        cuda_mod.manual_seed_all(seed)
    return reset_items

def _transition_worker_memory(engine: Any, tags: list[str], *, release: bool) -> dict[str, Any]:
    py_executor = getattr(getattr(engine, "_executor", None), "engine", None) or engine
    if release and getattr(py_executor, "active_requests", None):
        return findings.not_ok("cannot quiesce while requests are active")
    vram_before, touched, start_time = _cuda_vram_mib(), [], time.perf_counter()
    if "kv_cache" in tags:
        kv_manager = getattr(py_executor, "kv_cache_manager", None)
        kv_impl, req_method = getattr(kv_manager, "impl", None), ("release_pools" if release else "allocate_pools")
        if kv_impl is None or not hasattr(kv_impl, req_method):
            return findings.not_ok(f"kv_cache_manager.impl.{req_method} is unavailable")
        if release:
            kv_impl.release_pools()
        else:
            kv_impl.allocate_pools(False)
            for attr, getter in (("kv_cache_pool_pointers", "get_block_pool_pointers"), ("kv_cache_pool_mapping", "get_layer_to_pool_mapping")):
                if hasattr(kv_impl, getter):
                    setattr(kv_manager, attr, getattr(kv_impl, getter)())
        touched.append("kv_cache")
    if "cuda_graphs" in tags:
        model_engine, resource_manager = getattr(py_executor, "model_engine", None), getattr(py_executor, "resource_manager", None)
        if release and model_engine is not None and hasattr(model_engine, "_release_cuda_graphs"):
            model_engine._release_cuda_graphs()
            touched.append("cuda_graphs")
        elif not release and model_engine is not None and hasattr(model_engine, "warmup") and resource_manager is not None:
            model_engine.warmup(resource_manager)
            touched.append("cuda_graphs")
    if vm_tags := [tag for tag in tags if tag not in ("kv_cache", "cuda_graphs")]:
        vm_mod = importlib.import_module("tensorrt_llm._torch.virtual_memory")
        (vm_mod.release_with_tag if release else vm_mod.materialize_with_tag)(*vm_tags)
        touched.extend(vm_tags)
    cuda_mod = _cuda_module()
    if release:
        gc.collect()
        if cuda_mod is not None:
            cuda_mod.empty_cache()
    if cuda_mod is not None:
        cuda_mod.synchronize()
    return findings.ok(**{("released" if release else "resumed"): touched}, vram_before_mib=vram_before,
                vram_after_mib=_cuda_vram_mib(), seconds=round(time.perf_counter() - start_time, 6))

def _serve_worker(config: dict[str, Any]) -> None:
    mode, rank_index, seed = config["mode"], int(config.get("rank", 0)), int(config.get("seed", 0))
    engine = _build_engine(mode, config.get("engine_config") or {})
    for module_name in config.get("modules") or ():
        importlib.import_module(module_name)
    py_executor = getattr(getattr(engine, "_executor", None), "engine", None) or engine
    reset_hook = lambda: _reset_worker_state(mode, engine, seed)
    participant = coordinator.LocalParticipant(f"rank{rank_index}", scope={"engine": engine, "py_executor": py_executor},
                                     resets={"state": reset_hook, "R3": reset_hook})
    staged_modules: dict[str, set[str]] = {}
    fault_state: dict[str, Any] = {}
    should_stop = False

    def handle_request(request: dict[str, Any]) -> dict[str, Any]:
        nonlocal should_stop
        method, payload = request.get("method"), request.get("payload")
        if fault_state.get("lost") and method != "shutdown":
            return findings.not_ok(f"rank{rank_index}: simulated lost participant")
        if method == "ping":
            return findings.ok(rank=rank_index, pid=os.getpid())
        if method in ("park", "release_gap", "shutdown"):
            should_stop = should_stop or (method == "shutdown")
            return findings.ok(rank=rank_index, parked=(method == "park"))
        if method == "inject":
            fault_kwargs = dict(payload or {})
            fault = fault_kwargs.pop("fault", None)
            if fault in ("fail_apply", "fail_commit", "tamper"):
                return harness.inject_participant_fault(participant, fault, **fault_kwargs)
            if fault in ("crash", "hang", "lost"):
                fault_state[fault] = fault_kwargs.get("hang_s", True)
                return findings.ok(fault=fault, rank=rank_index)
            return findings.not_ok(f"unknown fault {fault!r}")
        if method == "generate":
            if fault_state.pop("crash", False):
                os._exit(1)
            if hang_seconds := fault_state.pop("hang", None):
                time.sleep(float(hang_seconds))
            return findings.ok(output=_run_engine_generate(mode, engine, payload or {}))
        if method in ("quiesce_memory", "resume_memory"):
            return _transition_worker_memory(engine, list((payload or {}).get("tags") or ()), release=(method == "quiesce_memory"))
        if method == "prepare":
            specs = (payload or {}).get("modules") or {}
            for module_name, module_spec in specs.items():
                file_path, source_text = getattr(sys.modules.get(module_name), "__file__", None) or harness.module_file(module_name), module_spec.get("source")
                if file_path and source_text is not None:
                    linecache.cache[file_path] = (len(source_text), None, source_text.splitlines(True), file_path)
            reply = participant.prepare(payload)
            staged_modules[str(reply.get("txn"))] = set(specs)
            if not reply.get("ok"):
                _sync_linecache(participant, list(specs))
            return reply
        if hasattr(participant, str(method)):
            reply = getattr(participant, str(method))(payload)
            if method in ("commit", "abort", "undo"):
                _sync_linecache(participant, {name for names in staged_modules.values() for name in names})
            return reply
        return findings.not_ok(f"unknown method {method!r}")

    try:
        if config.get("transport", "unix") == "file":
            with ipc.FileChannelServer(config["endpoint"], handle_request, poll_s=0.005, start=True):
                while not should_stop:
                    time.sleep(0.01)
        else:
            with ipc.UnixJsonServer(config["endpoint"], handler=None) as server:
                while not should_stop:
                    request, send_reply = server.next()
                    try:
                        reply = handle_request(request)
                    except BaseException as error:
                        reply = findings.not_ok(f"{request.get('method')}: {type(error).__name__}: {error}")
                    send_reply(reply)
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()

def main(argv: list[str] | None = None) -> int:
    with open((sys.argv[1:] if argv is None else argv)[0], encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("action") != "fresh":
        _serve_worker(spec)
        return 0
    engine = _build_engine(spec["mode"], spec.get("engine_config") or {})
    try:
        tokens = _run_engine_generate(spec["mode"], engine, spec.get("workload") or {})
        print("FRESH " + json.dumps({"output": {f"rank{idx}": tokens for idx in range(int(spec.get("rank_count", 1)))}}), flush=True)
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
