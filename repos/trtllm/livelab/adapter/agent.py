"""Worker-side RSIBox agent for TensorRT-LLM (PyTorch backend) and the CPU toy fixture."""
from __future__ import annotations

import gc
import importlib
import json
import linecache
import os
import queue
import random
import sys
import time
from typing import Any

from kernel import coordinator, findings
from rsibox import harness, ipc

PARTICIPANT_METHODS = frozenset({
    "prepare",
    "validate",
    "commit",
    "abort",
    "undo",
    "status",
    "proof",
    "witness_check",
    "reset",
    "at_base",
    "fingerprint",
})
BLOCKED_WHILE_QUIESCED = frozenset({"generate", "prepare", "validate", "commit", "reset"})
CAPTURED_PREFIXES = (
    "tensorrt_llm._torch.models.",
    "tensorrt_llm._torch.modules.",
    "tensorrt_llm._torch.attention_backend.",
)


def _cuda_module() -> Any:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and getattr(torch_mod.cuda, "is_initialized", lambda: False)():
        return torch_mod.cuda
    return None


def _cuda_vram_mib() -> float | None:
    cuda_mod = _cuda_module()
    if cuda_mod is None:
        return None
    free_bytes, total_bytes = cuda_mod.mem_get_info()
    return round((total_bytes - free_bytes) / (1024 ** 2), 2)


def _snapshot_pool_pointers(kv_impl: Any) -> Any:
    if kv_impl is None or not hasattr(kv_impl, "get_block_pool_pointers"):
        return None
    raw_pointers = kv_impl.get_block_pool_pointers()
    if hasattr(raw_pointers, "detach"):
        return raw_pointers.detach().cpu().tolist()
    if isinstance(raw_pointers, (list, tuple)):
        return list(raw_pointers)
    return raw_pointers


def _sync_linecache(participant: coordinator.LocalParticipant, module_names: list[str] | set[str]) -> None:
    for module_name in module_names:
        file_path = getattr(sys.modules.get(module_name), "__file__", None) or harness.module_file(module_name)
        live_source = participant._live_source(module_name)
        if not file_path:
            continue
        if live_source is not None:
            linecache.cache[file_path] = (len(live_source), None, live_source.splitlines(True), file_path)
        else:
            linecache.cache.pop(file_path, None)


def _extract_py_executor(engine: Any) -> Any:
    return getattr(getattr(engine, "_executor", None), "engine", None) or engine


def _build_engine(mode: str, engine_config: dict[str, Any]) -> Any:
    if mode == "toy":
        from adapter.fixture.py_executor import ToyPyExecutor

        extra = engine_config.get("extra_kwargs") or {}
        return ToyPyExecutor(shift_pointers_on_alloc=bool(extra.get("shift_pointers_on_alloc", False)))

    tp_size = int(engine_config.get("tensor_parallel_size", 1))
    if tp_size == 1:
        os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    else:
        os.environ.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig

    kv_config = KvCacheConfig(
        free_gpu_memory_fraction=float(engine_config.get("free_gpu_memory_fraction", 0.15)),
        enable_block_reuse=bool(engine_config.get("enable_block_reuse", False)),
    )
    return LLM(
        model=engine_config["model"],
        backend="pytorch",
        tensor_parallel_size=int(engine_config.get("tensor_parallel_size", 1)),
        kv_cache_config=kv_config,
        max_batch_size=int(engine_config.get("max_batch_size", 4)),
        max_num_tokens=int(engine_config.get("max_num_tokens", 2048)),
        disable_overlap_scheduler=bool(engine_config.get("disable_overlap_scheduler", True)),
        **(engine_config.get("extra_kwargs") or {}),
    )


def _run_engine_generate(mode: str, engine: Any, workload: dict[str, Any]) -> list[list[int]]:
    if mode == "toy":
        prompts = workload.get("prompts") or [[2, 5, 1], [4, 7, 3]]
        return engine.generate(prompts, max_tokens=int(workload.get("max_tokens", 6)))

    from tensorrt_llm import SamplingParams

    prompts = list(workload.get("prompts") or ["The capital of France is"])
    params = SamplingParams(
        max_tokens=int(workload.get("max_tokens", 16)),
        temperature=float(workload.get("temperature", 0.0)),
    )
    outputs = engine.generate(prompts, sampling_params=params, use_tqdm=False)
    return [list(item.outputs[0].token_ids) for item in outputs]


def _reset_worker_state(mode: str, engine: Any, seed: int) -> list[str]:
    py_executor = _extract_py_executor(engine)
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


def _release_single_tag(
    py_executor: Any, tag: str, memory_state: dict[str, Any]
) -> None:
    if tag == "kv_cache":
        kv_manager = getattr(py_executor, "kv_cache_manager", None)
        kv_impl = getattr(kv_manager, "impl", None)
        if kv_impl is None or not hasattr(kv_impl, "release_pools"):
            raise RuntimeError("kv_cache_manager.impl.release_pools is unavailable")
        memory_state["pool_pointers_before"] = _snapshot_pool_pointers(kv_impl)
        kv_impl.release_pools()
        kv_manager.kv_cache_pool_pointers = []
        kv_manager.kv_cache_pool_mapping = []
        return
    if tag == "cuda_graphs":
        model_engine = getattr(py_executor, "model_engine", None)
        if model_engine is None or not hasattr(model_engine, "_release_cuda_graphs"):
            raise RuntimeError("model_engine._release_cuda_graphs is unavailable")
        model_engine._release_cuda_graphs()
        return
    vm_mod = importlib.import_module("tensorrt_llm._torch.virtual_memory")
    vm_mod.release_with_tag(tag)


def _resume_single_tag(
    py_executor: Any,
    tag: str,
    tags_being_resumed: list[str],
    memory_state: dict[str, Any],
) -> None:
    if tag == "kv_cache":
        kv_manager = getattr(py_executor, "kv_cache_manager", None)
        kv_impl = getattr(kv_manager, "impl", None)
        if kv_impl is None or not hasattr(kv_impl, "allocate_pools"):
            raise RuntimeError("kv_cache_manager.impl.allocate_pools is unavailable")
        kv_impl.allocate_pools(False)
        if hasattr(kv_impl, "get_block_pool_pointers"):
            kv_manager.kv_cache_pool_pointers = kv_impl.get_block_pool_pointers()
        if hasattr(kv_impl, "get_layer_to_pool_mapping"):
            kv_manager.kv_cache_pool_mapping = kv_impl.get_layer_to_pool_mapping()
        pointers_before = memory_state.get("pool_pointers_before")
        pointers_after = _snapshot_pool_pointers(kv_impl)
        if (
            "cuda_graphs" not in tags_being_resumed
            and pointers_before is not None
            and pointers_after != pointers_before
        ):
            raise RuntimeError(
                f"KV cache pool pointers shifted across quiesce/resume ({pointers_before!r} -> {pointers_after!r})"
            )
        return
    if tag == "cuda_graphs":
        model_engine = getattr(py_executor, "model_engine", None)
        resource_manager = getattr(py_executor, "resource_manager", None)
        if model_engine is None or not hasattr(model_engine, "warmup") or resource_manager is None:
            raise RuntimeError("model_engine.warmup is unavailable")
        model_engine.warmup(resource_manager)
        return
    vm_mod = importlib.import_module("tensorrt_llm._torch.virtual_memory")
    vm_mod.materialize_with_tag(tag)


def _ordered_resume_tags(tags: list[str]) -> list[str]:
    ordered = list(reversed(list(dict.fromkeys(tags))))
    if "kv_cache" in ordered and "cuda_graphs" in ordered:
        kv_idx = ordered.index("kv_cache")
        graph_idx = ordered.index("cuda_graphs")
        if kv_idx > graph_idx:
            ordered.pop(kv_idx)
            ordered.insert(graph_idx, "kv_cache")
    return ordered


def _transition_worker_memory(
    engine: Any,
    tags: list[str],
    *,
    release: bool,
    enable_block_reuse: bool,
    memory_state: dict[str, Any],
) -> dict[str, Any]:
    py_executor = _extract_py_executor(engine)
    quiesced_tags: set[str] = memory_state["quiesced_tags"]
    if not tags:
        return findings.not_ok("memory transition requires at least one tag", rejected=True)
    if release:
        ordered_tags = list(dict.fromkeys(tags))
        if "kv_cache" in ordered_tags and enable_block_reuse:
            return findings.not_ok("quiesce_memory is unsupported when enable_block_reuse=True", rejected=True)
        if quiesced_tags:
            return findings.not_ok(f"worker is already quiesced ({sorted(quiesced_tags)})", rejected=True)
        if getattr(py_executor, "active_requests", None):
            return findings.not_ok("cannot quiesce while requests are active", rejected=True)
    else:
        ordered_tags = _ordered_resume_tags(tags)
        missing = [tag for tag in ordered_tags if tag not in quiesced_tags]
        if missing:
            return findings.not_ok(f"cannot resume tag(s) not quiesced: {missing}", rejected=True)

    vram_before = _cuda_vram_mib()
    touched: list[str] = []
    start_time = time.perf_counter()
    try:
        for tag in ordered_tags:
            if release:
                _release_single_tag(py_executor, tag, memory_state)
            else:
                _resume_single_tag(py_executor, tag, ordered_tags, memory_state)
            touched.append(tag)
    except Exception as error:
        if release and touched:
            unwound: list[str] = []
            unwind_errors: list[str] = []
            for prev_tag in _ordered_resume_tags(touched):
                try:
                    _resume_single_tag(py_executor, prev_tag, touched, memory_state)
                    unwound.append(prev_tag)
                except Exception as unwind_error:
                    unwind_errors.append(f"{prev_tag}: {type(unwind_error).__name__}: {unwind_error}")
            remaining = [tag for tag in touched if tag not in unwound]
            if remaining:
                quiesced_tags.update(remaining)
                return findings.not_ok(
                    f"{type(error).__name__}: {error} (partial unwind failed: {'; '.join(unwind_errors)})",
                    partial_released=remaining,
                )
        return findings.not_ok(f"{type(error).__name__}: {error}")

    if release:
        quiesced_tags.update(touched)
        gc.collect()
    else:
        quiesced_tags.difference_update(touched)
        if not quiesced_tags:
            memory_state["pool_pointers_before"] = None
    cuda_mod = _cuda_module()
    if cuda_mod is not None:
        if release:
            cuda_mod.empty_cache()
        cuda_mod.synchronize()
    result_key = "released" if release else "resumed"
    return findings.ok(
        **{result_key: touched},
        vram_before_mib=vram_before,
        vram_after_mib=_cuda_vram_mib(),
        seconds=round(time.perf_counter() - start_time, 6),
    )


def _is_captured_module(module_name: str) -> bool:
    return any(module_name.startswith(prefix) for prefix in CAPTURED_PREFIXES)


def _recapture_cuda_graphs(engine: Any, rank_index: int, *, eager: bool = False) -> dict[str, Any]:
    start_time = time.perf_counter()
    py_executor = _extract_py_executor(engine)
    model_engine = getattr(py_executor, "model_engine", None)
    cuda_graphs = getattr(model_engine, "_cuda_graphs", None)
    released = sum(
        len(drafts) if isinstance(drafts, dict) else 1
        for drafts in (cuda_graphs.values() if isinstance(cuda_graphs, dict) else ())
    )
    if model_engine is not None and hasattr(model_engine, "_release_cuda_graphs"):
        model_engine._release_cuda_graphs()
    elif isinstance(cuda_graphs, dict):
        cuda_graphs.clear()
    gc.collect()
    cuda_mod = _cuda_module()
    if cuda_mod is not None:
        cuda_mod.empty_cache()
        cuda_mod.synchronize()
    if eager and model_engine is not None and hasattr(model_engine, "warmup"):
        resource_manager = getattr(py_executor, "resource_manager", None)
        if resource_manager is not None:
            model_engine.warmup(resource_manager)
    return findings.ok(
        rank=rank_index,
        graphs=released,
        eager=bool(eager),
        seconds=round(time.perf_counter() - start_time, 6),
    )


def make_worker_handler(
    mode: str,
    engine: Any,
    rank_index: int,
    *,
    seed: int = 0,
    enable_block_reuse: bool = False,
    tracked_modules: tuple[str, ...] | list[str] = (),
) -> tuple[Any, Any]:
    """Build `(handle_request, should_stop_fn)` shared by single-process and MPI workers."""
    for module_name in tracked_modules:
        importlib.import_module(module_name)

    py_executor = _extract_py_executor(engine)
    reset_hook = lambda: _reset_worker_state(mode, engine, seed)
    participant = coordinator.LocalParticipant(
        f"rank{rank_index}",
        scope={"engine": engine, "py_executor": py_executor},
        resets={"state": reset_hook, "R3": reset_hook},
    )
    staged_modules: dict[str, set[str]] = {}
    committed_modules: dict[str, set[str]] = {}
    recaptured_txns: dict[str, bool] = {}
    touched_modules: set[str] = set()
    fault_state: dict[str, Any] = {}
    memory_state: dict[str, Any] = {"quiesced_tags": set(), "pool_pointers_before": None}
    stop_flag = [False]

    def handle_request(request: dict[str, Any]) -> dict[str, Any]:
        method = str(request.get("method") or "")
        payload = request.get("payload")

        if fault_state.get("lost") and method != "shutdown":
            return findings.not_ok(f"rank{rank_index}: simulated lost participant")
        if memory_state["quiesced_tags"] and method in BLOCKED_WHILE_QUIESCED:
            return findings.not_ok(f"rank{rank_index}: worker is quiesced; resume_memory required before {method}")
        if method == "ping":
            return findings.ok(rank=rank_index, pid=os.getpid())
        if method == "read_source":
            module_name = str((payload or {}).get("module") or "")
            file_path = getattr(sys.modules.get(module_name), "__file__", None) or harness.module_file(module_name)
            cached_text = "".join(linecache.getlines(file_path)) if file_path else ""
            return findings.ok(module=module_name, source=harness.read_module(module_name), linecache=cached_text)
        if method in ("park", "release_gap", "shutdown"):
            if method == "shutdown":
                stop_flag[0] = True
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
            hang_seconds = fault_state.pop("hang", None)
            if hang_seconds:
                time.sleep(float(hang_seconds))
            return findings.ok(output=_run_engine_generate(mode, engine, payload or {}))
        if method == "recapture":
            spec = dict(payload or {})
            txn_id = spec.get("txn")
            reply = _recapture_cuda_graphs(engine, rank_index, eager=bool(spec.get("eager", False)))
            if reply.get("ok") and txn_id is not None:
                recaptured_txns[str(txn_id)] = bool(spec.get("eager", False))
            return reply
        if method in ("quiesce_memory", "resume_memory"):
            tags = list((payload or {}).get("tags") or ())
            return _transition_worker_memory(
                engine,
                tags,
                release=(method == "quiesce_memory"),
                enable_block_reuse=enable_block_reuse,
                memory_state=memory_state,
            )
        if method == "prepare":
            specs = (payload or {}).get("modules") or {}
            touched_modules.update(specs)
            for module_name, module_spec in specs.items():
                file_path = getattr(sys.modules.get(module_name), "__file__", None) or harness.module_file(module_name)
                source_text = module_spec.get("source")
                if file_path and source_text is not None:
                    linecache.cache[file_path] = (len(source_text), None, source_text.splitlines(True), file_path)
            reply = participant.prepare(payload)
            staged_modules[str(reply.get("txn"))] = set(specs)
            if not reply.get("ok"):
                _sync_linecache(participant, list(specs))
                staged_modules.pop(str(reply.get("txn")), None)
            return reply
        if method in PARTICIPANT_METHODS:
            if method == "commit":
                _sync_linecache(participant, touched_modules)
            reply = getattr(participant, method)(payload)
            if method in ("commit", "abort", "undo"):
                txn_id, _deadline = coordinator.txn_ref(payload)
                txn_key = str(txn_id)
                popped = staged_modules.pop(txn_key, set())
                if method == "commit" and reply.get("ok"):
                    committed_modules[txn_key] = set(popped)
                elif method == "undo":
                    undone_mods = committed_modules.pop(txn_key, set()) | popped
                    was_eager = recaptured_txns.pop(txn_key, None)
                    if reply.get("ok") and (was_eager is not None or any(_is_captured_module(m) for m in undone_mods)):
                        recap = _recapture_cuda_graphs(engine, rank_index, eager=bool(was_eager))
                        if not recap.get("ok"):
                            return recap
                tracked = touched_modules | popped | {name for names in staged_modules.values() for name in names}
                _sync_linecache(participant, tracked)
            return reply
        return findings.not_ok(f"unknown method {method!r}")

    return handle_request, (lambda: stop_flag[0])


def _serve_worker(config: dict[str, Any]) -> None:
    mode = config["mode"]
    rank_index = int(config.get("rank", 0))
    seed = int(config.get("seed", 0))
    parent_pid = int(config.get("parent_pid") or os.getppid())
    engine_config = config.get("engine_config") or {}
    enable_block_reuse = bool(engine_config.get("enable_block_reuse", False))

    engine = _build_engine(mode, engine_config)
    handle_request, should_stop_fn = make_worker_handler(
        mode=mode,
        engine=engine,
        rank_index=rank_index,
        seed=seed,
        enable_block_reuse=enable_block_reuse,
        tracked_modules=list(config.get("modules") or ()),
    )

    try:
        if config.get("transport", "unix") == "file":
            with ipc.FileChannelServer(config["endpoint"], handle_request, poll_s=0.005, start=True):
                while not should_stop_fn() and os.getppid() == parent_pid:
                    time.sleep(0.05)
        else:
            with ipc.UnixJsonServer(config["endpoint"], handler=None) as server:
                while not should_stop_fn() and os.getppid() == parent_pid:
                    try:
                        request, send_reply = server.next(timeout=1.0)
                    except queue.Empty:
                        continue
                    try:
                        reply = handle_request(request)
                    except BaseException as error:
                        reply = findings.not_ok(f"{request.get('method')}: {type(error).__name__}: {error}")
                    send_reply(reply)
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()


def _serve_driver(config: dict[str, Any]) -> None:
    """Driver process for `TP >= 2` MPI mode: owns `LLM` and serves `generate` / `ping` / `shutdown`."""
    mode = config["mode"]
    parent_pid = int(config.get("parent_pid") or os.getppid())
    engine_config = config.get("engine_config") or {}
    engine = _build_engine(mode, engine_config)
    fault_state: dict[str, Any] = {}
    should_stop = False

    def handle_request(request: dict[str, Any]) -> dict[str, Any]:
        nonlocal should_stop
        method = str(request.get("method") or "")
        payload = request.get("payload")
        if method == "ping":
            return findings.ok(driver=True, pid=os.getpid())
        if method == "shutdown":
            should_stop = True
            return findings.ok(driver=True, shutdown=True)
        if method == "inject":
            fault_kwargs = dict(payload or {})
            fault = fault_kwargs.pop("fault", None)
            if fault in ("crash", "hang"):
                fault_state[fault] = fault_kwargs.get("hang_s", True)
                return findings.ok(fault=fault, driver=True)
            return findings.not_ok(f"driver does not handle fault {fault!r}")
        if method == "generate":
            if fault_state.pop("crash", False):
                os._exit(1)
            hang_seconds = fault_state.pop("hang", None)
            if hang_seconds:
                time.sleep(float(hang_seconds))
            return findings.ok(output=_run_engine_generate(mode, engine, payload or {}))
        return findings.not_ok(f"unknown driver method {method!r}")

    try:
        with ipc.UnixJsonServer(config["endpoint"], handler=None) as server:
            while not should_stop and os.getppid() == parent_pid:
                try:
                    request, send_reply = server.next(timeout=1.0)
                except queue.Empty:
                    continue
                try:
                    reply = handle_request(request)
                except BaseException as error:
                    reply = findings.not_ok(f"{request.get('method')}: {type(error).__name__}: {error}")
                send_reply(reply)
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        if mode == "real":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    with open(arguments[0], encoding="utf-8") as handle:
        spec = json.load(handle)
    action = spec.get("action")
    if action == "driver":
        _serve_driver(spec)
        return 0
    if action != "fresh":
        _serve_worker(spec)
        return 0
    engine = _build_engine(spec["mode"], spec.get("engine_config") or {})
    try:
        tokens = _run_engine_generate(spec["mode"], engine, spec.get("workload") or {})
        rank_count = int(spec.get("rank_count", 1))
        print("FRESH " + json.dumps({"output": {f"rank{idx}": tokens for idx in range(rank_count)}}), flush=True)
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
    if spec["mode"] == "real":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
