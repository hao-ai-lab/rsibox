"""Persistent warm worker subprocess for HuggingFace Transformers RSIBox live labs."""
from __future__ import annotations

import gc
import hashlib
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
from rsibox.lab import ENDPOINT_METHODS

DTYPE_ALIASES = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16",
    "float32": "float32", "fp32": "float32",
}
PARTICIPANT_METHODS = frozenset(ENDPOINT_METHODS)
QUIESCE_BLOCKED = frozenset({"generate", "prepare", "validate", "commit"})


def _sync_linecache(participant: Any, module_names: Any) -> None:
    for module_name in module_names:
        try:
            loaded = sys.modules.get(module_name)
            file_path = getattr(loaded, "__file__", None) or harness.module_file(module_name)
            if not file_path:
                continue
            live_source = participant._live_source(module_name)
            if isinstance(live_source, str):
                linecache.cache[file_path] = (len(live_source), None, live_source.splitlines(True), file_path)
            else:
                linecache.cache.pop(file_path, None)
        except Exception:
            continue


def _seed_torch(seed: int) -> Any:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)
    return torch


def _move_and_sync(model: Any, target_device: Any, torch_module: Any) -> None:
    is_cuda = target_device.type == "cuda"
    model.to(target_device, non_blocking=is_cuda)
    if is_cuda and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(target_device)


def load_backend(config: dict[str, Any]) -> dict[str, Any]:
    mode, seed = str(config.get("mode", "real")), int(config.get("seed", 0))
    if mode == "toy":
        from adapter.fixture.toy_hf.modeling import ToyCausalLM
        model = ToyCausalLM(vocab_size=int(config.get("vocab_size", 64)), seed=seed)
        return {
            "mode": "toy", "model": model, "tokenizer": None, "device": "cpu",
            "seed": seed, "model_module": type(model).__module__, "quiesced": False,
        }
    import torch
    import transformers
    model_path = str(config.get("model_path") or "")
    if not model_path:
        raise RuntimeError("config['model_path'] is required for real TransformersLab")
    device = str(config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype_key = str(config.get("dtype") or "bfloat16").lower()
    if dtype_key != "auto" and dtype_key not in DTYPE_ALIASES:
        raise ValueError(f"unsupported dtype {dtype_key!r}; expected auto or {sorted(DTYPE_ALIASES)}")
    resolved_dtype = "auto" if dtype_key == "auto" else getattr(torch, DTYPE_ALIASES[dtype_key])
    trust_remote = bool(config.get("trust_remote_code", False))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote}
    if config.get("attn_implementation"):
        load_kwargs["attn_implementation"] = config["attn_implementation"]
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, dtype=resolved_dtype, **load_kwargs)
    except TypeError:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=resolved_dtype, **load_kwargs)
    return {
        "mode": "real", "model": model.to(device).eval(), "tokenizer": tokenizer,
        "device": device, "seed": seed, "model_module": type(model).__module__, "quiesced": False,
    }


def reset_runtime_state(runtime: dict[str, Any]) -> list[str]:
    seed = int(runtime.get("seed", 0))
    if runtime["mode"] == "toy":
        random.seed(seed)
        runtime["model"].clear_cache(seed=seed)
        return ["kv_cache", "rng"]
    torch_module = _seed_torch(seed)
    if torch_module.cuda.is_initialized():
        torch_module.cuda.empty_cache()
    gc.collect()
    return ["kv_cache", "cuda_cache", "rng"]


def generate_tokens(runtime: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    model = runtime["model"]
    if runtime["mode"] == "toy":
        tokens = model.generate(
            workload["input_ids"], max_new_tokens=int(workload.get("max_new_tokens", 6)),
            temperature=float(workload.get("temperature", 0.0)),
        )
        return {"tokens": tokens, "digest": hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()[:16]}
    torch_module = _seed_torch(int(workload.get("seed", runtime.get("seed", 0))))
    tokenizer, device = runtime["tokenizer"], runtime["device"]
    temperature = float(workload.get("temperature", 0.0))
    do_sample = bool(workload.get("do_sample", temperature > 0.0))
    encoded = tokenizer(list(workload["prompts"]), return_tensors="pt", padding=True).to(device)
    gen_kwargs = {
        "max_new_tokens": int(workload.get("max_new_tokens", 16)), "do_sample": do_sample,
        "use_cache": True, "pad_token_id": tokenizer.pad_token_id, **({"temperature": temperature} if do_sample else {}),
    }
    with torch_module.inference_mode():
        output_ids = model.generate(**encoded, **gen_kwargs)
    generated_slice = output_ids[:, int(encoded["input_ids"].shape[1]):].detach().cpu()
    token_lists = generated_slice.tolist()
    texts = tokenizer.batch_decode(generated_slice, skip_special_tokens=True)
    digest = hashlib.sha256(json.dumps(token_lists).encode("utf-8")).hexdigest()[:16]
    return {"tokens": token_lists, "texts": texts, "digest": digest}


def _quiesce_runtime(runtime: dict[str, Any], offload_to_cpu: bool, pin_memory: bool) -> dict[str, Any]:
    model, released = runtime["model"], ["kv_cache"]
    if runtime["mode"] == "toy":
        (model.offload_to_cpu if offload_to_cpu else model.clear_cache)()
        runtime["quiesced"] = True
        return {"ok": True, "released": released + (["weights"] if offload_to_cpu else []), "device": "cpu"}
    import torch
    target_device = torch.device(runtime["device"])
    use_cuda, offloaded = (target_device.type == "cuda" or torch.cuda.is_initialized()), False
    try:
        if offload_to_cpu:
            model.to("cpu")
            offloaded = True
            if pin_memory and use_cuda and torch.cuda.is_available():
                unique_tensors = list(dict.fromkeys([*model.parameters(), *model.buffers()]))
                counts: dict[int, int] = {}
                for tensor in unique_tensors:
                    storage_ptr = tensor.untyped_storage().data_ptr()
                    counts[storage_ptr] = counts.get(storage_ptr, 0) + 1
                for tensor in unique_tensors:
                    storage = tensor.untyped_storage()
                    if (counts[storage.data_ptr()] == 1 and tensor.device.type == "cpu" and not tensor.is_pinned()
                            and tensor.is_contiguous() and tensor.storage_offset() == 0 and tensor.nbytes == storage.nbytes()):
                        tensor.data = tensor.pin_memory()
            released.append("weights")
        gc.collect()
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(target_device if target_device.type == "cuda" else None)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            released.append("cuda_cache")
    except Exception as error:
        message = f"quiesce failed: {type(error).__name__}: {error}"
        if offloaded:
            try:
                _move_and_sync(model, target_device, torch)
            except Exception as rollback_error:
                message = f"{message} (rollback failed: {rollback_error})"
        runtime["quiesced"] = False
        return findings.not_ok(message)
    runtime["quiesced"] = True
    return {"ok": True, "released": released, "device": "cpu" if offload_to_cpu else runtime["device"]}


def _resume_runtime(runtime: dict[str, Any], released: list[str]) -> dict[str, Any]:
    model, target_name, resumed = runtime["model"], runtime["device"], ["kv_cache"]
    if runtime["mode"] == "toy":
        if "weights" in released:
            model.restore_to_device(target_name)
            resumed.append("weights")
        runtime["quiesced"] = False
        return {"ok": True, "resumed": resumed, "device": target_name}
    import torch
    target_device = torch.device(target_name)
    if "weights" in released or any(tensor.device != target_device for tensor in (*model.parameters(), *model.buffers())):
        _move_and_sync(model, target_device, torch)
        resumed.append("weights")
    elif target_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(target_device)
    if target_device.type == "cuda" and "cuda_cache" in released:
        resumed.append("cuda_cache")
    runtime["quiesced"] = False
    return {"ok": True, "resumed": resumed, "device": target_name}


def _dispatch_request(
    runtime: dict[str, Any], participant: Any, planned_modules: dict[str, set[str]],
    fault_state: dict[str, Any], method: str, payload: Any,
) -> tuple[dict[str, Any], bool]:
    if method == "shutdown":
        return {"ok": True}, True
    if method == "ping":
        return {
            "ok": True, "pid": os.getpid(), "mode": runtime["mode"],
            "device": runtime["device"], "model_module": runtime["model_module"],
        }, False
    if runtime.get("quiesced") and method in QUIESCE_BLOCKED:
        return findings.not_ok(f"{method} refused: worker is quiesced; resume required"), False
    if method == "generate":
        if fault_state.pop("crash", False):
            os._exit(139)
        if hang_seconds := fault_state.pop("hang_s", 0.0):
            time.sleep(float(hang_seconds))
        return {"ok": True, "output": generate_tokens(runtime, payload or {})}, False
    if method == "quiesce":
        info = payload or {}
        return _quiesce_runtime(runtime, bool(info.get("offload_to_cpu", True)), bool(info.get("pin_memory", True))), False
    if method == "resume":
        return _resume_runtime(runtime, list((payload or {}).get("released") or ())), False
    if method == "inject":
        info = dict(payload or {})
        fault_name = info.pop("fault", "")
        if fault_name == "crash":
            fault_state["crash"] = True
            return {"ok": True, "fault": "crash"}, False
        if fault_name == "hang":
            fault_state["hang_s"] = float(info.get("hang_s", 5.0))
            return {"ok": True, "fault": "hang"}, False
        if fault_name == "lost":
            return {"ok": True, "fault": "lost"}, True
        return harness.inject_participant_fault(participant, fault_name, **info), False
    if method not in PARTICIPANT_METHODS:
        return findings.not_ok(f"unknown worker method {method!r}"), False
    if method == "prepare":
        module_specs = (payload or {}).get("modules", {})
        for module_name, spec in module_specs.items():
            origin = getattr(sys.modules.get(module_name), "__file__", None) or harness.module_file(module_name)
            source_text = spec.get("source") if isinstance(spec, dict) else None
            if origin and isinstance(source_text, str):
                linecache.cache[origin] = (len(source_text), None, source_text.splitlines(True), origin)
        reply = participant.prepare(payload)
        planned_modules[str(reply.get("txn"))] = set(module_specs)
        if not reply.get("ok"):
            _sync_linecache(participant, module_specs)
        return reply, False
    reply = getattr(participant, method)(payload)
    if method in ("commit", "abort", "undo"):
        txn_id, _ = coordinator.txn_ref(payload)
        affected = planned_modules.pop(str(txn_id), set()) if method in ("abort", "undo") else set()
        _sync_linecache(participant, affected | {name for names in planned_modules.values() for name in names})
    return reply, False


def serve_worker(sock_path: str, config: dict[str, Any]) -> int:
    parent_pid, runtime = int(config.get("parent_pid") or os.getppid()), load_backend(config)
    for module_name in config.get("modules") or ():
        importlib.import_module(module_name)
    participant = coordinator.LocalParticipant(
        "worker0", scope={"model": runtime["model"], "tokenizer": runtime["tokenizer"]},
        resets={"state": lambda: reset_runtime_state(runtime)},
    )
    planned_modules, fault_state, server = {}, {}, ipc.UnixJsonServer(sock_path, handler=None)
    try:
        while True:
            try:
                request, reply_fn = server.next(timeout=1.0)
            except queue.Empty:
                if parent_pid and os.getppid() != parent_pid:
                    return 0
                continue
            method = str(request.get("method") or "")
            try:
                reply, should_exit = _dispatch_request(
                    runtime, participant, planned_modules, fault_state, method, request.get("payload")
                )
            except Exception as error:
                reply, should_exit = findings.not_ok(f"{method}: {type(error).__name__}: {error}"), False
            reply_fn(reply)
            if should_exit:
                return 0
    finally:
        server.close()


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as handle:
        spec = json.load(handle)
    raise SystemExit(serve_worker(spec["sock_path"], spec["config"]))
