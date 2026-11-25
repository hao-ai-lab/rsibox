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


def _is_ascii_digits(text: str) -> bool:
    return bool(text) and text.isascii() and text.isdigit()


def _normalize_device_map(raw: Any) -> str | dict[str, int | str] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON device_map: {error}") from error
            return _normalize_device_map(parsed)
        return text
    if isinstance(raw, dict):
        if not raw:
            raise ValueError("device_map dict must not be empty")
        cleaned: dict[str, int | str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"device_map keys must be non-empty strings, got {key!r}")
            if isinstance(value, bool) or not isinstance(value, (int, str)) or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"device_map[{key!r}] must be a non-negative int GPU index or non-empty device string, got {value!r}")
            if isinstance(value, int) and value < 0:
                raise ValueError(f"device_map[{key!r}] GPU index must be >= 0, got {value!r}")
            cleaned[key] = value.strip() if isinstance(value, str) else int(value)
        return cleaned
    raise ValueError(f"device_map must be a string, dict, or None; got {type(raw).__name__}")


def _normalize_max_memory(raw: Any) -> dict[int | str, int | str] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON max_memory: {error}") from error
        return _normalize_max_memory(parsed)
    if isinstance(raw, dict):
        if not raw:
            raise ValueError("max_memory dict must not be empty")
        cleaned: dict[int | str, int | str] = {}
        for key, value in raw.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise ValueError(f"max_memory key must be an int or string, got {key!r}")
            if isinstance(key, int):
                if key < 0:
                    raise ValueError(f"max_memory GPU key must be >= 0, got {key!r}")
                norm_key: int | str = int(key)
            else:
                stripped_key = key.strip()
                if not stripped_key or not stripped_key.isascii():
                    raise ValueError(f"max_memory string key must be non-empty ASCII, got {key!r}")
                if stripped_key.startswith("-") and _is_ascii_digits(stripped_key[1:]):
                    raise ValueError(f"max_memory GPU key must be >= 0, got {key!r}")
                norm_key = int(stripped_key) if _is_ascii_digits(stripped_key) else stripped_key
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError(f"max_memory[{key!r}] must be a positive int or non-empty string, got {value!r}")
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"max_memory[{key!r}] must be > 0, got {value!r}")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"max_memory[{key!r}] must not be empty")
            cleaned[norm_key] = value.strip() if isinstance(value, str) else int(value)
        return cleaned
    raise ValueError(f"max_memory must be a JSON string, dict, or None; got {type(raw).__name__}")


def _format_device_entry(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"cuda:{value}"
    text = str(value).strip()
    if _is_ascii_digits(text):
        return f"cuda:{int(text)}"
    return text


def _stringify_device_map(device_map: Any) -> dict[str, str] | None:
    if not isinstance(device_map, dict) or not device_map:
        return None
    return {str(k): _format_device_entry(v) for k, v in device_map.items()}


def _resolve_input_device(model: Any, fallback_device: str, torch_module: Any) -> Any:
    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb):
        try:
            emb = get_emb()
            if emb is not None:
                for tensor in (*emb.parameters(), *emb.buffers()):
                    if tensor.device.type != "meta":
                        return tensor.device
        except Exception:
            pass
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for value in hf_map.values():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return torch_module.device(f"cuda:{value}")
            if isinstance(value, str):
                text = value.strip()
                if text and text not in ("disk", "meta"):
                    return torch_module.device(f"cuda:{int(text)}" if _is_ascii_digits(text) else text)
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    return torch_module.device(fallback_device)


def _model_devices(model: Any, fallback_device: str, torch_module: Any) -> list[Any]:
    seen: list[Any] = []
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != "meta" and tensor.device not in seen:
            seen.append(tensor.device)
    if not seen:
        seen.append(torch_module.device(fallback_device))
    return seen


def _sync_cuda_devices(devices: list[Any], torch_module: Any) -> None:
    if not (torch_module.cuda.is_available() and torch_module.cuda.is_initialized()):
        return
    cuda_devs: list[Any] = []
    for dev in devices:
        if dev.type == "cuda" and dev not in cuda_devs:
            cuda_devs.append(dev)
    if cuda_devs:
        for dev in cuda_devs:
            torch_module.cuda.synchronize(dev)
    else:
        torch_module.cuda.synchronize()


def _is_sharded_model(model: Any, runtime: dict[str, Any]) -> bool:
    if runtime.get("hf_device_map") or getattr(model, "hf_device_map", None):
        return True
    devices = {
        tensor.device
        for tensor in (*model.parameters(), *model.buffers())
        if tensor.device.type != "meta"
    }
    return len(devices) > 1


def _move_and_sync(model: Any, target_device: Any, torch_module: Any) -> None:
    is_cuda = target_device.type == "cuda"
    model.to(target_device, non_blocking=is_cuda)
    if is_cuda and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(target_device)


def _pin_cpu_tensors(unique_tensors: list[Any]) -> None:
    counts: dict[int, int] = {}
    for tensor in unique_tensors:
        storage_ptr = tensor.untyped_storage().data_ptr()
        counts[storage_ptr] = counts.get(storage_ptr, 0) + 1
    for tensor in unique_tensors:
        storage = tensor.untyped_storage()
        if (
            counts[storage.data_ptr()] == 1
            and tensor.device.type == "cpu"
            and not tensor.is_pinned()
            and tensor.is_contiguous()
            and tensor.storage_offset() == 0
            and tensor.nbytes == storage.nbytes()
        ):
            tensor.data = tensor.pin_memory()


def _move_tensors_preserving_storage(
    unique_tensors: list[Any],
    target_for_tensor: dict[int, Any],
    torch_module: Any,
) -> list[Any]:
    """Move each tensor's data to its target device while preserving shared storages and views."""
    moved_storages: dict[tuple[int, str, str], Any] = {}
    moved_devices: list[Any] = []
    for tensor in unique_tensors:
        target_dev = target_for_tensor.get(id(tensor))
        if target_dev is None or tensor.device.type == "meta":
            continue
        if target_dev not in moved_devices:
            moved_devices.append(target_dev)
        if tensor.device == target_dev:
            continue
        src_storage = tensor.untyped_storage()
        cache_key = (src_storage.data_ptr(), str(tensor.device), str(target_dev))
        dst_storage = moved_storages.get(cache_key)
        if dst_storage is None:
            non_blocking = target_dev.type == "cuda"
            byte_view = torch_module.empty(0, dtype=torch_module.uint8, device=tensor.device).set_(src_storage)
            dst_storage = byte_view.to(target_dev, non_blocking=non_blocking).untyped_storage()
            moved_storages[cache_key] = dst_storage
        rebound = torch_module.empty(0, dtype=tensor.dtype, device=target_dev).set_(
            dst_storage, tensor.storage_offset(), tensor.size(), tensor.stride()
        )
        tensor.data = rebound
    return moved_devices


def _restore_sharded_tensors(
    model: Any, saved_devices: dict[int, Any], torch_module: Any
) -> list[Any]:
    unique_tensors = list(dict.fromkeys([*model.parameters(), *model.buffers()]))
    restored_devices = _move_tensors_preserving_storage(unique_tensors, saved_devices, torch_module)
    _sync_cuda_devices(restored_devices, torch_module)
    return restored_devices


def load_backend(config: dict[str, Any]) -> dict[str, Any]:
    mode, seed = str(config.get("mode", "real")), int(config.get("seed", 0))
    device_map = _normalize_device_map(config.get("device_map"))
    max_memory = _normalize_max_memory(config.get("max_memory"))
    if mode == "toy":
        from adapter.fixture.toy_hf.modeling import ToyCausalLM
        model = ToyCausalLM(vocab_size=int(config.get("vocab_size", 64)), seed=seed, device_map=device_map)
        hf_map = dict(model.hf_device_map) if model.hf_device_map else None
        devices = sorted(set(hf_map.values())) if hf_map else [model.device]
        return {
            "mode": "toy", "model": model, "tokenizer": None, "device": model.device,
            "devices": devices, "hf_device_map": hf_map,
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
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, dtype=resolved_dtype, **load_kwargs)
    except TypeError:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=resolved_dtype, **load_kwargs)
    if device_map is None:
        model = model.to(device).eval()
        hf_map = None
        input_device = device
        devices = [device]
    else:
        model = model.eval()
        hf_map = _stringify_device_map(getattr(model, "hf_device_map", None))
        input_device = str(_resolve_input_device(model, device, torch))
        devices = sorted({str(d) for d in _model_devices(model, input_device, torch)})
    return {
        "mode": "real", "model": model, "tokenizer": tokenizer,
        "device": input_device, "devices": devices, "hf_device_map": hf_map,
        "seed": seed, "model_module": type(model).__module__, "quiesced": False,
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
    tokenizer = runtime["tokenizer"]
    input_device = _resolve_input_device(model, runtime["device"], torch_module)
    temperature = float(workload.get("temperature", 0.0))
    do_sample = bool(workload.get("do_sample", temperature > 0.0))
    encoded = tokenizer(list(workload["prompts"]), return_tensors="pt", padding=True).to(input_device)
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
    if runtime.get("quiesced"):
        return findings.not_ok("quiesce refused: worker is already quiesced; resume required")
    model, released = runtime["model"], ["kv_cache"]
    if runtime["mode"] == "toy":
        (model.offload_to_cpu if offload_to_cpu else model.clear_cache)()
        runtime["quiesced"] = True
        devices = ["cpu"] if offload_to_cpu else list(runtime.get("devices") or [model.device])
        return {
            "ok": True, "released": released + (["weights"] if offload_to_cpu else []),
            "device": "cpu" if offload_to_cpu else model.device,
            "devices": devices, "hf_device_map": runtime.get("hf_device_map"),
        }
    import torch
    sharded = _is_sharded_model(model, runtime)
    initial_devices = _model_devices(model, runtime["device"], torch)
    target_device = torch.device(runtime["device"])
    use_cuda = any(d.type == "cuda" for d in initial_devices) or target_device.type == "cuda" or torch.cuda.is_initialized()
    offloaded = False
    saved_devices: dict[int, Any] | None = None
    try:
        if offload_to_cpu:
            if sharded:
                unique_tensors = list(dict.fromkeys([*model.parameters(), *model.buffers()]))
                saved_devices = {id(t): t.device for t in unique_tensors if t.device.type != "meta"}
                offloaded = True
                cpu_targets = {tid: torch.device("cpu") for tid in saved_devices}
                _move_tensors_preserving_storage(unique_tensors, cpu_targets, torch)
                if pin_memory and use_cuda and torch.cuda.is_available():
                    _pin_cpu_tensors(unique_tensors)
                runtime["saved_devices"] = saved_devices
            else:
                model.to("cpu")
                offloaded = True
                if pin_memory and use_cuda and torch.cuda.is_available():
                    _pin_cpu_tensors(list(dict.fromkeys([*model.parameters(), *model.buffers()])))
            released.append("weights")
        gc.collect()
        if use_cuda and torch.cuda.is_available():
            _sync_cuda_devices(initial_devices, torch)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            released.append("cuda_cache")
    except Exception as error:
        message = f"quiesce failed: {type(error).__name__}: {error}"
        if offloaded:
            try:
                if sharded and saved_devices is not None:
                    _restore_sharded_tensors(model, saved_devices, torch)
                    runtime.pop("saved_devices", None)
                else:
                    _move_and_sync(model, target_device, torch)
            except Exception as rollback_error:
                message = f"{message} (rollback failed: {rollback_error})"
        runtime["quiesced"] = False
        return findings.not_ok(message)
    runtime["quiesced"] = True
    devices = ["cpu"] if offload_to_cpu else sorted({str(d) for d in initial_devices})
    return {
        "ok": True, "released": released,
        "device": "cpu" if offload_to_cpu else runtime["device"],
        "devices": devices, "hf_device_map": runtime.get("hf_device_map"),
    }


def _resume_runtime(runtime: dict[str, Any], released: list[str]) -> dict[str, Any]:
    model, target_name, resumed = runtime["model"], runtime["device"], ["kv_cache"]
    if runtime["mode"] == "toy":
        if "weights" in released:
            model.restore_to_device(target_name)
            resumed.append("weights")
        runtime["quiesced"] = False
        hf_map = dict(model.hf_device_map) if getattr(model, "hf_device_map", None) else None
        devices = sorted(set(hf_map.values())) if hf_map else [model.device]
        return {
            "ok": True, "resumed": resumed, "device": model.device,
            "devices": devices, "hf_device_map": hf_map,
        }
    import torch
    saved_devices: dict[int, Any] | None = runtime.get("saved_devices")
    if saved_devices is not None or _is_sharded_model(model, runtime):
        restore_weights = "weights" in released or not released
        if saved_devices is not None and restore_weights:
            restored_devices = _restore_sharded_tensors(model, saved_devices, torch)
            runtime.pop("saved_devices", None)
            resumed.append("weights")
        elif saved_devices is None and "weights" in released:
            return findings.not_ok("resume refused: sharded model has no saved device placement for weights")
        else:
            restored_devices = _model_devices(model, target_name, torch)
            _sync_cuda_devices(restored_devices, torch)
        if any(d.type == "cuda" for d in restored_devices) and "cuda_cache" in released:
            resumed.append("cuda_cache")
        devices = sorted({str(d) for d in restored_devices})
        runtime["devices"] = devices
        runtime["quiesced"] = False
        return {
            "ok": True, "resumed": resumed, "device": target_name,
            "devices": devices, "hf_device_map": runtime.get("hf_device_map"),
        }
    target_device = torch.device(target_name)
    if "weights" in released or any(tensor.device != target_device for tensor in (*model.parameters(), *model.buffers())):
        _move_and_sync(model, target_device, torch)
        resumed.append("weights")
    elif target_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(target_device)
    if target_device.type == "cuda" and "cuda_cache" in released:
        resumed.append("cuda_cache")
    runtime["quiesced"] = False
    return {
        "ok": True, "resumed": resumed, "device": target_name,
        "devices": [target_name], "hf_device_map": runtime.get("hf_device_map"),
    }


def _dispatch_request(
    runtime: dict[str, Any], participant: Any, planned_modules: dict[str, set[str]],
    fault_state: dict[str, Any], method: str, payload: Any,
) -> tuple[dict[str, Any], bool]:
    if method == "shutdown":
        return {"ok": True}, True
    if method == "ping":
        return {
            "ok": True, "pid": os.getpid(), "mode": runtime["mode"],
            "device": runtime["device"], "devices": list(runtime.get("devices") or [runtime["device"]]),
            "hf_device_map": runtime.get("hf_device_map"),
            "model_module": runtime["model_module"],
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
