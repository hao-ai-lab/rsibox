"""Read a `vllm serve` argv with vLLM's own parser, so vLLM stays the one source of truth for its
flags: the launcher passes the argv through verbatim and only reads back what it needs.

  parse(argv) -> {"ok": True, "args": {field: value}, "explicit": [fields set in argv],
                  "seconds": {"import", "build", "parse"}}
              or {"ok": False, "error": "<vLLM's argparse message>"}
  override(base_argv, patch_argv) -> merged `vllm serve` argv with patch flags overriding base
  classify_reconfig(old_v, new_v) -> 4-tier classification (see kb/reconfiguration_matrix.md):
      - in_process_instant (< 100 ms scheduler/runtime config mutation, no process restart)
      - in_process_graph   (0.5-2 s live dispatch toggle or CUDA graph recapture, no process restart)
      - warm_weights_ipc   (7-20 s worker restart with WeightHolder CUDA IPC, zero disk read)
      - live_reshard_or_cold (WeightCacheKey sharding or weight-format change)

Contract:
- The parser is `make_arg_parser(FlexibleArgumentParser())`, the one `vllm serve` builds, so
  aliases (`-tp`), `--x=v`, underscores, `--config` YAML files and dotted JSON flags read the
  same as in the server.
- model follows `vllm serve`'s rule (vllm/entrypoints/cli/serve.py): a positional model lands in
  `model_tag` and replaces `model`; reading `.model` alone gives vLLM's default model.
- "explicit" means given in the argv (or in its --config file), even when the value equals
  vLLM's default: `--load-format auto` is explicit. It is measured by parsing a second time
  into a namespace that holds a sentinel for each field; argparse leaves a field it did not see
  at the sentinel. The model is explicit when model_tag or --model is given.
- Importing vLLM's parser pulls torch, transformers and triton but does not initialize CUDA
  (test_vllm_args.py checks this in a fresh process). Cost on our host: import 4.7-4.9 s, build
  0.16 s, parse 1 ms. The zygote has these imported already (zygote.py, request "parse").
"""
import argparse, contextlib, dataclasses, enum, io, sys, time

# Tier 2a: flags that change how weights are sharded across GPUs in WeightCacheKey.
RESHARD_FIELDS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "enable_expert_parallel",
    "expert_placement_strategy",
)

# Tier 2b: flags that change weight checkpoint, dtype, quantization, KV scale tensors, or
# kernel-specific post-processed weight layout in WeightCacheKey.
WEIGHT_FORMAT_FIELDS = (
    "model",
    "revision",
    "dtype",
    "quantization",
    "hf_overrides",
    "kv_cache_dtype",
    "kv_cache_dtype_skip_layers",
    "moe_backend",
    "linear_backend",
    "linear_backend_per_quant",
)

# Tier 1A (in_process_instant, < 100 ms): scheduler and admission flags mutated in-process at the
# step gap without restarting workers or recapturing CUDA graphs.
IN_PROCESS_INSTANT_FIELDS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "enable_prefix_caching",
    "scheduling_policy",
    "enable_chunked_prefill",
    "async_scheduling",
)

# Tier 1B (in_process_graph, 0.5-2 s): execution/compilation flags applied in-process by toggling
# eager vs CUDAGraphWrapper dispatch or recapturing CUDA graphs via RecaptureCoordinator.
IN_PROCESS_GRAPH_FIELDS = (
    "enforce_eager",
    "cudagraph_capture_sizes",
    "max_cudagraph_capture_size",
    "compilation_config",
    "optimization_level",
    "speculative_config",
)

# Tier 2 (warm_weights_ipc, 7-20 s): KV pool sizing flags that require restarting the worker
# process while WeightHolder keeps post-processed weights resident in GPU memory over CUDA IPC.
WARM_WEIGHTS_IPC_FIELDS = (
    "gpu_memory_utilization",
    "block_size",
    "swap_space",
)

# All WeightCacheKey-preserving runtime flags (in_process_instant + in_process_graph + warm_weights_ipc).
WARM_IPC_FIELDS = (
    *IN_PROCESS_INSTANT_FIELDS,
    *IN_PROCESS_GRAPH_FIELDS,
    *WARM_WEIGHTS_IPC_FIELDS,
)

# Server/holder plumbing fields read by the launcher.
LAUNCHER_FIELDS = (
    "port",
    "host",
    "load_format",
    "worker_cls",
    "model_loader_extra_config",
)

# What the launcher reads back: dest names of vLLM's parser.
FIELDS = (
    "model", "port", "host", "tensor_parallel_size", "load_format", "enforce_eager",
    "worker_cls", "model_loader_extra_config",
    *(field_name for field_name in WARM_IPC_FIELDS + RESHARD_FIELDS + WEIGHT_FORMAT_FIELDS
      if field_name not in ("model", "tensor_parallel_size", "enforce_eager")),
)
_UNSET = object()
_KERNEL_CFG_FIELDS = ("moe_backend", "linear_backend", "linear_backend_per_quant")

# Short/long aliases and boolean flags for clean argv overriding without parser state mutation.
_FLAG_ALIASES = {
    "-tp": "--tensor-parallel-size",
    "-pp": "--pipeline-parallel-size",
    "-dp": "--data-parallel-size",
    "-q": "--quantization",
    "-cc": "--compilation-config",
    "-O": "--optimization-level",
}
_BOOL_DEFAULT_FALSE = {
    "--enforce-eager",
    "--enable-expert-parallel",
    "--async-scheduling",
    "--disable-log-stats",
}
_BOOL_FLAGS = _BOOL_DEFAULT_FALSE | {
    "--no-enforce-eager",
    "--no-enable-expert-parallel",
    "--no-async-scheduling",
    "--enable-prefix-caching",
    "--no-enable-prefix-caching",
    "--enable-chunked-prefill",
    "--no-enable-chunked-prefill",
    "--disable-chunked-prefill",
}


def _normalize_value(value):
    """Normalize parsed values into JSON-safe primitives so direct and zygote parses match."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return _normalize_value(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _normalize_value(item_val) for key, item_val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    return str(value)


def _field_value(namespace, field_name):
    value = getattr(namespace, field_name, None)
    if value is None and field_name in _KERNEL_CFG_FIELDS:
        kernel_cfg = getattr(namespace, "kernel_config", None)
        if kernel_cfg is not None:
            value = (kernel_cfg.get(field_name)
                     if isinstance(kernel_cfg, dict)
                     else getattr(kernel_cfg, field_name, None))
    return _normalize_value(value)


def build():
    """vLLM's `vllm serve` parser and the seconds its import and build took. What the import
    and build print goes to stderr: vLLM's logger writes to stdout (with VLLM_LOGGING_LEVEL=DEBUG,
    vLLM 0.15.1 printed 16 lines at import), and the launcher's stdout is one JSON line."""
    t = time.time()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from vllm.entrypoints.common.cli_args import make_arg_parser
        except ImportError:                   # vLLM releases before the common/ split
            from vllm.entrypoints.openai.cli_args import make_arg_parser
        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError:
            from vllm.utils import FlexibleArgumentParser
        t1 = time.time()
        parser = make_arg_parser(FlexibleArgumentParser(prog="vllm serve"))
    return parser, {"import": round(t1 - t, 3), "build": round(time.time() - t1, 3)}


def parse(argv, parser=None):
    seconds = {}
    if parser is None:
        parser, seconds = build()
    t = time.time()
    out = io.StringIO()
    try:
        with contextlib.redirect_stderr(out), contextlib.redirect_stdout(out):
            ns = parser.parse_args(list(argv))
            seen = parser.parse_args(list(argv), namespace=argparse.Namespace(
                kernel_config=_UNSET, **{field_name: _UNSET for field_name in FIELDS}))
    except SystemExit:                        # argparse error (or --help): its last line
        lines = out.getvalue().strip().splitlines()
        return {"ok": False, "error": lines[-1] if lines else "vllm serve: exited parsing"}
    except Exception as e:                    # FlexibleArgumentParser raises ValueError too
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    args = {field_name: _field_value(ns, field_name) for field_name in FIELDS}
    explicit = [field_name for field_name in FIELDS if getattr(seen, field_name, _UNSET) is not _UNSET]
    seen_kernel_cfg = getattr(seen, "kernel_config", _UNSET)
    if seen_kernel_cfg is not _UNSET and seen_kernel_cfg is not None:
        default_kernel_cfg = getattr(parser.parse_args([]), "kernel_config", None)
        for field_name in _KERNEL_CFG_FIELDS:
            if field_name not in explicit and _field_value(ns, field_name) != (
                _normalize_value(getattr(default_kernel_cfg, field_name, None))
                if default_kernel_cfg is not None else None
            ):
                explicit.append(field_name)
    if getattr(ns, "model_tag", None) is not None:         # serve.py: model_tag wins
        args["model"] = ns.model_tag
        explicit = sorted(set(explicit) | {"model"}, key=FIELDS.index)
    seconds["parse"] = round(time.time() - t, 4)
    return {"ok": True, "args": args, "explicit": explicit, "seconds": seconds}


def _canonical_flag(token):
    """Canonical flag key for an option token (unifies underscores, aliases, --no-*, --flag=v)."""
    option = token.split("=", 1)[0]
    if option.startswith("--"):
        option = "--" + option[2:].replace("_", "-")
    option = _FLAG_ALIASES.get(option, option)
    if option.startswith("--no-"):
        return "--" + option[5:]
    if option.startswith("--disable-"):
        return "--enable-" + option[10:]
    return option


def _tokenize_argv(argv):
    """Split a `vllm serve` argv into (positional_model, [(canonical_opt, raw_tokens), ...])."""
    model = None
    items = []
    index = 0
    argv = list(argv)
    while index < len(argv):
        token = argv[index]
        if not token.startswith("-"):
            if model is None:
                model = token
                index += 1
                continue
            items.append((None, [token]))
            index += 1
            continue
        option = token.split("=", 1)[0]
        normalized_opt = ("--" + option[2:].replace("_", "-")) if option.startswith("--") else option
        canonical_opt = _canonical_flag(token)
        if "=" in token or normalized_opt in _BOOL_FLAGS:
            chunk = [token]
            index += 1
        else:
            chunk = [token]
            index += 1
            while index < len(argv) and not argv[index].startswith("-"):
                chunk.append(argv[index])
                index += 1
                if canonical_opt != "--cudagraph-capture-sizes":
                    break
        if canonical_opt == "--model" and len(chunk) >= 2:
            model = chunk[1] if "=" not in chunk[0] else chunk[0].split("=", 1)[1]
        elif canonical_opt == "--model" and "=" in chunk[0]:
            model = chunk[0].split("=", 1)[1]
        items.append((canonical_opt, chunk))
    return model, items


def override(base_argv, patch_argv):
    """Apply `patch_argv` overrides onto `base_argv`, returning a clean `vllm serve` argv.
    - Preserves the model from `base_argv` unless `patch_argv` provides a positional model or
      `--model`.
    - Drops any option in `base_argv` whose canonical flag is overridden in `patch_argv`.
    - When `patch_argv` negates a default-false boolean flag (`--no-enforce-eager`,
      `--no-enable-expert-parallel`, `--no-async-scheduling`), drops the flag from `base_argv`
      and omits the `--no-*` token so both real vLLM and stub parsers resolve `False`."""
    base_model, base_items = _tokenize_argv(base_argv)
    patch_model, patch_items = _tokenize_argv(patch_argv)
    overridden = {canonical_opt for canonical_opt, _ in patch_items if canonical_opt is not None}
    if patch_model is not None:
        overridden.add("--model")
    merged_argv = []
    model = patch_model if patch_model is not None else base_model
    has_model_flag = any(canonical_opt == "--model" for canonical_opt, _ in patch_items) or (
        "--model" not in overridden
        and any(canonical_opt == "--model" for canonical_opt, _ in base_items)
    )
    if model is not None and not has_model_flag:
        merged_argv.append(model)
    for canonical_opt, chunk in base_items:
        if canonical_opt not in overridden:
            merged_argv.extend(chunk)
    for canonical_opt, chunk in patch_items:
        raw_opt = chunk[0].split("=", 1)[0]
        normalized_opt = (
            ("--" + raw_opt[2:].replace("_", "-")) if raw_opt.startswith("--") else raw_opt
        )
        if normalized_opt.startswith("--no-") and canonical_opt in _BOOL_DEFAULT_FALSE:
            continue
        merged_argv.extend(chunk)
    return merged_argv


def _build_control_actions(changed, new_args, instant_fields, graph_fields):
    """Build the ordered list of in-process control actions for `in_process_instant` and
    `in_process_graph` reconfigurations (see `repos/vllm/kb/reconfiguration_matrix.md`)."""
    actions = []
    sched_fields = [
        field_name for field_name in instant_fields
        if field_name in ("max_num_seqs", "max_num_batched_tokens", "scheduling_policy",
                          "enable_chunked_prefill", "async_scheduling")
    ]
    if sched_fields:
        actions.append({
            "action": "mutate_scheduler_config",
            "target": "EngineCore.scheduler.scheduler_config",
            "fields": {field_name: new_args.get(field_name) for field_name in sched_fields},
            "target_latency_s": 0.05,
        })
    if "max_model_len" in instant_fields:
        actions.append({
            "action": "mutate_max_model_len",
            "target": "EngineCore.model_config.max_model_len + Scheduler.max_model_len",
            "fields": {"max_model_len": new_args.get("max_model_len")},
            "target_latency_s": 0.05,
        })
    if "enable_prefix_caching" in instant_fields:
        actions.append({
            "action": "toggle_prefix_caching",
            "target": "KVCacheManager.enable_caching",
            "fields": {"enable_prefix_caching": new_args.get("enable_prefix_caching")},
            "target_latency_s": 0.05,
        })
    if "enforce_eager" in graph_fields:
        actions.append({
            "action": "toggle_cudagraph_dispatch",
            "target": "GPUModelRunner.compilation_config.cudagraph_mode",
            "fields": {"enforce_eager": new_args.get("enforce_eager")},
            "target_latency_s": 0.5,
        })
    recapture_fields = [
        field_name for field_name in graph_fields if field_name != "enforce_eager"
    ]
    if recapture_fields:
        actions.append({
            "action": "recapture_cudagraphs",
            "target": "RecaptureCoordinator.recapture",
            "fields": {field_name: new_args.get(field_name) for field_name in recapture_fields},
            "target_latency_s": 1.5,
        })
    return actions


def classify_reconfig(old_parsed, new_parsed):
    """Classify the config delta between two parsed `vllm serve` argvs (`old_parsed`, `new_parsed`
    from `parse()`) into 4 operational tiers (documented in `repos/vllm/kb/reconfiguration_matrix.md`):
    - `in_process_instant` (< 100 ms): scheduler/admission/KV-policy mutation in-process (`in_process=True`).
    - `in_process_graph` (0.5-2 s): eager/CUDAGraph dispatch toggle or graph recapture in-process
      (`in_process=True`).
    - `warm_weights_ipc` (7-20 s): worker restart with resident `WeightHolder` CUDA IPC weights,
      zero disk reload (`in_process=False`, `weights_compatible=True`).
    - `live_reshard_or_cold`: `WeightCacheKey` sharding (`subtier='tier2a_reshard'`) or
      weight-format (`subtier='tier2b_weight_format'`) changed (`in_process=False`,
      `weights_compatible=False`).
    - `noop`: no engine or weight flag changed (`in_process=True`, `weights_compatible=True`)."""
    old_args = old_parsed.get("args") or {}
    new_args = new_parsed.get("args") or {}
    changed = {
        field_name: [old_args.get(field_name), new_args.get(field_name)]
        for field_name in FIELDS
        if field_name not in LAUNCHER_FIELDS
        and old_args.get(field_name) != new_args.get(field_name)
    }
    reshard_fields = [field_name for field_name in RESHARD_FIELDS if field_name in changed]
    weight_format_fields = [
        field_name for field_name in WEIGHT_FORMAT_FIELDS if field_name in changed
    ]
    in_process_instant_fields = [
        field_name for field_name in IN_PROCESS_INSTANT_FIELDS if field_name in changed
    ]
    in_process_graph_fields = [
        field_name for field_name in IN_PROCESS_GRAPH_FIELDS if field_name in changed
    ]
    warm_restart_fields = [
        field_name for field_name in changed
        if field_name not in RESHARD_FIELDS
        and field_name not in WEIGHT_FORMAT_FIELDS
        and field_name not in IN_PROCESS_INSTANT_FIELDS
        and field_name not in IN_PROCESS_GRAPH_FIELDS
    ]
    warm_ipc_fields = [
        field_name for field_name in changed
        if field_name not in RESHARD_FIELDS and field_name not in WEIGHT_FORMAT_FIELDS
    ]
    weight_key_changed = reshard_fields + weight_format_fields
    if not changed:
        tier = "noop"
        subtier = "noop"
        in_process = True
    elif weight_key_changed:
        tier = "live_reshard_or_cold"
        subtier = "tier2a_reshard" if reshard_fields else "tier2b_weight_format"
        in_process = False
    elif warm_restart_fields:
        tier = "warm_weights_ipc"
        subtier = "tier1_warm_ipc"
        in_process = False
    elif in_process_graph_fields:
        tier = "in_process_graph"
        subtier = "tier1_in_process_graph"
        in_process = True
    else:
        tier = "in_process_instant"
        subtier = "tier1_in_process_instant"
        in_process = True
    control_actions = (
        _build_control_actions(
            changed, new_args, in_process_instant_fields, in_process_graph_fields
        )
        if in_process and changed else []
    )
    return {
        "tier": tier,
        "subtier": subtier,
        "in_process": in_process,
        "weights_compatible": not weight_key_changed,
        "changed": changed,
        "in_process_instant_fields": in_process_instant_fields,
        "in_process_graph_fields": in_process_graph_fields,
        "warm_restart_fields": warm_restart_fields,
        "warm_ipc_fields": warm_ipc_fields,
        "reshard_fields": reshard_fields,
        "weight_format_fields": weight_format_fields,
        "weight_key_changed": weight_key_changed,
        "control_actions": control_actions,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(parse(sys.argv[1:])))
