"""Worker subprocess & GMS-resident inference engine for NVIDIA Dynamo RSIBox live labs."""
from __future__ import annotations
import contextlib, hashlib, importlib, json, linecache, os, queue, subprocess, sys, tempfile, time
from typing import Any
from kernel import coordinator, findings
from rsibox import harness, ipc, proc
from rsibox.lab import ENDPOINT_METHODS, RoleEndpoint

PARTICIPANT_METHODS = frozenset(ENDPOINT_METHODS)
def kv_score_prefix(prompt: str, kv_cache: dict[str, Any]) -> int:
    """Return cached prefix length for `prompt` in this worker's KV cache (hot-swappable)."""
    return max((len(os.path.commonprefix((prompt, c))) for c in kv_cache), default=0)
def format_worker_tokens(token_ids: list[int], token_offset: int = 0) -> list[int]:
    """Normalize generated token IDs with optional offset (hot-swappable)."""
    return [int(t) + int(token_offset) for t in token_ids]
def worker_infer_step(prompt: str, max_new_tokens: int, token_offset: int, runtime: dict[str, Any]) -> dict[str, Any]:
    """Execute one deterministic inference step on CPU toy or GMS-backed GPU worker."""
    kv_cache, hit_chars = runtime["kv_cache"], kv_score_prefix(prompt, runtime["kv_cache"])
    if runtime["mode"] == "toy":
        vocab, sb = int(runtime.get("vocab_size", 128)), hashlib.sha256(f"{prompt}:{max_new_tokens}".encode()).digest()
        token_ids = format_worker_tokens([((sb[i % len(sb)] + i * 7) % vocab) for i in range(int(max_new_tokens))], token_offset)
        kv_cache[prompt], runtime["step_counter"] = token_ids, int(runtime.get("step_counter", 0)) + 1
        return {"token_ids": token_ids, "text": f"toy[{','.join(str(x) for x in token_ids[:4])}]", "kv_hit": hit_chars, "gms_mode": runtime["gms_mode"]}
    torch, tok, model = importlib.import_module("torch"), runtime["tokenizer"], runtime["model"]
    inputs = tok(prompt, return_tensors="pt").to(f"cuda:{runtime['device_idx']}")
    with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    token_ids = format_worker_tokens([int(x) for x in out[0][inputs["input_ids"].shape[1]:].tolist()], token_offset)
    kv_cache[prompt], runtime["step_counter"] = token_ids, int(runtime.get("step_counter", 0)) + 1
    return {"token_ids": token_ids, "text": tok.decode(out[0], skip_special_tokens=True), "kv_hit": hit_chars, "gms_mode": runtime["gms_mode"], "gms_committed_bytes": runtime["gms_committed_bytes"]}
def _sync_linecache(ep: RoleEndpoint, module_names: Any) -> None:
    for mod in module_names:
        with contextlib.suppress(Exception):
            if fpath := (getattr(sys.modules.get(mod), "__file__", None) or harness.module_file(mod)):
                src = ep.lp._live_source(mod)
                linecache.cache[fpath] = (len(src), None, src.splitlines(True), fpath) if isinstance(src, str) else linecache.cache.pop(fpath, None)  # type: ignore[assignment]
def _tail_text(path: str, limit: int = 500) -> str:
    if not os.path.exists(path): return ""
    with open(path, encoding="utf-8", errors="replace") as fh: return fh.read().strip()[-limit:]
def _stop_proc(p: Any, sock: str, *, clean: bool = True) -> None:
    if clean:
        with contextlib.suppress(Exception):
            if p.poll() is None: ipc.unix_json_call(sock, {"method": "shutdown"}, timeout=2.0); p.wait(timeout=2.0)
    with contextlib.suppress(Exception):
        if p.poll() is None: proc.stop_session(p.pid, grace=3.0, reap=p.poll)
    with contextlib.suppress(Exception):
        if p.poll() is None: proc.stop_tree(p.pid, grace_s=2.0, include_pg=True)
    with contextlib.suppress(Exception):
        if p.poll() is None: p.kill()
    with contextlib.suppress(Exception): p.wait(timeout=2.0)
def _close_gms(runtime: dict[str, Any]) -> None:
    if runtime.get("gms_manager") is not None:
        with contextlib.suppress(Exception): runtime["gms_manager"].close(best_effort=True)
def load_backend(config: dict[str, Any]) -> dict[str, Any]:
    """Load toy CPU runtime or real GMS-backed PyTorch/Transformers worker."""
    mode, worker_name = str(config.get("mode", "real")), str(config.get("worker_name", "worker0"))
    if mode == "toy": return {"mode": "toy", "worker_name": worker_name, "vocab_size": int(config.get("vocab_size", 128)), "kv_cache": {}, "step_counter": 0, "gms_mode": "toy", "gms_committed_bytes": 0, "gms_disk_load": False, "device_idx": 0}
    from rsibox.warm.dynamo_gms import ENV_GMS_DEVICE_UUID_MAP, install_visible_devices_uuid_shim
    if gdir := str(config.get("gms_socket_dir") or "").strip(): os.environ["GMS_SOCKET_DIR"] = gdir
    if umap := config.get("gms_device_uuid_map"): os.environ[ENV_GMS_DEVICE_UUID_MAP] = umap if isinstance(umap, str) else json.dumps(umap)
    install_visible_devices_uuid_shim()
    torch, tf = importlib.import_module("torch"), importlib.import_module("transformers")
    ga, gm, gl, gu, gc = (importlib.import_module(f"gpu_memory_service.{s}") for s in ("client.torch.allocator", "client.torch.module", "common.locks", "common.utils", "integrations.common.utils"))
    if not (model_path := str(config.get("model_path") or "").strip()): raise RuntimeError("config['model_path'] is required for real DynamoLab")
    device_idx, lock_req = int(config.get("device_idx", 0)), (gl.RequestedLockType.RO if config.get("gms_read_only") else gl.RequestedLockType.RW_OR_RO)
    sock, timeout_ms = gu.get_socket_path(device_idx, "weights"), config.get("ro_connect_timeout_ms", 10_000)
    mgr = ga.get_or_create_gms_client_memory_manager(sock, device_idx, lock_req, tag="weights", timeout_ms=int(timeout_ms) if lock_req == gl.RequestedLockType.RO and timeout_ms is not None else None)
    tok = tf.AutoTokenizer.from_pretrained(model_path)
    if mgr.granted_lock_type == gl.GrantedLockType.RW:
        model = tf.AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        with ga.gms_use_mem_pool("weights", device_idx): model = model.to(f"cuda:{device_idx}")
        gms_mode, committed_bytes, disk_load = "rw_committed", int(gc.finalize_gms_write(mgr, model).committed_bytes), True
    else:
        with torch.device("meta"): model = tf.AutoModelForCausalLM.from_config(tf.AutoConfig.from_pretrained(model_path), dtype=torch.bfloat16)
        gm.materialize_module_from_gms(mgr, model, device_index=device_idx)
        gms_mode, committed_bytes, disk_load = "ro_attached", int(mgr.total_bytes), False
    return {"mode": "real", "worker_name": worker_name, "model": model, "tokenizer": tok, "gms_manager": mgr, "gms_socket": sock, "gms_mode": gms_mode, "gms_committed_bytes": committed_bytes, "gms_disk_load": disk_load, "device_idx": device_idx, "kv_cache": {}, "step_counter": 0}
class _DynamoRoleEndpoint(RoleEndpoint):
    """Preserve per-transaction abstain state across `post_commit_reset='state'`."""
    def _dispatch(self, method: str, payload: Any) -> dict[str, Any]:
        if method != "reset": return super()._dispatch(method, payload)
        saved = dict(self._abstain)
        try: return super()._dispatch(method, payload)
        finally: self._abstain.update(saved)
def _dispatch(runtime: dict[str, Any], ep: RoleEndpoint, planned: dict[str, set[str]], fault_state: dict[str, Any], method: str, payload: Any) -> tuple[dict[str, Any], bool]:
    if method == "shutdown": return {"ok": True}, True
    if method == "ping": return {"ok": True, "pid": os.getpid(), **{k: runtime[k] for k in ("mode", "worker_name", "device_idx", "gms_mode", "gms_committed_bytes", "gms_disk_load")}}, False
    if method == "infer":
        if fault_state.pop("crash", False): os._exit(139)
        if hang_s := fault_state.pop("hang_s", 0.0): time.sleep(float(hang_s))
        req = dict(payload or {})
        return {"ok": True, "output": worker_infer_step(str(req.get("prompt", "")), int(req.get("max_new_tokens", 8)), int(req.get("token_offset", 0)), runtime)}, False
    if method == "inject":
        info, fault = dict(payload or {}), str((payload or {}).get("fault", "")); info.pop("fault", None)
        if fault in ("crash", "hang", "lost"):
            fault_state["crash" if fault == "crash" else "hang_s"] = True if fault == "crash" else float(info.get("hang_s", 5.0))
            return {"ok": True, "fault": fault}, fault == "lost"
        return harness.inject_participant_fault(ep.lp, fault, **info), False
    if method not in PARTICIPANT_METHODS: return findings.not_ok(f"unknown worker method {method!r}"), False
    if method == "prepare":
        specs = (payload or {}).get("modules", {})
        for mod, spec in specs.items():
            if (origin := getattr(sys.modules.get(mod), "__file__", None) or harness.module_file(mod)) and isinstance(src := (spec.get("source") if isinstance(spec, dict) else None), str):
                linecache.cache[origin] = (len(src), None, src.splitlines(True), origin)
        reply = ep.dispatch(method, payload); planned[str((payload or {}).get("id") or reply.get("txn"))] = set(specs)
        if not reply.get("ok"): _sync_linecache(ep, specs)
        return reply, False
    reply = ep.dispatch(method, payload)
    if method in ("commit", "abort", "undo"): _sync_linecache(ep, (planned.pop(str(coordinator.txn_ref(payload)[0]), set()) if method in ("abort", "undo") else set()) | {m for ms in planned.values() for m in ms})
    return reply, False
def serve_worker(sock_path: str, config: dict[str, Any]) -> int:
    """Serve live-swap and inference RPCs for one Dynamo worker over `AF_UNIX`."""
    parent_pid = int(config.get("parent_pid") or os.getppid())
    for mod in config.get("modules") or (): importlib.import_module(mod)
    runtime = load_backend(config)
    ep = _DynamoRoleEndpoint("worker", scope={"runtime": runtime}, rank=int(config.get("device_idx", 0)), resets={"state": lambda: (runtime["kv_cache"].clear(), runtime.__setitem__("step_counter", 0), ["kv_cache", "step_counter"])[2]})
    planned, fault_state, server = {}, {}, ipc.UnixJsonServer(sock_path, handler=None)
    try:
        while True:
            try: req, reply_fn = server.next(timeout=1.0)
            except queue.Empty:
                if parent_pid and os.getppid() != parent_pid: return 0
                continue
            try: reply, should_exit = _dispatch(runtime, ep, planned, fault_state, str(req.get("method") or ""), req.get("payload"))
            except Exception as exc: reply, should_exit = findings.not_ok(f"{type(exc).__name__}: {exc}"), False
            reply_fn(reply)
            if should_exit: return 0
    finally: _close_gms(runtime); server.close()

def run_fresh(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one fresh isolated `PackageOverlay` run of the Dynamo planner + worker pipeline."""
    from adapter import planner_ops
    cfg, wl = dict(spec.get("config") or {}), dict(spec.get("workload") or {})
    if cfg.get("mode") == "real": cfg["gms_read_only"] = True
    planner_cfg = planner_ops.PlannerConfig(policy=str(cfg.get("policy", "kv_aware")), max_new_tokens=int(cfg.get("max_new_tokens", 8)))
    workers, prompts = [f"worker{i}" for i in range(max(1, int(cfg.get("num_workers", 1))))], [str(p) for p in (wl.get("prompts") or ("The capital of France is",))]
    max_tokens, policy = int(wl.get("max_new_tokens", planner_cfg.max_new_tokens)), str(wl.get("policy", planner_cfg.policy))
    def _route(fn):
        return [planner_ops.format_route_response(m["worker"], p, fn(m["worker"], d), m) for idx, p in enumerate(prompts) for m in (planner_ops.select_worker(p, workers, None, policy=policy, step_idx=idx),) for d in (planner_ops.shape_decode_params(p, max_tokens, route_meta=m),)]
    if len(workers) == 1 or cfg.get("mode") != "real":
        runtimes = {w: load_backend({**cfg, "worker_name": w, "device_idx": i}) for i, w in enumerate(workers)}
        try: return {"ok": True, "output": {"policy": policy, "routing_scale": planner_cfg.routing_scale, "responses": _route(lambda w, d: worker_infer_step(d["prompt"], d["max_new_tokens"], d["token_offset"], runtimes[w]))}}
        finally:
            for rt in runtimes.values(): _close_gms(rt)
    procs, socks, specs = [], {}, []
    try:
        for i, w in enumerate(workers):
            sock = ipc.short_sock_path(f"rsidynf{i}")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as sf:
                json.dump({"sock_path": sock, "config": {**cfg, "worker_name": w, "device_idx": i, "parent_pid": os.getpid()}}, sf); specs.append(sf.name)
            procs.append((w, sock, subprocess.Popen([sys.executable, "-m", "adapter.worker", specs[-1]], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)))
        deadline = time.monotonic() + float(cfg.get("startup_timeout", 120.0))
        for w, sock, p in procs:
            while time.monotonic() < deadline and not os.path.exists(sock):
                if p.poll() is not None: raise RuntimeError(f"fresh {w} exited ({p.returncode}): {p.stderr.read().decode(errors='replace')[-400:] if p.stderr else ''}")
                time.sleep(0.05)
            if not ipc.unix_json_call(sock, {"method": "ping"}, timeout=30.0).get("ok"): raise RuntimeError(f"fresh {w} ping failed")
            socks[w] = sock
        return {"ok": True, "output": {"policy": policy, "routing_scale": planner_cfg.routing_scale, "responses": _route(lambda w, d: ipc.unix_json_call(socks[w], {"method": "infer", "payload": d}, timeout=60.0)["output"])}}
    finally:
        for _, sock, p in procs:
            _stop_proc(p, sock)
            if p.stderr is not None: p.stderr.close()
            with contextlib.suppress(OSError): os.unlink(sock)
        for sp in specs:
            with contextlib.suppress(OSError): os.unlink(sp)

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as _sf: spec = json.load(_sf)
    if "sock_path" in spec:
        import adapter.worker as _self_mod
        raise SystemExit(_self_mod.serve_worker(spec["sock_path"], spec["config"]))
    print("FRESH " + json.dumps(run_fresh(spec), default=str), flush=True)
