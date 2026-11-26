"""DynamoLab: RSIBox live lab over NVIDIA Dynamo Frontend + Python Planner + GMS GPU Workers."""
from __future__ import annotations
import argparse, contextlib, json, os, subprocess, sys, tempfile, time
from typing import Any
from adapter import planner_ops
from adapter import worker as _worker_mod  # noqa: F401
from adapter.frontend_driver import FrontendDriver
from kernel import coordinator
from rsibox import harness, ipc
from rsibox.lab import BaseLab, RecaptureCoordinator

DEFAULT_MODULES = ("adapter.planner_ops", "adapter.worker")
class DynamoLab(BaseLab):
    """Live lab composing Frontend HTTP ingress, in-process Python Planner, and GMS GPU workers."""
    name, package, threaded_check = "dynamo", "adapter", True
    def __init__(self, config: dict[str, Any] | None = None, modules: tuple[str, ...] | list[str] | None = None, *, receipts: str | None = None, rpc_timeout: float = 30.0, check_timeout: float | None = None, startup_timeout: float = 120.0, weight_store: Any = None, recovery_driver: Any = None, auto_spawn: bool = True) -> None:
        g = os.environ.get
        env_cfg = {"mode": g("RSIBOX_DYNAMO_MODE", "real"), "model_path": g("RSIBOX_DYNAMO_MODEL", ""), "num_workers": int(g("RSIBOX_DYNAMO_NUM_WORKERS", "1")), "max_new_tokens": int(g("RSIBOX_DYNAMO_MAX_NEW_TOKENS", "8")), "policy": g("RSIBOX_DYNAMO_POLICY", "kv_aware"), "frontend_mode": g("RSIBOX_DYNAMO_FRONTEND_MODE", "auto"), "gms_socket_dir": g("RSIBOX_DYNAMO_GMS_SOCKET_DIR", ""), "gms_read_only": g("RSIBOX_DYNAMO_GMS_READ_ONLY", "") in ("1", "true", "yes")}
        self.config = {**env_cfg, **(json.loads(json.dumps(config)) if config is not None else {})}; self.config.setdefault("mode", "real")
        self._init_modules, self.startup_timeout, self._workers, self._gen = list(modules if modules is not None else DEFAULT_MODULES), float(startup_timeout), {}, 0
        self._planner_ep: _worker_mod._DynamoRoleEndpoint | None = None
        self._planner_cfg: planner_ops.PlannerConfig | None = None
        self._frontend: FrontendDriver | None = None
        self._owned_gms_store: Any = None
        super().__init__(receipts=receipts, check_timeout=float(check_timeout if check_timeout is not None else g("RSIBOX_DYNAMO_CHECK_TIMEOUT", "60.0")), rpc_timeout=rpc_timeout, weight_store=weight_store, recovery_driver=recovery_driver, auto_spawn=auto_spawn)
    @classmethod
    def launch_environment(cls, args: list[str]) -> dict[str, str]:
        p = argparse.ArgumentParser(prog="rsibox conformance dynamo --real", allow_abbrev=False)
        for flag, default, typ in (("--model", "", str), ("--num-workers", 1, int), ("--max-new-tokens", 8, int), ("--frontend-mode", "auto", str), ("--gms-socket-dir", "", str)): p.add_argument(flag, default=default, type=typ)
        p.add_argument("--gms-read-only", action="store_true")
        ns = p.parse_args(list(args[1:] if args[:1] == ["--"] else args))
        if not ns.model: raise ValueError("--model is required")
        return {"RSIBOX_DYNAMO_MODE": "real", "RSIBOX_DYNAMO_MODEL": ns.model, "RSIBOX_DYNAMO_NUM_WORKERS": str(ns.num_workers), "RSIBOX_DYNAMO_MAX_NEW_TOKENS": str(ns.max_new_tokens), "RSIBOX_DYNAMO_FRONTEND_MODE": ns.frontend_mode, **({"RSIBOX_DYNAMO_GMS_SOCKET_DIR": ns.gms_socket_dir} if ns.gms_socket_dir else {}), **({"RSIBOX_DYNAMO_GMS_READ_ONLY": "1"} if ns.gms_read_only else {})}
    def default_workload(self) -> dict[str, Any]:
        n, policy = max(1, int(self.config.get("num_workers", 1))), str(self.config.get("policy", "kv_aware"))
        prompts, workers, seen, extra = ["The capital of France is", "Water boils at"], [f"worker{i}" for i in range(n)], set(), 0
        for idx, p in enumerate(prompts): seen.add(planner_ops.select_worker(p, workers, None, policy=policy, step_idx=idx)["worker"])
        while len(seen) < n:
            cand = f"Dynamo probe {extra}"; extra += 1
            if (w := planner_ops.select_worker(cand, workers, None, policy=policy, step_idx=len(prompts))["worker"]) not in seen: prompts.append(cand); seen.add(w)
        return {"prompts": prompts, "max_new_tokens": int(self.config.get("max_new_tokens", 8)), "policy": policy}
    def _livelab_paths(self) -> tuple[str, str]:
        ll = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return ll, os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(ll, "..", "..", "..", "core", "livelab"))
    def _gms_uuid_map(self, num_workers: int, visible_devices: str | None) -> dict[str, str]:
        from rsibox.warm.dynamo_gms import build_device_uuid_map, resolve_target_gpus
        targets = resolve_target_gpus(visible_devices=visible_devices, tp=num_workers)
        return build_device_uuid_map(targets) if targets else {}
    def _worker_env(self) -> dict[str, str]:
        ll, core = self._livelab_paths()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **({"GMS_SOCKET_DIR": str(self.config["gms_socket_dir"])} if self.config.get("gms_socket_dir") else {})}
        if self.config.get("mode") == "real":
            if umap := (self.config.get("gms_device_uuid_map") or self._gms_uuid_map(max(1, int(self.config.get("num_workers", 1))), os.environ.get("CUDA_VISIBLE_DEVICES"))):
                env["RSIBOX_GMS_DEVICE_UUID_MAP"] = umap if isinstance(umap, str) else json.dumps(umap)
        env["PYTHONPATH"] = os.pathsep.join([p for p in (ll, core) if p not in (ex := [x for x in (env.get("PYTHONPATH") or "").split(os.pathsep) if x])] + ex)
        return env
    def _ensure_gms_store(self) -> None:
        if self.config.get("mode") != "real": return
        num_workers, vis = max(1, int(self.config.get("num_workers", 1))), os.environ.get("CUDA_VISIBLE_DEVICES")
        if self.weight_store is not None and getattr(self.weight_store, "name", "none") != "none":
            if not (pr := self.weight_store.probe(tp=num_workers, visible_devices=vis)).ok: raise RuntimeError(f"weight store ({self.weight_store.name}) not ready: {pr.error}")
            return
        from rsibox.warm.dynamo_gms import DynamoGmsStore
        uuid_map = self._gms_uuid_map(num_workers, vis)
        store = DynamoGmsStore(socket_dir=str(self.config.get("gms_socket_dir") or "") or None, tags=("weights",), read_only=bool(self.config.get("gms_read_only")), allow_remapped_visible_devices=bool(uuid_map), extra_pythonpath=tuple(self.config.get("extra_pythonpath") or ()))
        if not (pr := store.ensure_ready(tp=num_workers, visible_devices=vis)).ok and not (pr.finding and pr.finding.get("check") == "dynamo_gms_not_committed"): raise RuntimeError(f"DynamoGmsStore failed: {pr.error}")
        self.config["gms_socket_dir"], self.config["gms_device_uuid_map"], self._owned_gms_store = store.socket_dir, uuid_map, store

    def _start_engine(self) -> tuple[Any, list[str]]:
        self._ensure_gms_store(); self._gen += 1
        num_workers, env = max(1, int(self.config.get("num_workers", 1))), self._worker_env()
        self._planner_cfg = planner_ops.PlannerConfig(policy=str(self.config.get("policy", "kv_aware")), max_new_tokens=int(self.config.get("max_new_tokens", 8)))
        self._planner_ep = _worker_mod._DynamoRoleEndpoint("planner", scope={"planner_cfg": self._planner_cfg}, resets={"state": lambda: ["planner_state"]})
        pending: list[str] = []
        try:
            for idx in range(num_workers):
                wname, sock = f"worker{idx}", ipc.short_sock_path(f"rsidyn{idx}")
                with tempfile.NamedTemporaryFile(mode="w", prefix=f"rsidyn-{wname}-", suffix=".json", delete=False, encoding="utf-8") as sf:
                    json.dump({"sock_path": sock, "config": {**self.config, "worker_name": wname, "device_idx": idx, "modules": list(self._init_modules), "parent_pid": os.getpid()}}, sf)
                    pending.append(spec_path := sf.name)
                with tempfile.NamedTemporaryFile(mode="w", prefix=f"rsidyn-{wname}-", suffix=".log", delete=False, encoding="utf-8") as ef:
                    pending.append(err_path := ef.name)
                    p = subprocess.Popen([sys.executable, "-m", "adapter.worker", spec_path], env=env, stdout=subprocess.DEVNULL, stderr=ef, start_new_session=True)
                self._workers[wname] = {"proc": p, "sock": sock, "spec": spec_path, "err": err_path, "device_idx": idx}; pending.clear()
            deadline, ready = time.monotonic() + self.startup_timeout, set()
            while len(ready) < len(self._workers) and time.monotonic() < deadline:
                for wn, w in self._workers.items():
                    if wn not in ready and w["proc"].poll() is not None: raise RuntimeError(f"dynamo {wn} exited (rc={w['proc'].returncode}): {_worker_mod._tail_text(w['err'])}")
                    if wn not in ready and os.path.exists(w["sock"]):
                        with contextlib.suppress(OSError, ValueError):
                            if ipc.unix_json_call(w["sock"], {"method": "ping"}, timeout=5.0).get("ok"): ready.add(wn)
                if len(ready) < len(self._workers): time.sleep(0.05)
            if len(ready) < len(self._workers): raise TimeoutError(f"dynamo workers not ready in {self.startup_timeout} s")
            fe = self._frontend = FrontendDriver(self._handle_frontend_request, mode=str(self.config.get("frontend_mode", "auto")), cache_dir=self.config.get("frontend_cache_dir"))
            fe.start(); planner_ep = self._planner_ep
            parts: list[Any] = [coordinator.CallParticipant("planner0", lambda m, pay, ep=planner_ep: ep.dispatch(m, pay)), *(coordinator.CallParticipant(wn, lambda m, pay, wn=wn: self._worker_call(wn, m, pay)) for wn in sorted(self._workers))]
            return RecaptureCoordinator(parts, gap=lambda fn: fn(), writer=self.writer, interface_version="1-draft", engine={"name": self.name, "version": "dynamo-gms", "commit": None}, phase2_deadline_s=self.rpc_timeout, post_commit_reset="state"), list(self._init_modules)
        except BaseException:
            for path in pending:
                with contextlib.suppress(OSError): os.unlink(path)
            self._stop_engine(); raise

    def _stop_engine(self) -> None:
        self._gen += 1
        if (ep := self._planner_ep) is not None:
            self._planner_ep = None
            with contextlib.suppress(Exception):
                for txn in reversed(list(ep.lp._done)): ep.dispatch("undo", txn)
        if (fe := self._frontend) is not None:
            self._frontend = None
            with contextlib.suppress(Exception): fe.stop()
        workers, self._workers = self._workers, {}
        for w in workers.values():
            _worker_mod._stop_proc(w["proc"], w["sock"], clean=not self._dirty)
            for key in ("sock", "spec", "err"):
                with contextlib.suppress(OSError): os.unlink(w[key])
        if (store := self._owned_gms_store) is not None:
            self._owned_gms_store = None
            with contextlib.suppress(Exception): store.stop()
    def _worker_call(self, wname: str, method: str, payload: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        gen = self._gen
        if self._dirty or wname not in self._workers: raise ConnectionError("worker channel dirty; respawn required")
        w = self._workers[wname]
        try: return ipc.unix_json_call(w["sock"], {"method": method, "payload": payload}, timeout=self.rpc_timeout if timeout is None else timeout, alive_fn=lambda proc_obj=w["proc"]: proc_obj.poll() is None)
        except (TimeoutError, ConnectionError, OSError, ValueError) as exc:
            if self._gen == gen: self._dirty = True
            raise ConnectionError(f"{wname} {method}: {type(exc).__name__}: {exc}") from exc
    def _handle_frontend_request(self, req: dict[str, Any]) -> dict[str, Any]:
        prompt, max_tokens, policy, step_idx = str(req.get("prompt", "")), int(req.get("max_new_tokens", 8)), str(req.get("policy", "kv_aware")), int(req.get("step_idx", 0))
        meta = planner_ops.select_worker(prompt, sorted(self._workers), None, policy=policy, step_idx=step_idx)
        dec = planner_ops.shape_decode_params(prompt, max_tokens, route_meta=meta)
        if not (rep := self._worker_call(meta["worker"], "infer", dec, timeout=self.check_timeout + 5.0)).get("ok"): raise RuntimeError(str(rep.get("error") or f"{meta['worker']} infer failed"))
        return {"ok": True, "response": planner_ops.format_route_response(meta["worker"], prompt, rep["output"], meta)}
    def _probe_participants(self) -> tuple[bool, str | None]:
        for wn in sorted(self._workers):
            try:
                if not self._worker_call(wn, "ping", timeout=min(5.0, self.rpc_timeout)).get("ok"): return False, f"{wn} ping failed"
            except Exception as exc: return False, f"{wn}: {type(exc).__name__}: {exc}"
        return True, None
    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        assert self._frontend is not None and self._planner_cfg is not None
        wl = {**self.default_workload(), **(workload or {})}
        max_tokens, policy, responses = int(wl.get("max_new_tokens", 8)), str(wl.get("policy", "kv_aware")), []
        for idx, prompt in enumerate(wl["prompts"]):
            if not (rep := self._frontend.complete({"prompt": str(prompt), "max_new_tokens": max_tokens, "policy": policy, "step_idx": idx}, timeout=self.check_timeout + 5.0)).get("ok"): raise RuntimeError(str(rep.get("error") or "frontend request failed"))
            responses.append(rep["response"])
        return {"policy": policy, "routing_scale": self._planner_cfg.routing_scale, "responses": responses}
    def _reset_state(self) -> dict[str, Any]:
        reset_items: list[str] = []
        for wn in sorted(self._workers):
            if not (rep := self._worker_call(wn, "reset", "state")).get("ok"): return {"ok": False, "error": rep.get("error"), "reset": []}
            reset_items = list(rep.get("result") or rep.get("reset") or ["kv_cache", "step_counter"])
        return {"ok": True, "reset": reset_items}
    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        plan, findings = super()._build_plan(patch, live_map)
        for mod, spec in plan["modules"].items():
            if mod in ("adapter.planner_ops", "adapter.worker"): spec["on"] = ["planner" if mod == "adapter.planner_ops" else "worker"]
        plan["undo_resets"] = list(patch.get("undo_resets", ["state"])); return plan, findings
    def restart_frontend(self) -> dict[str, Any]:
        """Micro-restart ONLY the HTTP frontend while verifying GMS-resident worker survival."""
        with self.lock:
            if not self._frontend or not self._workers: return {"ok": False, "error": "lab not running"}
            before, fe_info = {wn: self._worker_call(wn, "ping") for wn in sorted(self._workers)}, self._frontend.restart_frontend()
            after = {wn: self._worker_call(wn, "ping") for wn in sorted(self._workers)}
            pids_unchanged, gms_info = all(before[wn]["pid"] == after[wn]["pid"] for wn in self._workers), {}
            if self.config.get("mode") == "real" and self.config.get("gms_socket_dir"):
                from rsibox.warm.dynamo_gms import DynamoGmsStore, _probe_gms_socket
                for path in DynamoGmsStore(socket_dir=self.config["gms_socket_dir"], tags=("weights",), read_only=True, allow_remapped_visible_devices=bool(self.config.get("gms_device_uuid_map"))).socket_paths(tp=len(self._workers), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")) or ():
                    alive, state = _probe_gms_socket(path)
                    gms_info[os.path.basename(path)] = {"alive": alive, "committed": bool(state and state.get("committed") is True), "alloc_count": (state or {}).get("allocation_count")}
            return {**fe_info, "worker_pids_unchanged": pids_unchanged, "weight_reload_count": 0 if pids_unchanged else None, "gms_sockets": gms_info}
    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        ll, core = self._livelab_paths()
        try: overrides = self.files(patch, against="disk")
        except ValueError as exc: return {"ok": False, "output": None, "error": str(exc)}
        with harness.PackageOverlay(self.package, overrides) as overlay:
            assert overlay.root is not None
            return harness.run_fresh_subprocess("adapter.worker", {"config": dict(self.config), "workload": {**self.default_workload(), **(workload or {})}}, overlay.root, extra_pythonpath=(ll, core), timeout=self.check_timeout + 120.0, env=self._worker_env())
    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        if fault == "hang": kwargs.setdefault("hang_s", self.check_timeout + 2.0)
        if fault == "tamper":
            mod = kwargs.setdefault("module", "adapter.worker")
            kwargs.setdefault("qualname", "worker_infer_step" if mod == "adapter.worker" else "shape_decode_params")
        target = str(kwargs.pop("participant", "worker0"))
        with self.lock:
            if target == "planner0" and fault in ("fail_apply", "fail_commit", "tamper"):
                assert self._planner_ep is not None; return harness.inject_participant_fault(self._planner_ep.lp, fault, **kwargs)
            return self._worker_call(target, "inject", {"fault": fault, **kwargs})
    def patches(self) -> dict[str, Any]:
        p_src, w_src = self._base_source("adapter.planner_ops"), self._base_source("adapter.worker")
        e1 = harness.edit_once(p_src, '"token_offset": int(round(bias))', '"token_offset": int(round(bias)) + 1')
        e2 = harness.edit_once(e1, '"max_new_tokens": int(base_max_tokens)', '"max_new_tokens": max(1, int(base_max_tokens) - 2)')
        p_line = '    return {"worker": chosen_worker, "policy": (route_meta or {}).get("policy", "kv_aware"),'
        e3 = harness.edit_once(p_src, p_line, '    return {"worker": chosen_worker, "policy": "edited:" + str((route_meta or {}).get("policy", "kv_aware")),')
        w_multi = harness.edit_once(harness.edit_once(harness.edit_once(w_src, "default=0)", "default=1)"), "return [int(t) + int(token_offset) for t in token_ids]", "return [int(t) + int(token_offset) + 2 for t in token_ids]"), 'kv_cache, hit_chars = runtime["kv_cache"], kv_score_prefix(prompt, runtime["kv_cache"])', 'kv_cache, hit_chars = runtime["kv_cache"], kv_score_prefix(prompt, runtime["kv_cache"]) + 1')
        return {"noop": {"files": {"adapter.planner_ops": harness.edit_once(p_src, 'bias = float((route_meta or {}).get("route_bias", 0.0))', 'bias = float((route_meta or {}).get("route_bias", 0.0)) + 0.0')}}, "visible": {"files": {"adapter.planner_ops": e1}}, "chain": [{"files": {"adapter.planner_ops": e1}}, {"files": {"adapter.planner_ops": e2}}, {"files": {"adapter.planner_ops": harness.edit_once(e2, p_line, '    return {"worker": chosen_worker, "policy": "chain3:" + str((route_meta or {}).get("policy", "kv_aware")),')}}], "multi": {"files": {"adapter.planner_ops": e3, "adapter.worker": w_multi}}, "unreached": {"files": {"adapter.planner_ops": harness.edit_once(p_src, "return 1.0", "return 2.0")}, "accept": ["constructor_edit"]}, "refused": {"syntax": {"files": {"adapter.planner_ops": harness.edit_once(p_src, "return 1.0", "return (")}}, "closure": {"files": {"adapter.planner_ops": harness.edit_once(p_src, "base_bias - 0.01 * step_idx", "base_bias - 0.01 * step_idx + decay")}}, "module_state": {"files": {"adapter.planner_ops": harness.edit_once(p_src, "ROUTING_VERSION = 1", "ROUTING_VERSION = 2")}}, "empty": {"files": {"adapter.planner_ops": harness.edit_once(p_src, "def init_routing_scale() -> float:", "def init_routing_scale() -> float:  # noop")}}}}
