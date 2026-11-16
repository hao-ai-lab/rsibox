"""The `fastrsi` verl trainer: verl's `PPOTrainerSync` plus a `StepGapCoordinator` edit gate."""
import contextlib, importlib.util, json, os, sys, time, types
from pathlib import Path
import ray_participants as RP
from kernel import coordinator as C, locate as L, receipt as R, session as SS
from adapter import settings, state as S0

GATE_ENV, ROOTS_ENV, MODE, _TRAINER = "FASTRSI_RL_GATE", "FASTRSI_RL_ROOTS", "fastrsi", None
_DRIVER = types.SimpleNamespace(__ray_call__=types.SimpleNamespace(remote=lambda fn, *a: fn(None, *a)))


def _serve_verl(host, name, method, payload):
    if method == "prepare" and isinstance(payload, dict):
        local = RP._LOCAL.get(name) or RP._LOCAL.setdefault(name, C.LocalParticipant(name))
        base_map = local._sources.setdefault("__base__", {})
        for mkey, spec in (payload.get("modules") or {}).items():
            if "base_source" in spec:
                base_map.setdefault(mkey, spec["base_source"])
                base_map.setdefault(mkey + RP.ACTOR, spec["base_source"])
    return RP.serve(host, name, method, payload)


class VerlActorParticipant(RP.RayActorParticipant):
    def __init__(self, name, handle, get=None, timeout_s=None, bounded=False):
        fetch = get or RP._ray_get
        C.CallParticipant.__init__(self, name, lambda m, p: fetch(handle.__ray_call__.remote(_serve_verl, name, m, p), timeout_s), bounded)


def _write_json(path, data):
    tmp = path.with_name("." + path.name)
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    tmp.rename(path)


def module_for_file(file_path, roots=()):
    real = os.path.realpath(file_path)
    for mname, loaded in list(sys.modules.items()):
        origin = getattr(loaded, "__file__", None)
        if isinstance(origin, str) and os.path.realpath(origin) == real and not mname.startswith(("fastrsi_file_", "custom_module_")):
            return mname, False
    for entry in [e for e in list(sys.path) + [os.path.dirname(os.path.realpath(r)) for r in roots if r] if e]:
        base = os.path.realpath(entry)
        if real.startswith(base + os.sep) and real.endswith(".py"):
            parts = [p for p in os.path.relpath(real, base)[:-3].split(os.sep) if p != "__init__"]
            if parts and all(p.isidentifier() for p in parts):
                with contextlib.suppress(ImportError, ValueError):
                    if (spec := importlib.util.find_spec(".".join(parts))) and spec.origin and os.path.realpath(spec.origin) == real:
                        return ".".join(parts), False
    return RP.file_module(real), True


class GateMixin:
    def on_init_end(self):
        super().on_init_end()
        self._gate = Path(os.environ[GATE_ENV]) if os.environ.get(GATE_ENV) else None
        if self._gate is None:
            return
        self._gate.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self._roots = [r for r in os.environ.get(ROOTS_ENV, "").split(os.pathsep) if r]
        self._sources = {os.path.realpath(p): p.read_text(encoding="utf-8", errors="ignore") for r in self._roots for p in Path(r).rglob("*.py") if p.is_file()}
        self._live_sources, self._req_index, self._steps_left, self._pending, self._coordinator, self._session = [{}], 0, 0, None, None, None
        self._writer = R.ReceiptWriter(str(self._gate / "receipts.jsonl"))
        if not self.config.trainer.get("rollout_data_dir"):
            self.config.trainer.rollout_data_dir = str(self._gate / "rollouts")
        self._record_advantages()
        self._ray_workers(S0.worker_capture_s0)
        _write_json(self._gate / "gate.json", {"pid": os.getpid(), "started": time.time(), "snapshot_files": len(self._sources), "snapshot_secs": round(time.time() - t0, 2), "warm_processes": 1 + len(self._actor_handles())})

    def _actor_handles(self):
        return ([(f"reward{i}", h) for i, h in enumerate(getattr(getattr(self, "reward_loop_manager", None), "reward_loop_worker_handles", None) or ())]
                + [(f"trainer{i}", h) for i, h in enumerate(getattr(getattr(self, "actor_rollout_wg", None), "_workers", None) or ())])

    def _ray_workers(self, fn, *args):
        import ray
        workers = list(getattr(getattr(self, "actor_rollout_wg", None), "_workers", None) or ())
        return ray.get([h.__ray_call__.remote(fn, *args) for h in workers]) if workers else []

    def _ensure_session(self, tracked_modules=()):
        handles = self._actor_handles()
        if self._coordinator is None or [p.name for p in self._coordinator.participants] != ["driver"] + [n for n, _ in handles]:
            parts = [VerlActorParticipant("driver", _DRIVER, get=lambda r, _t: r)] + [VerlActorParticipant(n, h, timeout_s=60.0) for n, h in handles]
            self._coordinator = C.StepGapCoordinator(parts, gap=lambda job: job(), writer=self._writer, engine={"name": "verl", "version": None, "commit": None}, interface_version="1-draft", phase2_deadline_s=60.0)
            self._session = SS.Session(self._coordinator, list(tracked_modules))

    def _restore_s0(self):
        t0, replies = time.perf_counter(), self._ray_workers(S0.worker_restore_s0)
        if not all(r.get("ok") for r in replies):
            return {"ok": False, "error": "; ".join(r.get("error", "restore failed") for r in replies if not r.get("ok")), "cost_s": round(time.perf_counter() - t0, 6)}
        return {"ok": S0.sync_rollout_weights(self).get("ok", True), "reset": ["trainer_s0", "rollout_weights", "optimizer", "rng"], "digests": [r.get("digest") for r in replies], "cost_s": round(time.perf_counter() - t0, 6)}

    def _batch_sizes(self):
        cfg, mcfg = self.config, self.config.actor_rollout_ref.actor.megatron
        mp_sz = max(mcfg.tensor_model_parallel_size * mcfg.pipeline_model_parallel_size * mcfg.get("context_parallel_size", 1), 1)
        dp_sz, rn = max(len(self.actor_rollout_wg._workers) // mp_sz, 1), cfg.actor_rollout_ref.rollout.n
        return {"train": cfg.actor_rollout_ref.actor.ppo_mini_batch_size * rn // dp_sz, "infer": cfg.data.train_batch_size * rn // dp_sz}

    def _record_advantages(self):
        from verl.trainer.ppo.v1 import trainer_base
        orig, trainer = trainer_base.compute_advantage_for_multi_trajectories, self
        def recorded(data, batch_keys, **kw):
            import torch
            from omegaconf import OmegaConf
            inputs, out = {k: data.batch[k].clone() for k in ("token_level_rewards", "response_mask")}, orig(data, batch_keys, **kw)
            step_dir = trainer._gate / "steps" / str(trainer.global_steps)
            step_dir.mkdir(parents=True, exist_ok=True)
            skw = {k: str(v) if k == "adv_estimator" else v for k, v in kw.items() if k != "config"}
            torch.save({"inputs": inputs, "uid": list(data.non_tensor_batch["uid"]), "batch_keys": list(batch_keys), "kwargs": skw, "config": OmegaConf.to_container(kw["config"]) if kw.get("config") is not None else None, "advantages": out.batch["advantages"].clone()}, step_dir / "adv.pt")
            return out
        trainer_base.compute_advantage_for_multi_trajectories = recorded

    def _flush(self):
        if self._pending is None: return
        pending, self._pending = self._pending, None
        pending["verdict_s"] = round(time.time() - pending.pop("t_seen"), 3)
        if pending["steps_run"] and self._session is not None and self._session.stack:
            proofs, witness, totals = [self._coordinator.prove(tid) for tid in self._session.stack], {}, {}
            for prf in proofs:
                for pname, cnts in prf["receipt"]["witness"].items():
                    if cnts: witness.setdefault(pname, {}).update(cnts); [totals.update({k: totals.get(k, 0) + v}) for k, v in cnts.items()]
            findings = [f for prf in proofs for f in prf["receipt"]["findings"] if not (f["check"] == "zero_witness" and totals.get(f.get("subject"), 0) > 0)]
            pending.update(witness=witness, proven=not findings, findings=findings)
            if not pending["proven"]: pending.update(ok=False, error="edited code never ran: " + ", ".join(sorted({str(f.get("subject") or f["check"]) for f in findings})))
        _write_json(self._gate / f"resp-{pending['n']}.json", pending)

    def on_step_begin(self):
        if self._gate is None or self._steps_left > 0:
            if self._steps_left > 0:
                self._steps_left -= 1
            return super().on_step_begin()
        while True:
            self._flush()
            _write_json(self._gate / "ready.json", {"n": self._req_index, "step": self.global_steps, "t": time.time()})
            req_path = self._gate / f"req-{self._req_index}.json"
            while not req_path.exists():
                time.sleep(0.05)
            req, req_num, self._req_index = json.loads(req_path.read_text(encoding="utf-8")), self._req_index, self._req_index + 1
            if req.get("stop"):
                _write_json(self._gate / f"resp-{req_num}.json", {"n": req_num, "ok": True, "stopped": True})
                _write_json(self._gate / "stopped.json", {"t": time.time()})
                raise SystemExit("fastrsi gate: stop requested")
            self._pending = self._serve_request(req_num, req)
            if self._pending.get("ok") and self._pending.get("steps", 0) > 0:
                self._steps_left = self._pending["steps"] - 1
                return super().on_step_begin()

    def _serve_request(self, req_num, req):
        self._ensure_session()
        verb = req.get("verb", "install")
        reply = {"n": req_num, "req": req, "t_seen": time.time(), "steps": int(req.get("steps", 1 if verb == "install" else 0)), "steps_run": [], "processes": len(self._coordinator.participants)}
        if verb in ("status", "reset"):
            return {**reply, **(self._session.status() if verb == "status" else self._restore_s0()), "steps": 0}
        if verb == "uninstall":
            u_out = self._session.uninstall(req.get("id"))
            if u_out.get("undone"):
                del self._live_sources[-len(u_out["undone"]):]
            return {**reply, **u_out, "steps": 0}
        files = [os.path.realpath(p) for p in req.get("files", [])]
        if any(p not in self._sources for p in files):
            return {**reply, "ok": False, "error": "not under FASTRSI_RL_ROOTS"}
        try:
            live_vals = settings.plan(req.get("set") or {}, self._batch_sizes())
        except ValueError as err:
            return {**reply, "ok": False, "error": str(err)}
        if files:
            live_map = self._live_sources[-1]
            modules = {mkey: {"source": Path(p).read_text(encoding="utf-8"), "base_hash": L.text_hash(live_map.get(p) or self._sources[p]), "base_source": self._sources[p], **({"file": p} if is_f else {})}
                       for p in files for mkey, is_f in (module_for_file(p, self._roots),)}
            plan = {"modules": modules, "accept": req.get("accept", []), "undo_resets": req.get("undo_resets", []), "recipes": ["R0 swap"]}
            if RP.unloaded(self._coordinator.participants, plan):
                return {**reply, "ok": False, "outcome": "aborted", "codes": ["no_live_function"], "error": "not loaded in any warm process"}
            t_inst, res = time.perf_counter(), self._session.install(plan)
            outcome, rec = res.get("outcome", "refused"), res.get("receipt")
            reply.update(install_s=round(time.perf_counter() - t_inst, 3), outcome=outcome, txn=res.get("txn"), ok=outcome == "committed", restart_required=bool(self._session.poisoned()))
            if outcome != "committed":
                codes = rec["decisions"]["blocking"] if rec else sorted({f["check"] for f in res.get("findings", [])})
                return {**reply, "codes": codes, "error": f"{outcome}: {', '.join(codes)}"}
            self._live_sources.append({**live_map, **{p: Path(p).read_text(encoding="utf-8") for p in files}})
        else:
            reply.update(ok=True, install_s=0.0)
        if reply["ok"] and live_vals:
            from omegaconf import OmegaConf, open_dict
            prev = self._ray_workers(settings.apply, live_vals)[0]
            with open_dict(self.config):
                for k, v in live_vals.items():
                    OmegaConf.update(self.config, f"{settings.ROLE}.{k}", v)
            reply["set"] = {k: [prev[k], v] for k, v in live_vals.items()}
        if not getattr(self, "_log_hooked", False):
            self._log_hooked, orig = True, self.logger.log
            def log(data, step, **kw):
                if self._pending is not None:
                    metrics = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
                    self._pending["steps_run"].append({"step": step, "t": time.time(), "metrics": metrics})
                    self._pending["metrics"] = metrics
                return orig(data=data, step=step, **kw)
            self.logger.log = log
        return reply

    def on_train_end(self):
        if self._gate is not None:
            self._flush()
            _write_json(self._gate / "stopped.json", {"t": time.time()})
        return super().on_train_end()


def register():
    global _TRAINER
    if _TRAINER is None:
        from verl.trainer.ppo.v1.trainer_base import register_trainer
        from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
        trainer = type("PPOTrainerFastRSISync", (GateMixin, PPOTrainerSync), {"__module__": __name__})
        globals()[trainer.__name__] = trainer
        _TRAINER = register_trainer(MODE)(trainer)
    return _TRAINER
