"""The `fastrsi` verl trainer: verl's `PPOTrainerSync` plus a `StepGapCoordinator` edit gate.

Gate protocol (files in `$FASTRSI_RL_GATE`, served at `on_step_begin`, one request at a time):
`ready.json` names the next request number `n`; the client writes `req-<n>.json` and reads
`resp-<n>.json`. Verbs: `install` (files on disk under `$FASTRSI_RL_ROOTS`, or inline
`sources: {module or path: text}`), `check` (run `steps` steps, report metrics and witness),
`status`, `reset` (coupled S0), `uninstall`, `inject` (conformance faults), `stop`.
"""
import contextlib, copy, importlib.util, json, os, socket, sys, time, types
from pathlib import Path
import ray_participants as RP
from kernel import coordinator as C, locate as L, receipt as R, session as SS
from adapter import determinism, faults, settings, state as S0

GATE_ENV, ROOTS_ENV, TRACK_ENV, SEED_ENV = "FASTRSI_RL_GATE", "FASTRSI_RL_ROOTS", "FASTRSI_RL_TRACK", "FASTRSI_RL_SEED_PER_SAMPLE"
MODE, DEFAULT_TRACK, _TRAINER = "fastrsi", ("verl.trainer.ppo.core_algos",), None
_DRIVER = types.SimpleNamespace(__ray_call__=types.SimpleNamespace(remote=lambda fn, *a: fn(None, *a)))


def _serve_verl(host, name, method, payload):
    if faults.STATE.get("lost"):
        raise RuntimeError(f"{name}: participant lost (injected fault)")
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


def path_for_key(key):
    """The file behind a patch key: a path as given, or a module name's origin (None if neither)."""
    if os.path.isfile(key):
        return os.path.realpath(key)
    with contextlib.suppress(Exception):
        spec = importlib.util.find_spec(key)
        if spec and spec.origin and os.path.isfile(spec.origin):
            return os.path.realpath(spec.origin)
    return None


def tracked_modules(config=None, env=os.environ):
    """Modules the session fingerprints (`$FASTRSI_RL_TRACK`, `os.pathsep`-separated module names or
    `.py` paths); default: `core_algos` plus the custom reward file the job loads."""
    if env.get(TRACK_ENV):
        return [t for t in env[TRACK_ENV].split(os.pathsep) if t]
    reward = None
    with contextlib.suppress(Exception):
        reward = config.reward.custom_reward_function.path
    return list(DEFAULT_TRACK) + ([os.path.realpath(reward)] if reward else [])


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
        t_s0, s0 = time.perf_counter(), self._ray_workers(S0.worker_capture_s0)
        self._s0_dataloader = self._dataloader_state()
        self._info = {"pid": os.getpid(), "host": socket.gethostname(), "started": time.time(), "snapshot_files": len(self._sources), "snapshot_secs": round(time.time() - t0, 2),
                      "warm_processes": 1 + len(self._actor_handles()), "s0": s0, "s0_capture_s": round(time.perf_counter() - t_s0, 3),
                      "s0_dataloader": self._s0_dataloader is not None, "tracked": tracked_modules(self.config)}
        _write_json(self._gate / "gate.json", self._info)

    # -- participants ---------------------------------------------------------------------
    def _actor_handles(self):
        """Every warm Ray actor of the job: reward workers, agent-loop (rollout request) workers,
        and the colocated `WorkerDict` trainer ranks. Agent-loop workers exist only once `fit()`
        has set `agent_loop_manager`, so the list grows at the first step gap."""
        rewards = getattr(getattr(self, "reward_loop_manager", None), "reward_loop_worker_handles", None) or ()
        rollouts = getattr(getattr(self, "agent_loop_manager", None), "agent_loop_workers", None) or ()
        trainers = getattr(getattr(self, "actor_rollout_wg", None), "_workers", None) or ()
        return ([(f"reward{i}", h) for i, h in enumerate(rewards)] + [(f"rollout{i}", h) for i, h in enumerate(rollouts)]
                + [(f"trainer{i}", h) for i, h in enumerate(trainers)])

    def _ray_workers(self, fn, *args):
        import ray
        workers = list(getattr(getattr(self, "actor_rollout_wg", None), "_workers", None) or ())
        return ray.get([h.__ray_call__.remote(fn, *args) for h in workers]) if workers else []

    def _ray_call(self, handles, fn, *args):
        import ray
        return ray.get([h.__ray_call__.remote(fn, *args) for h in handles]) if handles else []

    def _ensure_session(self):
        handles = self._actor_handles()
        if self._coordinator is None or [p.name for p in self._coordinator.participants] != ["driver"] + [n for n, _ in handles]:
            if os.environ.get(SEED_ENV) and self._coordinator is None:
                seed = int(self.config.actor_rollout_ref.rollout.get("seed", 42))
                self._info["seed_per_sample"] = self._ray_call([h for n, h in handles if n.startswith("rollout")], determinism.seed_agent_loops, seed)
            parts = [VerlActorParticipant("driver", _DRIVER, get=lambda r, _t: r)] + [VerlActorParticipant(n, h, timeout_s=60.0) for n, h in handles]
            self._coordinator = C.StepGapCoordinator(parts, gap=lambda job: job(), writer=self._writer, engine={"name": "verl", "version": None, "commit": None}, interface_version="1-draft", phase2_deadline_s=60.0)
            self._session = SS.Session(self._coordinator, tracked_modules(self.config))
            self._info.update(warm_processes=len(parts), participants=[p.name for p in parts])
            _write_json(self._gate / "gate.json", self._info)

    # -- coupled S0 ----------------------------------------------------------------------
    def _dataloader_state(self):
        loader = getattr(self, "train_dataloader", None)
        if loader is None or not callable(getattr(loader, "state_dict", None)):
            return None
        try:
            return copy.deepcopy(loader.state_dict())
        except Exception:
            return None

    def _restore_s0(self):
        t0, replies = time.perf_counter(), self._ray_workers(S0.worker_restore_s0)
        if not all(r.get("ok") for r in replies):
            return {"ok": False, "error": "; ".join(r.get("error", "restore failed") for r in replies if not r.get("ok")), "cost_s": round(time.perf_counter() - t0, 6)}
        reset = ["trainer_s0", "rollout_weights", "optimizer", "rng"]
        if self._s0_dataloader is not None:
            self.train_dataloader.load_state_dict(copy.deepcopy(self._s0_dataloader))
            self.train_dataloader_it = None
            reset.append("dataloader")
        synced = S0.sync_rollout_weights(self)
        return {"ok": synced.get("ok", True), "reset": reset, "digests": [r.get("digest") for r in replies], "cost_s": round(time.perf_counter() - t0, 6)}

    def _fingerprint_timings(self):
        """Seconds one fingerprint call takes per participant (what `status` costs, and where)."""
        out = {}
        for p in self._coordinator.participants:
            t0 = time.perf_counter()
            C._call(p, "fingerprint", self._session.modules)
            out[p.name] = round(time.perf_counter() - t0, 3)
        return out

    def _batch_sizes(self):
        cfg, actor = self.config, self.config.actor_rollout_ref.actor
        mcfg = actor.get("megatron") if str(actor.get("strategy", "fsdp")).startswith("megatron") else None
        mp_sz = max(mcfg.tensor_model_parallel_size * mcfg.pipeline_model_parallel_size * mcfg.get("context_parallel_size", 1), 1) if mcfg else 1
        dp_sz, rn = max(len(self.actor_rollout_wg._workers) // mp_sz, 1), cfg.actor_rollout_ref.rollout.n
        return {"train": actor.ppo_mini_batch_size * rn // dp_sz, "infer": cfg.data.train_batch_size * rn // dp_sz}

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

    # -- the gate loop -------------------------------------------------------------------
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
            # ready first, then the previous response: a client that sees the response never reads a stale request number
            _write_json(self._gate / "ready.json", {"n": self._req_index, "step": self.global_steps, "t": time.time()})
            self._flush()
            req_path = self._gate / f"req-{self._req_index}.json"
            while not req_path.exists():
                time.sleep(0.05)
            req, req_num, self._req_index = json.loads(req_path.read_text(encoding="utf-8")), self._req_index, self._req_index + 1
            if req.get("stop"):
                _write_json(self._gate / f"resp-{req_num}.json", {"n": req_num, "ok": True, "stopped": True})
                _write_json(self._gate / "stopped.json", {"t": time.time()})
                raise SystemExit("fastrsi gate: stop requested")
            try:
                self._pending = self._serve_request(req_num, req)
            except Exception as err:
                self._pending = {"n": req_num, "req": req, "t_seen": time.time(), "steps": 0, "steps_run": [], "ok": False, "error": f"{type(err).__name__}: {err}"}
            if self._pending.get("ok") and self._pending.get("steps", 0) > 0:
                self._steps_left = self._pending["steps"] - 1
                if (hang_s := faults.STATE.pop("hang_s", None)) is not None:
                    time.sleep(hang_s)
                if faults.STATE.pop("crash", False):
                    raise RuntimeError("fastrsi gate: injected crash at the step boundary")
                return super().on_step_begin()

    def _edits(self, req):
        """{real path: new text} for the request's `files` (read from disk) and `sources` (inline)."""
        edits = {os.path.realpath(p): Path(os.path.realpath(p)).read_text(encoding="utf-8") for p in req.get("files", [])}
        for key, text in (req.get("sources") or {}).items():
            path = path_for_key(key)
            if path is None:
                raise ValueError(f"{key}: not a file and not an importable module")
            edits[path] = text
        return edits

    def _inject(self, req):
        fault, name, kw = req.get("fault"), req.get("participant", "driver"), dict(req.get("kwargs") or {})
        if fault == "tamper":
            kw.setdefault("module", DEFAULT_TRACK[0]), kw.setdefault("function", "compute_grpo_outcome_advantage")
        if name == "driver":
            return faults.inject("driver", fault, **kw)
        handles = dict(self._actor_handles())
        if name not in handles:
            return {"ok": False, "error": f"no participant {name}; have {['driver'] + sorted(handles)}"}
        return self._ray_call([handles[name]], faults.inject, name, fault, **kw)[0] if kw == {} else \
            self._ray_call([handles[name]], lambda host, n, f, k: faults.inject(n, f, **k), name, fault, kw)[0]

    def _serve_request(self, req_num, req):
        self._ensure_session()
        verb = req.get("verb", "install")
        reply = {"n": req_num, "req": req, "t_seen": time.time(), "steps": int(req.get("steps", 1 if verb in ("install", "check") else 0)), "steps_run": [], "processes": len(self._coordinator.participants)}
        if verb == "status":
            return {**reply, **self._session.status(), **({"fingerprint_s": self._fingerprint_timings()} if req.get("timing") else {}), "steps": 0}
        if verb == "reset":
            return {**reply, **self._restore_s0(), "steps": 0}
        if verb == "uninstall":
            u_out = self._session.uninstall(req.get("id"))
            if u_out.get("undone"):
                del self._live_sources[-len(u_out["undone"]):]
            return {**reply, **u_out, "steps": 0}
        if verb == "inject":
            return {**reply, **self._inject(req), "steps": 0}
        try:
            edits = self._edits(req)
        except Exception as err:
            return {**reply, "ok": False, "outcome": "refused", "codes": ["patch_invalid"], "error": f"patch_invalid: {err}"}
        if any(p not in self._sources for p in edits):
            return {**reply, "ok": False, "outcome": "refused", "codes": ["not_under_roots"], "error": f"not under {ROOTS_ENV}: {[p for p in edits if p not in self._sources]}"}
        try:
            live_vals = settings.plan(req.get("set") or {}, self._batch_sizes())
        except ValueError as err:
            return {**reply, "ok": False, "error": str(err)}
        if edits:
            live_map = self._live_sources[-1]
            modules = {mkey: {"source": text, "base_hash": L.text_hash(live_map.get(p) or self._sources[p]), "base_source": self._sources[p], **({"file": p} if is_f else {})}
                       for p, text in edits.items() for mkey, is_f in (module_for_file(p, self._roots),)}
            plan = {"modules": modules, "accept": req.get("accept", []), "undo_resets": req.get("undo_resets", []), "recipes": ["R0 swap"]}
            if RP.unloaded(self._coordinator.participants, plan):
                return {**reply, "ok": False, "outcome": "aborted", "codes": ["no_live_function"], "error": "not loaded in any warm process"}
            t_inst, res = time.perf_counter(), self._session.install(plan)
            outcome, rec = res.get("outcome", "refused"), res.get("receipt")
            reply.update(install_s=round(time.perf_counter() - t_inst, 3), outcome=outcome, txn=res.get("txn"), ok=outcome == "committed", restart_required=bool(self._session.poisoned()),
                         findings=rec["findings"] if rec else res.get("findings", []), mechanisms=rec["actual_recipes"] if rec else [])
            if outcome != "committed":
                codes = rec["decisions"]["blocking"] if rec else sorted({f["check"] for f in res.get("findings", [])})
                return {**reply, "codes": codes, "error": f"{outcome}: {', '.join(codes)}"}
            self._live_sources.append({**live_map, **edits})
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
                    dumped = next(iter(Path(self.config.trainer.rollout_data_dir).glob(f"{step}.jsonl")), None) if self.config.trainer.get("rollout_data_dir") else None
                    self._pending["steps_run"].append({"step": step, "t": time.time(), "metrics": metrics, "rollouts": str(dumped) if dumped else None})
                    self._pending["metrics"], self._pending["rollouts"] = metrics, str(dumped) if dumped else None
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
