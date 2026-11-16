"""VerlLab: the `LabUnderTest` conformance interface over a multi-actor synchronous RL cluster."""
import importlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path
import ray_participants as RP
from kernel import coordinator as C, findings as F, locate as L, receipt as R, session as SS


class VerlLab:
    name, package, fresh_module = "verl", "adapter", "adapter.toyverl.fresh"

    def __init__(self, config=None, receipts=None, modules=(), rpc_timeout=10.0, check_timeout=30.0):
        self.config, self.writer = dict(config or {}), (R.ReceiptWriter(receipts) if receipts else None)
        self.rpc_timeout, self.check_timeout, self.modules = rpc_timeout, check_timeout, list(modules)
        self.lock, self.costs = threading.RLock(), {}
        self._spawn()

    def make_cluster(self):
        raise NotImplementedError

    def default_workload(self):
        return {"prompts": [1, 2, 3, 4], "steps": 1}

    def format_output(self, step_result):
        return step_result

    def patches(self):
        raise NotImplementedError

    def _spawn(self):
        t0 = time.perf_counter()
        self.cluster = self.make_cluster()
        self.costs["load_s"] = round(time.perf_counter() - t0, 4)
        self.dirty, self.base_text, self.live_text = None, {}, [{}]
        parts = [RP.RayActorParticipant(n, h, get=self._make_getter(n), timeout_s=self.rpc_timeout)
                 for n, h in self.cluster.actor_handles()]
        self.coordinator = C.StepGapCoordinator(parts, gap=lambda job: job(), writer=self.writer, interface_version="1-draft",
            engine={"name": self.name, "version": None, "commit": os.environ.get("FASTRSI_ENGINE_COMMIT")}, phase2_deadline_s=self.rpc_timeout)
        if hasattr(self, "session"):
            self.session.recovered(self.coordinator)
        else:
            self.session = SS.Session(self.coordinator, self.modules)
        self.cluster.capture_s0()

    def _make_getter(self, actor_name):
        def getter(ref, timeout_s):
            if self.dirty is not None:
                raise ConnectionError(f"actor {self.dirty} missed a deadline; respawn required")
            try:
                return self.cluster.resolve_ref(ref, timeout_s or self.rpc_timeout)
            except TimeoutError:
                self.dirty = actor_name
                raise
        return getter

    def close(self):
        with self.lock:
            try:
                self.cluster.shutdown()
            except Exception:
                pass

    def _base(self, mod_or_path):
        if mod_or_path not in self.base_text:
            path = mod_or_path if os.path.isfile(mod_or_path) else importlib.util.find_spec(mod_or_path).origin
            self.base_text[mod_or_path] = Path(path).read_text(encoding="utf-8")
        return self.base_text[mod_or_path]

    def _refuse_install(self, outcome, codes, findings, t0):
        return {"ok": False, "id": None, "outcome": outcome, "codes": codes, "mechanisms": [], "findings": findings, "cost_s": round(time.perf_counter() - t0, 6)}

    def install(self, patch):
        t0 = time.perf_counter()
        with self.lock:
            if self.session.poisoned():
                return self._refuse_install("refused", ["session_poisoned"], self.session._refuse("install").get("findings", []), t0)
            live_map, modules = self.live_text[-1], {}
            try:
                for k, new_src in patch["files"].items():
                    bh = L.text_hash(live_map.get(k) or self._base(k))
                    modules[RP.file_module(k) if os.path.isfile(k) else k] = {"source": new_src, "base_hash": bh, **({"file": os.path.realpath(k)} if os.path.isfile(k) else {})}
            except Exception as err:
                return self._refuse_install("refused", ["patch_invalid"], [F.finding("patch_invalid", f"{type(err).__name__}: {err}", "importable modules")], t0)
            plan = {"modules": modules, "accept": patch.get("accept", []), "undo_resets": patch.get("undo_resets", []), "recipes": ["R0 swap"]}
            try:
                unloaded = RP.unloaded(self.coordinator.participants, plan)
            except Exception as err:
                self.session.fault("participant_failed", f"{type(err).__name__}: {err}", "every actor answers")
                return self._refuse_install("aborted", ["participant_failed"], list(self.session.poisoned()), t0)
            if unloaded:
                return self._refuse_install("aborted", ["no_live_function"], [F.finding("no_live_function", f"no warm actor loaded {unloaded}", "loaded module")], t0)
            res = self.session.install(plan, edit_time=patch.get("edit_time"))
            outcome, rec = res.get("outcome") or "refused", res.get("receipt")
            if outcome == "committed":
                self.live_text.append({**live_map, **patch["files"]})
            codes = rec["decisions"]["blocking"] if rec else sorted({f["check"] for f in res.get("findings", [])})
            return {"ok": outcome == "committed", "id": res.get("txn"), "outcome": outcome, "codes": [] if outcome == "committed" else codes,
                    "mechanisms": rec["actual_recipes"] if rec else [], "findings": rec["findings"] if rec else res.get("findings", []),
                    "edit_to_run_s": rec.get("edit_to_run_s") if rec else None, "receipt": rec, "cost_s": round(time.perf_counter() - t0, 6)}

    def _fail_check(self, observed, expected, t0):
        self.session.fault("participant_failed", observed, expected)
        return {"ok": False, "error": observed, "output": None, "proven": None, "witness": {}, "findings": list(self.session.poisoned()), "cost_s": round(time.perf_counter() - t0, 6)}

    def check(self, workload=None):
        t0 = time.perf_counter()
        with self.lock:
            try:
                self.cluster.ping_all(timeout_s=min(self.rpc_timeout, 5.0))
            except Exception as err:
                return self._fail_check(f"ping: {type(err).__name__}: {err}", "every actor answers", t0)
            r_s0 = self.cluster.restore_s0()
            if not r_s0.get("ok"):
                return self._fail_check(f"S0 restore failed: {r_s0.get('error')}", "coupled S0 restores", t0)
            req, box = dict(workload or self.default_workload()), {}
            def run_step():
                try:
                    box["result"] = self.cluster.run_steps(req)
                except BaseException as err:
                    box["error"] = f"{type(err).__name__}: {err}"
            th, t_step = threading.Thread(target=run_step, daemon=True), time.perf_counter()
            th.start()
            th.join(self.check_timeout)
            secs = round(time.perf_counter() - t_step, 4)
            if th.is_alive():
                self.dirty = "check"
                return self._fail_check(f"check: no result in {self.check_timeout} s", "step finishes in time", t0)
            if "error" in box:
                return self._fail_check(f"check: {box['error']}", "RL step completes", t0)
            out = self.format_output(box["result"])
            proofs = [self.coordinator.prove(tid, golden={"output": out}) for tid in self.session.stack]
            witness = {}
            for prf in proofs:
                for pname, cnts in prf["receipt"]["witness"].items():
                    if cnts:
                        witness.setdefault(pname, {}).update(cnts)
            return {"ok": True, "output": out, "seconds": secs, "proven": all(p["ok"] for p in proofs) if proofs else None,
                    "proven_by_txn": {t: p["ok"] for t, p in zip(self.session.stack, proofs)},
                    "witness": witness, "findings": [f for p in proofs for f in p["receipt"]["findings"]], "cost_s": round(time.perf_counter() - t0, 6)}

    def reset(self, kind="state"):
        t0 = time.perf_counter()
        with self.lock:
            if kind == "respawn":
                try:
                    self.cluster.shutdown()
                except Exception:
                    pass
                self._spawn()
                return {"ok": not self.session.poisoned(), "reset": ["ray_actors", "coupled_s0", "session"], "cost_s": round(time.perf_counter() - t0, 6)}
            res = self.cluster.restore_s0()
            return {"ok": bool(res.get("ok")), "error": res.get("error"), "reset": res.get("reset", ["trainer_s0", "rollout_weights", "optimizer", "rng"]), "cost_s": round(time.perf_counter() - t0, 6)}

    def uninstall(self, id=None):
        t0 = time.perf_counter()
        with self.lock:
            u = self.session.uninstall(id)
            if u.get("undone"):
                del self.live_text[-len(u["undone"]):]
            return {"ok": u["ok"], "undone": u.get("undone", []), "state": self.session.state(), "findings": u.get("findings", []), "cost_s": round(time.perf_counter() - t0, 6)}

    def status(self):
        with self.lock:
            return self.session.status()

    def fresh_check(self, patch, workload=None):
        root = os.path.dirname(os.path.dirname(importlib.import_module(self.package).__file__))
        tmp = tempfile.mkdtemp(prefix="verllab-fresh-")
        try:
            for d, dirs, files in os.walk(os.path.join(root, self.package)):
                dirs[:] = [x for x in dirs if x != "__pycache__"]
                os.makedirs(os.path.join(tmp, os.path.relpath(d, root)), exist_ok=True)
                for fn in files:
                    os.symlink(os.path.join(d, fn), os.path.join(tmp, os.path.relpath(d, root), fn))
            reward_override = None
            for k, text in ((patch or {}).get("files") or {}).items():
                if os.path.isfile(k):
                    reward_override = os.path.join(tmp, os.path.basename(k))
                    Path(reward_override).write_text(text, encoding="utf-8")
                else:
                    target_py = os.path.join(tmp, *k.split(".")) + ".py"
                    if os.path.islink(target_py) or os.path.exists(target_py):
                        os.unlink(target_py)
                    Path(target_py).write_text(text, encoding="utf-8")
            cfg = {**self.config, **({"reward_file": reward_override} if reward_override else {})}
            spec = os.path.join(tmp, "fresh.json")
            Path(spec).write_text(json.dumps({"config": cfg, "workload": dict(workload or self.default_workload())}), encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([tmp, os.environ.get("PYTHONPATH", "")])}
            proc = subprocess.run([sys.executable, "-m", self.fresh_module, spec], env=env, capture_output=True, text=True, timeout=self.check_timeout + 30.0, cwd=tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("FRESH ")]
        return {"ok": True, **json.loads(lines[-1][6:])} if (not proc.returncode and lines) else {"ok": False, "error": proc.stderr.strip().splitlines()[-5:], "output": None}

    def inject(self, fault, **kw):
        if fault == "hang":
            kw.setdefault("hang_s", self.check_timeout + 2.0)
        with self.lock:
            return self.cluster.inject_fault(fault, **kw)
