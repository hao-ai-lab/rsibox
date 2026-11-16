"""MilesLab: `LabUnderTest` implementation over `MilesStepGate` and `CoupledWeightReset`."""
import json, os, threading, time
import adapter  # noqa: F401
from adapter import gate as G
from kernel import coordinator as C, locate as L, receipt as R, session as SS


class MilesLab:
    """Live lab host for `miles` (`radixark/miles` / `THUDM/slime`). Never raises from a verb."""
    name = "miles"

    def __init__(self, receipts=None, mode="sync", rpc_timeout=30.0, check_timeout=60.0, **kwargs):
        self.receipts, self.writer = receipts, (R.ReceiptWriter(receipts) if receipts else None)
        self.mode, self.rpc_timeout, self.check_timeout = mode, rpc_timeout, check_timeout
        self.options, self.lock = dict(kwargs), threading.RLock()
        self._spawn()

    def make_runtime(self):
        if callable(launcher := self.options.get("runtime_factory")): return launcher()
        raise RuntimeError("MilesLab requires runtime_factory or ToyMilesLab for CPU conformance")

    def run_step(self, workload=None): return self.runtime["run_step"](workload)

    def _spawn(self):
        rt = self.runtime = self.make_runtime()
        self.modules, self.base_text = list(rt["modules"]), dict(rt["base_text"])
        self.live_text, self.data_buffer, self.coupled_reset = [dict(self.base_text)], rt.get("data_buffer"), rt.get("coupled_reset")
        self.gate = G.MilesStepGate(
            rt["rollout_actors"], rt["train_actors"], extra_actors=rt.get("extra_actors"),
            get=rt.get("get"), timeout_s=self.rpc_timeout, mode=self.mode,
            rollout_controller=rt.get("rollout_controller"), data_buffer=self.data_buffer,
            at_gap_fn=rt.get("at_gap"), writer=self.writer,
            engine_meta={"name": "miles", "version": "0.2.0", "commit": os.environ.get("FASTRSI_ENGINE_COMMIT")})
        self.coordinator = self.gate.coordinator
        if hasattr(self, "session"): self.session.recovered(self.coordinator)
        else: self.session = SS.Session(self.coordinator, self.modules)

    def _halt(self):
        if callable(shutdown := getattr(self, "runtime", {}).get("shutdown")): shutdown()

    def close(self):
        with self.lock: self._halt()

    def status(self):
        with self.lock: return self.session.status()

    def install(self, patch):
        with self.lock:
            started = time.perf_counter()
            if self.session.poisoned():
                refused = self.session.install({"modules": {}})
                codes = sorted({item["check"] for item in refused.get("findings", [])})
                return {"ok": False, "id": None, "outcome": "refused", "codes": codes, "mechanisms": [],
                        "findings": refused.get("findings", []), "cost_s": round(time.perf_counter() - started, 6)}
            base = self.live_text[-1]
            plan, plan_findings = self.gate.build_plan(patch, base)
            if plan is None:
                return {"ok": False, "id": None, "outcome": "refused", "mechanisms": [], "findings": plan_findings,
                        "codes": sorted({item["check"] for item in plan_findings}),
                        "cost_s": round(time.perf_counter() - started, 6)}
            result = self.session.install(plan, edit_time=patch.get("edit_time"))
            outcome, receipt_obj = result.get("outcome", "refused"), result.get("receipt")
            if outcome == "committed":
                self.live_text.append({**base, **(patch.get("files") or {})})
                if self.data_buffer is not None: self.data_buffer.advance_epoch()
            codes = (receipt_obj["decisions"]["blocking"] if receipt_obj
                     else sorted({item["check"] for item in result.get("findings", [])}))
            return {"ok": outcome == "committed", "id": result.get("txn"), "outcome": outcome,
                    "codes": [] if outcome == "committed" else codes,
                    "mechanisms": receipt_obj["actual_recipes"] if receipt_obj else [],
                    "findings": receipt_obj["findings"] if receipt_obj else result.get("findings", []),
                    "receipt": receipt_obj, "cost_s": round(time.perf_counter() - started, 6)}

    def check(self, workload=None):
        with self.lock:
            started = time.perf_counter()
            for participant in self.coordinator.participants:
                if not (ping := C._call(participant, "status", "ping"))["ok"]:
                    self.session.fault("participant_failed", f"{participant.name}: {ping.get('error')}",
                                       "every actor answers", subject=participant.name)
                    return {"ok": False, "error": ping.get("error"), "output": None, "proven": None,
                            "witness": {}, "findings": list(self.session.poisoned()),
                            "cost_s": round(time.perf_counter() - started, 6)}

            def step_job():
                if self.coupled_reset is not None: self.coupled_reset.restore_and_sync()
                return self.run_step(workload)

            try:
                output = json.loads(json.dumps(self.gate.at_gap(step_job, timeout=self.check_timeout)))
            except Exception as err:
                self.session.fault("participant_failed", f"check: {type(err).__name__}: {err}",
                                   f"the RL step finishes within {self.check_timeout} s")
                return {"ok": False, "error": f"{type(err).__name__}: {err}", "output": None, "proven": None,
                        "witness": {}, "findings": list(self.session.poisoned()),
                        "cost_s": round(time.perf_counter() - started, 6)}
            golden = {"output_hash": L.text_hash(json.dumps(output, sort_keys=True))}
            proofs = [self.coordinator.prove(txn_id, golden=golden) for txn_id in self.session.stack]
            witness = {}
            for proof in proofs:
                for part_name, counts in G.clean_witness(proof["receipt"]["witness"], proof["ok"]).items():
                    witness.setdefault(part_name, {}).update(counts)
            return {"ok": True, "output": output, "proven": all(p["ok"] for p in proofs) if proofs else None,
                    "witness": witness, "findings": [item for p in proofs for item in p["receipt"]["findings"]],
                    "cost_s": round(time.perf_counter() - started, 6)}

    def reset(self, kind="state"):
        with self.lock:
            started = time.perf_counter()
            if kind == "respawn":
                self._halt(); self._spawn()
                return {"ok": self.session.state() == "base", "reset": ["ray_actors", "session"],
                        "cost_s": round(time.perf_counter() - started, 6)}
            try:
                summary = self.gate.at_gap(
                    lambda: self.coupled_reset.restore_and_sync() if self.coupled_reset else {"reset": ["state"]},
                    timeout=self.check_timeout)
            except Exception as err:
                return {"ok": False, "error": f"{type(err).__name__}: {err}", "reset": [],
                        "cost_s": round(time.perf_counter() - started, 6)}
            return {"ok": True, "reset": summary["reset"], "cost_s": round(time.perf_counter() - started, 6)}

    def uninstall(self, id=None):
        with self.lock:
            started = time.perf_counter()
            result = self.session.uninstall(id)
            if undone_count := len(result.get("undone", [])):
                del self.live_text[-undone_count:]
                if self.data_buffer is not None: self.data_buffer.advance_epoch()
            return {"ok": result["ok"], "undone": result.get("undone", []), "state": self.session.state(),
                    "findings": result.get("findings", []), "cost_s": round(time.perf_counter() - started, 6)}

    def patches(self): raise NotImplementedError("Subclasses or configured runtimes provide patches()")
    def fresh_check(self, patch, workload=None): raise NotImplementedError("Subclasses provide fresh_check()")
    def inject(self, fault, **kwargs): raise NotImplementedError("Subclasses provide inject()")
