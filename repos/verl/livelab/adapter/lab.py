"""VerlLab: `BaseLab` (`LabUnderTest`) over a multi-actor synchronous RL cluster."""
import os

import ray_participants as RP
from adapter import LIVELAB
from fastrsi import harness
from fastrsi.lab import BaseLab
from kernel import coordinator as C, findings as F, locate as L


class VerlLab(BaseLab):
    name, package, fresh_module = "verl", "adapter", "adapter.toyverl.fresh"
    threaded_check = True

    def __init__(self, config=None, receipts=None, modules=(), rpc_timeout=10.0, check_timeout=30.0):
        self.config = dict(config or {})
        self._init_modules = list(modules)
        self.dirty = None
        self.cluster = None
        self.coordinator = None
        super().__init__(receipts=receipts, check_timeout=check_timeout, rpc_timeout=rpc_timeout)

    def make_cluster(self):
        raise NotImplementedError

    def default_workload(self):
        return {"prompts": [1, 2, 3, 4], "steps": 1}

    def format_output(self, step_result):
        return step_result

    def _base(self, mod_or_path):
        return self._base_source(mod_or_path)

    def _start_engine(self):
        self.dirty = None
        self.cluster = self.make_cluster()
        parts = [
            RP.RayActorParticipant(n, h, get=self._make_getter(n), timeout_s=self.rpc_timeout)
            for n, h in self.cluster.actor_handles()
        ]
        co = C.StepGapCoordinator(
            parts,
            gap=lambda job: job(),
            writer=self.writer,
            interface_version="1-draft",
            engine={"name": self.name, "version": None, "commit": os.environ.get("FASTRSI_ENGINE_COMMIT")},
            phase2_deadline_s=self.rpc_timeout,
        )
        self.coordinator = co
        self.cluster.capture_s0()
        return co, list(self._init_modules)

    def _stop_engine(self):
        if self.cluster is not None:
            self.cluster.shutdown()
            self.cluster = None

    def _make_getter(self, actor_name):
        def getter(ref, timeout_s):
            if self._dirty or self.dirty is not None:
                raise ConnectionError(f"actor {self.dirty or 'check'} missed a deadline; respawn required")
            try:
                return self.cluster.resolve_ref(ref, timeout_s or self.rpc_timeout)
            except TimeoutError:
                self.dirty = actor_name
                self._dirty = True
                raise
        return getter

    def _build_plan(self, patch, live_map):
        modules = {}
        for k, new_src in patch["files"].items():
            base_src = live_map.get(k)
            if base_src is None:
                base_src = self._base_source(k)
            is_file = os.path.isfile(k)
            modules[RP.file_module(k) if is_file else k] = {
                "source": new_src,
                "base_hash": L.text_hash(base_src),
                **({"file": os.path.realpath(k)} if is_file else {}),
            }
        plan = {
            "modules": modules,
            "accept": list(patch.get("accept", [])),
            "undo_resets": list(patch.get("undo_resets", [])),
            "recipes": list(patch.get("recipes", ["R0 swap"])),
        }
        try:
            unloaded = RP.unloaded(self.co.participants, plan)
        except Exception as err:
            self.session.fault("participant_failed", f"{type(err).__name__}: {err}", "every actor answers")
            return None, list(self.session.poisoned())
        if unloaded:
            return None, [F.finding("no_live_function", f"no warm actor loaded {unloaded}", "loaded module")]
        return plan, []

    def _probe_participants(self):
        try:
            self.cluster.ping_all(timeout_s=min(self.rpc_timeout, 5.0))
            return True, None
        except Exception as err:
            return False, f"ping: {type(err).__name__}: {err}"

    def _run_workload(self, workload):
        r_s0 = self.cluster.restore_s0()
        if not r_s0.get("ok"):
            raise RuntimeError(f"S0 restore failed: {r_s0.get('error')}")
        return self.cluster.run_steps(dict(workload or self.default_workload()))

    def _format_output(self, raw):
        return self.format_output(raw), {}

    def _golden_for(self, output, _extra):
        return {"output": output}

    def _clean_witness(self, raw_witness, _proof_ok):
        return {p: c for p, c in (raw_witness or {}).items() if c}

    def _reset_state(self):
        res = self.cluster.restore_s0()
        return {
            "ok": bool(res.get("ok")),
            "error": res.get("error"),
            "reset": res.get("reset", ["trainer_s0", "rollout_weights", "optimizer", "rng"]),
        }

    def fresh_check(self, patch, workload=None):
        files = (patch or {}).get("files") or {}
        with harness.PackageOverlay(self.package, files) as ov:
            reward_override = next(iter(ov.standalone.values()), None)
            cfg = {**self.config, **({"reward_file": reward_override} if reward_override else {})}
            spec = {"config": cfg, "workload": dict(workload or self.default_workload())}
            return harness.run_fresh_subprocess(
                self.fresh_module,
                spec,
                ov.root,
                extra_pythonpath=(LIVELAB,),
                timeout=self.check_timeout + 30.0,
            )

    def inject(self, fault, **kw):
        if fault == "hang":
            kw.setdefault("hang_s", self.check_timeout + 2.0)
        with self.lock:
            return self.cluster.inject_fault(fault, **kw)
