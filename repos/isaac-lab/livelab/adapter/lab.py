"""IsaacLabLiveLab: RSIBox live lab for Isaac Lab GPU robotics simulation and RL."""
from __future__ import annotations

import argparse, contextlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any

from adapter.worker import require_isaac_lab_deps, validate_positive_int
from kernel import coordinator
from rsibox import harness, ipc, proc
from rsibox.lab import BaseLab, RecaptureCoordinator

DEFAULT_MODULES = ("adapter.env_ops", "adapter.worker")


def _env_config() -> dict[str, Any]:
    g = os.environ.get
    return {"mode": g("RSIBOX_ISAACLAB_MODE", "isaaclab"), "task": g("RSIBOX_ISAACLAB_TASK", "Isaac-Cartpole-Direct"), "physics": g("RSIBOX_ISAACLAB_PHYSICS", "newton_mjwarp"), "num_envs": int(g("RSIBOX_ISAACLAB_NUM_ENVS", "4096")), "horizon_steps": int(g("RSIBOX_ISAACLAB_HORIZON_STEPS", "10")), "seed": int(g("RSIBOX_ISAACLAB_SEED", "42")), "device": g("RSIBOX_ISAACLAB_DEVICE", "cuda:0")}


class IsaacLabLiveLab(BaseLab):
    """RSIBox live lab over resident Isaac Lab GPU simulation or CPU batched robotics workers."""

    name, package, threaded_check = "isaac-lab", "adapter", True

    def __init__(self, config: dict[str, Any] | None = None, modules: tuple[str, ...] | list[str] | None = None, *, receipts: str | None = None, rpc_timeout: float = 30.0, check_timeout: float | None = None, startup_timeout: float | None = None, weight_store: Any = None, recovery_driver: Any = None, auto_spawn: bool = True) -> None:
        self.config = {**_env_config(), **(json.loads(json.dumps(config)) if config is not None else {})}
        self.config["num_envs"] = validate_positive_int(self.config["num_envs"], "num_envs")
        self.config["horizon_steps"] = validate_positive_int(self.config["horizon_steps"], "horizon_steps")
        self._init_modules = list(modules if modules is not None else DEFAULT_MODULES)
        is_cpu = str(self.config.get("mode", "isaaclab")) == "cpu"
        self.startup_timeout = float(startup_timeout if startup_timeout is not None else (15.0 if is_cpu else 120.0))
        ct = float(check_timeout if check_timeout is not None else os.environ.get("RSIBOX_ISAACLAB_CHECK_TIMEOUT", "2.0" if is_cpu else "30.0"))
        self._worker: dict[str, Any] | None = None
        self._gen = 0
        super().__init__(receipts=receipts, check_timeout=ct, rpc_timeout=rpc_timeout, weight_store=weight_store, recovery_driver=recovery_driver, auto_spawn=auto_spawn)

    @classmethod
    def launch_environment(cls, args: list[str]) -> dict[str, str]:
        """Parse `--real` CLI flags into `RSIBOX_ISAACLAB_*` env vars without eager GPU imports."""
        p = argparse.ArgumentParser(prog="rsibox conformance isaac-lab --real", allow_abbrev=False)
        for flag, default, typ in (("--mode", "isaaclab", str), ("--task", "Isaac-Cartpole-Direct", str), ("--physics", "newton_mjwarp", str), ("--num-envs", 4096, int), ("--horizon-steps", 10, int), ("--seed", 42, int), ("--device", "cuda:0", str), ("--check-timeout", 30.0, float)):
            p.add_argument(flag, default=default, type=typ)
        ns = p.parse_args(list(args[1:] if args[:1] == ["--"] else args))
        return {f"RSIBOX_ISAACLAB_{k.upper()}": str(getattr(ns, k)) for k in ("mode", "task", "physics", "num_envs", "horizon_steps", "seed", "device", "check_timeout")}

    def default_workload(self) -> dict[str, Any]:
        return {"num_envs": int(self.config["num_envs"]), "horizon_steps": int(self.config["horizon_steps"])}

    def _livelab_paths(self) -> tuple[str, str]:
        ll = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return ll, os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(ll, "..", "..", "..", "core", "livelab"))

    def _start_engine(self) -> tuple[Any, list[str]]:
        if str(self.config.get("mode", "isaaclab")) != "cpu":
            require_isaac_lab_deps(str(self.config.get("device", "cuda:0")))
        if self.weight_store is not None and getattr(self.weight_store, "name", "none") != "none":
            if not (probe := self.weight_store.probe(tp=1, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))).ok:
                raise RuntimeError(f"weight store ({self.weight_store.name}) not ready: {probe.error}")
        self._gen += 1
        ll, core = self._livelab_paths()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        existing = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join([p for p in (ll, core) if p not in existing] + existing)
        sock, wcfg = ipc.short_sock_path("rsiisaac"), {**self.config, "modules": list(self._init_modules), "parent_pid": os.getpid()}
        spec_path = err_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", prefix="rsiisaac-", suffix=".json", delete=False, encoding="utf-8") as sf:
                json.dump({"sock_path": sock, "config": wcfg}, sf)
                spec_path = sf.name
            with tempfile.NamedTemporaryFile(mode="w", prefix="rsiisaac-", suffix=".log", delete=False, encoding="utf-8") as ef:
                err_path = ef.name
                p = subprocess.Popen([sys.executable, "-m", "adapter.worker", spec_path], env=env, stdout=subprocess.DEVNULL, stderr=ef, start_new_session=True)
            self._worker = {"proc": p, "sock": sock, "spec": spec_path, "err": err_path}
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if p.poll() is not None:
                    tail = Path(err_path).read_text(encoding="utf-8", errors="replace").strip()[-500:] if os.path.exists(err_path) else ""
                    raise RuntimeError(f"isaac-lab worker exited (rc={p.returncode}): {tail}")
                if os.path.exists(sock):
                    with contextlib.suppress(OSError, ValueError):
                        if ipc.unix_json_call(sock, {"method": "ping"}, timeout=5.0).get("ok"):
                            break
                time.sleep(0.05)
            else:
                raise TimeoutError(f"isaac-lab worker not ready in {self.startup_timeout} s")
            return RecaptureCoordinator([coordinator.CallParticipant("sim0", lambda m, pay: self._call(m, pay))], gap=lambda fn: fn(), writer=self.writer, interface_version="1-draft", engine={"name": self.name, "version": str(self.config.get("physics", "newton_mjwarp")), "commit": None}, phase2_deadline_s=self.rpc_timeout, post_commit_reset="state"), list(self._init_modules)
        except BaseException:
            for pth in (spec_path, err_path):
                if pth:
                    with contextlib.suppress(OSError):
                        os.unlink(pth)
            self._stop_engine()
            raise

    def _stop_engine(self) -> None:
        self._gen += 1
        w, self._worker = self._worker, None
        if not w:
            return
        p = w["proc"]
        if p.poll() is None and not self._dirty:
            with contextlib.suppress(Exception):
                ipc.unix_json_call(w["sock"], {"method": "shutdown"}, timeout=3.0)
                p.wait(timeout=2.0)
        if p.poll() is None:
            for fn in (lambda: proc.stop_session(p.pid, grace=3.0, reap=p.poll), lambda: proc.stop_tree(p.pid, grace_s=2.0, include_pg=True)):
                with contextlib.suppress(Exception):
                    fn()
        with contextlib.suppress(Exception):
            p.wait(timeout=2.0)
        for pth in (w["sock"], w["spec"], w["err"]):
            with contextlib.suppress(OSError):
                os.unlink(pth)

    def _call(self, method: str, payload: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        gen = self._gen
        if self._dirty or not self._worker:
            raise ConnectionError("worker channel dirty; respawn required")
        try:
            return ipc.unix_json_call(self._worker["sock"], {"method": method, "payload": payload}, timeout=self.rpc_timeout if timeout is None else timeout)
        except (TimeoutError, ConnectionError, OSError, ValueError) as exc:
            if self._gen == gen:
                self._dirty = True
            raise ConnectionError(f"sim0 {method}: {type(exc).__name__}: {exc}") from exc

    def _probe_participants(self) -> tuple[bool, str | None]:
        try:
            return (True, None) if self._call("ping", timeout=min(5.0, self.rpc_timeout)).get("ok") else (False, "sim0 ping failed")
        except Exception as exc:
            return False, f"sim0: {type(exc).__name__}: {exc}"

    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        rep = self._call("step", {**self.default_workload(), **(workload or {})}, timeout=self.check_timeout + 5.0)
        if not rep.get("ok"):
            raise RuntimeError(str(rep.get("error") or "sim0 step failed"))
        return rep["output"]

    def _reset_state(self) -> dict[str, Any]:
        rep = self._call("reset", "state")
        return {"ok": True, "reset": list(rep.get("result") or rep.get("reset") or ["sim_s0"])} if rep.get("ok") else {"ok": False, "error": rep.get("error"), "reset": []}

    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]):
        plan, findings = super()._build_plan(patch, live_map)
        return {**plan, "undo_resets": list(patch.get("undo_resets", ["state"]))}, findings

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        ll, core = self._livelab_paths()
        try:
            overrides = self.files(patch, against="disk")
        except ValueError as exc:
            return {"ok": False, "output": None, "error": str(exc)}
        spec = {"fresh": True, "config": dict(self.config), "workload": {**self.default_workload(), **(workload or {})}}
        with harness.PackageOverlay(self.package, overrides) as overlay:
            assert overlay.root is not None
            return harness.run_fresh_subprocess("adapter.worker", spec, overlay.root, extra_pythonpath=(ll, core), timeout=self.check_timeout + 120.0)

    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        if fault == "hang":
            kwargs.setdefault("hang_s", self.check_timeout + 1.0)
        elif fault == "tamper":
            kwargs.setdefault("module", DEFAULT_MODULES[0])
            kwargs.setdefault("qualname", "compute_reward")
        with self.lock:
            try:
                return self._call("inject", {"fault": fault, **kwargs})
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def patches(self) -> dict[str, Any]:
        em, wm = DEFAULT_MODULES
        es, ws, e = self._base_source(em), self._base_source(wm), harness.edit_once
        r1 = e(es, "reward_gain = 1.0 * REWARD_SCALE", "reward_gain = 2.0 * REWARD_SCALE")
        r2 = e(r1, "reward_gain = 2.0 * REWARD_SCALE", "reward_gain = 3.0 * REWARD_SCALE")
        p3 = e(ws, "summary_scale = 1.0", "summary_scale = 0.5")
        multi_e = e(r1, "k_pole, k_vel = 1.2, 0.25", "k_pole, k_vel = 1.5, 0.35")
        return {
            "noop": {"files": {em: e(es, "reward_gain = 1.0 * REWARD_SCALE", "reward_gain = (1.0 + 0.0) * REWARD_SCALE")}},
            "visible": {"files": {em: r1}}, "chain": [{"files": {em: r1}}, {"files": {em: r2}}, {"files": {wm: p3}}],
            "multi": {"files": {em: multi_e, wm: p3}}, "unreached": {"files": {em: e(es, "return INIT_ACTION_SCALE", "return INIT_ACTION_SCALE * 2.0")}},
            "refused": {"syntax": {"files": {em: e(es, "reward_gain = 1.0 * REWARD_SCALE", "reward_gain = (")}}, "closure": {"files": {em: e(es, "max(floor_w, base_weight) if step < 2", "max(floor_w, base_weight * decay) if step < 2")}}, "module_state": {"files": {em: e(es, "INIT_ACTION_SCALE, REWARD_SCALE = 1.0, 1.0", "INIT_ACTION_SCALE, REWARD_SCALE = 1.0, 2.0")}}, "empty": {"files": {em: e(es, "def policy_step(obs: Any, action_scale: float = 1.0) -> Any:", "def policy_step(obs: Any, action_scale: float = 1.0) -> Any:  # pd")}}},
        }
