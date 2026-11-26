"""VllmOmniLab: RSIBox live-lab host for multi-stage multimodal vLLM-Omni pipelines."""
from __future__ import annotations

import argparse, contextlib, json, os, subprocess, sys, tempfile, threading, time
from typing import Any

from kernel import coordinator, findings as F
from rsibox import harness, ipc, proc
from rsibox.lab import BaseLab, RecaptureCoordinator

DEFAULT_MODULES = ("adapter.thinker_op", "adapter.talker_op", "adapter.dit_op", "adapter.vocoder_op", "adapter.stage_ops", "adapter.worker")
STAGE_ROLES = ("thinker", "talker", "dit", "vocoder")
MODULE_OWNER = {
    "adapter.thinker_op": "thinker", "adapter.talker_op": "talker",
    "adapter.dit_op": "dit", "adapter.vocoder_op": "vocoder",
    "adapter.stage_ops": "talker", "adapter.worker": "talker",
}


def _env_config() -> dict[str, Any]:
    g = os.environ.get
    mp = g("RSIBOX_VLLM_OMNI_MODEL", "").strip()
    return {
        "mode": g("RSIBOX_VLLM_OMNI_MODE", "real"), "model_path": os.path.abspath(mp) if mp else "",
        "hidden_dim": int(g("RSIBOX_VLLM_OMNI_HIDDEN_DIM", "8")), "dit_steps": int(g("RSIBOX_VLLM_OMNI_DIT_STEPS", "2")),
        "max_new_tokens": int(g("RSIBOX_VLLM_OMNI_MAX_NEW_TOKENS", "8")), "image_size": int(g("RSIBOX_VLLM_OMNI_IMAGE_SIZE", "64")),
        "use_aux_cache": g("RSIBOX_VLLM_OMNI_USE_AUX_CACHE", "1").strip().lower() not in ("0", "false", "no"), "seed": int(g("RSIBOX_VLLM_OMNI_SEED", "0")),
    }


class VllmOmniLab(BaseLab):
    """RSIBox live lab managing 4 multimodal stage participants with RO stage caching & S0 reset."""

    name, package, threaded_check = "vllm-omni", "adapter", True

    def __init__(self, config: dict[str, Any] | None = None, modules: tuple[str, ...] | list[str] | None = None, *, receipts: str | None = None, rpc_timeout: float = 30.0, check_timeout: float | None = None, startup_timeout: float | None = None, weight_store: Any = None, recovery_driver: Any = None, auto_spawn: bool = True) -> None:
        self.config = json.loads(json.dumps(config)) if config is not None else _env_config()
        self.config.setdefault("mode", "real")
        if mp := str(self.config.get("model_path") or "").strip():
            self.config["model_path"] = os.path.abspath(mp)
        self._init_modules = list(modules if modules is not None else DEFAULT_MODULES)
        self.startup_timeout = float(startup_timeout if startup_timeout is not None else os.environ.get("RSIBOX_VLLM_OMNI_STARTUP_TIMEOUT", "180.0"))
        ct = float(check_timeout if check_timeout is not None else os.environ.get("RSIBOX_VLLM_OMNI_CHECK_TIMEOUT", "90.0"))
        self._proc: subprocess.Popen[bytes] | None = None; self._sock: str | None = None; self._spec: str | None = None; self._err: str | None = None
        self._gen, self._lost_stages, self._lost_lock = 0, set(), threading.Lock()
        super().__init__(receipts=receipts, check_timeout=ct, rpc_timeout=rpc_timeout, weight_store=weight_store, recovery_driver=recovery_driver, auto_spawn=auto_spawn)

    @classmethod
    def launch_environment(cls, args: list[str]) -> dict[str, str]:
        """Translate `rsibox conformance vllm-omni --real ...` flags into env vars (zero torch imports)."""
        ap = argparse.ArgumentParser(prog="rsibox conformance vllm-omni --real", allow_abbrev=False)
        for flag, default, typ in (("--mode", "real", str), ("--model", "", str), ("--hidden-dim", 8, int), ("--dit-steps", 2, int), ("--max-new-tokens", 8, int), ("--image-size", 64, int), ("--check-timeout", 90.0, float), ("--startup-timeout", 180.0, float)):
            ap.add_argument(flag, default=default, type=typ)
        ap.add_argument("--no-aux-cache", action="store_true")
        ns = ap.parse_args(list(args[1:] if args[:1] == ["--"] else args))
        env = {f"RSIBOX_VLLM_OMNI_{k.upper()}": str(getattr(ns, k)) for k in ("mode", "hidden_dim", "dit_steps", "max_new_tokens", "image_size", "check_timeout", "startup_timeout")}
        env["RSIBOX_VLLM_OMNI_USE_AUX_CACHE"] = "0" if ns.no_aux_cache else "1"
        if ns.model.strip():
            env["RSIBOX_VLLM_OMNI_MODEL"] = os.path.abspath(ns.model.strip())
        return env

    def default_workload(self) -> dict[str, Any]:
        return {"prompt": "Describe color briefly.", "image_size": int(self.config.get("image_size", 64)), "max_new_tokens": int(self.config.get("max_new_tokens", 8)), "color": [128, 64, 32]}

    def _livelab_paths(self) -> tuple[str, str]:
        ll = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return ll, os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(ll, "..", "..", "..", "core", "livelab"))

    def _start_engine(self) -> tuple[Any, list[str]]:
        self._gen += 1
        with self._lost_lock:
            self._lost_stages.clear()
        ll, core = self._livelab_paths()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}
        existing = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join([p for p in (ll, core) if p not in existing] + existing)
        try:
            self._sock = ipc.short_sock_path("rsivlo")
            with tempfile.NamedTemporaryFile(mode="w", prefix="rsivlo-spec-", suffix=".json", delete=False, encoding="utf-8") as sf:
                json.dump({"sock_path": self._sock, "config": {**self.config, "modules": list(self._init_modules), "parent_pid": os.getpid()}}, sf)
                self._spec = sf.name
            with tempfile.NamedTemporaryFile(mode="w", prefix="rsivlo-err-", suffix=".log", delete=False, encoding="utf-8") as ef:
                self._err = ef.name
                self._proc = subprocess.Popen([sys.executable, "-c", "import sys; from adapter.worker import main; raise SystemExit(main(sys.argv[1:]))", self._spec], env=env, stdout=subprocess.DEVNULL, stderr=ef, start_new_session=True)
            self._wait_ready()
            parts = [coordinator.CallParticipant(role, lambda m, pay, r=role: self._stage_call(r, m, pay)) for role in STAGE_ROLES]
            return RecaptureCoordinator(parts, gap=lambda fn: fn(), writer=self.writer, interface_version="1-draft", engine={"name": self.name, "version": str(self.config.get("mode", "real")), "commit": None}, phase2_deadline_s=self.rpc_timeout, post_commit_reset="state"), list(self._init_modules)
        except BaseException:
            self._stop_engine(); raise

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                tail = ""
                if self._err and os.path.exists(self._err):
                    with open(self._err, encoding="utf-8", errors="replace") as ef:
                        tail = ef.read().strip()[-500:]
                raise RuntimeError(f"vllm-omni worker exited (rc={self._proc.returncode}): {tail}")
            if self._sock and os.path.exists(self._sock):
                with contextlib.suppress(OSError, ValueError):
                    if ipc.unix_json_call(self._sock, {"method": "ping"}, timeout=5.0).get("ok"):
                        return
            time.sleep(0.05)
        raise TimeoutError(f"vllm-omni worker not ready within {self.startup_timeout} s")

    def _stop_engine(self) -> None:
        self._gen += 1; p, self._proc = self._proc, None
        if p is not None:
            with contextlib.suppress(Exception):
                if p.poll() is None and not self._dirty and self._sock:
                    with contextlib.suppress(Exception):
                        ipc.unix_json_call(self._sock, {"method": "shutdown"}, timeout=2.0); p.wait(timeout=2.0)
                proc.stop_session(p.pid, grace=3.0, reap=p.poll)
                proc.stop_tree(p.pid, grace_s=2.0, include_pg=True)
                p.wait(timeout=2.0)
        for attr in ("_sock", "_spec", "_err"):
            if path := getattr(self, attr, None):
                with contextlib.suppress(OSError):
                    os.unlink(path)
            setattr(self, attr, None)

    def _stage_call(self, stage: str, method: str, payload: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        gen = self._gen
        with self._lost_lock:
            lost = stage in self._lost_stages
        if self._dirty or not self._sock or lost:
            self._dirty = True
            raise ConnectionError(f"{stage} {method}: worker unavailable or lost")
        alive = (lambda pr=self._proc: pr.poll() is None) if self._proc is not None else None
        try:
            return ipc.unix_json_call(self._sock, {"stage": stage, "method": method, "payload": payload}, timeout=self.rpc_timeout if timeout is None else timeout, alive_fn=alive)
        except (TimeoutError, ConnectionError, OSError, ValueError) as exc:
            if self._gen == gen:
                self._dirty = True
            raise ConnectionError(f"{stage} {method}: {type(exc).__name__}: {exc}") from exc

    def _probe_participants(self) -> tuple[bool, str | None]:
        for role in STAGE_ROLES:
            try:
                if not self._stage_call(role, "ping", timeout=min(5.0, self.rpc_timeout)).get("ok"):
                    return False, f"{role} ping failed"
            except Exception as exc:
                return False, f"{role}: {type(exc).__name__}: {exc}"
        return True, None

    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        rep = self._stage_call("talker", "run_pipeline", {**self.default_workload(), **(workload or {})}, timeout=self.check_timeout + 5.0)
        if not rep.get("ok"):
            raise RuntimeError(str(rep.get("error") or "run_pipeline failed"))
        return rep

    def _format_output(self, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return raw["output"], {"telemetry": raw.get("telemetry", {})}

    def _reset_state(self) -> dict[str, Any]:
        rep = self._stage_call("talker", "reset", "state")
        return {"ok": True, "reset": list(rep.get("result") or ["stage_buffers_s0", "rng"])} if rep.get("ok") else {"ok": False, "error": rep.get("error"), "reset": []}

    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        plan, plan_findings = super()._build_plan(patch, live_map)
        requested = list(patch["stages"]) if patch.get("stages") is not None else None
        for mod, spec in plan["modules"].items():
            owner = MODULE_OWNER.get(mod, "talker")
            if requested is not None and owner not in requested:
                return None, [F.finding("patch_invalid", f"{mod} belongs to stage {owner!r}, not {requested}", f"stages containing {owner!r}")]
            spec["on"] = [owner]
        plan["undo_resets"] = list(patch.get("undo_resets", ["state"]))
        return plan, plan_findings

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        ll, core = self._livelab_paths()
        try:
            overrides = self.files(patch, against="disk")
        except ValueError as exc:
            return {"ok": False, "output": None, "error": str(exc)}
        spec, env = {"fresh": True, "config": dict(self.config), "workload": {**self.default_workload(), **(workload or {})}}, {**os.environ, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}
        with harness.PackageOverlay(self.package, overrides) as overlay:
            assert overlay.root is not None
            return harness.run_fresh_subprocess("adapter.worker", spec, overlay.root, extra_pythonpath=(ll, core), timeout=self.check_timeout + 180.0, env=env)

    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        stage = str(kwargs.pop("participant", "talker"))
        if fault == "lost":
            with self._lost_lock:
                self._lost_stages.add(stage)
            return {"ok": True, "fault": fault, "participant": stage}
        if fault == "hang":
            kwargs.setdefault("hang_s", self.check_timeout + 1.5)
        elif fault == "tamper":
            kwargs.setdefault("module", "adapter.talker_op"); kwargs.setdefault("qualname", "talker_stage")
        with self.lock:
            try:
                return self._stage_call(stage, "inject", {"fault": fault, **kwargs})
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def patches(self) -> dict[str, Any]:
        t_mod, s_mod, w_mod = "adapter.talker_op", "adapter.stage_ops", "adapter.worker"
        t_src, s_src, w_src, e = self._base_source(t_mod), self._base_source(s_mod), self._base_source(w_mod), harness.edit_once
        v1 = e(t_src, "    prosody_gain = 1.0", "    prosody_gain = 1.5")
        v2 = e(v1, "    prosody_gain = 1.5", "    prosody_gain = 2.0")
        w_v3 = e(w_src, "    wave_rounded = [round(float(v), 6) for v in waveform]", "    wave_rounded = [round(float(v) * 2.0, 6) for v in waveform]")
        multi_t = e(v1, "* gain / (1.0 + 0.1 * mag)", "* gain * 1.1 / (1.0 + 0.1 * mag)")
        return {
            "noop": {"files": {t_mod: e(t_src, "    prosody_gain = 1.0", "    prosody_gain = 1.0 + 0.0")}},
            "visible": {"files": {t_mod: v1}},
            "chain": [{"files": {t_mod: v1}}, {"files": {t_mod: v2}}, {"files": {w_mod: w_v3}}],
            "multi": {"files": {t_mod: multi_t, w_mod: w_v3}},
            "unreached": {"files": {s_mod: e(s_src, "return INIT_STAGE_SCALE", "return INIT_STAGE_SCALE * 2.0")}},
            "refused": {
                "syntax": {"files": {t_mod: e(t_src, "    prosody_gain = 1.0", "    prosody_gain = (")}},
                "closure": {"files": {t_mod: e(t_src, "talker_prosody(val, prosody_gain)", "talker_prosody(val, prosody_gain + cfg.hidden_dim)")}},
                "module_state": {"files": {s_mod: e(s_src, "INIT_STAGE_SCALE = 0.25", "INIT_STAGE_SCALE = 0.5")}},
                "empty": {"files": {t_mod: e(t_src, "def talker_stage(thinker_cond: Any, cfg: Any) -> Any:", "def talker_stage(thinker_cond: Any, cfg: Any) -> Any:  # prosody")}},
            },
        }
