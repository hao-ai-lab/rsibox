"""KernelLab & ToyKernelLab: RSIBox live labs for standalone Triton `@triton.jit` and CUDA `.cu` operators."""
from __future__ import annotations

import argparse, contextlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any

from adapter.worker import require_kernel_deps, resolve_backend, split_kernel_output, validate_positive_int
from kernel import coordinator
from rsibox import harness, ipc, proc
from rsibox.lab import BaseLab, RecaptureCoordinator

CPU_MODULES = ("adapter.kernel_ops", "adapter.worker")
GPU_MODULES = ("adapter.kernel_ops", "adapter.triton_kernels", "adapter.worker")
_PATCH_SPECS = {
    "cpu": (CPU_MODULES[0], "kernel_gain = 1.0 * KERNEL_SCALE", "kernel_gain = 2.0 * KERNEL_SCALE", "kernel_gain = 3.0 * KERNEL_SCALE", "kernel_gain = (1.0 + 0.0) * KERNEL_SCALE", "bias_gain = 1.0", "bias_gain = 1.5"),
    "cuda": (CPU_MODULES[0], "cu_gain = 1.0", "cu_gain = 2.0", "cu_gain = 3.0", "cu_gain = 1.0 + 0.0", "scale = 1.0", "scale = 1.1"),
    "triton": ("adapter.triton_kernels", "gain = 1.0", "gain = 2.0", "gain = 3.0", "gain = 1.0 + 0.0", "float(beta)", "float(beta) + 0.05"),
}


def _env_config() -> dict[str, Any]:
    g = os.environ.get
    return {"backend": g("RSIBOX_KERNEL_BACKEND", "auto"), "numel": int(g("RSIBOX_KERNEL_NUMEL", "4096")), "block_size": int(g("RSIBOX_KERNEL_BLOCK_SIZE", "128")), "seed": int(g("RSIBOX_KERNEL_SEED", "42")), "iters": int(g("RSIBOX_KERNEL_ITERS", "5")), "device": g("RSIBOX_KERNEL_DEVICE", "cuda:0")}


def _enter_critical_region(hook: Any):
    if hook is None: return contextlib.nullcontext()
    if callable(getattr(hook, "critical_region", None)): return hook.critical_region()
    if hasattr(hook, "__enter__") and hasattr(hook, "__exit__"): return hook
    return hook() if callable(hook) else contextlib.nullcontext()


class KernelLab(BaseLab):
    """RSIBox live lab over resident Triton `@triton.jit` / CUDA `.cu` / CPU kernel workers."""

    name, package, threaded_check = "kernel", "adapter", True

    def __init__(self, config: dict[str, Any] | None = None, modules: tuple[str, ...] | list[str] | None = None, *, receipts: str | None = None, rpc_timeout: float = 30.0, check_timeout: float | None = None, startup_timeout: float | None = None, weight_store: Any = None, recovery_driver: Any = None, kernel_service: Any = None, critical_region: Any = None, auto_spawn: bool = True) -> None:
        self.config = {**_env_config(), **(json.loads(json.dumps(config)) if config is not None else {})}
        for key in ("numel", "block_size", "iters"):
            self.config[key] = validate_positive_int(self.config.get(key, 5 if key == "iters" else 128), key)
        self._resolved_backend = resolve_backend(str(self.config.get("backend", "auto")))
        is_cpu = self._resolved_backend == "cpu"
        self._init_modules = list(modules if modules is not None else (GPU_MODULES if self._resolved_backend == "triton" else CPU_MODULES))
        self.startup_timeout = float(startup_timeout if startup_timeout is not None else (15.0 if is_cpu else 120.0))
        ct = float(check_timeout if check_timeout is not None else os.environ.get("RSIBOX_KERNEL_CHECK_TIMEOUT", "2.0" if is_cpu else "30.0"))
        self.kernel_service, self.critical_region, self._worker, self._gen = kernel_service, critical_region, None, 0
        super().__init__(receipts=receipts, check_timeout=ct, rpc_timeout=rpc_timeout, weight_store=weight_store, recovery_driver=recovery_driver, auto_spawn=auto_spawn)

    @classmethod
    def launch_environment(cls, args: list[str]) -> dict[str, str]:
        """Parse `--real` CLI flags into `RSIBOX_KERNEL_*` env vars without eager GPU imports."""
        p = argparse.ArgumentParser(prog="rsibox conformance kernel --real", allow_abbrev=False)
        for flag, default, typ in (("--backend", "auto", str), ("--numel", 4096, int), ("--block-size", 128, int), ("--seed", 42, int), ("--iters", 5, int), ("--device", "cuda:0", str), ("--check-timeout", 30.0, float)):
            p.add_argument(flag, default=default, type=typ)
        ns = p.parse_args(list(args[1:] if args[:1] == ["--"] else args))
        return {f"RSIBOX_KERNEL_{k.upper()}": str(getattr(ns, k)) for k in ("backend", "numel", "block_size", "seed", "iters", "device", "check_timeout")}

    def default_workload(self) -> dict[str, Any]:
        return {k: int(self.config[k]) for k in ("numel", "block_size", "iters")}

    def _livelab_paths(self) -> tuple[str, str]:
        ll = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return ll, os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(ll, "..", "..", "..", "core", "livelab"))

    def _start_engine(self) -> tuple[Any, list[str]]:
        if str(self.config.get("backend", "auto")) != "cpu":
            require_kernel_deps(str(self.config.get("backend", "auto")), str(self.config.get("device", "cuda:0")))
        self._gen += 1
        ll, core = self._livelab_paths()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([ll, core, *(p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p)]))
        sock, wcfg, spec_path, err_path = ipc.short_sock_path("rsikern"), {**self.config, "modules": list(self._init_modules), "parent_pid": os.getpid()}, "", ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", prefix="rsikern-", suffix=".json", delete=False, encoding="utf-8") as sf: json.dump({"sock_path": sock, "config": wcfg}, sf); spec_path = sf.name
            with tempfile.NamedTemporaryFile(mode="w", prefix="rsikern-", suffix=".log", delete=False, encoding="utf-8") as ef: err_path = ef.name; p = subprocess.Popen([sys.executable, "-m", "adapter.worker", spec_path], env=env, stdout=subprocess.DEVNULL, stderr=ef, start_new_session=True)
            self._worker, deadline = {"proc": p, "sock": sock, "spec": spec_path, "err": err_path}, time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if p.poll() is not None: raise RuntimeError(f"kernel worker exited (rc={p.returncode}): {Path(err_path).read_text(encoding='utf-8', errors='replace').strip()[-500:] if os.path.exists(err_path) else ''}")
                if os.path.exists(sock):
                    with contextlib.suppress(OSError, ValueError):
                        if ipc.unix_json_call(sock, {"method": "ping"}, timeout=5.0).get("ok"): break
                time.sleep(0.05)
            else: raise TimeoutError(f"kernel worker not ready in {self.startup_timeout} s")
            def _gap(fn):
                with _enter_critical_region(self.critical_region or self.kernel_service): return fn()
            return RecaptureCoordinator([coordinator.CallParticipant("kernel0", lambda m, pay: self._call(m, pay))], gap=_gap, writer=self.writer, interface_version="1-draft", engine={"name": self.name, "version": str(self.config.get("backend", "auto")), "commit": None}, phase2_deadline_s=self.rpc_timeout, post_commit_reset="state"), list(self._init_modules)
        except BaseException:
            for pth in (spec_path, err_path):
                with contextlib.suppress(OSError):
                    if pth: os.unlink(pth)
            self._stop_engine()
            raise

    def _stop_engine(self) -> None:
        self._gen += 1
        w, self._worker = self._worker, None
        if not w: return
        p = w["proc"]
        if p.poll() is None and not self._dirty:
            with contextlib.suppress(Exception): ipc.unix_json_call(w["sock"], {"method": "shutdown"}, timeout=3.0); p.wait(timeout=2.0)
        if p.poll() is None:
            for fn in (lambda: proc.stop_session(p.pid, grace=3.0, reap=p.poll), lambda: proc.stop_tree(p.pid, grace_s=2.0, include_pg=True)):
                with contextlib.suppress(Exception): fn()
        with contextlib.suppress(Exception): p.wait(timeout=2.0)
        for pth in (w["sock"], w["spec"], w["err"]):
            with contextlib.suppress(OSError): os.unlink(pth)

    def _call(self, method: str, payload: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        gen = self._gen
        if self._dirty or not self._worker: raise ConnectionError("worker channel dirty; respawn required")
        try: return ipc.unix_json_call(self._worker["sock"], {"method": method, "payload": payload}, timeout=self.rpc_timeout if timeout is None else timeout)
        except (TimeoutError, ConnectionError, OSError, ValueError) as exc:
            if self._gen == gen: self._dirty = True
            raise ConnectionError(f"kernel0 {method}: {type(exc).__name__}: {exc}") from exc

    def _probe_participants(self) -> tuple[bool, str | None]:
        try: return (True, None) if self._call("ping", timeout=min(5.0, self.rpc_timeout)).get("ok") else (False, "kernel0 ping failed")
        except Exception as exc: return False, f"kernel0: {type(exc).__name__}: {exc}"

    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        with _enter_critical_region(self.critical_region or self.kernel_service):
            rep = self._call("step", {**self.default_workload(), **(workload or {})}, timeout=self.check_timeout + 5.0)
        if not rep.get("ok"):
            if rep.get("poison_worker"): self._dirty = True
            raise RuntimeError(str(rep.get("error") or "kernel0 step failed"))
        return rep["output"]

    def _format_output(self, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]: return split_kernel_output(raw)
    def _reset_state(self) -> dict[str, Any]:
        rep = self._call("reset", "state")
        return {"ok": True, "reset": list(rep.get("result") or rep.get("reset") or ["kernel_s0"])} if rep.get("ok") else {"ok": False, "error": rep.get("error"), "reset": []}
    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]):
        plan, findings = super()._build_plan(patch, live_map)
        return {**plan, "undo_resets": list(patch.get("undo_resets", ["state"]))}, findings

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        ll, core = self._livelab_paths()
        try: overrides = self.files(patch, against="disk")
        except ValueError as exc: return {"ok": False, "output": None, "error": str(exc)}
        spec = {"fresh": True, "config": dict(self.config), "workload": {**self.default_workload(), **(workload or {})}}
        with harness.PackageOverlay(self.package, overrides) as overlay:
            assert overlay.root is not None
            return harness.run_fresh_subprocess("adapter.worker", spec, overlay.root, extra_pythonpath=(ll, core), timeout=self.check_timeout + 120.0)

    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        if fault == "hang": kwargs.setdefault("hang_s", self.check_timeout + 1.0)
        elif fault == "tamper":
            kwargs.setdefault("module", CPU_MODULES[0]); kwargs.setdefault("qualname", "run_cpu_kernel" if self._resolved_backend == "cpu" else "cuda_kernel_source")
        with self.lock:
            try: return self._call("inject", {"fault": fault, **kwargs})
            except Exception as exc: return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def patches(self) -> dict[str, Any]:
        km, wm = CPU_MODULES
        ks, ws, e = self._base_source(km), self._base_source(wm), harness.edit_once
        mod, a0, a1, a2, n1, m0, m1 = _PATCH_SPECS.get(self._resolved_backend, _PATCH_SPECS["cuda"])
        src = ks if mod == km else self._base_source(mod)
        r1, r2, w_ed = e(src, a0, a1), e(e(src, a0, a1), a1, a2), e(ws, "summary_scale = 1.0", "summary_scale = 0.5")
        return {
            "noop": {"files": {mod: e(src, a0, n1)}}, "visible": {"files": {mod: r1}},
            "chain": [{"files": {mod: r1}}, {"files": {mod: r2}}, {"files": {wm: w_ed}}],
            "multi": {"files": {mod: e(r1, m0, m1), wm: w_ed}},
            "unreached": {"files": {km: e(ks, "return INIT_KERNEL_GAIN", "return INIT_KERNEL_GAIN * 2.0")}},
            "refused": {"syntax": {"files": {km: e(ks, "kernel_gain = 1.0 * KERNEL_SCALE", "kernel_gain = (")}}, "closure": {"files": {km: e(ks, "max(floor_w, base_weight) if step < 2", "max(floor_w, base_weight * decay) if step < 2")}}, "module_state": {"files": {km: e(ks, "INIT_KERNEL_GAIN, KERNEL_SCALE = 1.0, 1.0", "INIT_KERNEL_GAIN, KERNEL_SCALE = 1.0, 2.0")}}, "empty": {"files": {km: e(ks, "def run_cpu_kernel(xs: list[float], bias: list[float], alpha: float = 1.5, beta: float = 0.0) -> list[float]:", "def run_cpu_kernel(xs: list[float], bias: list[float], alpha: float = 1.5, beta: float = 0.0) -> list[float]:  # pd")}}},
        }


class ToyKernelLab(KernelLab):
    """Pure-Python standalone kernel conformance lab (zero eager GPU/Triton imports)."""

    name = "kernel-toy"
    def __init__(self, receipts: str | None = None, **kwargs: Any) -> None:
        super().__init__({"backend": "cpu", "numel": 256, "block_size": 128, "seed": 42, "iters": 3, "device": "cpu"}, CPU_MODULES, receipts=receipts, **{"rpc_timeout": 10.0, "check_timeout": 2.0, "startup_timeout": 15.0, **kwargs})
