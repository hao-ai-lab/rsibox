"""TorchTitanLab & ToyTorchTitanLab: RSIBox live labs for PyTorch Native FSDP2 / TorchTitan."""
from __future__ import annotations

import argparse, contextlib, json, os, subprocess, sys, tempfile, time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Any

from adapter.worker import FSDP2_SINGLE_RANK_ERR, aggregate_ranks, validate_world_size
from kernel import coordinator
from rsibox import harness, ipc, proc
from rsibox.lab import BaseLab, RecaptureCoordinator

DEFAULT_MODULES = ("adapter.model_ops", "adapter.optim_ops")


def _env_config() -> dict[str, Any]:
    g = os.environ.get
    return {"mode": g("RSIBOX_TORCHTITAN_MODE", "fsdp2"), "world_size": int(g("RSIBOX_TORCHTITAN_WORLD_SIZE", "1")), "hidden": int(g("RSIBOX_TORCHTITAN_HIDDEN", "64")), "ffn": int(g("RSIBOX_TORCHTITAN_FFN", "128")), "layers": int(g("RSIBOX_TORCHTITAN_LAYERS", "2")), "batch_size": int(g("RSIBOX_TORCHTITAN_BATCH_SIZE", "8")), "steps": int(g("RSIBOX_TORCHTITAN_STEPS", "2")), "seed": int(g("RSIBOX_TORCHTITAN_SEED", "0"))}


class TorchTitanLab(BaseLab):
    """RSIBox live lab over multi-rank CPU toy or single/multi-rank PyTorch FSDP2 training workers."""

    name, package, threaded_check = "torchtitan", "adapter", True

    def __init__(self, config: dict[str, Any] | None = None, modules: tuple[str, ...] | list[str] | None = None, *, receipts: str | None = None, rpc_timeout: float = 30.0, check_timeout: float | None = None, startup_timeout: float = 90.0, weight_store: Any = None, recovery_driver: Any = None, auto_spawn: bool = True) -> None:
        self.config = {"mode": "fsdp2", "world_size": 1, **(json.loads(json.dumps(config)) if config is not None else _env_config())}
        self.config["world_size"] = validate_world_size(self.config["world_size"])
        self._init_modules, self.startup_timeout, self._workers, self._gen = list(modules if modules is not None else DEFAULT_MODULES), float(startup_timeout), [], 0
        ct = float(check_timeout if check_timeout is not None else os.environ.get("RSIBOX_TORCHTITAN_CHECK_TIMEOUT", "30.0"))
        super().__init__(receipts=receipts, check_timeout=ct, rpc_timeout=rpc_timeout, weight_store=weight_store, recovery_driver=recovery_driver, auto_spawn=auto_spawn)

    @property
    def ranks(self) -> int:
        return int(self.config.get("world_size", 1))

    @classmethod
    def launch_environment(cls, args: list[str]) -> dict[str, str]:
        p = argparse.ArgumentParser(prog="rsibox conformance torchtitan --real", allow_abbrev=False)
        for flag, default, typ in (("--mode", "fsdp2", str), ("--world-size", 1, int), ("--hidden", 64, int), ("--ffn", 128, int), ("--layers", 2, int), ("--batch-size", 8, int), ("--steps", 2, int), ("--check-timeout", 30.0, float)):
            p.add_argument(flag, default=default, type=typ)
        a = p.parse_args(list(args[1:] if args[:1] == ["--"] else args))
        return {f"RSIBOX_TORCHTITAN_{k.upper()}": str(getattr(a, k)) for k in ("mode", "world_size", "hidden", "ffn", "layers", "batch_size", "steps", "check_timeout")}

    def default_workload(self) -> dict[str, Any]:
        return {"steps": int(self.config.get("steps", 2))}

    def _livelab_paths(self) -> tuple[str, str]:
        ll = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return ll, os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(ll, "..", "..", "..", "core", "livelab"))

    def _start_engine(self) -> tuple[Any, list[str]]:
        world_size = int(self.config.get("world_size", 1))
        if self.weight_store is not None and getattr(self.weight_store, "name", "none") != "none":
            probe_result = self.weight_store.probe(tp=world_size, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
            if not probe_result.ok:
                raise RuntimeError(f"weight store ({self.weight_store.name}) not ready: {probe_result.error}")
        self._gen += 1
        port, (ll, core) = proc.free_port(), self._livelab_paths()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}
        existing = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join([p for p in (ll, core) if p not in existing] + existing)
        pending_paths: list[str] = []
        try:
            for rank in range(world_size):
                sock = ipc.short_sock_path(f"rsitt{rank}")
                wcfg = {**self.config, "rank": rank, "local_rank": rank, "world_size": world_size, "master_port": port, "modules": list(self._init_modules), "parent_pid": os.getpid()}
                with tempfile.NamedTemporaryFile(mode="w", prefix=f"rsitt-r{rank}-", suffix=".json", delete=False, encoding="utf-8") as sf:
                    json.dump({"sock_path": sock, "config": wcfg}, sf)
                    spec_path = sf.name
                pending_paths.append(spec_path)
                with tempfile.NamedTemporaryFile(mode="w", prefix=f"rsitt-r{rank}-", suffix=".log", delete=False, encoding="utf-8") as ef:
                    err_path = ef.name
                    pending_paths.append(err_path)
                    p = subprocess.Popen([sys.executable, "-m", "adapter.worker", spec_path], env=env, stdout=subprocess.DEVNULL, stderr=ef, start_new_session=True)
                self._workers.append({"rank": rank, "proc": p, "sock": sock, "spec": spec_path, "err": err_path})
                pending_paths.clear()
            self._wait_for_workers()
            parts = [coordinator.CallParticipant(f"rank{w['rank']}", lambda m, pay, r=w["rank"]: self._rank_call(r, m, pay)) for w in self._workers]
            return RecaptureCoordinator(parts, gap=lambda fn: fn(), writer=self.writer, interface_version="1-draft", engine={"name": self.name, "version": str(self.config.get("mode", "fsdp2")), "commit": None}, phase2_deadline_s=self.rpc_timeout, post_commit_reset="state"), list(self._init_modules)
        except BaseException:
            for path in pending_paths:
                with contextlib.suppress(OSError):
                    os.unlink(path)
            self._stop_engine()
            raise

    def _wait_for_workers(self) -> None:
        deadline, ready = time.monotonic() + self.startup_timeout, set()
        while len(ready) < len(self._workers) and time.monotonic() < deadline:
            for w in self._workers:
                if w["rank"] in ready:
                    continue
                if w["proc"].poll() is not None:
                    tail = ""
                    if os.path.exists(w["err"]):
                        with open(w["err"], encoding="utf-8", errors="replace") as ef:
                            tail = ef.read().strip()[-500:]
                    raise RuntimeError(f"torchtitan rank{w['rank']} exited (rc={w['proc'].returncode}): {tail}")
                if os.path.exists(w["sock"]):
                    with contextlib.suppress(OSError, ValueError):
                        if ipc.unix_json_call(w["sock"], {"method": "ping"}, timeout=5.0).get("ok"):
                            ready.add(w["rank"])
            if len(ready) < len(self._workers):
                time.sleep(0.03)
        if len(ready) < len(self._workers):
            raise TimeoutError(f"torchtitan workers not ready in {self.startup_timeout} s")

    def _stop_engine(self) -> None:
        self._gen += 1
        workers, self._workers = self._workers, []
        for w in workers:
            p = w["proc"]
            if p.poll() is None and not self._dirty:
                with contextlib.suppress(Exception):
                    ipc.unix_json_call(w["sock"], {"method": "shutdown"}, timeout=2.0)
                    p.wait(timeout=1.0)
            if p.poll() is None:
                for fn in (lambda: proc.stop_session(p.pid, grace=3.0, reap=p.poll), lambda: proc.stop_tree(p.pid, grace_s=2.0, include_pg=True)):
                    with contextlib.suppress(Exception):
                        fn()
            with contextlib.suppress(Exception):
                p.wait(timeout=2.0)
            for path in (w["sock"], w["spec"], w["err"]):
                with contextlib.suppress(OSError):
                    os.unlink(path)

    def _rank_call(self, rank: int, method: str, payload: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        gen = self._gen
        if self._dirty or rank < 0 or rank >= len(self._workers):
            raise ConnectionError("worker channel dirty; respawn required")
        try:
            return ipc.unix_json_call(self._workers[rank]["sock"], {"method": method, "payload": payload}, timeout=self.rpc_timeout if timeout is None else timeout)
        except (TimeoutError, ConnectionError, OSError, ValueError) as exc:
            if self._gen == gen:
                self._dirty = True
            raise ConnectionError(f"rank{rank} {method}: {type(exc).__name__}: {exc}") from exc

    def _probe_participants(self) -> tuple[bool, str | None]:
        for w in self._workers:
            try:
                if not self._rank_call(w["rank"], "ping", timeout=min(5.0, self.rpc_timeout)).get("ok"):
                    return False, f"rank{w['rank']} ping failed"
            except Exception as exc:
                return False, f"rank{w['rank']}: {type(exc).__name__}: {exc}"
        return True, None

    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        gen, wl, pool = self._gen, {**self.default_workload(), **(workload or {})}, ThreadPoolExecutor(max_workers=max(1, len(self._workers)))
        futs = {pool.submit(self._rank_call, w["rank"], "step", wl, timeout=self.check_timeout + 5.0): w["rank"] for w in self._workers}
        pending, per_rank = set(futs), {}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_EXCEPTION)
                for fut in done:
                    if (exc := fut.exception()) is not None:
                        raise exc
                    rep = fut.result()
                    if not rep.get("ok"):
                        raise RuntimeError(str(rep.get("error") or f"rank{futs[fut]} step failed"))
                    per_rank[f"rank{futs[fut]}"] = rep["output"]
        except BaseException:
            if self._gen == gen:
                self._dirty = True
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return aggregate_ranks(per_rank)

    def _reset_state(self) -> dict[str, Any]:
        items: list[str] = []
        for w in self._workers:
            rep = self._rank_call(w["rank"], "reset", "state")
            if not rep.get("ok"):
                return {"ok": False, "error": rep.get("error"), "reset": []}
            items = list(rep.get("result") or rep.get("reset") or ["weights_s0", "optimizer_state", "rng"])
        return {"ok": True, "reset": items}

    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]):
        plan, findings = super()._build_plan(patch, live_map)
        return {**plan, "undo_resets": list(patch.get("undo_resets", ["state"]))}, findings

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        ll, core = self._livelab_paths()
        try:
            overrides = self.files(patch, against="disk")
        except ValueError as exc:
            return {"ok": False, "output": None, "error": str(exc)}
        outer_timeout = self.check_timeout + 120.0
        inner_timeout = max(5.0, outer_timeout - 15.0)
        spec, env = {
            "fresh": True,
            "config": {**self.config, "master_port": proc.free_port()},
            "workload": {**self.default_workload(), **(workload or {})},
            "timeout": inner_timeout,
        }, {**os.environ, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}
        with harness.PackageOverlay(self.package, overrides) as overlay:
            return harness.run_fresh_subprocess("adapter.worker", spec, overlay.root, extra_pythonpath=(ll, core), timeout=outer_timeout, env=env)

    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        rank = int(kwargs.pop("rank", 0))
        if fault == "hang":
            kwargs.setdefault("hang_s", self.check_timeout + 1.0)
        elif fault == "tamper":
            kwargs.setdefault("module", DEFAULT_MODULES[0])
            kwargs.setdefault("qualname", "compute_loss")
        with self.lock:
            try:
                return self._rank_call(rank, "inject", {"fault": fault, **kwargs})
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def patches(self) -> dict[str, Any]:
        model_module, optim_module = DEFAULT_MODULES
        m_src, o_src, e = self._base_source(model_module), self._base_source(optim_module), harness.edit_once
        loss_v1 = e(e(m_src, "return (diff * diff).mean()", "return (diff * diff).mean() * 2.0"), "return sum(d * d for d in diffs) / max(len(diffs), 1)", "return (sum(d * d for d in diffs) / max(len(diffs), 1)) * 2.0")
        loss_v2 = e(e(loss_v1, "return (diff * diff).mean() * 2.0", "return (diff * diff).mean() * 3.0"), "return (sum(d * d for d in diffs) / max(len(diffs), 1)) * 2.0", "return (sum(d * d for d in diffs) / max(len(diffs), 1)) * 3.0")
        clip_v3, multi_m = e(o_src, "grad_scale = 1.0", "grad_scale = 0.5"), e(e(loss_v1, "hidden = F.silu(w1(x))", "hidden = F.silu(w1(x)) * 1.1"), "d / (1.0 + math.exp", "1.1 * d / (1.0 + math.exp")
        return {
            "noop": {"files": {model_module: e(e(m_src, "diff = pred - target", "diff = (pred - target) + 0.0"), "(p_val - t_val)", "((p_val - t_val) + 0.0)")}},
            "visible": {"files": {model_module: loss_v1}}, "chain": [{"files": {model_module: loss_v1}}, {"files": {model_module: loss_v2}}, {"files": {optim_module: clip_v3}}],
            "multi": {"files": {model_module: multi_m, optim_module: clip_v3}}, "unreached": {"files": {model_module: e(m_src, "return INIT_WEIGHT_SCALE", "return INIT_WEIGHT_SCALE * 2.0")}},
            "refused": {"syntax": {"files": {model_module: e(m_src, "return (diff * diff).mean()", "return (")}}, "closure": {"files": {optim_module: e(o_src, "max(floor_lr, base_lr)", "max(floor_lr, base_lr * decay)")}}, "module_state": {"files": {optim_module: e(o_src, "GRAD_CLIP_NORM = 1.0", "GRAD_CLIP_NORM = 2.0")}}, "empty": {"files": {model_module: e(m_src, "def compute_loss(pred: Any, target: Any) -> Any:", "def compute_loss(pred: Any, target: Any) -> Any:  # mse")}}},
        }


class ToyTorchTitanLab(TorchTitanLab):
    """Zero-dependency 2-rank CPU conformance lab driving the real `TorchTitanLab` adapter."""

    name = "torchtitan-toy"

    def __init__(self, receipts: str | None = None, **kwargs: Any) -> None:
        kwargs = {"rpc_timeout": 5.0, "check_timeout": 2.0, "startup_timeout": 15.0, **kwargs}
        super().__init__({"mode": "cpu", "world_size": 2, "hidden": 8, "ffn": 16, "layers": 1, "steps": 2, "seed": 0}, DEFAULT_MODULES, receipts=receipts, **kwargs)
