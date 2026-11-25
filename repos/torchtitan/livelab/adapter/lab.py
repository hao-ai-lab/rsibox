"""TorchTitanLab & ToyTorchTitanLab: RSIBox live labs for PyTorch Native FSDP2 / TorchTitan."""
from __future__ import annotations

import argparse, contextlib, json, math, os, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Any

from adapter.worker import aggregate_ranks, validate_world_size
from kernel import coordinator
from rsibox import harness, ipc, proc
from rsibox.lab import BaseLab, RecaptureCoordinator

DEFAULT_MODULES = ("adapter.model_ops", "adapter.optim_ops")


def _env_config() -> dict[str, Any]:
    g = os.environ.get
    cfg: dict[str, Any] = {"mode": g("RSIBOX_TORCHTITAN_MODE", "fsdp2"), "world_size": int(g("RSIBOX_TORCHTITAN_WORLD_SIZE", "1")), "hidden": int(g("RSIBOX_TORCHTITAN_HIDDEN", "64")), "ffn": int(g("RSIBOX_TORCHTITAN_FFN", "128")), "layers": int(g("RSIBOX_TORCHTITAN_LAYERS", "2")), "batch_size": int(g("RSIBOX_TORCHTITAN_BATCH_SIZE", "8")), "steps": int(g("RSIBOX_TORCHTITAN_STEPS", "2")), "seed": int(g("RSIBOX_TORCHTITAN_SEED", "0"))}
    ckpt = (g("RSIBOX_TORCHTITAN_CHECKPOINT_DIR") or "").strip()
    if ckpt:
        cfg["checkpoint_dir"] = os.path.abspath(ckpt)
    return cfg


class TorchTitanLab(BaseLab):
    """RSIBox live lab over multi-rank CPU toy or single/multi-rank PyTorch FSDP2 training workers."""

    name, package, threaded_check = "torchtitan", "adapter", True

    def __init__(self, config: dict[str, Any] | None = None, modules: tuple[str, ...] | list[str] | None = None, *, receipts: str | None = None, rpc_timeout: float = 30.0, check_timeout: float | None = None, startup_timeout: float = 90.0, weight_store: Any = None, recovery_driver: Any = None, auto_spawn: bool = True) -> None:
        self.config = {"mode": "fsdp2", "world_size": 1, **(json.loads(json.dumps(config)) if config is not None else _env_config())}
        self.config["world_size"] = validate_world_size(self.config["world_size"])
        ckpt = self.config.get("checkpoint_dir")
        if isinstance(ckpt, str) and ckpt.strip():
            self.config["checkpoint_dir"] = os.path.abspath(ckpt.strip())
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
        p.add_argument("--checkpoint-dir", default="", type=str, help="Checkpoint directory to mount as S0 at startup")
        a = p.parse_args(list(args[1:] if args[:1] == ["--"] else args))
        env = {f"RSIBOX_TORCHTITAN_{k.upper()}": str(getattr(a, k)) for k in ("mode", "world_size", "hidden", "ffn", "layers", "batch_size", "steps", "check_timeout")}
        if a.checkpoint_dir.strip():
            env["RSIBOX_TORCHTITAN_CHECKPOINT_DIR"] = os.path.abspath(a.checkpoint_dir.strip())
        return env

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
        workers, self._workers = list(getattr(self, "_workers", ())), []
        if not workers:
            return
        was_dirty = bool(getattr(self, "_dirty", False))

        def _stop_one(w: dict[str, Any]) -> None:
            p = w.get("proc")
            try:
                if p is not None:
                    with contextlib.suppress(Exception):
                        if p.poll() is None and not was_dirty and w.get("sock"):
                            with contextlib.suppress(Exception):
                                ipc.unix_json_call(w["sock"], {"method": "shutdown"}, timeout=2.0)
                                p.wait(timeout=1.0)
                        if p.poll() is None:
                            for fn in (
                                lambda: proc.stop_session(p.pid, grace=3.0, reap=p.poll),
                                lambda: proc.stop_tree(p.pid, grace_s=2.0, include_pg=True),
                            ):
                                with contextlib.suppress(Exception):
                                    fn()
                        with contextlib.suppress(Exception):
                            p.wait(timeout=2.0)
            finally:
                for key in ("sock", "spec", "err"):
                    path = w.get(key)
                    if path:
                        with contextlib.suppress(OSError):
                            os.unlink(path)

        if len(workers) == 1:
            _stop_one(workers[0])
            return
        started: list[threading.Thread] = []
        for w in workers:
            thread = threading.Thread(target=_stop_one, args=(w,), daemon=True)
            try:
                thread.start()
                started.append(thread)
            except Exception:
                _stop_one(w)
        for thread in started:
            thread.join()

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

    def _fanout_rank_call(self, method: str, payload: Any, *, timeout: float | None = None) -> dict[int, dict[str, Any]]:
        if len(self._workers) <= 1:
            return {w["rank"]: self._rank_call(w["rank"], method, payload, timeout=timeout) for w in self._workers}
        with ThreadPoolExecutor(max_workers=len(self._workers)) as pool:
            futs = {pool.submit(self._rank_call, w["rank"], method, payload, timeout=timeout): w["rank"] for w in self._workers}
            return {rank: fut.result() for fut, rank in futs.items()}

    def _fault_mount(self, message: str) -> None:
        self._dirty = True
        if self.session is not None:
            self.session.fault(
                "participant_failed",
                message,
                "all ranks mount identical S0 checkpoint",
                subject="mount_checkpoint_as_s0",
            )

    def save_checkpoint(self, checkpoint_dir: str) -> dict[str, Any]:
        """Atomically save every rank's live weights, optimizer state, step, and RNG to `checkpoint_dir`."""
        if not isinstance(checkpoint_dir, str) or not checkpoint_dir.strip():
            return {"ok": False, "error": "checkpoint_dir must be a non-empty path string"}
        target_dir = os.path.abspath(checkpoint_dir.strip())
        with self.lock:
            if not self._workers or self._dirty:
                return {"ok": False, "error": "torchtitan workers are not healthy/running"}
            if self.session is not None and self.session.poisoned():
                return {"ok": False, "error": "torchtitan session is poisoned; respawn required"}
            if self.session is not None and getattr(self.session, "stack", None):
                return {"ok": False, "error": "uninstall active live edits before saving a checkpoint"}
            if os.path.exists(target_dir) and not os.path.isdir(target_dir):
                return {"ok": False, "error": f"checkpoint_dir is not a directory: {target_dir}"}
            t0 = time.perf_counter()
            try:
                os.makedirs(target_dir, exist_ok=True)
                stage_dir = tempfile.mkdtemp(prefix=".rsitt-stage-", dir=target_dir)
            except OSError as exc:
                return {"ok": False, "error": f"failed to prepare checkpoint_dir {target_dir}: {exc}"}
            try:
                try:
                    replies = self._fanout_rank_call("save_checkpoint", {"checkpoint_dir": stage_dir})
                except Exception as exc:
                    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                bad = [f"rank{r}: {rep.get('error')}" for r, rep in sorted(replies.items()) if not rep.get("ok")]
                if bad:
                    return {"ok": False, "error": "; ".join(bad)}
                steps = {int(rep["step"]) for rep in replies.values()}
                if len(steps) != 1:
                    return {"ok": False, "error": f"rank step mismatch while saving checkpoint: {sorted(steps)}"}
                step = next(iter(steps))
                sample_cfg = next(iter(replies.values())).get("cfg") or {}
                manifest = {
                    "format": "rsibox-torchtitan-ckpt-v1",
                    "mode": str(self.config.get("mode", "fsdp2")),
                    "world_size": len(self._workers),
                    "step": step,
                    "hidden": int(sample_cfg.get("hidden", self.config.get("hidden", 16))),
                    "ffn": int(sample_cfg.get("ffn", self.config.get("ffn", 32))),
                    "layers": int(sample_cfg.get("layers", self.config.get("layers", 2))),
                    "lr": float(sample_cfg.get("lr", self.config.get("lr", 0.01))),
                    "ranks": sorted(replies.keys()),
                }
                ext = "json" if str(self.config.get("mode", "fsdp2")) == "cpu" else "pt"
                try:
                    stage_manifest = os.path.join(stage_dir, "manifest.json")
                    ipc.atomic_write_json(stage_manifest, manifest)
                    for rank in sorted(replies.keys()):
                        fname = f"rank{rank}.{ext}"
                        os.replace(os.path.join(stage_dir, fname), os.path.join(target_dir, fname))
                    os.replace(stage_manifest, os.path.join(target_dir, "manifest.json"))
                except OSError as exc:
                    return {"ok": False, "error": f"failed to promote checkpoint files into {target_dir}: {exc}"}
                return {
                    "ok": True,
                    "checkpoint_dir": target_dir,
                    "step": step,
                    "ranks": len(self._workers),
                    "cost_s": round(time.perf_counter() - t0, 6),
                }
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)

    def mount_checkpoint_as_s0(self, checkpoint_dir: str) -> dict[str, Any]:
        """Mount a saved mid-run checkpoint directory as this live lab's deterministic `S0` anchor."""
        if not isinstance(checkpoint_dir, str) or not checkpoint_dir.strip():
            return {"ok": False, "error": "checkpoint_dir must be a non-empty path string"}
        target_dir = os.path.abspath(checkpoint_dir.strip())
        with self.lock:
            if not self._workers or self._dirty:
                return {"ok": False, "error": "torchtitan workers are not healthy/running"}
            if self.session is not None and self.session.poisoned():
                return {"ok": False, "error": "torchtitan session is poisoned; respawn required"}
            if self.session is not None and getattr(self.session, "stack", None):
                return {"ok": False, "error": "uninstall active live edits before mounting a new S0 checkpoint"}
            if not os.path.isdir(target_dir):
                return {"ok": False, "error": f"checkpoint_dir is not a directory: {target_dir}"}
            manifest_path = os.path.join(target_dir, "manifest.json")
            if not os.path.isfile(manifest_path):
                return {"ok": False, "error": f"missing checkpoint manifest: {manifest_path}"}
            try:
                manifest = ipc.read_json(manifest_path)
            except (OSError, ValueError) as exc:
                return {"ok": False, "error": f"failed to read checkpoint manifest {manifest_path}: {exc}"}
            if not isinstance(manifest, dict) or manifest.get("format") != "rsibox-torchtitan-ckpt-v1":
                return {"ok": False, "error": f"invalid checkpoint manifest in {manifest_path}"}
            expected_mode = str(self.config.get("mode", "fsdp2"))
            expected_ranks = [w["rank"] for w in self._workers]
            if manifest.get("world_size") != len(self._workers) or manifest.get("ranks") != expected_ranks:
                return {
                    "ok": False,
                    "error": f"checkpoint topology mismatch (world_size={manifest.get('world_size')}, ranks={manifest.get('ranks')} vs expected {len(self._workers)}/{expected_ranks})",
                }
            if manifest.get("mode") != expected_mode:
                return {
                    "ok": False,
                    "error": f"checkpoint mode mismatch ({manifest.get('mode')!r} != {expected_mode!r})",
                }
            manifest_step = manifest.get("step")
            if isinstance(manifest_step, bool) or not isinstance(manifest_step, int) or manifest_step < 0:
                return {"ok": False, "error": f"invalid manifest step: {manifest_step!r}"}
            expected_hidden = int(self.config.get("hidden", 16))
            expected_ffn = int(self.config.get("ffn", 32))
            expected_layers = int(self.config.get("layers", 2))
            expected_lr = float(self.config.get("lr", 0.01))
            if (
                manifest.get("hidden") != expected_hidden
                or manifest.get("ffn") != expected_ffn
                or manifest.get("layers") != expected_layers
            ):
                return {
                    "ok": False,
                    "error": (
                        f"checkpoint manifest dimensions mismatch "
                        f"({manifest.get('hidden')},{manifest.get('ffn')},{manifest.get('layers')} "
                        f"!= {expected_hidden},{expected_ffn},{expected_layers})"
                    ),
                }
            manifest_lr = manifest.get("lr")
            if (
                isinstance(manifest_lr, bool)
                or not isinstance(manifest_lr, (int, float))
                or not math.isfinite(float(manifest_lr))
                or not math.isclose(float(manifest_lr), expected_lr, rel_tol=1e-9, abs_tol=1e-12)
            ):
                return {"ok": False, "error": f"checkpoint manifest lr mismatch ({manifest_lr!r} != {expected_lr!r})"}
            # Pre-check every rank file exists before fanning out RPCs.
            ext = "json" if expected_mode == "cpu" else "pt"
            missing = [
                f"rank{w['rank']}.{ext}"
                for w in self._workers
                if not os.path.isfile(os.path.join(target_dir, f"rank{w['rank']}.{ext}"))
            ]
            if missing:
                return {"ok": False, "error": f"missing per-rank checkpoint file(s) in {target_dir}: {missing}"}
            t0 = time.perf_counter()
            # Phase 1: zero-mutation validation across every rank before any rank updates S0.
            try:
                val_replies = self._fanout_rank_call("validate_checkpoint", {"checkpoint_dir": target_dir})
            except Exception as exc:
                self._fault_mount(f"checkpoint validation RPC error: {type(exc).__name__}: {exc}")
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            val_bad = [f"rank{r}: {rep.get('error')}" for r, rep in sorted(val_replies.items()) if not rep.get("ok")]
            if val_bad:
                return {"ok": False, "error": "; ".join(val_bad)}
            val_steps = {int(rep["s0_step"]) for rep in val_replies.values()}
            if len(val_steps) != 1 or next(iter(val_steps)) != manifest_step:
                return {
                    "ok": False,
                    "error": f"checkpoint step mismatch across ranks/manifest: ranks={sorted(val_steps)} manifest={manifest_step}",
                }
            # Phase 2: transactional mount across every rank.
            try:
                replies = self._fanout_rank_call("mount_checkpoint", {"checkpoint_dir": target_dir})
            except Exception as exc:
                self._fault_mount(f"checkpoint mount RPC error: {type(exc).__name__}: {exc}")
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            bad = [f"rank{r}: {rep.get('error')}" for r, rep in sorted(replies.items()) if not rep.get("ok")]
            if bad:
                if any(rep.get("ok") for rep in replies.values()):
                    self._fault_mount(f"partial checkpoint mount failure: {'; '.join(bad)}")
                return {"ok": False, "error": "; ".join(bad)}
            s0_steps = {int(rep["s0_step"]) for rep in replies.values()}
            if len(s0_steps) != 1 or next(iter(s0_steps)) != manifest_step:
                self._fault_mount(f"inconsistent s0_step across ranks/manifest: ranks={sorted(s0_steps)} manifest={manifest_step}")
                return {"ok": False, "error": f"inconsistent s0_step across ranks: {sorted(s0_steps)}"}
            s0_step = next(iter(s0_steps))
            self.config["checkpoint_dir"] = target_dir
            self.last_check = None
            return {
                "ok": True,
                "checkpoint_dir": target_dir,
                "s0_step": s0_step,
                "ranks": len(self._workers),
                "cost_s": round(time.perf_counter() - t0, 6),
            }

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
