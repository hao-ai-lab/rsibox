"""NemoRlLab & ToyNemoRlLab: RSIBox live labs for NVIDIA NeMo-RL (AutoModel FSDP2 + SGLang GRPO)."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from adapter.worker import (
    build_nemo_rl_plan,
    clean_nemo_rl_witness,
    execute_grpo_episode,
    require_nemo_rl_deps,
    validate_positive_int,
)
from kernel import coordinator
from rsibox import harness, ipc, proc
from rsibox.lab import BaseLab, RecaptureCoordinator

DEFAULT_MODULES = ("adapter.grpo_ops", "adapter.rollout_ops", "adapter.shared_ops")
_ROLE_SPECS = (
    ("coordinator0", "coordinator"),
    ("trainer0", "trainer"),
    ("rollout0", "rollout"),
)


class _WorkerParticipant(coordinator.CallParticipant):
    """`CallParticipant` exposing `loaded(plan)` for `ray_participants.unloaded()`."""

    def loaded(self, plan: dict[str, Any]) -> dict[str, Any]:
        return self._do("loaded", plan)


def _env_config() -> dict[str, Any]:
    g = os.environ.get
    return {
        "mode": g("RSIBOX_NEMORL_MODE", "ray"),
        "model": g("RSIBOX_NEMORL_MODEL", ""),
        "data": g("RSIBOX_NEMORL_DATA", ""),
        "num_trainers": int(g("RSIBOX_NEMORL_NUM_TRAINERS", "1")),
        "num_rollouts": int(g("RSIBOX_NEMORL_NUM_ROLLOUTS", "1")),
        "hidden": int(g("RSIBOX_NEMORL_HIDDEN", "8")),
        "group_size": int(g("RSIBOX_NEMORL_GROUP_SIZE", "4")),
        "steps": int(g("RSIBOX_NEMORL_STEPS", "2")),
        "seed": int(g("RSIBOX_NEMORL_SEED", "0")),
    }


class NemoRlLab(BaseLab):
    """RSIBox live lab over NVIDIA NeMo-RL colocated Ray actors or 3-participant CPU toy workers."""

    name = "nemo-rl"
    package = "adapter"
    threaded_check = True

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        modules: tuple[str, ...] | list[str] | None = None,
        *,
        receipts: str | None = None,
        rpc_timeout: float = 30.0,
        check_timeout: float | None = None,
        startup_timeout: float = 90.0,
        weight_store: Any = None,
        recovery_driver: Any = None,
        runtime_factory: Any = None,
        auto_spawn: bool = True,
    ) -> None:
        merged = {**_env_config(), **(json.loads(json.dumps(config)) if config is not None else {})}
        merged["num_trainers"] = validate_positive_int(merged.get("num_trainers", 1), "num_trainers")
        merged["num_rollouts"] = validate_positive_int(merged.get("num_rollouts", 1), "num_rollouts")
        self.config = merged
        self._init_modules = list(modules if modules is not None else DEFAULT_MODULES)
        self.startup_timeout = float(startup_timeout)
        self._runtime_factory = runtime_factory
        self._runtime: dict[str, Any] = {}
        self._workers: dict[str, dict[str, Any]] = {}
        self._gen = 0
        ct = float(
            check_timeout if check_timeout is not None else os.environ.get("RSIBOX_NEMORL_CHECK_TIMEOUT", "30.0")
        )
        super().__init__(
            receipts=receipts,
            check_timeout=ct,
            rpc_timeout=rpc_timeout,
            weight_store=weight_store,
            recovery_driver=recovery_driver,
            auto_spawn=auto_spawn,
        )

    @property
    def ranks(self) -> int:
        return int(self.config.get("num_trainers", 1))

    @classmethod
    def launch_environment(cls, args: list[str]) -> dict[str, str]:
        """Parse `--real` CLI flags into `RSIBOX_NEMORL_*` env vars without importing GPU deps."""
        p = argparse.ArgumentParser(prog="rsibox conformance nemo-rl --real", allow_abbrev=False)
        for flag, default, typ in (
            ("--mode", "ray", str),
            ("--model", "", str),
            ("--data", "", str),
            ("--num-trainers", 1, int),
            ("--num-rollouts", 1, int),
            ("--hidden", 8, int),
            ("--group-size", 4, int),
            ("--steps", 2, int),
            ("--check-timeout", 30.0, float),
        ):
            p.add_argument(flag, default=default, type=typ)
        ns = p.parse_args(list(args[1:] if args[:1] == ["--"] else args))
        return {
            f"RSIBOX_NEMORL_{k.upper()}": str(getattr(ns, k))
            for k in (
                "mode",
                "model",
                "data",
                "num_trainers",
                "num_rollouts",
                "hidden",
                "group_size",
                "steps",
                "check_timeout",
            )
        }

    def default_workload(self) -> dict[str, Any]:
        return {"steps": int(self.config.get("steps", 2))}

    def _livelab_paths(self) -> tuple[str, str]:
        ll = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return ll, os.path.abspath(
            os.environ.get("RSIBOX_LIVELAB") or os.path.join(ll, "..", "..", "..", "core", "livelab")
        )

    def _start_engine(self) -> tuple[Any, list[str]]:
        if self.weight_store is not None and getattr(self.weight_store, "name", "none") != "none":
            probe_result = self.weight_store.probe(
                tp=self.ranks, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")
            )
            if not probe_result.ok:
                raise RuntimeError(f"weight store ({self.weight_store.name}) not ready: {probe_result.error}")

        if callable(self._runtime_factory):
            try:
                rt = self._runtime = self._runtime_factory()
                self.base_text = dict(rt["base_text"])
                parts = list(rt["participants"])
                co = RecaptureCoordinator(
                    parts,
                    gap=rt.get("gap", lambda fn: fn()),
                    writer=self.writer,
                    interface_version="1-draft",
                    engine={"name": self.name, "version": "nemo-rl", "commit": os.environ.get("RSIBOX_ENGINE_COMMIT")},
                    phase2_deadline_s=self.rpc_timeout,
                    post_commit_reset="state",
                )
                return co, list(rt["modules"])
            except BaseException:
                self._stop_engine()
                raise

        if str(self.config.get("mode", "ray")) != "cpu":
            require_nemo_rl_deps()
            raise RuntimeError(
                "NemoRlLab real mode requires a NeMo-RL Ray cluster runtime_factory or "
                "configured --model/--data GRPO driver (not yet staged on this host)."
            )

        self._gen += 1
        ll, core = self._livelab_paths()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        existing = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join([p for p in (ll, core) if p not in existing] + existing)
        pending_paths: list[str] = []
        try:
            for role_name, role_kind in _ROLE_SPECS:
                sock = ipc.short_sock_path(f"rsinrl_{role_kind[:4]}")
                wcfg = {
                    **self.config,
                    "role_name": role_name,
                    "role_kind": role_kind,
                    "parent_pid": os.getpid(),
                }
                with tempfile.NamedTemporaryFile(
                    mode="w", prefix=f"rsinrl-{role_name}-", suffix=".json", delete=False, encoding="utf-8"
                ) as sf:
                    json.dump({"sock_path": sock, "config": wcfg}, sf)
                    spec_path = sf.name
                pending_paths.append(spec_path)
                with tempfile.NamedTemporaryFile(
                    mode="w", prefix=f"rsinrl-{role_name}-", suffix=".log", delete=False, encoding="utf-8"
                ) as ef:
                    err_path = ef.name
                    pending_paths.append(err_path)
                    p = subprocess.Popen(
                        [sys.executable, "-m", "adapter.worker", spec_path],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=ef,
                        start_new_session=True,
                    )
                self._workers[role_name] = {
                    "role_name": role_name,
                    "role_kind": role_kind,
                    "proc": p,
                    "sock": sock,
                    "spec": spec_path,
                    "err": err_path,
                }
                pending_paths.clear()
            self._wait_for_workers()
            # Sync trainer0 S0 weights to rollout0 as initial refit_version=0.
            s0_weights = self._rpc_result("trainer0", "get_weights")
            self._rpc_result("rollout0", "set_s0_weights", s0_weights)
            parts = [
                _WorkerParticipant(
                    role_name, lambda m, pay, rn=role_name: self._worker_call(rn, m, pay)
                )
                for role_name, _ in _ROLE_SPECS
            ]
            co = RecaptureCoordinator(
                parts,
                gap=lambda fn: fn(),
                writer=self.writer,
                interface_version="1-draft",
                engine={"name": self.name, "version": "cpu-3role-grpo", "commit": None},
                phase2_deadline_s=self.rpc_timeout,
                post_commit_reset="state",
            )
            return co, list(self._init_modules)
        except BaseException:
            for path in pending_paths:
                with contextlib.suppress(OSError):
                    os.unlink(path)
            self._stop_engine()
            raise

    def _wait_for_workers(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        ready: set[str] = set()
        while len(ready) < len(self._workers) and time.monotonic() < deadline:
            for role_name, w in self._workers.items():
                if role_name in ready:
                    continue
                if w["proc"].poll() is not None:
                    tail = ""
                    if os.path.exists(w["err"]):
                        with open(w["err"], encoding="utf-8", errors="replace") as ef:
                            tail = ef.read().strip()[-500:]
                    raise RuntimeError(
                        f"nemo-rl {role_name} exited (rc={w['proc'].returncode}): {tail}"
                    )
                if os.path.exists(w["sock"]):
                    with contextlib.suppress(OSError, ValueError):
                        if ipc.unix_json_call(w["sock"], {"method": "ping"}, timeout=5.0).get("ok"):
                            ready.add(role_name)
            if len(ready) < len(self._workers):
                time.sleep(0.03)
        if len(ready) < len(self._workers):
            raise TimeoutError(f"nemo-rl workers not ready in {self.startup_timeout} s")

    def _stop_engine(self) -> None:
        rt, self._runtime = self._runtime, {}
        if callable(shutdown := rt.get("shutdown")):
            with contextlib.suppress(Exception):
                shutdown()
        self._gen += 1
        workers, self._workers = self._workers, {}
        for w in workers.values():
            p = w["proc"]
            if p.poll() is None and not self._dirty:
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
            for path in (w["sock"], w["spec"], w["err"]):
                with contextlib.suppress(OSError):
                    os.unlink(path)

    def _worker_call(
        self, role_name: str, method: str, payload: Any = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        gen = self._gen
        if self._dirty or role_name not in self._workers:
            raise ConnectionError("worker channel dirty; respawn required")
        try:
            return ipc.unix_json_call(
                self._workers[role_name]["sock"],
                {"method": method, "payload": payload},
                timeout=self.rpc_timeout if timeout is None else timeout,
            )
        except (TimeoutError, ConnectionError, OSError, ValueError) as exc:
            if self._gen == gen:
                self._dirty = True
            raise ConnectionError(f"{role_name} {method}: {type(exc).__name__}: {exc}") from exc

    def _rpc_result(self, role_name: str, method: str, payload: Any = None) -> Any:
        rep = self._worker_call(role_name, method, payload, timeout=self.check_timeout + 5.0)
        if not rep.get("ok"):
            raise RuntimeError(str(rep.get("error") or f"{role_name} {method} failed"))
        return rep.get("result")

    def _probe_participants(self) -> tuple[bool, str | None]:
        if callable(probe := self._runtime.get("probe")):
            return probe()
        if self._runtime:
            return super()._probe_participants()
        for role_name in self._workers:
            try:
                if not self._worker_call(role_name, "ping", timeout=min(5.0, self.rpc_timeout)).get("ok"):
                    return False, f"{role_name} ping failed"
            except Exception as exc:
                return False, f"{role_name}: {type(exc).__name__}: {exc}"
        return True, None

    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        if callable(run_step := self._runtime.get("run_step")):
            return run_step(workload)
        wl = {**self.default_workload(), **(workload or {})}
        if not wl.get("continue_state", False):
            rst = self._reset_state()
            if not rst.get("ok"):
                raise RuntimeError(f"S0 reset before workload failed: {rst}")
        return execute_grpo_episode(
            lambda m, p=None: self._rpc_result("rollout0", m, p),
            lambda m, p=None: self._rpc_result("coordinator0", m, p),
            lambda m, p=None: self._rpc_result("trainer0", m, p),
            workload=wl,
        )

    def _reset_state(self) -> dict[str, Any]:
        if callable(rst_fn := self._runtime.get("reset_state")):
            return rst_fn()
        items: list[str] = []
        for role_name in self._workers:
            rep = self._worker_call(role_name, "reset", "state")
            if not rep.get("ok"):
                return {"ok": False, "error": rep.get("error"), "reset": []}
            for item in rep.get("result") or rep.get("reset") or []:
                if item not in items:
                    items.append(item)
        return {"ok": True, "reset": items}

    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]):
        merged_base = {**self.base_text, **(live_map or {})}
        for key in (patch.get("files") or {}):
            if key not in merged_base:
                with contextlib.suppress(ImportError, OSError):
                    merged_base[key] = self._base_source(key)
        return build_nemo_rl_plan(self.co.participants, patch, merged_base)

    def _clean_witness(self, raw_witness: dict[str, dict[str, int]], proof_ok: bool) -> dict[str, dict[str, int]]:
        return clean_nemo_rl_witness(raw_witness, proof_ok)

    def _run_runtime_reset(self, phase: str) -> None:
        if self._runtime and callable(rst_fn := self._runtime.get("reset_state")):
            res = rst_fn()
            if isinstance(res, dict) and not res.get("ok", True):
                raise RuntimeError(f"{phase} S0 reset failed: {res}")

    def _after_commit(self, patch: dict[str, Any], receipt: dict | None) -> None:
        self._run_runtime_reset("post-commit")

    def _after_uninstall(self, undone: list[str]) -> None:
        self._run_runtime_reset("post-uninstall")

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        if callable(custom_fresh := self._runtime.get("fresh_check")):
            return custom_fresh(patch, workload)
        ll, core = self._livelab_paths()
        try:
            overrides = self.files(patch, against="disk")
        except ValueError as exc:
            return {"ok": False, "output": None, "error": str(exc)}
        outer_timeout = self.check_timeout + 60.0
        spec = {
            "fresh": True,
            "config": dict(self.config),
            "workload": {**self.default_workload(), **(workload or {})},
            "timeout": max(5.0, outer_timeout - 10.0),
        }
        with harness.PackageOverlay(self.package, overrides) as overlay:
            return harness.run_fresh_subprocess(
                "adapter.worker", spec, overlay.root, extra_pythonpath=(ll, core), timeout=outer_timeout
            )

    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        if callable(custom_inj := self._runtime.get("inject")):
            return custom_inj(fault, **kwargs)
        participant = str(kwargs.pop("participant", "trainer0"))
        if participant not in self._workers:
            return {"ok": False, "error": f"unknown participant {participant!r}"}
        if fault == "hang":
            kwargs.setdefault("hang_s", self.check_timeout + 1.0)
        elif fault == "tamper":
            kwargs.setdefault(
                "module",
                "adapter.rollout_ops" if participant == "rollout0" else (
                    "adapter.shared_ops" if participant == "coordinator0" else "adapter.grpo_ops"
                ),
            )
            kwargs.setdefault(
                "qualname",
                "generate_sglang_group" if participant == "rollout0" else (
                    "score_prompt_reward" if participant == "coordinator0" else "compute_grpo_loss"
                ),
            )
        with self.lock:
            try:
                return self._worker_call(participant, "inject", {"fault": fault, **kwargs})
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def patches(self) -> dict[str, Any]:
        if callable(custom_patches := self._runtime.get("patches")):
            return custom_patches()
        g_mod, r_mod, s_mod = DEFAULT_MODULES
        g_src = self._base_source(g_mod)
        r_src = self._base_source(r_mod)
        s_src = self._base_source(s_mod)
        e = harness.edit_once
        vis1 = e(g_src, "total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE", "total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE * 2.0")
        vis2 = e(vis1, "total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE * 2.0", "total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE * 3.0")
        roll_v3 = e(r_src, "0.05 * step_idx", "0.25 * step_idx")
        multi_g = e(vis1, "grad_scale = 1.0", "grad_scale = 0.5")
        multi_s = e(s_src, "0.1 * float(signal)", "0.4 * float(signal)")
        return {
            "noop": {"files": {g_mod: e(g_src, "kl_diff = float(ref_lp) - float(lp)", "kl_diff = (float(ref_lp) - float(lp)) + 0.0")}},
            "visible": {"files": {g_mod: vis1}},
            "chain": [
                {"files": {g_mod: vis1}},
                {"files": {g_mod: vis2}},
                {"files": {r_mod: roll_v3}},
            ],
            "multi": {"files": {g_mod: multi_g, s_mod: multi_s}},
            "unreached": {"files": {g_mod: e(g_src, "return INIT_POLICY_SCALE", "return INIT_POLICY_SCALE * 2.0")}},
            "refused": {
                "syntax": {"files": {g_mod: e(g_src, "return total_loss / n, grads", "return (")}},
                "closure": {"files": {g_mod: e(g_src, "max(floor_kl, base_kl)", "max(floor_kl, base_kl * decay)")}},
                "module_state": {"files": {g_mod: e(g_src, "GRPO_LOSS_SCALE = 1.0", "GRPO_LOSS_SCALE = 2.0")}},
                "empty": {"files": {g_mod: e(g_src, "def compute_grpo_loss(", "def compute_grpo_loss(  # note\n")}},
            },
        }


class ToyNemoRlLab(NemoRlLab):
    """Zero-dependency 3-participant CPU conformance lab (coordinator0 + trainer0 + rollout0)."""

    name = "nemo-rl-toy"

    def __init__(self, receipts: str | None = None, **kwargs: Any) -> None:
        kwargs = {"rpc_timeout": 5.0, "check_timeout": 2.0, "startup_timeout": 15.0, **kwargs}
        super().__init__(
            {"mode": "cpu", "num_trainers": 1, "num_rollouts": 1, "hidden": 8, "group_size": 4, "steps": 2, "seed": 0},
            DEFAULT_MODULES,
            receipts=receipts,
            **kwargs,
        )
