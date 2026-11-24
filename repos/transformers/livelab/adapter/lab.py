"""TransformersLab: live-lab host over `rsibox.lab.BaseLab` and a warm worker subprocess."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from adapter.fixture.patches import ARGMAX_ANCHOR, DEFAULT_REAL_MODULES, LOGITS_RETURN_ANCHOR, real_patches
from kernel import coordinator
from rsibox import harness, ipc, proc
from rsibox.lab import BaseLab, RecaptureCoordinator

__all__ = ["ARGMAX_ANCHOR", "DEFAULT_REAL_MODULES", "LOGITS_RETURN_ANCHOR", "TransformersLab"]


def _env_config() -> dict[str, Any]:
    model_path = os.environ.get("RSIBOX_TRANSFORMERS_MODEL")
    if not model_path:
        raise RuntimeError("set RSIBOX_TRANSFORMERS_MODEL to a model path or repo id")
    max_tokens_env = os.environ.get("RSIBOX_TRANSFORMERS_MAX_NEW_TOKENS")
    return {
        "mode": "real", "model_path": model_path, "device": os.environ.get("RSIBOX_TRANSFORMERS_DEVICE", ""),
        "dtype": os.environ.get("RSIBOX_TRANSFORMERS_DTYPE", "bfloat16"),
        "seed": int(os.environ.get("RSIBOX_TRANSFORMERS_SEED", "0")),
        **({"max_new_tokens": int(max_tokens_env)} if max_tokens_env else {}),
    }


class TransformersLab(BaseLab):
    """RSIBox live lab managing a persistent warm HuggingFace Transformers worker."""

    name, package, fresh_module, threaded_check = "transformers", "transformers", "adapter.fixture.fresh", True
    dirty = property(lambda self: self._dirty, lambda self, value: setattr(self, "_dirty", bool(value)))

    def __init__(
        self, config: dict[str, Any] | None = None, modules: tuple[str, ...] | list[str] | None = None, *,
        receipts: str | None = None, rpc_timeout: float = 30.0, check_timeout: float = 120.0,
        startup_timeout: float = 180.0, quiesce_timeout: float | None = None,
        weight_store: Any = None, recovery_driver: Any = None, auto_spawn: bool = True,
    ) -> None:
        self.config = json.loads(json.dumps(config)) if config is not None else _env_config()
        self.config.setdefault("mode", "real")
        self._init_modules = list(modules if modules is not None else DEFAULT_REAL_MODULES)
        self.startup_timeout = float(startup_timeout)
        env_quiesce = os.environ.get("RSIBOX_TRANSFORMERS_QUIESCE_TIMEOUT")
        self.quiesce_timeout = float(quiesce_timeout if quiesce_timeout is not None else (env_quiesce or 120.0))
        self.sock_path, self._spec_path, self._stderr_path = None, None, None
        self._worker_proc, self._worker_generation, self.model_module = None, 0, None
        super().__init__(
            receipts=receipts, check_timeout=check_timeout, rpc_timeout=rpc_timeout,
            auto_spawn=auto_spawn, weight_store=weight_store, recovery_driver=recovery_driver,
        )

    @classmethod
    def launch_environment(cls, args: list[str]) -> dict[str, str]:
        arguments = list(args[1:] if args[:1] == ["--"] else args)
        parser = argparse.ArgumentParser(prog="rsibox conformance transformers --real", allow_abbrev=False)
        parser.add_argument("--model", required=True, help="HuggingFace model path or repo id")
        parser.add_argument("--device", default="", help="Target device (e.g. cuda:0, cpu)")
        parser.add_argument("--dtype", default="bfloat16", help="Model dtype (bfloat16, float16, float32, auto)")
        parser.add_argument("--max-new-tokens", type=int, default=16, help="Tokens per generation check")
        parser.add_argument("--quiesce-timeout", type=float, default=None, help="Quiesce/resume RPC timeout (s)")
        parsed = parser.parse_args(arguments)
        env = {
            "RSIBOX_TRANSFORMERS_MODEL": parsed.model, "RSIBOX_TRANSFORMERS_DTYPE": parsed.dtype,
            "RSIBOX_TRANSFORMERS_MAX_NEW_TOKENS": str(parsed.max_new_tokens),
        }
        if parsed.device:
            env["RSIBOX_TRANSFORMERS_DEVICE"] = parsed.device
        if parsed.quiesce_timeout is not None:
            env["RSIBOX_TRANSFORMERS_QUIESCE_TIMEOUT"] = str(parsed.quiesce_timeout)
        return env

    def is_running(self) -> bool:
        return self._worker_proc is not None and self._worker_proc.poll() is None and self.sock_path is not None

    def _engine_version(self) -> str:
        with contextlib.suppress(Exception):
            return importlib.metadata.version("transformers")
        return "0"

    def default_workload(self) -> dict[str, Any]:
        return {
            "prompts": ["Write one sentence about lighthouses.", "Write one sentence about rivers."],
            "max_new_tokens": int(self.config.get("max_new_tokens", 16)), "temperature": 0.0,
        }

    def default_warmup_workload(self) -> dict[str, Any]:
        return {"prompts": ["Warmup."], "max_new_tokens": 2, "temperature": 0.0}

    def _livelab_paths(self) -> tuple[str, str]:
        livelab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_dir = os.path.abspath(
            os.environ.get("RSIBOX_LIVELAB") or os.path.join(livelab_dir, "..", "..", "..", "core", "livelab")
        )
        return livelab_dir, core_dir

    def _start_engine(self) -> tuple[Any, list[str]]:
        if self.weight_store is not None and getattr(self.weight_store, "name", "none") != "none":
            probe_result = self.weight_store.probe(tp=1, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
            if not probe_result.ok:
                raise RuntimeError(f"weight store ({self.weight_store.name}) not ready: {probe_result.error}")
        self._worker_generation += 1
        stderr_handle = None
        try:
            self.sock_path = ipc.short_sock_path("rsitf")
            worker_config = {**self.config, "modules": list(self._init_modules), "parent_pid": os.getpid()}
            with tempfile.NamedTemporaryFile(
                mode="w", prefix="rsitf-spec-", suffix=".json", delete=False, encoding="utf-8"
            ) as spec_file:
                json.dump({"sock_path": self.sock_path, "config": worker_config}, spec_file)
                self._spec_path = spec_file.name
            stderr_handle = tempfile.NamedTemporaryFile(
                mode="w", prefix="rsitf-err-", suffix=".log", delete=False, encoding="utf-8"
            )
            self._stderr_path = stderr_handle.name
            livelab_dir, core_dir = self._livelab_paths()
            worker_env = dict(os.environ)
            existing_pythonpath = [part for part in (worker_env.get("PYTHONPATH") or "").split(os.pathsep) if part]
            py_entries = [part for part in (livelab_dir, core_dir) if part not in existing_pythonpath]
            worker_env["PYTHONPATH"] = os.pathsep.join(py_entries + existing_pythonpath)
            worker_env.setdefault("PYTHONUNBUFFERED", "1")
            self._worker_proc = subprocess.Popen(
                [sys.executable, "-m", "adapter.agent", self._spec_path],
                env=worker_env, stdout=subprocess.DEVNULL, stderr=stderr_handle, start_new_session=True,
            )
            stderr_handle.close()
            stderr_handle = None
            handshake = self._wait_for_worker()
            self.model_module = handshake.get("model_module")
            tracked = list(dict.fromkeys([*self._init_modules, *([self.model_module] if self.model_module else [])]))
            participant = coordinator.CallParticipant("worker0", lambda method, payload: self.worker_call(method, payload))
            step_coordinator = RecaptureCoordinator(
                [participant], gap=lambda action: action(), writer=self.writer, interface_version="1-draft",
                engine={"name": self.name, "version": self._engine_version(), "commit": None},
                phase2_deadline_s=self.rpc_timeout, post_commit_reset="state",
            )
            return step_coordinator, tracked
        except BaseException:
            if stderr_handle is not None:
                with contextlib.suppress(OSError):
                    stderr_handle.close()
            self._stop_engine()
            raise

    def _read_stderr_tail(self) -> str:
        with contextlib.suppress(OSError, TypeError):
            with open(str(self._stderr_path), encoding="utf-8", errors="replace") as handle:
                return handle.read().strip()[-500:]
        return ""

    def _wait_for_worker(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._worker_proc is not None and self._worker_proc.poll() is not None:
                exit_code = self._worker_proc.returncode
                raise RuntimeError(f"transformers worker exited during startup (rc={exit_code}): {self._read_stderr_tail()}")
            if self.sock_path and os.path.exists(self.sock_path):
                with contextlib.suppress(OSError, ValueError):
                    reply = ipc.unix_json_call(self.sock_path, {"method": "ping"}, timeout=5.0)
                    if reply.get("ok"):
                        return reply
            time.sleep(0.05)
        raise TimeoutError(f"transformers worker not ready within {self.startup_timeout} s: {self._read_stderr_tail()}")

    def _stop_engine(self) -> None:
        worker, self._worker_proc = self._worker_proc, None
        if worker is not None:
            if worker.poll() is None and self.sock_path and not self.dirty:
                with contextlib.suppress(Exception):
                    ipc.unix_json_call(self.sock_path, {"method": "shutdown"}, timeout=2.0)
            if worker.poll() is None:
                with contextlib.suppress(Exception):
                    proc.stop_session(worker.pid, grace=5.0, reap=worker.poll)
                with contextlib.suppress(Exception):
                    proc.stop_tree(worker.pid, grace_s=2.0, include_pg=True)
            with contextlib.suppress(Exception):
                worker.wait(timeout=2.0)
        for cleanup_path in (self.sock_path, self._spec_path, self._stderr_path):
            if cleanup_path:
                with contextlib.suppress(OSError):
                    os.unlink(cleanup_path)
        self.sock_path, self._spec_path, self._stderr_path = None, None, None

    def worker_call(self, method: str, payload: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        generation, sock_path = self._worker_generation, self.sock_path
        if self.dirty or not sock_path:
            raise ConnectionError("worker channel dirty; respawn required" if self.dirty else "worker is not running")
        try:
            call_timeout = self.rpc_timeout if timeout is None else float(timeout)
            return ipc.unix_json_call(sock_path, {"method": method, "payload": payload}, timeout=call_timeout)
        except (TimeoutError, ConnectionError, OSError, ValueError) as error:
            if self._worker_generation == generation:
                self.dirty = True
            raise ConnectionError(f"worker0 {method}: {type(error).__name__}: {error}") from error

    def _probe_participants(self) -> tuple[bool, str | None]:
        try:
            reply = self.worker_call("ping", timeout=min(5.0, self.rpc_timeout))
            return (True, None) if reply.get("ok") else (False, str(reply.get("error") or "worker0 ping failed"))
        except Exception as error:
            return False, f"worker0: {type(error).__name__}: {error}"

    def run_generate(self, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        reply = self.worker_call("generate", {**self.default_workload(), **(workload or {})}, timeout=self.check_timeout + 5.0)
        if not reply.get("ok"):
            raise RuntimeError(str(reply.get("error") or "generate failed"))
        return reply["output"]

    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        generation = self._worker_generation
        try:
            return self.run_generate(workload)
        except BaseException:
            if self._worker_generation == generation:
                self.dirty = True
            raise

    def _reset_state(self) -> dict[str, Any]:
        reply = self.worker_call("reset", "state")
        if not reply.get("ok"):
            return {"ok": False, "error": reply.get("error"), "reset": []}
        return {"ok": True, "reset": list(reply.get("result") or reply.get("reset") or ["kv_cache", "rng"])}

    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        plan, plan_findings = super()._build_plan(patch, live_map)
        plan["undo_resets"] = list(patch.get("undo_resets", ["state"]))
        return plan, plan_findings

    def quiesce_worker(self, *, offload_to_cpu: bool = True, pin_memory: bool = True) -> dict[str, Any]:
        with self.lock:
            payload = {"offload_to_cpu": bool(offload_to_cpu), "pin_memory": bool(pin_memory)}
            return self.worker_call("quiesce", payload, timeout=self.quiesce_timeout)

    def resume_worker(self, *, released: tuple[str, ...] = ()) -> dict[str, Any]:
        with self.lock:
            return self.worker_call("resume", {"released": list(released)}, timeout=self.quiesce_timeout)

    def quiesce_hooks(
        self, *, offload_to_cpu: bool = True, pin_memory: bool = True, warmup_workload: dict[str, Any] | None = None
    ) -> Any:
        from adapter.snapshot_hooks import TransformersQuiesceHooks
        return TransformersQuiesceHooks(
            self, offload_to_cpu=offload_to_cpu, pin_memory=pin_memory, warmup_workload=warmup_workload
        )

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        livelab_dir, core_dir = self._livelab_paths()
        try:
            overrides = self.files(patch, against="disk")
        except ValueError as error:
            return {"ok": False, "output": None, "error": str(error)}
        spec = {"config": dict(self.config), "workload": {**self.default_workload(), **(workload or {})}}
        with harness.PackageOverlay(self.package, overrides) as overlay:
            return harness.run_fresh_subprocess(
                self.fresh_module, spec, overlay.root,
                extra_pythonpath=(livelab_dir, core_dir), timeout=self.check_timeout + 300.0,
            )

    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        if fault == "hang":
            kwargs.setdefault("hang_s", self.check_timeout + 2.0)
        if fault == "tamper":
            kwargs.setdefault("module", self._init_modules[0])
            kwargs.setdefault("qualname", "LogitsProcessorList.__call__")
        with self.lock:
            try:
                return self.worker_call("inject", {"fault": fault, **kwargs})
            except Exception as error:
                return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    def patches(self) -> dict[str, Any]:
        return real_patches(self._base_source)
