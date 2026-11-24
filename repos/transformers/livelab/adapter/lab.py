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

from kernel import coordinator
from rsibox import harness, ipc
from rsibox.lab import BaseLab, RecaptureCoordinator

DEFAULT_REAL_MODULES = ("transformers.generation.logits_process", "transformers.generation.utils")
ARGMAX_ANCHOR = "next_tokens = torch.argmax(next_token_scores, dim=-1)"
LOGITS_RETURN_ANCHOR = "            else:\n                scores = processor(input_ids, scores)\n\n        return scores\n"


def _env_config() -> dict[str, Any]:
    model_path = os.environ.get("RSIBOX_TRANSFORMERS_MODEL")
    if not model_path:
        raise RuntimeError("set RSIBOX_TRANSFORMERS_MODEL to a model path or repo id")
    extra = {"max_new_tokens": int(os.environ["RSIBOX_TRANSFORMERS_MAX_NEW_TOKENS"])} if os.environ.get("RSIBOX_TRANSFORMERS_MAX_NEW_TOKENS") else {}
    return {"mode": "real", "model_path": model_path, "device": os.environ.get("RSIBOX_TRANSFORMERS_DEVICE", ""), "dtype": os.environ.get("RSIBOX_TRANSFORMERS_DTYPE", "bfloat16"), "seed": int(os.environ.get("RSIBOX_TRANSFORMERS_SEED", "0")), **extra}


class TransformersLab(BaseLab):
    """RSIBox live lab managing a persistent warm HuggingFace Transformers worker."""

    name, package, fresh_module, threaded_check = "transformers", "transformers", "adapter.fixture.fresh", True
    dirty = property(lambda self: self._dirty, lambda self, value: setattr(self, "_dirty", bool(value)))

    def __init__(
        self, config: dict[str, Any] | None = None, modules: tuple[str, ...] | list[str] | None = None, *,
        receipts: str | None = None, rpc_timeout: float = 30.0, check_timeout: float = 120.0,
        startup_timeout: float = 180.0, weight_store: Any = None, recovery_driver: Any = None, auto_spawn: bool = True,
    ) -> None:
        self.config = json.loads(json.dumps(config)) if config is not None else _env_config()
        self.config.setdefault("mode", "real")
        self._init_modules = list(modules if modules is not None else DEFAULT_REAL_MODULES)
        self.startup_timeout, self.sock_path, self._spec_path, self._worker_proc, self.model_module = float(startup_timeout), None, None, None, None
        super().__init__(receipts=receipts, check_timeout=check_timeout, rpc_timeout=rpc_timeout, auto_spawn=auto_spawn, weight_store=weight_store, recovery_driver=recovery_driver)

    @classmethod
    def launch_environment(cls, args: list[str]) -> dict[str, str]:
        parser = argparse.ArgumentParser(prog="rsibox conformance transformers --real", add_help=False)
        for flag, default_value in (("--model", None), ("--device", ""), ("--dtype", "bfloat16")):
            parser.add_argument(flag, required=(default_value is None), default=default_value)
        parser.add_argument("--max-new-tokens", type=int, default=16)
        parsed, _unknown = parser.parse_known_args(args)
        env = {"RSIBOX_TRANSFORMERS_MODEL": parsed.model, "RSIBOX_TRANSFORMERS_DTYPE": parsed.dtype, "RSIBOX_TRANSFORMERS_MAX_NEW_TOKENS": str(parsed.max_new_tokens)}
        return {**env, **({"RSIBOX_TRANSFORMERS_DEVICE": parsed.device} if parsed.device else {})}

    def is_running(self) -> bool:
        return self._worker_proc is not None and self._worker_proc.poll() is None and self.sock_path is not None

    def default_workload(self) -> dict[str, Any]:
        return {"prompts": ["Write one sentence about lighthouses.", "Write one sentence about rivers."], "max_new_tokens": int(self.config.get("max_new_tokens", 16)), "temperature": 0.0}

    def default_warmup_workload(self) -> dict[str, Any]:
        return {"prompts": ["Warmup."], "max_new_tokens": 2, "temperature": 0.0}

    def _livelab_paths(self) -> tuple[str, str]:
        livelab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return livelab_dir, os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(livelab_dir, "..", "..", "..", "core", "livelab"))

    def _start_engine(self) -> tuple[Any, list[str]]:
        if self.weight_store is not None and getattr(self.weight_store, "name", "none") != "none":
            probe_result = self.weight_store.probe(tp=1, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
            if not probe_result.ok:
                raise RuntimeError(f"weight store ({self.weight_store.name}) not ready: {probe_result.error}")
        self.sock_path = ipc.short_sock_path("rsitf")
        with tempfile.NamedTemporaryFile(mode="w", prefix="rsitf-spec-", suffix=".json", delete=False, encoding="utf-8") as spec_file:
            json.dump({"sock_path": self.sock_path, "config": {**self.config, "modules": list(self._init_modules)}}, spec_file)
            self._spec_path = spec_file.name
        livelab_dir, core_dir = self._livelab_paths()
        worker_env = dict(os.environ)
        existing_py = [part for part in (worker_env.get("PYTHONPATH") or "").split(os.pathsep) if part]
        worker_env["PYTHONPATH"] = os.pathsep.join([part for part in (livelab_dir, core_dir) if part not in existing_py] + existing_py)
        worker_env.setdefault("PYTHONUNBUFFERED", "1")
        self._worker_proc = subprocess.Popen([sys.executable, "-m", "adapter.agent", self._spec_path], env=worker_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            handshake = self._wait_for_worker()
            self.model_module = handshake.get("model_module")
            tracked = list(dict.fromkeys([*self._init_modules, *([self.model_module] if self.model_module else [])]))
            version = "0"
            with contextlib.suppress(Exception):
                version = importlib.metadata.version("transformers")
            participant = coordinator.CallParticipant("worker0", lambda method, payload: self.worker_call(method, payload))
            return RecaptureCoordinator([participant], gap=lambda action: action(), writer=self.writer, interface_version="1-draft", engine={"name": self.name, "version": version, "commit": None}, phase2_deadline_s=self.rpc_timeout, post_commit_reset="state"), tracked
        except BaseException:
            self._stop_engine()
            raise

    def _wait_for_worker(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._worker_proc is not None and self._worker_proc.poll() is not None:
                stderr_tail = (self._worker_proc.stderr.read() if self._worker_proc.stderr else "").strip()
                raise RuntimeError(f"transformers worker exited during startup (rc={self._worker_proc.returncode}): {stderr_tail[-400:]}")
            if self.sock_path and os.path.exists(self.sock_path):
                with contextlib.suppress(OSError):
                    reply = ipc.unix_json_call(self.sock_path, {"method": "ping"}, timeout=5.0)
                    if reply.get("ok"):
                        return reply
            time.sleep(0.05)
        raise TimeoutError(f"transformers worker did not become ready within {self.startup_timeout} s")

    def _stop_engine(self) -> None:
        if self._worker_proc is not None:
            if self._worker_proc.poll() is None and self.sock_path and not self.dirty:
                with contextlib.suppress(Exception):
                    ipc.unix_json_call(self.sock_path, {"method": "shutdown"}, timeout=2.0)
            for signal_fn, wait_seconds in ((self._worker_proc.terminate, 5.0), (self._worker_proc.kill, 2.0)):
                if self._worker_proc.poll() is None:
                    with contextlib.suppress(Exception):
                        signal_fn()
                        self._worker_proc.wait(timeout=wait_seconds)
            if self._worker_proc.stderr is not None:
                with contextlib.suppress(OSError):
                    self._worker_proc.stderr.close()
            self._worker_proc = None
        for cleanup_path in (self.sock_path, self._spec_path):
            if cleanup_path:
                with contextlib.suppress(OSError):
                    os.unlink(cleanup_path)
        self.sock_path, self._spec_path = None, None

    def worker_call(self, method: str, payload: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        if self.dirty or not self.sock_path:
            raise ConnectionError("worker channel dirty; respawn required" if self.dirty else "transformers worker is not running")
        try:
            return ipc.unix_json_call(self.sock_path, {"method": method, "payload": payload}, timeout=self.rpc_timeout if timeout is None else float(timeout))
        except (TimeoutError, ConnectionError, OSError) as error:
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
        try:
            return self.run_generate(workload)
        except BaseException:
            self.dirty = True
            raise

    def _reset_state(self) -> dict[str, Any]:
        reply = self.worker_call("reset", "state")
        return {"ok": True, "reset": list(reply.get("reset") or ["kv_cache", "rng"])} if reply.get("ok") else {"ok": False, "error": reply.get("error"), "reset": []}

    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        plan, plan_findings = super()._build_plan(patch, live_map)
        plan["undo_resets"] = list(patch.get("undo_resets", ["state"]))
        return plan, plan_findings

    def quiesce_worker(self, *, offload_to_cpu: bool = True, pin_memory: bool = True) -> dict[str, Any]:
        with self.lock:
            return self.worker_call("quiesce", {"offload_to_cpu": bool(offload_to_cpu), "pin_memory": bool(pin_memory)})

    def resume_worker(self, *, released: tuple[str, ...] = ()) -> dict[str, Any]:
        with self.lock:
            return self.worker_call("resume", {"released": list(released)})

    def quiesce_hooks(self, *, offload_to_cpu: bool = True, pin_memory: bool = True, warmup_workload: dict[str, Any] | None = None) -> Any:
        from adapter.snapshot_hooks import TransformersQuiesceHooks
        return TransformersQuiesceHooks(self, offload_to_cpu=offload_to_cpu, pin_memory=pin_memory, warmup_workload=warmup_workload)

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        livelab_dir, core_dir = self._livelab_paths()
        try:
            overrides = self.files(patch, against="disk")
        except ValueError as error:
            return {"ok": False, "output": None, "error": str(error)}
        spec = {"config": dict(self.config), "workload": {**self.default_workload(), **(workload or {})}}
        with harness.PackageOverlay(self.package, overrides) as overlay:
            return harness.run_fresh_subprocess(self.fresh_module, spec, overlay.root, extra_pythonpath=(livelab_dir, core_dir), timeout=self.check_timeout + 300.0)

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
        logits_mod, utils_mod = DEFAULT_REAL_MODULES
        logits_src, utils_src = self._base_source(logits_mod), self._base_source(utils_mod)
        visible_v1 = harness.edit_once(utils_src, ARGMAX_ANCHOR, "next_tokens = (torch.argmax(next_token_scores, dim=-1) + 1) % next_token_scores.shape[-1]")
        visible_v2 = harness.edit_once(utils_src, ARGMAX_ANCHOR, "next_tokens = (torch.argmax(next_token_scores, dim=-1) + 2) % next_token_scores.shape[-1]")
        logits_flip = harness.edit_once(logits_src, LOGITS_RETURN_ANCHOR, LOGITS_RETURN_ANCHOR.replace("return scores\n", "return scores.flip(-1)\n"))
        return {
            "noop": {"files": {utils_mod: harness.edit_once(utils_src, ARGMAX_ANCHOR, ARGMAX_ANCHOR + ".clone()")}},
            "visible": {"files": {utils_mod: visible_v1}},
            "chain": [{"files": {utils_mod: visible_v1}}, {"files": {utils_mod: visible_v2}}, {"files": {logits_mod: logits_flip}}],
            "multi": {"files": {utils_mod: visible_v1, logits_mod: logits_flip}},
            "unreached": {"files": {logits_mod: logits_src}},
            "refused": {"syntax": {"files": {utils_mod: harness.edit_once(utils_src, ARGMAX_ANCHOR, "next_tokens = (")}}, "empty": {"files": {utils_mod: harness.edit_once(utils_src, ARGMAX_ANCHOR, ARGMAX_ANCHOR + "  # greedy")}}},
        }
