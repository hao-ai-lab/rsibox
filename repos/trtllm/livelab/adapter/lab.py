"""Real-engine `TrtllmLab` (`rsibox.BaseLab`) for NVIDIA TensorRT-LLM PyTorch backend."""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import adapter
from adapter import agent as trt_agent
from adapter import mpi_hook
from adapter.fixture.patches import (
    MODEL_ENGINE_MODULE,
    PY_EXECUTOR_MODULE,
    SAMPLER_MODULE,
    build_real_patches,
)
from adapter.snapshot_hooks import TrtllmQuiesceHooks
from kernel import coordinator, findings
from rsibox import BaseLab, harness, ipc, lab as lab_base, proc

DEFAULT_WORKLOAD: dict[str, Any] = {
    "prompts": [
        "The capital of France is",
        "In computer science, a hash table is a data structure that",
    ],
    "max_tokens": 16,
    "temperature": 0.0,
}


def _resolve_float_option(explicit: float | None, env_key: str, default: float) -> float:
    if explicit is not None:
        return float(explicit)
    raw_value = os.environ.get(env_key)
    return float(raw_value) if raw_value is not None else float(default)


def _package_src_dir(package_name: str) -> str | None:
    """Locate a top-level package directory via `find_spec` without importing the package."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return os.path.abspath(next(iter(spec.submodule_search_locations)))
    if spec.origin:
        return os.path.dirname(os.path.abspath(spec.origin))
    return None


def _cleanup_endpoint(transport: str, endpoint: str) -> None:
    if transport == "file":
        shutil.rmtree(endpoint, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            os.unlink(endpoint)


class _WorkerHandle:
    """Subprocess handle communicating with one `adapter.agent` worker over `rsibox.ipc`."""

    def __init__(
        self,
        name: str,
        rank_index: int,
        config: dict[str, Any],
        livelab_dir: str,
        rpc_timeout: float,
        boot_timeout: float,
        *,
        extra_env: dict[str, str] | None = None,
        extra_pythonpath: list[str] | None = None,
        single_process: bool = True,
    ) -> None:
        self.name = name
        self.transport = config["transport"]
        self.endpoint = config["endpoint"]
        self.rpc_timeout = float(rpc_timeout)
        self.work_dir = tempfile.mkdtemp(prefix=f"rsibox-trtllm-r{rank_index}-")
        self.stderr_path = os.path.join(self.work_dir, "worker.stderr.log")
        config_path = os.path.join(self.work_dir, "worker.json")
        try:
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            env = dict(os.environ)
            python_paths = list(extra_pythonpath or ()) + [livelab_dir, adapter.LIVELAB]
            if env.get("PYTHONPATH"):
                python_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_paths)
            if extra_env:
                env.update(extra_env)
            if config["mode"] == "real":
                if single_process:
                    env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
                else:
                    env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)

            with open(self.stderr_path, "w", encoding="utf-8") as log_handle:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "adapter.agent", config_path],
                    env=env,
                    stdout=log_handle,
                    stderr=log_handle,
                    start_new_session=True,
                    text=True,
                )
        except BaseException:
            shutil.rmtree(self.work_dir, ignore_errors=True)
            raise
        self._wait_ready(boot_timeout)

    def _stderr_tail(self) -> str:
        try:
            with open(self.stderr_path, encoding="utf-8", errors="replace") as handle:
                return handle.read()[-1600:]
        except OSError:
            return ""

    def _wait_ready(self, boot_timeout: float) -> None:
        deadline = time.monotonic() + boot_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr_tail = self._stderr_tail()
                returncode = self.process.returncode
                self.close(force=True)
                raise RuntimeError(f"{self.name} exited during boot (rc={returncode}): {stderr_tail}")
            if self.call("ping", timeout=1.0).get("ok"):
                return
            time.sleep(0.05)
        stderr_tail = self._stderr_tail()
        self.close(force=True)
        raise TimeoutError(f"{self.name} did not answer ping within {boot_timeout} s: {stderr_tail}")

    def call(self, method: str, payload: Any = None, timeout: float | None = None) -> dict[str, Any]:
        call_timeout = self.rpc_timeout if timeout is None else float(timeout)
        request = {"method": method, "payload": payload}
        try:
            if self.transport == "file":
                reply = ipc.file_channel_call(
                    self.endpoint,
                    request,
                    timeout=call_timeout,
                    alive_fn=lambda: self.process.poll() is None,
                )
                if not reply.get("ok") and reply.get("lost"):
                    return {**reply, "transport_error": True}
                return reply
            return ipc.unix_json_call(self.endpoint, request, timeout=call_timeout)
        except Exception as error:
            return findings.not_ok(
                f"{self.name} {method}: {type(error).__name__}: {error}",
                transport_error=True,
            )

    def close(self, *, force: bool = False) -> None:
        if self.process.poll() is None:
            if not force:
                with contextlib.suppress(Exception):
                    self.call("shutdown", timeout=1.5)
            proc.stop_tree(self.process.pid, grace_s=0.0 if force else 3.0, include_pg=True)
            with contextlib.suppress(Exception):
                self.process.wait(timeout=2.0)
        _cleanup_endpoint(self.transport, self.endpoint)
        shutil.rmtree(self.work_dir, ignore_errors=True)


class _RankClient:
    """RPC handle for an MPI-spawned TensorRT-LLM worker rank (`TP >= 2`)."""

    def __init__(self, name: str, endpoint: str, rpc_timeout: float, transport: str = "unix") -> None:
        self.name = name
        self.transport = transport
        self.endpoint = endpoint
        self.rpc_timeout = float(rpc_timeout)

    def wait_ready(self, driver: _WorkerHandle, boot_timeout: float) -> None:
        deadline = time.monotonic() + boot_timeout
        while time.monotonic() < deadline:
            if driver.process.poll() is not None:
                raise RuntimeError(
                    f"driver exited while waiting for {self.name} (rc={driver.process.returncode}): "
                    f"{driver._stderr_tail()}"
                )
            if os.path.exists(self.endpoint) and self.call("ping", timeout=1.0).get("ok"):
                return
            time.sleep(0.05)
        raise TimeoutError(f"{self.name} did not answer ping within {boot_timeout} s: {driver._stderr_tail()}")

    def call(self, method: str, payload: Any = None, timeout: float | None = None) -> dict[str, Any]:
        call_timeout = self.rpc_timeout if timeout is None else float(timeout)
        request = {"method": method, "payload": payload}
        try:
            caller = ipc.file_channel_call if self.transport == "file" else ipc.unix_json_call
            return caller(self.endpoint, request, timeout=call_timeout)
        except Exception as error:
            return findings.not_ok(
                f"{self.name} {method}: {type(error).__name__}: {error}",
                transport_error=True,
            )

    def close(self, *, force: bool = False) -> None:
        if not force:
            with contextlib.suppress(Exception):
                self.call("shutdown", timeout=1.0)
        _cleanup_endpoint(self.transport, self.endpoint)


class TrtllmLab(BaseLab):
    """Live RSIBox lab for NVIDIA TensorRT-LLM (PyTorch backend)."""

    name = "trtllm"
    engine_mode = "real"
    overlay_package = "tensorrt_llm"
    threaded_check = True
    default_tamper_module = SAMPLER_MODULE
    default_tamper_function = "TorchSampler._process_requests"

    def __init__(
        self,
        model: str | None = None,
        *,
        tensor_parallel_size: int | None = None,
        free_gpu_memory_fraction: float | None = None,
        max_batch_size: int = 4,
        max_num_tokens: int = 2048,
        disable_overlap_scheduler: bool = True,
        enable_block_reuse: bool = False,
        modules: tuple[str, ...] | list[str] | None = None,
        workload: dict[str, Any] | None = None,
        transport: str = "unix",
        boot_timeout: float = 240.0,
        fresh_timeout: float = 300.0,
        quiesce_timeout: float | None = None,
        extra_kwargs: dict[str, Any] | None = None,
        receipts: str | None = None,
        check_timeout: float | None = None,
        rpc_timeout: float | None = None,
        auto_spawn: bool = True,
        weight_store: Any = None,
        recovery_driver: Any = None,
    ) -> None:
        resolved_tp = (
            int(tensor_parallel_size)
            if tensor_parallel_size is not None
            else int(os.environ.get("RSIBOX_TRTLLM_TP", "1"))
        )
        if resolved_tp < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1 (got {resolved_tp})")
        self.rank_count = resolved_tp
        extra_config = dict(extra_kwargs or {})
        if self.engine_mode == "real":
            forbidden_parallel = sorted(
                key
                for key in ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size")
                if key in extra_config
            )
            if forbidden_parallel:
                raise ValueError(
                    f"extra_kwargs cannot override parallelism keys directly: {forbidden_parallel}"
                )
        self.livelab_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.transport = transport
        self.boot_timeout = float(boot_timeout)
        self.fresh_timeout = float(fresh_timeout)
        self.quiesce_timeout = _resolve_float_option(quiesce_timeout, "RSIBOX_TRTLLM_QUIESCE_TIMEOUT", 120.0)
        self.workload = dict(workload) if workload is not None else dict(DEFAULT_WORKLOAD)
        self.engine_config: dict[str, Any] = {
            "model": model if model is not None else os.environ.get("RSIBOX_TRTLLM_MODEL", ""),
            "tensor_parallel_size": self.rank_count,
            "free_gpu_memory_fraction": _resolve_float_option(
                free_gpu_memory_fraction, "RSIBOX_TRTLLM_MEM_FRACTION", 0.15
            ),
            "max_batch_size": int(max_batch_size),
            "max_num_tokens": int(max_num_tokens),
            "disable_overlap_scheduler": bool(disable_overlap_scheduler),
            "enable_block_reuse": bool(enable_block_reuse),
            "extra_kwargs": extra_config,
        }
        self._tracked_modules = (
            tuple(modules) if modules is not None else (SAMPLER_MODULE, PY_EXECUTOR_MODULE, MODEL_ENGINE_MODULE)
        )
        self.workers: list[Any] = []
        self._driver: _WorkerHandle | None = None
        self._mpi_hook_dir: str | None = None
        self._mpi_sock_dir: str | None = None
        super().__init__(
            receipts=receipts,
            check_timeout=_resolve_float_option(check_timeout, "RSIBOX_TRTLLM_CHECK_TIMEOUT", 60.0),
            rpc_timeout=_resolve_float_option(rpc_timeout, "RSIBOX_TRTLLM_RPC_TIMEOUT", 30.0),
            auto_spawn=auto_spawn,
            weight_store=weight_store,
            recovery_driver=recovery_driver,
        )

    @staticmethod
    def launch_environment(argv: list[str]) -> dict[str, str]:
        """Translate `rsibox conformance trtllm --real ...` flags into `RSIBOX_TRTLLM_*` env vars."""
        clean_args = list(argv)
        if clean_args and clean_args[0] == "--":
            clean_args = clean_args[1:]
        parser = argparse.ArgumentParser(prog="rsibox conformance trtllm --real", allow_abbrev=False)
        parser.add_argument("--model", type=str, default=None, help="local model path or repo id")
        parser.add_argument("--tp", type=int, default=None, help="tensor parallel degree (1=single-process, >=2=MPI)")
        parser.add_argument("--mem-fraction", type=float, default=None, help="KV cache free GPU memory fraction")
        parser.add_argument("--check-timeout", type=float, default=None, help="workload check timeout (s)")
        parser.add_argument("--rpc-timeout", type=float, default=None, help="control RPC timeout (s)")
        parser.add_argument("--quiesce-timeout", type=float, default=None, help="quiesce/resume timeout (s)")
        parsed = parser.parse_args(clean_args)
        env_mapping = (
            ("model", "RSIBOX_TRTLLM_MODEL"),
            ("tp", "RSIBOX_TRTLLM_TP"),
            ("mem_fraction", "RSIBOX_TRTLLM_MEM_FRACTION"),
            ("check_timeout", "RSIBOX_TRTLLM_CHECK_TIMEOUT"),
            ("rpc_timeout", "RSIBOX_TRTLLM_RPC_TIMEOUT"),
            ("quiesce_timeout", "RSIBOX_TRTLLM_QUIESCE_TIMEOUT"),
        )
        return {
            env_key: str(getattr(parsed, attr))
            for attr, env_key in env_mapping
            if getattr(parsed, attr) is not None
        }

    def default_warmup_workload(self) -> dict[str, Any]:
        prompts = self.workload.get("prompts") or DEFAULT_WORKLOAD["prompts"]
        return {"prompts": list(prompts[:1]), "max_tokens": 2, "temperature": 0.0}

    def quiesce_hooks(self, **kwargs: Any) -> TrtllmQuiesceHooks:
        return TrtllmQuiesceHooks(self, **kwargs)

    def _base_source(self, key: str) -> str:
        if key not in self.base_text:
            worker_error: str | None = None
            if self.workers:
                reply = self.workers[0].call("read_source", {"module": key})
                if reply.get("ok") and isinstance(reply.get("source"), str):
                    self.base_text[key] = reply["source"]
                    return self.base_text[key]
                worker_error = str(reply.get("error") or "read_source returned no source")
            if self.engine_mode == "real" and key.startswith(f"{self.overlay_package}."):
                raise RuntimeError(
                    f"failed to read {key!r} from worker without importing {self.overlay_package} in host"
                    + (f": {worker_error}" if worker_error else "")
                )
            self.base_text[key] = harness.read_module(key)
        return self.base_text[key]

    def _use_mpi_mode(self) -> bool:
        return self.engine_mode == "real" and self.rank_count > 1

    def _fanout(self, method: str, payload: Any = None, *, timeout: float | None = None) -> list[dict[str, Any]]:
        if len(self.workers) <= 1:
            return [w.call(method, payload, timeout=timeout) for w in self.workers]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            futs = [pool.submit(w.call, method, payload, timeout) for w in self.workers]
            return [f.result() for f in futs]

    def _step_gap(self, action: Any) -> Any:
        replies = self._fanout("park")
        for worker, reply in zip(self.workers, replies):
            if not reply.get("ok"):
                raise RuntimeError(f"failed to park {worker.name}: {reply.get('error')}")
        try:
            return action()
        finally:
            with contextlib.suppress(Exception):
                self._fanout("release_gap")

    def _rebuild_all_ranks(self, txn: str, spec: dict[str, Any]) -> dict[str, Any]:
        payload = {"txn": txn, **dict(spec or {})}
        replies = self._fanout("recapture", payload, timeout=self.quiesce_timeout)
        ok = all(r.get("ok") for r in replies)
        out: dict[str, Any] = {"ok": ok, "ranks": replies}
        if not ok:
            errs = [str(r.get("error") or "recapture failed") for r in replies if not r.get("ok")]
            out["error"] = "; ".join(errs)
        return out

    def _build_plan(self, patch: dict[str, Any], live_map: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        plan, extra = super()._build_plan(patch, live_map)
        if "recapture" in patch:
            plan["recapture"] = dict(patch["recapture"] or {})
        if "rebuild" in patch:
            plan["rebuild"] = list(patch["rebuild"] or ())
            return plan, extra
        extra_cfg = self.engine_config.get("extra_kwargs") or {}
        use_cg = bool(extra_cfg.get("use_cuda_graph", True))
        if use_cg and any(trt_agent._is_captured_module(mod) for mod in (plan.get("modules") or ())):
            plan["rebuild"] = ["recapture"]
        return plan, extra

    def _start_mpi_engine(self) -> tuple[lab_base.RecaptureCoordinator, list[str]]:
        self._mpi_hook_dir = tempfile.mkdtemp(prefix="rsibox-trtllm-hook-")
        self._mpi_sock_dir = tempfile.mkdtemp(prefix="rsibox-trtllm-mpi-", dir="/tmp")
        with open(os.path.join(self._mpi_hook_dir, "sitecustomize.py"), "w", encoding="utf-8") as handle:
            handle.write(mpi_hook.SITECUSTOMIZE_SOURCE)

        driver_endpoint = ipc.short_sock_path(prefix="rsibox-trtllm-drv")
        driver_config = {
            "action": "driver",
            "mode": self.engine_mode,
            "parent_pid": os.getpid(),
            "transport": "unix",
            "endpoint": driver_endpoint,
            "engine_config": self.engine_config,
        }
        mpi_env = {
            mpi_hook.ENV_MPI_SOCK_DIR: self._mpi_sock_dir,
            mpi_hook.ENV_MPI_MODULES: json.dumps(list(self._tracked_modules)),
            mpi_hook.ENV_MPI_ENABLE_BLOCK_REUSE: "1" if self.engine_config.get("enable_block_reuse") else "0",
            mpi_hook.ENV_MPI_SEED: "0",
        }
        self._driver = _WorkerHandle(
            "driver",
            0,
            driver_config,
            self.livelab_dir,
            self.rpc_timeout,
            self.boot_timeout,
            extra_env=mpi_env,
            extra_pythonpath=[self._mpi_hook_dir],
            single_process=False,
        )
        participants: list[coordinator.CallParticipant] = []
        for rank_index in range(self.rank_count):
            worker_name = f"rank{rank_index}"
            sock_path = os.path.join(self._mpi_sock_dir, f"rank-{rank_index}.sock")
            client = _RankClient(worker_name, sock_path, self.rpc_timeout, transport="unix")
            client.wait_ready(self._driver, self.boot_timeout)
            self.workers.append(client)
            participants.append(coordinator.CallParticipant(worker_name, client.call, bounded=True))
        for module_name in self._tracked_modules:
            self._base_source(module_name)
        return (
            lab_base.RecaptureCoordinator(
                participants,
                gap=self._step_gap,
                writer=self.writer,
                rebuild=self._rebuild_all_ranks,
            ),
            list(self._tracked_modules),
        )

    def _start_engine(self) -> tuple[coordinator.StepGapCoordinator, list[str]]:
        self.workers = []
        self._driver = None
        if self._use_mpi_mode():
            try:
                return self._start_mpi_engine()
            except BaseException:
                self._stop_engine()
                raise
        participants: list[coordinator.CallParticipant] = []
        parent_pid = os.getpid()
        try:
            for rank_index in range(self.rank_count):
                worker_name = f"rank{rank_index}"
                endpoint = (
                    tempfile.mkdtemp(prefix=f"rsibox-trtllm-chan-r{rank_index}-")
                    if self.transport == "file"
                    else ipc.short_sock_path(prefix=f"rsibox-trtllm-r{rank_index}")
                )
                worker_config = {
                    "action": "worker",
                    "mode": self.engine_mode,
                    "rank": rank_index,
                    "parent_pid": parent_pid,
                    "transport": self.transport,
                    "endpoint": endpoint,
                    "engine_config": self.engine_config,
                    "modules": list(self._tracked_modules),
                    "seed": 0,
                }
                try:
                    worker = _WorkerHandle(
                        worker_name,
                        rank_index,
                        worker_config,
                        self.livelab_dir,
                        self.rpc_timeout,
                        self.boot_timeout,
                    )
                except BaseException:
                    _cleanup_endpoint(self.transport, endpoint)
                    raise
                self.workers.append(worker)
                participants.append(coordinator.CallParticipant(worker_name, worker.call, bounded=True))
            for module_name in self._tracked_modules:
                self._base_source(module_name)
        except BaseException:
            self._stop_engine()
            raise
        return (
            lab_base.RecaptureCoordinator(
                participants,
                gap=self._step_gap,
                writer=self.writer,
                rebuild=self._rebuild_all_ranks,
            ),
            list(self._tracked_modules),
        )

    def _stop_engine(self) -> None:
        force_stop = bool(self._dirty)
        for worker in reversed(self.workers):
            worker.close(force=force_stop)
        self.workers = []
        if self._driver is not None:
            self._driver.close(force=force_stop)
            self._driver = None
        if self._mpi_sock_dir:
            shutil.rmtree(self._mpi_sock_dir, ignore_errors=True)
            self._mpi_sock_dir = None
        if self._mpi_hook_dir:
            shutil.rmtree(self._mpi_hook_dir, ignore_errors=True)
            self._mpi_hook_dir = None

    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        merged = {**self.workload, **(workload or {})}
        if self._driver is not None:
            reply = self._driver.call("generate", merged, timeout=self.check_timeout + 5.0)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error") or "driver: generate failed")
            tokens = reply["output"]
            return {worker.name: tokens for worker in self.workers}
        outputs: dict[str, Any] = {}
        for worker in self.workers:
            reply = worker.call("generate", merged, timeout=self.check_timeout + 5.0)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error") or f"{worker.name}: generate failed")
            outputs[worker.name] = reply["output"]
        return outputs

    def _reset_state(self) -> dict[str, Any]:
        reset_items: list[str] = []
        replies = self._fanout("reset", "state")
        for worker, reply in zip(self.workers, replies):
            if not reply.get("ok"):
                return {"ok": False, "error": reply.get("error") or f"{worker.name}: reset failed", "reset": []}
            for item in reply.get("result") or reply.get("reset") or ():
                if item not in reset_items:
                    reset_items.append(item)
        return {"ok": True, "reset": reset_items or ["kv_cache_events", "rng"]}

    def patches(self) -> dict[str, Any]:
        return build_real_patches(self._base_source)

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = {
            "action": "fresh",
            "mode": self.engine_mode,
            "rank_count": self.rank_count,
            "engine_config": self.engine_config,
            "workload": {**self.workload, **(workload or {})},
        }
        src_dir = _package_src_dir(self.overlay_package)
        with harness.PackageOverlay(self.overlay_package, (patch or {}).get("files"), src_dir=src_dir) as overlay:
            return harness.run_fresh_subprocess(
                "adapter.agent",
                spec,
                overlay.root,
                extra_pythonpath=[self.livelab_dir, adapter.LIVELAB],
                timeout=self.fresh_timeout,
            )

    def inject(self, fault: str, **kwargs: Any) -> dict[str, Any]:
        target_name = kwargs.pop("participant", self.workers[-1].name) if self.workers else None
        worker = next((item for item in self.workers if item.name == target_name), None)
        if worker is None:
            return findings.not_ok(f"unknown participant {target_name!r}")
        payload: dict[str, Any] = {"fault": fault, **kwargs}
        if fault == "hang":
            payload.setdefault("hang_s", self.check_timeout + 0.5)
        elif fault == "tamper":
            payload.setdefault("module", self.default_tamper_module)
            payload.setdefault("function", self.default_tamper_function)
        if self._driver is not None and fault in ("crash", "hang"):
            return self._driver.call("inject", payload)
        return worker.call("inject", payload)
