"""Real-engine `TrtllmLab` (`rsibox.BaseLab`) for NVIDIA TensorRT-LLM PyTorch backend."""
from __future__ import annotations

import argparse, contextlib, json, os, shutil, subprocess, sys, tempfile, time
from typing import Any
import adapter
from adapter.snapshot_hooks import TrtllmQuiesceHooks
from kernel import coordinator, findings
from rsibox import BaseLab, harness, ipc

SAMPLER_MODULE = "tensorrt_llm._torch.pyexecutor.sampler"
PY_EXECUTOR_MODULE = "tensorrt_llm._torch.pyexecutor.py_executor"
DEFAULT_WORKLOAD: dict[str, Any] = {"prompts": ["The capital of France is"], "max_tokens": 16, "temperature": 0.0}

class _WorkerHandle:
    """Subprocess handle communicating with one `adapter.agent` worker over `rsibox.ipc`."""
    def __init__(self, name: str, rank_index: int, config: dict[str, Any], livelab_dir: str, rpc_timeout: float, boot_timeout: float) -> None:
        self.name, self.transport, self.endpoint, self.rpc_timeout = name, config["transport"], config["endpoint"], float(rpc_timeout)
        self.work_dir = tempfile.mkdtemp(prefix=f"rsibox-trtllm-r{rank_index}-")
        config_path = os.path.join(self.work_dir, "worker.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([livelab_dir, adapter.LIVELAB] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        if config["mode"] == "real":
            env.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        self.process = subprocess.Popen([sys.executable, "-m", "adapter.agent", config_path], env=env, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + boot_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr_tail = (self.process.stderr.read() if self.process.stderr else "")[-1200:]
                raise RuntimeError(f"{self.name} exited during boot (rc={self.process.returncode}): {stderr_tail}")
            if self.call("ping", timeout=1.0).get("ok"):
                return
            time.sleep(0.05)
        self.close()
        raise TimeoutError(f"{self.name} did not answer ping within {boot_timeout} s")

    def call(self, method: str, payload: Any = None, timeout: float | None = None) -> dict[str, Any]:
        call_timeout, request = (self.rpc_timeout if timeout is None else float(timeout)), {"method": method, "payload": payload}
        try:
            caller = ipc.file_channel_call if self.transport == "file" else ipc.unix_json_call
            return caller(self.endpoint, request, timeout=call_timeout)
        except Exception as error:
            return findings.not_ok(f"{self.name} {method}: {type(error).__name__}: {error}")

    def close(self) -> None:
        if self.process.poll() is None:
            with contextlib.suppress(Exception):
                self.call("shutdown", timeout=1.5)
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                with contextlib.suppress(Exception):
                    self.process.wait(timeout=2.0)
        for cleanup in (lambda: self.process.stderr and self.process.stderr.close(),
                        lambda: self.transport == "unix" and os.unlink(self.endpoint)):
            with contextlib.suppress(Exception):
                cleanup()
        shutil.rmtree(self.work_dir, ignore_errors=True)

class TrtllmLab(BaseLab):
    """Live RSIBox lab for NVIDIA TensorRT-LLM (PyTorch backend)."""
    name, engine_mode, overlay_package, threaded_check = "trtllm", "real", "tensorrt_llm", True
    default_tamper_module, default_tamper_function = SAMPLER_MODULE, "TorchSampler._process_requests"

    def __init__(self, model: str | None = None, *, tensor_parallel_size: int = 1, free_gpu_memory_fraction: float = 0.15,
                 max_batch_size: int = 4, max_num_tokens: int = 2048, disable_overlap_scheduler: bool = True,
                 enable_block_reuse: bool = False, modules: tuple[str, ...] | list[str] | None = None,
                 workload: dict[str, Any] | None = None, transport: str = "unix", boot_timeout: float = 240.0,
                 fresh_timeout: float = 300.0, extra_kwargs: dict[str, Any] | None = None, receipts: str | None = None,
                 check_timeout: float = 60.0, rpc_timeout: float = 30.0, auto_spawn: bool = True,
                 weight_store: Any = None, recovery_driver: Any = None) -> None:
        self.livelab_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.rank_count, self.transport, self.boot_timeout, self.fresh_timeout = int(tensor_parallel_size), transport, float(boot_timeout), float(fresh_timeout)
        self.workload = dict(workload) if workload is not None else dict(DEFAULT_WORKLOAD)
        self.engine_config = {
            "model": model or os.environ.get("RSIBOX_TRTLLM_MODEL", ""), "tensor_parallel_size": self.rank_count,
            "free_gpu_memory_fraction": float(os.environ.get("RSIBOX_TRTLLM_MEM_FRACTION", free_gpu_memory_fraction)),
            "max_batch_size": int(max_batch_size), "max_num_tokens": int(max_num_tokens),
            "disable_overlap_scheduler": bool(disable_overlap_scheduler), "enable_block_reuse": bool(enable_block_reuse),
            "extra_kwargs": dict(extra_kwargs or {}),
        }
        self._tracked_modules = tuple(modules) if modules is not None else (SAMPLER_MODULE, PY_EXECUTOR_MODULE)
        self.workers: list[_WorkerHandle] = []
        super().__init__(receipts=receipts, check_timeout=float(os.environ.get("RSIBOX_TRTLLM_CHECK_TIMEOUT", check_timeout)),
                         rpc_timeout=float(os.environ.get("RSIBOX_TRTLLM_RPC_TIMEOUT", rpc_timeout)),
                         auto_spawn=auto_spawn, weight_store=weight_store, recovery_driver=recovery_driver)

    @staticmethod
    def launch_environment(argv: list[str]) -> dict[str, str]:
        parser = argparse.ArgumentParser(prog="rsibox conformance trtllm --real", allow_abbrev=False)
        for flag, arg_type in (("--model", str), ("--mem-fraction", float), ("--check-timeout", float), ("--rpc-timeout", float)):
            parser.add_argument(flag, type=arg_type, default=None)
        parsed = parser.parse_args([token for token in argv if token != "--"])
        pairs = (("model", "RSIBOX_TRTLLM_MODEL"), ("mem_fraction", "RSIBOX_TRTLLM_MEM_FRACTION"),
                 ("check_timeout", "RSIBOX_TRTLLM_CHECK_TIMEOUT"), ("rpc_timeout", "RSIBOX_TRTLLM_RPC_TIMEOUT"))
        return {env_key: str(getattr(parsed, attr)) for attr, env_key in pairs if getattr(parsed, attr) is not None}

    def default_warmup_workload(self) -> dict[str, Any]:
        return {"prompts": list((self.workload.get("prompts") or DEFAULT_WORKLOAD["prompts"])[:1]), "max_tokens": 2, "temperature": 0.0}

    def quiesce_hooks(self, **kwargs: Any) -> TrtllmQuiesceHooks:
        return TrtllmQuiesceHooks(self, **kwargs)

    def _step_gap(self, action):
        for worker in self.workers:
            reply = worker.call("park")
            if not reply.get("ok"):
                raise RuntimeError(f"failed to park {worker.name}: {reply.get('error')}")
        try:
            return action()
        finally:
            for worker in reversed(self.workers):
                with contextlib.suppress(Exception):
                    worker.call("release_gap")

    def _start_engine(self):
        self.workers, participants = [], []
        try:
            for rank_index in range(self.rank_count):
                worker_name = f"rank{rank_index}"
                endpoint = (tempfile.mkdtemp(prefix=f"rsibox-trtllm-chan-r{rank_index}-") if self.transport == "file"
                            else ipc.short_sock_path(prefix=f"rsibox-trtllm-r{rank_index}"))
                worker_config = {"action": "worker", "mode": self.engine_mode, "rank": rank_index, "transport": self.transport,
                                 "endpoint": endpoint, "engine_config": self.engine_config, "modules": list(self._tracked_modules), "seed": 0}
                worker = _WorkerHandle(worker_name, rank_index, worker_config, self.livelab_dir, self.rpc_timeout, self.boot_timeout)
                self.workers.append(worker)
                participants.append(coordinator.CallParticipant(worker_name, worker.call, bounded=True))
        except BaseException:
            self._stop_engine()
            raise
        for module_name in self._tracked_modules:
            self._base_source(module_name)
        return coordinator.StepGapCoordinator(participants, gap=self._step_gap, writer=self.writer), list(self._tracked_modules)

    def _stop_engine(self) -> None:
        for worker in reversed(self.workers):
            worker.close()
            if worker.transport == "file":
                shutil.rmtree(worker.endpoint, ignore_errors=True)
        self.workers = []

    def _run_workload(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        merged, outputs = {**self.workload, **(workload or {})}, {}
        for worker in self.workers:
            reply = worker.call("generate", merged, timeout=self.check_timeout + 5.0)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error") or f"{worker.name}: generate failed")
            outputs[worker.name] = reply["output"]
        return outputs

    def _reset_state(self) -> dict[str, Any]:
        reset_items: list[str] = []
        for worker in self.workers:
            reply = worker.call("reset", "state")
            if not reply.get("ok"):
                return {"ok": False, "error": reply.get("error") or f"{worker.name}: reset failed", "reset": []}
            reset_items.extend(item for item in (reply.get("result") or reply.get("reset") or ()) if item not in reset_items)
        return {"ok": True, "reset": reset_items or ["kv_cache_events", "rng"]}

    def patches(self) -> dict[str, Any]:
        sampler_src, exec_src = self._base_source(SAMPLER_MODULE), self._base_source(PY_EXECUTOR_MODULE)
        fast_anchor = "            logits = self._apply_embedding_bias(logits, requests)\n            next_tokens = torch.argmax(logits, dim=-1)"
        vis_one = harness.edit_once(sampler_src, fast_anchor, "            logits = self._apply_embedding_bias(logits, requests)\n            next_tokens = (torch.argmax(logits, dim=-1) + 1) % logits.shape[-1]")
        vis_two = harness.edit_once(vis_one, "            next_tokens = (torch.argmax(logits, dim=-1) + 1) % logits.shape[-1]", "            next_tokens = (torch.argmax(logits, dim=-1) + 2) % logits.shape[-1]")
        multi_sampler = harness.edit_once(vis_one, "        vocab_size = model_outputs[\"logits\"].shape[-1]", "        vocab_size = int(model_outputs[\"logits\"].shape[-1])")
        multi_exec = harness.edit_once(exec_src, "                return self.sampler.sample_async(scheduled_batch, batch_outputs)", "                sampled = self.sampler.sample_async(scheduled_batch, batch_outputs)\n                return sampled")
        return {
            "noop": {"files": {SAMPLER_MODULE: harness.edit_once(sampler_src, fast_anchor, "            logits = self._apply_embedding_bias(logits, requests)\n            chosen = torch.argmax(logits, dim=-1)\n            next_tokens = chosen")}},
            "visible": {"files": {SAMPLER_MODULE: vis_one}},
            "chain": [{"files": {SAMPLER_MODULE: vis_one}}, {"files": {SAMPLER_MODULE: vis_two}}, {"files": {PY_EXECUTOR_MODULE: multi_exec}}],
            "multi": {"files": {SAMPLER_MODULE: multi_sampler, PY_EXECUTOR_MODULE: multi_exec}},
            "unreached": {"files": {SAMPLER_MODULE: harness.edit_once(sampler_src, "        self.max_seq_len = args.max_seq_len", "        self.max_seq_len = int(args.max_seq_len)")}},
            "refused": {
                "syntax": {"files": {SAMPLER_MODULE: harness.edit_once(sampler_src, fast_anchor, "            logits = self._apply_embedding_bias(logits, requests)\n            next_tokens = torch.argmax(")}},
                "closure": {"files": {SAMPLER_MODULE: harness.edit_once(sampler_src, "def greedy_search_sampling_batch(logits):\n    next_tokens = torch.argmax(logits, dim=-1)", "def greedy_search_sampling_batch(logits):\n    extra = 1\n    next_tokens = (lambda value: value + extra)(torch.argmax(logits, dim=-1))")}},
                "module_state": {"files": {SAMPLER_MODULE: harness.edit_once(sampler_src, 'GREEDY: Greedy = ("greedy", None)', 'GREEDY: Greedy = ("greedy", 0)')}},
                "empty": {"files": {SAMPLER_MODULE: harness.edit_once(sampler_src, "def greedy_search_sampling_batch(logits):", "def greedy_search_sampling_batch(logits):  # note")}},
            },
        }

    def fresh_check(self, patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = {"action": "fresh", "mode": self.engine_mode, "rank_count": self.rank_count, "engine_config": self.engine_config, "workload": {**self.workload, **(workload or {})}}
        with harness.PackageOverlay(self.overlay_package, (patch or {}).get("files")) as overlay:
            return harness.run_fresh_subprocess("adapter.agent", spec, overlay.root, extra_pythonpath=[self.livelab_dir, adapter.LIVELAB], timeout=self.fresh_timeout)

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
        return worker.call("inject", payload)
