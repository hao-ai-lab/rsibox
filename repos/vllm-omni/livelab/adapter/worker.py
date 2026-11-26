"""Multi-stage vLLM-Omni worker with S4 frozen stage residency and S5 S0 stage-buffer reset."""
from __future__ import annotations

import contextlib, gc, hashlib, importlib, inspect, json, os, queue, random, sys, time
from typing import Any

from adapter import dit_op, stage_ops, talker_op, thinker_op, vocoder_op
from kernel import findings
from rsibox import harness, ipc
from rsibox.lab import ENDPOINT_METHODS, RoleEndpoint
from rsibox.services import AuxiliaryModelService, AuxStageKind, MemoryPoolKind, MemoryService, VersionedReplayBufferService

STAGE_ROLES = ("thinker", "talker", "dit", "vocoder")
PARTICIPANT_METHODS = frozenset(ENDPOINT_METHODS)


def assemble_omni_output(thinker_ids: list[int], waveform: list[float], dit_latents: list[float]) -> dict[str, Any]:
    """Assemble deterministic multimodal output digest across Thinker, DiT, and Vocoder stages."""
    wave_rounded = [round(float(v), 6) for v in waveform]
    dit_rounded = [round(float(v), 6) for v in dit_latents]
    payload = {"thinker_ids": list(thinker_ids), "dit": dit_rounded, "waveform": wave_rounded}
    return {**payload, "digest": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]}


class VllmOmniPipelineWorker:
    """Owns 4-stage pipeline, S1+S4 frozen stage residency, and S5 S0 stage-buffer state."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.mode, self.seed = str(config.get("mode", "cpu")), int(config.get("seed", 0))
        self.model_path, self.use_aux_cache = str(config.get("model_path", "")), bool(config.get("use_aux_cache", True))
        self.cfg = stage_ops.OmniPipelineConfig(int(config.get("hidden_dim", 8)), int(config.get("dit_steps", 2)), int(config.get("max_new_tokens", 8)), int(config.get("image_size", 64)))
        self.memory = MemoryService()
        self.aux = AuxiliaryModelService(memory_service=self.memory, default_cache_entries=64)
        self.stage_buffer = VersionedReplayBufferService(capacity=64, current_code_version="v0", drain_on_version_mismatch=True)
        self.devices, self.thinker_device, self.synth_device, self.frozen_vram_baseline = ["cpu"], "cpu", "cpu", {}
        self._rng_torch: Any = None; self._rng_cuda: dict[str, Any] | None = None
        self._llm: Any = None; self._processor: Any = None; self._vocoder_proj: Any = None
        self._at_s0, self._last_reset = False, ["stage_buffers_s0", "buffer_epoch_0", "rng"]
        self._init_stages()

    def _init_stages(self) -> None:
        random.seed(self.seed); self._rng_py = random.getstate()
        if self.mode != "real":
            voc_weights = [round(0.2 + 0.05 * i, 4) for i in range(self.cfg.hidden_dim)]
            th = self.memory.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "vllm-omni", "thinker", gpus=(), size_bytes=1024, stamp={"stage": "thinker", "mode": "cpu"})
            vh = self.memory.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "vllm-omni", "vocoder", gpus=(), size_bytes=512, stamp={"stage": "vocoder", "mode": "cpu"})
            self.aux.register_stage("thinker_backbone", kind=AuxStageKind.TEXT_ENCODER, engine="vllm-omni", model_id="toy-thinker", handle_id=th.handle_id, infer_fn=self._cpu_thinker_infer)
            self.aux.register_stage("vocoder_backbone", kind=AuxStageKind.CUSTOM, engine="vllm-omni", model_id="toy-vocoder", handle_id=vh.handle_id, infer_fn=lambda _p, _c: voc_weights)
            self._at_s0 = True
            return
        import torch  # type: ignore[import-not-found]
        if not torch.cuda.is_available():
            raise RuntimeError("mode='real' requires CUDA GPU")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.thinker_device, self.synth_device, self.devices = "cuda:0", "cuda:0", ["cuda:0"]
        torch.manual_seed(self.seed); torch.cuda.set_device("cuda:0"); torch.cuda.manual_seed(self.seed)
        self._llm, self._processor, self._vocoder_proj = stage_ops.build_gpu_stage_modules(self.cfg, self.model_path, self.synth_device, seed=self.seed)
        th = self.memory.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "vllm-omni", "thinker", gpus=(0,), size_bytes=max(1, int(torch.cuda.memory_allocated(0))), stamp={"model": os.path.basename(self.model_path)})
        vh = self.memory.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "vllm-omni", "vocoder", gpus=(0,), size_bytes=max(1, self._vocoder_proj.numel() * self._vocoder_proj.element_size()), stamp={"stage": "vocoder"})
        self.aux.register_stage("thinker_backbone", kind=AuxStageKind.TEXT_ENCODER, engine="vllm-omni", model_id=os.path.basename(self.model_path), handle_id=th.handle_id, infer_fn=self._gpu_thinker_infer)
        self.aux.register_stage("vocoder_backbone", kind=AuxStageKind.CUSTOM, engine="vllm-omni", model_id="frozen-vocoder", handle_id=vh.handle_id, infer_fn=lambda _p, _c: self._vocoder_proj)
        # Warm CUDA/vLLM workspaces with use_cache=False so S4 LRU cache stays cold until first check.
        self.aux.invoke("thinker_backbone", "thinker", {"prompt": "Describe color briefly.", "image_size": self.cfg.image_size, "max_new_tokens": self.cfg.max_new_tokens, "color": [128, 64, 32]}, use_cache=False)
        warm_tmp = torch.zeros(self.cfg.hidden_dim, device=self.synth_device, dtype=torch.float32) @ self._vocoder_proj.float()
        del warm_tmp; gc.collect()
        torch.cuda.synchronize("cuda:0"); torch.cuda.empty_cache()
        self.frozen_vram_baseline["cuda:0"] = int(torch.cuda.memory_allocated("cuda:0"))
        self._rng_torch, self._rng_cuda, self._at_s0 = torch.get_rng_state(), {"cuda:0": torch.cuda.get_rng_state("cuda:0")}, True

    def _cpu_thinker_infer(self, payload: dict[str, Any], _ctx: dict[str, Any]) -> dict[str, Any]:
        sz, toks = int(payload.get("image_size", self.cfg.image_size)), int(payload.get("max_new_tokens", self.cfg.max_new_tokens))
        color = list(payload.get("color", [128, 64, 32]))
        return {"ids": [(sz + toks + c * (i + 1)) % 97 for i, c in enumerate(color[:3])], "logits": [round(((sz % 17) + (i + 1) * 0.35 + (color[i % len(color)] / 255.0)) / (toks + 1), 8) for i in range(self.cfg.hidden_dim)]}

    def _gpu_thinker_infer(self, payload: dict[str, Any], _ctx: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image  # type: ignore[import-not-found]
        from vllm import SamplingParams  # type: ignore[import-not-found]
        sz, toks, color = int(payload.get("image_size", self.cfg.image_size)), int(payload.get("max_new_tokens", self.cfg.max_new_tokens)), tuple(int(c) for c in payload.get("color", [128, 64, 32]))
        img = Image.new("RGB", (sz, sz), color=color)
        prompt = self._processor.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": str(payload.get("prompt", "Describe color briefly."))}]}], tokenize=False, add_generation_prompt=True)
        sp = SamplingParams(temperature=0.0, max_tokens=toks, logprobs=self.cfg.hidden_dim, seed=self.seed)
        out = self._llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], sp, use_tqdm=False)[0].outputs[0]
        lp_vals = sorted((lp.logprob for lp in out.logprobs[-1].values()), reverse=True) if out.logprobs else []
        padded = (lp_vals + [0.0] * self.cfg.hidden_dim)[:self.cfg.hidden_dim]
        return {"ids": [int(x) for x in out.token_ids], "logits": [round(float(v), 6) for v in padded]}

    def reset_state(self) -> list[str]:
        """Reset S5 stage replay buffers, RNG, and CUDA cache while keeping S4 frozen stages resident."""
        if self._at_s0 and len(self.stage_buffer) == 0:
            return list(self._last_reset)
        self._at_s0 = False; random.setstate(self._rng_py)
        items = ["stage_buffers_s0", f"buffer_epoch_{self.stage_buffer.reset_state().epoch}", "rng"]
        if self.mode == "real":
            import torch  # type: ignore[import-not-found]
            if self._rng_torch is not None:
                torch.set_rng_state(self._rng_torch)
            for dev, state in (self._rng_cuda or {}).items():
                torch.cuda.set_rng_state(state, dev); torch.cuda.synchronize(dev); torch.cuda.empty_cache()
            items.append("cuda_cache")
        gc.collect(); self._at_s0, self._last_reset = True, items
        return list(items)

    def run_pipeline(self, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute Thinker -> Talker -> DiT -> Vocoder with RO stage leases and S0 buffer fence."""
        if not self._at_s0:
            self.reset_state()
        self._at_s0, wl = False, dict(workload or {})
        req = {"prompt": str(wl.get("prompt", "Describe color briefly.")), "image_size": int(wl.get("image_size", self.cfg.image_size)), "max_new_tokens": int(wl.get("max_new_tokens", self.cfg.max_new_tokens)), "color": list(wl.get("color", [128, 64, 32]))}
        t_rcpt = self.aux.invoke("thinker_backbone", "thinker", req, use_cache=bool(wl.get("use_aux_cache", self.use_aux_cache)))
        backbone = dict(t_rcpt.output)
        if self.mode == "real":
            import torch  # type: ignore[import-not-found]
            backbone["logits"] = torch.tensor(backbone["logits"], dtype=torch.float32, device=self.synth_device)
        thinker_cond = thinker_op.thinker_stage(backbone, self.cfg)
        talker_toks = talker_op.talker_stage(thinker_cond, self.cfg)
        dit_latents = dit_op.dit_stage(talker_toks, self.cfg)
        v_rcpt = self.aux.invoke("vocoder_backbone", "vocoder", {"stage": "vocoder"}, use_cache=True)
        waveform = vocoder_op.vocoder_stage(dit_latents, v_rcpt.output, self.cfg)
        dit_list = [float(x) for x in (dit_latents.detach().cpu().tolist() if hasattr(dit_latents, "shape") else dit_latents)]
        del backbone, thinker_cond, talker_toks, dit_latents
        output = assemble_omni_output(t_rcpt.output["ids"], waveform, dit_list)
        self.stage_buffer.push({"digest": output["digest"], "image_size": req["image_size"]})
        vram_now: dict[str, int] = {}
        if self.mode == "real":
            import torch  # type: ignore[import-not-found]
            torch.cuda.synchronize("cuda:0"); vram_now["cuda:0"] = int(torch.cuda.memory_allocated("cuda:0"))
        return {"output": output, "telemetry": {"devices": list(self.devices), "thinker_cache_hit": t_rcpt.cache_hit, "frozen_vram_baseline": dict(self.frozen_vram_baseline), "frozen_vram_now": vram_now, "aux_stats": self.aux.stats(), "buffer_stats": self.stage_buffer.stats()}}

    def close(self) -> None:
        if self._llm is not None:
            with contextlib.suppress(Exception):
                ec = getattr(getattr(self._llm, "llm_engine", None), "engine_core", None)
                if ec is not None and hasattr(ec, "shutdown"):
                    ec.shutdown()
            self._llm = self._processor = self._vocoder_proj = None
            gc.collect()
        for fn in (self.stage_buffer.close, self.aux.teardown, self.memory.teardown):
            with contextlib.suppress(Exception):
                fn()


def serve_worker(sock_path: str, config: dict[str, Any]) -> int:
    """Run Unix-socket JSON worker hosting 4 stage RoleEndpoints (thinker/talker/dit/vocoder)."""
    if (parent_pid := int(config.get("parent_pid") or os.getppid())) and os.getppid() != parent_pid:
        return 0
    worker: VllmOmniPipelineWorker | None = None
    server: ipc.UnixJsonServer | None = None
    try:
        worker = VllmOmniPipelineWorker(config)
        for mod in config.get("modules") or ():
            importlib.import_module(mod)
        endpoints = {r: RoleEndpoint(r, scope={"worker": worker, "cfg": worker.cfg}, resets={"state": worker.reset_state}) for r in STAGE_ROLES}
        faults: dict[str, Any] = {}
        server = ipc.UnixJsonServer(sock_path, handler=None)
        while True:
            try:
                req, reply_fn = server.next(timeout=1.0)
            except queue.Empty:
                if parent_pid and os.getppid() != parent_pid:
                    return 0
                continue
            method, payload, stage = str(req.get("method") or ""), req.get("payload"), str(req.get("stage") or "talker")
            try:
                rep, stop = _handle_rpc(worker, endpoints, faults, stage, method, payload)
            except Exception as exc:
                rep, stop = findings.not_ok(f"{method}: {type(exc).__name__}: {exc}"), False
            reply_fn(rep)
            if stop:
                return 0
    finally:
        if server is not None:
            server.close()
        if worker is not None:
            worker.close()


def _handle_rpc(worker: VllmOmniPipelineWorker, endpoints: dict[str, RoleEndpoint], faults: dict[str, Any], stage: str, method: str, payload: Any) -> tuple[dict[str, Any], bool]:
    if method == "shutdown":
        return {"ok": True}, True
    if method == "ping":
        return {"ok": True, "pid": os.getpid(), "devices": worker.devices, "mode": worker.mode}, False
    if method == "inspect_source":
        mod = sys.modules.get(str((payload or {}).get("module") or ""))
        return {"ok": True, "source": inspect.getsource(getattr(mod, str((payload or {}).get("attr") or "")))}, False
    if method == "run_pipeline":
        if faults.pop("crash", False):
            os._exit(139)
        if hang_s := faults.pop("hang_s", 0.0):
            time.sleep(float(hang_s))
        return {"ok": True, **worker.run_pipeline(payload)}, False
    if method == "inject":
        info, fault = dict(payload or {}), str((payload or {}).get("fault") or ""); info.pop("fault", None)
        if fault in ("crash", "hang", "lost"):
            faults["crash" if fault == "crash" else "hang_s"] = True if fault == "crash" else float(info.get("hang_s", 5.0))
            return {"ok": True, "fault": fault}, False
        return harness.inject_participant_fault(endpoints[stage].lp, fault, **info), False
    if method not in PARTICIPANT_METHODS:
        return findings.not_ok(f"unknown worker method {method!r}"), False
    if method == "reset" and payload == "state":
        return {**endpoints[stage].lp.reset(payload), "pid": os.getpid(), "role": stage}, False
    return endpoints[stage].dispatch(method, payload), False


def main(argv: list[str] | None = None) -> int:
    with open((argv or sys.argv[1:])[0], encoding="utf-8") as fh:
        spec = json.load(fh)
    if not spec.get("fresh"):
        return serve_worker(spec["sock_path"], spec["config"])
    worker = VllmOmniPipelineWorker(spec["config"])
    try:
        res = worker.run_pipeline(spec.get("workload"))
    finally:
        worker.close()
    print("FRESH " + json.dumps({"output": res["output"]}), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
