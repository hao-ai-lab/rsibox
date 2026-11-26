"""Multi-stage SGLang-Omni worker with S4 frozen stage residency and S5 S0 stage-buffer reset."""
from __future__ import annotations

import contextlib, gc, hashlib, importlib, json, linecache, os, queue, random, sys, time
from typing import Any

from adapter import dit_op, stage_ops, talker_op, thinker_op, vocoder_op
from kernel import coordinator, findings
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


def _sync_linecache(lp: Any, modules: Any) -> None:
    for mod in modules:
        with contextlib.suppress(Exception):
            if not (fpath := getattr(sys.modules.get(mod), "__file__", None) or harness.module_file(mod)):
                continue
            if isinstance(src := lp._live_source(mod), str):
                linecache.cache[fpath] = (len(src), None, src.splitlines(True), fpath)
            else:
                linecache.cache.pop(fpath, None)


class OmniPipelineWorker:
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
        self._model: Any = None; self._processor: Any = None; self._vocoder_proj: Any = None
        self._at_s0, self._last_reset = False, ["stage_buffers_s0", "buffer_epoch_0", "rng"]
        self._init_stages()

    def _init_stages(self) -> None:
        random.seed(self.seed); self._rng_py = random.getstate()
        if self.mode != "real":
            voc_weights = [round(0.2 + 0.05 * i, 4) for i in range(self.cfg.hidden_dim)]
            th = self.memory.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "sglang-omni", "thinker", gpus=(), size_bytes=1024, stamp={"stage": "thinker", "mode": "cpu"})
            vh = self.memory.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "sglang-omni", "vocoder", gpus=(), size_bytes=512, stamp={"stage": "vocoder", "mode": "cpu"})
            self.aux.register_stage("thinker_backbone", kind=AuxStageKind.TEXT_ENCODER, engine="sglang-omni", model_id="toy-thinker", handle_id=th.handle_id, infer_fn=self._cpu_thinker_infer)
            self.aux.register_stage("vocoder_backbone", kind=AuxStageKind.CUSTOM, engine="sglang-omni", model_id="toy-vocoder", handle_id=vh.handle_id, infer_fn=lambda _p, _c: voc_weights)
            self._at_s0 = True
            return
        import torch  # type: ignore[import-not-found]
        if not torch.cuda.is_available():
            raise RuntimeError("mode='real' requires CUDA GPU")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        ndev = torch.cuda.device_count()
        self.thinker_device, self.synth_device = "cuda:0", ("cuda:1" if ndev >= 2 else "cuda:0")
        self.devices = [self.thinker_device] if self.synth_device == self.thinker_device else [self.thinker_device, self.synth_device]
        torch.manual_seed(self.seed)
        for dev in self.devices:
            torch.cuda.set_device(dev); torch.cuda.manual_seed(self.seed)
        self._model, self._processor, self._vocoder_proj = stage_ops.build_gpu_stage_modules(self.cfg, self.model_path, self.thinker_device, self.synth_device)
        th = self.memory.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "sglang-omni", "thinker", gpus=(0,), size_bytes=max(1, sum(p.numel() * p.element_size() for p in self._model.parameters())), stamp={"model": os.path.basename(self.model_path)})
        vh = self.memory.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "sglang-omni", "vocoder", gpus=(1 if ndev >= 2 else 0,), size_bytes=max(1, self._vocoder_proj.numel() * self._vocoder_proj.element_size()), stamp={"stage": "vocoder"})
        self.aux.register_stage("thinker_backbone", kind=AuxStageKind.TEXT_ENCODER, engine="sglang-omni", model_id=os.path.basename(self.model_path), handle_id=th.handle_id, infer_fn=self._gpu_thinker_infer)
        self.aux.register_stage("vocoder_backbone", kind=AuxStageKind.CUSTOM, engine="sglang-omni", model_id="frozen-vocoder", handle_id=vh.handle_id, infer_fn=lambda _p, _c: self._vocoder_proj)
        # Warm CUDA/cuBLAS workspaces with use_cache=False so S4 LRU cache stays cold until c0/first check.
        self.aux.invoke("thinker_backbone", "thinker", {"prompt": "Describe color briefly.", "image_size": self.cfg.image_size, "max_new_tokens": self.cfg.max_new_tokens, "color": [128, 64, 32]}, use_cache=False)
        warm_tmp = torch.zeros(self.cfg.hidden_dim, device=self.synth_device, dtype=torch.float32) @ self._vocoder_proj.float()
        del warm_tmp; gc.collect()
        for dev in self.devices:
            torch.cuda.synchronize(dev); torch.cuda.empty_cache()
            self.frozen_vram_baseline[dev] = int(torch.cuda.memory_allocated(dev))
        self._rng_torch, self._rng_cuda, self._at_s0 = torch.get_rng_state(), {dev: torch.cuda.get_rng_state(dev) for dev in self.devices}, True

    def _cpu_thinker_infer(self, payload: dict[str, Any], _ctx: dict[str, Any]) -> dict[str, Any]:
        sz, toks = int(payload.get("image_size", self.cfg.image_size)), int(payload.get("max_new_tokens", self.cfg.max_new_tokens))
        color = list(payload.get("color", [128, 64, 32]))
        return {"ids": [(sz + toks + c * (i + 1)) % 97 for i, c in enumerate(color[:3])], "logits": [round(((sz % 17) + (i + 1) * 0.35 + (color[i % len(color)] / 255.0)) / (toks + 1), 8) for i in range(self.cfg.hidden_dim)]}

    def _gpu_thinker_infer(self, payload: dict[str, Any], _ctx: dict[str, Any]) -> dict[str, Any]:
        import torch  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
        sz, toks, color = int(payload.get("image_size", self.cfg.image_size)), int(payload.get("max_new_tokens", self.cfg.max_new_tokens)), tuple(int(c) for c in payload.get("color", [128, 64, 32]))
        img = Image.new("RGB", (sz, sz), color=color)
        text = self._processor.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": str(payload.get("prompt", "Describe color briefly."))}]}], tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[img], padding=True, return_tensors="pt").to(self.thinker_device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=toks, do_sample=False, return_dict_in_generate=True, output_scores=True)
        return {"ids": [int(x) for x in out.sequences[0, inputs["input_ids"].shape[1]:].tolist()], "logits": [round(float(v), 6) for v in out.scores[-1][0, : self.cfg.hidden_dim].detach().float().cpu().tolist()]}

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
            for dev in self.devices:
                torch.cuda.synchronize(dev); vram_now[dev] = int(torch.cuda.memory_allocated(dev))
        return {"output": output, "telemetry": {"devices": list(self.devices), "thinker_cache_hit": t_rcpt.cache_hit, "frozen_vram_baseline": dict(self.frozen_vram_baseline), "frozen_vram_now": vram_now, "aux_stats": self.aux.stats(), "buffer_stats": self.stage_buffer.stats()}}

    def close(self) -> None:
        for fn in (self.stage_buffer.close, self.aux.teardown, self.memory.teardown):
            with contextlib.suppress(Exception):
                fn()


def serve_worker(sock_path: str, config: dict[str, Any]) -> int:
    """Run Unix-socket JSON worker hosting 4 stage RoleEndpoints (thinker/talker/dit/vocoder)."""
    if (parent_pid := int(config.get("parent_pid") or os.getppid())) and os.getppid() != parent_pid:
        return 0
    worker = OmniPipelineWorker(config)
    for mod in config.get("modules") or ():
        importlib.import_module(mod)
    endpoints = {r: RoleEndpoint(r, scope={"worker": worker, "cfg": worker.cfg}, resets={"state": worker.reset_state}) for r in STAGE_ROLES}
    planned: dict[tuple[str, str], set[str]] = {}; tracked: dict[str, set[str]] = {r: set() for r in STAGE_ROLES}; faults: dict[str, Any] = {}
    server = ipc.UnixJsonServer(sock_path, handler=None)
    try:
        while True:
            try:
                req, reply_fn = server.next(timeout=1.0)
            except queue.Empty:
                if parent_pid and os.getppid() != parent_pid:
                    return 0
                continue
            method, payload, stage = str(req.get("method") or ""), req.get("payload"), str(req.get("stage") or "talker")
            try:
                rep, stop = _handle_rpc(worker, endpoints, planned, tracked, faults, stage, method, payload)
            except Exception as exc:
                rep, stop = findings.not_ok(f"{method}: {type(exc).__name__}: {exc}"), False
            reply_fn(rep)
            if stop:
                return 0
    finally:
        server.close(); worker.close()


def _handle_rpc(worker: OmniPipelineWorker, endpoints: dict[str, RoleEndpoint], planned: dict[tuple[str, str], set[str]], tracked: dict[str, set[str]], faults: dict[str, Any], stage: str, method: str, payload: Any) -> tuple[dict[str, Any], bool]:
    if method == "shutdown":
        return {"ok": True}, True
    if method == "ping":
        return {"ok": True, "pid": os.getpid(), "devices": worker.devices, "mode": worker.mode}, False
    if method == "run_pipeline":
        if faults.pop("crash", False): os._exit(139)
        if hang_s := faults.pop("hang_s", 0.0): time.sleep(float(hang_s))
        return {"ok": True, **worker.run_pipeline(payload)}, False
    if method == "inject":
        info, fault = dict(payload or {}), str((payload or {}).get("fault") or ""); info.pop("fault", None)
        if fault in ("crash", "hang", "lost"):
            faults["crash" if fault == "crash" else "hang_s"] = True if fault == "crash" else float(info.get("hang_s", 5.0))
            return {"ok": True, "fault": fault}, False
        return harness.inject_participant_fault(endpoints[stage].lp, fault, **info), False
    if method not in PARTICIPANT_METHODS: return findings.not_ok(f"unknown worker method {method!r}"), False
    if method == "reset" and payload == "state": return {**endpoints[stage].lp.reset(payload), "pid": os.getpid(), "role": stage}, False
    ep = endpoints[stage]
    if method == "prepare" and isinstance(payload, dict):
        mods = payload.get("modules") or {}
        for mod, spec in mods.items():
            origin = getattr(sys.modules.get(mod), "__file__", None) or harness.module_file(mod)
            if origin and isinstance(spec, dict) and isinstance(src := spec.get("source"), str): linecache.cache[origin] = (len(src), None, src.splitlines(True), origin)
        rep = ep.dispatch(method, payload)
        if rep.get("ok") and not rep.get("abstain") and rep.get("txn") is not None: planned[(stage, str(rep["txn"]))] = set(mods)
        elif not rep.get("ok") or rep.get("abstain"): _sync_linecache(ep.lp, mods)
        return rep, False
    rep = ep.dispatch(method, payload)
    if method in ("commit", "abort", "undo"):
        txn_id, _ = coordinator.txn_ref(payload)
        affected = planned.pop((stage, str(txn_id)), set())
        if method == "commit": tracked[stage].update(affected)
        _sync_linecache(ep.lp, (tracked[stage] | affected) if method == "undo" else affected)
    return rep, False

def main(argv: list[str] | None = None) -> int:
    with open((argv or sys.argv[1:])[0], encoding="utf-8") as fh: spec = json.load(fh)
    if not spec.get("fresh"): return serve_worker(spec["sock_path"], spec["config"])
    worker = OmniPipelineWorker(spec["config"])
    try: res = worker.run_pipeline(spec.get("workload"))
    finally: worker.close()
    print("FRESH " + json.dumps({"output": res["output"]}), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
