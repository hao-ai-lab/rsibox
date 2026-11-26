"""Persistent worker subprocess for Standalone Kernel Engine (Triton, CUDA `.cu`, & CPU)."""
from __future__ import annotations

import contextlib, hashlib, importlib, importlib.util, json, linecache, math, os, queue, random, sys, time
from typing import Any

from adapter import kernel_ops
from kernel import coordinator, findings
from rsibox import harness, ipc
from rsibox.lab import ENDPOINT_METHODS

PARTICIPANT_METHODS = frozenset(ENDPOINT_METHODS)
TIMING_KEYS = ("latency_ms", "compile_ms", "cuda_so_key", "cuda_cached")
_STICKY_FAULT_CU = '#include <cuda_runtime.h>\nextern "C" __global__ void bad_cu(int* p){if(!threadIdx.x)*p=1;}\nextern "C" int launch_fused_affine_silu(const float*,const float*,float*,int,float,float,int,void* s){bad_cu<<<1,1,0,(cudaStream_t)s>>>((int*)0x1);return (int)cudaGetLastError();}\n'


def validate_positive_int(value: Any, name: str) -> int:
    """Validate strict positive integer configuration fields."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1 (got {value!r})")
    return value


def split_kernel_output(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate bit-exact output fields from wall-clock timing metadata."""
    return {k: v for k, v in raw.items() if k not in TIMING_KEYS}, {k: raw[k] for k in TIMING_KEYS if k in raw}


def resolve_backend(backend: str = "auto") -> str:
    """Resolve `'auto'` backend lazily via `find_spec('triton')` without importing GPU runtime."""
    return ("triton" if importlib.util.find_spec("triton") is not None else "cuda") if backend == "auto" else backend


def require_kernel_deps(backend: str = "auto", device: str = "cuda:0") -> dict[str, str]:
    """Verify PyTorch/Triton/nvcc/CUDA runtime lazily; fail closed when absent."""
    if backend == "cpu": return {"backend": "cpu", "device": "cpu"}
    if importlib.util.find_spec("torch") is None: raise RuntimeError("KernelLab GPU mode requires torch")
    import torch  # type: ignore[import-not-found]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for KernelLab but torch.cuda.is_available() is False")
    resolved = resolve_backend(backend)
    if resolved == "triton" and importlib.util.find_spec("triton") is None: raise RuntimeError("KernelLab triton backend requires triton")
    if resolved == "cuda": kernel_ops.find_nvcc()
    elif resolved != "triton": raise ValueError(f"unsupported backend {backend!r}")
    return {"backend": resolved, "torch": getattr(torch, "__version__", "unknown"), "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu"}


def summarize_kernel_output(vals: list[float], ref_vals: list[float], meta: dict[str, Any]) -> dict[str, Any]:
    """Summarize output tensor/vector vs reference baseline into deterministic bit-exact JSON."""
    summary_scale = 1.0
    scaled = [round(v * summary_scale, 6) for v in vals]
    max_diff = round(max((abs(v - r) for v, r in zip(vals, ref_vals)), default=0.0), 6)
    return {**meta, "max_abs_diff": max_diff, "sample": scaled[:8], "total": round(sum(scaled), 6), "digest": hashlib.sha256(json.dumps(scaled).encode("utf-8")).hexdigest()[:16]}


def _sync_linecache(part: Any, modules: Any) -> None:
    for mod in modules:
        with contextlib.suppress(Exception):
            if fpath := (getattr(sys.modules.get(mod), "__file__", None) or harness.module_file(mod)):
                if isinstance(src := part._live_source(mod), str): linecache.cache[fpath] = (len(src), None, src.splitlines(True), fpath)
                else: linecache.cache.pop(fpath, None)


class KernelWorker:
    """Holds resident GPU CUDA context / seeded reference inputs across live kernel swaps."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.numel = validate_positive_int(int(config.get("numel", 256 if config.get("backend") == "cpu" else 4096)), "numel")
        self.block_size, self.seed = validate_positive_int(int(config.get("block_size", 128)), "block_size"), int(config.get("seed", 42))
        self.iters, req = validate_positive_int(int(config.get("iters", 5)), "iters"), str(config.get("backend", "auto"))
        self.device, self.build_root, self.cuda_cache = str(config.get("device", "cpu" if req == "cpu" else "cuda:0")), config.get("build_root"), bool(config.get("cuda_cache", True))
        self.backend = require_kernel_deps(req, self.device)["backend"]
        self.cfg = kernel_ops.KernelConfig(self.numel, self.block_size, self.seed, self.backend)
        self._shapes, self._cu_state, self._cu_by_src, self.step_count, self.poisoned_cuda = {}, {}, {}, 0, False
        self._rng_py = self._rng_torch = self._rng_cuda = None
        self._init_backend()

    def _get_shape_buffers(self, numel: int) -> tuple[Any, Any, Any, Any, list[float]]:
        if numel in self._shapes: return self._shapes[numel]
        if self.backend == "cpu":
            rng = random.Random(self.seed + numel)
            xs, bias = [round(rng.uniform(-1.5, 1.5), 6) for _ in range(numel)], [round(rng.uniform(-0.25, 0.25), 6) for _ in range(numel)]
            entry = (xs, bias, None, None, [(z / (1.0 + math.exp(-z)) if z >= -60.0 else 0.0) for xv, bv in zip(xs, bias) for z in (xv * 1.5 + bv,)])
        else:
            import torch  # type: ignore[import-not-found]
            gen = torch.Generator(device=self.device).manual_seed(self.seed + numel)
            x, b = (torch.rand(numel, device=self.device, dtype=torch.float32, generator=gen) * 3.0) - 1.5, (torch.rand(numel, device=self.device, dtype=torch.float32, generator=gen) * 0.5) - 0.25
            out, z = torch.empty_like(x), x * 1.5 + b
            ref_t = z * torch.sigmoid(z)
            entry = (x, b, out, ref_t, [float(v) for v in ref_t.detach().cpu().tolist()])
        return self._shapes.setdefault(numel, entry)

    def _init_backend(self) -> None:
        random.seed(self.seed); self._rng_py = random.getstate()
        x, b, out, _, _ = self._get_shape_buffers(self.numel)
        if self.backend == "cpu": return
        import torch  # type: ignore[import-not-found]
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
            torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        self._rng_torch, self._rng_cuda = torch.get_rng_state(), (torch.cuda.get_rng_state(self.device) if self.device.startswith("cuda") else None)
        with (torch.cuda.device(self.device) if self.device.startswith("cuda") else contextlib.nullcontext()):
            if self.backend == "triton":
                import adapter.triton_kernels as tk  # type: ignore[import-not-found]
                tk.launch_triton_affine_silu(x, b, out, block_size=self.block_size)
            elif self.backend == "cuda":
                src = kernel_ops.cuda_kernel_source()
                built = self._cu_by_src.setdefault(src, kernel_ops.compile_and_load_cuda(src, build_root=self.build_root, device=self.device, use_cache=self.cuda_cache))
                self._cu_state = {"source": src, **built}
                if (rc := built["lib"].launch_fused_affine_silu(x.data_ptr(), b.data_ptr(), out.data_ptr(), self.numel, 1.5, 0.0, self.block_size, torch.cuda.current_stream(self.device).cuda_stream)) != 0:
                    raise RuntimeError(f"CUDA warmup launch error {rc}")
            torch.cuda.synchronize(self.device)

    def reset_state(self) -> list[str]:
        """Restore deterministic S0 buffers and RNG state without destroying the CUDA context."""
        self.step_count = 0; random.setstate(self._rng_py); cleared = ["kernel_s0", "rng", "step_counter"]
        if self.backend != "cpu":
            import torch  # type: ignore[import-not-found]
            torch.set_rng_state(self._rng_torch)
            if self._rng_cuda is not None: torch.cuda.set_rng_state(self._rng_cuda, self.device)
            for _, _, out, _, _ in self._shapes.values():
                if out is not None: out.zero_()
            cleared.append("gpu_out_buffers")
        return cleared

    def run_kernel(self, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one deterministic multi-iteration kernel probe and reference diff."""
        wl = workload or {}
        numel, block_size, iters = validate_positive_int(int(wl.get("numel", self.numel)), "numel"), validate_positive_int(int(wl.get("block_size", self.block_size)), "block_size"), validate_positive_int(int(wl.get("iters", self.iters)), "iters")
        alpha, beta, (x, b, out, _, ref_vals), compile_ms, cu_key, cu_cached = float(wl.get("alpha", 1.5 * self.cfg.gain)), float(wl.get("beta", 0.0)), self._get_shape_buffers(numel), 0.0, None, None
        if self.backend == "cpu":
            t0, vals = time.perf_counter(), []
            for _ in range(iters): vals = kernel_ops.run_cpu_kernel(x, kernel_ops.prepare_cpu_bias(b), alpha=alpha, beta=beta)
        else:
            import torch  # type: ignore[import-not-found]
            with (torch.cuda.device(self.device) if self.device.startswith("cuda") else contextlib.nullcontext()):
                if self.backend == "triton":
                    import adapter.triton_kernels as tk  # type: ignore[import-not-found]
                    t0 = time.perf_counter()
                    for _ in range(iters): tk.launch_triton_affine_silu(x, b, out, alpha=alpha, beta=beta, block_size=block_size)
                else:
                    src = str(wl.get("cuda_source") or kernel_ops.cuda_kernel_source())
                    if src != self._cu_state.get("source"):
                        if src in self._cu_by_src: built, compile_ms, cu_cached = self._cu_by_src[src], 0.0, True
                        else: built = self._cu_by_src[src] = kernel_ops.compile_and_load_cuda(src, build_root=self.build_root, device=self.device, use_cache=self.cuda_cache); compile_ms, cu_cached = built["compile_ms"], built["cached"]
                        self._cu_state = {"source": src, **built}
                    cu_key, (a_val, b_val), lib, stream, t0 = self._cu_state["key"], kernel_ops.cuda_launch_params(alpha, beta), self._cu_state["lib"], torch.cuda.current_stream(self.device).cuda_stream, time.perf_counter()
                    for _ in range(iters):
                        if (rc := lib.launch_fused_affine_silu(x.data_ptr(), b.data_ptr(), out.data_ptr(), numel, a_val, b_val, block_size, stream)) != 0:
                            self.poisoned_cuda = True; raise RuntimeError(f"sticky CUDA launch error {rc}")
                try: torch.cuda.synchronize(self.device)
                except Exception as exc: self.poisoned_cuda = True; raise RuntimeError(f"sticky CUDA error: {type(exc).__name__}: {exc}") from exc
                vals = [float(v) for v in out.detach().cpu().tolist()]
        latency_ms = round(((time.perf_counter() - t0) * 1000.0) / iters, 4); self.step_count += 1
        res = summarize_kernel_output(vals, ref_vals, {"backend": self.backend, "numel": numel, "block_size": block_size, "alpha": alpha, "beta": beta})
        res["latency_ms"] = latency_ms
        if cu_key is not None: res.update({"compile_ms": compile_ms, "cuda_so_key": cu_key, "cuda_cached": cu_cached})
        return res

    def close(self) -> None: self._shapes.clear()

def _dispatch(worker: KernelWorker, part: Any, planned: dict[str, set[str]], tracked: set[str], faults: dict[str, Any], method: str, payload: Any) -> tuple[dict[str, Any], bool]:
    if method in ("shutdown", "ping"):
        return ({"ok": True}, True) if method == "shutdown" else ({"ok": True, "pid": os.getpid(), "backend": worker.backend, "numel": worker.numel}, False)
    if method == "step":
        if faults.pop("crash", False): os._exit(139)
        if hang_s := faults.pop("hang_s", 0.0): time.sleep(float(hang_s))
        if faults.pop("sticky_cuda", False):
            if worker.backend == "cpu": worker.poisoned_cuda = True; return {"ok": False, "poison_worker": True, "error": "sticky CUDA error: illegal memory access"}, False
            import torch  # type: ignore[import-not-found]
            with (torch.cuda.device(worker.device) if worker.device.startswith("cuda") else contextlib.nullcontext()):
                kernel_ops.compile_and_load_cuda(_STICKY_FAULT_CU, build_root=worker.build_root, device=worker.device, use_cache=True)["lib"].launch_fused_affine_silu(0, 0, 0, 1, 1.0, 0.0, 1, torch.cuda.current_stream(worker.device).cuda_stream)
                try: torch.cuda.synchronize(worker.device)
                except Exception as exc: worker.poisoned_cuda = True; return {"ok": False, "poison_worker": True, "error": f"sticky CUDA error: {exc}"}, False
        return {"ok": True, "output": worker.run_kernel(payload or {})}, False
    if method == "inject":
        info, fault = dict(payload or {}), str((payload or {}).get("fault") or "")
        info.pop("fault", None)
        if fault in ("crash", "hang", "lost", "sticky_cuda"):
            faults["crash" if fault == "crash" else ("sticky_cuda" if fault == "sticky_cuda" else "hang_s")] = True if fault in ("crash", "sticky_cuda") else float(info.get("hang_s", 5.0))
            return {"ok": True, "fault": fault}, fault == "lost"
        return harness.inject_participant_fault(part, fault, **info), False
    if method not in PARTICIPANT_METHODS: return findings.not_ok(f"unknown worker method {method!r}"), False
    if method == "prepare":
        mods = (payload or {}).get("modules", {})
        for mod, spec in mods.items():
            if (origin := getattr(sys.modules.get(mod), "__file__", None) or harness.module_file(mod)) and isinstance(src := spec.get("source") if isinstance(spec, dict) else None, str):
                linecache.cache[origin] = (len(src), None, src.splitlines(True), origin)
        reply = part.prepare(payload)
        (planned.__setitem__(str(reply.get("txn")), set(mods)) if reply.get("ok") else _sync_linecache(part, mods))
        return reply, False
    reply = getattr(part, method)(payload)
    if method in ("commit", "abort", "undo"):
        affected = planned.pop(str(coordinator.txn_ref(payload)[0]), set())
        tracked.update(affected if method == "commit" else ())
        _sync_linecache(part, (tracked | affected) if method == "undo" else affected)
    return reply, False

def main(argv: list[str] | None = None) -> int:
    with open((argv or sys.argv[1:])[0], encoding="utf-8") as fh: spec = json.load(fh)
    if spec.get("fresh"):
        if sys.platform.startswith("linux") and spec.get("parent_pid") and os.getppid() != int(spec["parent_pid"]): return 0
        worker = KernelWorker(dict(spec["config"]))
        try: print("FRESH " + json.dumps({"output": split_kernel_output(worker.run_kernel(spec.get("workload") or {}))[0]}), flush=True)
        finally: worker.close()
        return 0
    cfg, parent_pid = spec["config"], int(spec["config"].get("parent_pid") or os.getppid())
    if parent_pid and os.getppid() != parent_pid: return 0
    worker = KernelWorker(cfg)
    for mod in cfg.get("modules") or (): importlib.import_module(mod)
    part, planned, tracked, faults, server = coordinator.LocalParticipant("kernel0", scope={"worker": worker, "cfg": worker.cfg}, resets={"state": worker.reset_state}), {}, set(), {}, ipc.UnixJsonServer(spec["sock_path"], handler=None)
    try:
        while True:
            try: req, reply_fn = server.next(timeout=1.0)
            except queue.Empty:
                if parent_pid and os.getppid() != parent_pid: return 0
                continue
            try: reply, should_exit = _dispatch(worker, part, planned, tracked, faults, str(req.get("method") or ""), req.get("payload"))
            except Exception as exc: reply, should_exit = findings.not_ok(f"{req.get('method')}: {type(exc).__name__}: {exc}", poison_worker=worker.poisoned_cuda), False
            reply_fn(reply)
            if should_exit: return 0
    finally: server.close(); worker.close()

if __name__ == "__main__": raise SystemExit(importlib.import_module("adapter.worker").main())
