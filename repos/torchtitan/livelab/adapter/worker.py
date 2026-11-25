"""Step-gap worker subprocess for TorchTitan / FSDP2 RSIBox live labs."""
from __future__ import annotations

import contextlib
import ctypes
import gc, hashlib, importlib, json, linecache, math, os, queue, random, signal, subprocess, sys, tempfile, time
from typing import Any

from adapter import model_ops, optim_ops
from kernel import coordinator, findings
from rsibox import harness, ipc, proc
from rsibox.lab import ENDPOINT_METHODS

PARTICIPANT_METHODS = frozenset(ENDPOINT_METHODS)
WORLD_SIZE_ERR = "world_size must be an integer >= 1"
FSDP2_SINGLE_RANK_ERR = WORLD_SIZE_ERR


def validate_world_size(value: Any) -> int:
    """Validate `world_size` as a strict integer >= 1 (rejecting bools and floats)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{WORLD_SIZE_ERR} (got {value!r})")
    if value < 1:
        raise ValueError(f"{WORLD_SIZE_ERR} (got {value!r})")
    return value


def _validate_timeout(value: Any, default: float = 180.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"timeout must be a positive finite number (got {value!r})")
    timeout_s = float(value)
    if not (math.isfinite(timeout_s) and timeout_s > 0.0):
        raise ValueError(f"timeout must be a positive finite number (got {value!r})")
    return timeout_s


def _set_pdeathsig(parent_pid: int | None = None) -> None:
    """Arm Linux PR_SET_PDEATHSIG(SIGTERM) and close the parent-died-before-prctl race.

    Call only in short-lived child processes whose parent thread blocks until exit (e.g.
    `_run_fresh_rank`), because Linux binds PR_SET_PDEATHSIG to the spawning OS thread.
    """
    if sys.platform.startswith("linux"):
        with contextlib.suppress(Exception):
            libc = ctypes.CDLL(None)
            libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
            libc.prctl.restype = ctypes.c_int
            libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)
    if parent_pid and os.getppid() != int(parent_pid):
        os._exit(1)


def aggregate_ranks(per_rank: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-rank step losses, grad norms, and digest into one JSON output."""
    ordered = [per_rank[k] for k in sorted(per_rank)]
    n_steps, world_size = len(ordered[0]["losses"]), len(ordered)
    losses = [round(sum(r["losses"][i] for r in ordered) / world_size, 8) for i in range(n_steps)]
    grad_norms = [round(sum(r["grad_norms"][i] for r in ordered) / world_size, 8) for i in range(n_steps)]
    digest = hashlib.sha256(json.dumps(per_rank, sort_keys=True).encode()).hexdigest()[:16]
    return {"losses": losses, "grad_norms": grad_norms, "per_rank": per_rank, "digest": digest}


def _sync_linecache(participant: Any, module_names: Any) -> None:
    for module in module_names:
        with contextlib.suppress(Exception):
            fpath = getattr(sys.modules.get(module), "__file__", None) or harness.module_file(module)
            if not fpath:
                continue
            src = participant._live_source(module)
            if isinstance(src, str):
                linecache.cache[fpath] = (len(src), None, src.splitlines(True), fpath)
            else:
                linecache.cache.pop(fpath, None)


def _local(tensor: Any) -> Any:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor

def _seed_all(seed: int, use_torch: bool, device: str | None = None) -> None:
    random.seed(seed)
    if use_torch:
        import torch
        torch.manual_seed(seed)
        if device is not None and device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(seed)


class StepGapTrainer:
    """Owns one rank's training state, step-gap barrier hook, and deterministic `S0` snapshot."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.mode, self.rank, self.world_size = (
            str(config.get("mode", "cpu")),
            int(config.get("rank", 0)),
            validate_world_size(config.get("world_size", 1)),
        )
        self.seed, self.step, self.pg_initialized = int(config.get("seed", 0)), 0, False
        self.use_torch, self.use_fsdp2 = self.mode in ("fsdp2", "cuda", "torch_cpu"), self.mode == "fsdp2"
        self.cfg = model_ops.TitanStepConfig(int(config.get("hidden", 16)), int(config.get("ffn", 32)), int(config.get("layers", 2)), float(config.get("lr", 0.01)))
        try:
            self._init_backend(config)
        except BaseException:
            self.close()
            raise

    def _init_backend(self, config: dict[str, Any]) -> None:
        if not self.use_torch:
            rng, sc, h, f = random.Random(self.seed + 100), self.cfg.weight_scale, self.cfg.hidden, self.cfg.ffn
            self.w1, self.w2 = [[rng.uniform(-sc, sc) for _ in range(h)] for _ in range(f)], [[rng.uniform(-sc, sc) for _ in range(f)] for _ in range(h)]
            self._s0_w1, self._s0_w2, brng = [list(r) for r in self.w1], [list(r) for r in self.w2], random.Random(self.seed + 500 + self.rank)
            self.x_batch, self.y_batch, self.device = [[brng.uniform(-1.0, 1.0) for _ in range(h)] for _ in range(4)], [[brng.uniform(-0.5, 0.5) for _ in range(h)] for _ in range(4)], "cpu"
            _seed_all(self.seed + self.rank, False)
            self._rng_py = random.getstate()
            return
        import torch, torch.distributed as dist
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if self.mode in ("fsdp2", "cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
            device_count = torch.cuda.device_count()
            if self.world_size > device_count:
                raise RuntimeError(f"world_size={self.world_size} exceeds visible CUDA device count ({device_count})")
            local_rank = int(config.get("local_rank", self.rank))
            if local_rank < 0 or local_rank >= device_count:
                raise RuntimeError(f"local_rank={local_rank} out of range for visible CUDA device count ({device_count})")
            torch.cuda.set_device(local_rank)
            self.device = f"cuda:{local_rank}"
        else:
            self.device = "cpu"
        _seed_all(self.seed, True, self.device)
        if self.use_fsdp2 and not dist.is_initialized():
            pg_kwargs: dict[str, Any] = {
                "backend": "nccl",
                "init_method": f"tcp://127.0.0.1:{int(config.get('master_port', 29500))}",
                "rank": self.rank,
                "world_size": self.world_size,
            }
            try:
                dist.init_process_group(**pg_kwargs, device_id=torch.device(self.device))
            except TypeError:
                dist.init_process_group(**pg_kwargs)
            self.pg_initialized = True
        self.model = model_ops.build_torch_model(self.cfg, self.device, self.use_fsdp2)
        self.optimizer, self._s0_params = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr), [_local(p).detach().clone() for p in self.model.parameters()]
        gen, batch = torch.Generator(device="cpu").manual_seed(self.seed + 500 + self.rank), int(config.get("batch_size", 8))
        self.x_batch, self.y_batch = torch.randn(batch, self.cfg.hidden, generator=gen).to(self.device), torch.randn(batch, self.cfg.hidden, generator=gen).to(self.device)
        _seed_all(self.seed + self.rank, True, self.device)
        self._rng_py, self._rng_torch, self._rng_cuda = random.getstate(), torch.get_rng_state(), (torch.cuda.get_rng_state(self.device) if self.device.startswith("cuda") else None)

    def reset_state(self) -> list[str]:
        """Restore exact post-init `S0` weights, optimizer buffers, step counter, and RNG."""
        self.step = 0
        random.setstate(self._rng_py)
        if not self.use_torch:
            self.w1, self.w2 = [list(r) for r in self._s0_w1], [list(r) for r in self._s0_w2]
            return ["weights_s0", "optimizer_state", "rng", "step_counter"]
        import torch
        with torch.no_grad():
            for param, saved in zip(self.model.parameters(), self._s0_params):
                _local(param).copy_(saved)
                param.grad = None
        self.optimizer.state.clear()
        for group in self.optimizer.param_groups:
            group["lr"] = self.cfg.lr
        torch.set_rng_state(self._rng_torch)
        if self._rng_cuda is not None:
            torch.cuda.set_rng_state(self._rng_cuda, self.device)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        gc.collect()
        return ["weights_s0", "optimizer_state", "rng", *(["cuda_cache"] if self._rng_cuda is not None else []), "step_counter"]

    def _cpu_step(self, step_idx: int) -> tuple[float, float]:
        pred = model_ops.mlp_forward(self.x_batch, self.w1, self.w2)
        loss, n_elem = float(model_ops.compute_loss(pred, self.y_batch)), max(len(pred) * len(pred[0]), 1)
        g_w2, g_w1 = [[0.0] * len(self.w2[0]) for _ in self.w2], [[0.0] * len(self.w1[0]) for _ in self.w1]
        for p_row, t_row, x_row in zip(pred, self.y_batch, self.x_batch):
            h_row = [d / (1.0 + math.exp(-max(-20.0, min(20.0, d)))) for d in (sum(a * b for a, b in zip(x_row, c)) for c in self.w1)]
            d_out = [2.0 * (p - t) / n_elem for p, t in zip(p_row, t_row)]
            for o, dv in enumerate(d_out):
                g_w2[o] = [g + dv * hv for g, hv in zip(g_w2[o], h_row)]
            for j in range(len(self.w1)):
                dh = sum(d_out[o] * self.w2[o][j] for o in range(len(self.w2))) * 0.25
                g_w1[j] = [g + dh * xv for g, xv in zip(g_w1[j], x_row)]
        grad_norm, lr = float(optim_ops.clip_and_scale_grad([g_w1, g_w2])), float(optim_ops.optimizer_step_scale(step_idx, self.cfg.lr))
        for w, g in ((self.w1, g_w1), (self.w2, g_w2)):
            w[:] = [[wv - lr * gv for wv, gv in zip(wr, gr)] for wr, gr in zip(w, g)]
        return loss, grad_norm

    def _torch_step(self, step_idx: int) -> tuple[float, float]:
        import torch
        self.optimizer.zero_grad(set_to_none=True)
        loss = model_ops.compute_loss(self.model(self.x_batch), self.y_batch)
        loss.backward()
        grad_norm, lr = float(optim_ops.clip_and_scale_grad({"params": list(self.model.parameters())})), float(optim_ops.optimizer_step_scale(step_idx, self.cfg.lr))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return float(loss.detach().item()), grad_norm

    def run_steps(self, n_steps: int, *, reset_to_s0: bool = True) -> dict[str, Any]:
        """Run `n_steps` training iterations (`forward -> loss -> backward -> optimizer.step`)."""
        if reset_to_s0 and self.step != 0:
            self.reset_state()
        losses, grad_norms = [], []
        for _ in range(max(1, int(n_steps))):
            loss, gnorm = self._torch_step(self.step) if self.use_torch else self._cpu_step(self.step)
            losses.append(round(loss, 8))
            grad_norms.append(round(gnorm, 8))
            self.step += 1
        chk = [round(float(_local(p).detach().float().sum().item()), 6) for p in self.model.parameters()] if self.use_torch else [round(sum(sum(r) for r in self.w1), 6), round(sum(sum(r) for r in self.w2), 6)]
        digest = hashlib.sha256(json.dumps([losses, grad_norms, chk]).encode()).hexdigest()[:16]
        return {"rank": self.rank, "mode": self.mode, "fsdp2": self.use_fsdp2, "device": self.device, "step": self.step, "losses": losses, "grad_norms": grad_norms, "digest": digest}

    def close(self) -> None:
        if self.pg_initialized:
            with contextlib.suppress(Exception):
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
            self.pg_initialized = False


def _dispatch(trainer: StepGapTrainer, participant: Any, planned: dict[str, set[str]], tracked: set[str], faults: dict[str, Any], method: str, payload: Any) -> tuple[dict[str, Any], bool]:
    if method == "shutdown":
        return {"ok": True}, True
    if method == "ping":
        return {"ok": True, "pid": os.getpid(), "rank": trainer.rank, "mode": trainer.mode, "device": trainer.device}, False
    if method == "step":
        if faults.pop("crash", False):
            os._exit(139)
        if hang_s := faults.pop("hang_s", 0.0):
            time.sleep(float(hang_s))
        wl = payload or {}
        return {"ok": True, "output": trainer.run_steps(int(wl.get("steps", 2)), reset_to_s0=not bool(wl.get("continue_state", False)))}, False
    if method == "inject":
        info, fault = dict(payload or {}), str((payload or {}).get("fault") or "")
        info.pop("fault", None)
        if fault in ("crash", "hang", "lost"):
            faults["crash" if fault == "crash" else "hang_s"] = True if fault == "crash" else float(info.get("hang_s", 5.0))
            return {"ok": True, "fault": fault}, fault == "lost"
        return harness.inject_participant_fault(participant, fault, **info), False
    if method not in PARTICIPANT_METHODS:
        return findings.not_ok(f"unknown worker method {method!r}"), False
    if method == "prepare":
        modules = (payload or {}).get("modules", {})
        for module, spec in modules.items():
            origin, src = getattr(sys.modules.get(module), "__file__", None) or harness.module_file(module), spec.get("source") if isinstance(spec, dict) else None
            if origin and isinstance(src, str):
                linecache.cache[origin] = (len(src), None, src.splitlines(True), origin)
        reply = participant.prepare(payload)
        if reply.get("ok"):
            planned[str(reply.get("txn"))] = set(modules)
        else:
            _sync_linecache(participant, modules)
        return reply, False
    reply = getattr(participant, method)(payload)
    if method in ("commit", "abort", "undo"):
        txn_id, _ = coordinator.txn_ref(payload)
        affected = planned.pop(str(txn_id), set())
        tracked.update(affected if method == "commit" else ())
        _sync_linecache(participant, (tracked | affected) if method == "undo" else affected)
    return reply, False


def serve_worker(sock_path: str, config: dict[str, Any]) -> int:
    parent_pid = int(config.get("parent_pid") or os.getppid())
    if parent_pid and os.getppid() != parent_pid:
        return 0
    trainer = StepGapTrainer(config)
    for module in config.get("modules") or ():
        importlib.import_module(module)
    part, planned, tracked, faults, server = coordinator.LocalParticipant(f"rank{trainer.rank}", scope={"trainer": trainer, "cfg": trainer.cfg}, resets={"state": trainer.reset_state}), {}, set(), {}, ipc.UnixJsonServer(sock_path, handler=None)
    try:
        while True:
            try:
                req, reply_fn = server.next(timeout=1.0)
            except queue.Empty:
                if parent_pid and os.getppid() != parent_pid:
                    return 0
                continue
            method = str(req.get("method") or "")
            try:
                reply, should_exit = _dispatch(trainer, part, planned, tracked, faults, method, req.get("payload"))
            except Exception as exc:
                reply, should_exit = findings.not_ok(f"{method}: {type(exc).__name__}: {exc}"), False
            reply_fn(reply)
            if should_exit:
                return 0
    finally:
        server.close()
        trainer.close()


def _run_fresh_rank(spec: dict[str, Any]) -> int:
    _set_pdeathsig(spec.get("parent_pid"))
    cfg, n_steps = dict(spec["config"]), int((spec.get("workload") or {}).get("steps", 2))
    trainer = StepGapTrainer(cfg)
    try:
        out = trainer.run_steps(n_steps)
    finally:
        trainer.close()
    with open(spec["output_path"], "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return 0


def _read_tail(path: str, limit: int = 500) -> str:
    with contextlib.suppress(OSError):
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()[-limit:]
    return ""


def _stop_fresh_children(children: list[dict[str, Any]]) -> None:
    live = [item["proc"] for item in children if item["proc"].poll() is None]
    for child in live:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if all(child.poll() is not None for child in live):
            break
        time.sleep(0.03)
    for child in live:
        if child.poll() is None:
            with contextlib.suppress(Exception):
                proc.stop_tree(child.pid, grace_s=0.5, include_pg=True)
        with contextlib.suppress(Exception):
            child.wait(timeout=1.0)


def _run_fresh_parallel(cfg: dict[str, Any], workload: dict[str, Any], world_size: int, timeout_s: float) -> dict[str, dict[str, Any]]:
    world_size = validate_world_size(world_size)
    timeout_s = _validate_timeout(timeout_s)
    master_port = int(cfg.get("master_port") or proc.free_port())
    parent_pid = os.getpid()
    children: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rsitt-fresh-") as tmpdir:
        try:
            for rank in range(world_size):
                spec_path = os.path.join(tmpdir, f"rank{rank}.spec.json")
                out_path = os.path.join(tmpdir, f"rank{rank}.out.json")
                err_path = os.path.join(tmpdir, f"rank{rank}.err.log")
                rank_cfg = {**cfg, "rank": rank, "local_rank": rank, "world_size": world_size, "master_port": master_port}
                with open(spec_path, "w", encoding="utf-8") as sf:
                    json.dump(
                        {
                            "fresh_rank": True,
                            "parent_pid": parent_pid,
                            "config": rank_cfg,
                            "workload": workload,
                            "output_path": out_path,
                        },
                        sf,
                    )
                with open(err_path, "w", encoding="utf-8") as ef:
                    child = subprocess.Popen(
                        [sys.executable, "-m", "adapter.worker", spec_path],
                        env=dict(os.environ),
                        stdout=subprocess.DEVNULL,
                        stderr=ef,
                        start_new_session=True,
                    )
                children.append({"rank": rank, "proc": child, "out": out_path, "err": err_path})

            deadline = time.monotonic() + timeout_s
            while True:
                pending = 0
                for item in children:
                    rc = item["proc"].poll()
                    if rc is None:
                        pending += 1
                    elif rc != 0:
                        tail = _read_tail(item["err"])
                        raise RuntimeError(f"fresh_check rank{item['rank']} exited (rc={rc}): {tail}")
                if pending == 0:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"fresh_check timed out after {timeout_s} s across {world_size} ranks")
                time.sleep(0.05)

            per_rank: dict[str, dict[str, Any]] = {}
            for item in children:
                with open(item["out"], encoding="utf-8") as of:
                    per_rank[f"rank{item['rank']}"] = json.load(of)
            return per_rank
        finally:
            _stop_fresh_children(children)


def run_fresh(spec: dict[str, Any]) -> int:
    cfg = dict(spec["config"])
    workload = dict(spec.get("workload") or {})
    n_steps = int(workload.get("steps", 2))
    world_size = validate_world_size(cfg.get("world_size", 1))
    timeout_s = _validate_timeout(spec.get("timeout"))
    mode = str(cfg.get("mode", "cpu"))
    if world_size > 1 and (mode in ("fsdp2", "cuda") or bool(spec.get("parallel_fresh", False))):
        per_rank = _run_fresh_parallel(cfg, workload, world_size, timeout_s)
    else:
        per_rank = {}
        for rank in range(world_size):
            trainer = StepGapTrainer({**cfg, "rank": rank, "local_rank": rank})
            try:
                per_rank[f"rank{rank}"] = trainer.run_steps(n_steps)
            finally:
                trainer.close()
    print("FRESH " + json.dumps({"output": aggregate_ranks(per_rank)}), flush=True)
    return 0


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("fresh_rank"):
        raise SystemExit(_run_fresh_rank(spec))
    raise SystemExit(run_fresh(spec) if spec.get("fresh") else serve_worker(spec["sock_path"], spec["config"]))
