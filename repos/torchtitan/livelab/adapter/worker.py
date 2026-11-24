"""Step-gap worker subprocess for TorchTitan / FSDP2 RSIBox live labs."""
from __future__ import annotations

import contextlib
import gc, hashlib, importlib, json, linecache, math, os, queue, random, sys, time
from typing import Any

from adapter import model_ops, optim_ops
from kernel import coordinator, findings
from rsibox import harness, ipc
from rsibox.lab import ENDPOINT_METHODS

PARTICIPANT_METHODS = frozenset(ENDPOINT_METHODS)
FSDP2_SINGLE_RANK_ERR = "fsdp2 mode currently supports world_size=1; multi-rank FSDP2 fresh_check is not yet supported"


def aggregate_ranks(per_rank: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-rank step losses, grad norms, and digest into one JSON output."""
    n_steps, world_size = len(next(iter(per_rank.values()))["losses"]), len(per_rank)
    losses = [round(sum(r["losses"][i] for r in per_rank.values()) / world_size, 8) for i in range(n_steps)]
    grad_norms = [round(sum(r["grad_norms"][i] for r in per_rank.values()) / world_size, 8) for i in range(n_steps)]
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

def _seed_all(seed: int, use_torch: bool) -> None:
    random.seed(seed)
    if use_torch:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class StepGapTrainer:
    """Owns one rank's training state, step-gap barrier hook, and deterministic `S0` snapshot."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.mode, self.rank, self.world_size = str(config.get("mode", "cpu")), int(config.get("rank", 0)), int(config.get("world_size", 1))
        if self.mode == "fsdp2" and self.world_size > 1:
            raise ValueError(FSDP2_SINGLE_RANK_ERR)
        self.seed, self.step, self.pg_initialized = int(config.get("seed", 0)), 0, False
        self.use_torch, self.use_fsdp2 = self.mode in ("fsdp2", "cuda", "torch_cpu"), self.mode == "fsdp2"
        self.cfg = model_ops.TitanStepConfig(int(config.get("hidden", 16)), int(config.get("ffn", 32)), int(config.get("layers", 2)), float(config.get("lr", 0.01)))
        self._init_backend(config)

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
        _seed_all(self.seed, True)
        if self.mode in ("fsdp2", "cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
            local_rank = int(config.get("local_rank", self.rank % torch.cuda.device_count()))
            torch.cuda.set_device(local_rank)
            self.device = f"cuda:{local_rank}"
        else:
            self.device = "cpu"
        if self.use_fsdp2 and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{int(config.get('master_port', 29500))}", rank=self.rank, world_size=self.world_size)
            self.pg_initialized = True
        self.model = model_ops.build_torch_model(self.cfg, self.device, self.use_fsdp2)
        self.optimizer, self._s0_params = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr), [_local(p).detach().clone() for p in self.model.parameters()]
        gen, batch = torch.Generator(device="cpu").manual_seed(self.seed + 500 + self.rank), int(config.get("batch_size", 8))
        self.x_batch, self.y_batch = torch.randn(batch, self.cfg.hidden, generator=gen).to(self.device), torch.randn(batch, self.cfg.hidden, generator=gen).to(self.device)
        _seed_all(self.seed + self.rank, True)
        self._rng_py, self._rng_torch, self._rng_cuda = random.getstate(), torch.get_rng_state(), (torch.cuda.get_rng_state_all() if self.device.startswith("cuda") else None)

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
            torch.cuda.set_rng_state_all(self._rng_cuda)
            torch.cuda.synchronize()
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
            torch.cuda.synchronize()
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
    parent_pid, trainer = int(config.get("parent_pid") or os.getppid()), StepGapTrainer(config)
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

def run_fresh(spec: dict[str, Any]) -> int:
    cfg, n_steps, per_rank = dict(spec["config"]), int((spec.get("workload") or {}).get("steps", 2)), {}
    if str(cfg.get("mode", "cpu")) == "fsdp2" and int(cfg.get("world_size", 1)) > 1:
        raise ValueError(FSDP2_SINGLE_RANK_ERR)
    for rank in range(int(cfg.get("world_size", 1))):
        trainer = StepGapTrainer({**cfg, "rank": rank})
        try:
            per_rank[f"rank{rank}"] = trainer.run_steps(n_steps)
        finally:
            trainer.close()
    print("FRESH " + json.dumps({"output": aggregate_ranks(per_rank)}), flush=True)
    return 0

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        spec = json.load(fh)
    raise SystemExit(run_fresh(spec) if spec.get("fresh") else serve_worker(spec["sock_path"], spec["config"]))
