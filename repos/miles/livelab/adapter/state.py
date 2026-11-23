"""`code_epoch`-tagged `DataBuffer` drain and coupled trainer-to-rollout `S0` weight reset."""
from collections import deque
import copy, random, threading, time

ZERO_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step")
FP32_GROUPS = (
    "shard_fp32_from_float16_groups",
    "shard_fp32_groups",
    "fp32_from_float16_groups",
    "fp32_from_fp32_groups",
)


class EpochDataBuffer:
    """`code_epoch`-aware trajectory buffer wrapping or standing in for `miles` `DataBuffer`."""
    def __init__(self, capacity=64, unused_handler=None):
        self.capacity, self.unused_handler = max(1, int(capacity)), unused_handler
        self.code_epoch, self.paused, self.stale_evicted = 0, False, 0
        self._entries, self._retry_groups, self._lock = deque(), deque(), threading.RLock()

    def admits(self, *, in_flight=0):
        with self._lock: return (not self.paused) and (len(self._entries) + int(in_flight) < self.capacity)
    def pause(self):
        with self._lock: self.paused = True
    def resume(self):
        with self._lock: self.paused = False
    def advance_epoch(self):
        with self._lock: self.code_epoch += 1; return self.code_epoch

    def put_group(self, group, prompt_group=None, epoch=None):
        with self._lock:
            stamped = self.code_epoch if epoch is None else int(epoch)
            self._entries.append({"epoch": stamped, "prompt_group": copy.deepcopy(prompt_group), "group": copy.deepcopy(group)})

    def get_groups(self, num_groups):
        with self._lock:
            self.drain_stale(clear_all=False)
            if len(self._entries) < num_groups:
                raise RuntimeError(f"buffer holds {len(self._entries)} group(s), need {num_groups}")
            return [self._entries.popleft()["group"] for _ in range(num_groups)]

    def drain_stale(self, clear_all=False):
        with self._lock:
            kept, evicted = deque(), 0
            while self._entries:
                entry = self._entries.popleft()
                if clear_all or entry["epoch"] < self.code_epoch:
                    evicted += 1
                    self.stale_evicted += 0 if clear_all else 1
                    if self.unused_handler is not None:
                        self.unused_handler(entry["group"], "stale_code_epoch")
                else:
                    kept.append(entry)
            self._entries = kept
            if clear_all:
                evicted += len(self._retry_groups); self._retry_groups.clear()
            return {"evicted": evicted, "code_epoch": self.code_epoch, "remaining": len(self._entries)}


def sync_weights_backuper(actor, tag="actor"):
    """Refresh `actor.weights_backuper` after an S0 restore so subsequent steps do not revert to stale weights."""
    wb = getattr(actor, "weights_backuper", None)
    tags = getattr(wb, "backup_tags", None) or ()
    if wb is not None and tag in tags:
        wb.backup(tag)
        return True
    return False


class CoupledWeightReset:
    """Snapshot trainer `S0` state once and restore trainer + rollout weights together."""
    def __init__(self, get_trainer_states, set_trainer_states, sync_rollout_weights,
                 flush_rollout_cache=None, data_buffer=None, post_restore=None):
        self._get_trainer, self._set_trainer = get_trainer_states, set_trainer_states
        self._sync_rollout, self._flush_cache = sync_rollout_weights, flush_rollout_cache
        self.data_buffer, self._post_restore, self.initial_states = data_buffer, post_restore, None

    def capture_s0(self):
        self.initial_states = copy.deepcopy(self._get_trainer())
        return self.initial_states

    def restore_and_sync(self, broadcast_rollout=True):
        if self.initial_states is None:
            self.capture_s0()
        reset_items, evicted = ["trainer_s0"], 0
        if self.data_buffer is not None:
            evicted = self.data_buffer.drain_stale(clear_all=True)["evicted"]
            reset_items.insert(0, "data_buffer")
        self._set_trainer(copy.deepcopy(self.initial_states))
        if self._post_restore is not None:
            self._post_restore()
            reset_items.append("weights_backuper")
        if broadcast_rollout:
            self._sync_rollout(); reset_items.append("rollout_weights")
        if self._flush_cache is not None:
            self._flush_cache(); reset_items.append("kv_cache")
        return {"reset": reset_items, "evicted": evicted}


def _inner_optimizers(optimizer):
    return [] if optimizer is None else list(getattr(optimizer, "chained_optimizers", None) or [optimizer])


def _sync_cuda(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cpu_copy(obj, torch):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_copy(v, torch) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_copy(v, torch) for v in obj)
    return copy.deepcopy(obj)


def _collect_tensors(model, optimizer, torch):
    out, seen = {}, set()
    chunks = list(model) if isinstance(model, (list, tuple)) else ([model] if model is not None else [])

    def add(name, tensor):
        if not torch.is_tensor(tensor):
            return
        if type(tensor) is not torch.Tensor and hasattr(tensor, "__dict__"):
            if parts := sorted((k, v) for k, v in vars(tensor).items() if type(v) is torch.Tensor):
                for key, part in parts:
                    add(f"{name}.{key}", part)
                return
        key = (tensor.data_ptr(), tensor.numel(), tensor.dtype)
        if key not in seen and tensor.numel() > 0:
            seen.add(key); out[name] = tensor

    for ci, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            add(f"model{ci}.{name}", param.data)
        for name, buf in chunk.named_buffers():
            add(f"model{ci}.buffer.{name}", buf)
    for oi, opt in enumerate(_inner_optimizers(optimizer)):
        for group_attr in FP32_GROUPS:
            for gi, group in enumerate(getattr(opt, group_attr, None) or []):
                for pi, tensor in enumerate(group):
                    add(f"opt{oi}.{group_attr}.{gi}.{pi}", tensor)
        for si, (_, state) in enumerate((getattr(getattr(opt, "optimizer", opt), "state", None) or {}).items()):
            for k, v in state.items():
                add(f"opt{oi}.state.{si}.{k}", v)
    return out


def _cuda_rng_tracker():
    try:
        from megatron.core import tensor_parallel
        return tensor_parallel.get_cuda_rng_tracker()
    except Exception:
        return None


def _rng_state(torch, restore=None):
    try:
        import numpy as np
    except ImportError:
        np = None
    tracker = _cuda_rng_tracker()
    if restore is None:
        t_states = {
            k: (st.get_state() if isinstance(st, torch.Generator) else st.clone())
            for k, st in tracker.get_states().items()
        } if tracker is not None and tracker.get_states() else None
        return {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "tracker": t_states,
            "python": random.getstate(),
            "numpy": np.random.get_state() if np is not None else None,
        }
    torch.set_rng_state(restore["cpu"])
    if restore.get("cuda") is not None:
        torch.cuda.set_rng_state(restore["cuda"])
    if restore.get("tracker") is not None and tracker is not None:
        live = tracker.get_states()
        states = {}
        for name, st in restore["tracker"].items():
            if isinstance(live.get(name), torch.Generator):
                live[name].set_state(st.clone()); states[name] = live[name]
            else:
                states[name] = st.clone()
        tracker.set_states(states)
    random.setstate(restore["python"])
    if restore.get("numpy") is not None and np is not None:
        np.random.set_state(restore["numpy"])


class TrainerStateSnapshot:
    """Lazy-`torch` host snapshot of Megatron trainer weights, optimizer, scheduler, and RNG."""

    def __init__(self, model, optimizer, scheduler=None):
        import torch
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        t0 = time.perf_counter(); _sync_cuda(torch)
        live, pin = _collect_tensors(model, optimizer, torch), bool(torch.cuda.is_available())
        self.host = {}
        for name, tensor in live.items():
            buf = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=pin)
            buf.copy_(tensor, non_blocking=pin); self.host[name] = buf
        _sync_cuda(torch)
        self.groups = [
            [{k: copy.deepcopy(v) for k, v in g.items() if k != "params" and not torch.is_tensor(v)}
             for g in (getattr(getattr(opt, "optimizer", opt), "param_groups", None) or [])]
            for opt in _inner_optimizers(optimizer)
        ]
        self.num_steps = getattr(scheduler, "num_steps", None)
        self.scalers = [
            _cpu_copy(gs.state_dict(), torch)
            if (gs := getattr(opt, "grad_scaler", None)) is not None and hasattr(gs, "state_dict") else None
            for opt in _inner_optimizers(optimizer)
        ]
        self.rng = _rng_state(torch)
        self.bytes = sum(h.numel() * h.element_size() for h in self.host.values())
        self.take_s = round(time.perf_counter() - t0, 4)
        cats = {("buffers" if ".buffer." in n else "weights") if n.startswith("model")
                else ("optimizer" if ".state." in n else "master_weights") for n in self.host}
        self.restored = (
            [c for c in ("weights", "master_weights", "buffers", "optimizer") if c in cats]
            + ["optimizer_scalars"]
            + (["scheduler"] if self.num_steps is not None else [])
            + (["grad_scaler"] if any(x is not None for x in self.scalers) else [])
            + ["rng"]
        )

    def restore(self, allow_extra=False):
        import torch
        t0 = time.perf_counter()
        live, opts = _collect_tensors(self.model, self.optimizer, torch), _inner_optimizers(self.optimizer)
        missing, extra = sorted(set(self.host) - set(live)), sorted(set(live) - set(self.host))
        mismatch = sorted(n for n, h in self.host.items()
                          if n in live and (tuple(live[n].shape) != tuple(h.shape) or live[n].dtype != h.dtype))
        groups_err = ([f"{len(opts)} inner optimizers, snapshot has {len(self.groups)}"]
                      if len(opts) != len(self.groups) else [])
        for i, (opt, gs) in enumerate(zip(opts, self.groups)):
            if (n_pg := len(getattr(getattr(opt, "optimizer", opt), "param_groups", None) or [])) != len(gs):
                groups_err.append(f"optimizer {i}: {n_pg} param groups, snapshot has {len(gs)}")
            if hasattr(getattr(opt, "grad_scaler", None), "state_dict") != (self.scalers[i] is not None):
                groups_err.append(f"optimizer {i}: grad scaler presence changed")
        if allow_extra:
            non_opt_extra = [n for n in extra if not (n.startswith("opt") and ".state." in n)]
            unknown_state = [
                n for n in extra
                if n.startswith("opt") and ".state." in n and n.rsplit(".", 1)[1] not in ZERO_KEYS
            ]
            bad_extra = sorted(non_opt_extra + unknown_state)
        else:
            unknown_state = []
            bad_extra = extra
        out = {"tensors": 0, "missing": missing[:5], "extra": bad_extra[:5], "mismatch": mismatch[:5],
               "groups": groups_err, "n_missing": len(missing), "n_extra": len(bad_extra)}
        if unknown_state:
            out["unknown_state"] = unknown_state
        if missing or bad_extra or mismatch or groups_err:
            return {"ok": False, "restored": [], **out,
                    "error": "live state does not match snapshot; nothing written",
                    "seconds": round(time.perf_counter() - t0, 4)}
        with torch.no_grad():
            for name, host_tensor in self.host.items():
                live[name].copy_(host_tensor, non_blocking=host_tensor.is_pinned())
        for opt, groups in zip(opts, self.groups):
            for grp, saved_grp in zip(getattr(getattr(opt, "optimizer", opt), "param_groups", None) or [], groups):
                for key in [k for k in grp if k != "params" and not torch.is_tensor(grp[k]) and k not in saved_grp]:
                    del grp[key]
                grp.update(copy.deepcopy(saved_grp))
        if self.num_steps is not None and self.scheduler is not None:
            self.scheduler.num_steps = self.num_steps
            if callable(getattr(self.scheduler, "step", None)):
                self.scheduler.step(increment=0)
        for opt, scaler_sd in zip(opts, self.scalers):
            if scaler_sd is not None:
                opt.grad_scaler.load_state_dict(_cpu_copy(scaler_sd, torch))
        _rng_state(torch, restore=self.rng); _sync_cuda(torch)
        out["tensors"] = len(self.host)
        return {"ok": True, "restored": list(self.restored), **out,
                "n_extra_allowed": len(extra) if allow_extra else 0,
                "seconds": round(time.perf_counter() - t0, 4)}

    def zero_new_state(self):
        import torch
        candidates, unknown = [], []
        for name, tensor in _collect_tensors([], self.optimizer, torch).items():
            if name in self.host or not (name.startswith("opt") and ".state." in name):
                continue
            if name.rsplit(".", 1)[1] in ZERO_KEYS:
                candidates.append((name, tensor))
            else:
                unknown.append(name)
        if unknown:
            return [], unknown
        zeroed = []
        with torch.no_grad():
            for name, tensor in candidates:
                tensor.zero_()
                zeroed.append(name)
        for opt in _inner_optimizers(self.optimizer):
            for st in (getattr(getattr(opt, "optimizer", opt), "state", None) or {}).values():
                if "step" in st and not torch.is_tensor(st["step"]):
                    st["step"] = 0
        _sync_cuda(torch)
        return zeroed, unknown


def snapshot_trainer(actor):
    """Capture S0 state onto `actor._rsibox_s0` for in-place coupled resets."""
    snap = actor._rsibox_s0 = TrainerStateSnapshot(
        actor.model, actor.optimizer, getattr(actor, "opt_param_scheduler", None)
    )
    return {"take_s": snap.take_s, "restored": snap.restored, "bytes": snap.bytes}


def restore_trainer(actor, tag="actor", allow_extra=True):
    """Restore `actor._rsibox_s0`, zero lazy Adam state, and refresh `weights_backuper`."""
    res = (snap := actor._rsibox_s0).restore(allow_extra=allow_extra)
    if not res["ok"]:
        return res
    zeroed, unknown = snap.zero_new_state()
    if unknown:
        return {**res, "ok": False, "unknown_state": unknown, "error": f"unknown new optimizer state: {unknown}"}
    res["weights_backuper_refreshed"] = sync_weights_backuper(actor, tag=tag)
    res["zeroed_count"] = len(zeroed)
    return res
