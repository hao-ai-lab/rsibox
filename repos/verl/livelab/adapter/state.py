"""Coupled trainer + rollout start-state snapshot (`S0`) and restore."""
import copy, ctypes, hashlib, json, random, time

ZERO_KEYS, _SNAPSHOTS = frozenset({"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}), {}


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _is_tensor(val):
    t = _torch()
    return t is not None and t.is_tensor(val)


def _clone(val):
    return val.detach().cpu().clone() if _is_tensor(val) else copy.deepcopy(val)


def _meta(val):
    if _is_tensor(val):
        return ("tensor", str(val.dtype), tuple(val.shape))
    return (type(val).__name__, tuple(sorted(val)) if isinstance(val, dict) else len(val)) if isinstance(val, (list, dict)) else (type(val).__name__,)


def _hash_val(h, val):
    if _is_tensor(val):
        h.update(f"{val.dtype}{tuple(val.shape)}".encode())
        buf = val.detach().cpu().contiguous().reshape(-1).view(_torch().uint8)
        h.update(ctypes.string_at(buf.data_ptr(), buf.numel()) if hasattr(buf, "data_ptr") else bytes(buf.tolist()))
    elif isinstance(val, dict):
        for k in sorted(val, key=str):
            h.update(repr(k).encode())
            _hash_val(h, val[k])
    elif isinstance(val, (list, tuple)):
        h.update(f"{type(val).__name__}{len(val)}".encode())
        for item in val:
            _hash_val(h, item)
    else:
        h.update(repr(val).encode())


def collect_named_tensors(target):
    raw = getattr(target, "module", target) if target is not None else ()
    chunks, items = (list(raw) if isinstance(raw, (list, tuple)) else [raw]), {}
    for idx, chunk in enumerate(chunks):
        pfx = f"chunk{idx}." if len(chunks) > 1 else ""
        if hasattr(chunk, "named_parameters"):
            items.update({f"{pfx}param.{k}": p for k, p in chunk.named_parameters()})
            items.update({f"{pfx}buffer.{k}": b for k, b in chunk.named_buffers()})
        elif isinstance(getattr(chunk, "weights", chunk), dict):
            items.update({f"{pfx}weight.{k}": v for k, v in getattr(chunk, "weights", chunk).items()})
    return items


class StateSnapshot:
    def __init__(self, target, optimizer=None, scheduler=None):
        self.target, self.optimizer, self.scheduler = target, optimizer, scheduler
        live = collect_named_tensors(target)
        self.meta, self.saved = {k: _meta(v) for k, v in live.items()}, {k: _clone(v) for k, v in live.items()}
        opts = [] if optimizer is None else list(getattr(optimizer, "chained_optimizers", None) or [optimizer])
        self.opt_saved = copy.deepcopy(optimizer) if isinstance(optimizer, dict) else [
            {"state": {id(k): (k, {ek: _clone(ev) for ek, ev in e.items()}) for k, e in getattr(getattr(o, "optimizer", o), "state", {}).items() if isinstance(e, dict)},
             "groups": [{gk: _clone(gv) for gk, gv in g.items() if gk != "params"} for g in getattr(getattr(o, "optimizer", o), "param_groups", ())]} for o in opts]
        self.sched_step = copy.deepcopy(getattr(scheduler, "num_steps", getattr(scheduler, "last_epoch", None)))
        t = _torch()
        self.rng = {"python": random.getstate(), **({"torch_cpu": t.get_rng_state().clone()} if t is not None else {}),
                    **({"torch_cuda": [s.clone() for s in t.cuda.get_rng_state_all()]} if t is not None and t.cuda.is_available() and t.cuda.is_initialized() else {})}

    def restore(self):
        t0, live = time.perf_counter(), collect_named_tensors(self.target)
        missing, added = sorted(set(self.meta) - set(live)), sorted(set(live) - set(self.meta))
        changed = sorted(k for k in set(self.meta) & set(live) if _meta(live[k]) != self.meta[k])
        if missing or added or changed:
            return {"ok": False, "missing": missing, "added": added, "changed": changed, "error": f"state layout drifted: missing={missing}, added={added}, changed={changed}"}
        t = _torch()
        for k, dst in live.items():
            src = self.saved[k]
            if _is_tensor(dst):
                with t.no_grad():
                    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
            elif isinstance(dst, list):
                dst[:] = copy.deepcopy(src)
            elif isinstance(dst, dict):
                dst.clear()
                dst.update(copy.deepcopy(src))
        if isinstance(self.optimizer, dict):
            self.optimizer.clear()
            self.optimizer.update(copy.deepcopy(self.opt_saved))
        elif self.optimizer is not None:
            for opt, snap in zip(getattr(self.optimizer, "chained_optimizers", None) or [self.optimizer], self.opt_saved):
                inner = getattr(opt, "optimizer", opt)
                for pkey, entry in list((getattr(inner, "state", None) or {}).items()):
                    if id(pkey) not in snap["state"]:
                        for fk, fv in (entry.items() if isinstance(entry, dict) else ()):
                            if fk in ZERO_KEYS and _is_tensor(fv):
                                with t.no_grad():
                                    fv.zero_()
                            elif fk == "step" and not _is_tensor(fv):
                                entry[fk] = 0
                    else:
                        for fk, sv in snap["state"][id(pkey)][1].items():
                            if fk in entry and _is_tensor(entry[fk]):
                                with t.no_grad():
                                    entry[fk].copy_(sv.to(device=entry[fk].device, dtype=entry[fk].dtype))
                            else:
                                entry[fk] = _clone(sv)
                for grp, sgrp in zip(getattr(inner, "param_groups", ()), snap["groups"]):
                    grp.update({gk: _clone(gv) for gk, gv in sgrp.items()})
        if self.scheduler is not None and self.sched_step is not None:
            setattr(self.scheduler, "num_steps" if hasattr(self.scheduler, "num_steps") else "last_epoch", copy.deepcopy(self.sched_step))
        random.setstate(self.rng["python"])
        if t is not None and "torch_cpu" in self.rng:
            t.set_rng_state(self.rng["torch_cpu"])
            if "torch_cuda" in self.rng:
                t.cuda.set_rng_state_all(self.rng["torch_cuda"])
        return {"ok": True, "restored": ["weights", "optimizer", "scheduler", "rng"], "tensors": len(self.saved), "seconds": round(time.perf_counter() - t0, 6)}

    def digest(self):
        wh, oh = hashlib.sha256(), hashlib.sha256()
        for k, v in sorted(collect_named_tensors(self.target).items()):
            wh.update(k.encode())
            _hash_val(wh, v)
        _hash_val(oh, self.optimizer if isinstance(self.optimizer, dict) else [getattr(getattr(o, "optimizer", o), "state", {}) for o in (getattr(self.optimizer, "chained_optimizers", None) or ([self.optimizer] if self.optimizer else []))])
        out = {"weights": wh.hexdigest()[:16], "optimizer": oh.hexdigest()[:16]}
        return {**out, "all": hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]}


def _resolve_actor(host):
    wdict = getattr(host, "worker_dict", None)
    if isinstance(wdict, dict):
        for rkey in ("actor_rollout_ref", "actor_rollout", "actor"):
            if rkey in wdict:
                actor = getattr(wdict[rkey], "actor", wdict[rkey])
                eng = getattr(actor, "engine", actor)
                return eng, getattr(eng, "optimizer", getattr(actor, "optimizer", None)), getattr(eng, "lr_scheduler", getattr(eng, "optimizer_scheduler", getattr(actor, "scheduler", None)))
    return (host, getattr(host, "optimizer", None), getattr(host, "scheduler", None)) if hasattr(host, "weights") else None


def worker_capture_s0(host):
    resolved = _resolve_actor(host)
    if resolved is None:
        return {"ok": True, "absent": True}
    snap = StateSnapshot(*resolved)
    _SNAPSHOTS[id(host)] = snap
    return {"ok": True, "digest": snap.digest(), "tensors": len(snap.saved)}


def worker_restore_s0(host):
    snap = _SNAPSHOTS.get(id(host))
    if snap is None:
        return {"ok": False, "error": "S0 was never captured on this worker"}
    res = snap.restore()
    return {**res, "digest": snap.digest()} if res["ok"] else res


def sync_rollout_weights(trainer):
    mgr = getattr(trainer, "checkpoint_manager", None)
    if mgr is not None:
        if callable(getattr(mgr, "sleep_replicas", None)):
            mgr.sleep_replicas()
        if callable(getattr(mgr, "update_weights", None)):
            mgr.update_weights(getattr(trainer, "global_steps", 0))
        return {"ok": True, "synced": True}
    if hasattr(trainer, "sync_weights"):
        trainer.sync_weights()
        return {"ok": True, "synced": True}
    return {"ok": True, "synced": False}
