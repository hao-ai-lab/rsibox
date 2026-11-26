"""Step-gap worker subprocesses and Ray participant hooks for NVIDIA NeMo-RL live labs.

Pure Python stdlib + `core/livelab` at import time (zero eager `torch`, `ray`, `nemo_rl`, or
`sglang` imports). Hosts:
1. 3-participant CPU toy GRPO cluster (`coordinator0` + `trainer0` AutoModel FSDP2 stand-in +
   `rollout0` SGLang stand-in) over `AF_UNIX` JSON channels (`UnixJsonServer`) and
   `ray_participants.serve` (`#_actor` / `file_namespace` / `unloaded`).
2. Real NeMo-RL colocated Ray participant definitions (`NemoRlRayParticipant`,
   `serve_nemo_rl_actor`, `discover_nemo_rl_actors`, `snapshot_automodel_s0`,
   `restore_automodel_s0`, `coupled_nemo_rl_reset`) with fail-closed lazy dependency checks.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import linecache
import math
import os
import queue
import random
import sys
import time
import types
from typing import Any

import ray_participants as RP
from kernel import coordinator, findings, locate
from rsibox import harness, ipc
from rsibox.lab import ENDPOINT_METHODS

PARTICIPANT_METHODS = frozenset(ENDPOINT_METHODS)
REQUIRED_REAL_DEPS = ("ray", "torch", "nemo_rl", "sglang")
DEFAULT_PROMPTS = (1, 2, 3)


def validate_positive_int(value: Any, name: str) -> int:
    """Validate strict integer >= 1 (rejecting bools and floats)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1 (got {value!r})")
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


def require_nemo_rl_deps() -> None:
    """Fail closed with an actionable error when real NeMo-RL runtime deps are absent."""
    missing = [pkg for pkg in REQUIRED_REAL_DEPS if importlib.util.find_spec(pkg) is None]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"NemoRlLab real mode missing required runtime dependencies: {joined}. "
            "Install NVIDIA NeMo-RL (nemo_rl), Ray, PyTorch, and SGLang in the active Python "
            "environment, or run CPU conformance via ToyNemoRlLab (`rsibox conformance nemo-rl`)."
        )


# -- Fake @ray.remote wrapper for CPU actor-copy parity ----------------------------------


def _fake_remote(cls):
    """Mimic `@ray.remote` in-module ActorClass handle + `__ray_actor_class__` metadata."""
    modified = type(cls.__name__, (cls,), {"__ray_actor_class__": cls, "__module__": cls.__module__})
    handle = type(f"ActorClass({cls.__name__})", (), {})()
    handle.__dict__["__ray_metadata__"] = types.SimpleNamespace(modified_class=modified)
    return handle


def _by_value_class(cls):
    """Clone `cls` over a private globals dict (matching Ray cloudpickle-by-value actors)."""
    module = vars(sys.modules[cls.__module__])
    names: set[str] = set()

    def _collect(code):
        names.update(code.co_names)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                _collect(const)

    globals_dict = {k: module[k] for k in ("__name__", "__file__", "__package__") if k in module}
    body: dict[str, Any] = {}
    for key, value in vars(cls).items():
        if key in ("__dict__", "__weakref__"):
            continue
        if isinstance(value, types.FunctionType):
            _collect(value.__code__)
            fn_copy = types.FunctionType(
                value.__code__, globals_dict, value.__name__, value.__defaults__, value.__closure__
            )
            fn_copy.__qualname__, fn_copy.__kwdefaults__ = value.__qualname__, value.__kwdefaults__
            value = fn_copy
        body[key] = value
    globals_dict.update({n: module[n] for n in names if n in module})
    return type(cls.__name__, cls.__bases__, body)


def instantiate_ray_shaped_actor(handle, **kwargs):
    """Instantiate a deserialized by-value actor copy from a `@_fake_remote` handle."""
    original = vars(handle)["__ray_metadata__"].modified_class.__ray_actor_class__
    copied = _by_value_class(original)
    actor_cls = type(copied.__name__, (copied,), {"__ray_actor_class__": copied, "__module__": copied.__module__})
    return actor_cls(**kwargs)


# -- 3-participant CPU toy GRPO actors (Coordinator + AutoModel Trainer + SGLang Rollout)


@_fake_remote
class AutoModelTrainerActor:
    """Simulated AutoModel FSDP2 policy trainer with S0 weights, AdamW state, RNG, and step_counter."""

    def __init__(self, config: dict[str, Any]) -> None:
        from adapter import grpo_ops, shared_ops

        self.grpo_ops = grpo_ops
        self.shared_ops = shared_ops
        self.seed = int(config.get("seed", 0))
        self.rank = int(config.get("rank", 0))
        self.cfg = grpo_ops.NemoRlConfig(
            hidden=int(config.get("hidden", 8)),
            group_size=int(config.get("group_size", 4)),
            lr=float(config.get("lr", 0.05)),
            clip_eps=float(config.get("clip_eps", 0.2)),
            kl_coef=float(config.get("kl_coef", 0.01)),
        )
        init_rng = random.Random(self.seed + 100 + self.rank)
        sc = self.cfg.init_scale
        self.weights = [round(init_rng.uniform(-sc, sc), 8) for _ in range(self.cfg.hidden)]
        self.optimizer_state: dict[str, Any] = {
            "exp_avg": [0.0] * self.cfg.hidden,
            "exp_avg_sq": [0.0] * self.cfg.hidden,
            "adam_step": 0,
        }
        self.step_counter = 0
        self.rng = random.Random(self.seed + 200 + self.rank)
        self._s0_weights = list(self.weights)
        self._s0_optim = copy.deepcopy(self.optimizer_state)
        self._s0_rng = self.rng.getstate()

    def scope(self) -> dict[str, Any]:
        return {"trainer": self, "cfg": self.cfg}

    def get_weights(self) -> list[float]:
        return list(self.weights)

    def reset_state(self) -> list[str]:
        self.weights = list(self._s0_weights)
        self.optimizer_state = copy.deepcopy(self._s0_optim)
        self.step_counter = 0
        self.rng.setstate(self._s0_rng)
        return ["weights_s0", "optimizer_state", "rng", "step_counter"]

    def train_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        batch = payload["batch"]
        refit_version = int(payload.get("refit_version", self.step_counter))
        policy_lps: list[float] = []
        old_lps: list[float] = []
        ref_lps: list[float] = []
        advantages: list[float] = []
        w_sum = sum(self.weights)
        for item in batch:
            old_lp = float(item["old_logprob"])
            ref_lp = old_lp - 0.02
            signal = w_sum * 0.25 + float(item["logit"]) * 0.1
            _ = self.shared_ops.score_prompt_reward(int(item["prompt"]), refit_version, signal)
            pol_lp = old_lp + 0.05 * w_sum - 0.01 * int(item["sample_idx"])
            policy_lps.append(pol_lp)
            old_lps.append(old_lp)
            ref_lps.append(ref_lp)
            advantages.append(float(item["advantage"]))
        loss, sample_grads = self.grpo_ops.compute_grpo_loss(
            policy_lps, old_lps, ref_lps, advantages,
            clip_eps=self.cfg.clip_eps, kl_coef=self.cfg.kl_coef,
        )
        mean_g = sum(sample_grads) / max(len(sample_grads), 1)
        weight_grads = [
            mean_g * (0.5 + 0.1 * ((i + 1) % 3)) + (self.rng.random() - 0.5) * 1e-6
            for i in range(len(self.weights))
        ]
        grad_norm = self.grpo_ops.clip_and_step_adamw(
            self.weights, weight_grads, self.optimizer_state, self.cfg.lr
        )
        self.step_counter += 1
        return {
            "loss": round(float(loss), 8),
            "grad_norm": round(float(grad_norm), 8),
            "step_counter": self.step_counter,
            "adam_step": int(self.optimizer_state["adam_step"]),
            "weight_checksum": round(sum(self.weights), 8),
            "weights": [round(float(w), 10) for w in self.weights],
        }


@_fake_remote
class SGLangRolloutActor:
    """Simulated colocated SGLang generation worker with weight refit version & KV cache."""

    def __init__(self, config: dict[str, Any]) -> None:
        from adapter import rollout_ops, shared_ops

        self.rollout_ops = rollout_ops
        self.shared_ops = shared_ops
        self.seed = int(config.get("seed", 0))
        self.hidden = int(config.get("hidden", 8))
        self.group_size = int(config.get("group_size", 4))
        self.state: dict[str, Any] = {
            "weights": [0.0] * self.hidden,
            "refit_version": 0,
            "kv_cache": {},
        }
        self.step_counter = 0
        self.rng = random.Random(self.seed + 300)
        self._s0_weights = list(self.state["weights"])
        self._s0_rng = self.rng.getstate()

    def scope(self) -> dict[str, Any]:
        return {"rollout_state": self.state}

    def set_s0_weights(self, weights: list[float]) -> list[str]:
        self._s0_weights = [float(w) for w in weights]
        self.state["weights"] = list(self._s0_weights)
        self.state["refit_version"] = 0
        self.state["kv_cache"].clear()
        return ["rollout_weights", "rollout_refit_version", "kv_cache"]

    def reset_state(self) -> list[str]:
        self.state["weights"] = list(self._s0_weights)
        self.state["refit_version"] = 0
        self.state["kv_cache"].clear()
        self.step_counter = 0
        self.rng.setstate(self._s0_rng)
        return ["rollout_weights", "rollout_refit_version", "kv_cache", "rng", "step_counter"]

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompts = [int(p) for p in (payload.get("prompts") or DEFAULT_PROMPTS)]
        step_idx = int(payload.get("step_idx", self.step_counter))
        groups = self.rollout_ops.generate_sglang_group(
            prompts,
            self.state["weights"],
            int(self.state["refit_version"]),
            step_idx,
            self.group_size,
            self.state["kv_cache"],
            self.rng,
        )
        for grp in groups:
            for smp in grp["samples"]:
                smp["reward"] = self.shared_ops.score_prompt_reward(
                    smp["prompt"], int(self.state["refit_version"]), float(smp["logit"])
                )
        self.step_counter += 1
        return {
            "groups": groups,
            "refit_version": int(self.state["refit_version"]),
            "rollout_checksum": round(sum(self.state["weights"]), 8),
        }

    def refit_weights(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rollout_ops.apply_sglang_refit(
            self.state, payload["weights"], int(payload["refit_version"])
        )


@_fake_remote
class CoordinatorActor:
    """Simulated NeMo-RL GRPO loop coordinator computing group-normalized advantages."""

    def __init__(self, config: dict[str, Any]) -> None:
        from adapter import shared_ops

        self.shared_ops = shared_ops
        self.seed = int(config.get("seed", 0))
        self.step_counter = 0
        self.rng = random.Random(self.seed + 400)
        self._s0_rng = self.rng.getstate()

    def scope(self) -> dict[str, Any]:
        return {"coordinator": self}

    def reset_state(self) -> list[str]:
        self.step_counter = 0
        self.rng.setstate(self._s0_rng)
        return ["coordinator_step_counter", "rng"]

    def prepare_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        groups = payload["groups"]
        refit_version = int(payload.get("refit_version", 0))
        flat_batch: list[dict[str, Any]] = []
        all_rewards: list[float] = []
        for grp in groups:
            rewards = [
                self.shared_ops.score_prompt_reward(
                    int(s["prompt"]), refit_version, float(s["logit"])
                )
                for s in grp["samples"]
            ]
            all_rewards.extend(rewards)
            mean_r = sum(rewards) / max(len(rewards), 1)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = math.sqrt(var_r) + 1e-6
            for smp, rew in zip(grp["samples"], rewards):
                flat_batch.append({
                    **smp,
                    "reward": round(rew, 8),
                    "advantage": round((rew - mean_r) / std_r, 8),
                })
        self.step_counter += 1
        return {
            "batch": flat_batch,
            "mean_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 8),
        }


_ROLE_HANDLES = {
    "trainer": AutoModelTrainerActor,
    "rollout": SGLangRolloutActor,
    "coordinator": CoordinatorActor,
}


# -- AF_UNIX worker server & fresh subprocess runner ------------------------------------


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


def _dispatch_worker(
    actor: Any,
    role_name: str,
    planned: dict[str, set[str]],
    tracked: set[str],
    faults: dict[str, Any],
    method: str,
    payload: Any,
) -> tuple[dict[str, Any], bool]:
    if method == "shutdown":
        return {"ok": True}, True
    if method == "ping":
        return {"ok": True, "pid": os.getpid(), "role": role_name}, False
    if method in ("get_weights", "set_s0_weights", "generate", "prepare_batch", "train_step", "refit_weights"):
        if faults.pop("crash", False):
            os._exit(139)
        if hang_s := faults.pop("hang_s", 0.0):
            time.sleep(float(hang_s))
        fn = getattr(actor, method)
        res = fn() if payload is None else fn(payload)
        return {"ok": True, "result": res}, False
    if method == "inject":
        info = dict(payload or {})
        fault = str(info.pop("fault", ""))
        if fault in ("crash", "hang", "lost"):
            faults["crash" if fault == "crash" else "hang_s"] = (
                True if fault == "crash" else float(info.get("hang_s", 5.0))
            )
            return {"ok": True, "fault": fault}, fault == "lost"
        lp = RP._LOCAL.get(role_name)
        if lp is None:
            return findings.not_ok(f"participant {role_name!r} not initialized"), False
        return harness.inject_participant_fault(lp, fault, **info), False
    if method not in PARTICIPANT_METHODS and method != "loaded":
        return findings.not_ok(f"unknown worker method {method!r}"), False
    if method == "prepare":
        modules = (payload or {}).get("modules", {})
        for module, spec in modules.items():
            origin = getattr(sys.modules.get(module), "__file__", None) or harness.module_file(module)
            src = spec.get("source") if isinstance(spec, dict) else None
            if origin and isinstance(src, str):
                linecache.cache[origin] = (len(src), None, src.splitlines(True), origin)
        reply = RP.serve(actor, role_name, "prepare", payload)
        if reply.get("ok"):
            planned[str(reply.get("txn"))] = set(modules)
        else:
            _sync_linecache(RP._LOCAL[role_name], modules)
        return reply, False
    reply = RP.serve(actor, role_name, method, payload)
    if method in ("commit", "abort", "undo"):
        txn_id, _ = coordinator.txn_ref(payload)
        affected = planned.pop(str(txn_id), set())
        tracked.update(affected if method == "commit" else ())
        _sync_linecache(RP._LOCAL[role_name], (tracked | affected) if method == "undo" else affected)
    return reply, False


def serve_worker(sock_path: str, config: dict[str, Any]) -> int:
    parent_pid = int(config.get("parent_pid") or os.getppid())
    if parent_pid and os.getppid() != parent_pid:
        return 0
    role_kind = str(config["role_kind"])
    role_name = str(config["role_name"])
    handle = _ROLE_HANDLES[role_kind]
    actor = instantiate_ray_shaped_actor(handle, config=config)
    RP._LOCAL[role_name] = coordinator.LocalParticipant(
        role_name, scope=actor.scope(), resets={"state": actor.reset_state}
    )
    planned: dict[str, set[str]] = {}
    tracked: set[str] = set()
    faults: dict[str, Any] = {}
    server = ipc.UnixJsonServer(sock_path, handler=None)
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
                reply, should_exit = _dispatch_worker(
                    actor, role_name, planned, tracked, faults, method, req.get("payload")
                )
            except Exception as exc:
                reply, should_exit = findings.not_ok(f"{method}: {type(exc).__name__}: {exc}"), False
            reply_fn(reply)
            if should_exit:
                return 0
    finally:
        server.close()


def execute_grpo_episode(
    call_rollout,
    call_coordinator,
    call_trainer,
    workload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one GRPO rollout -> advantage -> AutoModel FSDP2 step -> SGLang refit episode."""
    wl = dict(workload or {})
    steps = max(1, int(wl.get("steps", 2)))
    prompts = [int(p) for p in (wl.get("prompts") or DEFAULT_PROMPTS)]
    losses: list[float] = []
    grad_norms: list[float] = []
    mean_rewards: list[float] = []
    refit_versions: list[int] = []
    last_train: dict[str, Any] = {}
    last_refit: dict[str, Any] = {}
    for step_idx in range(steps):
        gen = call_rollout("generate", {"prompts": prompts, "step_idx": step_idx})
        prep = call_coordinator(
            "prepare_batch",
            {"groups": gen["groups"], "refit_version": gen["refit_version"]},
        )
        tr = call_trainer(
            "train_step",
            {"batch": prep["batch"], "refit_version": gen["refit_version"]},
        )
        rf = call_rollout(
            "refit_weights",
            {"weights": tr["weights"], "refit_version": tr["step_counter"]},
        )
        losses.append(tr["loss"])
        grad_norms.append(tr["grad_norm"])
        mean_rewards.append(prep["mean_reward"])
        refit_versions.append(rf["refit_version"])
        last_train = tr
        last_refit = rf
    summary = {
        "losses": losses,
        "grad_norms": grad_norms,
        "mean_rewards": mean_rewards,
        "refit_versions": refit_versions,
        "trainer_step": int(last_train.get("step_counter", 0)),
        "adam_step": int(last_train.get("adam_step", 0)),
        "weight_checksum": last_train.get("weight_checksum", 0.0),
        "rollout_checksum": last_refit.get("weight_checksum", 0.0),
    }
    summary["digest"] = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()[:16]
    return summary


def run_fresh(spec: dict[str, Any]) -> int:
    """Fresh isolated subprocess entrypoint matching live `ToyNemoRlLab` / `NemoRlLab` output."""
    cfg = dict(spec["config"])
    mode = str(cfg.get("mode", "cpu"))
    if mode != "cpu":
        require_nemo_rl_deps()
        raise RuntimeError(
            "NemoRlLab real fresh_check driver must be invoked with real cluster config"
        )
    _validate_timeout(spec.get("timeout"))
    trainer = instantiate_ray_shaped_actor(AutoModelTrainerActor, config=cfg)
    rollout = instantiate_ray_shaped_actor(SGLangRolloutActor, config=cfg)
    coord = instantiate_ray_shaped_actor(CoordinatorActor, config=cfg)
    rollout.set_s0_weights(trainer.get_weights())

    def _call(actor):
        def _inner(method, payload=None):
            fn = getattr(actor, method)
            return fn() if payload is None else fn(payload)
        return _inner

    output = execute_grpo_episode(
        _call(rollout), _call(coord), _call(trainer), workload=spec.get("workload")
    )
    print("FRESH " + json.dumps({"output": output}), flush=True)
    return 0


# -- Real NeMo-RL Ray actor participant definitions (reusing core/livelab/ray_participants.py)


def serve_nemo_rl_actor(host: Any, name: str, method: str, payload: Any) -> dict[str, Any]:
    """Actor-side wrapper over `ray_participants.serve` for colocated NeMo-RL Ray actors."""
    own_mod = getattr(getattr(type(host), "__ray_actor_class__", None), "__module__", None)
    if own_mod and own_mod not in sys.modules:
        importlib.import_module(own_mod)
    local = RP._LOCAL.get(name)
    if local is None:
        reset_fn = getattr(host, "reset_state", None)

        def _missing_reset() -> list[str]:
            raise RuntimeError(
                f"actor {name!r} ({type(host).__name__}) has no reset_state() for coupled S0 reset"
            )

        resets = {"state": reset_fn if callable(reset_fn) else _missing_reset}
        local = RP._LOCAL.setdefault(name, coordinator.LocalParticipant(name, resets=resets))
    if (view := RP.actor_view(host)) is not None:
        local.alias[view[0] + RP.ACTOR] = view[1]
    if method == "fingerprint":
        scoped = []
        for entry in payload:
            if isinstance(entry, dict):
                path = entry.get("file")
                mod_name = entry.get("name")
                if isinstance(path, str) and path:
                    actual, loaded_mod = RP.file_namespace(path)
                    if loaded_mod is not None:
                        local.alias[mod_name or RP.file_module(path)] = actual
                        scoped.append(entry)
                elif isinstance(mod_name, str) and (mod_name in sys.modules or mod_name in local.alias):
                    scoped.append(entry)
            elif isinstance(entry, str) and (entry.endswith(".py") or os.sep in entry):
                actual, loaded_mod = RP.file_namespace(entry)
                if loaded_mod is not None:
                    key = RP.file_module(entry)
                    local.alias[key] = actual
                    scoped.append(key)
            elif isinstance(entry, str) and (entry in sys.modules or entry in local.alias):
                scoped.append(entry)
        payload = scoped
    return RP.serve(host, name, method, payload)


class NemoRlRayParticipant(RP.RayActorParticipant):
    """Host-side `RayActorParticipant` for NeMo-RL AutoModel FSDP2 & SGLang generation actors."""

    def __init__(self, name: str, handle: Any, get=None, timeout_s: float | None = None, bounded: bool = False):
        fetch = get or RP._ray_get
        coordinator.CallParticipant.__init__(
            self,
            name,
            lambda m, p: fetch(handle.__ray_call__.remote(serve_nemo_rl_actor, name, m, p), timeout_s),
            bounded,
        )


def discover_nemo_rl_actors(
    policy_group: Any = None,
    rollout_group: Any = None,
    coordinator_actor: Any = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extract `{name: ray_handle}` maps from NeMo-RL policy, SGLang rollout, and coordinator objects."""
    def _extract(source: Any, prefix: str) -> dict[str, Any]:
        if source is None:
            return {}
        if isinstance(source, dict):
            return {str(k): v for k, v in source.items()}
        for attr in ("workers", "_workers", "actor_handles", "_actor_handlers", "sglang_workers"):
            if isinstance(handles := getattr(source, attr, None), (list, tuple)):
                return {f"{prefix}{i}": h for i, h in enumerate(handles)}
        if isinstance(source, (list, tuple)):
            return {f"{prefix}{i}": h for i, h in enumerate(source)}
        return {f"{prefix}0": source}

    return (
        _extract(coordinator_actor, "coordinator"),
        _extract(policy_group, "trainer"),
        _extract(rollout_group, "rollout"),
    )


def _local_tensor(tensor: Any) -> Any:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _cpu_clone(obj: Any, torch: Any) -> Any:
    if torch.is_tensor(obj):
        return _local_tensor(obj).detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu_clone(v, torch) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_clone(v, torch) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_clone(v, torch) for v in obj)
    return copy.deepcopy(obj)


def snapshot_automodel_s0(actor: Any) -> dict[str, Any]:
    """Capture FSDP2 `DTensor`/tensor weights, CPU-offloaded AdamW optimizer buffers, and RNG."""
    import torch

    model = getattr(actor, "model", None) or getattr(actor, "policy", None)
    optimizer = getattr(actor, "optimizer", None)
    if model is None:
        raise RuntimeError("AutoModel policy actor exposes neither `.model` nor `.policy`")
    params = [_local_tensor(p).detach().cpu().clone() for p in model.parameters()]
    opt_state = (
        _cpu_clone(optimizer.state_dict(), torch)
        if optimizer is not None and hasattr(optimizer, "state_dict")
        else None
    )
    snap = {
        "params": params,
        "optimizer": opt_state,
        "scheduler": (
            copy.deepcopy(actor.scheduler.state_dict())
            if getattr(actor, "scheduler", None) is not None and hasattr(actor.scheduler, "state_dict")
            else None
        ),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "step_counter": int(getattr(actor, "step_counter", 0)),
    }
    actor._rsibox_s0_snapshot = snap
    return {"ok": True, "num_tensors": len(params)}


def restore_automodel_s0(actor: Any) -> dict[str, Any]:
    """Restore post-init `S0` FSDP2 parameters, optimizer state, step counter, and RNG."""
    import torch

    snap = getattr(actor, "_rsibox_s0_snapshot", None)
    if snap is None:
        raise RuntimeError("AutoModel S0 snapshot missing; call snapshot_automodel_s0 first")
    model = getattr(actor, "model", None) or getattr(actor, "policy", None)
    if model is None:
        raise RuntimeError("AutoModel policy actor exposes neither `.model` nor `.policy`")
    optimizer = getattr(actor, "optimizer", None)
    with torch.no_grad():
        for param, saved in zip(model.parameters(), snap["params"], strict=True):
            local_p = _local_tensor(param)
            local_p.copy_(saved.to(device=local_p.device, dtype=local_p.dtype))
            param.grad = None
    if optimizer is not None and snap.get("optimizer") is not None:
        if not snap["optimizer"].get("state") and hasattr(optimizer, "state"):
            optimizer.state.clear()
        optimizer.load_state_dict(copy.deepcopy(snap["optimizer"]))
    if getattr(actor, "scheduler", None) is not None and snap.get("scheduler") is not None:
        actor.scheduler.load_state_dict(copy.deepcopy(snap["scheduler"]))
    if hasattr(actor, "step_counter"):
        actor.step_counter = int(snap.get("step_counter", 0))
    random.setstate(snap["python_rng"])
    torch.set_rng_state(snap["torch_rng"])
    if snap.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(snap["cuda_rng"])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return {"ok": True, "reset": ["weights_s0", "optimizer_state", "scheduler_state", "rng", "step_counter"]}


def coupled_nemo_rl_reset(
    policy_group: Any,
    rollout_group: Any,
    *,
    ray_mod: Any = None,
    refit_fn: Any = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Restore AutoModel FSDP2 `S0` state, refit weights to SGLang workers, and flush KV cache."""
    if ray_mod is None:
        import ray as ray_mod
    trainers = discover_nemo_rl_actors(policy_group=policy_group)[1]
    if not trainers:
        raise RuntimeError("no trainer actors found for coupled S0 reset")
    rollouts = discover_nemo_rl_actors(rollout_group=rollout_group)[2]
    results = ray_mod.get(
        [h.__ray_call__.remote(restore_automodel_s0) for h in trainers.values()],
        timeout=timeout_s,
    )
    if len(results) != len(trainers) or not all(isinstance(r, dict) and r.get("ok") for r in results):
        raise RuntimeError(f"AutoModel S0 restore failed: {results}")
    reset_items = ["weights_s0", "optimizer_state", "rng", "step_counter"]
    if callable(refit_fn):
        refit_fn()
        reset_items.extend(["rollout_weights", "rollout_refit_version"])
    elif hasattr(policy_group, "refit_policy_generation"):
        policy_group.refit_policy_generation(rollout_group)
        reset_items.extend(["rollout_weights", "rollout_refit_version"])
    elif rollout_group is not None or rollouts:
        raise RuntimeError(
            "coupled S0 reset requires refit_fn or policy_group.refit_policy_generation for rollout actors"
        )
    if hasattr(rollout_group, "flush_cache"):
        rollout_group.flush_cache()
        reset_items.append("kv_cache")
    elif rollouts:
        missing_flush = [name for name, h in rollouts.items() if not hasattr(h, "flush_cache")]
        if missing_flush:
            raise RuntimeError(
                f"rollout actor(s) missing flush_cache for coupled S0 reset: {missing_flush}"
            )
        ray_mod.get([h.flush_cache.remote() for h in rollouts.values()], timeout=timeout_s)
        reset_items.append("kv_cache")
    elif rollout_group is not None:
        raise RuntimeError("rollout_group has no flush_cache or rollout actors for coupled S0 reset")
    return {"ok": True, "reset": reset_items}


def clean_nemo_rl_witness(raw_witness: dict[str, dict[str, int]], proof_ok: bool) -> dict[str, dict[str, int]]:
    """Strip dead `@ray.remote` class keys when `#actor` ran; omit abstaining roles (`{}`)."""
    cleaned: dict[str, dict[str, int]] = {}
    for part_name, counts in (raw_witness or {}).items():
        actor_keys = {str(k).split(RP.ACTOR + ":", 1)[1] for k in counts if RP.ACTOR + ":" in str(k)}
        filtered = {
            k: n
            for k, n in counts.items()
            if not (RP.ACTOR not in k and ":" in k and n == 0 and k.split(":", 1)[1] in actor_keys)
        }
        if filtered or not proof_ok:
            cleaned[part_name] = filtered
    return cleaned


def build_nemo_rl_plan(
    participants: list[Any],
    patch: dict[str, Any],
    base_text: dict[str, str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Build a transaction plan and fail closed (`unloaded_module`) if no participant loaded a target."""
    modules: dict[str, dict[str, Any]] = {}
    for target_key, source_text in (patch.get("files") or {}).items():
        is_path = target_key.endswith(".py") or os.sep in target_key
        module_key = RP.file_module(target_key) if is_path else target_key
        spec: dict[str, Any] = (
            {"file": target_key, "source": source_text} if is_path else {"source": source_text}
        )
        if target_key in base_text:
            spec["base_hash"] = locate.text_hash(base_text[target_key])
        modules[module_key] = spec
    plan = {
        "modules": modules,
        "accept": list(patch.get("accept", [])),
        "undo_resets": list(patch.get("undo_resets", ["state"])),
        "recipes": list(patch.get("recipes", ["R0 swap"])),
    }
    if missing := RP.unloaded(participants, plan):
        return None, [{
            "check": "unloaded_module",
            "observed": f"no participant loaded {missing}",
            "expected": "every edited module or file is loaded by at least one participant",
        }]
    return plan, []


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        spec = json.load(fh)
    raise SystemExit(run_fresh(spec) if spec.get("fresh") else serve_worker(spec["sock_path"], spec["config"]))
