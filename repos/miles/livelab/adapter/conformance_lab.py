"""MilesConformanceLab: real-engine `LabUnderTest` for `miles` / `THUDM/slime` Ray RL jobs."""
import argparse, contextlib, hashlib, json, os, shlex, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path

import adapter  # noqa: F401
from adapter.lab import MilesLab
from adapter.state import CoupledWeightReset, restore_trainer, snapshot_trainer
from rsibox import harness as H
from rsibox import proc as P

PPO_MOD = "slime.utils.ppo_utils"
LOSS_MOD = "slime.backends.megatron_utils.loss"
MODEL_MOD = "slime.backends.megatron_utils.model"
CONFORMANCE_MODULES = (PPO_MOD, LOSS_MOD, MODEL_MOD)

RAY_OTEL_ENV_VARS = {
    "RAY_ENABLE_OPEN_TELEMETRY": "0",
    "RAY_enable_open_telemetry": "0",
    "RAY_enable_metrics_collection": "0",
}
RAY_OTEL_SYSTEM_CONFIG = {"enable_open_telemetry": False, "enable_metrics_collection": False}
_DEFAULT_CUDA_ENV = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "1",
    "NCCL_CUMEM_ENABLE": "0", "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
}
_DEFAULT_SLIME_ARGS = shlex.split(
    "--megatron-to-hf-mode bridge --input-key messages --label-key label --apply-chat-template "
    "--num-rollout 4 --rollout-batch-size 4 --n-samples-per-prompt 4 --rollout-max-response-len 64 "
    "--rollout-temperature 0.8 --rollout-seed 42 --global-batch-size 16 --advantage-estimator grpo "
    "--kl-loss-coef 0.0 --kl-coef 0.0 --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 "
    "--optimizer adam --lr 1e-4 --lr-decay-style constant --weight-decay 0.1 "
    "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
    "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 --use-dynamic-batch-size "
    "--max-tokens-per-gpu 4096 --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.45 "
    "--sglang-cuda-graph-max-bs 16 --attention-dropout 0.0 --hidden-dropout 0.0 "
    "--attention-backend flash --actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 1 "
    "--swiglu --num-layers 28 --hidden-size 1024 --ffn-hidden-size 3072 --num-attention-heads 16 "
    "--group-query-attention --num-query-groups 8 --use-rotary-position-embeddings "
    "--disable-bias-linear --normalization RMSNorm --norm-epsilon 1e-6 --rotary-base 1000000 "
    "--vocab-size 151936 --kv-channels 128 --qk-layernorm"
)
_ENV_SPEC = (
    ("model", "RSIBOX_MILES_MODEL", False),
    ("data", "RSIBOX_MILES_DATA", True),
    ("custom_rm_path", "RSIBOX_MILES_CUSTOM_RM_PATH", False),
    ("slime_root", "RSIBOX_MILES_SLIME_ROOT", True),
    ("megatron_root", "RSIBOX_MILES_MEGATRON_ROOT", True),
    ("extra_py_path", "RSIBOX_MILES_EXTRA_PY_PATH", False),
    ("work", "RSIBOX_MILES_WORK", True),
    ("ray_gpus", "RSIBOX_MILES_RAY_GPUS", False),
    ("check_timeout", "RSIBOX_MILES_CHECK_TIMEOUT", False),
    ("rpc_timeout", "RSIBOX_MILES_RPC_TIMEOUT", False),
    ("boot_timeout", "RSIBOX_MILES_BOOT_TIMEOUT", False),
)


def build_pythonpath(*, slime_root=None, megatron_root=None, extra_py_path=None, overlay_dir=None, base_env=None):
    """Assemble deduplicated `PYTHONPATH` for live Ray workers and fresh subprocesses."""
    livelab = Path(__file__).resolve().parents[1]
    core = livelab.parents[2] / "core" / "livelab"
    existing = (os.environ if base_env is None else base_env).get("PYTHONPATH", "")
    raw = [
        *([overlay_dir] if overlay_dir else []),
        *(p for p in str(extra_py_path or "").split(os.pathsep) if p),
        *(str(p) for p in (slime_root, megatron_root, livelab, core) if p),
        *(p for p in existing.split(os.pathsep) if p),
    ]
    return os.pathsep.join(dict.fromkeys(raw))


def build_slime_cli_args(model, data, *, custom_rm_path=None, extra_args=None):
    """Build the slime/miles CLI argument vector from portable config (no hardcoded paths)."""
    args = ["--hf-checkpoint", model, "--load", model, "--prompt-data", data, *_DEFAULT_SLIME_ARGS]
    if custom_rm_path:
        args += ["--custom-rm-path", custom_rm_path]
    return args + list(extra_args or ())


def ray_init_kwargs(pythonpath, *, num_gpus=2):
    """Build `ray.init(...)` kwargs disabling Ray 2.54 C++ OpenTelemetry & metrics threads."""
    return {
        "num_gpus": int(num_gpus),
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "_system_config": dict(RAY_OTEL_SYSTEM_CONFIG),
        "runtime_env": {"env_vars": {**_DEFAULT_CUDA_ENV, **RAY_OTEL_ENV_VARS, "PYTHONPATH": pythonpath}},
    }


def _rollout_config_key(model, data, custom_rm_path=None, extra_args=None):
    payload = json.dumps(
        {
            "model": model,
            "data": data,
            "custom_rm_path": custom_rm_path,
            "extra_args": list(extra_args or ()),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _rollout_cache_valid(rollout_pt, cfg_key, *, require_meta=False):
    if not rollout_pt or not os.path.exists(rollout_pt):
        return False
    meta_path = f"{rollout_pt}.meta.json"
    if os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(meta, dict) and meta.get("config_key") == cfg_key
    return not require_meta


def _atomic_save_rollout(torch, payload, rollout_pt, cfg_key):
    parent = os.path.dirname(os.path.abspath(rollout_pt))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_pt = tempfile.mkstemp(dir=parent, prefix=".rollout0.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_pt)
        os.replace(tmp_pt, rollout_pt)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_pt)
        raise
    meta_path = f"{rollout_pt}.meta.json"
    fd_meta, tmp_meta = tempfile.mkstemp(dir=parent, prefix=".rollout0-meta.", suffix=".tmp")
    try:
        with os.fdopen(fd_meta, "w", encoding="utf-8") as fh:
            json.dump({"config_key": cfg_key}, fh)
        os.replace(tmp_meta, meta_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_meta)
        raise


def _collect_weight_digest(ray, trainers, *, timeout=None):
    names = list(trainers)
    refs = [trainers[name].__ray_call__.remote(trainer_weight_digest) for name in names]
    results = ray.get(refs) if timeout is None else ray.get(refs, timeout=timeout)
    if len(names) == 1:
        return results[0]
    h = hashlib.sha256()
    total = 0.0
    per_trainer = {}
    for name, item in zip(names, results):
        per_trainer[name] = item
        h.update(f"{name}:{item['digest']}:{item['abs_sum']}".encode("utf-8"))
        total += float(item["abs_sum"])
    return {"digest": h.hexdigest()[:16], "abs_sum": round(total, 6), "trainers": per_trainer}


def _build_fresh_overlay(files, dst):
    by_pkg = {}
    for key, text in files.items():
        if os.path.isfile(key) or os.sep in key:
            raise ValueError(f"{key}: fresh overlay only supports importable module names")
        by_pkg.setdefault(key.split(".")[0], {})[key] = text
    for top, overrides in by_pkg.items():
        H.build_package_overlay(top, overrides, dst=dst)


def _launch_slime_cluster(model, data, *, custom_rm_path=None, extra_args=None,
                          pythonpath, num_gpus=2, rollout_pt=None, save_rollout=False, prog="rsibox-miles"):
    for entry in reversed(pythonpath.split(os.pathsep)):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ.update(RAY_OTEL_ENV_VARS)
    import torch, ray
    if ray.is_initialized():
        ray.shutdown(); time.sleep(3.0)
    ray.init(**ray_init_kwargs(pythonpath, num_gpus=num_gpus))
    try:
        from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
        from slime.utils.arguments import parse_args
        from slime.utils.misc import Box

        prev_argv = sys.argv
        try:
            sys.argv = [prog] + build_slime_cli_args(model, data, custom_rm_path=custom_rm_path, extra_args=extra_args)
            args = parse_args()
        finally:
            sys.argv = prev_argv
        pgs = create_placement_groups(args)
        rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])
        actor_model, _ = create_training_models(args, pgs, rollout_manager)
        handlers = list(getattr(actor_model, "_actor_handlers", None) or ())
        if not handlers:
            raise RuntimeError("actor_model._actor_handlers is empty")
        trainers = {f"trainer{i}": h for i, h in enumerate(handlers)}
        actor_model.update_weights()
        t_ready = time.perf_counter()
        cfg_key = _rollout_config_key(model, data, custom_rm_path, extra_args)
        if _rollout_cache_valid(rollout_pt, cfg_key, require_meta=save_rollout):
            rollout_ref = [Box(ray.put(item)) for item in torch.load(rollout_pt, weights_only=False)]
        else:
            rollout_ref = ray.get(rollout_manager.generate.remote(0))
            if save_rollout and rollout_pt:
                _atomic_save_rollout(torch, [ray.get(box.inner) for box in rollout_ref], rollout_pt, cfg_key)
        return ray, rollout_manager, actor_model, trainers, rollout_ref, t_ready
    except BaseException:
        with contextlib.suppress(Exception):
            ray.shutdown()
        raise


def trainer_weight_digest(actor):
    """Compute a deterministic digest + L1 norm over actor model parameters (runs on trainer actor)."""
    h, total = hashlib.sha256(), 0.0
    for name, param in sorted(actor.model[0].named_parameters()):
        det = param.detach().float()
        total += float(det.abs().sum().item())
        if len(name) < 60 and ("word_embeddings" in name or "layers.0." in name or "final_layernorm" in name):
            h.update(det[:4].cpu().numpy().tobytes())
    return {"digest": h.hexdigest()[:16], "abs_sum": round(total, 6)}


def trainer_inject_fault(actor, fault, after=1, hang_s=25.0, participant="trainer0"):
    """Inject a conformance fault inside a trainer participant (runs on trainer actor)."""
    import ray_participants as RP
    if (lp := RP._LOCAL.get(participant)) is None:
        return {"ok": False, "error": f"{participant} participant not registered: {sorted(RP._LOCAL)}"}
    if fault == "fail_apply":
        return H.inject_participant_fault(lp, "fail_apply", after=after)
    if fault == "fail_commit":
        return H.inject_participant_fault(lp, "fail_commit")
    if fault == "tamper":
        return H.inject_participant_fault(lp, "tamper", module=PPO_MOD, qualname="get_grpo_returns")
    if fault in ("crash", "hang"):
        import slime.backends.megatron_utils.loss as loss_mod
        import slime.utils.ppo_utils as ppo_mod
        orig_ppo, orig_loss = ppo_mod.get_grpo_returns, loss_mod.get_grpo_returns

        def faulted(*args, **kwargs):
            ppo_mod.get_grpo_returns, loss_mod.get_grpo_returns = orig_ppo, orig_loss
            if fault == "crash":
                raise RuntimeError(f"{participant}: injected crash in get_grpo_returns")
            time.sleep(float(hang_s))
            return orig_ppo(*args, **kwargs)

        ppo_mod.get_grpo_returns = loss_mod.get_grpo_returns = faulted
        return {"ok": True, "fault": fault}
    return {"ok": False, "error": f"unknown fault {fault}"}


def _parse_extra_args(raw):
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("RSIBOX_MILES_SLIME_ARGS must be a JSON list of strings")
        return [str(x) for x in parsed]
    return shlex.split(text)


def _miles_patches(base_text):
    """Build the 11-case conformance patch table from base module sources."""
    edit = H.edit_once
    ppo, loss, model = base_text[PPO_MOD], base_text[LOSS_MOD], base_text[MODEL_MOD]
    vis1 = edit(ppo, "        returns.append(torch.ones_like(kl[i]) * rewards[i])",
                "        returns.append(torch.ones_like(kl[i]) * (rewards[i] * 2.0 + 1.0))")
    vis2 = edit(vis1, "        returns.append(torch.ones_like(kl[i]) * (rewards[i] * 2.0 + 1.0))",
                "        returns.append(torch.ones_like(kl[i]) * (rewards[i] * 3.0 + 2.0))")
    multi_ppo = edit(vis1, "    pg_losses1 = -ratio * advantages", "    pg_losses1 = -ratio * (advantages * 1.1)")
    vis3_loss = edit(loss, "        # TODO: is the copy necessary?\n        advantages = [r for r in returns]",
                     "        # TODO: is the copy necessary?\n        advantages = [r * 1.5 for r in returns]")
    unreached = edit(model, "    config = OptimizerConfig(**kwargs)\n    config.timers = None",
                     "    config = OptimizerConfig(**kwargs)\n    config.lr = config.lr * 5.0\n    config.timers = None")
    return {
        "noop": {"files": {PPO_MOD: edit(ppo, "        returns.append(torch.ones_like(kl[i]) * rewards[i])",
                                         "        returns.append(torch.ones_like(kl[i]) * (rewards[i] + 0.0))")}},
        "visible": {"files": {PPO_MOD: vis1}},
        "chain": [{"files": {PPO_MOD: vis1}}, {"files": {PPO_MOD: vis2}}, {"files": {LOSS_MOD: vis3_loss}}],
        "multi": {"files": {PPO_MOD: multi_ppo, LOSS_MOD: vis3_loss}},
        "unreached": {"files": {MODEL_MOD: unreached}},
        "refused": {
            "syntax": {"files": {PPO_MOD: edit(ppo, "        returns.append(torch.ones_like(kl[i]) * rewards[i])",
                                               "        returns.append(")}},
            "closure": {"files": {PPO_MOD: edit(
                ppo, "        def mul_reduce(a, b):\n            return (a * b).sum(dim=-1, keepdim=True)",
                "        def mul_reduce(a, b):\n            return (a * b + 0.0 * vocab_parallel_logits.sum()).sum(dim=-1, keepdim=True)",
            )}},
            "module_state": {"files": {PPO_MOD: edit(ppo, "import torch.nn.functional as F",
                                                     "import torch.nn.functional as F\nMILES_EXTRA_STATE = 1")}},
            "empty": {"files": {PPO_MOD: edit(ppo, "    returns = []", "    returns = []  # empty note")}},
        },
    }


class MilesConformanceLab(MilesLab):
    """Real-cluster `LabUnderTest` for `miles` / `THUDM/slime` (`rsibox conformance miles --real`)."""

    name = "miles"

    @staticmethod
    def launch_environment(argv):
        """Translate `rsibox conformance miles --real -- ...` flags into `RSIBOX_MILES_*` env vars."""
        clean = list(argv)[1:] if list(argv)[:1] == ["--"] else list(argv)
        ap = argparse.ArgumentParser(
            prog="rsibox conformance miles --real",
            description="Run conformance against a real miles / slime Ray RL cluster.",
            allow_abbrev=False,
        )
        ap.add_argument("--model", required=True, help="HF checkpoint path or repo id")
        ap.add_argument("--data", required=True, help="prompt dataset JSONL path")
        ap.add_argument("--custom-rm-path", default=None, help="custom reward function dotted path")
        ap.add_argument("--slime-root", default=None, help="slime / miles checkout root added to PYTHONPATH")
        ap.add_argument("--megatron-root", default=None, help="Megatron-LM checkout root added to PYTHONPATH")
        ap.add_argument("--extra-py-path", default=None, help="extra PYTHONPATH entries (os.pathsep-delimited)")
        ap.add_argument("--work", default=None, help="working directory for rollout0.pt and fresh runs")
        ap.add_argument("--ray-gpus", type=int, default=None, help="GPU count passed to ray.init(num_gpus=...)")
        ap.add_argument("--check-timeout", type=float, default=None, help="seconds one check step may take")
        ap.add_argument("--rpc-timeout", type=float, default=None, help="seconds one control RPC may take")
        ap.add_argument("--boot-timeout", type=float, default=None, help="seconds a fresh subprocess may take")
        ap.add_argument("slime_args", nargs=argparse.REMAINDER, help="-- followed by extra slime/miles CLI arguments")
        ns = ap.parse_args(clean)
        env = {
            key: (os.path.abspath(str(val)) if is_path else str(val))
            for attr, key, is_path in _ENV_SPEC
            if (val := getattr(ns, attr)) is not None
        }
        extra = ns.slime_args[1:] if ns.slime_args[:1] == ["--"] else list(ns.slime_args)
        if extra:
            env["RSIBOX_MILES_SLIME_ARGS"] = json.dumps(extra)
        return env

    def __init__(self, receipts=None, *, model=None, data=None, custom_rm_path=None, slime_root=None,
                 megatron_root=None, extra_py_path=None, work=None, ray_gpus=None, slime_args=None,
                 check_timeout=None, rpc_timeout=None, boot_timeout=None, mode="sync",
                 runtime_factory=None, fresh_runner=None):
        env = os.environ
        self.model = model if model is not None else env.get("RSIBOX_MILES_MODEL")
        self.data = data if data is not None else env.get("RSIBOX_MILES_DATA")
        self.custom_rm_path = custom_rm_path if custom_rm_path is not None else env.get("RSIBOX_MILES_CUSTOM_RM_PATH")
        self.slime_root = slime_root if slime_root is not None else env.get("RSIBOX_MILES_SLIME_ROOT")
        self.megatron_root = megatron_root if megatron_root is not None else env.get("RSIBOX_MILES_MEGATRON_ROOT")
        self.extra_py_path = extra_py_path if extra_py_path is not None else env.get("RSIBOX_MILES_EXTRA_PY_PATH")
        self.ray_gpus = int(ray_gpus if ray_gpus is not None else env.get("RSIBOX_MILES_RAY_GPUS", "2"))
        self.slime_args = _parse_extra_args(slime_args if slime_args is not None else env.get("RSIBOX_MILES_SLIME_ARGS"))
        self.boot_timeout = float(boot_timeout if boot_timeout is not None else env.get("RSIBOX_MILES_BOOT_TIMEOUT", "420.0"))
        self._fresh_runner = fresh_runner
        work_cfg = work if work is not None else env.get("RSIBOX_MILES_WORK")
        if work_cfg:
            self.work_dir = os.path.abspath(work_cfg); os.makedirs(self.work_dir, exist_ok=True); self._owns_work = False
        else:
            self.work_dir = tempfile.mkdtemp(prefix="rsibox-miles-"); self._owns_work = True
        self.rollout_pt = os.path.join(self.work_dir, "rollout0.pt")
        self._cluster = {
            "ray": None, "rollout_manager": None, "actor_model": None,
            "trainers": {}, "trainer0": None, "rollout_data_0": None,
        }
        eff_check = float(check_timeout if check_timeout is not None else env.get("RSIBOX_MILES_CHECK_TIMEOUT", "15.0"))
        eff_rpc = float(rpc_timeout if rpc_timeout is not None else env.get("RSIBOX_MILES_RPC_TIMEOUT", "60.0"))
        opts = {"runtime_factory": runtime_factory} if callable(runtime_factory) else {}
        try:
            super().__init__(receipts=receipts, mode=mode, rpc_timeout=eff_rpc, check_timeout=eff_check, **opts)
        except BaseException:
            if self._owns_work and os.path.isdir(self.work_dir):
                shutil.rmtree(self.work_dir, ignore_errors=True)
            raise

    def _pythonpath(self, overlay_dir=None):
        return build_pythonpath(
            slime_root=self.slime_root, megatron_root=self.megatron_root,
            extra_py_path=self.extra_py_path, overlay_dir=overlay_dir,
        )

    def _config_env(self, overlay_dir=None):
        env = {**os.environ, **RAY_OTEL_ENV_VARS, "PYTHONPATH": self._pythonpath(overlay_dir=overlay_dir)}
        for attr, key, _ in _ENV_SPEC:
            if attr != "work" and (val := getattr(self, attr, None)) is not None:
                env[key] = str(val)
        if self.slime_args:
            env["RSIBOX_MILES_SLIME_ARGS"] = json.dumps(self.slime_args)
        return env

    def _shutdown_cluster(self):
        ray = self._cluster.get("ray")
        try:
            if ray is not None and ray.is_initialized():
                ray.shutdown(); time.sleep(3.0)
        finally:
            self._cluster.update(
                ray=None, rollout_manager=None, actor_model=None,
                trainers={}, trainer0=None, rollout_data_0=None,
            )

    def make_runtime(self):
        if callable(launcher := self.options.get("runtime_factory")):
            return launcher()
        if not self.model or not self.data:
            raise ValueError("RSIBOX_MILES_MODEL and RSIBOX_MILES_DATA are required (--model / --data)")
        ray, rm, am, trainers, rollout_ref, _ = _launch_slime_cluster(
            self.model, self.data, custom_rm_path=self.custom_rm_path, extra_args=self.slime_args,
            pythonpath=self._pythonpath(), num_gpus=self.ray_gpus, rollout_pt=self.rollout_pt, save_rollout=True,
        )
        self._cluster.update(
            ray=ray, rollout_manager=rm, actor_model=am,
            trainers=trainers, trainer0=trainers["trainer0"], rollout_data_0=rollout_ref,
        )
        try:
            ray.get(
                [h.__ray_call__.remote(snapshot_trainer) for h in trainers.values()],
                timeout=self.rpc_timeout,
            )
            base_text = {m: H.read_module(m) for m in CONFORMANCE_MODULES}
        except BaseException:
            self._shutdown_cluster()
            raise

        def restore_s0():
            results = ray.get(
                [h.__ray_call__.remote(restore_trainer) for h in trainers.values()],
                timeout=self.rpc_timeout,
            )
            bad = {
                name: res for name, res in zip(trainers, results)
                if not isinstance(res, dict) or not res.get("ok") or not res.get("weights_backuper_refreshed")
            }
            if bad:
                raise RuntimeError(f"trainer S0 restore failed: {bad}")

        coupled = CoupledWeightReset(lambda: {"tag": "s0"}, lambda _: restore_s0(), am.update_weights)
        coupled.initial_states = {"tag": "s0"}

        def run_step(workload=None):
            ray.get(am.async_train(0, self._cluster["rollout_data_0"]), timeout=self.check_timeout)
            am.update_weights()
            return _collect_weight_digest(ray, trainers, timeout=self.check_timeout)

        return {
            "rollout_actors": {"rollout_mgr": rm}, "train_actors": dict(trainers),
            "modules": list(CONFORMANCE_MODULES), "base_text": base_text,
            "coupled_reset": coupled, "run_step": run_step, "shutdown": self._shutdown_cluster,
        }

    def patches(self):
        return _miles_patches(self.base_text)

    def inject(self, fault, **kwargs):
        if callable(custom := self.runtime.get("inject")):
            return custom(fault, **kwargs)
        ray = self._cluster.get("ray")
        trainers = self._cluster.get("trainers") or (
            {"trainer0": self._cluster["trainer0"]} if self._cluster.get("trainer0") is not None else {}
        )
        target_name = str(kwargs.get("participant", "trainer0"))
        target = trainers.get(target_name)
        if ray is None or target is None:
            return {"ok": False, "error": f"no live {target_name} actor"}
        try:
            if fault == "lost":
                ray.kill(target, no_restart=True)
                return {"ok": True, "fault": "lost"}
            hang_s = float(kwargs.get("hang_s", self.check_timeout + 10.0))
            return ray.get(
                target.__ray_call__.remote(
                    trainer_inject_fault, fault, kwargs.get("after", 1), hang_s, target_name
                ),
                timeout=self.rpc_timeout,
            )
        except Exception as err:
            return {"ok": False, "error": f"{type(err).__name__}: {err}"}

    def fresh_check(self, patch, workload=None):
        try:
            files = self.files(patch, against="disk")
        except ValueError as err:
            return {"ok": False, "error": str(err), "output": None}
        if callable(self._fresh_runner):
            return self._fresh_runner(files, workload)
        saved_diffs = [
            {m: src for m, src in self.live_text[i].items() if src != self.live_text[i - 1].get(m)}
            for i in range(1, len(self.live_text))
        ]
        tmp = tempfile.mkdtemp(prefix="rsibox-miles-fresh-")
        overlay = os.path.join(tmp, "overlay")
        stopped = False
        try:
            if files:
                _build_fresh_overlay(files, overlay)
            stopped = True
            self._stop_engine()
            fjson = os.path.join(tmp, "fresh.json")
            cmd = [sys.executable, "-u", "-m", "adapter.conformance_lab", "--fresh-json", fjson,
                   "--rollout-pt", self.rollout_pt, *(["--overlay", overlay] if files else [])]
            t0 = time.perf_counter()
            with subprocess.Popen(
                cmd,
                env=self._config_env(overlay_dir=overlay if files else None),
                text=True,
                start_new_session=True,
            ) as proc:
                try:
                    rc = proc.wait(timeout=self.boot_timeout)
                except subprocess.TimeoutExpired:
                    P.kill_tree(proc.pid, signal.SIGKILL, include_pg=True)
                    proc.wait()
                    return {"ok": False, "error": f"fresh timed out after {self.boot_timeout}s", "output": None}
                except BaseException:
                    if proc.poll() is None:
                        P.kill_tree(proc.pid, signal.SIGKILL, include_pg=True)
                        proc.wait()
                    raise
            if rc != 0:
                return {"ok": False, "error": f"fresh rc={rc}", "output": None}
            return {"ok": True, "output": json.loads(Path(fjson).read_text(encoding="utf-8"))["output"],
                    "cost_s": round(time.perf_counter() - t0, 4)}
        except Exception as err:
            return {"ok": False, "error": f"{type(err).__name__}: {err}", "output": None}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            if stopped:
                self._spawn()
                for diff_files in saved_diffs:
                    if not (re_res := self.install({"files": diff_files})).get("ok"):
                        raise RuntimeError(f"failed to replay live stack after fresh_check: {re_res}")

    def close(self):
        super().close()
        if getattr(self, "_owns_work", False) and os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)


def _run_fresh_mode(out_json, *, rollout_pt=None, overlay_dir=None):
    """Subprocess entrypoint for `MilesConformanceLab.fresh_check`."""
    t0, env = time.perf_counter(), os.environ
    py = build_pythonpath(
        slime_root=env.get("RSIBOX_MILES_SLIME_ROOT"), megatron_root=env.get("RSIBOX_MILES_MEGATRON_ROOT"),
        extra_py_path=env.get("RSIBOX_MILES_EXTRA_PY_PATH"), overlay_dir=overlay_dir,
    )
    ray, _, am, trainers, rollout_ref, t_ready = _launch_slime_cluster(
        env["RSIBOX_MILES_MODEL"], env["RSIBOX_MILES_DATA"],
        custom_rm_path=env.get("RSIBOX_MILES_CUSTOM_RM_PATH"),
        extra_args=_parse_extra_args(env.get("RSIBOX_MILES_SLIME_ARGS")),
        pythonpath=py, num_gpus=int(env.get("RSIBOX_MILES_RAY_GPUS", "2")),
        rollout_pt=rollout_pt, prog="rsibox-miles-fresh",
    )
    try:
        ray.get(am.async_train(0, rollout_ref))
        am.update_weights()
        t_step = time.perf_counter()
        output = _collect_weight_digest(ray, trainers)
    finally:
        ray.shutdown()
    Path(out_json).write_text(json.dumps({
        "boot_to_ready_s": round(t_ready - t0, 4), "step_s": round(t_step - t_ready, 4),
        "total_s": round(t_step - t0, 4), "output": output,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MilesConformanceLab fresh subprocess worker.")
    ap.add_argument("--fresh-json", required=True)
    ap.add_argument("--rollout-pt", default=None)
    ap.add_argument("--overlay", default=None)
    ns = ap.parse_args()
    _run_fresh_mode(ns.fresh_json, rollout_pt=ns.rollout_pt, overlay_dir=ns.overlay)
