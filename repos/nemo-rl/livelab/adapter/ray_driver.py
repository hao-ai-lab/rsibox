"""Real NVIDIA NeMo-RL (AutoModel FSDP2 + SGLang GRPO) Ray runtime driver for `NemoRlLab`.

Pure Python stdlib + `core/livelab` at import time (zero eager `torch`, `ray`, `nemo_rl`,
`sglang`, or `transformers` imports). Imported lazily by `NemoRlLab` in `mode="ray"`.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from adapter.worker import (
    NemoRlRayParticipant,
    discover_nemo_rl_actors,
    restore_automodel_s0,
    snapshot_automodel_s0,
)
from rsibox import harness as H
from rsibox import proc as P

CLIPPED_PG_MOD = "nemo_rl.algorithms.loss.clipped_pg"
LOSS_FN_MOD = "nemo_rl.algorithms.loss.loss_functions"
AUTOMODEL_TRAIN_MOD = "nemo_rl.models.automodel.train"
AUTOMODEL_SETUP_MOD = "nemo_rl.models.automodel.setup"
REAL_MODULES = (CLIPPED_PG_MOD, LOSS_FN_MOD, AUTOMODEL_TRAIN_MOD, AUTOMODEL_SETUP_MOD)

_RAY_ENV_DEFAULTS = {
    "NEMO_RL_PY_EXECUTABLES_SYSTEM": "1",
    "NRL_DISABLE_QWENVL_VISION_PATCH": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "RAY_ENABLE_OPEN_TELEMETRY": "0",
    "RAY_enable_open_telemetry": "0",
    "RAY_enable_metrics_collection": "0",
}


def _apply_runtime_env_defaults() -> None:
    for key, val in _RAY_ENV_DEFAULTS.items():
        os.environ.setdefault(key, val)
    ll = str(Path(__file__).resolve().parents[1])
    cur = [p for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if p]
    if ll not in cur:
        os.environ["PYTHONPATH"] = os.pathsep.join([*cur, ll])


def _ensure_sglang_safe_unpickler_compat() -> None:
    """Adapt NeMo-RL's SGLang SafeUnpickler compat patch for SGLang >= 0.5.21 (`ALLOWED_GLOBALS`)."""
    import sglang.srt.utils.common as sg_common
    import nemo_rl.models.generation.sglang.utils.patches as nrl_patches

    safe_cls = getattr(sg_common, "SafeUnpickler", None)
    if safe_cls is None or not hasattr(safe_cls, "ALLOWED_GLOBALS"):
        return
    common_path = sg_common.__file__
    marker = '("nemo_rl.models.generation.sglang.utils.train_utils", "_rebuild_cuda_tensor_modified")'
    with open(common_path, encoding="utf-8") as fh:
        src = fh.read()
    if marker not in src:
        anchor = '        ("sglang.srt.model_executor.model_runner", "LocalSerializedTensor"),\n'
        if anchor not in src:
            raise RuntimeError(f"SafeUnpickler ALLOWED_GLOBALS anchor not found in {common_path}")
        addition = (
            '        ("nemo_rl.models.generation.sglang.utils.train_utils", "_rebuild_cuda_tensor_modified"),\n'
            '        ("nemo_rl.models.generation.sglang.utils.train_utils", "FlattenedTensorMetadata"),\n'
            '        ("nemo_rl.models.generation.sglang.utils.train_utils", "FlattenedTensorBucket"),\n'
        )
        with open(common_path, "w", encoding="utf-8") as fh:
            fh.write(src.replace(anchor, anchor + addition, 1))

    def _compat_apply() -> None:
        nrl_patches._patch_sglang_transformers_compat_bootstrap()
        nrl_patches._override_sglang_imbalance_check_env()
        nrl_patches._patch_megatron_dynamic_context_hook_mode()

    nrl_patches._apply_sglang_compat_patches = _compat_apply


def _build_grpo_config(cfg: dict[str, Any]) -> dict[str, Any]:
    import nemo_rl
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    cfg_dir = Path(nemo_rl.__file__).resolve().parents[1] / "examples" / "configs"
    merged = OmegaConf.to_container(
        OmegaConf.merge(load_config(str(cfg_dir / "grpo_math_1B.yaml")), load_config(str(cfg_dir / "grpo_math_1B_sglang.yaml"))),
        resolve=False,
    )
    model, ngpus = str(cfg["model"]), int(cfg.get("num_trainers", 1))
    group_size = max(1, int(cfg.get("group_size", 4)))
    global_bs = 2 * group_size
    micro_bs = max(1, min(2, global_bs // max(ngpus, 1)))
    while global_bs % (micro_bs * max(ngpus, 1)) != 0 and micro_bs > 1:
        micro_bs -= 1
    ctx_len, max_new_tokens, mem_frac = int(cfg.get("context_length", 128)), int(cfg.get("max_new_tokens", 8)), float(cfg.get("mem_fraction", 0.35))

    pol = merged["policy"]
    pol.update({"model_name": model, "train_global_batch_size": global_bs, "train_micro_batch_size": micro_bs, "logprob_batch_size": micro_bs, "max_total_sequence_length": ctx_len})
    pol["tokenizer"]["name"] = model
    pol["automodel_cfg"].update({"enabled": True, "deterministic": True, "automodel_kwargs": {"attn_implementation": "sdpa"}})
    for key in ("sequence_packing", "dynamic_batching"):
        if key in pol:
            pol[key]["enabled"] = False

    gen = pol["generation"]
    gen["max_new_tokens"] = max_new_tokens
    sc = gen["sglang_cfg"]
    sc.update({"model_path": model, "context_length": ctx_len, "mem_fraction_static": mem_frac, "disable_cuda_graph": True, "tp_size": ngpus, "max_running_requests": 1})
    sc["sglang_server_config"].update({"num_gpus": ngpus, "num_gpus_per_engine": ngpus, "sglang_server_concurrency": 1})
    merged["cluster"].update({"gpus_per_node": ngpus, "num_nodes": 1})
    return OmegaConf.to_container(OmegaConf.create(merged), resolve=True)


def _init_trainer_actor(actor: Any, seed: int) -> dict[str, Any]:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    actor.step_counter = 0
    snap = snapshot_automodel_s0(actor)

    def _reset_s0() -> list[str]:
        return restore_automodel_s0(actor)["reset"]

    actor.reset_state = _reset_s0
    return snap


def _trainer_wsum(actor: Any) -> float:
    total = 0.0
    for param in actor.model.parameters():
        local_p = param.to_local() if hasattr(param, "to_local") else param
        total += float(local_p.detach().float().abs().sum().item())
    return round(total, 6)


def _init_rollout_actor(actor: Any) -> dict[str, Any]:
    def _reset_s0() -> list[str]:
        actor.invalidate_kv_cache()
        return ["kv_cache"]

    actor.reset_state = _reset_s0
    return {"ok": True}


def _inject_actor_fault(actor: Any, fault: str, hang_s: float = 240.0) -> dict[str, Any]:
    import nemo_rl.algorithms.loss.clipped_pg as clipped_pg
    import nemo_rl.algorithms.loss.loss_functions as loss_fn_mod

    orig = clipped_pg.clipped_pg_actor_objective

    def _faulted(*args: Any, **kwargs: Any) -> Any:
        clipped_pg.clipped_pg_actor_objective = orig
        loss_fn_mod.clipped_pg_actor_objective = orig
        if fault == "crash":
            raise RuntimeError("trainer injected crash in clipped_pg_actor_objective")
        time.sleep(float(hang_s))
        return orig(*args, **kwargs)

    clipped_pg.clipped_pg_actor_objective = _faulted
    loss_fn_mod.clipped_pg_actor_objective = _faulted
    return {"ok": True, "fault": fault}


def _inject_participant_fault(actor: Any, name: str, fault: str, **kwargs: Any) -> dict[str, Any]:
    import ray_participants as RP

    lp = RP._LOCAL.get(name)
    if lp is None:
        return {"ok": False, "error": f"participant {name!r} not initialized"}
    return H.inject_participant_fault(lp, fault, **kwargs)


def _run_grpo_steps(st: dict[str, Any], *, steps: int, group_size: int, rpc_timeout: float) -> dict[str, Any]:
    import torch
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    policy, gen, ws, tok, loss_fn = st["policy"], st["gen"], st["ws"], st["tok"], st["loss_fn"]
    workers, ray_mod = st["trainers"], st["ray"]
    prompts = [p for p in ("Solve: 1+1=", "Solve: 2+2=") for _ in range(group_size)]
    losses, grad_norms, out_token_sums, wsums = [], [], [], []
    for _ in range(max(1, int(steps))):
        gen.prepare_for_generation()
        ids = tok(prompts, return_tensors="pt", padding=True)
        in_lens = ids["attention_mask"].sum(dim=-1)
        out = gen.generate(BatchedDataDict({"input_ids": ids["input_ids"], "input_lengths": in_lens}), greedy=True)
        gen.finish_generation()

        out_ids, gen_lps, unpadded = out["output_ids"], out["logprobs"], out["unpadded_sequence_lengths"]
        bsz, seqlen = out_ids.shape
        tok_mask = torch.zeros((bsz, seqlen), dtype=torch.float32)
        for i in range(bsz):
            p_len, u_len = int(in_lens[i].item()), int(unpadded[i].item())
            if u_len > p_len:
                tok_mask[i, p_len:u_len] = 1.0
        sample_mask = torch.ones((bsz,), dtype=torch.float32)
        adv = torch.zeros((bsz, seqlen), dtype=torch.float32)
        for g0 in range(0, bsz, group_size):
            for j in range(group_size):
                adv[g0 + j] = 0.75 if j % 2 == 0 else -0.25

        lp_data = BatchedDataDict({"input_ids": out_ids, "input_lengths": unpadded, "token_mask": tok_mask, "sample_mask": sample_mask})
        policy.prepare_for_lp_inference()
        prev_lps = policy.get_logprobs(lp_data)["logprobs"]
        train_data = BatchedDataDict({
            "input_ids": out_ids, "input_lengths": unpadded, "token_mask": tok_mask, "sample_mask": sample_mask,
            "generation_logprobs": gen_lps, "prev_logprobs": prev_lps,
            "reference_policy_logprobs": prev_lps - 0.05, "advantages": adv,
        })
        policy.prepare_for_training()
        tr = policy.train(train_data, loss_fn)
        policy.prepare_for_lp_inference()
        ws.sync_weights()
        wsums = ray_mod.get([h.__ray_call__.remote(_trainer_wsum) for h in workers.values()], timeout=rpc_timeout)
        losses.append(round(float(tr["loss"].mean().item()), 6))
        grad_norms.append(round(float(tr["grad_norm"].mean().item()), 6))
        out_token_sums.append(int(out_ids.sum().item()))

    summary = {"losses": losses, "grad_norms": grad_norms, "out_token_sums": out_token_sums, "weight_checksum": round(sum(float(x) for x in wsums), 4)}
    summary["digest"] = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()[:16]
    return summary


def _start_cluster(cfg: dict[str, Any], *, rpc_timeout: float) -> dict[str, Any]:
    _apply_runtime_env_defaults()
    _ensure_sglang_safe_unpickler_compat()
    import ray
    from transformers import AutoTokenizer
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.weight_sync.factory import create_weight_synchronizer

    resolved = _build_grpo_config(cfg)
    ngpus = int(cfg.get("num_trainers", 1))
    seed = int(cfg.get("seed", 0))
    if ray.is_initialized():
        ray.shutdown()
        time.sleep(1.0)
    init_ray()
    gen = policy = cluster = None
    try:
        tok = AutoTokenizer.from_pretrained(str(cfg["model"]))
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        gen_cfg = configure_generation_config(resolved["policy"]["generation"], tok)
        cluster = RayVirtualCluster(
            name="rl_cluster",
            bundle_ct_per_node_list=[ngpus],
            use_gpus=True,
            num_gpus_per_node=ngpus,
            max_colocated_worker_groups=2,
        )
        gen = SGLangGeneration(cluster=cluster, sglang_cfg=gen_cfg)
        gen.finish_generation()
        policy = Policy(cluster=cluster, config=resolved["policy"], tokenizer=tok, init_optimizer=True, init_reference_model=False)
        ws = create_weight_synchronizer(
            policy=policy, generation=gen, generation_backend="sglang", colocated=True, train_cluster=cluster, inference_cluster=cluster
        )
        ws.init_communicator()
        policy.prepare_for_lp_inference()
        ws.sync_weights()

        _, trainers, rollouts = discover_nemo_rl_actors(
            policy_group=policy.worker_group, rollout_group=gen.engines
        )
        ray.get([h.__ray_call__.remote(_init_trainer_actor, seed) for h in trainers.values()], timeout=rpc_timeout)
        ray.get([h.__ray_call__.remote(_init_rollout_actor) for h in rollouts.values()], timeout=rpc_timeout)
        loss_fn = ClippedPGLossFn(ClippedPGLossConfig(
            reference_policy_kl_penalty=0.01, ratio_clip_min=0.2, ratio_clip_max=0.2, ratio_clip_c=None,
            loss_type="token_level", use_on_policy_kl_approximation=False, use_importance_sampling_correction=False,
            sequence_level_importance_ratios=False, force_on_policy_ratio=False,
        ))
        return {"ray": ray, "cluster": cluster, "gen": gen, "policy": policy, "ws": ws, "tok": tok, "loss_fn": loss_fn, "trainers": trainers, "rollouts": rollouts}
    except BaseException:
        for obj in (gen, policy, cluster):
            if obj is not None:
                with contextlib.suppress(Exception):
                    obj.shutdown()
        with contextlib.suppress(Exception):
            ray.shutdown()
        raise


def _shutdown_cluster(st: dict[str, Any], *, dirty: bool = False) -> None:
    if dirty:
        for handle in [*st["trainers"].values(), *st["rollouts"].values()]:
            with contextlib.suppress(Exception):
                st["ray"].kill(handle, no_restart=True)
    else:
        for obj in (st["gen"], st["policy"], st["cluster"]):
            with contextlib.suppress(Exception):
                obj.shutdown()
    with contextlib.suppress(Exception):
        if st["ray"].is_initialized():
            st["ray"].shutdown()
            time.sleep(1.5)
    gc.collect()


def run_fresh_subprocess(
    spec_path: str, out_path: str, *, env: dict[str, str], overlay_root: str | None = None, timeout_s: float = 360.0
) -> dict[str, Any]:
    popen_cwd = overlay_root if overlay_root and os.path.isdir(overlay_root) else os.path.dirname(spec_path)
    t0 = time.perf_counter()
    with subprocess.Popen([sys.executable, "-u", "-m", "adapter.ray_driver", spec_path], env=env, cwd=popen_cwd, start_new_session=True) as child:
        try:
            rc = child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": None, "error": f"fresh timed out after {timeout_s}s"}
        finally:
            with contextlib.suppress(Exception):
                P.kill_tree(child.pid, signal.SIGKILL, include_pg=True)
                child.wait(timeout=5)
    if rc != 0:
        return {"ok": False, "output": None, "error": f"fresh rc={rc}"}
    with open(out_path, encoding="utf-8") as fh:
        return {"ok": True, "output": json.load(fh)["output"], "cost_s": round(time.perf_counter() - t0, 4)}


def launch_nemo_rl_runtime(lab: Any) -> dict[str, Any]:
    """Launch real Ray + AutoModel FSDP2 + SGLang runtime for `NemoRlLab`."""
    cfg = dict(lab.config)
    if not str(cfg.get("model") or "").strip():
        raise ValueError("RSIBOX_NEMORL_MODEL (--model) is required for NemoRlLab real mode")
    st = _start_cluster(cfg, rpc_timeout=lab.rpc_timeout)
    try:
        ray, trainers, rollouts = st["ray"], st["trainers"], st["rollouts"]
        base_text = {mod: H.read_module(mod) for mod in lab._init_modules}
        parts = [NemoRlRayParticipant(name, handle, timeout_s=lab.rpc_timeout) for name, handle in [*trainers.items(), *rollouts.items()]]
    except BaseException:
        _shutdown_cluster(st, dirty=True)
        raise

    state_flags = {"at_s0": True}

    def reset_state() -> dict[str, Any]:
        ray.get([h.__ray_call__.remote(restore_automodel_s0) for h in trainers.values()], timeout=lab.rpc_timeout)
        st["policy"].prepare_for_lp_inference()
        st["ws"].sync_weights()
        st["gen"].invalidate_kv_cache()
        state_flags["at_s0"] = True
        return {"ok": True, "reset": ["weights_s0", "optimizer_state", "scheduler_state", "rng", "rollout_weights", "kv_cache"]}

    def run_step(workload: dict[str, Any] | None = None) -> dict[str, Any]:
        wl = {**lab.default_workload(), **(workload or {})}
        if not wl.get("continue_state", False) and not state_flags["at_s0"]:
            reset_state()
        state_flags["at_s0"] = False
        return _run_grpo_steps(st, steps=int(wl.get("steps", cfg.get("steps", 2))), group_size=int(cfg.get("group_size", 4)), rpc_timeout=lab.rpc_timeout)

    def probe() -> tuple[bool, str | None]:
        try:
            for name, handle in [*trainers.items(), *rollouts.items()]:
                rep = ray.get(handle.__ray_call__.remote(lambda _a, n=name: {"ok": True, "name": n}), timeout=min(10.0, lab.rpc_timeout))
                if not rep.get("ok"):
                    return False, f"{name} probe failed"
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def inject(fault: str, **kwargs: Any) -> dict[str, Any]:
        target_name = str(kwargs.pop("participant", "trainer0"))
        target = trainers.get(target_name) or rollouts.get(target_name)
        if target is None:
            return {"ok": False, "error": f"unknown participant {target_name!r}"}
        try:
            if fault == "lost":
                ray.kill(target, no_restart=True)
                return {"ok": True, "fault": "lost"}
            if fault in ("crash", "hang"):
                default_hang = float(lab.check_timeout) + float(lab.rpc_timeout) + 60.0
                hang_s = float(kwargs.get("seconds") or kwargs.get("hang_s") or default_hang)
                return ray.get(target.__ray_call__.remote(_inject_actor_fault, fault, hang_s), timeout=lab.rpc_timeout)
            if fault == "tamper":
                kwargs.setdefault("module", CLIPPED_PG_MOD)
                kwargs.setdefault("qualname", "clipped_pg_actor_objective")
            return ray.get(target.__ray_call__.remote(_inject_participant_fault, target_name, fault, **kwargs), timeout=lab.rpc_timeout)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def fresh_check(patch: dict[str, Any] | None, workload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            files = lab.files(patch, against="disk")
        except ValueError as exc:
            return {"ok": False, "output": None, "error": str(exc)}
        saved_diffs = [
            {m: src for m, src in lab.live_text[i].items() if src != lab.live_text[i - 1].get(m)}
            for i in range(1, len(lab.live_text))
        ]
        ll, core = lab._livelab_paths()
        tmp = tempfile.mkdtemp(prefix="rsibox-nemorl-fresh-")
        overlay = os.path.join(tmp, "overlay")
        stopped = False
        try:
            if files:
                H.build_package_overlay("nemo_rl", files, dst=overlay)
                import nemo_rl

                repo_root = Path(nemo_rl.__file__).resolve().parents[1]
                for extra_name in ("pyproject.toml", "examples"):
                    src_extra = repo_root / extra_name
                    dst_extra = Path(overlay) / extra_name
                    if src_extra.exists() and not dst_extra.exists():
                        os.symlink(src_extra, dst_extra)
            stopped = True
            lab._stop_engine()
            spec_path = os.path.join(tmp, "spec.json")
            out_path = os.path.join(tmp, "fresh.json")
            overlay_root = overlay if files else ""
            with open(spec_path, "w", encoding="utf-8") as fh:
                json.dump({"config": dict(cfg), "workload": {**lab.default_workload(), **(workload or {})}, "out": out_path, "rpc_timeout": lab.rpc_timeout, "overlay_root": overlay_root}, fh)
            existing = [p for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if p]
            py_entries = [*([overlay] if files else []), ll, core, *existing]
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.pathsep.join(dict.fromkeys(py_entries))}
            boot_to = float(cfg.get("boot_timeout", 300.0)) + lab.check_timeout
            return run_fresh_subprocess(spec_path, out_path, env=env, overlay_root=overlay_root or None, timeout_s=boot_to)
        except Exception as exc:
            return {"ok": False, "output": None, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            if stopped:
                lab._spawn()
                for diff_files in saved_diffs:
                    if not (re_res := lab.install({"files": diff_files})).get("ok"):
                        raise RuntimeError(f"failed to replay live stack after fresh_check: {re_res}")

    def shutdown() -> None:
        _shutdown_cluster(st, dirty=bool(lab._dirty))

    return {
        "participants": parts, "modules": list(lab._init_modules), "base_text": base_text,
        "reset_state": reset_state, "run_step": run_step, "probe": probe, "inject": inject,
        "fresh_check": fresh_check, "patches": lambda: real_patches(lab._base_source), "shutdown": shutdown,
    }


def real_patches(base_source: Any) -> dict[str, Any]:
    """Return the 11-case conformance patch table over upstream `nemo_rl` modules."""
    e = H.edit_once
    cpg = base_source(CLIPPED_PG_MOD)
    lf = base_source(LOSS_FN_MOD)
    tr = base_source(AUTOMODEL_TRAIN_MOD)
    st = base_source(AUTOMODEL_SETUP_MOD)

    vis1 = e(cpg, "        clip_loss = torch.maximum(-advantages * ratios, -advantages * ratios_clamped)", "        clip_loss = torch.maximum(-advantages * ratios, -advantages * ratios_clamped) * 1.5")
    vis2 = e(vis1, "        clip_loss = torch.maximum(-advantages * ratios, -advantages * ratios_clamped) * 1.5", "        clip_loss = torch.maximum(-advantages * ratios, -advantages * ratios_clamped) * 2.5")
    tr_v3 = e(tr, "        return loss, loss_metrics", "        return loss * 1.25, loss_metrics")
    multi_cpg = e(vis1, "    gen_kl_error = masked_mean(", "    gen_kl_error = 2.0 * masked_mean(")
    return {
        "noop": {"files": {CLIPPED_PG_MOD: e(cpg, "        log_ratios = curr_logprobs - prev_logprobs", "        log_ratios = (curr_logprobs - prev_logprobs) + 0.0")}},
        "visible": {"files": {CLIPPED_PG_MOD: vis1}},
        "chain": [{"files": {CLIPPED_PG_MOD: vis1}}, {"files": {CLIPPED_PG_MOD: vis2}}, {"files": {AUTOMODEL_TRAIN_MOD: tr_v3}}],
        "multi": {"files": {CLIPPED_PG_MOD: multi_cpg, AUTOMODEL_TRAIN_MOD: tr_v3}},
        "unreached": {"files": {AUTOMODEL_SETUP_MOD: e(st, "                **optimizer_kwargs,", '                **{**optimizer_kwargs, "lr": optimizer_kwargs["lr"] * 5.0},')}},
        "refused": {
            "syntax": {"files": {CLIPPED_PG_MOD: e(cpg, "    return actor_loss, ratios, ratios_clamped", "    return (")}},
            "closure": {"files": {LOSS_FN_MOD: e(lf, "            return self.objective_reduction.reduce(\n                per_token_values,", "            return self.objective_reduction.reduce(\n                per_token_values + 0.0 * opd_full_divergence.sum(),")}},
            "module_state": {"files": {CLIPPED_PG_MOD: e(cpg, "from typing import Optional", "from typing import Optional\nNEMORL_EXTRA_STATE = 1")}},
            "empty": {"files": {CLIPPED_PG_MOD: e(cpg, "def clipped_pg_actor_objective(", "def clipped_pg_actor_objective(  # note\n")}},
        },
    }


def run_fresh_ray(spec: dict[str, Any]) -> int:
    ov = spec.get("overlay_root")
    if ov:
        sys.path[:] = [ov, *[p for p in sys.path if p != ov]]
    cfg, wl, rpc_to = dict(spec["config"]), dict(spec.get("workload") or {}), float(spec.get("rpc_timeout", 60.0))
    st = _start_cluster(cfg, rpc_timeout=rpc_to)
    try:
        output = _run_grpo_steps(st, steps=int(wl.get("steps", cfg.get("steps", 2))), group_size=int(cfg.get("group_size", 4)), rpc_timeout=rpc_to)
        with open(spec["out"], "w", encoding="utf-8") as fh:
            json.dump({"output": output}, fh)
        return 0
    finally:
        _shutdown_cluster(st)


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        raise SystemExit(run_fresh_ray(json.load(fh)))
