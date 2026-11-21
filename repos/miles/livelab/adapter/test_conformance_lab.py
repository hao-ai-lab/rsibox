"""CPU unit tests for `MilesConformanceLab` and lazy-`torch` `TrainerStateSnapshot`."""
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import adapter  # noqa: F401
from adapter import conformance_lab as CL
from adapter import gate as G
from adapter import state as S
from conformance import cases as CC
from fastrsi import harness as H
from fastrsi import proc as P
import ray_participants as RP

_PPO_SRC = '''import torch
import torch.nn.functional as F

def _mul_helper(vocab_parallel_logits):
    if True:
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)
        return mul_reduce

def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns

def compute_policy_loss(ratio, advantages):
    pg_losses1 = -ratio * advantages
    return pg_losses1
'''

_LOSS_SRC = '''from slime.utils.ppo_utils import compute_policy_loss, get_grpo_returns

def compute_advantages_and_returns(rewards, kl):
    returns = get_grpo_returns(rewards, kl)
    if True:
        # TODO: is the copy necessary?
        advantages = [r for r in returns]
    return sum(compute_policy_loss(1.0, a) for a in advantages)
'''

_MODEL_SRC = '''class OptimizerConfig:
    def __init__(self, **kwargs):
        self.lr = kwargs.get("lr", 1e-4)
        self.timers = "unset"

def setup_model_and_optimizer(kwargs):
    config = OptimizerConfig(**kwargs)
    config.timers = None
    return config
'''


class _FakeTensor:
    """Minimal pure-Python stand-in for `torch.Tensor` to exercise `TrainerStateSnapshot` on CPU."""

    def __init__(self, values, dtype="float32", pinned=False):
        self._data = list(values)
        self.dtype = dtype
        self.shape = (len(self._data),)
        self._pinned = pinned

    def data_ptr(self):
        return id(self._data)

    def numel(self):
        return len(self._data)

    def element_size(self):
        return 4

    def is_pinned(self):
        return self._pinned

    def detach(self):
        return self

    def to(self, device, copy=False):
        return _FakeTensor(self._data, dtype=self.dtype)

    def copy_(self, other, non_blocking=False):
        self._data[:] = list(other._data)
        return self

    def zero_(self):
        self._data[:] = [0.0] * len(self._data)
        return self

    def clone(self):
        return _FakeTensor(self._data, dtype=self.dtype)

    def float(self):
        return self

    def abs(self):
        return _FakeTensor([abs(x) for x in self._data], dtype=self.dtype)

    def sum(self):
        return types.SimpleNamespace(item=lambda: float(sum(self._data)))

    def cpu(self):
        return self

    def numpy(self):
        return types.SimpleNamespace(tobytes=lambda: repr(self._data).encode("utf-8"))

    def __getitem__(self, item):
        sub = self._data[item]
        return _FakeTensor(sub if isinstance(sub, list) else [sub], dtype=self.dtype)

    @property
    def data(self):
        return self


def _make_fake_torch():
    rng = {"cpu": [1, 2, 3]}
    mod = types.ModuleType("torch")
    mod.__path__ = []
    mod.Tensor = _FakeTensor
    mod.Generator = type("Generator", (), {})
    mod.is_tensor = lambda obj: isinstance(obj, _FakeTensor)
    mod.empty = lambda shape, dtype="float32", pin_memory=False: _FakeTensor([0.0] * shape[0], dtype=dtype, pinned=pin_memory)
    mod.ones_like = lambda val: 1.0
    mod.no_grad = contextlib.nullcontext
    mod.get_rng_state = lambda: list(rng["cpu"])
    mod.set_rng_state = lambda st: rng.__setitem__("cpu", list(st))
    mod.cuda = types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None)
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.__path__ = []
    fn_mod = types.ModuleType("torch.nn.functional")
    nn_mod.functional = fn_mod
    mod.nn = nn_mod
    return mod, nn_mod, fn_mod


class _InProcessActor:
    def __init__(self, name, hidden_modules=()):
        self.name = name
        self.hidden_modules = tuple(hidden_modules)
        self._alive = True


class _FakeRemoteHandle:
    def __init__(self, actor):
        self._actor = actor

    @property
    def __ray_call__(self):
        actor = self._actor

        def _remote(fn, *args, **kwargs):
            if not actor._alive:
                raise RuntimeError(f"actor {actor.name} lost")
            stashed = {m: sys.modules.pop(m) for m in actor.hidden_modules if m in sys.modules}
            try:
                return fn(actor, *args, **kwargs)
            finally:
                sys.modules.update(stashed)

        return types.SimpleNamespace(remote=_remote)


class MilesConformanceLabTest(unittest.TestCase):

    def test_launch_environment_strips_separator_maps_flags_and_trailing_slime_args(self):
        env = CL.MilesConformanceLab.launch_environment([
            "--",
            "--model", "Qwen/Qwen3-0.6B",
            "--data", "/tmp/prompts.jsonl",
            "--custom-rm-path", "my_rm.score",
            "--slime-root", "/tmp/slime",
            "--megatron-root", "/tmp/megatron",
            "--extra-py-path", "/tmp/extra1:/tmp/extra2",
            "--work", "/tmp/miles-work",
            "--ray-gpus", "4",
            "--check-timeout", "25",
            "--rpc-timeout", "90",
            "--boot-timeout", "600",
            "--", "--num-layers", "12", "--lr", "2e-4",
        ])
        self.assertEqual(env["FASTRSI_MILES_MODEL"], "Qwen/Qwen3-0.6B")
        self.assertEqual(env["FASTRSI_MILES_DATA"], os.path.abspath("/tmp/prompts.jsonl"))
        self.assertEqual(env["FASTRSI_MILES_CUSTOM_RM_PATH"], "my_rm.score")
        self.assertEqual(env["FASTRSI_MILES_SLIME_ROOT"], os.path.abspath("/tmp/slime"))
        self.assertEqual(env["FASTRSI_MILES_MEGATRON_ROOT"], os.path.abspath("/tmp/megatron"))
        self.assertEqual(env["FASTRSI_MILES_EXTRA_PY_PATH"], "/tmp/extra1:/tmp/extra2")
        self.assertEqual(env["FASTRSI_MILES_WORK"], os.path.abspath("/tmp/miles-work"))
        self.assertEqual(env["FASTRSI_MILES_RAY_GPUS"], "4")
        self.assertEqual(env["FASTRSI_MILES_CHECK_TIMEOUT"], "25.0")
        self.assertEqual(env["FASTRSI_MILES_RPC_TIMEOUT"], "90.0")
        self.assertEqual(env["FASTRSI_MILES_BOOT_TIMEOUT"], "600.0")
        self.assertEqual(json.loads(env["FASTRSI_MILES_SLIME_ARGS"]), ["--num-layers", "12", "--lr", "2e-4"])

    def test_ray_init_kwargs_disables_lowercase_otel_and_system_config(self):
        py = CL.build_pythonpath(
            slime_root="/tmp/slime",
            megatron_root="/tmp/megatron",
            extra_py_path="/tmp/extra1:/tmp/slime",
            overlay_dir="/tmp/overlay",
            base_env={"PYTHONPATH": "/tmp/base"},
        )
        parts = py.split(os.pathsep)
        self.assertEqual(parts[0], "/tmp/overlay")
        self.assertEqual(parts.count("/tmp/slime"), 1)
        self.assertFalse(any("repos/megatron" in p for p in parts), parts)

        kw = CL.ray_init_kwargs(py, num_gpus=2)
        self.assertEqual(kw["num_gpus"], 2)
        self.assertFalse(kw["include_dashboard"])
        self.assertEqual(
            kw["_system_config"],
            {"enable_open_telemetry": False, "enable_metrics_collection": False},
        )
        env_vars = kw["runtime_env"]["env_vars"]
        self.assertEqual(env_vars["RAY_ENABLE_OPEN_TELEMETRY"], "0")
        self.assertEqual(env_vars["RAY_enable_open_telemetry"], "0")
        self.assertEqual(env_vars["RAY_enable_metrics_collection"], "0")
        self.assertEqual(env_vars["PYTHONPATH"], py)

    def test_real_help_cli_imports_zero_gpu_or_ray_modules(self):
        livelab_miles = Path(__file__).resolve().parents[1]
        repo_root = livelab_miles.parents[2]
        core_livelab = repo_root / "core" / "livelab"
        probe = (
            "import sys\n"
            "from fastrsi import cli\n"
            "before = set(sys.modules)\n"
            "try:\n"
            "    rc = cli.main(['conformance', 'miles', '--real', '--repo-root', sys.argv[1], '--', '--help'])\n"
            "except SystemExit as exc:\n"
            "    rc = exc.code\n"
            "forbidden = {'torch', 'ray', 'slime', 'miles', 'megatron', 'cuda', 'triton'}\n"
            "loaded = sorted({m.split('.')[0] for m in (set(sys.modules) - before) if m.split('.')[0] in forbidden})\n"
            "assert not loaded, f'eager runtime import(s): {loaded}'\n"
            "raise SystemExit(rc)\n"
        )
        env = {**os.environ, "PYTHONPATH": str(core_livelab)}
        proc = subprocess.run(
            [sys.executable, "-c", probe, str(repo_root)],
            capture_output=True, text=True, env=env, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, f"stdout={proc.stdout} stderr={proc.stderr}")
        self.assertIn("fastrsi conformance miles --real", proc.stdout)
        self.assertIn("--model", proc.stdout)
        self.assertIn("--data", proc.stdout)

    def test_miles_patches_c10_mutates_config_lr_and_refusals_are_well_formed(self):
        patches = CL._miles_patches({
            CL.PPO_MOD: _PPO_SRC,
            CL.LOSS_MOD: _LOSS_SRC,
            CL.MODEL_MOD: _MODEL_SRC,
        })
        unreached_src = patches["unreached"]["files"][CL.MODEL_MOD]
        self.assertIn("config.lr = config.lr * 5.0", unreached_src)
        self.assertNotIn("args.lr", unreached_src)
        compile(unreached_src, CL.MODEL_MOD, "exec")
        for key in ("noop", "visible", "multi"):
            for mod, src in patches[key]["files"].items():
                compile(src, mod, "exec")
        self.assertEqual(len(patches["chain"]), 3)
        self.assertEqual(set(patches["refused"]), {"syntax", "closure", "module_state", "empty"})
        with self.assertRaises(SyntaxError):
            compile(patches["refused"]["syntax"]["files"][CL.PPO_MOD], CL.PPO_MOD, "exec")

    def test_trainer_state_snapshot_restores_weights_zeroes_adam_and_refreshes_backuper(self):
        fake_torch, fake_nn, fake_fn = _make_fake_torch()
        prev = {k: sys.modules.get(k) for k in ("torch", "torch.nn", "torch.nn.functional")}
        sys.modules.update({"torch": fake_torch, "torch.nn": fake_nn, "torch.nn.functional": fake_fn})
        self.addCleanup(
            lambda: [
                sys.modules.__setitem__(k, v) if v is not None else sys.modules.pop(k, None)
                for k, v in prev.items()
            ]
        )

        w = _FakeTensor([1.0, 2.0])
        b = _FakeTensor([0.5])
        m = _FakeTensor([1.0, 2.0])
        chunk = types.SimpleNamespace(
            named_parameters=lambda: [("weight", w)],
            named_buffers=lambda: [("running_mean", b)],
        )
        inner_opt = types.SimpleNamespace(
            state={},
            param_groups=[{"params": [w], "lr": 1e-4}],
        )
        scaler_state = {"scale": 1024.0}
        grad_scaler = types.SimpleNamespace(
            state_dict=lambda: dict(scaler_state),
            load_state_dict=lambda sd: (scaler_state.clear(), scaler_state.update(sd)),
        )
        opt = types.SimpleNamespace(
            shard_fp32_from_float16_groups=[[m]],
            optimizer=inner_opt,
            grad_scaler=grad_scaler,
        )
        scheduler = types.SimpleNamespace(num_steps=0, step=lambda increment=0: None)
        backed = {"actor": list(w._data)}
        wb = types.SimpleNamespace(
            backup_tags=("actor",),
            backup=lambda tag: backed.__setitem__(tag, list(w._data)),
            restore=lambda tag: w._data.__setitem__(slice(None), backed[tag]),
        )
        actor = types.SimpleNamespace(
            model=[chunk],
            optimizer=opt,
            opt_param_scheduler=scheduler,
            weights_backuper=wb,
        )

        snap_info = S.snapshot_trainer(actor)
        self.assertIn("weights", snap_info["restored"])
        self.assertIn("master_weights", snap_info["restored"])
        self.assertIn("buffers", snap_info["restored"])
        self.assertIn("scheduler", snap_info["restored"])

        # Simulate a training step mutating everything + lazy Adam moments + stale weights_backuper.
        w._data[:] = [9.0, 10.0]
        b._data[:] = [7.5]
        m._data[:] = [9.0, 10.0]
        inner_opt.param_groups[0]["lr"] = 9e-4
        inner_opt.state[w] = {
            "exp_avg": _FakeTensor([3.0, 4.0]),
            "exp_avg_sq": _FakeTensor([5.0, 6.0]),
            "step": 4,
        }
        scheduler.num_steps = 5
        scaler_state["scale"] = 512.0
        wb.backup("actor")
        self.assertEqual(backed["actor"], [9.0, 10.0])

        res = S.restore_trainer(actor)
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["weights_backuper_refreshed"])
        self.assertEqual(res["zeroed_count"], 2)
        self.assertEqual(w._data, [1.0, 2.0])
        self.assertEqual(b._data, [0.5])
        self.assertEqual(m._data, [1.0, 2.0])
        self.assertEqual(inner_opt.param_groups[0]["lr"], 1e-4)
        self.assertEqual(scheduler.num_steps, 0)
        self.assertEqual(scaler_state["scale"], 1024.0)
        self.assertEqual(inner_opt.state[w]["exp_avg"]._data, [0.0, 0.0])
        self.assertEqual(inner_opt.state[w]["exp_avg_sq"]._data, [0.0, 0.0])
        self.assertEqual(inner_opt.state[w]["step"], 0)
        wb.restore("actor")
        self.assertEqual(w._data, [1.0, 2.0])

        # Unknown new optimizer state must fail closed.
        inner_opt.state[w]["unknown_buf"] = _FakeTensor([1.0])
        bad = S.restore_trainer(actor)
        self.assertFalse(bad["ok"])
        self.assertIn("unknown_state", bad)

    def test_conformance_lab_fake_runtime_passes_all_11_conformance_cases(self):
        tmp_pkg = tempfile.mkdtemp(prefix="fastrsi-miles-fakeslime-")
        self.addCleanup(shutil.rmtree, tmp_pkg, True)
        for rel, content in (
            ("slime/__init__.py", ""),
            ("slime/utils/__init__.py", ""),
            ("slime/utils/ppo_utils.py", _PPO_SRC),
            ("slime/backends/__init__.py", ""),
            ("slime/backends/megatron_utils/__init__.py", ""),
            ("slime/backends/megatron_utils/loss.py", _LOSS_SRC),
            ("slime/backends/megatron_utils/model.py", _MODEL_SRC),
        ):
            target = Path(tmp_pkg, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        fake_torch, fake_nn, fake_fn = _make_fake_torch()
        prev_mods = {
            k: sys.modules.get(k)
            for k in (
                "torch", "torch.nn", "torch.nn.functional", "slime", "slime.utils", "slime.backends",
                "slime.backends.megatron_utils", *CL.CONFORMANCE_MODULES,
            )
        }
        sys.path.insert(0, tmp_pkg)
        sys.modules.update({"torch": fake_torch, "torch.nn": fake_nn, "torch.nn.functional": fake_fn})

        def _cleanup():
            if tmp_pkg in sys.path:
                sys.path.remove(tmp_pkg)
            for name, old in prev_mods.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
            RP._LOCAL.clear()

        self.addCleanup(_cleanup)

        def _reload_base_modules():
            for name in ("slime", "slime.utils", "slime.backends", "slime.backends.megatron_utils", *CL.CONFORMANCE_MODULES):
                sys.modules.pop(name, None)
            importlib.invalidate_caches()
            for mod in CL.CONFORMANCE_MODULES:
                importlib.import_module(mod)

        state = {"trainer": None, "rollout": None, "weight": 1.0, "rollout_w": 1.0, "fault": None}

        def _make_runtime():
            RP._LOCAL.clear()
            _reload_base_modules()
            state["weight"] = 1.0
            state["rollout_w"] = 1.0
            state["fault"] = None
            trainer = _InProcessActor("trainer0")
            rollout = _InProcessActor("rollout_mgr", hidden_modules=CL.CONFORMANCE_MODULES)
            state["trainer"] = trainer
            state["rollout"] = rollout
            base_text = {mod: H.read_module(mod) for mod in CL.CONFORMANCE_MODULES}
            coupled = S.CoupledWeightReset(
                get_trainer_states=lambda: {"w": state["weight"]},
                set_trainer_states=lambda saved: state.__setitem__("weight", saved["w"]),
                sync_rollout_weights=lambda: state.__setitem__("rollout_w", state["weight"]),
            )
            coupled.capture_s0()

            def _run_step(workload=None):
                if not trainer._alive:
                    raise RuntimeError("trainer0 lost")
                if state["fault"] == "crash":
                    state["fault"] = None
                    raise RuntimeError("trainer0 injected crash")
                if state["fault"] == "hang":
                    state["fault"] = None
                    raise TimeoutError("trainer0 injected hang")
                loss_mod = sys.modules[CL.LOSS_MOD]
                model_mod = sys.modules[CL.MODEL_MOD]
                cfg = state.get("_cfg")
                if cfg is None:
                    cfg = model_mod.setup_model_and_optimizer({"lr": 1e-4})
                    state["_cfg"] = cfg
                val = loss_mod.compute_advantages_and_returns([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
                state["weight"] += val * cfg.lr
                state["rollout_w"] = state["weight"]
                digest = hashlib.sha256(f"{val:.6f}:{state['weight']:.8f}".encode()).hexdigest()[:16]
                return {"digest": digest, "abs_sum": round(state["weight"], 8)}

            def _inject(fault, **kwargs):
                if fault == "lost":
                    trainer._alive = False
                    return {"ok": True, "fault": "lost"}
                if fault in ("crash", "hang"):
                    state["fault"] = fault
                    return {"ok": True, "fault": fault}
                lp = RP._LOCAL["trainer0"]
                if fault == "fail_apply":
                    return H.inject_participant_fault(lp, "fail_apply", after=kwargs.get("after", 1))
                if fault == "fail_commit":
                    return H.inject_participant_fault(lp, "fail_commit")
                if fault == "tamper":
                    return H.inject_participant_fault(lp, "tamper", module=CL.PPO_MOD, qualname="get_grpo_returns")
                return {"ok": False, "error": f"unknown {fault}"}

            return {
                "rollout_actors": {"rollout_mgr": _FakeRemoteHandle(rollout)},
                "train_actors": {"trainer0": _FakeRemoteHandle(trainer)},
                "get": lambda ref, timeout=None: ref,
                "modules": list(CL.CONFORMANCE_MODULES),
                "base_text": base_text,
                "coupled_reset": coupled,
                "run_step": _run_step,
                "inject": _inject,
                "shutdown": lambda: state.pop("_cfg", None),
            }

        def _fresh_runner(files, workload=None):
            ppo_mod = types.ModuleType(CL.PPO_MOD)
            exec(files.get(CL.PPO_MOD, _PPO_SRC), ppo_mod.__dict__)
            saved_ppo = sys.modules.get(CL.PPO_MOD)
            sys.modules[CL.PPO_MOD] = ppo_mod
            try:
                loss_mod = types.ModuleType(CL.LOSS_MOD)
                exec(files.get(CL.LOSS_MOD, _LOSS_SRC), loss_mod.__dict__)
            finally:
                if saved_ppo is None:
                    sys.modules.pop(CL.PPO_MOD, None)
                else:
                    sys.modules[CL.PPO_MOD] = saved_ppo
            model_mod = types.ModuleType(CL.MODEL_MOD)
            exec(files.get(CL.MODEL_MOD, _MODEL_SRC), model_mod.__dict__)
            cfg = model_mod.setup_model_and_optimizer({"lr": 1e-4})
            val = loss_mod.compute_advantages_and_returns([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
            weight = 1.0 + val * cfg.lr
            digest = hashlib.sha256(f"{val:.6f}:{weight:.8f}".encode()).hexdigest()[:16]
            return {"ok": True, "output": {"digest": digest, "abs_sum": round(weight, 8)}, "cost_s": 0.001}

        lab = CL.MilesConformanceLab(runtime_factory=_make_runtime, fresh_runner=_fresh_runner)
        auto_work = lab.work_dir
        self.assertTrue(os.path.isdir(auto_work))
        try:
            for cname, case_fn in CC.CASES.items():
                with self.subTest(case=cname):
                    lab.reset()
                    ev = case_fn(lab)
                    self.assertTrue(ev, cname)
        finally:
            lab.close()
        self.assertFalse(os.path.exists(auto_work))

    def _install_fake_torch(self):
        fake_torch, fake_nn, fake_fn = _make_fake_torch()
        prev = {k: sys.modules.get(k) for k in ("torch", "torch.nn", "torch.nn.functional")}
        sys.modules.update({"torch": fake_torch, "torch.nn": fake_nn, "torch.nn.functional": fake_fn})
        self.addCleanup(
            lambda: [
                sys.modules.__setitem__(k, v) if v is not None else sys.modules.pop(k, None)
                for k, v in prev.items()
            ]
        )
        return fake_torch

    def _make_snapshot_actor(self, *, wrapped=True):
        w = _FakeTensor([1.0, 2.0])
        b = _FakeTensor([0.5])
        m = _FakeTensor([1.0, 2.0])
        params = [("weight", w)]
        buffers = [("running_mean", b)]
        chunk = types.SimpleNamespace(
            named_parameters=lambda: list(params),
            named_buffers=lambda: list(buffers),
        )
        inner_opt = types.SimpleNamespace(
            state={},
            param_groups=[{"params": [w], "lr": 1e-4, "betas": [0.9, 0.999], "meta": {"tag": "s0"}}],
        )
        scaler_state = {"scale": 1024.0}
        grad_scaler = types.SimpleNamespace(
            state_dict=lambda: dict(scaler_state),
            load_state_dict=lambda sd: (scaler_state.clear(), scaler_state.update(sd)),
        )
        if wrapped:
            opt = types.SimpleNamespace(
                shard_fp32_from_float16_groups=[[m]],
                optimizer=inner_opt,
                grad_scaler=grad_scaler,
            )
        else:
            inner_opt.grad_scaler = grad_scaler
            opt = inner_opt
        scheduler = types.SimpleNamespace(num_steps=0, step=lambda increment=0: None)
        backed = {"actor": list(w._data)}
        wb = types.SimpleNamespace(
            backup_tags=("actor",),
            backup=lambda tag: backed.__setitem__(tag, list(w._data)),
            restore=lambda tag: w._data.__setitem__(slice(None), backed[tag]),
        )
        actor = types.SimpleNamespace(
            model=[chunk],
            optimizer=opt,
            opt_param_scheduler=scheduler,
            weights_backuper=wb,
        )
        return actor, {
            "w": w, "b": b, "m": m, "params": params, "buffers": buffers,
            "inner_opt": inner_opt, "opt": opt, "scaler_state": scaler_state,
            "scheduler": scheduler, "backed": backed, "wb": wb,
        }

    def test_m4_restore_trainer_fail_closed_on_non_optimizer_and_unknown_state(self):
        fake_torch = self._install_fake_torch()

        for case in ("extra_param", "extra_buffer", "extra_master", "extra_param_with_state_in_name", "unknown_opt_key"):
            with self.subTest(case=case):
                actor, ctx = self._make_snapshot_actor(wrapped=True)
                S.snapshot_trainer(actor)

                # Mutate every piece of live state after S0.
                ctx["w"]._data[:] = [9.0, 10.0]
                ctx["b"]._data[:] = [7.5]
                ctx["m"]._data[:] = [9.0, 10.0]
                ctx["inner_opt"].param_groups[0]["lr"] = 9e-4
                ctx["inner_opt"].param_groups[0]["betas"][0] = 0.5
                ctx["scheduler"].num_steps = 5
                ctx["scaler_state"]["scale"] = 256.0
                fake_torch.set_rng_state([9, 9, 9])
                random.seed(12345)
                mutated_py_rng = random.getstate()
                ctx["wb"].backup("actor")
                exp_avg = _FakeTensor([3.0, 4.0])
                ctx["inner_opt"].state[ctx["w"]] = {"exp_avg": exp_avg, "step": 7}

                if case == "extra_param":
                    ctx["params"].append(("extra_weight", _FakeTensor([42.0])))
                elif case == "extra_buffer":
                    ctx["buffers"].append(("lazy_buf", _FakeTensor([42.0])))
                elif case == "extra_master":
                    ctx["opt"].shard_fp32_from_float16_groups[0].append(_FakeTensor([42.0]))
                elif case == "extra_param_with_state_in_name":
                    ctx["params"].append(("layer.state.step", _FakeTensor([42.0])))
                else:
                    ctx["inner_opt"].state[ctx["w"]]["unknown_buf"] = _FakeTensor([1.0])

                res = S.restore_trainer(actor)
                self.assertFalse(res["ok"], res)
                self.assertIn("nothing written", res["error"])
                if case == "unknown_opt_key":
                    self.assertIn("unknown_state", res)
                    self.assertIn("opt0.state.0.unknown_buf", res["unknown_state"])
                # Verify NOTHING was mutated.
                self.assertEqual(ctx["w"]._data, [9.0, 10.0])
                self.assertEqual(ctx["b"]._data, [7.5])
                self.assertEqual(ctx["m"]._data, [9.0, 10.0])
                self.assertEqual(ctx["inner_opt"].param_groups[0]["lr"], 9e-4)
                self.assertEqual(ctx["inner_opt"].param_groups[0]["betas"][0], 0.5)
                self.assertEqual(ctx["scheduler"].num_steps, 5)
                self.assertEqual(ctx["scaler_state"]["scale"], 256.0)
                self.assertEqual(fake_torch.get_rng_state(), [9, 9, 9])
                self.assertEqual(random.getstate(), mutated_py_rng)
                self.assertEqual(ctx["backed"]["actor"], [9.0, 10.0])
                self.assertEqual(exp_avg._data, [3.0, 4.0])
                self.assertEqual(ctx["inner_opt"].state[ctx["w"]]["step"], 7)

                # Direct zero_new_state() with an unknown key must also leave exp_avg/step untouched.
                if case == "unknown_opt_key":
                    zeroed, unknown = actor._fastrsi_s0.zero_new_state()
                    self.assertEqual(zeroed, [])
                    self.assertEqual(unknown, ["opt0.state.0.unknown_buf"])
                    self.assertEqual(exp_avg._data, [3.0, 4.0])
                    self.assertEqual(ctx["inner_opt"].state[ctx["w"]]["step"], 7)

    def test_m5_unwrapped_optimizer_and_deepcopy_param_groups(self):
        self._install_fake_torch()
        actor, ctx = self._make_snapshot_actor(wrapped=False)
        S.snapshot_trainer(actor)

        # In-place mutation of mutable param_groups values must not corrupt the snapshot.
        ctx["w"]._data[:] = [8.0, 9.0]
        ctx["inner_opt"].param_groups[0]["lr"] = 5e-3
        ctx["inner_opt"].param_groups[0]["betas"].append(0.123)
        ctx["inner_opt"].param_groups[0]["meta"]["tag"] = "mutated"
        ctx["inner_opt"].state[ctx["w"]] = {
            "exp_avg": _FakeTensor([2.0, 3.0]),
            "step": 3,
        }

        res = S.restore_trainer(actor)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["zeroed_count"], 1)
        self.assertEqual(ctx["w"]._data, [1.0, 2.0])
        self.assertEqual(ctx["inner_opt"].param_groups[0]["lr"], 1e-4)
        self.assertEqual(ctx["inner_opt"].param_groups[0]["betas"], [0.9, 0.999])
        self.assertEqual(ctx["inner_opt"].param_groups[0]["meta"], {"tag": "s0"})
        self.assertEqual(ctx["inner_opt"].state[ctx["w"]]["exp_avg"]._data, [0.0, 0.0])
        self.assertEqual(ctx["inner_opt"].state[ctx["w"]]["step"], 0)

    def _install_fake_ray_and_slime(self, *, fail_in=None, handler_count=1, fail_save=False):
        fake_torch = self._install_fake_torch()
        saved_store = {}

        def _fake_save(obj, path):
            if fail_save:
                Path(path).write_bytes(b"partial")
                raise RuntimeError("disk full")
            Path(path).write_text(json.dumps(obj), encoding="utf-8")
            saved_store["last"] = obj

        def _fake_load(path, weights_only=False):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        fake_torch.save = _fake_save
        fake_torch.load = _fake_load

        ray_state = {"initialized": False, "init_calls": 0, "shutdown_calls": 0, "get_timeouts": [], "killed": []}
        fake_ray = types.ModuleType("ray")

        def _is_initialized():
            return ray_state["initialized"]

        def _init(**kwargs):
            ray_state["initialized"] = True
            ray_state["init_calls"] += 1

        def _shutdown():
            ray_state["shutdown_calls"] += 1
            ray_state["initialized"] = False

        def _get(refs, timeout=None):
            ray_state["get_timeouts"].append(timeout)
            return refs

        def _put(obj):
            return obj

        def _kill(actor_handle, no_restart=True):
            ray_state["killed"].append((actor_handle, no_restart))

        fake_ray.is_initialized = _is_initialized
        fake_ray.init = _init
        fake_ray.shutdown = _shutdown
        fake_ray.get = _get
        fake_ray.put = _put
        fake_ray.kill = _kill

        class _Box:
            def __init__(self, inner):
                self.inner = inner

        slime_pkg = types.ModuleType("slime")
        slime_ray = types.ModuleType("slime.ray")
        slime_pg = types.ModuleType("slime.ray.placement_group")
        slime_utils = types.ModuleType("slime.utils")
        slime_args = types.ModuleType("slime.utils.arguments")
        slime_misc = types.ModuleType("slime.utils.misc")
        slime_misc.Box = _Box

        def _parse_args():
            if fail_in == "parse_args":
                raise SystemExit(2)
            return types.SimpleNamespace(argv=list(sys.argv))

        slime_args.parse_args = _parse_args

        rollout_gen_calls = [0]

        def _create_pgs(args):
            return {"rollout": "pg-rollout"}

        def _create_rm(args, pg):
            def _gen(idx):
                rollout_gen_calls[0] += 1
                return [_Box([rollout_gen_calls[0], idx])]

            rm = types.SimpleNamespace(
                generate=types.SimpleNamespace(remote=_gen),
                __ray_call__=types.SimpleNamespace(remote=lambda fn, *a, **kw: fn(_InProcessActor("rollout_mgr"), *a, **kw)),
            )
            return rm, None

        train_actors = []
        train_handles = []
        for idx in range(handler_count):
            actor, ctx = self._make_snapshot_actor(wrapped=True)
            actor.name = f"trainer{idx}"
            actor._alive = True
            actor.hidden_modules = ()
            ctx["w"]._data[:] = [float(idx + 1), float(idx + 2)]
            ctx["backed"]["actor"] = list(ctx["w"]._data)
            actor._ctx = ctx
            train_actors.append(actor)
            train_handles.append(_FakeRemoteHandle(actor))

        def _create_models(args, pgs, rm):
            if fail_in == "create_training_models":
                raise RuntimeError("OOM creating training models")

            def _async_train(step_idx, rollout_ref):
                for idx, actor in enumerate(train_actors):
                    actor._ctx["w"]._data[0] += 10.0 * (idx + 1)
                return ["trained"]

            am = types.SimpleNamespace(
                _actor_handlers=list(train_handles),
                update_weights=lambda: None,
                async_train=_async_train,
            )
            return am, None

        slime_pg.create_placement_groups = _create_pgs
        slime_pg.create_rollout_manager = _create_rm
        slime_pg.create_training_models = _create_models

        injected = {
            "ray": fake_ray,
            "slime": slime_pkg,
            "slime.ray": slime_ray,
            "slime.ray.placement_group": slime_pg,
            "slime.utils": slime_utils,
            "slime.utils.arguments": slime_args,
            "slime.utils.misc": slime_misc,
        }
        prev = {k: sys.modules.get(k) for k in injected}
        sys.modules.update(injected)
        self.addCleanup(
            lambda: [
                sys.modules.__setitem__(k, v) if v is not None else sys.modules.pop(k, None)
                for k, v in prev.items()
            ]
        )
        return fake_ray, ray_state, train_actors, rollout_gen_calls

    def test_m1_launch_and_make_runtime_shutdown_ray_on_partial_boot_failure(self):
        with mock.patch.object(CL.time, "sleep", lambda _s: None):
            # 1) parse_args() raises SystemExit inside _launch_slime_cluster -> ray.shutdown() runs.
            _, ray_state, _, _ = self._install_fake_ray_and_slime(fail_in="parse_args")
            with self.assertRaises(SystemExit):
                CL._launch_slime_cluster("m", "d", pythonpath="", num_gpus=2)
            self.assertEqual(ray_state["init_calls"], 1)
            self.assertEqual(ray_state["shutdown_calls"], 1)
            self.assertFalse(ray_state["initialized"])

            # 2) create_training_models() raises OOM -> ray.shutdown() runs.
            _, ray_state, _, _ = self._install_fake_ray_and_slime(fail_in="create_training_models")
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                CL._launch_slime_cluster("m", "d", pythonpath="", num_gpus=2)
            self.assertEqual(ray_state["shutdown_calls"], 1)
            self.assertFalse(ray_state["initialized"])

            # 3) snapshot_trainer fails inside make_runtime() -> _shutdown_cluster() runs and owned work_dir cleaned.
            _, ray_state, _, _ = self._install_fake_ray_and_slime(handler_count=1)
            with mock.patch.object(CL, "snapshot_trainer", side_effect=RuntimeError("snap boom")):
                with self.assertRaisesRegex(RuntimeError, "snap boom"):
                    CL.MilesConformanceLab(model="m", data="d")
            self.assertEqual(ray_state["shutdown_calls"], 1)
            self.assertFalse(ray_state["initialized"])

    def test_m3_empty_handlers_fail_closed_and_multi_trainer_snapshot_restore_digest(self):
        with mock.patch.object(CL.time, "sleep", lambda _s: None):
            # Empty _actor_handlers fails closed and shuts down Ray.
            _, ray_state, _, _ = self._install_fake_ray_and_slime(handler_count=0)
            with self.assertRaisesRegex(RuntimeError, "_actor_handlers is empty"):
                CL._launch_slime_cluster("m", "d", pythonpath="", num_gpus=2)
            self.assertEqual(ray_state["shutdown_calls"], 1)

            # 2 trainer handlers: register, snapshot, restore, and digest across BOTH trainers.
            fake_ray, ray_state, train_actors, _ = self._install_fake_ray_and_slime(handler_count=2)
            fake_src = {CL.PPO_MOD: _PPO_SRC, CL.LOSS_MOD: _LOSS_SRC, CL.MODEL_MOD: _MODEL_SRC}
            with mock.patch.object(CL.H, "read_module", side_effect=lambda mod: fake_src[mod]):
                lab = CL.MilesConformanceLab(model="m", data="d", rpc_timeout=17.5, check_timeout=12.0)
            try:
                self.assertEqual(sorted(lab.runtime["train_actors"]), ["trainer0", "trainer1"])
                self.assertEqual(sorted(lab._cluster["trainers"]), ["trainer0", "trainer1"])
                self.assertTrue(hasattr(train_actors[0], "_fastrsi_s0"))
                self.assertTrue(hasattr(train_actors[1], "_fastrsi_s0"))
                self.assertIn(17.5, ray_state["get_timeouts"])

                # Run one step -> both trainers mutated and reflected in combined digest.
                d1 = lab.runtime["run_step"]()
                self.assertIn("trainers", d1)
                self.assertEqual(sorted(d1["trainers"]), ["trainer0", "trainer1"])
                self.assertEqual(train_actors[0]._ctx["w"]._data, [11.0, 2.0])
                self.assertEqual(train_actors[1]._ctx["w"]._data, [22.0, 3.0])

                # Coupled S0 reset restores BOTH trainers to initial weights.
                lab.runtime["coupled_reset"].restore_and_sync()
                self.assertEqual(train_actors[0]._ctx["w"]._data, [1.0, 2.0])
                self.assertEqual(train_actors[1]._ctx["w"]._data, [2.0, 3.0])

                # Mutating only trainer1 changes combined digest & abs_sum.
                d_base = CL._collect_weight_digest(fake_ray, lab._cluster["trainers"])
                train_actors[1]._ctx["w"]._data[0] = 99.0
                d_mut = CL._collect_weight_digest(fake_ray, lab._cluster["trainers"])
                self.assertNotEqual(d_base["digest"], d_mut["digest"])
                self.assertNotEqual(d_base["abs_sum"], d_mut["abs_sum"])

                # If trainer1 fails restore, restore_s0 raises RuntimeError.
                train_actors[1]._ctx["inner_opt"].state[train_actors[1]._ctx["w"]] = {
                    "unknown_buf": _FakeTensor([1.0]),
                }
                with self.assertRaisesRegex(RuntimeError, "trainer1"):
                    lab.runtime["coupled_reset"].restore_and_sync()
            finally:
                lab.close()

            # Single trainer preserves "trainer0" key and flat digest schema.
            fake_ray1, _, _, _ = self._install_fake_ray_and_slime(handler_count=1)
            with mock.patch.object(CL.H, "read_module", side_effect=lambda mod: fake_src[mod]):
                lab1 = CL.MilesConformanceLab(model="m", data="d")
            try:
                self.assertEqual(list(lab1.runtime["train_actors"]), ["trainer0"])
                d_single = CL._collect_weight_digest(fake_ray1, lab1._cluster["trainers"])
                self.assertEqual(sorted(d_single), ["abs_sum", "digest"])
            finally:
                lab1.close()

    def test_m2_fresh_check_bad_diff_pre_stop_failure_and_timeout_kills_process_group(self):
        events = {"stop": 0, "spawn": 0, "install": []}

        def _make_runtime():
            events["spawn"] += 1
            actor = _InProcessActor("trainer0")
            return {
                "rollout_actors": {},
                "train_actors": {"trainer0": _FakeRemoteHandle(actor)},
                "get": lambda ref, timeout=None: ref,
                "modules": [],
                "base_text": {CL.PPO_MOD: _PPO_SRC},
                "run_step": lambda _w=None: {"ok": True},
                "shutdown": lambda: events.__setitem__("stop", events["stop"] + 1),
            }

        lab = CL.MilesConformanceLab(runtime_factory=_make_runtime, boot_timeout=0.6)
        self.addCleanup(lab.close)
        self.assertEqual(events["spawn"], 1)
        self.assertEqual(events["stop"], 0)

        # 1) ValueError in self.files (bad unified diff) returns ok=False without touching engine.
        bad_diff = lab.fresh_check({"diff": "--- a/slime/utils/ppo_utils.py\n+++ b/slime/utils/ppo_utils.py\n@@ -1,1 +1,1 @@\n-NONEXISTENT\n+X\n"})
        self.assertFalse(bad_diff["ok"])
        self.assertIsNone(bad_diff["output"])
        self.assertEqual(events["stop"], 0)
        self.assertEqual(events["spawn"], 1)

        # 2) Pre-stop failure in _build_fresh_overlay must NOT call _stop_engine or _spawn.
        with mock.patch.object(CL, "_build_fresh_overlay", side_effect=RuntimeError("overlay boom")):
            bad_ov = lab.fresh_check({"files": {CL.PPO_MOD: _PPO_SRC + "\n# edit\n"}})
        self.assertFalse(bad_ov["ok"])
        self.assertIn("overlay boom", bad_ov["error"])
        self.assertIsNone(bad_ov["output"])
        self.assertEqual(events["stop"], 0)
        self.assertEqual(events["spawn"], 1)

        # 3) Timeout in fresh subprocess kills entire process group (including grandchild) and respawns once.
        tmp_pid_dir = tempfile.mkdtemp(prefix="fastrsi-miles-pgtest-")
        self.addCleanup(shutil.rmtree, tmp_pid_dir, True)
        grandchild_pid_file = os.path.join(tmp_pid_dir, "grandchild.pid")
        spawner_code = (
            "import subprocess, sys, time\n"
            "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "with open(sys.argv[1], 'w') as fh:\n"
            "    fh.write(str(gc.pid))\n"
            "    fh.flush()\n"
            "time.sleep(60)\n"
        )
        real_popen = subprocess.Popen
        seen_start_new_session = []

        def _fake_popen(cmd, *args, **kwargs):
            if isinstance(cmd, (list, tuple)) and "adapter.conformance_lab" in cmd:
                seen_start_new_session.append(bool(kwargs.get("start_new_session")))
                return real_popen(
                    [sys.executable, "-c", spawner_code, grandchild_pid_file],
                    *args,
                    **{k: v for k, v in kwargs.items() if k != "env"},
                )
            return real_popen(cmd, *args, **kwargs)

        with mock.patch.object(CL.subprocess, "Popen", side_effect=_fake_popen):
            timed_out = lab.fresh_check({"files": {}})
        self.assertEqual(seen_start_new_session, [True])
        self.assertFalse(timed_out["ok"])
        self.assertIn("timed out", timed_out["error"])
        self.assertIsNone(timed_out["output"])
        self.assertEqual(events["stop"], 1)
        self.assertEqual(events["spawn"], 2)
        self.assertTrue(os.path.exists(grandchild_pid_file), "expected fresh child to spawn grandchild before timeout")
        gc_pid = int(Path(grandchild_pid_file).read_text(encoding="utf-8").strip())
        deadline = time.monotonic() + 3.0
        while P.pid_alive(gc_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(P.pid_alive(gc_pid), f"grandchild pid {gc_pid} leaked after fresh timeout")

    def test_m6_multi_package_overlay_inject_timeout_and_atomic_rollout_cache(self):
        # 1) _build_fresh_overlay groups files by top-level package (slime + megatron).
        tmp_src = tempfile.mkdtemp(prefix="fastrsi-miles-pkgs-")
        tmp_dst = tempfile.mkdtemp(prefix="fastrsi-miles-ov-")
        self.addCleanup(shutil.rmtree, tmp_src, True)
        self.addCleanup(shutil.rmtree, tmp_dst, True)
        for rel, content in (
            ("fake_slime/__init__.py", ""),
            ("fake_slime/a.py", "X = 1\n"),
            ("fake_slime/sibling.py", "S = 1\n"),
            ("fake_megatron/__init__.py", ""),
            ("fake_megatron/b.py", "Y = 1\n"),
            ("fake_megatron/sibling.py", "T = 1\n"),
        ):
            p = Path(tmp_src, rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        sys.path.insert(0, tmp_src)
        self.addCleanup(lambda: sys.path.remove(tmp_src) if tmp_src in sys.path else None)
        self.addCleanup(lambda: [sys.modules.pop(k, None) for k in ("fake_slime", "fake_megatron")])
        CL._build_fresh_overlay({"fake_slime.a": "X = 2\n", "fake_megatron.b": "Y = 2\n"}, tmp_dst)
        self.assertEqual(Path(tmp_dst, "fake_slime/a.py").read_text(encoding="utf-8"), "X = 2\n")
        self.assertTrue(os.path.lexists(os.path.join(tmp_dst, "fake_slime/sibling.py")))
        self.assertEqual(Path(tmp_dst, "fake_megatron/b.py").read_text(encoding="utf-8"), "Y = 2\n")
        self.assertTrue(os.path.lexists(os.path.join(tmp_dst, "fake_megatron/sibling.py")))

        # 2) inject() passes rpc_timeout and catches exceptions cleanly.
        with mock.patch.object(CL.time, "sleep", lambda _s: None):
            fake_ray, ray_state, _, _ = self._install_fake_ray_and_slime(handler_count=1)
            fake_src = {CL.PPO_MOD: _PPO_SRC, CL.LOSS_MOD: _LOSS_SRC, CL.MODEL_MOD: _MODEL_SRC}
            with mock.patch.object(CL.H, "read_module", side_effect=lambda mod: fake_src[mod]):
                lab = CL.MilesConformanceLab(model="m", data="d", rpc_timeout=23.0)
            try:
                ray_state["get_timeouts"].clear()
                res_missing = lab.inject("tamper")
                self.assertFalse(res_missing["ok"])
                self.assertEqual(ray_state["get_timeouts"], [23.0])

                fake_ray.get = mock.Mock(side_effect=TimeoutError("wedged actor"))
                res_err = lab.inject("tamper")
                self.assertFalse(res_err["ok"])
                self.assertIn("TimeoutError", res_err["error"])
                self.assertEqual(fake_ray.get.call_args.kwargs.get("timeout"), 23.0)
            finally:
                lab.close()

        # 3) Atomic rollout0.pt save & stale cache invalidation across different model/data.
        work = tempfile.mkdtemp(prefix="fastrsi-miles-rollout-")
        self.addCleanup(shutil.rmtree, work, True)
        rollout_pt = os.path.join(work, "rollout0.pt")
        with mock.patch.object(CL.time, "sleep", lambda _s: None):
            # Failed save leaves no rollout0.pt and no .tmp files.
            self._install_fake_ray_and_slime(handler_count=1, fail_save=True)
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                CL._launch_slime_cluster("modelA", "dataA", pythonpath="", rollout_pt=rollout_pt, save_rollout=True)
            self.assertFalse(os.path.exists(rollout_pt))
            self.assertEqual(os.listdir(work), [])

            # First save writes rollout0.pt + meta; second launch with same config reuses cache.
            _, _, _, gen_calls = self._install_fake_ray_and_slime(handler_count=1)
            CL._launch_slime_cluster("modelA", "dataA", pythonpath="", rollout_pt=rollout_pt, save_rollout=True)
            self.assertEqual(gen_calls[0], 1)
            self.assertTrue(os.path.exists(rollout_pt))
            self.assertTrue(os.path.exists(f"{rollout_pt}.meta.json"))
            CL._launch_slime_cluster("modelA", "dataA", pythonpath="", rollout_pt=rollout_pt, save_rollout=True)
            self.assertEqual(gen_calls[0], 1)

            # Different model/data invalidates stale rollout0.pt and regenerates.
            CL._launch_slime_cluster("modelB", "dataA", pythonpath="", rollout_pt=rollout_pt, save_rollout=True)
            self.assertEqual(gen_calls[0], 2)


if __name__ == "__main__":
    unittest.main()
