"""Golden tests for `repos/verl/livelab/adapter/`.

Covers:
1. Coupled `S0` weight + optimizer + rollout weight snapshot/restore (`state.py`): bit-exact
   restore across multiple RL steps, and fail-closed refusal when parameter shapes/names drift.
2. Live `--set` performance settings (`settings.py`): micro-batch divisibility validation and
   live application onto `WorkerDict` trainer engine configs.
3. `GateMixin` (`gate.py`) + `hook.setup` (`hook.py`): Ray setup hook registration, step-gap
   live swap across rollout (`@ray.remote` copy) + reward (`sys.modules`-less file) + trainer
   actors under real Ray, and coupled `S0` reset.

Run from repo root:
    PYTHONPATH=repos/verl/livelab python3 -m unittest discover -s repos/verl/livelab/adapter -t repos/verl/livelab -v
"""
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

from adapter import gate, hook, settings
from adapter import state as S0
from adapter.toy_lab import ToyVerlLab


class CoupledStateSnapshotTest(unittest.TestCase):
    def test_coupled_s0_restores_trainer_and_rollout_weights_bit_exact(self):
        """Input: `ToyVerlLab` runs 2 RL steps that mutate trainer weights, momentum, step counter,
        and rollout KV cache + weights. Action: `reset('state')` then `check()` again.
        Expected: second `check()` produces bit-exact tokens, scores, losses, and final weights;
        without `sync_rollout_weights`, rollout keeps post-step weights and diverges."""
        lab = ToyVerlLab()
        self.addCleanup(lab.close)
        first = lab.check()
        self.assertTrue(first["ok"], first)
        for rollout_actor in lab.cluster.rollouts:
            rollout_actor.call_method("sync_weights", [9.1, 9.2, 9.3])
        reset_reply = lab.reset("state")
        self.assertTrue(reset_reply["ok"], reset_reply)
        post_reset = lab.cluster.run_steps({"prompts": [1, 2, 3, 4], "steps": 2})
        self.assertEqual(post_reset, first["output"])
        second = lab.check()
        self.assertEqual(second["output"], first["output"])

    def test_s0_restore_fails_closed_when_parameter_layout_drifts(self):
        """Input: a `StateSnapshot` taken over `{'policy': [1.0, 0.5, -0.25]}`, then a parameter
        vector changes length (`[1.0, 0.5]`) or adds an extra key. Expected: `restore()` returns
        `ok=False` naming the drifted key and leaves live weights untouched."""
        model = types.SimpleNamespace(weights={"policy": [1.0, 0.5, -0.25]})
        optimizer = {"step": 0, "momentum": [0.0, 0.0, 0.0]}
        snap = S0.StateSnapshot(model, optimizer)
        base_digest = snap.digest()["all"]

        model.weights["policy"] = [2.0, 1.5]
        optimizer["step"] = 7
        bad = snap.restore()
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["changed"], ["weight.policy"])
        self.assertEqual(model.weights["policy"], [2.0, 1.5])
        self.assertEqual(optimizer["step"], 7)

        model.weights["policy"] = [2.0, 1.5, 0.25]
        good = snap.restore()
        self.assertTrue(good["ok"], good)
        self.assertEqual(model.weights["policy"], [1.0, 0.5, -0.25])
        self.assertEqual(optimizer["step"], 0)
        self.assertEqual(snap.digest()["all"], base_digest)


class LiveSettingsTest(unittest.TestCase):
    def test_live_settings_plan_validates_divisibility_and_refuses_learning_hyperparams(self):
        """Input: per-GPU train batch = 16, infer batch = 32. Valid micro-batches (4, 8, dynamic
        bsz) pass and update `WorkerDict` engine_config; non-divisors (6) or non-live keys
        (`actor.optim.lr`) raise `ValueError`."""
        sizes = {"train": 16, "infer": 32}
        planned = settings.plan(
            {
                "actor.ppo_micro_batch_size_per_gpu": "4",
                "rollout.log_prob_micro_batch_size_per_gpu": "8",
                "actor.use_dynamic_bsz": "true",
            },
            sizes,
        )
        self.assertEqual(
            planned,
            {
                "actor.ppo_micro_batch_size_per_gpu": 4,
                "rollout.log_prob_micro_batch_size_per_gpu": 8,
                "actor.use_dynamic_bsz": True,
            },
        )
        with self.assertRaisesRegex(ValueError, "does not divide"):
            settings.plan({"actor.ppo_micro_batch_size_per_gpu": "6"}, sizes)
        with self.assertRaisesRegex(ValueError, "not a live setting"):
            settings.plan({"actor.optim.lr": "1e-5"}, sizes)

        actor_cfg = types.SimpleNamespace(
            micro_batch_size_per_gpu=2,
            infer_micro_batch_size_per_gpu=4,
            use_dynamic_bsz=False,
        )
        host = types.SimpleNamespace(
            worker_dict={
                "actor_rollout_ref": types.SimpleNamespace(
                    actor=types.SimpleNamespace(engine_config=actor_cfg),
                    ref=types.SimpleNamespace(engine_config=types.SimpleNamespace()),
                )
            }
        )
        old = settings.apply(host, planned)
        self.assertEqual(old["actor.ppo_micro_batch_size_per_gpu"], 2)
        self.assertEqual(actor_cfg.micro_batch_size_per_gpu, 4)
        self.assertEqual(actor_cfg.infer_micro_batch_size_per_gpu, 8)
        self.assertTrue(actor_cfg.use_dynamic_bsz)


class HookRegistrationTest(unittest.TestCase):
    def test_setup_hook_registers_rsibox_trainer_mode_once_on_import(self):
        """Input: synthetic `verl.trainer.ppo.v1` package with `register_trainer` and
        `PPOTrainerSync`. Action: `hook.setup()` then import `verl.trainer.ppo.v1`.
        Expected: `PPOTrainerrsiboxSync` is registered under `'rsibox'` and is picklable."""
        import pickle

        tmp_pkg = tempfile.mkdtemp(prefix="rsibox-fakeverl-")
        self.addCleanup(shutil.rmtree, tmp_pkg, True)
        v1_dir = Path(tmp_pkg) / "verl" / "trainer" / "ppo" / "v1"
        v1_dir.mkdir(parents=True)
        for init_dir in (
            Path(tmp_pkg) / "verl",
            Path(tmp_pkg) / "verl" / "trainer",
            Path(tmp_pkg) / "verl" / "trainer" / "ppo",
            v1_dir,
        ):
            (init_dir / "__init__.py").write_text("", encoding="utf-8")
        (v1_dir / "trainer_base.py").write_text(
            "REGISTRY = {}\n"
            "def register_trainer(name):\n"
            "    def wrap(cls):\n"
            "        REGISTRY[name] = cls\n"
            "        return cls\n"
            "    return wrap\n"
            "def compute_advantage_for_multi_trajectories(data, keys, **kw):\n"
            "    return data\n",
            encoding="utf-8",
        )
        (v1_dir / "trainer_sync.py").write_text(
            "class PPOTrainerSync:\n"
            "    def on_init_end(self): pass\n"
            "    def on_step_begin(self): return 'step-ok'\n"
            "    def on_train_end(self): pass\n",
            encoding="utf-8",
        )
        sys.path.insert(0, tmp_pkg)
        self.addCleanup(sys.path.remove, tmp_pkg)
        for mod_key in [k for k in list(sys.modules) if k == "verl" or k.startswith("verl.")]:
            sys.modules.pop(mod_key, None)
        gate._TRAINER = None
        self.addCleanup(
            lambda: [
                sys.modules.pop(k, None)
                for k in list(sys.modules)
                if k == "verl" or k.startswith("verl.")
            ]
        )

        hook.setup()
        import verl.trainer.ppo.v1  # noqa: F401
        from verl.trainer.ppo.v1 import trainer_base

        self.assertIn("rsibox", trainer_base.REGISTRY)
        trainer_cls = trainer_base.REGISTRY["rsibox"]
        self.assertIs(pickle.loads(pickle.dumps(trainer_cls)), trainer_cls)


class RealRayVerlClusterTest(unittest.TestCase):
    @unittest.skipUnless(
        __import__("importlib.util").util.find_spec("ray") is not None,
        "ray is not installed",
    )
    def test_real_ray_actor_cluster_step_gap_swap_and_coupled_s0_reset(self):
        """Input: a real Ray cluster running a `@ray.remote` rollout actor, a `WorkerDict`
        trainer actor, and a `sys.modules`-less custom reward worker actor.
        Action: run a step, swap BOTH the `@ray.remote` rollout `generate` method and the
        custom reward file via `RayActorParticipant` + `Session`, run a step, verify witness
        and changed output, `uninstall()`, and restore coupled `S0` weights.
        Expected: bit-exact return to base output and `"base"` session state."""
        import ray
        import ray_participants as RP
        from kernel import coordinator as C, locate as L, session as SS

        tmp = tempfile.mkdtemp(prefix="rsibox-verl-realray-")
        self.addCleanup(shutil.rmtree, tmp, True)
        mod_name = "verl_ray_rollout_test_mod"
        mod_file = Path(tmp) / f"{mod_name}.py"
        base_rollout = (
            "import ray\n\n"
            "@ray.remote\n"
            "class RolloutWorker:\n"
            "    def __init__(self):\n"
            "        self.weights = [1.0, 0.5]\n"
            "    def sync_weights(self, w):\n"
            "        self.weights = list(w)\n"
            "    def generate(self, x):\n"
            "        return int(round(x * 10 + sum(self.weights) * 10))\n"
        )
        mod_file.write_text(base_rollout, encoding="utf-8")
        reward_file = Path(tmp) / "custom_reward.py"
        base_reward = "def compute_score(tok):\n    return float(tok) * 0.1\n"
        reward_file.write_text(base_reward, encoding="utf-8")

        from adapter import LIVELAB
        verl_livelab = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, tmp)
        self.addCleanup(sys.path.remove, tmp)
        self.addCleanup(sys.modules.pop, mod_name, None)

        ray.init(
            address="local",
            num_cpus=3,
            include_dashboard=False,
            log_to_driver=False,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": os.pathsep.join([tmp, verl_livelab, LIVELAB]),
                    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                }
            },
        )
        self.addCleanup(ray.shutdown)

        rollout_mod = __import__(mod_name)

        @ray.remote
        class TrainerWorker:
            def __init__(self):
                self.weights = {"policy": [1.0, 0.5]}
                self.optimizer = {"step": 0}

            def step(self, score):
                self.optimizer["step"] += 1
                self.weights["policy"] = [round(w + score * 0.01, 6) for w in self.weights["policy"]]
                return list(self.weights["policy"])

            def get_weights(self):
                return list(self.weights["policy"])

        @ray.remote
        class RewardWorker:
            def __init__(self, rpath):
                import importlib.util as iu
                spec = iu.spec_from_file_location("custom_reward_ray", rpath)
                loaded = iu.module_from_spec(spec)
                spec.loader.exec_module(loaded)
                self._fn = loaded.compute_score

            def score(self, tok):
                return self._fn(tok)

        r_actor = rollout_mod.RolloutWorker.remote()
        t_actor = TrainerWorker.remote()
        rw_actor = RewardWorker.remote(str(reward_file))

        ray.get(t_actor.__ray_call__.remote(S0.worker_capture_s0))

        def run_one():
            ray.get(t_actor.__ray_call__.remote(S0.worker_restore_s0))
            ray.get(r_actor.sync_weights.remote(ray.get(t_actor.get_weights.remote())))
            tok = ray.get(r_actor.generate.remote(2))
            sc = ray.get(rw_actor.score.remote(tok))
            new_w = ray.get(t_actor.step.remote(sc))
            ray.get(r_actor.sync_weights.remote(new_w))
            return {"tok": tok, "score": round(sc, 6), "weights": new_w}

        base_out = run_one()
        self.assertEqual(base_out["tok"], 35)

        parts = [
            RP.RayActorParticipant("rollout0", r_actor, timeout_s=30),
            RP.RayActorParticipant("trainer0", t_actor, timeout_s=30),
            RP.RayActorParticipant("reward0", rw_actor, timeout_s=30),
        ]
        coord = C.StepGapCoordinator(parts, gap=lambda job: job(), phase2_deadline_s=30)
        sess = SS.Session(coord, [mod_name, str(reward_file)])
        self.assertEqual(sess.state(), "base", sess.poisoned())

        plan = {
            "modules": {
                mod_name: {
                    "source": base_rollout.replace("x * 10", "x * 20"),
                    "base_hash": L.text_hash(base_rollout),
                },
                RP.file_module(str(reward_file)): {
                    "file": str(reward_file),
                    "source": base_reward.replace("0.1", "0.5"),
                    "base_hash": L.text_hash(base_reward),
                },
            }
        }
        inst = sess.install(plan)
        self.assertEqual(inst["outcome"], "committed", inst)
        edited_out = run_one()
        self.assertEqual(edited_out["tok"], 55)
        self.assertAlmostEqual(edited_out["score"], 27.5, places=5)
        self.assertTrue(coord.prove(inst["txn"])["ok"])

        undo = sess.uninstall()
        self.assertTrue(undo["ok"], undo)
        self.assertEqual(sess.state(), "base", sess.poisoned())
        self.assertEqual(run_one(), base_out)

    def test_gate_in_place_reward_file_outside_sys_path(self):
        import importlib.util as iu
        from adapter.gate import VerlActorParticipant, _DRIVER, module_for_file
        from kernel import coordinator as C, locate as L, session as SS

        tmp = tempfile.mkdtemp(prefix="rsibox-verl-inplace-")
        self.addCleanup(shutil.rmtree, tmp, True)
        reward_dir = Path(tmp) / "rewards"
        reward_dir.mkdir()
        reward_path = reward_dir / "gsm8k.py"
        orig_src = "def compute_score(val):\n    return 0.0\n"
        reward_path.write_text(orig_src, encoding="utf-8")

        spec = iu.spec_from_file_location("custom_module_inplace_test", str(reward_path))
        loaded = iu.module_from_spec(spec)
        sys.modules["custom_module_inplace_test"] = loaded
        self.addCleanup(sys.modules.pop, "custom_module_inplace_test", None)
        spec.loader.exec_module(loaded)

        mkey, is_file = module_for_file(str(reward_path), [str(reward_dir)])
        self.assertTrue(is_file)

        part = VerlActorParticipant("reward_local", _DRIVER, get=lambda r, _t: r)
        coord = C.StepGapCoordinator([part], gap=lambda job: job())
        sess = SS.Session(coord, [])

        edited_src = "def compute_score(val):\n    return 0.1\n"
        reward_path.write_text(edited_src, encoding="utf-8")
        plan = {
            "modules": {
                mkey: {
                    "source": edited_src,
                    "base_hash": L.text_hash(orig_src),
                    "base_source": orig_src,
                    "file": os.path.realpath(str(reward_path)),
                }
            },
            "recipes": ["R0 swap"],
        }
        res = sess.install(plan)
        self.assertEqual(res["outcome"], "committed", res)
        self.assertEqual(loaded.compute_score(1), 0.1)

        undone = sess.uninstall()
        self.assertTrue(undone["ok"], undone)
        self.assertEqual(loaded.compute_score(1), 0.0)
        self.assertEqual(sess.state(), "base")
        self.assertTrue(coord.at_base()["ok"])


if __name__ == "__main__":
    unittest.main()
