"""Golden unit tests and 11/11 conformance suite for `IsaacLabLiveLab` / `ToyIsaacLab`."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

import adapter  # noqa: F401 - puts core/livelab on sys.path
from adapter.lab import DEFAULT_MODULES, IsaacLabLiveLab
from adapter.toy_lab import ToyIsaacLab
from adapter.worker import require_isaac_lab_deps, validate_positive_int
from conformance import cases
from rsibox import harness


def _isaaclab_gpu_available() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "":
        return False
    try:
        info = require_isaac_lab_deps("cuda:0")
        return bool(info.get("device"))
    except Exception:
        return False


class IsaacLabConformanceTests(unittest.TestCase):
    """Runs all 11/11 `core/livelab/conformance` cases (`c0`–`c10`) on `ToyIsaacLab`."""

    def test_full_conformance_c0_to_c10_cpu(self) -> None:
        failures = []
        for name in cases.CASES:
            result = cases.run(name, ToyIsaacLab)
            if result["result"] != "pass":
                failures.append((name, result["result"], result["error"]))
        self.assertEqual(failures, [], f"conformance failures: {failures}")


class IsaacLabGoldenTests(unittest.TestCase):
    """Robotics simulation, S0 determinism, and fail-closed safety tests."""

    def setUp(self) -> None:
        self.lab = ToyIsaacLab()

    def tearDown(self) -> None:
        self.lab.close()

    def test_zero_eager_gpu_or_sim_imports(self) -> None:
        livelab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_dir = os.path.abspath(os.path.join(livelab_dir, "..", "..", "..", "core", "livelab"))
        probe = (
            "import importlib, sys\n"
            "for m in ('adapter.env_ops', 'adapter.worker', 'adapter.lab', 'adapter.toy_lab'):\n"
            "    importlib.import_module(m)\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('torch', 'isaaclab', 'isaacsim', 'mujoco', 'mujoco_warp', 'newton', 'warp', 'cuda', 'gymnasium')]\n"
            "assert not bad, f'eager imports: {bad}'\n"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([livelab_dir, core_dir]), "CUDA_VISIBLE_DEVICES": ""}
        proc = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)

    def test_live_swap_obs_reward_policy_and_bit_exact_s0_rollback(self) -> None:
        base_fp = self.lab.status()["fingerprint"]
        base_out = self.lab.check({"horizon_steps": 6})["output"]
        em, _ = DEFAULT_MODULES
        src = self.lab._base_source(em)
        edited = harness.edit_once(
            harness.edit_once(src, "reward_gain = 1.0 * REWARD_SCALE", "reward_gain = 1.75 * REWARD_SCALE"),
            "k_pole, k_vel = 1.2, 0.25",
            "k_pole, k_vel = 1.6, 0.4",
        )
        res = self.lab.install({"files": {em: edited}})
        self.assertTrue(res["ok"], res)
        chk = self.lab.check({"horizon_steps": 6})
        self.assertIs(chk["proven"], True)
        self.assertNotEqual(chk["output"]["reward_total"], base_out["reward_total"])
        w_sim0 = chk["witness"]["sim0"]
        self.assertTrue(any(k.startswith(f"{em}:compute_reward:") and v >= 6 for k, v in w_sim0.items()), w_sim0)
        self.assertTrue(any(k.startswith(f"{em}:policy_step:") and v >= 6 for k, v in w_sim0.items()), w_sim0)

        undo = self.lab.uninstall()
        self.assertTrue(undo["ok"], undo)
        self.assertEqual(self.lab.status()["fingerprint"], base_fp)
        restored = self.lab.check({"horizon_steps": 6})["output"]
        self.assertEqual(restored, base_out)

    def test_continue_state_advances_and_s0_reset_restores_trajectory(self) -> None:
        c0 = self.lab.check({"horizon_steps": 3, "continue_state": True})
        c1 = self.lab.check({"horizon_steps": 3, "continue_state": True})
        self.assertNotEqual(c0["output"]["step_rewards"], c1["output"]["step_rewards"])
        self.assertEqual(c1["output"]["step"], 6)
        rst = self.lab.reset("state")
        self.assertTrue(rst["ok"], rst)
        self.assertIn("sim_s0", rst["reset"])
        c2 = self.lab.check({"horizon_steps": 3, "continue_state": True})
        self.assertEqual(c2["output"], c0["output"])

    def test_refuses_constructor_edit_and_invalid_configs_fail_closed(self) -> None:
        em = DEFAULT_MODULES[0]
        src = self.lab._base_source(em)
        ctor_patch = harness.edit_once(src, "self.action_scale = init_action_scale()", "self.action_scale = 2.0")
        res = self.lab.install({"files": {em: ctor_patch}})
        self.assertFalse(res["ok"])
        self.assertIn("constructor_edit", res["codes"])

        for bad in (0, -4, True, 1.5, "64"):
            with self.assertRaises(ValueError):
                validate_positive_int(bad, "num_envs")
            with self.assertRaises(ValueError):
                IsaacLabLiveLab({"mode": "cpu", "num_envs": bad}, auto_spawn=False)

        mismatch = self.lab.check({"num_envs": 128})
        self.assertFalse(mismatch["ok"])
        self.assertIn("num_envs mismatch", mismatch["error"])
        self.assertTrue(self.lab.reset("respawn")["ok"])

        env_vars = IsaacLabLiveLab.launch_environment(["--num-envs", "4096", "--horizon-steps", "10", "--physics", "newton_mjwarp"])
        self.assertEqual(env_vars["RSIBOX_ISAACLAB_NUM_ENVS"], "4096")
        self.assertEqual(env_vars["RSIBOX_ISAACLAB_PHYSICS"], "newton_mjwarp")

        if not _isaaclab_gpu_available():
            with self.assertRaisesRegex(RuntimeError, "missing required package|CUDA requested"):
                IsaacLabLiveLab({"mode": "isaaclab", "num_envs": 64}, auto_spawn=True)


if __name__ == "__main__":
    unittest.main()
