"""Golden unit tests for the `miles` (`radixark/miles` / `THUDM/slime`) live-lab adapter.

Each case tests one concrete RL live-reload pain point end-to-end (no mocks of the mechanism
under test):
1. Stale pre-edit trajectories queued in `EpochDataBuffer` during async rollout are evicted on
   `code_epoch` advance rather than consumed by post-edit training steps.
2. `CoupledWeightReset` restores trainer `S0` weights AND broadcasts them to rollout actors
   (whereas trainer-only `S0` restore leaves rollout weights drifted).
3. Async rollout mode (`mode="async"`) quiesces `RolloutManager` / `InferenceController`
   (`pause_generation` + `flush_cache` -> Phase 2 -> `continue_generation`) and advances
   `code_epoch` on both `install` and `uninstall`.
4. `@ray.remote` actor method swaps (`RolloutManager.generate_batch`,
   `MegatronTrainRayActor.train_step`) reach live deserialized actor copies via `actor_view`.
5. Custom reward `.py` files loaded without a `sys.modules` entry swap live via `file_namespace`.
6. Editing a module or `.py` file that no actor loaded fails closed (`unloaded_module`) instead
   of committing an all-abstaining silent no-op.
7. `discover_actors` extracts handles from both `miles` (`TrainerController._cells`) and
   `THUDM/slime` (`RayTrainGroup._actor_handlers`).
8. Real Ray cluster smoke test when `ray` is installed (`skipUnless`).

Run from the repo root:
    PYTHONWARNINGS=error python3 -m unittest discover -s repos/miles/livelab/adapter -t repos/miles/livelab -v
"""
import importlib.util
import os
import shutil
import tempfile
import types
import unittest

import adapter
from adapter import gate as G
from adapter import state as S
from adapter import toy_lab as TL

CUSTOM_REWARD = """def bonus():
    return 1


def compute_reward(token):
    return token * bonus()
"""


class MilesAdapterTest(unittest.TestCase):

    def test_epoch_data_buffer_evicts_pre_edit_trajectories_on_epoch_advance(self):
        """Pain point: `DefaultDataBuffer` filters by `weight_version`, not code identity, so
        trajectories buffered before a reward swap contaminate post-edit steps unless tagged with
        `code_epoch` and drained."""
        dropped = []
        buffer = S.EpochDataBuffer(capacity=8, unused_handler=lambda grp, why: dropped.append((grp, why)))
        buffer.put_group([{"reward": 1, "tag": "pre-edit-0"}])
        buffer.put_group([{"reward": 1, "tag": "pre-edit-1"}])
        self.assertTrue(buffer.admits())
        buffer.pause()
        self.assertFalse(buffer.admits())
        new_epoch = buffer.advance_epoch()
        self.assertEqual(new_epoch, 1)
        buffer.resume()
        buffer.put_group([{"reward": 10, "tag": "post-edit-0"}])
        batch = buffer.get_groups(1)
        self.assertEqual(batch, [[{"reward": 10, "tag": "post-edit-0"}]])
        self.assertEqual(buffer.stale_evicted, 2)
        self.assertEqual(len(dropped), 2)
        self.assertEqual(dropped[0][1], "stale_code_epoch")

    def test_coupled_weight_reset_syncs_rollout_weights_with_restored_trainer_s0(self):
        """Pain point: restoring trainer `S0` weights without broadcasting `S0` to rollout leaves
        rollout actors generating from drifted step-k weights."""
        trainer_weights = {"trainer0": {"weight": 4, "step_count": 0},
                           "trainer1": {"weight": 4, "step_count": 0}}
        rollout_state = {"weight": 4, "cache": {"p0": 11}}
        buffer = S.EpochDataBuffer(capacity=8)
        coupled = S.CoupledWeightReset(
            get_trainer_states=lambda: trainer_weights,
            set_trainer_states=lambda saved: trainer_weights.update(saved),
            sync_rollout_weights=lambda: rollout_state.__setitem__("weight", trainer_weights["trainer0"]["weight"]),
            flush_rollout_cache=lambda: rollout_state["cache"].clear(),
            data_buffer=buffer,
        )
        coupled.capture_s0()
        # Simulate a training step mutating trainer + rollout weights and populating buffer/cache.
        trainer_weights["trainer0"] = {"weight": 99, "step_count": 3}
        trainer_weights["trainer1"] = {"weight": 99, "step_count": 3}
        rollout_state["weight"] = 99
        rollout_state["cache"]["p1"] = 42
        buffer.put_group([{"token": 7}])
        # Fail-before-fix demonstration: uncoupled restore (`broadcast_rollout=False`) leaves rollout at 99.
        coupled.restore_and_sync(broadcast_rollout=False)
        self.assertEqual(trainer_weights["trainer0"]["weight"], 4)
        self.assertEqual(rollout_state["weight"], 99)
        # Coupled restore (`broadcast_rollout=True`) restores trainer S0 AND syncs rollout back to 4.
        rollout_state["cache"]["p2"] = 55
        buffer.put_group([{"token": 8}])
        summary = coupled.restore_and_sync(broadcast_rollout=True)
        self.assertEqual(rollout_state["weight"], 4)
        self.assertEqual(rollout_state["cache"], {})
        self.assertEqual(summary["evicted"], 1)
        self.assertEqual(summary["reset"], ["data_buffer", "trainer_s0", "rollout_weights", "kv_cache"])

    def test_async_mode_quiesces_rollout_controller_and_drains_stale_epoch(self):
        """Input: `ToyMilesLab(mode='async')` with pre-edit trajectories buffered under epoch 0.
        Expected: `install` pauses generation + flushes cache before Phase 2 and resumes after,
        bumps `code_epoch` to 1, and `uninstall` bumps `code_epoch` to 2 returning to base."""
        lab = TL.ToyMilesLab(mode="async")
        self.addCleanup(lab.close)
        base_out = lab.check()["output"]
        events = lab.runtime["rollout_controller"].events
        events.clear()
        lab.data_buffer.put_group([{"prompt": 1, "token": -999, "reward": -999}])
        self.assertEqual(lab.data_buffer.code_epoch, 0)
        res = lab.install(lab.patches()["visible"])
        self.assertTrue(res["ok"], res)
        self.assertEqual(lab.data_buffer.code_epoch, 1)
        self.assertEqual(events[:3], ["pause", "flush", "resume"])
        edited = lab.check()
        self.assertTrue(edited["proven"])
        self.assertNotEqual(edited["output"], base_out)
        self.assertEqual(lab.uninstall()["state"], "base")
        self.assertEqual(lab.data_buffer.code_epoch, 2)
        self.assertEqual(lab.check()["output"], base_out)

    def test_rollout_actor_method_swap_reaches_deserialized_actor_copy(self):
        """Input: edit `@remote class RolloutManager.generate_batch` (which runs only in `rollout0`,
        while `trainer0`/`trainer1` never imported `rollout`). Expected: `rollout0` swaps its
        deserialized `#actor` class copy and witnesses it; trainers abstain cleanly; undo restores base."""
        lab = TL.ToyMilesLab()
        self.addCleanup(lab.close)
        base_out = lab.check()["output"]
        rollout_src = lab.base_text[TL.ROLLOUT]
        edited_rollout = TL._edit(rollout_src, "score_prompt(p + 1, self.weight)",
                                  "score_prompt(p + 3, self.weight)")
        res = lab.install({"files": {TL.ROLLOUT: edited_rollout}})
        self.assertTrue(res["ok"], res)
        checked = lab.check()
        self.assertTrue(checked["proven"], checked)
        self.assertIn("rollout0", checked["witness"])
        self.assertNotIn("trainer0", checked["witness"])
        self.assertNotEqual(checked["output"], base_out)
        self.assertEqual(lab.uninstall()["state"], "base")
        self.assertEqual(lab.check()["output"], base_out)

    def test_custom_reward_file_without_sys_modules_swaps_and_undoes_live(self):
        """Input: custom reward `.py` file loaded in `RolloutManager` via `spec_from_file_location`
        without registering in `sys.modules`. Expected: live edit to `bonus()` commits via
        `file_namespace`, changes RL step output, proves witness > 0, and undoes to base."""
        temp_dir = tempfile.mkdtemp(prefix="miles-reward-")
        self.addCleanup(shutil.rmtree, temp_dir, True)
        reward_path = os.path.join(temp_dir, "reward_fn.py")
        with open(reward_path, "w", encoding="utf-8") as fh:
            fh.write(CUSTOM_REWARD)
        lab = TL.ToyMilesLab(reward_file=reward_path)
        self.addCleanup(lab.close)
        base_out = lab.check()["output"]
        edited_reward = CUSTOM_REWARD.replace("return 1", "return 2")
        res = lab.install({"files": {reward_path: edited_reward}})
        self.assertTrue(res["ok"], res)
        checked = lab.check()
        self.assertTrue(checked["proven"], checked)
        self.assertNotEqual(checked["output"], base_out)
        self.assertEqual(lab.uninstall()["state"], "base")
        self.assertEqual(lab.check()["output"], base_out)

    def test_unloaded_module_or_file_fails_closed_instead_of_silent_noop_commit(self):
        """Pain point: a plan targeting a `.py` file or module that no actor loaded would abstain
        everywhere and report `committed` while changing nothing unless caught by `unloaded`."""
        temp_dir = tempfile.mkdtemp(prefix="miles-unloaded-")
        self.addCleanup(shutil.rmtree, temp_dir, True)
        unused_path = os.path.join(temp_dir, "never_loaded.py")
        with open(unused_path, "w", encoding="utf-8") as fh:
            fh.write(CUSTOM_REWARD)
        lab = TL.ToyMilesLab()
        self.addCleanup(lab.close)
        res = lab.install({"files": {unused_path: CUSTOM_REWARD.replace("return 1", "return 3")}})
        self.assertFalse(res["ok"])
        self.assertEqual(res["outcome"], "refused")
        self.assertEqual(res["codes"], ["unloaded_module"])
        self.assertEqual(lab.status()["state"], "base")

    def test_discover_actors_supports_miles_cells_and_slime_actor_handlers(self):
        miles_ctl = types.SimpleNamespace(_cells=[
            types.SimpleNamespace(cell_id="c0", workers=["t0", "t1"]),
        ])
        slime_group = types.SimpleNamespace(_actor_handlers=["r0", "r1"])
        rollout_map, train_map = G.discover_actors(slime_group, miles_ctl)
        self.assertEqual(rollout_map, {"rollout0": "r0", "rollout1": "r1"})
        self.assertEqual(train_map, {"trainer_c0_0": "t0", "trainer_c0_1": "t1"})

    @unittest.skipUnless(importlib.util.find_spec("ray"), "ray is not installed")
    def test_real_ray_actors_swap_and_undo_through_miles_step_gate(self):
        import ray
        from kernel import session as SS
        temp_dir = tempfile.mkdtemp(prefix="miles-real-ray-")
        self.addCleanup(shutil.rmtree, temp_dir, True)
        actor_src = (
            "import ray\n"
            "@ray.remote\n"
            "class MilesWorker:\n"
            "    def run_step(self, x):\n"
            "        return x + 1\n"
        )
        mod_name = "miles_real_ray_worker"
        with open(os.path.join(temp_dir, mod_name + ".py"), "w", encoding="utf-8") as fh:
            fh.write(actor_src)
        sys_path_added = temp_dir not in os.sys.path
        if sys_path_added:
            os.sys.path.insert(0, temp_dir)
            self.addCleanup(os.sys.path.remove, temp_dir)
        self.addCleanup(os.sys.modules.pop, mod_name, None)
        worker_mod = __import__(mod_name)
        pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        py_path = os.pathsep.join([adapter.LIVELAB, pkg_root, temp_dir])
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
                 runtime_env={"env_vars": {"PYTHONPATH": py_path}})
        self.addCleanup(ray.shutdown)
        r_handle = worker_mod.MilesWorker.remote()
        t_handle = worker_mod.MilesWorker.remote()
        gate = G.MilesStepGate({"rollout0": r_handle}, {"trainer0": t_handle}, timeout_s=30.0)
        sess = SS.Session(gate.coordinator, [mod_name])
        plan, _ = gate.build_plan(
            {"files": {mod_name: actor_src.replace("x + 1", "x + 10")}},
            {mod_name: actor_src},
        )
        installed = sess.install(plan)
        self.assertEqual(installed["outcome"], "committed", installed)
        self.assertEqual(ray.get([r_handle.run_step.remote(5), t_handle.run_step.remote(5)]), [15, 15])
        self.assertTrue(gate.coordinator.prove(installed["txn"])["ok"])
        self.assertTrue(sess.uninstall()["ok"])
        self.assertEqual(ray.get([r_handle.run_step.remote(5), t_handle.run_step.remote(5)]), [6, 6])


if __name__ == "__main__":
    unittest.main()
