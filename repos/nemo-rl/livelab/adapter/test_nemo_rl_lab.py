"""Golden unit tests and full 11/11 conformance suite for `NemoRlLab` / `ToyNemoRlLab`."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import unittest

import adapter  # noqa: F401 - puts core/livelab on sys.path
from adapter.lab import DEFAULT_MODULES, NemoRlLab, ToyNemoRlLab
from adapter.worker import (
    NemoRlRayParticipant,
    discover_nemo_rl_actors,
    serve_nemo_rl_actor,
)
from conformance import cases
from kernel import test_ray_participants as TRP
from rsibox import harness
from rsibox.warm.recovery import DefaultRespawnDriver
from rsibox.warm.types import StoreProbe


class NemoRlConformanceTests(unittest.TestCase):
    """Runs all 11/11 `core/livelab/conformance` cases (`c0`–`c10`) on 3-participant `ToyNemoRlLab`."""

    def test_full_conformance_c0_to_c10_cpu_three_participants(self):
        """Input: `ToyNemoRlLab` (coordinator0 + AutoModel trainer0 + SGLang rollout0).
        Expected: all 11 conformance cases c0..c10 pass with 0 failures."""
        failures = []
        for name in cases.CASES:
            result = cases.run(name, ToyNemoRlLab)
            if result["result"] != "pass":
                failures.append((name, result["result"], result["error"]))
        self.assertEqual(failures, [], f"conformance failures: {failures}")


class NemoRlGoldenTests(unittest.TestCase):
    """GRPO-specific golden tests targeting coupled trainer+rollout S0, witnesses, and fault recovery."""

    def setUp(self) -> None:
        self.lab = ToyNemoRlLab()

    def tearDown(self) -> None:
        self.lab.close()

    def test_coupled_s0_reset_restores_trainer_weights_adamw_rollout_refit_and_step_counter(self):
        """Input: run 2 GRPO steps with `continue_state=True`, then 2 more steps, then `reset('state')`.
        Expected: `step_counter`, `adam_step`, `refit_versions`, and `losses` advance on continued steps,
        and `reset('state')` restores exact S0 trainer weights/optimizer/step + SGLang rollout refit=0."""
        c0 = self.lab.check({"steps": 2, "continue_state": True})
        self.assertTrue(c0["ok"], c0)
        self.assertEqual(c0["output"]["refit_versions"], [1, 2])
        self.assertEqual(c0["output"]["trainer_step"], 2)
        self.assertEqual(c0["output"]["adam_step"], 2)

        c1 = self.lab.check({"steps": 2, "continue_state": True})
        self.assertTrue(c1["ok"], c1)
        self.assertEqual(c1["output"]["refit_versions"], [3, 4])
        self.assertEqual(c1["output"]["trainer_step"], 4)
        self.assertEqual(c1["output"]["adam_step"], 4)
        self.assertNotEqual(c1["output"]["losses"], c0["output"]["losses"])
        self.assertNotEqual(c1["output"]["weight_checksum"], c0["output"]["weight_checksum"])
        self.assertNotEqual(c1["output"]["rollout_checksum"], c0["output"]["rollout_checksum"])

        rst = self.lab.reset("state")
        self.assertTrue(rst["ok"], rst)
        for key in (
            "weights_s0",
            "optimizer_state",
            "rollout_weights",
            "rollout_refit_version",
            "kv_cache",
            "rng",
            "step_counter",
        ):
            self.assertIn(key, rst["reset"], rst)

        c2 = self.lab.check({"steps": 2, "continue_state": True})
        self.assertTrue(c2["ok"], c2)
        self.assertEqual(c2["output"], c0["output"])

    def test_multi_participant_swap_witnesses_and_bit_exact_rollback(self):
        """Input: hot-swap `adapter.grpo_ops` (trainer0), `adapter.rollout_ops` (rollout0), and
        `adapter.shared_ops` (all 3 participants) together.
        Expected: witness > 0 on coordinator0, trainer0, and rollout0, output shifts, and
        uninstall restores base fingerprint and bit-exact GRPO losses (`max_abs_loss_diff == 0.0`)."""
        base_status = self.lab.status()
        base_check = self.lab.check({"steps": 3})
        self.assertTrue(base_check["ok"], base_check)
        base_losses = base_check["output"]["losses"]
        self.assertEqual(len(base_losses), 3)

        g_mod, r_mod, s_mod = DEFAULT_MODULES
        g_src = self.lab._base_source(g_mod)
        r_src = self.lab._base_source(r_mod)
        s_src = self.lab._base_source(s_mod)

        g_edited = harness.edit_once(
            harness.edit_once(
                g_src,
                "total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE",
                "total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE * 1.5",
            ),
            "grad_scale = 1.0",
            "grad_scale = 0.5",
        )
        r_edited = harness.edit_once(r_src, "0.05 * step_idx", "0.25 * step_idx")
        s_edited = harness.edit_once(s_src, "0.1 * float(signal)", "0.35 * float(signal)")

        res = self.lab.install({"files": {g_mod: g_edited, r_mod: r_edited, s_mod: s_edited}})
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["outcome"], "committed")

        edited_check = self.lab.check({"steps": 3})
        self.assertTrue(edited_check["ok"], edited_check)
        self.assertIs(edited_check["proven"], True)
        self.assertNotEqual(edited_check["output"]["losses"], base_losses)
        for p_name in ("coordinator0", "trainer0", "rollout0"):
            self.assertIn(p_name, edited_check["witness"], edited_check["witness"])
            self.assertTrue(
                all(cnt > 0 for cnt in edited_check["witness"][p_name].values()),
                edited_check["witness"][p_name],
            )

        undo = self.lab.uninstall()
        self.assertTrue(undo["ok"], undo)
        self.assertEqual(undo["state"], "base")
        self.assertEqual(self.lab.status()["fingerprint"], base_status["fingerprint"])
        after_check = self.lab.check({"steps": 3})
        max_loss_diff = max(abs(a - b) for a, b in zip(base_losses, after_check["output"]["losses"]))
        self.assertEqual(max_loss_diff, 0.0)

    def test_sglang_rollout_only_edit_witnesses_rollout_and_abstains_others(self):
        """Input: edit only `adapter.rollout_ops:generate_sglang_group`.
        Expected: `rollout0` witnesses execution, `trainer0` and `coordinator0` abstain cleanly,
        output diverges, and uninstall restores exact base output."""
        base_out = self.lab.check()["output"]
        r_mod = DEFAULT_MODULES[1]
        r_edited = harness.edit_once(self.lab._base_source(r_mod), "0.05 * step_idx", "0.2 * step_idx")
        res = self.lab.install({"files": {r_mod: r_edited}})
        self.assertTrue(res["ok"], res)
        checked = self.lab.check()
        self.assertTrue(checked["ok"], checked)
        self.assertIs(checked["proven"], True)
        self.assertIn("rollout0", checked["witness"])
        self.assertNotIn("trainer0", checked["witness"])
        self.assertNotIn("coordinator0", checked["witness"])
        self.assertNotEqual(checked["output"], base_out)
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], base_out)

    def test_refuses_constructor_edit_and_unloaded_module_fail_closed(self):
        """Input: edit `NemoRlConfig.__init__` and an unloaded module.
        Expected: refused with `constructor_edit` / `unloaded_module` without mutating state."""
        fp0 = self.lab.status()["fingerprint"]
        out0 = self.lab.check()["output"]
        g_mod = DEFAULT_MODULES[0]
        ctor_patch = harness.edit_once(
            self.lab._base_source(g_mod),
            "self.init_scale = init_policy_scale()",
            "self.init_scale = init_policy_scale() * 2.0",
        )
        res = self.lab.install({"files": {g_mod: ctor_patch}})
        self.assertFalse(res["ok"], res)
        self.assertIn("constructor_edit", res["codes"], res)

        unloaded_res = self.lab.install({"files": {"nonexistent.nemo_rl.mod": "def f():\n    return 1\n"}})
        self.assertFalse(unloaded_res["ok"], unloaded_res)
        self.assertEqual(unloaded_res["codes"], ["unloaded_module"])

        st = self.lab.status()
        self.assertEqual(st["state"], "base")
        self.assertEqual(st["fingerprint"], fp0)
        self.assertEqual(self.lab.check()["output"], out0)

    def test_fault_and_tamper_across_participants_poison_and_respawn(self):
        """Input: inject `crash` on `rollout0` and `tamper` on `coordinator0`.
        Expected: session poisons in both cases, refuses install with `session_poisoned`,
        and `reset('respawn')` restores clean base state and bit-exact output."""
        out0 = self.lab.check()["output"]
        for participant, fault in (("rollout0", "crash"), ("coordinator0", "tamper")):
            self.assertTrue(self.lab.inject(fault, participant=participant)["ok"])
            self.lab.check()
            st = self.lab.status()
            self.assertEqual(st["state"], "poisoned", (participant, fault, st))
            refused = self.lab.install(self.lab.patches()["visible"])
            self.assertFalse(refused["ok"])
            self.assertEqual(refused["codes"], ["session_poisoned"])
            respawn = self.lab.reset("respawn")
            self.assertTrue(respawn["ok"], respawn)
            self.assertEqual(self.lab.status()["state"], "base")
            self.assertEqual(self.lab.check()["output"], out0)

    def test_pluggable_weight_store_recovery_driver_validation_and_thread_spawn(self):
        """Input: invalid trainer/rollout counts, pluggable `weight_store` / `recovery_driver`,
        and spawning on a transient thread.
        Expected: fail-closed validation, `weight_store.probe(tp=1)` + `recovery_driver` invoked,
        and workers survive spawning-thread exit."""
        for bad in (0, -1, True, 1.5, "1"):
            with self.assertRaises(ValueError):
                NemoRlLab({"mode": "cpu", "num_trainers": bad}, auto_spawn=False)
            with self.assertRaises(ValueError):
                NemoRlLab({"mode": "cpu", "num_rollouts": bad}, auto_spawn=False)

        probe_calls: list[int | None] = []

        class _RejectStore:
            name = "reject-store"

            def probe(self, *, tp=None, visible_devices=None):
                probe_calls.append(tp)
                return StoreProbe(ok=False, backend=self.name, error="store offline")

        with self.assertRaisesRegex(RuntimeError, "store offline"):
            NemoRlLab({"mode": "cpu"}, weight_store=_RejectStore(), auto_spawn=True)
        self.assertEqual(probe_calls, [1])

        respawn_calls: list[int] = []

        class _TrackStore:
            name = "track-store"

            def probe(self, *, tp=None, visible_devices=None):
                respawn_calls.append(int(tp or 0))
                return StoreProbe(ok=True, backend=self.name)

        class _TrackDriver(DefaultRespawnDriver):
            def respawn(self, lab, *args, **kwargs):
                respawn_calls.append(-lab.ranks)
                return super().respawn(lab, *args, **kwargs)

        custom_lab = ToyNemoRlLab(weight_store=_TrackStore(), recovery_driver=_TrackDriver())
        self.addCleanup(custom_lab.close)
        rst = custom_lab.reset("respawn")
        self.assertTrue(rst["ok"], rst)
        self.assertIn(-1, respawn_calls)
        self.assertGreaterEqual(respawn_calls.count(1), 2)
        self.assertFalse(custom_lab.inject("crash", participant="unknown0")["ok"])

        spawned: list[ToyNemoRlLab] = []
        t_spawn = threading.Thread(target=lambda: spawned.append(ToyNemoRlLab()))
        t_spawn.start()
        t_spawn.join(timeout=15.0)
        self.assertEqual(len(spawned), 1)
        self.addCleanup(spawned[0].close)
        time.sleep(0.15)
        self.assertTrue(spawned[0].check()["ok"])

    def test_real_mode_fails_closed_without_nemo_rl_deps_and_zero_eager_gpu_imports(self):
        """Input: `NemoRlLab` in default `--real` mode without `nemo_rl`/`ray`/`torch`, plus
        subprocess probe of `launch_environment`.
        Expected: fails closed with actionable missing-deps error and zero eager GPU/Ray imports."""
        with self.assertRaisesRegex(RuntimeError, "missing required runtime dependencies"):
            NemoRlLab({"mode": "ray"})

        livelab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        probe = (
            "import json, sys; from adapter.lab import NemoRlLab; "
            "env = NemoRlLab.launch_environment(['--', '--mode', 'ray', '--num-trainers', '2', '--steps', '4']); "
            "bad = [m for m in ('torch', 'ray', 'nemo_rl', 'sglang') if m in sys.modules]; "
            "print(json.dumps({'env': env, 'bad': bad}))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            env={**os.environ, "PYTHONPATH": livelab_dir},
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout.strip())
        self.assertEqual(data["env"]["RSIBOX_NEMORL_MODE"], "ray")
        self.assertEqual(data["env"]["RSIBOX_NEMORL_NUM_TRAINERS"], "2")
        self.assertEqual(data["env"]["RSIBOX_NEMORL_STEPS"], "4")
        self.assertEqual(data["bad"], [])

    def test_real_ray_participant_adapter_supports_actor_view_via_runtime_factory(self):
        """Input: `NemoRlLab` wired via `runtime_factory` with `NemoRlRayParticipant` over
        by-value Ray-shaped actor handles (`serve_nemo_rl_actor` + `discover_nemo_rl_actors`).
        Expected: live swap reaches deserialized actor copies, proves witness, and undoes to base."""
        import importlib
        import shutil
        import tempfile
        import types

        temp_dir = tempfile.mkdtemp(prefix="nemorl-fake-ray-")
        self.addCleanup(shutil.rmtree, temp_dir, True)
        mod_name = "test_nemorl_fake_ray_actors"
        src = (
            "from kernel import test_ray_participants as TRP\n"
            "@TRP.fake_remote\n"
            "class PolicyWorker:\n"
            "    def step(self, x):\n"
            "        return x + 1\n"
            "    def reset_state(self):\n"
            "        return ['weights_s0']\n"
            "@TRP.fake_remote\n"
            "class SGLangWorker:\n"
            "    def generate(self, x):\n"
            "        return x * 2\n"
            "    def reset_state(self):\n"
            "        return ['rollout_weights']\n"
        )
        with open(os.path.join(temp_dir, f"{mod_name}.py"), "w", encoding="utf-8") as fh:
            fh.write(src)
        sys.path.insert(0, temp_dir)
        self.addCleanup(lambda: temp_dir in sys.path and sys.path.remove(temp_dir))
        self.addCleanup(sys.modules.pop, mod_name, None)
        importlib.invalidate_caches()
        mod = importlib.import_module(mod_name)

        policy_actor = TRP.FakeActor(mod.PolicyWorker)
        rollout_actor = TRP.FakeActor(mod.SGLangWorker)
        _, trainers, rollouts = discover_nemo_rl_actors(
            policy_group=types.SimpleNamespace(workers=[policy_actor]),
            rollout_group=types.SimpleNamespace(sglang_workers=[rollout_actor]),
        )
        self.assertEqual(list(trainers), ["trainer0"])
        self.assertEqual(list(rollouts), ["rollout0"])

        def _factory():
            parts = [
                NemoRlRayParticipant("trainer0", trainers["trainer0"], get=TRP.same_ref),
                NemoRlRayParticipant("rollout0", rollouts["rollout0"], get=TRP.same_ref),
            ]
            return {
                "participants": parts,
                "modules": [mod_name],
                "base_text": {mod_name: src},
                "run_step": lambda wl=None: {
                    "val": policy_actor.instance.step(rollout_actor.instance.generate(3))
                },
                "reset_state": lambda: {"ok": True, "reset": ["weights_s0", "rollout_weights"]},
            }

        lab = NemoRlLab({"mode": "ray"}, modules=[mod_name], runtime_factory=_factory)
        self.addCleanup(lab.close)
        self.assertEqual(lab.check()["output"], {"val": 7})
        edited = src.replace("return x + 1", "return x + 10")
        res = lab.install({"files": {mod_name: edited}})
        self.assertTrue(res["ok"], res)
        checked = lab.check()
        self.assertTrue(checked["proven"], checked)
        self.assertEqual(checked["output"], {"val": 16})
        self.assertIn("trainer0", checked["witness"])
        self.assertTrue(lab.uninstall()["ok"])
        self.assertEqual(lab.check()["output"], {"val": 7})

    def test_build_plan_preserves_base_hash_unprimed_and_across_multi_module_chain(self):
        """Input: fresh `ToyNemoRlLab` before `patches()`/`check()` (empty `base_text`), then
        chained edits across `adapter.grpo_ops` and `adapter.rollout_ops`.
        Expected: `_build_plan` populates `base_hash` on both the unprimed first edit and the
        second-module chained edit."""
        from rsibox import harness as H

        g_mod, r_mod, _ = DEFAULT_MODULES
        fresh_lab = ToyNemoRlLab()
        self.addCleanup(fresh_lab.close)
        # Clear any pre-seeded base_text to simulate an unprimed session.
        fresh_lab.base_text.clear()
        fresh_lab.live_text = [{}]

        g_src = H.read_module(g_mod)
        g_edit = H.edit_once(
            g_src,
            "total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE",
            "total_loss += (pg_loss + kl_pen) * GRPO_LOSS_SCALE * 2.0",
        )
        plan1, findings1 = fresh_lab._build_plan({"files": {g_mod: g_edit}}, fresh_lab.live_text[-1])
        self.assertEqual(findings1, [])
        self.assertIsNotNone(plan1)
        self.assertIn("base_hash", plan1["modules"][g_mod])
        res1 = fresh_lab.install({"files": {g_mod: g_edit}})
        self.assertTrue(res1["ok"], res1)

        r_src = H.read_module(r_mod)
        r_edit = H.edit_once(r_src, "0.05 * step_idx", "0.25 * step_idx")
        plan2, findings2 = fresh_lab._build_plan({"files": {r_mod: r_edit}}, fresh_lab.live_text[-1])
        self.assertEqual(findings2, [])
        self.assertIsNotNone(plan2)
        self.assertIn("base_hash", plan2["modules"][r_mod])

    def test_real_mode_s0_reset_fail_closed_and_cpu_optimizer_offload(self):
        """Input: `coupled_nemo_rl_reset` with empty trainers or missing `flush_cache`,
        `serve_nemo_rl_actor` on an actor without `reset_state()`, and `snapshot_automodel_s0` /
        `restore_automodel_s0` with fake CUDA tensors.
        Expected: missing trainers, missing `flush_cache`, and missing `reset_state()` fail closed,
        and optimizer tensor state is offloaded to CPU via `.detach().cpu().clone()`."""
        import contextlib
        import types
        from unittest import mock
        from adapter.worker import (
            coupled_nemo_rl_reset,
            restore_automodel_s0,
            snapshot_automodel_s0,
        )

        # 1. Empty trainers must fail closed (never vacuous ok=True).
        fake_ray = types.SimpleNamespace(get=lambda refs, timeout=None: list(refs))
        with self.assertRaisesRegex(RuntimeError, "no trainer actors"):
            coupled_nemo_rl_reset(
                policy_group=types.SimpleNamespace(workers=[]),
                rollout_group=None,
                ray_mod=fake_ray,
            )

        # 2. Missing flush_cache on rollout actor must fail closed.
        class _DummyTrainer:
            def __init__(self):
                self.__ray_call__ = types.SimpleNamespace(
                    remote=lambda fn: {"ok": True, "reset": ["weights_s0"]}
                )

        with self.assertRaisesRegex(RuntimeError, "flush_cache"):
            coupled_nemo_rl_reset(
                policy_group=types.SimpleNamespace(
                    workers=[_DummyTrainer()],
                    refit_policy_generation=lambda rg: None,
                ),
                rollout_group=types.SimpleNamespace(sglang_workers=[types.SimpleNamespace()]),
                ray_mod=fake_ray,
            )

        # 3. Actor without reset_state() must fail closed on reset('state') instead of no-op ok=True.
        import ray_participants as RP

        no_reset_host = types.SimpleNamespace()
        self.addCleanup(RP._LOCAL.pop, "missing_reset_actor", None)
        reply = serve_nemo_rl_actor(no_reset_host, "missing_reset_actor", "reset", "state")
        self.assertFalse(reply["ok"], reply)
        self.assertEqual(reply["findings"][0]["check"], "reset_failed", reply)
        self.assertIn("reset_state", reply["findings"][0]["observed"], reply)

        # 4. snapshot_automodel_s0 offloads parameter and optimizer tensors to CPU; restore uses strict zip.
        class _FakeTensor:
            def __init__(self, data, device="cuda:0"):
                self.data = list(data)
                self.device = device
                self.dtype = "float32"
                self.grad = [0.1]
                self.cpu_calls = 0

            def detach(self):
                return self

            def cpu(self):
                self.cpu_calls += 1
                return _FakeTensor(self.data, device="cpu")

            def clone(self):
                return _FakeTensor(self.data, device=self.device)

            def to(self, device=None, dtype=None):
                return _FakeTensor(self.data, device=device or self.device)

            def copy_(self, other):
                self.data = list(other.data)

        p_cuda = _FakeTensor([1.0, 2.0], device="cuda:0")
        opt_m = _FakeTensor([0.5, -0.5], device="cuda:0")
        opt_v = _FakeTensor([0.25, 0.25], device="cuda:0")
        loaded_states = []

        class _FakeModel:
            def parameters(self):
                return [p_cuda]

        class _FakeOptimizer:
            def state_dict(self):
                return {"state": {0: {"exp_avg": opt_m, "exp_avg_sq": opt_v}}, "param_groups": [{"lr": 0.01}]}

            def load_state_dict(self, st):
                loaded_states.append(st)

        fake_torch = types.SimpleNamespace(
            is_tensor=lambda obj: isinstance(obj, _FakeTensor),
            get_rng_state=lambda: ("torch", 0),
            set_rng_state=lambda s: None,
            no_grad=contextlib.nullcontext,
            cuda=types.SimpleNamespace(
                is_available=lambda: False,
                get_rng_state=lambda: None,
                set_rng_state=lambda s: None,
                synchronize=lambda: None,
                empty_cache=lambda: None,
            ),
        )
        actor = types.SimpleNamespace(model=_FakeModel(), optimizer=_FakeOptimizer(), step_counter=3)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            snap_rep = snapshot_automodel_s0(actor)
            self.assertTrue(snap_rep["ok"])
            self.assertEqual(opt_m.cpu_calls, 1)
            self.assertEqual(opt_v.cpu_calls, 1)
            self.assertEqual(actor._rsibox_s0_snapshot["optimizer"]["state"][0]["exp_avg"].device, "cpu")
            p_cuda.data = [99.0, 99.0]
            actor.step_counter = 10
            rst_rep = restore_automodel_s0(actor)
            self.assertTrue(rst_rep["ok"])
            self.assertEqual(p_cuda.data, [1.0, 2.0])
            self.assertEqual(actor.step_counter, 3)
            self.assertIsNone(p_cuda.grad)
            self.assertEqual(len(loaded_states), 1)
            # strict zip rejects mismatched parameter count
            actor._rsibox_s0_snapshot["params"].append(_FakeTensor([3.0], device="cpu"))
            with self.assertRaises(ValueError):
                restore_automodel_s0(actor)


if __name__ == "__main__":
    unittest.main()
