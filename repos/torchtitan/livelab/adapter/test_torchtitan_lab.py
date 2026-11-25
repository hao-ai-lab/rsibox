"""Golden unit tests and full 11/11 conformance suite for `TorchTitanLab` / `ToyTorchTitanLab`."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

import adapter  # noqa: F401 - puts core/livelab on sys.path
from adapter.lab import DEFAULT_MODULES, TorchTitanLab, ToyTorchTitanLab
from conformance import cases
from rsibox import harness


def _cuda_fsdp2_available() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "":
        return False
    try:
        import torch
        from torch.distributed.fsdp import fully_shard  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class TorchTitanConformanceTests(unittest.TestCase):
    """Runs all 11/11 `core/livelab/conformance` cases (`c0`–`c10`) on 2-rank `ToyTorchTitanLab`."""

    def test_full_conformance_c0_to_c10_cpu_two_ranks(self):
        """Input: `ToyTorchTitanLab` (2 worker ranks, pure-Python fake-distributed).
        Expected: all 11 conformance cases c0..c10 pass with 0 failures."""
        failures = []
        for name in cases.CASES:
            result = cases.run(name, ToyTorchTitanLab)
            if result["result"] != "pass":
                failures.append((name, result["result"], result["error"]))
        self.assertEqual(failures, [], f"conformance failures: {failures}")


class TorchTitanGoldenTests(unittest.TestCase):
    """Training-loop and FSDP2-specific golden tests targeting real failure modes."""

    def setUp(self) -> None:
        self.lab = ToyTorchTitanLab()

    def tearDown(self) -> None:
        self.lab.close()

    def test_step_gap_hotswap_loss_mlp_optimizer_and_bit_exact_rollback(self):
        """Input: hot-swap `compute_loss`, `mlp_forward`, and `optimizer_step_scale` across 2 ranks.
        Expected: witness > 0 on both ranks for all 3 targets, losses/grad_norms shift, and
        uninstall restores base fingerprint and bit-exact losses (`max_abs_loss_diff == 0.0`)."""
        base_status = self.lab.status()
        base_check = self.lab.check({"steps": 3})
        self.assertTrue(base_check["ok"])
        base_losses = base_check["output"]["losses"]
        base_grads = base_check["output"]["grad_norms"]
        self.assertEqual(len(base_losses), 3)

        m_mod, o_mod = DEFAULT_MODULES
        m_src = self.lab._base_source(m_mod)
        o_src = self.lab._base_source(o_mod)
        m_edited = harness.edit_once(
            harness.edit_once(m_src, "d / (1.0 + math.exp", "1.25 * d / (1.0 + math.exp"),
            "return sum(d * d for d in diffs) / max(len(diffs), 1)",
            "return (sum(d * d for d in diffs) / max(len(diffs), 1)) * 1.5",
        )
        o_edited = harness.edit_once(
            o_src,
            "return base_lr / (1.0 + 0.05 * step_idx)",
            "return base_lr / (1.0 + 0.25 * step_idx)",
        )
        res = self.lab.install({"files": {m_mod: m_edited, o_mod: o_edited}})
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["outcome"], "committed")

        edited_check = self.lab.check({"steps": 3})
        self.assertTrue(edited_check["ok"])
        self.assertIs(edited_check["proven"], True)
        self.assertNotEqual(edited_check["output"]["losses"], base_losses)
        self.assertNotEqual(edited_check["output"]["grad_norms"], base_grads)
        # Witness must fire on every rank for every edited op.
        for rank_name in ("rank0", "rank1"):
            rank_w = edited_check["witness"][rank_name]
            self.assertEqual(len(rank_w), 3, rank_w)
            self.assertTrue(all(cnt >= 3 for cnt in rank_w.values()), rank_w)

        undo = self.lab.uninstall()
        self.assertTrue(undo["ok"], undo)
        self.assertEqual(undo["state"], "base")
        after_status = self.lab.status()
        self.assertEqual(after_status["fingerprint"], base_status["fingerprint"])
        after_check = self.lab.check({"steps": 3})
        max_loss_diff = max(abs(a - b) for a, b in zip(base_losses, after_check["output"]["losses"]))
        max_grad_diff = max(abs(a - b) for a, b in zip(base_grads, after_check["output"]["grad_norms"]))
        self.assertEqual(max_loss_diff, 0.0)
        self.assertEqual(max_grad_diff, 0.0)

    def test_stateful_continuation_advances_and_reset_restores_s0(self):
        """Input: run 2 steps with `continue_state=True`, then 2 more steps, then `reset('state')`.
        Expected: continued steps diverge from step 0..1, and `reset('state')` restores exact S0."""
        c0 = self.lab.check({"steps": 2, "continue_state": True})
        c1 = self.lab.check({"steps": 2, "continue_state": True})
        self.assertNotEqual(c0["output"]["losses"], c1["output"]["losses"])
        self.assertEqual(c1["output"]["per_rank"]["rank0"]["step"], 4)

        rst = self.lab.reset("state")
        self.assertTrue(rst["ok"], rst)
        self.assertIn("weights_s0", rst["reset"])
        c2 = self.lab.check({"steps": 2, "continue_state": True})
        self.assertEqual(c2["output"], c0["output"])

    def test_refuses_init_constructor_edit_without_mutating_state(self):
        """Input: edit `TitanStepConfig.__init__` (startup-only constructor).
        Expected: refused with `constructor_edit`, state stays `base`, fingerprint and output unchanged."""
        fp0 = self.lab.status()["fingerprint"]
        out0 = self.lab.check()["output"]
        m_mod = DEFAULT_MODULES[0]
        m_src = self.lab._base_source(m_mod)
        ctor_patch = harness.edit_once(
            m_src,
            "float(lr), init_weight_scale()",
            "float(lr), init_weight_scale() * 2.0",
        )
        res = self.lab.install({"files": {m_mod: ctor_patch}})
        self.assertFalse(res["ok"], res)
        self.assertIn("constructor_edit", res["codes"], res)
        state = self.lab.status()
        self.assertEqual(state["state"], "base")
        self.assertEqual(state["fingerprint"], fp0)
        self.assertEqual(self.lab.check()["output"], out0)

    def test_crash_poisons_session_and_respawn_recovers_base(self):
        """Input: inject worker crash on rank0.
        Expected: `check()` fails and poisons session, subsequent `install()` refuses with
        `session_poisoned`, and `reset('respawn')` restores clean base state and bit-exact output."""
        out0 = self.lab.check()["output"]
        self.assertTrue(self.lab.inject("crash", rank=0)["ok"])
        failed = self.lab.check()
        self.assertFalse(failed["ok"])
        state = self.lab.status()
        self.assertEqual(state["state"], "poisoned")
        refused = self.lab.install(self.lab.patches()["visible"])
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["codes"], ["session_poisoned"])

        respawn = self.lab.reset("respawn")
        self.assertTrue(respawn["ok"], respawn)
        self.assertEqual(self.lab.status()["state"], "base")
        self.assertEqual(self.lab.check()["output"], out0)

    def test_world_size_validation_pluggable_hooks_and_parallel_fresh_parity(self):
        """Input: invalid `world_size`/`timeout`, pluggable `weight_store`/`recovery_driver`,
        transient-thread spawn, parallel `PackageOverlay` propagation, and asymmetric sibling hang.
        Expected: strict fail-closed validation, `weight_store.probe(tp=2)` + `recovery_driver` work,
        workers survive spawning-thread exit, and rank-0 crash kills hanging rank-1 fast."""
        import threading
        import time
        from adapter.worker import _run_fresh_parallel, aggregate_ranks, run_fresh

        for bad_ws in (0, -1, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                TorchTitanLab({"mode": "fsdp2", "world_size": bad_ws}, auto_spawn=False)
            with self.assertRaises(ValueError):
                run_fresh({"config": {"mode": "fsdp2", "world_size": bad_ws}, "workload": {"steps": 1}})
        for bad_to in (0, 0.0, -5.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                run_fresh({"config": {"mode": "cpu", "world_size": 2}, "workload": {"steps": 1}, "timeout": bad_to})

        # Pluggable weight_store fail-closed probe with tp == ranks == 2, and recovery_driver delegation.
        probe_calls: list[int] = []

        from rsibox.warm.recovery import DefaultRespawnDriver
        from rsibox.warm.types import StoreProbe

        class _RejectStore:
            name = "test-store"

            def probe(self, *, tp=None, visible_devices=None):
                probe_calls.append(tp)
                return StoreProbe(ok=False, backend=self.name, error="store offline")

        with self.assertRaisesRegex(RuntimeError, "store offline"):
            TorchTitanLab({"mode": "cpu", "world_size": 2}, weight_store=_RejectStore(), auto_spawn=True)
        self.assertEqual(probe_calls, [2])

        respawn_calls: list[int] = []

        class _TrackStore:
            name = "track-store"

            def probe(self, *, tp=None, visible_devices=None):
                respawn_calls.append(tp)
                return StoreProbe(ok=True, backend=self.name)

        class _TrackDriver(DefaultRespawnDriver):
            def respawn(self, lab, *args, **kwargs):
                respawn_calls.append(-lab.ranks)
                return super().respawn(lab, *args, **kwargs)

        custom_lab = ToyTorchTitanLab(weight_store=_TrackStore(), recovery_driver=_TrackDriver())
        self.addCleanup(custom_lab.close)
        self.assertEqual(custom_lab.ranks, 2)
        rst = custom_lab.reset("respawn")
        self.assertTrue(rst["ok"], rst)
        # Initial boot probes tp=2; respawn invokes _TrackDriver (-2) + DefaultRespawnDriver weight_store probe (2) + _start_engine probe (2).
        self.assertIn(-2, respawn_calls)
        self.assertGreaterEqual(respawn_calls.count(2), 2)

        # Negative rank must fail closed instead of indexing self._workers[-1].
        self.assertFalse(custom_lab.inject("crash", rank=-1)["ok"])

        # Spawning live workers on a transient thread must NOT trigger PR_SET_PDEATHSIG SIGTERM on thread exit.
        spawned: list[ToyTorchTitanLab] = []
        t_spawn = threading.Thread(target=lambda: spawned.append(ToyTorchTitanLab()))
        t_spawn.start()
        t_spawn.join(timeout=15.0)
        self.assertEqual(len(spawned), 1)
        self.addCleanup(spawned[0].close)
        time.sleep(0.15)
        self.assertTrue(spawned[0].check()["ok"])

        # Real 2-rank parallel subprocess fresh_check with PackageOverlay patch propagation!
        patch = self.lab.patches()["visible"]
        self.assertTrue(self.lab.install(patch)["ok"])
        live_edited = self.lab.check()["output"]
        self.assertTrue(self.lab.uninstall()["ok"])
        overrides = self.lab.files(patch, against="disk")
        livelab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_dir = os.path.abspath(os.path.join(livelab_dir, "..", "..", "..", "core", "livelab"))
        old_pp = os.environ.get("PYTHONPATH")
        try:
            with harness.PackageOverlay("adapter", overrides) as overlay:
                os.environ["PYTHONPATH"] = os.pathsep.join([overlay.root, livelab_dir, core_dir] + ([old_pp] if old_pp else []))
                per_rank = _run_fresh_parallel(
                    {"mode": "cpu", "world_size": 2, "hidden": 8, "ffn": 16, "layers": 1, "steps": 2, "seed": 0},
                    {"steps": 2},
                    world_size=2,
                    timeout_s=30.0,
                )
            self.assertEqual(aggregate_ranks(per_rank), live_edited)

            # Asymmetric sibling hang: one rank fails immediately while sibling sleeps 20 s -> must fail fast (< 8 s)!
            m_mod = DEFAULT_MODULES[0]
            # Rank 0 and rank 1 have distinct deterministic y_batch[0][0] (seed 500 vs 501): check exact sign split.
            import random as _r_chk
            r0_y = _r_chk.Random(500).uniform(-0.5, 0.5)
            r1_y = _r_chk.Random(501).uniform(-0.5, 0.5)
            mid_y = (r0_y + r1_y) / 2.0
            asym_src = harness.edit_once(
                self.lab._base_source(m_mod),
                "diffs = [(p_val - t_val) for p_row, t_row in zip(pred, target) for p_val, t_val in zip(p_row, t_row)]",
                "import time as _time\n"
                f"    if target[0][0] < {mid_y!r}:\n"
                "        raise RuntimeError('rank-crash-fast')\n"
                "    _time.sleep(20.0)\n"
                "    diffs = [(p_val - t_val) for p_row, t_row in zip(pred, target) for p_val, t_val in zip(p_row, t_row)]",
            )
            asym_overrides = self.lab.files({"files": {m_mod: asym_src}}, against="disk")
            t0 = time.monotonic()
            with harness.PackageOverlay("adapter", asym_overrides) as asym_ov:
                os.environ["PYTHONPATH"] = os.pathsep.join([asym_ov.root, livelab_dir, core_dir] + ([old_pp] if old_pp else []))
                with self.assertRaisesRegex(RuntimeError, "rank-crash-fast"):
                    _run_fresh_parallel(
                        {"mode": "cpu", "world_size": 2, "hidden": 8, "ffn": 16, "layers": 1, "steps": 1, "seed": 0},
                        {"steps": 1},
                        world_size=2,
                        timeout_s=30.0,
                    )
            self.assertLess(time.monotonic() - t0, 8.0)
        finally:
            if old_pp is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pp

    def test_rng_state_restored_bit_exact_for_stochastic_hotswapped_op(self):
        """Input: hot-swap `compute_loss` to add `random.random() * 1e-3` noise.
        Expected: repeated checks with `reset('state')` produce bit-exact identical stochastic losses,
        and `uninstall()` restores exact base output."""
        base_out = self.lab.check()["output"]
        module_name = DEFAULT_MODULES[0]
        src = self.lab._base_source(module_name)
        stochastic = harness.edit_once(
            src,
            "return sum(d * d for d in diffs) / max(len(diffs), 1)",
            "import random as _r\n    return sum(d * d for d in diffs) / max(len(diffs), 1) + _r.random() * 1e-3",
        )
        res = self.lab.install({"files": {module_name: stochastic}})
        self.assertTrue(res["ok"], res)
        c1 = self.lab.check()
        self.assertTrue(c1["ok"] and c1["proven"] is True)
        self.assertNotEqual(c1["output"]["losses"], base_out["losses"])
        rst = self.lab.reset("state")
        self.assertTrue(rst["ok"] and "rng" in rst["reset"], rst)
        c2 = self.lab.check()
        self.assertEqual(c2["output"], c1["output"])
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], base_out)

    def test_launch_environment_parses_flags_without_importing_torch(self):
        """Input: `TorchTitanLab.launch_environment(['--', '--mode', 'fsdp2', '--world-size', '2'])` in a clean subprocess.
        Expected: returns expected RSIBOX_TORCHTITAN_* env vars without importing `torch`."""
        livelab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        probe = (
            "import json, sys; from adapter.lab import TorchTitanLab; "
            "env = TorchTitanLab.launch_environment(['--', '--mode', 'fsdp2', '--world-size', '2', '--steps', '3']); "
            "print(json.dumps({'env': env, 'torch_loaded': 'torch' in sys.modules}))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            env={**os.environ, "PYTHONPATH": livelab_dir},
            capture_output=True, text=True, check=True,
        )
        data = json.loads(proc.stdout.strip())
        self.assertEqual(data["env"]["RSIBOX_TORCHTITAN_MODE"], "fsdp2")
        self.assertEqual(data["env"]["RSIBOX_TORCHTITAN_WORLD_SIZE"], "2")
        self.assertEqual(data["env"]["RSIBOX_TORCHTITAN_STEPS"], "3")
        self.assertFalse(data["torch_loaded"])


class TorchTitanRealCudaFsdp2Tests(unittest.TestCase):
    """Real PyTorch CUDA + FSDP2 (`fully_shard`) verification (auto-skips on CPU-only hosts)."""

    def test_real_fsdp2_cuda_hotswap_and_bit_exact_rollback(self):
        """Input: real `TorchTitanLab(mode='fsdp2')` on CUDA with `torch.distributed.fsdp.fully_shard`.
        Expected: FSDP2 step runs on CUDA, live loss hot-swap shifts loss with witness > 0, and
        uninstall restores bit-exact loss (`max_abs_loss_diff == 0.0`)."""
        if not _cuda_fsdp2_available():
            self.skipTest("requires CUDA GPU and PyTorch FSDP2")
        lab = TorchTitanLab(
            {"mode": "fsdp2", "world_size": 1, "hidden": 64, "ffn": 128, "layers": 2, "steps": 2, "seed": 0},
            check_timeout=30.0,
            startup_timeout=60.0,
        )
        try:
            c0 = lab.check()
            self.assertTrue(c0["ok"], c0)
            self.assertTrue(c0["output"]["per_rank"]["rank0"]["fsdp2"])
            self.assertTrue(c0["output"]["per_rank"]["rank0"]["device"].startswith("cuda"))
            res = lab.install(lab.patches()["visible"])
            self.assertTrue(res["ok"], res)
            c1 = lab.check()
            self.assertTrue(c1["ok"] and c1["proven"] is True, c1)
            self.assertNotEqual(c1["output"]["losses"], c0["output"]["losses"])
            u = lab.uninstall()
            self.assertTrue(u["ok"], u)
            c2 = lab.check()
            max_diff = max(abs(a - b) for a, b in zip(c0["output"]["losses"], c2["output"]["losses"]))
            self.assertEqual(max_diff, 0.0)
        finally:
            lab.close()

    def test_real_fsdp2_cuda_two_rank_nccl_hotswap_fresh_parity_and_rollback(self):
        """Input: real 2-rank `TorchTitanLab(mode='fsdp2', world_size=2)` over NCCL on 2 CUDA GPUs.
        Expected: both ranks shard via `fully_shard` on distinct GPUs (`cuda:0`, `cuda:1`), live
        hot-swap proves execution on both ranks, matches parallel 2-rank `fresh_check()`, and
        uninstall restores bit-exact losses (`max_abs_loss_diff == 0.0`)."""
        if not _cuda_fsdp2_available():
            self.skipTest("requires CUDA GPU and PyTorch FSDP2")
        import torch

        if torch.cuda.device_count() < 2:
            self.skipTest("requires >= 2 visible CUDA GPUs for world_size=2 FSDP2 NCCL test")
        lab = TorchTitanLab(
            {"mode": "fsdp2", "world_size": 2, "hidden": 64, "ffn": 128, "layers": 2, "steps": 2, "seed": 0},
            check_timeout=45.0,
            startup_timeout=90.0,
        )
        try:
            c0 = lab.check()
            self.assertTrue(c0["ok"], c0)
            self.assertEqual(c0["output"]["per_rank"]["rank0"]["device"], "cuda:0")
            self.assertEqual(c0["output"]["per_rank"]["rank1"]["device"], "cuda:1")
            self.assertTrue(c0["output"]["per_rank"]["rank0"]["fsdp2"])
            self.assertTrue(c0["output"]["per_rank"]["rank1"]["fsdp2"])
            patch = lab.patches()["visible"]
            res = lab.install(patch)
            self.assertTrue(res["ok"], res)
            c1 = lab.check()
            self.assertTrue(c1["ok"] and c1["proven"] is True, c1)
            self.assertNotEqual(c1["output"]["losses"], c0["output"]["losses"])
            self.assertIn("rank0", c1["witness"])
            self.assertIn("rank1", c1["witness"])
            fresh = lab.fresh_check(patch)
            self.assertTrue(fresh["ok"], fresh)
            self.assertEqual(c1["output"], fresh["output"])
            u = lab.uninstall()
            self.assertTrue(u["ok"], u)
            c2 = lab.check()
            max_diff = max(abs(a - b) for a, b in zip(c0["output"]["losses"], c2["output"]["losses"]))
            self.assertEqual(max_diff, 0.0)
        finally:
            lab.close()


if __name__ == "__main__":
    unittest.main()
