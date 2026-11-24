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
            "self.weight_scale = init_weight_scale()",
            "self.weight_scale = init_weight_scale() * 2.0",
        )
        res = self.lab.install({"files": {m_mod: ctor_patch}})
        self.assertFalse(res["ok"], res)
        self.assertIn("constructor_edit", res["codes"], res)
        st = self.lab.status()
        self.assertEqual(st["state"], "base")
        self.assertEqual(st["fingerprint"], fp0)
        self.assertEqual(self.lab.check()["output"], out0)

    def test_crash_poisons_session_and_respawn_recovers_base(self):
        """Input: inject worker crash on rank0.
        Expected: `check()` fails and poisons session, subsequent `install()` refuses with
        `session_poisoned`, and `reset('respawn')` restores clean base state and bit-exact output."""
        out0 = self.lab.check()["output"]
        self.assertTrue(self.lab.inject("crash", rank=0)["ok"])
        failed = self.lab.check()
        self.assertFalse(failed["ok"])
        st = self.lab.status()
        self.assertEqual(st["state"], "poisoned")
        refused = self.lab.install(self.lab.patches()["visible"])
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["codes"], ["session_poisoned"])

        respawn = self.lab.reset("respawn")
        self.assertTrue(respawn["ok"], respawn)
        self.assertEqual(self.lab.status()["state"], "base")
        self.assertEqual(self.lab.check()["output"], out0)

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


if __name__ == "__main__":
    unittest.main()
