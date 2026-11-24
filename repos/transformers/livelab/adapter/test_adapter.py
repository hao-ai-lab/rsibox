"""Comprehensive CPU golden unit tests for the HuggingFace Transformers RSIBox adapter."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

LIVELAB_DIR = Path(__file__).resolve().parents[1]
CORE_LIVELAB = Path(__file__).resolve().parents[4] / "core" / "livelab"
for search_path in (str(LIVELAB_DIR), str(CORE_LIVELAB)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from adapter import TransformersQuiesceHooks  # noqa: E402
from adapter.lab import TransformersLab  # noqa: E402
from adapter.toy_lab import ToyTransformersLab  # noqa: E402
from rsibox.warm import (  # noqa: E402
    DefaultRespawnDriver,
    DiskStore,
    DynamoK8sSnapshotDriver,
    QuiesceHooks,
    QuiesceReceipt,
    StoreProbe,
)


class LazyImportAndCliTests(unittest.TestCase):
    def test_adapter_imports_do_not_eagerly_load_torch_or_transformers(self) -> None:
        probe_code = (
            "import importlib, json, sys\n"
            "forbidden = {'torch', 'transformers', 'cuda', 'cupy', 'triton', 'pynvml', 'nvidia'}\n"
            "before = set(sys.modules)\n"
            "for mod_name in ('adapter', 'adapter.lab', 'adapter.toy_lab', 'adapter.snapshot_hooks'):\n"
            "    importlib.import_module(mod_name)\n"
            "loaded = sorted({name.split('.')[0] for name in (set(sys.modules) - before) if name.split('.')[0] in forbidden})\n"
            "print(json.dumps({'loaded': loaded}))\n"
        )
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(LIVELAB_DIR), str(CORE_LIVELAB)]),
            "CUDA_VISIBLE_DEVICES": "",
        }
        proc = subprocess.run([sys.executable, "-c", probe_code], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["loaded"], [])

    def test_launch_environment_parses_cli_flags(self) -> None:
        env = TransformersLab.launch_environment(
            ["--model", "Qwen/Qwen3-0.6B", "--device", "cuda:0", "--dtype", "bfloat16", "--max-new-tokens", "32"]
        )
        self.assertEqual(
            env,
            {
                "RSIBOX_TRANSFORMERS_MODEL": "Qwen/Qwen3-0.6B",
                "RSIBOX_TRANSFORMERS_DEVICE": "cuda:0",
                "RSIBOX_TRANSFORMERS_DTYPE": "bfloat16",
                "RSIBOX_TRANSFORMERS_MAX_NEW_TOKENS": "32",
            },
        )


class ToyTransformersLiveLabTests(unittest.TestCase):
    def test_live_swap_witness_rollback_and_fresh_parity(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)

        base_status = lab.status()
        self.assertEqual(base_status["state"], "base")
        base_check = lab.check()
        self.assertTrue(base_check["ok"], base_check)
        base_output = base_check["output"]

        fresh_base = lab.fresh_check(None)
        self.assertTrue(fresh_base["ok"], fresh_base)
        self.assertEqual(fresh_base["output"], base_output)

        visible_patch = lab.patches()["visible"]
        install_reply = lab.install(visible_patch)
        self.assertTrue(install_reply["ok"], install_reply)
        self.assertEqual(install_reply["outcome"], "committed")

        edited_check = lab.check()
        self.assertTrue(edited_check["ok"], edited_check)
        self.assertTrue(edited_check["proven"])
        self.assertNotEqual(edited_check["output"], base_output)
        worker_witness = edited_check["witness"].get("worker0", {})
        self.assertTrue(worker_witness and min(worker_witness.values()) > 0, edited_check["witness"])

        fresh_edited = lab.fresh_check(visible_patch)
        self.assertTrue(fresh_edited["ok"], fresh_edited)
        self.assertEqual(fresh_edited["output"], edited_check["output"])

        uninstall_reply = lab.uninstall()
        self.assertTrue(uninstall_reply["ok"], uninstall_reply)
        self.assertEqual(uninstall_reply["state"], "base")
        self.assertEqual(lab.status()["fingerprint"], base_status["fingerprint"])
        after_rollback = lab.check()
        self.assertTrue(after_rollback["ok"])
        self.assertEqual(after_rollback["output"], base_output)
        self.assertIsNone(after_rollback["proven"])
        self.assertEqual(after_rollback["witness"], {})

    def test_fault_injection_crash_hang_lost_tamper_and_respawn_recovery(self) -> None:
        lab = ToyTransformersLab(recovery_driver=DefaultRespawnDriver(), weight_store=DiskStore())
        self.addCleanup(lab.close)
        base_output = lab.check()["output"]
        visible_patch = lab.patches()["visible"]

        for fault_name in ("crash", "hang", "lost", "tamper"):
            with self.subTest(fault=fault_name):
                injected = lab.inject(fault_name)
                self.assertTrue(injected["ok"], injected)
                lab.check()
                poisoned_status = lab.status()
                self.assertEqual(poisoned_status["state"], "poisoned", poisoned_status)
                refused = lab.install(visible_patch)
                self.assertFalse(refused["ok"])
                self.assertEqual(refused["codes"], ["session_poisoned"])
                respawn_reply = lab.reset("respawn")
                self.assertTrue(respawn_reply["ok"], respawn_reply)
                self.assertEqual(respawn_reply["driver"], "default_respawn")
                self.assertEqual(lab.status()["state"], "base")
                self.assertEqual(lab.check()["output"], base_output)

    def test_unhealthy_weight_store_fails_closed_at_startup(self) -> None:
        class _BrokenWeightStore:
            name = "broken_store"

            def probe(self, **kwargs: Any) -> StoreProbe:
                return StoreProbe(ok=False, backend="broken_store", error="store offline")

        with self.assertRaisesRegex(RuntimeError, "store offline"):
            ToyTransformersLab(weight_store=_BrokenWeightStore())


class TransformersQuiesceHooksTests(unittest.TestCase):
    def test_full_snapshot_driver_lifecycle_with_real_worker_and_post_resume_live_edit(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        base_output = lab.check()["output"]
        base_fingerprint = lab.status()["fingerprint"]

        hooks = lab.quiesce_hooks()
        self.assertIsInstance(hooks, TransformersQuiesceHooks)
        self.assertIsInstance(hooks, QuiesceHooks)
        self.assertEqual(hooks.mutability, "readonly_weights")

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="tfsnap-") as control_dir:
            control_path = Path(control_dir)
            ready_sentinel = control_path / "ready-for-snapshot"
            restore_complete_sentinel = control_path / "restore-complete"
            restore_ready_sentinel = control_path / "transformers-restore-ready"
            # Plant stale sentinels to confirm lifecycle clears them before warmup.
            for stale_file in (ready_sentinel, restore_complete_sentinel, restore_ready_sentinel):
                stale_file.write_text("stale", encoding="utf-8")

            observed: dict[str, Any] = {}

            def simulate_snapshot_agent() -> None:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if ready_sentinel.exists() and not restore_complete_sentinel.exists():
                        observed["saw_clean_ready"] = True
                        restore_complete_sentinel.write_text("ok\n", encoding="utf-8")
                        return
                    time.sleep(0.01)

            agent_thread = threading.Thread(target=simulate_snapshot_agent, daemon=True)
            agent_thread.start()
            driver = DynamoK8sSnapshotDriver(
                engine_name="transformers", control_dir=control_dir, wait_timeout_s=5.0, poll_s=0.01
            )
            lifecycle = driver.checkpoint_and_wait(hooks)
            agent_thread.join(timeout=2.0)

            self.assertTrue(lifecycle.ok, lifecycle.to_dict())
            self.assertTrue(observed.get("saw_clean_ready"))
            self.assertTrue(restore_ready_sentinel.exists())
            self.assertEqual(lifecycle.quiesce.released, ("kv_cache", "weights"))
            self.assertEqual(lifecycle.quiesce.token, "transformers:cpu_offload:kv_cache,weights")
            self.assertTrue(lifecycle.resume_details["participants_ok"])
            self.assertEqual(lifecycle.resume_details["fingerprint"], base_fingerprint)

        # Generation remains bit-exact after quiesce + resume.
        post_resume_check = lab.check()
        self.assertTrue(post_resume_check["ok"])
        self.assertEqual(post_resume_check["output"], base_output)

        # RSIBox live hot-swap still succeeds after snapshot resume.
        visible_patch = lab.patches()["visible"]
        install_reply = lab.install(visible_patch)
        self.assertTrue(install_reply["ok"], install_reply)
        edited_check = lab.check()
        self.assertTrue(edited_check["proven"])
        self.assertNotEqual(edited_check["output"], base_output)
        self.assertTrue(lab.uninstall()["ok"])
        self.assertEqual(lab.check()["output"], base_output)

    def test_cache_clear_only_mode_skips_weight_offload(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        hooks = lab.quiesce_hooks(offload_to_cpu=False, pin_memory=False)
        receipt = hooks.quiesce_for_snapshot()
        self.assertTrue(receipt.ok, receipt)
        self.assertEqual(receipt.released, ("kv_cache",))
        self.assertEqual(receipt.token, "transformers:cache_clear:kv_cache")
        resume_reply = hooks.resume_after_restore(receipt)
        self.assertTrue(resume_reply["ok"], resume_reply)

    def test_stopped_and_dirty_labs_fail_closed_across_all_hooks(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        hooks = lab.quiesce_hooks()
        sample_receipt = QuiesceReceipt(ok=True, released=("kv_cache", "weights"))

        # Dirty channel fails closed across all three hooks.
        lab.dirty = True
        with self.assertRaisesRegex(RuntimeError, "dirty"):
            hooks.warmup_for_snapshot()
        dirty_quiesce = hooks.quiesce_for_snapshot()
        self.assertFalse(dirty_quiesce.ok)
        self.assertIn("dirty", dirty_quiesce.error or "")
        dirty_resume = hooks.resume_after_restore(sample_receipt)
        self.assertFalse(dirty_resume["ok"])
        self.assertIn("dirty", dirty_resume["error"])

        # Stopped worker fails closed across all three hooks.
        lab.dirty = False
        lab.close()
        with self.assertRaisesRegex(RuntimeError, "not running"):
            hooks.warmup_for_snapshot()
        stopped_quiesce = hooks.quiesce_for_snapshot()
        self.assertFalse(stopped_quiesce.ok)
        self.assertIn("not running", stopped_quiesce.error or "")
        stopped_resume = hooks.resume_after_restore(sample_receipt)
        self.assertFalse(stopped_resume["ok"])
        self.assertIn("not running", stopped_resume["error"])

    def test_quiesce_and_resume_failures_block_snapshot_sentinels(self) -> None:
        lab = ToyTransformersLab()
        self.addCleanup(lab.close)
        hooks = lab.quiesce_hooks()

        # 1. Quiesce failure must NEVER write ready-for-snapshot.
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="tfsnap-") as control_dir:
            driver = DynamoK8sSnapshotDriver(
                engine_name="transformers", control_dir=control_dir, wait_timeout_s=0.5, poll_s=0.01
            )
            with mock.patch.object(lab, "quiesce_worker", return_value={"ok": False, "error": "offload OOM"}):
                lifecycle = driver.checkpoint_and_wait(hooks)
            self.assertFalse(lifecycle.ok)
            self.assertIn("offload OOM", lifecycle.error or "")
            self.assertFalse((Path(control_dir) / "ready-for-snapshot").exists())
            self.assertFalse((Path(control_dir) / "transformers-restore-ready").exists())

        # 2. Post-quiesce probe failure must NEVER write ready-for-snapshot.
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="tfsnap-") as control_dir:
            driver = DynamoK8sSnapshotDriver(
                engine_name="transformers", control_dir=control_dir, wait_timeout_s=0.5, poll_s=0.01
            )
            with mock.patch.object(lab, "_probe_participants", side_effect=[(True, None), (False, "worker0 gone")]):
                lifecycle = driver.checkpoint_and_wait(hooks)
            self.assertFalse(lifecycle.ok)
            self.assertIn("worker0 gone", lifecycle.error or "")
            self.assertFalse((Path(control_dir) / "ready-for-snapshot").exists())

        # 3. Resume failure or poisoned status after restore must NEVER write transformers-restore-ready.
        for label, patch_target, patch_value, expected_text in (
            ("resume_error", "resume_worker", {"ok": False, "error": "cuda H2D failure"}, "cuda H2D failure"),
            ("poisoned_status", "status", {"ok": True, "state": "poisoned", "poison": [{"check": "tampered"}]}, "tampered"),
        ):
            with self.subTest(case=label), tempfile.TemporaryDirectory(dir="/tmp", prefix="tfsnap-") as control_dir:
                control_path = Path(control_dir)

                def write_restore_complete() -> None:
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        if (control_path / "ready-for-snapshot").exists():
                            (control_path / "restore-complete").write_text("ok\n", encoding="utf-8")
                            return
                        time.sleep(0.01)

                agent_thread = threading.Thread(target=write_restore_complete, daemon=True)
                agent_thread.start()
                driver = DynamoK8sSnapshotDriver(
                    engine_name="transformers", control_dir=control_dir, wait_timeout_s=5.0, poll_s=0.01
                )
                with mock.patch.object(lab, patch_target, return_value=patch_value):
                    lifecycle = driver.checkpoint_and_wait(hooks)
                agent_thread.join(timeout=2.0)

                self.assertFalse(lifecycle.ok)
                self.assertIn(expected_text, lifecycle.error or "")
                self.assertFalse((control_path / "transformers-restore-ready").exists())
                self.assertFalse(lifecycle.resume_details["participants_ok"])


if __name__ == "__main__":
    unittest.main()
