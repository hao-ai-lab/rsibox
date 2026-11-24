"""CPU golden unit tests for the NVIDIA TensorRT-LLM (PyTorch backend) RSIBox adapter."""
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
from unittest import mock

import adapter
from adapter.lab import TrtllmLab
from adapter.snapshot_hooks import TrtllmQuiesceHooks
from adapter.toy_lab import ToyTrtllmLab
from rsibox.warm import DynamoK8sSnapshotDriver, QuiesceReceipt, SENTINEL_READY, SENTINEL_RESTORE_COMPLETE


class LazyImportTest(unittest.TestCase):
    """Importing the TRT-LLM adapter modules must never eagerly import GPU or TRT-LLM libraries."""

    def test_adapter_imports_stay_gpu_free(self) -> None:
        livelab_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        probe = (
            "import importlib, json, sys\n"
            "sys.path[:0] = [sys.argv[1], sys.argv[2]]\n"
            "forbidden = {'torch', 'cuda', 'triton', 'pynvml', 'nvidia', 'tensorrt', 'tensorrt_llm'}\n"
            "for mod_name in ('adapter', 'adapter.lab', 'adapter.toy_lab', 'adapter.snapshot_hooks'):\n"
            "    importlib.import_module(mod_name)\n"
            "loaded = sorted({name.split('.')[0] for name in sys.modules if name.split('.')[0] in forbidden})\n"
            "print(json.dumps(loaded))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", probe, livelab_dir, adapter.LIVELAB],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(json.loads(completed.stdout.strip()), [])

    def test_launch_environment_translates_cli_flags(self) -> None:
        translated = TrtllmLab.launch_environment(
            ["--", "--model", "Qwen/Qwen3-0.6B", "--mem-fraction", "0.2", "--check-timeout", "45", "--rpc-timeout", "15"]
        )
        self.assertEqual(
            translated,
            {
                "RSIBOX_TRTLLM_MODEL": "Qwen/Qwen3-0.6B",
                "RSIBOX_TRTLLM_MEM_FRACTION": "0.2",
                "RSIBOX_TRTLLM_CHECK_TIMEOUT": "45.0",
                "RSIBOX_TRTLLM_RPC_TIMEOUT": "15.0",
            },
        )


class ToyTrtllmAdapterTest(unittest.TestCase):
    """End-to-end CPU golden tests against spawned `ToyTrtllmLab` worker subprocesses."""

    def setUp(self) -> None:
        self.lab = ToyTrtllmLab(ranks=2)

    def tearDown(self) -> None:
        self.lab.close()

    def test_live_swap_witness_and_exact_rollback(self) -> None:
        base_status = self.lab.status()
        base_check = self.lab.check()
        self.assertTrue(base_status["ok"])
        self.assertEqual(base_status["state"], "base")
        self.assertTrue(base_check["ok"])

        visible_patch = self.lab.patches()["visible"]
        install_reply = self.lab.install(visible_patch)
        self.assertTrue(install_reply["ok"])
        self.assertEqual(install_reply["outcome"], "committed")

        edited_check = self.lab.check()
        self.assertTrue(edited_check["ok"])
        self.assertTrue(edited_check["proven"])
        self.assertNotEqual(edited_check["output"], base_check["output"])
        for rank_name in ("rank0", "rank1"):
            self.assertIn(rank_name, edited_check["witness"])
            self.assertGreater(min(edited_check["witness"][rank_name].values()), 0)

        uninstall_reply = self.lab.uninstall()
        self.assertTrue(uninstall_reply["ok"])
        self.assertEqual(uninstall_reply["state"], "base")
        restored_status = self.lab.status()
        restored_check = self.lab.check()
        self.assertEqual(restored_status["fingerprint"], base_status["fingerprint"])
        self.assertEqual(restored_check["output"], base_check["output"])

    def test_fresh_subprocess_parity_and_unreached_detection(self) -> None:
        base_output = self.lab.check()["output"]
        fresh_base = self.lab.fresh_check(None)
        self.assertTrue(fresh_base["ok"])
        self.assertEqual(fresh_base["output"], base_output)

        visible_patch = self.lab.patches()["visible"]
        self.assertTrue(self.lab.install(visible_patch)["ok"])
        live_visible = self.lab.check()["output"]
        fresh_visible = self.lab.fresh_check(visible_patch)
        self.assertTrue(fresh_visible["ok"])
        self.assertEqual(fresh_visible["output"], live_visible)
        self.assertTrue(self.lab.uninstall()["ok"])

        unreached_patch = self.lab.patches()["unreached"]
        self.assertTrue(self.lab.install(unreached_patch)["ok"])
        unreached_check = self.lab.check()
        self.assertIs(unreached_check["proven"], False)
        self.assertEqual(unreached_check["output"], base_output)
        fresh_unreached = self.lab.fresh_check(unreached_patch)
        self.assertTrue(fresh_unreached["ok"])
        self.assertNotEqual(fresh_unreached["output"], base_output)

    def test_fault_rollback_and_crash_hang_respawn(self) -> None:
        base_output = self.lab.check()["output"]
        multi_patch = self.lab.patches()["multi"]
        self.assertTrue(self.lab.inject("fail_apply", after=2)["ok"])
        failed_reply = self.lab.install(multi_patch)
        self.assertFalse(failed_reply["ok"])
        self.assertIn("participant_failed", failed_reply["codes"])
        self.assertEqual(self.lab.status()["state"], "base")
        self.assertEqual(self.lab.check()["output"], base_output)

        for fault in ("crash", "hang", "lost", "tamper"):
            self.assertTrue(self.lab.inject(fault)["ok"])
            self.lab.check()
            self.assertEqual(self.lab.status()["state"], "poisoned")
            refused = self.lab.install(self.lab.patches()["visible"])
            self.assertFalse(refused["ok"])
            self.assertEqual(refused["codes"], ["session_poisoned"])
            respawn = self.lab.reset("respawn")
            self.assertTrue(respawn["ok"])
            self.assertEqual(self.lab.status()["state"], "base")
            self.assertEqual(self.lab.check()["output"], base_output)

    def test_file_channel_transport_live_swap(self) -> None:
        file_lab = ToyTrtllmLab(ranks=1, transport="file")
        try:
            base_output = file_lab.check()["output"]
            self.assertTrue(file_lab.install(file_lab.patches()["visible"])["ok"])
            edited_check = file_lab.check()
            self.assertTrue(edited_check["proven"])
            self.assertNotEqual(edited_check["output"], base_output)
            self.assertTrue(file_lab.uninstall()["ok"])
            self.assertEqual(file_lab.check()["output"], base_output)
        finally:
            file_lab.close()


class TrtllmQuiesceHooksTest(unittest.TestCase):
    """Golden tests for `TrtllmQuiesceHooks` + `DynamoK8sSnapshotDriver` sentinel lifecycle."""

    def setUp(self) -> None:
        self.lab = ToyTrtllmLab(ranks=2)

    def tearDown(self) -> None:
        self.lab.close()

    def test_full_snapshot_sentinel_lifecycle_and_post_restore_live_edit(self) -> None:
        base_output = self.lab.check()["output"]
        hooks = self.lab.quiesce_hooks()
        self.assertIsInstance(hooks, TrtllmQuiesceHooks)

        with tempfile.TemporaryDirectory(prefix="rsibox-trtllm-snap-") as control_dir:
            control_path = Path(control_dir)
            driver = DynamoK8sSnapshotDriver(
                engine_name="trtllm",
                control_dir=control_dir,
                wait_timeout_s=5.0,
                poll_s=0.02,
            )

            def simulate_snapshot_agent() -> None:
                ready_file = control_path / SENTINEL_READY
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline and not ready_file.exists():
                    time.sleep(0.02)
                if ready_file.exists():
                    (control_path / SENTINEL_RESTORE_COMPLETE).write_text("restored\n", encoding="utf-8")

            agent_thread = threading.Thread(target=simulate_snapshot_agent, daemon=True)
            agent_thread.start()
            lifecycle = driver.checkpoint_and_wait(hooks)
            agent_thread.join(timeout=5.0)

            self.assertTrue(lifecycle.ok, lifecycle.error)
            self.assertEqual(lifecycle.quiesce.released, ("kv_cache",))
            self.assertEqual(lifecycle.quiesce.token, "trtllm:kv_cache")
            self.assertTrue((control_path / "trtllm-restore-ready").exists())
            self.assertTrue(lifecycle.resume_details["participants_ok"])

        self.assertEqual(self.lab.check()["output"], base_output)

        visible_patch = self.lab.patches()["visible"]
        self.assertTrue(self.lab.install(visible_patch)["ok"])
        post_restore_check = self.lab.check()
        self.assertTrue(post_restore_check["proven"])
        self.assertNotEqual(post_restore_check["output"], base_output)
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], base_output)

    def test_quiesce_blocks_generation_until_resumed(self) -> None:
        hooks = self.lab.quiesce_hooks()
        receipt = hooks.quiesce_for_snapshot()
        self.assertTrue(receipt.ok)
        self.assertEqual(receipt.released, ("kv_cache",))

        with self.assertRaises(RuntimeError):
            self.lab._run_workload(None)

        resume_reply = hooks.resume_after_restore(receipt)
        self.assertTrue(resume_reply["ok"])
        self.assertEqual(resume_reply["resumed"], ["kv_cache"])
        self.assertTrue(self.lab.check()["ok"])

    def test_fail_closed_validation_and_partial_quiesce_rollback(self) -> None:
        with self.assertRaises(TypeError):
            TrtllmQuiesceHooks(self.lab, release_tags="kv_cache")

        hooks = self.lab.quiesce_hooks()
        original_rank1_call = self.lab.workers[1].call
        with mock.patch.object(
            self.lab.workers[1],
            "call",
            side_effect=lambda method, payload=None, timeout=None: (
                {"ok": False, "error": "injected rank1 quiesce failure"}
                if method == "quiesce_memory"
                else original_rank1_call(method, payload, timeout)
            ),
        ):
            with tempfile.TemporaryDirectory(prefix="rsibox-trtllm-snap-fail-") as control_dir:
                driver = DynamoK8sSnapshotDriver(
                    engine_name="trtllm", control_dir=control_dir, wait_timeout_s=1.0, poll_s=0.02
                )
                failed = driver.checkpoint_and_wait(hooks)
                self.assertFalse(failed.ok)
                self.assertIn("injected rank1 quiesce failure", failed.error or "")
                self.assertFalse((Path(control_dir) / SENTINEL_READY).exists())

        # rank0 must have been rolled back so generation still works cleanly.
        self.assertTrue(self.lab.check()["ok"])

        # Dirty and stopped labs must fail closed.
        self.lab._dirty = True
        self.assertFalse(hooks.quiesce_for_snapshot().ok)
        self.assertFalse(hooks.resume_after_restore(QuiesceReceipt(ok=True, released=("kv_cache",)))["ok"])
        with self.assertRaises(RuntimeError):
            hooks.warmup_for_snapshot()
        self.lab._dirty = False

        self.lab.close()
        self.assertFalse(hooks.quiesce_for_snapshot().ok)
        self.assertFalse(hooks.resume_after_restore(QuiesceReceipt(ok=True, released=("kv_cache",)))["ok"])
        with self.assertRaises(RuntimeError):
            hooks.warmup_for_snapshot()


if __name__ == "__main__":
    unittest.main()
