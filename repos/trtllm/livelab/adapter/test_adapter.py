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
from adapter.fixture.py_executor import ToyPyExecutor
from adapter.lab import TrtllmLab
from adapter.snapshot_hooks import TrtllmQuiesceHooks
from adapter.toy_lab import ToyTrtllmLab
from rsibox.warm import DynamoK8sSnapshotDriver, QuiesceReceipt, SENTINEL_READY, SENTINEL_RESTORE_COMPLETE


class LazyImportAndConfigTest(unittest.TestCase):
    """Importing the TRT-LLM adapter must never eagerly import GPU or TRT-LLM libraries."""

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
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(completed.stdout.strip()), [])

    def test_launch_environment_flags_leading_delimiter_help_and_unknown(self) -> None:
        translated = TrtllmLab.launch_environment(
            [
                "--",
                "--model",
                "Qwen/Qwen3-0.6B",
                "--mem-fraction",
                "0.2",
                "--check-timeout",
                "45",
                "--rpc-timeout",
                "15",
                "--quiesce-timeout",
                "90",
            ]
        )
        self.assertEqual(
            translated,
            {
                "RSIBOX_TRTLLM_MODEL": "Qwen/Qwen3-0.6B",
                "RSIBOX_TRTLLM_MEM_FRACTION": "0.2",
                "RSIBOX_TRTLLM_CHECK_TIMEOUT": "45.0",
                "RSIBOX_TRTLLM_RPC_TIMEOUT": "15.0",
                "RSIBOX_TRTLLM_QUIESCE_TIMEOUT": "90.0",
            },
        )
        with self.assertRaises(SystemExit) as help_exit:
            TrtllmLab.launch_environment(["--help"])
        self.assertEqual(help_exit.exception.code, 0)

        with self.assertRaises(SystemExit) as unknown_exit:
            TrtllmLab.launch_environment(["--unknown-flag", "1"])
        self.assertEqual(unknown_exit.exception.code, 2)

        with self.assertRaises(SystemExit) as double_delim_exit:
            TrtllmLab.launch_environment(["--", "--", "--model", "Qwen/Qwen3-0.6B"])
        self.assertEqual(double_delim_exit.exception.code, 2)

    def test_real_mode_tp_validation_and_explicit_args_override_env(self) -> None:
        with self.assertRaises(ValueError):
            TrtllmLab(model="dummy", tensor_parallel_size=0, auto_spawn=False)
        tp2 = TrtllmLab(model="dummy", tensor_parallel_size=2, auto_spawn=False)
        self.assertEqual(tp2.rank_count, 2)
        self.assertTrue(tp2._use_mpi_mode())
        self.assertEqual(TrtllmLab.launch_environment(["--tp", "2"]), {"RSIBOX_TRTLLM_TP": "2"})
        for parallel_key in ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"):
            with self.assertRaises(ValueError):
                TrtllmLab(model="dummy", extra_kwargs={parallel_key: 2}, auto_spawn=False)

        no_worker_lab = TrtllmLab(model="dummy", auto_spawn=False)
        with self.assertRaises(RuntimeError):
            no_worker_lab._base_source("tensorrt_llm._torch.pyexecutor.sampler")

        with mock.patch.dict(
            os.environ,
            {
                "RSIBOX_TRTLLM_MEM_FRACTION": "0.5",
                "RSIBOX_TRTLLM_CHECK_TIMEOUT": "99",
                "RSIBOX_TRTLLM_RPC_TIMEOUT": "88",
                "RSIBOX_TRTLLM_QUIESCE_TIMEOUT": "77",
            },
            clear=False,
        ):
            explicit_lab = TrtllmLab(
                model="dummy",
                free_gpu_memory_fraction=0.2,
                check_timeout=12.0,
                rpc_timeout=8.0,
                quiesce_timeout=30.0,
                auto_spawn=False,
            )
            self.assertEqual(explicit_lab.engine_config["free_gpu_memory_fraction"], 0.2)
            self.assertEqual(explicit_lab.check_timeout, 12.0)
            self.assertEqual(explicit_lab.rpc_timeout, 8.0)
            self.assertEqual(explicit_lab.quiesce_timeout, 30.0)

    def test_toy_executor_validates_refreshed_pool_pointers_and_mapping(self) -> None:
        executor = ToyPyExecutor()
        self.assertTrue(executor.generate([[1, 2]], max_tokens=2))

        executor.kv_cache_manager.kv_cache_pool_pointers = []
        with self.assertRaises(RuntimeError):
            executor.generate([[1, 2]], max_tokens=2)

        executor.kv_cache_manager.kv_cache_pool_pointers = [1]
        executor.kv_cache_manager.kv_cache_pool_mapping = []
        with self.assertRaises(RuntimeError):
            executor.generate([[1, 2]], max_tokens=2)


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

    def test_worker_linecache_tracks_install_and_restores_base_on_uninstall(self) -> None:
        sampler_module = "adapter.fixture.sampler"
        base_source = self.lab._base_source(sampler_module)
        visible_patch = self.lab.patches()["visible"]
        patched_source = visible_patch["files"][sampler_module]

        self.assertTrue(self.lab.install(visible_patch)["ok"])
        after_install = self.lab.workers[0].call("read_source", {"module": sampler_module})
        self.assertTrue(after_install["ok"])
        self.assertEqual(after_install["linecache"], patched_source)

        self.assertTrue(self.lab.uninstall()["ok"])
        after_uninstall = self.lab.workers[0].call("read_source", {"module": sampler_module})
        self.assertTrue(after_uninstall["ok"])
        self.assertEqual(after_uninstall["linecache"], base_source)

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

    def test_quiesce_state_machine_refuses_duplicate_quiesce_unreleased_resume_and_check(self) -> None:
        hooks = self.lab.quiesce_hooks()

        # Resume without prior quiesce must fail closed.
        self.assertFalse(hooks.resume_after_restore(QuiesceReceipt(ok=False, error="failed"))["ok"])
        self.assertFalse(hooks.resume_after_restore(None)["ok"])
        unquiesced_resume = hooks.resume_after_restore(QuiesceReceipt(ok=True, released=("kv_cache",)))
        self.assertFalse(unquiesced_resume["ok"])
        self.assertIn("not quiesced", unquiesced_resume["error"])

        receipt = hooks.quiesce_for_snapshot()
        self.assertTrue(receipt.ok)
        self.assertEqual(receipt.released, ("kv_cache",))

        # Double quiesce must fail closed.
        duplicate_receipt = hooks.quiesce_for_snapshot()
        self.assertFalse(duplicate_receipt.ok)
        self.assertIn("already quiesced", duplicate_receipt.error or "")

        # Generation / check while quiesced must fail cleanly.
        with self.assertRaises(RuntimeError):
            self.lab._run_workload(None)

        resume_reply = hooks.resume_after_restore(receipt)
        self.assertTrue(resume_reply["ok"])
        self.assertEqual(resume_reply["resumed"], ["kv_cache"])
        self.assertTrue(self.lab.check()["ok"])

        # Verify lab.check() itself rejects while quiesced (and recovers cleanly via respawn).
        receipt_two = hooks.quiesce_for_snapshot()
        self.assertTrue(receipt_two.ok)
        quiesced_check = self.lab.check()
        self.assertFalse(quiesced_check["ok"])
        self.assertIn("quiesced", quiesced_check.get("error", ""))

    def test_fail_closed_validation_partial_rollback_and_dirty_poison_on_failed_rollback(self) -> None:
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

        # rank0 must have been rolled back cleanly so generation still works.
        self.assertTrue(self.lab.check()["ok"])

        # If rank0 rollback ITSELF fails when rank1 quiesce fails, lab must become _dirty and poisoned.
        original_rank0_call = self.lab.workers[0].call
        with (
            mock.patch.object(
                self.lab.workers[1],
                "call",
                side_effect=lambda method, payload=None, timeout=None: (
                    {"ok": False, "error": "rank1 quiesce boom"}
                    if method == "quiesce_memory"
                    else original_rank1_call(method, payload, timeout)
                ),
            ),
            mock.patch.object(
                self.lab.workers[0],
                "call",
                side_effect=lambda method, payload=None, timeout=None: (
                    {"ok": False, "error": "rank0 rollback boom"}
                    if method == "resume_memory"
                    else original_rank0_call(method, payload, timeout)
                ),
            ),
        ):
            poisoned_receipt = hooks.quiesce_for_snapshot()
            self.assertFalse(poisoned_receipt.ok)
            self.assertIn("rank0 rollback boom", poisoned_receipt.error or "")
            self.assertTrue(self.lab._dirty)
            self.assertEqual(self.lab.status()["state"], "poisoned")

        # Dirty/poisoned and stopped labs must fail closed.
        self.assertFalse(hooks.quiesce_for_snapshot().ok)
        self.assertFalse(hooks.resume_after_restore(QuiesceReceipt(ok=True, released=("kv_cache",)))["ok"])
        with self.assertRaises(RuntimeError):
            hooks.warmup_for_snapshot()

        self.lab.close()
        self.assertFalse(hooks.quiesce_for_snapshot().ok)
        self.assertFalse(hooks.resume_after_restore(QuiesceReceipt(ok=True, released=("kv_cache",)))["ok"])
        with self.assertRaises(RuntimeError):
            hooks.warmup_for_snapshot()

    def test_pointer_shift_on_resume_fails_closed_and_poisons_lab(self) -> None:
        shift_lab = ToyTrtllmLab(ranks=1, extra_kwargs={"shift_pointers_on_alloc": True})
        try:
            hooks = shift_lab.quiesce_hooks()
            receipt = hooks.quiesce_for_snapshot()
            self.assertTrue(receipt.ok)

            resume_reply = hooks.resume_after_restore(receipt)
            self.assertFalse(resume_reply["ok"])
            self.assertIn("shifted", resume_reply.get("error", ""))
            self.assertTrue(shift_lab._dirty)
            self.assertEqual(shift_lab.status()["state"], "poisoned")
        finally:
            shift_lab.close()

    def test_block_reuse_enabled_fails_closed_on_quiesce(self) -> None:
        reuse_lab = ToyTrtllmLab(ranks=1, enable_block_reuse=True)
        try:
            hooks = reuse_lab.quiesce_hooks()
            receipt = hooks.quiesce_for_snapshot()
            self.assertFalse(receipt.ok)
            self.assertIn("enable_block_reuse=True", receipt.error or "")
            self.assertTrue(reuse_lab.check()["ok"])
        finally:
            reuse_lab.close()

    def test_file_transport_worker_sigkill_fast_fails_with_transport_error_and_recovers(self) -> None:
        """Golden regression: under `transport='file'` (used for CRIU/snapshot and SandboxPool),
        a real `SIGKILL` of the TRT-LLM worker must fail fast (< 2.0 s vs. 60 s timeout) with
        `lost=True` and `transport_error=True` (so `TrtllmQuiesceHooks` poisons cleanly instead of
        stalling), and `reset('respawn')` must recover a clean worker."""
        import signal
        import threading
        import time
        from rsibox import ipc

        lab = ToyTrtllmLab(ranks=1, transport="file", rpc_timeout=60.0, check_timeout=60.0)
        try:
            self.assertTrue(lab.check()["ok"])
            worker = lab.workers[0]
            # 1. Pre-send SIGKILL fast-fails in milliseconds with lost=True + transport_error=True.
            os.kill(worker.process.pid, signal.SIGKILL)
            worker.process.wait(timeout=5.0)
            t0 = time.perf_counter()
            reply = worker.call("ping", timeout=60.0)
            elapsed_s = time.perf_counter() - t0
            self.assertFalse(reply["ok"])
            self.assertTrue(reply.get("lost"))
            self.assertTrue(reply.get("transport_error"))
            self.assertLess(elapsed_s, 2.0, f"file-transport call stalled for {elapsed_s:.3f}s")

            # 2. Mid-quiesce SIGKILL (worker dies right after preflight passes, before quiesce_memory)
            #    must fast-fail with transport_error=True so TrtllmQuiesceHooks._poison_lab sets _dirty=True.
            respawn0 = lab.reset("respawn")
            self.assertTrue(respawn0["ok"], respawn0)
            self.assertFalse(lab._dirty)
            worker_q = lab.workers[0]
            killed_after_preflight = False
            orig_status = lab.status

            def status_then_kill(*args: object, **kwargs: object) -> dict[str, object]:
                nonlocal killed_after_preflight
                result = orig_status(*args, **kwargs)
                if not killed_after_preflight:
                    killed_after_preflight = True
                    os.kill(worker_q.process.pid, signal.SIGKILL)
                    worker_q.process.wait(timeout=5.0)
                return result

            hooks = lab.quiesce_hooks()
            t_q = time.perf_counter()
            with mock.patch.object(lab, "status", side_effect=status_then_kill):
                receipt = hooks.quiesce_for_snapshot()
            self.assertFalse(receipt.ok)
            self.assertLess(time.perf_counter() - t_q, 2.0)
            self.assertTrue(lab._dirty)
            self.assertEqual(lab.status()["state"], "poisoned")

            # 3. Mid-call SIGKILL on a freshly respawned worker fails fast instead of waiting 60 s.
            respawn = lab.reset("respawn")
            self.assertTrue(respawn["ok"], respawn)
            self.assertTrue(lab.check()["ok"])
            worker2 = lab.workers[0]
            # Arm a 30 s hang synchronously first, then SIGKILL mid-generate once ctl.json advances.
            self.assertTrue(worker2.call("inject", {"fault": "hang", "hang_s": 30.0}, timeout=10.0)["ok"])
            ctl_path = os.path.join(worker2.endpoint, "ctl.json")
            before_seq = ipc.read_json(ctl_path).get("seq", 0) if os.path.exists(ctl_path) else 0
            stop_killer = threading.Event()

            def kill_after_next_ctl() -> None:
                while not stop_killer.is_set():
                    cur = ipc.read_json(ctl_path)
                    if isinstance(cur, dict) and cur.get("seq", 0) > before_seq:
                        os.kill(worker2.process.pid, signal.SIGKILL)
                        worker2.process.wait(timeout=5.0)
                        return
                    time.sleep(0.002)

            killer = threading.Thread(target=kill_after_next_ctl, daemon=True)
            killer.start()
            t_mid = time.perf_counter()
            try:
                mid_reply = worker2.call("generate", lab.workload, timeout=60.0)
                mid_elapsed = time.perf_counter() - t_mid
            finally:
                stop_killer.set()
                killer.join(timeout=5.0)
            self.assertFalse(mid_reply["ok"])
            self.assertTrue(mid_reply.get("lost"))
            self.assertTrue(mid_reply.get("transport_error"))
            self.assertLess(mid_elapsed, 2.0, f"mid-call SIGKILL stalled for {mid_elapsed:.3f}s")

            # 4. Respawn recovers cleanly after mid-call SIGKILL.
            lab.session.fault("participant_failed", "worker killed mid-call", "clean")
            self.assertTrue(lab.reset("respawn")["ok"])
            self.assertTrue(lab.check()["ok"])
        finally:
            lab.close()


if __name__ == "__main__":
    unittest.main()


class TrtllmRecaptureAndMpiTest(unittest.TestCase):
    def test_recapture_and_undo_invalidate_cuda_graphs_on_all_ranks(self) -> None:
        lab = ToyTrtllmLab(ranks=2)
        try:
            patch = dict(lab.patches()["visible"])
            patch["rebuild"] = ["recapture"]
            installed = lab.install(patch)
            self.assertTrue(installed["ok"], installed)
            receipt = installed.get("receipt") or {}
            rebuild_info = receipt.get("rebuild") or {}
            self.assertEqual(len(rebuild_info.get("ranks") or []), 2)
            self.assertTrue(all(r.get("ok") for r in (rebuild_info["ranks"].values() if isinstance(rebuild_info["ranks"], dict) else rebuild_info["ranks"])))
            checked = lab.check()
            self.assertTrue(checked["ok"] and checked["proven"])
            undone = lab.uninstall(installed["id"])
            self.assertTrue(undone["ok"], undone)
            restored = lab.check()
            self.assertTrue(restored["ok"])
        finally:
            lab.close()

    def test_captured_prefix_auto_attaches_r1_rebuild(self) -> None:
        lab = TrtllmLab(model="dummy", tensor_parallel_size=2, auto_spawn=False)
        plan, _extra = lab._build_plan(
            {"files": {"tensorrt_llm._torch.models.modeling_qwen3": "x = 1\n"}},
            {"tensorrt_llm._torch.models.modeling_qwen3": "x = 0\n"},
        )
        self.assertEqual(plan.get("rebuild"), ["recapture"])
        plan_optout, _ = lab._build_plan(
            {"files": {"tensorrt_llm._torch.models.modeling_qwen3": "x = 1\n"}, "rebuild": []},
            {"tensorrt_llm._torch.models.modeling_qwen3": "x = 0\n"},
        )
        self.assertEqual(plan_optout.get("rebuild"), [])

    def test_mpi_hook_stays_lazy_without_gpu_imports(self) -> None:
        import adapter.mpi_hook as mh
        self.assertIn("RSIBOX_TRTLLM_MPI_SOCK_DIR", mh.SITECUSTOMIZE_SOURCE)
        self.assertNotIn("tensorrt_llm", sys.modules)
