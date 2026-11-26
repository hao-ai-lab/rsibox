"""Conformance and golden unit tests for `DynamoLab` and `ToyDynamoLab`."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import adapter  # noqa: F401
from adapter import planner_ops, worker as worker_mod
from adapter.frontend_driver import FrontendDriver
from adapter.lab import DEFAULT_MODULES, DynamoLab
from adapter.toy_lab import ToyDynamoLab
from conformance import cases
from rsibox import harness
from rsibox.warm.types import StoreProbe


class DynamoConformanceTests(unittest.TestCase):
    """Runs all 11/11 `core/livelab/conformance` cases on `ToyDynamoLab`."""

    def test_full_conformance_c0_to_c10_cpu(self) -> None:
        failures = []
        for name in cases.CASES:
            result = cases.run(name, ToyDynamoLab)
            if result["result"] != "pass":
                failures.append((name, result["result"], result["error"]))
        self.assertEqual(failures, [], f"conformance failures: {failures}")


class DynamoGoldenTests(unittest.TestCase):
    """Golden tests for multi-shape planner swap, frontend micro-restart, and fail-closed guards."""

    def setUp(self) -> None:
        self.lab = ToyDynamoLab(num_workers=2)

    def tearDown(self) -> None:
        self.lab.close()

    def test_planner_live_swap_two_shapes_fresh_parity_and_rollback(self) -> None:
        base_fp = self.lab.status()["fingerprint"]
        shape_a = {"prompts": ["The capital of France is"], "max_new_tokens": 4, "policy": "kv_aware"}
        shape_b = {"prompts": ["Alpha beta gamma", "Delta epsilon"], "max_new_tokens": 6, "policy": "round_robin"}
        base_a = self.lab.check(shape_a)
        base_b = self.lab.check(shape_b)
        self.assertTrue(base_a["ok"] and base_b["ok"])

        patch = self.lab.patches()["visible"]
        ins = self.lab.install(patch)
        self.assertTrue(ins["ok"], ins)

        ed_a = self.lab.check(shape_a)
        ed_b = self.lab.check(shape_b)
        self.assertIs(ed_a["proven"], True)
        self.assertIs(ed_b["proven"], True)
        self.assertNotEqual(ed_a["output"], base_a["output"])
        self.assertNotEqual(ed_b["output"], base_b["output"])
        self.assertEqual(self.lab.fresh_check(patch, shape_a)["output"], ed_a["output"])
        self.assertEqual(self.lab.fresh_check(patch, shape_b)["output"], ed_b["output"])

        undo = self.lab.uninstall()
        self.assertTrue(undo["ok"], undo)
        self.assertEqual(self.lab.status()["fingerprint"], base_fp)
        self.assertEqual(self.lab.check(shape_a)["output"], base_a["output"])

    def test_frontend_micro_restart_preserves_workers_and_parity(self) -> None:
        base_out = self.lab.check()["output"]
        res = self.lab.restart_frontend()
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["worker_pids_unchanged"])
        self.assertEqual(res["weight_reload_count"], 0)
        self.assertEqual(self.lab.check()["output"], base_out)

    def test_refuses_constructor_edit_and_rejecting_weight_store(self) -> None:
        p_mod = DEFAULT_MODULES[0]
        ctor_patch = harness.edit_once(self.lab._base_source(p_mod), ", init_routing_scale()", ", 9.0")
        refused = self.lab.install({"files": {p_mod: ctor_patch}})
        self.assertFalse(refused["ok"])
        self.assertIn("constructor_edit", refused["codes"])

        class _OfflineStore:
            name = "test-gms"

            def probe(self, *, tp=None, visible_devices=None):
                return StoreProbe(ok=False, backend=self.name, error="gms offline")

        with self.assertRaisesRegex(RuntimeError, "gms offline"):
            DynamoLab({"mode": "real", "model_path": "dummy"}, weight_store=_OfflineStore(), auto_spawn=True)

    def test_lazy_imports_without_torch_or_gms(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("gpu_memory_service", sys.modules)
        env = DynamoLab.launch_environment(["--model", "test-model", "--num-workers", "2", "--gms-read-only"])
        self.assertEqual(env["RSIBOX_DYNAMO_NUM_WORKERS"], "2")
        self.assertEqual(env["RSIBOX_DYNAMO_GMS_READ_ONLY"], "1")
        self.assertTrue(os.path.isdir(os.path.dirname(__file__)))

    def test_stop_engine_and_run_fresh_reap_on_shutdown_errors(self) -> None:
        sleep_a = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        sleep_b = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with tempfile.TemporaryDirectory() as tmp:
            sock_a, spec_a, err_a = (os.path.join(tmp, f) for f in ("a.sock", "a.json", "a.log"))
            for p in (sock_a, spec_a, err_a):
                with open(p, "w", encoding="utf-8"):
                    pass
            lab = DynamoLab({"mode": "toy", "frontend_mode": "python"}, auto_spawn=False)
            lab._workers = {"worker0": {"proc": sleep_a, "sock": sock_a, "spec": spec_a, "err": err_a, "device_idx": 0}}
            with mock.patch("adapter.worker.ipc.unix_json_call", side_effect=FileNotFoundError("no sock")):
                lab._stop_engine()
            self.assertIsNotNone(sleep_a.poll())
            self.assertFalse(os.path.exists(sock_a))
            with mock.patch("adapter.worker.ipc.unix_json_call", side_effect=TimeoutError("hung")):
                worker_mod._stop_proc(sleep_b, os.path.join(tmp, "b.sock"), clean=True)
            self.assertIsNotNone(sleep_b.poll())

    def test_frontend_driver_start_failure_cleans_up_resources(self) -> None:
        with mock.patch("shutil.which", return_value=None), mock.patch("os.path.isfile", return_value=False):
            fe = FrontendDriver(lambda r: {"ok": True}, mode="rust")
            with self.assertRaisesRegex(RuntimeError, "rustc not found"):
                fe.start()
            self.assertIsNone(fe._be_http)
            self.assertIsNone(fe._be_thread)

        fe2 = FrontendDriver(lambda r: {"ok": True}, mode="rust")
        fake_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        with (
            mock.patch.object(fe2, "_ensure_rust_binary", return_value="/bin/true"),
            mock.patch("subprocess.Popen", return_value=fake_proc),
            mock.patch.object(fe2, "health", return_value={"ok": False}),
        ):
            with self.assertRaises(TimeoutError):
                fe2.start(timeout_s=0.05)
        self.assertIsNotNone(fake_proc.poll())
        self.assertIsNone(fe2._proc)
        self.assertIsNone(fe2._be_http)
        self.assertIsNone(fe2._be_thread)
        self.assertIsNone(fe2._owned_temp_dir)

    def test_remapped_visible_devices_populates_uuid_map_and_fails_closed(self) -> None:
        lab = DynamoLab({"mode": "real", "model_path": "dummy", "num_workers": 2}, auto_spawn=False)
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}),
            mock.patch("rsibox.warm.dynamo_gms.resolve_target_gpus", return_value=[(2, "GPU-aaa"), (3, "GPU-bbb")]),
        ):
            env = lab._worker_env()
            self.assertEqual(env["RSIBOX_GMS_DEVICE_UUID_MAP"], '{"0": "GPU-aaa", "1": "GPU-bbb"}')
            with mock.patch("adapter.lab.harness.run_fresh_subprocess", return_value={"ok": True, "output": {}}) as m_fresh:
                lab.fresh_check(None)
                self.assertEqual(m_fresh.call_args.kwargs["env"]["RSIBOX_GMS_DEVICE_UUID_MAP"], '{"0": "GPU-aaa", "1": "GPU-bbb"}')

        seen_kwargs: dict[str, object] = {}
        class _FailStore:
            def __init__(self, **kw: object) -> None:
                seen_kwargs.update(kw)
            socket_dir = "/tmp"
            def ensure_ready(self, **kw: object) -> StoreProbe:
                return StoreProbe(ok=False, backend="dynamo_gms", error="remapped GPU refused")

        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}),
            mock.patch("rsibox.warm.dynamo_gms.resolve_target_gpus", return_value=None),
            mock.patch("rsibox.warm.dynamo_gms.DynamoGmsStore", _FailStore),
        ):
            with self.assertRaisesRegex(RuntimeError, "remapped GPU refused"):
                lab._ensure_gms_store()
        self.assertIs(seen_kwargs["allow_remapped_visible_devices"], False)

    def test_three_worker_conformance_c5_and_c8(self) -> None:
        lab3 = ToyDynamoLab(num_workers=3)
        try:
            wl = lab3.default_workload()
            workers = ["worker0", "worker1", "worker2"]
            routed = {planner_ops.select_worker(p, workers, None, policy=wl["policy"], step_idx=i)["worker"] for i, p in enumerate(wl["prompts"])}
            self.assertEqual(routed, set(workers))
        finally:
            lab3.close()
        for cname in ("c5_failed_install", "c8_crash_and_hang"):
            res = cases.run(cname, lambda receipts=None: ToyDynamoLab(receipts=receipts, num_workers=3))
            self.assertEqual(res["result"], "pass", res)


if __name__ == "__main__":
    unittest.main()
