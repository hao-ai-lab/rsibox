"""Unit tests for `rsibox.warm` (pluggable WeightStore, Dynamo GMS, RecoveryDriver, Snapshot contract).

Pure CPU / stdlib tests — no GPU, torch, vLLM, or Dynamo install required.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

LIVELAB_DIR = Path(__file__).resolve().parents[2]
if str(LIVELAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVELAB_DIR))

from rsibox import BaseLab  # noqa: E402
from rsibox.warm import (  # noqa: E402
    GMS_TAGS,
    GMS_VLLM_LOAD_FORMAT,
    GMS_VLLM_WORKER_CLS,
    DefaultRespawnDriver,
    DiskStore,
    DynamoGmsStore,
    DynamoK8sSnapshotDriver,
    QuiesceReceipt,
    StandbyShadowDriver,
    get_weight_store,
    list_weight_stores,
    run_snapshot_lifecycle,
    stamp_diff,
)
from rsibox.warm.dynamo_gms import long_socket_path_finding, socket_name  # noqa: E402


class _UnixSocketGuard:
    """Minimal AF_UNIX listener for probing tests (accepts and closes cleanly)."""

    def __init__(self, path: str):
        self.path = path
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.bind(path)
            self._sock.listen(8)
            self._sock.settimeout(0.05)
        except Exception:
            self._sock.close()
            raise
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()
        if os.path.exists(self.path):
            os.unlink(self.path)


class WarmTypesAndRegistryTests(unittest.TestCase):
    def test_registry_contains_none_and_dynamo_gms(self):
        stores = list_weight_stores()
        self.assertIn("none", stores)
        self.assertIn("dynamo_gms", stores)
        self.assertIsInstance(get_weight_store("none"), DiskStore)
        self.assertIsInstance(get_weight_store("dynamo_gms", socket_dir="/tmp"), DynamoGmsStore)

    def test_stamp_diff_detects_loader_and_env_edits(self):
        base = {
            "files": {"loader": "aaa", "quant": "bbb"},
            "weights": {"model.safetensors": [100, 1]},
            "env": {"GMS_MODE": "ro"},
            "extra": None,
        }
        same = json.loads(json.dumps(base))
        self.assertEqual(stamp_diff(base, same), ())
        edited = json.loads(json.dumps(base))
        edited["files"]["loader"] = "ccc"
        edited["env"]["GMS_MODE"] = "rw"
        self.assertEqual(stamp_diff(base, edited), ("files:loader", "env:GMS_MODE"))


class DynamoGmsStoreTests(unittest.TestCase):
    def test_long_socket_path_rejected_before_start(self):
        long_dir = "/tmp/" + ("a" * 90)
        finding = long_socket_path_finding(long_dir)
        self.assertIsNotNone(finding)
        self.assertEqual(finding["check"], "gms_socket_path")

        store = DynamoGmsStore(socket_dir=long_dir, uuid_resolver=lambda *a, **k: ["GPU-1"])
        pr = store.probe()
        self.assertFalse(pr.ok)
        self.assertEqual(pr.finding["check"], "gms_socket_path")

    def test_probe_succeeds_when_all_gpu_tag_sockets_answer_and_stamp_matches(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            uuid = "GPU-11111111-2222-3333-4444-555555555555"
            guards = [_UnixSocketGuard(os.path.join(tmp, socket_name(uuid, tag))) for tag in GMS_TAGS]
            try:
                store = DynamoGmsStore(socket_dir=tmp, uuid_resolver=lambda *a, **k: [uuid])
                pr = store.probe(gpus=[uuid])
                self.assertTrue(pr.ok, pr.to_dict())
                self.assertEqual(len(pr.sockets), 2)

                stamp = {"files": {"model": "abc"}, "weights": {}, "env": {}, "extra": None}
                store.write_stamp(stamp)
                self.assertTrue(store.probe(gpus=[uuid], expected_stamp=stamp).ok)

                drifted = {"files": {"model": "xyz"}, "weights": {}, "env": {}, "extra": None}
                stale_pr = store.probe(gpus=[uuid], expected_stamp=drifted)
                self.assertFalse(stale_pr.ok)
                self.assertEqual(stale_pr.stale_keys, ("files:model",))
                self.assertEqual(stale_pr.finding["check"], "dynamo_gms_stale")
            finally:
                for g in guards:
                    g.close()

    def test_probe_reports_missing_sockets_cleanly(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            uuid = "GPU-11111111-2222-3333-4444-555555555555"
            store = DynamoGmsStore(socket_dir=tmp, uuid_resolver=lambda *a, **k: [uuid])
            pr = store.probe(gpus=[uuid])
            self.assertFalse(pr.ok)
            self.assertEqual(len(pr.missing), 2)
            self.assertEqual(pr.finding["check"], "dynamo_gms_daemon")

    def test_launch_delta_vllm_and_conflicts(self):
        store = DynamoGmsStore(
            socket_dir="/tmp/gms-test",
            read_only=True,
            ro_connect_timeout_ms=5000,
            scratch_kv=True,
        )
        delta = store.launch_delta(engine="vllm")
        self.assertTrue(delta.ok)
        self.assertIn("--load-format", delta.args)
        self.assertIn(GMS_VLLM_LOAD_FORMAT, delta.args)
        self.assertIn("--worker-cls", delta.args)
        self.assertIn(GMS_VLLM_WORKER_CLS, delta.args)
        self.assertEqual(delta.requires_worker_cls, GMS_VLLM_WORKER_CLS)
        self.assertEqual(delta.env["GMS_SOCKET_DIR"], os.path.abspath("/tmp/gms-test"))
        self.assertEqual(delta.env["DYN_GMS_SCRATCH_KV_ENABLED"], "1")
        cfg_idx = delta.args.index("--model-loader-extra-config")
        self.assertEqual(
            json.loads(delta.args[cfg_idx + 1]),
            {"gms_read_only": True, "gms_ro_connect_timeout_ms": 5000},
        )

        # Explicit conflicting load_format and worker_cls must emit structured findings.
        conflict = store.launch_delta(
            engine="vllm",
            explicit_flags={"load_format", "worker_cls"},
            explicit_values={"load_format": "ipc_cache", "worker_cls": "kvplan.KVPlanWorker"},
        )
        self.assertFalse(conflict.ok)
        checks = {f["check"] for f in conflict.findings}
        self.assertEqual(checks, {"gms_load_format", "gms_worker_cls"})

    def test_launch_delta_sglang_and_fastvideo_and_unsupported(self):
        store = DynamoGmsStore(socket_dir="/tmp/gms-test", read_only=False)
        sgl = store.launch_delta(engine="sglang")
        self.assertTrue(sgl.ok)
        self.assertEqual(sgl.args, ("--load-format", "gms"))

        fv = store.launch_delta(engine="fastvideo")
        self.assertTrue(fv.ok)
        self.assertEqual(fv.env["RSIBOX_GMS_MODE"], "rw_or_ro")

        bad = store.launch_delta(engine="megatron")
        self.assertFalse(bad.ok)
        self.assertEqual(bad.findings[0]["check"], "gms_unsupported_engine")


class SnapshotContractAndRecoveryTests(unittest.TestCase):
    def test_snapshot_lifecycle_happy_path(self):
        events: list[str] = []

        class _FakeHooks:
            mutability = "readonly_weights"

            def warmup_for_snapshot(self) -> None:
                events.append("warmup")

            def quiesce_for_snapshot(self) -> QuiesceReceipt:
                events.append("quiesce")
                return QuiesceReceipt(ok=True, released=("kv_cache", "gms_va"), token="tok-1")

            def resume_after_restore(self, receipt: QuiesceReceipt) -> dict:
                events.append(f"resume:{receipt.token}")
                return {"ok": True, "reattached": ["weights", "kv_cache"]}

        with tempfile.TemporaryDirectory(prefix="snap-") as tmp:
            ctrl = Path(tmp)
            # Plant a stale ready-for-snapshot to verify it is cleared before warmup.
            (ctrl / "ready-for-snapshot").write_text("stale", encoding="utf-8")

            def agent_side():
                # Wait for freshly written ready-for-snapshot after quiesce.
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if (ctrl / "ready-for-snapshot").exists() and "quiesce" in events:
                        (ctrl / "restore-complete").write_text("ok\n", encoding="utf-8")
                        return
                    time.sleep(0.01)

            t = threading.Thread(target=agent_side, daemon=True)
            t.start()
            driver = DynamoK8sSnapshotDriver(
                engine_name="fastvideo", control_dir=str(ctrl), wait_timeout_s=5.0, poll_s=0.01
            )
            res = driver.checkpoint_and_wait(_FakeHooks())
            t.join(timeout=2.0)

            self.assertTrue(res.ok, res.to_dict())
            self.assertEqual(events, ["warmup", "quiesce", "resume:tok-1"])
            self.assertTrue((ctrl / "fastvideo-restore-ready").exists())

    def test_snapshot_lifecycle_fails_closed_when_quiesce_fails(self):
        class _FailingHooks:
            mutability = "mutable_step_state"

            def warmup_for_snapshot(self) -> None:
                return None

            def quiesce_for_snapshot(self) -> QuiesceReceipt:
                return QuiesceReceipt(ok=False, error="in-flight collective did not drain")

            def resume_after_restore(self, receipt: QuiesceReceipt) -> dict:
                raise AssertionError("resume must not run when quiesce fails")

        with tempfile.TemporaryDirectory(prefix="snap-") as tmp:
            res = run_snapshot_lifecycle(_FailingHooks(), engine_name="megatron", control_dir=tmp)
            self.assertFalse(res.ok)
            self.assertFalse((Path(tmp) / "ready-for-snapshot").exists())
            self.assertFalse((Path(tmp) / "megatron-restore-ready").exists())

    def test_base_lab_delegates_respawn_to_recovery_driver(self):
        import contextlib
        from kernel import coordinator as C

        calls: list[str] = []

        class _DummyLab(BaseLab):
            name = "dummy"

            def _start_engine(self):
                calls.append("start")
                part = C.LocalParticipant("p")
                return C.StepGapCoordinator([part], contextlib.nullcontext), []

            def _stop_engine(self):
                calls.append("stop")

            def _run_workload(self, workload):
                return {"ok": True}

        promoted: list[str] = []

        def promote(lab, store):
            promoted.append(store.name if store else "none")
            lab._spawn()
            return {"ok": True, "reset": ["shadow promoted"]}

        driver = StandbyShadowDriver(promote_fn=promote)
        lab = _DummyLab(weight_store=DiskStore(), recovery_driver=driver)
        res = lab.reset("respawn")
        self.assertTrue(res["ok"])
        self.assertEqual(res["driver"], "standby_shadow")
        self.assertEqual(promoted, ["none"])
        self.assertEqual(calls, ["start", "stop", "start"])

        # DefaultRespawnDriver also works cleanly
        lab.recovery_driver = DefaultRespawnDriver()
        res2 = lab.reset("respawn")
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["driver"], "default_respawn")


if __name__ == "__main__":
    unittest.main()
