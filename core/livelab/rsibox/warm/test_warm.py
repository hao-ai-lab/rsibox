"""Unit tests for `rsibox.warm` (pluggable WeightStore, Dynamo GMS, RecoveryDriver, Snapshot contract).

Pure CPU / stdlib tests — no GPU, torch, vLLM, or Dynamo install required.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

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
from rsibox.warm.dynamo_gms import (  # noqa: E402
    _GET_RUNTIME_STATE_FRAME,
    _decode_msgpack,
    _probe_gms_socket,
    long_socket_path_finding,
    resolve_target_gpus,
    socket_name,
)


def _encode_runtime_state_reply(
    state: str, committed: bool, msg_type: str = "get_runtime_state_response"
) -> bytes:
    """Minimal MsgPack frame matching `GetRuntimeStateResponse` (`type`, `state`, `committed`)."""
    tp_b = msg_type.encode("utf-8")
    st_b = state.encode("utf-8")
    payload = (
        b"\x83"
        + b"\xa4type"
        + bytes([0xA0 | len(tp_b)]) + tp_b
        + b"\xa5state"
        + bytes([0xA0 | len(st_b)]) + st_b
        + b"\xa9committed"
        + (b"\xc3" if committed else b"\xc2")
    )
    return struct.pack("!I", len(payload)) + payload


class _UnixSocketGuard:
    """AF_UNIX listener simulating either a bare socket or a GMSRPCServer runtime-state probe."""

    def __init__(self, path: str, *, gms_state: tuple[str, bool] | None = None):
        self.path = path
        self.gms_state = gms_state
        self.received_frames: list[bytes] = []
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
            with conn:
                if self.gms_state is not None:
                    conn.settimeout(0.5)
                    try:
                        data = conn.recv(64)
                        if data:
                            self.received_frames.append(data)
                            if data == _GET_RUNTIME_STATE_FRAME:
                                conn.sendall(_encode_runtime_state_reply(*self.gms_state))
                    except OSError:
                        pass

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

    def test_resolve_target_gpus_handles_empty_visible_uuids_and_tp(self):
        nvml = [(0, "GPU-000"), (4, "GPU-444"), (5, "GPU-555")]
        # CUDA_VISIBLE_DEVICES="" means zero GPUs (never fall back to all host GPUs).
        self.assertIsNone(resolve_target_gpus(visible_devices="", nvml_query=lambda: nvml))
        # Physical indices and UUIDs map cleanly to (phys_idx, uuid).
        self.assertEqual(
            resolve_target_gpus(visible_devices="4,5", tp=2, nvml_query=lambda: nvml),
            [(4, "GPU-444"), (5, "GPU-555")],
        )
        self.assertEqual(
            resolve_target_gpus(gpus=["GPU-555"], nvml_query=lambda: nvml),
            [(5, "GPU-555")],
        )
        # Unset visible_devices slices to tp.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            self.assertEqual(
                resolve_target_gpus(tp=2, nvml_query=lambda: nvml),
                [(0, "GPU-000"), (4, "GPU-444")],
            )
            self.assertIsNone(resolve_target_gpus(tp=4, nvml_query=lambda: nvml))

    def test_bare_metal_nonzero_cuda_visible_devices_emits_mismatch_finding(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            store = DynamoGmsStore(
                socket_dir=tmp,
                target_resolver=lambda *a, **k: [(4, "GPU-444")],
            )
            pr = store.probe(visible_devices="4")
            self.assertFalse(pr.ok)
            self.assertEqual(pr.finding["check"], "gms_v0_visible_devices_mismatch")

            # Container-isolated callers can explicitly opt in.
            store_ok = DynamoGmsStore(
                socket_dir=tmp,
                allow_remapped_visible_devices=True,
                target_resolver=lambda *a, **k: [(4, "GPU-444")],
            )
            pr2 = store_ok.probe(visible_devices="4")
            self.assertEqual(pr2.finding["check"], "dynamo_gms_daemon")

    def test_probe_uses_msgpack_runtime_state_and_fails_closed_on_uncommitted_ro_and_missing_stamp(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            uuid = "GPU-11111111-2222-3333-4444-555555555555"
            w_guard = _UnixSocketGuard(
                os.path.join(tmp, socket_name(uuid, "weights")), gms_state=("empty", False)
            )
            kv_guard = _UnixSocketGuard(
                os.path.join(tmp, socket_name(uuid, "kv_cache")), gms_state=("empty", False)
            )
            try:
                # RW_OR_RO store succeeds against an EMPTY GMS server with zero wire errors.
                rw_store = DynamoGmsStore(socket_dir=tmp, uuid_resolver=lambda *a, **k: [uuid])
                pr_rw = rw_store.probe(gpus=[uuid])
                self.assertTrue(pr_rw.ok, pr_rw.to_dict())
                self.assertIn(_GET_RUNTIME_STATE_FRAME, w_guard.received_frames)

                # RO store fails closed when weights socket reports committed=False!
                ro_store = DynamoGmsStore(
                    socket_dir=tmp, read_only=True, uuid_resolver=lambda *a, **k: [uuid]
                )
                pr_ro_empty = ro_store.probe(gpus=[uuid])
                self.assertFalse(pr_ro_empty.ok)
                self.assertEqual(pr_ro_empty.finding["check"], "dynamo_gms_not_committed")

                # Once committed, RO probe passes.
                w_guard.gms_state = ("committed", True)
                self.assertTrue(ro_store.probe(gpus=[uuid]).ok)

                # ErrorResponse fails closed under read_only=True even if committed is absent/True.
                w_guard.gms_state = ("error", True, "error_response")
                pr_err = ro_store.probe(gpus=[uuid])
                self.assertFalse(pr_err.ok)
                self.assertEqual(pr_err.finding["check"], "dynamo_gms_not_committed")
                w_guard.gms_state = ("committed", True)

                # Missing stamp must FAIL CLOSED when expected_stamp is requested.
                stamp = {"files": {"model": "abc"}, "weights": {}, "env": {}, "extra": None}
                pr_missing_stamp = rw_store.probe(gpus=[uuid], expected_stamp=stamp)
                self.assertFalse(pr_missing_stamp.ok)
                self.assertEqual(pr_missing_stamp.stale_keys, ("stamp_missing",))
                self.assertEqual(pr_missing_stamp.finding["check"], "dynamo_gms_stale")

                # Writing matching stamp passes; drifting stamp fails closed.
                rw_store.write_stamp(stamp)
                self.assertTrue(rw_store.probe(gpus=[uuid], expected_stamp=stamp).ok)
                drifted = {"files": {"model": "xyz"}, "weights": {}, "env": {}, "extra": None}
                stale_pr = rw_store.probe(gpus=[uuid], expected_stamp=drifted)
                self.assertFalse(stale_pr.ok)
                self.assertEqual(stale_pr.stale_keys, ("files:model",))
            finally:
                w_guard.close()
                kv_guard.close()

    def test_probe_reports_missing_sockets_cleanly(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            uuid = "GPU-11111111-2222-3333-4444-555555555555"
            store = DynamoGmsStore(socket_dir=tmp, uuid_resolver=lambda *a, **k: [uuid])
            pr = store.probe(gpus=[uuid])
            self.assertFalse(pr.ok)
            self.assertEqual(len(pr.missing), 2)
            self.assertEqual(pr.finding["check"], "dynamo_gms_daemon")

    def test_stop_reaps_owned_children_promptly_without_zombie_stall(self):
        store = DynamoGmsStore(socket_dir="/tmp")
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        store._owned_procs.append(child)
        t0 = time.monotonic()
        store.stop()
        elapsed = time.monotonic() - t0
        self.assertIsNotNone(child.poll(), "child process was not reaped")
        self.assertLess(elapsed, 2.0, f"stop() stalled for {elapsed:.2f}s waiting on zombie")

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

        # Explicit conflicting load_format, worker_cls, and model_loader_extra_config emit findings.
        conflict = store.launch_delta(
            engine="vllm",
            explicit_flags={"load_format", "worker_cls", "model_loader_extra_config"},
            explicit_values={
                "load_format": "ipc_cache",
                "worker_cls": "kvplan.KVPlanWorker",
                "model_loader_extra_config": {"gms_read_only": False},
            },
        )
        self.assertFalse(conflict.ok)
        checks = {f["check"] for f in conflict.findings}
        self.assertEqual(checks, {"gms_load_format", "gms_worker_cls", "gms_loader_extra_config"})

        # Explicit non-conflicting user extra config merges with GMS keys cleanly.
        merged_delta = store.launch_delta(
            engine="vllm",
            explicit_flags={"model_loader_extra_config"},
            explicit_values={"model_loader_extra_config": {"foo": 1, "gms_read_only": True}},
        )
        self.assertTrue(merged_delta.ok, merged_delta.findings)
        m_idx = merged_delta.args.index("--model-loader-extra-config")
        self.assertEqual(
            json.loads(merged_delta.args[m_idx + 1]),
            {"foo": 1, "gms_read_only": True, "gms_ro_connect_timeout_ms": 5000},
        )

        # Malformed non-dict JSON in model_loader_extra_config emits finding without crashing.
        bad_json_delta = store.launch_delta(
            engine="vllm",
            explicit_flags={"model_loader_extra_config"},
            explicit_values={"model_loader_extra_config": "[1, 2]"},
        )
        self.assertFalse(bad_json_delta.ok)
        self.assertEqual([f["check"] for f in bad_json_delta.findings], ["gms_loader_extra_config"])

    def test_hung_socket_error_response_ext_msgpack_and_partial_tag_spawn(self):
        # 1. Hung/SIGSTOP'd listening Unix socket (never replies) fails probe() on timeout.
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            uuid = "GPU-11111111-2222-3333-4444-555555555555"
            w_path = os.path.join(tmp, socket_name(uuid, "weights"))
            kv_path = os.path.join(tmp, socket_name(uuid, "kv_cache"))
            hung = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            hung.bind(w_path)
            hung.listen(4)
            kv_guard = _UnixSocketGuard(kv_path, gms_state=("empty", False))
            try:
                alive, state = _probe_gms_socket(w_path, timeout_s=0.05)
                self.assertFalse(alive)
                self.assertIsNone(state)
                store_ro = DynamoGmsStore(
                    socket_dir=tmp, read_only=True, target_resolver=lambda *a, **k: [(0, uuid)]
                )
                pr_hung = store_ro.probe(gpus=[uuid], connect_timeout_s=0.05)
                self.assertFalse(pr_hung.ok)
                self.assertEqual(pr_hung.finding["check"], "dynamo_gms_daemon")
            finally:
                hung.close()
                kv_guard.close()

        # 2. MsgPack fixext/ext decoding + truncated bounds check + malformed stamp_diff.
        # Map of 2 entries: {"ts": fixext1(type=-1, 0x2a), "committed": True}
        payload = b"\x82\xa2ts\xd4\xff\x2a\xa9committed\xc3"
        decoded, pos = _decode_msgpack(payload, 0)
        self.assertEqual(pos, len(payload))
        self.assertEqual(decoded["committed"], True)
        self.assertEqual(decoded["ts"], (-1, b"\x2a"))
        with self.assertRaises(ValueError):
            _decode_msgpack(b"\xa5abc", 0)
        self.assertEqual(stamp_diff({"files": "corrupt"}, {"files": {"a": "1"}}), ("files:invalid",))
        self.assertEqual(stamp_diff("bad", {"files": {}}), ("stamp_invalid",))

        # 3. _gms_server_env strips BOTH CUDA_VISIBLE_DEVICES and DYN_GMS_USE_V1.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4", "DYN_GMS_USE_V1": "true"}):
            env = DynamoGmsStore(socket_dir="/tmp/gms")._gms_server_env()
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            self.assertNotIn("DYN_GMS_USE_V1", env)

        # 4. ensure_ready() spawns ONLY the missing tag (kv_cache) when weights is already live.
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            uuid = "GPU-22222222-2222-3333-4444-555555555555"
            w_guard = _UnixSocketGuard(os.path.join(tmp, socket_name(uuid, "weights")), gms_state=("committed", True))
            spawned_cmds: list[list[str]] = []
            try:
                store = DynamoGmsStore(
                    socket_dir=tmp,
                    target_resolver=lambda *a, **k: [(0, uuid), (0, uuid)],
                )

                class _FakePopen:
                    pid = 999999
                    def poll(self):
                        return 0
                    def wait(self, timeout=None):
                        return 0
                    def kill(self):
                        return None

                def fake_popen(cmd, **kw):
                    spawned_cmds.append(list(cmd))
                    return _FakePopen()

                with mock.patch("subprocess.Popen", side_effect=fake_popen):
                    store.ensure_ready(gpus=["0"], timeout_s=0.1)
                self.assertEqual(len(spawned_cmds), 1)
                cmd = spawned_cmds[0]
                self.assertIn("kv_cache", cmd)
                self.assertNotIn("weights", cmd)
            finally:
                w_guard.close()

    def test_launch_delta_sglang_and_fastvideo_and_unsupported(self):
        store = DynamoGmsStore(socket_dir="/tmp/gms-test", read_only=False)
        sgl = store.launch_delta(engine="sglang")
        self.assertTrue(sgl.ok)
        self.assertEqual(sgl.args, ())

        fv = store.launch_delta(engine="fastvideo")
        self.assertTrue(fv.ok)
        self.assertEqual(fv.env["RSIBOX_GMS_MODE"], "rw_or_ro")

        bad = store.launch_delta(engine="megatron")
        self.assertFalse(bad.ok)
        self.assertEqual(bad.findings[0]["check"], "gms_unsupported_engine")


class SnapshotContractAndRecoveryTests(unittest.TestCase):
    def test_snapshot_lifecycle_clears_stale_restore_complete_and_succeeds(self):
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

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="snap-") as tmp:
            ctrl = Path(tmp)
            # Plant stale sentinels (including restore-complete!) to verify Step 1 clears them.
            (ctrl / "ready-for-snapshot").write_text("stale", encoding="utf-8")
            (ctrl / "restore-complete").write_text("stale", encoding="utf-8")

            def agent_side():
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if (ctrl / "ready-for-snapshot").exists() and "quiesce" in events:
                        # Stale restore-complete must have been removed before waiting!
                        assert not (ctrl / "restore-complete").exists()
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

    def test_snapshot_lifecycle_cleans_ready_sentinel_on_timeout_and_rejects_bad_name(self):
        class _OkHooks:
            mutability = "readonly_weights"

            def warmup_for_snapshot(self) -> None:
                return None

            def quiesce_for_snapshot(self) -> QuiesceReceipt:
                return QuiesceReceipt(ok=True)

            def resume_after_restore(self, receipt: QuiesceReceipt) -> dict:
                return {"ok": True}

        with self.assertRaises(ValueError):
            DynamoK8sSnapshotDriver(engine_name="../escape")

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="snap-") as tmp:
            res = run_snapshot_lifecycle(
                _OkHooks(), engine_name="vllm", control_dir=tmp, wait_timeout_s=0.05, poll_s=0.01
            )
            self.assertFalse(res.ok)
            self.assertFalse((Path(tmp) / "ready-for-snapshot").exists())

    def test_snapshot_lifecycle_fails_closed_when_quiesce_fails(self):
        class _FailingHooks:
            mutability = "mutable_step_state"

            def warmup_for_snapshot(self) -> None:
                return None

            def quiesce_for_snapshot(self) -> QuiesceReceipt:
                return QuiesceReceipt(ok=False, error="in-flight collective did not drain")

            def resume_after_restore(self, receipt: QuiesceReceipt) -> dict:
                raise AssertionError("resume must not run when quiesce fails")

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="snap-") as tmp:
            res = run_snapshot_lifecycle(_FailingHooks(), engine_name="megatron", control_dir=tmp)
            self.assertFalse(res.ok)
            self.assertFalse((Path(tmp) / "ready-for-snapshot").exists())
            self.assertFalse((Path(tmp) / "megatron-restore-ready").exists())

    def test_base_lab_recovery_drivers_fail_closed_and_skip_replenish_on_failed_promote(self):
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

        # 1. StandbyShadowDriver: failed promote skips replenish & handles reserved keys safely.
        replenished: list[str] = []
        bad_driver = StandbyShadowDriver(
            promote_fn=lambda lab, store: {"ok": False, "driver": "override", "reason": "bad"},
            replenish_fn=lambda lab, store: replenished.append("called"),
        )
        lab = _DummyLab(weight_store=DiskStore(), recovery_driver=bad_driver)
        res_bad = lab.reset("respawn")
        self.assertFalse(res_bad["ok"])
        self.assertEqual(res_bad["driver"], "standby_shadow")
        self.assertEqual(replenished, [])

        # 2. StandbyShadowDriver: successful promote invokes replenish.
        good_driver = StandbyShadowDriver(
            promote_fn=lambda lab, store: (lab._spawn(), {"ok": True, "reset": ["shadow promoted"]})[1],
            replenish_fn=lambda lab, store: replenished.append("ok"),
        )
        lab.recovery_driver = good_driver
        res_good = lab.reset("respawn")
        self.assertTrue(res_good["ok"])
        self.assertEqual(replenished, ["ok"])

        # 3. DefaultRespawnDriver: fails closed when WeightStore probe fails (does NOT respawn).
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            broken_store = DynamoGmsStore(
                socket_dir=tmp, uuid_resolver=lambda *a, **k: ["GPU-00000000-0000-0000-0000-000000000000"]
            )
            lab.weight_store = broken_store
            lab.recovery_driver = DefaultRespawnDriver()
            before = list(calls)
            res_fail = lab.reset("respawn")
            self.assertFalse(res_fail["ok"])
            self.assertEqual(calls, before, "DefaultRespawnDriver must not stop/spawn on failed store probe")

        # 4. DefaultRespawnDriver resolves SglangLab-shaped TP (`config['tp_size']`) & visible_devices.
        probed_kwargs: list[dict] = []

        class _SpyStore(DiskStore):
            name = "spy"

            def probe(self, **kw):
                probed_kwargs.append(dict(kw))
                return super().probe(**kw)

        lab.config = {"tp_size": 2}
        lab.weight_store = _SpyStore()
        lab.recovery_driver = DefaultRespawnDriver()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            res_tp = lab.reset("respawn")
        self.assertTrue(res_tp["ok"])
        self.assertEqual(probed_kwargs[-1]["tp"], 2)
        self.assertEqual(probed_kwargs[-1]["visible_devices"], "0,1")


if __name__ == "__main__":
    unittest.main()
