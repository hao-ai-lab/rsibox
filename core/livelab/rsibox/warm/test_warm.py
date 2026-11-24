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
    DEFAULT_MIN_NOFILE,
    ENV_GMS_MIN_NOFILE,
    GMS_TAGS,
    GMS_VLLM_LOAD_FORMAT,
    GMS_VLLM_WORKER_CLS,
    RECOMMENDED_NOFILE,
    AsyncShadowReplenisher,
    DefaultRespawnDriver,
    DiskStore,
    DynamoGmsStore,
    DynamoK8sSnapshotDriver,
    QuiesceReceipt,
    StandbyShadowDriver,
    get_weight_store,
    list_weight_stores,
    nofile_limit_finding,
    run_snapshot_lifecycle,
    stamp_diff,
)
from rsibox.warm.dynamo_gms import (  # noqa: E402
    ENV_GMS_DEVICE_UUID_MAP,
    _GET_RUNTIME_STATE_FRAME,
    _decode_msgpack,
    _probe_gms_socket,
    build_device_uuid_map,
    install_visible_devices_uuid_shim,
    long_socket_path_finding,
    resolve_target_gpus,
    socket_name,
    uninstall_visible_devices_uuid_shim,
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

    def test_visible_devices_uuid_shim_routes_logical_ordinals_and_survives_invalidate(self):
        import types

        fake_utils = types.ModuleType("gpu_memory_service.common.utils")
        fake_utils._uuid_cache = {}

        def orig_get_socket_path(device: int, tag: str = "weights") -> str:
            u = fake_utils._uuid_cache.get(device, f"WRONG-PHYS-{device}")
            return os.path.join(os.environ.get("GMS_SOCKET_DIR", "/tmp"), f"gms_{u}_{tag}.sock")

        def orig_invalidate() -> None:
            fake_utils._uuid_cache.clear()

        fake_utils.get_socket_path = orig_get_socket_path
        fake_utils.invalidate_uuid_cache = orig_invalidate
        fake_mods = {
            "gpu_memory_service": types.ModuleType("gpu_memory_service"),
            "gpu_memory_service.common": types.ModuleType("gpu_memory_service.common"),
            "gpu_memory_service.common.utils": fake_utils,
        }
        mapping = build_device_uuid_map([(4, "GPU-444"), (5, "GPU-555")])
        self.assertEqual(mapping, {"0": "GPU-444", "1": "GPU-555"})

        with mock.patch.dict(sys.modules, fake_mods), mock.patch.dict(
            os.environ, {"GMS_SOCKET_DIR": "/tmp/gms-test", ENV_GMS_DEVICE_UUID_MAP: json.dumps(mapping)}
        ):
            self.assertTrue(install_visible_devices_uuid_shim())
            want0 = os.path.join("/tmp/gms-test", socket_name("GPU-444", "weights"))
            want1_kv = os.path.join("/tmp/gms-test", socket_name("GPU-555", "kv_cache"))
            want1_w = os.path.join("/tmp/gms-test", socket_name("GPU-555", "weights"))
            # Both wrapped utils.get_socket_path AND pre-imported orig_get_socket_path resolve GPU-444/555!
            self.assertEqual(fake_utils.get_socket_path(0, "weights"), want0)
            self.assertEqual(orig_get_socket_path(1, "kv_cache"), want1_kv)
            # Survives invalidate_uuid_cache()!
            fake_utils.invalidate_uuid_cache()
            self.assertEqual(fake_utils.get_socket_path(0, "weights"), want0)
            self.assertEqual(orig_get_socket_path(1, "weights"), want1_w)

            # Rejects JSON null/bool entries and evicts old keys on uninstall!
            self.assertFalse(install_visible_devices_uuid_shim({"0": None}))
            self.assertTrue(uninstall_visible_devices_uuid_shim())
            self.assertIs(fake_utils.get_socket_path, orig_get_socket_path)
            self.assertNotIn(0, fake_utils._uuid_cache)
            self.assertNotIn(1, fake_utils._uuid_cache)

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
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-det-") as tmp:
            store = DynamoGmsStore(socket_dir=tmp)
            # 1. Detached daemon spawned via _spawn_gms_daemon is reparented away from us
            #    (so SGLang's kill_process_tree(os.getpid()) cannot kill it) and stops cleanly.
            ppid_file = os.path.join(tmp, "ppid.txt")
            log_file = os.path.join(tmp, "daemon.log")
            daemon_pid = store._spawn_gms_daemon(
                [
                    sys.executable,
                    "-c",
                    f"import os, time; open({ppid_file!r}, 'w').write(str(os.getppid())); time.sleep(60)",
                ],
                os.environ.copy(),
                log_file,
            )
            store._owned_pids.append(daemon_pid)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not os.path.exists(ppid_file):
                time.sleep(0.02)
            self.assertTrue(os.path.exists(ppid_file))
            self.assertNotEqual(int(Path(ppid_file).read_text().strip()), os.getpid())

            # 2. Direct Popen fallback also reaps cleanly without zombie stall.
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

        # 3. _gms_server_env strips CUDA_VISIBLE_DEVICES, DYN_GMS_USE_V1, and RSIBOX_GMS_DEVICE_UUID_MAP.
        with mock.patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "4",
                "DYN_GMS_USE_V1": "true",
                ENV_GMS_DEVICE_UUID_MAP: '{"0":"GPU-444"}',
            },
        ):
            env = DynamoGmsStore(socket_dir="/tmp/gms")._gms_server_env()
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            self.assertNotIn("DYN_GMS_USE_V1", env)
            self.assertNotIn(ENV_GMS_DEVICE_UUID_MAP, env)

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

                def fake_spawn(cmd, env, log_path):
                    spawned_cmds.append(list(cmd))
                    return 999999

                with mock.patch.object(store, "_spawn_gms_daemon", side_effect=fake_spawn):
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

    def test_nofile_preflight_fails_closed_on_low_cap_and_blocks_daemon_spawn(self):
        """Golden case: RLIMIT_NOFILE preflight is strictly read-only in probe() and blocks unsafe spawns.

        Proves `probe()` never calls `set_nofile`, `nofile_limit_finding(..., auto_raise=True)`
        never mutates limits when `hard_limit < min_nofile`, `ensure_ready()` spawns zero daemons
        on low hard caps, `soft > hard` and missing `set_nofile` fail closed cleanly, `min_nofile > 65536`
        clamps the recommended ulimit, and `RSIBOX_GMS_MIN_NOFILE=0` disables the gate.
        """
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-nofile-") as temp_dir:
            gpu_uuid = "GPU-33333333-2222-3333-4444-555555555555"
            raised_calls: list[tuple[int, int]] = []
            spawned_daemons: list[list[str]] = []

            # 1. probe() on raisable (1024, 1048576): non-blocking, strictly read-only, attaches advisory!
            readonly_store = DynamoGmsStore(
                socket_dir=temp_dir,
                min_nofile=DEFAULT_MIN_NOFILE,
                auto_raise_nofile=True,
                get_nofile=lambda: (1024, 1048576),
                set_nofile=lambda soft, hard: raised_calls.append((soft, hard)),
                target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
            )
            probe_raisable = readonly_store.probe(gpus=[gpu_uuid])
            self.assertFalse(probe_raisable.ok)
            self.assertEqual(probe_raisable.finding["check"], "dynamo_gms_daemon")
            self.assertEqual(raised_calls, [], "probe() must never mutate RLIMIT_NOFILE")
            self.assertIsNotNone(probe_raisable.details)
            self.assertEqual(probe_raisable.details["nofile_advisory"]["soft"], 1024)
            self.assertEqual(probe_raisable.details["nofile_advisory"]["hard"], 1048576)
            self.assertEqual(probe_raisable.details["nofile_advisory"]["recommended"], RECOMMENDED_NOFILE)

            # ensure_ready() with auto_raise_nofile=False fails closed on raisable shortfall!
            strict_ensure_store = DynamoGmsStore(
                socket_dir=temp_dir,
                min_nofile=DEFAULT_MIN_NOFILE,
                auto_raise_nofile=False,
                get_nofile=lambda: (1024, 1048576),
                set_nofile=lambda soft, hard: raised_calls.append((soft, hard)),
                target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
            )
            strict_ensure = strict_ensure_store.ensure_ready(gpus=[gpu_uuid], timeout_s=0.1)
            self.assertFalse(strict_ensure.ok)
            self.assertEqual(strict_ensure.finding["check"], "gms_nofile_limit")
            self.assertEqual(raised_calls, [])
            self.assertTrue(
                any(f"{ENV_GMS_MIN_NOFILE}=0" in action for action in strict_ensure.finding["actions"])
            )

            # ensure_ready() checks long_socket_path_finding BEFORE prepare_nofile() (zero set_nofile on bad path).
            bad_path_store = DynamoGmsStore(
                socket_dir="/tmp/" + ("a" * 90),
                min_nofile=DEFAULT_MIN_NOFILE,
                auto_raise_nofile=True,
                get_nofile=lambda: (1024, 1048576),
                set_nofile=lambda soft, hard: raised_calls.append((soft, hard)),
            )
            bad_path_ensure = bad_path_store.ensure_ready(gpus=[gpu_uuid], timeout_s=0.1)
            self.assertFalse(bad_path_ensure.ok)
            self.assertEqual(bad_path_ensure.finding["check"], "gms_socket_path")
            self.assertEqual(raised_calls, [], "ensure_ready must not mutate rlimit when socket_dir is invalid")

            # 2. Hard cap == 1024 or 2048 (< 4096): probe() & ensure_ready() fail closed, NEVER mutate rlimit.
            for hard_cap in (1024, 2048):
                raised_calls.clear()
                finding = nofile_limit_finding(
                    4096,
                    auto_raise=True,
                    get_nofile=lambda cap=hard_cap: (1024, cap),
                    set_nofile=lambda soft, hard: raised_calls.append((soft, hard)),
                )
                self.assertIsNotNone(finding)
                self.assertEqual(finding["check"], "gms_nofile_limit")
                self.assertEqual(raised_calls, [], "must not call set_nofile when hard < min_nofile")

            capped_store = DynamoGmsStore(
                socket_dir=temp_dir,
                min_nofile=DEFAULT_MIN_NOFILE,
                auto_raise_nofile=True,
                get_nofile=lambda: (1024, 1024),
                set_nofile=lambda soft, hard: raised_calls.append((soft, hard)),
                target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
            )
            probe_capped = capped_store.probe(gpus=[gpu_uuid])
            self.assertFalse(probe_capped.ok)
            self.assertEqual(probe_capped.finding["check"], "gms_nofile_limit")
            with mock.patch.object(
                capped_store,
                "_spawn_gms_daemon",
                side_effect=lambda cmd, env, log_path: spawned_daemons.append(list(cmd)) or 999999,
            ):
                ensure_result = capped_store.ensure_ready(gpus=[gpu_uuid], timeout_s=0.1)
            self.assertFalse(ensure_result.ok)
            self.assertEqual(ensure_result.finding["check"], "gms_nofile_limit")
            self.assertEqual(spawned_daemons, [])
            self.assertEqual(raised_calls, [])

            # 3. Custom get_nofile without set_nofile, soft > hard, and min_nofile > 65536 recommendation clamp.
            no_setter = nofile_limit_finding(4096, auto_raise=True, get_nofile=lambda: (1024, 1048576))
            self.assertIsNotNone(no_setter)
            self.assertIn("set_nofile hook not provided", no_setter["observed"])

            invalid_soft_gt_hard = nofile_limit_finding(
                4096,
                auto_raise=True,
                get_nofile=lambda: (8192, 1024),
                set_nofile=lambda soft, hard: raised_calls.append((soft, hard)),
            )
            self.assertIsNotNone(invalid_soft_gt_hard)
            self.assertIn("invalid: soft > hard", invalid_soft_gt_hard["observed"])
            self.assertEqual(raised_calls, [])

            high_min_finding = nofile_limit_finding(131072, auto_raise=False, get_nofile=lambda: (65536, 1048576))
            self.assertIsNotNone(high_min_finding)
            self.assertTrue(
                any("--ulimit nofile=131072:131072" in action for action in high_min_finding["actions"])
            )

            # 4. RSIBOX_GMS_MIN_NOFILE env override and validation.
            weights_guard = _UnixSocketGuard(
                os.path.join(temp_dir, socket_name(gpu_uuid, "weights")), gms_state=("committed", True)
            )
            kv_guard = _UnixSocketGuard(
                os.path.join(temp_dir, socket_name(gpu_uuid, "kv_cache")), gms_state=("empty", False)
            )
            try:
                with mock.patch.dict(os.environ, {ENV_GMS_MIN_NOFILE: "0"}):
                    env_disabled = DynamoGmsStore(
                        socket_dir=temp_dir,
                        get_nofile=lambda: (1024, 1024),
                        target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
                    )
                    self.assertEqual(env_disabled.min_nofile, 0)
                    self.assertTrue(env_disabled.probe(gpus=[gpu_uuid]).ok)
                with mock.patch.dict(os.environ, {ENV_GMS_MIN_NOFILE: "8192"}):
                    self.assertEqual(DynamoGmsStore(socket_dir=temp_dir).min_nofile, 8192)
                    self.assertEqual(DynamoGmsStore(socket_dir=temp_dir, min_nofile=2048).min_nofile, 2048)
                with mock.patch.dict(os.environ, {ENV_GMS_MIN_NOFILE: "not-a-number"}):
                    with self.assertRaises(ValueError):
                        DynamoGmsStore(socket_dir=temp_dir)
                with mock.patch.dict(os.environ, {ENV_GMS_MIN_NOFILE: "-5"}):
                    with self.assertRaises(ValueError):
                        DynamoGmsStore(socket_dir=temp_dir)
            finally:
                weights_guard.close()
                kv_guard.close()

            with self.assertRaises(ValueError):
                DynamoGmsStore(socket_dir=temp_dir, min_nofile=-1)
            with self.assertRaises(ValueError):
                DynamoGmsStore(socket_dir=temp_dir, min_nofile=True)

    def test_nofile_preflight_auto_raises_in_ensure_ready_and_real_rlimit_subprocesses(self):
        """Golden case: prepare_nofile()/ensure_ready() lifts soft limit when hard >= min_nofile.

        Proves (a) ensure_ready() raises soft=1024 -> 4096 once and passes with live sockets,
        (b) RLIM_INFINITY and PermissionError behave cleanly, (c) real POSIX subprocess with
        (1024, high_hard) keeps probe() read-only while prepare_nofile() raises to 4096, and
        (d) real POSIX subprocess with (1024, 1024) fails closed by default and bypasses via
        RSIBOX_GMS_MIN_NOFILE=0.
        """
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-raise-") as temp_dir:
            gpu_uuid = "GPU-44444444-2222-3333-4444-555555555555"
            weights_guard = _UnixSocketGuard(
                os.path.join(temp_dir, socket_name(gpu_uuid, "weights")), gms_state=("committed", True)
            )
            kv_guard = _UnixSocketGuard(
                os.path.join(temp_dir, socket_name(gpu_uuid, "kv_cache")), gms_state=("empty", False)
            )
            try:
                current_limits = [1024, 1048576]
                applied_limits: list[tuple[int, int]] = []

                def apply_stateful(soft: int, hard: int) -> None:
                    applied_limits.append((soft, hard))
                    current_limits[0], current_limits[1] = soft, hard

                store = DynamoGmsStore(
                    socket_dir=temp_dir,
                    min_nofile=4096,
                    auto_raise_nofile=True,
                    get_nofile=lambda: (current_limits[0], current_limits[1]),
                    set_nofile=apply_stateful,
                    target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
                )
                # probe() on raisable (1024, 1048576) succeeds read-only with nofile_advisory!
                probe_before = store.probe(gpus=[gpu_uuid])
                self.assertTrue(probe_before.ok, probe_before.to_dict())
                self.assertEqual(applied_limits, [])
                self.assertEqual(probe_before.details["nofile_advisory"]["soft"], 1024)
                self.assertIn("details", probe_before.to_dict())

                # ensure_ready() raises soft limit once to min(hard, max(RECOMMENDED_NOFILE, min_nofile))!
                ready_probe = store.ensure_ready(gpus=[gpu_uuid], timeout_s=1.0)
                self.assertTrue(ready_probe.ok, ready_probe.to_dict())
                self.assertEqual(applied_limits, [(RECOMMENDED_NOFILE, 1048576)])
                # Subsequent probe() calls succeed cleanly with no advisory and zero further set_nofile calls.
                probe_after = store.probe(gpus=[gpu_uuid])
                self.assertTrue(probe_after.ok)
                self.assertIsNone(probe_after.details)
                self.assertEqual(len(applied_limits), 1)
            finally:
                weights_guard.close()
                kv_guard.close()

        # When hard is between min_nofile and RECOMMENDED_NOFILE (e.g. 8192), raise clamps to hard.
        clamped_calls: list[tuple[int, int]] = []
        self.assertIsNone(
            nofile_limit_finding(
                4096,
                auto_raise=True,
                get_nofile=lambda: (1024, 8192),
                set_nofile=lambda soft, hard: clamped_calls.append((soft, hard)),
            )
        )
        self.assertEqual(clamped_calls, [(8192, 8192)])

        # RLIM_INFINITY (-1 on Linux) as soft passes untouched; as hard allows raising to RECOMMENDED_NOFILE.
        infinity_calls: list[tuple[int, int]] = []
        self.assertIsNone(
            nofile_limit_finding(
                4096,
                auto_raise=True,
                get_nofile=lambda: (-1, -1),
                set_nofile=lambda soft, hard: infinity_calls.append((soft, hard)),
            )
        )
        self.assertEqual(infinity_calls, [])
        self.assertIsNone(
            nofile_limit_finding(
                4096,
                auto_raise=True,
                get_nofile=lambda: (1024, -1),
                set_nofile=lambda soft, hard: infinity_calls.append((soft, hard)),
            )
        )
        self.assertEqual(infinity_calls, [(RECOMMENDED_NOFILE, -1)])

        # PermissionError during set_nofile fails closed with the underlying OS error.
        def deny_raise(soft: int, hard: int) -> None:
            raise PermissionError("EPERM denied by seccomp")

        denied = nofile_limit_finding(
            4096,
            auto_raise=True,
            get_nofile=lambda: (1024, 1048576),
            set_nofile=deny_raise,
        )
        self.assertIsNotNone(denied)
        self.assertEqual(denied["check"], "gms_nofile_limit")
        self.assertIn("EPERM denied by seccomp", denied["observed"])

        # Real POSIX kernel test 1: (1024, high_hard) -> probe() stays ok with advisory & soft=1024;
        # prepare_nofile() raises soft to >= 4096.
        raise_script = "\n".join(
            [
                "import os, resource, socket, sys, tempfile, threading",
                f"sys.path.insert(0, {str(LIVELAB_DIR)!r})",
                "from rsibox.warm import DynamoGmsStore, RECOMMENDED_NOFILE",
                "from rsibox.warm.dynamo_gms import socket_name",
                "soft_orig, hard_orig = resource.getrlimit(resource.RLIMIT_NOFILE)",
                "if hard_orig != resource.RLIM_INFINITY and hard_orig < 4096:",
                "    print('SKIP_LOW_HARD')",
                "    raise SystemExit(0)",
                "resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard_orig))",
                "with tempfile.TemporaryDirectory(dir='/tmp', prefix='gms-sub-') as tmp:",
                "    uuid = 'GPU-00000000-0000-0000-0000-000000000001'",
                "    socks = []",
                "    for tag in ('weights', 'kv_cache'):",
                "        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)",
                "        sock.bind(os.path.join(tmp, socket_name(uuid, tag)))",
                "        sock.listen(4)",
                "        socks.append(sock)",
                "    stop_evt = threading.Event()",
                "    def accept_loop(listener):",
                "        listener.settimeout(0.05)",
                "        while not stop_evt.is_set():",
                "            try:",
                "                conn, _ = listener.accept()",
                "                conn.close()",
                "            except OSError:",
                "                continue",
                "    threads = [threading.Thread(target=accept_loop, args=(sock,), daemon=True) for sock in socks]",
                "    for thread in threads:",
                "        thread.start()",
                "    try:",
                "        store = DynamoGmsStore(socket_dir=tmp, min_nofile=4096, auto_raise_nofile=True, target_resolver=lambda *a, **k: [(0, uuid)])",
                "        probe_before = store.probe(gpus=[uuid])",
                "        assert probe_before.ok, probe_before.to_dict()",
                "        assert probe_before.details and probe_before.details['nofile_advisory']['soft'] == 1024",
                "        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 1024",
                "        assert store.prepare_nofile() is None",
                "        expected_soft = RECOMMENDED_NOFILE if hard_orig == resource.RLIM_INFINITY else min(hard_orig, RECOMMENDED_NOFILE)",
                "        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == expected_soft",
                "        probe_after = store.probe(gpus=[uuid])",
                "        assert probe_after.ok and probe_after.details is None",
                "    finally:",
                "        stop_evt.set()",
                "        for thread in threads:",
                "            thread.join(timeout=1.0)",
                "        for sock in socks:",
                "            sock.close()",
                "print('RAISED_OK')",
            ]
        )
        raise_result = subprocess.run(
            [sys.executable, "-c", raise_script],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=True,
        )
        self.assertIn(raise_result.stdout.strip(), ("RAISED_OK", "SKIP_LOW_HARD"))

        # Real POSIX kernel test 2: hard-capped (1024, 1024) -> default fails closed; RSIBOX_GMS_MIN_NOFILE=0 bypasses.
        capped_script = "\n".join(
            [
                "import os, resource, sys",
                f"sys.path.insert(0, {str(LIVELAB_DIR)!r})",
                "from rsibox.warm import DynamoGmsStore, ENV_GMS_MIN_NOFILE",
                "resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))",
                "os.environ.pop(ENV_GMS_MIN_NOFILE, None)",
                "default_store = DynamoGmsStore(socket_dir='/tmp', target_resolver=lambda *a, **k: [(0, 'GPU-0')])",
                "probe_default = default_store.probe(gpus=['GPU-0'])",
                "assert not probe_default.ok and probe_default.finding['check'] == 'gms_nofile_limit'",
                "assert resource.getrlimit(resource.RLIMIT_NOFILE) == (1024, 1024)",
                "os.environ[ENV_GMS_MIN_NOFILE] = '0'",
                "bypass_store = DynamoGmsStore(socket_dir='/tmp', target_resolver=lambda *a, **k: [(0, 'GPU-0')])",
                "probe_bypass = bypass_store.probe(gpus=['GPU-0'])",
                "assert probe_bypass.finding['check'] == 'dynamo_gms_daemon', probe_bypass.to_dict()",
                "print('CAPPED_BYPASS_OK')",
            ]
        )
        capped_result = subprocess.run(
            [sys.executable, "-c", capped_script],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=True,
        )
        self.assertEqual(capped_result.stdout.strip(), "CAPPED_BYPASS_OK")


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

    def test_async_shadow_replenisher_runs_off_critical_path_with_standby_driver(self):
        """Golden case: StandbyShadowDriver.respawn() returns immediately while replenishment runs concurrently.

        Proves failover latency stays decoupled from background shadow boot time, and that
        telemetry (`attempts`, `succeeded`, `failed`, `last_cost_s`, `in_flight`, `history`)
        updates atomically once the background thread completes.
        """
        from kernel import coordinator as kernel_coordinator

        class _StandbyDummyLab(BaseLab):
            name = "standby_dummy"

            def _start_engine(self):
                participant = kernel_coordinator.LocalParticipant("participant")
                return kernel_coordinator.StepGapCoordinator([participant], contextlib.nullcontext), []

            def _stop_engine(self):
                return None

            def _run_workload(self, workload):
                return {"ok": True}

        worker_entered = threading.Event()
        release_worker = threading.Event()
        received_stores: list[str] = []

        def slow_replenish(lab, store):
            received_stores.append(store.name if store is not None else "none")
            worker_entered.set()
            release_worker.wait(timeout=5.0)
            return {"ok": True, "shadow": "ready"}

        with AsyncShadowReplenisher(slow_replenish) as replenisher:
            driver = StandbyShadowDriver(
                promote_fn=lambda lab, store: (lab._spawn(), {"ok": True, "reset": ["shadow promoted"]})[1],
                replenish_fn=replenisher,
            )
            lab = _StandbyDummyLab(weight_store=DiskStore(), recovery_driver=driver)

            started_at = time.perf_counter()
            respawn_result = lab.reset("respawn")
            respawn_elapsed_s = time.perf_counter() - started_at

            self.assertTrue(respawn_result["ok"])
            self.assertEqual(respawn_result["replenish"], "started")
            # Critical path returned while slow_replenish is still blocked on release_worker.
            self.assertLess(respawn_elapsed_s, 1.0)
            self.assertTrue(worker_entered.wait(timeout=2.0))
            self.assertTrue(replenisher.in_flight)
            self.assertFalse(replenisher.wait(timeout_s=0.02))

            release_worker.set()
            self.assertTrue(replenisher.wait(timeout_s=2.0))
            self.assertFalse(replenisher.in_flight)
            self.assertEqual(received_stores, ["none"])
            self.assertEqual(replenisher.attempts, 1)
            self.assertEqual(replenisher.succeeded, 1)
            self.assertEqual(replenisher.failed, 0)
            self.assertIsNone(replenisher.last_error)
            self.assertIsNotNone(replenisher.last_cost_s)
            self.assertGreaterEqual(replenisher.last_cost_s, 0.0)
            self.assertEqual(len(replenisher.history), 1)
            self.assertTrue(replenisher.history[0]["ok"])

        self.assertTrue(replenisher.closed)

    def test_async_shadow_replenisher_single_flight_coalescing_and_busy_mode(self):
        """Golden case: rapid back-to-back failovers never spawn overlapping replenishments.

        Proves `queue_when_busy=True` executes strictly single-flight (`max_concurrent == 1`)
        and coalesces multiple in-flight triggers into at most one follow-up with the latest
        request, while `queue_when_busy=False` reports `"busy"` without queueing.
        """
        state_lock = threading.Lock()
        active_count = 0
        max_concurrent = 0
        executed_tags: list[str] = []
        first_entered = threading.Event()
        release_first = threading.Event()

        def tracked_worker(lab, store):
            nonlocal active_count, max_concurrent
            with state_lock:
                active_count += 1
                max_concurrent = max(max_concurrent, active_count)
                executed_tags.append(str(lab))
            try:
                if lab == "failover-1":
                    first_entered.set()
                    release_first.wait(timeout=5.0)
            finally:
                with state_lock:
                    active_count -= 1

        # 1. Default coalescing queue: 4 rapid triggers -> run #1 + single coalesced follow-up (#4).
        replenisher = AsyncShadowReplenisher(tracked_worker, queue_when_busy=True)
        try:
            self.assertEqual(replenisher("failover-1", None), "started")
            self.assertTrue(first_entered.wait(timeout=2.0))
            self.assertEqual(replenisher("failover-2", None), "queued")
            self.assertTrue(replenisher.pending)
            self.assertEqual(replenisher("failover-3", None), "coalesced")
            self.assertEqual(replenisher("failover-4", None), "coalesced")
            self.assertEqual(replenisher.coalesced, 2)

            release_first.set()
            self.assertTrue(replenisher.wait(timeout_s=2.0))
            self.assertEqual(max_concurrent, 1)
            self.assertEqual(executed_tags, ["failover-1", "failover-4"])
            self.assertEqual(replenisher.attempts, 2)
            self.assertEqual(replenisher.succeeded, 2)
            self.assertFalse(replenisher.pending)
        finally:
            self.assertTrue(replenisher.close(timeout_s=2.0))

        # 2. queue_when_busy=False: rejects triggers while active with "busy".
        busy_entered = threading.Event()
        release_busy = threading.Event()

        def busy_worker(lab, store):
            busy_entered.set()
            release_busy.wait(timeout=5.0)

        busy_replenisher = AsyncShadowReplenisher(busy_worker, queue_when_busy=False)
        try:
            self.assertEqual(busy_replenisher("lab-a", None), "started")
            self.assertTrue(busy_entered.wait(timeout=2.0))
            self.assertEqual(busy_replenisher("lab-b", None), "busy")
            self.assertEqual(busy_replenisher.busy_rejected, 1)
            self.assertFalse(busy_replenisher.pending)
            release_busy.set()
            self.assertTrue(busy_replenisher.wait(timeout_s=2.0))
            self.assertEqual(busy_replenisher.attempts, 1)
        finally:
            self.assertTrue(busy_replenisher.close(timeout_s=2.0))

    def test_async_shadow_replenisher_failure_accounting_history_cap_and_clean_close(self):
        """Golden case: background worker errors never leak uncaught thread exceptions.

        Proves exceptions and `{"ok": False}` outcomes increment `failed` with structured
        `last_error`, `max_history` bounds telemetry retention, and `close()` drops queued
        work and refuses subsequent triggers cleanly.
        """
        unhandled_thread_errors: list[object] = []
        orig_excepthook = threading.excepthook
        threading.excepthook = lambda args: unhandled_thread_errors.append(args)
        try:
            call_index = 0
            gate_entered = threading.Event()
            release_gate = threading.Event()

            def scripted_worker(lab, store):
                nonlocal call_index
                call_index += 1
                if call_index == 1:
                    raise RuntimeError("vmm handle exhausted")
                if call_index == 2:
                    return {"ok": False, "error": "shadow port busy"}
                if call_index == 3:
                    return {"ok": True}
                gate_entered.set()
                release_gate.wait(timeout=5.0)
                return {"ok": True}

            replenisher = AsyncShadowReplenisher(scripted_worker, max_history=2)
            self.assertEqual(replenisher("run-1", None), "started")
            self.assertTrue(replenisher.wait(timeout_s=2.0))
            self.assertEqual(replenisher.attempts, 1)
            self.assertEqual(replenisher.failed, 1)
            self.assertEqual(replenisher.succeeded, 0)
            self.assertEqual(replenisher.last_error, "RuntimeError: vmm handle exhausted")

            self.assertEqual(replenisher("run-2", None), "started")
            self.assertTrue(replenisher.wait(timeout_s=2.0))
            self.assertEqual(replenisher.failed, 2)
            self.assertEqual(replenisher.last_error, "shadow port busy")

            self.assertEqual(replenisher("run-3", None), "started")
            self.assertTrue(replenisher.wait(timeout_s=2.0))
            self.assertEqual(replenisher.attempts, 3)
            self.assertEqual(replenisher.succeeded, 1)
            self.assertEqual(replenisher.failed, 2)
            self.assertIsNone(replenisher.last_error)

            # History capped at max_history=2 (attempts 2 and 3) and status() is JSON-serializable.
            snapshot = replenisher.status()
            json.dumps(snapshot)
            self.assertEqual([entry["attempt"] for entry in snapshot["history"]], [2, 3])
            self.assertEqual(snapshot["history"][0]["error"], "shadow port busy")
            self.assertTrue(snapshot["history"][1]["ok"])

            # close() drops pending queued work, joins active worker, and refuses new triggers.
            self.assertEqual(replenisher("run-4", None), "started")
            self.assertTrue(gate_entered.wait(timeout=2.0))
            self.assertEqual(replenisher("run-5-dropped", None), "queued")

            def release_once_closed() -> None:
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not replenisher.closed:
                    time.sleep(0.005)
                release_gate.set()

            releaser_thread = threading.Thread(target=release_once_closed, daemon=True)
            releaser_thread.start()
            self.assertTrue(replenisher.close(timeout_s=2.0))
            releaser_thread.join(timeout=2.0)
            self.assertEqual(replenisher.attempts, 4)
            self.assertEqual(replenisher("run-6-after-close", None), "closed")
            self.assertEqual(replenisher.attempts, 4)
            self.assertEqual(unhandled_thread_errors, [])
        finally:
            threading.excepthook = orig_excepthook

        with self.assertRaises(TypeError):
            AsyncShadowReplenisher("not-callable")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AsyncShadowReplenisher(lambda lab, store: None, max_history=-1)

    def test_standby_shadow_driver_guards_back_to_back_failover_and_fallback(self):
        """Golden case: back-to-back failover during in-flight or failed replenish never kills primary blindly.

        Proves (a) bounded `replenish_wait_timeout_s` waits for replenishment and promotes cleanly,
        (b) wait timeout returns `standby_not_ready` WITHOUT stopping the primary, (c) `fallback_driver`
        cleanly delegates on timeout, (d) failed replenishment refuses to stop primary and schedules a
        background retry, and (e) bound `.trigger` and reserved `"replenish"` keys are handled cleanly.
        """
        from kernel import coordinator as kernel_coordinator

        stop_calls: list[str] = []
        promote_calls: list[str] = []

        class _GuardedLab(BaseLab):
            name = "guarded_lab"

            def _start_engine(self):
                participant = kernel_coordinator.LocalParticipant("participant")
                return kernel_coordinator.StepGapCoordinator([participant], contextlib.nullcontext), []

            def _stop_engine(self):
                stop_calls.append("stop")

            def _run_workload(self, workload):
                return {"ok": True}

        def promote_with_spoofed_key(lab, store):
            promote_calls.append("promote")
            lab._spawn()
            return {"ok": True, "reset": ["shadow promoted"], "replenish": "spoofed"}

        # Case A: back-to-back failover succeeds when gated replenish completes within wait timeout.
        gate_entered = threading.Event()
        release_gate = threading.Event()
        replenish_runs = 0

        def gated_worker(lab, store):
            nonlocal replenish_runs
            replenish_runs += 1
            if replenish_runs == 1:
                gate_entered.set()
                release_gate.wait(timeout=5.0)
            return {"ok": True}

        with AsyncShadowReplenisher(gated_worker) as replenisher:
            # Also exercises bound `.trigger` detection and reserved "replenish" key protection!
            driver = StandbyShadowDriver(
                promote_fn=promote_with_spoofed_key,
                replenish_fn=replenisher.trigger,
                replenish_wait_timeout_s=2.0,
            )
            lab = _GuardedLab(weight_store=DiskStore(), recovery_driver=driver)
            first_failover = lab.reset("respawn")
            self.assertTrue(first_failover["ok"])
            self.assertEqual(first_failover["replenish"], "started")
            self.assertTrue(gate_entered.wait(timeout=2.0))

            def release_soon() -> None:
                time.sleep(0.05)
                release_gate.set()

            releaser = threading.Thread(target=release_soon, daemon=True)
            releaser.start()
            second_failover = lab.reset("respawn")
            releaser.join(timeout=2.0)
            self.assertTrue(second_failover["ok"])
            self.assertEqual(len(promote_calls), 2)
            self.assertTrue(replenisher.wait(timeout_s=2.0))

        # Case B: wait timeout returns standby_not_ready WITHOUT calling _stop_engine() on primary!
        stop_calls.clear()
        promote_calls.clear()
        stuck_entered = threading.Event()
        release_stuck = threading.Event()

        def stuck_worker(lab, store):
            stuck_entered.set()
            release_stuck.wait(timeout=5.0)
            return {"ok": True}

        with AsyncShadowReplenisher(stuck_worker) as stuck_replenisher:
            strict_driver = StandbyShadowDriver(
                promote_fn=promote_with_spoofed_key,
                replenish_fn=stuck_replenisher,
                replenish_wait_timeout_s=0.05,
            )
            lab_b = _GuardedLab(weight_store=DiskStore(), recovery_driver=strict_driver)
            self.assertTrue(lab_b.reset("respawn")["ok"])
            self.assertTrue(stuck_entered.wait(timeout=2.0))
            stops_before = len(stop_calls)

            blocked_failover = lab_b.reset("respawn")
            self.assertFalse(blocked_failover["ok"])
            self.assertEqual(blocked_failover["recovery_mode"], "standby_shadow")
            self.assertIn("standby_not_ready", blocked_failover["error"])
            self.assertEqual(len(stop_calls), stops_before, "primary must NOT be stopped when standby is not ready")

            # Case C: with fallback_driver configured, timeout delegates cleanly to DefaultRespawnDriver.
            lab_b.recovery_driver = StandbyShadowDriver(
                promote_fn=promote_with_spoofed_key,
                replenish_fn=stuck_replenisher,
                replenish_wait_timeout_s=0.02,
                fallback_driver=DefaultRespawnDriver(),
            )
            fallback_failover = lab_b.reset("respawn")
            self.assertTrue(fallback_failover["ok"])
            self.assertEqual(fallback_failover["driver"], "default_respawn")
            self.assertEqual(fallback_failover["fallback_from"], "standby_shadow")
            release_stuck.set()

        # Case D: previous replenish failed -> refuses to kill primary and triggers background retry.
        stop_calls.clear()
        promote_calls.clear()
        flaky_attempts = 0

        def flaky_worker(lab, store):
            nonlocal flaky_attempts
            flaky_attempts += 1
            if flaky_attempts == 1:
                raise RuntimeError("standby boot OOM")
            return {"ok": True}

        with AsyncShadowReplenisher(flaky_worker) as flaky_replenisher:
            flaky_driver = StandbyShadowDriver(
                promote_fn=promote_with_spoofed_key,
                replenish_fn=flaky_replenisher,
            )
            lab_d = _GuardedLab(weight_store=DiskStore(), recovery_driver=flaky_driver)
            self.assertTrue(lab_d.reset("respawn")["ok"])
            self.assertTrue(flaky_replenisher.wait(timeout_s=2.0))
            self.assertFalse(flaky_replenisher.last_ok)

            stops_before_retry = len(stop_calls)
            retry_result = lab_d.reset("respawn")
            self.assertFalse(retry_result["ok"])
            self.assertIn("standby_not_ready", retry_result["error"])
            self.assertEqual(retry_result["replenish"], "started")
            self.assertEqual(len(stop_calls), stops_before_retry)
            self.assertTrue(flaky_replenisher.wait(timeout_s=2.0))
            self.assertTrue(flaky_replenisher.last_ok)

            # Once the background retry succeeded, failover promotes cleanly again!
            recovered = lab_d.reset("respawn")
            self.assertTrue(recovered["ok"])

        # Case E: closed replenisher refuses to kill primary into an empty pool.
        stop_calls.clear()
        closed_replenisher = AsyncShadowReplenisher(lambda lab, store: {"ok": True})
        self.assertTrue(closed_replenisher.close())
        lab_e = _GuardedLab(
            weight_store=DiskStore(),
            recovery_driver=StandbyShadowDriver(
                promote_fn=promote_with_spoofed_key, replenish_fn=closed_replenisher
            ),
        )
        closed_failover = lab_e.reset("respawn")
        self.assertFalse(closed_failover["ok"])
        self.assertIn("replenisher is closed", closed_failover["error"])
        self.assertEqual(closed_failover["replenish"], "closed")
        self.assertEqual(stop_calls, [])

    def test_async_shadow_replenisher_cancel_dispose_timeout_warning_and_thread_start_error(self):
        """Golden case: cooperative cancellation, post-close disposal, live-worker close warning, and start failure.

        Proves (a) 0-arg, 1-arg `cancel_event`, and 3-arg workers receive cancellation signals,
        (b) `dispose_fn` cleans up outcomes returned after `close()`, (c) `close()` timeout sets
        `closed_with_live_worker=True` and `__exit__` logs a warning, (d) `Thread.start()`
        `RuntimeError` resets `in_flight` cleanly, and (e) calling `wait()` inside the worker
        raises `RuntimeError` instead of deadlocking.
        """
        # 1. 0-arg worker & 1-arg cooperative cancel_event worker + dispose_fn on post-close outcome.
        zero_arg_calls: list[str] = []
        with AsyncShadowReplenisher(lambda: zero_arg_calls.append("ran")) as zero_replenisher:
            self.assertEqual(zero_replenisher("lab", None), "started")
            self.assertTrue(zero_replenisher.wait(timeout_s=2.0))
        self.assertEqual(zero_arg_calls, ["ran"])

        worker_started = threading.Event()
        disposed_items: list[dict] = []

        def cooperative_worker(cancel_event: threading.Event) -> dict:
            worker_started.set()
            cancel_event.wait(timeout=5.0)
            return {"shadow_pid": 4242, "cancelled": cancel_event.is_set()}

        coop_replenisher = AsyncShadowReplenisher(
            cooperative_worker,
            dispose_fn=lambda outcome: disposed_items.append(dict(outcome)),
        )
        self.assertEqual(coop_replenisher("lab", None), "started")
        self.assertTrue(worker_started.wait(timeout=2.0))
        self.assertTrue(coop_replenisher.close(timeout_s=2.0))
        self.assertTrue(coop_replenisher.cancel_event.is_set())
        self.assertEqual(disposed_items, [{"shadow_pid": 4242, "cancelled": True}])
        self.assertEqual(coop_replenisher.disposed_after_close, 1)
        self.assertTrue(coop_replenisher.history[-1]["disposed"])

        # 2. close() timeout sets closed_with_live_worker=True and __exit__ logs a warning.
        unblock_stuck = threading.Event()
        stuck_started = threading.Event()

        def uninterruptible_worker(lab, store, cancel_event):
            stuck_started.set()
            unblock_stuck.wait(timeout=5.0)
            return {"ok": True}

        stuck_replenisher = AsyncShadowReplenisher(uninterruptible_worker)
        orig_close = stuck_replenisher.close
        try:
            with self.assertLogs("rsibox.warm.recovery", level="WARNING") as captured_logs:
                with mock.patch.object(
                    stuck_replenisher, "close", side_effect=lambda: orig_close(timeout_s=0.02)
                ):
                    with stuck_replenisher:
                        self.assertEqual(stuck_replenisher("lab", None), "started")
                        self.assertTrue(stuck_started.wait(timeout=2.0))
            self.assertTrue(stuck_replenisher.status()["closed_with_live_worker"])
            self.assertTrue(any("timed out" in message for message in captured_logs.output))
        finally:
            unblock_stuck.set()
            orig_close(timeout_s=2.0)
        self.assertFalse(stuck_replenisher.status()["closed_with_live_worker"])

        # 3. Thread.start() RuntimeError resets in_flight cleanly and records failure.
        start_fail_replenisher = AsyncShadowReplenisher(lambda lab, store: {"ok": True})
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("can't start new thread")):
            dispatch_status = start_fail_replenisher("lab", None)
        self.assertEqual(dispatch_status, "failed")
        self.assertFalse(start_fail_replenisher.in_flight)
        self.assertEqual(start_fail_replenisher.failed, 1)
        self.assertIn("can't start new thread", start_fail_replenisher.last_error)
        self.assertTrue(start_fail_replenisher.close())

        # 4. Calling wait() from inside the active replenishment thread raises RuntimeError (no deadlock).
        self_wait_holder: list[AsyncShadowReplenisher] = []

        def self_waiting_worker(lab, store):
            self_wait_holder[0].wait(timeout_s=1.0)

        with AsyncShadowReplenisher(self_waiting_worker) as self_wait_replenisher:
            self_wait_holder.append(self_wait_replenisher)
            self.assertEqual(self_wait_replenisher("lab", None), "started")
            self.assertTrue(self_wait_replenisher.wait(timeout_s=2.0))
            self.assertEqual(self_wait_replenisher.failed, 1)
            self.assertIn("cannot call wait()", self_wait_replenisher.last_error)


if __name__ == "__main__":
    unittest.main()
