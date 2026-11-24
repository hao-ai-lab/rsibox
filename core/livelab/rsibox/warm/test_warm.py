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
        """Golden case: Docker-like RLIMIT_NOFILE=1024 fails closed before GMS daemon spawn.

        Proves `probe()` and `ensure_ready()` emit `gms_nofile_limit` (recommending
        `--ulimit nofile=65536:65536`) and spawn zero daemons when hard_limit < min_nofile
        or auto_raise_nofile=False, while `min_nofile=0` explicitly opts out.
        """
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-nofile-") as temp_dir:
            gpu_uuid = "GPU-33333333-2222-3333-4444-555555555555"
            raised_calls: list[tuple[int, int]] = []
            spawned_daemons: list[list[str]] = []

            # 1. Hard cap == 1024 (Docker --ulimit nofile=1024:1024): cannot raise, fails closed.
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
            self.assertIn("soft=1024, hard=1024", probe_capped.error)
            self.assertTrue(
                any(f"--ulimit nofile={RECOMMENDED_NOFILE}:{RECOMMENDED_NOFILE}" in action for action in probe_capped.finding["actions"])
            )
            self.assertEqual(raised_calls, [])

            with mock.patch.object(
                capped_store,
                "_spawn_gms_daemon",
                side_effect=lambda cmd, env, log_path: spawned_daemons.append(list(cmd)) or 999999,
            ):
                ensure_result = capped_store.ensure_ready(gpus=[gpu_uuid], timeout_s=0.1)
            self.assertFalse(ensure_result.ok)
            self.assertEqual(ensure_result.finding["check"], "gms_nofile_limit")
            self.assertEqual(spawned_daemons, [])

            # 2. Hard cap == 2048 (< 4096): raises soft to 2048 but STILL fails closed.
            partial_store = DynamoGmsStore(
                socket_dir=temp_dir,
                min_nofile=4096,
                auto_raise_nofile=True,
                get_nofile=lambda: (1024, 2048),
                set_nofile=lambda soft, hard: raised_calls.append((soft, hard)),
                target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
            )
            probe_partial = partial_store.probe(gpus=[gpu_uuid])
            self.assertFalse(probe_partial.ok)
            self.assertEqual(probe_partial.finding["check"], "gms_nofile_limit")
            self.assertEqual(raised_calls, [(2048, 2048)])
            self.assertIn("soft=2048, hard=2048", probe_partial.error)

            # 3. auto_raise_nofile=False never calls set_nofile even when hard_limit is high.
            raised_calls.clear()
            strict_store = DynamoGmsStore(
                socket_dir=temp_dir,
                min_nofile=4096,
                auto_raise_nofile=False,
                get_nofile=lambda: (1024, 1048576),
                set_nofile=lambda soft, hard: raised_calls.append((soft, hard)),
                target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
            )
            probe_strict = strict_store.probe(gpus=[gpu_uuid])
            self.assertFalse(probe_strict.ok)
            self.assertEqual(probe_strict.finding["check"], "gms_nofile_limit")
            self.assertEqual(raised_calls, [])

            # 4. min_nofile=0 disables the preflight guard cleanly; negative/bool rejected.
            weights_guard = _UnixSocketGuard(
                os.path.join(temp_dir, socket_name(gpu_uuid, "weights")), gms_state=("committed", True)
            )
            kv_guard = _UnixSocketGuard(
                os.path.join(temp_dir, socket_name(gpu_uuid, "kv_cache")), gms_state=("empty", False)
            )
            try:
                opt_out_store = DynamoGmsStore(
                    socket_dir=temp_dir,
                    min_nofile=0,
                    get_nofile=lambda: (1024, 1024),
                    target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
                )
                self.assertTrue(opt_out_store.probe(gpus=[gpu_uuid]).ok)
            finally:
                weights_guard.close()
                kv_guard.close()

            with self.assertRaises(ValueError):
                DynamoGmsStore(socket_dir=temp_dir, min_nofile=-1)
            with self.assertRaises(ValueError):
                DynamoGmsStore(socket_dir=temp_dir, min_nofile=True)

    def test_nofile_preflight_auto_raises_soft_limit_and_handles_infinity_and_real_rlimit(self):
        """Golden case: auto_raise lifts soft RLIMIT_NOFILE to min_nofile without breaking high hard caps.

        Proves soft=1024 with hard=1048576 (or RLIM_INFINITY) raises to 4096 and passes probe(),
        PermissionError during set_nofile fails closed cleanly, and real kernel getrlimit/setrlimit
        works end-to-end inside an isolated subprocess.
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
                applied_limits: list[tuple[int, int]] = []
                store = DynamoGmsStore(
                    socket_dir=temp_dir,
                    min_nofile=4096,
                    auto_raise_nofile=True,
                    get_nofile=lambda: (1024, 1048576),
                    set_nofile=lambda soft, hard: applied_limits.append((soft, hard)),
                    target_resolver=lambda *args, **kwargs: [(0, gpu_uuid)],
                )
                probe_result = store.probe(gpus=[gpu_uuid])
                self.assertTrue(probe_result.ok, probe_result.to_dict())
                self.assertEqual(applied_limits, [(4096, 1048576)])
            finally:
                weights_guard.close()
                kv_guard.close()

        # RLIM_INFINITY (-1 on Linux) as soft passes untouched; as hard allows raising to min_nofile.
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
        self.assertEqual(infinity_calls, [(4096, -1)])

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

        # Real POSIX kernel getrlimit/setrlimit round-trip in an isolated child process.
        child_script = "\n".join(
            [
                "import resource, sys",
                f"sys.path.insert(0, {str(LIVELAB_DIR)!r})",
                "from rsibox.warm import DynamoGmsStore",
                "soft_orig, hard_orig = resource.getrlimit(resource.RLIMIT_NOFILE)",
                "if hard_orig != resource.RLIM_INFINITY and hard_orig < 4096:",
                "    print('SKIP_LOW_HARD')",
                "    raise SystemExit(0)",
                "resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard_orig))",
                "store = DynamoGmsStore(socket_dir='/tmp', min_nofile=4096, auto_raise_nofile=True)",
                "finding = store.check_nofile()",
                "soft_after, _ = resource.getrlimit(resource.RLIMIT_NOFILE)",
                "assert finding is None, finding",
                "assert soft_after == 4096, soft_after",
                "print('RAISED_OK')",
            ]
        )
        proc_result = subprocess.run(
            [sys.executable, "-c", child_script],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=True,
        )
        self.assertIn(proc_result.stdout.strip(), ("RAISED_OK", "SKIP_LOW_HARD"))


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


if __name__ == "__main__":
    unittest.main()
