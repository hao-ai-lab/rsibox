"""CPU unit tests for pluggable `DynamoGmsStore` wiring in `SglangLab`.

Faithfully models SGLang >=0.5.21 `ServerArgs` + `declare_resolution` (storing decisions in
`_resolved_overrides` WITHOUT mutating raw `ServerArgs` fields) + `resolution_result` +
`gpu_memory_service.integrations.sglang.setup_gms`, verifying both parent and spawned
scheduler child (`agent.run_scheduler_entry`) lock-mode synchronization, JSON-string config
merging, disabled pauseable CUDA graphs, `extra_pythonpath` re-entry/restoration, MsgPack
socket probing, `fresh_check` GMS stripping, and conflict refusals.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

LIVELAB_DIR = Path(__file__).resolve().parents[1]
CORE_LIVELAB = Path(__file__).resolve().parents[4] / "core" / "livelab"
for p in (str(LIVELAB_DIR), str(CORE_LIVELAB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter import agent  # noqa: E402
from adapter.lab import SglangLab  # noqa: E402
from rsibox.warm import DiskStore, DynamoGmsStore, GMS_TAGS, LaunchDelta  # noqa: E402
from rsibox.warm.dynamo_gms import socket_name  # noqa: E402


def _pack_runtime_state(committed: bool) -> bytes:
    parts = [b"\x82"]
    for key, val in (("type", "get_runtime_state_response"), ("committed", committed)):
        kb = key.encode("utf-8")
        parts.append(bytes([0xA0 | len(kb)]) + kb)
        if isinstance(val, bool):
            parts.append(b"\xc3" if val else b"\xc2")
        else:
            vb = val.encode("utf-8")
            parts.append(bytes([0xA0 | len(vb)]) + vb)
    payload = b"".join(parts)
    return struct.pack(">I", len(payload)) + payload


class _MockGmsSock:
    def __init__(self, path: str, *, committed: bool = True):
        self.path = path
        self.committed = committed
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(4)
        self._sock.settimeout(0.05)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            try:
                conn.settimeout(0.5)
                hdr = conn.recv(4)
                if len(hdr) == 4:
                    n = struct.unpack(">I", hdr)[0]
                    _ = conn.recv(n)
                    conn.sendall(_pack_runtime_state(self.committed))
            except OSError:
                pass
            finally:
                conn.close()

    def close(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        self._sock.close()
        if os.path.exists(self.path):
            os.unlink(self.path)


def _fake_sglang_and_gms_modules():
    """Return synthetic `sglang.*` and `gpu_memory_service.*` modules matching SGLang >=0.5.21.

    Crucially, `declare_resolution` appends to `_resolved_overrides` WITHOUT mutating
    `server_args.model_loader_extra_config` (which stays '{}'), mirroring upstream SGLang.
    """
    events: list[tuple] = []

    class FakeGMSModelLoader:
        pass

    class FakeServerArgs:
        def __init__(self, model_path, host="127.0.0.1", port=30000):
            self.model_path = model_path
            self.host = host
            self.port = port
            self.enable_memory_saver = False
            self.model_loader_extra_config = "{}"
            self.load_format = "auto"
            self._resolved_overrides: list[tuple[str, dict]] = []

    def declare_resolution(server_args, source, **kwargs):
        events.append(("declare", source, dict(kwargs)))
        server_args._resolved_overrides.append((source, dict(kwargs)))

    def resolution_result(server_args, field, default=None):
        for _src, declared in reversed(getattr(server_args, "_resolved_overrides", None) or ()):
            if field in declared:
                return declared[field]
        return getattr(server_args, field, default)

    def get_gms_lock_mode(extra):
        return "RO" if extra.get("gms_read_only") else "RW_OR_RO"

    def get_gms_ro_connect_timeout_ms(extra):
        return extra.get("gms_ro_connect_timeout_ms")

    gms_sglang_mod = types.ModuleType("gpu_memory_service.integrations.sglang")
    gms_sglang_mod._gms_lock_mode = None
    gms_sglang_mod._gms_ro_connect_timeout_ms = None
    gms_sglang_mod._gms_initialized = False

    def setup_gms(server_args):
        events.append(("setup_gms", server_args.model_loader_extra_config))
        declare_resolution(server_args, "dynamo.gms", enable_memory_saver=True)
        # Upstream setup_gms reads raw getattr (which is '{}' under declare_resolution!).
        raw = getattr(server_args, "model_loader_extra_config", None)
        extra = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
        gms_sglang_mod._gms_lock_mode = get_gms_lock_mode(extra)
        gms_sglang_mod._gms_ro_connect_timeout_ms = get_gms_ro_connect_timeout_ms(extra)
        gms_sglang_mod._gms_initialized = True
        return FakeGMSModelLoader

    gms_sglang_mod.setup_gms = setup_gms

    gms_common_utils = types.ModuleType("gpu_memory_service.integrations.common.utils")
    gms_common_utils.get_gms_lock_mode = get_gms_lock_mode
    gms_common_utils.get_gms_ro_connect_timeout_ms = get_gms_ro_connect_timeout_ms

    mods = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
        "sglang.srt.arg_groups": types.ModuleType("sglang.srt.arg_groups"),
        "sglang.srt.arg_groups.overrides": types.ModuleType("sglang.srt.arg_groups.overrides"),
        "gpu_memory_service": types.ModuleType("gpu_memory_service"),
        "gpu_memory_service.integrations": types.ModuleType("gpu_memory_service.integrations"),
        "gpu_memory_service.integrations.common": types.ModuleType("gpu_memory_service.integrations.common"),
        "gpu_memory_service.integrations.common.utils": gms_common_utils,
        "gpu_memory_service.integrations.sglang": gms_sglang_mod,
    }
    mods["sglang.srt.server_args"].ServerArgs = FakeServerArgs
    mods["sglang.srt.arg_groups.overrides"].declare_resolution = declare_resolution
    mods["sglang.srt.arg_groups.overrides"].resolution_result = resolution_result
    return mods, events, FakeGMSModelLoader, gms_sglang_mod, resolution_result


class SglangDynamoGmsTests(unittest.TestCase):
    def test_prepare_weight_store_syncs_gms_globals_in_parent_and_spawned_child(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        mods, events, loader_cls, gms_sglang_mod, resolution_result = _fake_sglang_and_gms_modules()
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            overlay = os.path.join(tmp, "gms-overlay")
            os.makedirs(overlay)
            guards = [_MockGmsSock(os.path.join(tmp, socket_name(uuid, tag)), committed=True) for tag in GMS_TAGS]
            try:
                store = DynamoGmsStore(
                    socket_dir=tmp,
                    read_only=True,
                    extra_pythonpath=(overlay,),
                    target_resolver=lambda *a, **k: [(0, uuid)],
                )
                lab = SglangLab.__new__(SglangLab)
                # Non-conflicting user model_loader_extra_config merges cleanly with GMS RO keys!
                lab.config = {
                    "model_path": "m",
                    "host": "127.0.0.1",
                    "port": 31000,
                    "tp_size": 1,
                    "model_loader_extra_config": {"foo": 1},
                }
                lab.weight_store = store
                lab._weight_cache_env_restore = None

                prev_gms = os.environ.get("GMS_SOCKET_DIR")
                prev_py = os.environ.get("PYTHONPATH")
                with mock.patch.dict(sys.modules, mods):
                    cfg = lab._prepare_weight_store()
                    # Re-entry must preserve extra_pythonpath in sys.path!
                    cfg = lab._prepare_weight_store()

                    self.assertIn("server_args", cfg)
                    sa = cfg["server_args"]
                    self.assertEqual(sa.model_path, "m")
                    self.assertEqual(sa.port, 31000)
                    self.assertTrue(resolution_result(sa, "enable_memory_saver"))
                    self.assertIs(resolution_result(sa, "load_format"), loader_cls)
                    self.assertEqual(resolution_result(sa, "cuda_graph_backend_decode"), "disabled")
                    self.assertEqual(resolution_result(sa, "cuda_graph_backend_prefill"), "disabled")
                    resolved_extra = resolution_result(sa, "model_loader_extra_config")
                    self.assertIsInstance(resolved_extra, str)
                    self.assertEqual(
                        json.loads(resolved_extra),
                        {"foo": 1, "gms_read_only": True, "gms_ro_connect_timeout_ms": 10000},
                    )
                    # Parent process globals MUST reflect RO + timeout even though raw attr stayed '{}'!
                    self.assertEqual(sa.model_loader_extra_config, "{}")
                    self.assertEqual(gms_sglang_mod._gms_lock_mode, "RO")
                    self.assertEqual(gms_sglang_mod._gms_ro_connect_timeout_ms, 10000)

                    # Simulate spawned scheduler child: globals reset to None before run_scheduler_entry.
                    gms_sglang_mod._gms_lock_mode = None
                    gms_sglang_mod._gms_ro_connect_timeout_ms = None
                    fake_sched_mod = types.ModuleType("fake_sched_mod")

                    class _FakeScheduler:
                        def __init__(self):
                            pass

                    fake_sched_mod.Scheduler = _FakeScheduler
                    seen_in_child: list[tuple] = []

                    def fake_real_entry(child_sa):
                        seen_in_child.append(
                            (gms_sglang_mod._gms_lock_mode, gms_sglang_mod._gms_ro_connect_timeout_ms)
                        )
                        return "ok"

                    with mock.patch.dict(sys.modules, {"fake_sched_mod": fake_sched_mod}):
                        ret = agent.run_scheduler_entry("/tmp/unused.sock", "fake_sched_mod", fake_real_entry, sa)
                    self.assertEqual(ret, "ok")
                    self.assertEqual(seen_in_child, [("RO", 10000)])

                self.assertEqual(os.environ.get("GMS_SOCKET_DIR"), os.path.abspath(tmp))
                self.assertIn(overlay, os.environ.get("PYTHONPATH", "").split(os.pathsep))
                self.assertIn(overlay, sys.path)

                lab._restore_weight_cache_env()
                self.assertEqual(os.environ.get("GMS_SOCKET_DIR"), prev_gms)
                self.assertEqual(os.environ.get("PYTHONPATH"), prev_py)
                self.assertNotIn(overlay, sys.path)
            finally:
                for g in guards:
                    g.close()

    def test_prepare_weight_store_handles_none_extra_config_and_rejects_custom_delta_args(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        mods, _, _, _, resolution_result = _fake_sglang_and_gms_modules()
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            guards = [_MockGmsSock(os.path.join(tmp, socket_name(uuid, tag)), committed=True) for tag in GMS_TAGS]
            try:
                store = DynamoGmsStore(
                    socket_dir=tmp,
                    read_only=True,
                    target_resolver=lambda *a, **k: [(0, uuid)],
                )
                lab = SglangLab.__new__(SglangLab)
                lab.config = {"model_path": "m", "tp_size": 1, "model_loader_extra_config": None}
                lab.weight_store = store
                lab._weight_cache_env_restore = None
                with mock.patch.dict(sys.modules, mods):
                    cfg = lab._prepare_weight_store()
                self.assertEqual(
                    json.loads(resolution_result(cfg["server_args"], "model_loader_extra_config")),
                    {"gms_read_only": True, "gms_ro_connect_timeout_ms": 10000},
                )
                lab._restore_weight_cache_env()
            finally:
                for g in guards:
                    g.close()

        # Custom non-GMS WeightStore returning CLI args fails closed rather than silently dropping them.
        class _CustomArgStore(DiskStore):
            name = "custom_cli"

            def launch_delta(self, **kw):
                return LaunchDelta(args=("--custom-flag", "1"))

        lab2 = SglangLab.__new__(SglangLab)
        lab2.config = {"model_path": "m", "tp_size": 1}
        lab2.weight_store = _CustomArgStore()
        lab2._weight_cache_env_restore = None
        with self.assertRaisesRegex(RuntimeError, "CLI args"):
            lab2._prepare_weight_store()

    def test_fresh_check_strips_gms_only_config_keys(self):
        lab = SglangLab.__new__(SglangLab)
        lab.config = {
            "model_path": "m",
            "load_format": "gms",
            "model_loader_extra_config": json.dumps({"gms_read_only": True, "foo": 1}),
        }
        lab.check_timeout = 10.0
        lab.live_text = [{}]
        captured_specs: list[dict] = []

        class _NoopOverlay:
            root = "/tmp"
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def fake_run(mod, spec, overlay_root, **kw):
            captured_specs.append(spec)
            return {"ok": True, "output": {}}

        with (
            mock.patch("adapter.fixture.harness.H.PackageOverlay", return_value=_NoopOverlay()),
            mock.patch("adapter.fixture.harness.H.run_fresh_subprocess", side_effect=fake_run),
        ):
            res = lab.fresh_check({"files": {}}, workload={"prompts": ["hi"]})
        self.assertTrue(res["ok"])
        cfg = captured_specs[0]["config"]
        self.assertNotIn("load_format", cfg)
        self.assertEqual(json.loads(cfg["model_loader_extra_config"]), {"foo": 1})

    def test_prepare_weight_store_rejects_cuda_graphs_and_uncommitted_ro_and_restores_env(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            overlay = os.path.join(tmp, "gms-overlay")
            os.makedirs(overlay)
            guards = [_MockGmsSock(os.path.join(tmp, socket_name(uuid, tag)), committed=False) for tag in GMS_TAGS]
            try:
                ro_store = DynamoGmsStore(
                    socket_dir=tmp,
                    read_only=True,
                    target_resolver=lambda *a, **k: [(0, uuid)],
                )
                lab_ro = SglangLab.__new__(SglangLab)
                lab_ro.config = {"model_path": "m", "tp_size": 1}
                lab_ro.weight_store = ro_store
                lab_ro._weight_cache_env_restore = None
                with self.assertRaisesRegex(RuntimeError, "EMPTY / uncommitted"):
                    lab_ro._prepare_weight_store()

                rw_store = DynamoGmsStore(
                    socket_dir=tmp,
                    read_only=False,
                    extra_pythonpath=(overlay,),
                    target_resolver=lambda *a, **k: [(0, uuid)],
                )
                lab_cg = SglangLab.__new__(SglangLab)
                lab_cg.config = {
                    "model_path": "m",
                    "tp_size": 1,
                    "cuda_graph_backend_decode": "flashinfer",
                }
                lab_cg.weight_store = rw_store
                lab_cg._weight_cache_env_restore = None
                prev_gms = os.environ.get("GMS_SOCKET_DIR")
                with self.assertRaisesRegex(RuntimeError, "CUDA graphs"):
                    lab_cg._prepare_weight_store()
                self.assertEqual(os.environ.get("GMS_SOCKET_DIR"), prev_gms)
                self.assertNotIn(overlay, sys.path)
            finally:
                for g in guards:
                    g.close()

    def test_init_rejects_simultaneous_weight_cache_dir_and_weight_store(self):
        store = DynamoGmsStore(socket_dir="/tmp/gms")
        with self.assertRaisesRegex(ValueError, "cannot pass both weight_cache_dir and weight_store"):
            SglangLab(
                config={"model_path": "m"},
                modules=[],
                weight_cache_dir="/tmp/wc",
                weight_store=store,
            )

    def test_prepare_weight_store_fails_fast_when_sockets_missing(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            store = DynamoGmsStore(socket_dir=tmp, target_resolver=lambda *a, **k: [(0, uuid)])
            lab = SglangLab.__new__(SglangLab)
            lab.config = {"model_path": "m", "tp_size": 1}
            lab.weight_store = store
            lab._weight_cache_env_restore = None
            with self.assertRaises(RuntimeError):
                lab._prepare_weight_store()


if __name__ == "__main__":
    unittest.main()
