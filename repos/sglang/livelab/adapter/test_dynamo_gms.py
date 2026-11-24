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

                    # Resetting back to empty extra config on a GMS server_args clears RO lock mode back to RW_OR_RO!
                    sa_rw = mods["sglang.srt.server_args"].ServerArgs("m")
                    sa_rw.load_format = loader_cls
                    agent.sync_sglang_gms_globals(sa_rw)
                    self.assertEqual(gms_sglang_mod._gms_lock_mode, "RW_OR_RO")
                    self.assertIsNone(gms_sglang_mod._gms_ro_connect_timeout_ms)

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

    def test_stale_uuid_map_is_cleared_on_unremapped_sglang_lab(self):
        uuid0 = "GPU-00000000-0000-0000-0000-000000000000"
        mods, _, _, _, _ = _fake_sglang_and_gms_modules()
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            guards = [_MockGmsSock(os.path.join(tmp, socket_name(uuid0, tag)), committed=True) for tag in GMS_TAGS]
            try:
                store = DynamoGmsStore(
                    socket_dir=tmp,
                    target_resolver=lambda *a, **k: [(0, uuid0)],
                )
                lab = SglangLab.__new__(SglangLab)
                lab.config = {"model_path": "m", "tp_size": 1}
                lab.weight_store = store
                lab._weight_cache_env_restore = None
                with mock.patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": "0", "RSIBOX_GMS_DEVICE_UUID_MAP": '{"0":"GPU-stale"}'},
                ), mock.patch.dict(sys.modules, mods):
                    lab._prepare_weight_store()
                    self.assertNotIn("RSIBOX_GMS_DEVICE_UUID_MAP", os.environ)
                    lab._restore_weight_cache_env()
                    self.assertEqual(os.environ.get("RSIBOX_GMS_DEVICE_UUID_MAP"), '{"0":"GPU-stale"}')
            finally:
                for g in guards:
                    g.close()

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

    def test_nonpauseable_decode_cuda_graphs_delegates_and_guards_pause_in_parent_and_child(self):
        """Golden case: allow_nonpauseable_cuda_graphs=True enables decode='full' under GMS,
        delegates GMSMemorySaverImpl.cuda_graph to torch.cuda.graph in parent + child,
        refuses pause('cuda_graph'), and restores cleanly on teardown.
        """
        import contextlib

        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        mods, _, _, _, resolution_result = _fake_sglang_and_gms_modules()
        graph_calls: list[tuple] = []
        pause_calls: list[str | None] = []

        class _FakeGMSMemorySaverImpl:
            def cuda_graph(self, cuda_graph, pool, stream, capture_error_mode, tag, **backup_options):
                raise RuntimeError("SGLang with GMS does not support pauseable CUDA graphs.")

            def pause(self, tag=None):
                pause_calls.append(tag)

        ms_mod = types.ModuleType("gpu_memory_service.integrations.sglang.memory_saver")
        ms_mod.GMSMemorySaverImpl = _FakeGMSMemorySaverImpl
        mods["gpu_memory_service.integrations.sglang.memory_saver"] = ms_mod

        fake_torch = types.ModuleType("torch")
        fake_cuda = types.SimpleNamespace()

        @contextlib.contextmanager
        def _fake_cuda_graph(cuda_graph, pool=None, stream=None, capture_error_mode="global"):
            graph_calls.append((cuda_graph, pool, stream, capture_error_mode))
            yield

        fake_cuda.graph = _fake_cuda_graph
        fake_torch.cuda = fake_cuda
        mods["torch"] = fake_torch

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            guards = [_MockGmsSock(os.path.join(tmp, socket_name(uuid, tag)), committed=True) for tag in GMS_TAGS]
            try:
                store = DynamoGmsStore(
                    socket_dir=tmp,
                    read_only=True,
                    target_resolver=lambda *a, **k: [(0, uuid)],
                )
                lab = SglangLab.__new__(SglangLab)
                lab.config = {"model_path": "m", "tp_size": 1}
                lab.weight_store = store
                lab._weight_cache_env_restore = None
                lab.allow_nonpauseable_cuda_graphs = True

                prev_cg = os.environ.get(agent.ENV_SGLANG_GMS_NONPAUSEABLE_CG)
                with mock.patch.dict(sys.modules, mods):
                    cfg = lab._prepare_weight_store()
                    sa = cfg["server_args"]
                    self.assertEqual(resolution_result(sa, "cuda_graph_backend_decode"), "full")
                    self.assertEqual(resolution_result(sa, "cuda_graph_backend_prefill"), "disabled")
                    self.assertEqual(os.environ.get(agent.ENV_SGLANG_GMS_NONPAUSEABLE_CG), "1")

                    impl = _FakeGMSMemorySaverImpl()
                    with impl.cuda_graph("g1", pool="p1", stream="s1", capture_error_mode="global", tag="cuda_graph"):
                        pass
                    self.assertEqual(graph_calls, [("g1", "p1", "s1", "global")])

                    # Pausing weights/kv_cache/None works (positional & keyword), but pausing cuda_graph MUST fail closed!
                    impl.pause("weights")
                    impl.pause(tag="kv_cache")
                    impl.pause()
                    self.assertEqual(pause_calls, ["weights", "kv_cache", None])
                    with self.assertRaisesRegex(RuntimeError, "cannot pause 'cuda_graph'"):
                        impl.pause("cuda_graph")
                    with self.assertRaisesRegex(RuntimeError, "cannot pause 'cuda_graph'"):
                        impl.pause(tag="cuda_graph")

                    # Simulate spawned scheduler child reinstalling the shim via run_scheduler_entry.
                    agent.uninstall_sglang_gms_nonpauseable_cuda_graph_shim()
                    with self.assertRaisesRegex(RuntimeError, "does not support pauseable CUDA graphs"):
                        impl.cuda_graph("g0", "p0", "s0", "global", "cuda_graph")

                    fake_sched_mod = types.ModuleType("fake_sched_mod_cg")

                    class _FakeScheduler:
                        def __init__(self):
                            pass

                    fake_sched_mod.Scheduler = _FakeScheduler
                    child_captured: list[str] = []

                    def fake_real_entry(_child_sa):
                        with _FakeGMSMemorySaverImpl().cuda_graph("g_child", pool="pc", stream="sc", capture_error_mode="global", tag="cuda_graph"):
                            child_captured.append("captured")
                        return "child_ok"

                    with mock.patch.dict(sys.modules, {"fake_sched_mod_cg": fake_sched_mod}):
                        self.assertEqual(
                            agent.run_scheduler_entry("/tmp/unused.sock", "fake_sched_mod_cg", fake_real_entry, sa),
                            "child_ok",
                        )
                    self.assertEqual(child_captured, ["captured"])
                    self.assertEqual(graph_calls[-1], ("g_child", "pc", "sc", "global"))

                    # Teardown restores env and original GMSMemorySaverImpl.cuda_graph!
                    lab._restore_weight_cache_env()
                    self.assertEqual(os.environ.get(agent.ENV_SGLANG_GMS_NONPAUSEABLE_CG), prev_cg)
                    with self.assertRaisesRegex(RuntimeError, "does not support pauseable CUDA graphs"):
                        _FakeGMSMemorySaverImpl().cuda_graph("g2", "p2", "s2", "global", "cuda_graph")
            finally:
                agent.uninstall_sglang_gms_nonpauseable_cuda_graph_shim()
                for g in guards:
                    g.close()

    def test_stale_nonpauseable_cg_env_is_cleared_when_flag_is_false(self):
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
                lab.config = {"model_path": "m", "tp_size": 1}
                lab.weight_store = store
                lab._weight_cache_env_restore = None
                lab.allow_nonpauseable_cuda_graphs = False
                with mock.patch.dict(
                    os.environ,
                    {agent.ENV_SGLANG_GMS_NONPAUSEABLE_CG: "1"},
                ), mock.patch.dict(sys.modules, mods):
                    cfg = lab._prepare_weight_store()
                    self.assertNotIn(agent.ENV_SGLANG_GMS_NONPAUSEABLE_CG, os.environ)
                    self.assertEqual(resolution_result(cfg["server_args"], "cuda_graph_backend_decode"), "disabled")
                    lab._restore_weight_cache_env()
                    self.assertEqual(os.environ.get(agent.ENV_SGLANG_GMS_NONPAUSEABLE_CG), "1")
            finally:
                agent.uninstall_sglang_gms_nonpauseable_cuda_graph_shim()
                for g in guards:
                    g.close()

    def test_gms_meta_loader_shim_avoids_cuda_dummy_weight_allocation_and_falls_back(self):
        events: list[tuple] = []
        active_devices: list[str] = []

        class _DeviceCtx:
            def __init__(self, dev):
                self.dev = dev

            def __enter__(self):
                active_devices.append(self.dev)
                return self

            def __exit__(self, *exc):
                active_devices.pop()
                return False

        class _FakeTorch:
            @staticmethod
            def device(dev):
                return _DeviceCtx(dev)

        class _FakeTorchDev:
            def __init__(self):
                self.cur = 0
                self.set_calls = []

            def current_device(self):
                return self.cur

            def set_device(self, d):
                self.set_calls.append(d)
                self.cur = d

        fake_dev = _FakeTorchDev()

        class _FakeQuant:
            def process_weights_after_loading(self, mod):
                events.append(("quant_post", mod.name, tuple(active_devices)))

        class _FakeModel:
            def __init__(self):
                self.eval_called = False
                self._subs = [
                    ("lin0", types.SimpleNamespace(
                        name="lin0", quant_method=_FakeQuant(), is_weights_quantized=lambda: False,
                    )),
                    ("quant_fn", types.SimpleNamespace(
                        name="quant_fn", quant_method=_FakeQuant(), is_weights_quantized=lambda: True,
                    )),
                    ("quant_bool", types.SimpleNamespace(
                        name="quant_bool", quant_method=_FakeQuant(), is_weights_quantized=True,
                    )),
                ]

            def named_modules(self):
                return self._subs

            def eval(self):
                self.eval_called = True
                return self

        class _FakeGMSModelLoader:
            def __init__(self):
                self.load_config = {"load_format": "gms"}

            def _create_meta_model(self, model_config, device_config):
                events.append(("orig_create_meta", device_config.device))
                return "orig_model"

        sgl_loader_mod = types.ModuleType("sglang.srt.model_loader.loader")
        sgl_utils_mod = types.ModuleType("sglang.srt.model_loader.utils")
        ml_mod = types.ModuleType("gpu_memory_service.integrations.sglang.model_loader")
        common_utils = types.ModuleType("gpu_memory_service.integrations.common.utils")
        fail_init = [False]
        fail_post = [False]

        def _init_model(model_config, load_cfg, quant_cfg):
            fake_dev.cur = 7  # verify finally restores original device 0
            if fail_init[0]:
                raise RuntimeError("exotic init failure")
            events.append(("init_model", tuple(active_devices), load_cfg, quant_cfg))
            return _FakeModel()

        import contextlib

        @contextlib.contextmanager
        def _set_dtype(dt):
            events.append(("dtype_enter", dt))
            try:
                yield
            finally:
                events.append(("dtype_exit", dt))

        def _post_load(m):
            events.append(("post_load", tuple(active_devices)))
            if fail_post[0]:
                raise NotImplementedError("meta tensor op")

        sgl_loader_mod._initialize_model = _init_model
        sgl_loader_mod._get_quantization_config = lambda mc, lc: "qcfg"
        sgl_loader_mod.set_default_torch_dtype = _set_dtype
        sgl_loader_mod.post_load_weights = _post_load
        sgl_utils_mod.process_model_weights_after_loading = lambda m, mc: events.append(("proc_after",))
        ml_mod.GMSModelLoader = _FakeGMSModelLoader
        ml_mod.strip_gms_model_loader_config = lambda lc, load_format="dummy": {"load_format": load_format}
        ml_mod.torch_device = lambda: fake_dev
        common_utils.setup_meta_tensor_workaround = lambda: events.append(("meta_workaround",))

        mods, _, _, gms_sglang_mod, _ = _fake_sglang_and_gms_modules()
        mods.update({
            "torch": _FakeTorch,
            "sglang.srt.model_loader": types.ModuleType("sglang.srt.model_loader"),
            "sglang.srt.model_loader.loader": sgl_loader_mod,
            "sglang.srt.model_loader.utils": sgl_utils_mod,
            "gpu_memory_service.integrations.sglang.model_loader": ml_mod,
        })
        mods["gpu_memory_service.integrations.common.utils"].setup_meta_tensor_workaround = (
            common_utils.setup_meta_tensor_workaround
        )
        mc = types.SimpleNamespace(dtype="bfloat16")
        dc = types.SimpleNamespace(device="cuda")

        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            guards = [_MockGmsSock(os.path.join(tmp, socket_name(uuid, tag)), committed=True) for tag in GMS_TAGS]
            try:
                store = DynamoGmsStore(
                    socket_dir=tmp,
                    read_only=True,
                    target_resolver=lambda *a, **k: [(0, uuid)],
                )
                lab = SglangLab.__new__(SglangLab)
                lab.config = {"model_path": "m", "tp_size": 1}
                lab.weight_store = store
                lab._weight_cache_env_restore = None
                lab.allow_nonpauseable_cuda_graphs = False

                with mock.patch.dict(sys.modules, mods):
                    loader = _FakeGMSModelLoader()
                    # Before _prepare_weight_store: original path runs with cuda device_config.
                    self.assertEqual(loader._create_meta_model(mc, dc), "orig_model")
                    self.assertEqual(events, [("orig_create_meta", "cuda")])
                    events.clear()

                    # _prepare_weight_store -> sync_sglang_gms_globals installs zero-copy meta shim (NO direct install call!).
                    cfg = lab._prepare_weight_store()
                    sa = cfg["server_args"]
                    model = loader._create_meta_model(mc, dc)
                    self.assertIsInstance(model, _FakeModel)
                    self.assertTrue(model.eval_called)
                    self.assertEqual(fake_dev.cur, 0)
                    self.assertEqual(fake_dev.set_calls, [0])
                    # Only lin0 runs quant_post; callable True & bool True skip cleanly without TypeError!
                    self.assertEqual(
                        events,
                        [
                            ("meta_workaround",),
                            ("dtype_enter", "bfloat16"),
                            ("init_model", ("meta",), {"load_format": "dummy"}, "qcfg"),
                            ("post_load", ("meta",)),
                            ("quant_post", "lin0", ("meta",)),
                            ("dtype_exit", "bfloat16"),
                            ("proc_after",),
                        ],
                    )
                    self.assertTrue(agent.install_sglang_gms_meta_loader_shim())  # idempotent

                    # post_load_weights meta error falls back to orig_create_meta so quantized/MLA layouts never corrupt.
                    events.clear()
                    fake_dev.set_calls.clear()
                    fail_post[0] = True
                    self.assertEqual(loader._create_meta_model(mc, dc), "orig_model")
                    self.assertEqual(fake_dev.cur, 0)
                    self.assertEqual(fake_dev.set_calls, [0])
                    self.assertEqual(events[-1], ("orig_create_meta", "cuda"))
                    fail_post[0] = False

                    # If _initialize_model fails, device is still restored and falls back cleanly to original.
                    events.clear()
                    fake_dev.set_calls.clear()
                    fail_init[0] = True
                    self.assertEqual(loader._create_meta_model(mc, dc), "orig_model")
                    self.assertEqual(fake_dev.cur, 0)
                    self.assertEqual(fake_dev.set_calls, [0])
                    self.assertEqual(events[-1], ("orig_create_meta", "cuda"))
                    fail_init[0] = False

                    # Spawned scheduler child reinstalls the shim via run_scheduler_entry even after uninstall.
                    agent.uninstall_sglang_gms_meta_loader_shim()
                    events.clear()
                    self.assertEqual(_FakeGMSModelLoader()._create_meta_model(mc, dc), "orig_model")
                    fake_sched_mod = types.ModuleType("fake_sched_mod_meta")

                    class _FakeScheduler:
                        def __init__(self):
                            pass

                    fake_sched_mod.Scheduler = _FakeScheduler
                    child_res = []

                    def fake_real_entry(_child_sa):
                        events.clear()
                        m = _FakeGMSModelLoader()._create_meta_model(mc, dc)
                        child_res.append(isinstance(m, _FakeModel))
                        return "child_ok"

                    with mock.patch.dict(sys.modules, {"fake_sched_mod_meta": fake_sched_mod}):
                        self.assertEqual(
                            agent.run_scheduler_entry("/tmp/unused.sock", "fake_sched_mod_meta", fake_real_entry, sa),
                            "child_ok",
                        )
                    self.assertEqual(child_res, [True])

                    # Switching to a non-GMS server_args or None uninstalls the shim cleanly.
                    sa_non_gms = mods["sglang.srt.server_args"].ServerArgs("m")
                    agent.sync_sglang_gms_globals(sa_non_gms)
                    events.clear()
                    self.assertEqual(_FakeGMSModelLoader()._create_meta_model(mc, dc), "orig_model")
                    self.assertEqual(events, [("orig_create_meta", "cuda")])

                    # Re-prepare and verify teardown via _restore_weight_cache_env uninstalls the meta-loader shim!
                    lab._prepare_weight_store()
                    lab._restore_weight_cache_env()
                    events.clear()
                    self.assertEqual(_FakeGMSModelLoader()._create_meta_model(mc, dc), "orig_model")
                    self.assertEqual(events, [("orig_create_meta", "cuda")])
            finally:
                agent.uninstall_sglang_gms_meta_loader_shim()
                for g in guards:
                    g.close()


if __name__ == "__main__":
    unittest.main()
