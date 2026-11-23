"""CPU unit tests for pluggable `DynamoGmsStore` wiring in `SglangLab`.

Verifies the upstream SGLang GMS contract (`ServerArgs` + `declare_resolution` +
`setup_gms`, JSON-string `model_loader_extra_config`, disabled pauseable CUDA graphs,
`extra_pythonpath` propagation/restoration, MsgPack socket probing, and conflict refusals)
without requiring GPUs or an installed SGLang/Dynamo runtime.
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

from adapter.lab import SglangLab  # noqa: E402
from rsibox.warm import DynamoGmsStore, GMS_TAGS  # noqa: E402
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
    """Return synthetic `sglang.*` and `gpu_memory_service.integrations.sglang` modules
    modeling SGLang >=0.5.21 `ServerArgs` + `declare_resolution` + `setup_gms`."""
    events: list[tuple] = []

    class FakeGMSModelLoader:
        pass

    class FakeServerArgs:
        def __init__(self, model_path, host="127.0.0.1", port=30000):
            self.model_path = model_path
            self.host = host
            self.port = port
            self.enable_memory_saver = False
            self.model_loader_extra_config = None
            self.load_format = "auto"
            self.resolutions: list[tuple[str, dict]] = []

    def declare_resolution(server_args, source, **kwargs):
        events.append(("declare", source, dict(kwargs)))
        server_args.resolutions.append((source, dict(kwargs)))
        for k, v in kwargs.items():
            setattr(server_args, k, v)

    def setup_gms(server_args):
        events.append(("setup_gms", server_args.model_loader_extra_config))
        declare_resolution(server_args, "dynamo.gms", enable_memory_saver=True)
        # Upstream setup_gms parses JSON string model_loader_extra_config.
        extra = server_args.model_loader_extra_config
        if extra is not None:
            if not isinstance(extra, str):
                raise TypeError("model_loader_extra_config must be a JSON string")
            json.loads(extra)
        return FakeGMSModelLoader

    mods = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
        "sglang.srt.arg_groups": types.ModuleType("sglang.srt.arg_groups"),
        "sglang.srt.arg_groups.overrides": types.ModuleType("sglang.srt.arg_groups.overrides"),
        "gpu_memory_service": types.ModuleType("gpu_memory_service"),
        "gpu_memory_service.integrations": types.ModuleType("gpu_memory_service.integrations"),
        "gpu_memory_service.integrations.sglang": types.ModuleType("gpu_memory_service.integrations.sglang"),
    }
    mods["sglang.srt.server_args"].ServerArgs = FakeServerArgs
    mods["sglang.srt.arg_groups.overrides"].declare_resolution = declare_resolution
    mods["gpu_memory_service.integrations.sglang"].setup_gms = setup_gms
    return mods, events, FakeGMSModelLoader


class SglangDynamoGmsTests(unittest.TestCase):
    def test_prepare_weight_store_wires_setup_gms_declare_resolution_and_restores_env(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        mods, events, loader_cls = _fake_sglang_and_gms_modules()
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
                lab.config = {"model_path": "m", "host": "127.0.0.1", "port": 31000, "tp_size": 1}
                lab.weight_store = store
                lab._weight_cache_env_restore = None

                prev_gms = os.environ.get("GMS_SOCKET_DIR")
                prev_py = os.environ.get("PYTHONPATH")
                with mock.patch.dict(sys.modules, mods):
                    cfg = lab._prepare_weight_store()

                self.assertIn("server_args", cfg)
                sa = cfg["server_args"]
                self.assertEqual(sa.model_path, "m")
                self.assertEqual(sa.port, 31000)
                self.assertTrue(sa.enable_memory_saver)
                self.assertIs(sa.load_format, loader_cls)
                self.assertEqual(sa.cuda_graph_backend_decode, "disabled")
                self.assertEqual(sa.cuda_graph_backend_prefill, "disabled")
                self.assertIsInstance(sa.model_loader_extra_config, str)
                self.assertEqual(
                    json.loads(sa.model_loader_extra_config),
                    {"gms_read_only": True, "gms_ro_connect_timeout_ms": 10000},
                )
                # Ordering: rsibox.gms overrides -> setup_gms -> rsibox.gms.loader
                self.assertEqual([e[0] for e in events], ["declare", "setup_gms", "declare", "declare"])
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

    def test_prepare_weight_store_rejects_cuda_graphs_and_uncommitted_ro_and_restores_env(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            overlay = os.path.join(tmp, "gms-overlay")
            os.makedirs(overlay)
            guards = [_MockGmsSock(os.path.join(tmp, socket_name(uuid, tag)), committed=False) for tag in GMS_TAGS]
            try:
                # 1. read_only on uncommitted GMS fails closed before env mutation
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

                # 2. explicit CUDA graph backend rejected and env/sys.path restored cleanly
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
