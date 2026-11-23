"""CPU unit tests for pluggable `DynamoGmsStore` wiring in `SglangLab`."""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

LIVELAB_DIR = Path(__file__).resolve().parents[1]
CORE_LIVELAB = Path(__file__).resolve().parents[4] / "core" / "livelab"
for p in (str(LIVELAB_DIR), str(CORE_LIVELAB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter.lab import SglangLab  # noqa: E402
from rsibox.warm import DynamoGmsStore, GMS_TAGS  # noqa: E402
from rsibox.warm.dynamo_gms import socket_name  # noqa: E402


class _Sock:
    def __init__(self, path: str):
        self.path = path
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
            conn.close()

    def close(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        self._sock.close()
        if os.path.exists(self.path):
            os.unlink(self.path)


class SglangDynamoGmsTests(unittest.TestCase):
    def test_prepare_weight_store_sets_gms_env_and_config_and_restores(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            guards = [_Sock(os.path.join(tmp, socket_name(uuid, tag))) for tag in GMS_TAGS]
            try:
                store = DynamoGmsStore(
                    socket_dir=tmp,
                    read_only=True,
                    uuid_resolver=lambda *a, **k: [uuid],
                )
                lab = SglangLab.__new__(SglangLab)
                lab.config = {"model_path": "m", "tp_size": 1}
                lab.weight_store = store
                lab._weight_cache_env_restore = None

                prev = os.environ.get("GMS_SOCKET_DIR")
                cfg = lab._prepare_weight_store()
                self.assertEqual(cfg["load_format"], "gms")
                self.assertEqual(cfg["model_loader_extra_config"], {"gms_read_only": True})
                self.assertEqual(os.environ.get("GMS_SOCKET_DIR"), os.path.abspath(tmp))

                lab._restore_weight_cache_env()
                self.assertEqual(os.environ.get("GMS_SOCKET_DIR"), prev)
            finally:
                for g in guards:
                    g.close()

    def test_prepare_weight_store_fails_fast_when_sockets_missing(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglgms-") as tmp:
            store = DynamoGmsStore(socket_dir=tmp, uuid_resolver=lambda *a, **k: [uuid])
            lab = SglangLab.__new__(SglangLab)
            lab.config = {"model_path": "m", "tp_size": 1}
            lab.weight_store = store
            lab._weight_cache_env_restore = None
            with self.assertRaises(RuntimeError):
                lab._prepare_weight_store()


if __name__ == "__main__":
    unittest.main()
