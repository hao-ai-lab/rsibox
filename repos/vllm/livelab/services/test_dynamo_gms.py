"""Unit tests for pluggable Dynamo GMS support in `repos/vllm/livelab/services/engine.py`.

CPU-only tests (no GPU, vLLM, or Dynamo installation required). Verifies flag wiring,
conflict findings, GMS socket preflight probing, and zero regression on `--weights` (`ipc_cache`).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import engine  # noqa: E402
from rsibox.warm import GMS_TAGS, GMS_VLLM_LOAD_FORMAT, GMS_VLLM_WORKER_CLS  # noqa: E402
from rsibox.warm.dynamo_gms import socket_name  # noqa: E402


def _parsed(explicit=(), **overrides):
    args = {
        "model": "m",
        "port": 18500,
        "host": None,
        "tensor_parallel_size": 1,
        "load_format": "auto",
        "worker_cls": "auto",
        "model_loader_extra_config": {},
    }
    args.update(overrides)
    return {"ok": True, "args": args, "explicit": list(explicit)}


class _UnixSock:
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


class EngineDynamoGmsTests(unittest.TestCase):
    def test_serve_cmd_and_env_inject_dynamo_gms_flags(self):
        a, vargv = engine.parse([
            "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights-backend", "dynamo_gms",
            "--gms-socket-dir", "/tmp/gms",
            "--gms-read-only", "--gms-scratch-kv",
            "--gms-pythonpath", "/opt/gms-overlay",
            "--", "Qwen/Qwen3-0.6B", "--port", "18500",
            "--worker-extension-cls", "adapter.worker_ext.KernelWorkerExtension",
        ])
        v = _parsed(explicit=("model", "port"))
        self.assertEqual(engine.weights_backend(a, v), "dynamo_gms")
        self.assertEqual(engine.load_format(a, v), GMS_VLLM_LOAD_FORMAT)
        self.assertEqual(engine.argv_findings(a, v), [])

        cmd = engine.serve_cmd(a, vargv, v)
        self.assertIn("--load-format", cmd)
        self.assertIn("gms", cmd)
        self.assertIn("--worker-cls", cmd)
        self.assertIn(GMS_VLLM_WORKER_CLS, cmd)
        self.assertIn("--worker-extension-cls", cmd)
        self.assertIn("adapter.worker_ext.KernelWorkerExtension", cmd)
        cfg_idx = cmd.index("--model-loader-extra-config")
        self.assertEqual(json.loads(cmd[cfg_idx + 1]), {"gms_read_only": True})

        env = engine.engine_env(a, base={"PATH": "/usr/bin"})
        self.assertEqual(env["GMS_SOCKET_DIR"], os.path.abspath("/tmp/gms"))
        self.assertEqual(env["DYN_GMS_SCRATCH_KV_ENABLED"], "1")
        self.assertIn("/opt/gms-overlay", env["PYTHONPATH"].split(os.pathsep))

    def test_conflicts_emit_structured_findings(self):
        # 1. --weights + --weights-backend dynamo_gms
        a1, _ = engine.parse([
            "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights", "--weights-backend", "dynamo_gms",
            "--gms-socket-dir", "/tmp/gms", "--", "m",
        ])
        f1 = engine.argv_findings(a1, _parsed(explicit=("model",)))
        self.assertIn("weights_backend_conflict", {f["check"] for f in f1})

        # 2. --kvplan + --weights-backend dynamo_gms (--worker-cls collision)
        a2, _ = engine.parse([
            "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--kvplan=/tmp/kv", "--weights-backend", "dynamo_gms",
            "--gms-socket-dir", "/tmp/gms", "--", "m",
        ])
        f2 = engine.argv_findings(a2, _parsed(explicit=("model",)))
        self.assertIn("kvplan_gms_worker_cls", {f["check"] for f in f2})

        # 3. explicit conflicting --load-format
        a3, _ = engine.parse([
            "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights-backend", "dynamo_gms", "--gms-socket-dir", "/tmp/gms", "--", "m",
        ])
        f3 = engine.argv_findings(
            a3, _parsed(explicit=("model", "load_format"), load_format="safetensors")
        )
        self.assertIn("gms_load_format", {f["check"] for f in f3})

    def test_preflight_probes_gms_sockets_and_fails_fast_when_missing(self):
        uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            a, _ = engine.parse([
                "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
                "--weights-backend", "dynamo_gms", "--gms-socket-dir", tmp, "--", "m",
            ])
            v = _parsed(explicit=("model",))
            with mock.patch("rsibox.warm.dynamo_gms.resolve_gpu_uuids", return_value=[uuid]):
                found, rep = engine.preflight(a, v, {"CUDA_VISIBLE_DEVICES": "0"})
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["check"], "dynamo_gms_daemon")
                self.assertFalse(rep["gms_probe"]["ok"])

                # Start both weights and kv_cache listeners -> preflight passes.
                guards = [_UnixSock(os.path.join(tmp, socket_name(uuid, tag))) for tag in GMS_TAGS]
                try:
                    found2, rep2 = engine.preflight(a, v, {"CUDA_VISIBLE_DEVICES": "0"})
                    self.assertEqual(found2, [])
                    self.assertTrue(rep2["gms_probe"]["ok"])
                finally:
                    for g in guards:
                        g.close()

    def test_legacy_weights_flag_stays_ipc_cache(self):
        a, vargv = engine.parse([
            "start", "--cache-root", "/tmp/c", "--log", "/tmp/l", "--weights", "--", "m",
        ])
        v = _parsed(explicit=("model",))
        self.assertEqual(engine.weights_backend(a, v), "ipc_cache")
        self.assertEqual(engine.load_format(a, v), "ipc_cache")
        cmd = engine.serve_cmd(a, vargv, v)
        self.assertIn("ipc_cache", cmd)
        self.assertNotIn(GMS_VLLM_WORKER_CLS, cmd)


if __name__ == "__main__":
    unittest.main()
