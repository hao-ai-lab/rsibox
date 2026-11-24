"""Unit tests for pluggable Dynamo GMS support in `repos/vllm/livelab/services/engine.py`
and `VllmLab.launch_environment()`.

CPU-only tests (no GPU, vLLM, or Dynamo installation required). Verifies flag wiring,
PYTHONPATH ordering, conflict findings, Tier-2 reconfig protection, `/proc` fallback
recovery, `_inherit_launcher` backend switching, MsgPack GMS socket preflight probing,
and zero regression on `--weights` (`ipc_cache`).
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
LIVELAB_ROOT = HERE.parent
for _p in (str(HERE), str(LIVELAB_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine  # noqa: E402
from adapter.conformance_lab import VllmLab  # noqa: E402
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


def _pack_runtime_state(committed: bool) -> bytes:
    """Minimal MsgPack map: {"type": "get_runtime_state_response", "committed": bool}."""
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


class EngineDynamoGmsTests(unittest.TestCase):
    def test_serve_cmd_and_env_inject_dynamo_gms_flags_without_shadowing_live_site(self):
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
        self.assertEqual(
            json.loads(cmd[cfg_idx + 1]),
            {"gms_read_only": True, "gms_ro_connect_timeout_ms": 10000},
        )

        env = engine.engine_env(a, base={"PATH": "/usr/bin", "PYTHONPATH": "/tmp/hooks:/tmp/farm"})
        self.assertEqual(env["GMS_SOCKET_DIR"], os.path.abspath("/tmp/gms"))
        self.assertEqual(env["DYN_GMS_SCRATCH_KV_ENABLED"], "1")
        py_parts = env["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(py_parts[0], engine.LIVE_SITE)
        self.assertLess(py_parts.index("/tmp/farm"), py_parts.index("/opt/gms-overlay"))

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

        # 4. --gms-read-only + contradicting --model-loader-extra-config emits finding,
        #    while non-conflicting user keys merge cleanly in serve_cmd().
        a4, vargv4 = engine.parse([
            "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights-backend", "dynamo_gms", "--gms-read-only",
            "--gms-socket-dir", "/tmp/gms", "--", "m",
            "--model-loader-extra-config", '{"foo": 1}',
        ])
        f4_bad = engine.argv_findings(
            a4, _parsed(explicit=("model", "model_loader_extra_config"), model_loader_extra_config={"gms_read_only": False})
        )
        self.assertIn("gms_loader_extra_config", {f["check"] for f in f4_bad})

        v4_merge = _parsed(explicit=("model", "model_loader_extra_config"), model_loader_extra_config={"foo": 1})
        self.assertEqual(engine.argv_findings(a4, v4_merge), [])
        cmd4 = engine.serve_cmd(a4, vargv4, v4_merge)
        self.assertEqual(cmd4.count("--model-loader-extra-config"), 1)
        self.assertEqual(
            json.loads(cmd4[cmd4.index("--model-loader-extra-config") + 1]),
            {"foo": 1, "gms_read_only": True, "gms_ro_connect_timeout_ms": 10000},
        )

        # 5. --weights-backend none + explicit --load-format gms/ipc_cache is rejected
        a5_none, _ = engine.parse([
            "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights-backend", "none", "--", "m",
        ])
        f5_none = engine.argv_findings(
            a5_none, _parsed(explicit=("model", "load_format"), load_format=GMS_VLLM_LOAD_FORMAT)
        )
        self.assertIn("weights_backend_conflict", {f["check"] for f in f5_none})

        # 6. --weights-backend ipc_cache enforces weights_load_format & weights_fallback too
        a5, _ = engine.parse([
            "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights-backend", "ipc_cache", "--", "m",
        ])
        f5 = engine.argv_findings(
            a5,
            _parsed(
                explicit=("model", "load_format", "model_loader_extra_config"),
                load_format="safetensors",
                model_loader_extra_config={"fallback": True},
            ),
        )
        checks5 = {f["check"] for f in f5}
        self.assertIn("weights_load_format", checks5)
        self.assertIn("weights_fallback", checks5)

    def test_reconfig_findings_guards_tier2_changes_for_dynamo_gms(self):
        a, _ = engine.parse([
            "reconfigure", "--pid", "123", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights-backend", "dynamo_gms", "--gms-socket-dir", "/tmp/gms",
        ])
        v = _parsed(explicit=("model",))
        reconfig = {
            "tier": "tier2",
            "weights_compatible": False,
            "weight_key_changed": ["model"],
            "changed": {"model": ("A", "B")},
        }
        findings = engine.reconfig_findings(a, v, reconfig)
        self.assertEqual([f["check"] for f in findings], ["weights_reconfig_tier2"])

        a.weights_accept_reconfig = True
        self.assertEqual(engine.reconfig_findings(a, v, reconfig), [])

    def test_load_engine_state_proc_fallback_recovers_gms_fields(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="proc-") as tmp:
            pid_dir = Path(tmp) / "4242"
            pid_dir.mkdir()
            # Exercise `--flag=value` syntax AND RSIBOX_GMS_PYTHONPATH recovery.
            cmdline = [
                "/usr/bin/vllm", "serve", "m", "--port", "18500",
                "--host=127.0.0.1",
                f"--load-format={GMS_VLLM_LOAD_FORMAT}",
                f"--worker-cls={GMS_VLLM_WORKER_CLS}",
                "--model-loader-extra-config=" + json.dumps({"gms_read_only": True}),
            ]
            (pid_dir / "cmdline").write_bytes("\0".join(cmdline).encode())
            (pid_dir / "environ").write_bytes(
                b"GMS_SOCKET_DIR=/tmp/gms-test\0DYN_GMS_SCRATCH_KV_ENABLED=1\0RSIBOX_GMS_PYTHONPATH=/opt/gms\0"
            )
            st = engine.load_engine_state(os.path.join(tmp, "cache"), 4242, proc_root=tmp)
            self.assertIsNotNone(st)
            self.assertEqual(st["vargv"], ["m", "--port", "18500"])
            self.assertEqual(st["launcher"]["weights_backend"], "dynamo_gms")
            self.assertTrue(st["launcher"]["gms_read_only"])
            self.assertTrue(st["launcher"]["gms_scratch_kv"])
            self.assertEqual(st["launcher"]["gms_socket_dir"], "/tmp/gms-test")
            self.assertEqual(st["launcher"]["gms_pythonpath"], "/opt/gms")

    def test_inherit_launcher_handles_equals_syntax_and_backend_switches(self):
        # Switch from saved --weights (ipc_cache) to --weights-backend=dynamo_gms
        # must NOT inherit old ipc_cache --weights-socket-dir!
        raw1 = [
            "reconfigure", "--pid", "1", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights-backend=dynamo_gms",
        ]
        a1, _ = engine.parse(raw1)
        engine._inherit_launcher(
            a1,
            {
                "weights": True, "weights_backend": "",
                "weights_socket_dir": "/tmp/old-ipc-dir",
                "gms_socket_dir": "", "kvplan": "",
            },
            raw1,
        )
        self.assertFalse(a1.weights)
        self.assertEqual(a1.weights_backend, "dynamo_gms")
        self.assertEqual(a1.weights_socket_dir, "")
        self.assertIsNone(a1.kvplan)

        # Launcher-only backend signature detects switches (prevents noop / live corruption).
        saved_gms = {
            "weights": False, "weights_backend": "dynamo_gms",
            "gms_socket_dir": "/tmp/a", "gms_read_only": False, "gms_scratch_kv": False,
        }
        raw_flip = [
            "reconfigure", "--pid", "1", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--gms-scratch-kv",
        ]
        a_flip, _ = engine.parse(raw_flip)
        engine._inherit_launcher(a_flip, saved_gms, raw_flip)
        self.assertNotEqual(
            engine._launcher_backend_signature(saved_gms),
            engine._launcher_backend_signature(a_flip),
        )

        # Switch from saved dynamo_gms to --weights-backend=none drops gms_* inheritance
        raw2 = [
            "reconfigure", "--pid", "1", "--cache-root", "/tmp/c", "--log", "/tmp/l",
            "--weights-backend=none",
        ]
        a2, _ = engine.parse(raw2)
        engine._inherit_launcher(
            a2,
            {"weights": False, "weights_backend": "dynamo_gms", "gms_socket_dir": "/tmp/a", "gms_read_only": True},
            raw2,
        )
        self.assertEqual(a2.weights_backend, "none")
        self.assertEqual(a2.gms_socket_dir, "")
        self.assertFalse(a2.gms_read_only)
        env2 = engine.engine_env(a2, base={"PATH": "/usr/bin"})
        self.assertNotIn("GMS_SOCKET_DIR", env2)

    def test_preflight_probes_gms_sockets_and_fails_closed_on_uncommitted_ro(self):
        uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            a, _ = engine.parse([
                "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
                "--weights-backend", "dynamo_gms", "--gms-socket-dir", tmp, "--", "m",
            ])
            v = _parsed(explicit=("model",))
            with mock.patch("rsibox.warm.dynamo_gms.resolve_target_gpus", return_value=[(0, uuid)]):
                found, rep = engine.preflight(a, v, {"CUDA_VISIBLE_DEVICES": "0"})
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["check"], "dynamo_gms_daemon")
                self.assertFalse(rep["gms_probe"]["ok"])

                # Start uncommitted listeners: RW_OR_RO passes, --gms-read-only fails closed.
                guards = [
                    _MockGmsSock(os.path.join(tmp, socket_name(uuid, tag)), committed=False)
                    for tag in GMS_TAGS
                ]
                try:
                    found2, rep2 = engine.preflight(a, v, {"CUDA_VISIBLE_DEVICES": "0"})
                    self.assertEqual(found2, [])
                    self.assertTrue(rep2["gms_probe"]["ok"])

                    a_ro, _ = engine.parse([
                        "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
                        "--weights-backend", "dynamo_gms", "--gms-read-only",
                        "--gms-socket-dir", tmp, "--", "m",
                    ])
                    found3, rep3 = engine.preflight(a_ro, v, {"CUDA_VISIBLE_DEVICES": "0"})
                    self.assertEqual([f["check"] for f in found3], ["dynamo_gms_not_committed"])
                    self.assertFalse(rep3["gms_probe"]["ok"])
                finally:
                    for g in guards:
                        g.close()

    def test_vllm_lab_launch_environment_translates_gms_flags(self):
        env = VllmLab.launch_environment([
            "--model", "m",
            "--weights-backend", "dynamo_gms",
            "--gms-socket-dir", "/tmp/gms",
            "--gms-pythonpath", "/opt/gms",
            "--gms-read-only",
            "--gms-scratch-kv",
        ])
        self.assertEqual(env["RSIBOX_VLLM_WEIGHTS_BACKEND"], "dynamo_gms")
        self.assertEqual(env["RSIBOX_VLLM_GMS_SOCKET_DIR"], os.path.abspath("/tmp/gms"))
        self.assertEqual(env["RSIBOX_VLLM_GMS_PYTHONPATH"], "/opt/gms")
        self.assertEqual(env["RSIBOX_VLLM_GMS_READ_ONLY"], "1")
        self.assertEqual(env["RSIBOX_VLLM_GMS_SCRATCH_KV"], "1")

    def test_bare_metal_remapped_cuda_visible_devices_exports_uuid_map_and_passes_preflight(self):
        u4 = "GPU-44444444-4444-4444-4444-444444444444"
        u5 = "GPU-55555555-5555-5555-5555-555555555555"
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="gms-") as tmp:
            guards = [
                _MockGmsSock(os.path.join(tmp, socket_name(u, tag)), committed=True)
                for u in (u4, u5)
                for tag in GMS_TAGS
            ]
            try:
                a, _ = engine.parse([
                    "start", "--cache-root", "/tmp/c", "--log", "/tmp/l",
                    "--weights-backend", "dynamo_gms", "--gms-socket-dir", tmp,
                    "--", "m", "-tp", "2",
                ])
                v = _parsed(explicit=("model", "tensor_parallel_size"), tensor_parallel_size=2)
                with (
                    mock.patch("engine.resolve_target_gpus", return_value=[(4, u4), (5, u5)]),
                    mock.patch("rsibox.warm.dynamo_gms.resolve_target_gpus", return_value=[(4, u4), (5, u5)]),
                ):
                    env = engine.engine_env(a, base={"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "4,5"})
                    self.assertEqual(
                        json.loads(env["RSIBOX_GMS_DEVICE_UUID_MAP"]),
                        {"0": u4, "1": u5},
                    )
                    found, rep = engine.preflight(a, v, env)
                    self.assertEqual(found, [])
                    self.assertTrue(rep["gms_probe"]["ok"])
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
