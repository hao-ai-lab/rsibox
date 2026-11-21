"""CPU unit tests for `services.weights` (SGLang native IPC weight-cache daemon helper).

Run:
    PYTHONPATH=repos/sglang/livelab:core/livelab python3 -W error::ResourceWarning \
        -m unittest discover -s repos/sglang/livelab/services -p 'test_weights.py' -t repos/sglang/livelab -v
"""
import contextlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from services import weights as W

_FAKE_DAEMON_SCRIPT = """
import contextlib, os, signal, socket, sys, time
socks = []
stop = False
def _on_term(_signum, _frame):
    raise SystemExit(0)
signal.signal(signal.SIGTERM, _on_term)
sock_tmpl = os.environ["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"]
ready_tmpl = os.environ["SGLANG_WEIGHT_CACHE_READY_TEMPLATE"]
uuids = os.environ["FAKE_UUIDS"].split(",")
if os.environ.get("FAKE_EXIT"):
    sys.exit(int(os.environ["FAKE_EXIT"]))
delay = float(os.environ.get("FAKE_DELAY", "0"))
if delay:
    time.sleep(delay)
for uuid in uuids:
    sp = sock_tmpl.format(device_uuid=uuid)
    rp = ready_tmpl.format(device_uuid=uuid)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(sp)
    s.listen(4)
    socks.append(s)
    with open(rp, "w", encoding="utf-8") as fh:
        fh.write(f"pid={os.getpid()}\\n")
try:
    while not stop:
        time.sleep(0.05)
finally:
    for s in socks:
        with contextlib.suppress(OSError):
            s.close()
"""


def _short_tmp_dir(prefix="sglwc-"):
    base = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def _listen_unix(path):
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        sock.listen(4)
    except BaseException:
        sock.close()
        raise
    return sock


class SglangWeightsServiceTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.sock_dir = _short_tmp_dir()
        self.addCleanup(shutil.rmtree, self.sock_dir, True)

    def test_templates_env_config_and_daemon_cmd(self):
        env = W.client_env(self.sock_dir)
        self.assertIn("{device_uuid}", env[W.SOCKET_ENV])
        self.assertIn("{device_uuid}", env[W.READY_ENV])
        self.assertTrue(env[W.SOCKET_ENV].endswith(".sock"))
        self.assertTrue(env[W.READY_ENV].endswith(".ready"))

        base_cfg = {"model_path": "/models/qwen", "tp_size": 2}
        client_cfg = W.client_config(base_cfg)
        self.assertEqual(client_cfg["weight_cache_mode"], "client")
        self.assertNotIn("weight_cache_mode", base_cfg)

        cmd = W.daemon_cmd(
            "/models/qwen",
            self.sock_dir,
            tp_size=2,
            python="/usr/bin/python3",
            extra=["--quantization", "fp8"],
        )
        self.assertEqual(
            cmd,
            [
                "/usr/bin/python3",
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                "/models/qwen",
                "--tp-size",
                "2",
                "--quantization",
                "fp8",
            ],
        )

    def test_long_socket_path_finding_and_start_refusal(self):
        u0 = "GPU-" + ("a" * 36)
        self.assertIsNone(W.long_socket_path_finding(self.sock_dir, [u0]))

        deep = os.path.join(self.sock_dir, "x" * 75, "nested")
        finding = W.long_socket_path_finding(deep, [u0])
        self.assertIsNotNone(finding)
        self.assertEqual(finding["check"], "weights_socket_path_too_long")

        with self.assertRaises(ValueError):
            W.start("/models/qwen", deep, uuids=[u0])

    def test_gpu_uuids_handles_explicit_uuids_and_nvidia_smi(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-111,GPU-222"}, clear=False):
            self.assertEqual(W.gpu_uuids(), ["111", "222"])

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "MIG-xyz"}, clear=False):
            self.assertEqual(W.gpu_uuids(), ["MIG-xyz"])

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}, clear=False):
            with mock.patch("services.weights.subprocess.check_output", return_value="GPU-aaa\nGPU-bbb\n") as m_out:
                self.assertEqual(W.gpu_uuids(), ["aaa", "bbb"])
        m_out.assert_called_once()
        self.assertEqual(
            m_out.call_args.args[0],
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "4,5"],
        )

        # Normalized socket paths match SGLang's unprefixed CudaSRTPlatform UUID filenames.
        self.assertEqual(
            W.socket_paths(self.sock_dir, ["GPU-aaa", "bbb"]),
            [os.path.join(os.path.abspath(self.sock_dir), "wc-aaa.sock"),
             os.path.join(os.path.abspath(self.sock_dir), "wc-bbb.sock")],
        )

    def test_probe_validates_live_pid_and_connectable_socket(self):
        uuids = ["GPU-100", "GPU-101"]
        sock0, sock1 = W.socket_paths(self.sock_dir, uuids)
        ready0, ready1 = W.ready_paths(self.sock_dir, uuids)

        # Missing ready and sockets -> fails closed.
        missing_res = W.probe(self.sock_dir, uuids=uuids)
        self.assertFalse(missing_res["ok"])
        self.assertEqual(missing_res["missing_ready"], [ready0, ready1])
        self.assertEqual(W.daemon_finding(missing_res, self.sock_dir)["check"], "weights_daemon")

        # Rank 0 ready with current PID + listening socket, rank 1 missing -> still fails.
        with open(ready0, "w", encoding="utf-8") as fh:
            fh.write(f"pid={os.getpid()}\n")
        with contextlib.closing(_listen_unix(sock0)):
            partial = W.probe(self.sock_dir, uuids=uuids)
            self.assertFalse(partial["ok"])
            self.assertEqual(partial["missing_ready"], [ready1])
            self.assertEqual(partial["unconnectable_sockets"], [sock1])

            # Dead PID in rank 1 ready -> stale_pids caught.
            with open(ready1, "w", encoding="utf-8") as fh:
                fh.write("pid=9999999\n")
            with (
                mock.patch("services.weights.proc.pid_alive", side_effect=lambda p: p == os.getpid()),
                contextlib.closing(_listen_unix(sock1)),
            ):
                stale = W.probe(self.sock_dir, uuids=uuids)
                self.assertFalse(stale["ok"])
                self.assertEqual(len(stale["stale_pids"]), 1)

            # Both ready with alive PID + connectable sockets -> ok=True.
            with open(ready1, "w", encoding="utf-8") as fh:
                fh.write(f"pid={os.getpid()}\n")
            with contextlib.closing(_listen_unix(sock1)):
                ok_res = W.probe(self.sock_dir, uuids=uuids)
                self.assertTrue(ok_res["ok"])
                self.assertTrue(ok_res["matched_gpus"])
                self.assertIsNone(W.daemon_finding(ok_res, self.sock_dir))

    def test_start_wait_ready_and_stop_lifecycle_plus_cli_probe(self):
        uuids = ["GPU-live-0", "GPU-live-1"]
        script_path = os.path.join(self.sock_dir, "fake_daemon.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(_FAKE_DAEMON_SCRIPT)
        log_path = os.path.join(self.sock_dir, "daemon.log")

        with mock.patch(
            "services.weights.daemon_cmd",
            return_value=[sys.executable, script_path],
        ):
            started = W.start(
                "/models/qwen",
                self.sock_dir,
                tp_size=2,
                uuids=uuids,
                env={"FAKE_UUIDS": "live-0,live-1"},
                log_path=log_path,
                timeout=10.0,
                poll_interval=0.05,
            )
        proc_obj = started["process"]
        try:
            self.assertTrue(started["probe"]["ok"])
            self.assertEqual(set(started["probe"]["pids"]), {"live-0", "live-1"})

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = W.main(["probe", "--sock-dir", self.sock_dir, "--uuids", ",".join(uuids)])
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["weights_probe"]["ok"])
            self.assertEqual(payload["findings"], [])
        finally:
            W.stop(started["pid"], grace=3.0, reap=proc_obj.poll)
            proc_obj.wait(timeout=5.0)

        # After stop, CLI probe must exit 2 with a weights_daemon finding.
        buf_after = io.StringIO()
        with contextlib.redirect_stdout(buf_after):
            rc_after = W.main(["probe", "--sock-dir", self.sock_dir, "--uuids", ",".join(uuids)])
        self.assertEqual(rc_after, 2)
        payload_after = json.loads(buf_after.getvalue())
        self.assertFalse(payload_after["weights_probe"]["ok"])
        self.assertEqual(payload_after["findings"][0]["check"], "weights_daemon")

    def test_start_cli_accepts_flags_after_model_and_forwards_extra(self):
        captured = {}

        def fake_start(model, sock_dir, *, tp_size=1, uuids=None, python=None, extra=(), log_path=None, timeout=180.0):
            captured.update(
                model=model, sock_dir=sock_dir, tp_size=tp_size, uuids=uuids, extra=list(extra),
            )
            return {"pid": 123, "sock_dir": sock_dir, "uuids": ["aaa"], "probe": {"ok": True}}

        buf = io.StringIO()
        with mock.patch("services.weights.start", side_effect=fake_start), contextlib.redirect_stdout(buf):
            rc = W.main([
                "start", "/models/qwen", "--sock-dir", self.sock_dir, "--tp-size", "2", "--", "--quantization", "fp8",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["model"], "/models/qwen")
        self.assertEqual(captured["sock_dir"], self.sock_dir)
        self.assertEqual(captured["tp_size"], 2)
        self.assertEqual(captured["extra"], ["--quantization", "fp8"])

    def test_start_fails_fast_on_daemon_exit_and_timeout_cleans_up(self):
        uuids = ["GPU-fail-0"]
        script_path = os.path.join(self.sock_dir, "fake_daemon.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(_FAKE_DAEMON_SCRIPT)

        with mock.patch(
            "services.weights.daemon_cmd",
            return_value=[sys.executable, script_path],
        ):
            with self.assertRaises(RuntimeError):
                W.start(
                    "/models/qwen",
                    self.sock_dir,
                    uuids=uuids,
                    env={"FAKE_UUIDS": "fail-0", "FAKE_EXIT": "3"},
                    timeout=10.0,
                    poll_interval=0.05,
                )

            with self.assertRaises(TimeoutError):
                W.start(
                    "/models/qwen",
                    self.sock_dir,
                    uuids=uuids,
                    env={"FAKE_UUIDS": "fail-0", "FAKE_DELAY": "5.0"},
                    timeout=0.2,
                    poll_interval=0.05,
                )

    def test_zero_top_level_gpu_imports(self):
        livelab_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        core_root = os.path.abspath(os.path.join(livelab_root, "..", "..", "..", "core", "livelab"))
        env = {**os.environ, "PYTHONPATH": f"{livelab_root}{os.pathsep}{core_root}"}
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import sys; from services import weights; "
                "assert not any(m in sys.modules for m in ('torch', 'sglang', 'pynvml')); "
                "print('clean')",
            ],
            env=env,
            text=True,
        )
        self.assertEqual(out.strip(), "clean")

    def test_gpu_uuids_fails_closed_on_empty_visible_devices_and_honors_override(self):
        with mock.patch("services.weights.subprocess.check_output") as m_out:
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    W.gpu_uuids()
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": " , "}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    W.gpu_uuids()
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    W.gpu_uuids(visible_devices="")
            with self.assertRaises(RuntimeError):
                W.gpu_uuids([])
            with self.assertRaises(RuntimeError):
                W.gpu_uuids(["GPU-"])
        m_out.assert_not_called()

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            with mock.patch("services.weights.subprocess.check_output", return_value="GPU-c\nGPU-d\n") as m_out2:
                self.assertEqual(W.gpu_uuids(visible_devices="2,3"), ["c", "d"])
        self.assertEqual(
            m_out2.call_args.args[0],
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "2,3"],
        )

    def test_start_and_probe_bound_uuids_to_tp_size_and_reject_empty_uuids(self):
        with mock.patch("services.weights.subprocess.Popen") as m_popen:
            with self.assertRaises(ValueError):
                W.start("/models/qwen", self.sock_dir, uuids=[])
            with self.assertRaises(ValueError):
                W.start("/models/qwen", self.sock_dir, uuids=["GPU-"])
            with self.assertRaises(ValueError):
                W.start("/models/qwen", self.sock_dir, tp_size=0, uuids=["GPU-0"])
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-0,GPU-1"}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "requires 3 visible GPUs"):
                    W.start("/models/qwen", self.sock_dir, tp_size=3)
        m_popen.assert_not_called()

        # 4 visible GPUs with tp_size=2 binds and probes only the first 2 GPUs.
        script_path = os.path.join(self.sock_dir, "fake_daemon.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(_FAKE_DAEMON_SCRIPT)
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-g0,GPU-g1,GPU-g2,GPU-g3"}, clear=False),
            mock.patch("services.weights.daemon_cmd", return_value=[sys.executable, script_path]),
        ):
            started = W.start(
                "/models/qwen",
                self.sock_dir,
                tp_size=2,
                env={"FAKE_UUIDS": "g0,g1"},
                timeout=10.0,
                poll_interval=0.05,
            )
            proc_obj = started["process"]
            try:
                self.assertEqual(started["uuids"], ["g0", "g1"])
                ok_tp2 = W.probe(self.sock_dir, tp_size=2)
                self.assertTrue(ok_tp2["ok"])
                self.assertEqual(ok_tp2["uuids"], ["g0", "g1"])

                too_many = W.probe(self.sock_dir, tp_size=5)
                self.assertFalse(too_many["ok"])
                self.assertIn("requires 5 visible GPUs", too_many["error"])
            finally:
                stopped = W.stop(started["pid"], grace=3.0, reap=proc_obj.poll)
                proc_obj.wait(timeout=5.0)
            self.assertIsInstance(stopped, bool)

            # env["CUDA_VISIBLE_DEVICES"] overrides parent env in start().
            started_ov = W.start(
                "/models/qwen",
                self.sock_dir,
                tp_size=2,
                env={"CUDA_VISIBLE_DEVICES": "GPU-g2,GPU-g3", "FAKE_UUIDS": "g2,g3"},
                timeout=10.0,
                poll_interval=0.05,
            )
            proc_ov = started_ov["process"]
            try:
                self.assertEqual(started_ov["uuids"], ["g2", "g3"])
            finally:
                W.stop(started_ov["pid"], grace=3.0, reap=proc_ov.poll)
                proc_ov.wait(timeout=5.0)

    def test_start_timeout_passes_reap_to_stop_and_avoids_zombie_hang(self):
        import time as _time

        script_path = os.path.join(self.sock_dir, "fake_daemon.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(_FAKE_DAEMON_SCRIPT)

        with (
            mock.patch("services.weights.daemon_cmd", return_value=[sys.executable, script_path]),
            mock.patch("services.weights.stop", wraps=W.stop) as m_stop,
        ):
            t0 = _time.monotonic()
            with self.assertRaises(TimeoutError):
                W.start(
                    "/models/qwen",
                    self.sock_dir,
                    uuids=["GPU-slow-0"],
                    env={"FAKE_UUIDS": "slow-0", "FAKE_DELAY": "5.0"},
                    timeout=0.15,
                    poll_interval=0.03,
                )
            elapsed = _time.monotonic() - t0
        m_stop.assert_called_once()
        self.assertTrue(callable(m_stop.call_args.kwargs.get("reap")))
        self.assertLess(elapsed, 1.5)


if __name__ == "__main__":
    unittest.main()
