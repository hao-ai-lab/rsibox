"""CPU unit tests for `adapter.snapshot_hooks`, `adapter.client`, and `VllmLab` sleep/shadow hooks.

Runs real `ThreadingHTTPServer` instances speaking vLLM 0.33.0 dev endpoints (`/v1/completions`,
`/pause`, `/sleep`, `/release_kv_cache_memory`, `/is_sleeping`, `/wake_up`, `/resume`), real
`rsibox.ipc.FileChannelServer` instances serving real `adapter.host.Host` objects, real
`DynamoK8sSnapshotDriver` sentinel-file handshakes, and real `StandbyShadowDriver` promotion.
"""
from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rsibox import ipc
from rsibox.warm import (
    DefaultRespawnDriver,
    DynamoK8sSnapshotDriver,
    QuiesceHooks,
    QuiesceReceipt,
    StoreProbe,
)

from . import (
    VllmQuiesceHooks,
    VllmShadowPromoter,
    client,
    conformance_lab,
    make_shadow_recovery_driver,
    park_shadow_server,
)
from .test_conformance_lab import FAKE_SOURCES, FakeBackend


class _VllmDevServerState:
    """Mutable state machine matching vLLM 0.33.0 dev sleep/pause routes."""

    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.paused = False
        self.sleeping_tags: set[str] = set()
        self.requests: list[tuple[str, str, dict[str, list[str]]]] = []
        self.fail_endpoints: dict[str, tuple[int, str]] = {}
        self.override_is_sleeping: bool | None = None
        self.lock = threading.Lock()


def _make_vllm_handler(state: _VllmDevServerState):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            return

        def _send_json(self, status_code: int, payload: dict) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _handle(self, method: str) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            route = parsed.path
            query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(length) if length > 0 else b""
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}

            with state.lock:
                state.requests.append((method, route, query))
                injected = state.fail_endpoints.get(route)
                if injected is not None:
                    self._send_json(injected[0], {"error": injected[1]})
                    return

                if method == "GET" and route == "/is_sleeping":
                    sleeping = (
                        state.override_is_sleeping
                        if state.override_is_sleeping is not None
                        else bool(state.paused or state.sleeping_tags)
                    )
                    self._send_json(200, {"is_sleeping": sleeping})
                    return

                if method == "POST" and route == "/v1/completions":
                    if state.paused or state.sleeping_tags:
                        self._send_json(503, {"error": "engine is paused or sleeping"})
                        return
                    prompt = body.get("prompt", "")
                    max_tokens = int(body.get("max_tokens", 2))
                    output = state.backend.complete([prompt], max_tokens)["worker0"][0]
                    self._send_json(200, {"choices": [{"text": output}]})
                    return

                if method == "POST" and route == "/pause":
                    mode = (query.get("mode") or ["abort"])[0]
                    if mode not in ("keep", "abort", "wait"):
                        self._send_json(400, {"error": f"invalid pause mode: {mode}"})
                        return
                    state.paused = True
                    self._send_json(200, {"status": "paused"})
                    return

                if method == "POST" and route == "/sleep":
                    level = int((query.get("level") or ["1"])[0])
                    mode = (query.get("mode") or ["abort"])[0]
                    if level not in (0, 1, 2) or mode not in ("keep", "abort", "wait"):
                        self._send_json(400, {"error": "invalid sleep args"})
                        return
                    state.paused = True
                    if level >= 1:
                        state.sleeping_tags |= {"weights", "kv_cache"}
                    self._send_json(200, {"status": "sleeping", "level": level})
                    return

                if method == "POST" and route == "/release_kv_cache_memory":
                    if not state.paused:
                        self._send_json(500, {"error": "release_kv_cache_memory requires pause first"})
                        return
                    if state.sleeping_tags:
                        self._send_json(500, {"error": "executor already sleeping"})
                        return
                    state.sleeping_tags.add("kv_cache")
                    self._send_json(200, {"status": "kv_cache_released"})
                    return

                if method == "POST" and route == "/wake_up":
                    tags = query.get("tags")
                    if tags:
                        for tag in tags:
                            if tag not in state.sleeping_tags:
                                self._send_json(500, {"error": f"tag {tag} not sleeping"})
                                return
                            state.sleeping_tags.remove(tag)
                    else:
                        if not state.sleeping_tags and not state.paused:
                            self._send_json(500, {"error": "GMS managers are not unmapped"})
                            return
                        state.sleeping_tags.clear()
                    fully_awake = not state.sleeping_tags
                    if fully_awake:
                        state.paused = False
                    self._send_json(
                        200, {"status": "awake" if fully_awake else "sleeping", "tags": tags}
                    )
                    return

                if method == "POST" and route == "/resume":
                    state.paused = False
                    self._send_json(200, {"status": "resumed"})
                    return

                self._send_json(404, {"error": f"unknown route {method} {route}"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


class _LiveFakeVllmInstance:
    """Real localhost HTTP vLLM dev server + real `FileChannelServer` over `adapter.host.Host`."""

    def __init__(self, pid: int = 10001) -> None:
        self.backend = FakeBackend()
        self.state = _VllmDevServerState(self.backend)
        self.kernel_dir = tempfile.mkdtemp(prefix="rsibox-vllm-snap-kdir-")

        def dispatch_control(command: dict) -> dict:
            # Mirror `host._pre_step`: bind `loop_thread` to the serving thread.
            self.backend.host.loop_thread = threading.current_thread()
            return self.backend.ctl(command)

        self.channel = ipc.FileChannelServer(self.kernel_dir, dispatch_control, poll_s=0.005)
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_vllm_handler(self.state))
        self.port = int(self.httpd.server_address[1])
        self._http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._http_thread.start()
        self._alive = True
        self.stopped_calls = 0
        self.pid = pid

        instance = self

        class Handle:
            def __init__(self) -> None:
                self.pid = instance.pid
                self.port = instance.port
                self.kernel_dir = instance.kernel_dir
                self.start_s = 0.01

            def alive(self) -> bool:
                return instance._alive

            def stop(self) -> None:
                instance.stopped_calls += 1
                instance.stop_server_only()

        self.handle = Handle()

    def stop_server_only(self) -> None:
        if self._alive:
            self._alive = False
            self.channel.stop()
            self.httpd.shutdown()
            self.httpd.server_close()

    def close(self) -> None:
        self.stop_server_only()
        self.backend.close()
        shutil.rmtree(self.kernel_dir, ignore_errors=True)


def _attach_live_handle_to_lab(
    instance: _LiveFakeVllmInstance,
    *,
    weights_backend: str = "dynamo_gms",
    weight_store=None,
    recovery_driver=None,
    share_as_class_shared: bool = False,
) -> conformance_lab.VllmLab:
    lab = conformance_lab.VllmLab.__new__(conformance_lab.VllmLab)
    lab.receipts = None
    lab.check_timeout = 5.0
    lab.weight_store = weight_store
    lab.recovery_driver = recovery_driver
    lab.enable_sleep_mode = True
    lab._was_shared = False
    lab._backend = None
    lab._fault_txns = []
    lab._pending_inject = None
    lab._stack_files = []
    lab.weights_backend = weights_backend
    lab.server = instance.handle
    lab.port = instance.port
    lab.tp = 1
    lab.base_text = dict(FAKE_SOURCES)
    if share_as_class_shared:
        conformance_lab.VllmLab._shared = instance.handle
    lab.bridge = conformance_lab.HostBridge(lab._ctl)
    lab.session = conformance_lab.SS.Session(lab.bridge, conformance_lab.MODULES)
    return lab


class VllmQuiesceHooksTest(unittest.TestCase):
    def test_protocol_conformance_and_argument_validation(self):
        hooks = VllmQuiesceHooks(base_url="http://127.0.0.1:18500")
        self.assertIsInstance(hooks, QuiesceHooks)
        self.assertEqual(hooks.mutability, "readonly_weights")
        with self.assertRaises(ValueError):
            VllmQuiesceHooks(base_url="http://127.0.0.1:18500", sleep_level=3)
        with self.assertRaises(ValueError):
            VllmQuiesceHooks(base_url="http://127.0.0.1:18500", pause_mode="invalid")

    def test_snapshot_lifecycle_end_to_end_clears_stale_sentinels_and_resumes(self):
        instance = _LiveFakeVllmInstance()
        self.addCleanup(instance.close)
        lab = _attach_live_handle_to_lab(instance, weights_backend="dynamo_gms")
        hooks = lab.quiesce_hooks(sleep_level=1, pause_mode="keep")

        with tempfile.TemporaryDirectory(prefix="rsibox-snap-ctrl-") as control_dir:
            control_path = Path(control_dir)
            # Pre-seed stale sentinels to confirm step 1 clears them before warmup/quiesce.
            (control_path / "ready-for-snapshot").write_text("stale\n")
            (control_path / "restore-complete").write_text("stale\n")
            (control_path / "vllm-restore-ready").write_text("stale\n")

            seen_ready = threading.Event()
            observed_sleeping_at_checkpoint: list[bool] = []

            def simulate_snapshot_agent():
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    ready_file = control_path / "ready-for-snapshot"
                    try:
                        content = ready_file.read_text()
                    except FileNotFoundError:
                        content = ""
                    if content == "ok\n":
                        seen_ready.set()
                        observed_sleeping_at_checkpoint.append(
                            bool(instance.state.paused and instance.state.sleeping_tags)
                        )
                        (control_path / "restore-complete").write_text("restored\n")
                        return
                    time.sleep(0.01)

            watcher = threading.Thread(target=simulate_snapshot_agent, daemon=True)
            watcher.start()

            driver = DynamoK8sSnapshotDriver(
                engine_name="vllm", control_dir=control_dir, wait_timeout_s=5.0, poll_s=0.01
            )
            result = driver.checkpoint_and_wait(hooks)
            watcher.join(timeout=2.0)

            self.assertTrue(result.ok, result.to_dict())
            self.assertTrue(seen_ready.is_set())
            self.assertEqual(observed_sleeping_at_checkpoint, [True])
            self.assertTrue((control_path / "vllm-restore-ready").exists())
            self.assertTrue(result.quiesce.ok)
            self.assertEqual(result.quiesce.released, ("weights", "kv_cache"))
            self.assertEqual(result.quiesce.mutability, "readonly_weights")
            self.assertEqual(result.resume_details["resumed"], ["weights", "kv_cache"])
            self.assertTrue(result.resume_details["host_ping_ok"])
            self.assertFalse(instance.state.paused)
            self.assertEqual(instance.state.sleeping_tags, set())

            recorded = [(method, route, query) for method, route, query in instance.state.requests]
            self.assertEqual(
                [(method, route) for method, route, _ in recorded],
                [
                    ("POST", "/v1/completions"),
                    ("POST", "/pause"),
                    ("POST", "/sleep"),
                    ("GET", "/is_sleeping"),
                    ("POST", "/wake_up"),
                    ("POST", "/resume"),
                    ("GET", "/is_sleeping"),
                ],
            )
            self.assertEqual(recorded[1][2], {"mode": ["keep"], "clear_cache": ["false"]})
            self.assertEqual(recorded[2][2], {"level": ["1"], "mode": ["keep"]})
            self.assertEqual(recorded[4][2], {})  # No `tags` param -> wakes all tags on GMSWorker

    def test_snapshot_lifecycle_release_kv_cache_memory_mode(self):
        instance = _LiveFakeVllmInstance()
        self.addCleanup(instance.close)
        lab = _attach_live_handle_to_lab(instance, weights_backend="none")
        hooks = lab.quiesce_hooks(release_kv_only=True)

        with tempfile.TemporaryDirectory(prefix="rsibox-snap-kv-") as control_dir:
            control_path = Path(control_dir)

            def trigger_restore():
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if (control_path / "ready-for-snapshot").exists():
                        (control_path / "restore-complete").write_text("restored\n")
                        return
                    time.sleep(0.01)

            watcher = threading.Thread(target=trigger_restore, daemon=True)
            watcher.start()
            driver = DynamoK8sSnapshotDriver(
                engine_name="vllm", control_dir=control_dir, wait_timeout_s=5.0, poll_s=0.01
            )
            result = driver.checkpoint_and_wait(hooks)
            watcher.join(timeout=2.0)

            self.assertTrue(result.ok, result.to_dict())
            self.assertEqual(result.quiesce.released, ("kv_cache",))
            wake_calls = [
                query for _method, route, query in instance.state.requests if route == "/wake_up"
            ]
            self.assertEqual(wake_calls, [{"tags": ["kv_cache"]}])
            self.assertIn(("POST", "/release_kv_cache_memory", {}), instance.state.requests)

    def test_criu_cleared_kernel_dir_recovers_via_min_seq_floor(self):
        """Proves CRIU restore with an empty `RSIBOX_KERNEL_DIR` does not stall on `seq <= _last`."""
        instance = _LiveFakeVllmInstance()
        self.addCleanup(instance.close)
        lab = _attach_live_handle_to_lab(instance)
        hooks = lab.quiesce_hooks(timeout=1.5)

        hooks.warmup_for_snapshot()
        receipt = hooks.quiesce_for_snapshot()
        self.assertTrue(receipt.ok, receipt)
        self.assertGreater(instance.channel._last, 0)

        # Simulate CRIU restore onto a fresh emptyDir: ctl.json & reply/ vanish while
        # the restored `FileChannelServer` keeps its pre-checkpoint in-memory `_last`.
        os.unlink(os.path.join(instance.kernel_dir, "ctl.json"))
        shutil.rmtree(os.path.join(instance.kernel_dir, "reply"))

        # Without the `min_seq` floor, a plain `file_channel_call` starts at seq=1 <= _last and times out.
        naive_reply = ipc.file_channel_call(instance.kernel_dir, {"op": "ping"}, timeout=0.15, poll_s=0.005)
        self.assertFalse(naive_reply["ok"])
        self.assertTrue(naive_reply.get("lost"))

        # `resume_after_restore` advances sequence past `ctl_seq` in `receipt.token` and succeeds!
        resumed = hooks.resume_after_restore(receipt)
        self.assertTrue(resumed["ok"], resumed)
        self.assertTrue(resumed["host_ping_ok"])
        self.assertGreater(client.current_seq(instance.kernel_dir), instance.channel._last - 1)

    def test_fail_closed_pause_failure_writes_no_ready_sentinel(self):
        instance = _LiveFakeVllmInstance()
        self.addCleanup(instance.close)
        instance.state.fail_endpoints["/pause"] = (500, "pause exploded")
        lab = _attach_live_handle_to_lab(instance)
        hooks = lab.quiesce_hooks()

        with tempfile.TemporaryDirectory() as control_dir:
            driver = DynamoK8sSnapshotDriver(
                engine_name="vllm", control_dir=control_dir, wait_timeout_s=1.0, poll_s=0.01
            )
            result = driver.checkpoint_and_wait(hooks)
            self.assertFalse(result.ok)
            self.assertFalse((Path(control_dir) / "ready-for-snapshot").exists())
            self.assertIn("pause failed", result.error or "")
            routes = [route for _method, route, _query in instance.state.requests]
            self.assertNotIn("/sleep", routes)

    def test_fail_closed_sleep_failure_rolls_back_with_resume_and_writes_no_ready_sentinel(self):
        instance = _LiveFakeVllmInstance()
        self.addCleanup(instance.close)
        instance.state.fail_endpoints["/sleep"] = (500, "sleep exploded")
        lab = _attach_live_handle_to_lab(instance)
        hooks = lab.quiesce_hooks()

        with tempfile.TemporaryDirectory() as control_dir:
            driver = DynamoK8sSnapshotDriver(
                engine_name="vllm", control_dir=control_dir, wait_timeout_s=1.0, poll_s=0.01
            )
            result = driver.checkpoint_and_wait(hooks)
            self.assertFalse(result.ok)
            self.assertFalse((Path(control_dir) / "ready-for-snapshot").exists())
            routes = [route for _method, route, _query in instance.state.requests]
            self.assertIn("/pause", routes)
            self.assertIn("/sleep", routes)
            self.assertEqual(routes[-1], "/resume")
            self.assertFalse(instance.state.paused)

    def test_fail_closed_is_sleeping_false_rolls_back_and_writes_no_ready_sentinel(self):
        instance = _LiveFakeVllmInstance()
        self.addCleanup(instance.close)
        instance.state.override_is_sleeping = False
        lab = _attach_live_handle_to_lab(instance)
        hooks = lab.quiesce_hooks()

        with tempfile.TemporaryDirectory() as control_dir:
            driver = DynamoK8sSnapshotDriver(
                engine_name="vllm", control_dir=control_dir, wait_timeout_s=1.0, poll_s=0.01
            )
            result = driver.checkpoint_and_wait(hooks)
            self.assertFalse(result.ok)
            self.assertFalse((Path(control_dir) / "ready-for-snapshot").exists())
            self.assertIn("quiesce verification failed", result.error or "")
            self.assertEqual(instance.state.requests[-1][1], "/resume")

    def test_fail_closed_wake_up_failure_writes_no_restore_ready_sentinel(self):
        instance = _LiveFakeVllmInstance()
        self.addCleanup(instance.close)
        instance.state.fail_endpoints["/wake_up"] = (500, "GMS remap failed")
        lab = _attach_live_handle_to_lab(instance)
        hooks = lab.quiesce_hooks()

        with tempfile.TemporaryDirectory() as control_dir:
            control_path = Path(control_dir)

            def trigger_restore():
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if (control_path / "ready-for-snapshot").exists():
                        (control_path / "restore-complete").write_text("restored\n")
                        return
                    time.sleep(0.01)

            watcher = threading.Thread(target=trigger_restore, daemon=True)
            watcher.start()
            driver = DynamoK8sSnapshotDriver(
                engine_name="vllm", control_dir=control_dir, wait_timeout_s=5.0, poll_s=0.01
            )
            result = driver.checkpoint_and_wait(hooks)
            watcher.join(timeout=2.0)

            self.assertFalse(result.ok)
            self.assertFalse((control_path / "vllm-restore-ready").exists())
            self.assertIn("wake_up failed", result.error or "")

    def test_fail_closed_post_restore_host_ping_failure_writes_no_restore_ready_sentinel(self):
        instance = _LiveFakeVllmInstance()
        self.addCleanup(instance.close)
        lab = _attach_live_handle_to_lab(instance)
        hooks = lab.quiesce_hooks(timeout=0.3)

        with tempfile.TemporaryDirectory() as control_dir:
            control_path = Path(control_dir)

            def break_host_then_restore():
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if (control_path / "ready-for-snapshot").exists():
                        # Stop adapter host file channel before restore completes.
                        instance.channel.stop()
                        (control_path / "restore-complete").write_text("restored\n")
                        return
                    time.sleep(0.01)

            watcher = threading.Thread(target=break_host_then_restore, daemon=True)
            watcher.start()
            driver = DynamoK8sSnapshotDriver(
                engine_name="vllm", control_dir=control_dir, wait_timeout_s=5.0, poll_s=0.01
            )
            result = driver.checkpoint_and_wait(hooks)
            watcher.join(timeout=2.0)

            self.assertFalse(result.ok)
            self.assertFalse((control_path / "vllm-restore-ready").exists())
            self.assertIn("host check failed", result.error or "")

    def test_resume_after_restore_refreshes_server_start_ticks_only_after_host_ping(self):
        """CRIU restore preserves PID but assigns a new /proc/<pid>/stat starttime."""
        instance = _LiveFakeVllmInstance(pid=os.getpid())
        self.addCleanup(instance.close)
        current_ticks = [154497550]

        with (
            mock.patch.object(conformance_lab.engine, "_start_time", side_effect=lambda _pid: current_ticks[0]),
            mock.patch.object(conformance_lab, "_alive", return_value=True),
        ):
            server_handle = conformance_lab._ServerHandle(
                os.getpid(), instance.port, instance.kernel_dir, 0.01, ""
            )
            lab = _attach_live_handle_to_lab(instance)
            lab.server = server_handle
            hooks = lab.quiesce_hooks(timeout=0.5)

            hooks.warmup_for_snapshot()
            receipt = hooks.quiesce_for_snapshot()
            self.assertTrue(receipt.ok, receipt)
            self.assertTrue(lab.server.alive())

            # Simulate CRIU restore: same PID, new /proc/<pid>/stat field 22 starttime.
            current_ticks[0] = 154501520
            self.assertFalse(lab.server.alive())

            # Case 1: if post-restore host ping fails, start_ticks MUST NOT be refreshed.
            instance.channel.stop()
            failed_resume = hooks.resume_after_restore(receipt)
            self.assertFalse(failed_resume["ok"])
            self.assertEqual(lab.server.start_ticks, 154497550)
            self.assertFalse(lab.server.alive())

            # Re-enter restored sleeping state and bring host channel back up.
            instance.state.paused = True
            instance.state.sleeping_tags = {"weights", "kv_cache"}
            instance.channel = ipc.FileChannelServer(
                instance.kernel_dir,
                lambda command: (
                    setattr(instance.backend.host, "loop_thread", threading.current_thread())
                    or instance.backend.ctl(command)
                ),
                poll_s=0.005,
            )
            self.assertFalse(lab.server.alive())

            # Case 2: once wake_up + resume + adapter.host ping/fingerprint succeed,
            # start_ticks refreshes so `lab.server.alive()` and `lab.check()` work again.
            resumed = hooks.resume_after_restore(receipt)
            self.assertTrue(resumed["ok"], resumed)
            self.assertEqual(lab.server.start_ticks, 154501520)
            self.assertTrue(lab.server.alive())
            post_check = lab.check()
            self.assertTrue(post_check["ok"], post_check)


class VllmLabSleepModeAndShadowPromotionTest(unittest.TestCase):
    def tearDown(self):
        conformance_lab.VllmLab._shared = None

    def test_launch_environment_and_start_server_opt_in_sleep_mode(self):
        default_env = conformance_lab.VllmLab.launch_environment(["--model", "test-model"])
        self.assertNotIn("RSIBOX_VLLM_ENABLE_SLEEP_MODE", default_env)

        sleep_env = conformance_lab.VllmLab.launch_environment(
            ["--model", "test-model", "--enable-sleep-mode"]
        )
        self.assertEqual(sleep_env["RSIBOX_VLLM_ENABLE_SLEEP_MODE"], "1")

        # Verify `_start_server` subprocess env/argv for off, on, and already-in-extra cases.
        with tempfile.TemporaryDirectory() as work_dir:
            captured: list[tuple[list[str], dict[str, str]]] = []

            def fake_subprocess_run(command, *, capture_output, text, env, timeout):
                captured.append((list(command), dict(env)))
                return SimpleNamespace(returncode=0, stdout=json.dumps({"healthy_after_s": 0.1, "pid": 4242}))

            for enable_flag, extra_flags, expect_dev, expect_count in (
                (False, ["--enforce-eager"], False, 0),
                (True, ["--enforce-eager"], True, 1),
                (True, ["--enforce-eager", "--enable-sleep-mode"], True, 1),
            ):
                captured.clear()
                lab = conformance_lab.VllmLab.__new__(conformance_lab.VllmLab)
                lab.work_dir = work_dir
                lab.hooks_dir = os.path.join(work_dir, "hooks")
                lab.livelab_root = os.path.dirname(os.path.dirname(os.path.abspath(conformance_lab.__file__)))
                lab.farm_dir = os.path.join(work_dir, "farm")
                lab.python_bin = "/usr/bin/python3"
                lab.engine_py = os.path.join(lab.livelab_root, "services", "engine.py")
                lab.cache_root = os.path.join(work_dir, "cache")
                lab.vllm_bin = "/usr/bin/vllm"
                lab.weights_backend = ""
                lab.gms_socket_dir = ""
                lab.gms_pythonpath = ""
                lab.gms_read_only = False
                lab.gms_scratch_kv = False
                lab.model = "test-model"
                lab.tp = 1
                lab.extra = list(extra_flags)
                lab.enable_sleep_mode = enable_flag
                lab.server_key = ("test-model", enable_flag)

                with (
                    mock.patch.dict(os.environ, {}, clear=False),
                    mock.patch.object(conformance_lab.subprocess, "run", side_effect=fake_subprocess_run),
                    mock.patch.object(conformance_lab.engine, "load_engine_state", return_value={}),
                    mock.patch.object(conformance_lab.engine, "_start_time", return_value=1),
                    mock.patch.object(conformance_lab, "_http_post", return_value={"choices": [{"text": "ok"}]}),
                    mock.patch.object(
                        conformance_lab.client,
                        "send",
                        return_value={"ok": True, "parallel": {"world": 1, "expected": [0]}},
                    ),
                ):
                    os.environ.pop("VLLM_SERVER_DEV_MODE", None)
                    server_handle = lab._start_server(18500, "live", use_farm=False)
                    self.assertEqual(server_handle.pid, 4242)

                command, child_env = captured[0]
                self.assertEqual(command.count("--enable-sleep-mode"), expect_count)
                if expect_dev:
                    self.assertEqual(child_env.get("VLLM_SERVER_DEV_MODE"), "1")
                else:
                    self.assertNotIn("VLLM_SERVER_DEV_MODE", child_env)

    def test_server_key_changes_when_enable_sleep_mode_toggles(self):
        with tempfile.TemporaryDirectory() as work_dir:
            fake_site = os.path.join(work_dir, "site")
            for module_name, source_text in FAKE_SOURCES.items():
                rel_path = module_name.replace(".", os.sep) + ".py"
                full_path = os.path.join(fake_site, rel_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                Path(full_path).write_text(source_text)

            def start_stub(self_lab, port, tag, use_farm):
                handle = SimpleNamespace(alive=lambda: True, stop=lambda: None, port=port, key=self_lab.server_key)
                return handle

            base_env = {
                "RSIBOX_VLLM_MODEL": "test-model",
                "RSIBOX_VLLM_WORK": os.path.join(work_dir, "labwork"),
                "RSIBOX_VLLM_PORT": "18500",
            }
            with (
                mock.patch.dict(os.environ, base_env, clear=False),
                mock.patch.object(conformance_lab, "_site_packages", return_value=fake_site),
                mock.patch.object(conformance_lab.VllmLab, "_start_server", start_stub),
                mock.patch.object(conformance_lab.VllmLab, "_ctl", return_value={"ok": True, "workers": {"ok": True, "code": {}}, "core": {"ok": True, "code": {}}}),
            ):
                os.environ.pop("RSIBOX_VLLM_ENABLE_SLEEP_MODE", None)
                lab_off = conformance_lab.VllmLab(enable_sleep_mode=False)
                lab_on = conformance_lab.VllmLab(enable_sleep_mode=True)
                self.assertFalse(lab_off.enable_sleep_mode)
                self.assertTrue(lab_on.enable_sleep_mode)
                self.assertNotEqual(lab_off.server_key, lab_on.server_key)
                self.assertIsNot(lab_off.server, lab_on.server)

    def test_standby_shadow_promotion_end_to_end_on_vllm_lab(self):
        primary = _LiveFakeVllmInstance(pid=20001)
        shadow = _LiveFakeVllmInstance(pid=20002)
        self.addCleanup(primary.close)
        self.addCleanup(shadow.close)

        parked = park_shadow_server(shadow.handle, sleep_level=1, pause_mode="keep", timeout=5.0)
        self.assertTrue(parked["ok"], parked)
        self.assertTrue(parked["is_sleeping"])
        self.assertEqual(shadow.state.sleeping_tags, {"weights", "kv_cache"})

        replenished_calls: list[tuple[int, object]] = []
        driver = make_shadow_recovery_driver(
            shadow.handle,
            replenish_fn=lambda lab_obj, store: replenished_calls.append((lab_obj.port, store)),
            timeout=5.0,
        )
        lab = _attach_live_handle_to_lab(
            primary,
            weights_backend="dynamo_gms",
            recovery_driver=driver,
            share_as_class_shared=True,
        )

        baseline_check = lab.check()
        self.assertTrue(baseline_check["ok"], baseline_check)

        # Poison primary session to mirror a real crash/hang failover trigger.
        lab.session.fault("participant_failed", "primary worker crashed", "worker answers")
        self.assertEqual(lab.status()["state"], "poisoned")

        recovered = lab.reset("respawn")
        self.assertTrue(recovered["ok"], recovered)
        self.assertEqual(recovered["driver"], "standby_shadow")
        self.assertEqual(recovered["promoted_port"], shadow.port)
        self.assertEqual(recovered["promoted_pid"], shadow.pid)
        self.assertTrue(recovered["host_ping_ok"])

        # Primary was stopped; lab and VllmLab._shared switched to shadow; replenish_fn fired.
        self.assertFalse(primary.handle.alive())
        self.assertEqual(primary.stopped_calls, 1)
        self.assertIs(lab.server, shadow.handle)
        self.assertIs(conformance_lab.VllmLab._shared, shadow.handle)
        self.assertEqual(lab.port, shadow.port)
        self.assertEqual(len(replenished_calls), 1)
        self.assertEqual(replenished_calls[0][0], shadow.port)
        self.assertEqual(lab.status()["state"], "base")

        # Verify check(), install(), and uninstall() all work end-to-end on the promoted shadow.
        post_check = lab.check()
        self.assertTrue(post_check["ok"], post_check)
        self.assertEqual(post_check["output"], baseline_check["output"])

        edit_result = lab.install(lab.patches()["visible"])
        self.assertTrue(edit_result["ok"], edit_result)
        edited_check = lab.check()
        self.assertTrue(edited_check["ok"], edited_check)
        self.assertTrue(edited_check["proven"])
        self.assertNotEqual(edited_check["output"], baseline_check["output"])

        undo_result = lab.uninstall()
        self.assertTrue(undo_result["ok"], undo_result)
        restored_check = lab.check()
        self.assertTrue(restored_check["ok"], restored_check)
        self.assertEqual(restored_check["output"], baseline_check["output"])

    def test_standby_shadow_promotion_fails_closed_when_shadow_wake_up_fails(self):
        primary = _LiveFakeVllmInstance(pid=30001)
        shadow = _LiveFakeVllmInstance(pid=30002)
        self.addCleanup(primary.close)
        self.addCleanup(shadow.close)

        self.assertTrue(park_shadow_server(shadow.handle)["ok"])
        shadow.state.fail_endpoints["/wake_up"] = (500, "GMS RO lock timeout")

        replenished: list[bool] = []
        driver = make_shadow_recovery_driver(
            shadow.handle, replenish_fn=lambda _lab, _store: replenished.append(True), timeout=5.0
        )
        lab = _attach_live_handle_to_lab(primary, recovery_driver=driver)
        result = lab.reset("respawn")

        self.assertFalse(result["ok"])
        self.assertEqual(result["driver"], "standby_shadow")
        self.assertIn("wake_up failed", result.get("error", ""))
        self.assertIs(lab.server, primary.handle)
        self.assertEqual(replenished, [])

    def test_default_respawn_driver_fails_closed_on_unhealthy_weight_store(self):
        class FailingStore:
            name = "dynamo_gms"

            def probe(self, **_kwargs):
                return StoreProbe(ok=False, backend="dynamo_gms", error="GMS weights socket uncommitted")

        backend = FakeBackend()
        self.addCleanup(backend.close)
        lab = conformance_lab.VllmLab(
            check_timeout=1.0,
            backend=backend,
            base_texts=FAKE_SOURCES,
            weight_store=FailingStore(),
            recovery_driver=DefaultRespawnDriver(),
        )
        with mock.patch.object(lab, "_stop_engine") as stop_mock, mock.patch.object(lab, "_spawn") as spawn_mock:
            reply = lab.reset("respawn")
            self.assertFalse(reply["ok"])
            self.assertEqual(reply["driver"], "default_respawn")
            self.assertIn("uncommitted", reply["error"])
            stop_mock.assert_not_called()
            spawn_mock.assert_not_called()

    def test_vllm_ctl_fast_fails_when_server_exits_mid_request(self):
        """Regression for the 180 s SIGKILL race: if the vLLM server is alive when `_ctl()`
        starts but gets SIGKILLed before writing a reply, `_ctl()` must return `lost=True`
        within milliseconds instead of stalling for the full `check_timeout`."""
        with tempfile.TemporaryDirectory(prefix="rsibox-vllm-ctl-kill-") as kdir:
            sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            self.addCleanup(lambda: (sleeper.kill(), sleeper.wait()) if sleeper.poll() is None else None)

            class _FakeProcHandle:
                kernel_dir = kdir
                port = 19999

                def alive(self) -> bool:
                    return sleeper.poll() is None

            lab = conformance_lab.VllmLab.__new__(conformance_lab.VllmLab)
            lab._backend = None
            lab.check_timeout = 60.0
            lab.server = _FakeProcHandle()

            ctl_path = os.path.join(kdir, "ctl.json")
            stop_killer = threading.Event()

            def kill_mid_call() -> None:
                while not stop_killer.is_set() and not os.path.exists(ctl_path):
                    time.sleep(0.002)
                if not stop_killer.is_set():
                    sleeper.kill()
                    sleeper.wait()

            killer = threading.Thread(target=kill_mid_call, daemon=True)
            killer.start()
            t0 = time.perf_counter()
            try:
                reply = lab._ctl({"op": "status"})
                elapsed_s = time.perf_counter() - t0
            finally:
                stop_killer.set()
                killer.join(timeout=3.0)

            self.assertFalse(reply["ok"])
            self.assertTrue(reply.get("lost"))
            self.assertIn("exited before reply", reply.get("error", ""))
            self.assertLess(elapsed_s, 2.0, f"_ctl stalled for {elapsed_s:.3f}s instead of fast-failing")


if __name__ == "__main__":
    unittest.main()
