"""Unit, concurrency, and real `os.fork()` + `AF_UNIX` `SCM_RIGHTS` tests for `rsibox.services.zygote`."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

LIVELAB_DIR = Path(__file__).resolve().parents[2]
if str(LIVELAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVELAB_DIR))

import forksafe  # noqa: E402
from rsibox.services import (  # noqa: E402
    ForkPoisonedError,
    SpawnedChild,
    SpawnError,
    ZygoteClosedError,
    ZygoteError,
    ZygoteProfile,
    ZygoteService,
    ZygoteState,
    list_profiles,
    profile_for_engine,
)

_FAKE_PROBE = lambda: [{"name": "FakeGPU", "major": 10, "minor": 0, "total_memory": 1024, "multi_processor_count": 8}]


def _make_fake_cuda() -> types.SimpleNamespace:
    def _uninit(*_args, **_kwargs):
        raise AssertionError("unpatched fake CUDA queried while uninitialized")

    return types.SimpleNamespace(
        is_initialized=lambda: False,
        get_device_capability=_uninit,
        current_device=_uninit,
        get_device_properties=_uninit,
    )


def _read_pipe_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(fd, 4096):
        chunks.append(chunk)
    return b"".join(chunks)


class ZygoteProfileTests(unittest.TestCase):
    def test_all_ten_engines_and_aliases(self) -> None:
        """All 10 canonical RSIBox engines, `vllm` fork default, and aliases resolve cleanly."""
        canonical = (
            "vllm", "sglang", "trtllm", "megatron", "torchtitan",
            "verl", "fastvideo", "transformers", "miles", "nemo-rl",
        )
        profiles = list_profiles()
        self.assertEqual([p.engine for p in profiles], list(canonical))
        self.assertEqual(profile_for_engine("vllm").env_defaults.get("VLLM_WORKER_MULTIPROC_METHOD"), "fork")
        for eng in canonical:
            prof = profile_for_engine(eng)
            self.assertEqual(prof.engine, eng)
            self.assertTrue(prof.preload_modules)
            prof.env_defaults["MUTATED"] = "1"
            self.assertNotIn("MUTATED", profile_for_engine(eng).env_defaults)

        alias_cases = {
            "vLLM": "vllm", "SGLang": "sglang", "trt-llm": "trtllm", "TensorRT-LLM": "trtllm",
            "tensorrt_llm": "trtllm", "Megatron-LM": "megatron", "rsibox_megatron": "megatron",
            "torch_titan": "torchtitan", "veRL": "verl", "fast_video": "fastvideo",
            "hf-transformers": "transformers", "nemo_rl": "nemo-rl", "nemorl": "nemo-rl",
        }
        for raw, expected in alias_cases.items():
            self.assertEqual(profile_for_engine(raw).engine, expected)
        for bad in ("", "   ", "unknown-engine"):
            with self.assertRaises(ValueError):
                profile_for_engine(bad)


class ZygoteGuardAndPoisonTests(unittest.TestCase):
    def tearDown(self) -> None:
        if hasattr(gc, "unfreeze"):
            gc.unfreeze()

    def test_fail_closed_fork_poisoned_on_arm_and_spawn(self) -> None:
        """`arm_guards()` and `spawn_child()` fail closed with `ForkPoisonedError` when poisoned."""
        prof = ZygoteProfile("vllm", preload_modules=("json",), freeze_gc=False)
        with mock.patch.dict(os.environ, {}, clear=False):
            with mock.patch.object(forksafe, "fork_poisoned", return_value=True), \
                 mock.patch.object(forksafe, "poison_reasons", return_value=["bad fork"]):
                with self.assertRaises(ForkPoisonedError):
                    with ZygoteService(prof) as _svc:
                        pass
                svc = ZygoteService(prof)
                try:
                    with self.assertRaises(ForkPoisonedError):
                        svc.arm_guards(capability=(10, 0), probe=_FAKE_PROBE)
                    with self.assertRaises(ForkPoisonedError):
                        svc.start(capability=(10, 0), probe=_FAKE_PROBE)
                finally:
                    svc.close()

            svc_ok = ZygoteService(prof, cuda_module_fn=_make_fake_cuda)
            try:
                svc_ok.arm_guards(capability=(10, 0), probe=_FAKE_PROBE, preload=False)
                with self.assertRaises(ZygoteError):
                    svc_ok.arm_guards(capability=(10, 0), probe=_FAKE_PROBE, preload=False)
                svc_ok.start(capability=(10, 0), probe=_FAKE_PROBE, preload=False)
                with mock.patch.object(forksafe, "fork_poisoned", return_value=True), \
                     mock.patch.object(forksafe, "poison_reasons", return_value=["cuInit called"]):
                    with self.assertRaises(ForkPoisonedError):
                        svc_ok.spawn_child("worker")
                    state = svc_ok.status()
                    self.assertTrue(state.poisoned)
                    self.assertIn("cuInit called", state.poison_reasons)
            finally:
                svc_ok.close()

    def test_arm_guards_triton_restore_cuda_init_check_and_rollback(self) -> None:
        """`arm_guards()` restores Triton pin after preload, checks `cuda_initialized()`, and rolls back on error."""
        fake_cuda = _make_fake_cuda()
        prof = ZygoteProfile("sglang", preload_modules=("json", "math"), freeze_gc=True)
        events: list[str] = []

        def _fake_pin_triton(**_kw):
            events.append("pin_triton")
            def _restore():
                events.append("restore_triton")
            return _restore

        def _importer(name: str):
            self.assertTrue(getattr(fake_cuda.get_device_capability, "_rsibox_forksafe", False))
            events.append(f"import:{name}")
            return __import__(name)

        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(forksafe, "pin_triton_target", side_effect=_fake_pin_triton):
            svc = ZygoteService(prof, cuda_module_fn=lambda: fake_cuda, importer=_importer)
            try:
                svc.arm_guards(capability=(10, 0), probe=_FAKE_PROBE)
                svc.start()
                self.assertEqual(events, ["pin_triton", "import:json", "import:math", "restore_triton"])
                st = svc.status()
                self.assertIsInstance(st, ZygoteState)
                self.assertTrue(st.guards_armed and st.started)
                self.assertTrue(st.preload_report["json"]["loaded"])
            finally:
                svc.close()

        # Rollback on CUDA init during preload + restore_triton still called.
        events.clear()
        fake_cuda2 = _make_fake_cuda()
        prof2 = ZygoteProfile("sglang", preload_modules=("evil_cuda_mod",), env_defaults={"Z_TEST_ROLLBACK": "1"}, freeze_gc=False)
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(forksafe, "pin_triton_target", side_effect=_fake_pin_triton):
            os.environ.pop("Z_TEST_ROLLBACK", None)
            svc2 = ZygoteService(prof2, cuda_module_fn=lambda: fake_cuda2, importer=lambda _n: object())
            try:
                with mock.patch.object(forksafe, "cuda_initialized", side_effect=[False, True]):
                    with self.assertRaises(ForkPoisonedError):
                        svc2.arm_guards(capability=(10, 0), probe=_FAKE_PROBE)
                self.assertIn("restore_triton", events)
                self.assertFalse(getattr(fake_cuda2.get_device_capability, "_rsibox_forksafe", False))
                self.assertNotIn("Z_TEST_ROLLBACK", os.environ)
                self.assertFalse(svc2.status().guards_armed)
            finally:
                svc2.close()

    def test_unarmed_teardown_does_not_unpatch_armed_sibling(self) -> None:
        """Closing an unarmed or second service leaves pre-existing `torch_cuda` guards intact."""
        fake_cuda = _make_fake_cuda()
        prof = ZygoteProfile("vllm", preload_modules=(), freeze_gc=False)
        with mock.patch.dict(os.environ, {}, clear=False):
            svc_armed = ZygoteService(prof, cuda_module_fn=lambda: fake_cuda)
            svc_unarmed = ZygoteService(prof, cuda_module_fn=lambda: fake_cuda)
            try:
                svc_armed.arm_guards(capability=(10, 0), probe=_FAKE_PROBE, preload=False)
                self.assertTrue(getattr(fake_cuda.get_device_capability, "_rsibox_forksafe", False))
                svc_unarmed.close()
                self.assertTrue(getattr(fake_cuda.get_device_capability, "_rsibox_forksafe", False))
            finally:
                svc_armed.close()
            self.assertFalse(getattr(fake_cuda.get_device_capability, "_rsibox_forksafe", False))

        # Pre-existing forksafe.install(fake_cuda) + unset PYTORCH_NVML_BASED_CUDA_CHECK survives close() and rollback.
        fake_pre = _make_fake_cuda()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
            forksafe.install(fake_pre, capability=lambda: ((10, 0), 1))
            try:
                svc_pre = ZygoteService(ZygoteProfile("vllm", preload_modules=("evil_mod",), freeze_gc=False),
                                        cuda_module_fn=lambda: fake_pre, importer=lambda _n: object())
                try:
                    with mock.patch.object(forksafe, "cuda_initialized", side_effect=[False, True]):
                        with self.assertRaises(ForkPoisonedError):
                            svc_pre.arm_guards(capability=(10, 0), probe=_FAKE_PROBE)
                    self.assertTrue(getattr(fake_pre.get_device_capability, "_rsibox_forksafe", False))
                finally:
                    svc_pre.close()
                self.assertTrue(getattr(fake_pre.get_device_capability, "_rsibox_forksafe", False))

                svc_pre2 = ZygoteService(prof, cuda_module_fn=lambda: fake_pre)
                try:
                    svc_pre2.arm_guards(capability=(10, 0), probe=_FAKE_PROBE, preload=False)
                    self.assertEqual(os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"), "1")
                finally:
                    svc_pre2.close()
                self.assertTrue(getattr(fake_pre.get_device_capability, "_rsibox_forksafe", False))
                self.assertNotIn("PYTORCH_NVML_BASED_CUDA_CHECK", os.environ)
            finally:
                forksafe.unpatch(fake_pre, clear_env=True)


class ZygoteForkAndScmRightsTests(unittest.TestCase):
    def tearDown(self) -> None:
        if hasattr(gc, "unfreeze"):
            gc.unfreeze()

    def test_real_fork_scm_rights_fd_passing_and_parent_fd_isolation(self) -> None:
        """Real `os.fork()` + `AF_UNIX` `SCM_RIGHTS` transfers FDs, closes parent listener FD in child, and reaps."""
        fake_cuda = _make_fake_cuda()
        prof = ZygoteProfile("vllm", preload_modules=("json",), env_defaults={"OMP_NUM_THREADS": "1"}, freeze_gc=True)

        def _child_handler(payload: dict, fds: list[int]) -> int:
            listener_fd = int(os.environ["RSIBOX_PARENT_LISTENER_FD"])
            try:
                os.fstat(listener_fd)
                listener_open_in_child = True
            except OSError:
                listener_open_in_child = False
            out = {
                "pid": os.getpid(),
                "payload_pid": payload["pid"],
                "role": payload["role"],
                "engine": payload["engine"],
                "incarnation": payload["incarnation"],
                "env_tag": os.environ.get("RSIBOX_ZYGOTE_TEST_TAG"),
                "child_guard_report": payload["child_guard_report"],
                "affinity_report": payload["affinity_report"],
                "still_patched": getattr(fake_cuda.get_device_capability, "_rsibox_forksafe", False),
                "listener_open_in_child": listener_open_in_child,
            }
            os.write(fds[0], json.dumps(out).encode("utf-8"))
            return 0

        with mock.patch.dict(os.environ, {}, clear=False):
            svc = ZygoteService(prof, cuda_module_fn=lambda: fake_cuda)
            try:
                svc.start(capability=(9, 0), probe=_FAKE_PROBE)
                svc_sock = svc.socket_path
                assert svc_sock is not None and svc._listener is not None
                self.assertTrue(svc_sock.exists())
                r_fd, w_fd = os.pipe()
                try:
                    child = svc.spawn_child(
                        "actor",
                        env={"RSIBOX_ZYGOTE_TEST_TAG": "scm-ok", "RSIBOX_PARENT_LISTENER_FD": str(svc._listener.fileno())},
                        pass_fds=(w_fd,),
                        handler=_child_handler,
                    )
                finally:
                    os.close(w_fd)
                try:
                    raw = _read_pipe_all(r_fd)
                finally:
                    os.close(r_fd)

                self.assertIsInstance(child, SpawnedChild)
                self.assertNotEqual(child.pid, os.getpid())
                data = json.loads(raw.decode("utf-8"))
                self.assertEqual(data["pid"], child.pid)
                self.assertEqual(data["env_tag"], "scm-ok")
                self.assertFalse(data["still_patched"])
                self.assertFalse(data["listener_open_in_child"])
                self.assertEqual(
                    sorted(data["child_guard_report"]["torch_cuda"]),
                    ["current_device", "get_device_capability", "get_device_properties"],
                )
                self.assertEqual(svc.wait_child(child.pid, timeout_s=5.0), 0)
                self.assertEqual(svc.reap_finished()[child.pid], 0)
                self.assertEqual(svc.status().active_children, ())
            finally:
                svc.close()
            self.assertFalse(svc_sock.exists())
            self.assertTrue(svc.status().closed)

    def test_concurrent_spawns_nonzero_exit_max_children_and_closed_errors(self) -> None:
        """Concurrent spawns, non-zero exit codes, `max_children`, concurrent `teardown()`, and `ZygoteClosedError`."""
        prof = ZygoteProfile("megatron", preload_modules=(), freeze_gc=False)

        def _default_worker(payload: dict, fds: list[int]) -> int:
            if payload["role"] == "fail42":
                return 42
            if payload["role"] == "sleep":
                time.sleep(5.0)
            return 0

        with mock.patch.dict(os.environ, {}, clear=False):
            svc = ZygoteService(prof, worker_handler=_default_worker, max_children=4)
            try:
                with self.assertRaises(ZygoteError):
                    svc.spawn_child("before-start")
                svc.start(capability=(10, 0), probe=_FAKE_PROBE, preload=False)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda i: svc.spawn_child(f"rank-{i}", wait=True, timeout_s=5.0), range(4)))
                self.assertEqual(sorted(c.incarnation for c in results), [1, 2, 3, 4])
                self.assertEqual({c.exit_code for c in results}, {0})
                self.assertEqual(svc.spawn_child("fail42", wait=True, timeout_s=5.0).exit_code, 42)
            finally:
                svc.close()

            svc.close()
            for fn in (svc.arm_guards, svc.start, lambda: svc.spawn_child("after-close")):
                with self.assertRaises(ZygoteClosedError):
                    fn()

        with mock.patch.dict(os.environ, {}, clear=False):
            with ZygoteService(prof, worker_handler=_default_worker, max_children=2) as svc2:
                c1 = svc2.spawn_child("sleep")
                c2 = svc2.spawn_child("sleep")
                with self.assertRaises(SpawnError):
                    svc2.spawn_child("sleep")
                with ThreadPoolExecutor(max_workers=2) as pool:
                    r1, r2 = list(pool.map(lambda _: svc2.teardown(grace_s=0.1), range(2)))
                self.assertIn(c1.pid, r1)
                self.assertIn(c2.pid, r2)
                self.assertEqual(r1, r2)
                self.assertEqual(svc2.status().active_children, ())

    def test_stalled_client_early_child_death_and_fds_mismatch(self) -> None:
        """Stalled client cancels cleanly on teardown, child dying pre-report leaves 0 zombies, `fds_count` mismatch fails."""
        prof = ZygoteProfile("transformers", preload_modules=(), freeze_gc=False)
        with mock.patch.dict(os.environ, {}, clear=False):
            svc = ZygoteService(prof)
            try:
                svc.start(capability=(9, 0), probe=_FAKE_PROBE, preload=False)
                sock_path = svc.socket_path
                assert sock_path is not None

                # P2: fds_count mismatch (and bool fds_count) over raw AF_UNIX frame.
                for bad_fds_count in (2, True):
                    body = json.dumps({"role": "w", "fds_count": bad_fds_count}).encode("utf-8")
                    frame = struct.pack("!I", len(body)) + body
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw_cli:
                        raw_cli.settimeout(2.0)
                        raw_cli.connect(os.fspath(sock_path))
                        raw_cli.sendall(frame)
                        hdr = raw_cli.recv(4)
                        (rlen,) = struct.unpack("!I", hdr)
                        rep = json.loads(raw_cli.recv(rlen).decode("utf-8"))
                    self.assertFalse(rep["ok"])
                    self.assertIn("FD count mismatch", rep["error"])

                # P1-4: Child crashes before writing report_w -> SpawnError + 0 zombie left behind.
                with mock.patch.object(forksafe, "unpatch_child", side_effect=RuntimeError("boom-pre-report")):
                    with self.assertRaises(SpawnError):
                        svc.spawn_child("crash-pre-report", timeout_s=2.0)
                self.assertEqual(svc.status().active_children, ())

                # P1-3: Stalled client (1 byte sent, never completed) does not hang teardown().
                stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    stalled.connect(os.fspath(sock_path))
                    stalled.sendall(b"\x00")
                    t0 = time.monotonic()
                    svc.teardown(grace_s=0.1)
                    self.assertLess(time.monotonic() - t0, 1.5)
                finally:
                    stalled.close()
            finally:
                svc.close()

    def test_validation_child_process_error_and_start_failure_propagation(self) -> None:
        """NaN/inf/bool timeouts, fast `ChildProcessError` fail, external listener, and bad-listener `start()` fail."""
        prof = profile_for_engine("transformers")
        prof_plain = ZygoteProfile(prof.engine, preload_modules=(), freeze_gc=False)
        with tempfile.TemporaryDirectory(prefix="zygext_") as tdir:
            ext_sock = Path(tdir) / "ext.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
                srv.bind(os.fspath(ext_sock))
                srv.listen(4)
                with mock.patch.dict(os.environ, {}, clear=False):
                    svc_conc = ZygoteService(prof_plain)
                    try:
                        with ThreadPoolExecutor(max_workers=4) as pool:
                            paths = list(pool.map(lambda _: svc_conc.start(capability=(9, 0), probe=_FAKE_PROBE, preload=False), range(4)))
                        self.assertEqual(len(set(paths)), 1)
                    finally:
                        svc_conc.close()

                    # start()-vs-teardown() race: neither leaks FDs/threads nor leaves started=True, closed=True.
                    for _ in range(5):
                        svc_race = ZygoteService(prof_plain)
                        try:
                            with ThreadPoolExecutor(max_workers=2) as pool:
                                f_start = pool.submit(svc_race.start, capability=(9, 0), probe=_FAKE_PROBE, preload=False)
                                f_stop = pool.submit(svc_race.teardown, grace_s=0.1)
                                try:
                                    f_start.result(timeout=5.0)
                                except ZygoteClosedError:
                                    pass
                                f_stop.result(timeout=5.0)
                            st = svc_race.status()
                            self.assertTrue(st.closed)
                            self.assertFalse(st.started and st.closed)
                        finally:
                            svc_race.close()

                    svc = ZygoteService(prof_plain)
                    try:
                        svc.start(listener=srv, capability=(9, 0), probe=_FAKE_PROBE, preload=False)
                        c = svc.spawn_child("eval")
                        # Externally reap child -> wait_child raises SpawnError immediately (<0.5s).
                        os.waitpid(c.pid, 0)
                        t0 = time.monotonic()
                        with self.assertRaises(SpawnError):
                            svc.wait_child(c.pid, timeout_s=5.0)
                        self.assertLess(time.monotonic() - t0, 0.5)

                        for bad_tmo in (float("nan"), float("inf"), -1.0, 0.0, True):
                            with self.assertRaises(ValueError):
                                svc.spawn_child("x", timeout_s=bad_tmo)
                            with self.assertRaises(ValueError):
                                svc.wait_child(c.pid, timeout_s=bad_tmo)
                        for bad_grace in (float("nan"), float("inf"), -0.1, True):
                            with self.assertRaises(ValueError):
                                svc.teardown(grace_s=bad_grace)
                    finally:
                        svc.close()
                self.assertNotEqual(srv.fileno(), -1)

            # Closed listener & _serve_loop init failure both propagate ZygoteError cleanly.
            bad_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            bad_sock.close()
            with mock.patch.dict(os.environ, {}, clear=False):
                svc_bad = ZygoteService(prof_plain)
                try:
                    with self.assertRaises(ZygoteError):
                        svc_bad.start(listener=bad_sock, capability=(9, 0), probe=_FAKE_PROBE, preload=False)
                finally:
                    svc_bad.close()

            class _FailingListener:
                def getsockname(self) -> str:
                    return str(Path(tdir) / "fail.sock")

                def setblocking(self, _flag: bool) -> None:
                    raise RuntimeError("boom in _serve_loop")

            with mock.patch.dict(os.environ, {}, clear=False):
                svc_fail = ZygoteService(prof_plain)
                try:
                    with self.assertRaises(ZygoteError) as ctx:
                        svc_fail.start(listener=_FailingListener(), capability=(9, 0), probe=_FAKE_PROBE, preload=False)  # type: ignore[arg-type]
                    self.assertIn("boom in _serve_loop", str(ctx.exception))
                finally:
                    svc_fail.close()

            long_path = Path(tdir) / ("x" * 120 + ".sock")
            with mock.patch.dict(os.environ, {}, clear=False):
                svc_long = ZygoteService(prof_plain, socket_path=long_path)
                try:
                    with self.assertRaises(ZygoteError):
                        svc_long.start(capability=(9, 0), probe=_FAKE_PROBE, preload=False)
                finally:
                    svc_long.close()


if __name__ == "__main__":
    unittest.main()
