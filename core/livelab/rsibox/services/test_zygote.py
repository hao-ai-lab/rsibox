"""Unit, concurrency, and real `os.fork()` + `AF_UNIX` `SCM_RIGHTS` tests for `rsibox.services.zygote`."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import socket
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
        """All 10 canonical RSIBox engines and aliases resolve cleanly with isolated env dicts."""
        canonical = (
            "vllm", "sglang", "trt-llm", "megatron", "torchtitan",
            "verl", "fastvideo", "transformers", "miles", "nemo-rl",
        )
        profiles = list_profiles()
        self.assertEqual([p.engine for p in profiles], list(canonical))
        for eng in canonical:
            prof = profile_for_engine(eng)
            self.assertEqual(prof.engine, eng)
            self.assertTrue(prof.preload_modules)
            prof.env_defaults["MUTATED"] = "1"
            self.assertNotIn("MUTATED", profile_for_engine(eng).env_defaults)

        alias_cases = {
            "vLLM": "vllm",
            "SGLang": "sglang",
            "trtllm": "trt-llm",
            "TensorRT-LLM": "trt-llm",
            "tensorrt_llm": "trt-llm",
            "Megatron-LM": "megatron",
            "rsibox_megatron": "megatron",
            "torch_titan": "torchtitan",
            "veRL": "verl",
            "fast_video": "fastvideo",
            "hf-transformers": "transformers",
            "nemo_rl": "nemo-rl",
            "nemorl": "nemo-rl",
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
                svc = ZygoteService(prof)
                with self.assertRaises(ForkPoisonedError):
                    svc.arm_guards()
                with self.assertRaises(ForkPoisonedError):
                    svc.start()
                svc.close()

            # Clean start, then runtime becomes poisoned before spawn_child().
            with ZygoteService(prof, cuda_module_fn=_make_fake_cuda) as svc:
                svc.arm_guards(
                    capability=(10, 0),
                    probe=lambda: [{"name": "FakeGPU", "major": 10, "minor": 0, "total_memory": 1024, "multi_processor_count": 8}],
                    preload=False,
                )
                with mock.patch.object(forksafe, "fork_poisoned", return_value=True), \
                     mock.patch.object(forksafe, "poison_reasons", return_value=["cuInit called"]):
                    with self.assertRaises(ForkPoisonedError):
                        svc.spawn_child("worker")
                    state = svc.status()
                    self.assertTrue(state.poisoned)
                    self.assertIn("cuInit called", state.poison_reasons)

    def test_arm_guards_preloads_only_after_guards_and_detects_cuda_init(self) -> None:
        """`arm_guards()` installs `forksafe` guards before preloading and refuses CUDA-initializing imports."""
        fake_cuda = _make_fake_cuda()
        prof = ZygoteProfile("sglang", preload_modules=("json", "math"), freeze_gc=True)
        order: list[str] = []

        def _importer(name: str):
            self.assertTrue(getattr(fake_cuda.get_device_capability, "_rsibox_forksafe", False))
            order.append(name)
            return __import__(name)

        with mock.patch.dict(os.environ, {}, clear=False):
            with ZygoteService(prof, cuda_module_fn=lambda: fake_cuda, importer=_importer) as svc:
                st = svc.status()
                self.assertIsInstance(st, ZygoteState)
                self.assertTrue(st.guards_armed)
                self.assertTrue(st.started)
                self.assertEqual(order, ["json", "math"])
                self.assertTrue(st.preload_report["json"]["loaded"])
                self.assertTrue(st.gc_report.get("frozen_count", 0) >= 0)

        # If preloading a module initializes CUDA, arm_guards raises ForkPoisonedError.
        prof2 = ZygoteProfile("sglang", preload_modules=("evil_cuda_mod",), freeze_gc=False)
        with mock.patch.dict(os.environ, {}, clear=False):
            svc2 = ZygoteService(prof2, importer=lambda _name: object())
            try:
                with mock.patch.object(forksafe, "cuda_initialized", return_value=True):
                    with self.assertRaises(ForkPoisonedError):
                        svc2.arm_guards()
            finally:
                svc2.close()


class ZygoteForkAndScmRightsTests(unittest.TestCase):
    def tearDown(self) -> None:
        if hasattr(gc, "unfreeze"):
            gc.unfreeze()

    def test_real_fork_scm_rights_fd_passing_and_unpatch_child(self) -> None:
        """Real `os.fork()` + `AF_UNIX` `SCM_RIGHTS` transfers pipe FD, unpatches child, and reaps cleanly."""
        fake_cuda = _make_fake_cuda()
        prof = ZygoteProfile("vllm", preload_modules=("json",), env_defaults={"OMP_NUM_THREADS": "1"}, freeze_gc=True)

        def _child_handler(payload: dict, fds: list[int]) -> int:
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
            }
            os.write(fds[0], json.dumps(out).encode("utf-8"))
            return 0

        with mock.patch.dict(os.environ, {}, clear=False):
            with ZygoteService(prof, cuda_module_fn=lambda: fake_cuda) as svc:
                svc_sock = svc.socket_path
                self.assertIsNotNone(svc_sock)
                assert svc_sock is not None
                self.assertTrue(svc_sock.exists())
                svc.arm_guards(
                    capability=(9, 0),
                    probe=lambda: [{"name": "H100", "major": 9, "minor": 0, "total_memory": 80_000, "multi_processor_count": 132}],
                    preload=False,
                )
                r_fd, w_fd = os.pipe()
                try:
                    child = svc.spawn_child(
                        "actor",
                        env={"RSIBOX_ZYGOTE_TEST_TAG": "scm-ok"},
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
                self.assertEqual(child.role, "actor")
                self.assertEqual(child.engine, "vllm")
                self.assertEqual(child.incarnation, 1)
                data = json.loads(raw.decode("utf-8"))
                self.assertEqual(data["pid"], child.pid)
                self.assertEqual(data["payload_pid"], child.pid)
                self.assertEqual(data["env_tag"], "scm-ok")
                self.assertFalse(data["still_patched"])
                self.assertEqual(
                    sorted(data["child_guard_report"]["torch_cuda"]),
                    ["current_device", "get_device_capability", "get_device_properties"],
                )
                self.assertEqual(svc.wait_child(child.pid, timeout_s=5.0), 0)
                reaped = svc.reap_finished()
                self.assertEqual(reaped[child.pid], 0)
                self.assertEqual(svc.status().active_children, ())

            # Outside `with`: socket directory cleaned and service closed.
            self.assertFalse(svc_sock.exists())
            self.assertTrue(svc.status().closed)

    def test_concurrent_spawns_nonzero_exit_max_children_and_closed_errors(self) -> None:
        """Concurrent spawns, non-zero exit codes, `max_children` cap, and `ZygoteClosedError`."""
        prof = ZygoteProfile("megatron", preload_modules=(), freeze_gc=False)

        def _default_worker(payload: dict, fds: list[int]) -> int:
            if payload["role"] == "fail42":
                return 42
            if payload["role"] == "sleep":
                time.sleep(5.0)
                return 0
            if fds:
                os.write(fds[0], f"{payload['incarnation']}:{os.getpid()}".encode("utf-8"))
            return 0

        with mock.patch.dict(os.environ, {}, clear=False):
            svc = ZygoteService(prof, worker_handler=_default_worker, max_children=4)
            with self.assertRaises(ZygoteError):
                svc.spawn_child("before-start")
            svc.start(capability=(10, 0), preload=False)
            try:
                # Concurrent wait=True spawns
                def _spawn_one(i: int) -> SpawnedChild:
                    return svc.spawn_child(f"rank-{i}", wait=True, timeout_s=5.0)

                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(_spawn_one, range(4)))
                self.assertEqual(sorted(c.incarnation for c in results), [1, 2, 3, 4])
                self.assertEqual({c.exit_code for c in results}, {0})

                # Non-zero exit code
                bad_child = svc.spawn_child("fail42", wait=True, timeout_s=5.0)
                self.assertEqual(bad_child.exit_code, 42)
            finally:
                svc.close()

            # Idempotent close + ZygoteClosedError after close
            svc.close()
            with self.assertRaises(ZygoteClosedError):
                svc.arm_guards()
            with self.assertRaises(ZygoteClosedError):
                svc.start()
            with self.assertRaises(ZygoteClosedError):
                svc.spawn_child("after-close")

        # Enforce max_children=2 and clean SIGTERM/reap on teardown
        with mock.patch.dict(os.environ, {}, clear=False):
            svc2 = ZygoteService(prof, worker_handler=_default_worker, max_children=2)
            svc2.start(capability=(10, 0), preload=False)
            c1 = svc2.spawn_child("sleep")
            c2 = svc2.spawn_child("sleep")
            with self.assertRaises(SpawnError):
                svc2.spawn_child("sleep")
            exits = svc2.teardown(grace_s=0.1)
            self.assertIn(c1.pid, exits)
            self.assertIn(c2.pid, exits)
            self.assertEqual(svc2.status().active_children, ())

    def test_external_listener_and_overlong_socket_path(self) -> None:
        """External listener, explicit socket path, multi-FD SCM_RIGHTS, handler exception, and validation."""
        prof = profile_for_engine("transformers")
        prof_plain = ZygoteProfile(prof.engine, preload_modules=(), freeze_gc=False)
        with tempfile.TemporaryDirectory(prefix="zygext_") as tdir:
            ext_sock = Path(tdir) / "ext.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
                srv.bind(os.fspath(ext_sock))
                srv.listen(4)
                with mock.patch.dict(os.environ, {}, clear=False):
                    svc = ZygoteService(prof_plain)
                    svc.start(listener=srv, capability=(9, 0), preload=False)
                    child = svc.spawn_child("eval", wait=True, timeout_s=5.0)
                    self.assertEqual(child.exit_code, 0)
                    svc.close()
                self.assertNotEqual(srv.fileno(), -1)

            explicit_path = Path(tdir) / "explicit.sock"
            with mock.patch.dict(os.environ, {}, clear=False):
                with ZygoteService(prof_plain, socket_path=explicit_path) as svc_exp:
                    self.assertTrue(explicit_path.exists())
                    r1, w1 = os.pipe()
                    r2, w2 = os.pipe()
                    try:
                        c_multi = svc_exp.spawn_child(
                            "multi-fd",
                            pass_fds=(w1, w2),
                            handler=lambda _p, fds: (os.write(fds[0], b"A"), os.write(fds[1], b"B"), 0)[2],
                            wait=True,
                        )
                    finally:
                        os.close(w1)
                        os.close(w2)
                    try:
                        self.assertEqual(_read_pipe_all(r1), b"A")
                        self.assertEqual(_read_pipe_all(r2), b"B")
                    finally:
                        os.close(r1)
                        os.close(r2)
                    self.assertEqual(c_multi.exit_code, 0)

                    def _boom(_p, _fds):
                        raise RuntimeError("boom")

                    c_boom = svc_exp.spawn_child("boom", handler=_boom, wait=True)
                    self.assertEqual(c_boom.exit_code, 1)
                self.assertFalse(explicit_path.exists())

            long_path = Path(tdir) / ("x" * 120 + ".sock")
            with mock.patch.dict(os.environ, {}, clear=False):
                svc_long = ZygoteService(prof_plain, socket_path=long_path)
                try:
                    with self.assertRaises(ZygoteError):
                        svc_long.start(capability=(9, 0), preload=False)
                finally:
                    svc_long.close()

        with self.assertRaises(TypeError):
            ZygoteService("not-a-profile")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ZygoteService(prof_plain, max_children=0)


if __name__ == "__main__":
    unittest.main()
