"""CPU unit tests for `SglangQuiesceHooks` and `DynamoK8sSnapshotDriver` integration."""
from __future__ import annotations

import asyncio
import dataclasses
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

LIVELAB_DIR = Path(__file__).resolve().parents[1]
CORE_LIVELAB = Path(__file__).resolve().parents[4] / "core" / "livelab"
for search_path in (str(LIVELAB_DIR), str(CORE_LIVELAB)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from adapter import SglangQuiesceHooks  # noqa: E402
from adapter.lab import SglangLab  # noqa: E402
from adapter.snapshot_hooks import DEFAULT_WARMUP_WORKLOAD  # noqa: E402
from rsibox.warm import (  # noqa: E402
    DiskStore,
    DynamoGmsStore,
    DynamoK8sSnapshotDriver,
    QuiesceHooks,
    QuiesceReceipt,
)

FAKE_IO_MODULE = "fake_sglang_snapshot_io_struct"


def _make_fake_io_module() -> types.ModuleType:
    """Create a synthetic SGLang `io_struct` module with `kw_only=True` request types."""

    @dataclasses.dataclass(kw_only=True)
    class PauseGenerationReqInput:
        mode: str = "abort"

    @dataclasses.dataclass(kw_only=True)
    class ContinueGenerationReqInput:
        torch_empty_cache: bool = True

    io_module = types.ModuleType(FAKE_IO_MODULE)
    io_module.PauseGenerationReqInput = PauseGenerationReqInput
    io_module.ContinueGenerationReqInput = ContinueGenerationReqInput
    return io_module


class _FakeTokenizerManager:
    """Mirrors `sglang.srt.managers.tokenizer_manager.TokenizerManager` pause/continue hooks."""

    def __init__(
        self,
        events: list[tuple[str, Any]],
        *,
        async_mode: bool = True,
        pause_result: Any = None,
        continue_result: Any = None,
        pause_error: Exception | None = None,
        continue_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.async_mode = async_mode
        self.pause_result = pause_result
        self.continue_result = continue_result
        self.pause_error = pause_error
        self.continue_error = continue_error
        self.paused = False
        self.pause_loop_ids: list[int] = []
        self.continue_loop_ids: list[int] = []

    def pause_generation(self, request: Any) -> Any:
        def execute() -> Any:
            self.events.append(("pause_generation", request.mode))
            if self.pause_error is not None:
                raise self.pause_error
            self.paused = True
            return self.pause_result

        if not self.async_mode:
            return execute()

        async def run_async() -> Any:
            self.pause_loop_ids.append(id(asyncio.get_running_loop()))
            return execute()

        return run_async()

    def continue_generation(self, request: Any) -> Any:
        def execute() -> Any:
            self.events.append(("continue_generation", request.torch_empty_cache))
            if self.continue_error is not None:
                raise self.continue_error
            self.paused = False
            return self.continue_result

        if not self.async_mode:
            return execute()

        async def run_async() -> Any:
            self.continue_loop_ids.append(id(asyncio.get_running_loop()))
            return execute()

        return run_async()


class _FakeSglangEngine:
    """Mirrors `sglang.srt.entrypoints.engine.Engine` control methods and offload-tag tracking."""

    def __init__(
        self,
        events: list[tuple[str, Any]],
        *,
        event_loop: asyncio.AbstractEventLoop | None = None,
        async_tokenizer: bool = True,
        async_memory: bool = False,
        pause_result: Any = None,
        continue_result: Any = None,
        release_result: Any = None,
        resume_result: Any = None,
        pause_error: Exception | None = None,
        continue_error: Exception | None = None,
        release_error: Exception | None = None,
        resume_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.loop = event_loop
        self.async_memory = async_memory
        self.release_result = release_result
        self.resume_result = resume_result
        self.release_error = release_error
        self.resume_error = resume_error
        self.offload_tags: set[str] = set()
        self.tokenizer_manager = _FakeTokenizerManager(
            events,
            async_mode=async_tokenizer,
            pause_result=pause_result,
            continue_result=continue_result,
            pause_error=pause_error,
            continue_error=continue_error,
        )

    def generate(self, prompt: list[str], sampling_params: dict[str, Any]) -> list[dict[str, Any]]:
        self.events.append(("generate", list(prompt), dict(sampling_params)))
        return [{"output_ids": [1, 2], "text": "ok"} for _ in prompt]

    def release_memory_occupation(self, tags: list[str] | None = None) -> Any:
        def execute() -> Any:
            resolved_tags = list(tags or [])
            self.events.append(("release_memory_occupation", resolved_tags))
            if self.release_error is not None:
                raise self.release_error
            for tag in resolved_tags:
                self.offload_tags.add(tag)
            return self.release_result

        if not self.async_memory:
            return execute()

        async def run_async() -> Any:
            return execute()

        return run_async()

    def resume_memory_occupation(self, tags: list[str] | None = None) -> Any:
        def execute() -> Any:
            resolved_tags = list(tags or [])
            self.events.append(("resume_memory_occupation", resolved_tags))
            if self.resume_error is not None:
                raise self.resume_error
            # Mirrors SGLang `weight_updater.py`: resuming an unreleased tag raises KeyError.
            for tag in resolved_tags:
                self.offload_tags.remove(tag)
            return self.resume_result

        if not self.async_memory:
            return execute()

        async def run_async() -> Any:
            return execute()

        return run_async()


def _make_fake_lab(
    engine: Any,
    events: list[tuple[str, Any]],
    *,
    weight_store: Any = None,
    dirty: bool = False,
    probe_outcomes: list[tuple[bool, str | None] | Exception] | None = None,
    rpc_timeout: float = 5.0,
) -> SglangLab:
    lab = SglangLab.__new__(SglangLab)
    lab.engine = engine
    lab._dirty = dirty
    lab.weight_store = weight_store
    lab.rpc_timeout = rpc_timeout
    lab.io_module = FAKE_IO_MODULE
    lab.ranks = 1

    outcomes = list(probe_outcomes) if probe_outcomes is not None else []

    def probe_participants() -> tuple[bool, str | None]:
        events.append(("probe_participants", None))
        if outcomes:
            next_outcome = outcomes.pop(0)
            if isinstance(next_outcome, Exception):
                raise next_outcome
            return next_outcome
        return True, None

    lab._probe_participants = probe_participants
    return lab


class SglangQuiesceHooksTests(unittest.TestCase):
    def test_protocol_conformance_and_default_release_tags_selection(self) -> None:
        events: list[tuple[str, Any]] = []
        engine = _FakeSglangEngine(events)

        gms_store = DynamoGmsStore(socket_dir="/tmp/fake-gms")
        gms_lab = _make_fake_lab(engine, events, weight_store=gms_store)
        gms_hooks = gms_lab.quiesce_hooks()
        self.assertIsInstance(gms_hooks, SglangQuiesceHooks)
        self.assertIsInstance(gms_hooks, QuiesceHooks)
        self.assertEqual(gms_hooks.mutability, "readonly_weights")
        self.assertEqual(gms_hooks.release_tags, ("weights", "kv_cache"))

        disk_lab = _make_fake_lab(engine, events, weight_store=DiskStore())
        disk_hooks = disk_lab.quiesce_hooks()
        self.assertIsInstance(disk_hooks, QuiesceHooks)
        self.assertEqual(disk_hooks.release_tags, ("kv_cache",))

        no_store_lab = _make_fake_lab(engine, events, weight_store=None)
        self.assertEqual(no_store_lab.quiesce_hooks().release_tags, ("kv_cache",))

        override_hooks = gms_lab.quiesce_hooks(release_tags=["kv_cache"], pause_mode="in_place")
        self.assertEqual(override_hooks.release_tags, ("kv_cache",))
        self.assertEqual(override_hooks.pause_mode, "in_place")

        empty_hooks = gms_lab.quiesce_hooks(release_tags=())
        self.assertEqual(empty_hooks.release_tags, ())

        with self.assertRaises(TypeError):
            gms_lab.quiesce_hooks(release_tags="kv_cache")
        with self.assertRaises(TypeError):
            gms_lab.quiesce_hooks(release_tags=b"kv_cache")

    def test_full_snapshot_driver_lifecycle_with_sentinels_and_stopped_engine_loop(self) -> None:
        events: list[tuple[str, Any]] = []
        stopped_loop = asyncio.new_event_loop()
        self.addCleanup(stopped_loop.close)
        engine = _FakeSglangEngine(events, event_loop=stopped_loop, async_tokenizer=True)
        gms_store = DynamoGmsStore(socket_dir="/tmp/fake-gms")
        lab = _make_fake_lab(engine, events, weight_store=gms_store)
        hooks = lab.quiesce_hooks()

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglsnap-") as control_dir:
            control_path = Path(control_dir)
            ready_sentinel = control_path / "ready-for-snapshot"
            restore_complete_sentinel = control_path / "restore-complete"
            restore_ready_sentinel = control_path / "sglang-restore-ready"
            # Plant stale sentinels to confirm step 1 clears them before warmup.
            ready_sentinel.write_text("stale", encoding="utf-8")
            restore_complete_sentinel.write_text("stale", encoding="utf-8")
            restore_ready_sentinel.write_text("stale", encoding="utf-8")

            agent_observations: dict[str, Any] = {}

            def simulate_snapshot_agent() -> None:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if ready_sentinel.exists() and any(
                        entry[0] == "release_memory_occupation" for entry in events
                    ):
                        agent_observations["restore_complete_cleared"] = (
                            not restore_complete_sentinel.exists()
                        )
                        agent_observations["events_at_ready"] = list(events)
                        agent_observations["offload_tags_at_ready"] = set(engine.offload_tags)
                        restore_complete_sentinel.write_text("ok\n", encoding="utf-8")
                        return
                    time.sleep(0.01)

            agent_thread = threading.Thread(target=simulate_snapshot_agent, daemon=True)
            agent_thread.start()
            driver = DynamoK8sSnapshotDriver(
                engine_name="sglang",
                control_dir=control_dir,
                wait_timeout_s=5.0,
                poll_s=0.01,
            )
            with mock.patch.dict(sys.modules, {FAKE_IO_MODULE: _make_fake_io_module()}):
                lifecycle_result = driver.checkpoint_and_wait(hooks)
            agent_thread.join(timeout=2.0)

            self.assertTrue(lifecycle_result.ok, lifecycle_result.to_dict())
            self.assertTrue(agent_observations.get("restore_complete_cleared"))
            self.assertEqual(
                agent_observations["offload_tags_at_ready"], {"weights", "kv_cache"}
            )
            self.assertTrue(restore_ready_sentinel.exists())
            self.assertEqual(lifecycle_result.restore_ready_path, str(restore_ready_sentinel))
            self.assertEqual(lifecycle_result.quiesce.released, ("weights", "kv_cache"))
            self.assertEqual(lifecycle_result.quiesce.token, "sglang:retract:weights,kv_cache")
            self.assertEqual(
                lifecycle_result.resume_details["resumed"], ["weights", "kv_cache"]
            )
            self.assertTrue(lifecycle_result.resume_details["participants_ok"])
            self.assertEqual(engine.offload_tags, set())
            self.assertFalse(engine.tokenizer_manager.paused)

            event_names = [entry[0] for entry in events]
            self.assertEqual(
                event_names,
                [
                    "generate",
                    "probe_participants",
                    "pause_generation",
                    "release_memory_occupation",
                    "resume_memory_occupation",
                    "continue_generation",
                    "probe_participants",
                ],
            )
            self.assertEqual(events[0][1], DEFAULT_WARMUP_WORKLOAD["prompts"])
            self.assertEqual(events[2], ("pause_generation", "retract"))
            self.assertEqual(events[3], ("release_memory_occupation", ["weights", "kv_cache"]))
            self.assertEqual(events[4], ("resume_memory_occupation", ["weights", "kv_cache"]))
            self.assertEqual(events[5], ("continue_generation", True))
            # Coroutines ran directly on the stopped `engine.loop`, matching offline `sglang.Engine`.
            self.assertEqual(engine.tokenizer_manager.pause_loop_ids, [id(stopped_loop)])
            self.assertEqual(engine.tokenizer_manager.continue_loop_ids, [id(stopped_loop)])

    def test_empty_receipt_released_skips_resume_memory_occupation(self) -> None:
        events: list[tuple[str, Any]] = []
        engine = _FakeSglangEngine(events, async_tokenizer=False)
        gms_store = DynamoGmsStore(socket_dir="/tmp/fake-gms")
        lab = _make_fake_lab(engine, events, weight_store=gms_store)
        hooks = lab.quiesce_hooks()
        self.assertEqual(hooks.release_tags, ("weights", "kv_cache"))

        # Even though hooks.release_tags is non-empty, an empty receipt.released must NOT
        # invoke resume_memory_occupation (which would raise KeyError on unreleased tags).
        empty_receipt = QuiesceReceipt(ok=True, released=(), token="sglang:retract:")
        with mock.patch.dict(sys.modules, {FAKE_IO_MODULE: _make_fake_io_module()}):
            resume_result = hooks.resume_after_restore(empty_receipt)

        self.assertTrue(resume_result["ok"], resume_result)
        self.assertEqual(resume_result["resumed"], [])
        self.assertTrue(resume_result["participants_ok"])
        self.assertEqual(
            [entry[0] for entry in events],
            ["continue_generation", "probe_participants"],
        )

        # Also verify hooks configured with release_tags=() skip both release and resume.
        events.clear()
        no_release_hooks = lab.quiesce_hooks(release_tags=())
        with mock.patch.dict(sys.modules, {FAKE_IO_MODULE: _make_fake_io_module()}):
            quiesce_receipt = no_release_hooks.quiesce_for_snapshot()
            resume_after_empty = no_release_hooks.resume_after_restore(quiesce_receipt)

        self.assertTrue(quiesce_receipt.ok)
        self.assertEqual(quiesce_receipt.released, ())
        self.assertTrue(resume_after_empty["ok"])
        self.assertEqual(resume_after_empty["resumed"], [])
        self.assertEqual(
            [entry[0] for entry in events],
            ["pause_generation", "continue_generation", "probe_participants"],
        )

    def test_event_loop_resolution_modes_and_same_loop_fail_closed(self) -> None:
        io_modules = {FAKE_IO_MODULE: _make_fake_io_module()}

        # 1. Synchronous return values (non-awaitable tokenizer and memory methods).
        sync_events: list[tuple[str, Any]] = []
        sync_engine = _FakeSglangEngine(sync_events, async_tokenizer=False, async_memory=False)
        sync_lab = _make_fake_lab(sync_engine, sync_events)
        sync_hooks = sync_lab.quiesce_hooks()
        with mock.patch.dict(sys.modules, io_modules):
            sync_receipt = sync_hooks.quiesce_for_snapshot()
            sync_resume = sync_hooks.resume_after_restore(sync_receipt)
        self.assertTrue(sync_receipt.ok)
        self.assertTrue(sync_resume["ok"])

        # 2. Coroutines when `engine.loop` is None (falls back to `asyncio.run`).
        none_loop_events: list[tuple[str, Any]] = []
        none_loop_engine = _FakeSglangEngine(
            none_loop_events, event_loop=None, async_tokenizer=True, async_memory=True
        )
        none_loop_lab = _make_fake_lab(none_loop_engine, none_loop_events)
        none_loop_hooks = none_loop_lab.quiesce_hooks()
        with mock.patch.dict(sys.modules, io_modules):
            none_loop_receipt = none_loop_hooks.quiesce_for_snapshot()
            none_loop_resume = none_loop_hooks.resume_after_restore(none_loop_receipt)
        self.assertTrue(none_loop_receipt.ok)
        self.assertTrue(none_loop_resume["ok"])
        self.assertEqual(len(none_loop_engine.tokenizer_manager.pause_loop_ids), 1)
        self.assertEqual(len(none_loop_engine.tokenizer_manager.continue_loop_ids), 1)

        # 3. Coroutines when `engine.loop` is running on a background thread (`run_coroutine_threadsafe`).
        thread_loop = asyncio.new_event_loop()
        loop_ready = threading.Event()

        def run_background_loop() -> None:
            asyncio.set_event_loop(thread_loop)
            thread_loop.call_soon(loop_ready.set)
            thread_loop.run_forever()

        loop_thread = threading.Thread(target=run_background_loop, daemon=True)
        loop_thread.start()
        self.assertTrue(loop_ready.wait(timeout=2.0))
        try:
            threaded_events: list[tuple[str, Any]] = []
            threaded_engine = _FakeSglangEngine(
                threaded_events, event_loop=thread_loop, async_tokenizer=True, async_memory=True
            )
            threaded_lab = _make_fake_lab(threaded_engine, threaded_events)
            threaded_hooks = threaded_lab.quiesce_hooks()
            with mock.patch.dict(sys.modules, io_modules):
                threaded_receipt = threaded_hooks.quiesce_for_snapshot()
                threaded_resume = threaded_hooks.resume_after_restore(threaded_receipt)
            self.assertTrue(threaded_receipt.ok, threaded_receipt.error)
            self.assertTrue(threaded_resume["ok"], threaded_resume)
            self.assertEqual(threaded_engine.tokenizer_manager.pause_loop_ids, [id(thread_loop)])
            self.assertEqual(
                threaded_engine.tokenizer_manager.continue_loop_ids, [id(thread_loop)]
            )
        finally:
            thread_loop.call_soon_threadsafe(thread_loop.stop)
            loop_thread.join(timeout=2.0)
            thread_loop.close()

        # 4. Fail closed cleanly (and without unawaited-coroutine RuntimeWarning) when invoked
        #    on the same running event loop.
        same_loop_events: list[tuple[str, Any]] = []

        async def invoke_on_running_loop() -> QuiesceReceipt:
            current_loop = asyncio.get_running_loop()
            same_loop_engine = _FakeSglangEngine(
                same_loop_events, event_loop=current_loop, async_tokenizer=True
            )
            same_loop_lab = _make_fake_lab(same_loop_engine, same_loop_events)
            same_loop_hooks = same_loop_lab.quiesce_hooks()
            with mock.patch.dict(sys.modules, io_modules):
                return same_loop_hooks.quiesce_for_snapshot()

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            same_loop_receipt = asyncio.run(invoke_on_running_loop())

        self.assertFalse(same_loop_receipt.ok)
        self.assertIn("own running event loop", same_loop_receipt.error or "")
        self.assertEqual(same_loop_events, [])
        unawaited_warnings = [
            warning for warning in captured_warnings if "was never awaited" in str(warning.message)
        ]
        self.assertEqual(unawaited_warnings, [])

    def test_pause_failure_fails_closed_without_release_or_ready_sentinel(self) -> None:
        events: list[tuple[str, Any]] = []
        engine = _FakeSglangEngine(
            events,
            async_tokenizer=False,
            pause_error=RuntimeError("scheduler refused pause"),
        )
        lab = _make_fake_lab(engine, events)
        hooks = lab.quiesce_hooks()

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglsnap-") as control_dir:
            driver = DynamoK8sSnapshotDriver(
                engine_name="sglang", control_dir=control_dir, wait_timeout_s=0.2, poll_s=0.01
            )
            with mock.patch.dict(sys.modules, {FAKE_IO_MODULE: _make_fake_io_module()}):
                lifecycle_result = driver.checkpoint_and_wait(hooks)

            self.assertFalse(lifecycle_result.ok)
            self.assertIn("scheduler refused pause", lifecycle_result.error or "")
            self.assertFalse((Path(control_dir) / "ready-for-snapshot").exists())
            self.assertFalse((Path(control_dir) / "sglang-restore-ready").exists())
            event_names = [entry[0] for entry in events]
            self.assertNotIn("release_memory_occupation", event_names)
            self.assertNotIn("continue_generation", event_names)

    def test_release_failure_triggers_continue_rollback_and_blocks_ready_sentinel(self) -> None:
        events: list[tuple[str, Any]] = []
        engine = _FakeSglangEngine(
            events,
            async_tokenizer=False,
            release_error=RuntimeError("cuda VMM unmap failed"),
        )
        lab = _make_fake_lab(engine, events)
        hooks = lab.quiesce_hooks()

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglsnap-") as control_dir:
            driver = DynamoK8sSnapshotDriver(
                engine_name="sglang", control_dir=control_dir, wait_timeout_s=0.2, poll_s=0.01
            )
            with mock.patch.dict(sys.modules, {FAKE_IO_MODULE: _make_fake_io_module()}):
                lifecycle_result = driver.checkpoint_and_wait(hooks)

            self.assertFalse(lifecycle_result.ok)
            self.assertIn("cuda VMM unmap failed", lifecycle_result.error or "")
            self.assertFalse((Path(control_dir) / "ready-for-snapshot").exists())
            self.assertFalse(engine.tokenizer_manager.paused)
            self.assertEqual(
                [entry[0] for entry in events],
                [
                    "generate",
                    "probe_participants",
                    "pause_generation",
                    "release_memory_occupation",
                    "continue_generation",
                ],
            )

        # If best-effort rollback also fails, preserve both the primary and rollback errors.
        double_fail_events: list[tuple[str, Any]] = []
        double_fail_engine = _FakeSglangEngine(
            double_fail_events,
            async_tokenizer=False,
            release_error=RuntimeError("unmap boom"),
            continue_error=RuntimeError("continue boom"),
        )
        double_fail_lab = _make_fake_lab(double_fail_engine, double_fail_events)
        with mock.patch.dict(sys.modules, {FAKE_IO_MODULE: _make_fake_io_module()}):
            receipt = double_fail_lab.quiesce_hooks().quiesce_for_snapshot()
        self.assertFalse(receipt.ok)
        self.assertIn("unmap boom", receipt.error or "")
        self.assertIn("continue boom", receipt.error or "")

    def test_explicit_negative_control_results_fail_closed(self) -> None:
        negative_results = [
            False,
            {"ok": False, "error": "dict ok false"},
            {"success": False, "message": "dict success false"},
            types.SimpleNamespace(success=False, message="object success false"),
            types.SimpleNamespace(ok=False, error="object ok false"),
        ]
        io_modules = {FAKE_IO_MODULE: _make_fake_io_module()}

        for negative_value in negative_results:
            with self.subTest(stage="pause_generation", negative_value=negative_value):
                events: list[tuple[str, Any]] = []
                engine = _FakeSglangEngine(
                    events, async_tokenizer=False, pause_result=negative_value
                )
                lab = _make_fake_lab(engine, events)
                with mock.patch.dict(sys.modules, io_modules):
                    receipt = lab.quiesce_hooks().quiesce_for_snapshot()
                self.assertFalse(receipt.ok)
                self.assertNotIn("release_memory_occupation", [entry[0] for entry in events])

            with self.subTest(stage="release_memory_occupation", negative_value=negative_value):
                events = []
                engine = _FakeSglangEngine(
                    events, async_tokenizer=False, release_result=negative_value
                )
                lab = _make_fake_lab(engine, events)
                with mock.patch.dict(sys.modules, io_modules):
                    receipt = lab.quiesce_hooks().quiesce_for_snapshot()
                self.assertFalse(receipt.ok)
                self.assertIn("continue_generation", [entry[0] for entry in events])

            with self.subTest(stage="resume_memory_occupation", negative_value=negative_value):
                events = []
                engine = _FakeSglangEngine(
                    events, async_tokenizer=False, resume_result=negative_value
                )
                engine.offload_tags.add("kv_cache")
                lab = _make_fake_lab(engine, events)
                with mock.patch.dict(sys.modules, io_modules):
                    resume_result = lab.quiesce_hooks().resume_after_restore(
                        QuiesceReceipt(ok=True, released=("kv_cache",))
                    )
                self.assertFalse(resume_result["ok"])
                self.assertNotIn("continue_generation", [entry[0] for entry in events])

            with self.subTest(stage="continue_generation", negative_value=negative_value):
                events = []
                engine = _FakeSglangEngine(
                    events, async_tokenizer=False, continue_result=negative_value
                )
                engine.offload_tags.add("kv_cache")
                lab = _make_fake_lab(engine, events)
                with mock.patch.dict(sys.modules, io_modules):
                    resume_result = lab.quiesce_hooks().resume_after_restore(
                        QuiesceReceipt(ok=True, released=("kv_cache",))
                    )
                self.assertFalse(resume_result["ok"])
                self.assertNotIn("probe_participants", [entry[0] for entry in events])

    def test_resume_continue_and_probe_failures_prevent_restore_ready_sentinel(self) -> None:
        io_modules = {FAKE_IO_MODULE: _make_fake_io_module()}

        failure_cases = [
            (
                "resume_memory_occupation",
                dict(resume_error=RuntimeError("gms remap timeout")),
                None,
                "gms remap timeout",
            ),
            (
                "continue_generation",
                dict(continue_error=RuntimeError("continue dispatch failed")),
                None,
                "continue dispatch failed",
            ),
            (
                "probe_returns_false",
                {},
                [(True, None), (False, "rank0: socket disconnected")],
                "rank0: socket disconnected",
            ),
            (
                "probe_raises_exception",
                {},
                [(True, None), ConnectionError("rank0: broken pipe")],
                "broken pipe",
            ),
        ]

        for case_name, engine_kwargs, probe_outcomes, expected_error in failure_cases:
            with self.subTest(case=case_name):
                events: list[tuple[str, Any]] = []
                engine = _FakeSglangEngine(events, async_tokenizer=False, **engine_kwargs)
                lab = _make_fake_lab(engine, events, probe_outcomes=probe_outcomes)
                hooks = lab.quiesce_hooks()

                with tempfile.TemporaryDirectory(dir="/tmp", prefix="sglsnap-") as control_dir:
                    control_path = Path(control_dir)

                    def write_restore_complete() -> None:
                        deadline = time.monotonic() + 5.0
                        while time.monotonic() < deadline:
                            if (control_path / "ready-for-snapshot").exists():
                                (control_path / "restore-complete").write_text(
                                    "ok\n", encoding="utf-8"
                                )
                                return
                            time.sleep(0.01)

                    agent_thread = threading.Thread(target=write_restore_complete, daemon=True)
                    agent_thread.start()
                    driver = DynamoK8sSnapshotDriver(
                        engine_name="sglang",
                        control_dir=control_dir,
                        wait_timeout_s=5.0,
                        poll_s=0.01,
                    )
                    with mock.patch.dict(sys.modules, io_modules):
                        lifecycle_result = driver.checkpoint_and_wait(hooks)
                    agent_thread.join(timeout=2.0)

                    self.assertFalse(lifecycle_result.ok)
                    self.assertIn(expected_error, lifecycle_result.error or "")
                    self.assertFalse((control_path / "sglang-restore-ready").exists())
                    self.assertFalse(lifecycle_result.resume_details["participants_ok"])

    def test_dirty_lab_and_stopped_engine_fail_closed_across_all_hooks(self) -> None:
        io_modules = {FAKE_IO_MODULE: _make_fake_io_module()}
        receipt = QuiesceReceipt(ok=True, released=("kv_cache",))

        for label, engine, dirty, expected_fragment in (
            ("stopped_engine", None, False, "not running"),
            ("dirty_lab", _FakeSglangEngine([], async_tokenizer=False), True, "dirty"),
        ):
            with self.subTest(case=label):
                events: list[tuple[str, Any]] = []
                lab = _make_fake_lab(engine, events, dirty=dirty)
                hooks = lab.quiesce_hooks()
                with mock.patch.dict(sys.modules, io_modules):
                    with self.assertRaisesRegex(RuntimeError, expected_fragment):
                        hooks.warmup_for_snapshot()

                    quiesce_receipt = hooks.quiesce_for_snapshot()
                    self.assertFalse(quiesce_receipt.ok)
                    self.assertIn(expected_fragment, quiesce_receipt.error or "")

                    resume_result = hooks.resume_after_restore(receipt)
                    self.assertFalse(resume_result["ok"])
                    self.assertIn(expected_fragment, resume_result["error"])

    def test_positional_only_memory_methods_are_invoked_cleanly(self) -> None:
        recorded_calls: list[tuple[str, list[str]]] = []

        class _PositionalOnlyEngine:
            loop = None

            def __init__(self) -> None:
                self.tokenizer_manager = _FakeTokenizerManager(
                    recorded_calls, async_mode=False
                )

            def release_memory_occupation(self, tags: list[str], /) -> None:
                recorded_calls.append(("release", list(tags)))

            def resume_memory_occupation(self, tags: list[str], /) -> None:
                recorded_calls.append(("resume", list(tags)))

        engine = _PositionalOnlyEngine()
        lab = _make_fake_lab(engine, recorded_calls)
        hooks = lab.quiesce_hooks(release_tags=("weights", "kv_cache"))
        with mock.patch.dict(sys.modules, {FAKE_IO_MODULE: _make_fake_io_module()}):
            receipt = hooks.quiesce_for_snapshot()
            resume_result = hooks.resume_after_restore(receipt)

        self.assertTrue(receipt.ok, receipt.error)
        self.assertTrue(resume_result["ok"], resume_result)
        self.assertIn(("release", ["weights", "kv_cache"]), recorded_calls)
        self.assertIn(("resume", ["weights", "kv_cache"]), recorded_calls)

    def test_nonpauseable_cuda_graphs_under_gms_forbids_cuda_graph_release_tag(self) -> None:
        """Fail closed when caller tries to release 'cuda_graph' under non-pauseable GMS graphs."""
        events: list[tuple[str, Any]] = []
        engine = _FakeSglangEngine(events, async_tokenizer=False)
        lab = _make_fake_lab(engine, events, weight_store=DynamoGmsStore(socket_dir="/tmp/gms-unused"))
        lab.allow_nonpauseable_cuda_graphs = True

        # Default ('weights', 'kv_cache') works cleanly.
        hooks = lab.quiesce_hooks()
        self.assertEqual(hooks.release_tags, ("weights", "kv_cache"))

        # Explicit 'cuda_graph' tag MUST fail closed at construction before touching engine.
        with self.assertRaisesRegex(ValueError, "cannot include 'cuda_graph'"):
            lab.quiesce_hooks(release_tags=("weights", "kv_cache", "cuda_graph"))

        # Post-construction mutation of release_tags also fails closed inside quiesce_for_snapshot
        # without calling pause_generation or release_memory_occupation.
        hooks.release_tags = ("weights", "cuda_graph")
        with mock.patch.dict(sys.modules, {FAKE_IO_MODULE: _make_fake_io_module()}):
            receipt = hooks.quiesce_for_snapshot()
        self.assertFalse(receipt.ok)
        self.assertIn("cannot include 'cuda_graph'", receipt.error or "")
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
