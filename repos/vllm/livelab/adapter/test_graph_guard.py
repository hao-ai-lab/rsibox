"""CPU golden tests for O(1) CUDA-graph staleness guard, explicit R1 pool teardown,
and non-blocking background native build subprocesses.

Runs cleanly in CPU-only CI without `torch` or `vllm` installed, exercising the real
`GraphReplayGuard`, `Recapture`/`teardown_graph_pool`, `KernelWorkerExtension`, and
`BackgroundBuildPool` (`subprocess.Popen`) implementations end to end.
"""
import gc
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
import weakref
from types import SimpleNamespace as NS
from unittest import mock

from . import graph_guard as GG
from . import native as N
from . import recapture as RC
from . import worker_ext as WE
from .test_host import make_host, plan
from .test_recapture import Runner


class _TensorLike:
    """Pure-Python duck-typed tensor exposing `.shape`, `.dtype`, `.data_ptr()`."""

    def __init__(self, shape, dtype="float16", ptr=0x1000):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


class _TrackedGraph:
    """Weakref-trackable captured graph stand-in with an explicit `.reset()` counter."""

    def __init__(self, label, reset_log):
        self.label = label
        self._reset_log = reset_log

    def reset(self):
        self._reset_log.append(self.label)


class GraphGuardGoldenTests(unittest.TestCase):

    def test_stale_replay_raises_after_code_edit_without_recapture_and_passes_after_recapture(self):
        """Input: graph captured & stamped at epoch 0; live edit bumps `code_epoch` to 1 without
        recapture, then R1 recapture stamps epoch 1. Expected: replay raises `StaleGraphReplayError`
        (fail-closed) before recapture; after recapture, `check_before_replay` passes in O(1) and
        re-verifies cleanly across an end-to-end Host/Side transaction. Fails without guard:
        stale epoch-0 graph silently replays old kernels."""
        guard = GG.GraphReplayGuard()
        key = GG.graph_key("FULL", 4)
        guard.stamp(key)
        self.assertEqual(guard.epoch, (0, 0))
        guard.check_before_replay(key)

        # Edit bumps code_epoch without recapturing `key` -> fail-closed!
        new_epoch = guard.bump_code_epoch("commit:txn-1")
        self.assertEqual(new_epoch, 1)
        self.assertFalse(guard.is_fresh(key))
        with self.assertRaises(GG.StaleGraphReplayError) as ctx:
            guard.check_before_replay(key)
        self.assertEqual(ctx.exception.reason, "epoch_mismatch")
        self.assertEqual(ctx.exception.actual_epoch, (0, 0))
        self.assertEqual(ctx.exception.expected_epoch, (1, 0))
        self.assertEqual(guard.stats()["stale_rejections"], 1)

        # Recapture stamps epoch 1 -> O(1) check passes immediately.
        guard.stamp(key)
        self.assertTrue(guard.is_fresh(key))
        t0 = time.perf_counter()
        for _ in range(10000):
            guard.check_before_replay(key)
        per_call_us = ((time.perf_counter() - t0) / 10000.0) * 1e6
        self.assertLess(per_call_us, 50.0, f"O(1) check unexpectedly slow: {per_call_us:.2f} us/call")

        # End-to-end via Host + Side: commit without R1 bumps code_epoch -> stale replay blocked;
        # commit with R1 auto-stamps captured descriptors -> replay allowed.
        h, names, sides = make_host(self, "guard_e2e")
        runners = []
        for n, s in zip(names, sides):
            r = Runner(n, modes=("FULL",), sizes=(4, 2))
            s.scope["r1_engine"] = r.engine
            s.graph_guard.stamp(GG.graph_key("FULL", 4))
            runners.append(r)

        res_no_r1 = h.handle({"op": "txn", "plan": plan("g_no_r1")})
        self.assertEqual(res_no_r1["outcome"], "committed")
        for s in sides:
            self.assertEqual(s.graph_guard.code_epoch, 1)
            with self.assertRaises(GG.StaleGraphReplayError):
                s.graph_guard.check_before_replay(GG.graph_key("FULL", 4))
        self.assertTrue(h.handle({"op": "undo", "txn": "g_no_r1"})["ok"])

        res_r1 = h.handle({"op": "txn", "plan": plan("g_with_r1", rebuild=["R1"], r1={"sizes": [4]})})
        self.assertEqual(res_r1["outcome"], "committed", res_r1["receipt"]["findings"])
        for s in sides:
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 4))
            with self.assertRaises(GG.StaleGraphReplayError):
                s.graph_guard.check_before_replay(GG.graph_key("FULL", 2))

    def test_input_signature_drift_raises_stale_graph_replay_error(self):
        """Input: graph stamped with tensor `(shape, dtype, ptr)` signature; replay invoked with
        drifted shape, dtype, or data pointer. Expected: matching signature passes in O(1), while
        every drift raises `StaleGraphReplayError(reason='input_sig_mismatch')`."""
        guard = GG.GraphReplayGuard()
        key = GG.graph_key("FULL", 8)
        t_orig = _TensorLike((8, 64), "torch.bfloat16", 0x7F001000)
        sig_orig = GG.input_signature(t_orig)
        guard.stamp(key, sig_orig)

        # Exact match passes
        guard.check_before_replay(key, GG.input_signature(_TensorLike((8, 64), "torch.bfloat16", 0x7F001000)))

        # Shape drift fails closed
        with self.assertRaises(GG.StaleGraphReplayError) as c_shape:
            guard.check_before_replay(key, GG.input_signature(_TensorLike((16, 64), "torch.bfloat16", 0x7F001000)))
        self.assertEqual(c_shape.exception.reason, "input_sig_mismatch")

        # Dtype drift fails closed
        with self.assertRaises(GG.StaleGraphReplayError) as c_dtype:
            guard.check_before_replay(key, GG.input_signature(_TensorLike((8, 64), "torch.float16", 0x7F001000)))
        self.assertEqual(c_dtype.exception.reason, "input_sig_mismatch")

        # Buffer pointer drift fails closed
        with self.assertRaises(GG.StaleGraphReplayError) as c_ptr:
            guard.check_before_replay(key, GG.input_signature(_TensorLike((8, 64), "torch.bfloat16", 0x7F009000)))
        self.assertEqual(c_ptr.exception.reason, "input_sig_mismatch")

        # Worker extension replay wrapper enforces the guard with zero overhead at epoch 0
        calls = []

        class _FakeMgr:
            runtime_mode = "FULL"
            def replay(self, desc, input_sig=None):
                calls.append(desc)
                return f"ok:{desc}"

        class _FakeWorker(WE.KernelWorkerExtension):
            rank = 0
            def __init__(self):
                self.model_runner = NS(cudagraph_manager=_FakeMgr())
            def execute_model(self, *a, **k): return None
            def sample_tokens(self, *a, **k): return None

        saved = list(WE._ENDPOINT)
        WE._ENDPOINT.clear()
        try:
            w = _FakeWorker()
            w.rsibox_kernel_endpoint()
            # Epoch 0, unstamped -> zero-overhead passthrough
            self.assertEqual(w.model_runner.cudagraph_manager.replay(4), "ok:4")
            # After code edit without recapture -> fail-closed
            w.rsibox_graph_guard().bump_code_epoch("live_edit")
            with self.assertRaises(GG.StaleGraphReplayError):
                w.model_runner.cudagraph_manager.replay(4)
            # Stamped after recapture -> succeeds again
            w.rsibox_graph_guard().stamp(GG.graph_key("FULL", 4))
            self.assertEqual(w.model_runner.cudagraph_manager.replay(4), "ok:4")
            w.rsibox_check_graph_replay(GG.graph_key("FULL", 4))
        finally:
            WE._ENDPOINT[:] = saved

    def test_repeated_r1_teardown_releases_old_graph_refs_and_increments_pool_epoch(self):
        """Input: repeated R1 recaptures with `release_old=True` holding tracked graph objects.
        Expected: every teardown clears old dicts, calls `.reset()`, drops strong references
        (verified via `weakref.ref(...)() is None`), increments `pool_epoch`, calls
        `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` when CUDA is available, and
        records `teardown_vram_before_mib`, `teardown_vram_after_mib`, `freed_graphs_count`, and
        `teardown_s` in the recapture receipt."""
        reset_log = []
        cuda_events = {"sync": 0, "empty": 0, "reserved": 512 * 1024 * 1024}

        class _FakeTorch(types.ModuleType):
            pass

        fake_torch = _FakeTorch("torch")

        def _empty_cache():
            cuda_events["empty"] += 1
            cuda_events["reserved"] = 128 * 1024 * 1024

        fake_torch.cuda = NS(
            is_available=lambda: True,
            synchronize=lambda: cuda_events.__setitem__("sync", cuda_events["sync"] + 1),
            empty_cache=_empty_cache,
            memory_reserved=lambda: cuda_events["reserved"],
        )

        class _StubJournal:
            def __init__(self):
                self.marks = []
                self.entries = []
            def mark(self, txn): self.marks.append((txn, len(self.entries)))
            def record_call(self, fn, label): self.entries.append((fn, label))

        class _StubWitness:
            def capture(self):
                import contextlib
                return contextlib.nullcontext()

        wrapper = NS(runtime_mode="FULL", concrete_cudagraph_entries={})
        keys = {"FULL": {4, 2}}
        gen = [0]

        def capture_fn(mode, desc):
            wrapper.concrete_cudagraph_entries[desc] = _TrackedGraph(f"g{gen[0]}_{desc}", reset_log)

        engine = RC.Engine(lambda: [wrapper], keys, capture_fn,
                           memory=lambda: {"reserved": cuda_events["reserved"]})
        # Initial epoch-0 capture
        for d in (4, 2):
            capture_fn("FULL", d)

        guard = GG.GraphReplayGuard()
        for d in (4, 2):
            guard.stamp(GG.graph_key("FULL", d))
        rec = RC.Recapture(guard=guard)
        journal, witness = _StubJournal(), _StubWitness()

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            for cycle in range(1, 4):
                old_refs = [weakref.ref(obj) for obj in wrapper.concrete_cudagraph_entries.values()]
                self.assertTrue(all(r() is not None for r in old_refs))
                cuda_events["reserved"] = 512 * 1024 * 1024
                gen[0] = cycle
                txn = f"txn-r1-{cycle}"
                journal.mark(txn)
                guard.bump_code_epoch(f"commit:{txn}")

                report = rec.run(engine, txn, journal, witness, release_old=True)
                self.assertTrue(report["ok"], report)
                gc.collect()

                # Every previous-cycle graph object was freed (no leaked strong references!)
                self.assertTrue(all(r() is None for r in old_refs),
                                f"cycle {cycle}: old graph reference leaked")
                self.assertEqual(guard.pool_epoch, cycle)
                self.assertEqual(report["pool_epoch"], cycle)
                self.assertEqual(report["freed_graphs_count"], 2)
                self.assertEqual(report["teardown_vram_before_mib"], 512.0)
                self.assertEqual(report["teardown_vram_after_mib"], 128.0)
                self.assertGreaterEqual(report["teardown_s"], 0.0)
                # Newly captured graphs are stamped at the new (code_epoch, pool_epoch)
                guard.check_before_replay(GG.graph_key("FULL", 4))
                guard.check_before_replay(GG.graph_key("FULL", 2))

        self.assertEqual(cuda_events["sync"], 3)
        self.assertEqual(cuda_events["empty"], 3)
        self.assertEqual(len(reset_log), 6)

    def test_background_build_runs_concurrently_with_live_work_and_cancel_leaves_no_orphan(self):
        """Input: real background subprocesses launched via `BackgroundBuildPool.compile_async`.
        Expected: (1) while a background build runs (~0.25 s) to emit an artifact, the live
        foreground executes steps without blocking and observes `poll() is None`, then `wait()`
        returns `ok=True` with the produced artifact; (2) `wait(timeout_s=0.05)` raises
        `TimeoutError` on a slow build that spawns a child process, and `cancel()` terminates the
        entire process group with zero orphan processes remaining."""
        with tempfile.TemporaryDirectory() as tmp, N.BackgroundBuildPool() as pool:
            artifact = os.path.join(tmp, "compiled_kernel.so")
            build_script = (
                "import sys, time\n"
                "time.sleep(0.25)\n"
                "with open(sys.argv[1], 'wb') as fh: fh.write(b'BUILT_OK')\n"
            )
            job = pool.compile_async(cmd=[sys.executable, "-c", build_script, artifact], job_id="fast-build")
            self.assertIsNone(job.poll())
            self.assertIn("fast-build", pool.active_jobs())

            # Foreground keeps serving live steps while background build runs
            foreground_steps = 0
            while job.poll() is None:
                foreground_steps += 1
                time.sleep(0.01)
            self.assertGreater(foreground_steps, 3, "foreground should execute multiple steps during build")
            res = job.wait(timeout_s=5.0)
            self.assertTrue(res["ok"], res)
            self.assertFalse(res["cancelled"])
            self.assertGreaterEqual(res["build_s"], 0.2)
            with open(artifact, "rb") as fh:
                self.assertEqual(fh.read(), b"BUILT_OK")

            # Slow build spawning a grandchild process: verify timeout + orphan-free cancel()
            child_pid_file = os.path.join(tmp, "grandchild.pid")
            slow_script = (
                "import subprocess, sys, time\n"
                "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "with open(sys.argv[1], 'w') as fh: fh.write(str(p.pid))\n"
                "time.sleep(60)\n"
            )
            slow = N.compile_async(pool=pool, cmd=[sys.executable, "-c", slow_script, child_pid_file],
                                   job_id="slow-build")
            # Wait until grandchild PID is written
            deadline = time.perf_counter() + 5.0
            while not os.path.isfile(child_pid_file) or os.path.getsize(child_pid_file) == 0:
                if time.perf_counter() > deadline:
                    self.fail("slow build grandchild never wrote its pid")
                time.sleep(0.01)
            with open(child_pid_file, "r", encoding="utf-8") as fh:
                grandchild_pid = int(fh.read().strip())

            with self.assertRaises(TimeoutError):
                pool.wait("slow-build", timeout_s=0.05)

            self.assertTrue(pool.cancel("slow-build", grace_s=0.3))
            self.assertTrue(slow.cancelled)
            self.assertFalse(pool.cancel("slow-build"))
            self.assertEqual(pool.active_jobs(), [])
            final = slow.poll()
            self.assertIsNotNone(final)
            self.assertFalse(final["ok"])
            self.assertTrue(final["cancelled"])

            # Neither the build process nor its grandchild may survive as an orphan
            for pid in (slow.pid, grandchild_pid):
                gone = False
                for _ in range(50):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        gone = True
                        break
                    time.sleep(0.02)
                self.assertTrue(gone, f"subprocess {pid} survived cancel()")


if __name__ == "__main__":
    unittest.main()
