"""Golden tests of the FastVideo lab host on the FastVideo-shaped toy (CPU, real processes).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab -v

Each worker is a spawned process in the pin's worker busy-loop protocol (toyfv/engine.py); the
lab reaches it through its pipe exactly as it reaches FastVideo's workers.
"""
import os
import signal
import time
import unittest

from adapter.toy_lab import ToyFastVideoLab

BASE = {"video": [123, 177, 231, 34], "audio": [5, 4, 3, 2, 1, 0]}


class LabTest(unittest.TestCase):
    def setUp(self):
        self.lab = ToyFastVideoLab(ranks=2)

    def tearDown(self):
        self.lab.close()

    def test_every_rank_switches_in_the_request_gap_and_witness_is_per_rank(self):
        """Input: the visible edit (decode `+ 1` -> `+ 2`) on a 2-rank engine. Expect: the
        next video changes on rank 0's output, both ranks report 1 call of the new code, and
        undo returns the base output. Proves: install reaches every worker process through
        its pipe, between two requests. Not: a switch inside a request (there is none)."""
        self.assertEqual(self.lab.check()["output"], BASE)
        r = self.lab.install(self.lab.patches()["visible"])
        self.assertEqual(r["outcome"], "committed", r["codes"])
        c = self.lab.check()
        self.assertEqual(c["output"]["video"], [124, 178, 232, 35])
        self.assertEqual(sorted(c["witness"]), ["rank0", "rank1"])
        self.assertEqual([list(w.values()) for w in c["witness"].values()], [[1], [1]])
        self.assertTrue(c["proven"])
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], BASE)

    def test_a_reply_that_misses_the_deadline_is_never_read_as_the_next_reply(self):
        """Input: a cell that sleeps 1.0 s on rank 0, called with a 0.2 s deadline. Expect:
        TimeoutError, then every later call refused until a respawn. Proves: the stale reply
        left in the pipe is never taken as the answer to the next call (FastVideo's
        collective_rpc ignores its timeout and would block, or read it). Not: a deadline
        inside the worker."""
        with self.assertRaises(TimeoutError):
            self.lab.rank_call(0, "repl", "import time; time.sleep(1.0); out = 'late'", timeout=0.2)
        time.sleep(1.2)                                     # the late reply is now in the pipe
        with self.assertRaises(ConnectionError):
            self.lab.rank_call(0, "ping")
        w = self.lab.executor.workers[0]
        self.assertTrue(w.pipe.poll(0))                      # it is there, unread
        self.assertIn("late", str(w.pipe.recv()))
        self.assertTrue(self.lab.reset("respawn")["ok"])
        self.assertEqual(self.lab.rank_call(0, "ping")["rank"], 0)

    def test_a_worker_killed_for_real_poisons_and_respawn_returns_to_base(self):
        """Input: SIGKILL of rank 1's process while an edit is installed. Expect: the next
        check fails and poisons the lab, install is refused with session_poisoned, a respawn
        starts new workers that run base code. Proves: a dead rank is a finding, never a hang
        or a silent pass, and respawn re-imports base code. Not: a hung CUDA context."""
        self.lab.install(self.lab.patches()["visible"])
        os.kill(self.lab.executor.workers[1].proc.pid, signal.SIGKILL)
        self.lab.executor.workers[1].proc.join(5)
        c = self.lab.check()
        self.assertFalse(c["ok"])
        self.assertEqual(self.lab.status()["state"], "poisoned")
        r = self.lab.install(self.lab.patches()["visible"])
        self.assertEqual(r["codes"], ["session_poisoned"])
        self.assertTrue(self.lab.reset("respawn")["ok"])
        self.assertEqual(self.lab.status()["state"], "base")
        self.assertEqual(self.lab.check()["output"], BASE)

    def test_a_startup_only_edit_installs_but_is_not_proven(self):
        """Input: an edit of `load_schedule`, which only the pipeline's constructor calls.
        Expect: committed, output unchanged, proven False with zero_witness on both ranks.
        Proves: an edit the workload never runs never reads as a pass (H3's DMD schedule edit
        has the same shape). Not: graph-captured code."""
        r = self.lab.install(self.lab.patches()["unreached"])
        self.assertEqual(r["outcome"], "committed")
        c = self.lab.check()
        self.assertEqual(c["output"], BASE)
        self.assertIs(c["proven"], False)
        self.assertEqual(sorted(f["participant"] for f in c["findings"]
                                if f["check"] == "zero_witness"), ["rank0", "rank1"])

    def test_source_readers_see_the_installed_text_until_undo(self):
        """Input: an edit that adds a function, then a cell that reads the function's source
        with inspect.getsource in the worker. Expect: the new source while installed; after
        undo, the source on disk is read again. Proves: decorators that read source at
        definition time (Triton's @jit, Dynamo) compile the edit, not the file on disk: a new
        @triton.jit kernel in an H3 edit compiled the base file's function and crashed the
        next video. Not: a real Triton compile (no GPU here)."""
        P = self.lab.patches()["visible"]
        (mod, text), = P["files"].items()
        text = text + "\n\ndef added_helper():\n    return 'new'\n"
        self.assertEqual(self.lab.install({"files": {mod: text}})["outcome"], "committed")
        cell = ("import inspect, linecache, sys\n"
                f"m = sys.modules['{mod}']\n"
                "out = linecache.getline(m.__file__, len(linecache.getlines(m.__file__)))")
        r = self.lab.repl(cell)
        self.assertTrue(r["ok"], r)
        self.assertEqual({v["out"] for v in r["ranks"].values()}, {repr("    return 'new'\n")})
        self.lab.uninstall()
        r = self.lab.repl(cell)
        self.assertNotIn("new", {v["out"] for v in r["ranks"].values()}.pop())

    def test_proof_is_per_transaction(self):
        """Input: an edit the workload never runs, then a visible edit on top. Expect: the
        check's `proven` is False (one live edit did not run), while `proven_by_txn` says the
        visible edit ran and the startup-only one did not. Proves: a judge can tell whether the
        NEW edit ran even when an older installed edit no longer runs (a combined reinstall of
        kept H3 edits held a function a later edit bypasses, and hid a proven new edit). Not:
        which function of a transaction did not run (the findings say)."""
        a = self.lab.install(self.lab.patches()["unreached"])
        b = self.lab.install(self.lab.patches()["visible"])
        c = self.lab.check()
        self.assertIs(c["proven"], False)
        self.assertEqual(c["proven_by_txn"], {a["id"]: False, b["id"]: True})

    def test_fresh_check_takes_a_package_init_and_a_new_module(self):
        """Input: fresh_check with a patch of the package `adapter.toyfv` (its `__init__.py`)
        and of a module that does not exist yet. Expect: an ok-dict with the base output, no
        exception. Proves: the overlay writes a package's `__init__.py` and new files (an edit
        that adds a module is checked, not crashed on)."""
        r = self.lab.fresh_check({"files": {"adapter.toyfv": '"""toy."""\nX = 1\n',
                                            "adapter.toyfv.new_helper": "Y = 2\n"}})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["output"], BASE)

    def test_sp2_asymmetric_crash_fails_fast_and_dirty_respawn_skips_shutdown_wait(self):
        """Input: SP2 toy lab with check_timeout=30 s; rank0 sleeps 60 s in transformer (simulating
        blocked NCCL collective) while rank1 injects `crash`. Expected: check() fails in <4 s via
        the worker fault beacon (not 30 s), poisons with participant_failed, refuses install, and
        reset('respawn') fast-kills dirty workers, cleans the temp fault file, and restores BASE."""
        from fastrsi import proc
        lab = ToyFastVideoLab(ranks=2, check_timeout=30.0)
        try:
            old_pids = [w.proc.pid for w in lab.executor.workers]
            old_fault_path = lab._fault_path
            self.assertIsNotNone(old_fault_path)
            self.assertTrue(lab.repl(
                "import time\n"
                "pipeline.modules['transformer'].register_forward_pre_hook(lambda *a: time.sleep(60.0))",
                rank=0,
            )["ok"])
            self.assertTrue(lab.inject("crash", participant=1)["ok"])
            t0 = time.perf_counter()
            c = lab.check()
            elapsed = time.perf_counter() - t0
            self.assertFalse(c["ok"])
            self.assertLess(elapsed, 4.0, f"expected fast-fail, took {elapsed:.2f}s: {c}")
            self.assertIn("rank1 failed during forward while peers blocked", c["error"])
            self.assertEqual(lab.status()["state"], "poisoned")
            self.assertEqual(lab.install(lab.patches()["visible"])["codes"], ["session_poisoned"])

            t_rs = time.perf_counter()
            rs = lab.reset("respawn")
            rs_elapsed = time.perf_counter() - t_rs
            self.assertTrue(rs["ok"], rs)
            self.assertLess(rs_elapsed, 6.0, f"dirty respawn too slow: {rs_elapsed:.2f}s")
            for pid in old_pids:
                self.assertFalse(proc.pid_alive(pid))
            self.assertFalse(os.path.exists(old_fault_path))
            self.assertEqual(lab.check()["output"], BASE)
        finally:
            lab.close()

    def test_sp2_worker_death_mid_forward_fails_fast_while_peer_blocked(self):
        """Input: SP2 toy lab with check_timeout=30 s; rank0 blocks 60 s in transformer while rank1
        exits abruptly (os._exit(9)) inside forward. Expected: check() fails in <4 s via worker
        liveness polling and respawn restores BASE."""
        lab = ToyFastVideoLab(ranks=2, check_timeout=30.0)
        try:
            self.assertTrue(lab.repl(
                "import time\n"
                "pipeline.modules['transformer'].register_forward_pre_hook(lambda *a: time.sleep(60.0))",
                rank=0,
            )["ok"])
            self.assertTrue(lab.repl(
                "import os\n"
                "pipeline.modules['transformer'].register_forward_pre_hook(lambda *a: os._exit(9))",
                rank=1,
            )["ok"])
            t0 = time.perf_counter()
            c = lab.check()
            elapsed = time.perf_counter() - t0
            self.assertFalse(c["ok"])
            self.assertLess(elapsed, 4.0, f"expected fast-fail on worker exit, took {elapsed:.2f}s")
            self.assertIn("rank1 exited during forward", c["error"])
            self.assertEqual(lab.status()["state"], "poisoned")
            self.assertTrue(lab.reset("respawn")["ok"])
            self.assertEqual(lab.check()["output"], BASE)
        finally:
            lab.close()

    def test_dirty_shutdown_fast_kills_hung_workers_before_executor_shutdown(self):
        """Input: hung worker (60 s sleep) after check_timeout=0.5 s expires (_dirty=True).
        Expected: _stop_engine kills hung workers BEFORE gen.shutdown() runs (avoiding FastVideo's
        30 s MultiprocExecutor graceful wait), cleans the temp fault file, and restores BASE."""
        lab = ToyFastVideoLab(ranks=2, check_timeout=0.5)
        try:
            alive_at_shutdown = []
            orig_shutdown = lab.gen.shutdown

            def spy_shutdown():
                alive_at_shutdown.extend(w.proc.is_alive() for w in lab.executor.workers)
                return orig_shutdown()

            lab.gen.shutdown = spy_shutdown
            old_fault_path = lab._fault_path
            self.assertTrue(lab.inject("hang", participant=1, hang_s=60.0)["ok"])
            c = lab.check()
            self.assertFalse(c["ok"])
            self.assertTrue(lab._dirty)

            t0 = time.perf_counter()
            rs = lab.reset("respawn")
            self.assertTrue(rs["ok"], rs)
            self.assertLess(time.perf_counter() - t0, 6.0)
            self.assertEqual(alive_at_shutdown, [False, False])
            self.assertFalse(os.path.exists(old_fault_path))
            self.assertEqual(lab.check()["output"], BASE)
        finally:
            lab.close()


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_have_torch(), "the holder process needs torch")
class HolderLabTest(unittest.TestCase):
    def test_close_takes_back_the_holder_socket_and_its_env(self):
        """Input: a toy lab with `holder=True`, then close. Expect: while open,
        FASTRSI_HOLDER_SOCK names this lab's socket; after close the variable is gone and the
        socket and its ready file are removed. Proves: a lab leaves no holder socket or
        environment behind for the next lab in the process."""
        before = {k: os.environ.get(k) for k in ("FASTVIDEO_WORKER_MULTIPROC_METHOD",
                                                 "PYTORCH_NVML_BASED_CUDA_CHECK", "FASTRSI_ZYGOTE_PRELOAD")}
        lab = ToyFastVideoLab(ranks=1, holder=True)
        try:
            sock = lab.holder_sock
            self.assertEqual(os.environ.get("FASTRSI_HOLDER_SOCK"), sock)
        finally:
            lab.close()
            for k, v in before.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertNotIn("FASTRSI_HOLDER_SOCK", os.environ)
        self.assertFalse(os.path.exists(sock))
        self.assertFalse(os.path.exists(sock + ".ready"))


if __name__ == "__main__":
    unittest.main()
