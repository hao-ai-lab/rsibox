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


if __name__ == "__main__":
    unittest.main()
