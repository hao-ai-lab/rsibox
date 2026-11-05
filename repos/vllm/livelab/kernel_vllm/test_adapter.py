"""CPU tests of the adapter's own logic (no vLLM, no GPU). Run from repos/vllm/livelab with the
kernel on the path:

    PYTHONPATH=../../../template/livelab python3 -m unittest kernel_vllm.test_adapter -v

The worker group's transport is replaced by a loop over in-process `Side`s: one
collective_rpc = one call on every rank, in rank order, as vLLM's executor does. Everything
the tests are named for (Side, WorkerGroup, the kernel participant) is the real code.
"""
import itertools
import sys
import time
import unittest

from kernel import locate as L
from kernel import toy

from .host import WorkerGroup
from .side import Side

ENG = toy.source("engine")
_n = itertools.count()


def plan(tag, faults=None):
    """A worker plan that stages a one-line edit of the toy engine's step (a fresh copy)."""
    name = f"kva{next(_n)}_{tag}"
    toy.load_copy(name)
    line = next(i for i, s in enumerate(ENG.splitlines(), 1) if "def step" in s)
    return {"id": f"txn-{name}", "faults": faults or {},
            "modules": {f"{name}.engine": {
                "on": ["worker"], "base_hash": L.text_hash(ENG),
                "source": ENG.replace('self.ran.append("base")', 'self.ran.append("v1")'),
                "targets": [["Engine.step", line]]}}}


class _Executor:
    def __init__(self, sides):
        self.sides, self.calls = sides, []

    def collective_rpc(self, name, timeout=None, args=()):
        method, payload = args
        self.calls.append(method)
        out = []
        for s in self.sides:
            r = s.dispatch(method, payload)
            r["rank"] = s.rank
            out.append(r)
        return out


class _Host:
    def __init__(self, sides):
        self.core = type("Core", (), {})()
        self.core.model_executor = _Executor(sides)

    def in_loop(self, fn, gap=False, timeout=60.0):
        return fn()


class WorkerPrepareTest(unittest.TestCase):

    def _side(self, rank):
        side = Side("worker", rank=rank)
        if side.lp.analyzer is not None:
            self.addCleanup(side.lp.analyzer.close)
        return side

    def test_ranks_that_finish_early_keep_their_vote(self):
        """Proves: with rank 3 preparing 0.3 s longer than ranks 0-2, the group keeps polling
        every rank and still gets four yes votes (before the fix, ranks 0-2 answered "no
        prepare started" on the polls after their first finished one: a false abort, 1 of 33
        live TP4 prepares). After abort, the result is gone. Not: the real collective_rpc."""
        sides = [self._side(r) for r in range(4)]
        g = WorkerGroup(_Host(sides))
        p = plan("slow3", {"worker:3": {"method": "prepare", "delay_s": 0.3, "fail": False}})
        t0 = time.perf_counter()
        r = g.prepare(p)
        took = time.perf_counter() - t0
        self.assertEqual([(x["rank"], x["ok"], x.get("error")) for x in r["ranks"]],
                         [(0, True, None), (1, True, None), (2, True, None), (3, True, None)])
        self.assertTrue(r["ok"])
        self.assertGreaterEqual(took, 0.3)
        polls = g.host.core.model_executor.calls.count("prepare_poll")
        self.assertGreater(polls, 2)                 # ranks 0-2 were asked again after done
        again = g._rpc("prepare_poll", p["id"])         # repeatable until the vote is over
        self.assertEqual((again["ok"], again.get("pending")), (True, None))
        self.assertTrue(g.abort(p["id"])["ok"])
        after = g._rpc("prepare_poll", p["id"])
        self.assertEqual([x["error"] for x in after["ranks"]],
                         [f"no prepare started for {p['id']}"] * 4)

    def test_side_prepares_with_the_analysis_in_a_child_process(self):
        """Proves: a Side stages through the kernel's ProcessAnalyzer by default, so the
        parses and compare do not hold the engine process's GIL (timings say "process").
        Not: the stall itself (measured live: kernel_vllm/README.md)."""
        side = self._side(0)
        p = plan("analyzer")
        r = side.dispatch("prepare", p)
        self.assertTrue(r["ok"], r.get("error"))
        (t,) = r["timings"].values()
        self.assertEqual(t["analyze"], "process", t)


if __name__ == "__main__":
    sys.exit(unittest.main())
