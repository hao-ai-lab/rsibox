"""CPU golden tests of R1, selective CUDA-graph recapture (recapture.py, the host's
R1Coordinator, the side's `recapture`). Run as test_host.py says.

The stand-in runner keeps vLLM's semantics where R1 depends on them, and nothing else:
- a graph wrapper per mode (one FULL wrapper, PIECEWISE wrappers per piece) with a dict
  descriptor -> entry; a missing entry is captured on the next call in its mode, and only
  while capture is enabled (vLLM's `validate_cudagraph_capturing_enabled`);
- a replay runs no Python: it returns what the capture recorded;
- piece i of a descriptor recorded the address of piece i-1's output: a replay whose previous
  piece is another capture's raises (the hazard of recapturing only some pieces);
- the dispatcher's keys: FULL first, then PIECEWISE, else eager.
Host, WorkerGroup, Side, recapture.py and the kernel (journal, witness) are the real code.
"""
import sys
import unittest
import weakref
from types import SimpleNamespace as NS

from . import recapture as RC
from .test_host import STEP_LINE, make_host, plan
from .test_side import worker

KEY = f"toy.engine:Engine.step:{STEP_LINE}"
LIVE = weakref.WeakSet()       # every recorded graph still referenced


class Graph:
    def __init__(self, value, src):
        self.value, self.src = value, src
        LIVE.add(self)


class Wrapper:
    def __init__(self, runner, mode, piece):
        self.runner, self.runtime_mode, self.piece = runner, mode, piece
        self.concrete_cudagraph_entries = {}

    def __call__(self, mode, desc, prev):
        if mode != self.runtime_mode:
            return self.runner.python(prev)
        e = self.concrete_cudagraph_entries.get(desc)
        if e is None:
            if not self.runner.capturing:
                raise RuntimeError("CUDA graph capturing detected at an inappropriate time")
            if self.runner.fail_at == (mode, desc):
                raise RuntimeError("out of memory during capture")
            e = self.concrete_cudagraph_entries[desc] = Graph(self.runner.python(prev).value, prev)
            return e
        if e.src is not prev:
            raise RuntimeError(f"piece {self.piece}: replay reads another capture's output")
        return e


class Runner:
    """One rank's model runner: `pieces` piecewise wrappers, a FULL wrapper around them."""

    def __init__(self, mod, modes=("FULL",), sizes=(8, 4, 2, 1), pieces=3):
        self.mod, self.capturing, self.fail_at, self.calls = mod, False, None, 0
        self.pw = [Wrapper(self, "PIECEWISE", i) for i in range(pieces)] \
            if "PIECEWISE" in modes else []
        self.full = Wrapper(self, "FULL", "full") if "FULL" in modes else None
        self.keys = {m: set(sizes) for m in modes}
        self.engine = RC.Engine(lambda: [w for w in [self.full] + self.pw if w], self.keys,
                                self.capture, self.session, lambda: {"graphs": len(LIVE)})
        for m in modes:
            with self.session():
                for d in sizes:
                    self.capture(m, d)

    def python(self, prev):
        self.calls += 1
        e = sys.modules[self.mod + ".engine"].Engine()
        e.step(1)
        return NS(value=e.ran[-1])

    def session(self):
        runner = self

        class S:
            def __enter__(self):
                runner.capturing = True

            def __exit__(self, *a):
                runner.capturing = False
        return S()

    def capture(self, mode, desc):
        self.forward(desc, mode)

    def forward(self, n, mode=None):
        if mode is None:
            mode = "FULL" if n in self.keys.get("FULL", ()) else \
                "PIECEWISE" if n in self.keys.get("PIECEWISE", ()) else "NONE"
        if mode == "FULL":
            return self.full(mode, n, None).value
        out = None
        for w in self.pw or [None]:
            out = w(mode, n, out) if w else self.python(out)
        return out.value

    def step(self, n):
        """(what the step ran, how it ran): ("v1", "FULL") ..."""
        mode = "FULL" if n in self.keys.get("FULL", ()) else \
            "PIECEWISE" if n in self.keys.get("PIECEWISE", ()) else "NONE"
        return self.forward(n), mode


def host(case, tag, **kw):
    h, names, sides = make_host(case, tag)
    runners = []
    for n, s in zip(names, sides):
        r = Runner(n, **kw)
        s.scope["r1_engine"] = r.engine
        runners.append(r)
    return h, runners, sides


def entries(r):
    return {w.piece: w.concrete_cudagraph_entries for w in [r.full] + r.pw if w}


class R1Test(unittest.TestCase):

    def test_one_bucket_is_recaptured_in_the_gap_and_undo_puts_the_base_graphs_back(self):
        """Input: 2 ranks, FULL graphs for 8/4/2/1 tokens; an edit of the captured code with
        `"rebuild": ["R1"], "r1": {"sizes": [4]}`. Expected: committed and rebuilt in the same
        gap: a 4-token step replays a graph of "v1" (no Python: live witness 0, capture calls
        > 0), 8/2/1-token steps run eager on "v1" (live calls); the receipt's rebuild names
        FULL 4 captured and FULL 8/2/1 eager, with recapture_s. Undo: 4 and 8 replay "base"
        again from the very entry dicts they had, every key is back, and the number of live
        graphs is what it was before the edit. Fails without R1: after the commit the 4-token
        step replays "base", the stale graph. Not: real CUDA graphs (the GPU receipt)."""
        h, rs, _ = host(self, "one")
        before = [entries(r) for r in rs]
        graphs0 = len(LIVE)
        self.assertEqual([r.step(4) for r in rs], [("base", "FULL")] * 2)
        r = h.handle({"op": "txn", "plan": plan("r1one", rebuild=["R1"], r1={"sizes": [4]})})
        rec = r["receipt"]
        self.assertEqual(r["outcome"], "committed", rec["findings"])
        rb = rec["rebuild"]
        self.assertEqual((rb["ok"], rb["routes"], sorted(rb["ranks"])), (True, ["R1"], ["0", "1"]))
        self.assertEqual([x[:2] for x in rb["ranks"]["0"]["captured"]], [["FULL", 4]])
        self.assertEqual(rb["ranks"]["1"]["eager"], {"FULL": [8, 2, 1]})
        self.assertEqual(rec["recapture_s"], rb["seconds"]["R1"])
        self.assertIn("rebuilt", rec["timeline"])
        self.assertEqual([r.step(4) for r in rs], [("v1", "FULL")] * 2)
        self.assertEqual([r.step(8) for r in rs], [("v1", "NONE")] * 2)
        p = h.handle({"op": "prove", "txn": "r1one"})["receipt"]
        self.assertEqual(p["witness"]["workers/rank0"][KEY], 1)          # the eager 8 only
        self.assertGreater(p["witness_capture"]["workers/rank0"][KEY], 0)
        g = h.handle({"op": "graphs"})["workers"]["ranks"][0]
        self.assertEqual(g["keys"], {"FULL": [4]})
        self.assertTrue(h.handle({"op": "undo", "txn": "r1one"})["ok"])
        self.assertEqual([r.step(4) for r in rs], [("base", "FULL")] * 2)
        self.assertEqual([r.step(8) for r in rs], [("base", "FULL")] * 2)
        for b, r in zip(before, rs):
            self.assertTrue(all(b[k] is v for k, v in entries(r).items()))
            self.assertEqual(r.keys, {"FULL": {8, 4, 2, 1}})
        self.assertEqual(len(LIVE), graphs0)
        self.assertTrue(h.handle({"op": "at_base"})["ok"])

    def test_extend_captures_a_left_out_bucket_later(self):
        """Input: R1 of size 4 only, then `r1_extend` of size 8. Expected: the 8-token step
        ran eager on "v1" before the extend and replays a "v1" graph after it; undo puts both
        back to the base graphs. Pain point: a bucket left eager forever, or captured outside
        a gap while a batch replays. Not: on-demand capture during a step."""
        h, rs, _ = host(self, "ext")
        graphs0 = len(LIVE)
        h.handle({"op": "txn", "plan": plan("r1ext", rebuild=["R1"], r1={"sizes": [4]})})
        self.assertEqual(rs[0].step(8), ("v1", "NONE"))
        for x in rs:
            x.fail_at = ("FULL", 2)
        self.assertFalse(h.handle({"op": "r1_extend", "txn": "r1ext", "sizes": [8, 2]})["ok"])
        self.assertEqual(rs[1].keys, {"FULL": {4}})
        self.assertEqual(set(rs[1].full.concrete_cudagraph_entries), {4})
        for x in rs:
            x.fail_at = None
        r = h.handle({"op": "r1_extend", "txn": "r1ext", "sizes": [8]})
        self.assertTrue(r["ok"], r)
        self.assertEqual([x[:2] for x in r["ranks"][0]["captured"]], [["FULL", 8]])
        self.assertEqual([r.step(8) for r in rs], [("v1", "FULL")] * 2)
        self.assertTrue(h.handle({"op": "undo", "txn": "r1ext"})["ok"])
        self.assertEqual([r.step(8) for r in rs], [("base", "FULL")] * 2)
        self.assertEqual(len(LIVE), graphs0)
        self.assertFalse(h.handle({"op": "r1_extend", "txn": "r1ext"})["ok"])

    def test_pieces_of_a_bucket_are_recaptured_together_and_other_modes_are_untouched(self):
        """Input: FULL_AND_PIECEWISE (a FULL wrapper and 3 piecewise wrappers, graphs for
        8/4/2/1 in both modes); R1 with modes ["PIECEWISE"], sizes [2]. Expected: the FULL
        wrapper keeps the very same entry dict (its replays still "base": the caller chose
        the mode), every piecewise wrapper got a new dict holding only size 2, a piecewise
        2-token run replays "v1" through all 3 pieces without an address error; undo gives
        every piece its old dict and the base replay works again. Pain point: recapturing
        only the piece that runs the edit leaves the next piece reading the old capture's
        output address. Not: which piece runs an op (vLLM's split)."""
        h, rs, _ = host(self, "pw", modes=("PIECEWISE", "FULL"))
        before = [entries(r) for r in rs]
        r = h.handle({"op": "txn", "plan": plan("r1pw", rebuild=["R1"],
                                                 r1={"modes": ["PIECEWISE"], "sizes": [2]})})
        self.assertEqual(r["outcome"], "committed", r["receipt"]["findings"])
        for b, x in zip(before, rs):
            e = entries(x)
            self.assertIs(e["full"], b["full"])
            self.assertEqual([sorted(e[i]) for i in range(3)], [[2]] * 3)
            self.assertEqual(x.forward(2, "PIECEWISE"), "v1")
            self.assertEqual(x.forward(2, "FULL"), "base")
        self.assertTrue(h.handle({"op": "undo", "txn": "r1pw"})["ok"])
        for b, x in zip(before, rs):
            self.assertTrue(all(b[k] is v for k, v in entries(x).items()))
            self.assertEqual(x.forward(2, "PIECEWISE"), "base")

    def test_a_capture_that_fails_on_one_rank_undoes_the_edit_on_every_rank(self):
        """Input: 2 ranks, R1 of every size; rank 1's capture of size 2 raises (out of
        memory). Expected: outcome rolled_back with a `rebuild_failed` finding from rank 1;
        in the same gap both ranks are back on "base" in Python and in every graph (rank 0's
        new graphs dropped by its undo, rank 1's by R1 itself), keys complete, at_base ok.
        Pain point: one rank replaying new graphs and another the old ones, or a dispatcher
        key left pointing at a descriptor with no graph (capture error at the next step).
        Not: a capture that hangs (the gap timeout)."""
        h, rs, _ = host(self, "fail")
        rs[1].fail_at = ("FULL", 2)
        r = h.handle({"op": "txn", "plan": plan("r1fail", rebuild=["R1"])})
        rec = r["receipt"]
        self.assertEqual(r["outcome"], "rolled_back")
        self.assertIn(("rebuild_failed", "workers"),
                      [(f["check"], f.get("participant")) for f in rec["findings"]])
        self.assertFalse(rec["rebuild"]["ok"])
        for x in rs:
            self.assertEqual(x.keys, {"FULL": {8, 4, 2, 1}})
            self.assertEqual([x.step(n) for n in (8, 4, 2, 1)], [("base", "FULL")] * 4)
            self.assertEqual(x.forward(3, "NONE"), "base")
        self.assertTrue(h.handle({"op": "at_base"})["ok"])

    def test_five_edit_recapture_undo_cycles_keep_the_graph_count(self):
        """Input: 5 cycles of commit with R1 (every size) and undo. Expected: after each
        commit 4 new graphs live next to the 4 base graphs per rank (kept for undo); after
        each undo the count is back to the start, and each step replays "base". Pain point:
        graphs of every past edit kept alive (FastVideo's 77 GB of graph pools after
        rebuilds). Not: the device pool's bytes (the GPU receipt measures them)."""
        h, rs, _ = host(self, "cyc")
        n0 = len(LIVE)
        for i in range(5):
            r = h.handle({"op": "txn", "plan": plan(f"r1cyc{i}", rebuild=["R1"])})
            self.assertEqual(r["outcome"], "committed")
            self.assertEqual(len(LIVE), n0 + 8)
            self.assertTrue(h.handle({"op": "undo", "txn": f"r1cyc{i}"})["ok"])
            self.assertEqual(len(LIVE), n0)
            self.assertEqual([x.step(4) for x in rs], [("base", "FULL")] * 2)

    def test_a_plan_asking_for_r1_turns_graph_findings_advisory(self):
        """Input: graph-mode workers (FULL_AND_PIECEWISE, torch.compile), the toy as model
        code. Expected: without "rebuild" the edit is a blocking captured_target (aborted);
        with "rebuild": ["R1"] the same finding is advisory, says the host recaptures, and the
        transaction commits and replays "v1". Pain point: the finding told users to restart
        though the host can recapture. Not: vLLM's real config."""
        h, rs, sides = host(self, "adv")
        for s in sides:
            s.scope["worker"] = worker()
            s.captured_prefixes = ("toy.",)
        self.assertEqual(h.handle({"op": "txn", "plan": plan("r1adv0")})["outcome"], "aborted")
        r = h.handle({"op": "txn", "plan": plan("r1adv", rebuild=["R1"])})
        fs = r["receipt"]["findings"]
        self.assertEqual(r["outcome"], "committed", fs)
        self.assertEqual({(f["check"], f.get("advisory")) for f in fs},
                         {("captured_target", True)})
        self.assertIn("recaptures the graphs", fs[0]["observed"])
        self.assertEqual(rs[0].step(4), ("v1", "FULL"))


if __name__ == "__main__":
    unittest.main()
