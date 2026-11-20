"""CPU golden tests of selective CUDA-graph recapture (recapture.py, the host's
RecaptureCoordinator, the side's `recapture`). Run as test_host.py says.

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
import queue
import sys
import threading
import time
import unittest
import weakref
from types import SimpleNamespace as NS

from . import host as H
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


class RecaptureTest(unittest.TestCase):

    def test_one_bucket_is_recaptured_in_the_gap_and_undo_puts_the_base_graphs_back(self):
        """Input: 2 ranks, FULL graphs for 8/4/2/1 tokens; an edit of the captured code with
        `"rebuild": ["recapture"], "recapture": {"sizes": [4]}`. Expected: committed and rebuilt in the same
        gap: a 4-token step replays a graph of "v1" (no Python: live witness 0, capture calls
        > 0), 8/2/1-token steps run eager on "v1" (live calls); the receipt's rebuild names
        FULL 4 captured and FULL 8/2/1 eager, with recapture_s. Undo: 4 and 8 replay "base"
        again from the very entry dicts they had, every key is back, and the number of live
        graphs is what it was before the edit. Fails without recapture: after the commit the 4-token
        step replays "base", the stale graph. Not: real CUDA graphs (the GPU receipt)."""
        h, rs, _ = host(self, "one")
        before = [entries(r) for r in rs]
        graphs0 = len(LIVE)
        self.assertEqual([r.step(4) for r in rs], [("base", "FULL")] * 2)
        r = h.handle({"op": "txn", "plan": plan("r1one", rebuild=["recapture"], recapture={"sizes": [4]})})
        rec = r["receipt"]
        self.assertEqual(r["outcome"], "committed", rec["findings"])
        rb = rec["rebuild"]
        self.assertEqual((rb["ok"], rb["routes"], sorted(rb["ranks"])), (True, ["recapture"], ["0", "1"]))
        self.assertEqual([x[:2] for x in rb["ranks"]["0"]["captured"]], [["FULL", 4]])
        self.assertEqual(rb["ranks"]["1"]["eager"], {"FULL": [8, 2, 1]})
        self.assertEqual(rec["recapture_s"], rb["seconds"]["recapture"])
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
        """Input: recapture of size 4 only, then `recapture_extend` of size 8. Expected: the 8-token step
        ran eager on "v1" before the extend and replays a "v1" graph after it; undo puts both
        back to the base graphs. Pain point: a bucket left eager forever, or captured outside
        a gap while a batch replays. Not: on-demand capture during a step."""
        h, rs, _ = host(self, "ext")
        graphs0 = len(LIVE)
        h.handle({"op": "txn", "plan": plan("r1ext", rebuild=["recapture"], recapture={"sizes": [4]})})
        self.assertEqual(rs[0].step(8), ("v1", "NONE"))
        for x in rs:
            x.fail_at = ("FULL", 2)
        self.assertFalse(h.handle({"op": "recapture_extend", "txn": "r1ext", "sizes": [8, 2]})["ok"])
        self.assertEqual(rs[1].keys, {"FULL": {4}})
        self.assertEqual(set(rs[1].full.concrete_cudagraph_entries), {4})
        for x in rs:
            x.fail_at = None
        r = h.handle({"op": "recapture_extend", "txn": "r1ext", "sizes": [8]})
        self.assertTrue(r["ok"], r)
        self.assertEqual([x[:2] for x in r["ranks"][0]["captured"]], [["FULL", 8]])
        self.assertEqual([r.step(8) for r in rs], [("v1", "FULL")] * 2)
        self.assertTrue(h.handle({"op": "undo", "txn": "r1ext"})["ok"])
        self.assertEqual([r.step(8) for r in rs], [("base", "FULL")] * 2)
        self.assertEqual(len(LIVE), graphs0)
        self.assertFalse(h.handle({"op": "recapture_extend", "txn": "r1ext"})["ok"])

    def test_an_extend_that_fails_on_one_rank_is_dropped_on_every_rank(self):
        """Input: 2 ranks, FULL graphs for 8/4/2/1; recapture of size 1, then
        `recapture_extend` of size 4 where rank 1's capture of 4 raises. Expected: the extend
        is not ok and says what it dropped; in the same gap rank 0 drops the 4-token graph it
        captured, so both ranks run a 4-token step eager on "v1" (same keys, same entries);
        a second extend captures 4 on both ranks and undo puts the base graphs back. Pain
        point: rank 0 replaying a FULL graph while rank 1 runs the same step eager (TP ranks
        out of step). Fails before the fix: rank 0 keeps keys {1, 4}. Not: real NCCL ranks."""
        lab, runners, _ = host(self, "extfail")
        lab.handle({"op": "txn", "plan": plan("r1xf", rebuild=["recapture"], recapture={"sizes": [1]})})
        runners[1].fail_at = ("FULL", 4)
        reply = lab.handle({"op": "recapture_extend", "txn": "r1xf", "sizes": [4]})
        runners[1].fail_at = None
        self.assertFalse(reply["ok"])
        self.assertEqual([runner.keys for runner in runners], [{"FULL": {1}}] * 2)
        self.assertEqual([set(runner.full.concrete_cudagraph_entries) for runner in runners],
                         [{1}] * 2)
        self.assertEqual([runner.step(4) for runner in runners], [("v1", "NONE")] * 2)
        self.assertEqual([rank["dropped"] for rank in reply["dropped"]["ranks"]], [True, False])
        self.assertFalse(lab.co.poison)
        self.assertTrue(lab.handle({"op": "recapture_extend", "txn": "r1xf", "sizes": [4]})["ok"])
        self.assertEqual([runner.step(4) for runner in runners], [("v1", "FULL")] * 2)
        self.assertTrue(lab.handle({"op": "undo", "txn": "r1xf"})["ok"])
        self.assertEqual([runner.step(4) for runner in runners], [("base", "FULL")] * 2)

    def test_a_drop_that_fails_poisons_the_host(self):
        """Input: 2 ranks; recapture of size 1, then `recapture_extend` of size 4 where rank
        1's capture raises and rank 0 cannot drop its captured 4 (its drop replies not ok).
        Expected: the extend is not ok with a `rebuild_failed` finding, the host is poisoned
        (rank 0 replays FULL 4, rank 1 runs it eager), the next `txn` and the next
        `recapture_extend` are refused, and undo still puts both ranks back on the base
        graphs. Pain point: ranks left replaying different graphs with the host still taking
        transactions. Fails without the poison branch: no poison, the next txn commits.
        Not: why a real drop fails (the drop is made to fail here)."""
        lab, runners, sides = host(self, "dropfail")
        lab.handle({"op": "txn", "plan": plan("r1df", rebuild=["recapture"], recapture={"sizes": [1]})})
        runners[1].fail_at = ("FULL", 4)
        sides[0].recapture.drop_extend = lambda txn, journal, extend_id: {
            "ok": False, "error": "drop refused on rank 0", "findings": []}
        reply = lab.handle({"op": "recapture_extend", "txn": "r1df", "sizes": [4]})
        runners[1].fail_at = None
        self.assertFalse(reply["ok"])
        self.assertIn("rebuild_failed", [finding["check"] for finding in reply["findings"]])
        self.assertEqual([finding["participant"] for finding in lab.co.poison], ["workers"])
        self.assertEqual([runner.step(4) for runner in runners], [("v1", "FULL"), ("v1", "NONE")])
        self.assertEqual(lab.handle({"op": "txn", "plan": plan("r1df2")})["outcome"], "refused")
        again = lab.handle({"op": "recapture_extend", "txn": "r1df", "sizes": [4]})
        self.assertIn("poisoned", again["error"])
        self.assertTrue(lab.handle({"op": "undo", "txn": "r1df"})["ok"])
        for runner in runners:
            self.assertEqual(runner.keys, {"FULL": {8, 4, 2, 1}})
            self.assertEqual([runner.step(size) for size in (8, 4, 2, 1)], [("base", "FULL")] * 4)

    def test_a_dropped_extend_gives_every_mode_its_left_out_buckets_back(self):
        """Input: FULL_AND_PIECEWISE, sizes 4/2/1; recapture of size 1; an extend of 4 that
        fails on rank 1 at FULL 4 (after its PIECEWISE 4), so rank 0 drops both of its new
        graphs; then the same extend again. Expected: the second extend captures PIECEWISE 4
        and FULL 4 on both ranks (keys {1, 4} in both modes on both ranks). Pain point: the
        drop restored the left-out list of the last mode only, so rank 0 never captured
        PIECEWISE 4 again. Fails before the fix: rank 0's PIECEWISE keys stay {1}."""
        lab, runners, _ = host(self, "extmm", modes=("PIECEWISE", "FULL"), sizes=(4, 2, 1))
        lab.handle({"op": "txn", "plan": plan("r1mm", rebuild=["recapture"], recapture={"sizes": [1]})})
        runners[1].fail_at = ("FULL", 4)
        self.assertFalse(lab.handle({"op": "recapture_extend", "txn": "r1mm", "sizes": [4]})["ok"])
        runners[1].fail_at = None
        reply = lab.handle({"op": "recapture_extend", "txn": "r1mm", "sizes": [4]})
        self.assertTrue(reply["ok"], reply)
        self.assertEqual([runner.keys for runner in runners],
                         [{"PIECEWISE": {1, 4}, "FULL": {1, 4}}] * 2)

    def test_undo_after_an_extend_with_nothing_to_capture(self):
        """Input: FULL-only ranks; a recapture plan with modes ["PIECEWISE"] (no such graphs:
        nothing chosen), then `recapture_extend`, then undo. Expected: the extend is ok with
        nothing captured and undo is ok, both ranks back on "base". Pain point: the extend's
        undo entry read a loop variable that was never set (UnboundLocalError), so undo of the
        transaction was `undo_incomplete`. Fails before the fix: undo ok False."""
        lab, runners, _ = host(self, "extnone", sizes=(4, 2, 1))
        reply = lab.handle({"op": "txn", "plan": plan("r1none", rebuild=["recapture"],
                                                       recapture={"modes": ["PIECEWISE"]})})
        self.assertEqual(reply["outcome"], "committed")
        self.assertTrue(lab.handle({"op": "recapture_extend", "txn": "r1none"})["ok"])
        undo = lab.handle({"op": "undo", "txn": "r1none"})
        self.assertTrue(undo["ok"], undo.get("findings") or undo)
        self.assertEqual([runner.step(4) for runner in runners], [("base", "FULL")] * 2)

    def test_an_extend_that_outlives_the_gap_poisons_the_host(self):
        """Input: the host driven from this thread with a stand-in busy loop on its own
        thread; `recapture_extend` whose collective_rpc takes 0.5 s, gap_timeout_s 0.2.
        Expected: the op is not ok, the host is poisoned (workers), the worker group refuses
        later calls, and a later extend is refused. Pain point: an extend still capturing in
        the loop after its caller gave up, with the host still taking transactions. Fails
        before the fix: no poison, the group stays usable. Not: vLLM's real busy loop."""
        lab, _, _ = host(self, "extlate")
        lab.handle({"op": "txn", "plan": plan("r1late", rebuild=["recapture"], recapture={"sizes": [4]})})
        jobs = queue.SimpleQueue()
        lab.core.input_queue = jobs

        def loop():
            while True:
                _, job = jobs.get()
                if job is None:
                    return
                job.run()
        loop_thread = threading.Thread(target=loop, daemon=True)
        loop_thread.start()
        lab.loop_thread = loop_thread
        self.addCleanup(loop_thread.join, 5)
        self.addCleanup(jobs.put, (H.TAG, None))
        executor = lab.core.model_executor
        full_rpc = executor.collective_rpc

        def slow_extend_rpc(name, timeout=None, args=()):
            if args[0] == "recapture_extend":
                time.sleep(0.5)
            return full_rpc(name, timeout, args)
        executor.collective_rpc = slow_extend_rpc
        reply = lab.handle({"op": "recapture_extend", "txn": "r1late", "sizes": [8],
                            "gap_timeout_s": 0.2})
        self.assertFalse(reply["ok"])
        self.assertEqual([finding["participant"] for finding in lab.co.poison], ["workers"])
        self.assertTrue(lab.workers.broken)
        time.sleep(0.5)                          # the late extend finishes in the loop
        again = lab.handle({"op": "recapture_extend", "txn": "r1late", "sizes": [2]})
        self.assertFalse(again["ok"])
        self.assertIn("poisoned", again["error"])

    def test_an_undo_after_a_failed_recapture_has_the_phase2_deadline(self):
        """Input: 2 ranks; rank 1's capture raises, and rank 0's undo of the rolled-back edit
        takes 0.6 s; phase-2 deadline 0.4 s. Expected: rollback status partial, rank 0 named
        `deadline_missed`, the host poisoned and a pending rollback held; once rank 0's late
        undo lands, both ranks run "base", the pending rollback clears and the next
        transaction is still refused until `recovered`. Pain point: that undo ran with no
        deadline, so one slow rank held the gap as long as it liked and a late rank never
        poisoned the host. Fails before the fix: status undone, no poison."""
        lab, runners, _ = host(self, "rbdl")
        runners[1].fail_at = ("FULL", 2)
        slow_undo = plan("r1dl", rebuild=["R1"], faults={
            "worker:0": {"method": "undo", "delay_s": 0.6, "fail": False}})
        reply = lab.handle({"op": "txn", "plan": slow_undo, "phase2_deadline_s": 0.4})
        receipt = reply["receipt"]
        self.assertEqual((reply["outcome"], receipt["rollback"]["status"]),
                         ("rolled_back", "partial"))
        undo = receipt["participants"]["workers"]["undo"]
        self.assertEqual([finding["check"] for finding in undo["ranks"][0]["findings"]],
                         ["deadline_missed"])
        self.assertEqual([finding["participant"] for finding in lab.co.poison], ["workers"])
        self.assertTrue(lab.co.has_pending_rollback())
        time.sleep(0.6)                          # rank 0's late undo lands
        self.assertFalse(lab.co.has_pending_rollback())
        for runner in runners:
            self.assertEqual([runner.step(size) for size in (8, 4, 2, 1)], [("base", "FULL")] * 4)
        self.assertEqual(lab.handle({"op": "txn", "plan": plan("r1dl2")})["outcome"], "refused")

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


R1Test = RecaptureTest


class V2RecaptureTest(unittest.TestCase):
    """Pure-Python tests of vLLM V2 `ModelCudaGraphManager` binding & selective R1 recapture."""

    def _make_v2_stubs(self):
        import contextlib
        import enum
        import types

        class CUDAGraphMode(enum.Enum):
            NONE = 0
            PIECEWISE = 1
            FULL = 2

        class BatchExecutionDescriptor:
            def __init__(self, cg_mode, num_tokens):
                self.cg_mode = cg_mode
                self.num_tokens = num_tokens

            def __hash__(self):
                return hash((self.cg_mode, self.num_tokens))

            def __eq__(self, other):
                return isinstance(other, BatchExecutionDescriptor) and (
                    self.cg_mode, self.num_tokens) == (other.cg_mode, other.num_tokens)

            def __repr__(self):
                return f"Desc({self.cg_mode.name},{self.num_tokens})"

        events = {"streams": [], "captured_streams": [], "gc_freeze": 0, "lora_enter": 0,
                  "kv_reset": 0, "capturing_states": [], "groups": []}

        class FakeStream:
            def __init__(self, device=None):
                self.device = device
                events["streams"].append(self)

        class FakeTorch(types.ModuleType):
            pass

        torch_mod = FakeTorch("torch")
        torch_mod.cuda = NS(
            Stream=FakeStream,
            synchronize=lambda: None,
            mem_get_info=lambda: (1024, 2048),
            memory_reserved=lambda: 512,
            memory_allocated=lambda: 256,
        )

        class FakeWrapper:
            _all_instances = []

        class FakeBreakable:
            _all_instances = []

        class GraphCaptureContext:
            def __init__(self, stream):
                self.stream = stream

        class FakeGroup:
            def __init__(self, tag):
                self.tag = tag

            @contextlib.contextmanager
            def graph_capture(self, ctx):
                events["groups"].append((self.tag, "enter", ctx.stream))
                try:
                    yield
                finally:
                    events["groups"].append((self.tag, "exit", ctx.stream))

        def set_capturing(flag):
            events["capturing_states"].append(flag)

        @contextlib.contextmanager
        def freeze_gc():
            events["gc_freeze"] += 1
            yield

        cg_utils = types.ModuleType("vllm.v1.worker.gpu.cudagraph_utils")

        @contextlib.contextmanager
        def default_graph_capture(device=None):
            # Emulate vLLM's unpatched behavior allocating a brand-new stream per capture.
            yield GraphCaptureContext(FakeStream(device=device))

        cg_utils.graph_capture = default_graph_capture

        class FakeV2Manager:
            def __init__(self, mod_name, sizes=(4, 2, 1), with_piecewise=False):
                self.mod_name = mod_name
                self.device = f"cuda:v2-{id(self)}"
                self._needs = True
                self._graphs_captured = True
                self.ubatch_runner = None
                self.breakable_cg_runner = None
                self.fail_at = None
                self.graphs = {}
                self._candidates = {}
                self._capture_descs = {CUDAGraphMode.FULL: []}
                if with_piecewise:
                    self.breakable_cg_runner = NS(runtime_mode=CUDAGraphMode.PIECEWISE, entries={})
                    self._capture_descs[CUDAGraphMode.PIECEWISE] = []
                for n in sizes:
                    d_full = BatchExecutionDescriptor(CUDAGraphMode.FULL, n)
                    self._capture_descs[CUDAGraphMode.FULL].append(d_full)
                    cands = [d_full]
                    if with_piecewise:
                        d_pw = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, n)
                        self._capture_descs[CUDAGraphMode.PIECEWISE].append(d_pw)
                        cands.append(d_pw)
                    self._candidates[(n, 0)] = cands
                # Initial base capture
                for mode, descs in self._capture_descs.items():
                    for d in descs:
                        self._record(mode, d)

            def needs_capture(self):
                return self._needs

            def _run_python(self):
                e = sys.modules[self.mod_name + ".engine"].Engine()
                e.step(1)
                return e.ran[-1]

            def _record(self, mode, desc):
                val = self._run_python()
                if mode == CUDAGraphMode.FULL:
                    self.graphs[desc] = Graph(val, None)
                else:
                    self.breakable_cg_runner.entries[desc] = Graph(val, None)

            def capture(self, model, model_state, input_buffers, intermediate_tensors,
                        block_tables, attn_groups, kv_cache_config, *, pcp_manager=None,
                        has_lora=False, use_aux_hidden_state_outputs=False,
                        lora_capture_hook=None):
                with cg_utils.graph_capture(device=self.device) as ctx:
                    events["captured_streams"].append(ctx.stream)
                    for mode, descs in self._capture_descs.items():
                        for d in descs:
                            if self.fail_at == (mode.name, d.num_tokens):
                                raise RuntimeError(f"boom on {mode.name} {d.num_tokens}")
                            self._record(mode, d)
                self._graphs_captured = True

            def dispatch(self, num_tokens):
                cands = self._candidates.get((num_tokens, 0), [])
                if not cands:
                    return (self._run_python(), CUDAGraphMode.NONE.name)
                chosen = cands[0]
                if chosen.cg_mode == CUDAGraphMode.FULL:
                    assert chosen in self.graphs, f"missing graph for {chosen}"
                    return (self.graphs[chosen].value, chosen.cg_mode.name)
                assert chosen in self.breakable_cg_runner.entries
                return (self.breakable_cg_runner.entries[chosen].value, chosen.cg_mode.name)

        def make_worker(mod_name, sizes=(4, 2, 1), with_piecewise=False):
            mgr = FakeV2Manager(mod_name, sizes=sizes, with_piecewise=with_piecewise)

            @contextlib.contextmanager
            def setup_loras(_cfg):
                events["lora_enter"] += 1
                yield

            runner = NS(
                cudagraph_manager=mgr,
                device=mgr.device,
                model=NS(),
                model_state=NS(supports_mm_inputs=False),
                input_buffers=NS(),
                pcp_manager=None,
                intermediate_tensors=None,
                block_tables=None,
                attn_groups=None,
                kv_cache_config=None,
                lora_config=None,
                use_aux_hidden_state_outputs=False,
                maybe_setup_dummy_loras=setup_loras,
                kv_connector=NS(reset_capture_state=lambda: events.__setitem__(
                    "kv_reset", events["kv_reset"] + 1)),
            )
            return NS(model_runner=runner), mgr

        mods = {
            "torch": torch_mod,
            "vllm": types.ModuleType("vllm"),
            "vllm.config": NS(CUDAGraphMode=CUDAGraphMode),
            "vllm.compilation": types.ModuleType("vllm.compilation"),
            "vllm.compilation.cuda_graph": NS(CUDAGraphWrapper=FakeWrapper),
            "vllm.compilation.breakable_cudagraph": NS(BreakableCUDAGraphWrapper=FakeBreakable),
            "vllm.compilation.monitor": NS(set_cudagraph_capturing_enabled=set_capturing),
            "vllm.distributed": types.ModuleType("vllm.distributed"),
            "vllm.distributed.parallel_state": NS(
                GraphCaptureContext=GraphCaptureContext,
                get_tp_group=lambda: FakeGroup("tp"),
                get_pp_group=lambda: FakeGroup("pp"),
                get_dp_group=lambda: FakeGroup("dp"),
            ),
            "vllm.utils": types.ModuleType("vllm.utils"),
            "vllm.utils.gc_utils": NS(freeze_gc_for_cudagraph_capture=freeze_gc),
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
            "vllm.v1.worker.gpu": types.ModuleType("vllm.v1.worker.gpu"),
            "vllm.v1.worker.gpu.lora_utils": NS(create_lora_capture_hook=lambda _c, _r: None),
            "vllm.v1.worker.gpu.cudagraph_utils": cg_utils,
        }
        return mods, cg_utils, default_graph_capture, make_worker, events, CUDAGraphMode

    def test_v2_selective_recapture_extend_priority_undo_and_stream_reuse(self):
        """Input: 2-rank V2 ModelCudaGraphManager (FULL + PIECEWISE for sizes 4/2/1).
        Expected: selective FULL sizes=[1] removes FULL(4)/FULL(2) from _candidates (dispatch falls
        back to PIECEWISE or NONE); extend([4, 2]) restores FULL ahead of PIECEWISE in original
        priority order; O(1) undo restores original `mgr.graphs` dict & `_candidates`; 5 cycles
        preserve live graph count and reuse a single CUDA stream via `graph_capture` intercept."""
        from unittest import mock
        mods, cg_utils, orig_gc, make_worker, events, _ = self._make_v2_stubs()
        h, names, sides = make_host(self, "v2sel")
        workers, mgrs = [], []
        with mock.patch.dict(sys.modules, mods):
            for n, s in zip(names, sides):
                w, mgr = make_worker(n, sizes=(4, 2, 1), with_piecewise=True)
                workers.append(w)
                mgrs.append(mgr)
                s.scope["r1_engine"] = RC.vllm_engine(w)
            old_dicts = [m.graphs for m in mgrs]
            orig_cands = [{k: list(v) for k, v in m._candidates.items()} for m in mgrs]
            orig_descs = [m._capture_descs for m in mgrs]

            # Selective FULL recapture of size 1 only: FULL(4) & FULL(2) drop out -> fall back to PIECEWISE
            r = h.handle({"op": "txn", "plan": plan(
                "v2one", rebuild=["R1"], r1={"modes": ["FULL"], "sizes": [1]})})
            self.assertEqual(r["outcome"], "committed", r["receipt"]["findings"])
            for m, od in zip(mgrs, orig_descs):
                self.assertEqual(m.dispatch(1), ("v1", "FULL"))
                self.assertEqual(m.dispatch(4), ("base", "PIECEWISE"))
                self.assertEqual(m.dispatch(2), ("base", "PIECEWISE"))
                self.assertIs(m._capture_descs, od)
            self.assertIs(cg_utils.graph_capture, orig_gc)

            # Extend sizes [4, 2]: FULL restored AHEAD of PIECEWISE (original priority order!)
            ext = h.handle({"op": "recapture_extend", "txn": "v2one", "sizes": [4, 2]})
            self.assertTrue(ext["ok"], ext)
            for m, oc in zip(mgrs, orig_cands):
                self.assertEqual(m._candidates, oc)
                self.assertEqual([m.dispatch(n) for n in (4, 2, 1)], [("v1", "FULL")] * 3)

            # O(1) undo restores original dict identity and candidates
            self.assertTrue(h.handle({"op": "undo", "txn": "v2one"})["ok"])
            for m, od_dict, oc in zip(mgrs, old_dicts, orig_cands):
                self.assertIs(m.graphs, od_dict)
                self.assertEqual(m._candidates, oc)
                self.assertEqual([m.dispatch(n) for n in (4, 2, 1)], [("base", "FULL")] * 3)
            self.assertTrue(h.handle({"op": "at_base"})["ok"])

            # 5 full recapture/undo cycles on FULL-only: stable graph count & single stream per device
            h2, names2, sides2 = make_host(self, "v2cyc")
            events["streams"].clear()
            events["captured_streams"].clear()
            events["kv_reset"] = 0
            w2, m2 = make_worker(names2[0], sizes=(4, 2, 1), with_piecewise=False)
            w2b, m2b = make_worker(names2[1], sizes=(4, 2, 1), with_piecewise=False)
            sides2[0].scope["r1_engine"] = RC.vllm_engine(w2)
            sides2[1].scope["r1_engine"] = RC.vllm_engine(w2b)
            n0 = len(LIVE)
            for i in range(5):
                res = h2.handle({"op": "txn", "plan": plan(
                    f"v2c{i}", rebuild=["R1"], r1={"sizes": [1]})})
                self.assertEqual(res["outcome"], "committed")
                # Uncaptured sizes 4 and 2 fall through dispatch() to NONE (eager on v1)
                self.assertEqual(m2.dispatch(1), ("v1", "FULL"))
                self.assertEqual(m2.dispatch(4), ("v1", "NONE"))
                self.assertEqual(m2.dispatch(2), ("v1", "NONE"))
                self.assertEqual(len(LIVE), n0 + 2)
                self.assertTrue(h2.handle({"op": "undo", "txn": f"v2c{i}"})["ok"])
                self.assertEqual(len(LIVE), n0)
                self.assertEqual([m2.dispatch(n) for n in (4, 2, 1)], [("base", "FULL")] * 3)
            # Exactly 1 stream created per device across all 5 cycles (reused via graph_capture patch)
            self.assertEqual(len(events["streams"]), 2)
            self.assertEqual(set(events["captured_streams"]), set(events["streams"]))
            self.assertEqual(events["kv_reset"], 10)
            self.assertIs(cg_utils.graph_capture, orig_gc)

    def test_v2_failure_rollback_and_unsupported_refusals(self):
        """Input: V2 capture error mid-batch; plus DBO (ubatch_runner), encoder cudagraph, and
        eager/uncaptured V2 managers. Expected: failure rolls back graphs/_candidates/graph_capture;
        DBO and encoder graphs raise NotImplementedError; eager returns None."""
        from unittest import mock
        mods, cg_utils, orig_gc, make_worker, _, _ = self._make_v2_stubs()
        h, names, sides = make_host(self, "v2fail")
        with mock.patch.dict(sys.modules, mods):
            mgrs = []
            for n, s in zip(names, sides):
                w, mgr = make_worker(n, sizes=(4, 2, 1))
                mgrs.append(mgr)
                s.scope["r1_engine"] = RC.vllm_engine(w)
            orig_dicts = [m.graphs for m in mgrs]
            orig_cands = [{k: list(v) for k, v in m._candidates.items()} for m in mgrs]
            mgrs[1].fail_at = ("FULL", 2)
            r = h.handle({"op": "txn", "plan": plan("v2boom", rebuild=["R1"])})
            self.assertEqual(r["outcome"], "rolled_back")
            self.assertIn("rebuild_failed", [f["check"] for f in r["receipt"]["findings"]])
            for m, od, oc in zip(mgrs, orig_dicts, orig_cands):
                self.assertIs(m.graphs, od)
                self.assertEqual(m._candidates, oc)
                self.assertEqual([m.dispatch(n) for n in (4, 2, 1)], [("base", "FULL")] * 3)
            self.assertIs(cg_utils.graph_capture, orig_gc)
            self.assertTrue(h.handle({"op": "at_base"})["ok"])

            # DBO refusal
            w_dbo, m_dbo = make_worker(names[0])
            m_dbo.ubatch_runner = object()
            with self.assertRaises(NotImplementedError):
                RC.vllm_engine(w_dbo)

            # Encoder CUDA graph refusal
            w_enc, _ = make_worker(names[0])
            w_enc.model_runner.model_state = NS(
                supports_mm_inputs=True, encoder_runner=NS(has_cudagraph=lambda: True)
            )
            with self.assertRaises(NotImplementedError):
                RC.vllm_engine(w_enc)

            # Eager / uncaptured returns None
            w_eager, m_eager = make_worker(names[0])
            m_eager._needs = False
            self.assertIsNone(RC.vllm_engine(w_eager))
            w_empty, m_empty = make_worker(names[0])
            m_empty._graphs_captured = False
            m_empty.graphs.clear()
            self.assertIsNone(RC.vllm_engine(w_empty))


if __name__ == "__main__":
    unittest.main()
