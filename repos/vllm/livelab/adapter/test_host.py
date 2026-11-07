"""CPU golden tests of the adapter's host, worker group and endpoints (no vLLM, no GPU).

Run from the repo root:
    python3 -m unittest discover -s repos/vllm/livelab/adapter -t repos/vllm/livelab -v

Only the transport is replaced: `collective_rpc` becomes a loop over in-process `Endpoint`s,
one call on every rank in rank order, as vLLM's executor does. Each rank has its own copy of
the toy engine (its own module objects, as in its own process). Host, WorkerGroup, Endpoint
and the kernel are the real code.
"""
import importlib
import itertools
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from kernel import locate as L
from kernel import toy

from . import host as H
from .side import Side

ENG = toy.source("engine")
STEP_LINE = next(i for i, x in enumerate(ENG.splitlines(), 1) if "def step" in x)
_n = itertools.count()


class Executor:
    """vLLM's collective_rpc, in process: every rank answers, in rank order."""

    def __init__(self, endpoints):
        self.endpoints, self.calls = endpoints, []

    def collective_rpc(self, name, timeout=None, args=()):
        method, payload = args
        self.calls.append(method)
        return [ep.dispatch(method, payload) for ep in self.endpoints]


def plan(tid, **kw):
    """A worker-only plan: Engine.step appends "v1" instead of "base"."""
    return {"id": tid, **kw, "modules": {"toy.engine": {
        "on": ["worker"], "base_hash": L.text_hash(ENG),
        "source": ENG.replace('self.ran.append("base")', 'self.ran.append("v1")'),
        "targets": [["Engine.step", STEP_LINE]]}}}


def make_host(case, tag, ranks=2):
    """The real Host over `ranks` worker Sides (each its own toy copy) and a core."""
    names, eps = [], []
    for r in range(ranks):
        name = f"adh{next(_n)}_{tag}{r}"
        toy.load_copy(name)
        names.append(name)
        eps.append(Side("worker", scope={"worker": NS()}, rank=r, alias={"toy": name}))
    d = tempfile.mkdtemp()
    case.addCleanup(shutil.rmtree, d, True)
    core = NS(model_executor=Executor(eps))
    with mock.patch.dict(sys.modules, {"vllm": types.SimpleNamespace(__version__="cpu")}):
        h = H.Host(core, d)
    return h, names, eps


def ran(name):
    e = sys.modules[name + ".engine"].Engine()
    e.step(1)
    return e.ran[-1]


class TransactionTest(unittest.TestCase):

    def test_two_ranks_switch_in_one_gap_and_undo_restores_the_same_code(self):
        """Input: the host over 2 worker ranks; a worker-only edit of Engine.step. Expected:
        committed; both ranks run "v1"; the receipt keeps each rank's code hash and
        ranks_agree {"workers": True}; the core abstained (no module for it), so it does not
        veto with empty_change; one step per rank gives witness 1 for workers/rank0 and
        workers/rank1; undo puts back the very code object each rank ran before, and at_base
        is ok. Not: real processes (the GPU receipt runs this on vLLM)."""
        h, names, _ = make_host(self, "t0")
        base = [sys.modules[n + ".engine"].Engine.step.__code__ for n in names]
        r = h.handle({"op": "txn", "plan": plan("t0")})
        rec = r["receipt"]
        self.assertEqual((r["outcome"], rec["commit_hashes"]["ranks_agree"]),
                         ("committed", {"workers": True}))
        self.assertEqual(sorted(rec["commit_hashes"]["ranks"]["workers"]), ["0", "1"])
        self.assertTrue(rec["participants"]["core"]["prepare"]["abstain"])
        self.assertEqual([ran(n) for n in names], ["v1", "v1"])
        p = h.handle({"op": "prove", "txn": "t0"})["receipt"]
        key = f"toy.engine:Engine.step:{STEP_LINE}"
        self.assertEqual({w: c.get(key) for w, c in p["witness"].items()},
                         {"workers/rank0": 1, "workers/rank1": 1, "core": None})
        self.assertTrue(h.handle({"op": "undo", "txn": "t0"})["ok"])
        self.assertEqual([ran(n) for n in names], ["base", "base"])
        self.assertEqual([sys.modules[n + ".engine"].Engine.step.__code__ for n in names],
                         base)
        self.assertTrue(h.handle({"op": "at_base"})["ok"])

    def test_a_rank_that_raises_replies_and_the_others_roll_back(self):
        """Input: rank 1's commit raises RuntimeError. Expected: every rank still replies (the
        commit reply has ranks 0 and 1; rank 1's is not-ok and names "worker rank 1"), the
        transaction is rolled_back, rank 0 runs "base" again, and the next call reads its own
        replies (pstatus: both ranks "undone"). Pain point: a raising RPC method leaves the
        other ranks' replies queued, so every later call reads the previous call's replies.
        Not: vLLM's real executor queue."""
        h, names, eps = make_host(self, "raise")

        def boom(txn):
            raise RuntimeError("rank 1 lost its device")
        eps[1].lp.commit = boom
        r = h.handle({"op": "txn", "plan": plan("rx")})
        commit = r["receipt"]["participants"]["workers"]["commit"]
        self.assertEqual([(x["rank"], x["ok"]) for x in commit["ranks"]], [(0, True), (1, False)])
        self.assertIn("worker rank 1", commit["ranks"][1]["error"])
        self.assertEqual(r["outcome"], "rolled_back")
        self.assertEqual([ran(n) for n in names], ["base", "base"])
        ps = h.handle({"op": "pstatus", "txn": "rx"})["workers"]
        self.assertEqual([(x["rank"], x["state"]) for x in ps["ranks"]],
                         [(0, "undone"), (1, "undone")])

    def test_a_timed_out_rpc_poisons_the_host_and_refuses_later_calls(self):
        """Input: collective_rpc raises TimeoutError on commit (a rank that never answers).
        Expected: a `deadline_missed` finding, the host poisoned, the next transaction
        refused with `session_poisoned`, and the worker group refusing every later call
        (its late reply would be read by the next call). Not: a real hung rank."""
        h, names, _ = make_host(self, "hang")
        ex = h.core.model_executor
        full = ex.collective_rpc

        def rpc(name, timeout=None, args=()):
            if args[0] == "commit":
                raise TimeoutError("no reply")
            return full(name, timeout, args)
        ex.collective_rpc = rpc
        r = h.handle({"op": "txn", "plan": plan("hx")})
        checks = {f["check"] for f in r["receipt"]["findings"]}
        self.assertIn("deadline_missed", checks)
        self.assertTrue(h.co.poison)
        again = h.handle({"op": "txn", "plan": plan("hy")})
        self.assertEqual((again["outcome"], again["findings"][0]["check"]),
                         ("refused", "session_poisoned"))
        self.assertIn("unusable", h.workers.status("hx")["error"])

    def test_a_loop_job_that_outlives_its_wait_poisons_the_host(self):
        """Input: the host driven from a control thread with a stand-in busy loop on its own
        thread; the first collective_rpc (prepare) takes 0.5 s, the worker group waits 0.2 s
        for its job. Expected: the transaction aborts with `deadline_missed` from the
        workers, the host is poisoned, the next transaction is refused with
        `session_poisoned`, and the worker group refuses every later call. Fails before the
        fix: the late prepare was an ordinary no vote, the group stayed usable and the next
        transaction ran while the first prepare was still staging in the loop. Not: vLLM's
        real busy loop (LoopTest covers the patched loop)."""
        h, names, _ = make_host(self, "late")
        q = queue.SimpleQueue()
        h.core.input_queue = q

        def loop():
            while True:
                _, job = q.get()
                if job is None:
                    return
                job.run()
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        h.loop_thread = th
        self.addCleanup(th.join, 5)
        self.addCleanup(q.put, (H.TAG, None))
        wait = h.in_loop        # the group's wait for its job: 0.2 s instead of 60 s
        h.in_loop = lambda fn, gap=False, timeout=60.0: wait(fn, gap, timeout if gap else 0.2)
        ex = h.core.model_executor
        full, slow = ex.collective_rpc, [0.5]

        def rpc(name, timeout=None, args=()):
            if slow:                     # the first call: prepare
                time.sleep(slow.pop())
            return full(name, timeout, args)
        ex.collective_rpc = rpc
        r = h.handle({"op": "txn", "plan": plan("late")})
        prep = r["receipt"]["participants"]["workers"]["prepare"]
        self.assertEqual((r["outcome"], [f["check"] for f in prep["findings"]]),
                         ("aborted", ["deadline_missed"]))
        self.assertEqual([f["participant"] for f in h.co.poison], ["workers"])
        again = h.handle({"op": "txn", "plan": plan("late2")})
        self.assertEqual((again["outcome"], again["findings"][0]["check"]),
                         ("refused", "session_poisoned"))
        self.assertIn("unusable", h.workers.status("late")["error"])
        self.assertEqual([ran(n) for n in names], ["base", "base"])

    def test_a_rank_that_raises_systemexit_still_replies(self):
        """Input: rank 1's commit raises SystemExit (a repair cell or method calling
        sys.exit). Expected: every rank replies (rank 1 not-ok naming SystemExit), the
        transaction is rolled_back and both ranks run "base". Fails before the fix: dispatch
        caught only Exception, so SystemExit escaped the RPC (here: out of handle itself)."""
        h, names, eps = make_host(self, "exit")

        def bye(txn):
            raise SystemExit(3)
        eps[1].lp.commit = bye
        r = h.handle({"op": "txn", "plan": plan("ex")})
        commit = r["receipt"]["participants"]["workers"]["commit"]
        self.assertEqual([(x["rank"], x["ok"]) for x in commit["ranks"]], [(0, True), (1, False)])
        self.assertIn("SystemExit", commit["ranks"][1]["error"])
        self.assertEqual((r["outcome"], [ran(n) for n in names]),
                         ("rolled_back", ["base", "base"]))

    def test_a_missing_rank_or_dp_group_refuses_the_transaction(self):
        """Input: (a) the parallel config says TP 2 x DP 2 (this is DP rank 0); (b) DP 1, but
        the executor drops rank 1's reply. Expected: (a) parallel() expects 0, 1 and "dp1"
        and says DP>1 is not supported; the transaction aborts with a blocking, not
        overridable participant_missing naming dp1; (b) aborts naming rank 1. Neither rank
        runs the edit. Pain point: at DP>1 only DP group 0 replied, and witness, receipt and
        undo covered 1 of 4 groups with no word. Not: a real DP launch."""
        pc = NS(world_size=2, data_parallel_size=2, data_parallel_rank=0)
        d = H.parallel(NS(vllm_config=NS(parallel_config=pc)))
        self.assertEqual(d["expected"], [0, 1, "dp1"])
        self.assertIn("not supported", d["note"])
        h, names, _ = make_host(self, "dp")
        h.workers.expected = d["expected"]
        r = h.handle({"op": "txn", "plan": plan("dp", accept="all")})
        fs = [f for f in r["receipt"]["findings"] if f["check"] == "participant_missing"]
        self.assertEqual((r["outcome"], fs[0]["missing"], fs[0]["overridable"]),
                         ("aborted", ["dp1"], False))
        h2, names2, _ = make_host(self, "drop")
        h2.workers.expected = [0, 1]
        ex = h2.core.model_executor
        full = ex.collective_rpc
        ex.collective_rpc = lambda name, timeout=None, args=(): full(name, timeout, args)[:1]
        r = h2.handle({"op": "txn", "plan": plan("drop", accept="all")})
        fs = [f for f in r["receipt"]["findings"] if f["check"] == "participant_missing"]
        self.assertEqual((r["outcome"], fs[0]["missing"]), ("aborted", ["1"]))
        self.assertEqual([ran(n) for n in names + names2], ["base"] * 4)

    def test_a_cell_for_one_role_commits_while_the_other_abstains(self):
        """Input: a plan with only a repair cell `"on": ["worker"]` that sets an attribute on
        each rank's worker. Expected: committed (the core abstains instead of vetoing with
        empty_change), every rank's worker has the attribute, and undo removes it. Not: the
        cell's own journal calls (kernel tests)."""
        h, _, eps = make_host(self, "cell")
        p = {"id": "cell", "modules": {},
             "repairs": [{"cell": "worker.fastrsi_fixed = 1", "declares": ["worker"],
                          "on": ["worker"]}]}
        r = h.handle({"op": "txn", "plan": p})
        self.assertEqual(r["outcome"], "committed", r["receipt"]["findings"])
        self.assertEqual([vars(ep.scope["worker"]) for ep in eps], [{"fastrsi_fixed": 1}] * 2)
        self.assertTrue(h.handle({"op": "undo", "txn": "cell"})["ok"])
        self.assertEqual([vars(ep.scope["worker"]) for ep in eps], [{}, {}])

    def test_status_without_a_txn_is_the_hosts_overall_status(self):
        """Input: the status op with no txn, before and after a committed and undone
        transaction. Expected: ok, state "idle", the last transactions, at_base True. Pain
        point: status without a txn id raised KeyError in the prototype."""
        h, _, _ = make_host(self, "st")
        r = h.handle({"op": "status"})
        self.assertEqual((r["ok"], r["state"], r["last"]), (True, "idle", []))
        self.assertEqual(h.handle({"op": "txn", "plan": plan("s1")})["outcome"], "committed")
        self.assertTrue(h.handle({"op": "undo", "txn": "s1"})["ok"])
        r = h.handle({"op": "status"})
        self.assertEqual((r["ok"], r["state"], [x["txn"] for x in r["last"]], r["at_base"]),
                         (True, "idle", ["s1"], True))


class LoopTest(unittest.TestCase):
    """The patched EngineCoreProc: where jobs run relative to batches in flight."""

    def test_a_gap_job_waits_for_the_batch_in_flight_and_nothing_new_is_scheduled(self):
        """Input: a patched stand-in EngineCoreProc with one batch in flight; a gap job (phase
        2) and a plain job arrive. Expected: the plain job runs at once; the gap job waits;
        the next step runs with `scheduler.has_requests()` False (pause keep: nothing new is
        scheduled while the batch drains), then the gap job runs right after that step, once
        `batch_queue` is empty; the shadow is gone afterwards. Pain point: a commit while a
        batch is in flight lets the batch finish on mixed code. Not: vLLM's real loop."""
        seen = []

        class Scheduler:
            def has_requests(self):
                return True

        class EngineCoreProc:
            def __init__(self):
                self.batch_queue, self.scheduler = ["b0"], Scheduler()

            def _handle_client_request(self, request_type, request):
                seen.append(("client", request_type))

            def _process_engine_step(self):
                seen.append(("step", self.scheduler.has_requests(), list(self.batch_queue)))
                self.batch_queue.clear()

            def run_busy_loop(self):
                pass

        H._patch_core(types.SimpleNamespace(EngineCoreProc=EngineCoreProc))
        core = EngineCoreProc()
        fake = NS(pending=[], held_steps=0, steps=[])
        with mock.patch.object(H, "_HOST", [fake]):
            gap = H._Job(lambda: seen.append("gap job"), gap=True)
            plain = H._Job(lambda: seen.append("plain job"), gap=False)
            core._handle_client_request(H.TAG, gap)
            core._handle_client_request(H.TAG, plain)
            self.assertEqual(seen, ["plain job"])
            core._process_engine_step()
            core._handle_client_request("ADD", None)
        self.assertEqual(seen, ["plain job", ("step", False, ["b0"]), "gap job",
                                ("client", "ADD")])
        self.assertEqual((fake.held_steps, core.scheduler.has_requests()), (1, True))


class OverlayTest(unittest.TestCase):

    def test_the_core_comes_from_the_overlay_and_is_patched(self):
        """Input: an installed `vllm/v1/engine/core.py` (SRC = "installed") and an overlay
        finder serving its own core.py (SRC = "overlay"), then the host's finder in front.
        Expected: importing vllm.v1.engine.core gives the overlay's module, patched. Pain point:
        the prototype's finder asked PathFinder directly, so the engine core ran the installed
        file while the plan's base was the overlay's: a false base_mismatch. Not: a real vLLM
        import."""
        import importlib.abc
        import importlib.util
        core = ("SRC = {!r}\n\n\nclass EngineCoreProc:\n"
                "    def _handle_client_request(self, t, r):\n        pass\n\n"
                "    def _process_engine_step(self):\n        pass\n\n"
                "    def run_busy_loop(self):\n        pass\n")
        inst, over = tempfile.mkdtemp(), tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, inst, True)
        self.addCleanup(shutil.rmtree, over, True)
        for d, src in ((inst, "installed"), (over, "overlay")):
            with open(os.path.join(d, "core.py"), "w") as f:
                f.write(core.format(src))

        class Overlay(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "vllm.v1.engine.core":
                    return importlib.util.spec_from_file_location(
                        name, os.path.join(over, "core.py"))
                return None
        pkgs = {}
        for n in ("vllm", "vllm.v1", "vllm.v1.engine"):
            m = types.ModuleType(n)
            m.__path__ = [inst] if n == "vllm.v1.engine" else []
            pkgs[n] = m
        meta = [H._After(), Overlay()] + list(sys.meta_path)
        with mock.patch.dict(sys.modules, pkgs), mock.patch.object(sys, "meta_path", meta):
            sys.modules.pop("vllm.v1.engine.core", None)
            mod = importlib.import_module("vllm.v1.engine.core")
            self.assertEqual(mod.SRC, "overlay")
            self.assertIn("_patch_core", mod.EngineCoreProc.run_busy_loop.__qualname__)


if __name__ == "__main__":
    unittest.main()
