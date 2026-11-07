"""Golden tests of the transaction host (`coordinator.Coordinator`, `StepGapCoordinator`) on
toy ranks (CPU, standard library only).

Run from the repo root:  python3 -m unittest discover -s repos/vllm/livelab/kernel -t repos/vllm/livelab -v
(or python3 -m pytest -q repos/vllm/livelab/kernel)
"""
import itertools
import os
import shutil
import tempfile
import threading
import time
import unittest

from . import coordinator as C
from . import findings as F
from . import locate as L
from . import receipt as R
from . import toy
from .test_detectors import temp_modules
from .test_locate import ENG, at, edit, fresh
from .test_participant import step_target
from .test_witness import wait_for

_n = itertools.count()


class HostTest(unittest.TestCase):

    def _three(self, tag, quiesce=None, core_path=None):
        out = []
        for pname in ("rank0", "rank1", "core"):
            name, eng, _ = fresh(tag + pname, core_path if pname == "core" else None)
            e = eng.Engine()
            out.append((C.LocalParticipant(pname, alias={"toy": name},
                                           quiesce=(quiesce or {}).get(pname),
                                           scope={"engine": e}), eng, e))
        return out

    def _group(self, ranks, fail_commit=None):
        """One participant made of several ranks (like vLLM's worker group: one collective
        call per method); `fail_commit`: the rank whose commit returns not-ok."""
        def call(method, payload):
            rs = []
            for i, (p, _, _) in enumerate(ranks):
                r = (F.not_ok(f"rank{i} commit failed") if method == "commit" and i == fail_commit
                     else getattr(p, method)(payload))
                rs.append(r)
            return {"ok": all(r["ok"] for r in rs), "ranks": rs,
                    "state": rs[0].get("state"), "error": "; ".join(
                        str(r.get("error")) for r in rs if not r["ok"]) or None}
        return C.CallParticipant("workers", call)

    def test_half_committed_group_is_undone(self):
        """Proves: when one rank of a group participant fails its commit, the ranks of that group that did commit are undone in the same phase 2, so every rank runs its base code. Not: real RPC (TP4 live run shows it on vLLM)."""
        parts = self._three("half")
        ranks = parts[:2]
        base = [(eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__))
                for _, eng, _ in ranks]
        spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        res = C.Coordinator([self._group(ranks, fail_commit=1), parts[2][0]]).run(
            {"modules": {"toy.engine": spec}})
        self.assertEqual(res["outcome"], "rolled_back")
        # one state for the group: the rank that never committed got the group's undo too
        self.assertEqual([p._state[res["txn"]] for p, _, _ in ranks], ["undone", "undone"])
        self.assertEqual([(eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__))
                          for _, eng, _ in ranks], base)

    def test_vote_no_changes_nobody(self):
        """Proves: one participant on another build votes no and every participant keeps its base code objects. Not: a vote over real RPC."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        for f in ("__init__.py", "ops.py", "engine.py"):
            shutil.copy(os.path.join(toy.HERE, f), tmp)
        with open(os.path.join(tmp, "engine.py"), "w") as fh:
            fh.write(edit(ENG, ("make_shift(10)", "make_shift(20)")))
        parts = self._three("vote", core_path=tmp)
        base = [eng.Engine.step.__code__ for _, eng, _ in parts]
        spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        path = os.path.join(tmp, "receipts.jsonl")
        res = C.Coordinator([p for p, _, _ in parts], R.ReceiptWriter(path)).run(
            {"modules": {"toy.engine": spec}})
        self.assertEqual(res["outcome"], "aborted")
        rec = res["receipt"]
        self.assertEqual({n: r["prepare"]["ok"] for n, r in rec["participants"].items()},
                         {"rank0": True, "rank1": True, "core": False})
        self.assertEqual([(f["participant"], f["check"]) for f in rec["findings"]],
                         [("core", "base_mismatch"), ("core", "outside_changes")])
        self.assertEqual([eng.Engine.step.__code__ for _, eng, _ in parts], base)
        self.assertEqual([e.step(3) for _, _, e in parts], [17, 17, 27])
        self.assertEqual((rec["rollback"]["status"], rec["actual_recipes"], rec["edit_to_run_s"]),
                         ("not_needed", [], None))
        self.assertEqual(R.read(path)[0]["outcome"], "aborted")

    def test_commit_failure_undoes_the_committed(self):
        """Proves: a participant lost at commit gets rank0 undone (and itself undone: a no-op here) and core aborted, and the receipt says so. Not: a half-switched participant (next test)."""
        def lost(fn):
            raise RuntimeError("rank1: worker process lost")
        parts = self._three("fail", quiesce={"rank1": lost})
        base = [(eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__))
                for _, eng, _ in parts]
        spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        res = C.Coordinator([p for p, _, _ in parts]).run({"modules": {"toy.engine": spec}})
        rec = res["receipt"]
        self.assertEqual(res["outcome"], "rolled_back")
        self.assertEqual({n: sorted(r) for n, r in rec["participants"].items()},
                         {"rank0": ["commit", "prepare", "undo", "validate"],
                          "rank1": ["abort", "commit", "prepare", "undo", "validate"],
                          "core": ["abort", "prepare", "validate"]})
        self.assertEqual(rec["participants"]["rank1"]["commit"]["error"],
                         "commit: RuntimeError: rank1: worker process lost")
        self.assertEqual((rec["rollback"]["status"], list(rec["rollback"]["participants"])),
                         ("undone", ["rank1", "rank0"]))
        self.assertIn(("rank1", "participant_failed"),
                      [(f["participant"], f["check"]) for f in rec["findings"]])
        self.assertEqual([(eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__))
                          for _, eng, _ in parts], base)
        self.assertEqual([e.step(3) for _, _, e in parts], [17, 17, 17])
        self.assertEqual(parts[0][0].journal.entries, [])

    # -- step-gap transactions: ranks that step together --------------------------------------

    def _cluster(self, tag, gap_mode, quiesce=None, slow=None):
        """Three ranks, each its own toy copy, stepped together by one Cluster loop (the engine
        core). gap_mode: participants switch when called (the loop owner calls them in its
        gap); otherwise each switches at its own engine's boundary (the old way). The
        coordinator reaches each rank through CallParticipant over a loopback call that takes
        1 ms, like an RPC: in one process without it, the GIL would make three back-to-back
        commits look atomic and hide the race. slow: {rank: {method: extra seconds}}.
        Returns (cluster, call participants, locals)."""
        engines, ps, lps = {}, [], []
        for pname in ("rank0", "rank1", "rank2"):
            name, eng, _ = fresh(tag + pname)
            e = engines[pname] = eng.Engine()
            q = (quiesce or {}).get(pname) or (None if gap_mode else e.at_boundary)
            lp = C.LocalParticipant(pname, alias={"toy": name}, quiesce=q)

            def rpc(method, payload, lp=lp, extra=(slow or {}).get(pname, {})):
                time.sleep(0.001 + extra.get(method, 0))
                return getattr(lp, method)(payload)
            lps.append(lp)
            ps.append(C.CallParticipant(pname, rpc))
        return eng.Cluster(engines), ps, lps

    def _start(self, cl):
        stop = threading.Event()
        th = threading.Thread(target=cl.run, args=(stop,), daemon=True)
        th.start()

        def halt():
            stop.set()
            th.join()
        self.addCleanup(halt)
        wait_for(lambda: len(cl.log) >= 3)
        return halt

    def _steps(self, cl, n=5):
        k = len(cl.log)
        wait_for(lambda: len(cl.log) >= k + n)

    V1 = {"toy.engine": {"source": edit(ENG, ('self.ran.append("base")', 'self.ran.append("v1")')),
                         "base_hash": L.text_hash(ENG),
                         "targets": [["Engine.step", at(ENG, "def step")]]}}

    def test_step_gap_commit_and_undo_never_mix_versions(self):
        """Proves: with steps running, a step-gap commit then undo switch all 3 ranks between the same two steps, never mid-step. Not: real RPC."""
        cl, ps, lps = self._cluster("gap", gap_mode=True)
        co = C.StepGapCoordinator(ps, cl.at_gap)
        halt = self._start(cl)
        res = co.run({"modules": self.V1})
        self._steps(cl)
        u = co.undo(res["txn"])
        self._steps(cl)
        halt()
        self.assertEqual((res["outcome"], u["ok"]), ("committed", True))
        self.assertEqual([t for t in cl.log if len(set(t)) > 1], [])
        self.assertEqual([k for k, _ in itertools.groupby(cl.log)],
                         [("base",) * 3, ("v1",) * 3, ("base",) * 3])
        self.assertLess(res["receipt"]["gap_s"], 0.1)   # how long the gap stalled the loop

    def test_step_gap_failed_commit_is_undone_in_the_same_gap(self):
        """Proves: when rank1 fails at commit, rank0's commit is undone inside the same gap, so no step ever ran v1. Not: a rank that hangs."""
        def lost(fn):
            raise RuntimeError("rank1: worker process lost")
        cl, ps, lps = self._cluster("gapfail", gap_mode=True, quiesce={"rank1": lost})
        co = C.StepGapCoordinator(ps, cl.at_gap)
        halt = self._start(cl)
        res = co.run({"modules": self.V1})
        self._steps(cl)
        halt()
        self.assertEqual((res["outcome"], res["receipt"]["rollback"]["status"]),
                         ("rolled_back", "undone"))
        self.assertEqual(set(cl.log), {("base",) * 3})
        self.assertEqual(lps[0].journal.entries, [])


class ParticipantMissingTest(unittest.TestCase):
    def test_a_dp_client_that_drops_replies_blocks_the_transaction(self):
        """Input: a worker group of 2 data-parallel groups x 2 ranks (each rank its own copy of
        a module), reached through a fake DP client that delivers every call to every rank
        but returns only DP group 0's replies (as vLLM's DP client did at EP4DP4); the group
        declares the 4 ranks it expects. Expected: the transaction aborts with a blocking,
        not-overridable participant_missing naming dp1/0 and dp1/1, even with accept "all";
        no rank runs the edit afterwards. Fails before this change: the transaction committed
        on all 4 ranks while its witness, receipt and undo covered 2, with no finding."""
        mods = []
        for i in range(4):
            name = f"kt_dp{next(_n)}"
            (m,) = temp_modules(self, {name: "def f():\n    return 'base'\n"})
            mods.append(m)
        ranks = [(f"dp{i // 2}/{i % 2}", C.LocalParticipant(
            "workers", alias={"pkg": mods[i].__name__})) for i in range(4)]

        def dp_client(method, payload):
            rs = [{**getattr(lp, method)(payload), "rank": rid} for rid, lp in ranks]
            return C.merge_ranks(method, rs[:2])          # DP group 0's replies only
        g = C.CallParticipant("workers", dp_client)
        g.expected = [rid for rid, _ in ranks]
        co = C.Coordinator([g], phase2_deadline_s=None)
        r = co.run({"id": "dp", "accept": "all", "modules": {"pkg": {
            "source": "def f():\n    return 'edit'\n"}}})
        self.assertEqual(r["outcome"], "aborted")
        fs = [f for f in r["receipt"]["findings"] if f["check"] == "participant_missing"]
        self.assertEqual(sorted({f["observed"].split(":")[0] for f in fs}), ["abort", "prepare"])
        self.assertEqual({(tuple(f["missing"]), f["overridable"]) for f in fs},
                         {(("dp1/0", "dp1/1"), False)})
        self.assertEqual([m.f() for m in mods], ["base"] * 4)


if __name__ == "__main__":
    unittest.main()
