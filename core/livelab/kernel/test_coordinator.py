"""Golden tests of the transaction host (`coordinator.Coordinator`, `StepGapCoordinator`) on
toy ranks (CPU, standard library only).

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
(or python3 -m pytest -q core/livelab/kernel)
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
        members = []
        for index, (participant, _, _) in enumerate(ranks):
            if index == fail_commit:
                def call(method, payload, inner=participant, rank_index=index):
                    if method == "commit":
                        return F.not_ok(f"rank{rank_index} commit failed")
                    return getattr(inner, method)(payload)
                members.append(C.CallParticipant(participant.name, call))
            else:
                members.append(participant)
        return C.GroupParticipant("workers", members)

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

    def test_late_commit_poisons_and_steps_may_mix(self):
        """Input: 3 ranks stepping together, phase2_deadline_s=0.2 (commit by 0.1 s); rank1's
        commit starts in time and applies v1, then takes 0.5 s more to answer. Expected:
        rolled_back, rank1 in missed_deadline, the host poisoned and the next run refused with
        session_poisoned; after the gap, steps run with rank1 on v1 and the others on base
        until rank1's queued undo lands, then all base again. Proves what the README states:
        no mixed step only within the deadline; a miss poisons the host, and steps may mix
        until the lab respawns. Not: a host that keeps the loop stopped until the undo is
        confirmed (deferred)."""
        def slow_answer(fn):
            r = fn()
            time.sleep(0.5)
            return r
        cl, ps, lps = self._cluster("gaplate", gap_mode=True, quiesce={"rank1": slow_answer})
        co = C.StepGapCoordinator(ps, cl.at_gap, phase2_deadline_s=0.2)
        halt = self._start(cl)
        res = co.run({"modules": self.V1})
        wait_for(lambda: lps[1]._state.get(res["txn"]) == "undone", timeout=5)
        self._steps(cl)
        refused = co.run({"modules": self.V1})
        halt()
        self.assertEqual((res["outcome"], res["receipt"]["rollback"]["missed_deadline"],
                          res["receipt"]["rollback"]["poisoned"]),
                         ("rolled_back", ["rank1"], True))
        self.assertEqual((co.status()["state"], refused["outcome"],
                          [f["check"] for f in refused["findings"]]),
                         ("poisoned", "refused", ["session_poisoned"]))
        self.assertIn(("base", "v1", "base"), cl.log)        # mixed steps after the miss
        self.assertEqual(cl.log[-1], ("base",) * 3)          # until the queued undo landed

    def test_receipt_written_only_after_append_succeeds(self):
        """Input: a writer whose first append blocks, then fails; later appends succeed.
        Expected: status(txn) during the append says written=False (and does not raise); run
        returns the failed append; the next status(txn) retries and the file holds the
        receipt. Fails before the fix: status said written=True during and after the failed
        append, and the receipt was never written."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "receipts.jsonl")
        entered, release = threading.Event(), threading.Event()

        class Flaky(R.ReceiptWriter):
            calls = 0

            def append(self, rec):
                Flaky.calls += 1
                if Flaky.calls == 1:
                    entered.set()
                    release.wait(5)
                    return {"ok": False, "error": "disk full"}
                return super().append(rec)
        parts = self._three("flaky")
        co = C.Coordinator([p for p, _, _ in parts], Flaky(path))
        spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        box = {}
        th = threading.Thread(target=lambda: box.update(r=co.run(
            {"id": "flaky", "modules": {"toy.engine": spec}})))
        th.start()
        self.assertTrue(entered.wait(5))
        during = co.status("flaky")
        release.set()
        th.join()
        self.assertEqual((during["written"], box["r"]["outcome"], box["r"]["written"]),
                         (False, "committed", {"ok": False, "error": "disk full"}))
        self.assertEqual(co.status("flaky")["written"], True)
        self.assertEqual([r["id"] for r in R.read(path)], ["flaky"])

    def test_validate_deadline_miss_refreshes_for_healthy_participants_and_timed_out_undo_cancels(self):
        """Input: (a) rank0's validate exceeds commit_at (0.05 s) while rank1 and rank2 are
        healthy; (b) a committed transaction whose undo gap call times out before starting.
        Expected: (a) only rank0 is in missed_deadline and poison (rank1 and rank2 get fresh
        per-call validate deadlines instead of an already-expired commit_at); (b) the timed-out
        undo does not run when the gap drains later."""
        cluster, participants, local_participants = self._cluster(
            "valmiss", gap_mode=True, slow={"rank0": {"validate": 0.15}}
        )
        coordinator = C.StepGapCoordinator(participants, cluster.at_gap, phase2_deadline_s=0.1)
        halt = self._start(cluster)
        res = coordinator.run({"modules": self.V1})
        halt()
        self.assertEqual(res["outcome"], "aborted")
        self.assertEqual(res["receipt"]["rollback"]["missed_deadline"], ["rank0"])
        self.assertEqual([finding["participant"] for finding in coordinator.poison], ["rank0"])

        parts = self._three("undocancel")
        gate_Release = threading.Event()
        def timed_out_gap(_fn):
            raise TimeoutError("gap did not start undo in time")
        host_co = C.StepGapCoordinator([p for p, _, _ in parts], lambda fn: fn())
        spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        committed = host_co.run({"id": "u1", "modules": {"toy.engine": spec}})
        self.assertEqual(committed["outcome"], "committed")
        queued_jobs = []
        def capture_then_timeout(fn):
            queued_jobs.append(fn)
            raise TimeoutError("gap busy")
        host_co.gap = capture_then_timeout
        undo_reply = host_co.undo("u1")
        self.assertFalse(undo_reply["ok"])
        self.assertEqual(queued_jobs[0](), None)
        self.assertEqual([e.step(3) for _, _, e in parts], [1700, 1700, 1700])
        _ = gate_Release


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

    def test_a_rank_missing_from_proof_is_in_the_receipt(self):
        """Input: the same 4-rank group, but the client drops DP group 1's replies only for
        the post-commit `proof`. Expected: committed (every rank switched), and the receipt
        carries the proof's participant_missing naming dp1/0 and dp1/1, with
        commit_hashes.agree False. Fails before the fix: the proof's findings were dropped
        and the receipt said the ranks agree, from the 2 ranks that answered."""
        mods = []
        for i in range(4):
            (m,) = temp_modules(self, {f"kt_pm{next(_n)}": "def f():\n    return 'base'\n"})
            mods.append(m)
        ranks = [(f"dp{i // 2}/{i % 2}", C.LocalParticipant(
            "workers", alias={"pkg": mods[i].__name__})) for i in range(4)]

        def dp_client(method, payload):
            rs = [{**getattr(lp, method)(payload), "rank": rid} for rid, lp in ranks]
            return C.merge_ranks(method, rs[:2] if method == "proof" else rs)
        g = C.CallParticipant("workers", dp_client)
        g.expected = [rid for rid, _ in ranks]
        r = C.Coordinator([g], phase2_deadline_s=None).run({"id": "pm", "modules": {"pkg": {
            "source": "def f():\n    return 'edit'\n"}}})
        rec = r["receipt"]
        self.assertEqual((r["outcome"], [m.f() for m in mods]), ("committed", ["edit"] * 4))
        self.assertEqual([(f["check"], f["missing"]) for f in rec["findings"]
                          if f["check"] == "participant_missing"],
                         [("participant_missing", ["dp1/0", "dp1/1"])])
        self.assertEqual((rec["commit_hashes"]["ranks_agree"], rec["commit_hashes"]["agree"]),
                         ({"workers": False}, False))
        empty_group = C.CallParticipant("workers", lambda _m, _p: C.merge_ranks("prepare", []))
        empty_group.expected = [0, 1]
        empty_reply = C.expect_ranks(empty_group, "prepare", C.merge_ranks("prepare", []))
        self.assertIn("replies from none", empty_reply["findings"][0]["observed"])


if __name__ == "__main__":
    unittest.main()
