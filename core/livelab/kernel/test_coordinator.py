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
from . import session as SS
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
            slow_for_rank = (slow or {}).get(pname) or {}

            def rpc(method, payload, lp=lp, extra=slow_for_rank):
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
        Expected: (a) only rank0 is in missed_deadline and poison (rank1 and rank2, whose turn
        came after the deadline, are not asked to validate and are aborted without a wait: not
        blamed, and the gap stays bounded by the deadline, audit #19); (b) the timed-out
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


class UndoCascadeTest(unittest.TestCase):
    """Undo of an older transaction under a newer one: the journal undoes both."""

    def _two(self, tag, resets, *kinds_per_txn, coordinator_cls=C.Coordinator):
        """Commit t1, t2, ... (one per entry of kinds_per_txn, its undo_resets), each edit on
        top of the previous: Engine.step returns y * 10, then y * 100, ..."""
        name, eng, _ = fresh(tag)
        part = C.LocalParticipant("rank0", alias={"toy": name}, resets=resets)
        coordinator = coordinator_cls([part], phase2_deadline_s=None)
        src, factor = ENG, ""
        for index, kinds in enumerate(kinds_per_txn, 1):
            new_factor = " * 1" + "0" * index
            src = edit(src, (f"return y{factor}", f"return y{new_factor}"))
            factor = new_factor
            res = coordinator.run({"id": f"t{index}", "undo_resets": kinds, "modules": {
                "toy.engine": {"source": src, "targets": [["Engine.step", at(ENG, "def step")]]}}})
            self.assertEqual(res["outcome"], "committed", res)
        return coordinator, part, eng

    def test_a_cascade_in_doubt_leaves_no_later_transaction_committed(self):
        """Input: t1, t2, t3 committed; undo t1 in a gap that does the work and then raises
        (as a gap timeout after the undo ran). Expected: every participant undid all three
        (base code, step(3) = 17); the host gives t1, t2 and t3 the same not-ok state
        ("partial": the work finished but no reply came back), never committed. Pain point: later transactions took their state only from the
        participants' replies, which an undo in doubt never returns, so t2 and t3 stayed
        "committed" for code that was gone. Not: which undo reply was lost."""
        class GapDiesAfterWork(C.Coordinator):
            die = False

            def _in_gap(self, fn):
                out = fn()
                if self.die:
                    raise RuntimeError("gap timed out")
                return out
        coordinator, part, eng = self._two("casc_doubt", {}, [], [], [],
                                           coordinator_cls=GapDiesAfterWork)
        coordinator.die = True
        undo_res = coordinator.undo("t1")
        self.assertEqual((undo_res["status"], eng.Engine().step(3)), ("partial", 17))
        self.assertEqual([part.status(txn)["state"] for txn in ("t1", "t2", "t3")],
                         ["undone"] * 3)
        self.assertEqual([coordinator.status(txn)["state"] for txn in ("t1", "t2", "t3")],
                         ["partial"] * 3)

    def test_a_later_transaction_left_in_doubt_still_gets_its_resets(self):
        """Input: t1 (undo_resets ["state"]) and t2 (["cache"]); an undo of t2 that dies
        after it started and before any participant ran (t2 in-doubt on the host, still
        installed); then undo t1. Expected: undo(t2) latches recovery_required fail-closed,
        so subsequent undo(t1) is refused until recovered() clears the fence (and held
        committed later transactions still get their cascade resets in
        `test_cascade_runs_the_later_transactions_resets_too`)."""
        ran = []

        class FirstUndoDies(C.Coordinator):
            dies = 1

            def _undo_phases(self, txn, resets, tl):
                if self.dies:
                    self.dies -= 1
                    raise RuntimeError("gap died")
                return super()._undo_phases(txn, resets, tl)
        coordinator, part, eng = self._two("casc_partial", {
            "cache": lambda: ran.append("cache"), "state": lambda: ran.append("state")},
            ["state"], ["cache"], coordinator_cls=FirstUndoDies)
        u2 = coordinator.undo("t2")
        self.assertEqual(u2["status"], "in-doubt")
        self.assertTrue(u2["recovery_required"])
        self.assertTrue(coordinator.is_recovery_required())
        self.assertEqual((coordinator.status("t2")["state"], eng.Engine().step(3)),
                         ("in-doubt", 1700))
        undo_res = coordinator.undo("t1")
        self.assertEqual((undo_res["ok"], undo_res["status"], undo_res["recovery_required"], ran),
                         (False, "refused", True, []))
        rec = coordinator.recovered()
        self.assertEqual((rec["ok"], rec["incarnation"], coordinator.is_recovery_required()),
                         (True, 2, False))

    def test_cascade_runs_the_later_transactions_resets_too(self):
        """Input: t1 (undo_resets ["state"]) then t2 on top (undo_resets ["cache"]); undo t1.
        Expected: base code (step(3) = 17); both resets ran, t2's first (newest first); the
        undo receipt lists both; t2 is "undone" on the host. Pain point: only t1's resets
        ran, t2 was marked undone, so its cache reset could never run (a second undo of t2
        is refused). Not: resets that depend on each other's order beyond newest first."""
        ran = []
        coordinator, part, eng = self._two("casc_resets", {
            "cache": lambda: ran.append("cache"), "state": lambda: ran.append("state")},
            ["state"], ["cache"])
        undo_res = coordinator.undo("t1")
        self.assertEqual((undo_res["ok"], eng.Engine().step(3), ran), (True, 17, ["cache", "state"]))
        self.assertEqual(undo_res["receipt"]["resets"], ["cache", "state"])
        self.assertEqual(coordinator.status("t2")["state"], "undone")

    def test_a_failed_cascade_never_leaves_the_later_transaction_committed(self):
        """Input: t1 (undo_resets ["bad"], a reset that raises) then t2 on top; undo t1.
        Expected: undo not ok, the code is at base on the participant (step(3) = 17) and t2
        is "undone" there, and the host says t2 is "partial", the same as t1. Pain point:
        the host kept t2 "committed" for code that was gone. Not: an undo that never started
        (cancelled), where both keep their state."""
        def bad():
            raise RuntimeError("reset failed")
        coordinator, part, eng = self._two("casc_fail", {"bad": bad}, ["bad"], [])
        undo_res = coordinator.undo("t1")
        self.assertEqual((undo_res["ok"], eng.Engine().step(3), part.status("t2")["state"]),
                         (False, 17, "undone"))
        self.assertEqual((coordinator.status("t1")["state"], coordinator.status("t2")["state"]),
                         ("partial", "partial"))


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


class ParallelFanoutTest(unittest.TestCase):

    def _delayed_participants(self, tag, count=8, delay_s=0.05, delay_methods=(), fail_method=None):
        """Build `count` CallParticipants backed by distinct toy modules with optional per-method delay."""
        parts, engines, tracker = [], [], {"lock": threading.Lock(), "active": {}, "max_active": {}, "order": {}}
        for idx in range(count):
            pname = f"r{idx}"
            mod_name, eng, _ = fresh(f"{tag}_{next(_n)}_{pname}")
            lp = C.LocalParticipant(pname, alias={"toy": mod_name})
            engines.append(eng.Engine())

            def rpc(method, payload, lp=lp, idx=idx, pname=pname):
                with tracker["lock"]:
                    cur = tracker["active"].get(method, 0) + 1
                    tracker["active"][method] = cur
                    tracker["max_active"][method] = max(tracker["max_active"].get(method, 0), cur)
                    tracker["order"].setdefault(method, []).append(pname)
                try:
                    if method in delay_methods:
                        time.sleep(delay_s)
                    if fail_method is not None and (idx, method) == fail_method:
                        return F.not_ok(f"{pname} failed {method}")
                    return getattr(lp, method)(payload)
                finally:
                    with tracker["lock"]:
                        tracker["active"][method] -= 1

            parts.append(C.CallParticipant(pname, rpc))
        return parts, engines, tracker

    def test_parallel_fanout_speedup_order_and_serial_commit_undo(self):
        """8 x 100 ms non-local participants fan out concurrently (< 500 ms per stage) across
        prepare, validate, proof, witness_check, at_base, and Session._fingerprint while
        preserving exact participant order and keeping commit + undo strictly serial."""
        fanout_methods = ("prepare", "validate", "proof", "witness_check", "at_base", "fingerprint")
        parts, engines, tracker = self._delayed_participants(
            "fanout_ok", count=8, delay_s=0.10, delay_methods=fanout_methods,
        )
        names = [p.name for p in parts]
        co = C.Coordinator(parts, phase2_deadline_s=None)

        t0 = time.monotonic()
        sess = SS.Session(co, ["toy.engine"])
        fp_s = time.monotonic() - t0
        self.assertLess(fp_s, 0.50, f"Session._fingerprint took {fp_s:.3f}s")
        self.assertEqual(list(sess.expected[0].keys()), names)

        plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "base_hash": L.text_hash(ENG),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}}
        t0 = time.monotonic()
        res = co.run(plan)
        run_s = time.monotonic() - t0
        self.assertEqual(res["outcome"], "committed", res)
        self.assertLess(res["receipt"]["prepare_s"], 0.50, res["receipt"]["prepare_s"])
        self.assertLess(res["receipt"]["gap_s"], 0.50, res["receipt"]["gap_s"])
        # prepare + validate + proof = 3 fanned stages x 100ms (~300ms), well under serial 2.4s
        self.assertLess(run_s, 1.30, f"run() took {run_s:.3f}s")
        self.assertEqual(list(res["receipt"]["participants"].keys()), names)
        self.assertEqual(list(res["receipt"]["commit_hashes"]["participants"].keys()), names)
        self.assertEqual([e.step(3) for e in engines], [170] * 8)

        t0 = time.monotonic()
        proof = co.prove(res["txn"])
        prove_s = time.monotonic() - t0
        self.assertTrue(proof["ok"], proof)
        self.assertLess(prove_s, 0.50, f"prove() took {prove_s:.3f}s")
        self.assertEqual(list(proof["receipt"]["witness"].keys()), names)

        t0 = time.monotonic()
        ab = co.at_base()
        ab_s = time.monotonic() - t0
        self.assertFalse(ab["ok"])
        self.assertLess(ab_s, 0.50, f"at_base() took {ab_s:.3f}s")
        self.assertEqual(list(ab["participants"].keys()), names)

        undo_res = co.undo(res["txn"])
        self.assertTrue(undo_res["ok"], undo_res)
        self.assertEqual([e.step(3) for e in engines], [17] * 8)

        # Parallel stages observed real concurrency; commit & undo stayed strictly serial in order.
        for method in fanout_methods:
            self.assertGreater(tracker["max_active"][method], 1, f"{method} did not fan out")
        self.assertEqual(tracker["max_active"]["commit"], 1)
        self.assertEqual(tracker["max_active"]["undo"], 1)
        self.assertEqual(tracker["order"]["commit"], names)
        self.assertEqual(tracker["order"]["undo"], list(reversed(names)))

    def test_parallel_fanout_abort_speedup_and_fault_parity(self):
        """Phase-1 and phase-2 aborts fan out concurrently (< 500 ms for 8 x 100 ms) with exact
        participant ordering and untouched base code."""
        for fail_stage in ("prepare", "validate"):
            parts, engines, tracker = self._delayed_participants(
                f"fanout_abort_{fail_stage}", count=8, delay_s=0.10,
                delay_methods=("abort",), fail_method=(5, fail_stage),
            )
            co = C.Coordinator(parts, phase2_deadline_s=None)
            plan = {"modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "base_hash": L.text_hash(ENG),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }}}
            t0 = time.monotonic()
            res = co.run(plan)
            elapsed = time.monotonic() - t0
            self.assertEqual(res["outcome"], "aborted", (fail_stage, res))
            self.assertLess(elapsed, 0.50, f"{fail_stage} abort took {elapsed:.3f}s")
            self.assertEqual(list(res["receipt"]["participants"].keys()), [p.name for p in parts])
            self.assertGreater(tracker["max_active"]["abort"], 1)
            self.assertEqual([e.step(3) for e in engines], [17] * 8)

    def test_parallel_validate_deadline_missed_is_thread_safe(self):
        """Concurrent deadline misses across 8 participants record every participant once in
        `missed` and `poison` without races."""
        parts, _, _ = self._delayed_participants(
            "fanout_dl", count=8, delay_s=0.25, delay_methods=("validate",),
        )
        co = C.Coordinator(parts, phase2_deadline_s=0.20)
        plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "base_hash": L.text_hash(ENG),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}}
        res = co.run(plan)
        self.assertEqual(res["outcome"], "aborted", res)
        self.assertLess(res["receipt"]["gap_s"], 0.25, res["receipt"]["gap_s"])
        expected_names = [f"r{i}" for i in range(8)]
        self.assertEqual(res["receipt"]["rollback"]["missed_deadline"], expected_names)
        self.assertEqual(co.status(res["txn"])["missed"], expected_names)
        poison_names = [f["participant"] for f in co.poison]
        self.assertEqual(sorted(poison_names), expected_names)
        self.assertEqual(len(poison_names), 8)

    def test_all_local_participants_stay_on_caller_thread(self):
        """All-LocalParticipant lists bypass ThreadPoolExecutor and run on the caller thread."""
        caller = threading.get_ident()
        seen_threads = []
        locals_ = []
        for idx in range(4):
            mod_name, _, _ = fresh(f"local_sync_{next(_n)}_{idx}")
            locals_.append(C.LocalParticipant(f"l{idx}", alias={"toy": mod_name}))
        out = C._fanout(locals_, lambda p: (seen_threads.append(threading.get_ident()), p.name)[1])
        self.assertEqual(out, ["l0", "l1", "l2", "l3"])
        self.assertEqual(set(seen_threads), {caller})

    def test_loop_bound_custom_participant_stays_on_caller_thread(self):
        """A custom Participant subclass (like vLLM WorkerGroup) stays on the caller thread
        both when paired with one CallParticipant and when mixed with multiple CallParticipants."""
        caller = threading.get_ident()
        observed = {}

        class LoopBound(C.Participant):
            name = "loop_bound"

            def ping(self):
                observed[self.name] = threading.get_ident()
                return "loop"

        lb = LoopBound()
        c0 = C.CallParticipant("c0", lambda m, p: F.ok())
        c1 = C.CallParticipant("c1", lambda m, p: F.ok())
        self.assertEqual(C._fanout([lb, c0], lambda p: lb.ping() if p is lb else p.name),
                         ["loop", "c0"])
        self.assertEqual(observed["loop_bound"], caller)

        observed.clear()

        def probe(p):
            if p is lb:
                return lb.ping()
            time.sleep(0.03)
            observed[p.name] = threading.get_ident()
            return p.name

        self.assertEqual(C._fanout([lb, c0, c1], probe), ["loop", "c0", "c1"])
        self.assertEqual(observed["loop_bound"], caller)
        self.assertNotEqual(observed["c0"], caller)
        self.assertNotEqual(observed["c1"], caller)

    def test_parallel_fanout_false_runs_all_stages_serially(self):
        """Setting `co.parallel_fanout = False` serializes all Coordinator and Session fanout
        stages (`fingerprint`, `prepare`, `validate`, `proof`, `witness_check`, `at_base`, and
        `abort`) without spawning worker threads."""
        fanout_methods = ("prepare", "validate", "proof", "witness_check", "at_base", "fingerprint")
        parts, engines, tracker = self._delayed_participants(
            "serial_ok", count=4, delay_s=0.01, delay_methods=fanout_methods,
        )
        names = [p.name for p in parts]
        co = C.Coordinator(parts, phase2_deadline_s=None)
        co.parallel_fanout = False

        sess = SS.Session(co, ["toy.engine"])
        self.assertEqual(list(sess.expected[0].keys()), names)
        plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "base_hash": L.text_hash(ENG),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}}
        res = co.run(plan)
        self.assertEqual(res["outcome"], "committed", res)
        self.assertEqual([e.step(3) for e in engines], [170] * 4)
        self.assertTrue(co.prove(res["txn"])["ok"])
        self.assertFalse(co.at_base()["ok"])
        self.assertTrue(co.undo(res["txn"])["ok"])
        for method in fanout_methods:
            self.assertEqual(tracker["max_active"][method], 1, f"{method} fanned out when disabled")
            self.assertEqual(tracker["order"][method], names)

        for fail_stage in ("prepare", "validate"):
            ab_parts, _, ab_tracker = self._delayed_participants(
                f"serial_abort_{fail_stage}", count=4, delay_s=0.01,
                delay_methods=("abort",), fail_method=(2, fail_stage),
            )
            ab_co = C.Coordinator(ab_parts, phase2_deadline_s=None)
            ab_co.parallel_fanout = False
            ab_res = ab_co.run(plan)
            self.assertEqual(ab_res["outcome"], "aborted", ab_res)
            self.assertEqual(ab_tracker["max_active"]["abort"], 1)
            self.assertEqual(ab_tracker["order"]["abort"], [p.name for p in ab_parts])

    def test_concurrent_status_polling_never_steals_finisher_lock(self):
        """Input: multiple threads tightly polling `co.status(txn)` (including right at the
        window between `_done()` and `_finish()`) while `co.run()` and `Session.install()`
        execute across repeated transactions. Expected: `co.status(txn)` never steals
        `t['fin']` from `co.run()`, `co.run()` always returns the finalized `receipt`, and
        `Session.install()` always ends `'edited'` with zero poison."""
        from . import session as SS
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        name, _, _ = fresh("fin_stress")
        lp = C.LocalParticipant("rank0", alias={"toy": name})

        class YieldingDoneCoordinator(C.Coordinator):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.done_seen = threading.Event()
                self.slow_finish = True

            def _done(self, txn, out):
                super()._done(txn, out)
                self.done_seen.set()
                if self.slow_finish:
                    # Wide window after `t['done']=True` before `run()` enters `_finish()`
                    time.sleep(0.02)

        path = os.path.join(tmp, "r.jsonl")
        co = YieldingDoneCoordinator([lp], R.ReceiptWriter(path), phase2_deadline_s=None)
        sess = SS.Session(co, ["toy.engine"])
        base_eng = ENG
        cur_src = base_eng

        for i in range(1, 11):
            txn_id = f"race-{i}"
            nxt_src = edit(cur_src,
                           ("return y" if i == 1 else f"return y + {i - 1}",
                            f"return y + {i}"))
            plan = {"id": txn_id, "modules": {"toy.engine": {
                "source": nxt_src,
                "targets": [["Engine.step", at(nxt_src, "def step")]]}}}
            stop = threading.Event()
            seen_states = []

            def poller(tid=txn_id, ev=stop, states=seen_states):
                while not ev.is_set():
                    st = co.status(tid)
                    if st.get("ok"):
                        states.append(st.get("state"))
                    time.sleep(0.001)

            co.done_seen.clear()
            threads = [threading.Thread(target=poller, daemon=True) for _ in range(4)]
            for th in threads:
                th.start()
            res = sess.install(plan)
            stop.set()
            for th in threads:
                th.join(timeout=2.0)

            self.assertEqual(res["outcome"], "committed", res)
            self.assertEqual(res["state"], "edited", sess.poisoned())
            self.assertIsNotNone(res.get("receipt"), f"iter {i} lost receipt to status()!")
            self.assertTrue(res["written"]["ok"], res["written"])
            self.assertEqual(co.poison, [])
            cur_src = nxt_src

    def test_partial_rollback_poisons_participant_and_coordinator(self):
        """Input: a commit whose rollback cannot undo every change (`undo_incomplete` /
        `repair_not_undoable`). Expected: `LocalParticipant` marks `at_base(poisoned=True)` and
        refuses subsequent `prepare()` with `session_poisoned`; `Coordinator` appends
        `undo_incomplete` to `co.poison` and refuses subsequent `co.run()` with
        `session_poisoned`."""
        box_cls = type("Box", (), {"val": 1})
        box = box_cls()
        name, _, _ = fresh("part_poison")
        lp = C.LocalParticipant("core", alias={"toy": name}, scope={"box": box})
        bad_plan = {
            "id": "bad-rb",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 100")),
                "targets": [["Engine.step", at(ENG, "def step")]]}},
            "repairs": [{"cell": "box.val = 2\nraise RuntimeError('boom')\n", "declares": []}],
        }
        next_plan = {
            "id": "after-rb",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]]}},
        }
        co = C.Coordinator([lp], phase2_deadline_s=None)
        res = co.run(bad_plan)
        self.assertEqual(res["outcome"], "rolled_back")
        self.assertEqual(res["receipt"]["rollback"]["status"], "partial")
        self.assertTrue(lp.at_base()["poisoned"])
        self.assertTrue(any(f["check"] == "undo_incomplete" for f in co.poison), co.poison)

        prep_after = lp.prepare(next_plan)
        self.assertFalse(prep_after["ok"])
        self.assertEqual([f["check"] for f in prep_after["findings"]], ["session_poisoned"])

        run_after = co.run(next_plan)
        self.assertEqual(run_after["outcome"], "refused")
        self.assertEqual([f["check"] for f in run_after["findings"]], ["session_poisoned"])


class CoordinatorHardeningRegressionTest(unittest.TestCase):
    """Regression tests for C1-C9 coordinator hardening fixes."""

    def test_c1_poll_rollback_binds_txn_id_per_call_across_timeouts(self):
        """C1: `poll_rollback` binds `txn_id` and `incarnation` eagerly in the Lane lambda
        so a timed-out `status` call on txn-1 never compares against txn-2."""
        observed = []
        orig_check_reply = C._check_reply
        self.addCleanup(setattr, C, "_check_reply", orig_check_reply)

        def spy_check_reply(p, method, r, expected_txn=None, expected_incarnation=None):
            res = orig_check_reply(
                p, method, r, expected_txn=expected_txn, expected_incarnation=expected_incarnation
            )
            observed.append({
                "participant": p.name,
                "reply_txn": r.get("txn"),
                "expected_txn": expected_txn,
                "stale": res.get("stale"),
            })
            return res

        C._check_reply = spy_check_reply
        status_gate = threading.Event()
        p1_called = threading.Event()

        def p1_rpc(method, payload):
            if method == "status":
                p1_called.set()
                status_gate.wait(2.0)
                return {"ok": True, "txn": payload["txn"],
                        "incarnation": payload["incarnation"], "state": "undone"}
            return F.ok()

        def p2_rpc(method, payload):
            if method == "status":
                return {"ok": True, "txn": payload["txn"],
                        "incarnation": payload["incarnation"], "state": "undone"}
            return F.ok()

        p1 = C.CallParticipant("p1", p1_rpc)
        p2 = C.CallParticipant("p2", p2_rpc)
        co = C.Coordinator([p1, p2], phase2_deadline_s=0.1)
        co.poison.append(F.finding("deadline_missed", "late", "on time"))
        co.txns["txn-1"] = {"state": "aborted", "pending_rollback": [p1]}
        co.txns["txn-2"] = {"state": "aborted", "pending_rollback": [p2]}

        res = co.poll_rollback(deadline_s=1e-6)
        self.assertTrue(res)
        status_gate.set()
        self.assertTrue(p1_called.wait(2.0))
        wait_for(lambda: any(entry["participant"] == "p1" for entry in observed), timeout=2.0)
        p1_entries = [entry for entry in observed if entry["participant"] == "p1"]
        self.assertEqual(len(p1_entries), 1)
        self.assertEqual(p1_entries[0]["reply_txn"], "txn-1")
        self.assertEqual(p1_entries[0]["expected_txn"], "txn-1")
        self.assertIsNone(p1_entries[0]["stale"])

    def test_c2_undo_refuses_prepared_and_indoubt_and_done_preserves_superseded(self):
        """C2: `undo()` refuses `'prepared'` and `'in-doubt'` transactions without a receipt,
        and `_done()` never overwrites `'superseded'` or a stale incarnation."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "r1.jsonl")
        name, eng, _ = fresh("c2_prep")
        lp = C.LocalParticipant("r0", alias={"toy": name})
        gap_entered = threading.Event()
        gap_release = threading.Event()

        def blocking_gap(fn):
            gap_entered.set()
            gap_release.wait(5)
            return fn()

        co = C.StepGapCoordinator([lp], gap=blocking_gap, writer=R.ReceiptWriter(path),
                                  phase2_deadline_s=None)
        plan = {
            "id": "txn-prep",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        }
        run_out = {}
        th = threading.Thread(target=lambda: run_out.update(co.run(plan)), daemon=True)
        th.start()
        self.assertTrue(gap_entered.wait(5))
        self.assertEqual(co.status("txn-prep")["state"], "prepared")
        u_prep = co.undo("txn-prep")
        self.assertEqual((u_prep["ok"], u_prep["status"], u_prep["receipt"]),
                         (False, "refused", None))
        gap_release.set()
        th.join(5)
        self.assertEqual(run_out["outcome"], "committed")
        self.assertEqual(co.status("txn-prep")["state"], "committed")
        self.assertEqual([r["kind"] for r in R.read(path)], ["txn"])

        # Phase-2 in-doubt (run timed out waiting for gap while commit runs)
        name2, eng2, _ = fresh("c2_indoubt")
        lp2 = C.LocalParticipant("r0", alias={"toy": name2})
        commit_entered = threading.Event()
        commit_release = threading.Event()
        orig_commit = lp2.commit

        def slow_commit(ref):
            commit_entered.set()
            commit_release.wait(5)
            return orig_commit(ref)

        lp2.commit = slow_commit

        def timeout_gap(fn):
            done = threading.Event()
            box = {}

            def worker():
                box["r"] = fn()
                done.set()

            threading.Thread(target=worker, daemon=True).start()
            if not done.wait(0.05):
                raise TimeoutError("gap timed out")
            return box["r"]

        co2 = C.StepGapCoordinator([lp2], gap=timeout_gap, phase2_deadline_s=None)
        res2 = co2.run({
            "id": "txn-indoubt",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })
        self.assertEqual(res2["outcome"], "in-doubt")
        self.assertTrue(commit_entered.wait(5))
        self.assertEqual(co2.status("txn-indoubt")["state"], "in-doubt")
        u_indoubt = co2.undo("txn-indoubt")
        self.assertEqual((u_indoubt["ok"], u_indoubt["status"], u_indoubt["receipt"]),
                         (False, "refused", None))
        commit_release.set()
        wait_for(lambda: co2.status("txn-indoubt")["state"] == "committed", timeout=5)
        self.assertEqual(eng2.Engine().step(3), 170)

        # _done() must not overwrite superseded after recovered().
        co2.recovered()
        self.assertEqual(co2.txns["txn-indoubt"]["state"], "superseded")
        co2._done("txn-indoubt", {"outcome": "committed", "hashes": {}, "extra": [],
                                  "rollback": {"status": "not_needed", "participants": {}}})
        self.assertEqual(co2.txns["txn-indoubt"]["state"], "superseded")

    def test_c3_c9_recovered_isolates_incarnation_across_undo_and_run(self):
        """C3 & C9: `recovered()` isolates prior-incarnation in-flight `undo()` and `run()`,
        never overwriting `'superseded'` and never calling or poisoning the new incarnation."""
        # C3: undo() vs recovered()
        name1, _, _ = fresh("c3_rec1")
        lp1 = C.LocalParticipant("r0", alias={"toy": name1})
        co = C.StepGapCoordinator([lp1], gap=lambda fn: fn(), phase2_deadline_s=None)
        r1 = co.run({
            "id": "inc1-txn",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })
        self.assertEqual(r1["outcome"], "committed")

        undo_entered = threading.Event()
        undo_release = threading.Event()

        def gap_pause(fn):
            undo_entered.set()
            undo_release.wait(5)
            return fn()

        name2, _, _ = fresh("c3_rec2")
        lp2 = C.LocalParticipant("r0", alias={"toy": name2})
        lp2_calls = []

        def fail_undo(txn):
            lp2_calls.append(txn)
            return F.not_ok("inc2 should never be called",
                            [F.finding("undo_incomplete", "bad", "ok")])

        lp2.undo = fail_undo
        co.gap = gap_pause

        undo_res = {}
        th = threading.Thread(target=lambda: undo_res.update(co.undo("inc1-txn")), daemon=True)
        th.start()
        self.assertTrue(undo_entered.wait(5))

        co.participants = [lp2]
        rec = co.recovered()
        self.assertEqual(rec["incarnation"], 2)
        self.assertEqual(co.txns["inc1-txn"]["state"], "superseded")

        undo_release.set()
        th.join(5)
        self.assertFalse(undo_res["ok"])
        self.assertEqual(lp2_calls, [])
        self.assertEqual(co.txns["inc1-txn"]["state"], "superseded")
        self.assertFalse(co.is_recovery_required())
        self.assertEqual(co.poison, [])

        # C9: run() vs recovered()
        n_a, _, _ = fresh("c9_run1")
        lp_a = C.LocalParticipant("r0", alias={"toy": n_a})
        prep_entered = threading.Event()
        prep_release = threading.Event()
        orig_prep = lp_a.prepare

        def slow_prep(plan):
            prep_entered.set()
            prep_release.wait(5)
            return orig_prep(plan)

        lp_a.prepare = slow_prep
        co2 = C.Coordinator([lp_a], phase2_deadline_s=None)
        run_out = {}
        th2 = threading.Thread(target=lambda: run_out.update(co2.run({
            "id": "inc1-race",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })), daemon=True)
        th2.start()
        self.assertTrue(prep_entered.wait(5))

        n_b, _, _ = fresh("c9_run2")
        lp_b = C.LocalParticipant("r0", alias={"toy": n_b})
        lp_b_calls = []
        lp_b.abort = lambda ref: (lp_b_calls.append(("abort", ref)), F.not_ok("inc2 abort failed"))[1]
        lp_b.validate = lambda ref: (lp_b_calls.append(("validate", ref)), F.not_ok("bad"))[1]
        co2.participants = [lp_b]
        co2.recovered()
        self.assertEqual(co2.incarnation, 2)
        self.assertEqual(co2.txns["inc1-race"]["state"], "superseded")

        prep_release.set()
        th2.join(5)
        self.assertEqual(lp_b_calls, [])
        self.assertEqual(co2.txns["inc1-race"]["state"], "superseded")
        self.assertFalse(co2.is_recovery_required())
        self.assertEqual(co2.poison, [])

    def test_c4_undo_cascade_snapshots_inside_gap_and_preserves_undone(self):
        """C4: `undo()` computes `later` and `resets` inside `undo_job()` (in the gap),
        and never overwrites `'undone'` or `'superseded'` transactions during cascade."""
        name, _, _ = fresh("c4_drift1")
        reset_calls = []
        lp = C.LocalParticipant("r0", alias={"toy": name},
                                resets={"R2": lambda: reset_calls.append("R2") or "ok"})
        undo_gap_wait = threading.Event()
        undo_gap_go = threading.Event()
        in_undo_gap = [False]

        def gap(fn):
            if in_undo_gap[0]:
                undo_gap_wait.set()
                undo_gap_go.wait(5)
            return fn()

        co = C.StepGapCoordinator([lp], gap=gap, phase2_deadline_s=None)
        t1 = co.run({
            "id": "t1",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })
        self.assertEqual(t1["outcome"], "committed")

        in_undo_gap[0] = True
        u_out = {}
        th = threading.Thread(target=lambda: u_out.update(co.undo("t1")), daemon=True)
        th.start()
        self.assertTrue(undo_gap_wait.wait(5))

        in_undo_gap[0] = False
        t2 = co.run({
            "id": "t2",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 100")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
            "undo_resets": ["R2"],
        })
        self.assertEqual(t2["outcome"], "committed")

        undo_gap_go.set()
        th.join(5)
        self.assertTrue(u_out["ok"], u_out)
        self.assertEqual(u_out["receipt"]["resets"], ["R2"])
        self.assertEqual(reset_calls, ["R2"])
        self.assertEqual(co.status("t2")["state"], "undone")

        # Stale later snapshot must not overwrite an already-undone transaction to partial.
        name2, _, _ = fresh("c4_drift2")
        lp2 = C.LocalParticipant("r0", alias={"toy": name2})
        wait2 = threading.Event()
        go2 = threading.Event()
        hold_a = [False]

        def gap2(fn):
            if hold_a[0]:
                hold_a[0] = False
                wait2.set()
                go2.wait(5)
            return fn()

        co2 = C.StepGapCoordinator([lp2], gap=gap2, phase2_deadline_s=None)
        self.assertEqual(co2.run({"id": "a", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}})["outcome"], "committed")
        self.assertEqual(co2.run({"id": "b", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 20")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}})["outcome"], "committed")

        hold_a[0] = True
        ua_out = {}
        th_a = threading.Thread(target=lambda: ua_out.update(co2.undo("a")), daemon=True)
        th_a.start()
        self.assertTrue(wait2.wait(5))

        ub = co2.undo("b")
        self.assertTrue(ub["ok"])
        self.assertEqual(co2.status("b")["state"], "undone")

        lp2.undo = lambda _txn: F.not_ok("a undo failed", [F.finding("undo_incomplete", "fail", "ok")])
        go2.set()
        th_a.join(5)
        self.assertEqual(ua_out["status"], "partial")
        self.assertEqual(co2.status("b")["state"], "undone")

    def test_c5_undo_indoubt_latches_recovery_and_receipt_matches_status(self):
        """C5: `undo()` receipt `status` matches `status_label` (`cancelled`, `in-doubt`,
        `partial`, `undone`), `in-doubt` latches `recovery_required` under lock with incarnation
        guard, and `undo_state['done']` is read under lock."""
        name, _, _ = fresh("c5_main")
        lp = C.LocalParticipant("r0", alias={"toy": name})
        co = C.StepGapCoordinator([lp], gap=lambda fn: fn(), phase2_deadline_s=None)
        r = co.run({
            "id": "t-1",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })
        self.assertEqual(r["outcome"], "committed")

        # Cancelled undo receipt status must be 'cancelled', not 'partial'.
        co.gap = lambda _fn: (_ for _ in ()).throw(TimeoutError("gap busy"))
        u_cancel = co.undo("t-1")
        self.assertEqual((u_cancel["status"], u_cancel["receipt"]["status"]),
                         ("cancelled", "cancelled"))
        self.assertFalse(co.is_recovery_required())

        # In-doubt undo latches recovery_required and refuses run() and second undo().
        entered = threading.Event()
        release = threading.Event()
        orig_undo = lp.undo

        def slow_undo(txn):
            entered.set()
            release.wait(5)
            return orig_undo(txn)

        lp.undo = slow_undo

        def timeout_after_start(fn):
            threading.Thread(target=fn, daemon=True).start()
            self.assertTrue(entered.wait(5))
            raise TimeoutError("gap timed out mid-undo")

        co.gap = timeout_after_start
        u_indoubt = co.undo("t-1")
        self.assertEqual((u_indoubt["ok"], u_indoubt["status"], u_indoubt["receipt"]["status"]),
                         (False, "in-doubt", "in-doubt"))
        self.assertTrue(u_indoubt["recovery_required"])
        self.assertTrue(co.is_recovery_required())
        self.assertTrue(any(f["check"] == "deadline_missed" for f in co.poison), co.poison)
        self.assertEqual(co.status()["state"], "poisoned")
        self.assertEqual(co.run({"modules": {}})["outcome"], "refused")

        co.gap = lambda fn: fn()
        u_second = co.undo("t-1")
        self.assertEqual((u_second["ok"], u_second["status"], u_second["recovery_required"]),
                         (False, "refused", True))
        release.set()

        # Concurrent recovered() during in-doubt timeout must not poison new incarnation.
        name2, _, _ = fresh("c5_rec")
        lp2 = C.LocalParticipant("r0", alias={"toy": name2})
        co2 = C.StepGapCoordinator([lp2], gap=lambda fn: fn(), phase2_deadline_s=None)
        self.assertEqual(co2.run({
            "id": "t-2",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })["outcome"], "committed")

        started2 = threading.Event()
        release2 = threading.Event()
        lp2.undo = lambda _txn: (started2.set(), release2.wait(5), F.ok(undone=1))[2]

        def gap_rec(fn):
            threading.Thread(target=fn, daemon=True).start()
            self.assertTrue(started2.wait(5))
            co2.recovered()
            raise TimeoutError("gap timed out after recovered")

        co2.gap = gap_rec
        co2.undo("t-2")
        release2.set()
        self.assertEqual(co2.incarnation, 2)
        self.assertEqual(co2.poison, [])
        self.assertFalse(co2.is_recovery_required())
        self.assertEqual(co2.txns["t-2"]["state"], "superseded")

    def test_c6_post_deadline_queued_abort_failure_latches_recovery(self):
        """C6: Post-deadline queued `abort` stays non-blocking for the gap, and latches
        `recovery_required` (`participant_failed`) if the queued `Lane` abort fails or raises."""
        n0, _, _ = fresh("c6_ab0")
        n1, _, _ = fresh("c6_ab1")
        lp0 = C.LocalParticipant("r0", alias={"toy": n0})
        lp1 = C.LocalParticipant("r1", alias={"toy": n1})
        p1_abort_crashed = threading.Event()

        def p0_rpc(method, payload):
            if method == "validate":
                time.sleep(0.15)
                return F.not_ok("r0 validate late rejection")
            return getattr(lp0, method)(payload)

        def p1_rpc(method, payload):
            if method == "abort":
                p1_abort_crashed.set()
                raise ConnectionError("r1 crashed during abort!")
            return getattr(lp1, method)(payload)

        p0 = C.CallParticipant("r0", p0_rpc, bounded=True)
        p1 = C.CallParticipant("r1", p1_rpc, bounded=False)
        co = C.StepGapCoordinator([p0, p1], gap=lambda fn: fn(), phase2_deadline_s=0.1)
        res = co.run({
            "id": "txn-unconf-ab",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })
        self.assertEqual(res["outcome"], "aborted")
        self.assertTrue(p1_abort_crashed.wait(2))
        wait_for(co.is_recovery_required, timeout=2.0)
        self.assertTrue(co.is_recovery_required())
        self.assertTrue(any(f["check"] == "participant_failed" and f["participant"] == "r1"
                            for f in co.poison), co.poison)

    def test_c7_stepgap_timed_out_run_then_undo_writes_txn_receipt_first(self):
        """C7: When `StepGapCoordinator.run()` times out (`in-doubt`) and finishes committing,
        calling `undo(txn)` before `status(txn)` flushes the `txn` receipt (with valid
        `commit_hashes`) before writing the `undo` receipt."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "r.jsonl")
        name, _, _ = fresh("c7_ord")
        lp = C.LocalParticipant("r0", alias={"toy": name})
        done_gap = threading.Event()
        commit_started = threading.Event()
        commit_release = threading.Event()
        orig_commit = lp.commit

        def gated_commit(ref):
            commit_started.set()
            commit_release.wait(5)
            return orig_commit(ref)

        lp.commit = gated_commit

        def timeout_gap(fn):
            threading.Thread(target=lambda: (fn(), done_gap.set()), daemon=True).start()
            self.assertTrue(commit_started.wait(5))
            raise TimeoutError("gap timed out")

        co = C.StepGapCoordinator([lp], gap=timeout_gap, writer=R.ReceiptWriter(path),
                                  phase2_deadline_s=None)
        r = co.run({
            "id": "txn-ord",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })
        self.assertEqual(r["outcome"], "in-doubt")
        commit_release.set()
        self.assertTrue(done_gap.wait(5))

        co.gap = lambda fn: fn()
        u = co.undo("txn-ord")
        self.assertEqual(u["status"], "undone")
        recs = R.read(path)
        self.assertEqual([x["kind"] for x in recs], ["txn", "undo"])
        txn_rec = recs[0]
        self.assertEqual(txn_rec["outcome"], "committed")
        self.assertTrue(txn_rec["commit_hashes"]["participants"]["r0"])

    def test_c8_at_base_false_during_inflight_commit_and_indoubt(self):
        """C8: `LocalParticipant.at_base()` returns `ok=False` while a commit job is still
        in-flight in `quiesce`, and `Coordinator.at_base()` returns `ok=False` while any
        transaction is `in-doubt` or `recovery_required` is latched."""
        name, _, _ = fresh("c8_atbase")
        entered_q = threading.Event()
        release_q = threading.Event()

        def slow_quiesce(fn):
            entered_q.set()
            release_q.wait(5)
            return fn()

        lp = C.LocalParticipant("r0", alias={"toy": name}, quiesce=slow_quiesce)
        co = C.StepGapCoordinator([lp], gap=lambda fn: fn(), phase2_deadline_s=None)

        done_bg = threading.Event()

        def gap_starts_then_times_out(fn):
            threading.Thread(target=lambda: (fn(), done_bg.set()), daemon=True).start()
            self.assertTrue(entered_q.wait(5))
            raise TimeoutError("gap wait timed out")

        co.gap = gap_starts_then_times_out
        res = co.run({
            "id": "txn-ab",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 10")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
        })
        self.assertEqual(res["outcome"], "in-doubt")
        self.assertEqual(co.status()["state"], "in-doubt")
        self.assertFalse(lp.at_base()["ok"])
        self.assertFalse(co.at_base()["ok"])
        self.assertFalse(co.status()["at_base"])

        release_q.set()
        self.assertTrue(done_bg.wait(5))
        self.assertEqual(co.status("txn-ab")["state"], "committed")
        self.assertFalse(co.at_base()["ok"])


if __name__ == "__main__":
    unittest.main()
