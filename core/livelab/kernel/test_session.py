"""Golden tests of the session (base / edited / poisoned) on toy ranks (CPU, standard library
only). The whole lab is checked by the template's conformance cases against `toy_lab.ToyLab`
(CI step "Run the template conformance cases c0-c10 against the toy lab").

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
(or python3 -m pytest -q core/livelab/kernel)
"""
import time
import unittest

from . import coordinator as C
from . import session as SS
from . import test_coordinator as TC
from .test_locate import ENG, OPS, at, edit, fresh
from .test_witness import wait_for


class SessionTest(unittest.TestCase):
    _cluster, _start, _steps, V1 = (TC.HostTest._cluster, TC.HostTest._start,
                                    TC.HostTest._steps, TC.HostTest.V1)

    def test_code_changed_outside_the_journal_poisons_the_session(self):
        """Input: a session over one rank; a transaction commits (step returns y * 100);
        then someone rebinds Engine.step.__code__ by hand (a REPL edit, not journaled).
        Expected: status() says poisoned with a `tampered` finding naming the function; a
        new install is refused with session_poisoned; uninstall (undo) is still allowed and
        the session stays poisoned. Pain point: a lab that reports "edited" while it runs
        code no transaction installed. Not: tampering in another process (each rank checks
        its own fingerprint)."""
        name, eng, _ = fresh("tamper")
        e = eng.Engine()
        sess = SS.Session(C.Coordinator([C.LocalParticipant("rank0", alias={"toy": name})],
                                        phase2_deadline_s=None), ["toy.engine"])
        plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 100")),
            "targets": [["Engine.step", at(ENG, "def step")]]}}}
        res = sess.install(plan)
        self.assertEqual((res["outcome"], sess.status()["state"], e.step(3)),
                         ("committed", "edited", 1700))
        eng.Engine.step.__code__ = eng.Engine.drain.__code__.replace(
            co_name="step", co_qualname="Engine.step")                     # by hand
        st = sess.status()
        self.assertEqual((st["state"], [f["check"] for f in st["poison"]]),
                         ("poisoned", ["tampered"]))
        self.assertIn("toy.engine:Engine.step", st["poison"][0]["observed"])
        again = sess.install(plan)
        self.assertEqual((again["outcome"], [f["check"] for f in again["findings"]]),
                         ("refused", ["session_poisoned"]))
        self.assertTrue(sess.uninstall()["ok"])
        self.assertEqual(sess.state(), "poisoned")

    def test_group_participant_gives_every_ranks_fingerprint(self):
        """Input: a session over one GroupParticipant of two ranks (each its own copy of the
        toy module); an install of `return y * 100`; then a hand edit of Engine.step on rank 1
        only. Expected: the session starts at base and the install commits ("edited"); after
        the hand edit status() is poisoned with a `tampered` finding naming rank 1's
        Engine.step. Pain point: the group did not forward fingerprint, so it answered "no
        fingerprint" and a session over a group was poisoned before its first install. Not:
        ranks in other processes (adapter host test_side)."""
        n0, eng0, _ = fresh("grp0")
        n1, eng1, _ = fresh("grp1")
        grp = C.GroupParticipant("workers", [C.LocalParticipant("r0", alias={"toy": n0}),
                                             C.LocalParticipant("r1", alias={"toy": n1})])
        fp = grp.fingerprint(["toy.engine"])
        self.assertTrue(fp["ok"], fp.get("error"))
        self.assertIn(f"rank1/toy.engine:Engine.step:{at(ENG, 'def step')}", fp["code"])
        sess = SS.Session(C.Coordinator([grp], phase2_deadline_s=None), ["toy.engine"])
        self.assertEqual(sess.state(), "base", sess.status())
        plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 100")),
            "targets": [["Engine.step", at(ENG, "def step")]]}}}
        res = sess.install(plan)
        self.assertEqual((res["outcome"], sess.status()["state"]), ("committed", "edited"))
        self.assertEqual((eng0.Engine().step(3), eng1.Engine().step(3)), (1700, 1700))
        eng1.Engine.step.__code__ = eng1.Engine.drain.__code__.replace(
            co_name="step", co_qualname="Engine.step")                     # by hand, rank 1
        st = sess.status()
        self.assertEqual((st["state"], [f["check"] for f in st["poison"]]),
                         ("poisoned", ["tampered"]))
        self.assertIn("workers:rank1/toy.engine:Engine.step", st["poison"][0]["observed"])

    def test_install_after_an_unjournaled_edit_is_refused(self):
        """Input: a session over one rank at base; someone rebinds Engine.drain.__code__ by
        hand; with no status() call in between, an install of an edit to Engine.step.
        Expected: refused with session_poisoned, a `tampered` finding naming Engine.drain,
        Engine.step untouched. Fails before the fix: the install committed and recorded the
        hand edit as the code the session expects, so status() then said "edited"."""
        name, eng, _ = fresh("tamper2")
        e = eng.Engine()
        sess = SS.Session(C.Coordinator([C.LocalParticipant("rank0", alias={"toy": name})],
                                        phase2_deadline_s=None), ["toy.engine"])
        eng.Engine.drain.__code__ = eng.Engine.drain.__code__.replace(
            co_consts=eng.Engine.drain.__code__.co_consts + ("by hand",))   # by hand
        res = sess.install({"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 100")),
            "targets": [["Engine.step", at(ENG, "def step")]]}}})
        self.assertEqual((res["outcome"], [f["check"] for f in res.get("findings", [])], e.step(3)),
                         ("refused", ["session_poisoned"], 17))
        self.assertEqual([f["check"] for f in sess.poisoned()], ["tampered"])
        self.assertIn("toy.engine:Engine.drain", sess.poisoned()[0]["observed"])
        self.assertEqual(sess.status()["state"], "poisoned")

    def test_uninstall_past_a_tampered_function_still_poisons(self):
        """Input: two transactions commit (t1 edits Engine.step, t2 edits Engine.drain); then
        someone rebinds Engine.step.__code__ by hand; with no status() call in between,
        uninstall(t1) undoes t2 and t1. Expected: both undone, Engine.step back at base, and
        the session poisoned with `tampered` naming Engine.step. Fails before the fix: undoing
        t1 wrote the base code over the hand edit before the only check ran, so the session
        reported a clean base and the evidence was gone."""
        name, eng, _ = fresh("tamper3")
        e = eng.Engine()
        sess = SS.Session(C.Coordinator([C.LocalParticipant("rank0", alias={"toy": name})],
                                        phase2_deadline_s=None), ["toy.engine"])
        step = ("return y", "return y * 100")
        drain = ("self.tokens.clear()", "self.tokens.clear(); self.ran.append('drained')")
        t1 = sess.install({"modules": {"toy.engine": {
            "source": edit(ENG, step), "targets": [["Engine.step", at(ENG, "def step")]]}}})
        t2 = sess.install({"modules": {"toy.engine": {
            "source": edit(ENG, step, drain),
            "targets": [["Engine.drain", at(ENG, "def drain")]]}}})
        self.assertEqual((t1["outcome"], t2["outcome"]), ("committed", "committed"))
        eng.Engine.step.__code__ = eng.Engine.drain.__code__.replace(
            co_name="step", co_qualname="Engine.step")                     # by hand
        u = sess.uninstall(t1["txn"])
        self.assertEqual((u["undone"], e.step(3)), ([t2["txn"], t1["txn"]], 17))
        self.assertEqual(sess.state(), "poisoned")
        self.assertEqual([f["check"] for f in sess.poisoned()], ["tampered"])
        self.assertIn("toy.engine:Engine.step", sess.poisoned()[0]["observed"])

    def test_no_fingerprint_never_verifies(self):
        """Input: a session whose only participant cannot give a fingerprint (its call
        fails), so the expected one is None too. Expected: verify() is not ok and the session
        is poisoned with `tampered`. Fails before the fix: verify() said ok, because the
        missing fingerprint equalled the missing expected one (None == None)."""
        name, _, _ = fresh("nofp")
        lp = C.LocalParticipant("rank0", alias={"toy": name})

        def call(method, payload):
            if method == "fingerprint":
                return {"ok": False, "error": "rank0 lost"}
            return getattr(lp, method)(payload)
        sess = SS.Session(C.Coordinator([C.CallParticipant("rank0", call)],
                                        phase2_deadline_s=None), ["toy.engine"])
        v = sess.verify()
        self.assertEqual((v["ok"], v["fingerprint"], sess.state()), (False, None, "poisoned"))
        self.assertEqual({f["check"] for f in sess.poisoned()}, {"tampered"})

    def test_hung_commit_ends_the_gap_at_the_deadline(self):
        """Proves: with steps running, a rank that hangs 0.6 s in commit does not stall the loop: the gap ends within the 0.2 s deadline, rank0's commit is undone in it, no step ever runs mixed versions (not even after the late commit arrives: it is refused), the outcome is definite (rolled back, state aborted) and the host and session are poisoned (install refused, undo allowed). Not: a participant stuck inside apply (its late switch is undone by the queued undo, after it ran)."""
        cl, ps, lps = self._cluster("deadline", gap_mode=True, slow={"rank1": {"commit": 0.6}})
        co = C.StepGapCoordinator(ps, cl.at_gap, phase2_deadline_s=0.2)
        sess = SS.Session(co, ["toy.engine"])
        halt = self._start(cl)
        t0 = time.perf_counter()
        res = sess.install({"modules": self.V1})
        took = time.perf_counter() - t0
        st = co.status(res["txn"])
        rec = res["receipt"]
        self.assertEqual((res["outcome"], st["state"], st["poisoned"], st["missed"]),
                         ("rolled_back", "aborted", True, ["rank1"]))
        self.assertLess(rec["gap_s"], 0.2 + 0.05)
        self.assertLess(took, 0.5)
        self.assertEqual(rec["rollback"]["missed_deadline"], ["rank1"])
        self.assertIn(("rank1", "deadline_missed"),
                      [(f.get("participant"), f["check"]) for f in rec["findings"]])
        self.assertEqual(sess.state(), "poisoned")
        again = sess.install({"modules": self.V1})
        self.assertEqual([f["check"] for f in again["findings"]], ["session_poisoned"])
        self.assertEqual(co.run({"modules": self.V1})["outcome"], "refused")
        # the hung commit arrives late; then the undo and abort queued behind it run
        wait_for(lambda: lps[1].status(res["txn"])["state"] != "prepared")
        self._steps(cl)
        halt()
        self.assertLess(cl.longest_stall(), 0.2 + 0.05)  # measured by the loop itself
        self.assertEqual(set(cl.log), {("base",) * 3})    # never v1 anywhere, never mixed
        self.assertEqual([p.status(res["txn"])["state"] for p in lps],
                         ["undone", "undone", "aborted"])   # rank1: its queued undo ran
        self.assertEqual(lps[1].journal.entries, [])     # the late commit applied nothing
        self.assertTrue(sess.uninstall()["ok"])          # undo stays allowed while poisoned
        self.assertEqual(sess.state(), "poisoned")

    def test_ranks_that_disagree_after_commit_poison_the_session(self):
        """Input: a session over a group of 2 ranks; rank 1's client changes its own
        Engine.step code right after its commit returns ok (a rank that did not switch
        cleanly). Expected: the install commits, the receipt says commit_hashes.agree False
        with a ranks_disagree finding, and the session is poisoned (status() says so) and a
        second install is refused with session_poisoned. Pain point: the session recorded the
        split fingerprint as the expected one, so verify() passed and the lab reported
        "edited" while its ranks ran different code. Not: an accept, which never excuses it
        (test_a_rank_that_never_switched_poisons_even_with_accept_all)."""
        n0, _, _ = fresh("split0")
        n1, eng1, _ = fresh("split1")
        lp0 = C.LocalParticipant("r0", alias={"toy": n0})
        lp1 = C.LocalParticipant("r1", alias={"toy": n1})

        def rank1(method, payload):
            res = getattr(lp1, method)(payload)
            if method == "commit":
                step = eng1.Engine.step
                step.__code__ = step.__code__.replace(co_consts=step.__code__.co_consts + ("x",))
            return res
        group = C.GroupParticipant("workers", [lp0, C.CallParticipant("r1", rank1)])
        sess = SS.Session(C.Coordinator([group], phase2_deadline_s=None), ["toy.engine"])
        plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 100")),
            "targets": [["Engine.step", at(ENG, "def step")]]}}}
        res = sess.install(plan)
        rec = res["receipt"]
        self.assertEqual((res["outcome"], rec["commit_hashes"]["agree"]), ("committed", False))
        self.assertIn("ranks_disagree", [finding["check"] for finding in rec["findings"]])
        self.assertEqual((res["state"], sess.status()["state"]), ("poisoned", "poisoned"))
        self.assertEqual(sess.install(plan)["findings"][0]["check"], "session_poisoned")

    def test_a_rank_that_never_switched_poisons_even_with_accept_all(self):
        """Input: (a) a group of 2 ranks where rank 1's client answers commit ok without
        committing and answers proof with no code hashes; (b) 2 separate participants, the
        second doing the same. Each plan has accept "all". Expected: the install commits,
        the session is poisoned (ranks_disagree) and the silent rank still runs base code
        (step(3) = 17 vs 1700). Pain point: in (a) an overridable ranks_disagree was excused
        by the accept, and in (b) commit_hashes.agree was None (no target shared), so the
        session recorded the split as its expected state, "edited", verify() ok. Not: a role
        that abstains or stages nothing (a plan of repair cells only), which may show no
        hash."""
        plan = {"accept": "all", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 100")),
            "targets": [["Engine.step", at(ENG, "def step")]]}}}
        for grouped in (True, False):
            n0, eng0, _ = fresh("silent0")
            n1, eng1, _ = fresh("silent1")
            lp0 = C.LocalParticipant("r0", alias={"toy": n0})
            lp1 = C.LocalParticipant("r1", alias={"toy": n1})

            def silent(method, payload, lp1=lp1):
                if method == "commit":
                    return {"ok": True, "findings": []}         # says ok, installs nothing
                if method == "proof":
                    return {"ok": True, "findings": [], "hashes": {}, "witness": {}}
                return getattr(lp1, method)(payload)
            second = C.CallParticipant("r1", silent)
            parts = [C.GroupParticipant("workers", [lp0, second])] if grouped else [lp0, second]
            sess = SS.Session(C.Coordinator(parts, phase2_deadline_s=None), ["toy.engine"])
            res = sess.install(plan)
            self.assertEqual((res["outcome"], res["state"]), ("committed", "poisoned"), grouped)
            self.assertIn("ranks_disagree", [finding["check"] for finding in sess.poisoned()])
            self.assertEqual((eng0.Engine().step(3), eng1.Engine().step(3)), (1700, 17))

    def test_a_commit_with_no_receipt_to_read_poisons(self):
        """Input: a coordinator whose run reports committed with receipt None and keeps no
        record of the transaction. Expected: the session is poisoned. Pain point: no receipt
        read as "no split", so code nobody checked became the expected state. Not: the real
        coordinator, which keeps the receipt it built (`txns[txn]["rec"]`) and the session
        reads it from there."""
        name, _, _ = fresh("norec")
        coordinator = C.Coordinator([C.LocalParticipant("r0", alias={"toy": name})],
                                    phase2_deadline_s=None)
        real_run = coordinator.run

        def run_without_receipt(plan, **kw):
            res = real_run(plan, **kw)
            coordinator.txns.pop(res["txn"], None)
            return {**res, "receipt": None}
        coordinator.run = run_without_receipt
        sess = SS.Session(coordinator, ["toy.engine"])
        res = sess.install({"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 100")),
            "targets": [["Engine.step", at(ENG, "def step")]]}}})
        self.assertEqual((res["outcome"], res["state"]), ("committed", "poisoned"))

    def test_add_only_mixed_and_removal_only_edits_end_edited(self):
        """Input: one participant on toy.ops. (a) A plan that only adds a top-level
        `def added_helper(x): return x + 5`, then on top (c) a plan that only removes it
        again (removed_definition accepted); separately from base, (b) a plan that adds it
        and edits scale (x * 3). Expected: each commits and the session is "edited", never
        poisoned; after (a) added_helper(1) = 6; after (c) it is gone; after (b)
        added_helper(1) = 6 and scale(3) = 9. Pain point: the switch check read only the
        proof's code hashes, which cover swapped functions, so an add-only edit that did
        switch read as a rank that did not (poisoned). Not: a rank that lies about its
        witness keys."""
        added = OPS + "\n\ndef added_helper(x):\n    return x + 5\n"

        def session(tag):
            name, _, ops = fresh(tag)
            part = C.LocalParticipant("r0", alias={"toy": name})
            return SS.Session(C.Coordinator([part], phase2_deadline_s=None), ["toy.ops"]), ops

        def install(sess, source, accept=()):
            res = sess.install({"accept": list(accept), "modules": {"toy.ops": {"source": source}}})
            self.assertEqual((res["outcome"], res["state"]), ("committed", "edited"),
                             sess.poisoned() or res.get("findings"))
        sess, ops = session("addonly")
        install(sess, added)
        self.assertEqual(ops.added_helper(1), 6)
        install(sess, OPS, accept=["removed_definition"])
        self.assertFalse(hasattr(ops, "added_helper"))
        sess, ops = session("addswap")
        install(sess, added.replace("x * STEP_SCALE", "x * 3"))
        self.assertEqual((ops.added_helper(1), ops.scale(3)), (6, 9))
    def test_a_receipt_another_thread_is_finishing_is_waited_for(self):
        """Input: run() reports committed with receipt None because a status() call in
        another thread holds the finisher lock and is still building the receipt (emulated:
        a thread holds `fin` and puts `rec` back after 0.2 s). Expected: the session waits
        for it, reads the receipt and ends "edited". Pain point: the session poisoned a
        healthy commit when it lost the finisher race. Not: a finisher that never ends
        (after 30 s the session poisons: no receipt)."""
        import threading
        name, eng, _ = fresh("finrace")
        coordinator = C.Coordinator([C.LocalParticipant("r0", alias={"toy": name})],
                                    phase2_deadline_s=None)
        real_run = coordinator.run

        def run_losing_the_race(plan, **kw):
            res = real_run(plan, **kw)
            record = coordinator.txns[res["txn"]]
            rec, held = record.pop("rec"), threading.Event()

            def other_finisher():
                with record["fin"]:
                    held.set()
                    time.sleep(0.2)
                    record["rec"] = rec
            threading.Thread(target=other_finisher).start()
            held.wait()
            return {**res, "receipt": None}
        coordinator.run = run_losing_the_race
        sess = SS.Session(coordinator, ["toy.engine"])
        res = sess.install({"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 100")),
            "targets": [["Engine.step", at(ENG, "def step")]]}}})
        self.assertEqual((res["outcome"], res["state"], eng.Engine().step(3)),
                         ("committed", "edited", 1700))


if __name__ == "__main__":
    unittest.main()
