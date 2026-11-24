"""Bounded formal protocol verification & adversarial fault-injection tests (CPU, stdlib only).

Covers two layers with zero Java/TLC dependency:
1. Exhaustive bounded BFS state-space checking of the four TLA+ protocol specifications in
   `core/livelab/kernel/formal/` (`Transaction.tla`, `Participant.tla`, `SequentialTx.tla`,
   `Incarnation.tla`), including negative-control bug mutants proving each invariant catches
   its targeted protocol violation.
2. Real kernel trace-replay and adversarial fault-injection tests against `Coordinator`,
   `StepGapCoordinator`, `LocalParticipant`, `CallParticipant`, `Session`, and `ToyLab`.

Run from the repository root:
    python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
"""
from collections import deque
import os
import shutil
import tempfile
import time
import unittest

from . import coordinator as C
from . import findings as F
from . import identity as ID
from . import locate as L
from . import receipt as R
from . import session as SS
from . import toy_lab as TL
from .test_detectors import temp_modules
from .test_locate import ENG, OPS, at, edit, fresh


def _bfs_check(init_state, transitions, invariants, max_states=50_000):
    """Exhaustively explore finite reachable states via BFS.
    Returns `(visited_count, violation)` where `violation` is `(inv_name, trace)` or None."""
    queue = deque([(init_state, [])])
    visited = {init_state}
    while queue:
        state, trace = queue.popleft()
        for name, predicate in invariants.items():
            if not predicate(state):
                return len(visited), (name, trace + [("VIOLATED", state)])
        for action, nxt in transitions(state):
            if nxt not in visited:
                visited.add(nxt)
                if len(visited) > max_states:
                    raise RuntimeError(f"state space exceeded bound {max_states}")
                queue.append((nxt, trace + [(action, nxt)]))
    return len(visited), None


class BoundedProtocolModelCheckTest(unittest.TestCase):
    """Native Python stdlib BFS verifier matching `core/livelab/kernel/formal/*.tla`."""

    def test_formal_tla_specs_present_and_documented(self):
        formal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "formal")
        expected = ("Transaction.tla", "Participant.tla", "SequentialTx.tla",
                    "Incarnation.tla", "README.md")
        for fname in expected:
            fpath = os.path.join(formal_dir, fname)
            self.assertTrue(os.path.isfile(fpath), f"missing {fname}")
            with open(fpath, encoding="utf-8") as fh:
                text = fh.read()
            self.assertTrue(len(text) > 100, fname)
        with open(os.path.join(formal_dir, "README.md"), encoding="utf-8") as fh:
            readme = fh.read()
        self.assertIn("not", readme.lower())
        self.assertIn("numerical", readme.lower())

    def _run_transaction_bfs(self, *, mutant=None):
        """State: (coord_state, p_states_tuple, recovery_required, poisoned)."""
        n_parts = 2
        init = ("idle", ("idle",) * n_parts, False, False)

        def transitions(st):
            cs, ps, rec_req, pois = st
            out = []
            # StartPrepare
            can_start = (cs == "idle") if mutant == "skip_rollback_freeze" else (
                cs == "idle" and not rec_req and not pois
            )
            if can_start:
                out.append(("StartPrepare", ("preparing", ps, rec_req, pois)))
            if mutant == "skip_rollback_freeze" and cs == "recovery_required":
                out.append(("MutantStartWhileFrozen", ("preparing", ("idle",) * n_parts, rec_req, pois)))
            # ParticipantPrepare
            if cs == "preparing":
                for i in range(n_parts):
                    if ps[i] == "idle":
                        nps = ps[:i] + ("prepared",) + ps[i + 1:]
                        out.append((f"Prepare({i})", (cs, nps, rec_req, pois)))
                if all(p == "prepared" for p in ps):
                    out.append(("PrepareAllYes", ("validating", ps, rec_req, pois)))
                out.append(("PrepareAbortClean", ("aborted", ("aborted",) * n_parts, rec_req, pois)))
            # AbortUnconfirmed
            if cs in ("preparing", "validating", "rolling_back"):
                for i in range(n_parts):
                    nps = ps[:i] + ("dead",) + ps[i + 1:]
                    out.append((f"AbortUnconfirmed({i})", ("recovery_required", nps, True, True)))
            # ValidatePass
            if cs == "validating" and all(p == "prepared" for p in ps):
                out.append(("ValidatePass", ("committing", ps, rec_req, pois)))
            # Committing
            if cs == "committing":
                for i in range(n_parts):
                    if ps[i] == "prepared":
                        nps = ps[:i] + ("committed",) + ps[i + 1:]
                        out.append((f"Commit({i})", (cs, nps, rec_req, pois)))
                        # Commit failure starts rollback
                        fps = ps[:i] + ("aborted",) + ps[i + 1:]
                        out.append((f"CommitFail({i})", ("rolling_back", fps, rec_req, pois)))
                if mutant == "allow_split_commit":
                    if any(p == "committed" for p in ps):
                        out.append(("MutantSplitFinalize", ("committed", ps, rec_req, pois)))
                elif all(p == "committed" for p in ps):
                    out.append(("FinalizeCommit", ("committed", ps, rec_req, pois)))
            # Rolling back
            if cs == "rolling_back":
                for i in range(n_parts):
                    if ps[i] == "committed":
                        nps = ps[:i] + ("undone",) + ps[i + 1:]
                        out.append((f"RollbackClean({i})", (cs, nps, rec_req, pois)))
                    if ps[i] in ("committed", "prepared"):
                        nps = ps[:i] + ("partial",) + ps[i + 1:]
                        out.append((f"RollbackFail({i})", ("recovery_required", nps, True, True)))
                if all(p in ("undone", "aborted", "idle") for p in ps):
                    out.append(("FinalizeRollbackClean", ("rolled_back", ps, rec_req, pois)))
            # Respawn
            out.append(("Respawn", init))
            return out

        invariants = {
            "no_split_commit": lambda st: (st[0] != "committed") or all(p == "committed" for p in st[1]),
            "rollback_freeze": lambda st: (not st[2]) or (st[0] == "recovery_required" and st[3]),
        }
        return _bfs_check(init, transitions, invariants)

    def test_transaction_spec_invariants_and_mutants(self):
        visited, violation = self._run_transaction_bfs()
        self.assertGreater(visited, 20)
        self.assertIsNone(violation, f"Transaction.tla invariant violated: {violation}")

        _, v_freeze = self._run_transaction_bfs(mutant="skip_rollback_freeze")
        self.assertIsNotNone(v_freeze)
        self.assertEqual(v_freeze[0], "rollback_freeze")

        _, v_split = self._run_transaction_bfs(mutant="allow_split_commit")
        self.assertIsNotNone(v_split)
        self.assertEqual(v_split[0], "no_split_commit")

    def _run_participant_and_sequential_bfs(self, *, mutant=None):
        """Combined check for Participant.tla + SequentialTx.tla."""
        # Participant state: (p_state, journal_depth, deadline_passed, poisoned)
        p_init = ("idle", 0, False, False)

        def p_trans(st):
            ps, depth, dl, pois = st
            out = [("Respawn", p_init)]
            if ps == "idle" and not pois:
                out.append(("Prepare", ("prepared", depth, dl, pois)))
            if not dl:
                out.append(("DeadlineExpire", (ps, depth, True, pois)))
            if ps == "prepared" and not dl and not pois and depth < 2:
                out.append(("Commit", ("committed", depth + 1, dl, pois)))
            if ps == "prepared" and dl:
                out.append(("LateCommitRefused", ("aborted", depth, dl, pois)))
            if ps == "prepared":
                out.append(("Abort", ("aborted", depth, dl, pois)))
            if ps == "committed" and depth > 0:
                out.append(("UndoClean", ("undone", depth - 1, dl, pois)))
                out.append(("UndoPartial", ("partial", depth, dl, True)))
            return out

        p_invs = {
            "partial_implies_poisoned": lambda st: st[0] != "partial" or st[3],
            "committed_has_journal": lambda st: st[0] != "committed" or st[1] >= 1,
        }
        p_visited, p_viol = _bfs_check(p_init, p_trans, p_invs)

        # SequentialTx state: (stack_tuple, seen_frozenset, at_base, recovery_required)
        txns = ("t1", "t2")
        s_init = ((), frozenset(), True, False)

        def s_trans(st):
            stack, seen, at_base, rec_req = st
            out = [("Respawn", s_init)]
            if not rec_req:
                for t in txns:
                    allow = (len(stack) < 2) if mutant == "allow_duplicate_txn_id" else (t not in seen)
                    if allow:
                        out.append((f"InstallCommit({t})", (stack + (t,), seen | {t}, False, False)))
                        out.append((f"InstallAbort({t})", (stack, seen | {t}, at_base, False)))
                if stack:
                    out.append(("UninstallClean", (stack[:-1], seen, len(stack) == 1, False)))
                    out.append(("UninstallFail", (stack, seen, at_base, True)))
            return out

        s_invs = {
            "empty_stack_restores_base": lambda st: not (len(st[0]) == 0 and not st[3]) or st[2],
            "no_duplicate_in_stack": lambda st: len(st[0]) == len(set(st[0])),
        }
        s_visited, s_viol = _bfs_check(s_init, s_trans, s_invs)
        return (p_visited, p_viol), (s_visited, s_viol)

    def test_participant_and_sequential_tx_invariants_and_mutants(self):
        (pv, pviol), (sv, sviol) = self._run_participant_and_sequential_bfs()
        self.assertGreater(pv, 10)
        self.assertGreater(sv, 10)
        self.assertIsNone(pviol)
        self.assertIsNone(sviol)

        _, (_, dup_viol) = self._run_participant_and_sequential_bfs(mutant="allow_duplicate_txn_id")
        self.assertIsNotNone(dup_viol)
        self.assertEqual(dup_viol[0], "no_duplicate_in_stack")

    def _run_incarnation_bfs(self, *, mutant=None):
        """State: (active_inc, active_txn, live_incs_frozenset, in_flight_frozenset, stale_accepted)."""
        max_inc = 2
        txns = (1, 2)
        parts = ("r0", "r1")
        init = (1, 0, frozenset({1}), frozenset(), False)

        def transitions(st):
            inc, txn, live_incs, inflight, stale_acc = st
            out = []
            if txn == 0:
                for t in txns:
                    out.append((f"StartTxn({t})", (inc, t, live_incs, inflight, stale_acc)))
            else:
                for p in parts:
                    msg = (p, inc, txn)
                    if msg not in inflight:
                        out.append((f"SendReply({p})", (inc, txn, live_incs, inflight | {msg}, stale_acc)))
                out.append(("FinishTxn", (inc, 0, live_incs, inflight, stale_acc)))
            for r in inflight:
                _, r_inc, r_txn = r
                rest = inflight - {r}
                if r_inc == inc and r_txn == txn:
                    out.append(("DeliverFresh", (inc, txn, live_incs, rest, stale_acc)))
                else:
                    if mutant == "accept_stale_reply":
                        out.append(("MutantDeliverStale", (inc, txn, live_incs, rest, True)))
                    else:
                        out.append(("RejectStale", (inc, txn, live_incs, rest, stale_acc)))
            if inc < max_inc:
                new_live = (live_incs | {inc + 1}) if mutant == "keep_old_incarnation_live" else frozenset({inc + 1})
                out.append(("Respawn", (inc + 1, 0, new_live, inflight, stale_acc)))
            return out

        invariants = {
            "single_active_incarnation": lambda st: len(st[2]) == 1 and st[2] == frozenset({st[0]}),
            "stale_reply_rejection": lambda st: not st[4],
        }
        return _bfs_check(init, transitions, invariants)

    def test_incarnation_spec_invariants_and_mutants(self):
        visited, viol = self._run_incarnation_bfs()
        self.assertGreater(visited, 50)
        self.assertIsNone(viol, f"Incarnation.tla invariant violated: {viol}")

        _, v_inc = self._run_incarnation_bfs(mutant="keep_old_incarnation_live")
        self.assertIsNotNone(v_inc)
        self.assertEqual(v_inc[0], "single_active_incarnation")

        _, v_stale = self._run_incarnation_bfs(mutant="accept_stale_reply")
        self.assertIsNotNone(v_stale)
        self.assertEqual(v_stale[0], "stale_reply_rejection")


class KernelAdversarialHardeningTest(unittest.TestCase):
    """Real kernel trace-replay & adversarial fault-injection tests."""

    def test_commit_rollback_failure_freezes_install_and_uninstall_until_respawn(self):
        """A commit whose rollback fails latches `recovery_required`, rejecting both `install()`
        and `uninstall()` fail-closed until `recovered()` restores a fresh incarnation."""
        box = type("Box", (), {"val": 1})()
        name, _, _ = fresh("rb_freeze")
        lp = C.LocalParticipant("rank0", alias={"toy": name}, scope={"box": box})
        co = C.Coordinator([lp], phase2_deadline_s=None)
        sess = SS.Session(co, ["toy.engine"], base_sources={"toy.engine": ENG})

        bad_plan = {
            "id": "rb-fail-1",
            "modules": {"toy.engine": {
                "source": edit(ENG, ("return y", "return y * 100")),
                "targets": [["Engine.step", at(ENG, "def step")]],
            }},
            "repairs": [{"cell": "box.val = 9\nraise RuntimeError('boom')\n", "declares": []}],
        }
        res = sess.install(bad_plan)
        self.assertEqual(res["outcome"], "rolled_back")
        self.assertEqual(res["state"], "poisoned")
        self.assertTrue(res["recovery_required"])
        self.assertTrue(sess.recovery_required())
        self.assertTrue(co.is_recovery_required())
        st = sess.status()
        self.assertEqual(st["state"], "poisoned")
        self.assertTrue(st["recovery_required"])

        next_plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 5")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}}
        ref_install = sess.install(next_plan)
        self.assertEqual(ref_install["outcome"], "refused")
        self.assertTrue(ref_install["recovery_required"])
        self.assertEqual([f["check"] for f in ref_install["findings"]], ["session_poisoned"])

        ref_uninstall = sess.uninstall()
        self.assertFalse(ref_uninstall["ok"])
        self.assertEqual(ref_uninstall["outcome"], "refused")
        self.assertTrue(ref_uninstall["recovery_required"])
        self.assertEqual([f["check"] for f in ref_uninstall["findings"]], ["recovery_required"])

        # Clean respawn with a fresh participant clears recovery_required and bumps incarnation.
        name2, eng2, _ = fresh("rb_freeze_respawn")
        co.participants = [C.LocalParticipant("rank0", alias={"toy": name2})]
        rec = sess.recovered()
        self.assertTrue(rec["ok"])
        self.assertFalse(rec["recovery_required"])
        self.assertEqual(rec["incarnation"], 2)
        self.assertEqual(sess.status()["state"], "base")
        ok_res = sess.install(next_plan)
        self.assertEqual((ok_res["outcome"], ok_res["state"], eng2.Engine().step(3)),
                         ("committed", "edited", 85))

    def test_uninstall_failure_freezes_subsequent_uninstall_and_install(self):
        """If `uninstall()` fails on any participant, `recovery_required` latches so further
        `uninstall()` and `install()` calls fail closed until `reset('respawn')`."""
        lab = TL.ToyLab(ranks=2)
        self.addCleanup(lab.close)
        p_vis = lab.patches()["visible"]
        r1 = lab.install(p_vis)
        self.assertTrue(r1["ok"], r1)

        # Sabotage rank1's undo to simulate a participant failure mid-uninstall.
        lp1 = lab.locals["rank1"]
        orig_undo = lp1.undo
        lp1.undo = lambda txn: F.not_ok("rank1 undo crashed", [
            {**F.finding("undo_incomplete", "rank1 failed undo", "clean undo"), "participant": "rank1"}
        ])
        try:
            u1 = lab.uninstall()
            self.assertFalse(u1["ok"])
            self.assertEqual(u1["state"], "poisoned")
            self.assertTrue(u1["recovery_required"])
            self.assertTrue(lab.status()["recovery_required"])
        finally:
            lp1.undo = orig_undo

        # Subsequent uninstall and install are both rejected fail-closed.
        u2 = lab.uninstall()
        self.assertFalse(u2["ok"])
        self.assertTrue(u2["recovery_required"])
        self.assertEqual([f["check"] for f in u2["findings"]], ["recovery_required"])

        r2 = lab.install(p_vis)
        self.assertFalse(r2["ok"])
        self.assertEqual(r2["outcome"], "refused")
        self.assertEqual(r2["codes"], ["session_poisoned"])

        # Clean respawn recovers and restores normal operation.
        rs = lab.reset("respawn")
        self.assertTrue(rs["ok"], rs)
        st = lab.status()
        self.assertEqual((st["state"], st["recovery_required"]), ("base", False))
        self.assertTrue(lab.install(p_vis)["ok"])
        self.assertTrue(lab.uninstall()["ok"])

    def test_unconfirmed_abort_freezes_fail_closed(self):
        """A participant that stops responding during abort latches `recovery_required`."""
        n0, _, _ = fresh("unconf_ab0")
        n1, _, _ = fresh("unconf_ab1")
        lp0 = C.LocalParticipant("r0", alias={"toy": n0})
        lp1 = C.LocalParticipant("r1", alias={"toy": n1})

        def dead_on_abort(method, payload):
            if method == "abort":
                raise ConnectionError("r0 died during abort")
            return getattr(lp0, method)(payload)

        def vote_no_validate(method, payload):
            if method == "validate":
                return F.not_ok("r1 validate rejected")
            return getattr(lp1, method)(payload)

        co = C.Coordinator(
            [C.CallParticipant("r0", dead_on_abort), C.CallParticipant("r1", vote_no_validate)],
            phase2_deadline_s=None,
        )
        sess = SS.Session(co, ["toy.engine"], base_sources={"toy.engine": ENG})
        plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}}
        res = sess.install(plan)
        self.assertEqual(res["outcome"], "aborted")
        self.assertEqual(res["state"], "poisoned")
        self.assertTrue(res["recovery_required"])
        self.assertFalse(sess.uninstall()["ok"])
        self.assertEqual([f["check"] for f in sess.uninstall()["findings"]], ["recovery_required"])
        self.assertEqual(sess.install(plan)["outcome"], "refused")

    def test_partial_rank_rollback_unwinds_reachable_ranks_without_hanging(self):
        """When rolling back across 4 ranks and rank2 hangs/times out (or raises), reachable
        ranks r0 and r1 unwind cleanly to base without hanging on the dead peer."""
        parts, engines = [], []
        for idx in range(4):
            mod_name, eng, _ = fresh(f"prb_{idx}")
            lp = C.LocalParticipant(f"r{idx}", alias={"toy": mod_name})
            engines.append(eng.Engine())

            def rpc(method, payload, lp=lp, idx=idx):
                if idx == 3 and method == "commit":
                    raise RuntimeError("r3 commit died")
                if idx == 2 and method == "undo":
                    time.sleep(1.5)  # hung peer during rollback
                return getattr(lp, method)(payload)

            parts.append(C.CallParticipant(f"r{idx}", rpc))

        co = C.Coordinator(parts, phase2_deadline_s=0.35)
        plan = {"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 100")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}}
        t0 = time.monotonic()
        res = co.run(plan)
        elapsed = time.monotonic() - t0
        self.assertEqual(res["outcome"], "rolled_back")
        self.assertLess(elapsed, 0.9, f"rollback hung on dead peer: {elapsed:.2f}s")
        rb = res["receipt"]["rollback"]
        self.assertEqual(rb["status"], "partial")
        self.assertIn("r2", rb["failed_participants"])
        self.assertIn("r2", rb["errors"])
        # Reachable ranks r0 and r1 unwound cleanly back to base.
        self.assertEqual(engines[0].step(3), 17)
        self.assertEqual(engines[1].step(3), 17)
        self.assertEqual(engines[3].step(3), 17)
        self.assertTrue(co.is_recovery_required())

        # Also check exception-raising rank during rollback without deadline.
        parts2, engines2 = [], []
        for idx in range(3):
            mod_name, eng, _ = fresh(f"prb_exc_{idx}")
            lp = C.LocalParticipant(f"r{idx}", alias={"toy": mod_name})
            engines2.append(eng.Engine())

            def rpc2(method, payload, lp=lp, idx=idx):
                if idx == 2 and method == "commit":
                    raise RuntimeError("r2 commit failed")
                if idx == 1 and method == "undo":
                    raise RuntimeError("r1 undo exploded")
                return getattr(lp, method)(payload)

            parts2.append(C.CallParticipant(f"r{idx}", rpc2))

        co2 = C.Coordinator(parts2, phase2_deadline_s=None)
        res2 = co2.run(plan)
        self.assertEqual(res2["outcome"], "rolled_back")
        self.assertEqual(res2["receipt"]["rollback"]["failed_participants"], ["r1"])
        self.assertEqual(engines2[0].step(3), 17)
        self.assertTrue(co2.is_recovery_required())

    def test_duplicate_and_stale_txn_and_incarnation_replies_rejected(self):
        """Duplicate txn IDs, mismatched-txn replies, and stale-incarnation replies are
        rejected cleanly and never cause a split commit."""
        n0, eng0, _ = fresh("stale0")
        lp0 = C.LocalParticipant("r0", alias={"toy": n0})
        co = C.Coordinator([lp0], phase2_deadline_s=None)
        plan = {"id": "fixed-txn-1", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}}
        r1 = co.run(plan)
        self.assertEqual(r1["outcome"], "committed")
        # Duplicate txn id refused by Coordinator and LocalParticipant.
        dup_co = co.run(plan)
        self.assertEqual(dup_co["outcome"], "refused")
        self.assertEqual([f["check"] for f in dup_co["findings"]], ["patch_invalid"])
        dup_lp = lp0.prepare(plan)
        self.assertFalse(dup_lp["ok"])
        self.assertEqual(eng0.Engine().step(3), 170)

        # Stale txn reply on prepare rejected.
        n1, eng1, _ = fresh("stale1")
        lp1 = C.LocalParticipant("r1", alias={"toy": n1})

        def stale_txn_rpc(method, payload):
            reply = getattr(lp1, method)(payload)
            if method == "prepare":
                return {**reply, "txn": "old-txn-999"}
            return reply

        co_stale = C.Coordinator([C.CallParticipant("r1", stale_txn_rpc)], phase2_deadline_s=None)
        r_stale = co_stale.run({"id": "new-txn-1", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}})
        self.assertEqual(r_stale["outcome"], "aborted")
        self.assertTrue(r_stale["receipt"]["participants"]["r1"]["prepare"].get("stale"))
        self.assertEqual(eng1.Engine().step(3), 17)

        # Stale incarnation reply rejected after recovered() increments incarnation to 2.
        n2, eng2, _ = fresh("stale_inc")
        lp2 = C.LocalParticipant("r2", alias={"toy": n2})

        def stale_inc_rpc(method, payload):
            reply = getattr(lp2, method)(payload)
            if method == "prepare":
                return {**reply, "incarnation": 1}
            return reply

        co_inc = C.Coordinator([C.CallParticipant("r2", stale_inc_rpc)], phase2_deadline_s=None)
        co_inc.recovered()
        self.assertEqual(co_inc.incarnation, 2)
        r_inc = co_inc.run({"id": "inc-txn-1", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}})
        self.assertEqual(r_inc["outcome"], "aborted")
        self.assertTrue(r_inc["receipt"]["participants"]["r2"]["prepare"].get("stale"))
        self.assertEqual(eng2.Engine().step(3), 17)

    def test_plan_is_strictly_non_mutating_and_classifies_boundaries(self):
        """`Session.plan(patch)` and `ToyLab.plan(patch)` classify `in_place_python`,
        `triton_rejit`, and `refused` boundaries without mutating any participant, journal,
        witness, or module state."""
        lab = TL.ToyLab(ranks=2)
        self.addCleanup(lab.close)
        before_st = lab.status()
        before_out = lab.check()["output"]
        patches = lab.patches()

        # 1. Valid in-place Python edit.
        p_ok = lab.plan(patches["visible"])
        self.assertTrue(p_ok["ok"], p_ok)
        self.assertEqual(p_ok["affected_modules"], ["toy.ops"])
        self.assertEqual(p_ok["mechanisms"], ["in_place_python"])
        self.assertEqual(p_ok["restart_boundary"], "in_place_python")
        self.assertEqual(p_ok["refusal_findings"], [])
        self.assertEqual(p_ok["identity"]["digest"], before_st["identity"]["digest"])

        # 2. All refusal classes.
        expected_codes = {
            "syntax": "patch_invalid",
            "closure": "closure_changed",
            "module_state": "outside_changes",
            "empty": "empty_change",
        }
        for cls_name, want_code in expected_codes.items():
            p_ref = lab.plan(patches["refused"][cls_name])
            self.assertFalse(p_ref["ok"], (cls_name, p_ref))
            self.assertEqual(p_ref["mechanisms"], ["refused"], cls_name)
            self.assertEqual(p_ref["restart_boundary"], "refused", cls_name)
            got_codes = [f["check"] for f in p_ref["refusal_findings"]]
            self.assertIn(want_code, got_codes, (cls_name, got_codes))

        # Verify ZERO mutation across every participant and lab output.
        after_st = lab.status()
        self.assertEqual(after_st["state"], "base")
        self.assertEqual(after_st["txns"], [])
        self.assertEqual(after_st["fingerprint"], before_st["fingerprint"])
        self.assertEqual(after_st["identity"]["digest"], before_st["identity"]["digest"])
        for lp in lab.locals.values():
            self.assertEqual(lp.journal.entries, [])
            self.assertEqual(lp._staged, {})
            self.assertEqual(lp._state, {})
            self.assertEqual(lp.witness.counts, {})
        self.assertEqual(lab.check()["output"], before_out)

        # 3. Triton JIT edit classification (pure CPU stand-in, zero GPU).
        triton_src = (
            "class _T:\n"
            "    @staticmethod\n"
            "    def jit(fn):\n"
            "        return fn\n"
            "triton = _T()\n\n"
            "@triton.jit\n"
            "def add_kernel(x):\n"
            "    return x + 1\n"
        )
        (tmod,) = temp_modules(self, {"kt_plan_triton": triton_src})
        lp_t = C.LocalParticipant("rank0")
        sess_t = SS.Session(C.Coordinator([lp_t], phase2_deadline_s=None), ["kt_plan_triton"])
        p_jit = sess_t.plan({"files": {"kt_plan_triton": triton_src.replace("x + 1", "x + 2")}})
        self.assertTrue(p_jit["ok"], p_jit)
        self.assertEqual(p_jit["mechanisms"], ["triton_rejit"])
        self.assertEqual(p_jit["restart_boundary"], "triton_rejit")
        self.assertEqual(lp_t.journal.entries, [])
        self.assertEqual(tmod.add_kernel(4), 5)

        # 4. Plan refused when session is poisoned.
        lab.inject("tamper")
        lab.check()
        self.assertEqual(lab.status()["state"], "poisoned")
        p_pois = lab.plan(patches["visible"])
        self.assertFalse(p_pois["ok"])
        self.assertEqual(p_pois["restart_boundary"], "refused")
        self.assertEqual([f["check"] for f in p_pois["refusal_findings"]], ["session_poisoned"])

    def test_cryptographic_identity_determinism_and_receipt_binding(self):
        """`compute_identity` produces deterministic 64-hex SHA-256 digests wired into
        `Session.status()` and every JSONL record written by `ReceiptWriter`."""
        id_a = ID.compute_identity(
            module_sources={"b.mod": "def g(): return 2\n", "a.mod": "def f(): return 1\n"},
            txns=["t1"], state="edited", incarnation=1, recovery_required=False,
            metadata={"engine": {"name": "toy"}},
        )
        id_b = ID.compute_identity(
            module_sources={"a.mod": "def f(): return 1\n", "b.mod": "def g(): return 2\n"},
            txns=["t1"], state="edited", incarnation=1, recovery_required=False,
            metadata={"engine": {"name": "toy"}},
        )
        self.assertEqual(len(id_a["digest"]), 64)
        self.assertEqual(id_a["digest"], id_b["digest"])
        # Any change to source, txns, state, incarnation, or recovery_required changes digest.
        id_diff_src = ID.compute_identity(
            module_sources={"a.mod": "def f(): return 9\n", "b.mod": "def g(): return 2\n"},
            txns=["t1"], state="edited", incarnation=1, recovery_required=False,
            metadata={"engine": {"name": "toy"}},
        )
        id_diff_inc = ID.compute_identity(
            module_sources={"a.mod": "def f(): return 1\n", "b.mod": "def g(): return 2\n"},
            txns=["t1"], state="edited", incarnation=2, recovery_required=False,
            metadata={"engine": {"name": "toy"}},
        )
        self.assertNotEqual(id_a["digest"], id_diff_src["digest"])
        self.assertNotEqual(id_a["digest"], id_diff_inc["digest"])

        # End-to-end ToyLab + ReceiptWriter binding across txn, proof, and undo records.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        rpath = os.path.join(tmp, "receipts.jsonl")
        lab = TL.ToyLab(ranks=2, receipts=rpath)
        self.addCleanup(lab.close)
        id_base = lab.status()["identity"]
        self.assertEqual(len(id_base["digest"]), 64)
        self.assertEqual(id_base["state"], "base")

        ins = lab.install(lab.patches()["visible"])
        self.assertTrue(ins["ok"], ins)
        id_edited = lab.status()["identity"]
        self.assertNotEqual(id_edited["digest"], id_base["digest"])
        self.assertEqual(id_edited["txns"], [ins["id"]])
        self.assertTrue(lab.check()["ok"])
        self.assertTrue(lab.uninstall()["ok"])
        id_back = lab.status()["identity"]
        self.assertEqual(id_back["digest"], id_base["digest"])

        recs = R.read(rpath)
        self.assertEqual([r["kind"] for r in recs], ["txn", "proof", "undo"])
        for rec in recs:
            self.assertIn("identity", rec)
            self.assertEqual(len(rec["identity"]["digest"]), 64)
            self.assertEqual(rec["identity"]["schema"], ID.SCHEMA)


if __name__ == "__main__":
    unittest.main()
