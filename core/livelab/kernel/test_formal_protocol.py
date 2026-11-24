"""Bounded formal protocol verification & adversarial fault-injection tests (CPU, stdlib only).

Covers two layers with zero Java/TLC dependency:
1. Exhaustive bounded BFS state-space checking of the four TLA+ protocol specifications in
   `core/livelab/kernel/formal/` (`Transaction.tla`, `Participant.tla`, `SequentialTx.tla`,
   `Incarnation.tla`), including `TypeOK` and negative-control guard-removal mutants proving
   each invariant catches its targeted protocol violation.
2. Real kernel trace-replay and adversarial fault-injection tests against `Coordinator`,
   `StepGapCoordinator`, `LocalParticipant`, `CallParticipant`, `GroupParticipant`,
   `Session`, and `ToyLab`.

Run from the repository root:
    PYTHONPATH=core/livelab python3 -m unittest kernel.test_formal_protocol -v
"""
from collections import deque
import os
import threading
import time
import unittest

from . import coordinator as C
from . import findings as F
from . import identity as ID
from . import session as SS
from . import toy_lab as TL
from .test_detectors import temp_modules
from .test_locate import ENG, at, edit, fresh


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
            self.assertGreater(len(text), 100, fname)
        with open(os.path.join(formal_dir, "README.md"), encoding="utf-8") as fh:
            readme = fh.read()
        self.assertIn("not", readme.lower())
        self.assertIn("numerical", readme.lower())
        self.assertIn("PYTHONPATH=core/livelab python3 -m unittest kernel.test_formal_protocol", readme)

    def _run_transaction_bfs(self, *, mutant=None):
        """State: (coord_state, p_states_tuple, recovery_required, poisoned)."""
        n_parts = 2
        terminal = {"idle", "committed", "aborted", "rolled_back", "recovery_required"}
        valid_cs = terminal | {"preparing", "validating", "committing", "rolling_back"}
        valid_ps = {"idle", "prepared", "committed", "aborted", "undone", "partial", "dead"}
        init = ("idle", ("idle",) * n_parts, False, False)

        def transitions(state):
            cs, ps, rec_req, pois = state
            out = []
            # StartPrepare: mutant removes the `not rec_req and not pois` guard.
            can_start = (cs in terminal) if mutant == "skip_rollback_freeze" else (
                cs in terminal and not rec_req and not pois
            )
            if can_start:
                out.append(("StartPrepare", ("preparing", ("idle",) * n_parts, rec_req, pois)))
            if cs == "preparing":
                for i in range(n_parts):
                    if ps[i] == "idle":
                        nps = ps[:i] + ("prepared",) + ps[i + 1:]
                        out.append((f"Prepare({i})", (cs, nps, rec_req, pois)))
                if all(p == "prepared" for p in ps):
                    out.append(("PrepareAllYes", ("validating", ps, rec_req, pois)))
                out.append(("PrepareAbortClean", ("aborted", ("aborted",) * n_parts, rec_req, pois)))
            if cs in ("preparing", "validating", "rolling_back"):
                for i in range(n_parts):
                    nps = ps[:i] + ("dead",) + ps[i + 1:]
                    out.append((f"AbortUnconfirmed({i})", ("recovery_required", nps, True, True)))
            if cs == "validating" and all(p == "prepared" for p in ps):
                out.append(("ValidatePass", ("committing", ps, rec_req, pois)))
            if cs == "committing":
                for i in range(n_parts):
                    if ps[i] == "prepared":
                        nps = ps[:i] + ("committed",) + ps[i + 1:]
                        out.append((f"Commit({i})", (cs, nps, rec_req, pois)))
                        fps = ps[:i] + ("aborted",) + ps[i + 1:]
                        out.append((f"CommitFail({i})", ("rolling_back", fps, rec_req, pois)))
                # FinalizeCommit: mutant weakens `\A p : committed` to `\E p : committed`.
                ready = any(p == "committed" for p in ps) if mutant == "allow_split_commit" else (
                    all(p == "committed" for p in ps)
                )
                if ready:
                    out.append(("FinalizeCommit", ("committed", ps, rec_req, pois)))
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
            out.append(("Respawn", init))
            return out

        invariants = {
            "type_ok": lambda s: (
                s[0] in valid_cs and all(p in valid_ps for p in s[1])
                and isinstance(s[2], bool) and isinstance(s[3], bool)
            ),
            "no_split_commit": lambda s: (s[0] != "committed") or all(p == "committed" for p in s[1]),
            "rollback_freeze": lambda s: (not s[2]) or (s[0] == "recovery_required" and s[3]),
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

    def _run_participant_bfs(self, *, mutant=None):
        """State: (p_state, journal_depth, deadline_passed, committed_late, poisoned)."""
        max_depth = 2
        valid_ps = {"idle", "prepared", "committed", "aborted", "undone", "partial"}
        init = ("idle", 0, False, False, False)

        def transitions(state):
            ps, depth, dl, committed_late, pois = state
            out = [("Respawn", init)]
            if ps in ("idle", "committed", "aborted", "undone") and not pois and depth < max_depth:
                out.append(("Prepare", ("prepared", depth, False, committed_late, pois)))
            if ps == "prepared" and not dl:
                out.append(("DeadlineExpire", (ps, depth, True, committed_late, pois)))
            # Commit: mutant removes the `not dl` guard.
            can_commit = (ps == "prepared" and not pois and depth < max_depth) if mutant == "allow_late_commit" else (
                ps == "prepared" and not dl and not pois and depth < max_depth
            )
            if can_commit:
                out.append(("Commit", ("committed", depth + 1, dl, committed_late or dl, pois)))
            if ps == "prepared" and dl:
                out.append(("LateCommitRefused", ("aborted", depth, dl, committed_late, pois)))
            if ps == "prepared":
                out.append(("Abort", ("aborted", depth, dl, committed_late, pois)))
            if ps in ("committed", "undone") and depth > 0:
                next_ps = "idle" if depth == 1 else "undone"
                out.append(("UndoClean", (next_ps, depth - 1, dl, committed_late, pois)))
                out.append(("UndoPartial", ("partial", depth, dl, committed_late, True)))
            return out

        invariants = {
            "type_ok": lambda s: (
                s[0] in valid_ps and 0 <= s[1] <= max_depth
                and isinstance(s[2], bool) and isinstance(s[3], bool) and isinstance(s[4], bool)
            ),
            "partial_implies_poisoned": lambda s: s[0] != "partial" or s[4],
            "no_commit_after_deadline": lambda s: not s[3],
            "committed_has_journal": lambda s: s[0] != "committed" or s[1] >= 1,
        }
        return _bfs_check(init, transitions, invariants)

    def _run_sequential_bfs(self, *, mutant=None):
        """State: (stack_tuple, seen_frozenset, at_base, recovery_required)."""
        txns = ("t1", "t2")
        init = ((), frozenset(), True, False)

        def transitions(state):
            stack, seen, at_base, rec_req = state
            out = [("Respawn", init)]
            if not rec_req:
                for t in txns:
                    # InstallCommit: mutant removes `t not in seen` guard.
                    allow = (len(stack) < 2) if mutant == "allow_duplicate_txn_id" else (t not in seen)
                    if allow:
                        out.append((f"InstallCommit({t})", (stack + (t,), seen | {t}, False, False)))
                        out.append((f"InstallAbort({t})", (stack, seen | {t}, at_base, False)))
                if stack:
                    out.append(("UninstallClean", (stack[:-1], seen, len(stack) == 1, False)))
                    out.append(("UninstallFail", (stack, seen, at_base, True)))
            return out

        invariants = {
            "type_ok": lambda s: (
                s[1].issubset(set(txns)) and isinstance(s[2], bool) and isinstance(s[3], bool)
            ),
            "empty_stack_restores_base": lambda s: not (len(s[0]) == 0 and not s[3]) or s[2],
            "no_duplicate_in_stack": lambda s: len(s[0]) == len(set(s[0])),
        }
        return _bfs_check(init, transitions, invariants)

    def test_participant_and_sequential_tx_invariants_and_mutants(self):
        pv, pviol = self._run_participant_bfs()
        self.assertGreaterEqual(pv, 12)
        self.assertIsNone(pviol, f"Participant.tla violated: {pviol}")

        _, late_viol = self._run_participant_bfs(mutant="allow_late_commit")
        self.assertIsNotNone(late_viol)
        self.assertEqual(late_viol[0], "no_commit_after_deadline")

        sv, sviol = self._run_sequential_bfs()
        self.assertGreater(sv, 10)
        self.assertIsNone(sviol, f"SequentialTx.tla violated: {sviol}")

        _, dup_viol = self._run_sequential_bfs(mutant="allow_duplicate_txn_id")
        self.assertIsNotNone(dup_viol)
        self.assertEqual(dup_viol[0], "no_duplicate_in_stack")

    def _run_incarnation_bfs(self, *, mutant=None):
        """State: (active_inc, active_txn, live_incs, in_flight, accepted_records)."""
        max_inc = 2
        txns = (1, 2)
        parts = ("r0",)
        init = (1, 0, frozenset({1}), frozenset(), frozenset())

        def transitions(state):
            inc, txn, live_incs, inflight, accepted = state
            out = []
            if txn == 0:
                for t in txns:
                    out.append((f"StartTxn({t})", (inc, t, live_incs, inflight, accepted)))
            else:
                for p in parts:
                    msg = (p, inc, txn)
                    if msg not in inflight:
                        out.append((f"SendReply({p})", (inc, txn, live_incs, inflight | {msg}, accepted)))
                out.append(("FinishTxn", (inc, 0, live_incs, inflight, accepted)))
            for reply in inflight:
                p, r_inc, r_txn = reply
                rest = inflight - {reply}
                # DeliverFreshReply: mutant removes `r_inc == inc and r_txn == txn` guard.
                fresh_ok = True if mutant == "accept_stale_reply" else (r_inc == inc and r_txn == txn)
                if fresh_ok:
                    rec = (p, r_inc, r_txn, inc, txn)
                    out.append(("DeliverFreshReply", (inc, txn, live_incs, rest, accepted | {rec})))
                if r_inc != inc or r_txn != txn:
                    out.append(("RejectStaleReply", (inc, txn, live_incs, rest, accepted)))
            if inc < max_inc:
                new_live = (live_incs | {inc + 1}) if mutant == "keep_old_incarnation_live" else frozenset({inc + 1})
                out.append(("Respawn", (inc + 1, 0, new_live, inflight, accepted)))
            return out

        invariants = {
            "type_ok": lambda s: (
                1 <= s[0] <= max_inc and s[1] in (0, *txns)
                and s[2].issubset(set(range(1, max_inc + 1)))
            ),
            "single_active_incarnation": lambda s: len(s[2]) == 1 and s[2] == frozenset({s[0]}),
            "stale_reply_rejection": lambda s: all(
                r_inc == acc_inc and r_txn == acc_txn
                for _, r_inc, r_txn, acc_inc, acc_txn in s[4]
            ),
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
        status = sess.status()
        self.assertEqual(status["state"], "poisoned")
        self.assertTrue(status["recovery_required"])

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

        u2 = lab.uninstall()
        self.assertFalse(u2["ok"])
        self.assertTrue(u2["recovery_required"])
        self.assertEqual([f["check"] for f in u2["findings"]], ["recovery_required"])

        r2 = lab.install(p_vis)
        self.assertFalse(r2["ok"])
        self.assertEqual(r2["outcome"], "refused")
        self.assertEqual(r2["codes"], ["session_poisoned"])

        rs = lab.reset("respawn")
        self.assertTrue(rs["ok"], rs)
        status = lab.status()
        self.assertEqual((status["state"], status["recovery_required"]), ("base", False))
        self.assertTrue(lab.install(p_vis)["ok"])
        self.assertTrue(lab.uninstall()["ok"])

    def test_toylab_respawn_increments_incarnation_and_supersedes_old_txns(self):
        """`ToyLab.reset('respawn')` carries `incarnation` 1 -> 2 across new coordinators,
        changes base `identity['digest']`, and `Coordinator.recovered()` refuses undo of
        superseded prior-incarnation transactions."""
        lab = TL.ToyLab(ranks=2)
        self.addCleanup(lab.close)
        st1 = lab.status()
        self.assertEqual(st1["incarnation"], 1)
        self.assertEqual(st1["identity"]["incarnation"], 1)

        rs = lab.reset("respawn")
        self.assertTrue(rs["ok"], rs)
        st2 = lab.status()
        self.assertEqual(st2["incarnation"], 2)
        self.assertEqual(st2["identity"]["incarnation"], 2)
        self.assertNotEqual(st1["identity"]["digest"], st2["identity"]["digest"])

        # Coordinator.recovered() supersedes prior-incarnation committed txns.
        name, _, _ = fresh("superseded_txn")
        co = C.Coordinator([C.LocalParticipant("r0", alias={"toy": name})], phase2_deadline_s=None)
        committed = co.run({"id": "inc1-txn", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}})
        self.assertEqual(committed["outcome"], "committed")
        co.recovered()
        self.assertEqual(co.incarnation, 2)
        old_undo = co.undo("inc1-txn")
        self.assertEqual((old_undo["ok"], old_undo["status"]), (False, "refused"))
        self.assertIn("superseded", old_undo["error"])

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
                    time.sleep(1.5)
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
        self.assertEqual(engines[0].step(3), 17)
        self.assertEqual(engines[1].step(3), 17)
        self.assertEqual(engines[3].step(3), 17)
        self.assertTrue(co.is_recovery_required())

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

    def test_poll_rollback_toctou_group_mixed_and_no_rpc_in_readonly_queries(self):
        """(a) `poll_rollback()` discards stale results if `recovered()` ran concurrently;
        (b) `poll_rollback()` detects a `"partial"` rank inside a `GroupParticipant` `"mixed"`
        reply and latches `recovery_required`; (c) read-only `is_recovery_required()`,
        `Session.identity()`, and `Session.plan()` emit zero `status` RPCs."""
        # (a) TOCTOU: concurrent recovered() during poll_rollback() must not poison inc 2.
        entered, release = threading.Event(), threading.Event()
        status_calls = []

        def slow_status(method, payload):
            if method == "status":
                status_calls.append(payload)
                entered.set()
                release.wait(5)
                return F.not_ok("dead from inc 1")
            return F.ok(code={"toy.engine:Engine.step:1": ["h", 1]})

        part = C.CallParticipant("r0", slow_status)
        co = C.Coordinator([part], phase2_deadline_s=None)
        co.poison.append(F.finding("deadline_missed", "late", "on time"))
        co.txns["t-old"] = {"state": "aborted", "pending_rollback": [part]}

        poll_res = {}
        th = threading.Thread(target=lambda: poll_res.update(v=co.poll_rollback()), daemon=True)
        th.start()
        self.assertTrue(entered.wait(5))
        co.recovered()
        self.assertEqual(co.incarnation, 2)
        release.set()
        th.join(5)
        self.assertFalse(co.is_recovery_required())
        self.assertEqual(co.poison, [])

        # (b) GroupParticipant returning state='mixed' with rank 1 'partial' latches recovery_required.
        def mixed_group(method, payload):
            if method == "status":
                return C.merge_ranks("status", [
                    {"ok": True, "state": "undone", "rank": 0},
                    {"ok": True, "state": "partial", "rank": 1},
                ])
            return F.ok()

        grp = C.CallParticipant("workers", mixed_group)
        co_g = C.Coordinator([grp], phase2_deadline_s=None)
        co_g.poison.append(F.finding("deadline_missed", "late", "on time"))
        co_g.txns["t-mix"] = {"state": "aborted", "pending_rollback": [grp]}
        self.assertTrue(co_g.poll_rollback())
        self.assertTrue(co_g.is_recovery_required())

        # (c) Read-only queries make ZERO status RPCs.
        status_calls.clear()
        sess = SS.Session(co, ["toy.engine"], base_sources={"toy.engine": ENG})
        _ = co.is_recovery_required()
        _ = sess.recovery_required()
        _ = sess.identity()
        _ = sess.plan({"files": {"toy.engine": edit(ENG, ("return y", "return y + 1"))}})
        self.assertEqual(status_calls, [])

    def test_duplicate_stale_replies_and_real_incarnation_echo(self):
        """Duplicate txn IDs, mismatched-txn replies, and stale-incarnation replies are
        rejected; real `LocalParticipant` echoes `txn` and `incarnation` tags."""
        n0, eng0, _ = fresh("stale0")
        lp0 = C.LocalParticipant("r0", alias={"toy": n0})
        co = C.Coordinator([lp0], phase2_deadline_s=None)
        plan = {"id": "fixed-txn-1", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}}
        r1 = co.run(plan)
        self.assertEqual(r1["outcome"], "committed")
        # Real LocalParticipant echoed txn & incarnation on prepare, validate, and commit.
        p_rec = r1["receipt"]["participants"]["r0"]
        for phase in ("prepare", "validate", "commit"):
            self.assertEqual(p_rec[phase].get("txn"), "fixed-txn-1", phase)
            self.assertEqual(p_rec[phase].get("incarnation"), 1, phase)

        # Duplicate txn id refused by Coordinator and LocalParticipant.
        dup_co = co.run(plan)
        self.assertEqual(dup_co["outcome"], "refused")
        self.assertEqual([f["check"] for f in dup_co["findings"]], ["patch_invalid"])
        self.assertFalse(lp0.prepare(plan)["ok"])
        self.assertEqual(eng0.Engine().step(3), 170)

        # Stale txn reply rejected.
        n1, eng1, _ = fresh("stale1")
        lp1 = C.LocalParticipant("r1", alias={"toy": n1})
        co_stale = C.Coordinator([C.CallParticipant("r1", lambda m, p: (
            {**getattr(lp1, m)(p), "txn": "old-txn-999"} if m == "prepare" else getattr(lp1, m)(p)
        ))], phase2_deadline_s=None)
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
        co_inc = C.Coordinator([C.CallParticipant("r2", lambda m, p: (
            {**getattr(lp2, m)(p), "incarnation": 1} if m == "prepare" else getattr(lp2, m)(p)
        ))], phase2_deadline_s=None)
        co_inc.recovered()
        self.assertEqual(co_inc.incarnation, 2)
        r_inc = co_inc.run({"id": "inc-txn-1", "modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}})
        self.assertEqual(r_inc["outcome"], "aborted")
        self.assertTrue(r_inc["receipt"]["participants"]["r2"]["prepare"].get("stale"))
        self.assertEqual(eng2.Engine().step(3), 17)

    def test_cancelled_undo_never_latches_recovery_required(self):
        """A cancelled undo (gap timed out before starting) preserves `cancelled` and does
        NOT falsely latch `recovery_required`."""
        name, eng, _ = fresh("cancel_undo")
        co = C.StepGapCoordinator([C.LocalParticipant("r0", alias={"toy": name})],
                                  lambda fn: fn(), phase2_deadline_s=None)
        sess = SS.Session(co, ["toy.engine"], base_sources={"toy.engine": ENG})
        ins = sess.install({"modules": {"toy.engine": {
            "source": edit(ENG, ("return y", "return y * 10")),
            "targets": [["Engine.step", at(ENG, "def step")]],
        }}})
        self.assertEqual(ins["outcome"], "committed")

        def timeout_before_start(_fn):
            raise TimeoutError("gap busy")

        co.gap = timeout_before_start
        u = sess.uninstall()
        self.assertFalse(u["ok"])
        self.assertFalse(u["recovery_required"])
        self.assertFalse(sess.recovery_required())
        self.assertFalse(co.is_recovery_required())
        self.assertEqual(eng.Engine().step(3), 170)

    def test_plan_non_mutating_boundaries_malformed_inputs_and_constant_accept_all(self):
        """`Session.plan` and `ToyLab.plan` classify `in_place_python`, `triton_rejit`, and
        `refused`, never raise on malformed inputs, refuse constant-only edits even with
        `accept='all'`, and never mutate any participant, journal, witness, or module state."""
        lab = TL.ToyLab(ranks=2)
        self.addCleanup(lab.close)
        before_st = lab.status()
        before_out = lab.check()["output"]
        patches = lab.patches()

        p_ok = lab.plan(patches["visible"])
        self.assertTrue(p_ok["ok"], p_ok)
        self.assertEqual(p_ok["affected_modules"], ["toy.ops"])
        self.assertEqual(p_ok["mechanisms"], ["in_place_python"])
        self.assertEqual(p_ok["restart_boundary"], "in_place_python")
        self.assertEqual(p_ok["refusal_findings"], [])
        self.assertEqual(p_ok["identity"]["digest"], before_st["identity"]["digest"])

        expected_codes = {
            "syntax": "patch_invalid", "closure": "closure_changed",
            "module_state": "outside_changes", "empty": "empty_change",
        }
        for cls_name, want_code in expected_codes.items():
            p_ref = lab.plan(patches["refused"][cls_name])
            self.assertFalse(p_ref["ok"], (cls_name, p_ref))
            self.assertEqual(p_ref["mechanisms"], ["refused"], cls_name)
            self.assertEqual(p_ref["restart_boundary"], "refused", cls_name)
            self.assertIn(want_code, [f["check"] for f in p_ref["refusal_findings"]], cls_name)

        # Constant-only edit with accept='all' MUST still be refused with empty_change.
        p_const_all = lab.plan({**patches["refused"]["module_state"], "accept": "all"})
        self.assertFalse(p_const_all["ok"], p_const_all)
        self.assertEqual(p_const_all["restart_boundary"], "refused")
        self.assertIn("empty_change", [f["check"] for f in p_const_all["refusal_findings"]])

        # Malformed inputs never raise and return ok=False.
        for bad in (
            "not-a-dict",
            {"modules": "not-a-dict"},
            {"files": 123},
            {"modules": {"toy.ops": {"source": patches["visible"]["files"]["toy.ops"], "targets": "bad"}}},
            {"modules": {"toy.ops": {"source": patches["visible"]["files"]["toy.ops"]}}, "accept": 42},
            {"repairs": "not-a-list"},
            {"repairs": [{"no_cell": 1}]},
        ):
            r_bad = lab.plan(bad)
            self.assertFalse(r_bad["ok"], bad)
            self.assertEqual(r_bad["restart_boundary"], "refused", bad)

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

        # Triton JIT edit classification (pure CPU stand-in, zero GPU).
        triton_src = (
            "class _T:\n    @staticmethod\n    def jit(fn):\n        return fn\n"
            "triton = _T()\n\n@triton.jit\ndef add_kernel(x):\n    return x + 1\n"
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

    def test_cryptographic_identity_determinism_strict_json_and_live_preference(self):
        """`compute_identity` is deterministic, rejects NaN and non-JSON objects, prefers live
        in-memory code hashes over edited on-disk files, and tracks `ToyLab` state transitions."""
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

        with self.assertRaises(ValueError):
            ID.sha256_canonical({"bad": float("nan")})
        with self.assertRaises(TypeError):
            ID.sha256_canonical({"bad": object()})

        # Prefer live in-memory code hashes over edited on-disk __file__.
        (mod,) = temp_modules(self, {"kt_id_disk": "def f():\n    return 1\n"})
        fp = {"r0": {"kt_id_disk:f:1": ["deadbeef12345678", 999]}}
        d_before = ID.module_source_digests(["kt_id_disk"], fingerprint=fp)["kt_id_disk"]
        with open(mod.__file__, "w", encoding="utf-8") as fh:
            fh.write("def f():\n    return 9999\n")
        d_after = ID.module_source_digests(["kt_id_disk"], fingerprint=fp)["kt_id_disk"]
        self.assertEqual(d_before, d_after)

        # ToyLab live status identity tracks install -> uninstall cleanly.
        lab = TL.ToyLab(ranks=2)
        self.addCleanup(lab.close)
        id_base = lab.status()["identity"]
        ins = lab.install(lab.patches()["visible"])
        self.assertTrue(ins["ok"])
        id_ed = lab.status()["identity"]
        self.assertNotEqual(id_ed["digest"], id_base["digest"])
        self.assertEqual(id_ed["txns"], [ins["id"]])
        self.assertTrue(lab.uninstall()["ok"])
        self.assertEqual(lab.status()["identity"]["digest"], id_base["digest"])


if __name__ == "__main__":
    unittest.main()
