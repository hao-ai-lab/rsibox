"""Regression tests for RecaptureCoordinator post-rebuild partial rollback latching."""

from __future__ import annotations

import threading
import unittest

from kernel import coordinator as C
from kernel import findings as F
from rsibox.lab import RecaptureCoordinator


def _make_participant(name: str, *, undo_ok: bool) -> C.CallParticipant:
    def rpc(method: str, _payload):
        if method == "prepare":
            return F.ok(staged={})
        if method == "undo" and not undo_ok:
            return F.not_ok(f"{name} undo failed")
        return F.ok()

    return C.CallParticipant(name, rpc)


class RecaptureRecoveryTest(unittest.TestCase):
    def test_partial_post_rebuild_undo_latches_recovery_and_poison(self) -> None:
        p0 = _make_participant("p0", undo_ok=True)
        p1 = _make_participant("p1", undo_ok=False)
        coord = RecaptureCoordinator(
            [p0, p1],
            lambda fn: fn(),
            rebuild=lambda _txn, _spec: {"ok": False, "error": "recapture failed"},
            phase2_deadline_s=None,
        )
        res = coord.run({"id": "recapture-partial-1", "rebuild": ["recapture"], "modules": {}})
        self.assertEqual(res["outcome"], "rolled_back")
        self.assertEqual(res["receipt"]["rollback"]["status"], "partial")
        self.assertTrue(res["receipt"]["rollback"]["poisoned"])
        self.assertTrue(coord.is_recovery_required())
        self.assertEqual(coord.txns["recapture-partial-1"]["pending_rollback"], [p1])
        self.assertTrue(
            any(
                f.get("check") == "undo_incomplete" and f.get("participant") == "p1"
                for f in coord.poison
            ),
            coord.poison,
        )

    def test_clean_post_rebuild_undo_does_not_latch_recovery(self) -> None:
        p0 = _make_participant("p0", undo_ok=True)
        coord = RecaptureCoordinator(
            [p0],
            lambda fn: fn(),
            rebuild=lambda _txn, _spec: {"ok": False, "error": "recapture failed"},
            phase2_deadline_s=None,
        )
        res = coord.run({"id": "recapture-clean-1", "rebuild": ["recapture"], "modules": {}})
        self.assertEqual(res["outcome"], "rolled_back")
        self.assertEqual(res["receipt"]["rollback"]["status"], "undone")
        self.assertFalse(coord.is_recovery_required())
        self.assertEqual(coord.poison, [])

    def test_late_post_rebuild_undo_after_recovered_does_not_poison_new_incarnation(self) -> None:
        # (a) Concurrent recovered() during undo leaves inc=2 clean and pending_rollback empty.
        undo_entered, undo_release = threading.Event(), threading.Event()

        def slow_fail_undo(method: str, _payload):
            if method == "prepare":
                return F.ok(staged={})
            if method == "undo":
                undo_entered.set()
                undo_release.wait(timeout=5.0)
                return F.not_ok("late undo failed")
            return F.ok()

        p0 = C.CallParticipant("p0", slow_fail_undo)
        coord = RecaptureCoordinator(
            [p0],
            lambda fn: fn(),
            rebuild=lambda _txn, _spec: {"ok": False, "error": "recapture failed"},
            phase2_deadline_s=None,
        )
        run_res: dict = {}
        th = threading.Thread(
            target=lambda: run_res.update(
                coord.run({"id": "recapture-race-undo", "rebuild": ["recapture"], "modules": {}})
            ),
            daemon=True,
        )
        th.start()
        self.assertTrue(undo_entered.wait(timeout=5.0))
        rec = coord.recovered()
        self.assertEqual(rec["incarnation"], 2)
        undo_release.set()
        th.join(timeout=5.0)
        self.assertFalse(run_res["ok"])
        self.assertEqual(run_res["outcome"], "rolled_back")
        self.assertFalse(coord.is_recovery_required())
        self.assertEqual(coord.poison, [])
        self.assertEqual(coord.txns["recapture-race-undo"].get("pending_rollback"), [])
        self.assertEqual(coord.txns["recapture-race-undo"].get("state"), "superseded")

        # (b) Concurrent recovered() during rebuild skips late rollback and returns aborted.
        rb_entered, rb_release = threading.Event(), threading.Event()
        undo_calls: list[str] = []

        def track_rpc(method: str, _payload):
            if method == "prepare":
                return F.ok(staged={})
            if method == "undo":
                undo_calls.append(method)
                return F.not_ok("should not run")
            return F.ok()

        coord2 = RecaptureCoordinator(
            [C.CallParticipant("p0", track_rpc)],
            lambda fn: fn(),
            rebuild=lambda _txn, _spec: (
                rb_entered.set(),
                rb_release.wait(timeout=5.0),
                {"ok": False, "error": "late rebuild fail"},
            )[2],
            phase2_deadline_s=None,
        )
        run_res2: dict = {}
        th2 = threading.Thread(
            target=lambda: run_res2.update(
                coord2.run({"id": "recapture-race-rb", "rebuild": ["recapture"], "modules": {}})
            ),
            daemon=True,
        )
        th2.start()
        self.assertTrue(rb_entered.wait(timeout=5.0))
        self.assertEqual(coord2.recovered()["incarnation"], 2)
        rb_release.set()
        th2.join(timeout=5.0)
        self.assertEqual(undo_calls, [])
        self.assertFalse(run_res2["ok"])
        self.assertEqual(run_res2["outcome"], "aborted")
        self.assertFalse(run_res2["receipt"]["rebuild"]["ok"])
        self.assertFalse(coord2.is_recovery_required())
        self.assertEqual(coord2.poison, [])

    def test_post_commit_reset_timeout_after_recovered_does_not_poison_new_incarnation(self) -> None:
        reset_entered, reset_release = threading.Event(), threading.Event()

        def slow_reset_rpc(method: str, _payload):
            if method == "prepare":
                return F.ok(staged={})
            if method == "reset":
                reset_entered.set()
                reset_release.wait(timeout=5.0)
                return F.ok()
            return F.ok()

        coord = RecaptureCoordinator(
            [C.CallParticipant("p0", slow_reset_rpc)],
            lambda fn: fn(),
            rebuild=lambda _txn, _spec: {"ok": True},
            post_commit_reset="state",
            phase2_deadline_s=0.05,
        )
        th = threading.Thread(
            target=lambda: coord.run(
                {"id": "recapture-reset-race", "rebuild": ["recapture"], "modules": {}}
            ),
            daemon=True,
        )
        th.start()
        self.assertTrue(reset_entered.wait(timeout=5.0))
        self.assertEqual(coord.recovered()["incarnation"], 2)
        th.join(timeout=5.0)
        reset_release.set()
        self.assertEqual(coord.incarnation, 2)
        self.assertEqual(coord.poison, [])
        self.assertFalse(coord.is_recovery_required())

        # Concurrent recovered() during succeeding rebuild skips post_commit_reset completely.
        rb_entered, rb_release = threading.Event(), threading.Event()
        reset_calls: list[str] = []

        def track_reset_rpc(method: str, _payload):
            if method == "prepare":
                return F.ok(staged={})
            if method == "reset":
                reset_calls.append(method)
            return F.ok()

        coord2 = RecaptureCoordinator(
            [C.CallParticipant("p0", track_reset_rpc)],
            lambda fn: fn(),
            rebuild=lambda _txn, _spec: (
                rb_entered.set(),
                rb_release.wait(timeout=5.0),
                {"ok": True},
            )[2],
            post_commit_reset="state",
            phase2_deadline_s=0.05,
        )
        th2 = threading.Thread(
            target=lambda: coord2.run(
                {"id": "recapture-reset-skip", "rebuild": ["recapture"], "modules": {}}
            ),
            daemon=True,
        )
        th2.start()
        self.assertTrue(rb_entered.wait(timeout=5.0))
        self.assertEqual(coord2.recovered()["incarnation"], 2)
        rb_release.set()
        th2.join(timeout=5.0)
        self.assertEqual(reset_calls, [])
        self.assertEqual(coord2.incarnation, 2)
        self.assertEqual(coord2.poison, [])
        self.assertFalse(coord2.is_recovery_required())


if __name__ == "__main__":
    unittest.main()
