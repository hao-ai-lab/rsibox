"""Spec-to-code conformance tests validating live state transitions against the TLA+ models."""
from __future__ import annotations

import pathlib
import sys
import tempfile
import threading
import time
import unittest

from kernel import coordinator as C
from kernel import journal as J
from kernel import receipt as R
from kernel import swap as S
from kernel import toy
from kernel import witness as W
from rsibox.pool import PoolError, PoolSlot, SandboxPool


class _TrackingLab:
    def __init__(self, entered: threading.Event | None = None, release: threading.Event | None = None):
        self.closed = 0
        self._entered = entered
        self._release = release

    def status(self):
        return {"ok": True, "state": "base", "txns": []}

    def check(self, *_args):
        if self._entered is not None:
            self._entered.set()
        if self._release is not None:
            self._release.wait(timeout=5.0)
        return {"ok": True}

    def reset(self, _kind="state"):
        return {"ok": True, "state": "base"}

    def uninstall(self, _txn):
        return {"ok": True, "state": "base"}

    def close(self):
        self.closed += 1


class TlaConformanceTests(unittest.TestCase):
    """Verify real core modules uphold the invariants proved in core/livelab/formal/*.tla."""

    def test_sandbox_pool_tla_undrained_close_and_caller_ownership(self) -> None:
        entered, release = threading.Event(), threading.Event()
        lab = _TrackingLab(entered=entered, release=release)
        pool = SandboxPool([PoolSlot("s0", lab)], drain_timeout_s=0.05, close_labs=True)
        lease = pool.acquire("agent-a")

        in_flight = threading.Thread(target=lease.lab.check)
        in_flight.start()
        self.assertTrue(entered.wait(timeout=2.0))

        # Release times out waiting for the active call -> slot quarantined, close deferred.
        lease.release()
        self.assertEqual(lab.closed, 0)  # NeverCloseLabWhileCallActive
        with self.assertRaises(PoolError):
            pool.heal_slot("s0")  # NeverHealWhileCallActive

        pool.close(close_labs=True)
        self.assertEqual(lab.closed, 0)  # NeverCloseLabWhileCallActive

        release.set()
        in_flight.join(timeout=2.0)
        self.assertEqual(lab.closed, 1)  # DeferredCloseCompletes

        # Caller ownership invariant on aborted heal_slot (PR #256 contract).
        lab2 = _TrackingLab()
        lab2.status = lambda: {"ok": True, "state": "poisoned", "txns": []}  # type: ignore[method-assign]
        lab2.reset = lambda _k="state": {"ok": False, "error": "down"}  # type: ignore[method-assign]
        rep = _TrackingLab()
        rep_entered, rep_release = threading.Event(), threading.Event()
        rep.reset = lambda _k="state": (  # type: ignore[method-assign]
            rep_entered.set(), rep_release.wait(timeout=5.0), {"ok": True, "state": "base"}
        )[2]
        pool2 = SandboxPool([PoolSlot("s1", lab2)], close_labs=False)
        self.addCleanup(pool2.close)
        pool2.run_isolated("a", lambda _l, _ls: {"ok": True})
        h_res: list[dict] = []
        th = threading.Thread(target=lambda: h_res.append(pool2.heal_slot("s1", lab=rep)))
        th.start()
        self.assertTrue(rep_entered.wait(timeout=2.0))
        d_res: list[dict] = []
        td = threading.Thread(target=lambda: d_res.append(pool2.drain_slot("s1", timeout_s=5.0, close_lab=True)))
        td.start()
        deadline = time.monotonic() + 2.0
        while "s1" not in pool2.metrics()["draining_slots"] and time.monotonic() < deadline:
            time.sleep(0.005)
        rep_release.set()
        th.join(timeout=2.0)
        td.join(timeout=2.0)
        self.assertFalse(h_res[0]["healed"])
        self.assertTrue(d_res[0]["drained"])
        self.assertEqual(rep.closed, 0)  # Replacement remains caller-owned and reusable.

    def test_coordinator_tla_indoubt_latches_recovery_and_incarnation_fence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            name = f"tla_coord_{int(time.time() * 1e6)}"
            toy.load_copy(name)
            orig_mod = sys.modules[f"{name}.ops"]
            # Build simple edit on ToyLab participant via Coordinator directly
            gap_entered, gap_allow = threading.Event(), threading.Event()
            gap_threads: list[threading.Thread] = []

            class _SlowUndoParticipant(C.LocalParticipant):
                def undo(self, txn):
                    gap_entered.set()
                    gap_allow.wait(timeout=5.0)
                    return super().undo(txn)

            class _GatedGapCoordinator(C.Coordinator):
                def _in_gap(self, fn):
                    res = fn()
                    return res

            p0 = _SlowUndoParticipant("r0", alias={"toy": name})
            co = _GatedGapCoordinator([p0], writer=R.ReceiptWriter(f"{td}/receipts.jsonl"))
            base_src = pathlib.Path(orig_mod.__file__).read_text(encoding="utf-8")
            new_src = base_src.replace("return x * STEP_SCALE", "return x + 5")
            r = co.run({"id": "t1", "modules": {"toy.ops": {"source": new_src, "targets": "auto"}}})
            self.assertEqual(r["outcome"], "committed", r)

            # Simulate timed-out gap during undo -> InDoubtUndoLatchesRecovery + NoUndoAfterInDoubt
            def _raising_gap(fn):
                t = threading.Thread(target=fn)
                gap_threads.append(t)
                t.start()
                gap_entered.wait(timeout=2.0)
                raise TimeoutError("gap timed out")

            co._in_gap = _raising_gap  # type: ignore[method-assign]
            u1 = co.undo("t1")
            self.assertEqual(u1["status"], "in-doubt")
            self.assertTrue(co.is_recovery_required())  # InDoubtUndoLatchesRecovery

            u2 = co.undo("t1")
            self.assertEqual(u2["status"], "refused")  # NoUndoAfterInDoubt
            self.assertTrue(u2["recovery_required"])

            # RecoverRespawn supersedes t1; finishing late undo thread must NOT overwrite Superseded.
            co.recovered()
            self.assertEqual(co.txns["t1"]["state"], "superseded")
            gap_allow.set()
            for t in gap_threads:
                t.join(timeout=2.0)
            self.assertEqual(co.txns["t1"]["state"], "superseded")  # SupersededNeverOverwritten
            self.assertFalse(co.is_recovery_required())

    def test_swap_lifecycle_tla_failed_apply_restores_all(self) -> None:
        class _FailCodeJournal(J.Journal):
            def set_code(self, fn, code, defaults=J.KEEP, kwdefaults=J.KEEP):
                raise RuntimeError("injected set_code failure")

        with tempfile.TemporaryDirectory() as td:
            mod_file = pathlib.Path(td) / "tla_swap_mod.py"
            base_src = (
                "def outer(x):\n"
                "    def inner(y):\n"
                "        return y + 1\n"
                "    return inner(x)\n"
            )
            patch_src = (
                "def outer(x):\n"
                "    def inner(y):\n"
                "        return y + 2\n"
                "    return inner(x)\n"
                "\n"
                "def added_helper():\n"
                "    return 42\n"
            )
            mod_file.write_text(base_src, encoding="utf-8")
            sys.path.insert(0, td)
            try:
                import tla_swap_mod

                st = S.stage("tla_swap_mod", patch_src)["staged"]
                self.assertTrue(S.validate(st)["ok"])
                nested_before = [
                    c.co_filename for _, code, _, _ in st.swaps for c in code.co_consts if hasattr(c, "co_filename")
                ]

                wit = W.Witness()
                outer_key = next(k for k in st.keys if ":outer:" in k)
                wit.capture_counts[outer_key] = 9
                wit.probe_errors[outer_key] = 3

                res = S.apply(st, _FailCodeJournal(), wit)
                self.assertFalse(res["ok"])
                # FailedApplyRestoresAll: witness untouched, st.swaps unmutated, W.HIT not leaked.
                self.assertEqual(wit.capture_counts.get(outer_key), 9)
                self.assertEqual(wit.probe_errors.get(outer_key), 3)
                self.assertNotIn(W.HIT, tla_swap_mod.__dict__)
                nested_after = [
                    c.co_filename for _, code, _, _ in st.swaps for c in code.co_consts if hasattr(c, "co_filename")
                ]
                self.assertEqual(nested_before, nested_after)
                self.assertEqual(st.adds, [])
            finally:
                if td in sys.path:
                    sys.path.remove(td)
                sys.modules.pop("tla_swap_mod", None)
