"""Golden unit & concurrency tests for `PoolAutoScaler` (`core/livelab/rsibox/autoscaler.py`)."""
from __future__ import annotations

import threading
import time
import unittest
from typing import Any

from kernel import toy_lab
from rsibox.autoscaler import AutoScalerPolicy, PoolAutoScaler
from rsibox.pool import PoolSlot, SandboxPool


class PoolAutoScalerTests(unittest.TestCase):
    def _make_toy_lab(self) -> toy_lab.ToyLab:
        lab = toy_lab.ToyLab(ranks=2)
        self.addCleanup(lab.close)
        return lab

    @staticmethod
    def _wait_for_waiters(pool: SandboxPool, count: int, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with pool._lock:
                if sum(1 for w in pool._waiters if not w.cancelled) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"timed out waiting for {count} waiter(s)")

    def test_policy_and_constructor_validation(self):
        """Invalid policy bounds, bool/NaN/Inf values, and bad constructor args fail closed."""
        with self.assertRaises(ValueError):
            AutoScalerPolicy(min_slots=3, max_slots=2)
        with self.assertRaises(ValueError):
            AutoScalerPolicy(min_slots=-1)
        with self.assertRaises(ValueError):
            AutoScalerPolicy(max_slots=0)
        with self.assertRaises(ValueError):
            AutoScalerPolicy(role="   ")
        with self.assertRaises(ValueError):
            AutoScalerPolicy(scale_up_queue_depth=0)
        with self.assertRaises(ValueError):
            AutoScalerPolicy(heal_backoff_s=10.0, max_heal_backoff_s=5.0)
        with self.assertRaises(ValueError):
            AutoScalerPolicy(poll_interval_s=0.0)
        with self.assertRaises(ValueError):
            AutoScalerPolicy(scale_down_idle_s=float("nan"))
        with self.assertRaises(ValueError):
            AutoScalerPolicy(scale_up_cooldown_s=float("inf"))
        with self.assertRaises(TypeError):
            AutoScalerPolicy(min_slots=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            AutoScalerPolicy(auto_heal=1)  # type: ignore[arg-type]

        pool = SandboxPool([PoolSlot("s0", self._make_toy_lab())])
        self.addCleanup(pool.close)
        with self.assertRaises(TypeError):
            PoolAutoScaler(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            PoolAutoScaler(pool, slot_factory="bad")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            PoolAutoScaler(pool, clock="not-callable")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PoolAutoScaler(pool, slot_id_prefix="   ")

    def test_scale_up_on_backlog_hysteresis_cooldown_and_idle_scale_down(self):
        """Golden case: queue backlog >= scale_up_queue_depth scales pool 1 -> 2 -> 3 respecting
        scale_up_cooldown_s and max_slots, then scales down auto-provisioned idle slots cleanly
        after scale_down_idle_s while preserving min_slots and closing drained labs."""
        now = [100.0]
        closed_labs: list[str] = []

        def factory(slot_id: str, role: str | None) -> PoolSlot:
            lab = self._make_toy_lab()
            orig_close = lab.close
            lab.close = lambda: (closed_labs.append(slot_id), orig_close())[1]  # type: ignore[method-assign]
            return PoolSlot(slot_id, lab, role=role)

        pool = SandboxPool([PoolSlot("static-0", self._make_toy_lab())], clock=lambda: now[0])
        self.addCleanup(pool.close)
        scaler = PoolAutoScaler(
            pool,
            factory,
            policy=AutoScalerPolicy(
                min_slots=1,
                max_slots=3,
                scale_up_queue_depth=2,
                scale_up_cooldown_s=5.0,
                scale_down_idle_s=10.0,
                scale_down_cooldown_s=5.0,
                close_on_drain=True,
            ),
            clock=lambda: now[0],
            slot_id_prefix="dyn",
        )

        # Idle at min_slots=1 -> noop
        ev0 = scaler.reconcile_once()
        self.assertEqual(ev0["action"], "noop")

        # Hold static-0 and enqueue 2 waiters -> backlog=2 triggers scale_up 1 -> 2 (dyn-1)
        h0 = pool.acquire("holder")
        w_done: list[str] = []
        hold_w1, rel_w1 = threading.Event(), threading.Event()
        self.addCleanup(rel_w1.set)

        def run_w1() -> None:
            with pool.lease("w1", timeout_s=5.0) as lease:
                w_done.append(f"w1:{lease.slot_id}")
                hold_w1.set()
                rel_w1.wait(timeout=5.0)

        def run_w2() -> None:
            with pool.lease("w2", timeout_s=5.0) as lease:
                w_done.append(f"w2:{lease.slot_id}")

        t1 = threading.Thread(target=run_w1)
        t2 = threading.Thread(target=run_w2)
        t1.start()
        self._wait_for_waiters(pool, 1)
        t2.start()
        self._wait_for_waiters(pool, 2)

        ev_up1 = scaler.reconcile_once()
        self.assertEqual(ev_up1["action"], "scale_up")
        self.assertEqual(ev_up1["slot_id"], "dyn-1")
        self.assertEqual(ev_up1["slots"], 2)
        # dyn-1 immediately receives w1!
        self.assertTrue(hold_w1.wait(timeout=5.0))
        self.assertIn("w1:dyn-1", w_done)

        # Enqueue another waiter so backlog=2 again (w2 + w3), but cooldown=5.0s blocks scale-up at t=102!
        t3 = threading.Thread(target=lambda: pool.run_isolated("w3", lambda _l, s: w_done.append(f"w3:{s.slot_id}"), timeout_s=5.0))
        t3.start()
        self._wait_for_waiters(pool, 2)
        now[0] = 102.0
        self.assertEqual(scaler.reconcile_once()["action"], "noop")

        # Advance past scale_up_cooldown_s (t=105.0) -> scales 2 -> 3 (dyn-2)
        now[0] = 105.0
        ev_up2 = scaler.reconcile_once()
        self.assertEqual(ev_up2["action"], "scale_up")
        self.assertEqual(ev_up2["slot_id"], "dyn-2")
        self.assertEqual(ev_up2["slots"], 3)

        # Release all active work so all 3 slots go idle
        rel_w1.set()
        h0.release()
        for t in (t1, t2, t3):
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive())
        self.assertEqual(len(w_done), 3)

        # First idle tick starts idle timers (t=110.0) -> no scale-down yet
        now[0] = 110.0
        self.assertEqual(scaler.reconcile_once()["action"], "noop")
        # Before scale_down_idle_s (t=119.0) -> still noop
        now[0] = 119.0
        self.assertEqual(scaler.reconcile_once()["action"], "noop")
        # At t=120.0 (10.0s continuous idle) -> prefers draining auto-provisioned slot dyn-1 (NOT static-0!)
        now[0] = 120.0
        ev_dn1 = scaler.reconcile_once()
        self.assertEqual(ev_dn1["action"], "scale_down")
        self.assertIn(ev_dn1["slot_id"], ("dyn-1", "dyn-2"))
        self.assertEqual(ev_dn1["slots"], 2)
        self.assertIn(ev_dn1["slot_id"], closed_labs)

        # Scale-down cooldown blocks at t=122.0
        now[0] = 122.0
        self.assertEqual(scaler.reconcile_once()["action"], "noop")
        # At t=125.0 -> drains second dyn slot, leaving static-0 (min_slots=1) untouched
        now[0] = 125.0
        ev_dn2 = scaler.reconcile_once()
        self.assertEqual(ev_dn2["action"], "scale_down")
        self.assertEqual(ev_dn2["slots"], 1)
        self.assertEqual(sorted(closed_labs), ["dyn-1", "dyn-2"])
        # Never scales below min_slots=1 even after long idle
        now[0] = 200.0
        self.assertEqual(scaler.reconcile_once()["action"], "noop")
        st = scaler.status()
        self.assertEqual((st["scale_ups"], st["scale_downs"], st["errors"]), (2, 2, 0))

    def test_auto_heal_with_exponential_backoff_and_dead_slot_replacement(self):
        """Golden case: transient respawn failures back off exponentially before succeeding,
        and persistently dead slots are drained and replaced when replace_on_heal_exhausted=True."""
        now = [0.0]
        lab0 = self._make_toy_lab()
        orig_reset = lab0.reset
        fail_remaining = [2]  # Fail first 2 respawn attempts (1 initial quarantine + 1 heal), succeed on 2nd heal!

        def _flaky_reset(kind: str = "state") -> dict[str, Any]:
            if kind == "respawn" and fail_remaining[0] > 0:
                fail_remaining[0] -= 1
                return {"ok": False, "error": "transient-gms-err"}
            return orig_reset(kind)

        lab0.reset = _flaky_reset  # type: ignore[method-assign]
        pool = SandboxPool([PoolSlot("s0", lab0)], clock=lambda: now[0])
        self.addCleanup(pool.close)

        def _poison(plab, _l):
            plab.session.fault("participant_failed", "boom", "clean")
            raise RuntimeError("boom")

        pool.run_isolated("rogue", _poison, timeout_s=2.0)
        self.assertEqual(pool.metrics()["quarantined_slots"], ["s0"])

        scaler = PoolAutoScaler(
            pool,
            policy=AutoScalerPolicy(
                min_slots=1,
                max_slots=2,
                auto_heal=True,
                heal_backoff_s=5.0,
                max_heal_backoff_s=20.0,
                max_heal_attempts=3,
            ),
            clock=lambda: now[0],
        )

        # Attempt 1 at t=0.0 fails -> schedules backoff until t=5.0
        ev1 = scaler.reconcile_once()
        self.assertEqual(ev1["action"], "heal_failed")
        self.assertEqual(ev1["attempts"], 1)
        # t=4.0: backoff skips retry cleanly
        now[0] = 4.0
        self.assertEqual(scaler.reconcile_once()["action"], "noop")
        # t=5.0: Attempt 2 succeeds -> s0 restored online!
        now[0] = 5.0
        ev2 = scaler.reconcile_once()
        self.assertEqual(ev2["action"], "heal")
        self.assertTrue(ev2["ok"])
        self.assertEqual(pool.metrics()["quarantined_slots"], [])
        self.assertEqual(scaler.status()["heals_succeeded"], 1)

        # Persistently dead slot with replace_on_heal_exhausted=True drains + replaces!
        dead_lab = self._make_toy_lab()
        orig_dead_reset = dead_lab.reset
        dead_closed = [0]
        orig_dead_close = dead_lab.close
        dead_lab.close = lambda: (dead_closed.__setitem__(0, dead_closed[0] + 1), orig_dead_close())[1]  # type: ignore[method-assign]
        dead_lab.reset = lambda kind="state": {"ok": False, "error": "perm-dead"} if kind == "respawn" else orig_dead_reset(kind)  # type: ignore[method-assign]

        pool2 = SandboxPool([PoolSlot("dead-0", dead_lab)], clock=lambda: now[0])
        self.addCleanup(pool2.close)
        pool2.run_isolated("rogue", _poison, timeout_s=2.0)
        self.assertEqual(pool2.metrics()["quarantined_slots"], ["dead-0"])

        scaler2 = PoolAutoScaler(
            pool2,
            lambda sid, role: PoolSlot(sid, self._make_toy_lab(), role=role),
            policy=AutoScalerPolicy(
                min_slots=1,
                max_slots=2,
                auto_heal=True,
                heal_backoff_s=2.0,
                max_heal_attempts=2,
                replace_on_heal_exhausted=True,
                close_on_drain=True,
            ),
            clock=lambda: now[0],
            slot_id_prefix="repl",
        )
        now[0] = 10.0
        self.assertEqual(scaler2.reconcile_once()["action"], "heal_failed")
        now[0] = 12.0
        ev_rep = scaler2.reconcile_once()
        self.assertEqual(ev_rep["action"], "replace")
        self.assertEqual(ev_rep["replaced_slot_id"], "dead-0")
        self.assertEqual(ev_rep["new_slot_id"], "repl-1")
        self.assertEqual(dead_closed[0], 1)
        self.assertEqual(scaler2.status()["replacements"], 1)
        # Replacement slot serves verified work immediately!
        ok_res = pool2.run_isolated("healthy", lambda plab, _l: plab.check(), timeout_s=2.0)
        self.assertTrue(ok_res["ok"] and ok_res["slot_id"] == "repl-1", ok_res)

    def test_role_isolation_background_loop_and_zero_leak_on_stop_during_provision(self):
        """Golden case: role-scoped autoscaler ignores other roles' waitlists, background
        context manager auto-heals and exits cleanly on pool.close(), and stopping while
        slot_factory is provisioning closes the late lab cleanly without leaking."""
        pool = SandboxPool([
            PoolSlot("dev-0", self._make_toy_lab(), role="dev"),
            PoolSlot("eval-0", self._make_toy_lab(), role="eval"),
        ])
        self.addCleanup(pool.close)

        # Hold dev-0 and enqueue 2 dev-only waiters -> eval-scoped autoscaler must NOT scale up!
        h_dev = pool.acquire("d-hold", role="dev")
        t_d1 = threading.Thread(target=lambda: pool.run_isolated("d1", lambda _l, _s: None, role="dev", timeout_s=5.0))
        t_d2 = threading.Thread(target=lambda: pool.run_isolated("d2", lambda _l, _s: None, role="dev", timeout_s=5.0))
        t_d1.start()
        self._wait_for_waiters(pool, 1)
        t_d2.start()
        self._wait_for_waiters(pool, 2)

        eval_scaler = PoolAutoScaler(
            pool,
            lambda sid, role: PoolSlot(sid, self._make_toy_lab(), role=role),
            policy=AutoScalerPolicy(role="eval", min_slots=1, max_slots=3, scale_up_queue_depth=2),
        )
        self.assertEqual(eval_scaler.reconcile_once()["action"], "noop")
        self.assertEqual(eval_scaler.status()["scale_ups"], 0)
        h_dev.release()
        t_d1.join(timeout=5.0)
        t_d2.join(timeout=5.0)

        # Zero-leak abort when stop() is called while slot_factory is mid-provision!
        prov_entered, prov_allow = threading.Event(), threading.Event()
        self.addCleanup(prov_allow.set)
        leaked_closed = [0]

        def slow_factory(sid: str, role: str | None) -> PoolSlot:
            lab = self._make_toy_lab()
            orig_c = lab.close
            lab.close = lambda: (leaked_closed.__setitem__(0, leaked_closed[0] + 1), orig_c())[1]  # type: ignore[method-assign]
            prov_entered.set()
            prov_allow.wait(timeout=5.0)
            return PoolSlot(sid, lab, role=role)

        # Deficit pool: 1 slot present, min_slots=2 -> triggers provision immediately on start()
        p_def = SandboxPool([PoolSlot("base-0", self._make_toy_lab())])
        self.addCleanup(p_def.close)
        bg = PoolAutoScaler(
            p_def,
            slow_factory,
            policy=AutoScalerPolicy(min_slots=2, max_slots=3, poll_interval_s=0.02),
        )
        bg.start()
        self.assertTrue(bg.status()["running"])
        self.assertTrue(prov_entered.wait(timeout=5.0))
        stop_t = threading.Thread(target=bg.stop)
        stop_t.start()
        time.sleep(0.02)
        prov_allow.set()
        stop_t.join(timeout=5.0)
        self.assertFalse(bg.status()["running"])
        self.assertEqual(leaked_closed[0], 1)
        self.assertEqual(bg.status()["scale_ups"], 0)
        self.assertEqual(len(p_def.metrics()["per_slot_utilization"]), 1)

        # Context manager exits cleanly when pool closes!
        p_cm = SandboxPool([PoolSlot("c0", self._make_toy_lab())])
        with PoolAutoScaler(p_cm, policy=AutoScalerPolicy(min_slots=1, max_slots=1, poll_interval_s=0.02)) as cm_s:
            self.assertTrue(cm_s.status()["running"])
            p_cm.close()
            deadline = time.monotonic() + 2.0
            while cm_s.status()["running"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(cm_s.status()["running"])
        cm_s.stop()  # idempotent

    def test_concurrency_races_deadlock_guard_clock_skew_and_capped_waiters(self):
        """Regression suite for P0/P1/P2 concurrency & boundary hazards:
        1. queue_depth_fn touching pool.metrics() never deadlocks on pool._lock.
        2. Concurrent manual add_slot() during slot_factory enforces hard max_slots cap & closes late lab.
        3. Static slot named 'auto-1' is NOT evicted ahead of auto-provisioned 'auto-2'.
        4. max_slots_per_agent capped waiters do NOT trigger runaway scale-up or block scale-down.
        5. Backward clock jump & extreme 1e308 timestamps never wedge heal backoff or idle timers.
        6. Duplicate-lab slot_factory rejection does NOT close the live pool slot's lab.
        7. Concurrent drain/remove during heal_slot skips cleanly without error & heals sibling slot.
        8. Duck-typed slot_states() pool, heal_exhausted, below_min_slots, and scale_down_skipped paths.
        """
        # 1. P0-1: queue_depth_fn calling pool.metrics() must NOT deadlock!
        p_dl = SandboxPool([PoolSlot("s0", self._make_toy_lab())])
        self.addCleanup(p_dl.close)
        s_dl = PoolAutoScaler(
            p_dl,
            policy=AutoScalerPolicy(min_slots=1, max_slots=2),
            queue_depth_fn=lambda: int(p_dl.metrics().get("total_leases", 0)),
        )
        done_dl = threading.Event()
        res_dl: list[dict[str, Any]] = []
        t_dl = threading.Thread(target=lambda: (res_dl.append(s_dl.reconcile_once()), done_dl.set()))
        t_dl.start()
        self.assertTrue(done_dl.wait(timeout=2.0), "queue_depth_fn calling pool.metrics() deadlocked!")
        t_dl.join(timeout=2.0)
        self.assertEqual(res_dl[0]["action"], "noop")

        # 2. P1-1: Concurrent manual add_slot() during slot_factory enforces max_slots=2 & closes extra lab!
        p_cap = SandboxPool([PoolSlot("s0", self._make_toy_lab())])
        self.addCleanup(p_cap.close)
        entered_cap, allow_cap = threading.Event(), threading.Event()
        self.addCleanup(allow_cap.set)
        aborted_closed = [0]

        def racing_factory(sid: str, role: str | None) -> PoolSlot:
            lab = self._make_toy_lab()
            oc = lab.close
            lab.close = lambda: (aborted_closed.__setitem__(0, aborted_closed[0] + 1), oc())[1]  # type: ignore[method-assign]
            entered_cap.set()
            allow_cap.wait(timeout=5.0)
            return PoolSlot(sid, lab, role=role)

        s_cap = PoolAutoScaler(
            p_cap,
            racing_factory,
            policy=AutoScalerPolicy(min_slots=2, max_slots=2),
        )
        cap_out: list[dict[str, Any]] = []
        t_cap = threading.Thread(target=lambda: cap_out.append(s_cap.reconcile_once()))
        t_cap.start()
        self.assertTrue(entered_cap.wait(timeout=5.0))
        p_cap.add_slot(PoolSlot("manual-1", self._make_toy_lab()))
        allow_cap.set()
        t_cap.join(timeout=5.0)
        self.assertEqual(cap_out[0]["action"], "scale_up_aborted")
        self.assertEqual(cap_out[0]["reason"], "max_slots_reached")
        self.assertEqual(aborted_closed[0], 1)
        self.assertEqual(len(p_cap.metrics()["per_slot_utilization"]), 2)

        # 3. P2-1: Static slot 'auto-1' is preserved while auto-provisioned 'auto-2' is drained first!
        now_own = [0.0]
        p_own = SandboxPool([PoolSlot("auto-1", self._make_toy_lab())], clock=lambda: now_own[0])
        self.addCleanup(p_own.close)
        s_own = PoolAutoScaler(
            p_own,
            lambda sid, role: PoolSlot(sid, self._make_toy_lab(), role=role),
            policy=AutoScalerPolicy(
                min_slots=1, max_slots=2, scale_up_queue_depth=1,
                scale_up_cooldown_s=0.0, scale_down_idle_s=5.0, scale_down_cooldown_s=0.0,
            ),
            clock=lambda: now_own[0],
            slot_id_prefix="auto",
            queue_depth_fn=lambda: own_q[0],
        )
        own_q = [1]
        up_own = s_own.reconcile_once()
        self.assertEqual(up_own["action"], "scale_up")
        self.assertEqual(up_own["slot_id"], "auto-2")
        own_q[0] = 0
        # Start idle timer at t=10 for both auto-1 and auto-2, then drain at t=15 -> MUST drain auto-2!
        now_own[0] = 10.0
        self.assertEqual(s_own.reconcile_once()["action"], "noop")
        now_own[0] = 15.0
        dn_own = s_own.reconcile_once()
        self.assertEqual(dn_own["action"], "scale_down")
        self.assertEqual(dn_own["slot_id"], "auto-2")
        self.assertEqual(list(p_own.metrics()["per_slot_utilization"].keys()), ["auto-1"])

        # 4. P1-4: max_slots_per_agent=1 capped waiters must NOT trigger scale-up!
        p_wc = SandboxPool(
            [PoolSlot("s0", self._make_toy_lab()), PoolSlot("s1", self._make_toy_lab())],
            max_slots_per_agent=1,
        )
        self.addCleanup(p_wc.close)
        h_greedy = p_wc.acquire("greedy")
        w_t1 = threading.Thread(target=lambda: p_wc.run_isolated("greedy", lambda _l, _s: None, timeout_s=5.0))
        w_t2 = threading.Thread(target=lambda: p_wc.run_isolated("greedy", lambda _l, _s: None, timeout_s=5.0))
        w_t1.start()
        self._wait_for_waiters(p_wc, 1)
        w_t2.start()
        self._wait_for_waiters(p_wc, 2)
        s_wc = PoolAutoScaler(
            p_wc,
            lambda sid, role: PoolSlot(sid, self._make_toy_lab(), role=role),
            policy=AutoScalerPolicy(min_slots=1, max_slots=4, scale_up_queue_depth=1),
        )
        ev_wc = s_wc.reconcile_once()
        self.assertEqual(ev_wc["action"], "noop")
        self.assertEqual(ev_wc["queue_depth"], 0)
        self.assertEqual(s_wc.status()["scale_ups"], 0)
        h_greedy.release()
        w_t1.join(timeout=5.0)
        w_t2.join(timeout=5.0)

        # 5. P1-3: Backward clock jump & huge float64 (1e308) recover cleanly without wedging backoff!
        now_ck = [1000.0]
        lab_ck = self._make_toy_lab()
        orig_ck_reset = lab_ck.reset
        ck_fail = [2]  # 1 quarantine + 1 heal fail, then succeed!
        lab_ck.reset = lambda kind="state": (  # type: ignore[method-assign]
            (ck_fail.__setitem__(0, ck_fail[0] - 1), {"ok": False, "error": "err"})[1]
            if kind == "respawn" and ck_fail[0] > 0
            else orig_ck_reset(kind)
        )
        p_ck = SandboxPool([PoolSlot("s0", lab_ck)], clock=lambda: now_ck[0])
        self.addCleanup(p_ck.close)
        p_ck.run_isolated("rogue", lambda pl, _l: (pl.session.fault("participant_failed", "b", "c"), 1 / 0), timeout_s=2.0)
        s_ck = PoolAutoScaler(
            p_ck,
            policy=AutoScalerPolicy(min_slots=1, max_slots=1, heal_backoff_s=5.0, max_heal_attempts=3),
            clock=lambda: now_ck[0],
        )
        self.assertEqual(s_ck.reconcile_once()["action"], "heal_failed")
        # Jump clock backwards 1000 -> 10: clamps last_fail_mono to 10; at t=16 (6s > 5s) heals!
        now_ck[0] = 10.0
        self.assertEqual(s_ck.reconcile_once()["action"], "noop")
        now_ck[0] = 16.0
        self.assertEqual(s_ck.reconcile_once()["action"], "heal")

        # 6. P2-2: Duplicate-lab rejection from buggy slot_factory must NOT close live pool lab!
        live_lab = self._make_toy_lab()
        live_closed = [0]
        oc_live = live_lab.close
        live_lab.close = lambda: (live_closed.__setitem__(0, live_closed[0] + 1), oc_live())[1]  # type: ignore[method-assign]
        p_dup = SandboxPool([PoolSlot("s0", live_lab)])
        self.addCleanup(p_dup.close)
        s_dup = PoolAutoScaler(
            p_dup,
            lambda sid, role: PoolSlot(sid, live_lab, role=role),
            policy=AutoScalerPolicy(min_slots=2, max_slots=2),
        )
        ev_dup = s_dup.reconcile_once()
        self.assertEqual(ev_dup["action"], "scale_up_failed")
        self.assertEqual(live_closed[0], 0, "buggy duplicate-lab factory closed live pool lab!")

        # 7. P1-2 + P2-3: Duck-typed slot_states(), concurrent draining skip -> heals sibling,
        #    heal_exhausted (replace=False), and scale_down_skipped on drain timeout!
        class _DuckPool:
            def __init__(self) -> None:
                self._closed = False
                self.healed: list[str] = []
                self.rows = [
                    {"slot_id": "d0", "role": None, "in_use": False, "quarantined": True, "draining": False},
                    {"slot_id": "d1", "role": None, "in_use": False, "quarantined": True, "draining": False},
                ]

            def slot_states(self) -> list[dict[str, Any]]:
                return [dict(r) for r in self.rows]

            def metrics(self) -> dict[str, Any]:
                return {"queue_depth": 0}

            def heal_slot(self, sid: str) -> dict[str, Any]:
                if sid == "d0":
                    return {"ok": False, "healed": False, "status": "draining", "error": "draining"}
                self.healed.append(sid)
                self.rows[1]["quarantined"] = False
                return {"ok": True, "healed": True, "recovery_s": 0.01}

            def drain_slot(self, sid: str, **_kw: Any) -> dict[str, Any]:
                from rsibox.pool import PoolTimeoutError
                raise PoolTimeoutError(f"busy {sid}")

        duck = _DuckPool()
        now_d = [0.0]
        s_duck = PoolAutoScaler(
            duck,  # type: ignore[arg-type]
            policy=AutoScalerPolicy(min_slots=0, max_slots=2, scale_down_idle_s=0.0, scale_down_cooldown_s=0.0),
            clock=lambda: now_d[0],
        )
        ev_d = s_duck.reconcile_once()
        self.assertEqual(ev_d["action"], "heal")
        self.assertEqual(ev_d["slot_id"], "d1")
        self.assertEqual(s_duck.status()["errors"], 0)
        # Now d0 is gone/healthy and d1 is healthy idle -> drain_slot raises PoolTimeoutError -> scale_down_skipped!
        duck.rows = [{"slot_id": "d1", "role": None, "in_use": False, "quarantined": False, "draining": False}]
        ev_skip = s_duck.reconcile_once()
        self.assertEqual(ev_skip["action"], "scale_down_skipped")
        self.assertEqual(ev_skip["slot_id"], "d1")

        # 8. heal_exhausted with replace_on_heal_exhausted=False records exhausted_slots!
        dead_l = self._make_toy_lab()
        orig_dr = dead_l.reset
        dead_l.reset = lambda kind="state": {"ok": False, "error": "dead"} if kind == "respawn" else orig_dr(kind)  # type: ignore[method-assign]
        p_ex = SandboxPool([PoolSlot("x0", dead_l)])
        self.addCleanup(p_ex.close)
        p_ex.run_isolated("rogue", lambda pl, _l: (pl.session.fault("participant_failed", "b", "c"), 1 / 0), timeout_s=2.0)
        s_ex = PoolAutoScaler(
            p_ex,
            policy=AutoScalerPolicy(min_slots=1, max_slots=1, max_heal_attempts=1, replace_on_heal_exhausted=False),
        )
        ev_ex = s_ex.reconcile_once()
        self.assertEqual(ev_ex["action"], "heal_exhausted")
        self.assertEqual(s_ex.status()["exhausted_slots"], ["x0"])
        self.assertEqual(s_ex.reconcile_once()["action"], "noop")


if __name__ == "__main__":
    unittest.main()
