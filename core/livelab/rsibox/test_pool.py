"""Golden concurrency, isolation, failover, and telemetry tests for `rsibox.pool`.

Exercises `SandboxPool` end-to-end against real `ToyLab` instances, `BaseLab` with
`StandbyShadowDriver`, and `Box` wrappers on CPU (pure stdlib).

Run from the repository root:
    python3 -m unittest discover -s core/livelab/rsibox -t core/livelab -v
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
import threading
import time
import unittest

from kernel import coordinator as kernel_coordinator
from kernel import toy as kernel_toy
from kernel import toy_lab

from rsibox import (
    BaseLab,
    PoolClosedError,
    PoolSlot,
    PoolTimeoutError,
    SandboxPool,
    edit_once,
    verify_trace,
)
from rsibox.box import Box
from rsibox.warm import DiskStore, StandbyShadowDriver

_SHADOW_SEQUENCE = itertools.count(1)


def _patch_activate_offset(base_ops_source: str, offset: int) -> dict[str, dict[str, str]]:
    """Build a valid `toy.ops` patch changing `activate(x)` to `return x + <offset>`."""
    patched_source = edit_once(base_ops_source, "    return x + 1", f"    return x + {offset}")
    return {"files": {"toy.ops": patched_source}}


def _expected_toy_output(ranks: list[str], offset: int) -> dict[str, list[int]]:
    """Exact output of `ToyLab` workload when `activate(x)` returns `x + offset`."""
    inputs = toy_lab.WORKLOAD["inputs"]
    return {rank: [(value * 2) + offset + 10 for value in inputs] for rank in ranks}


class _ShadowToyLab(BaseLab):
    """Real `BaseLab` backed by `kernel.toy` so `StandbyShadowDriver` runs live on CPU."""

    name = "shadow_toy"

    def __init__(self, *, recovery_driver=None, weight_store=None):
        self._package_name: str | None = None
        self._engine = None
        self.fail_next_check = False
        super().__init__(
            check_timeout=5.0,
            rpc_timeout=5.0,
            weight_store=weight_store or DiskStore(),
            recovery_driver=recovery_driver,
        )

    def _start_engine(self):
        package_name = f"shadow_pool_toy_{next(_SHADOW_SEQUENCE)}"
        kernel_toy.load_copy(package_name)
        self._package_name = package_name
        self.base_text = {
            "toy.ops": kernel_toy.source("ops"),
            "toy.engine": kernel_toy.source("engine"),
        }
        engine = sys.modules[f"{package_name}.engine"].Engine()
        self._engine = engine
        participant = kernel_coordinator.LocalParticipant(
            "rank0",
            alias={"toy": package_name},
            scope={"engine": engine},
            resets={"R2": engine.drain, "R3": engine.cache.clear},
        )
        coordinator = kernel_coordinator.StepGapCoordinator(
            [participant],
            lambda callback: callback(),
            interface_version="1-draft",
            engine={"name": "shadow_toy", "version": "0", "commit": None},
        )
        return coordinator, ("toy.ops", "toy.engine")

    def _stop_engine(self) -> None:
        if self._package_name:
            prefix = self._package_name + "."
            for module_name in list(sys.modules):
                if module_name == self._package_name or module_name.startswith(prefix):
                    sys.modules.pop(module_name, None)
            self._package_name = None
        self._engine = None

    def _run_workload(self, workload=None):
        if self.fail_next_check:
            self.fail_next_check = False
            raise RuntimeError("simulated shadow worker OOM")
        inputs = (workload or toy_lab.WORKLOAD)["inputs"]
        return {"rank0": [self._engine.step(value) for value in inputs]}

    def _reset_state(self):
        self._engine.drain()
        self._engine.cache.clear()
        return {"ok": True, "reset": ["tokens", "cache"]}


class _FakeClock:
    """Thread-safe deterministic monotonic clock for exact metrics verification."""

    def __init__(self, start: float = 100.0) -> None:
        self._lock = threading.Lock()
        self._now = float(start)

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, delta_s: float) -> float:
        with self._lock:
            self._now += float(delta_s)
            return self._now


class SandboxPoolTests(unittest.TestCase):
    def _make_toy_lab(self) -> toy_lab.ToyLab:
        lab = toy_lab.ToyLab(ranks=2, check_timeout=2.0, hang_s=0.5)
        self.addCleanup(lab.close)
        return lab

    def test_eight_concurrent_agents_on_two_toylabs_zero_cross_agent_leakage(self):
        """Pain point: Agent A leaves stacked edits and dirty cache on a warm slot; Agent B
        must never observe Agent A's code or state while 8 threads contend on 2 real ToyLabs."""
        slot_zero_lab = self._make_toy_lab()
        slot_one_lab = self._make_toy_lab()
        base_output = slot_zero_lab.check()["output"]
        self.assertEqual(base_output, _expected_toy_output(slot_zero_lab.ranks, offset=1))

        pool = SandboxPool([
            PoolSlot("slot-0", slot_zero_lab, metadata={"gpu": 0}),
            PoolSlot("slot-1", slot_one_lab, metadata={"gpu": 1}),
        ])
        self.addCleanup(pool.close)

        results: list[dict] = []
        results_lock = threading.Lock()
        start_barrier = threading.Barrier(8)

        def run_agent(agent_index: int) -> None:
            offset = agent_index + 3
            expected_output = _expected_toy_output(slot_zero_lab.ranks, offset=offset)
            start_barrier.wait(timeout=10.0)

            def action(lab, lease):
                initial_status = lab.status()
                self.assertEqual(initial_status["state"], "base", initial_status)
                self.assertEqual(initial_status["txns"], [])
                initial_check = lab.check()
                self.assertEqual(initial_check["output"], base_output)
                for engine in lab.engines.values():
                    self.assertEqual(engine.cache, {})

                # Install first edit (and on even agents stack a second edit) without uninstalling.
                first_patch = _patch_activate_offset(lab.base_text["toy.ops"], offset=offset + 100)
                second_patch = _patch_activate_offset(lab.base_text["toy.ops"], offset=offset)
                if agent_index % 2 == 0:
                    first_install = lab.install(first_patch)
                    self.assertTrue(first_install["ok"], first_install)
                final_install = lab.install(second_patch)
                self.assertTrue(final_install["ok"], final_install)

                for engine in lab.engines.values():
                    engine.cache["leaked_key"] = f"agent-{agent_index}"

                after_check = lab.check()
                self.assertTrue(after_check["ok"], after_check)
                self.assertTrue(after_check["proven"], after_check)
                self.assertNotEqual(after_check["output"], base_output)
                self.assertEqual(after_check["output"], expected_output)
                return {"agent_index": agent_index, "slot_id": lease.slot_id, "output": after_check["output"]}

            outcome = pool.run_isolated(
                f"agent-{agent_index}",
                action,
                idea_id=f"idea-{agent_index}",
                timeout_s=30.0,
            )
            with results_lock:
                results.append(outcome)

        threads = [threading.Thread(target=run_agent, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)
            self.assertFalse(thread.is_alive())

        self.assertEqual(len(results), 8)
        for outcome in results:
            self.assertTrue(outcome["ok"], outcome)
            self.assertEqual(outcome["status_after"], "base")
            self.assertFalse(outcome["recovered"])
        self.assertEqual({outcome["slot_id"] for outcome in results}, {"slot-0", "slot-1"})

        for lab in (slot_zero_lab, slot_one_lab):
            self.assertEqual(lab.status()["state"], "base")
            self.assertEqual(lab.status()["txns"], [])
            self.assertEqual(lab.check()["output"], base_output)
            for engine in lab.engines.values():
                self.assertEqual(engine.cache, {})

    def test_poisoned_slot_auto_respawns_while_healthy_slot_serves_without_stall(self):
        """Pain point: crashing/poisoning slot 0 must trigger automatic `reset("respawn")`
        out of the pool lock so slot 1 keeps serving other agents with zero stall."""
        slot_zero_lab = self._make_toy_lab()
        slot_one_lab = self._make_toy_lab()
        base_output = slot_zero_lab.check()["output"]

        respawn_entered = threading.Event()
        allow_respawn_finish = threading.Event()
        original_reset = slot_zero_lab.reset

        def gated_reset(kind: str = "state"):
            if kind == "respawn":
                respawn_entered.set()
                allow_respawn_finish.wait(timeout=10.0)
            return original_reset(kind)

        slot_zero_lab.reset = gated_reset

        pool = SandboxPool([PoolSlot("slot-0", slot_zero_lab), PoolSlot("slot-1", slot_one_lab)])
        self.addCleanup(pool.close)
        self.addCleanup(allow_respawn_finish.set)

        poison_result_box: dict = {}

        def crash_slot_zero() -> None:
            def crash_action(lab, lease):
                self.assertEqual(lease.slot_id, "slot-0")
                lab.inject("crash")
                failed_check = lab.check()
                self.assertFalse(failed_check["ok"])
                self.assertEqual(lab.status()["state"], "poisoned")
                return "poisoned"

            poison_result_box["outcome"] = pool.run_isolated("crasher", crash_action, timeout_s=10.0)

        crash_thread = threading.Thread(target=crash_slot_zero)
        crash_thread.start()
        self.assertTrue(respawn_entered.wait(timeout=10.0), "slot-0 should enter reset('respawn')")

        # While slot-0 is blocked inside respawn, slot-1 must serve an agent immediately.
        healthy_outcome = pool.run_isolated(
            "healthy-agent",
            lambda lab, lease: (
                lab.install(_patch_activate_offset(lab.base_text["toy.ops"], offset=9)),
                lab.check(),
            )[1],
            timeout_s=5.0,
        )
        self.assertTrue(healthy_outcome["ok"], healthy_outcome)
        self.assertEqual(healthy_outcome["slot_id"], "slot-1")
        self.assertTrue(healthy_outcome["result"]["proven"])
        self.assertEqual(
            healthy_outcome["result"]["output"],
            _expected_toy_output(slot_one_lab.ranks, offset=9),
        )

        # Let slot-0 complete respawn and verify it recovers to clean base.
        allow_respawn_finish.set()
        crash_thread.join(timeout=10.0)
        self.assertFalse(crash_thread.is_alive())

        crash_outcome = poison_result_box["outcome"]
        self.assertTrue(crash_outcome["ok"], crash_outcome)
        self.assertTrue(crash_outcome["recovered"])
        self.assertGreater(crash_outcome["recovery_s"], 0.0)
        self.assertEqual(crash_outcome["status_after"], "base")
        self.assertEqual(slot_zero_lab.status()["state"], "base")
        self.assertEqual(slot_zero_lab.check()["output"], base_output)
        self.assertEqual(pool.metrics()["recoveries"], 1)
        self.assertEqual(pool.metrics()["quarantined_slots"], [])

    def test_standby_shadow_driver_failover_invoked_automatically_on_poison(self):
        """Proves `SandboxPool` transparently invokes `StandbyShadowDriver.respawn()` on a
        `BaseLab` slot when an agent poisons the worker, running promote + replenish cleanly."""
        driver_events: list[str] = []

        def promote(lab, weight_store):
            driver_events.append(f"promote:{weight_store.name}")
            lab._spawn()
            return {"ok": True, "reset": ["shadow promoted", "session"]}

        def replenish(lab, weight_store):
            driver_events.append(f"replenish:{lab.status()['state']}")

        shadow_driver = StandbyShadowDriver(promote_fn=promote, replenish_fn=replenish)
        shadow_lab = _ShadowToyLab(recovery_driver=shadow_driver)
        self.addCleanup(shadow_lab.close)
        base_output = shadow_lab.check()["output"]

        pool = SandboxPool([PoolSlot("shadow-0", shadow_lab, role="eval")])
        self.addCleanup(pool.close)

        def trigger_worker_crash(lab, lease):
            lab.fail_next_check = True
            failed_check = lab.check()
            self.assertFalse(failed_check["ok"])
            self.assertEqual(lab.status()["state"], "poisoned")
            return failed_check

        poison_outcome = pool.run_isolated(
            "eval-agent-1",
            trigger_worker_crash,
            role="eval",
            idea_id="shadow-failover",
        )
        self.assertTrue(poison_outcome["ok"], poison_outcome)
        self.assertTrue(poison_outcome["recovered"])
        self.assertEqual(poison_outcome["status_after"], "base")
        self.assertEqual(driver_events, ["promote:none", "replenish:base"])

        # Follow-up agent uses the freshly promoted shadow slot cleanly at base.
        followup = pool.run_isolated(
            "eval-agent-2",
            lambda lab, lease: (
                lab.install(_patch_activate_offset(lab.base_text["toy.ops"], offset=6)),
                lab.check(),
            )[1],
            role="eval",
        )
        self.assertTrue(followup["ok"], followup)
        self.assertFalse(followup["recovered"])
        self.assertTrue(followup["result"]["proven"])
        self.assertNotEqual(followup["result"]["output"], base_output)
        self.assertEqual(pool.metrics()["recoveries"], 1)

    def test_role_aware_routing_isolates_roles_enforces_fifo_and_times_out_cleanly(self):
        """Pain point: dev/toolsmith/eval role splits must never cross-route, must fail
        cleanly on missing or busy roles, and must serve queued waiters in strict FIFO order."""
        dev_lab = self._make_toy_lab()
        toolsmith_lab = self._make_toy_lab()
        eval_lab = self._make_toy_lab()

        pool = SandboxPool([
            PoolSlot("gpu-dev", dev_lab, role="dev"),
            PoolSlot("gpu-toolsmith", toolsmith_lab, role="toolsmith"),
            PoolSlot("gpu-eval", eval_lab, role="eval"),
        ])
        self.addCleanup(pool.close)

        # Exact role routing across all 3 roles.
        for requested_role, expected_slot in (
            ("dev", "gpu-dev"),
            ("toolsmith", "gpu-toolsmith"),
            ("eval", "gpu-eval"),
        ):
            outcome = pool.run_isolated(
                f"agent-{requested_role}",
                lambda lab, lease: lease.slot_id,
                role=requested_role,
            )
            self.assertTrue(outcome["ok"], outcome)
            self.assertEqual(outcome["slot_id"], expected_slot)

        # Unmatched role fails fast with PoolTimeoutError (a TimeoutError) and ok=False in run_isolated.
        with self.assertRaises(PoolTimeoutError) as error_context:
            pool.acquire("agent-unknown", role="nonexistent", timeout_s=0.1)
        self.assertIsInstance(error_context.exception, TimeoutError)

        missing_role_outcome = pool.run_isolated(
            "agent-unknown", lambda lab, lease: None, role="nonexistent", timeout_s=0.1
        )
        self.assertFalse(missing_role_outcome["ok"])
        self.assertIsNone(missing_role_outcome["slot_id"])
        self.assertIn("no healthy slot", missing_role_outcome["error"])

        # Holding gpu-dev must block a second dev request until timeout even though toolsmith & eval are idle.
        with pool.lease("dev-holder", role="dev") as held_lease:
            self.assertEqual(held_lease.slot_id, "gpu-dev")
            with self.assertRaises(PoolTimeoutError):
                pool.acquire("dev-waiter", role="dev", timeout_s=0.05)
            # Meanwhile role=None ("any") can still acquire an idle non-dev slot.
            with pool.lease("any-agent", role=None, timeout_s=1.0) as any_lease:
                self.assertIn(any_lease.slot_id, {"gpu-toolsmith", "gpu-eval"})

        # Strict FIFO ordering check on single role="eval" slot.
        order_acquired: list[str] = []
        first_lease = pool.acquire("eval-0", role="eval")
        waiter_threads: list[threading.Thread] = []
        for index in range(1, 5):
            def queued_worker(worker_name: str) -> None:
                with pool.lease(worker_name, role="eval", timeout_s=10.0) as acquired:
                    order_acquired.append(worker_name)
                    self.assertEqual(acquired.slot_id, "gpu-eval")

            thread = threading.Thread(target=queued_worker, args=(f"eval-{index}",))
            thread.start()
            waiter_threads.append(thread)
            deadline = time.monotonic() + 2.0
            while len(pool._waiters) < index and time.monotonic() < deadline:
                time.sleep(0.002)
        first_lease.release()
        for thread in waiter_threads:
            thread.join(timeout=10.0)
        self.assertEqual(order_acquired, ["eval-1", "eval-2", "eval-3", "eval-4"])

    def test_tamper_evident_jsonl_trace_and_exact_metrics_calculation(self):
        """Proves atomic hash-chained JSONL telemetry records every required field, detects
        tampering, and computes exact queue wait, p95, hold, recovery, and utilization stats."""
        slot_zero_lab = self._make_toy_lab()
        slot_one_lab = self._make_toy_lab()
        fake_clock = _FakeClock(start=1000.0)
        fake_wall_clock = _FakeClock(start=1700000000.0)

        with tempfile.TemporaryDirectory(prefix="rsibox-pool-trace-") as temp_dir:
            trace_path = os.path.join(temp_dir, "leases.jsonl")
            pool = SandboxPool(
                [PoolSlot("slot-0", slot_zero_lab, role="dev"), PoolSlot("slot-1", slot_one_lab, role="eval")],
                trace_path=trace_path,
                clock=fake_clock,
                wall_clock=fake_wall_clock,
            )
            self.addCleanup(pool.close)

            # Lease 1: hold=2.0s, cleanup=0.5s (slot-0)
            lease_one = pool.acquire("agent-a", role="dev", idea_id="idea-1")
            fake_clock.advance(2.0)
            fake_wall_clock.advance(2.0)
            original_reset_zero = slot_zero_lab.reset

            def timed_reset_zero(kind: str = "state"):
                fake_clock.advance(0.5 if kind == "state" else 1.5)
                return original_reset_zero(kind)

            slot_zero_lab.reset = timed_reset_zero
            record_one = lease_one.release()
            # Idempotent second release returns the same record without double-counting.
            self.assertEqual(lease_one.release(), record_one)

            # Lease 2: poisoned slot-1 -> hold=1.0s, recovery=3.0s
            lease_two = pool.acquire("agent-b", role="eval", idea_id="idea-2")
            slot_one_lab.inject("crash")
            slot_one_lab.check()
            fake_clock.advance(1.0)
            fake_wall_clock.advance(1.0)
            original_reset_one = slot_one_lab.reset

            def timed_reset_one(kind: str = "state"):
                fake_clock.advance(3.0 if kind == "respawn" else 0.2)
                return original_reset_one(kind)

            slot_one_lab.reset = timed_reset_one
            lease_two.release()

            # Lease 3: queued behind lease_three_holder on slot-0 -> exact non-zero queue_wait_s=1.0s
            holder = pool.acquire("agent-c-holder", role="dev", idea_id="idea-3a")
            waiter_outcome_box: dict = {}

            def queued_acquire() -> None:
                waiter_lease = pool.acquire("agent-c-waiter", role="dev", idea_id="idea-3b", timeout_s=10.0)
                fake_clock.advance(1.5)
                fake_wall_clock.advance(1.5)
                waiter_outcome_box["record"] = waiter_lease.release()

            waiter_thread = threading.Thread(target=queued_acquire)
            waiter_thread.start()
            deadline = time.monotonic() + 2.0
            while not pool._waiters and time.monotonic() < deadline:
                time.sleep(0.002)
            # Advance clock 1.0s while agent-c-waiter is queued, then release holder (cleanup adds 0.5s -> queue_wait=1.5s).
            fake_clock.advance(1.0)
            fake_wall_clock.advance(1.0)
            holder.release()
            waiter_thread.join(timeout=5.0)
            self.assertFalse(waiter_thread.is_alive())

            # Idle gap of 1.5s: total elapsed = 2.0+0.5 (L1) + 1.0+3.0 (L2) + 1.0+0.5 (L3a) + 1.5+0.5 (L3b) = 10.0s!
            # Wait: check exact elapsed:
            # L1: hold 2.0 + cleanup 0.5 = 2.5 (slot-0)
            # L2: hold 1.0 + recovery 3.0 = 4.0 (slot-1)
            # L3a: hold 1.0 + cleanup 0.5 = 1.5 (slot-0)
            # L3b: hold 1.5 + cleanup 0.5 = 2.0 (slot-0), queued for 1.0 + 0.5 = 1.5s
            # Total elapsed = 2.5 + 4.0 + 1.5 + 2.0 = 10.0s!
            verified = verify_trace(trace_path)
            self.assertTrue(verified["ok"], verified)
            self.assertEqual(verified["count"], 4)
            required_keys = {
                "agent_id", "role", "idea_id", "slot_id", "acquired_at", "released_at",
                "queue_wait_s", "hold_s", "cleanup_s", "recovery_s", "recovered",
                "error", "status_after", "seq", "prev_sha256", "record_sha256",
            }
            for trace_entry in verified["records"]:
                self.assertTrue(required_keys.issubset(trace_entry.keys()), trace_entry.keys())

            self.assertEqual(verified["records"][0]["agent_id"], "agent-a")
            self.assertEqual(verified["records"][0]["hold_s"], 2.0)
            self.assertEqual(verified["records"][0]["cleanup_s"], 0.5)
            self.assertFalse(verified["records"][0]["recovered"])
            self.assertEqual(verified["records"][1]["agent_id"], "agent-b")
            self.assertEqual(verified["records"][1]["hold_s"], 1.0)
            self.assertEqual(verified["records"][1]["recovery_s"], 3.0)
            self.assertTrue(verified["records"][1]["recovered"])
            self.assertEqual(waiter_outcome_box["record"]["queue_wait_s"], 1.5)

            stats = pool.metrics()
            self.assertEqual(stats["total_leases"], 4)
            # queue_waits = [0.0, 0.0, 0.0, 1.5] -> mean = 0.375, nearest-rank p95 = 1.5
            self.assertEqual(stats["mean_queue_wait_s"], 0.375)
            self.assertEqual(stats["p95_queue_wait_s"], 1.5)
            # hold_times = [2.0, 1.0, 1.0, 1.5] -> mean = 1.375
            self.assertEqual(stats["mean_hold_s"], 1.375)
            self.assertEqual(stats["total_recovery_s"], 3.0)
            self.assertEqual(stats["recoveries"], 1)
            self.assertEqual(stats["quarantined_slots"], [])
            # slot-0 busy = 2.5 + 1.5 + 2.0 = 6.0 / 10.0 = 0.6; slot-1 busy = 4.0 / 10.0 = 0.4
            self.assertEqual(stats["per_slot_utilization"], {"slot-0": 0.6, "slot-1": 0.4})

            # Tampering with a single field in the JSONL trace must be detected.
            with open(trace_path, encoding="utf-8") as handle:
                lines = handle.readlines()
            tampered_first = json.loads(lines[0])
            tampered_first["hold_s"] = 999.0
            lines[0] = json.dumps(tampered_first, sort_keys=True) + "\n"
            with open(trace_path, "w", encoding="utf-8") as handle:
                handle.writelines(lines)
            tamper_check = verify_trace(trace_path)
            self.assertFalse(tamper_check["ok"])
            self.assertEqual(tamper_check["line"], 1)

    def test_quarantine_fails_closed_routes_to_healthy_slot_and_wakes_unservable_waiters(self):
        """Pain point: if recovery fails on slot 0, slot 0 must be quarantined fail-closed,
        subsequent leases must route to healthy slot 1 without hanging, and waiters with zero
        surviving slots must wake with PoolTimeoutError instead of deadlocking."""
        broken_lab = self._make_toy_lab()
        healthy_lab = self._make_toy_lab()
        original_reset = broken_lab.reset

        def failing_respawn(kind: str = "state"):
            if kind == "respawn":
                return {"ok": False, "reset": [], "error": "unrecoverable device fault"}
            return original_reset(kind)

        broken_lab.reset = failing_respawn

        slot_zero = PoolSlot("slot-0", broken_lab)
        slot_one = PoolSlot("slot-1", healthy_lab)
        pool = SandboxPool([slot_zero, slot_one])
        self.addCleanup(pool.close)

        # Agent 1 poisons slot-0; respawn fails -> slot-0 quarantined.
        first_outcome = pool.run_isolated(
            "agent-poison",
            lambda lab, lease: (lab.inject("crash"), lab.check())[1],
            timeout_s=5.0,
        )
        self.assertEqual(first_outcome["slot_id"], "slot-0")
        self.assertEqual(first_outcome["status_after"], "quarantined")
        self.assertFalse(first_outcome["recovered"])
        self.assertIn("unrecoverable device fault", first_outcome["error"])
        self.assertTrue(slot_zero.quarantined)
        self.assertEqual(pool.metrics()["quarantined_slots"], ["slot-0"])

        # Subsequent agents route exclusively to healthy slot-1 without hanging.
        for index in range(4):
            routed = pool.run_isolated(
                f"survivor-{index}",
                lambda lab, lease: lab.check()["ok"],
                timeout_s=5.0,
            )
            self.assertTrue(routed["ok"], routed)
            self.assertTrue(routed["result"])
            self.assertEqual(routed["slot_id"], "slot-1")
            self.assertEqual(routed["status_after"], "base")

        # Now make slot-1's respawn also fail while a waiter is blocked with timeout_s=None:
        # waiter must wake immediately with PoolTimeoutError rather than hanging forever.
        healthy_lab.reset = failing_respawn
        last_lease = pool.acquire("last-holder", timeout_s=5.0)
        self.assertEqual(last_lease.slot_id, "slot-1")
        waiter_error_box: list[BaseException] = []

        def blocked_waiter() -> None:
            try:
                pool.acquire("doomed-waiter", timeout_s=None)
            except BaseException as error:
                waiter_error_box.append(error)

        waiter_thread = threading.Thread(target=blocked_waiter)
        waiter_thread.start()
        deadline = time.monotonic() + 2.0
        while not pool._waiters and time.monotonic() < deadline:
            time.sleep(0.002)

        healthy_lab.inject("crash")
        healthy_lab.check()
        last_lease.release()
        waiter_thread.join(timeout=5.0)
        self.assertFalse(waiter_thread.is_alive())
        self.assertEqual(len(waiter_error_box), 1)
        self.assertIsInstance(waiter_error_box[0], PoolTimeoutError)
        self.assertEqual(pool.metrics()["quarantined_slots"], ["slot-0", "slot-1"])

    def test_pre_acquire_dirty_recovery_box_wrapper_and_exception_rollback(self):
        """Proves (a) PoolSlot wraps a live `Box` seamlessly, (b) pre-dirty slots are cleaned
        defensively before lease handoff, (c) action exceptions roll back cleanly, and (d)
        `close()` wakes waiting threads with `PoolClosedError`."""
        dirty_lab = self._make_toy_lab()
        base_output = dirty_lab.check()["output"]
        dirty_install = dirty_lab.install(_patch_activate_offset(dirty_lab.base_text["toy.ops"], offset=42))
        self.assertTrue(dirty_install["ok"])
        self.assertEqual(dirty_lab.status()["state"], "edited")

        boxed = Box(lambda receipts=None: toy_lab.ToyLab(ranks=2, receipts=receipts), spec="toy")
        boxed.boot()
        self.addCleanup(boxed.close)

        pool = SandboxPool([
            PoolSlot("dirty-slot", dirty_lab, role="dev"),
            PoolSlot("boxed-slot", boxed, role="eval"),
        ])
        self.addCleanup(pool.close)

        # Defensive acquire cleans the pre-edited lab before the agent sees it.
        cleaned_outcome = pool.run_isolated(
            "dev-clean",
            lambda lab, lease: (lab.status()["state"], lab.check()["output"]),
            role="dev",
        )
        self.assertTrue(cleaned_outcome["ok"], cleaned_outcome)
        self.assertEqual(cleaned_outcome["result"], ("base", base_output))

        # Action exception inside run_isolated rolls back installed patch on a Box-backed slot.
        def failing_box_action(lab, lease):
            installed = lab.install(_patch_activate_offset(lab.base_text["toy.ops"], offset=77))
            self.assertTrue(installed["ok"], installed)
            raise ValueError("agent crashed mid-experiment")

        failed_run = pool.run_isolated("eval-fail", failing_box_action, role="eval")
        self.assertFalse(failed_run["ok"])
        self.assertEqual(failed_run["slot_id"], "boxed-slot")
        self.assertEqual(failed_run["status_after"], "base")
        self.assertIn("ValueError: agent crashed mid-experiment", failed_run["error"])

        # Box-backed slot is back at clean base and recorded events.
        self.assertEqual(boxed.call("status")[1]["state"], "base")
        self.assertEqual(boxed.call("check")[1]["output"], base_output)
        self.assertGreater(len(boxed.events), 0)

        # Closing pool wakes blocked waiter with PoolClosedError.
        held = pool.acquire("holder", role="eval")
        closed_errors: list[BaseException] = []
        waiter = threading.Thread(
            target=lambda: self._capture_error(lambda: pool.acquire("waiter", role="eval"), closed_errors)
        )
        waiter.start()
        deadline = time.monotonic() + 2.0
        while not pool._waiters and time.monotonic() < deadline:
            time.sleep(0.002)
        pool.close()
        waiter.join(timeout=5.0)
        held.release()
        self.assertEqual(len(closed_errors), 1)
        self.assertIsInstance(closed_errors[0], PoolClosedError)

    def test_pre_acquire_broken_slot_is_quarantined_and_fails_over_to_healthy_slot(self):
        """Pain point: if slot 0 enters the pool already poisoned with broken respawn,
        `acquire()` must fail-closed quarantine slot 0 and transparently grant slot 1."""
        broken_lab = self._make_toy_lab()
        healthy_lab = self._make_toy_lab()
        broken_lab.inject("crash")
        broken_lab.check()
        self.assertEqual(broken_lab.status()["state"], "poisoned")
        broken_lab.reset = lambda kind="state": {"ok": False, "reset": [], "error": "dead GPU"}

        slot_zero = PoolSlot("broken-0", broken_lab)
        slot_one = PoolSlot("healthy-1", healthy_lab)
        pool = SandboxPool([slot_zero, slot_one])
        self.addCleanup(pool.close)

        outcome = pool.run_isolated("agent-failover", lambda lab, lease: lab.check()["ok"], timeout_s=5.0)
        self.assertTrue(outcome["ok"], outcome)
        self.assertTrue(outcome["result"])
        self.assertEqual(outcome["slot_id"], "healthy-1")
        self.assertTrue(slot_zero.quarantined)
        self.assertEqual(pool.metrics()["quarantined_slots"], ["broken-0"])

    @staticmethod
    def _capture_error(callable_fn, bucket: list[BaseException]) -> None:
        try:
            callable_fn()
        except BaseException as error:
            bucket.append(error)


if __name__ == "__main__":
    unittest.main()
