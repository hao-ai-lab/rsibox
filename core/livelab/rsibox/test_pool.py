"""Golden concurrency, isolation, failover, and telemetry tests for `rsibox.pool`.

Exercises `SandboxPool` end-to-end against real `ToyLab` instances, `BaseLab` with
`StandbyShadowDriver`, and `Box` wrappers on CPU (pure stdlib).

Run from the repository root:
    python3 -m unittest discover -s core/livelab/rsibox -t core/livelab -v
"""
from __future__ import annotations

import contextlib
import itertools
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from kernel import coordinator as kernel_coordinator
from kernel import toy as kernel_toy
from kernel import toy_lab

from rsibox import (
    BaseLab,
    PoolClosedError,
    PoolError,
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

    def _wait_for_waiters(self, pool: SandboxPool, expected_count: int, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        observed = 0
        while time.monotonic() < deadline:
            with pool._lock:
                observed = len(pool._waiters)
            if observed >= expected_count:
                return
            time.sleep(0.002)
        self.fail(f"expected at least {expected_count} waiter(s), observed {observed}")

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
            self._wait_for_waiters(pool, index)
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
            self._wait_for_waiters(pool, 1)
            # Advance clock 1.0s while agent-c-waiter is queued; holder cleanup adds 0.5s -> queue_wait_s=1.5s.
            fake_clock.advance(1.0)
            fake_wall_clock.advance(1.0)
            holder.release()
            waiter_thread.join(timeout=5.0)
            self.assertFalse(waiter_thread.is_alive())

            # Total elapsed = 2.5 (L1) + 4.0 (L2) + 1.5 (L3a) + 2.0 (L3b) = 10.0s.
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
        self._wait_for_waiters(pool, 1)

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
        self._wait_for_waiters(pool, 1)
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

    def test_trace_write_failure_does_not_double_release_or_corrupt_next_owner(self):
        """Regression for P0-1: OSError in `_append_trace()` must still return the slot once,
        must never double-clean or double-count metrics, and must not corrupt the queued owner."""
        slot_lab = self._make_toy_lab()
        reset_state_calls = 0
        original_reset = slot_lab.reset

        def counting_reset(kind: str = "state"):
            nonlocal reset_state_calls
            if kind == "state":
                reset_state_calls += 1
            return original_reset(kind)

        slot_lab.reset = counting_reset

        with tempfile.TemporaryDirectory(prefix="rsibox-trace-dir-") as trace_dir:
            # Pointing trace_path at a directory forces IsADirectoryError on _append_trace().
            pool = SandboxPool([PoolSlot("slot-0", slot_lab)], trace_path=trace_dir)
            self.addCleanup(pool.close)

            second_acquired = threading.Event()
            allow_second_release = threading.Event()
            second_outcome: dict = {}

            def second_agent() -> None:
                second_lease = pool.acquire("agent-2", timeout_s=5.0)
                second_acquired.set()
                try:
                    installed = second_lease.lab.install(
                        _patch_activate_offset(second_lease.lab.base_text["toy.ops"], offset=13)
                    )
                    self.assertTrue(installed["ok"], installed)
                    allow_second_release.wait(timeout=5.0)
                    checked = second_lease.lab.check()
                    second_outcome["proven"] = checked["proven"]
                    second_outcome["output"] = checked["output"]
                finally:
                    with contextlib.suppress(OSError):
                        second_lease.release()

            waiter_thread = threading.Thread(target=second_agent)
            captured_first_lease = None
            with self.assertRaises(OSError):
                with pool.lease("agent-1") as first_lease:
                    captured_first_lease = first_lease
                    first_lease.lab.install(
                        _patch_activate_offset(first_lease.lab.base_text["toy.ops"], offset=5)
                    )
                    waiter_thread.start()
                    self._wait_for_waiters(pool, 1)

            # First acquire cleaned dirty initial state (1) + first release cleaned once (2).
            self.assertEqual(reset_state_calls, 2)
            self.assertTrue(second_acquired.wait(timeout=5.0))
            self.assertEqual(pool.metrics()["total_leases"], 1)

            # Duplicate release on first_lease re-raises the saved OSError without touching slot-0!
            with self.assertRaises(OSError):
                captured_first_lease.release()
            self.assertEqual(reset_state_calls, 2)
            self.assertEqual(pool.metrics()["total_leases"], 1)

            allow_second_release.set()
            waiter_thread.join(timeout=5.0)
            self.assertFalse(waiter_thread.is_alive())
            self.assertTrue(second_outcome["proven"])
            self.assertEqual(second_outcome["output"], _expected_toy_output(slot_lab.ranks, offset=13))
            self.assertEqual(reset_state_calls, 3)
            self.assertEqual(pool.metrics()["total_leases"], 2)

    def test_infinite_and_huge_timeouts_and_base_exception_cleanup_leak_zero_slots(self):
        """Regression for P0-2: inf/huge timeouts never raise OverflowError, NaN raises ValueError,
        and BaseException/clock failure during wait or acquire leaks zero waiters or slots."""
        slot_lab = self._make_toy_lab()
        pool = SandboxPool([PoolSlot("slot-0", slot_lab)])
        self.addCleanup(pool.close)

        with self.assertRaises(ValueError):
            pool.acquire("agent-nan", timeout_s=float("nan"))
        with self.assertRaises(TypeError):
            pool.acquire("agent-bool", timeout_s=True)
        with self.assertRaises(TypeError):
            SandboxPool([PoolSlot("slot-bad-drain", self._make_toy_lab())], drain_timeout_s=False)

        # Duplicate underlying Lab or Box across two PoolSlots in one pool must fail fast.
        shared_lab = self._make_toy_lab()
        with self.assertRaises(ValueError):
            SandboxPool([PoolSlot("dup-0", shared_lab), PoolSlot("dup-1", shared_lab)])
        shared_box = Box(lambda receipts=None: toy_lab.ToyLab(ranks=2, receipts=receipts), spec="toy")
        shared_box.boot()
        self.addCleanup(shared_box.close)
        with self.assertRaises(ValueError):
            SandboxPool([PoolSlot("box-0", shared_box), PoolSlot("box-1", shared_box)])
        with self.assertRaises(ValueError):
            SandboxPool([PoolSlot("box-0", shared_box), PoolSlot("lab-1", shared_box.lab)])

        # Huge and infinite timeouts work cleanly on queued handoff.
        holder = pool.acquire("holder")
        acquired_slots: list[str] = []

        def infinite_waiter() -> None:
            with pool.lease("inf-agent", timeout_s=float("inf")) as leased:
                acquired_slots.append(leased.slot_id)

        def huge_timeout_waiter() -> None:
            with pool.lease("huge-agent", timeout_s=1e20) as leased:
                acquired_slots.append(leased.slot_id)

        thread_one = threading.Thread(target=infinite_waiter)
        thread_two = threading.Thread(target=huge_timeout_waiter)
        thread_one.start()
        self._wait_for_waiters(pool, 1)
        thread_two.start()
        self._wait_for_waiters(pool, 2)
        holder.release()
        thread_one.join(timeout=5.0)
        thread_two.join(timeout=5.0)
        self.assertEqual(acquired_slots, ["slot-0", "slot-0"])

        # Simulate KeyboardInterrupt inside waiter.event.wait() BOTH before and after handoff.
        from rsibox import pool as pool_module

        held = pool.acquire("holder-2")
        real_waiter_cls = pool_module._Waiter

        # Case A: interrupt while still waiting (no slot assigned yet) -> waiter removed, zero leaks.
        def interrupting_waiter_unassigned(role, priority=0):
            waiter = real_waiter_cls(role, priority)
            waiter.event.wait = lambda timeout=None: (_ for _ in ()).throw(
                KeyboardInterrupt("interrupt before handoff")
            )
            return waiter

        with mock.patch.object(pool_module, "_Waiter", side_effect=interrupting_waiter_unassigned):
            with self.assertRaises(KeyboardInterrupt):
                pool.acquire("interrupted-unassigned", timeout_s=5.0)
        with pool._lock:
            self.assertEqual(len(pool._waiters), 0)

        # Case B: interrupt right AFTER holder hands off slot -> assigned slot returned cleanly!
        interrupted_errors: list[BaseException] = []

        def interrupting_waiter_after_handoff(role, priority=0):
            waiter = real_waiter_cls(role, priority)
            real_wait = waiter.event.wait

            def wait_then_interrupt(timeout=None):
                real_wait(timeout=timeout)
                raise KeyboardInterrupt("interrupt after handoff")

            waiter.event.wait = wait_then_interrupt
            return waiter

        with mock.patch.object(pool_module, "_Waiter", side_effect=interrupting_waiter_after_handoff):
            wait_thread = threading.Thread(
                target=lambda: self._capture_error(
                    lambda: pool.acquire("interrupted-after-handoff", timeout_s=5.0),
                    interrupted_errors,
                )
            )
            wait_thread.start()
            self._wait_for_waiters(pool, 1)
            held.release()
            wait_thread.join(timeout=5.0)
            self.assertFalse(wait_thread.is_alive())

        self.assertEqual(len(interrupted_errors), 1)
        self.assertIsInstance(interrupted_errors[0], KeyboardInterrupt)
        with pool._lock:
            self.assertEqual(len(pool._waiters), 0)
        with pool.lease("post-interrupt", timeout_s=1.0) as clean_lease:
            self.assertEqual(clean_lease.slot_id, "slot-0")

        # Failing custom clock right after _claim_slot() must NOT leak slot_state.in_use=True.
        call_counter = [0]

        def flaky_clock() -> float:
            call_counter[0] += 1
            if call_counter[0] == 4:
                raise RuntimeError("clock failed after claim")
            return float(call_counter[0])

        flaky_pool = SandboxPool([PoolSlot("slot-flaky", self._make_toy_lab())], clock=flaky_clock)
        self.addCleanup(flaky_pool.close)
        with self.assertRaises(RuntimeError):
            flaky_pool.acquire("flaky-caller")
        with flaky_pool.lease("recovery-caller", timeout_s=1.0) as recovered_lease:
            self.assertEqual(recovered_lease.slot_id, "slot-flaky")

        # Clock failure at pre-acquire quarantine timestamp must still quarantine the broken slot.
        broken_lab = self._make_toy_lab()
        broken_lab.status = lambda: {"ok": False, "state": "poisoned", "error": "broken"}
        broken_lab.reset = lambda kind="state": {"ok": False, "error": "respawn failed"}
        quarantine_ticks = [0]

        def clock_failing_at_quarantine() -> float:
            quarantine_ticks[0] += 1
            if quarantine_ticks[0] == 9:
                raise RuntimeError("clock failed at quarantine")
            return float(quarantine_ticks[0])

        quarantine_pool = SandboxPool([PoolSlot("broken-clock-0", broken_lab)], clock=clock_failing_at_quarantine)
        self.addCleanup(quarantine_pool.close)
        with self.assertRaises(PoolTimeoutError):
            quarantine_pool.acquire("caller", timeout_s=1.0)
        self.assertEqual(quarantine_pool.metrics()["quarantined_slots"], ["broken-clock-0"])

    def test_close_races_with_handoff_and_is_idempotent_with_close_labs(self):
        """Regression for P1-1: close(close_labs=True) closes every lab at most once, is not
        undone by a later close(close_labs=False), and never grants a lease after close()."""
        idle_lab = self._make_toy_lab()
        busy_lab = self._make_toy_lab()
        close_counts = {"idle": 0, "busy": 0}
        orig_idle_close, orig_busy_close = idle_lab.close, busy_lab.close

        def spy_idle_close():
            close_counts["idle"] += 1
            orig_idle_close()

        def spy_busy_close():
            close_counts["busy"] += 1
            orig_busy_close()

        idle_lab.close = spy_idle_close
        busy_lab.close = spy_busy_close

        pool = SandboxPool(
            [PoolSlot("busy-0", busy_lab), PoolSlot("idle-1", idle_lab)],
            close_labs=False,
        )
        busy_lease = pool.acquire("holder")
        self.assertEqual(busy_lease.slot_id, "busy-0")

        # First close(close_labs=True) closes idle-1 once; second close(close_labs=False) must NOT
        # double-close idle-1 OR prevent busy-0 from closing on release.
        pool.close(close_labs=True)
        pool.close(close_labs=False)
        pool.close(close_labs=True)
        self.assertEqual(close_counts, {"idle": 1, "busy": 0})
        busy_lease.release()
        self.assertEqual(close_counts, {"idle": 1, "busy": 1})

        # Race: close(close_labs=True) called while acquire() is inside pre-acquire status().
        race_lab = self._make_toy_lab()
        race_closes = [0]
        orig_race_close = race_lab.close
        race_lab.close = lambda: (race_closes.__setitem__(0, race_closes[0] + 1), orig_race_close())[1]
        entered_status = threading.Event()
        allow_status_finish = threading.Event()
        orig_status = race_lab.status

        def slow_status():
            if not entered_status.is_set():
                entered_status.set()
                allow_status_finish.wait(timeout=5.0)
            return orig_status()

        race_lab.status = slow_status
        race_pool = SandboxPool([PoolSlot("race-0", race_lab)], close_labs=True)
        race_errors: list[BaseException] = []
        acquire_thread = threading.Thread(
            target=lambda: self._capture_error(lambda: race_pool.acquire("racer"), race_errors)
        )
        acquire_thread.start()
        self.assertTrue(entered_status.wait(timeout=5.0))
        race_pool.close(close_labs=True)
        allow_status_finish.set()
        acquire_thread.join(timeout=5.0)
        self.assertEqual(len(race_errors), 1)
        self.assertIsInstance(race_errors[0], PoolClosedError)
        self.assertEqual(race_closes[0], 1)

    def test_fenced_lease_proxy_blocks_post_release_access_and_first_acquire_cleans_cache(self):
        """Regression for P1-2 / P1-A: initial acquire cleans pre-existing engine cache, stale
        references on `lease.lab` raise `PoolError` after `release()`, `lease.slot` is a read-only
        view without `.lab`/`.box`, and external `PoolSlot` mutation cannot corrupt scheduling."""
        slot_lab = self._make_toy_lab()
        slot_lab.engines["rank0"].cache["preexisting_stale"] = "dirty"
        pool_slot = PoolSlot("slot-0", slot_lab, role="dev", metadata={"gpu": 0})
        pool = SandboxPool([pool_slot])
        self.addCleanup(pool.close)

        first_lease = pool.acquire("agent-1", role="dev")
        self.assertEqual(first_lease.lab.engines["rank0"].cache, {})
        self.assertEqual(first_lease.slot.slot_id, "slot-0")
        self.assertEqual(first_lease.slot.role, "dev")
        self.assertEqual(first_lease.slot.metadata["gpu"], 0)
        self.assertFalse(hasattr(first_lease.slot, "lab"))
        self.assertFalse(hasattr(first_lease.slot, "box"))
        self.assertFalse(hasattr(first_lease.slot, "quarantined"))
        with self.assertRaises(AttributeError):
            first_lease.slot.role = "eval"
        with self.assertRaises(TypeError):
            first_lease.slot.metadata["gpu"] = 99
        for attr_name, bad_value in (
            ("slot_id", "other"),
            ("slot", None),
            ("agent_id", "other"),
            ("role", "eval"),
            ("idea_id", "tampered"),
            ("acquired_at", 0.0),
            ("queue_wait_s", 0.0),
            ("lab", None),
        ):
            with self.assertRaises(AttributeError):
                setattr(first_lease, attr_name, bad_value)
        with self.assertRaises(PoolError):
            first_lease.lab.close()

        stale_lab_ref = first_lease.lab
        stale_check_method = first_lease.lab.check
        first_lease.release()

        # Post-release access via stale lab ref, pre-bound method, or released mutation must fail closed.
        with self.assertRaises(AttributeError):
            first_lease.released = False
        with self.assertRaises(PoolError):
            stale_lab_ref.status()
        with self.assertRaises(PoolError):
            stale_lab_ref.install(_patch_activate_offset(slot_lab.base_text["toy.ops"], offset=8))
        with self.assertRaises(PoolError):
            stale_check_method()
        with self.assertRaises(PoolError):
            _ = stale_lab_ref.base_text
        with self.assertRaises(PoolError):
            stale_lab_ref.custom_flag = True

        # External mutation of original PoolSlot must NOT corrupt role routing or quarantine state.
        pool_slot.role = "tampered"
        pool_slot.quarantined = True
        self.assertEqual(pool.metrics()["quarantined_slots"], [])
        with pool.lease("agent-2", role="dev", timeout_s=1.0) as second_lease:
            self.assertEqual(second_lease.slot.role, "dev")
            self.assertEqual(second_lease.lab.status()["state"], "base")
            self.assertTrue(second_lease.lab.check()["ok"])

        # Conversely, clearing PoolSlot.quarantined externally cannot un-quarantine a failed slot.
        broken_lab = self._make_toy_lab()
        broken_lab.status = lambda: {"ok": False, "state": "poisoned", "error": "fatal"}
        broken_lab.reset = lambda kind="state": {"ok": False, "error": "respawn broken"}
        broken_slot = PoolSlot("broken-0", broken_lab, role="dev")
        broken_pool = SandboxPool([broken_slot])
        self.addCleanup(broken_pool.close)
        with self.assertRaises(PoolTimeoutError):
            broken_pool.acquire("agent-3", role="dev", timeout_s=0.1)
        self.assertEqual(broken_pool.metrics()["quarantined_slots"], ["broken-0"])
        broken_slot.quarantined = False
        self.assertEqual(broken_pool.metrics()["quarantined_slots"], ["broken-0"])
        with self.assertRaises(PoolTimeoutError):
            broken_pool.acquire("agent-4", role="dev", timeout_s=0.05)

    def test_release_drains_in_flight_fenced_call_or_quarantines_on_drain_timeout(self):
        """Regression for P1-B: release() waits for an in-flight fenced call to finish before
        cleanup, and quarantines the slot fail-closed without concurrent cleanup if drain times out."""
        # Case A: in-flight check() completes within drain_timeout_s -> strictly ordered before reset().
        drain_lab = self._make_toy_lab()
        events: list[str] = []
        events_lock = threading.Lock()
        entered_check = threading.Event()
        allow_check_finish = threading.Event()
        self.addCleanup(allow_check_finish.set)
        orig_check, orig_reset = drain_lab.check, drain_lab.reset

        def gated_check(workload=None):
            with events_lock:
                events.append("check_start")
            entered_check.set()
            allow_check_finish.wait(timeout=5.0)
            reply = orig_check(workload)
            with events_lock:
                events.append("check_end")
            return reply

        def tracked_reset(kind="state"):
            with events_lock:
                events.append(f"reset_{kind}")
            return orig_reset(kind)

        drain_lab.check = gated_check
        drain_lab.reset = tracked_reset

        drain_pool = SandboxPool([PoolSlot("drain-0", drain_lab)], drain_timeout_s=5.0)
        self.addCleanup(drain_pool.close)
        leased = drain_pool.acquire("agent-drain")
        with events_lock:
            events.clear()

        background_replies: list[dict] = []
        background_thread = threading.Thread(target=lambda: background_replies.append(leased.lab.check()))
        background_thread.start()
        self.assertTrue(entered_check.wait(timeout=5.0))

        release_records: list[dict] = []
        release_thread = threading.Thread(target=lambda: release_records.append(leased.release()))
        release_thread.start()

        deadline = time.monotonic() + 2.0
        while not leased.released and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(leased.released)
        time.sleep(0.05)
        self.assertTrue(release_thread.is_alive())
        with self.assertRaises(PoolError):
            leased.lab.status()
        with events_lock:
            self.assertNotIn("reset_state", events)

        allow_check_finish.set()
        background_thread.join(timeout=5.0)
        release_thread.join(timeout=5.0)
        self.assertFalse(background_thread.is_alive())
        self.assertFalse(release_thread.is_alive())
        self.assertTrue(background_replies and background_replies[0]["ok"])
        self.assertEqual(release_records[0]["status_after"], "base")
        self.assertLess(events.index("check_end"), events.index("reset_state"))
        self.assertEqual(drain_pool.metrics()["quarantined_slots"], [])
        with drain_pool.lease("agent-next", timeout_s=1.0) as next_lease:
            self.assertEqual(next_lease.slot_id, "drain-0")
            self.assertEqual(next_lease.lab.status()["state"], "base")

        # Case B: hung in-flight check() exceeds short drain_timeout_s -> quarantine fail-closed,
        # ZERO concurrent cleanup on the occupied lab, and healthy slot 1 keeps serving.
        hung_lab = self._make_toy_lab()
        healthy_lab = self._make_toy_lab()
        entered_hang = threading.Event()
        release_hang = threading.Event()
        self.addCleanup(release_hang.set)
        hung_resets = [0]
        orig_hung_check, orig_hung_reset = hung_lab.check, hung_lab.reset

        def hung_check(workload=None):
            entered_hang.set()
            release_hang.wait(timeout=5.0)
            return orig_hung_check(workload)

        def counted_hung_reset(kind="state"):
            hung_resets[0] += 1
            return orig_hung_reset(kind)

        hung_lab.check = hung_check
        hung_lab.reset = counted_hung_reset

        hang_pool = SandboxPool(
            [PoolSlot("slot-0", hung_lab, role="eval"), PoolSlot("slot-1", healthy_lab, role="dev")],
            drain_timeout_s=0.05,
        )
        self.addCleanup(hang_pool.close)
        hung_lease = hang_pool.acquire("agent-hung", role="eval")
        resets_before_release = hung_resets[0]
        hung_thread = threading.Thread(target=lambda: hung_lease.lab.check())
        hung_thread.start()
        self.assertTrue(entered_hang.wait(timeout=5.0))

        # Queue a waiter specifically for role="eval"; undrained quarantine must wake it promptly.
        waiter_errors: list[BaseException] = []
        eval_waiter = threading.Thread(
            target=lambda: self._capture_error(
                lambda: hang_pool.acquire("agent-eval-waiter", role="eval", timeout_s=5.0),
                waiter_errors,
            )
        )
        eval_waiter.start()
        self._wait_for_waiters(hang_pool, 1)

        release_record = hung_lease.release()
        self.assertEqual(release_record["status_after"], "quarantined")
        self.assertIn("in-flight fenced call did not drain", release_record["error"])
        self.assertEqual(hung_resets[0], resets_before_release)
        self.assertEqual(hang_pool.metrics()["quarantined_slots"], ["slot-0"])

        eval_waiter.join(timeout=2.0)
        self.assertFalse(eval_waiter.is_alive())
        self.assertEqual(len(waiter_errors), 1)
        self.assertIsInstance(waiter_errors[0], PoolTimeoutError)

        # Healthy slot-1 continues serving immediately with zero stall.
        healthy_outcome = hang_pool.run_isolated("agent-healthy", lambda lab, lease: lab.check()["ok"])
        self.assertTrue(healthy_outcome["ok"], healthy_outcome)
        self.assertEqual(healthy_outcome["slot_id"], "slot-1")
        release_hang.set()
        hung_thread.join(timeout=5.0)
        self.assertFalse(hung_thread.is_alive())

        # Case C (P1-2): KeyboardInterrupt during drain wait must quarantine fail-closed,
        # never leak in_use=True, wake unservable waiters, and re-raise on duplicate release().
        interrupt_lab = self._make_toy_lab()
        entered_interrupt = threading.Event()
        release_interrupt = threading.Event()
        self.addCleanup(release_interrupt.set)
        interrupt_resets = [0]
        orig_interrupt_check, orig_interrupt_reset = interrupt_lab.check, interrupt_lab.reset

        def interrupt_check(workload=None):
            entered_interrupt.set()
            release_interrupt.wait(timeout=5.0)
            return orig_interrupt_check(workload)

        def counted_interrupt_reset(kind="state"):
            interrupt_resets[0] += 1
            return orig_interrupt_reset(kind)

        interrupt_lab.check = interrupt_check
        interrupt_lab.reset = counted_interrupt_reset

        interrupt_pool = SandboxPool([PoolSlot("interrupt-0", interrupt_lab)], drain_timeout_s=5.0)
        self.addCleanup(interrupt_pool.close)
        interrupt_lease = interrupt_pool.acquire("agent-interrupt")
        resets_before_interrupt = interrupt_resets[0]
        active_thread = threading.Thread(target=lambda: interrupt_lease.lab.check())
        active_thread.start()
        self.assertTrue(entered_interrupt.wait(timeout=5.0))

        blocked_errors: list[BaseException] = []
        blocked_waiter = threading.Thread(
            target=lambda: self._capture_error(
                lambda: interrupt_pool.acquire("agent-waiter", timeout_s=5.0),
                blocked_errors,
            )
        )
        blocked_waiter.start()
        self._wait_for_waiters(interrupt_pool, 1)

        with mock.patch.object(
            interrupt_lease._call_cond,
            "wait",
            side_effect=KeyboardInterrupt("interrupted during drain"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                interrupt_lease.release()

        with interrupt_pool._lock:
            slot_state = interrupt_pool._slots["interrupt-0"]
            self.assertFalse(slot_state.in_use)
            self.assertTrue(slot_state.quarantined)
        self.assertEqual(interrupt_pool.metrics()["quarantined_slots"], ["interrupt-0"])
        self.assertEqual(interrupt_resets[0], resets_before_interrupt)
        blocked_waiter.join(timeout=2.0)
        self.assertFalse(blocked_waiter.is_alive())
        self.assertEqual(len(blocked_errors), 1)
        self.assertIsInstance(blocked_errors[0], PoolTimeoutError)
        # Duplicate release() must re-raise the original KeyboardInterrupt rather than returning {}.
        with self.assertRaises(KeyboardInterrupt):
            interrupt_lease.release()
        release_interrupt.set()
        active_thread.join(timeout=5.0)
        self.assertFalse(active_thread.is_alive())

        # Failing _format_error(error) inside release() must also quarantine fail-closed and re-raise.
        class _BrokenStr:
            def __str__(self) -> str:
                raise RuntimeError("unprintable error")

        bad_err_pool = SandboxPool([PoolSlot("bad-err-0", self._make_toy_lab())])
        self.addCleanup(bad_err_pool.close)
        bad_err_lease = bad_err_pool.acquire("agent-bad-err")
        with self.assertRaises(RuntimeError):
            bad_err_lease.release(error=_BrokenStr())
        self.assertEqual(bad_err_pool.metrics()["quarantined_slots"], ["bad-err-0"])
        with self.assertRaises(RuntimeError):
            bad_err_lease.release()

    def test_multi_writer_shared_trace_chain_and_expected_head_truncation_detection(self):
        """Regression for P1-3: two independent SandboxPool instances writing concurrently to
        the same trace_path produce a valid unbroken chain, and expected_head catches truncation."""
        lab_a = self._make_toy_lab()
        lab_b = self._make_toy_lab()

        with tempfile.TemporaryDirectory(prefix="rsibox-shared-trace-") as temp_dir:
            trace_path = os.path.join(temp_dir, "shared.jsonl")
            pool_a = SandboxPool([PoolSlot("pool-a-0", lab_a)], trace_path=trace_path)
            pool_b = SandboxPool([PoolSlot("pool-b-0", lab_b)], trace_path=trace_path)
            self.addCleanup(pool_a.close)
            self.addCleanup(pool_b.close)

            barrier = threading.Barrier(8)

            def worker(target_pool: SandboxPool, worker_name: str) -> None:
                barrier.wait(timeout=10.0)
                outcome = target_pool.run_isolated(worker_name, lambda lab, lease: lab.check()["ok"])
                self.assertTrue(outcome["ok"], outcome)

            threads = [
                threading.Thread(
                    target=worker,
                    args=(pool_a if index < 4 else pool_b, f"multi-agent-{index}"),
                )
                for index in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=15.0)
                self.assertFalse(thread.is_alive())

            verified = verify_trace(trace_path)
            self.assertTrue(verified["ok"], verified)
            self.assertEqual(verified["count"], 8)
            self.assertEqual([entry["seq"] for entry in verified["records"]], list(range(1, 9)))

            # Verify expected_head matches live head and detects tail truncation.
            saved_head = verified["head"]
            self.assertTrue(verify_trace(trace_path, expected_head=saved_head)["ok"])
            with open(trace_path, encoding="utf-8") as handle:
                lines = handle.readlines()
            with open(trace_path, "w", encoding="utf-8") as handle:
                handle.writelines(lines[:-1])
            self.assertTrue(verify_trace(trace_path)["ok"])
            truncated_check = verify_trace(trace_path, expected_head=saved_head)
            self.assertFalse(truncated_check["ok"])
            self.assertIn("trace head mismatch", truncated_check["error"])

    def test_verify_trace_rejects_malformed_json_bool_seq_reordered_and_pool_fails_closed(self):
        """Regression for P1-4: verify_trace rejects non-dict JSON, bool seq, syntax errors,
        deletions, and reorders; SandboxPool resumes valid traces and fails closed on corrupt ones."""
        slot_lab = self._make_toy_lab()
        with tempfile.TemporaryDirectory(prefix="rsibox-malformed-trace-") as temp_dir:
            trace_path = os.path.join(temp_dir, "trace.jsonl")

            # Write 3 valid records, then resume in a second pool instance and append a 4th.
            pool_one = SandboxPool([PoolSlot("slot-0", slot_lab)], trace_path=trace_path)
            for index in range(3):
                self.assertTrue(pool_one.run_isolated(f"a-{index}", lambda lab, lease: True)["ok"])
            pool_one.close()

            pool_two = SandboxPool([PoolSlot("slot-0", slot_lab)], trace_path=trace_path)
            self.assertTrue(pool_two.run_isolated("a-3", lambda lab, lease: True)["ok"])
            expected_head = pool_two.metrics()["trace_head"]
            pool_two.close()
            resumed = verify_trace(trace_path, expected_head=expected_head)
            self.assertTrue(resumed["ok"], resumed)
            self.assertEqual(resumed["count"], 4)

            with open(trace_path, encoding="utf-8") as handle:
                valid_lines = handle.readlines()

            # Non-dict JSON lines must return ok=False without raising AttributeError.
            for bad_payload in ("[]\n", "123\n", "null\n", '"text"\n'):
                bad_path = os.path.join(temp_dir, "non_dict.jsonl")
                with open(bad_path, "w", encoding="utf-8") as handle:
                    handle.write(bad_payload)
                result = verify_trace(bad_path)
                self.assertFalse(result["ok"])
                self.assertEqual(result["line"], 1)

            # Boolean seq (True == 1 in Python) must be rejected.
            bool_record = json.loads(valid_lines[0])
            bool_record["seq"] = True
            bool_path = os.path.join(temp_dir, "bool_seq.jsonl")
            with open(bool_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(bool_record) + "\n")
            self.assertFalse(verify_trace(bool_path)["ok"])

            # Deleted middle line and swapped lines must fail verification.
            deleted_path = os.path.join(temp_dir, "deleted.jsonl")
            with open(deleted_path, "w", encoding="utf-8") as handle:
                handle.writelines([valid_lines[0], valid_lines[2], valid_lines[3]])
            self.assertFalse(verify_trace(deleted_path)["ok"])

            swapped_path = os.path.join(temp_dir, "swapped.jsonl")
            with open(swapped_path, "w", encoding="utf-8") as handle:
                handle.writelines([valid_lines[1], valid_lines[0]])
            self.assertFalse(verify_trace(swapped_path)["ok"])

            # Malformed JSON reports line number and causes SandboxPool.__init__ to fail closed.
            corrupt_path = os.path.join(temp_dir, "corrupt.jsonl")
            with open(corrupt_path, "w", encoding="utf-8") as handle:
                handle.write(valid_lines[0] + "{not valid json\n")
            corrupt_check = verify_trace(corrupt_path)
            self.assertFalse(corrupt_check["ok"])
            self.assertEqual(corrupt_check["line"], 2)
            with self.assertRaises(PoolError):
                SandboxPool([PoolSlot("slot-0", slot_lab)], trace_path=corrupt_path)

    def test_pool_slot_rejects_non_ready_box_states_and_forwards_receipts(self):
        """Regression for P1-5: PoolSlot rejects booting/failed/closed Boxes cleanly instead of
        mistaking a booting engine for a poisoned worker, and forwards ready Box receipts."""
        booting_box = Box(lambda receipts=None: toy_lab.ToyLab(ranks=2, receipts=receipts), spec="toy")
        self.assertEqual(booting_box.state, "booting")
        with self.assertRaises(ValueError):
            PoolSlot("booting-0", booting_box)

        failed_box = Box(lambda receipts=None: (_ for _ in ()).throw(RuntimeError("boot failed")), spec="toy")
        failed_box.boot()
        self.assertEqual(failed_box.state, "failed")
        with self.assertRaises(ValueError):
            PoolSlot("failed-0", failed_box)

        with tempfile.TemporaryDirectory(prefix="rsibox-ready-box-") as state_dir:
            ready_box = Box(
                lambda receipts=None: toy_lab.ToyLab(ranks=2, receipts=receipts),
                spec="toy",
                state_dir=state_dir,
            )
            ready_box.boot()
            self.addCleanup(ready_box.close)
            pool = SandboxPool([PoolSlot("ready-0", ready_box)])
            self.addCleanup(pool.close)
            with pool.lease("box-agent") as leased:
                receipts_reply = leased.lab.receipts()
                self.assertTrue(receipts_reply["ok"])
                self.assertIsInstance(receipts_reply["receipts"], list)

            ready_box.close()
            with self.assertRaises(ValueError):
                PoolSlot("closed-0", ready_box)

    def test_priority_scheduling_and_fifo_tie_break_with_trace(self):
        """Golden case: higher-priority waiters (e.g., Eval) wake ahead of lower-priority waiters
        queued earlier, equal-priority waiters preserve strict FIFO order, negative priorities work,
        invalid priorities fail closed across acquire/lease/run_isolated, and priority is recorded
        in the hash-chained trace (while remaining backward-compatible with legacy traces)."""
        slot_lab = toy_lab.ToyLab(ranks=2)
        self.addCleanup(slot_lab.close)
        with tempfile.TemporaryDirectory(prefix="rsibox-priority-") as temp_dir:
            trace_path = os.path.join(temp_dir, "priority_trace.jsonl")
            # Seed a legacy JSONL trace without the "priority" key to lock down backward compatibility.
            from rsibox import pool as pool_module

            legacy_entry, _, _ = pool_module._chain_record(
                {
                    "agent_id": "legacy-agent", "role": None, "slot_role": None,
                    "idea_id": "legacy-0", "slot_id": "slot-0",
                    "acquired_at": 1.0, "released_at": 2.0, "queue_wait_s": 0.0,
                    "hold_s": 1.0, "cleanup_s": 0.0, "recovery_s": 0.0,
                    "recovered": False, "error": None, "status_after": "base",
                },
                0,
                pool_module._GENESIS_HASH,
            )
            with open(trace_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(legacy_entry, sort_keys=True) + "\n")
            self.assertTrue(verify_trace(trace_path)["ok"])

            pool = SandboxPool([PoolSlot("slot-0", slot_lab)], trace_path=trace_path)
            self.addCleanup(pool.close)

            # Invalid priorities fail closed with TypeError across acquire, lease, and run_isolated.
            for bad_priority in (True, False, 1.5, "10", None):
                with self.assertRaises(TypeError):
                    pool.acquire("bad-agent", priority=bad_priority)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    with pool.lease("bad-agent", priority=bad_priority):  # type: ignore[arg-type]
                        pass
                with self.assertRaises(TypeError):
                    pool.run_isolated("bad-agent", lambda _lab, _lease: None, priority=bad_priority)  # type: ignore[arg-type]

            holder = pool.acquire("holder", priority=0)
            self.assertEqual(holder.priority, 0)
            with self.assertRaises(AttributeError):
                holder.priority = 99  # type: ignore[misc]

            wake_order: list[tuple[str, int]] = []
            order_lock = threading.Lock()
            errors: list[BaseException] = []

            def enqueue_waiter(agent_id: str, priority: int) -> threading.Thread:
                def action(_lab, lease) -> None:
                    with order_lock:
                        wake_order.append((agent_id, lease.priority))

                def runner() -> None:
                    try:
                        outcome = pool.run_isolated(
                            agent_id,
                            action,
                            priority=priority,
                            timeout_s=5.0,
                        )
                        if not outcome["ok"]:
                            raise AssertionError(outcome)
                    except BaseException as err:
                        errors.append(err)

                worker_thread = threading.Thread(target=runner)
                worker_thread.start()
                return worker_thread

            # Queue: bg-low (-5), dev-1 (0), dev-2 (0), eval-1 (10), fix-1 (5), eval-2 (10)
            waiter_threads = []
            for expected_len, (agent_name, prio) in enumerate(
                [("bg-low", -5), ("dev-1", 0), ("dev-2", 0), ("eval-1", 10), ("fix-1", 5), ("eval-2", 10)],
                start=1,
            ):
                waiter_threads.append(enqueue_waiter(agent_name, prio))
                self._wait_for_waiters(pool, expected_len)

            # A high-priority waiter that times out is skipped cleanly without blocking lower priorities.
            with self.assertRaises(PoolTimeoutError):
                pool.acquire("timed-out-eval", priority=50, timeout_s=0.01)

            holder.release()
            for worker_thread in waiter_threads:
                worker_thread.join(timeout=5.0)
                self.assertFalse(worker_thread.is_alive())
            self.assertEqual(errors, [])

            # Expected wake order: eval-1 (10), eval-2 (10), fix-1 (5), dev-1 (0), dev-2 (0), bg-low (-5)
            self.assertEqual(
                wake_order,
                [("eval-1", 10), ("eval-2", 10), ("fix-1", 5), ("dev-1", 0), ("dev-2", 0), ("bg-low", -5)],
            )

            verified = verify_trace(trace_path)
            self.assertTrue(verified["ok"], verified)
            self.assertEqual(verified["count"], 8)
            self.assertNotIn("priority", verified["records"][0])
            self.assertEqual(
                [(entry["agent_id"], entry["priority"]) for entry in verified["records"][1:]],
                [
                    ("holder", 0),
                    ("eval-1", 10),
                    ("eval-2", 10),
                    ("fix-1", 5),
                    ("dev-1", 0),
                    ("dev-2", 0),
                    ("bg-low", -5),
                ],
            )

            # Post-handoff cancellation of the highest-priority waiter must hand off to the next-highest
            # priority waiter (not FIFO).
            real_waiter_cls = pool_module._Waiter
            holder_two = pool.acquire("holder-2", priority=0)
            handoff_order: list[str] = []
            cancel_errors: list[BaseException] = []

            def make_interrupting_waiter(role, priority=0):
                waiter = real_waiter_cls(role, priority)
                if priority == 20:
                    real_wait = waiter.event.wait

                    def wait_then_interrupt(timeout=None):
                        real_wait(timeout=timeout)
                        raise KeyboardInterrupt("cancel high-priority after handoff")

                    waiter.event.wait = wait_then_interrupt
                return waiter

            with mock.patch.object(pool_module, "_Waiter", side_effect=make_interrupting_waiter):
                low_thread = threading.Thread(
                    target=lambda: pool.run_isolated("low-0", lambda _l, _s: handoff_order.append("low-0"), priority=0, timeout_s=5.0)
                )
                low_thread.start()
                self._wait_for_waiters(pool, 1)
                mid_thread = threading.Thread(
                    target=lambda: pool.run_isolated("mid-5", lambda _l, _s: handoff_order.append("mid-5"), priority=5, timeout_s=5.0)
                )
                mid_thread.start()
                self._wait_for_waiters(pool, 2)
                cancelled_high = threading.Thread(
                    target=lambda: self._capture_error(
                        lambda: pool.acquire("cancel-20", priority=20, timeout_s=5.0),
                        cancel_errors,
                    )
                )
                cancelled_high.start()
                self._wait_for_waiters(pool, 3)

                holder_two.release()
                for worker_thread in (cancelled_high, mid_thread, low_thread):
                    worker_thread.join(timeout=5.0)
                    self.assertFalse(worker_thread.is_alive())

            self.assertEqual(len(cancel_errors), 1)
            self.assertIsInstance(cancel_errors[0], KeyboardInterrupt)
            self.assertEqual(handoff_order, ["mid-5", "low-0"])

    def test_priority_respects_role_isolation(self):
        """Golden case: a high-priority eval waiter does not starve a dev slot or block fast-path
        acquisition of an idle dev slot, while still preempting a lower-priority generalist waiter
        on an eval slot."""
        dev_lab = toy_lab.ToyLab(ranks=2)
        eval_lab = toy_lab.ToyLab(ranks=2)
        self.addCleanup(dev_lab.close)
        self.addCleanup(eval_lab.close)
        pool = SandboxPool([
            PoolSlot("dev-0", dev_lab, role="dev"),
            PoolSlot("eval-0", eval_lab, role="eval"),
        ])
        self.addCleanup(pool.close)

        dev_holder = pool.acquire("holder-dev", role="dev")
        eval_holder = pool.acquire("holder-eval", role="eval")

        dev_acquired: list[str] = []
        eval_order: list[str] = []
        errors: list[BaseException] = []
        dev_entered = threading.Event()
        dev_release_gate = threading.Event()
        self.addCleanup(dev_release_gate.set)

        def wait_role(
            agent_id: str,
            role: str | None,
            priority: int,
            bucket: list[str],
            *,
            hold_gate: tuple[threading.Event, threading.Event] | None = None,
        ) -> threading.Thread:
            def runner() -> None:
                try:
                    with pool.lease(agent_id, role=role, priority=priority, timeout_s=5.0) as lease:
                        bucket.append(f"{agent_id}:{lease.slot_id}")
                        if hold_gate is not None:
                            hold_gate[0].set()
                            hold_gate[1].wait(timeout=5.0)
                except BaseException as err:
                    errors.append(err)

            worker_thread = threading.Thread(target=runner)
            worker_thread.start()
            return worker_thread

        generalist_thread = wait_role("gen-low", None, 0, eval_order)
        self._wait_for_waiters(pool, 1)
        eval_thread = wait_role("eval-high", "eval", 10, eval_order)
        self._wait_for_waiters(pool, 2)
        dev_thread = wait_role("dev-mid", "dev", 5, dev_acquired, hold_gate=(dev_entered, dev_release_gate))
        self._wait_for_waiters(pool, 3)

        # Releasing dev-0 wakes dev-mid (priority 5 > gen-low priority 0), ignoring eval-high (wrong role).
        dev_holder.release()
        self.assertTrue(dev_entered.wait(timeout=5.0))
        self.assertEqual(dev_acquired, ["dev-mid:dev-0"])

        # While dev-mid holds dev-0, releasing eval-0 must wake eval-high (priority 10) before gen-low (priority 0).
        eval_holder.release()
        eval_thread.join(timeout=5.0)
        self.assertFalse(eval_thread.is_alive())
        dev_release_gate.set()
        for worker_thread in (dev_thread, generalist_thread):
            worker_thread.join(timeout=5.0)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(eval_order, ["eval-high:eval-0", "gen-low:eval-0"])

        # Fast-path cross-role check: with eval-0 busy and a high-priority eval waiter queued,
        # a low-priority generalist caller claims idle dev-0 immediately even with timeout_s=0.
        eval_holder_two = pool.acquire("holder-eval-2", role="eval")
        queued_eval_got: list[str] = []
        queued_eval_thread = wait_role("eval-queued", "eval", 10, queued_eval_got)
        self._wait_for_waiters(pool, 1)
        with pool.lease("gen-fast", role=None, priority=0, timeout_s=0.0) as fast_lease:
            self.assertEqual(fast_lease.slot_id, "dev-0")
        eval_holder_two.release()
        queued_eval_thread.join(timeout=5.0)
        self.assertFalse(queued_eval_thread.is_alive())
        self.assertEqual(queued_eval_got, ["eval-queued:eval-0"])
        self.assertEqual(errors, [])

    @staticmethod
    def _capture_error(operation, bucket: list[BaseException]) -> None:
        try:
            operation()
        except BaseException as error:
            bucket.append(error)


if __name__ == "__main__":
    unittest.main()
