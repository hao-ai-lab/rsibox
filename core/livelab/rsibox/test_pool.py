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
    AgentQuarantinedError,
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

    def test_priority_aging_prevents_low_priority_starvation_while_preserving_burst_priority(self):
        """Golden case (starvation vs. aging): under continuous high-priority Eval arrivals,
        static priority starves queued Dev (`priority=0`) behind every arriving Eval (`priority=10`).
        With `priority_aging_s=0.5`, short bursts still favor Eval (`eff 12.0 > 6.0` at t=3.0),
        but once Dev's wait exceeds `10 * 0.5 = 5.0 s`, Dev (`eff 12.0`) overtakes a newly arriving
        Eval (`eff 10.0` at t=6.0) to guarantee bounded tail latency."""
        lab_bad = self._make_toy_lab()
        for bad_type in (True, False, "0.5"):
            with self.assertRaises(TypeError):
                SandboxPool([PoolSlot("s0", lab_bad)], priority_aging_s=bad_type)  # type: ignore[arg-type]
        for bad_val in (0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), 10**400):
            with self.assertRaises(ValueError):
                SandboxPool([PoolSlot("s0", lab_bad)], priority_aging_s=bad_val)
        # Positive integer priority_aging_s is accepted.
        int_aging_pool = SandboxPool([PoolSlot("s-int", self._make_toy_lab())], priority_aging_s=1)
        int_aging_pool.close()

        # Deterministic fake-clock comparison: static (None) vs aged (0.5 s per +1 priority).
        def run_scenario(priority_aging_s: float | None, *, late_arrival_s: float = 6.0) -> list[str]:
            now = [0.0]
            pool = SandboxPool(
                [PoolSlot("slot-0", self._make_toy_lab())],
                clock=lambda: now[0],
                priority_aging_s=priority_aging_s,
            )
            self.addCleanup(pool.close)
            order: list[str] = []
            errors: list[BaseException] = []
            eval_early_entered = threading.Event()
            eval_early_release = threading.Event()
            self.addCleanup(eval_early_release.set)

            holder = pool.acquire("holder")

            def run_agent(
                agent_id: str,
                priority: int,
                gate: tuple[threading.Event, threading.Event] | None = None,
            ) -> threading.Thread:
                def worker() -> None:
                    try:
                        with pool.lease(agent_id, priority=priority, timeout_s=60.0):
                            order.append(agent_id)
                            if gate is not None:
                                gate[0].set()
                                gate[1].wait(timeout=5.0)
                    except BaseException as err:
                        errors.append(err)

                worker_thread = threading.Thread(target=worker)
                worker_thread.start()
                return worker_thread

            # t=0.0: dev-0 (priority=0) queues first.
            dev_thread = run_agent("dev-0", 0)
            self._wait_for_waiters(pool, 1)

            # t=2.0: eval-early (priority=10) queues.
            now[0] = 2.0
            eval_early_thread = run_agent("eval-early", 10, gate=(eval_early_entered, eval_early_release))
            self._wait_for_waiters(pool, 2)

            # t=3.0: holder releases -> eval-early wins (aged eff 12.0 > dev eff 6.0; static 10 > 0).
            now[0] = 3.0
            holder.release()
            self.assertTrue(eval_early_entered.wait(timeout=5.0))
            self.assertEqual(order, ["eval-early"])

            # At late_arrival_s: eval-late (priority=10) arrives while dev-0 has waited late_arrival_s.
            now[0] = late_arrival_s
            eval_late_thread = run_agent("eval-late", 10)
            self._wait_for_waiters(pool, 2)
            eval_early_release.set()

            for worker_thread in (dev_thread, eval_early_thread, eval_late_thread):
                worker_thread.join(timeout=5.0)
                self.assertFalse(worker_thread.is_alive())
            self.assertEqual(errors, [])
            return order

        self.assertEqual(run_scenario(None), ["eval-early", "eval-late", "dev-0"])
        self.assertEqual(run_scenario(0.5, late_arrival_s=6.0), ["eval-early", "dev-0", "eval-late"])
        # Exact effective-priority tie-break: at t=5.0, dev-0 (0 + 5.0/0.5 = 10.0) ties eval-late (10.0) -> FIFO wins.
        self.assertEqual(run_scenario(0.5, late_arrival_s=5.0), ["eval-early", "dev-0", "eval-late"])

        # Non-monotonic clock guard: high-10 enqueues at t=10.0, clock steps back to t=1.0, then mid-5
        # enqueues at t=1.0. Without max(0.0, ...), high-10 would drop to 1.0 < 5.0 and lose to mid-5!
        now_back = [10.0]
        pool_back = SandboxPool(
            [PoolSlot("slot-back", self._make_toy_lab())],
            clock=lambda: now_back[0],
            priority_aging_s=1.0,
        )
        self.addCleanup(pool_back.close)
        holder_back = pool_back.acquire("holder")
        back_order: list[str] = []
        back_errors: list[BaseException] = []

        def spawn_back(agent_id: str, priority: int) -> threading.Thread:
            def run() -> None:
                try:
                    with pool_back.lease(agent_id, priority=priority, timeout_s=60.0):
                        back_order.append(agent_id)
                except BaseException as err:
                    back_errors.append(err)
            worker_thread = threading.Thread(target=run)
            worker_thread.start()
            return worker_thread

        high_thread = spawn_back("high-10", 10)
        self._wait_for_waiters(pool_back, 1)
        now_back[0] = 1.0  # stepped backwards relative to high-10's enqueued_mono=10.0
        mid_thread = spawn_back("mid-5", 5)
        self._wait_for_waiters(pool_back, 2)
        holder_back.release()
        for worker_thread in (high_thread, mid_thread):
            worker_thread.join(timeout=5.0)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(back_errors, [])
        self.assertEqual(back_order, ["high-10", "mid-5"])

        # Subnormal/tiny priority_aging_s & large int priority never collapse to float('inf') or lose ordering.
        now_ext = [0.0]
        pool_ext = SandboxPool(
            [PoolSlot("slot-ext", self._make_toy_lab())],
            clock=lambda: now_ext[0],
            priority_aging_s=1e-308,
        )
        self.addCleanup(pool_ext.close)
        holder_ext = pool_ext.acquire("holder")
        ext_order: list[str] = []
        ext_errors: list[BaseException] = []

        def spawn_ext(agent_id: str, priority: int) -> threading.Thread:
            def run() -> None:
                try:
                    with pool_ext.lease(agent_id, priority=priority, timeout_s=60.0):
                        ext_order.append(agent_id)
                except BaseException as err:
                    ext_errors.append(err)
            worker_thread = threading.Thread(target=run)
            worker_thread.start()
            return worker_thread

        low_ext_thread = spawn_ext("low-neg100", -100)
        self._wait_for_waiters(pool_ext, 1)
        high_ext_thread = spawn_ext("high-pos100", 100)
        self._wait_for_waiters(pool_ext, 2)
        now_ext[0] = 2.0  # 2.0 / 1e-308 would overflow IEEE-754 float64 to inf!
        holder_ext.release()
        for worker_thread in (low_ext_thread, high_ext_thread):
            worker_thread.join(timeout=5.0)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(ext_errors, [])
        self.assertEqual(ext_order, ["high-pos100", "low-neg100"])

        # Exact large-int priority ordering when priority_aging_s=None (> 2**53 precision check).
        pool_big = SandboxPool([PoolSlot("slot-big", self._make_toy_lab())], priority_aging_s=None)
        self.addCleanup(pool_big.close)
        holder_big = pool_big.acquire("holder")
        big_order: list[str] = []
        big_errors: list[BaseException] = []

        def spawn_big(agent_id: str, priority: int) -> threading.Thread:
            def run() -> None:
                try:
                    with pool_big.lease(agent_id, priority=priority, timeout_s=60.0):
                        big_order.append(agent_id)
                except BaseException as err:
                    big_errors.append(err)
            worker_thread = threading.Thread(target=run)
            worker_thread.start()
            return worker_thread

        big_a = spawn_big("big-0", (1 << 60))
        self._wait_for_waiters(pool_big, 1)
        big_b = spawn_big("big-1", (1 << 60) + 1)
        self._wait_for_waiters(pool_big, 2)
        holder_big.release()
        for worker_thread in (big_a, big_b):
            worker_thread.join(timeout=5.0)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(big_errors, [])
        self.assertEqual(big_order, ["big-1", "big-0"])

        # Exact large-int priority ordering WITH aging enabled (no float64 rounding inversion near 2**52).
        now_big_aged = [0.0]
        pool_big_aged = SandboxPool(
            [PoolSlot("slot-big-aged", self._make_toy_lab())],
            clock=lambda: now_big_aged[0],
            priority_aging_s=1.0,
        )
        self.addCleanup(pool_big_aged.close)
        holder_big_aged = pool_big_aged.acquire("holder")
        big_aged_order: list[str] = []
        big_aged_errors: list[BaseException] = []

        def spawn_big_aged(agent_id: str, priority: int) -> threading.Thread:
            def run() -> None:
                try:
                    with pool_big_aged.lease(agent_id, priority=priority, timeout_s=60.0):
                        big_aged_order.append(agent_id)
                except BaseException as err:
                    big_aged_errors.append(err)
            worker_thread = threading.Thread(target=run)
            worker_thread.start()
            return worker_thread

        t_lower = spawn_big_aged("near52-lower", (1 << 52) - 1)
        self._wait_for_waiters(pool_big_aged, 1)
        now_big_aged[0] = 0.8  # true eff = (1<<52) - 0.2 < (1<<52)
        t_higher = spawn_big_aged("near52-higher", (1 << 52))
        self._wait_for_waiters(pool_big_aged, 2)
        holder_big_aged.release()
        for worker_thread in (t_lower, t_higher):
            worker_thread.join(timeout=5.0)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(big_aged_errors, [])
        self.assertEqual(big_aged_order, ["near52-higher", "near52-lower"])

    def test_elastic_role_borrowing_prefers_primary_blocks_theft_and_survives_quarantine(self):
        """Golden case (work-conserving Triad): on a 3-role pool (`eval`, `fixer`, `dev`),
        `fallback_roles=("fixer", "dev")` lets `eval` prefer `eval-0` first, borrow `fixer-0`
        then `dev-0` when `eval-0` is busy, never steal a home slot from a queued primary-role
        waiter (even with higher priority), survive quarantine of its primary slot via fallback,
        and record `borrowed` provenance in the SHA-256 hash-chained trace and `metrics()`."""
        with tempfile.TemporaryDirectory(prefix="rsibox-elastic-") as tmpdir:
            trace_path = os.path.join(tmpdir, "elastic-roles.jsonl")
            self._run_elastic_role_borrowing_assertions(trace_path)

    def _run_elastic_role_borrowing_assertions(self, trace_path: str) -> None:
        eval_lab = self._make_toy_lab()
        fixer_lab = self._make_toy_lab()
        dev_lab = self._make_toy_lab()
        pool = SandboxPool(
            [
                PoolSlot("dev-0", dev_lab, role="dev"),
                PoolSlot("fixer-0", fixer_lab, role="fixer"),
                PoolSlot("eval-0", eval_lab, role="eval"),
            ],
            trace_path=trace_path,
        )
        self.addCleanup(pool.close)

        # 1. Fail-closed validation (including bytearray/memoryview, empty role, and empty sequence no-op).
        with self.assertRaises(ValueError):
            pool.acquire("bad-none", role=None, fallback_roles=("dev",))
        with self.assertRaises(ValueError):
            pool.acquire("bad-empty-role", role="  ", fallback_roles=("dev",))
        with self.assertRaises(TypeError):
            pool.acquire("bad-int-role", role=123, fallback_roles=("dev",))  # type: ignore[arg-type]
        for bad_fb in ("dev", b"dev", bytearray(b""), memoryview(b""), {"dev"}, 123):
            with self.assertRaises(TypeError):
                pool.acquire("bad-type", role="eval", fallback_roles=bad_fb)  # type: ignore[arg-type]
        for bad_entry in (["  "], [""], [123]):
            exc_type = TypeError if isinstance(bad_entry[0], int) else ValueError
            with self.assertRaises(exc_type):
                pool.acquire("bad-entry", role="eval", fallback_roles=bad_entry)  # type: ignore[arg-type]
        with pool.lease("gen-empty-fb", role=None, fallback_roles=(), timeout_s=2.0) as gen_lease:
            self.assertFalse(gen_lease.borrowed)
            self.assertEqual(gen_lease.fallback_roles, ())
            with self.assertRaises(AttributeError):
                gen_lease.borrowed = True  # type: ignore[misc]
            with self.assertRaises(AttributeError):
                gen_lease.fallback_roles = ("dev",)  # type: ignore[misc]

        # 2. Primary-first preference & ordered fallback (`eval-0` -> `fixer-0` -> `dev-0`),
        #    deduplicating repeated/primary entries.
        l_primary = pool.acquire("eval-a", role="eval", fallback_roles=("eval", "fixer", "dev", "fixer"), timeout_s=2.0)
        self.assertEqual(l_primary.slot_id, "eval-0")
        self.assertFalse(l_primary.borrowed)
        self.assertEqual(l_primary.fallback_roles, ("fixer", "dev"))

        l_fb1 = pool.acquire("eval-b", role="eval", fallback_roles=("fixer", "dev"), timeout_s=2.0)
        self.assertEqual(l_fb1.slot_id, "fixer-0")
        self.assertTrue(l_fb1.borrowed)

        l_fb2 = pool.acquire("eval-c", role="eval", fallback_roles=("fixer", "dev"), timeout_s=2.0)
        self.assertEqual(l_fb2.slot_id, "dev-0")
        self.assertTrue(l_fb2.borrowed)

        # 3. Zero primary-role theft on handoff: while all 3 slots are held, queue a HIGH-priority
        #    `eval` borrower (priority=10, fallback=("dev",)) AND a LOW-priority primary `dev` waiter
        #    (priority=0, role="dev"). Releasing `dev-0` must hand off to primary `dev` FIRST!
        handoff_order: list[str] = []
        handoff_errors: list[BaseException] = []
        dev_entered = threading.Event()
        dev_release = threading.Event()
        self.addCleanup(dev_release.set)

        def queue_worker(
            agent_id: str,
            role: str,
            priority: int,
            fallback_roles: tuple[str, ...] | None,
            gate: tuple[threading.Event, threading.Event] | None = None,
        ) -> threading.Thread:
            def runner() -> None:
                try:
                    with pool.lease(
                        agent_id, role=role, priority=priority, fallback_roles=fallback_roles, timeout_s=5.0,
                    ) as leased:
                        handoff_order.append(f"{agent_id}:{leased.slot_id}:borrowed={leased.borrowed}")
                        if gate is not None:
                            gate[0].set()
                            gate[1].wait(timeout=5.0)
                except BaseException as err:
                    handoff_errors.append(err)

            worker_thread = threading.Thread(target=runner)
            worker_thread.start()
            return worker_thread

        borrow_eval_t = queue_worker("eval-borrower", "eval", 10, ("dev",))
        self._wait_for_waiters(pool, 1)
        primary_dev_t = queue_worker("dev-home", "dev", 0, None, gate=(dev_entered, dev_release))
        self._wait_for_waiters(pool, 2)

        l_fb2.release()  # releases dev-0
        self.assertTrue(dev_entered.wait(timeout=5.0))
        self.assertEqual(handoff_order, ["dev-home:dev-0:borrowed=False"])

        # Once dev-home releases dev-0 and no primary dev waiter remains, dev-0 is lent to eval-borrower!
        dev_release.set()
        for worker_thread in (primary_dev_t, borrow_eval_t):
            worker_thread.join(timeout=5.0)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(handoff_errors, [])
        self.assertEqual(
            handoff_order,
            ["dev-home:dev-0:borrowed=False", "eval-borrower:dev-0:borrowed=True"],
        )

        # Release remaining initial leases.
        l_fb1.release()
        l_primary.release()

        # 4. Quarantine resilience: quarantine `eval-0`, confirm `eval` with `fallback_roles=("dev",)`
        #    still completes via `run_isolated` by borrowing `dev-0`!
        eval_lab.status = lambda: {"ok": False, "state": "poisoned", "error": "eval gpu dead"}
        eval_lab.reset = lambda kind="state": {"ok": False, "error": "respawn failed"}
        res_iso = pool.run_isolated(
            "eval-resilient",
            lambda leased_lab, lease: (lease.slot_id, lease.borrowed, leased_lab.status()["state"]),
            role="eval",
            fallback_roles=("dev",),
            timeout_s=5.0,
        )
        self.assertTrue(res_iso["ok"], res_iso)
        self.assertEqual(res_iso["slot_id"], "dev-0")
        self.assertTrue(res_iso["borrowed"])
        self.assertEqual(res_iso["result"], ("dev-0", True, "base"))
        self.assertEqual(pool.metrics()["quarantined_slots"], ["eval-0"])

        # Strict eval caller without fallback fails immediately once eval-0 is quarantined.
        strict_fail = pool.run_isolated("eval-strict", lambda _l, _s: "nope", role="eval", timeout_s=5.0)
        self.assertFalse(strict_fail["ok"])
        self.assertFalse(strict_fail["borrowed"])
        self.assertIn("no healthy slot", strict_fail["error"])

        # 5. Competing borrowers (Tier 2 priority + FIFO) AND generalist Tier 1 precedence over borrowers.
        hold_dev = pool.acquire("hold-dev", role="dev", timeout_s=2.0)
        hold_fix = pool.acquire("hold-fix", role="fixer", timeout_s=2.0)
        comp_order: list[str] = []
        comp_errors: list[BaseException] = []

        def spawn_comp(agent_id: str, role: str | None, priority: int, fb: tuple[str, ...] | None) -> threading.Thread:
            def run() -> None:
                try:
                    with pool.lease(agent_id, role=role, priority=priority, fallback_roles=fb, timeout_s=5.0) as leased:
                        comp_order.append(f"{agent_id}:{leased.slot_id}:b={leased.borrowed}")
                except BaseException as err:
                    comp_errors.append(err)
            th = threading.Thread(target=run)
            th.start()
            return th

        t_b_low = spawn_comp("borrow-low", "eval", 2, ("dev",))
        self._wait_for_waiters(pool, 1)
        t_b_hi1 = spawn_comp("borrow-hi1", "eval", 10, ("dev",))
        self._wait_for_waiters(pool, 2)
        t_b_hi2 = spawn_comp("borrow-hi2", "eval", 10, ("dev",))
        self._wait_for_waiters(pool, 3)
        t_gen = spawn_comp("gen-t1", None, 0, None)
        self._wait_for_waiters(pool, 4)

        hold_dev.release()  # Tier 1 generalist beats all borrowers; then Tier 2 borrowers run hi1 -> hi2 -> low
        for th in (t_gen, t_b_hi1, t_b_hi2, t_b_low):
            th.join(timeout=5.0)
            self.assertFalse(th.is_alive())
        hold_fix.release()
        self.assertEqual(comp_errors, [])
        self.assertEqual(
            comp_order,
            [
                "gen-t1:dev-0:b=False",
                "borrow-hi1:dev-0:b=True",
                "borrow-hi2:dev-0:b=True",
                "borrow-low:dev-0:b=True",
            ],
        )

        # 6. Queued-borrower quarantine wakeup:
        #    (a) Primary `eval-0` dies while `dev-0` is held & healthy -> strict eval waiter wakes with
        #        PoolTimeoutError, while co-queued borrower stays alive and borrows `dev-0` on release.
        #    (b) All primary+fallback slots (`eval-0` + `dev-0`) die while `fixer-0` is healthy ->
        #        queued borrower (`timeout_s=None`) wakes immediately with PoolTimeoutError.
        q_eval = self._make_toy_lab()
        q_dev = self._make_toy_lab()
        q_fixer = self._make_toy_lab()
        q_pool = SandboxPool(
            [
                PoolSlot("q-eval", q_eval, role="eval"),
                PoolSlot("q-dev", q_dev, role="dev"),
                PoolSlot("q-fixer", q_fixer, role="fixer"),
            ]
        )
        self.addCleanup(q_pool.close)
        h_q_eval = q_pool.acquire("h-eval", role="eval", timeout_s=2.0)
        h_q_dev = q_pool.acquire("h-dev", role="dev", timeout_s=2.0)
        strict_errs: list[BaseException] = []
        borrow_got: list[tuple[str, bool]] = []
        borrow_errs: list[BaseException] = []

        t_strict = threading.Thread(
            target=lambda: self._capture_error(
                lambda: q_pool.acquire("q-strict", role="eval", timeout_s=None),
                strict_errs,
            )
        )
        def run_q_borrow() -> None:
            with q_pool.lease("q-borrow", role="eval", fallback_roles=("dev",), timeout_s=None) as leased:
                borrow_got.append((leased.slot_id, leased.borrowed))

        t_borrow = threading.Thread(target=run_q_borrow)
        t_strict.start()
        self._wait_for_waiters(q_pool, 1)
        t_borrow.start()
        self._wait_for_waiters(q_pool, 2)

        # Kill q-eval so releasing h_q_eval quarantines it.
        q_eval.status = lambda: {"ok": False, "state": "poisoned", "error": "dead"}
        q_eval.reset = lambda kind="state": {"ok": False, "error": "respawn failed"}
        h_q_eval.release()
        t_strict.join(timeout=5.0)
        self.assertFalse(t_strict.is_alive())
        self.assertEqual(len(strict_errs), 1)
        self.assertIsInstance(strict_errs[0], PoolTimeoutError)
        # Borrower stays queued until q-dev is released, then borrows q-dev!
        self.assertTrue(t_borrow.is_alive())
        h_q_dev.release()
        t_borrow.join(timeout=5.0)
        self.assertFalse(t_borrow.is_alive())
        self.assertEqual(borrow_got, [("q-dev", True)])

        # Now hold q-dev, queue another borrower (timeout_s=None), and kill q-dev -> immediate wakeup!
        h_q_dev2 = q_pool.acquire("h-dev-2", role="dev", timeout_s=2.0)
        t_borrow_all_dead = threading.Thread(
            target=lambda: self._capture_error(
                lambda: q_pool.acquire("q-borrow-dead", role="eval", fallback_roles=("dev",), timeout_s=None),
                borrow_errs,
            )
        )
        t_borrow_all_dead.start()
        self._wait_for_waiters(q_pool, 1)
        q_dev.status = lambda: {"ok": False, "state": "poisoned", "error": "dev dead"}
        q_dev.reset = lambda kind="state": {"ok": False, "error": "dev respawn failed"}
        h_q_dev2.release()
        t_borrow_all_dead.join(timeout=5.0)
        self.assertFalse(t_borrow_all_dead.is_alive())
        self.assertEqual(len(borrow_errs), 1)
        self.assertIsInstance(borrow_errs[0], PoolTimeoutError)
        self.assertEqual(q_pool.metrics()["quarantined_slots"], ["q-eval", "q-dev"])

        # 7. SHA-256 trace & metrics verification on main pool.
        metrics = pool.metrics()
        self.assertEqual(metrics["total_leases"], 13)
        self.assertEqual(metrics["borrowed_leases"], 7)
        verified = verify_trace(trace_path, expected_head=metrics["trace_head"])
        self.assertTrue(verified["ok"], verified)
        self.assertEqual(verified["count"], 13)
        self.assertEqual(sum(1 for rec in verified["records"] if rec["borrowed"]), 7)

    def test_per_agent_fault_circuit_breaker_window_cooldown_and_queued_wakeup(self) -> None:
        """Golden regression for SandboxPool per-agent fault circuit breaker:
        1. Parameter validation rejects invalid max_agent_faults / window / cooldown combos.
        2. Normal fail-closed install refusals, clean caller exceptions, and pre-existing dirty
           slot preparation recoveries NEVER blame the acquiring agent.
        3. Repeated slot-destabilizing faults trip AgentQuarantinedError for only the rogue agent,
           protecting healthy teammates; reset_agent_circuit() re-enables cleanly.
        4. Sliding window (agent_fault_window_s) expires stale faults, and timed cooldown
           (agent_cooldown_s) auto-recovers with clean fault history.
        5. Atomic trip-before-handoff immediately rejects queued duplicate waiters from the
           tripped agent and hands the recovered slot to a waiting healthy teammate."""
        lab_a = self._make_toy_lab()
        lab_b = self._make_toy_lab()

        # 1. Input validation
        for bad_max in (0, -1, True, 1.5, "2"):
            with self.assertRaises((TypeError, ValueError)):
                SandboxPool([PoolSlot("s0", lab_a)], max_agent_faults=bad_max)  # type: ignore[arg-type]
        for bad_win in (0.0, -1.0, float("nan"), float("inf"), True, "5", 10**400):
            with self.assertRaises((TypeError, ValueError)):
                SandboxPool([PoolSlot("s0", lab_a)], max_agent_faults=2, agent_fault_window_s=bad_win)  # type: ignore[arg-type]
            with self.assertRaises((TypeError, ValueError)):
                SandboxPool([PoolSlot("s0", lab_a)], max_agent_faults=2, agent_cooldown_s=bad_win)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SandboxPool([PoolSlot("s0", lab_a)], agent_fault_window_s=10.0)
        with self.assertRaises(ValueError):
            SandboxPool([PoolSlot("s0", lab_a)], agent_cooldown_s=5.0)

        now = [100.0]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        trace_path = os.path.join(tmpdir.name, "circuit_breaker.jsonl")
        pool = SandboxPool(
            [PoolSlot("s0", lab_a), PoolSlot("s1", lab_b)],
            trace_path=trace_path,
            clock=lambda: now[0],
            max_agent_faults=2,
            agent_fault_window_s=10.0,
            agent_cooldown_s=None,
        )
        self.addCleanup(pool.close)

        with self.assertRaises(ValueError):
            pool.reset_agent_circuit("")
        self.assertFalse(pool.reset_agent_circuit("nobody"))

        # 2. Zero false positives: fail-closed install refusal, clean caller error, and inherited dirty slot.
        #    First poison lab_a externally BEFORE innocent acquires it so preparation recovers it.
        lab_a.session.fault("participant_failed", "pre-existing dirty state", "clean")
        res_inherited = pool.run_isolated("innocent", lambda lab, _l: lab.check(), timeout_s=2.0)
        self.assertTrue(res_inherited["ok"], res_inherited)
        self.assertTrue(res_inherited["recovered"])
        self.assertEqual(pool.metrics()["agent_fault_counts"], {})
        self.assertEqual(pool.metrics()["quarantined_agents"], [])

        # Fail-closed Session-level bad patch refusal (syntax error on toy.ops) + caller RuntimeError
        # leaving slot clean must NOT count as slot faults.
        bad_patch = {"files": {"toy.ops": "def broken(:\n"}}
        for _ in range(3):
            res_refusal = pool.run_isolated("innocent", lambda lab, _l: lab.install(bad_patch), timeout_s=2.0)
            self.assertTrue(res_refusal["ok"])
            self.assertFalse(res_refusal["result"]["ok"])
            res_exc = pool.run_isolated("innocent", lambda lab, _l: (_ for _ in ()).throw(RuntimeError("user bug")), timeout_s=2.0)
            self.assertFalse(res_exc["ok"])
            self.assertFalse(res_exc["recovered"])
        self.assertEqual(pool.metrics()["agent_fault_counts"], {})
        self.assertEqual(pool.metrics()["quarantined_agents"], [])

        # 3. Sliding window expires stale fault (t=100 then t=115 > 10s window -> does NOT trip).
        def poison_slot(lab, _l):
            lab.session.fault("participant_failed", "rogue crash", "clean")
            return "poisoned"

        r_f1 = pool.run_isolated("rogue", poison_slot, timeout_s=2.0)
        self.assertTrue(r_f1["ok"])
        self.assertTrue(r_f1["recovered"])
        self.assertEqual(pool.metrics()["agent_fault_counts"], {"rogue": 1})
        self.assertEqual(pool.metrics()["quarantined_agents"], [])

        now[0] = 115.0  # > 10.0 s window: first fault expires!
        r_f2 = pool.run_isolated("rogue", poison_slot, timeout_s=2.0)
        self.assertTrue(r_f2["recovered"])
        self.assertEqual(pool.metrics()["agent_fault_counts"], {"rogue": 1})
        self.assertEqual(pool.metrics()["quarantined_agents"], [])

        # Second fault within window (t=120, delta=5s <= 10s) TRIPS the circuit breaker!
        now[0] = 120.0
        r_f3 = pool.run_isolated("rogue", poison_slot, timeout_s=2.0)
        self.assertTrue(r_f3["recovered"])
        self.assertEqual(pool.metrics()["quarantined_agents"], ["rogue"])

        # Rogue is now rejected immediately on acquire() AND run_isolated(), while innocent keeps running!
        with self.assertRaises(AgentQuarantinedError) as ctx:
            pool.acquire("rogue", timeout_s=2.0)
        self.assertEqual(ctx.exception.agent_id, "rogue")
        self.assertEqual(ctx.exception.fault_count, 2)
        self.assertIsNone(ctx.exception.retry_after_s)

        r_rej = pool.run_isolated("rogue", lambda lab, _l: lab.check(), timeout_s=2.0)
        self.assertFalse(r_rej["ok"])
        self.assertIsNone(r_rej["slot_id"])
        self.assertIn("AgentQuarantinedError", r_rej["error"])
        self.assertEqual(pool.metrics()["circuit_breaker_rejections"], 2)

        r_healthy = pool.run_isolated("innocent", lambda lab, _l: lab.check(), timeout_s=2.0)
        self.assertTrue(r_healthy["ok"], r_healthy)

        # Manual reset_agent_circuit() re-enables rogue cleanly and returns False on second call.
        self.assertTrue(pool.reset_agent_circuit("rogue"))
        self.assertFalse(pool.reset_agent_circuit("rogue"))
        self.assertEqual(pool.metrics()["quarantined_agents"], [])
        self.assertEqual(pool.metrics()["agent_fault_counts"], {})
        self.assertTrue(pool.run_isolated("rogue", lambda lab, _l: lab.check(), timeout_s=2.0)["ok"])

        # 4. Timed cooldown auto-recovery & clean post-cooldown history + queued duplicate-waiter wakeup.
        lab_c = self._make_toy_lab()
        cd_now = [10.0]
        cd_pool = SandboxPool(
            [PoolSlot("sc", lab_c)],
            clock=lambda: cd_now[0],
            max_agent_faults=1,
            agent_cooldown_s=5.0,
        )
        self.addCleanup(cd_pool.close)

        # Hold sole slot as rogue-dup; queue rogue-dup (priority=10) AND healthy (priority=0).
        held = cd_pool.acquire("rogue-dup", timeout_s=2.0)
        rogue_errs: list[BaseException] = []
        healthy_got: list[str] = []

        t_rogue2 = threading.Thread(
            target=lambda: self._capture_error(
                lambda: cd_pool.acquire("rogue-dup", priority=10, timeout_s=2.0), rogue_errs
            )
        )
        t_healthy = threading.Thread(
            target=lambda: healthy_got.append(
                cd_pool.run_isolated("healthy-peer", lambda lab, _l: "ok", priority=0, timeout_s=2.0)["result"]
            )
        )
        t_rogue2.start()
        self._wait_for_waiters(cd_pool, 1)
        t_healthy.start()
        self._wait_for_waiters(cd_pool, 2)

        # Rogue poisons sole slot and releases -> circuit trips BEFORE handoff, rejecting t_rogue2
        # with AgentQuarantinedError and handing the recovered slot to healthy-peer!
        held.lab.session.fault("participant_failed", "boom", "clean")
        held.release()
        t_rogue2.join(timeout=5.0)
        t_healthy.join(timeout=5.0)
        self.assertFalse(t_rogue2.is_alive())
        self.assertFalse(t_healthy.is_alive())
        self.assertEqual(len(rogue_errs), 1)
        self.assertIsInstance(rogue_errs[0], AgentQuarantinedError)
        self.assertEqual(rogue_errs[0].retry_after_s, 5.0)
        self.assertEqual(healthy_got, ["ok"])
        self.assertEqual(cd_pool.metrics()["quarantined_agents"], ["rogue-dup"])

        # Advance clock past 5.0 s cooldown -> rogue-dup auto-recovers with clean fault history!
        cd_now[0] = 15.0
        self.assertEqual(cd_pool.metrics()["quarantined_agents"], [])
        self.assertEqual(cd_pool.metrics()["agent_fault_counts"], {})
        self.assertTrue(cd_pool.run_isolated("rogue-dup", lambda lab, _l: lab.check(), timeout_s=2.0)["ok"])

        # 5. Latched quarantine + sliding window: fault_count must NOT decay below max_agent_faults
        #    while still quarantined, and stale pre-reset timestamps must be ignored.
        now[0] = 200.0
        self.assertTrue(pool.run_isolated("latched-rogue", poison_slot, timeout_s=2.0)["recovered"])
        now[0] = 204.0
        self.assertTrue(pool.run_isolated("latched-rogue", poison_slot, timeout_s=2.0)["recovered"])
        self.assertEqual(pool.metrics()["quarantined_agents"], ["latched-rogue"])
        for future_t in (211.0, 215.0, 500.0):
            now[0] = future_t
            with self.assertRaises(AgentQuarantinedError) as ctx_l:
                pool.acquire("latched-rogue", timeout_s=2.0)
            self.assertEqual(ctx_l.exception.fault_count, 2)
            self.assertEqual(pool.metrics()["agent_fault_counts"].get("latched-rogue"), 2)
            self.assertEqual(pool.metrics()["quarantined_agents"], ["latched-rogue"])
        self.assertTrue(pool.reset_agent_circuit("latched-rogue"))
        # Stale fault timestamp (< reset_mono=500.0) must be ignored; fresh fault at 500.0 counts.
        with pool._lock:
            pool._record_agent_fault_locked("latched-rogue", 499.0)
        self.assertEqual(pool.metrics()["agent_fault_counts"], {})
        # Out-of-order timestamps inside window must expire properly when clock advances.
        lab_jit = self._make_toy_lab()
        jit_now = [120.0]
        jit_pool = SandboxPool(
            [PoolSlot("sj", lab_jit)],
            clock=lambda: jit_now[0],
            max_agent_faults=3,
            agent_fault_window_s=10.0,
        )
        self.addCleanup(jit_pool.close)
        with jit_pool._lock:
            jit_pool._record_agent_fault_locked("jitter", 120.0)
            jit_pool._record_agent_fault_locked("jitter", 100.0)
        jit_now[0] = 112.0  # 100.0 expired (delta 12s > 10s), while 120.0 remains (clamped delta 0s <= 10s)
        self.assertEqual(jit_pool.metrics()["agent_fault_counts"], {"jitter": 1})
        self.assertEqual(jit_pool.metrics()["quarantined_agents"], [])
        now[0] = 500.0

        # 6. Float64 ULP absorption (clock=1e9, cooldown=1e-8) must NOT self-expire or turn a
        #    queued duplicate waiter into PoolTimeoutError.
        lab_ulp = self._make_toy_lab()
        ulp_now = [1e9]
        ulp_pool = SandboxPool(
            [PoolSlot("su", lab_ulp)],
            clock=lambda: ulp_now[0],
            max_agent_faults=1,
            agent_cooldown_s=1e-8,
        )
        self.addCleanup(ulp_pool.close)
        held_ulp = ulp_pool.acquire("ulp-rogue", timeout_s=2.0)
        ulp_errs: list[BaseException] = []
        t_ulp_wait = threading.Thread(
            target=lambda: self._capture_error(
                lambda: ulp_pool.acquire("ulp-rogue", timeout_s=None), ulp_errs
            )
        )
        t_ulp_wait.start()
        self._wait_for_waiters(ulp_pool, 1)
        held_ulp.lab.session.fault("participant_failed", "ulp-boom", "clean")
        held_ulp.release()
        t_ulp_wait.join(timeout=5.0)
        self.assertFalse(t_ulp_wait.is_alive())
        self.assertEqual(len(ulp_errs), 1)
        self.assertIsInstance(ulp_errs[0], AgentQuarantinedError)
        self.assertEqual(ulp_pool.metrics()["quarantined_agents"], ["ulp-rogue"])
        ulp_now[0] = 1e9 + 1.0
        self.assertEqual(ulp_pool.metrics()["quarantined_agents"], [])

        # 7. Pre-acquire TOCTOU: concurrent lease from same agent trips circuit while a second
        #    acquire() is in pre-acquire cleanup -> returns clean slot and raises AgentQuarantinedError.
        lab_toctou_a = self._make_toy_lab()
        lab_toctou_b = self._make_toy_lab()
        prep_entered = threading.Event()
        prep_release = threading.Event()
        orig_reset = lab_toctou_b.reset
        blocked_once = [False]

        def slow_prep_reset(kind: str = "state") -> dict[str, Any]:
            if not blocked_once[0]:
                blocked_once[0] = True
                prep_entered.set()
                prep_release.wait(timeout=5.0)
            return orig_reset(kind)

        lab_toctou_b.reset = slow_prep_reset  # type: ignore[method-assign]
        toctou_pool = SandboxPool(
            [PoolSlot("ta", lab_toctou_a), PoolSlot("tb", lab_toctou_b)],
            max_agent_faults=1,
            agent_cooldown_s=None,
        )
        self.addCleanup(toctou_pool.close)
        lease_a = toctou_pool.acquire("racing-rogue", timeout_s=2.0)
        toctou_errs: list[BaseException] = []
        t_racing = threading.Thread(
            target=lambda: self._capture_error(
                lambda: toctou_pool.acquire("racing-rogue", timeout_s=2.0), toctou_errs
            )
        )
        t_racing.start()
        self.assertTrue(prep_entered.wait(timeout=5.0))
        lease_a.lab.session.fault("participant_failed", "trip-during-prep", "clean")
        lease_a.release()
        prep_release.set()
        t_racing.join(timeout=5.0)
        self.assertFalse(t_racing.is_alive())
        self.assertEqual(len(toctou_errs), 1)
        self.assertIsInstance(toctou_errs[0], AgentQuarantinedError)
        # Returned slot tb must remain healthy and immediately usable by a healthy agent.
        self.assertTrue(toctou_pool.run_isolated("healthy-after-toctou", lambda lab, _l: lab.check(), timeout_s=2.0)["ok"])

        # 8. Undrained fenced call exceeding drain_timeout_s attributes a slot fault to the offending agent.
        lab_hang = self._make_toy_lab()
        hang_started = threading.Event()
        hang_release = threading.Event()
        orig_check = lab_hang.check

        def hanging_check(workload: Any = None) -> dict[str, Any]:
            hang_started.set()
            hang_release.wait(timeout=5.0)
            return orig_check(workload)

        lab_hang.check = hanging_check  # type: ignore[method-assign]
        hang_pool = SandboxPool(
            [PoolSlot("sh", lab_hang)],
            drain_timeout_s=0.05,
            max_agent_faults=1,
            agent_cooldown_s=None,
        )
        self.addCleanup(hang_pool.close)
        hang_lease = hang_pool.acquire("hang-rogue", timeout_s=2.0)
        t_hang = threading.Thread(target=lambda: self._capture_error(lambda: hang_lease.lab.check(), []))
        t_hang.start()
        self.assertTrue(hang_started.wait(timeout=5.0))
        hang_lease.release()
        hang_release.set()
        t_hang.join(timeout=5.0)
        self.assertEqual(hang_pool.metrics()["quarantined_agents"], ["hang-rogue"])
        self.assertEqual(hang_pool.metrics()["quarantined_slots"], ["sh"])

        # 9. SHA-256 trace verification on primary pool.
        metrics = pool.metrics()
        verified = verify_trace(trace_path, expected_head=metrics["trace_head"])
        self.assertTrue(verified["ok"], verified)
        self.assertEqual(verified["count"], metrics["total_leases"])

    def test_max_slots_per_agent_blast_radius_fairness_and_cross_role_wakeup(self) -> None:
        # 1. Input validation for max_slots_per_agent.
        for bad_val, expected_exc in (
            (True, TypeError),
            (False, TypeError),
            (1.5, TypeError),
            ("1", TypeError),
            (0, ValueError),
            (-2, ValueError),
        ):
            with self.assertRaises(expected_exc):
                SandboxPool([PoolSlot("v", self._make_toy_lab())], max_slots_per_agent=bad_val)  # type: ignore[arg-type]

        # 2. Concurrent-rogue blast-radius containment (C=3, max_slots_per_agent=1, max_agent_faults=1):
        #    3 parallel threads from "rogue" attack simultaneously while 2 healthy agents run live edits.
        #    Without max_slots_per_agent=1, rogue would claim and poison all 3 slots concurrently before
        #    lease #1 released to trip max_agent_faults=1. With cap=1, blast radius is strictly 1 slot.
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        trace_path = os.path.join(tmpdir.name, "cap_blast_radius.jsonl")
        pool = SandboxPool(
            [PoolSlot(f"g{i}", self._make_toy_lab()) for i in range(3)],
            trace_path=trace_path,
            max_slots_per_agent=1,
            max_agent_faults=1,
            agent_cooldown_s=None,
        )
        self.addCleanup(pool.close)
        rogue_entered = threading.Event()
        rogue_release = threading.Event()
        rogue_lock = threading.Lock()
        rogue_in_flight = [0]
        rogue_max_concurrent = [0]
        rogue_results: list[dict[str, Any]] = []

        def rogue_attack(idx: int) -> None:
            def _action(lab, _lease):
                with rogue_lock:
                    rogue_in_flight[0] += 1
                    rogue_max_concurrent[0] = max(rogue_max_concurrent[0], rogue_in_flight[0])
                rogue_entered.set()
                rogue_release.wait(timeout=5.0)
                lab.session.fault("participant_failed", f"rogue-burst-{idx}", "clean")
                with rogue_lock:
                    rogue_in_flight[0] -= 1
                raise RuntimeError(f"rogue-burst-{idx}")

            res = pool.run_isolated("rogue", _action, idea_id=f"rogue-{idx}", timeout_s=5.0)
            with rogue_lock:
                rogue_results.append(res)

        rogue_threads = [threading.Thread(target=rogue_attack, args=(i,)) for i in range(3)]
        for th in rogue_threads:
            th.start()
        self.assertTrue(rogue_entered.wait(timeout=5.0))
        # Give sibling rogue threads time to enqueue behind the cap; verify live active-slot count == 1
        # and that healthy agents immediately claim the other 2 slots with zero head-of-line blocking.
        deadline = time.monotonic() + 2.0
        while len(pool._waiters) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(pool.metrics()["agent_active_slots"], {"rogue": 1})
        self.assertEqual(pool.metrics()["max_slots_per_agent"], 1)

        healthy_ok: list[bool] = []
        h_lock = threading.Lock()

        def run_healthy(aid: str) -> None:
            r = pool.run_isolated(aid, lambda lab, _l: lab.check(), idea_id=f"{aid}-edit", timeout_s=2.0)
            with h_lock:
                healthy_ok.append(bool(r["ok"]))

        h_threads = [threading.Thread(target=run_healthy, args=(f"healthy-{i}",)) for i in range(2)]
        for th in h_threads:
            th.start()
        for th in h_threads:
            th.join(timeout=5.0)
        self.assertEqual(healthy_ok, [True, True])

        # Release rogue lease #1: trips circuit breaker and rejects both queued rogue siblings in-lock!
        rogue_release.set()
        for th in rogue_threads:
            th.join(timeout=5.0)
            self.assertFalse(th.is_alive())
        self.assertEqual(rogue_max_concurrent[0], 1)
        m = pool.metrics()
        self.assertEqual(m["recoveries"], 1)
        self.assertEqual(m["circuit_breaker_rejections"], 2)
        self.assertEqual(m["quarantined_agents"], ["rogue"])
        self.assertEqual(m["agent_active_slots"], {})
        self.assertEqual(sum(1 for r in rogue_results if "AgentQuarantinedError" in str(r.get("error", ""))), 2)
        vt = verify_trace(trace_path, expected_head=m["trace_head"])
        self.assertTrue(vt["ok"] and vt["count"] == m["total_leases"], vt)

        # 3. Fairness: capped high-priority waiter must NOT head-of-line block uncapped low-priority agent;
        #    timeout_s=0.0 at cap fails fast with PoolTimeoutError.
        fair_pool = SandboxPool(
            [PoolSlot("s0", self._make_toy_lab()), PoolSlot("s1", self._make_toy_lab())],
            max_slots_per_agent=1,
        )
        self.addCleanup(fair_pool.close)
        g_lease = fair_pool.acquire("greedy", timeout_s=2.0)
        with self.assertRaises(PoolTimeoutError):
            fair_pool.acquire("greedy", timeout_s=0.0)
        g2_acquired = threading.Event()
        g2_errors: list[BaseException] = []

        def greedy_second() -> None:
            try:
                with fair_pool.lease("greedy", priority=10, timeout_s=5.0):
                    g2_acquired.set()
            except BaseException as err:
                g2_errors.append(err)

        t_g2 = threading.Thread(target=greedy_second)
        t_g2.start()
        deadline = time.monotonic() + 2.0
        while len(fair_pool._waiters) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(g2_acquired.is_set())
        # Uncapped low-priority agent immediately claims idle s1 despite greedy's priority=10 waiter!
        with fair_pool.lease("fair-agent", priority=0, timeout_s=2.0) as fl:
            self.assertEqual(fl.slot_id, "s1")
            self.assertEqual(fair_pool.metrics()["agent_active_slots"], {"fair-agent": 1, "greedy": 1})
        g_lease.release()
        t_g2.join(timeout=5.0)
        self.assertFalse(t_g2.is_alive())
        self.assertEqual(g2_errors, [])
        self.assertTrue(g2_acquired.is_set())
        self.assertEqual(fair_pool.metrics()["agent_active_slots"], {})

        # 4. Cross-role idle-slot wakeup (clean release AND quarantined release):
        #    agent-x holds role="dev" while waiting for idle role="eval"; releasing dev MUST wake eval!
        for quarantine_dev in (False, True):
            lab_dev = self._make_toy_lab()
            if quarantine_dev:
                orig_dev_reset = lab_dev.reset
                lab_dev.reset = lambda kind="state", _orig=orig_dev_reset: (  # type: ignore[method-assign]
                    {"ok": False, "error": "forced-respawn-fail"} if kind == "respawn" else _orig(kind)
                )
            role_pool = SandboxPool(
                [PoolSlot("s-dev", lab_dev, role="dev"), PoolSlot("s-eval", self._make_toy_lab(), role="eval")],
                max_slots_per_agent=1,
            )
            self.addCleanup(role_pool.close)
            dev_lease = role_pool.acquire("agent-x", role="dev", timeout_s=2.0)
            eval_got: list[str] = []
            eval_errs: list[BaseException] = []

            def wait_eval() -> None:
                try:
                    with role_pool.lease("agent-x", role="eval", timeout_s=3.0) as el:
                        eval_got.append(el.slot_id)
                except BaseException as err:
                    eval_errs.append(err)

            t_ev = threading.Thread(target=wait_eval)
            t_ev.start()
            deadline = time.monotonic() + 2.0
            while len(role_pool._waiters) < 1 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(eval_got, [])
            if quarantine_dev:
                dev_lease.lab.session.fault("participant_failed", "kill-dev", "clean")
            dev_lease.release()
            t_ev.join(timeout=5.0)
            self.assertFalse(t_ev.is_alive())
            self.assertEqual(eval_errs, [], f"quarantine_dev={quarantine_dev}")
            self.assertEqual(eval_got, ["s-eval"], f"quarantine_dev={quarantine_dev}")
            self.assertEqual(role_pool.metrics()["agent_active_slots"], {})
            if quarantine_dev:
                self.assertEqual(role_pool.metrics()["quarantined_slots"], ["s-dev"])

        # 5. Pre-acquire failure accounting zero-leak: failed pre-acquire recovery must release agent permit.
        lab_bad = self._make_toy_lab()
        lab_bad.reset = lambda kind="state": {"ok": False, "error": "prep-fail"}  # type: ignore[method-assign]
        lab_bad.session.fault("participant_failed", "pre-poisoned", "clean")
        lab_good = self._make_toy_lab()
        prep_pool = SandboxPool(
            [PoolSlot("bad", lab_bad), PoolSlot("good", lab_good)],
            max_slots_per_agent=1,
        )
        self.addCleanup(prep_pool.close)
        # First claim hits bad (quarantines it), loops cleanly to good without leaking agent permit!
        with prep_pool.lease("agent-p", timeout_s=2.0) as pl:
            self.assertEqual(pl.slot_id, "good")
            self.assertEqual(prep_pool.metrics()["agent_active_slots"], {"agent-p": 1})
        self.assertEqual(prep_pool.metrics()["agent_active_slots"], {})
        self.assertEqual(prep_pool.metrics()["quarantined_slots"], ["bad"])

        # 6. Regression for P1-1: global priority across idle slots (no cross-role priority inversion).
        #    agent-prio holds s-dev while queueing (dev, priority=0) then (eval, priority=100) with s-eval idle.
        #    Releasing s-dev MUST wake (eval, priority=100) FIRST instead of re-assigning s-dev to priority=0.
        prio_pool = SandboxPool(
            [PoolSlot("s-dev", self._make_toy_lab(), role="dev"), PoolSlot("s-eval", self._make_toy_lab(), role="eval")],
            max_slots_per_agent=1,
        )
        self.addCleanup(prio_pool.close)
        p_dev = prio_pool.acquire("agent-prio", role="dev", timeout_s=2.0)
        prio_order: list[tuple[int, str]] = []
        prio_lock = threading.Lock()
        first_done = threading.Event()

        def prio_waiter(role_name: str, prio: int) -> None:
            with prio_pool.lease("agent-prio", role=role_name, priority=prio, timeout_s=3.0) as lease:
                with prio_lock:
                    prio_order.append((prio, lease.slot_id))
                    if len(prio_order) == 1:
                        first_done.set()

        t_low = threading.Thread(target=prio_waiter, args=("dev", 0))
        t_high = threading.Thread(target=prio_waiter, args=("eval", 100))
        t_low.start()
        deadline = time.monotonic() + 2.0
        while len(prio_pool._waiters) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        t_high.start()
        while len(prio_pool._waiters) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        p_dev.release()
        t_high.join(timeout=5.0)
        t_low.join(timeout=5.0)
        self.assertEqual(prio_order, [(100, "s-eval"), (0, "s-dev")])
        self.assertEqual(prio_pool.metrics()["agent_active_slots"], {})

        # 7. Regression for P1-2: primary-before-borrow when releasing a fallback slot while primary is idle.
        pb_pool = SandboxPool(
            [PoolSlot("s-dev", self._make_toy_lab(), role="dev"), PoolSlot("s-eval", self._make_toy_lab(), role="eval")],
            max_slots_per_agent=1,
        )
        self.addCleanup(pb_pool.close)
        pb_dev = pb_pool.acquire("agent-pb", role="dev", timeout_s=2.0)
        pb_got: list[tuple[str, bool]] = []

        def pb_waiter() -> None:
            with pb_pool.lease("agent-pb", role="eval", fallback_roles=("dev",), timeout_s=3.0) as lease:
                pb_got.append((lease.slot_id, lease.borrowed))

        t_pb = threading.Thread(target=pb_waiter)
        t_pb.start()
        deadline = time.monotonic() + 2.0
        while len(pb_pool._waiters) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        pb_dev.release()
        t_pb.join(timeout=5.0)
        self.assertEqual(pb_got, [("s-eval", False)])

        # 8. Regression for P2-1: caller-specified fallback_roles order + uncapped borrower bypassing capped waiter.
        fb_pool = SandboxPool(
            [
                PoolSlot("s-c", self._make_toy_lab(), role="c"),
                PoolSlot("s-b", self._make_toy_lab(), role="b"),
                PoolSlot("s-a", self._make_toy_lab(), role="a"),
            ],
            max_slots_per_agent=1,
        )
        self.addCleanup(fb_pool.close)
        fb_hold = fb_pool.acquire("agent-fb", role="c", timeout_s=2.0)
        fb_got: list[tuple[str, bool]] = []

        def fb_waiter() -> None:
            with fb_pool.lease("agent-fb", role="missing", fallback_roles=("a", "b"), priority=50, timeout_s=3.0) as lease:
                fb_got.append((lease.slot_id, lease.borrowed))

        t_fb = threading.Thread(target=fb_waiter)
        t_fb.start()
        deadline = time.monotonic() + 2.0
        while len(fb_pool._waiters) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        # Uncapped borrower (priority=0) must NOT be blocked on idle s-a by capped agent-fb (priority=50).
        with fb_pool.lease("uncapped-borrower", role="missing", fallback_roles=("a",), priority=0, timeout_s=2.0) as ub:
            self.assertEqual((ub.slot_id, ub.borrowed), ("s-a", True))
        # After uncapped-borrower releases s-a and agent-fb releases s-c, agent-fb MUST pick "a" over "b"!
        fb_hold.release()
        t_fb.join(timeout=5.0)
        self.assertEqual(fb_got, [("s-a", True)])
        self.assertEqual(fb_pool.metrics()["agent_active_slots"], {})

        # 9. max_slots_per_agent > 1 (cap=2 on C=3), timeout_s=inf at cap, priority_aging_s interaction,
        #    reset_agent_circuit() preserving active permits, post-handoff cancel, and close(close_labs=True).
        now_t = [100.0]
        cap2_pool = SandboxPool(
            [PoolSlot(f"c{i}", self._make_toy_lab()) for i in range(3)],
            clock=lambda: now_t[0],
            priority_aging_s=1.0,
            max_slots_per_agent=2,
            max_agent_faults=2,
            close_labs=True,
        )
        self.addCleanup(cap2_pool.close)
        l1 = cap2_pool.acquire("multi", timeout_s=2.0)
        l2 = cap2_pool.acquire("multi", timeout_s=2.0)
        self.assertEqual(cap2_pool.metrics()["agent_active_slots"], {"multi": 2})
        # 3rd concurrent acquire with timeout_s=inf blocks behind cap=2 while uncapped peer grabs c2.
        l3_got: list[str] = []

        def multi_third() -> None:
            with cap2_pool.lease("multi", priority=0, timeout_s=float("inf")) as l3:
                l3_got.append(l3.slot_id)

        t_m3 = threading.Thread(target=multi_third)
        t_m3.start()
        deadline = time.monotonic() + 2.0
        while len(cap2_pool._waiters) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        now_t[0] += 50.0  # Aged capped waiter has huge effective priority, still must NOT block uncapped peer!
        with cap2_pool.lease("peer", priority=0, timeout_s=2.0) as pl:
            self.assertEqual(pl.slot_id, "c2")
        self.assertFalse(cap2_pool.reset_agent_circuit("multi"))
        self.assertEqual(cap2_pool.metrics()["agent_active_slots"], {"multi": 2})
        l1.release()
        t_m3.join(timeout=5.0)
        self.assertEqual(len(l3_got), 1)
        self.assertEqual(cap2_pool.metrics()["agent_active_slots"], {"multi": 1})
        l2.release()
        self.assertEqual(cap2_pool.metrics()["agent_active_slots"], {})

        # Post-handoff cancellation (_cancel_waiter_locked with assigned_state != None) leaks zero permits.
        cancel_pool = SandboxPool(
            [PoolSlot("k0", self._make_toy_lab())],
            max_slots_per_agent=1,
            close_labs=True,
        )
        k_hold = cancel_pool.acquire("agent-k", timeout_s=2.0)
        from rsibox.pool import _Waiter
        fake_waiter = _Waiter(None, priority=0, agent_id="agent-k")
        with cancel_pool._lock:
            cancel_pool._waiters.append(fake_waiter)
        k_hold.release()
        self.assertIsNotNone(fake_waiter.assigned_state)
        self.assertEqual(cancel_pool.metrics()["agent_active_slots"], {"agent-k": 1})
        with cancel_pool._lock:
            cancel_pool._cancel_waiter_locked(fake_waiter)
        self.assertEqual(cancel_pool.metrics()["agent_active_slots"], {})
        # Close with active lease and queued capped waiter must close cleanly with zero leaked permits.
        k_hold2 = cancel_pool.acquire("agent-k", timeout_s=2.0)
        closed_errs: list[BaseException] = []
        t_closed = threading.Thread(
            target=lambda: self._capture_error(lambda: cancel_pool.acquire("agent-k", timeout_s=5.0), closed_errs)
        )
        t_closed.start()
        deadline = time.monotonic() + 2.0
        while len(cancel_pool._waiters) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        cancel_pool.close(close_labs=True)
        t_closed.join(timeout=5.0)
        self.assertEqual(len(closed_errs), 1)
        self.assertIsInstance(closed_errs[0], PoolClosedError)
        k_hold2.release()
        self.assertEqual(cancel_pool.metrics()["agent_active_slots"], {})

    @staticmethod
    def _capture_error(operation, bucket: list[BaseException]) -> None:
        try:
            operation()
        except BaseException as error:
            bucket.append(error)


if __name__ == "__main__":
    unittest.main()
