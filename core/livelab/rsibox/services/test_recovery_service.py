"""Unit, composition, and concurrency tests for `rsibox.services.recovery_service` (S6).

Pure CPU / stdlib tests — zero GPU, torch, numpy, or engine imports.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Sequence

LIVELAB_DIR = Path(__file__).resolve().parents[2]
if str(LIVELAB_DIR) not in sys.path:
    sys.path.insert(0, str(LIVELAB_DIR))

from kernel import coordinator as kernel_coordinator  # noqa: E402
from rsibox import BaseLab, PoolSlot, SandboxPool, SlotLease  # noqa: E402
from rsibox.autoscaler import AutoScalerPolicy, PoolAutoScaler  # noqa: E402
from rsibox.services import (  # noqa: E402
    EndpointNotFoundError,
    EscalationFailedError,
    RecoveryClosedError,
    RecoveryInProgressError,
    RecoveryReceipt,
    RecoveryService,
    RecoveryServiceError,
    RecoveryTier,
    ServiceHealth,
    ServiceHealthStatus,
)
from rsibox.warm import (  # noqa: E402
    AsyncShadowReplenisher,
    DefaultRespawnDriver,
    DiskStore,
    DynamoK8sSnapshotDriver,
    LaunchDelta,
    QuiesceReceipt,
    StandbyShadowDriver,
    StoreProbe,
)


class _StubLab(BaseLab):
    """Minimal CPU `BaseLab` tracking spawn/stop counts and poison injection."""

    name = "stub_lab"

    def __init__(self, *, poison_on_spawn: bool = False, **kwargs: Any) -> None:
        self.spawn_calls = 0
        self.stop_calls = 0
        self.closed = False
        self.poison_on_spawn = poison_on_spawn
        super().__init__(auto_spawn=True, **kwargs)

    def _start_engine(self):
        self.spawn_calls += 1
        participant = kernel_coordinator.LocalParticipant("worker-0")
        coord = kernel_coordinator.StepGapCoordinator([participant], contextlib.nullcontext)
        return coord, []

    def _spawn(self) -> None:
        super()._spawn()
        if self.poison_on_spawn:
            self.session.fault("ranks_disagree", "poisoned", "clean")

    def _stop_engine(self) -> None:
        self.stop_calls += 1

    def _run_workload(self, workload: Any) -> dict[str, Any]:
        return {"ok": True, "workload": workload}

    def close(self) -> dict[str, Any]:
        self.closed = True
        return super().close()


class _StubStore:
    name = "stub_store"

    def __init__(self, *, probe_ok: bool = True, error: str | None = None) -> None:
        self.probe_ok = probe_ok
        self.error = error
        self.probe_calls = 0
        self.stop_calls = 0

    def ensure_ready(self, **_kw: Any) -> StoreProbe:
        return StoreProbe(ok=self.probe_ok, backend=self.name, error=self.error)

    def probe(
        self,
        *,
        gpus: Sequence[str] = (),
        visible_devices: str | None = None,
        tp: int | None = None,
        expected_stamp: dict[str, Any] | None = None,
    ) -> StoreProbe:
        self.probe_calls += 1
        return StoreProbe(ok=self.probe_ok, backend=self.name, error=self.error)

    def launch_delta(self, **_kw: Any) -> LaunchDelta:
        return LaunchDelta()

    def stop(self) -> None:
        self.stop_calls += 1


class _StubHooks:
    mutability = "readonly_weights"

    def __init__(self, *, ok: bool = True, fail_warmup: bool = False) -> None:
        self.ok = ok
        self.fail_warmup = fail_warmup
        self.events: list[str] = []

    def warmup_for_snapshot(self) -> None:
        self.events.append("warmup")
        if self.fail_warmup:
            raise RuntimeError("warmup cuda error")

    def quiesce_for_snapshot(self) -> QuiesceReceipt:
        self.events.append("quiesce")
        if not self.ok:
            return QuiesceReceipt(ok=False, error="quiesce drain failed")
        return QuiesceReceipt(ok=True, released=("kv_pool",), token="snap-tok-1")

    def resume_after_restore(self, receipt: QuiesceReceipt) -> dict[str, Any]:
        self.events.append(f"resume:{receipt.token}")
        return {"ok": True}


class RecoveryServiceValidationTests(unittest.TestCase):
    def test_input_validation_and_error_guards(self) -> None:
        with RecoveryService(quarantine_threshold=2) as svc:
            for bad_thresh in (True, False, 0, -1, "2"):
                with self.assertRaises((TypeError, ValueError)):
                    RecoveryService(quarantine_threshold=bad_thresh)  # type: ignore[arg-type]

            for bad_str in ("", "   ", None, 123):
                with self.assertRaises(ValueError):
                    svc.register_endpoint(bad_str, engine="vllm")  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    svc.register_endpoint("ep-bad-eng", engine=bad_str)  # type: ignore[arg-type]

            with self.assertRaises(ValueError):
                svc.register_endpoint("ep-shadow-nodrv", engine="vllm", tier=RecoveryTier.STANDBY_SHADOW)
            with self.assertRaises(TypeError):
                svc.register_endpoint("ep-bad-drv", engine="vllm", driver="not-a-driver")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.register_endpoint("ep-bad-hooks", engine="vllm", quiesce_hooks=object())  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.register_endpoint("ep-bool-own", engine="vllm", own_driver="yes")  # type: ignore[arg-type]

            h = svc.register_endpoint("ep-1", engine="vllm", lab_factory=_StubLab)
            self.assertIsInstance(h, ServiceHealth)
            self.assertEqual(h.status, ServiceHealthStatus.HEALTHY)
            with self.assertRaises(RecoveryServiceError):
                svc.register_endpoint("ep-1", engine="vllm")
            with self.assertRaises(EndpointNotFoundError):
                svc.get_health("missing")
            with self.assertRaises(EndpointNotFoundError):
                svc.record_failure("missing", "boom")
            with self.assertRaises(ValueError):
                svc.record_failure("ep-1", "   ")
            with self.assertRaises(TypeError):
                svc.recover("ep-1", allow_fallback="yes")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                svc.attach_pool("p1", None)  # type: ignore[arg-type]


class RecoveryServiceFailoverAndConcurrencyTests(unittest.TestCase):
    def test_single_flight_same_endpoint_and_parallel_distinct_endpoints(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class _BlockingDriver:
            name = "default_respawn"

            def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                entered.set()
                release.wait(timeout=5.0)
                return DefaultRespawnDriver().respawn(lab, store, reason=reason)

        with RecoveryService() as svc:
            svc.register_endpoint("ep-sf", engine="sglang", lab_factory=_StubLab, driver=_BlockingDriver())
            results: list[RecoveryReceipt] = []
            errors: list[BaseException] = []

            def run_first() -> None:
                try:
                    results.append(svc.recover("ep-sf", reason="oom"))
                except BaseException as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=run_first)
            t1.start()
            self.assertTrue(entered.wait(timeout=2.0))
            self.assertEqual(svc.get_health("ep-sf").status, ServiceHealthStatus.RECOVERING)
            with self.assertRaises(RecoveryInProgressError):
                svc.recover("ep-sf", reason="duplicate")
            release.set()
            t1.join(timeout=2.0)
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            self.assertEqual(results[0].incarnation, 2)

            # Distinct endpoints must recover concurrently (lock released across driver.respawn).
            barrier = threading.Barrier(2, timeout=3.0)

            class _BarrierDriver:
                name = "default_respawn"

                def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                    barrier.wait()
                    return {"ok": True, "driver": self.name, "reason": reason}

            svc.register_endpoint("ep-a", engine="vllm", driver=_BarrierDriver())
            svc.register_endpoint("ep-b", engine="megatron", driver=_BarrierDriver())
            pair_errors: list[BaseException] = []

            def run_pair(eid: str) -> None:
                try:
                    rec = svc.recover(eid)
                    self.assertTrue(rec.ok)
                except BaseException as exc:
                    pair_errors.append(exc)

            ta = threading.Thread(target=run_pair, args=("ep-a",))
            tb = threading.Thread(target=run_pair, args=("ep-b",))
            ta.start()
            tb.start()
            ta.join(timeout=4.0)
            tb.join(timeout=4.0)
            self.assertEqual(pair_errors, [])

    def test_standby_shadow_fallback_to_cold_respawn_and_quarantine(self) -> None:
        with RecoveryService(quarantine_threshold=2) as svc:
            shadow_driver = StandbyShadowDriver(
                promote_fn=lambda lab, store: {"ok": False, "error": "shadow_promote_failed"},
                ready_count_fn=lambda: 0,
            )
            svc.register_endpoint(
                "ep-failover",
                engine="vllm",
                lab_factory=_StubLab,
                tier=RecoveryTier.STANDBY_SHADOW,
                driver=shadow_driver,
                fallback_driver=DefaultRespawnDriver(),
            )
            # Without fallback allowed, recovery returns ok=False and degrades endpoint.
            no_fb = svc.recover("ep-failover", reason="fault", allow_fallback=False)
            self.assertFalse(no_fb.ok)
            self.assertFalse(no_fb.fallback_used)
            self.assertIsNone(no_fb.tier_succeeded)
            self.assertEqual(svc.get_health("ep-failover").status, ServiceHealthStatus.DEGRADED)
            self.assertEqual(svc.get_health("ep-failover").consecutive_failures, 1)

            # With fallback allowed, StandbyShadowDriver fails over to DefaultRespawnDriver.
            rec = svc.recover("ep-failover", reason="fault", allow_fallback=True)
            self.assertTrue(rec.ok)
            self.assertTrue(rec.fallback_used)
            self.assertEqual(rec.tier_attempted, RecoveryTier.STANDBY_SHADOW)
            self.assertEqual(rec.tier_succeeded, RecoveryTier.COLD_RESPAWN)
            self.assertEqual(rec.incarnation, 2)
            self.assertIn("standby_not_ready", rec.error or "")
            health = svc.get_health("ep-failover")
            self.assertEqual(health.status, ServiceHealthStatus.HEALTHY)
            self.assertEqual(health.consecutive_failures, 0)
            self.assertIn("standby_not_ready", health.last_error or "")

            # All tiers failing raises EscalationFailedError and quarantines at threshold=2.
            poisoned_lab = _StubLab(poison_on_spawn=True)
            svc.register_endpoint(
                "ep-all-fail",
                engine="sglang",
                lab=poisoned_lab,
                tier=RecoveryTier.STANDBY_SHADOW,
                driver=shadow_driver,
                fallback_driver=DefaultRespawnDriver(),
            )
            with self.assertRaises(EscalationFailedError):
                svc.recover("ep-all-fail")
            self.assertEqual(svc.get_health("ep-all-fail").status, ServiceHealthStatus.DEGRADED)
            with self.assertRaises(EscalationFailedError):
                svc.recover("ep-all-fail")
            q_health = svc.get_health("ep-all-fail")
            self.assertEqual(q_health.status, ServiceHealthStatus.QUARANTINED)
            self.assertEqual(q_health.consecutive_failures, 2)
            self.assertIn("primary (standby_shadow) failed", q_health.last_error or "")
            self.assertIn("fallback (cold_respawn) failed", q_health.last_error or "")

    def test_record_failure_and_reset_after_false(self) -> None:
        with RecoveryService(quarantine_threshold=2) as svc:
            svc.register_endpoint("ep-rf", engine="verl", lab_factory=_StubLab)
            h1 = svc.record_failure("ep-rf", RuntimeError("nccl timeout"))
            self.assertEqual(h1.status, ServiceHealthStatus.DEGRADED)
            self.assertEqual(h1.consecutive_failures, 1)
            self.assertIn("RuntimeError: nccl timeout", h1.last_error or "")

            h2 = svc.record_failure("ep-rf", "cuda oom")
            self.assertEqual(h2.status, ServiceHealthStatus.QUARANTINED)
            self.assertEqual(h2.consecutive_failures, 2)

            rec = svc.recover("ep-rf", reset_after=False)
            self.assertTrue(rec.ok)
            self.assertEqual(svc.get_health("ep-rf").status, ServiceHealthStatus.QUARANTINED)
            self.assertEqual(svc.get_health("ep-rf").consecutive_failures, 2)

            rec2 = svc.recover("ep-rf", reset_after=True)
            self.assertTrue(rec2.ok)
            self.assertEqual(svc.get_health("ep-rf").status, ServiceHealthStatus.HEALTHY)
            self.assertEqual(svc.get_health("ep-rf").consecutive_failures, 0)


class RecoveryServicePoolQuiesceAndTeardownTests(unittest.TestCase):
    def test_pool_autoscaler_health_aggregation_and_owner_scoped_teardown(self) -> None:
        lab_owned = _StubLab()
        lab_foreign = _StubLab()
        owned_pool = SandboxPool([PoolSlot("own-0", lab_owned, role="eval")], close_labs=True)
        foreign_pool = SandboxPool([PoolSlot("ext-0", lab_foreign, role="dev")], close_labs=True)
        owned_scaler = PoolAutoScaler(
            owned_pool,
            policy=AutoScalerPolicy(min_slots=1, max_slots=1, poll_interval_s=0.05),
        )
        owned_scaler.start()
        owned_store = _StubStore(probe_ok=True)
        foreign_store = _StubStore(probe_ok=False, error="stale gms socket")
        replenisher = AsyncShadowReplenisher(lambda _lab, _store: {"ok": True})
        shadow_drv = StandbyShadowDriver(
            promote_fn=lambda lab, store: {"ok": True, "reset": ["shadow promoted"]},
            replenish_fn=replenisher.trigger,
            ready_count_fn=lambda: 1,
        )

        svc = RecoveryService()
        svc.register_endpoint(
            "ep-owned",
            engine="vllm",
            lab=lab_owned,
            store=owned_store,
            tier=RecoveryTier.STANDBY_SHADOW,
            driver=shadow_drv,
            own_driver=True,
            own_store=True,
            metadata={"gpu": "0"},
        )
        svc.register_endpoint(
            "ep-foreign",
            engine="sglang",
            lab=lab_foreign,
            store=foreign_store,
            own_driver=False,
            own_store=False,
        )
        svc.attach_pool("owned-pool", owned_pool, autoscaler=owned_scaler, own=True)
        svc.attach_pool("foreign-pool", foreign_pool, own=False)

        with owned_pool.acquire("agent-1", timeout_s=2.0) as lease:
            self.assertIsInstance(lease, SlotLease)
            snap = svc.health_snapshot()
            self.assertTrue(snap["ok"])
            self.assertFalse(snap["closed"])
            self.assertEqual(snap["endpoint_count"], 2)
            self.assertEqual(snap["pool_count"], 2)
            self.assertEqual(snap["pools"]["owned-pool"]["slot_states"][0]["state"], "leased")
            self.assertTrue(snap["pools"]["owned-pool"]["autoscaler"]["running"])

        probed = svc.probe_health()
        self.assertFalse(probed["ok"])
        self.assertTrue(probed["stores"]["ep-owned"]["ok"])
        self.assertFalse(probed["stores"]["ep-foreign"]["ok"])
        self.assertEqual(probed["stores"]["ep-foreign"]["error"], "stale gms socket")

        summary = svc.teardown()
        self.assertFalse(summary["already_closed"])
        self.assertEqual(summary["stopped_pools"], ["owned-pool"])
        self.assertEqual(summary["errors"], [])
        self.assertTrue(replenisher.closed)
        self.assertEqual(owned_store.stop_calls, 1)
        self.assertEqual(foreign_store.stop_calls, 0)
        self.assertTrue(lab_owned.closed)
        self.assertFalse(lab_foreign.closed)
        self.assertFalse(owned_scaler.status()["running"])

        # Idempotent teardown and fail-closed post-close guards.
        again = svc.teardown()
        self.assertTrue(again["already_closed"])
        self.assertEqual(svc.get_health("ep-owned").status, ServiceHealthStatus.CLOSED)
        self.assertTrue(svc.health_snapshot()["closed"])
        with self.assertRaises(RecoveryClosedError):
            svc.recover("ep-owned")
        with self.assertRaises(RecoveryClosedError):
            svc.record_failure("ep-owned", "after close")
        with self.assertRaises(RecoveryClosedError):
            svc.register_endpoint("ep-new", engine="vllm")
        with self.assertRaises(RecoveryClosedError):
            svc.attach_pool("p-new", foreign_pool)
        with self.assertRaises(RecoveryClosedError):
            svc.quiesce_and_checkpoint("ep-owned")

        foreign_pool.close()
        self.assertTrue(lab_foreign.closed)

    def test_quiesce_and_checkpoint_and_dynamo_snapshot_fail_closed(self) -> None:
        with RecoveryService() as svc:
            hooks = _StubHooks(ok=True)
            svc.register_endpoint("ep-q", engine="vllm", quiesce_hooks=hooks, store=DiskStore())
            receipt = svc.quiesce_and_checkpoint("ep-q")
            self.assertTrue(receipt.ok)
            self.assertEqual(receipt.token, "snap-tok-1")
            self.assertEqual(hooks.events, ["warmup", "quiesce"])

            bad_hooks = _StubHooks(ok=False)
            svc.register_endpoint("ep-q-bad", engine="megatron", quiesce_hooks=bad_hooks)
            with self.assertRaises(RecoveryServiceError):
                svc.quiesce_and_checkpoint("ep-q-bad")
            self.assertEqual(svc.get_health("ep-q-bad").status, ServiceHealthStatus.DEGRADED)

            svc.register_endpoint("ep-no-hooks", engine="vllm")
            with self.assertRaises(RecoveryServiceError):
                svc.quiesce_and_checkpoint("ep-no-hooks")

            with tempfile.TemporaryDirectory(dir="/tmp", prefix="s6-snap-") as tmp:
                ctrl = Path(tmp)
                snap_drv = DynamoK8sSnapshotDriver(
                    engine_name="vllm", control_dir=str(ctrl), wait_timeout_s=2.0, poll_s=0.01
                )
                snap_hooks = _StubHooks(ok=True)
                svc.register_endpoint(
                    "ep-k8s",
                    engine="vllm",
                    tier=RecoveryTier.SNAPSHOT_RESTORE,
                    driver=snap_drv,
                    quiesce_hooks=snap_hooks,
                )

                def _agent_restore() -> None:
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        if (ctrl / "ready-for-snapshot").exists():
                            (ctrl / "restore-complete").write_text("ok\n", encoding="utf-8")
                            return
                        time.sleep(0.01)

                agent = threading.Thread(target=_agent_restore, daemon=True)
                agent.start()
                k8s_receipt = svc.quiesce_and_checkpoint("ep-k8s")
                agent.join(timeout=2.0)
                self.assertTrue(k8s_receipt.ok)
                self.assertTrue((ctrl / "vllm-restore-ready").exists())

                # Timeout fails closed without holding lock.
                timeout_drv = DynamoK8sSnapshotDriver(
                    engine_name="sglang", control_dir=str(ctrl), wait_timeout_s=0.03, poll_s=0.01
                )
                svc.register_endpoint(
                    "ep-k8s-timeout",
                    engine="sglang",
                    tier=RecoveryTier.SNAPSHOT_RESTORE,
                    driver=timeout_drv,
                    quiesce_hooks=_StubHooks(ok=True),
                )
                with self.assertRaises(RecoveryServiceError):
                    svc.quiesce_and_checkpoint("ep-k8s-timeout")
                self.assertEqual(svc.get_health("ep-k8s-timeout").status, ServiceHealthStatus.DEGRADED)

    def test_concurrent_recover_vs_teardown_fails_closed(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class _SlowDriver:
            name = "default_respawn"

            def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                entered.set()
                release.wait(timeout=5.0)
                return {"ok": True, "driver": self.name}

        svc = RecoveryService()
        svc.register_endpoint("ep-race", engine="vllm", driver=_SlowDriver())
        caught: list[BaseException] = []

        def run_recover() -> None:
            try:
                svc.recover("ep-race")
            except BaseException as exc:
                caught.append(exc)

        t = threading.Thread(target=run_recover)
        t.start()
        self.assertTrue(entered.wait(timeout=2.0))
        svc.teardown()
        release.set()
        t.join(timeout=2.0)
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0], RecoveryClosedError)
        self.assertEqual(svc.get_health("ep-race").status, ServiceHealthStatus.CLOSED)


if __name__ == "__main__":
    unittest.main()
