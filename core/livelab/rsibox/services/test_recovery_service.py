"""Unit, composition, and concurrency tests for `rsibox.services.recovery_service` (S6).

Pure CPU / stdlib tests — zero GPU, torch, numpy, or engine imports.
"""
from __future__ import annotations

import contextlib
import gc
import os
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


class _PipeLab(_StubLab):
    """`_StubLab` holding an open `os.pipe()` pair so any leaked lab triggers `ResourceWarning`."""

    def __init__(self, **kwargs: Any) -> None:
        r_fd, w_fd = os.pipe()
        self._r_file = os.fdopen(r_fd, "rb")
        self._w_file = os.fdopen(w_fd, "wb")
        super().__init__(**kwargs)

    def close(self) -> dict[str, Any]:
        for fh in (self._r_file, self._w_file):
            if not fh.closed:
                fh.close()
        return super().close()


class RecoveryServiceHardeningTests(unittest.TestCase):
    def test_pipe_lab_ownership_closes_on_fail_replacement_race_and_teardown(self) -> None:
        class _FailDriver:
            name = "default_respawn"

            def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                return {"ok": False, "error": "spawn failed"}

        class _ReplacingDriver:
            name = "default_respawn"

            def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                return {"ok": True, "driver": self.name, "lab": _PipeLab()}

        created_fail: list[_PipeLab] = []
        created_ok: list[_PipeLab] = []
        foreign_lab = _PipeLab()
        try:
            with RecoveryService() as svc:
                svc.register_endpoint(
                    "ep-fail-pipe",
                    engine="vllm",
                    lab_factory=lambda: created_fail.append(p := _PipeLab()) or p,
                    driver=_FailDriver(),
                )
                rec_fail = svc.recover("ep-fail-pipe")
                self.assertFalse(rec_fail.ok)
                self.assertTrue(created_fail[0].closed)

                svc.register_endpoint(
                    "ep-replace-pipe",
                    engine="vllm",
                    lab_factory=lambda: created_ok.append(p := _PipeLab()) or p,
                    driver=_ReplacingDriver(),
                )
                rec_rep = svc.recover("ep-replace-pipe")
                self.assertTrue(rec_rep.ok)
                # Initial lab_factory lab was replaced & closed immediately; replacement stays open until teardown.
                self.assertTrue(created_ok[0].closed)

                svc.register_endpoint("ep-foreign-pipe", engine="sglang", lab=foreign_lab, own_lab=False)

            # Service exited context manager -> teardown closed replacement lab, left foreign lab open.
            self.assertFalse(foreign_lab.closed)

            # Concurrent teardown during recover() closes created_lab before raising RecoveryClosedError.
            entered = threading.Event()
            release = threading.Event()
            raced: list[_PipeLab] = []

            class _SlowPipeDriver:
                name = "default_respawn"

                def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                    entered.set()
                    release.wait(timeout=5.0)
                    return {"ok": True, "driver": self.name}

            svc2 = RecoveryService()
            svc2.register_endpoint(
                "ep-race-pipe",
                engine="vllm",
                lab_factory=lambda: raced.append(p := _PipeLab()) or p,
                driver=_SlowPipeDriver(),
            )
            race_exc: list[BaseException] = []

            def run_raced() -> None:
                try:
                    svc2.recover("ep-race-pipe")
                except BaseException as exc:
                    race_exc.append(exc)

            t = threading.Thread(target=run_raced)
            t.start()
            self.assertTrue(entered.wait(timeout=2.0))
            svc2.teardown()
            release.set()
            t.join(timeout=2.0)
            self.assertIsInstance(race_exc[0], RecoveryClosedError)
            self.assertTrue(raced[0].closed)

            # Concurrent teardown when driver returns a replacement "lab" closes returned_lab too.
            rep_entered = threading.Event()
            rep_release = threading.Event()
            returned_labs: list[_PipeLab] = []
            initial_labs: list[_PipeLab] = []

            class _SlowReplacingDriver:
                name = "default_respawn"

                def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                    rep_entered.set()
                    rep_release.wait(timeout=5.0)
                    new_lab = _PipeLab()
                    returned_labs.append(new_lab)
                    return {"ok": True, "driver": self.name, "lab": new_lab}

            svc3 = RecoveryService()
            svc3.register_endpoint(
                "ep-race-rep-pipe",
                engine="vllm",
                lab_factory=lambda: initial_labs.append(p := _PipeLab()) or p,
                driver=_SlowReplacingDriver(),
            )
            rep_exc: list[BaseException] = []

            def run_rep_raced() -> None:
                try:
                    svc3.recover("ep-race-rep-pipe")
                except BaseException as exc:
                    rep_exc.append(exc)

            t3 = threading.Thread(target=run_rep_raced)
            t3.start()
            self.assertTrue(rep_entered.wait(timeout=2.0))
            svc3.teardown()
            rep_release.set()
            t3.join(timeout=2.0)
            self.assertIsInstance(rep_exc[0], RecoveryClosedError)
            self.assertTrue(initial_labs[0].closed)
            self.assertTrue(returned_labs[0].closed)
            gc.collect()
        finally:
            foreign_lab.close()

    def test_quiesce_concurrency_record_failure_and_teardown(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class _SlowHooks(_StubHooks):
            def quiesce_for_snapshot(self) -> QuiesceReceipt:
                entered.set()
                release.wait(timeout=5.0)
                return super().quiesce_for_snapshot()

        with RecoveryService(quarantine_threshold=2) as svc:
            svc.register_endpoint("ep-q-race", engine="vllm", quiesce_hooks=_SlowHooks(ok=True))
            out: list[QuiesceReceipt] = []

            t = threading.Thread(target=lambda: out.append(svc.quiesce_and_checkpoint("ep-q-race")))
            t.start()
            self.assertTrue(entered.wait(timeout=2.0))
            # Concurrent fault must NOT be clobbered when quiesce finishes.
            svc.record_failure("ep-q-race", "concurrent worker fault")
            release.set()
            t.join(timeout=2.0)
            self.assertTrue(out[0].ok)
            self.assertEqual(svc.get_health("ep-q-race").status, ServiceHealthStatus.DEGRADED)

        # Concurrent teardown during failing quiesce raises RecoveryClosedError.
        entered2 = threading.Event()
        release2 = threading.Event()

        class _FailOnCloseHooks(_StubHooks):
            def quiesce_for_snapshot(self) -> QuiesceReceipt:
                entered2.set()
                release2.wait(timeout=5.0)
                return QuiesceReceipt(ok=False, error="aborted")

        svc2 = RecoveryService()
        svc2.register_endpoint("ep-q-close", engine="vllm", quiesce_hooks=_FailOnCloseHooks())
        errs: list[BaseException] = []

        def run_q() -> None:
            try:
                svc2.quiesce_and_checkpoint("ep-q-close")
            except BaseException as exc:
                errs.append(exc)

        t2 = threading.Thread(target=run_q)
        t2.start()
        self.assertTrue(entered2.wait(timeout=2.0))
        svc2.teardown()
        release2.set()
        t2.join(timeout=2.0)
        self.assertIsInstance(errs[0], RecoveryClosedError)

    def test_standby_shadow_promote_tier_inference_and_fallback_receipts(self) -> None:
        with RecoveryService() as svc:
            # Tier inferred automatically from StandbyShadowDriver when tier is omitted.
            shadow_ok = StandbyShadowDriver(
                promote_fn=lambda lab, store: {"ok": True, "reset": ["shadow promoted"]},
                ready_count_fn=lambda: 1,
            )
            h = svc.register_endpoint("ep-sh-ok", engine="vllm", lab_factory=_StubLab, driver=shadow_ok)
            self.assertEqual(h.tier, RecoveryTier.STANDBY_SHADOW)
            rec = svc.recover("ep-sh-ok")
            self.assertTrue(rec.ok)
            self.assertFalse(rec.fallback_used)
            self.assertEqual(rec.tier_succeeded, RecoveryTier.STANDBY_SHADOW)

            # StandbyShadowDriver with internal failing fallback preserves promote_error & fallback_used.
            class _FailFallback:
                name = "default_respawn"

                def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                    return {"ok": False, "driver": self.name, "error": "cold_respawn_oom"}

            shadow_internal_fb = StandbyShadowDriver(
                promote_fn=lambda lab, store: {"ok": False, "error": "promote_ipc_err"},
                fallback_driver=_FailFallback(),
            )
            svc.register_endpoint("ep-sh-fb-fail", engine="vllm", lab_factory=_StubLab, driver=shadow_internal_fb)
            rec_fb_fail = svc.recover("ep-sh-fb-fail")
            self.assertFalse(rec_fb_fail.ok)
            self.assertTrue(rec_fb_fail.fallback_used)
            self.assertIn("promote_ipc_err", rec_fb_fail.error or "")
            self.assertIn("cold_respawn_oom", rec_fb_fail.error or "")

            # Fallback lab_factory error is reported cleanly instead of masked as None.
            svc.register_endpoint(
                "ep-fb-factory-err",
                engine="sglang",
                lab_factory=lambda: (_ for _ in ()).throw(RuntimeError("factory disk full")),
                tier=RecoveryTier.STANDBY_SHADOW,
                driver=StandbyShadowDriver(promote_fn=lambda lab, store: {"ok": False}, ready_count_fn=lambda: 0),
                fallback_driver=DefaultRespawnDriver(),
            )
            with self.assertRaises(EscalationFailedError) as ctx:
                svc.recover("ep-fb-factory-err")
            self.assertIn("factory disk full", str(ctx.exception))

    def test_closed_and_raising_pool_health_snapshot_and_recovering_readiness(self) -> None:
        # Store missing stop() rejected when own_store=True.
        with RecoveryService() as svc:
            with self.assertRaises(TypeError):
                svc.register_endpoint("ep-bad-store", engine="vllm", store=type("NoStop", (), {"probe": lambda self: None})(), own_store=True)  # type: ignore[arg-type]

            lab = _StubLab()
            closed_pool = SandboxPool([PoolSlot("s0", lab)], close_labs=True)
            closed_pool.close()
            svc.attach_pool("closed-pool", closed_pool, own=False)

            class _RaisingPool:
                _closed = False

                def slot_states(self) -> list[dict[str, Any]]:
                    raise RuntimeError("pool rpc broken")

                def metrics(self) -> dict[str, Any]:
                    return {}

            svc.attach_pool("bad-pool", _RaisingPool(), own=False)  # type: ignore[arg-type]
            snap = svc.health_snapshot()
            self.assertFalse(snap["ok"])
            self.assertFalse(snap["pools"]["closed-pool"]["ok"])
            self.assertTrue(snap["pools"]["closed-pool"]["closed"])
            self.assertFalse(snap["pools"]["bad-pool"]["ok"])
            self.assertIn("pool rpc broken", snap["pools"]["bad-pool"]["error"])

        # RECOVERING marks health_snapshot()["ok"] == False and BaseException restores concurrent fault status.
        entered = threading.Event()
        release = threading.Event()

        class _AbortDriver:
            name = "default_respawn"

            def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
                entered.set()
                release.wait(timeout=5.0)
                raise SystemExit("simulated base exception")

        with RecoveryService(quarantine_threshold=1) as svc_rec:
            svc_rec.register_endpoint("ep-rec-ok", engine="vllm", driver=_AbortDriver())
            caught_base: list[BaseException] = []

            def run_abort() -> None:
                try:
                    svc_rec.recover("ep-rec-ok")
                except BaseException as exc:
                    caught_base.append(exc)

            t = threading.Thread(target=run_abort)
            t.start()
            self.assertTrue(entered.wait(timeout=2.0))
            snap_rec = svc_rec.health_snapshot()
            self.assertFalse(snap_rec["ok"])
            self.assertEqual(snap_rec["endpoints"]["ep-rec-ok"]["status"], "recovering")
            svc_rec.record_failure("ep-rec-ok", "fatal fault during recovery")
            release.set()
            t.join(timeout=2.0)
            self.assertIsInstance(caught_base[0], SystemExit)
            self.assertEqual(svc_rec.get_health("ep-rec-ok").status, ServiceHealthStatus.QUARANTINED)


if __name__ == "__main__":
    unittest.main()
