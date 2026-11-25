"""Spec-to-code conformance tests validating live state transitions against the TLA+ models."""
from __future__ import annotations

import gc
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import forksafe
from kernel import coordinator as C
from kernel import journal as J
from kernel import receipt as R
from kernel import swap as S
from kernel import toy
from kernel import witness as W
from rsibox.autoscaler import AutoScalerPolicy, PoolAutoScaler
from rsibox.lab import BaseLab
from rsibox.pool import PoolError, PoolSlot, PoolTimeoutError, SandboxPool
from rsibox.services import ForkPoisonedError, ZygoteProfile, ZygoteService
from rsibox.warm import (
    AsyncShadowReplenisher,
    DefaultRespawnDriver,
    DiskStore,
    DynamoK8sSnapshotDriver,
    QuiesceReceipt,
    StandbyShadowDriver,
    run_snapshot_lifecycle,
)


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

    def test_standby_shadow_tla_single_flight_fallback_and_close(self) -> None:
        class _DummyShadowLab(BaseLab):
            name = "tla_shadow_dummy"

            def _start_engine(self):
                part = C.CallParticipant("r0", lambda *_a: {"ok": True, "code": {"m": "x"}})
                return C.StepGapCoordinator([part], lambda fn: fn()), []

            def _stop_engine(self):
                return None

            def _run_workload(self, _workload):
                return {"ok": True}

        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        disposed: list[dict] = []
        started = [0]

        def _slow_rep(_lab, _store=None):
            started[0] += 1
            entered.set()
            release.wait(timeout=5.0)
            return {"ok": True, "shadow": f"s-{started[0]}"}

        rep = AsyncShadowReplenisher(_slow_rep, queue_when_busy=True, dispose_fn=disposed.append)
        self.addCleanup(rep.close)

        # SingleFlightReplenisher: coalesces concurrent triggers while in flight.
        self.assertEqual(rep.trigger(None), "started")
        self.assertTrue(entered.wait(timeout=2.0))
        self.assertEqual(rep.trigger(None), "queued")
        self.assertEqual(rep.trigger(None), "coalesced")

        # NeverWaitWhenClosed + ZeroOrphanOnClose: close() wakes waiter, skips queued item, and disposes outcome.
        wait_res: list[bool] = []
        t_wait = threading.Thread(target=lambda: wait_res.append(rep.wait(timeout_s=5.0, until=lambda: False)))
        t_wait.start()
        time.sleep(0.03)
        close_ok = rep.close(timeout_s=0.05)
        t_wait.join(timeout=2.0)
        self.assertFalse(t_wait.is_alive())
        self.assertEqual(wait_res, [False])
        release.set()
        rep.wait(timeout_s=2.0)
        if not close_ok:
            # Wait briefly for worker thread post-close cleanup to finish disposing outcome.
            deadline = time.monotonic() + 2.0
            while rep.status()["in_flight"] and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertEqual(rep.status()["attempts"], 1)
        self.assertEqual(len(disposed), 1)
        self.assertEqual(rep.status()["disposed_after_close"], 1)

        # Failed promote_fn with fallback_driver recovers cleanly via DefaultRespawnDriver.
        rep_calls = [0]
        driver = StandbyShadowDriver(
            promote_fn=lambda _lab, _store: {"ok": False, "error": "shadow vanished"},
            replenish_fn=lambda _lab, _store: (rep_calls.__setitem__(0, rep_calls[0] + 1), {"ok": True})[1],
            fallback_driver=DefaultRespawnDriver(),
        )
        lab = _DummyShadowLab(weight_store=DiskStore(), recovery_driver=driver)
        self.addCleanup(lab.close)
        res = lab.reset("respawn")
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["fallback_from"], "standby_shadow")
        self.assertEqual(res["driver"], "default_respawn")
        self.assertEqual(rep_calls[0], 1)

    def test_autoscaler_tla_undrained_replace_retry_and_start_stop_lock(self) -> None:
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        lab = _TrackingLab(entered=entered, release=release)
        pool = SandboxPool([PoolSlot("s0", lab)], drain_timeout_s=0.05, close_labs=True)
        self.addCleanup(pool.close)

        lease = pool.acquire("agent-a")
        t_call = threading.Thread(target=lease.lab.check)
        t_call.start()
        self.assertTrue(entered.wait(timeout=2.0))
        lease.release()

        now = [0.0]
        fail_respawn = [True]
        orig_reset = lab.reset

        def _reset(kind="state"):
            if kind == "respawn" and fail_respawn[0]:
                return {"ok": False, "error": "down"}
            return orig_reset(kind)

        lab.status = lambda: {"ok": True, "state": "poisoned", "txns": []}  # type: ignore[method-assign]
        lab.reset = _reset  # type: ignore[method-assign]
        orig_drain = pool.drain_slot
        fail_drain = [True]

        def _drain(slot_id, **kw):
            if fail_drain[0]:
                raise PoolTimeoutError("transient drain timeout")
            return orig_drain(slot_id, **kw)

        pool.drain_slot = _drain  # type: ignore[method-assign]
        scaler = PoolAutoScaler(
            pool,
            lambda sid, role: PoolSlot(sid, _TrackingLab(), role=role),
            policy=AutoScalerPolicy(
                min_slots=1, max_slots=1, max_heal_attempts=2,
                heal_backoff_s=4.0, replace_on_heal_exhausted=True,
                poll_interval_s=0.01,
            ),
            clock=lambda: now[0],
        )
        self.addCleanup(scaler.stop)

        # UndrainedNeverExhaustsDirectly: undrained PoolError backs off with heals_attempted == 0.
        self.assertEqual(scaler.reconcile_once()["action"], "noop")
        self.assertEqual(scaler.status()["heals_attempted"], 0)
        self.assertEqual(scaler.status()["exhausted_slots"], [])

        # Once active call exits, 2 heal failures exhaust -> drain_slot fails -> retries replace without re-healing.
        release.set()
        t_call.join(timeout=2.0)
        now[0] = 5.0
        self.assertEqual(scaler.reconcile_once()["action"], "heal_failed")
        now[0] = 10.0
        self.assertEqual(scaler.reconcile_once()["action"], "replace_failed")
        self.assertEqual(scaler.status()["heals_attempted"], 2)
        self.assertEqual(scaler.status()["exhausted_slots"], ["s0"])

        fail_drain[0] = False
        now[0] = 15.0
        self.assertEqual(scaler.reconcile_once()["action"], "replace")
        self.assertEqual(scaler.status()["heals_attempted"], 2)
        self.assertEqual(scaler.status()["exhausted_slots"], [])  # RemovedClearsTracking

        # Concurrent start()/stop() lock ordering never leaves running thread after stop().
        for _ in range(15):
            scaler.start()
            scaler.stop(timeout_s=2.0)
            self.assertFalse(scaler.status()["running"])

        pool.close(close_labs=True)
        self.assertEqual(scaler.reconcile_once()["action"], "noop")  # ClosedYieldsNoop

    def test_snapshot_lifecycle_tla_fail_closed_sentinels(self) -> None:
        class _Hooks:
            mutability = "readonly_weights"

            def __init__(self, *, quiesce_ok: bool, resume_ok: bool) -> None:
                self._quiesce_ok = quiesce_ok
                self._resume_ok = resume_ok
                self.events: list[str] = []

            def warmup_for_snapshot(self) -> None:
                self.events.append("warmup")

            def quiesce_for_snapshot(self) -> QuiesceReceipt:
                self.events.append("quiesce")
                if not self._quiesce_ok:
                    return QuiesceReceipt(ok=False, error="quiesce failed")
                return QuiesceReceipt(ok=True, released=("kv_cache",), token="tok-tla")

            def resume_after_restore(self, _receipt: QuiesceReceipt) -> dict[str, object]:
                self.events.append("resume")
                if not self._resume_ok:
                    return {"ok": False, "error": "wake_up / host ping failed"}
                return {"ok": True}

        with tempfile.TemporaryDirectory() as td:
            ctrl = pathlib.Path(td)
            ready = ctrl / "ready-for-snapshot"
            restore_complete = ctrl / "restore-complete"
            restore_ready = ctrl / "vllm-restore-ready"
            driver = DynamoK8sSnapshotDriver(
                engine_name="vllm", control_dir=str(ctrl), wait_timeout_s=1.0, poll_s=0.01,
            )

            # 1. Failing quiesce_for_snapshot() -> stale sentinels cleared, ready-for-snapshot never written.
            ready.write_text("stale", encoding="utf-8")
            restore_complete.write_text("stale", encoding="utf-8")
            restore_ready.write_text("stale", encoding="utf-8")
            q_fail = _Hooks(quiesce_ok=False, resume_ok=True)
            res_q = driver.checkpoint_and_wait(q_fail)
            self.assertFalse(res_q.ok)
            self.assertEqual(q_fail.events, ["warmup", "quiesce"])
            self.assertFalse(ready.exists())  # ReadySentinelRequiresQuiesce
            self.assertFalse(restore_complete.exists())
            self.assertFalse(restore_ready.exists())  # FailedNoResumeReady

            # 2. Timeout waiting for restore-complete -> ready-for-snapshot unlinked, no restore-ready.
            tmo_hooks = _Hooks(quiesce_ok=True, resume_ok=True)
            res_tmo = run_snapshot_lifecycle(
                tmo_hooks, engine_name="vllm", control_dir=str(ctrl), wait_timeout_s=0.04, poll_s=0.01,
            )
            self.assertFalse(res_tmo.ok)
            self.assertEqual(tmo_hooks.events, ["warmup", "quiesce"])
            self.assertFalse(ready.exists())
            self.assertFalse(restore_ready.exists())  # ResumeReadyRequiresRestore

            def _spawn_agent(expected_hooks: _Hooks) -> threading.Thread:
                def _agent() -> None:
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        if ready.exists() and "quiesce" in expected_hooks.events:
                            restore_complete.write_text("ok\n", encoding="utf-8")
                            return
                        time.sleep(0.01)

                t = threading.Thread(target=_agent, daemon=True)
                t.start()
                return t

            # 3. Failing wake_up / host ping in resume_after_restore() -> no <engine>-restore-ready written.
            r_fail = _Hooks(quiesce_ok=True, resume_ok=False)
            t1 = _spawn_agent(r_fail)
            res_r = driver.checkpoint_and_wait(r_fail)
            t1.join(timeout=2.0)
            self.assertFalse(res_r.ok)
            self.assertTrue(restore_complete.exists())
            self.assertFalse(restore_ready.exists())  # FailedNoResumeReady

            # 4. Clean pass writes ready-for-snapshot after quiesce and <engine>-restore-ready after restore.
            ok_hooks = _Hooks(quiesce_ok=True, resume_ok=True)
            t2 = _spawn_agent(ok_hooks)
            res_ok = driver.checkpoint_and_wait(ok_hooks)
            t2.join(timeout=2.0)
            self.assertTrue(res_ok.ok, res_ok.to_dict())
            self.assertEqual(ok_hooks.events, ["warmup", "quiesce", "resume"])
            self.assertTrue(ready.exists())
            self.assertTrue(restore_complete.exists())
            self.assertTrue(restore_ready.exists())

    def test_forksafe_lock_order_tla_guard_preload_and_poison_refusal(self) -> None:
        if hasattr(gc, "unfreeze"):
            self.addCleanup(gc.unfreeze)

        def _make_fake_cuda() -> types.SimpleNamespace:
            def _uninit(*_a: object, **_kw: object) -> None:
                raise AssertionError("unpatched fake CUDA queried while uninitialized")

            return types.SimpleNamespace(
                is_initialized=lambda: False,
                get_device_capability=_uninit,
                current_device=_uninit,
                get_device_properties=_uninit,
            )

        def _fake_probe() -> list[dict[str, object]]:
            return [{"name": "FakeGPU", "major": 10, "minor": 0, "total_memory": 1024, "multi_processor_count": 8}]

        prof = ZygoteProfile("vllm", preload_modules=("json",), freeze_gc=True)

        # 1. Poisoned parent refuses arm_guards() and start() fail-closed (ForkRequiresCleanDriver).
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(forksafe, "fork_poisoned", return_value=True), \
             mock.patch.object(forksafe, "poison_reasons", return_value=["cuInit called"]):
            svc_poisoned = ZygoteService(prof, cuda_module_fn=_make_fake_cuda)
            self.addCleanup(svc_poisoned.close)
            with self.assertRaises(ForkPoisonedError):
                svc_poisoned.arm_guards(capability=(10, 0), probe=_fake_probe)
            with self.assertRaises(ForkPoisonedError):
                svc_poisoned.start(capability=(10, 0), probe=_fake_probe)

        # 2. Guards armed & Triton pinned before preload, Triton unpinned after preload, poison blocks fork,
        #    and clean fork runs unpatch_child() before child payload (PreloadAfterGuards / ReadyChildUnpatched).
        fake_cuda = _make_fake_cuda()
        events: list[str] = []

        def _fake_pin_triton(**_kw: object):
            events.append("pin_triton")

            def _restore() -> None:
                events.append("restore_triton")

            return _restore

        def _importer(name: str) -> object:
            self.assertTrue(getattr(fake_cuda.get_device_capability, "_rsibox_forksafe", False))
            events.append(f"import:{name}")
            return __import__(name)

        def _child_handler(payload: dict, fds: list[int]) -> int:
            out = {
                "still_patched": getattr(fake_cuda.get_device_capability, "_rsibox_forksafe", False),
                "torch_cuda": sorted(payload["child_guard_report"].get("torch_cuda", [])),
            }
            os.write(fds[0], json.dumps(out).encode("utf-8"))
            return 0

        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(forksafe, "pin_triton_target", side_effect=_fake_pin_triton):
            svc = ZygoteService(prof, cuda_module_fn=lambda: fake_cuda, importer=_importer)
            self.addCleanup(svc.close)
            svc.start(capability=(10, 0), probe=_fake_probe)
            self.assertEqual(events, ["pin_triton", "import:json", "restore_triton"])

            # Subsequent CUDA initialization refuses spawn_child() fail-closed.
            with mock.patch.object(forksafe, "fork_poisoned", return_value=True), \
                 mock.patch.object(forksafe, "poison_reasons", return_value=["CUDA driver initialized"]):
                with self.assertRaises(ForkPoisonedError):
                    svc.spawn_child("worker")

            r_fd, w_fd = os.pipe()
            try:
                child = svc.spawn_child("worker", pass_fds=(w_fd,), handler=_child_handler, wait=True)
            finally:
                os.close(w_fd)
            try:
                chunks: list[bytes] = []
                while chunk := os.read(r_fd, 4096):
                    chunks.append(chunk)
                child_info = json.loads(b"".join(chunks).decode("utf-8"))
            finally:
                os.close(r_fd)

            self.assertEqual(child.exit_code, 0)
            self.assertFalse(child_info["still_patched"])
            self.assertEqual(
                child_info["torch_cuda"],
                ["current_device", "get_device_capability", "get_device_properties"],
            )
