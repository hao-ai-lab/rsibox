"""Golden CPU tests for `SglangShadowPromoter`, targeted engine stop, and failover."""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
import unittest
from unittest import mock

_ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
_LIVELAB_DIR = os.path.dirname(_ADAPTER_DIR)
_CORE_LIVELAB = os.path.abspath(os.path.join(_LIVELAB_DIR, "..", "..", "..", "core", "livelab"))
for _path in (_CORE_LIVELAB, _LIVELAB_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from adapter.fixture.harness import edit_once
from adapter.shadow import (
    ParkedSglangShadow,
    SglangShadowPromoter,
    make_shadow_recovery_driver,
    park_shadow_lab,
    stop_sglang_engine_targeted,
)
from adapter.snapshot_hooks import SglangQuiesceHooks
from adapter.toy_lab import LAYERNORM, SAMPLER, ToySglangLab
from rsibox.warm import QuiesceReceipt, StandbyShadowDriver


class _FakeTokenizerManager:
    def __init__(self, log: list[tuple[str, object]], *, fail_continue: bool = False) -> None:
        self._log = log
        self._fail_continue = fail_continue

    def pause_generation(self, req: object) -> dict[str, object]:
        self._log.append(("pause_generation", getattr(req, "mode", None)))
        return {"ok": True}

    def continue_generation(self, _req: object) -> dict[str, object]:
        self._log.append(("continue_generation", None))
        if self._fail_continue:
            return {"ok": False, "error": "continue refused"}
        return {"ok": True}


def _attach_fake_memory_saver(
    lab: ToySglangLab,
    log: list[tuple[str, object]],
    *,
    fail_release: bool = False,
    fail_resume: bool = False,
    fail_continue: bool = False,
) -> None:
    """Attach SGLang's memory-occupation and pause/continue hooks onto a real ToySglangLab."""
    lab.engine.tokenizer_manager = _FakeTokenizerManager(log, fail_continue=fail_continue)

    def release_memory_occupation(tags: list[str] | None = None) -> dict[str, object]:
        log.append(("release_memory_occupation", tuple(tags or ())))
        if fail_release:
            return {"ok": False, "error": "release refused"}
        return {"ok": True}

    def resume_memory_occupation(tags: list[str] | None = None) -> dict[str, object]:
        log.append(("resume_memory_occupation", tuple(tags or ())))
        if fail_resume:
            return {"ok": False, "error": "resume refused"}
        return {"ok": True}

    lab.engine.release_memory_occupation = release_memory_occupation
    lab.engine.resume_memory_occupation = resume_memory_occupation


@contextlib.contextmanager
def _patch_pause_io_types(io_module_name: str):
    mod = importlib.import_module(io_module_name)

    class PauseGenerationReqInput:
        def __init__(self, mode: str = "retract") -> None:
            self.mode = mode

    class ContinueGenerationReqInput:
        pass

    with (
        mock.patch.object(mod, "PauseGenerationReqInput", PauseGenerationReqInput, create=True),
        mock.patch.object(mod, "ContinueGenerationReqInput", ContinueGenerationReqInput, create=True),
    ):
        yield mod


class SglangShadowPromoterTests(unittest.TestCase):
    def test_park_promote_and_live_edit_on_real_toy_sglang_labs(self) -> None:
        """Golden case: poisoned primary ToySglangLab fails over to a parked standby ToySglangLab
        via `StandbyShadowDriver`, preserves process-global env restore closures until final close,
        detaches the shell shadow without killing the promoted engine, and immediately executes
        real `R0` and `R1` live edits + rollback cleanly."""
        primary = ToySglangLab(tp_size=2)
        shadow = ToySglangLab(tp_size=2)
        shadow_log: list[tuple[str, object]] = []
        replenish_calls: list[str] = []
        env_restores: list[str] = []
        primary._weight_cache_env_restore = lambda: env_restores.append("primary_env_restored")
        shadow._weight_cache_env_restore = lambda: env_restores.append("shadow_env_restored")
        _attach_fake_memory_saver(shadow, shadow_log)
        shadow_engine_obj = shadow.engine

        try:
            with _patch_pause_io_types(shadow.io_module):
                base_chk = primary.check()
                self.assertTrue(base_chk["ok"], base_chk)
                base_out = base_chk["output"]
                driver = make_shadow_recovery_driver(
                    replenish_fn=lambda lab, _store: replenish_calls.append(lab.name),
                )
                self.assertIsInstance(driver, StandbyShadowDriver)
                primary.recovery_driver = driver
                promoter: SglangShadowPromoter = driver.promoter  # type: ignore[attr-defined]

                parked = promoter.park_shadow(shadow)
                self.assertTrue(parked["ok"], parked)
                self.assertEqual(parked["released"], ["kv_cache"])
                self.assertEqual(promoter.pending_count(), 1)
                self.assertEqual(
                    shadow_log,
                    [("pause_generation", "retract"), ("release_memory_occupation", ("kv_cache",))],
                )

                # Poison primary and mark its control channel dirty (simulating crash / sticky fault).
                primary.session.fault("participant_failed", "simulated primary SIGKILL", "clean")
                primary.dirty = True
                self.assertTrue(primary.session.poisoned())

                res = primary.reset("respawn")
                self.assertTrue(res["ok"], res)
                self.assertEqual(res["driver"], "standby_shadow")
                self.assertEqual(res["promoted_ranks"], 2)
                self.assertEqual(res["resumed"], ["kv_cache"])
                self.assertTrue(res["participants_ok"])
                self.assertEqual(promoter.pending_count(), 0)
                self.assertEqual(replenish_calls, [primary.name])
                self.assertIn("promote_s", primary.costs)
                self.assertGreaterEqual(primary.costs["promote_s"], 0.0)
                self.assertFalse(primary.session.poisoned())
                self.assertFalse(primary.dirty)
                self.assertIs(primary.engine, shadow_engine_obj)
                # Process-global env restore must NOT have fired during failover!
                self.assertEqual(env_restores, [])

                # Shell shadow lab is fully detached: closing it must NOT kill promoted engine or mutate env!
                self.assertIsNone(shadow.engine)
                self.assertTrue(shadow.dirty)
                shadow.close()
                self.assertEqual(env_restores, [])
                self.assertEqual(len(primary.rank_conns), 2)
                self.assertTrue(primary.rank_call(0, "ping")["ok"])
                self.assertTrue(primary.rank_call(1, "ping")["ok"])

                # Promoted primary immediately serves bit-exact baseline and real R0 + R1 live edits!
                post_chk = primary.check()
                self.assertTrue(post_chk["ok"], post_chk)
                self.assertEqual(post_chk["output"], base_out)
                edited_sampler = edit_once(primary._base(SAMPLER), "if value > row[best]:", "if value < row[best]:")
                inst_r0 = primary.install({"files": {SAMPLER: edited_sampler}})
                self.assertTrue(inst_r0["ok"], inst_r0)
                chk_r0 = primary.check()
                self.assertTrue(chk_r0["ok"], chk_r0)
                self.assertTrue(chk_r0["proven"])
                self.assertNotEqual(chk_r0["output"], base_out)
                self.assertTrue(primary.uninstall()["ok"])
                rb0_chk = primary.check()
                self.assertTrue(rb0_chk["ok"], rb0_chk)
                self.assertEqual(rb0_chk["output"], base_out)

                edited_norm = edit_once(primary._base(LAYERNORM), "(h * 7 + 1) % MODULUS", "(h * 11 + 1) % MODULUS")
                inst_r1 = primary.install({"files": {LAYERNORM: edited_norm}, "sizes": [1, 2]})
                self.assertTrue(inst_r1["ok"], inst_r1)
                self.assertIn("R1 recapture", inst_r1["mechanisms"])
                chk_r1 = primary.check()
                self.assertTrue(chk_r1["ok"], chk_r1)
                self.assertTrue(chk_r1["proven"])
                self.assertNotEqual(chk_r1["output"], base_out)
                self.assertTrue(primary.uninstall()["ok"])
                rb1_chk = primary.check()
                self.assertTrue(rb1_chk["ok"], rb1_chk)
                self.assertEqual(rb1_chk["output"], base_out)
        finally:
            primary.close()
            shadow.close()
        # Final primary.close() runs the original pre-lab process env restore exactly once.
        self.assertEqual(env_restores, ["primary_env_restored"])

    def test_targeted_engine_stop_kills_only_owned_watchdog_processes(self) -> None:
        """Golden case: when `engine.tokenizer_manager._subprocess_watchdog._processes` is present,
        `stop_sglang_engine_targeted` terminates only that engine's owned processes and never
        invokes `engine.shutdown()` (which upstream calls `kill_process_tree(os.getpid())`)."""
        events: list[str] = []

        class _FakeProc:
            def __init__(self, pid: int, name: str) -> None:
                self.pid = pid
                self.name = name
                self._alive = True

            def is_alive(self) -> bool:
                return self._alive

            def terminate(self) -> None:
                events.append(f"term:{self.name}")
                self._alive = False

            def kill(self) -> None:
                events.append(f"kill:{self.name}")
                self._alive = False

            def join(self, timeout: float | None = None) -> None:
                events.append(f"join:{self.name}")

        class _FakeWatchdog:
            def __init__(self, procs) -> None:
                self._processes = procs

            def stop(self) -> None:
                events.append("watchdog.stop")

        class _FakeRpc:
            def close(self, linger: int = 0) -> None:
                events.append(f"rpc.close:{linger}")

        class _FakeEngine:
            def __init__(self) -> None:
                self.tokenizer_manager = type(
                    "_TM",
                    (),
                    {
                        "_subprocess_watchdog": _FakeWatchdog(
                            [_FakeProc(101, "sched0"), _FakeProc(102, "detok0")]
                        ),
                        "mm_processor": type("_MM", (), {"shutdown": lambda _s: events.append("mm.shutdown")})(),
                        "cuda_vmm_feature_transport": type(
                            "_VMM", (), {"shutdown": lambda _s: events.append("vmm.shutdown")}
                        )(),
                    },
                )()
                self.send_to_rpc = _FakeRpc()

            def shutdown(self) -> None:
                events.append("BLANKET_SHUTDOWN_MUST_NOT_RUN")

        class _FakeLab:
            def __init__(self) -> None:
                self.engine = _FakeEngine()
                self.rank_conns = {}
                self.rank_locks = {}
                self.sock_path = None
                self.ranks = 1
                self.dirty = False
                self._fork_subprocs_restore = lambda: events.append("fork_restore")

            def _restore_weight_cache_env(self) -> None:
                events.append("env_restore")

        lab = _FakeLab()
        self.assertTrue(stop_sglang_engine_targeted(lab, preserve_process_env=True))
        self.assertIsNone(lab.engine)
        self.assertTrue(lab.dirty)
        self.assertNotIn("BLANKET_SHUTDOWN_MUST_NOT_RUN", events)
        self.assertNotIn("env_restore", events)
        self.assertNotIn("fork_restore", events)
        self.assertIn("watchdog.stop", events)
        self.assertIn("term:sched0", events)
        self.assertIn("term:detok0", events)
        self.assertIn("rpc.close:0", events)
        self.assertIn("mm.shutdown", events)
        self.assertIn("vmm.shutdown", events)

    def test_shadow_promoter_fails_closed_on_poisoned_edited_mismatch_and_resume_error(self) -> None:
        """Fail closed without leaking healthy queued shadows or mutating primary on empty queue,
        poisoned/edited/dirty shadows, tp_size mismatch, resume error, or self-promotion."""
        primary = ToySglangLab(tp_size=2)
        shadow_tp1 = ToySglangLab(tp_size=1)
        shadow_bad = ToySglangLab(tp_size=2)
        shadow_poison = ToySglangLab(tp_size=2)
        log_tp1: list[tuple[str, object]] = []
        log_bad: list[tuple[str, object]] = []
        log_poison: list[tuple[str, object]] = []
        _attach_fake_memory_saver(shadow_tp1, log_tp1)
        _attach_fake_memory_saver(shadow_bad, log_bad, fail_resume=True)
        _attach_fake_memory_saver(shadow_poison, log_poison)

        try:
            with _patch_pause_io_types(primary.io_module):
                # String release_tags refused cleanly.
                bad_tags = park_shadow_lab(shadow_tp1, release_tags="kv_cache")  # type: ignore[arg-type]
                self.assertFalse(bad_tags["ok"])
                self.assertIn("sequence", bad_tags["error"])

                # Edited shadow refused at park time!
                edited_sampler = edit_once(
                    shadow_poison._base(SAMPLER), "if value > row[best]:", "if value < row[best]:"
                )
                self.assertTrue(shadow_poison.install({"files": {SAMPLER: edited_sampler}})["ok"])
                edited_park = park_shadow_lab(shadow_poison)
                self.assertFalse(edited_park["ok"])
                self.assertIn("active live edit", edited_park["error"])
                self.assertTrue(shadow_poison.uninstall()["ok"])

                # Poisoned shadow refused at park time!
                shadow_poison.session.fault("participant_failed", "stale poison", "clean")
                poison_park = park_shadow_lab(shadow_poison)
                self.assertFalse(poison_park["ok"])
                self.assertIn("poisoned", poison_park["error"])

                promoter = SglangShadowPromoter()
                replenish_calls: list[str] = []
                driver = SglangShadowPromoter(
                    replenish_fn=lambda lab, _s: replenish_calls.append(lab.name)
                ).recovery_driver()
                primary.recovery_driver = driver
                # Empty queue fails closed without invoking replenish.
                empty_res = primary.reset("respawn")
                self.assertFalse(empty_res["ok"])
                self.assertIn("no parked standby shadow", empty_res["error"])
                self.assertEqual(replenish_calls, [])

                # Unsuccessful receipt rejected on registration.
                hooks_tp1 = SglangQuiesceHooks(shadow_tp1, release_tags=("kv_cache",))
                with self.assertRaises(ValueError):
                    promoter.register_shadow(
                        ParkedSglangShadow(
                            lab=shadow_tp1,
                            hooks=hooks_tp1,
                            receipt=QuiesceReceipt(ok=False, error="quiesce failed"),
                        )
                    )

                # TP mismatch fails closed and KEEPS the healthy shadow in queue (no leak)!
                self.assertTrue(promoter.park_shadow(shadow_tp1)["ok"])
                self.assertEqual(promoter.pending_count(), 1)
                # Duplicate park of already-queued shadow fails closed!
                dup_park = promoter.park_shadow(shadow_tp1)
                self.assertFalse(dup_park["ok"])
                self.assertIn("already parked", dup_park["error"])
                orig_primary_engine = primary.engine
                mismatch_res = promoter.promote(primary)
                self.assertFalse(mismatch_res["ok"])
                self.assertIn("tp_size mismatch", mismatch_res["error"])
                self.assertIs(primary.engine, orig_primary_engine)
                self.assertEqual(promoter.pending_count(), 1)

                # Mismatched (shadow_B, hooks_A, receipt) rejected!
                with self.assertRaises(ValueError):
                    promoter.register_shadow(
                        (shadow_bad, hooks_tp1, QuiesceReceipt(ok=True, released=("kv_cache",)))
                    )

                # close_parked closes the queued healthy shadow cleanly and rejects subsequent parks.
                self.assertEqual(promoter.close_parked(), 1)
                self.assertEqual(promoter.pending_count(), 0)
                self.assertIsNone(shadow_tp1.engine)
                self.assertEqual(promoter.close_parked(), 0)
                closed_park = promoter.park_shadow(shadow_bad)
                self.assertFalse(closed_park["ok"])
                self.assertIn("closed", closed_park["error"])

                # Fresh promoter for self-promotion and resume-failure tests.
                promoter2 = SglangShadowPromoter()
                _attach_fake_memory_saver(primary, [])
                self.assertTrue(promoter2.park_shadow(primary)["ok"])
                self_res = promoter2.promote(primary)
                self.assertFalse(self_res["ok"])
                self.assertIn("distinct", self_res["error"])
                with promoter2._lock:
                    promoter2._shadows.clear()

                # Resume failure closes the failed half-resumed shadow and marks primary dirty.
                self.assertTrue(promoter2.park_shadow(shadow_bad)["ok"])
                resume_fail_res = promoter2.promote(primary)
                self.assertFalse(resume_fail_res["ok"])
                self.assertIn("resume_memory_occupation failed", resume_fail_res["error"])
                self.assertTrue(primary.dirty)
                self.assertIsNone(shadow_bad.engine)
        finally:
            primary.close()
            shadow_tp1.close()
            shadow_bad.close()
            shadow_poison.close()

    def test_shadow_provider_fallback_and_session_recovery_failure(self) -> None:
        """`shadow_provider` supplies a parked shadow when queue is empty; provider errors and
        session recovery failures fail closed with actionable poison findings."""
        primary = ToySglangLab(tp_size=1)
        shadow = ToySglangLab(tp_size=1)
        log: list[tuple[str, object]] = []
        _attach_fake_memory_saver(shadow, log)
        try:
            with _patch_pause_io_types(primary.io_module):
                bad_promoter = SglangShadowPromoter(
                    shadow_provider=lambda _lab, _store: (_ for _ in ()).throw(RuntimeError("pool offline"))
                )
                bad_res = bad_promoter.promote(primary)
                self.assertFalse(bad_res["ok"])
                self.assertIn("shadow_provider failed", bad_res["error"])

                parked = park_shadow_lab(shadow)
                self.assertTrue(parked["ok"], parked)
                good_promoter = SglangShadowPromoter(
                    shadow_provider=lambda _lab, _store: (shadow, parked["hooks"], parked["receipt"])
                )
                primary.recovery_driver = good_promoter.recovery_driver()
                primary.session.fault("participant_failed", "simulated crash", "clean")
                res = primary.reset("respawn")
                self.assertTrue(res["ok"], res)
                self.assertEqual(res["promoted_ranks"], 1)
                self.assertFalse(primary.session.poisoned())
        finally:
            primary.close()
            shadow.close()


if __name__ == "__main__":
    unittest.main()
