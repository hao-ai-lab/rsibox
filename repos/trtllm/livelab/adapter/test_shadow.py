"""Golden CPU tests for `TrtllmShadowPromoter` (`repos/trtllm/livelab/adapter/shadow.py`)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from adapter import ParkedTrtllmShadow, TrtllmShadowPromoter, park_shadow_lab
from adapter.toy_lab import ToyTrtllmLab
from rsibox.warm import AsyncShadowReplenisher, DefaultRespawnDriver, QuiesceReceipt


class TrtllmShadowPromoterTests(unittest.TestCase):
    def test_park_promote_after_crash_preserves_parity_live_swap_and_clean_shell_close(self) -> None:
        """Golden case: parking a 2-rank ToyTrtllmLab shadow releases kv_cache; after primary
        crash/poison, lab.reset('respawn') promotes the shadow cleanly, restores bit-exact
        output, supports subsequent live swaps + fresh_check parity, and survives shell close."""
        primary = ToyTrtllmLab(ranks=2)
        self.addCleanup(primary.close)
        base_out = primary.check()["output"]

        shadow = ToyTrtllmLab(ranks=2)
        promoter = TrtllmShadowPromoter()
        self.addCleanup(promoter.close)
        parked = promoter.park_shadow(shadow)
        self.assertTrue(parked["ok"], parked)
        self.assertEqual(parked["released"], ["kv_cache"])
        self.assertEqual(promoter.pending_count(), 1)
        primary.recovery_driver = promoter.as_recovery_driver()

        # Poison primary via real crash fault, confirm install is refused while poisoned.
        self.assertTrue(primary.inject("crash")["ok"])
        primary.check()
        self.assertEqual(primary.status()["state"], "poisoned")
        self.assertFalse(primary.install(primary.patches()["visible"])["ok"])

        # Promote parked shadow!
        rec = primary.reset("respawn")
        self.assertTrue(rec["ok"], rec)
        self.assertEqual(rec["driver"], "standby_shadow")
        self.assertEqual(rec["resumed"], ["kv_cache"])
        self.assertEqual(promoter.pending_count(), 0)
        self.assertEqual(primary.status()["state"], "base")
        self.assertEqual(primary.check()["output"], base_out)

        # Closing the detached shadow shell again must NOT kill the promoted primary workers!
        shadow.close()
        self.assertEqual(primary.check()["output"], base_out)

        # Promoted lab supports full live edit + witness proof + fresh_check parity + uninstall.
        visible = primary.patches()["visible"]
        inst = primary.install(visible)
        self.assertTrue(inst["ok"], inst)
        chk = primary.check()
        self.assertTrue(chk["ok"] and chk["proven"], chk)
        self.assertNotEqual(chk["output"], base_out)
        fresh = primary.fresh_check(visible)
        self.assertTrue(fresh["ok"] and fresh["output"] == chk["output"], fresh)
        self.assertTrue(primary.uninstall()["ok"])
        self.assertEqual(primary.check()["output"], base_out)

    def test_dead_head_shadow_is_filtered_and_skipped_to_promote_healthy_shadow2(self) -> None:
        """Golden case: SIGKILLing parked shadow #1 drops pending_count 2 -> 1 and causes
        promote() to skip dead shadow #1 cleanly and promote healthy shadow #2."""
        primary = ToyTrtllmLab(ranks=2)
        self.addCleanup(primary.close)
        base_out = primary.check()["output"]

        sh1 = ToyTrtllmLab(ranks=2)
        sh2 = ToyTrtllmLab(ranks=2)
        with TrtllmShadowPromoter() as promoter:
            self.assertTrue(promoter.park_shadow(sh1)["ok"])
            self.assertTrue(promoter.park_shadow(sh2)["ok"])
            self.assertEqual(promoter.pending_count(), 2)

            # SIGKILL shadow #1 rank0 worker while parked!
            os.kill(sh1.workers[0].process.pid, signal.SIGKILL)
            sh1.workers[0].process.wait(timeout=5.0)
            self.assertEqual(promoter.pending_count(), 1)

            primary.recovery_driver = promoter.as_recovery_driver()
            self.assertTrue(primary.inject("tamper")["ok"])
            self.assertEqual(primary.status()["state"], "poisoned")

            rec = primary.reset("respawn")
            self.assertTrue(rec["ok"], rec)
            self.assertEqual(rec["skipped_dead_shadows"], 1)
            self.assertEqual(promoter.pending_count(), 0)
            self.assertEqual(primary.check()["output"], base_out)

    def test_multi_shadow_async_replenisher_early_wake_and_fallback_on_exhaustion(self) -> None:
        """Golden case: ready_count_fn=promoter.pending_count skips in_flight replenisher stall
        when shadow #2 is already parked, and falls back cleanly to DefaultRespawnDriver when empty."""
        primary = ToyTrtllmLab(ranks=2)
        self.addCleanup(primary.close)
        base_out = primary.check()["output"]

        sh1 = ToyTrtllmLab(ranks=2)
        sh2 = ToyTrtllmLab(ranks=2)
        promoter = TrtllmShadowPromoter()
        self.addCleanup(promoter.close)
        self.assertTrue(promoter.park_shadow(sh1)["ok"])
        self.assertTrue(promoter.park_shadow(sh2)["ok"])

        rep_started, rep_release = threading.Event(), threading.Event()
        self.addCleanup(rep_release.set)

        def slow_replenish(_lab, _store):
            rep_started.set()
            rep_release.wait(timeout=5.0)
            return "restocked"

        replenisher = AsyncShadowReplenisher(slow_replenish)
        self.addCleanup(replenisher.close)
        primary.recovery_driver = promoter.as_recovery_driver(
            replenish_fn=replenisher,
            replenish_wait_timeout_s=5.0,
            fallback_driver=DefaultRespawnDriver(),
        )

        # Crash #1 promotes sh1 and triggers slow background replenishment.
        self.assertTrue(primary.inject("crash")["ok"])
        primary.check()
        rec1 = primary.reset("respawn")
        self.assertTrue(rec1["ok"], rec1)
        self.assertEqual(rec1["driver"], "standby_shadow")
        self.assertTrue(rep_started.wait(timeout=5.0))

        # Crash #2 happens WHILE replenisher is still in_flight, but sh2 is ready -> promotes IMMEDIATELY!
        self.assertTrue(primary.inject("crash")["ok"])
        primary.check()
        t0 = time.monotonic()
        rec2 = primary.reset("respawn")
        elapsed = time.monotonic() - t0
        self.assertTrue(rec2["ok"], rec2)
        self.assertEqual(rec2["driver"], "standby_shadow")
        self.assertLess(elapsed, 2.5, f"should NOT wait 5s for in_flight replenisher when sh2 is ready (took {elapsed:.2f}s)")
        self.assertEqual(primary.check()["output"], base_out)

        # Release replenisher and close promoter -> Crash #3 cleanly falls back to DefaultRespawnDriver!
        rep_release.set()
        replenisher.wait(timeout_s=5.0)
        promoter.close()
        self.assertTrue(primary.inject("crash")["ok"])
        primary.check()
        rec3 = primary.reset("respawn")
        self.assertTrue(rec3["ok"], rec3)
        self.assertEqual(rec3["driver"], "default_respawn")
        self.assertEqual(rec3["fallback_from"], "standby_shadow")
        self.assertEqual(primary.check()["output"], base_out)

    def test_validation_rank_mismatch_active_edit_and_close_cleanup(self) -> None:
        """Golden case: invalid inputs, active-edit shadows, rank mismatches, duplicates, and
        closed promoters fail closed cleanly without leaking workers."""
        with self.assertRaises(TypeError):
            TrtllmShadowPromoter(shadow_provider="bad")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            TrtllmShadowPromoter(replenish_fn=123)  # type: ignore[arg-type]

        # Active edit on shadow is refused by park_shadow_lab.
        dirty_shadow = ToyTrtllmLab(ranks=2)
        self.addCleanup(dirty_shadow.close)
        self.assertTrue(dirty_shadow.install(dirty_shadow.patches()["visible"])["ok"])
        refused = park_shadow_lab(dirty_shadow)
        self.assertFalse(refused["ok"])
        self.assertIn("active live edit", refused["error"])
        self.assertTrue(dirty_shadow.uninstall()["ok"])
        self.assertFalse(park_shadow_lab(dirty_shadow, release_tags="kv_cache")["ok"])  # type: ignore[arg-type]

        # Duplicate registration & bad receipt fail closed.
        ok_park = park_shadow_lab(dirty_shadow)
        self.assertTrue(ok_park["ok"], ok_park)
        promoter = TrtllmShadowPromoter(ok_park["shadow"])
        self.addCleanup(promoter.close)
        with self.assertRaises(ValueError):
            promoter.register_shadow(ok_park["shadow"])
        with self.assertRaises(ValueError):
            promoter.register_shadow(
                ParkedTrtllmShadow(dirty_shadow, ok_park["hooks"], QuiesceReceipt(ok=False, error="bad"))
            )

        # Rank mismatch (1-rank shadow vs 2-rank primary) is skipped/refused cleanly!
        sh_r1 = ToyTrtllmLab(ranks=1)
        self.addCleanup(sh_r1.close)
        p_mm = TrtllmShadowPromoter()
        self.addCleanup(p_mm.close)
        self.assertTrue(p_mm.park_shadow(sh_r1)["ok"])
        primary_r2 = ToyTrtllmLab(ranks=2)
        self.addCleanup(primary_r2.close)
        res_mm = p_mm.promote(primary_r2)
        self.assertFalse(res_mm["ok"])
        self.assertEqual(res_mm["skipped_dead_shadows"], 1)
        # Primary was NOT stopped because candidate failed pre-checks before stop!
        self.assertTrue(primary_r2.check()["ok"])

        # Closing promoter tears down all remaining parked shadow workers cleanly.
        promoter.close()
        self.assertTrue(promoter.is_closed())
        self.assertEqual(promoter.pending_count(), 0)
        self.assertEqual(dirty_shadow.workers, [])

    def test_shadow_provider_resume_exception_failover_fallback_and_close_race(self) -> None:
        """Regression suite for P0/P1/P2 fixes:
        1. shadow_provider works via as_recovery_driver() even when queue starts empty.
        2. resume_after_restore() raising an exception closes bad shadow #1 and fails over to shadow #2.
        3. Failed resume on sole shadow preserves real error, sets lab._dirty=True, and falls back when configured.
        4. Mismatched engine_mode / model shadow is rejected without stopping primary (and triggers fallback via driver).
        5. Concurrent promoter.close() during park_shadow() closes the quiesced shadow without leaking workers.
        6. Importing adapter.shadow never eagerly imports torch or tensorrt_llm.
        """
        # 1. shadow_provider invoked cleanly through as_recovery_driver()!
        primary = ToyTrtllmLab(ranks=2)
        self.addCleanup(primary.close)
        base_out = primary.check()["output"]
        provider_calls: list[int] = []

        def _provider(_lab, _store):
            provider_calls.append(1)
            sh = ToyTrtllmLab(ranks=2)
            parked = park_shadow_lab(sh)
            assert parked["ok"], parked
            return parked["shadow"]

        p_prov = TrtllmShadowPromoter(shadow_provider=_provider)
        self.addCleanup(p_prov.close)
        primary.recovery_driver = p_prov.as_recovery_driver()
        self.assertTrue(primary.inject("crash")["ok"])
        primary.check()
        rec_p = primary.reset("respawn")
        self.assertTrue(rec_p["ok"], rec_p)
        self.assertEqual(rec_p["driver"], "standby_shadow")
        self.assertEqual(len(provider_calls), 1)
        self.assertEqual(primary.check()["output"], base_out)

        # 2. resume_after_restore() raising an exception closes shadow #1 and promotes shadow #2!
        sh_bad = ToyTrtllmLab(ranks=2)
        sh_good = ToyTrtllmLab(ranks=2)
        p_ex = TrtllmShadowPromoter()
        self.addCleanup(p_ex.close)
        pk_bad = p_ex.park_shadow(sh_bad)
        self.assertTrue(pk_bad["ok"])
        self.assertTrue(p_ex.park_shadow(sh_good)["ok"])
        pk_bad["hooks"].resume_after_restore = lambda _r: (_ for _ in ()).throw(RuntimeError("resume-exploded"))
        primary.recovery_driver = p_ex.as_recovery_driver()
        self.assertTrue(primary.inject("crash")["ok"])
        primary.check()
        rec_ex = primary.reset("respawn")
        self.assertTrue(rec_ex["ok"], rec_ex)
        self.assertEqual(rec_ex["skipped_dead_shadows"], 1)
        self.assertEqual(sh_bad.workers, [])
        self.assertEqual(primary.check()["output"], base_out)

        # 3. Sole shadow failing resume preserves real error, sets lab._dirty=True, and falls back when configured!
        sh_only = ToyTrtllmLab(ranks=2)
        p_fail = TrtllmShadowPromoter()
        self.addCleanup(p_fail.close)
        pk_only = p_fail.park_shadow(sh_only)
        self.assertTrue(pk_only["ok"])
        pk_only["hooks"].resume_after_restore = lambda _r: {"ok": False, "error": "pointer-shift-boom"}
        direct_fail = p_fail.promote(primary)
        self.assertFalse(direct_fail["ok"])
        self.assertIn("pointer-shift-boom", direct_fail["error"])
        self.assertTrue(direct_fail["primary_stopped"])
        self.assertTrue(primary._dirty)
        self.assertTrue(primary.session.poisoned())
        # Recover via DefaultRespawnDriver!
        primary.recovery_driver = DefaultRespawnDriver()
        self.assertTrue(primary.reset("respawn")["ok"])

        # 4. Mismatched model shadow is rejected via promote() without stopping primary, and triggers fallback via driver!
        sh_mm = ToyTrtllmLab(ranks=2)
        self.addCleanup(sh_mm.close)
        sh_mm.engine_config["model"] = "other-model"
        p_mm = TrtllmShadowPromoter()
        self.addCleanup(p_mm.close)
        self.assertTrue(p_mm.park_shadow(sh_mm)["ok"])
        res_mm = p_mm.promote(primary)
        self.assertFalse(res_mm["ok"])
        self.assertIn("model mismatch", res_mm["error"])
        self.assertFalse(res_mm["primary_stopped"])
        self.assertTrue(primary.check()["ok"])

        # Via driver + fallback_driver, rank mismatch cleanly falls back to DefaultRespawnDriver!
        sh_r1 = ToyTrtllmLab(ranks=1)
        self.addCleanup(sh_r1.close)
        p_fb = TrtllmShadowPromoter()
        self.addCleanup(p_fb.close)
        self.assertTrue(p_fb.park_shadow(sh_r1)["ok"])
        primary.recovery_driver = p_fb.as_recovery_driver(fallback_driver=DefaultRespawnDriver())
        rec_fb = primary.reset("respawn")
        self.assertTrue(rec_fb["ok"], rec_fb)
        self.assertEqual(rec_fb["driver"], "default_respawn")
        self.assertEqual(rec_fb["fallback_from"], "standby_shadow")
        self.assertEqual(primary.check()["output"], base_out)

        # 5. Concurrent promoter.close() during park_shadow() closes the quiesced shadow cleanly!
        sh_race = ToyTrtllmLab(ranks=2)
        self.addCleanup(sh_race.close)
        p_race = TrtllmShadowPromoter()
        orig_park = park_shadow_lab

        def _park_then_close(lab, **kw):
            out = orig_park(lab, **kw)
            p_race.close()
            return out

        with mock.patch("adapter.shadow.park_shadow_lab", side_effect=_park_then_close):
            race_res = p_race.park_shadow(sh_race)
        self.assertFalse(race_res["ok"])
        self.assertIn("closed", race_res["error"])
        self.assertEqual(sh_race.workers, [])

        # 6. Lazy import check: importing adapter.shadow never loads torch or tensorrt_llm!
        livelab_dir = str(Path(__file__).resolve().parents[1])
        core_dir = str(Path(__file__).resolve().parents[4] / "core" / "livelab")
        probe = (
            "import importlib, json, sys\n"
            "importlib.import_module('adapter.shadow')\n"
            "bad = [m for m in ('torch', 'tensorrt_llm', 'cuda', 'triton') if m in sys.modules]\n"
            "print(json.dumps({'bad': bad}))\n"
        )
        proc_res = subprocess.run(
            [sys.executable, "-c", probe],
            env={**os.environ, "PYTHONPATH": os.pathsep.join([livelab_dir, core_dir]), "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True, text=True, timeout=15, check=True,
        )
        self.assertEqual(proc_res.stdout.strip().splitlines()[-1], '{"bad": []}')


if __name__ == "__main__":
    unittest.main()
