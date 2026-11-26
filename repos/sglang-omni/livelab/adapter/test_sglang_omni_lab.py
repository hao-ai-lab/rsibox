"""Unit, golden, and conformance tests for `SglangOmniLab` / `ToySglangOmniLab`."""
from __future__ import annotations

import json, linecache, os, subprocess, sys, unittest

import adapter  # noqa: F401 - puts core/livelab on sys.path
from adapter.lab import SglangOmniLab
from adapter.toy_lab import ToySglangOmniLab
from conformance import cases
from rsibox import harness


def _cuda_omni_ready() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") == "":
        return False
    model = os.environ.get("RSIBOX_SGLANG_OMNI_MODEL", "").strip()
    if not model or not os.path.isdir(model):
        return False
    try:
        import torch, transformers  # type: ignore[import-not-found]  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class SglangOmniConformanceTests(unittest.TestCase):
    """Runs all 11/11 `core/livelab/conformance` cases (`c0`–`c10`) on `ToySglangOmniLab`."""

    def test_full_conformance_c0_to_c10_cpu(self) -> None:
        failures = []
        for name in cases.CASES:
            res = cases.run(name, ToySglangOmniLab)
            if res["result"] != "pass":
                failures.append((name, res["result"], res["error"]))
        self.assertEqual(failures, [], f"conformance failures: {failures}")


class SglangOmniGoldenTests(unittest.TestCase):
    """Multi-stage targeted swap, cross-stage stacking, S4/S5 services, and boundary checks."""

    def setUp(self) -> None:
        self.lab = ToySglangOmniLab()

    def tearDown(self) -> None:
        self.lab.close()

    def test_targeted_stage_swap_preserves_frozen_stages_and_resets_s0_buffer(self) -> None:
        c0 = self.lab.check()
        self.assertTrue(c0["ok"], c0)
        tel0 = c0["telemetry"]
        self.assertFalse(tel0["thinker_cache_hit"])
        self.assertEqual(tel0["aux_stats"]["stage_count"], 2)
        epoch0 = tel0["buffer_stats"]["epoch"]

        c_warm = self.lab.check()
        self.assertTrue(c_warm["ok"] and c_warm["output"] == c0["output"])
        self.assertTrue(c_warm["telemetry"]["thinker_cache_hit"])

        # Targeted swap on talker abstains thinker/dit/vocoder and resets S0 stage buffer.
        res = self.lab.install(self.lab.patches()["visible"])
        self.assertTrue(res["ok"], res)
        c1 = self.lab.check()
        self.assertTrue(c1["ok"] and c1["proven"] is True)
        self.assertEqual(c1["abstained"], ["dit", "thinker", "vocoder"])
        self.assertEqual([k for k, v in c1["witness"].items() if v], ["talker"])
        self.assertNotEqual(c1["output"], c0["output"])
        self.assertGreater(c1["telemetry"]["buffer_stats"]["epoch"], epoch0)
        self.assertTrue(c1["telemetry"]["thinker_cache_hit"])

        # Targeted swap on dit stage only abstains talker/thinker/vocoder.
        self.assertTrue(self.lab.uninstall()["ok"])
        d_src = self.lab._base_source("adapter.dit_op")
        dit_patch = {"files": {"adapter.dit_op": harness.edit_once(d_src, "steps = max(1, int(cfg.dit_steps))", "steps = max(1, int(cfg.dit_steps) + 2)")}, "stages": ["dit"]}
        r_dit = self.lab.install(dit_patch)
        self.assertTrue(r_dit["ok"], r_dit)
        c_dit = self.lab.check()
        self.assertTrue(c_dit["ok"] and c_dit["proven"] is True)
        self.assertEqual(c_dit["abstained"], ["talker", "thinker", "vocoder"])
        self.assertEqual([k for k, v in c_dit["witness"].items() if v], ["dit"])
        self.assertNotEqual(c_dit["output"], c0["output"])
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], c0["output"])

    def test_cross_stage_stacked_and_multistage_swaps_no_code_collision(self) -> None:
        """Gate 2 P1-1 regression: both simultaneous multi-stage and stacked cross-stage swaps work cleanly."""
        c0 = self.lab.check()
        t_src = self.lab._base_source("adapter.talker_op")
        d_src = self.lab._base_source("adapter.dit_op")
        t_edit = harness.edit_once(t_src, "    prosody_gain = 1.0", "    prosody_gain = 1.75")
        d_edit = harness.edit_once(d_src, "steps = max(1, int(cfg.dit_steps))", "steps = max(1, int(cfg.dit_steps) + 3)")

        # 1. Single multi-stage patch across talker_op + dit_op (stages=["dit", "talker"]).
        multi_res = self.lab.install({"files": {"adapter.talker_op": t_edit, "adapter.dit_op": d_edit}, "stages": ["dit", "talker"]})
        self.assertTrue(multi_res["ok"], multi_res)
        cm = self.lab.check()
        self.assertTrue(cm["ok"] and cm["proven"] is True, cm)
        self.assertEqual(sorted(k for k, v in cm["witness"].items() if v), ["dit", "talker"])
        self.assertEqual(cm["abstained"], ["thinker", "vocoder"])
        fresh_m = self.lab.fresh_check({"files": {"adapter.talker_op": t_edit, "adapter.dit_op": d_edit}})
        self.assertTrue(fresh_m["ok"] and fresh_m["output"] == cm["output"], fresh_m)
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], c0["output"])

        # 2. Stacked cross-stage edits: install talker, then stack dit -> undo one by one to base.
        self.assertTrue(self.lab.install({"files": {"adapter.talker_op": t_edit}, "stages": ["talker"]})["ok"])
        c_t = self.lab.check()
        self.assertTrue(c_t["ok"] and c_t["proven"] is True)
        self.assertTrue(self.lab.install({"files": {"adapter.dit_op": d_edit}, "stages": ["dit"]})["ok"])
        c_td = self.lab.check()
        self.assertTrue(c_td["ok"] and c_td["proven"] is True)
        self.assertEqual(c_td["output"], cm["output"])
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], c_t["output"])
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], c0["output"])

        # 3. Mismatched stage owner is refused fail-closed.
        bad = self.lab.install({"files": {"adapter.dit_op": d_edit}, "stages": ["talker"]})
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["codes"], ["patch_invalid"])

    def test_refuses_constructor_edit_ro_lease_and_restores_linecache_on_refusal(self) -> None:
        from rsibox.services import MemoryPoolKind, MemoryService, PoolAccessError
        mem = MemoryService()
        self.addCleanup(mem.teardown)
        h = mem.register_region(MemoryPoolKind.RO_AUX_WEIGHTS, "sglang-omni", "thinker", size_bytes=64)
        with self.assertRaises(PoolAccessError):
            mem.acquire_lease(h.handle_id, "talker", writable=True)

        fp0 = self.lab.status()["fingerprint"]
        s_mod = "adapter.stage_ops"
        ctor_patch = {"files": {s_mod: harness.edit_once(self.lab._base_source(s_mod), "self.stage_scale = init_stage_scale()", "self.stage_scale = init_stage_scale() * 2.0")}}
        res = self.lab.install(ctor_patch)
        self.assertFalse(res["ok"])
        self.assertIn("constructor_edit", res["codes"])
        self.assertEqual(self.lab.status()["fingerprint"], fp0)

        # P2-1 regression: syntax refusal leaves worker linecache clean (not holding broken text).
        bad_syn = self.lab.patches()["refused"]["syntax"]
        ref = self.lab.install(bad_syn)
        self.assertFalse(ref["ok"])
        t_path = harness.module_file("adapter.talker_op")
        assert t_path is not None
        linecache.checkcache(t_path)
        self.assertNotIn("prosody_gain = (", "".join(linecache.getlines(t_path)))

    def test_launch_environment_zero_eager_gpu_imports(self) -> None:
        ll = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        probe = "import json, sys; from adapter.lab import SglangOmniLab; env = SglangOmniLab.launch_environment(['--', '--mode', 'real', '--image-size', '128', '--max-new-tokens', '12']); print(json.dumps({'env': env, 'gpu_mods': [m for m in ('torch', 'sglang', 'transformers') if m in sys.modules]}))"
        proc = subprocess.run([sys.executable, "-c", probe], env={**os.environ, "PYTHONPATH": ll}, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout.strip())
        self.assertEqual(data["env"]["RSIBOX_SGLANG_OMNI_IMAGE_SIZE"], "128")
        self.assertEqual(data["env"]["RSIBOX_SGLANG_OMNI_MAX_NEW_TOKENS"], "12")
        self.assertEqual(data["gpu_mods"], [])


class SglangOmniRealGpuTests(unittest.TestCase):
    """Real CUDA multimodal test (auto-skips on CPU-only hosts)."""

    def test_real_b200_multistage_frozen_vram_parity_and_rollback(self) -> None:
        if not _cuda_omni_ready():
            self.skipTest("requires CUDA GPU and RSIBOX_SGLANG_OMNI_MODEL")
        lab = SglangOmniLab({"mode": "real", "model_path": os.environ["RSIBOX_SGLANG_OMNI_MODEL"], "image_size": 64, "max_new_tokens": 8, "dit_steps": 2, "seed": 0})
        self.addCleanup(lab.close)
        c0 = lab.check()
        self.assertTrue(c0["ok"], c0)
        self.assertFalse(c0["telemetry"]["thinker_cache_hit"])
        base_vram = c0["telemetry"]["frozen_vram_baseline"]
        self.assertEqual(c0["telemetry"]["frozen_vram_now"], base_vram)
        c_warm = lab.check()
        self.assertTrue(c_warm["ok"] and c_warm["output"] == c0["output"])
        self.assertTrue(c_warm["telemetry"]["thinker_cache_hit"])
        patch = lab.patches()["visible"]
        self.assertTrue(lab.install(patch)["ok"])
        c1 = lab.check()
        self.assertTrue(c1["ok"] and c1["proven"] is True)
        self.assertEqual(c1["telemetry"]["frozen_vram_now"], base_vram)
        self.assertNotEqual(c1["output"], c0["output"])
        fresh = lab.fresh_check(patch)
        self.assertTrue(fresh["ok"] and fresh["output"] == c1["output"], fresh)
        self.assertTrue(lab.uninstall()["ok"])
        c2 = lab.check()
        self.assertEqual(c2["output"], c0["output"])
        self.assertEqual(c2["telemetry"]["frozen_vram_now"], base_vram)


if __name__ == "__main__":
    unittest.main()
