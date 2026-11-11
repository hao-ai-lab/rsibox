"""Golden tests of the lab's `rebuild` verb (in-place rebuild) on the FastVideo-shaped toy (CPU,
real spawned workers reached through their pipes).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab -v
"""
import os
import types
import unittest

from adapter.toy_lab import STAGES, OPS, ToyFastVideoLab

BASE = {"video": [123, 177, 231, 34], "audio": [5, 4, 3, 2, 1, 0]}


def edit(src, old, new):
    assert src.count(old) == 1, old
    return src.replace(old, new)


class RebuildTest(unittest.TestCase):
    def setUp(self):
        self.lab = ToyFastVideoLab(ranks=2)
        st = self.lab._base(STAGES)
        # a constructor edit: the stage gets a field, its forward reads it
        self.ctor = edit(edit(st, "class VideoDecodingStage:\n",
                              "class VideoDecodingStage:\n    def __init__(self):\n        self.bias = 2\n\n"),
                         "(v * SCALE + 1) % 251", "(v * SCALE + self.bias) % 251")

    def tearDown(self):
        self.lab.close()

    def test_constructor_edit_refused_by_install_runs_after_rebuild(self):
        """Input: VideoDecodingStage gains `__init__` setting `self.bias = 2`, used in forward.
        Expect: install refuses it (a live swap cannot run a constructor again); rebuild applies it on
        both ranks without a respawn (same worker pids); the next video is the +2 output
        [124, 178, 232, 35]; revert gives the base output back. Proves: the edit the live swap routes to
        a restart runs in the warm workers. Not: a GPU engine (see kb/costs.md receipts)."""
        pids = [self.lab.rank_call(i, "ping")["pid"] for i in range(2)]
        self.assertEqual(self.lab.check()["output"], BASE)
        r = self.lab.install({"files": {STAGES: self.ctor}})
        self.assertIn(r["outcome"], ("refused", "aborted"), r["codes"])
        b = self.lab.rebuild({"files": {STAGES: self.ctor}})
        self.assertEqual(b["outcome"], "rebuilt", b["findings"])
        self.assertEqual(sorted(b["ranks"]), ["rank0", "rank1"])
        self.assertIn(STAGES, b["ranks"]["rank0"]["refreshed"])
        self.assertEqual([self.lab.rank_call(i, "ping")["pid"] for i in range(2)], pids)
        c = self.lab.check()
        self.assertEqual(c["output"]["video"], [124, 178, 232, 35])
        self.assertEqual(self.lab.status()["state"], "base")       # the rebuilt text is the base
        self.assertEqual(self.lab.rebuild({"revert": True})["outcome"], "rebuilt")
        self.assertEqual(self.lab.check()["output"], BASE)

    def test_module_constant_reaches_the_stage_that_imported_it(self):
        """Input: SCALE = 3 in ops (refused by install: module state). Expect: rebuild
        refreshes ops and stages (stages did `from ops import SCALE`), and the video is
        (v * 3 + 1) % 251. Proves: importers of an edited module are rebuilt too."""
        new = edit(self.lab._base(OPS), "SCALE = 2", "SCALE = 3")
        self.assertIn(self.lab.install({"files": {OPS: new}})["outcome"], ("refused", "aborted"))
        b = self.lab.rebuild({"files": {OPS: new}})
        self.assertEqual(b["outcome"], "rebuilt", b["findings"])
        self.assertIn(STAGES, b["ranks"]["rank0"]["refreshed"])
        latents = [61, 88, 115, 142]         # the toy DiT output for seed 1, 4 frames
        self.assertEqual(self.lab.check()["output"]["video"], [(v * 3 + 1) % 251 for v in latents])

    def test_broken_edit_is_refused_and_the_lab_keeps_running_base(self):
        """Input: an edit whose import raises in the workers. Expect: outcome refused with
        finding refresh_failed on every rank, the lab not poisoned, the next video the base
        output. Proves: a failed rebuild leaves the warm workers as they were."""
        bad = self.ctor + "\nraise RuntimeError('import-time failure')\n"
        b = self.lab.rebuild({"files": {STAGES: bad}})
        self.assertEqual(b["outcome"], "refused")
        self.assertEqual({f["check"] for f in b["findings"]}, {"refresh_failed"})
        self.assertNotEqual(self.lab.status()["state"], "poisoned")
        self.assertEqual(self.lab.check()["output"], BASE)

    def test_rebuild_refused_while_live_edits_are_installed(self):
        """Input: a committed live-swap edit, then a rebuild. Expect: refused with
        live_edits_installed (the rebuild would make the edit's base stale)."""
        self.assertEqual(self.lab.install(self.lab.patches()["visible"])["outcome"], "committed")
        b = self.lab.rebuild({"files": {STAGES: self.ctor}})
        self.assertEqual([f["check"] for f in b["findings"]], ["live_edits_installed"])

    def test_live_edit_on_a_rebuilt_module_swaps_against_the_rebuilt_text(self):
        """Input: rebuild SCALE = 3 in ops, then install an edit made on that rebuilt text
        (Transformer `v * 3` -> `v * 5`). Expect: the install commits (no base_mismatch), the
        video is (v * 3 + 1) % 251 over the x5 DiT latents, and undo gives the rebuilt output
        back. Proves: a worker stages a live swap against the text the rebuild runs, not the
        file on disk."""
        rebuilt = edit(self.lab._base(OPS), "SCALE = 2", "SCALE = 3")
        self.assertEqual(self.lab.rebuild({"files": {OPS: rebuilt}})["outcome"], "rebuilt")
        out_rebuilt = self.lab.check()["output"]
        r = self.lab.install({"files": {OPS: edit(rebuilt, "v * 3 + step", "v * 5 + step")}})
        self.assertEqual(r["outcome"], "committed", r["findings"])
        latents = [1, 2, 3, 4, 5, 6]                 # seed 1, 4 frames: geometry 6
        for s in (3, 2, 1):
            latents = [v * 5 + s for v in latents]
        self.assertEqual(self.lab.check()["output"]["video"], [(v * 3 + 1) % 251 for v in latents[:4]])
        self.assertTrue(self.lab.uninstall()["ok"])
        self.assertEqual(self.lab.check()["output"], out_rebuilt)

    def broken_stage_build(self):
        """ops: the DiT edited (a class swap); stages: VideoDecodingStage.__init__ raises, so
        the stage build fails after the swap."""
        ops = edit(self.lab._base(OPS), "v * 3 + step", "v * 99 + step")
        stages = edit(self.lab._base(STAGES), "class VideoDecodingStage:\n",
                      "class VideoDecodingStage:\n    def __init__(self):\n        raise RuntimeError('boom')\n\n")
        return {"files": {OPS: ops, STAGES: stages}}

    def test_stage_build_failure_rolls_back_every_rank(self):
        """Input: one rebuild that edits the DiT (`v * 3` -> `v * 99`) and gives
        VideoDecodingStage an `__init__` that raises. Expect: refused with rebuild_failed, the
        next video the base output, the lab at base (not tampered), and in the workers the DiT
        has its old class and the old stage class is loaded again. Proves: a rebuild that fails
        after it started to change the workers puts them back."""
        b = self.lab.rebuild(self.broken_stage_build())
        self.assertEqual(b["outcome"], "refused")
        self.assertEqual({f["check"] for f in b["findings"]}, {"rebuild_failed"})
        self.assertEqual(self.lab.check()["output"], BASE)
        probe = self.lab.repl("import sys\nout = type(pipeline.modules['transformer']) is "
                              "sys.modules['adapter.toyfv.ops'].Transformer and "
                              "'__init__' not in vars(sys.modules['adapter.toyfv.stages'].VideoDecodingStage)")
        self.assertEqual([v["out"] for v in probe["ranks"].values()], ["True", "True"], probe)
        self.assertEqual(self.lab.status()["state"], "base")

    def test_a_rank_in_unknown_state_after_a_rebuild_poisons_the_lab(self):
        """Input: rank 1 is lost (its calls fail), then a rebuild that rank 0 refuses (its stage
        build fails). Expect: outcome failed and the lab poisoned. Proves: a refused rebuild
        keeps the lab trusted only when every rank says it changed nothing; rank 1 cannot."""
        self.assertTrue(self.lab.inject("lost", participant="rank1")["ok"])
        b = self.lab.rebuild(self.broken_stage_build())
        self.assertEqual(b["outcome"], "failed")
        self.assertEqual(self.lab.status()["state"], "poisoned")

    def test_a_later_install_keeps_the_rebuilt_source_visible(self):
        """Input: install and uninstall an edit of ops, rebuild ops (a marker comment), then
        install an unrelated edit of stages. Expect: `inspect.getsource(ops)` in each worker is
        the rebuilt text after the last install. Proves: an install never drops the linecache
        entry of a rebuilt module (what inspect, Triton and Dynamo read)."""
        ops = self.lab._base(OPS)
        self.assertTrue(self.lab.install({"files": {OPS: edit(ops, "v * 3 + step", "v * 5 + step")}})["ok"])
        self.assertTrue(self.lab.uninstall()["ok"])
        rebuilt = edit(ops, "SCALE = 2", "SCALE = 2  # REBUILT")
        self.assertEqual(self.lab.rebuild({"files": {OPS: rebuilt}})["outcome"], "rebuilt")
        self.assertTrue(self.lab.install(self.lab.patches()["visible"])["ok"])
        src = self.lab.repl("import inspect, adapter.toyfv.ops as m\nout = 'REBUILT' in inspect.getsource(m)")
        self.assertEqual([v["out"] for v in src["ranks"].values()], ["True", "True"], src)


class RebuildInProcessTest(unittest.TestCase):
    """agent.rebuild in this process, on a plain (non-torch) component."""

    def setUp(self):
        import importlib
        import sys
        import tempfile
        self.dir = tempfile.mkdtemp(prefix="fvrb-")
        os.makedirs(os.path.join(self.dir, "slotpkg"))
        for n, t in (("__init__.py", ""), ("m.py", "class Comp:\n    def run(self):\n        return 1\n")):
            with open(os.path.join(self.dir, "slotpkg", n), "w") as f:
                f.write(t)
        sys.path.insert(0, self.dir)
        importlib.invalidate_caches()
        import slotpkg.m
        self.old_module = slotpkg.m
        comp = slotpkg.m.Comp()
        self.worker = types.SimpleNamespace(rank=0, pipeline=types.SimpleNamespace(modules={"comp": comp}))

    def tearDown(self):
        import shutil
        import sys
        for n in [n for n in sys.modules if n.split(".")[0] == "slotpkg"]:
            del sys.modules[n]
        sys.path.remove(self.dir)
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_failed_class_swap_refuses_and_restores_the_modules(self):
        """Input: the edit gives Comp `__slots__` (a live object cannot take a class of another
        layout). Expect: ok False with class_swap_failed, `clean` (nothing left changed), the
        live object keeps its class and `sys.modules` holds the old module again. Proves: a
        class swap that fails is not reported as a rebuild."""
        import sys
        from adapter import agent
        r = agent.rebuild(self.worker, {"files": {"slotpkg.m": "class Comp:\n    __slots__ = ('a',)\n\n"
                                                    "    def run(self):\n        return 2\n"}})
        self.assertFalse(r["ok"])
        self.assertEqual([f["check"] for f in r["findings"]], ["class_swap_failed"])
        self.assertTrue(r.get("clean"))
        self.assertIs(sys.modules["slotpkg.m"], self.old_module)
        self.assertIs(type(self.worker.pipeline.modules["comp"]), self.old_module.Comp)
        self.assertEqual(self.worker.pipeline.modules["comp"].run(), 1)


if __name__ == "__main__":
    unittest.main()
