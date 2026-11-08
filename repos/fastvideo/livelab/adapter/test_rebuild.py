"""Golden tests of the lab's `rebuild` verb (in-place rebuild) on the FastVideo-shaped toy (CPU,
real spawned workers reached through their pipes).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab -v
"""
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


if __name__ == "__main__":
    unittest.main()
