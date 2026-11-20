"""Golden tests of diff patches (`adapter/diffs.py`, `FastVideoLab.files`) on the toy (CPU).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/adapter -t repos/fastvideo/livelab -v

A `.patch` file (`patches/h3-hillclimb/`) is installed as `lab.install({"diff": text})`. The
diff is applied to the lab's live text, so a stack of patches made one on top of the other
installs in order, and a diff made against other code is refused before any rank changes.
"""
import os
import unittest

from adapter import diffs
from adapter.toy_lab import STAGES, ToyFastVideoLab

BASE = {"video": [123, 177, 231, 34], "audio": [5, 4, 3, 2, 1, 0]}
PATH = "adapter/toyfv/stages.py"


def diff(old, new, path=PATH):
    """A `git diff`-style hunk replacing line 8, `old`, by `new` (one context line each side)."""
    return (f"--- a/{path}\n+++ b/{path}\n@@ -7,3 +7,3 @@\n"
            f"         n = layout(batch)\n-{old}\n+{new}\n \n")


LINE = '        return [(v * SCALE + {k}) % 251 for v in batch["latents"][:n]]'


class DiffPatchTest(unittest.TestCase):
    def setUp(self):
        self.lab = ToyFastVideoLab(ranks=2)

    def tearDown(self):
        self.lab.close()

    def test_two_stacked_patch_files_install_in_order_and_fresh_equals_live(self):
        """Input: patch 1 (`+ 1` -> `+ 2`), then patch 2 made on top of it (`+ 2` -> `+ 3`), as
        `{"diff": text}`. Expect: both commit; the video is [125, 179, 233, 36] (each value +2
        over base), every rank ran the new code, a fresh process started from base files plus
        both diffs gives the same video, and undoing both returns the base output. Proves: a
        diff applies to the live text (patch 2 does not apply to the file on disk) and to the
        disk for fresh_check. Not: H3 (the GPU run installs patches/h3-hillclimb this way)."""
        p1 = {"diff": diff(LINE.format(k=1), LINE.format(k=2))}
        p2 = {"diff": diff(LINE.format(k=2), LINE.format(k=3))}
        self.assertEqual(self.lab.install(p1)["outcome"], "committed")
        r = self.lab.install(p2)
        self.assertEqual(r["outcome"], "committed", r["codes"])
        c = self.lab.check()
        self.assertEqual(c["output"]["video"], [125, 179, 233, 36])
        self.assertTrue(c["proven"])
        live = self.lab.live_text[-1][STAGES]                # base + p1 + p2
        self.assertIn("SCALE + 3", live)
        both = {"files": {STAGES: live}}
        self.assertEqual(self.lab.fresh_check(both)["output"], c["output"])
        self.assertEqual(self.lab.uninstall(self.lab.status()["txns"][0])["ok"], True)
        self.assertEqual(self.lab.check()["output"], BASE)

    def test_a_diff_made_against_other_code_is_refused_and_nothing_changes(self):
        """Input: patch 2 alone (its context line says `+ 2`; the live code says `+ 1`).
        Expect: outcome refused with `patch_invalid` naming git's error, no transaction, the
        base output. Proves: a stale or misordered patch file is never applied with fuzz to
        the wrong lines. Not: two diffs that both apply but conflict in meaning."""
        r = self.lab.install({"diff": diff(LINE.format(k=2), LINE.format(k=3))})
        self.assertEqual(r["outcome"], "refused")
        self.assertEqual(r["codes"], ["patch_invalid"])
        self.assertIn("does not apply", r["findings"][0]["observed"])
        self.assertEqual(self.lab.status()["state"], "base")
        self.assertEqual(self.lab.check()["output"], BASE)

    def test_a_rebuild_takes_a_diff_and_the_next_diff_applies_to_the_rebuilt_text(self):
        """Input: rebuild with a diff that changes `SCALE = 2` to `SCALE = 3` (a module
        constant: the live swap refuses it), then install a diff made on top of it. Expect:
        rebuilt, then committed; the video uses SCALE 3. Proves: rebuild's text becomes the
        text later diffs apply to. Not: rebuild's rollback (test_rebuild.py)."""
        ops = "adapter/toyfv/ops.py"
        scale = (f"--- a/{ops}\n+++ b/{ops}\n@@ -1,3 +1,3 @@\n"
                 ' """Toy engine code the conformance edits change: a pipeline of stages, as '
                 "FastVideo's.\"\"\"\n-SCALE = 2\n+SCALE = 3\n \n")
        r = self.lab.rebuild({"diff": scale})
        self.assertEqual(r["outcome"], "rebuilt", r["findings"])
        on_top = scale.replace("-SCALE = 2\n+SCALE = 3", "-SCALE = 3\n+SCALE = 4")
        self.assertEqual(self.lab.files({"diff": on_top})["adapter.toyfv.ops"].count("SCALE = 4"), 1)
        self.assertNotEqual(self.lab.check()["output"], BASE)

    def test_paths_and_refusals_of_the_parser(self):
        """Input: diffs that create a module, delete one, and touch a non-Python file. Expect:
        a created module gets its full text; deletion and non-module files raise ValueError.
        Proves: a diff is never half-applied. Not: binary diffs (git apply refuses them)."""
        new = "--- /dev/null\n+++ b/pkg/new_mod.py\n@@ -0,0 +1 @@\n+X = 1\n"
        self.assertEqual(diffs.apply(new, lambda m: None), {"pkg.new_mod": "X = 1\n"})
        gone = "--- a/pkg/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-X = 1\n"
        with self.assertRaises(ValueError):
            diffs.apply(gone, lambda m: "X = 1\n")
        yaml = "--- a/cfg.yaml\n+++ b/cfg.yaml\n@@ -1 +1 @@\n-a: 1\n+a: 2\n"
        with self.assertRaises(ValueError):
            diffs.apply(yaml, lambda m: None)
        self.assertEqual(diffs.module_of("pkg/sub/__init__.py"), "pkg.sub")


if __name__ == "__main__":
    unittest.main()
