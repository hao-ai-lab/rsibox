"""Golden tests of the in-place rebuild (T1 restart tier) on real torch modules (CPU).

Run from the repo root:
    python3 -m unittest discover -s repos/fastvideo/livelab/services -t repos/fastvideo/livelab -v

Each case writes a tiny package to a temp dir, imports it, builds a module with real weights,
then refreshes an edited copy of the source in the same process, as a warm worker would.
"""
import importlib
import os
import sys
import tempfile
import unittest

import torch

from services import rebuild as RB

BLOCK = '''import torch
from torch import nn

EPS = 1e-6


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = EPS

    def forward(self, x):
        return x * self.weight / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = Norm(dim)
        self.proj = nn.Linear(dim, dim)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.proj(self.norm(x))
'''
USER = "from rbpkg.block import EPS\n\n\ndef eps():\n    return EPS\n"


def edit(src, old, new):
    assert src.count(old) == 1, old
    return src.replace(old, new)


class RebuildTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rbtest-")
        os.makedirs(os.path.join(self.dir, "rbpkg"))
        for name, text in (("__init__.py", ""), ("block.py", BLOCK), ("user.py", USER)):
            with open(os.path.join(self.dir, "rbpkg", name), "w") as f:
                f.write(text)
        sys.path.insert(0, self.dir)
        importlib.invalidate_caches()
        import rbpkg.block
        import rbpkg.user  # noqa: F401  (an importer of block: refreshed with it)
        torch.manual_seed(0)
        self.live = rbpkg.block.Block(8)
        self.live(torch.randn(2, 8))                     # state set after construction
        self.x = torch.randn(2, 8)

    def tearDown(self):
        for n in [n for n in sys.modules if n.split(".")[0] == "rbpkg"]:
            del sys.modules[n]
        sys.path.remove(self.dir)

    def rebuild(self, new_src):
        old_cls = type(self.live)
        r = RB.refresh({"rbpkg.block": new_src}, prefixes=("rbpkg",))
        self.assertTrue(r["ok"], r["findings"])
        cmap = RB.class_map(r["old"])
        with torch.device("meta"):
            base, new = old_cls(8), cmap[old_cls](8)
        return r, RB.merge_fields(self.live, base, new)

    def test_constructor_edit_keeps_weights_and_live_state(self):
        """Input: Norm's constructor now sets eps = 1e-2 (a constructor edit R0 refuses).
        Expect: the live Norm gets eps 1e-2 and the new class; the weights are the same tensor
        objects (no load); `calls` (set after construction, ctor unchanged) stays 1; the output
        equals a fresh eps=1e-2 module with the same weights. Proves: the 3-way merge applies
        the edit and nothing else. Not: an edit that adds weights (next case)."""
        w = self.live.proj.weight
        before = self.live(self.x)
        r, m = self.rebuild(edit(BLOCK, "self.eps = EPS", "self.eps = 1e-2"))
        self.assertTrue(m["ok"], m["findings"])
        self.assertIn("norm.eps", m["set"])
        self.assertEqual(self.live.norm.eps, 1e-2)
        self.assertIs(self.live.proj.weight, w)
        self.assertEqual(self.live.calls, 2)              # 1 before + the call above, kept
        self.assertIs(type(self.live), sys.modules["rbpkg.block"].Block)
        after = self.live(self.x)
        self.assertFalse(torch.equal(before, after))
        n = self.live.norm
        ref = self.x * n.weight / (self.x.pow(2).mean(-1, keepdim=True) + 1e-2).sqrt()
        self.assertTrue(torch.equal(after, self.live.proj(ref)))

    def test_module_constant_reaches_importers(self):
        """Input: EPS = 1e-3 at module level. Expect: rbpkg.user (which did `from block import
        EPS`) is refreshed too and returns 1e-3; the live Norm's eps changes (its constructor
        value changed). Proves: importers are re-imported in order."""
        r, m = self.rebuild(edit(BLOCK, "EPS = 1e-6", "EPS = 1e-3"))
        self.assertIn("rbpkg.user", r["refreshed"])
        self.assertEqual(sys.modules["rbpkg.user"].eps(), 1e-3)
        self.assertEqual(self.live.norm.eps, 1e-3)

    def test_new_weights_are_a_finding_and_change_nothing(self):
        """Input: Block's constructor adds a Linear `gate` (new weights). Expect: finding
        `new_submodule`, no attribute set, the live class unchanged. Proves: the merge never
        invents tensors from the meta build (they need a load)."""
        src = edit(BLOCK, "        self.calls = 0\n", "        self.calls = 0\n        self.gate = nn.Linear(dim, dim)\n")
        cls = type(self.live)
        r, m = self.rebuild(src)
        self.assertFalse(m["ok"])
        self.assertEqual([f["check"] for f in m["findings"]], ["new_submodule"])
        self.assertIs(type(self.live), cls)
        self.assertFalse(hasattr(self.live, "gate"))

    def test_failed_import_puts_the_old_modules_back(self):
        """Input: an edit whose import raises. Expect: finding `refresh_failed`, and
        sys.modules holds the very same module objects as before. Proves: a refused refresh
        leaves the process as it was."""
        old = sys.modules["rbpkg.block"], sys.modules["rbpkg.user"]
        r = RB.refresh({"rbpkg.block": BLOCK + "\nraise RuntimeError('boom')\n"}, prefixes=("rbpkg",))
        self.assertFalse(r["ok"])
        self.assertEqual(r["findings"][0]["check"], "refresh_failed")
        self.assertIs(sys.modules["rbpkg.block"], old[0])
        self.assertIs(sys.modules["rbpkg.user"], old[1])


if __name__ == "__main__":
    unittest.main()
