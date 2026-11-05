"""A second transaction on a module whose first swap changed its line count (CPU, stdlib only).

Run from template/livelab:  python3 -m unittest kernel.test_line_shift -v

Case: swap `scale` from text v1 (one line longer), then edit `make_shift` (below it) from text v2
with base_source = v1. The live `make_shift` still carries the original file's line numbers, so
(qualname, v1 line) matches no running code. Proves: the kernel finds it by its unique qualname,
reports the alias in timings["line_aliases"], swaps it, and the new code runs. Not: qualnames
defined twice (overloads), which still report no_live_function.
"""
import unittest

from . import coordinator as C
from . import findings as F
from . import journal as J
from . import swap as S
from . import witness as W
from .test_kernel import at, edit, fresh, OPS


class LineShiftedSecondEdit(unittest.TestCase):
    def _run(self, explicit):
        _, _, ops = fresh("lineshift")
        j, w = J.Journal(), W.Witness()
        v1 = edit(OPS, ("    return x * STEP_SCALE", "    y = x * STEP_SCALE\n    return y"))
        r1 = S.stage(ops.__name__, v1, [["scale", at(OPS, "def scale")]])
        self.assertTrue(r1["ok"] and F.decide(r1["findings"], ())["proceed"], r1["findings"])
        S.apply(r1["staged"], j, w)
        self.assertEqual(ops.scale(3), 6)
        v2 = edit(v1, ("    bias = 0", "    bias = 5"))
        targets = [["make_shift", at(v1, "def make_shift")]] if explicit else "auto"
        own = [ops.scale.__code__]
        r2 = S.stage(ops.__name__, v2, targets, base_source=v1, own_codes=own)
        checks = [f["check"] for f in r2["findings"]]
        self.assertNotIn("no_live_function", checks)
        self.assertNotIn("base_mismatch", checks)
        self.assertTrue(r2["ok"] and F.decide(r2["findings"], ())["proceed"], r2["findings"])
        self.assertEqual(r2["timings"].get("line_aliases"),
                         [["make_shift", at(v1, "def make_shift"), at(OPS, "def make_shift")]])
        S.apply(r2["staged"], j, w)
        self.assertIn(5, ops.make_shift.__code__.co_consts)
        self.assertEqual(ops.make_shift(10)(1), 11)

    def test_explicit_targets(self):
        self._run(True)

    def test_auto_targets(self):
        self._run(False)


class LineMapChain(unittest.TestCase):
    """The exact fix: the participant composes its committed swaps' line maps, so a later
    target keyed at the installed text's line finds the live code with no qualname guess."""

    def _chain(self, targets2):
        name, _, ops = fresh("linechain")
        p = C.LocalParticipant("core", alias={"toy": name}, analyzer=None)
        v1 = edit(OPS, ("    return x * STEP_SCALE",
                        "    y = x * STEP_SCALE\n    y = y + 0\n    return y"))
        r1 = p.prepare({"id": "e1", "modules": {"toy.ops": {"source": v1}}})
        self.assertTrue(r1["ok"], r1["findings"])
        self.assertTrue(p.commit("e1")["ok"])
        v2 = edit(v1, ("    bias = 0", "    bias = 5"))
        line_v1 = at(v1, "def make_shift")
        spec = {"source": v2}
        if targets2 == "explicit":
            spec["targets"] = [["make_shift", line_v1]]
        r2 = p.prepare({"id": "e2", "modules": {"toy.ops": spec}})
        checks = [f["check"] for f in r2["findings"]]
        self.assertTrue(r2["ok"], r2["findings"])
        self.assertNotIn("line_aliased", checks)   # exact: no qualname fallback
        (t,) = r2["timings"].values()
        self.assertIn(["make_shift", line_v1, at(OPS, "def make_shift")],
                      t["line_map_aliases"])
        self.assertEqual(list(r2["staged"]), [f"toy.ops:make_shift:{line_v1}"])
        self.assertTrue(p.commit("e2")["ok"])
        self.assertIn(5, ops.make_shift.__code__.co_consts)
        self.assertEqual((ops.scale(3), ops.make_shift(10)(1)), (6, 11))
        self.assertTrue(p.undo("e2")["ok"])                 # make_shift back, scale still v1
        self.assertNotIn(5, ops.make_shift.__code__.co_consts)
        self.assertIn("y = y + 0", p._live_source("toy.ops"))
        self.assertTrue(p.undo("e1")["ok"])                 # back at the file
        self.assertIsNone(p._live_source("toy.ops"))
        self.assertEqual((ops.scale(3), ops.make_shift(10)(1)), (6, 11))
        self.assertEqual(ops.scale.__code__.co_firstlineno, at(OPS, "def scale"))

    def test_three_edit_chain_explicit(self):
        """Input: edit 1 adds 2 lines in `scale` (above), edit 2 changes `bias` in
        `make_shift` (below) with the target at the installed text's line, then undo edit 2,
        then undo edit 1. Expected: edit 2 stages through the line map (no line_aliased),
        runs the new code, and each undo restores its base. Pain point: edit 2 was refused
        with no_live_function + outside_changes (live run, a scheduler module). Not: a
        qualname defined twice in the module (not covered here)."""
        self._chain("explicit")

    def test_three_edit_chain_auto(self):
        """The same chain with edit 2's targets left to the kernel ("auto")."""
        self._chain("auto")


if __name__ == "__main__":
    unittest.main()
