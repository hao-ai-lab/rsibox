"""Golden tests of the swap's detectors: fields only the new __init__ sets, and line maps of
a second edit after a line shift (CPU, standard library only).

Run from the repo root:  python3 -m unittest discover -s repos/vllm/livelab/kernel -t repos/vllm/livelab -v
(or python3 -m pytest -q repos/vllm/livelab/kernel)
"""
import gc
import importlib
import itertools
import os
import shutil
import sys
import tempfile
import unittest

from . import findings as F
from . import journal as J
from . import swap as S
from . import witness as W
from .test_locate import OPS, at, edit, fresh

_n = itertools.count()


def temp_modules(test, files):
    """Write {name: text} into a new directory on sys.path and import them in order."""
    tmp = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, tmp, True)
    for name, text in files.items():
        with open(os.path.join(tmp, name + ".py"), "w") as f:
            f.write(text)
    sys.path.insert(0, tmp)
    test.addCleanup(sys.path.remove, tmp)
    for name in files:
        test.addCleanup(sys.modules.pop, name, None)
    return [importlib.import_module(n) for n in files]


ESTIMATOR = '''class Estimator:
    def __init__(self, max_num_reqs, steps):
        self.steps = steps
        self.seen = 0

    def step(self, n):
        self.seen += n
        return min(n, 256)


class Speculator:
    def __init__(self, max_num_reqs):
        self.max_num_reqs = max_num_reqs
        self.acceptance_estimator = Estimator(max_num_reqs, 3)


class Runner:
    def __init__(self):
        self.speculator = Speculator(1000)
'''

ESTIMATOR_NEW = ESTIMATOR.replace(
    "        self.seen = 0\n",
    "        self.seen = 0\n        block = max(256, max_num_reqs * 2)\n"
    "        self._block = block\n").replace("return min(n, 256)", "return min(n, self._block)")


class NewFieldsTest(unittest.TestCase):

    def test_a_field_only_the_new_init_sets_blocks_and_routes_to_t1(self):
        """Input (vLLM #63819's shape): Estimator.__init__ gains `self._block = max(256,
        max_num_reqs * 2)` and step() reads self._block; the live estimator was built by the
        old __init__ (gc heap frozen, as in vLLM). Expected: stage gives one new_fields
        finding (subject `<module>:Estimator._block`, route T1, no backfill action), so
        stage_and_apply applies nothing and the live estimator still runs the old step
        (step(5000) = 256). Fails without the detector: the swap applies with no finding and
        the next step raises AttributeError (a live run killed the engine this way). Not:
        setting the field on live objects (a backfill is a later PR)."""
        name = f"kt_est{next(_n)}"
        (m,) = temp_modules(self, {name: ESTIMATOR})
        est = m.Runner().speculator.acceptance_estimator
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        r = S.stage(name, ESTIMATOR_NEW)
        (f,) = r["findings"]
        self.assertEqual((f["check"], f["subject"], f["route"], f["overridable"]),
                         ("new_fields", f"{name}:Estimator._block", "T1", True))
        self.assertIn("Estimator.step read self._block", f["observed"])
        self.assertFalse(any("backfill" in a["action"] for a in f["actions"]))
        j, w = J.Journal(), W.Witness()
        r = S.stage_and_apply(name, ESTIMATOR_NEW, j, w)
        self.assertEqual((r["ok"], [(x["check"], x["accepted"]) for x in r["findings"]]),
                         (False, [("new_fields", False)]))
        self.assertEqual((est.step(5000), j.entries), (256, []))

    def test_a_class_default_or_an_old_setter_is_not_a_new_field(self):
        """Input: the same edit when the base class has `_block = 256` in its body, and when
        an old method already sets self._block. Expected: no new_fields finding (old
        instances have the attribute). Not: fields set by code outside the class."""
        for base in (ESTIMATOR.replace("class Estimator:\n", "class Estimator:\n    _block = 256\n\n"),
                     ESTIMATOR.replace("        self.seen += n\n",
                                       "        self.seen += n\n        self._block = 256\n")):
            name = f"kt_est{next(_n)}"
            (m,) = temp_modules(self, {name: base})
            new = base.replace("        self.seen = 0\n", "        self.seen = 0\n"
                               "        self._block = max(256, max_num_reqs)\n").replace(
                "return min(n, 256)", "return min(n, self._block)")
            r = S.stage(name, new)
            self.assertEqual([f["check"] for f in r["findings"]], [])


class LineShiftTest(unittest.TestCase):
    """Case: swap `scale` from text v1 (lines longer), then edit `make_shift` (below it) from
    text v2 with base_source = v1. The live `make_shift` still carries the original file's
    line numbers, so (qualname, v1 line) matches no running code."""

    def _first_edit(self, extra):
        _, _, ops = fresh("lineshift")
        j, w = J.Journal(), W.Witness()
        v1 = edit(OPS, ("    return x * STEP_SCALE", "    y = x * STEP_SCALE\n" + extra
                        + "    return y"))
        r1 = S.stage(ops.__name__, v1, [["scale", at(OPS, "def scale")]])
        self.assertTrue(r1["ok"] and F.decide(r1["findings"], ())["proceed"], r1["findings"])
        self.assertTrue(S.apply(r1["staged"], j, w)["ok"])
        v2 = edit(v1, ("    bias = 0", "    bias = 5"))
        return ops, j, w, v1, v2, r1["staged"]

    def test_edit_again_after_a_line_shift_follows_the_line_map(self):
        """Input: edit 1 adds 2 lines in `scale` (above make_shift); edit 2 changes `bias` in
        make_shift, staged with base_source = edit 1's text, edit 1's line map, and targets
        explicit (make_shift at its line in edit 1's text) or auto. Expected: no finding (no
        qualname guess: no line_aliased), timings["line_map_aliases"] holds the step from
        edit 1's line of make_shift to its live line, the swap applies and make_shift(10)(1) = 11 with the
        new constant 5 in its code. Pain point: edit 2 was refused with no_live_function +
        outside_changes (a live run on a scheduler module). Not: a qualname defined twice."""
        for targets in ("explicit", "auto"):
            ops, j, w, v1, v2, st1 = self._first_edit("    y = y + 0\n")
            line_v1 = at(v1, "def make_shift")
            t = [["make_shift", line_v1]] if targets == "explicit" else "auto"
            r2 = S.stage(ops.__name__, v2, t, base_source=v1, own_codes=[ops.scale.__code__],
                         line_maps=[st1.line_map])
            self.assertEqual((r2["ok"], r2["findings"], r2["targets"]),
                             (True, [], [["make_shift", line_v1]]), targets)
            self.assertIn(["make_shift", line_v1, at(OPS, "def make_shift")],
                          r2["timings"]["line_map_aliases"])
            self.assertTrue(S.apply(r2["staged"], j, w)["ok"])
            self.assertIn(5, ops.make_shift.__code__.co_consts)
            self.assertEqual((ops.scale(3), ops.make_shift(10)(1)), (6, 11))

    def test_without_a_line_map_the_one_live_qualname_is_an_advisory_alias(self):
        """Input: the same two edits (edit 1 adds 1 line), edit 2 staged with no line map.
        Expected: the one live make_shift is taken by its qualname, with an advisory
        line_aliased finding naming base line and live line; the swap applies and runs.
        Not: qualnames defined twice (overloads), which still report no_live_function."""
        ops, j, w, v1, v2, _ = self._first_edit("")
        line_v1, line_live = at(v1, "def make_shift"), at(OPS, "def make_shift")
        r2 = S.stage(ops.__name__, v2, [["make_shift", line_v1]], base_source=v1,
                     own_codes=[ops.scale.__code__])
        self.assertEqual([(f["check"], f.get("advisory")) for f in r2["findings"]],
                         [("line_aliased", True)])
        self.assertTrue(F.decide(r2["findings"], ())["proceed"])
        self.assertEqual(r2["timings"]["line_aliases"], [["make_shift", line_v1, line_live]])
        self.assertTrue(S.apply(r2["staged"], j, w)["ok"])
        self.assertEqual(ops.make_shift(10)(1), 11)


if __name__ == "__main__":
    unittest.main()
