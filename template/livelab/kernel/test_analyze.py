"""Golden tests of the swap's change analysis (`swap.analyze`, `swap.changed_targets`) on the
toy's source texts (CPU, standard library only). The analysis touches no live object, so
these tests import no toy module: they read its files.

Run from the repo root:  python3 -m unittest discover -s template/livelab/kernel -t template/livelab -v
(or python3 -m pytest -q template/livelab/kernel)
"""
import ast
import unittest

from . import locate as L
from . import swap as S
from . import witness as W
from .test_locate import OPS, at, edit

FILE = "ops.py"
SCALE, SHIFT = at(OPS, "def scale"), at(OPS, "def shift")
MAKE_SHIFT = at(OPS, "def make_shift")


def analyze(new, targets="auto", live=None):
    """analyze() of an edit of the toy's ops.py; by default every definition is live."""
    if live is None:
        live = {(q, line): True for q, line in [("scale", SCALE), ("make_shift", MAKE_SHIFT),
                                                ("make_shift.<locals>.shift", SHIFT)]}
    return S.analyze(OPS, new, targets, FILE, "toy.ops", live)


class ChangeAnalysisTest(unittest.TestCase):

    def test_a_changed_body_is_a_target_and_its_new_code_counts_its_calls(self):
        """Input: scale's body edited (x * STEP_SCALE -> x * STEP_SCALE * 5), and separately
        shift's (a closure made by make_shift). Expected: changed_targets gives [scale] and
        [make_shift, make_shift.<locals>.shift] (the enclosing function is listed too: it
        makes the closures). analyze("auto") resolves the same targets, changed, with no
        outside change; the compiled new module, run with the witness global bound, gives
        scale(3) = 30 and one call counted under "toy.ops:scale:<line>". Pain point: an
        analysis that misses the closure's maker (closures made later keep the old code), or
        compiles a body that never says it ran. Not: installing the code (`apply`, test_swap.py)."""
        new = edit(OPS, ("return x * STEP_SCALE", "return x * STEP_SCALE * 5"))
        self.assertEqual(S.changed_targets(OPS, new), [["scale", SCALE]])
        shift = edit(OPS, ("return x + k", "return x + 2 * k"))
        self.assertEqual(S.changed_targets(OPS, shift),
                         [["make_shift", MAKE_SHIFT], ["make_shift.<locals>.shift", SHIFT]])

        a = analyze(new)
        self.assertEqual((a["resolved"], [(t["status"], t["changed"]) for t in a["targets"]],
                          a["outside_changes"], a["added"], a["removed"], a["new_error"]),
                         ([["scale", SCALE]], [("ok", True)], False, [], [], None))
        w = W.Witness()
        ns = {"__name__": "toy_ops_copy", W.HIT: w.hook}
        exec(a["new_units"][0], ns)
        self.assertEqual(ns["scale"](3), 30)
        self.assertEqual(w.counts, {L.key_of("toy.ops", "scale", SCALE): 1})

    def test_comment_and_docstring_edits_are_no_target_and_a_constant_edit_is_outside(self):
        """Input: three edits of ops.py: a comment on `def scale`, a docstring added to
        scale, and STEP_SCALE = 2 -> 3. Expected: none has a target; the comment and the
        docstring have no outside change; the constant edit is `outside_changes` (True).
        Pain points: a docstring-only edit swapped and then reported as a false zero_witness
        (a live run), and a module-constant edit that a code swap silently does not apply.
        Not: the transaction-level `empty_change` verdict (the coordinator's)."""
        comment = edit(OPS, ("def scale(x):", "def scale(x):  # faster"))
        doc = edit(OPS, ("def scale(x):\n", 'def scale(x):\n    "Scale x."\n'))
        const = edit(OPS, ("STEP_SCALE = 2", "STEP_SCALE = 3"))
        got = [(S.changed_targets(OPS, src), analyze(src)["resolved"],
                analyze(src)["outside_changes"]) for src in (comment, doc, const)]
        self.assertEqual(got, [([], [], False), ([], [], False), ([], [], True)])

    def test_targets_that_do_not_resolve_are_reported_and_new_names_are_found(self):
        """Input: an edit that changes scale's body, adds `def double(x)`, imports `math`
        and deletes `make_shift`; explicit targets scale at the `def` line + 1 (no
        definition there), and scale with no live function. Expected: statuses
        no_definition and no_live_function; the auto analysis finds `double` as added, the
        `math` import, `make_shift` as removed, and no outside change (added, imported and
        removed names are not outside changes). Not: whether `stage` blocks on them (it
        turns these into findings, test_swap.py)."""
        start = OPS.index("def make_shift")
        new = edit(OPS[:start].rstrip() + "\n",
                   ("import functools", "import functools\nimport math"),
                   ("return x * STEP_SCALE", "return math.floor(x * STEP_SCALE)"))
        new += "\n\ndef double(x):\n    return 2 * x\n"
        self.assertEqual([t["status"] for t in analyze(new, [["scale", SCALE + 1]])["targets"]],
                         ["no_definition"])
        self.assertEqual([t["status"] for t in analyze(new, [["scale", SCALE]], {})["targets"]],
                         ["no_live_function"])
        a = analyze(new)
        self.assertEqual((a["resolved"], [x["q"] for x in a["added"]], a["removed"],
                          [i["names"] for i in a["imports"]], a["outside_changes"]),
                         ([["scale", SCALE]], ["double"], ["make_shift"],
                          [["functools"], ["math"]], False))

    def test_the_callers_trees_come_back_unchanged(self):
        """Input: a planner's own parse of the base and the edited text, passed as `trees`.
        Expected: the analysis gives the same result as from its own parse (targets, line
        map, outside changes, the new scale code's content hash), and both trees dump
        exactly as before, although the analysis inserted witness calls and blanked bodies
        in them. Pain point: a planner that reuses its trees for the next edit and plans
        against a tree that still holds the last edit's witness calls. Not: thread safety
        (one analysis per tree at a time)."""
        new = edit(OPS, ("return x + 1", "return x + 3"), ("return x + k", "return x - k"))
        trees = (ast.parse(OPS), ast.parse(new))
        before = [ast.dump(t, include_attributes=True) for t in trees]
        live = {(q, line): True for q, line in S.changed_targets(OPS, new)}
        mine = S.analyze(OPS, new, "auto", FILE, "toy.ops", live, trees=trees)
        own = analyze(new, live=live)
        self.assertEqual([t["status"] for t in mine["targets"]], ["ok"] * 3)
        self.assertEqual([ast.dump(t, include_attributes=True) for t in trees], before)

        def summary(a):
            codes = {}
            for c in a["new_units"]:
                L.code_tree(c, None, (codes, {}))
            return (a["resolved"], a["line_map"], a["outside_changes"],
                    [L.content_hash(codes[tuple(t)][0]) for t in a["resolved"]])
        self.assertEqual(summary(mine), summary(own))
        self.assertEqual(mine["resolved"], [["activate", at(OPS, "@traced")],
                                            ["make_shift", MAKE_SHIFT],
                                            ["make_shift.<locals>.shift", SHIFT]])


class TargetEdgeCaseTest(unittest.TestCase):
    """Edits at the edge of a target: a signature annotation, a nested function's docstring."""

    def test_a_signature_annotation_edit_is_not_outside_the_target(self):
        """Input: scale gains a return annotation (`-> int`), or type parameters
        (`def scale[T](x: T)`), with a body edit. Expected: scale is the one target and
        outside_changes is False, as for a parameter annotation (a code swap installs
        neither). Pain point: the edit was refused as outside_changes. Not: annotations on
        the live function (they stay the old ones)."""
        for sig in ("def scale(x) -> int:", "def scale[T](x: T):"):
            a = analyze(edit(OPS, ("def scale(x):", sig), ("x * STEP_SCALE", "x * 3")))
            self.assertEqual((a["resolved"], a["outside_changes"]),
                             ([["scale", SCALE]], False), sig)

    def test_a_docstring_edit_of_a_nested_function_targets_nothing(self):
        """Input: the nested `shift` in make_shift gains a docstring. Expected: no target
        (not make_shift either) and no outside_changes. Pain point: make_shift, which runs
        only at engine init, was swapped and then reported a false zero_witness. Not: a
        docstring that the code reads (`__doc__`)."""
        new = edit(OPS, ("    def shift(x):\n", '    def shift(x):\n        """Add k."""\n'))
        self.assertEqual(S.changed_targets(OPS, new), [])
        self.assertFalse(analyze(new)["outside_changes"])


if __name__ == "__main__":
    unittest.main()
