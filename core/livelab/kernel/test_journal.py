"""Golden tests of the journal on the toy engine (CPU, standard library only; one test
imports torch on CPU and is skipped without it).

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
(or python3 -m pytest -q core/livelab/kernel)

Toy arithmetic and helpers: see test_locate.py (step(3) = 17 at base).
"""
import gc
import importlib
import unittest

from . import journal as J
from . import locate as L
from .test_locate import ENG, OPS, at, edit, fresh


def compiled(module, source, qualname, line):
    """The code object of `qualname` at `line` in `source` compiled as `module`'s file: what a
    swap would install. (The swap's own staging is tested in test_swap.py.)"""
    return L.code_tree(compile(source, module.__file__, "exec"))[0][(qualname, line)][0]


class JournalCodeTest(unittest.TestCase):
    """Pain point: an undo that leaves one holder on the new code (a `from x import f` copy, a
    saved callback, a decorator's wrapped original) while the module looks restored."""

    def test_set_code_reaches_every_holder_and_undo_restores_exact_base(self):
        """Proves: `set_code` on the located function objects changes what every holder runs (a from-import copy, the decorated original under its wrapper, a bound-method callback), and `undo_to(mark)` puts back the same code objects (identity and content hash) so the engine computes its base values again. Not: swap staging, closures made at run time, data the new code wrote (the token it appended stays)."""
        name, eng, ops = fresh("journal")
        e = eng.Engine()
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        self.assertEqual(e.step(3), 17)
        ops_new = edit(OPS, ("return x * STEP_SCALE", "return x * 3"),
                       ("return x + 1", "return x + 2"))
        eng_new = edit(ENG, ("self.tokens.append(t)", "self.tokens.append(-t)"))
        targets = [(ops, ops_new, "scale", at(OPS, "def scale")),
                   (ops, ops_new, "activate", at(OPS, "@traced")),
                   (eng, eng_new, "Engine.on_token", at(ENG, "def on_token"))]
        j = J.Journal()
        mark = j.mark("txn-1")
        base = []
        for mod, src, q, line in targets:
            loc = L.locate(mod.__name__, q, line)
            (fn,) = loc["functions"]
            base.append((fn, fn.__code__, loc["codes"][0]["hash"]))
            j.set_code(fn, compiled(mod, src, q, line))
        self.assertEqual([x["kind"] for x in j.describe()], ["code", "code", "code"])

        self.assertEqual(e.step(3), 21)                  # 3*3=9, +2=11, +10=21
        self.assertEqual(e.tokens, [17, -21])            # the saved callback runs the new code
        self.assertEqual(eng.scale(3), 9)                # the from-import copy
        self.assertEqual((ops.activate(3), ops.activate.calls), (5, 3))   # wrapper kept

        u = j.undo_to(mark)
        self.assertEqual((u["ok"], u["undone"], j.entries), (True, 3, []))
        for fn, code, h in base:
            self.assertIs(fn.__code__, code)
            self.assertEqual(L.content_hash(fn.__code__), h)
        self.assertEqual((e.step(3), e.tokens), (17, [17, -21, 17]))
        self.assertEqual((eng.scale(3), ops.activate(3)), (6, 4))


class RecordAttrTest(unittest.TestCase):
    """Pain point: `record_attr` read and restored only `__dict__` entries, so a repair that
    set a function's `__doc__`, a descriptor (torch `Parameter.data`) or an `nn.Module`
    submodule (kept in `_modules` by its `__setattr__`) was journaled as "added" and undo left
    the new value in place."""

    class Prop:
        def __init__(self):
            self._v = "old"

        @property
        def val(self):
            return self._v

        @val.setter
        def val(self, v):
            self._v = v

    class Side:
        """Stores attributes in a side dict through `__setattr__`, like nn.Module."""

        def __init__(self):
            object.__setattr__(self, "_side", {"sub": "old sub"})

        def __getattr__(self, n):
            try:
                return self.__dict__["_side"][n]
            except KeyError:
                raise AttributeError(n) from None

        def __setattr__(self, n, v):
            self._side[n] = v

        def __delattr__(self, n):
            del self._side[n]

    def test_undo_restores_descriptor_side_dict_and_doc(self):
        """Proves: a property, a name kept by a custom `__setattr__` (replaced and added), a function's `__doc__`, and a plain `__dict__` entry (replaced and added) all come back exactly after `record_attr` + assign + undo. Not: torch (next test)."""
        def fn():
            """old doc"""
        p, sd = self.Prop(), self.Side()

        class Plain:
            pass
        pl = Plain()
        pl.x = old_x = ["x"]
        j = J.Journal()
        for obj, name, value in [(p, "val", "new"), (sd, "sub", "new sub"),
                                 (sd, "extra", "added"), (fn, "__doc__", "new doc"),
                                 (pl, "x", ["new"]), (pl, "y", "added")]:
            j.record_attr(obj, name)
            setattr(obj, name, value)
        self.assertEqual((p.val, sd.sub, sd.extra, fn.__doc__, pl.y),
                         ("new", "new sub", "added", "new doc", "added"))
        u = j.undo_to(0)
        self.assertTrue(u["ok"], u)
        self.assertEqual((p.val, sd.sub, fn.__doc__), ("old", "old sub", "old doc"))
        self.assertEqual(sd._side, {"sub": "old sub"})        # the added name is gone
        self.assertIs(pl.x, old_x)
        self.assertNotIn("y", vars(pl))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_undo_restores_parameter_data_and_module_submodules(self):
        """Proves (CPU torch): `Parameter.data` swapped to another tensor comes back to the old storage and values, a replaced `nn.Module` submodule comes back as the same object, an added one is removed from `_modules`. Not: CUDA tensors, buffers."""
        import torch
        from torch import nn
        m = nn.Module()
        m.lin = old_lin = nn.Linear(2, 2)
        w = m.lin.weight
        ptr, old_vals = w.data_ptr(), w.detach().clone()
        j = J.Journal()
        j.record_attr(w, "data")
        w.data = torch.full((2, 2), 7.0)
        j.record_attr(m, "lin")
        m.lin = nn.Linear(2, 2)
        j.record_attr(m, "extra")
        m.extra = nn.ReLU()
        self.assertTrue(j.undo_to(0)["ok"])
        self.assertEqual(w.data_ptr(), ptr)
        self.assertTrue(torch.equal(w.detach(), old_vals))
        self.assertIs(m.lin, old_lin)
        self.assertEqual(list(m._modules), ["lin"])


class UndoCallTest(unittest.TestCase):
    """Pain point: an in-place re-init (a Triton kernel rebuilt) had no undo entry, so undo
    needed a hand-run `repair()`."""

    def test_record_call_runs_in_reverse_order_and_a_failure_is_a_finding(self):
        """Proves: `record_call` entries run at their place in reverse order with the attribute entries (the first call sees the later attribute already undone), and one that raises is an `undo_incomplete` finding naming its label while the rest still run. Not: calls that take arguments (bind them in a closure)."""
        class O:
            pass
        o, log = O(), []
        j = J.Journal()
        j.set_attr(o, "a", 1)
        j.record_call(lambda: log.append(("k1", vars(o).get("b"))), "rebuild k1")

        def boom():
            raise RuntimeError("kernel cache gone")
        j.set_attr(o, "b", 2)
        j.record_call(boom, "rebuild k2")
        j.record_call(lambda: log.append(("k3", vars(o).get("b"))), "rebuild k3")
        u = j.undo_to(0)
        self.assertEqual(log, [("k3", 2), ("k1", None)])
        self.assertEqual(vars(o), {})
        self.assertFalse(u["ok"])
        self.assertEqual([f["check"] for f in u["findings"]], ["undo_incomplete"])
        self.assertIn("rebuild k2", u["findings"][0]["observed"])
        self.assertIn("kernel cache gone", u["findings"][0]["observed"])
        self.assertTrue(j.poisoned)
        self.assertEqual(len(j.entries), 1)
        self.assertEqual(j.entries[0].name, "rebuild k2")

    def test_on_undo_keyed_to_mark_epoch_and_marks_truncated(self):
        """Input: txn1 with 0 entries + on_undo(cb1), then txn2 at the same position + entry +
        on_undo(cb2); undo_to(m2), then undo_to(m1). Expected: undo_to(m2) runs ONLY cb2, slices
        txn2 out of journal.marks while keeping txn1, and cb1 fires only on undo_to(m1)."""
        d, events = {}, []
        j = J.Journal()
        m1 = j.mark("txn1")
        j.on_undo(lambda: events.append("cb1"))
        m2 = j.mark("txn2")
        j.set_item(d, "x", 1)
        j.on_undo(lambda: events.append("cb2"))
        self.assertEqual(m1, m2)   # both at entry offset 0

        u2 = j.undo_to(m2)
        self.assertTrue(u2["ok"], u2)
        self.assertEqual(events, ["cb2"])
        self.assertEqual(d, {})
        self.assertEqual(j.marks, [("txn1", m1)])

        u1 = j.undo_to(m1)
        self.assertTrue(u1["ok"], u1)
        self.assertEqual(events, ["cb2", "cb1"])
        self.assertEqual(j.marks, [])

    def test_failing_invert_preserves_entry_poisons_and_supports_retry(self):
        """Input: txn1 (clean), txn2 (one normal entry + one record_call whose inverse fails on
        first attempt). Expected: first undo_to(m2) undoes the normal entry, preserves the failing
        entry in `journal.entries`, sets `journal.poisoned=True`, and reports `undo_incomplete`;
        retrying undo_to(m2) once the inverse stops raising completes cleanly and truncates m2."""
        d, state = {}, {"fail": True, "calls": 0}

        def flaky():
            state["calls"] += 1
            if state["fail"]:
                raise RuntimeError("transient invert failure")

        j = J.Journal()
        m1 = j.mark("t1")
        j.set_item(d, "base", 10)
        m2 = j.mark("t2")
        j.set_item(d, "tmp", 20)
        j.record_call(flaky, "flaky_op")

        u_fail = j.undo_to(m2)
        self.assertFalse(u_fail["ok"])
        self.assertEqual([f["check"] for f in u_fail["findings"]], ["undo_incomplete"])
        self.assertTrue(j.poisoned)
        self.assertEqual(d, {"base": 10})
        self.assertEqual([e.name for e in j.entries], ["base", "flaky_op"])

        state["fail"] = False
        u_retry = j.undo_to(m2)
        self.assertTrue(u_retry["ok"], u_retry)
        self.assertEqual(u_retry["undone"], 1)
        self.assertEqual([e.name for e in j.entries], ["base"])
        self.assertEqual(j.marks, [("t1", m1)])


class ClosureUndoTest(unittest.TestCase):
    """S3: live closures created by swapped code are restored even when the outer new code drops."""

    def test_undo_restores_closure_when_outer_code_drops_and_cancels_refcount(self):
        """Input: swap `outer` -> `inner` without retaining `staged`, call `c1 = mod.outer()` so
        `c1` holds `inner_new` (+1) while `Entry('code')` drops `outer_new` (-1), then `undo_to`.
        Expected: `_restore_closures` still scans `gc.get_referrers` and restores `c1.__code__`
        so `c1()` returns 1 instead of staying on 2."""
        from . import swap as S
        from . import witness as W
        from .test_detectors import temp_modules

        (mod,) = temp_modules(
            self,
            {"kt_closure_undo_s3": "def outer():\n    def inner():\n        return 1\n    return inner\n"},
        )
        j, w = J.Journal(), W.Witness()
        mark = j.mark("t1")
        res = S.apply(
            S.stage("kt_closure_undo_s3", "def outer():\n    def inner():\n        return 2\n    return inner\n")[
                "staged"
            ],
            j,
            w,
        )
        self.assertTrue(res["ok"], res)
        c1 = mod.outer()
        self.assertEqual(c1(), 2)
        u = j.undo_to(mark)
        self.assertTrue(u["ok"], u)
        self.assertGreaterEqual(u.get("closures_restored", 0), 1)
        self.assertEqual(c1(), 1)
        self.assertEqual(mod.outer()(), 1)
