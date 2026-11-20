"""Golden tests of the swap (R0: a function-body swap) on the toy engine (CPU, standard
library only).

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
(or python3 -m pytest -q core/livelab/kernel)

Toy arithmetic: see test_locate.py (step(3) = 17 at base).
"""
import gc
import itertools
import threading
import unittest

from . import findings as F
from . import journal as J
from . import locate as L
from . import swap as S
from . import witness as W
from .test_detectors import temp_modules
from .test_locate import ENG, OPS, at, edit, fresh
from .test_witness import wait_for

_n = itertools.count()


def install(module, source, targets, journal, witness, accept=()):
    """stage -> decide -> apply, as a participant does."""
    r = S.stage(module, source, targets)
    d = F.decide(r["findings"], accept)
    if not (r["ok"] and d["proceed"]):
        return r, d, None
    return r, d, S.apply(r["staged"], journal, witness)


class SwapTest(unittest.TestCase):

    def test_swap_reaches_every_holder(self):
        """Proves: one swap reaches from-import copies, a decorator wrapper, a saved bound method and closures. Not: copied code objects."""
        name, eng, ops = fresh("holders")
        e = eng.Engine()
        gc.freeze()                       # engines freeze the heap after startup (vLLM does)
        self.addCleanup(gc.unfreeze)
        self.assertEqual(e.step(3), 17)
        holders = {q: {(h["kind"], h["where"]) for h in L.locate(m, q, line)["holders"]}
                   for m, q, line in [(name + ".ops", "scale", at(OPS, "def scale")),
                                      (name + ".ops", "activate", at(OPS, "@traced")),
                                      (name + ".engine", "Engine.on_token",
                                       at(ENG, "def on_token"))]}
        self.assertEqual(holders["scale"], {("module_dict", f"{name}.ops.scale"),
                                            ("module_dict", f"{name}.engine.scale")})
        self.assertIn(("closure_cell", "traced.<locals>.wrapper"), holders["activate"])
        self.assertIn(("bound_method", "Engine().on_token"), holders["Engine.on_token"])

        j, w = J.Journal(), W.Witness()
        ops_new = edit(OPS, ("return x * STEP_SCALE", "return x * 3"),
                       ("return x + 1", "return x + 2"), ("return x + k", "return x + 2 * k"))
        r, d, a = install(name + ".ops", ops_new,
                          [("scale", at(OPS, "def scale")), ("activate", at(OPS, "@traced")),
                           ("make_shift.<locals>.shift", at(OPS, "def shift"))], j, w)
        self.assertEqual((r["findings"], a["ok"]), ([], True))
        r, d, a = install(name + ".engine", edit(ENG, ("self.tokens.append(t)",
                                                       "self.tokens.append(-t)")),
                          [("Engine.on_token", at(ENG, "def on_token"))], j, w)
        self.assertEqual((r["findings"], a["ok"]), ([], True))

        self.assertEqual(e.step(3), 31)                  # 3*3=9, +2=11, +2*10=31
        self.assertEqual(e.tokens, [17, -31])            # callback runs the new on_token
        self.assertEqual(eng.scale(3), 9)                # the from-import copy
        self.assertEqual((ops.activate(3), ops.activate.calls), (5, 3))   # wrapper kept
        self.assertEqual(e.shift(0), 0 + 2 * 10)         # closure made before the swap
        self.assertEqual(eng.make_shift(1)(0), 2)        # closure made after the swap
        self.assertEqual(w.counts[f"{name}.ops:scale:{at(OPS, 'def scale')}"], 2)

    def test_undo_restores_exact_base(self):
        """Proves: undo_to(mark) restores the same code objects and removes added methods, imports and witness keys; the witness global stays, bound to the no-op stub (compiled code may still call it). Not: data the new code wrote."""
        name, eng, _ = fresh("undo")
        e = eng.Engine()
        base_code, base_hash = eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__)
        mod_keys, cls_keys = set(vars(eng)), set(vars(eng.Engine))
        new = edit(ENG, ("import threading", "import statistics\nimport threading"),
                   ("self.steps += 1", "self.steps += 2"),
                   ("    def drain(self):", "    def mean_token(self):\n"
                    "        return statistics.mean(self.tokens)\n\n    def drain(self):"))
        j, w = J.Journal(), W.Witness()
        mark = j.mark("txn-1")
        r, d, a = install(name + ".engine", new, [("Engine.step", at(ENG, "def step"))], j, w)
        self.assertEqual((r["findings"], a["ok"]), ([], True))
        e.step(1)
        e.step(3)
        self.assertEqual((e.steps, e.mean_token()), (4, 15))         # (13 + 17) / 2
        self.assertEqual(sorted(set(vars(eng)) - mod_keys), ["__fastrsi_hit__", "statistics"])
        self.assertEqual(sorted(set(vars(eng.Engine)) - cls_keys), ["mean_token"])
        # the import, the witness global, the counter keys of step and mean_token, step's
        # code, the added method
        self.assertEqual([x["kind"] for x in j.describe()],
                         ["added", "object", "added", "added", "code", "added"])

        u = j.undo_to(mark)
        self.assertEqual((u["ok"], u["undone"], j.entries), (True, 6, []))
        self.assertIs(eng.Engine.step.__code__, base_code)
        self.assertEqual(L.content_hash(eng.Engine.step.__code__), base_hash)
        self.assertEqual((set(vars(eng)), set(vars(eng.Engine))),
                         (mod_keys | {"__fastrsi_hit__"}, cls_keys))
        self.assertIs(eng.__fastrsi_hit__, W.stub)
        self.assertFalse(hasattr(e, "mean_token"))
        self.assertEqual(w.counts, {})
        e.step(1)
        self.assertEqual(e.steps, 5)

    def test_closure_change_is_a_finding_not_an_exception(self):
        """Proves: a new closure variable is a non-overridable closure_changed finding and nothing changes. Not: changed cell values."""
        name, eng, _ = fresh("closure")
        e = eng.Engine()
        new = edit(OPS, ("return x + k", "return x + k + bias"))
        r = S.stage(name + ".ops", new, [("make_shift.<locals>.shift", at(OPS, "def shift"))])
        self.assertTrue(r["ok"])
        (f,) = r["findings"]
        self.assertEqual((f["check"], f["observed"], f["expected"], f["overridable"]),
                         ("closure_changed", "free variables ('bias', 'k')",
                          "the running closure's ('k',)", False))
        d = F.decide(r["findings"], accept=["closure_changed"])
        self.assertEqual((d["proceed"], d["blocking"]), (False, ["closure_changed"]))
        self.assertEqual(e.step(3), 17)

    def test_an_edit_a_code_swap_cannot_install_is_a_finding(self):
        """Input: two edits of ops staged with auto targets: a comment on `def scale`, and
        STEP_SCALE = 2 -> 3 (a module constant, read by scale). Expected: the comment edit
        has no target and no finding (no code changed); the constant edit has no target and
        an `outside_changes` finding, so `stage_and_apply` applies nothing unless it is
        accepted, and scale(3) stays 6. Pain point: a swap that reports success while the
        engine runs exactly what it ran before. Not: the transaction-level `empty_change`
        verdict (no target, nothing added or deleted), which the coordinator gives."""
        name, _, ops = fresh("empty")
        comment = edit(OPS, ("def scale(x):", "def scale(x):  # faster"))
        const = edit(OPS, ("STEP_SCALE = 2", "STEP_SCALE = 3"))
        r = S.stage(name + ".ops", comment)
        self.assertEqual((r["ok"], r["targets"], r["findings"]), (True, [], []))
        j, w = J.Journal(), W.Witness()
        r = S.stage_and_apply(name + ".ops", const, j, w)
        self.assertEqual((r["ok"], r["stage"]["targets"]), (False, []))
        self.assertEqual([(f["check"], f["subject"], f["accepted"]) for f in r["findings"]],
                         [("outside_changes", name + ".ops", False)])
        self.assertEqual((ops.scale(3), ops.STEP_SCALE, j.entries), (6, 2, []))

    def test_a_loop_frame_live_at_apply_blocks_through_the_helper(self):
        """Input: the toy engine's run_loop running on a thread, and an edit of run_loop
        applied in one call by `swap.stage_and_apply` (no two-phase host). Expected: stage's
        running_frame is advisory, the helper's validate at the apply point finds the frame
        live and blocks (running_frame not accepted): nothing applied, run_loop's code
        unchanged; with "running_frame" accepted it applies. Pain point: a caller that
        applied right after stage swapped a loop that never leaves the old code, with no
        blocking finding. Not: the two-phase host's quiesce point (the coordinator)."""
        name, eng, _ = fresh("saa")
        e, stop = eng.Engine(delay=0.001), threading.Event()
        th = threading.Thread(target=eng.run_loop, args=(e, stop), daemon=True)
        th.start()
        try:
            wait_for(lambda: e.steps > 0)
            new = ENG.replace("engine.step(x)\n", "engine.step(x + 1)\n")
            code = eng.run_loop.__code__
            j, w = J.Journal(), W.Witness()
            r = S.stage_and_apply(name + ".engine", new, j, w)
            self.assertFalse(r["ok"])
            self.assertEqual([(f["check"], f.get("advisory", False), f["accepted"])
                              for f in r["findings"]],
                             [("running_frame", True, True), ("running_frame", False, False)])
            self.assertIs(eng.run_loop.__code__, code)
            self.assertEqual(j.entries, [])
            r = S.stage_and_apply(name + ".engine", new, j, w, accept=["running_frame"])
            self.assertTrue(r["ok"], r["findings"])
            self.assertIsNot(eng.run_loop.__code__, code)
        finally:
            stop.set()
            th.join()


class SwapEdgeCaseTest(unittest.TestCase):
    """Renamed keyword-only parameters, nested generators and lambdas of a live function, and
    added top-level definitions that refer to each other."""

    def _apply(self, base, new, call):
        name = f"kt_sw{next(_n)}"
        (m,) = temp_modules(self, {name: base})
        r = S.stage_and_apply(name, new, J.Journal(), W.Witness())
        return r, call(m)

    def test_a_renamed_keyword_only_parameter_gets_its_default(self):
        """Input: `def f(*, old=False)` edited to `def f(*, new=False)` (same default value).
        Expected: the swap applies and f() returns False. Pain point: __kwdefaults__ stayed
        {"old": False} and every call raised TypeError (missing keyword-only argument
        'new'). Not: positional defaults (a tuple, names do not matter)."""
        r, out = self._apply("def f(*, old=False):\n    return old\n",
                             "def f(*, new=False):\n    return new\n", lambda m: m.f())
        self.assertEqual((r["ok"], out), (True, False))

    def test_a_nested_generator_or_lambda_of_a_live_function_is_live(self):
        """Input: (a) the inner generator of `sum(sum(z * 2 for z in row) for row in mat)`
        edited to z * 3; (b) a lambda inside a local helper of a live function edited.
        Expected: both swap through the live outer function (total([[1, 2]]) = 9,
        outer(1) = 3). Pain point: a blocking no_live_function for code two levels inside a
        live function. Not: code inside a function that nothing holds."""
        base = "def total(mat):\n    return sum(sum(z * 2 for z in row) for row in mat)\n"
        r, out = self._apply(base, base.replace("z * 2", "z * 3"),
                             lambda m: m.total([[1, 2]]))
        self.assertEqual((r["ok"], [f["check"] for f in r["findings"]], out), (True, [], 9))
        base = ("def outer(v):\n    def inner(w):\n        return (lambda y: y * 2)(w)\n"
                "    return inner(v)\n")
        r, out = self._apply(base, base.replace("y * 2", "y * 3"), lambda m: m.outer(1))
        self.assertEqual((r["ok"], [f["check"] for f in r["findings"]], out), (True, [], 3))

    def test_added_top_level_definitions_see_each_other(self):
        """Input: the edit adds (a) a decorator and a helper decorated by it, (b) a class and
        a subclass of it, (c) a helper used as the new default of an edited function.
        Expected: each stages and applies, and the edited f runs. Pain point: patch_invalid
        (NameError) for (a) and (b), defaults_not_evaluated for (c). Not: an added
        definition that uses one added below it (the file order is kept)."""
        cases = [
            ("def my_dec(fn):\n    return fn\n\n@my_dec\ndef helper(x):\n    return x + 1\n\n"
             "def f(x):\n    return helper(x)\n", 2),
            ("class A:\n    k = 1\n\nclass B(A):\n    pass\n\ndef f(x):\n    return x + B.k\n", 2),
            ("def default_fn(x):\n    return x + 1\n\ndef f(x, fn=default_fn):\n"
             "    return fn(x)\n", 2)]
        for new, want in cases:
            r, out = self._apply("def f(x, fn=None):\n    return x\n", new, lambda m: m.f(1))
            self.assertEqual((r["ok"], [f["check"] for f in r["findings"]], out),
                             (True, [], want), new)

    def test_a_changed_top_level_import_updates_the_existing_module_binding_and_undoes(self):
        """Input: a module with `from math import floor as round_op`, edited to
        `from math import ceil as round_op` while keeping the binding name `round_op`.
        Expected: stage and apply replace `round_op` in the module dict, and `undo_to`
        restores the original `floor` binding."""
        module_name = f"kt_sw{next(_n)}"
        base_source = "from math import floor as round_op\n\ndef f(x):\n    return round_op(x)\n"
        edited_source = "from math import ceil as round_op\n\ndef f(x):\n    return round_op(x)\n"
        (module_obj,) = temp_modules(self, {module_name: base_source})
        self.assertEqual(module_obj.f(1.5), 1)
        journal, witness = J.Journal(), W.Witness()
        mark = journal.mark("txn-import")
        result = S.stage_and_apply(module_name, edited_source, journal, witness)
        self.assertEqual((result["ok"], [f["check"] for f in result["findings"]], module_obj.f(1.5)),
                         (True, [], 2))
        undo_result = journal.undo_to(mark)
        self.assertEqual((undo_result["ok"], module_obj.f(1.5)), (True, 1))

    def test_apply_syncs_linecache_for_runtime_inspect_getsourcelines_and_restores_on_undo(self):
        """Input: a swapped function that lazily defines an inner kernel and calls
        `inspect.getsourcelines` on it at runtime (as `@triton.jit` does when compiling a lazy
        kernel inside a function in Megatron climb iter 33 and 55). Expected: while installed,
        `inspect.getsourcelines` reads the patched source lines; after `undo_to`, `linecache`
        is restored to the base file lines without adding extra entries to `journal.describe()`."""
        import inspect
        module_name = f"kt_sw{next(_n)}"
        base_source = (
            "def compute(val):\n"
            "    return val + 1\n"
        )
        edited_source = (
            "import inspect\n\n"
            "def compute(val):\n"
            "    def _lazy_triton_kernel(inp):\n"
            "        return inp * 10\n"
            "    lines, _ = inspect.getsourcelines(_lazy_triton_kernel)\n"
            "    return ''.join(lines).strip(), _lazy_triton_kernel(val)\n"
        )
        (module_obj,) = temp_modules(self, {module_name: base_source})
        journal, witness = J.Journal(), W.Witness()
        mark = journal.mark("txn-linecache")
        result = S.stage_and_apply(module_name, edited_source, journal, witness)
        self.assertTrue(result["ok"], result["findings"])
        src_text, val_out = module_obj.compute(3)
        self.assertIn("def _lazy_triton_kernel(inp):", src_text)
        self.assertIn("return inp * 10", src_text)
        self.assertEqual(val_out, 30)
        undo_result = journal.undo_to(mark)
        self.assertEqual((undo_result["ok"], module_obj.compute(3)), (True, 4))
        base_lines, _ = inspect.getsourcelines(module_obj.compute)
        self.assertIn("return val + 1", "".join(base_lines))

    def test_shifted_lines_in_later_swap_do_not_break_earlier_lazy_triton_builder_source_lookup(self):
        """Input: txn 1 adds a lazy Triton-like builder (NOT called yet) at the bottom of a
        module; txn 2 edits an earlier function and inserts 15 lines above the builder, leaving
        the builder unchanged (reproducing the FastVideo H3 patches 02-08 stacked live-install
        `CompilationError ... StopIteration` failure). Expected: calling txn 1's lazy builder
        AFTER txn 2 still reads the exact nested kernel source via `inspect.getsourcelines`, and
        undoing both transactions removes all `<fastrsi-live:...>` linecache entries."""
        import inspect
        import linecache
        from . import coordinator as C

        module_name = f"kt_sw{next(_n)}"
        base_source = (
            "def head(x):\n"
            "    return x + 1\n"
        )
        txn1_source = (
            "import inspect\n\n"
            "def head(x):\n"
            "    return x + 1\n\n\n"
            "def lazy_builder():\n"
            "    def _lazy_kernel(inp):\n"
            "        return inp * 7\n"
            "    lines, _ = inspect.getsourcelines(_lazy_kernel)\n"
            "    return ''.join(lines).strip(), _lazy_kernel(3)\n"
        )
        padding = "\n".join(f"    x = x + {i}" for i in range(15))
        txn2_source = txn1_source.replace("    return x + 1\n", f"{padding}\n    return x + 2\n")

        (module_obj,) = temp_modules(self, {module_name: base_source})
        part = C.LocalParticipant("core")
        self.assertTrue(part.prepare({"id": "t1", "modules": {module_name: {"source": txn1_source}}})["ok"])
        self.assertTrue(part.commit("t1")["ok"])
        # Do NOT call lazy_builder() yet: install txn 2 shifting lines above lazy_builder first.
        self.assertTrue(part.prepare({"id": "t2", "modules": {module_name: {"source": txn2_source}}})["ok"])
        self.assertTrue(part.commit("t2")["ok"])

        src_text, out_val = module_obj.lazy_builder()
        self.assertIn("def _lazy_kernel(inp):", src_text)
        self.assertIn("return inp * 7", src_text)
        self.assertEqual(out_val, 21)

        self.assertTrue(part.undo("t2")["ok"])
        self.assertTrue(part.undo("t1")["ok"])
        self.assertTrue(part.at_base()["ok"])
        live_keys = [k for k in linecache.cache if k.startswith(f"<fastrsi-live:{module_obj.__file__}#")]
        self.assertEqual(live_keys, [])

    def test_a_method_added_to_or_removed_from_a_nested_class_is_refused(self):
        """Input: `class Outer: class Inner: def f(self, x): return x + 1`; edits that (a) add
        `Inner.helper` and make f call it, (b) remove a second method `Inner.g`. Expected:
        stage blocks with an outside_changes finding that is not overridable and names the
        method, so even accept="all" does not proceed; nothing is applied. Pain point: with
        outside_changes accepted, the swap installed f calling self.helper, which the live
        Inner never got: AttributeError on the next call. Not: editing an existing nested
        method's body (a normal target)."""
        base = ("class Outer:\n    class Inner:\n        def f(self, x):\n"
                "            return x + 1\n\n        def g(self):\n            return 0\n")
        added = base.replace("return x + 1", "return self.helper(x)\n\n"
                             "        def helper(self, x):\n            return x + 10")
        removed = base.replace("\n\n        def g(self):\n            return 0", "")
        for new, member in ((added, "Outer.Inner.helper"), (removed, "Outer.Inner.g")):
            name = f"kt_sw{next(_n)}"
            (mod_obj,) = temp_modules(self, {name: base})
            res = S.stage(name, new, "auto")
            blocking = [finding for finding in res["findings"] if finding["check"] == "outside_changes"
                        and not finding["overridable"]]
            self.assertEqual(len(blocking), 1, res["findings"])
            self.assertIn(member, blocking[0]["observed"])
            self.assertFalse(F.decide(res["findings"], "all")["proceed"])
            self.assertEqual(mod_obj.Outer.Inner().f(1), 2)


if __name__ == "__main__":
    unittest.main()
