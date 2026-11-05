"""Golden tests for gaps a live vLLM run found in the transaction kernel (CPU).

Run from template/livelab:  python3 -m unittest kernel.test_gaps -v
(The real-Triton case runs only where `triton` imports; it needs no GPU.)

Each case names the live failure it is for:
- JitTargetTest: editing a @triton.jit kernel committed a silent no-op (the GPU kept the old
  compiled kernel; only a misleading zero_witness at proof).
- NewFieldsTest: vLLM #63819, a new step() read a field only the edited __init__ sets; the
  kernel committed with no finding and the engine died with AttributeError.
- WitnessStubTest: undo deleted __fastrsi_hit__ while compiled code still called it: NameError.
- CellJournalTest: a repair cell that failed after releasing engine state was reported rolled
  back while that state stayed; a cell-only transaction proved with an empty witness.
- DocstringTest / AddedMethodWitnessTest: smaller ones from the same runs.
"""
import gc
import itertools
import os
import shutil
import sys
import tempfile
import importlib
import unittest

from . import coordinator as C
from . import journal as J
from . import locate as L
from . import swap as S
from . import witness as W

_n = itertools.count()

FAKE_JIT = '''
"""A Triton stand-in: compiles from the source TEXT read at build time, caches the compiled
function per device, inlines the @jit functions it calls when it compiles, runs pre-run hooks
at every launch. Like Triton, it never runs the function's bytecode."""
import collections, inspect, re, textwrap


class JITFunction:
    def __init__(self, fn, debug=None):
        self.fn = fn
        self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        src = textwrap.dedent("".join(self.raw_src))
        self._src = src[re.search(r"^def\\s+\\w+\\s*\\(", src, re.MULTILINE).start():]
        self.debug = debug
        self.hash = None
        self.device_caches = collections.defaultdict(self.create_binder)
        self.pre_run_hooks = []

    def create_binder(self):
        return {}

    def compiled(self):
        cache = self.device_caches["cpu"]
        if "k" not in cache:
            g = dict(self.fn.__globals__)
            g.update({n: v.compiled() for n, v in g.items() if isinstance(v, JITFunction)
                      and v is not self and re.search(n + r"\\(", self._src)})
            ns = {}
            exec(self._src, g, ns)
            cache["k"] = ns[self.fn.__name__]
        return cache["k"]

    def __getitem__(self, grid):
        return self.run

    def run(self, *args):
        for h in self.pre_run_hooks:
            h(*args)
        return self.compiled()(*args)


def jit(fn=None, **kw):
    return JITFunction(fn, **kw) if fn is not None else (lambda f: JITFunction(f, **kw))
'''

KERNELS = '''from {fake} import jit


@jit
def helper(x):
    return x + 1


@jit(debug=False)
def kernel(x):
    return helper(x) * 2


@jit
def scale_kernel(x):
    return x * 3
'''


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


def line_of(src, needle):
    return next(i for i, s in enumerate(src.splitlines(), 1) if needle in s)


class JitTargetTest(unittest.TestCase):
    def _mods(self):
        fake, kern = f"tritonfake_{next(_n)}", f"kt_kernels{next(_n)}"
        text = KERNELS.format(fake=fake)
        _, k = temp_modules(self, {fake: FAKE_JIT, kern: text})
        return kern, k, text

    def test_a_jit_kernel_edit_runs_on_the_next_launch_and_undo_restores_the_old_kernel(self):
        """Input: kernels compiled and launched (kernel(1) = (1+1)*2 = 4, scale_kernel(1) = 3),
        then an edit of the @jit helper (x + 1 -> x + 100) and of scale_kernel (x * 3 ->
        x * 5), targets auto, through a participant. Expected: an advisory jit_rebuild finding
        per target at prepare (no heap scan: the namespace finds the function inside the JIT
        object); after commit kernel(1) = 202 (the caller that inlines the helper was reset)
        and scale_kernel(1) = 5; the witness counts launches (helper: 2 launches of its caller,
        scale_kernel: 1); undo gives 4 and 3 again from the SAME compiled-kernel caches as
        before the edit. Fails before this change: the swap changed only fn.__code__, so the
        launches kept returning 4 and 3, with witness 0 and no finding at prepare."""
        kern, k, text = self._mods()
        self.assertEqual((k.kernel[(1,)](1), k.scale_kernel[(1,)](1)), (4, 3))
        caches = (k.kernel.device_caches, k.helper.device_caches, k.scale_kernel.device_caches)
        new = text.replace("return x + 1", "return x + 100").replace("x * 3", "x * 5")
        p = C.LocalParticipant("workers")
        r = p.prepare({"id": "jit", "modules": {kern: {"source": new}}})
        self.assertTrue(r["ok"], r)
        jf = sorted(f["subject"].split(":")[1] for f in r["findings"]
                    if f["check"] == "jit_rebuild" and f["advisory"])
        self.assertEqual(jf, ["helper", "scale_kernel"])
        self.assertEqual(r["timings"][kern]["scan"], "namespace")
        self.assertTrue(p.commit("jit")["ok"])
        self.assertEqual((k.kernel[(1,)](1), k.kernel[(1,)](2)), (202, 204))
        self.assertEqual(k.scale_kernel[(1,)](1), 5)
        counts = {key.split(":")[1]: n for key, n in p.witness_check("jit")["counts"].items()}
        self.assertEqual(counts, {"helper": 2, "scale_kernel": 1})
        # a chained edit of the same kernel: one launch counts once (its hook replaces the
        # first edit's), and undo of the chain goes back to the first edit's kernel
        self.assertTrue(p.prepare({"id": "jit2", "modules": {kern: {
            "source": new.replace("x * 5", "x * 7")}}})["ok"])
        self.assertTrue(p.commit("jit2")["ok"])
        self.assertEqual(k.scale_kernel[(1,)](1), 7)
        counts = {key.split(":")[1]: n for key, n in p.witness_check("jit2")["counts"].items()}
        self.assertEqual(counts, {"scale_kernel": 1})
        self.assertTrue(p.undo("jit2")["ok"])
        self.assertEqual(k.scale_kernel[(1,)](1), 5)
        self.assertTrue(p.undo("jit")["ok"])
        self.assertEqual((k.kernel[(1,)](1), k.scale_kernel[(1,)](1)), (4, 3))
        self.assertEqual((k.kernel.device_caches, k.helper.device_caches,
                          k.scale_kernel.device_caches), caches)
        self.assertTrue(all(a is b for a, b in zip(
            (k.kernel.device_caches, k.helper.device_caches, k.scale_kernel.device_caches),
            caches)))
        self.assertEqual((k.kernel.pre_run_hooks, k.helper.pre_run_hooks), ([], []))

    @unittest.skipUnless(importlib.util.find_spec("triton"), "triton is not installed")
    def test_real_triton_jit_is_rebuilt_from_the_new_text(self):
        """Input: a real @triton.jit kernel (no launch, no GPU), edit `+ 1` -> `+ 2`.
        Expected: after commit its src holds `+ 2` and its cache_key (what Triton keys the
        compiled kernel on) differs; its device caches are new; undo restores the old src,
        cache_key and the same cache object. Not: a launch (GPU receipts)."""
        name = f"kt_triton{next(_n)}"
        text = ("import triton\nimport triton.language as tl\n\n\n@triton.jit\n"
                "def add_one(x_ptr):\n    tl.store(x_ptr, tl.load(x_ptr) + 1)\n")
        (m,) = temp_modules(self, {name: text})
        old_key, old_caches = m.add_one.cache_key, m.add_one.device_caches
        p = C.LocalParticipant("workers")
        self.assertTrue(p.prepare({"id": "t", "modules": {name: {
            "source": text.replace("+ 1", "+ 2")}}})["ok"])
        self.assertTrue(p.commit("t")["ok"])
        self.assertIn("+ 2", m.add_one.src)
        self.assertNotEqual(m.add_one.cache_key, old_key)
        self.assertIsNot(m.add_one.device_caches, old_caches)
        self.assertEqual(len(m.add_one.pre_run_hooks), 1)
        self.assertTrue(p.undo("t")["ok"])
        self.assertIn("+ 1", m.add_one.src)
        self.assertEqual(m.add_one.cache_key, old_key)
        self.assertIs(m.add_one.device_caches, old_caches)


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
    def test_a_field_only_the_new_init_sets_blocks_and_a_backfill_makes_it_run(self):
        """Input (vLLM #63819's shape): Estimator.__init__ gains `self._block = max(256,
        max_num_reqs * 2)` and step() reads self._block; the live estimator, built by the old
        __init__, hangs off worker.speculator (which keeps max_num_reqs = 1000); the gc heap
        is frozen, as in vLLM. Expected: prepare votes no with new_fields (1 of 1 live
        instance lacks it; max_num_reqs from its owner); accepted with "backfill": true, the
        commit sets _block = 2000 on the live estimator and step(5000) returns 2000; undo
        removes the field and step(5000) is 256 again. Fails before this change: prepare voted
        yes with no finding and step raised AttributeError."""
        name = f"kt_est{next(_n)}"
        (m,) = temp_modules(self, {name: ESTIMATOR})
        worker = m.Runner()
        est = worker.speculator.acceptance_estimator
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        p = C.LocalParticipant("workers", scope={"worker": worker})
        plan = {"id": "nf", "modules": {name: {"source": ESTIMATOR_NEW}}}
        r = p.prepare(plan)
        self.assertFalse(r["ok"])
        (f,) = [f for f in r["findings"] if f["check"] == "new_fields"]
        self.assertEqual((f["subject"], f["accepted"]), (f"{name}:Estimator._block", False))
        self.assertIn("1 of 1 live instance(s)", f["observed"])
        self.assertEqual(f["backfill"]["params"],
                         {"worker.speculator.acceptance_estimator":
                          {"max_num_reqs": "owner -1.max_num_reqs"}})
        self.assertTrue(f["backfill"]["backfill"])
        r = p.prepare({**plan, "id": "nf2", "accept": ["new_fields"], "backfill": True})
        self.assertTrue(r["ok"], r)
        c = p.commit("nf2")
        self.assertTrue(c["ok"], c)
        self.assertEqual(c["backfill"]["set"],
                         [["worker.speculator.acceptance_estimator", "_block"]])
        self.assertEqual((est._block, est.step(5000)), (2000, 2000))
        self.assertEqual(m.Estimator(10, 3)._block, 256)       # new objects: the new __init__
        self.assertTrue(p.undo("nf2")["ok"])
        self.assertNotIn("_block", vars(est))
        self.assertEqual(est.step(5000), 256)

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


class WitnessStubTest(unittest.TestCase):
    def test_code_compiled_while_installed_still_runs_after_undo(self):
        """Input: a swap, then code made while it is installed that calls __fastrsi_hit__ (as
        Dynamo's transformed bytecode of a traced edit did), then undo. Expected: the old
        code is back, the global is the no-op stub (counts nothing), the compiled code still
        runs. Fails before this change: undo deleted the global, and the call raised NameError
        (a live graph-mode engine died at the next step)."""
        name = f"kt_stub{next(_n)}"
        text = "def f(x):\n    return x + 1\n"
        (m,) = temp_modules(self, {name: text})
        j, w = J.Journal(), W.Witness()
        mark = j.mark("t")
        r = S.stage(name, text.replace("x + 1", "x + 2"))
        self.assertTrue(S.apply(r["staged"], j, w)["ok"])
        compiled = eval(compile("lambda x: (__fastrsi_hit__('k'), x)[1]", m.__file__, "eval"),
                        m.__dict__)
        self.assertEqual((m.f(1), compiled(7)), (3, 7))
        self.assertTrue(j.undo_to(mark)["ok"])
        before = dict(w.counts)
        self.assertEqual((m.f(1), compiled(7)), (2, 7))
        self.assertIs(m.__fastrsi_hit__, W.stub)
        self.assertEqual(w.counts, before)            # the stub counts nothing
        r = S.stage(name, text.replace("x + 1", "x + 3"))      # a second swap rebinds it
        mark = j.mark("t2")
        self.assertTrue(S.apply(r["staged"], j, w)["ok"])
        self.assertEqual(m.f(1), 4)
        self.assertEqual(w.counts[r["staged"].keys[0]], 1)
        j.undo_to(mark)
        self.assertIs(m.__fastrsi_hit__, W.stub)


class CellJournalTest(unittest.TestCase):
    def test_a_failed_cell_rolls_back_after_the_code_and_says_what_it_left(self):
        """Input: a swap plus a repair cell that releases the engine's graphs (journaling
        "recapture" with record_call(after_code=True)), changes an undeclared object, and
        then raises. Expected: the commit fails with repair_failed AND a repair_not_undoable
        finding for the undeclared change; the recapture ran during the rollback once the
        code was back at base (it captured the base code), so the graphs are back. Fails
        before this change: the recapture ran before the code undo (it captured the edit),
        and the undeclared change was not reported."""
        name = f"kt_cell{next(_n)}"
        text = "VERSION = 'base'\n\n\ndef forward():\n    return 'base'\n"
        (m,) = temp_modules(self, {name: text})

        class Engine:
            def __init__(self):
                self.graphs = ["g:base"]
        eng, other = Engine(), Engine()
        cell = ("engine.graphs.clear()\n"
                "journal.record_call(lambda: engine.graphs.append('g:' + mod.forward()),"
                " 'recapture', after_code=True)\n"
                "other.graphs = []\n"
                "raise RuntimeError('capture refused')\n")
        p = C.LocalParticipant("workers", scope={"engine": eng, "other": other, "mod": m})
        plan = {"id": "c", "modules": {name: {"source": text.replace("return 'base'",
                                                                     "return 'edit'")}},
                "repairs": [{"cell": cell, "declares": []}]}
        self.assertTrue(p.prepare(plan)["ok"])
        c = p.commit("c")
        self.assertFalse(c["ok"])
        checks = [f["check"] for f in c["findings"]]
        self.assertEqual(checks[0], "repair_failed")
        self.assertIn("repair_not_undoable", checks)
        self.assertEqual((m.forward(), eng.graphs), ("base", ["g:base"]))

    def test_a_cell_that_needs_a_rebuild_gets_it_outside_the_gap(self):
        """Input: a host with an R1 rebuild; a cell that changes engine state only a rebuild
        restores calls journal.need_rebuild("R1"). Expected: when the cell fails, the
        transaction rolls back and R1 runs once from the base code (why rollback) after the
        gap; when it succeeds, undo runs R1 (why undo) after the code undo, not inside it.
        Fails before this change: need_rebuild did not exist; the alternative, an inverse that
        recompiles inside undo, held the gap past the 5 s deadline and killed a live engine."""
        name = f"kt_nr{next(_n)}"
        text = "def forward():\n    return 'base'\n"
        (m,) = temp_modules(self, {name: text})
        calls = []

        def r1(txn, why, deadline_at):
            calls.append((why, m.forward()))
            return {"ok": True, "findings": []}
        p = C.LocalParticipant("workers")
        co = C.Coordinator([p], phase2_deadline_s=None, rebuilds={"R1": r1})
        edit = {name: {"source": text.replace("'base'", "'edit'")}}
        cell = "journal.need_rebuild('R1')\n"
        r = co.run({"id": "f", "modules": edit, "repairs": [
            {"cell": cell + "raise RuntimeError('capture refused')", "declares": []}]})
        self.assertEqual((r["outcome"], calls), ("rolled_back", [("rollback", "base")]))
        del calls[:]
        r = co.run({"id": "ok", "modules": edit, "repairs": [{"cell": cell, "declares": []}]})
        self.assertEqual((r["outcome"], calls, r["receipt"]["actual_recipes"]),
                         ("committed", [], ["R0 swap", "R10 repair", "R1 recapture"]))
        self.assertTrue(co.undo("ok")["ok"])
        self.assertEqual(calls, [("undo", "base")])

    def test_a_cell_only_transaction_does_not_prove_silently(self):
        """Input: a transaction made only of a repair cell. Expected: its proof has a
        no_witness_possible finding; a cell that arms a counter (`arm(key)`) and calls it
        proves with that count, and undo removes the key. Fails before this change: the
        proof had an empty witness and no finding."""
        class Engine:
            pass
        e = Engine()
        p = C.LocalParticipant("workers", scope={"engine": e})
        self.assertTrue(p.prepare({"id": "c1", "repairs": [
            {"cell": "engine.x = 1", "declares": ["engine"]}]})["ok"])
        self.assertTrue(p.commit("c1")["ok"])
        w = p.witness_check("c1")
        self.assertEqual([f["check"] for f in w["findings"]], ["no_witness_possible"])
        cell = "engine.launch = arm('cell:launch')\n"
        self.assertTrue(p.prepare({"id": "c2", "repairs": [{"cell": cell,
                                                             "declares": ["engine"]}]})["ok"])
        self.assertTrue(p.commit("c2")["ok"])
        e.launch()
        e.launch()
        w = p.witness_check("c2")
        self.assertEqual((w["counts"], w["findings"]), ({"cell:launch": 2}, []))
        self.assertTrue(p.undo("c2")["ok"])
        self.assertNotIn("cell:launch", p.witness.counts)


class DocstringTest(unittest.TestCase):
    def test_a_docstring_edit_is_not_a_target(self):
        """Input: one function's docstring edited, another function's body edited. Expected:
        auto targets = only the body edit, no outside_changes; a docstring-only edit alone is
        an empty_change. Fails before this change: the docstring edit was swapped, then its
        witness read 0 (a false zero_witness)."""
        name = f"kt_doc{next(_n)}"
        text = ('"""Module."""\n\n\ndef a():\n    """Old doc."""\n    return 1\n\n\n'
                'def b():\n    return 2\n')
        temp_modules(self, {name: text})
        both = text.replace("Old doc.", "New doc.").replace("return 2", "return 3")
        r = S.stage(name, both)
        self.assertEqual((r["targets"], [f["check"] for f in r["findings"]]),
                         ([["b", line_of(text, "def b")]], []))
        p = C.LocalParticipant("x")
        r = p.prepare({"id": "d", "modules": {name: {"source": text.replace("Old doc.",
                                                                           "New doc.")}}})
        self.assertEqual([f["check"] for f in r["findings"]], ["empty_change"])


class AddedMethodWitnessTest(unittest.TestCase):
    def test_an_added_method_has_a_witness(self):
        """Input: an edit that adds a method and calls it from a changed one. Expected: the
        witness check counts the added method's calls. Fails before this change: added
        definitions were left out of the witness check (#63819: _refit ran, unproven)."""
        name = f"kt_add{next(_n)}"
        text = "class A:\n    def run(self):\n        return 1\n"
        (m,) = temp_modules(self, {name: text})
        new = ("class A:\n    def run(self):\n        return self._refit()\n\n"
               "    def _refit(self):\n        return 2\n")
        p = C.LocalParticipant("x")
        self.assertTrue(p.prepare({"id": "a", "modules": {name: {"source": new}}})["ok"])
        self.assertTrue(p.commit("a")["ok"])
        a = m.A()
        self.assertEqual((a.run(), a.run()), (2, 2))
        counts = {k.split(":")[1]: n for k, n in p.witness_check("a")["counts"].items()}
        self.assertEqual(counts, {"A.run": 2, "A._refit": 2})


class StageAndApplyTest(unittest.TestCase):
    def test_a_loop_frame_live_at_apply_blocks_through_the_helper(self):
        """Input: the toy engine's run_loop running on a thread, and an edit of run_loop
        applied in one call by `swap.stage_and_apply` (no two-phase host). Expected: stage's
        running_frame is advisory, the helper's validate at the apply point finds the frame
        live and blocks (running_frame not accepted): nothing applied, run_loop's code
        unchanged; with "running_frame" accepted it applies. Fails before this change: the
        helper did not exist, and an adapter that called stage then apply swapped a loop that
        never leaves the old code, with no blocking finding (since running frames became
        advisory at stage)."""
        from . import toy
        name = f"kt_saa{next(_n)}"
        toy.load_copy(name)
        self.addCleanup(lambda: [sys.modules.pop(k) for k in list(sys.modules)
                                 if k == name or k.startswith(name + ".")])
        eng = sys.modules[name + ".engine"]
        src = toy.source("engine")
        e, stop = eng.Engine(delay=0.001), __import__("threading").Event()
        th = __import__("threading").Thread(target=eng.run_loop, args=(e, stop), daemon=True)
        th.start()
        try:
            while e.steps == 0:
                pass
            new = src.replace("engine.step(x)\n", "engine.step(x + 1)\n")
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


class AnalyzerStartTest(unittest.TestCase):
    def test_start_says_whether_it_spawned(self):
        """Input: ProcessAnalyzer.start() twice, then after the child is killed. Expected:
        True (spawned), False (a live child kept), True (respawned); a LocalParticipant
        records it (`analyzer_spawned`). Fails before this change: start() returned the
        analyzer itself, so callers could not record a spawn."""
        a = S.ProcessAnalyzer()
        self.addCleanup(a.close)
        self.assertEqual((a.start(), a.start()), (True, False))
        a._p.kill()
        a._p.wait()
        self.assertIs(a.start(), True)
        b = S.ProcessAnalyzer()
        self.addCleanup(b.close)
        p = C.LocalParticipant("x", analyzer=b)
        self.assertIs(p.analyzer_spawned, True)
        self.assertIs(C.LocalParticipant("y", analyzer=b).analyzer_spawned, False)


if __name__ == "__main__":
    unittest.main()
