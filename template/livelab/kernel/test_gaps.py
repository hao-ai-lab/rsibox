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


class RecipeRebuildsTest(unittest.TestCase):
    def test_a_recipe_that_rebuilds_the_class_resolves_its_new_fields(self):
        """Input (vLLM #63827's shape through core_txn's U4): Estimator.__init__ gains
        `self._block`, step() reads it, a live estimator exists; the plan says its recipe
        builds every live Estimator again with the edited __init__ (`"rebuilds":
        ["Estimator"]`, as U4 / R4 does for the scheduler, R5 / R5m for model objects).
        Expected: prepare ok with no accept; new_fields is advisory, resolved_by "recipe", and
        no backfill is planned (commit sets nothing on the live estimator); without
        "rebuilds" the same plan still votes no. Fails before this change: the plan voted no
        with new_fields although the rebuild sets the field (tip-63827 needed accept)."""
        name = f"kt_est{next(_n)}"
        (m,) = temp_modules(self, {name: ESTIMATOR})
        worker = m.Runner()
        p = C.LocalParticipant("workers", scope={"worker": worker})
        plan = {"id": "rb", "modules": {name: {"source": ESTIMATOR_NEW,
                                                "rebuilds": ["Estimator"]}}}
        r = p.prepare(plan)
        self.assertTrue(r["ok"], r)
        (f,) = [f for f in r["findings"] if f["check"] == "new_fields"]
        self.assertEqual((f["subject"], f["advisory"], f["resolved_by"], f["accepted"]),
                         (f"{name}:Estimator._block", True, "recipe", True))
        c = p.commit("rb")
        self.assertTrue(c["ok"], c)
        self.assertFalse(c.get("backfill"))
        self.assertNotIn("_block", vars(worker.speculator.acceptance_estimator))
        self.assertTrue(p.undo("rb")["ok"])
        other = {"id": "rb2", "modules": {name: {"source": ESTIMATOR_NEW,
                                                 "rebuilds": ["Speculator"]}}}
        self.assertFalse(p.prepare(other)["ok"])


TRITON_VLLM_SHAPE = """import triton
import triton.language as tl


@triton.jit
def _helper(x):
    return x + 1


@triton.jit(do_not_specialize_on_alignment=["BATCH_SIZE"])
def _mask_kernel(x_ptr, BATCH_SIZE, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    tl.store(x_ptr + i, _helper(tl.load(x_ptr + i)))


def launcher(x):
    return _mask_kernel


class Sampler:
    def run(self, x):
        return launcher(x)
"""


class TritonAutoTargetsTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("triton"), "triton is not installed")
    def test_triton_only_edits_are_staged_by_auto_targets(self):
        """Input: a module shaped like vLLM's topk_topp_triton.py (a @triton.jit helper, a
        kernel decorated `@triton.jit(do_not_specialize_on_alignment=[...])`, Python
        launchers), an edit that touches only the two @triton.jit bodies, targets left to the
        kernel (no "targets": auto). Expected: both JIT functions are the staged targets
        (keys at their decorator lines), each with a jit_rebuild finding; after commit the
        kernel's src holds the edit and its cache key changed. Not: a launch (GPU receipts).
        The NVL runs saw "targets 0" because the lab's live patch takes @triton.jit
        definitions out of the targets it hands the kernel (its own Triton path), not
        because auto staging misses them."""
        name = f"kt_trauto{next(_n)}"
        (m,) = temp_modules(self, {name: TRITON_VLLM_SHAPE})
        new = TRITON_VLLM_SHAPE.replace("x + 1", "x + 7").replace(
            "tl.store(x_ptr + i, _helper", "tl.store(x_ptr + i + 0, _helper")
        old_key = m._mask_kernel.cache_key
        p = C.LocalParticipant("workers")
        r = p.prepare({"id": "ta", "modules": {name: {"source": new}}})
        self.assertTrue(r["ok"], r)
        lines = TRITON_VLLM_SHAPE.splitlines()
        deco = {q: 1 + next(i for i, s in enumerate(lines) if s.startswith(f"def {q}(")) - 1
                for q in ("_helper", "_mask_kernel")}
        self.assertEqual(sorted(map(tuple, r["targets"][name])),
                         sorted((q, ln) for q, ln in deco.items()))
        self.assertEqual(sorted(f["subject"] for f in r["findings"]
                                if f["check"] == "jit_rebuild"),
                         sorted(f"{name}:{q}:{ln}" for q, ln in deco.items()))
        self.assertTrue(p.commit("ta")["ok"])
        self.assertIn("i + 0", m._mask_kernel.src)
        self.assertNotEqual(m._mask_kernel.cache_key, old_key)
        self.assertTrue(p.undo("ta")["ok"])
        self.assertEqual(m._mask_kernel.cache_key, old_key)


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
        self.assertIsNotNone(r["receipt"]["edit_to_run_s"])     # the cell rebuilt in the gap
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
        self.assertEqual([(f["check"], f["advisory"]) for f in r["findings"]],
                         [("docstring_only", True)])

    def test_class_and_module_docstring_edits_are_advisory_no_ops(self):
        """Input (#51899's shape, kv_events through core_txn): an edit of a class docstring
        only, then of the module docstring only. Expected: prepare ok with one advisory
        docstring_only finding each, commit swaps nothing (no code entry in the journal, the
        same code objects).
        A comment-only edit is still empty_change (no docstring differs). Fails before this
        change: the class docstring edit was refused (empty_change, or outside_changes
        through core_txn's planner)."""
        name = f"kt_cdoc{next(_n)}"
        text = ('"""Module doc."""\n\n\nclass Ev:\n    """Old class doc."""\n\n'
                '    def go(self):\n        return 1\n')
        (m,) = temp_modules(self, {name: text})
        code = m.Ev.go.__code__
        for i, new in enumerate((text.replace("Old class doc.", "New class doc."),
                                 text.replace("Module doc.", "Module doc, edited."))):
            p = C.LocalParticipant("x")
            r = p.prepare({"id": f"c{i}", "modules": {name: {"source": new}}})
            self.assertTrue(r["ok"], r)
            self.assertEqual([(f["check"], f["advisory"]) for f in r["findings"]],
                             [("docstring_only", True)])
            c = p.commit(f"c{i}")
            self.assertEqual((c["ok"], [e for e in c["journal"] if e["kind"] == "code"]),
                             (True, []))
            self.assertIs(m.Ev.go.__code__, code)
        r = C.LocalParticipant("x").prepare({"id": "cm", "modules": {name: {
            "source": text.replace("        return 1", "        return 1  # same")}}})
        self.assertEqual([f["check"] for f in r["findings"]], ["empty_change"])


FUTURE = '''from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import OrderedDict


def f(r: OrderedDict) -> int:
    return 1
'''


class FutureAnnotationsTest(unittest.TestCase):
    def test_future_annotations_and_type_checking_imports_stage(self):
        """Input (#62591's shape): a module with `from __future__ import annotations`; the
        edit adds a TYPE_CHECKING-only import (Counter), an added function annotated with it,
        and f() now calls that function. Expected: prepare ok with no finding (the new
        TYPE_CHECKING import counts as an import, the added unit keeps the annotations
        future), commit, f(None) == 7. Fails before this change: patch_invalid NameError
        'Counter' (the added unit compiled without the future) and outside_changes (the
        TYPE_CHECKING block)."""
        name = f"kt_fut{next(_n)}"
        (m,) = temp_modules(self, {name: FUTURE})
        new = FUTURE.replace("    from collections import OrderedDict\n",
                             "    from collections import Counter, OrderedDict\n").replace(
            "    return 1\n", "    return helper(r)\n\n\ndef helper(r: Counter) -> int:\n"
            "    return 7\n")
        p = C.LocalParticipant("x")
        r = p.prepare({"id": "f", "modules": {name: {"source": new}}})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["findings"], [])
        self.assertTrue(p.commit("f")["ok"])
        self.assertEqual(m.f(None), 7)


DECORATED = "import functools\n\n\nclass A:\n    @functools.lru_cache\n    def f(self):\n        return 1\n"


class DecoratedKeyMessageTest(unittest.TestCase):
    def test_not_in_patch_names_the_decorator_line(self):
        """Input: a decorated method targeted at its `def` line (6) instead of its key (the
        decorator line, 5). Expected: not_in_patch says A.f is keyed at line [5] and why.
        Fails before this change: 'no definition at this line in the base file' only."""
        name = f"kt_dec{next(_n)}"
        temp_modules(self, {name: DECORATED})
        r = S.stage(name, DECORATED.replace("return 1", "return 2"), [["A.f", 6]])
        (f,) = [f for f in r["findings"] if f["check"] == "not_in_patch"]
        self.assertIn("keyed at line(s) [5]", f["observed"])
        self.assertIn("first decorator", f["observed"])


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


class AddedMemberTest(unittest.TestCase):
    BASE = "class B:\n    def v(self):\n        return 40\n\n\nclass C(B):\n    def w(self):\n        return 1\n"

    def test_added_properties_and_descriptors_stage_and_work(self):
        """Input (vLLM #62359's shape, which failed on every rank at stage with "'property'
        object has no attribute '__closure__'"): a class gains a @property using zero-arg
        super(), a property with setter and deleter, a staticmethod, a classmethod using
        super(), a functools.cached_property, and a method whose decorator returns a constant
        (a member with no code). Expected: stage ok with no finding, apply ok, every member
        works on the live class; undo removes them all. Fails before this change: stage
        raised on the property (patch_invalid)."""
        name = f"kt_mem{next(_n)}"
        (m,) = temp_modules(self, {name: "import functools\n\n\n" + self.BASE})
        new = ("import functools\n\n\n" + self.BASE
               + "\n    @property\n    def q(self):\n        return super().v() + 2\n"
               "\n    @property\n    def r(self):\n        return self.__dict__.get('_r', 0)\n"
               "\n    @r.setter\n    def r(self, v):\n        self._r = v\n"
               "\n    @r.deleter\n    def r(self):\n        self.__dict__.pop('_r', None)\n"
               "\n    @staticmethod\n    def s():\n        return 5\n"
               "\n    @classmethod\n    def k(cls):\n        return cls.__name__\n"
               "\n    @functools.cached_property\n    def cp(self):\n        return super().v() * 2\n"
               "\n    @(lambda f: 11)\n    def const(self):\n        return 0\n")
        r = S.stage(name, new)
        self.assertTrue(r["ok"], r.get("error"))
        self.assertEqual([f["check"] for f in r["findings"]], [])
        j = J.Journal()
        mark = j.mark("t")
        self.assertTrue(S.apply(r["staged"], j, W.Witness())["ok"])
        c = m.C()
        c.r = 3
        self.assertEqual((c.q, c.r, m.C.s(), m.C.k(), c.cp, m.C.const), (42, 3, 5, "C", 80, 11))
        del c.r
        self.assertEqual(c.r, 0)
        self.assertTrue(j.undo_to(mark)["ok"])
        self.assertFalse(any(hasattr(m.C, n) for n in ("q", "r", "s", "k", "cp", "const")))

    def test_an_edited_property_getter_is_swapped(self):
        """Input: a class with a property (getter and setter); the getter's body is edited.
        Expected: the getter alone is the target, the namespace finds it (no heap scan), and
        the live property returns the new value; undo returns the old. Not: a property
        replaced by another object."""
        name = f"kt_prop{next(_n)}"
        text = ("class P:\n    @property\n    def x(self):\n        return 1\n\n"
                "    @x.setter\n    def x(self, v):\n        pass\n")
        (m,) = temp_modules(self, {name: text})
        r = S.stage(name, text.replace("return 1", "return 2"))
        self.assertEqual((r["ok"], r["targets"], r["timings"]["scan"]),
                         (True, [["P.x", 2]], "namespace"))
        j = J.Journal()
        mark = j.mark("t")
        self.assertTrue(S.apply(r["staged"], j, W.Witness())["ok"])
        self.assertEqual(m.P().x, 2)
        j.undo_to(mark)
        self.assertEqual(m.P().x, 1)


class GeneratorTargetTest(unittest.TestCase):
    def test_a_generator_target_needs_no_heap_scan_and_a_suspended_one_still_blocks(self):
        """Input: an edit of a generator function (as vLLM's safetensors_weights_iterator), first
        with no generator of it alive, then with one suspended (started, not finished), with
        the gc heap frozen as vLLM does. Expected: no whole-heap pass (`all_objects` is never
        called) in either case; with none alive no finding; with one suspended a blocking
        running_frame. Not: a generator made before gc.freeze() and still suspended (only objects
        made since the freeze are looked at). Fails before this change: stage called all_objects (one get_objects
        and a Python pass: 0.42 s of a 0.49 s stage per rank on DeepSeek-V4.1 TP4)."""
        from unittest import mock
        name = f"kt_gen{next(_n)}"
        text = "def it(n):\n    for i in range(n):\n        yield i\n"
        (m,) = temp_modules(self, {name: text})
        new = text.replace("yield i", "yield i * 2")
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        scans = []
        real = W.all_objects
        with mock.patch.object(W, "all_objects", lambda: scans.append(1) or real()):
            r = S.stage(name, new)
            self.assertEqual((r["ok"], r["findings"], scans), (True, [], []))
            g = m.it(3)
            next(g)                        # made after the freeze, as at serving time
            r = S.stage(name, new)
            self.assertEqual(([(f["check"], f.get("advisory", False)) for f in r["findings"]],
                              scans), ([("running_frame", False)], []))
            self.assertEqual(list(g), [1, 2])


class ParticipantMissingTest(unittest.TestCase):
    def test_a_dp_client_that_drops_replies_blocks_the_transaction(self):
        """Input: a worker group of 2 data-parallel groups x 2 ranks (each rank its own copy of
        a module), reached through a fake DP client that delivers every call to every rank
        but returns only DP group 0's replies (as vLLM's DP client did at EP4DP4); the group
        declares the 4 ranks it expects. Expected: the transaction aborts with a blocking,
        not-overridable participant_missing naming dp1/0 and dp1/1, even with accept "all";
        no rank runs the edit afterwards. Fails before this change: the transaction committed
        on all 4 ranks while its witness, receipt and undo covered 2, with no finding."""
        mods = []
        for i in range(4):
            name = f"kt_dp{next(_n)}"
            (m,) = temp_modules(self, {name: "def f():\n    return 'base'\n"})
            mods.append(m)
        ranks = [(f"dp{i // 2}/{i % 2}", C.LocalParticipant(
            "workers", alias={"pkg": mods[i].__name__})) for i in range(4)]

        def dp_client(method, payload):
            rs = [{**getattr(lp, method)(payload), "rank": rid} for rid, lp in ranks]
            return C.merge_ranks(method, rs[:2])          # DP group 0's replies only
        g = C.CallParticipant("workers", dp_client)
        g.expected = [rid for rid, _ in ranks]
        co = C.Coordinator([g], phase2_deadline_s=None)
        r = co.run({"id": "dp", "accept": "all", "modules": {"pkg": {
            "source": "def f():\n    return 'edit'\n"}}})
        self.assertEqual(r["outcome"], "aborted")
        fs = [f for f in r["receipt"]["findings"] if f["check"] == "participant_missing"]
        self.assertEqual(sorted({f["observed"].split(":")[0] for f in fs}), ["abort", "prepare"])
        self.assertEqual({(tuple(f["missing"]), f["overridable"]) for f in fs},
                         {(("dp1/0", "dp1/1"), False)})
        self.assertEqual([m.f() for m in mods], ["base"] * 4)


class AtBaseTest(unittest.TestCase):
    def test_apply_then_undo_is_at_base_and_a_leftover_is_not(self):
        """Input (a warm engine pool: every job applies a PR live, then undoes it): a swap of
        a module plus a repair cell, committed and undone. Expected: at_base ok (no pending
        transaction, journal empty, the module's live code digest equal to its base one, no
        extra names or attributes) on the participant and through the coordinator; while
        committed it is not ok (pending, digests differ). Then a cell that sets an attribute
        outside the journal (undeclared, its repair_not_undoable accepted) and an undo: not
        ok, extra_attrs names it. Fails before this change: there was no at_base, so a pool
        could not tell a clean engine from one an undo left dirty."""
        class Engine:
            pass
        e = Engine()
        name = f"kt_ab{next(_n)}"
        (m,) = temp_modules(self, {name: "def f():\n    return 'base'\n"})
        p = C.LocalParticipant("workers", scope={"engine": e})
        co = C.Coordinator([p], phase2_deadline_s=None)
        plan = {"id": "a", "modules": {name: {"source": "def f():\n    return 'edit'\n"}},
                "repairs": [{"cell": "engine.x = 1", "declares": ["engine"]}]}
        self.assertEqual(co.run(plan)["outcome"], "committed")
        b = p.at_base()
        self.assertEqual((b["ok"], b["pending_txns"]), (False, ["a"]))
        self.assertNotEqual(*b["modules"][name])
        self.assertTrue(co.undo("a")["ok"])
        b = p.at_base()
        self.assertEqual((b["ok"], b["pending_txns"], b["journal_len"], b["extra_attrs"],
                          b["extra_names"]), (True, [], 0, {}, {}))
        self.assertEqual(*b["modules"][name])
        self.assertTrue(co.at_base()["ok"])
        setattr(m, "__compiled_fn_1", object())    # a tracer's own global: apart, not left over
        m.sub__fastrsi_hit__ = W.stub
        p.engine_names = lambda n: n.startswith("__compiled_fn")
        b = p.at_base()
        self.assertEqual((b["ok"], b["engine_names"]), (True, {name: ["__compiled_fn_1"]}))
        m.stray = 1
        self.assertEqual(p.at_base()["extra_names"], {name: ["stray"]})
        del m.stray
        r = co.run({"id": "b", "accept": ["repair_not_undoable"], "repairs": [
            {"cell": "engine.leftover = 1", "declares": []}]})
        self.assertEqual(r["outcome"], "committed")
        self.assertTrue(co.undo("b")["ok"])
        b = co.at_base()
        self.assertFalse(b["ok"])
        self.assertEqual(b["participants"]["workers"]["extra_attrs"], {"engine": ["leftover"]})


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
