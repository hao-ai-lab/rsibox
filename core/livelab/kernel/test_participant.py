"""Golden tests of the participant side of a transaction (`coordinator.LocalParticipant`) on
the toy engine (CPU, standard library only). Here the test drives the two phases by hand, as
a host does; the host itself (`Coordinator`) is the next kernel PR.

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
(or python3 -m pytest -q core/livelab/kernel)
"""
import threading
import unittest

from . import coordinator as C
from . import locate as L
from . import swap as S
from .test_detectors import ESTIMATOR, ESTIMATOR_NEW, _n, temp_modules
from .test_locate import ENG, OPS, at, edit, fresh, loads_constant
from .test_witness import wait_for


def step_target():
    return {"source": None, "base_hash": L.text_hash(ENG),
            "targets": [["Engine.step", at(ENG, "def step")]]}


class ParticipantTest(unittest.TestCase):

    def test_two_ranks_switch_together_and_undo_to_the_same_code_objects(self):
        """Input: two ranks (two toy copies, each a LocalParticipant) and one plan: step()
        returns y * 100. The test is the host: prepare on both, validate on both, commit on
        both, then undo on both. Expected: both vote yes with no finding and state
        "prepared"; nothing runs the edit before commit (step(3) = 17); after commit both
        compute 1700, report the same code hash for the target and state "committed"; after
        undo both run their exact base code objects again (17), state "undone", empty
        journals. Not: the host's all-or-nothing rules (adapter/test_host.py), real RPC."""
        ranks = []
        for r in ("rank0", "rank1"):
            name, eng, _ = fresh(r)
            ranks.append((C.LocalParticipant(r, alias={"toy": name}), eng, eng.Engine()))
        base = [eng.Engine.step.__code__ for _, eng, _ in ranks]
        plan = {"id": "t", "modules": {"toy.engine": {
            **step_target(), "source": edit(ENG, ("return y", "return y * 100"))}}}
        prep = [p.prepare(plan) for p, _, _ in ranks]
        self.assertEqual([(r["ok"], r["findings"]) for r in prep], [(True, [])] * 2)
        self.assertEqual([p.status("t")["state"] for p, _, _ in ranks], ["prepared"] * 2)
        self.assertEqual([e.step(3) for _, _, e in ranks], [17, 17])
        self.assertTrue(all(p.validate("t")["ok"] for p, _, _ in ranks))
        self.assertTrue(all(p.commit("t")["ok"] for p, _, _ in ranks))
        self.assertEqual([e.step(3) for _, _, e in ranks], [1700, 1700])
        hashes = [p.proof("t")["hashes"] for p, _, _ in ranks]
        self.assertEqual((len(hashes[0]), hashes[0] == hashes[1]), (1, True))
        self.assertEqual([p.status("t")["state"] for p, _, _ in ranks], ["committed"] * 2)
        self.assertTrue(all(p.undo("t")["ok"] for p, _, _ in ranks))
        self.assertEqual([eng.Engine.step.__code__ for _, eng, _ in ranks], base)
        self.assertEqual([e.step(3) for _, _, e in ranks], [17, 17])
        self.assertEqual([(p.status("t")["state"], p.journal.entries) for p, _, _ in ranks],
                         [("undone", [])] * 2)

    def test_edit_that_changes_no_code_is_refused(self):
        """Proves: a comment-only edit, and a module-constant edit whose outside_changes is accepted, are both refused as empty_change instead of committing nothing. Not: a planner that lists unchanged targets on purpose."""
        name, _, ops = fresh("empty")
        p = C.LocalParticipant("rank0", alias={"toy": name})
        comment = edit(OPS, ("def scale(x):", "def scale(x):  # faster"))
        const = edit(OPS, ("STEP_SCALE = 2", "STEP_SCALE = 3"))
        for i, (src, accept) in enumerate([(comment, ()), (const, "all")]):
            r = p.prepare({"id": f"e{i}", "accept": accept, "modules": {"toy.ops": {
                "source": src, "targets": S.changed_targets(OPS, src)}}})
            self.assertEqual((r["ok"], r["staged"], [(f["check"], f["accepted"])
                                                     for f in r["findings"]]),
                             (False, {}, [("outside_changes", True)] * i
                              + [("empty_change", False)]))
        self.assertEqual((ops.scale(3), ops.STEP_SCALE, p.journal.entries), (6, 2, []))

    def test_running_frame_never_runs_the_new_code(self):
        """Proves: zero calls under a running loop frame give running_frame, an uncalled function zero_witness. Not: frames in C or other processes."""
        name, eng, _ = fresh("loop")
        e, stop = eng.Engine(delay=0.001), threading.Event()
        th = threading.Thread(target=eng.run_loop, args=(e, stop), daemon=True)
        th.start()
        try:
            wait_for(lambda: e.steps > 0)
            new = edit(ENG, ("engine.step(x)\n", "engine.step(x + 1)\n"),
                       ("self.steps += 1", "self.steps = self.steps + 1"),
                       ("self.tokens.clear()", "self.tokens = []"))
            p = C.LocalParticipant("core", alias={"toy": name}, quiesce=e.at_boundary)
            plan = {"id": "t", "accept": ["running_frame"], "modules": {"toy.engine": {
                "source": new, "targets": [["run_loop", at(ENG, "def run_loop")],
                                           ["Engine.step", at(ENG, "def step")],
                                           ["Engine.drain", at(ENG, "def drain")]]}}}
            prep = p.prepare(plan)
            self.assertEqual([(f["check"], f["accepted"]) for f in prep["findings"]],
                             [("running_frame", True)])
            self.assertTrue(p.commit("t")["ok"])
            n = e.steps
            wait_for(lambda: e.steps >= n + 5)
            chk = p.witness_check("t")
        finally:
            stop.set()
            th.join()
        k = {q: f"toy.engine:{q}:{at(ENG, s)}" for q, s in
             [("run_loop", "def run_loop"), ("Engine.step", "def step"),
              ("Engine.drain", "def drain")]}
        self.assertEqual(chk["counts"][k["run_loop"]], 0)
        self.assertGreaterEqual(chk["counts"][k["Engine.step"]], 5)
        self.assertEqual({f["subject"]: f["check"] for f in chk["findings"]},
                         {k["run_loop"]: "running_frame", k["Engine.drain"]: "zero_witness"})
        self.assertEqual(e.tokens[-1], 13)   # the loop still calls step(1), not step(2) -> 15

    def test_a_new_field_votes_no_and_routes_to_t1(self):
        """Input: the estimator edit (only the new __init__ sets _block, step reads it) as a
        plan, with a live estimator. Expected: the participant votes no with blocking
        new_fields and constructor_edit findings (the edit is in Estimator.__init__), route
        T1, tagged with the participant's name; nothing is staged,
        and the live step still returns 256. Not: a rebuild or restart that resolves it."""
        (m,) = temp_modules(self, {"kt_part_est": ESTIMATOR})
        est = m.Runner().speculator.acceptance_estimator
        p = C.LocalParticipant("workers")
        r = p.prepare({"id": "nf", "modules": {"kt_part_est": {"source": ESTIMATOR_NEW}}})
        self.assertEqual((r["ok"], [(f["check"], f["route"], f["accepted"], f["participant"])
                                    for f in r["findings"]]),
                         (False, [("new_fields", "T1", False, "workers"),
                                  ("constructor_edit", "T1", False, "workers")]))
        self.assertEqual((p.status("nf")["state"], est.step(5000)), ("aborted", 256))


class LineMapChainTest(unittest.TestCase):
    """The participant composes its committed swaps' line maps, so a later target keyed at
    the installed text's line finds the live code with no qualname guess."""

    def _chain(self, targets2):
        name, _, ops = fresh("linechain")
        p = C.LocalParticipant("core", alias={"toy": name})
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
        self.assertTrue(loads_constant(ops.make_shift.__code__, 5))
        self.assertEqual((ops.scale(3), ops.make_shift(10)(1)), (6, 11))
        self.assertTrue(p.undo("e2")["ok"])                 # make_shift back, scale still v1
        self.assertFalse(loads_constant(ops.make_shift.__code__, 5))
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


class EditAddedDefinitionTest(unittest.TestCase):

    def test_a_function_an_earlier_transaction_added_can_be_edited_again(self):
        """Input: transaction a adds `helper` (and calls it from scale), transaction b edits
        `helper`'s body. Expected: b commits and runs the new body; undoing b then a returns
        scale's base code and removes helper. Fails without the fix: b aborts with
        base_mismatch ("running code ..."), because the code of a definition a transaction
        added was not counted as the participant's own (an overnight H3 hill-climb lost most
        proposals this way: they tuned helpers that kept edits had added). Not: added classes'
        methods (same path, not exercised here)."""
        n = next(_n)
        mod = f"kt_added{n}"
        base = "def scale(x):\n    return x * 2\n"
        (mod_obj,) = temp_modules(self, {mod: base})
        p = C.LocalParticipant("core")
        t1 = ("def helper(x):\n    return x + 1\n\n\n"
              "def scale(x):\n    return helper(x) * 2\n")
        self.assertTrue(p.prepare({"id": "a", "modules": {mod: {"source": t1}}})["ok"])
        self.assertTrue(p.commit("a")["ok"])
        self.assertEqual(mod_obj.scale(1), 4)
        t2 = t1.replace("return x + 1", "return x + 5")
        r = p.prepare({"id": "b", "modules": {mod: {"source": t2}}})
        self.assertTrue(r["ok"], [f["check"] for f in r.get("findings", [])])
        self.assertTrue(p.commit("b")["ok"])
        self.assertEqual(mod_obj.scale(1), 12)
        self.assertTrue(p.undo("b")["ok"])
        self.assertEqual(mod_obj.scale(1), 4)
        self.assertTrue(p.undo("a")["ok"])
        self.assertEqual(mod_obj.scale(1), 2)
        self.assertFalse(hasattr(mod_obj, "helper"))

    def test_a_method_of_a_class_an_earlier_transaction_added_can_be_edited_again(self):
        """Input: transaction txn-a adds `class Helper` with method `compute`, transaction
        txn-b edits `Helper.compute`. Expected: txn-b commits and runs the new method body;
        undoing txn-b then txn-a restores base and `at_base()["ok"]` is True."""
        idx = next(_n)
        mod_name = f"kt_added_cls{idx}"
        base_src = "def scale(val):\n    return val * 2\n"
        (mod_obj,) = temp_modules(self, {mod_name: base_src})
        part = C.LocalParticipant("core")
        first_patch = (
            "class Helper:\n"
            "    def compute(self, val):\n"
            "        return val + 1\n\n\n"
            "def scale(val):\n"
            "    return Helper().compute(val) * 2\n"
        )
        self.assertTrue(part.prepare({"id": "txn-a", "modules": {mod_name: {"source": first_patch}}})["ok"])
        self.assertTrue(part.commit("txn-a")["ok"])
        self.assertEqual(mod_obj.scale(1), 4)
        second_patch = first_patch.replace("return val + 1", "return val + 5")
        prep_res = part.prepare({"id": "txn-b", "modules": {mod_name: {"source": second_patch}}})
        self.assertTrue(prep_res["ok"], [item["check"] for item in prep_res.get("findings", [])])
        self.assertTrue(part.commit("txn-b")["ok"])
        self.assertEqual(mod_obj.scale(1), 12)
        self.assertTrue(part.undo("txn-b")["ok"])
        self.assertEqual(mod_obj.scale(1), 4)
        self.assertTrue(part.undo("txn-a")["ok"])
        self.assertEqual((mod_obj.scale(1), part.at_base()["ok"]), (2, True))

    def test_cached_builder_and_local_class_patterns_undo_cleanly_to_base(self):
        """Input: an edit that (1) adds a lazy builder function caching a cyclic JIT-like
        wrapper on its own `__dict__`, (2) swaps a method that caches a kernel wrapper on its
        own `__dict__` and on its class, and (3) defines a local class and recursive helper
        inside a swapped function (reproducing the FastVideo H3 and Megatron climb iter 33, 55,
        and 157 undo-poisoning patterns). Expected: after running the edited functions, `undo`
        succeeds with no `closure_not_restored` finding and `at_base()["ok"]` is True."""
        idx = next(_n)
        mod_name = f"kt_cached_jit{idx}"
        base_src = (
            "class CrossEntropyOp:\n"
            "    @staticmethod\n"
            "    def apply_op(val):\n"
            "        return val * 2\n\n\n"
            "def embed_op(val):\n"
            "    return val + 1\n"
        )
        (mod_obj,) = temp_modules(self, {mod_name: base_src})
        part = C.LocalParticipant("core")
        patched_src = (
            "class _FakeJIT:\n"
            "    def __init__(self, func):\n"
            "        self.fn = func\n"
            "        self.binder = self._bind\n"
            "    def _bind(self):\n"
            "        return self.fn\n\n\n"
            "def _get_builder_kernel():\n"
            "    kern = getattr(_get_builder_kernel, 'kernel', None)\n"
            "    if kern is None:\n"
            "        @_FakeJIT\n"
            "        def _inner_kernel(val):\n"
            "            return val + 10\n"
            "        _get_builder_kernel.kernel = _inner_kernel\n"
            "        kern = _inner_kernel\n"
            "    return kern\n\n\n"
            "class CrossEntropyOp:\n"
            "    @staticmethod\n"
            "    def apply_op(val):\n"
            "        owner = CrossEntropyOp\n"
            "        kern = getattr(owner, '_shift_exp_sum_kernel', None)\n"
            "        if kern is None:\n"
            "            @_FakeJIT\n"
            "            def _cls_kernel(inp):\n"
            "                return inp * 3\n"
            "            owner._shift_exp_sum_kernel = _cls_kernel\n"
            "            CrossEntropyOp.apply_op._self_kernel = _cls_kernel\n"
            "            kern = _cls_kernel\n"
            "        return kern.fn(_get_builder_kernel().fn(val))\n\n\n"
            "def embed_op(val):\n"
            "    class LocalGradFn:\n"
            "        @staticmethod\n"
            "        def forward(inp):\n"
            "            return inp + 4\n"
            "    def walk(steps):\n"
            "        return 0 if steps <= 0 else 1 + walk(steps - 1)\n"
            "    return LocalGradFn.forward(val) + walk(2)\n"
        )
        self.assertTrue(part.prepare({"id": "txn-jit", "modules": {mod_name: {"source": patched_src}}})["ok"])
        self.assertTrue(part.commit("txn-jit")["ok"])
        self.assertEqual((mod_obj.CrossEntropyOp.apply_op(2), mod_obj.embed_op(3)), (36, 9))
        undo_res = part.undo("txn-jit")
        self.assertEqual((undo_res["ok"], undo_res.get("findings", [])), (True, []))
        self.assertFalse(hasattr(mod_obj.CrossEntropyOp, "_shift_exp_sum_kernel"))
        self.assertFalse(hasattr(mod_obj.CrossEntropyOp.apply_op, "_self_kernel"))
        self.assertEqual(
            (mod_obj.CrossEntropyOp.apply_op(2), mod_obj.embed_op(3), part.at_base()["ok"]),
            (4, 4, True),
        )

    def test_lru_cache_and_container_cached_jit_builders_undo_cleanly_to_base(self):
        """Input: an edit that (1) adds a `@functools.lru_cache` lazy JIT builder and
        (2) caches a tuple/dict of JIT wrappers on module globals and a class attribute
        (FastVideo H3 patches 09-13 pattern). Expected: after running the builders, `undo`
        clears LRU and container caches with zero `closure_not_restored` findings and
        `at_base()['ok']` is True."""
        idx = next(_n)
        mod_name = f"kt_lru_jit{idx}"
        base_src = (
            "class Block:\n"
            "    @staticmethod\n"
            "    def run(val):\n"
            "        return val + 1\n"
        )
        (mod_obj,) = temp_modules(self, {mod_name: base_src})
        part = C.LocalParticipant("core")
        patched_src = (
            "import functools\n\n"
            "class _FakeJIT:\n"
            "    def __init__(self, func):\n"
            "        self.fn = func\n"
            "        self.device_caches = {0: ({'k': func}, {})}\n"
            "        self.pre_run_hooks = [lambda *a: func]\n\n\n"
            "@functools.lru_cache(maxsize=4)\n"
            "def _cached_builder(tag):\n"
            "    @_FakeJIT\n"
            "    def _k_a(x):\n"
            "        return x + 10\n"
            "    @_FakeJIT\n"
            "    def _k_b(x):\n"
            "        return x * 2\n"
            "    return (_k_a, _k_b)\n\n\n"
            "class Block:\n"
            "    @staticmethod\n"
            "    def run(val):\n"
            "        pair = _cached_builder('default')\n"
            "        globals()['_RUNTIME_KERNEL_TUPLE'] = pair\n"
            "        Block._runtime_kernel_map = {'pair': pair}\n"
            "        return pair[1].fn(pair[0].fn(val))\n"
        )
        self.assertTrue(part.prepare({"id": "txn-lru", "modules": {mod_name: {"source": patched_src}}})["ok"])
        self.assertTrue(part.commit("txn-lru")["ok"])
        self.assertEqual(mod_obj.Block.run(3), 26)
        undo_res = part.undo("txn-lru")
        self.assertEqual((undo_res["ok"], undo_res.get("findings", [])), (True, []))
        self.assertFalse(hasattr(mod_obj, "_RUNTIME_KERNEL_TUPLE"))
        self.assertFalse(hasattr(mod_obj.Block, "_runtime_kernel_map"))
        self.assertEqual((mod_obj.Block.run(3), part.at_base()["ok"]), (4, True))

    def test_undo_keeps_base_globals_and_class_attributes_the_engine_set(self):
        """Input: a module with `make_shift(k)` (returns `shift`, x + k), a base global
        `DEFAULT_SHIFT = make_shift(1)` and an empty `class Config`. The edit makes shift
        x + 10 * k. While it is installed the engine rebinds `DEFAULT_SHIFT = make_shift(2)`
        and sets `Config.port = 8080` (a plain value). Expected: undo is ok; DEFAULT_SHIFT is
        still bound and runs base code (5 + 2 = 7); Config.port stays; at_base is ok. Then a
        base global deleted by hand makes at_base false and names it in `missing_names`. Pain
        point: undo deleted every module global holding a new nested code, even one the base
        defined, and every class attribute added since apply on any class of the module; and
        at_base did not look for base names that were gone. A name `engine_names` accepts is
        listed there instead. Not: a base global rebound to a closure that cannot go back
        (closure_not_restored)."""
        mod_name = f"kt_keep_globals{next(_n)}"
        base_src = ("class Config:\n    pass\n\n\n"
                    "def make_shift(k):\n    def shift(x):\n        return x + k\n"
                    "    return shift\n\n\nDEFAULT_SHIFT = make_shift(1)\n")
        (mod_obj,) = temp_modules(self, {mod_name: base_src})
        part = C.LocalParticipant("core")
        patched = base_src.replace("x + k", "x + 10 * k")
        self.assertTrue(part.prepare({"id": "g1", "modules": {mod_name: {"source": patched}}})["ok"])
        self.assertTrue(part.commit("g1")["ok"])
        mod_obj.DEFAULT_SHIFT = mod_obj.make_shift(2)
        mod_obj.Config.port = 8080
        self.assertEqual(mod_obj.DEFAULT_SHIFT(5), 25)
        undo_res = part.undo("g1")
        self.assertEqual((undo_res["ok"], undo_res["findings"]), (True, []))
        self.assertEqual((hasattr(mod_obj, "DEFAULT_SHIFT"), getattr(mod_obj.Config, "port", None)),
                         (True, 8080))
        self.assertEqual(mod_obj.DEFAULT_SHIFT(5), 7)
        del mod_obj.Config.port
        self.assertTrue(part.at_base()["ok"], part.at_base())
        del mod_obj.DEFAULT_SHIFT
        base_res = part.at_base()
        self.assertEqual((base_res["ok"], base_res["missing_names"]), (False, {mod_name: ["DEFAULT_SHIFT"]}))
        part.engine_names = lambda name: name == "DEFAULT_SHIFT"    # one the engine deletes
        base_res = part.at_base()
        # (at_base stays false here: the module's code digest lost DEFAULT_SHIFT's function)
        self.assertEqual((base_res["missing_names"], base_res["engine_names"]),
                         ({}, {mod_name: ["DEFAULT_SHIFT"]}))


LAZY = '''import importlib

DEP = "{dep}"


def ensure_dep():
    """vLLM's LazyLoader: at first use, bind the module under its dotted name."""
    globals()[DEP] = importlib.import_module(DEP)
    return globals()[DEP].VALUE


def scale(x):
    return x * 2
'''


class LazyImportAtBaseTest(unittest.TestCase):

    def test_a_module_the_engine_imports_lazily_after_base_is_not_an_extra_name(self):
        """Input (vLLM #62872's shape: backend_guidance's LazyLoader binds `llguidance.hf` and
        `llguidance.torch` into the module at first use): commit an edit of scale(), let
        the engine run the lazy import (the module gains the name of the dependency), undo.
        Expected: at_base is ok, the name is listed in `lazy_imports`, not `extra_names`.
        Then the same kind of value under a name our transaction added (a patch that adds
        `import json as kt_json`), left after undo as a failed undo would leave it: that
        one is an extra name and at_base is false. Fails without the fix: at_base is false
        after the clean undo (extra_names lists the dependency). Not: names a repair cell
        writes outside the journal (they would read as the engine's)."""
        n = next(_n)
        mod, dep = f"kt_lazy_mod{n}", f"kt_lazy_dep{n}"
        src = LAZY.format(dep=dep)
        _, lazy = temp_modules(self, {dep: "VALUE = 7\n", mod: src})
        p = C.LocalParticipant("core")
        new = edit(src, ("return x * 2", "return x * 3"))
        self.assertTrue(p.prepare({"id": "a", "modules": {mod: {"source": new}}})["ok"])
        self.assertTrue(p.commit("a")["ok"])
        self.assertEqual((lazy.scale(1), lazy.ensure_dep()), (3, 7))   # the engine runs
        self.assertTrue(p.undo("a")["ok"])
        ab = p.at_base()
        self.assertEqual((ab["ok"], ab["extra_names"]), (True, {}))
        self.assertEqual(ab["lazy_imports"], {mod: [dep]})

        imp = edit(src, ("import importlib\n", "import importlib\nimport json as kt_json\n"),
                   ("return x * 2", "return x * len(kt_json.dumps(1))"))
        self.assertTrue(p.prepare({"id": "b", "modules": {mod: {"source": imp}}})["ok"])
        self.assertTrue(p.commit("b")["ok"])
        import json
        self.assertIs(lazy.kt_json, json)
        self.assertTrue(p.undo("b")["ok"])
        lazy.kt_json = json                    # what a failed undo would leave
        ab = p.at_base()
        self.assertEqual((ab["ok"], ab["extra_names"], ab["lazy_imports"]),
                         (False, {mod: ["kt_json"]}, {mod: [dep]}))


class _Box:
    pass


SCALE3 = edit(OPS, ("x * STEP_SCALE", "x * 3"))       # scale(1): 2 -> 3


class ParticipantEdgeCaseTest(unittest.TestCase):
    """Malformed plans, reused txn ids, partial rollbacks and undos, and witness checks of a
    txn not committed here."""

    def _part(self, tag, **kw):
        name, _, ops = fresh(tag)
        return C.LocalParticipant("core", alias={"toy": name}, **kw), ops

    def test_a_plan_without_an_id_or_with_missing_fields_never_raises(self):
        """Input: a plan with no "id"; a module spec with no "source"; a repair with no
        "declares". Expected: the first is staged under a generated id that the reply names
        (`txn`) and commits (scale(1) = 3); the others vote no with a patch_invalid finding.
        Pain point: KeyError out of a call documented "never raises" (the id is optional).
        Not: a host that names the id itself (it does)."""
        p, ops = self._part("noid")
        r = p.prepare({"modules": {"toy.ops": {"source": SCALE3}}})
        self.assertTrue(r["ok"], r["findings"])
        self.assertEqual((p.status(r["txn"])["state"], p.commit(r["txn"])["ok"], ops.scale(1)),
                         ("prepared", True, 3))
        for i, bad in enumerate([{"modules": {"toy.ops": {}}},
                                 {"repairs": [{"cell": "x = 1"}]}]):
            r = p.prepare({"id": f"bad{i}", **bad})
            self.assertEqual((r["ok"], [f["check"] for f in r["findings"]],
                              p.status(f"bad{i}")["state"]),
                             (False, ["patch_invalid"], "aborted"))

    def test_an_id_already_in_use_is_refused(self):
        """Input: commit "c" (scale x 3), then prepare another plan with the same id "c".
        Expected: the second prepare votes no (patch_invalid naming the id); undo("c") then
        takes the first edit back and the participant is at base. Pain point: the second
        prepare overwrote the first one's journal mark, so its undo left scale x 3 running.
        Not: reusing the id of an aborted or undone transaction (allowed)."""
        p, ops = self._part("reuse")
        self.assertTrue(p.prepare({"id": "c", "modules": {"toy.ops": {"source": SCALE3}}})["ok"])
        self.assertTrue(p.commit("c")["ok"])
        again = edit(SCALE3, ("    return x + 1", "    return x + 2"))
        r = p.prepare({"id": "c", "modules": {"toy.ops": {"source": again}}})
        self.assertEqual((r["ok"], [f["check"] for f in r["findings"]], p.status("c")["state"]),
                         (False, ["patch_invalid"], "committed"))
        self.assertTrue(p.undo("c")["ok"])
        self.assertEqual((ops.scale(1), ops.activate(1), p.at_base()["ok"]), (2, 2, True))

    def test_a_commit_rollback_that_leaves_a_change_is_partial(self):
        """Input: a repair cell that changes an undeclared scope object (box.val = 2, which
        the journal cannot undo) and then raises, or does not raise (its
        repair_not_undoable blocks the commit). Expected: the commit fails, the state is
        "partial", at_base is not ok and poisoned, and undo repeats the finding instead of
        "nothing committed". Pain point: box.val stayed 2 while at_base said ok and undo
        said ok. Not: changes the audit cannot see (deeper than one level)."""
        for tail in ("raise RuntimeError('boom')\n", ""):
            box = _Box()
            box.val = 1
            p, _ = self._part("rollback", scope={"box": box})
            self.assertTrue(p.prepare({"id": "r", "modules": {"toy.ops": {"source": SCALE3}},
                                       "repairs": [{"cell": "box.val = 2\n" + tail,
                                                    "declares": []}]})["ok"])
            c = p.commit("r")
            self.assertIn("repair_not_undoable", [f["check"] for f in c["findings"]])
            ab, u = p.at_base(), p.undo("r")
            self.assertEqual((c["ok"], box.val, p.status("r")["state"], ab["ok"],
                              ab["poisoned"], u["ok"]),
                             (False, 2, "partial", False, True, False), tail)
            self.assertIn("repair_not_undoable", [f["check"] for f in u["findings"]])

    def test_an_undo_cascade_that_was_partial_stays_partial_for_later_txns(self):
        """Input: txn A records an undo hook that raises; txn B commits after A; undo(A)
        (partial, cascades over B), then undo(B). Expected: undo(B) is not ok and repeats
        the partial undo's findings; B stays "partial". Pain point: undo(B) answered ok,
        "nothing committed", and set B "undone". Not: which txn's entry failed (the journal
        does not say)."""
        p, _ = self._part("cascade", scope={"bad": lambda: 1 / 0})
        self.assertTrue(p.prepare({"id": "A", "modules": {"toy.ops": {"source": SCALE3}},
                                   "repairs": [{"cell": "journal.record_call(bad, 'bad')\n",
                                                "declares": []}]})["ok"])
        self.assertTrue(p.commit("A")["ok"])
        two = edit(SCALE3, ("    return x + 1", "    return x + 2"))
        self.assertTrue(p.prepare({"id": "B", "modules": {"toy.ops": {"source": two}}})["ok"])
        self.assertTrue(p.commit("B")["ok"])
        self.assertEqual((p.undo("A")["ok"], p.status("B")["state"]), (False, "partial"))
        ub = p.undo("B")
        self.assertEqual((ub["ok"], p.status("B")["state"]), (False, "partial"))
        self.assertIn("undo_incomplete", [f["check"] for f in ub["findings"]])

    def test_a_witness_check_of_a_txn_not_committed_here_is_not_a_pass(self):
        """Input: witness_check of an unknown id; of a committed txn whose target never ran.
        Expected: the unknown id is not ok with no_witness_possible; every finding names the
        participant. Pain point: an unknown id answered ok with no finding (a pass). Not: the
        host's per-rank merge."""
        p, _ = self._part("witness")
        r = p.witness_check("nope")
        self.assertEqual((r["ok"], [(f["check"], f["participant"]) for f in r["findings"]]),
                         (False, [("no_witness_possible", "core")]))
        self.assertTrue(p.prepare({"id": "w", "modules": {"toy.ops": {"source": SCALE3}}})["ok"])
        self.assertTrue(p.commit("w")["ok"])
        r = p.witness_check("w")
        self.assertEqual([(f["check"], f["participant"]) for f in r["findings"]],
                         [("zero_witness", "core")])


if __name__ == "__main__":
    unittest.main()
