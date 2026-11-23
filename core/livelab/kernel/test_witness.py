"""Golden tests of the witness on the toy engine (CPU, standard library only; one test
imports torch on CPU and is skipped without it).

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
(or python3 -m pytest -q core/livelab/kernel)

These tests use the witness without the swap, so `install` below does by hand what the swap
does for a witness: add the counter call to the new body (`witness.inject`),
compile the module text (`inject` leaves the
new name's location to `ast.fix_missing_locations`, as the swap's compile does), put the code on every live holder and bind `__rsibox_hit__` to the
participant's hook, each through the journal. Toy arithmetic: see test_locate.py.
"""
import ast
import gc
import inspect
import sys
import threading
import time
import types
import unittest

from . import findings as F
from . import journal as J
from . import locate as L
from . import swap as S
from . import witness as W
from .test_locate import ENG, OPS, at, edit, fresh, module_from


def install(mod, source, targets, journal, witness, key_module):
    """Witness `targets` [(qualname, first line)] of `source` (the module's text, maybe
    edited) in the live module `mod`. Returns the witness keys."""
    tree = ast.parse(source, mod.__file__)
    keys = []
    for q, line in targets:
        name = q.rsplit(".", 1)[-1]
        (node,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == name and (n.decorator_list[0].lineno if n.decorator_list
                                           else n.lineno) == line]
        k = L.key_of(key_module, q, line)
        W.inject(node, k)
        keys.append(k)
    codes = L.code_tree(compile(ast.fix_missing_locations(tree), mod.__file__, "exec"))[0]
    journal.set_item(mod.__dict__, W.HIT, witness.hook, absent=W.stub)
    for (q, line), k in zip(targets, keys):
        journal.set_item(witness.counts, k, 0)
        for fn in L.locate(mod.__name__, q, line)["functions"]:
            journal.set_code(fn, codes[(q, line)][0])
    return keys


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise AssertionError("timed out")
        time.sleep(0.005)


class WitnessCountTest(unittest.TestCase):
    """Pain point: a swap that never ran reported as a pass, because nothing counted the new
    code's calls on the participant that should have run it."""

    def test_calls_are_counted_per_participant_and_zero_calls_is_not_reached(self):
        """Input: two participants (rank0, rank1: two copies of the toy) witness Engine.step
        and Engine.drain under the shared key module `toy.engine`; rank0 steps 3 times,
        rank1 never. Expected: rank0 counts step 3, drain 0, and its check reports one
        zero_witness ("0 calls since the swap") for drain only; rank1 reports zero_witness for
        both; each says "ran during capture 0, ran live N". After undo the global is the no-op
        stub, the keys are gone, and steps count nothing. Not: the swap's staging (a later
        PR), calls in other processes (each participant checks its own)."""
        lines = {"Engine.step": at(ENG, "def step"), "Engine.drain": at(ENG, "def drain")}
        ranks = {}
        for rank in ("rank0", "rank1"):
            _, eng, _ = fresh(rank)
            j, w = J.Journal(rank), W.Witness()
            mark = j.mark("t")
            keys = install(eng, ENG, list(lines.items()), j, w, "toy.engine")
            ranks[rank] = (eng, eng.Engine(), j, w, mark)
        k_step, k_drain = (f"toy.engine:{q}:{n}" for q, n in lines.items())
        self.assertEqual(keys, [k_step, k_drain])
        eng0, e0, j0, w0, mark0 = ranks["rank0"]
        self.assertEqual([e0.step(3) for _ in range(3)], [17, 17, 17])  # same values, new code

        r0 = w0.check(keys)
        self.assertEqual((r0["ok"], r0["counts"]), (True, {k_step: 3, k_drain: 0}))
        self.assertEqual(r0["ran"], {k_step: "ran during capture 0, ran live 3",
                                     k_drain: "ran during capture 0, ran live 0"})
        self.assertEqual([(f["check"], f["subject"], f["observed"]) for f in r0["findings"]],
                         [("zero_witness", k_drain, "0 calls since the swap")])
        r1 = ranks["rank1"][3].check(keys)
        self.assertEqual(([f["check"] for f in r1["findings"]], r1["counts"]),
                         (["zero_witness"] * 2, {k_step: 0, k_drain: 0}))
        self.assertEqual(F.decide(r1["findings"])["proceed"], False)   # blocks until accepted

        self.assertTrue(j0.undo_to(mark0)["ok"])
        self.assertIs(eng0.__rsibox_hit__, W.stub)
        self.assertEqual(w0.counts, {})
        e0.step(3)
        self.assertEqual(w0.counts, {})                   # the base code has no counter


class RunningFrameTest(unittest.TestCase):
    """Pain point: a loop entered before the swap keeps executing the old bytecode, and a
    suspended generator or coroutine (an event loop waiting in `await`) is on no thread's
    stack; both make "0 calls" mean "cannot run", not "not reached yet"."""

    def test_a_running_loop_and_a_suspended_generator_are_running_frame(self):
        """Input: the toy's run_loop running in a thread (entered before any swap), and a
        generator suspended after its first item; check with each one's old code. Expected:
        running_frame for both ("0 calls; a frame of the old code is still running"). After
        the loop stops and the generator is exhausted: zero_witness for both. Not: frames in
        C, other processes, or a generator suspended since before `gc.freeze()` (see
        `live_codes`)."""
        _, eng, _ = fresh("loop")
        e, stop = eng.Engine(delay=0.001), threading.Event()
        th = threading.Thread(target=eng.run_loop, args=(e, stop), daemon=True)
        th.start()

        def gen():
            yield 1
            yield 2
        g = gen()
        next(g)
        k_loop, k_gen = "toy.engine:run_loop:1", "test:gen:1"
        old = {k_loop: [eng.run_loop.__code__], k_gen: [gen.__code__]}
        w = W.Witness()
        try:
            wait_for(lambda: e.steps > 0)
            r = w.check([k_loop, k_gen], old)
        finally:
            stop.set()
            th.join()
        self.assertEqual([(f["check"], f["subject"], f["observed"]) for f in r["findings"]],
                         [("running_frame", k, "0 calls; a frame of the old code is still "
                           "running") for k in (k_loop, k_gen)])
        list(g)
        r = w.check([k_loop, k_gen], old)
        self.assertEqual([(f["check"], f["subject"]) for f in r["findings"]],
                         [("zero_witness", k_loop), ("zero_witness", k_gen)])

    def test_a_post_startup_generator_stays_visible_after_a_scan_refreezes_the_heap(self):
        """Input (#42): gc.freeze() after startup, then a serving-time generator `g = mod.gen();
        next(g)` suspended on its first yield, then a heap scan (`L.all_objects()`) whose exit
        calls `gc.freeze()` again. Expected: `W.live_codes(codes=[mod.gen.__code__])` still
        contains `mod.gen.__code__`, and `S.stage` on an edit of `gen` still reports
        `running_frame` until `g` is exhausted."""
        src = "def gen():\n    yield 1\n    yield 2\n"
        name, mod = module_from(self, src)
        mod.__file__ = name + ".py"
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        g = mod.gen()
        self.assertEqual(next(g), 1)
        self.assertIn(mod.gen.__code__, W.live_codes(codes=[mod.gen.__code__]))
        L.all_objects()
        self.assertIn(mod.gen.__code__, W.live_codes(codes=[mod.gen.__code__]))
        patched = src.replace("yield 2", "yield 3")
        r = S.stage(name, patched, base_source=src)
        self.assertEqual([f["check"] for f in r["findings"]], ["running_frame"])
        self.assertEqual(list(g), [2])
        self.assertNotIn(mod.gen.__code__, W.live_codes(codes=[mod.gen.__code__]))
        r = S.stage(name, patched, base_source=src)
        self.assertEqual(r["findings"], [])


class CaptureWitnessTest(unittest.TestCase):
    """Pain point: after a CUDA-graph recapture (TP2, Qwen3.5-35B-A3B MoE) the witness read
    13280 calls per rank and no zero_witness, yet every call was made while graphs were
    captured: a replay runs no Python, so the counter proved capture, not a run."""

    def test_capture_calls_are_counted_apart_from_live_calls(self):
        """Input: Engine.step witnessed; 3 calls while the engine says it captures (the
        `capturing` probe), 2 more inside `witness.capture()` (a recapture's warm-up runs).
        Expected: live 0, capture 5, and a zero_witness routed to R1 whose note asks for a
        device-side probe; "ran during capture 5, ran live 0". After 2 live calls: no
        finding, "ran during capture 5, ran live 2". Not: CUDA itself (vLLM asks torch
        whether the stream is capturing)."""
        _, eng, _ = fresh("capwit")
        capturing = [False]
        j, w = J.Journal(), W.Witness(capturing=lambda: capturing[0])
        (k,) = install(eng, ENG, [("Engine.step", at(ENG, "def step"))], j, w, "toy.engine")
        e = eng.Engine()
        capturing[0] = True
        for _ in range(3):
            e.step(1)
        capturing[0] = False
        with w.capture():
            e.step(1)
            e.step(1)
        r = w.check([k])
        self.assertEqual((r["counts"], r["capture_counts"], r["ran"]),
                         ({k: 0}, {k: 5}, {k: "ran during capture 5, ran live 0"}))
        (f,) = r["findings"]
        self.assertEqual((f["check"], f.get("route")), ("zero_witness", "R1"))
        self.assertIn("5 calls while graphs were captured", f["observed"])
        self.assertIn("device-side probe", f["note"])
        e.step(1)
        e.step(1)
        r = w.check([k])
        self.assertEqual((r["findings"], r["ran"]), ([], {k: "ran during capture 5, ran live 2"}))


class InjectTest(unittest.TestCase):
    """Pain point: a counter that breaks what it counts (a docstring that is no longer one)
    or counts the wrong thing (a lambda or generator expression has no statement body)."""

    def test_the_counter_keeps_the_docstring_and_counts_lambdas_and_genexprs(self):
        """Input: a def with a docstring, a lambda and a generator expression, each given
        the counter by `inject`, compiled and run (f twice, the lambda 3 times, the
        generator expression over 4 items). Expected: f's __doc__ is unchanged, counts are
        f 2, lambda 3, genexpr 4 (once per item produced, not per call), and the results are
        those of the code without the counter. Not: a def whose body is only a docstring."""
        src = ('def f(x):\n    "Doc of f."\n    return x + 1\n'
               "g = lambda x: x * 2\n"
               "def h(n):\n    return sum(i for i in range(n))\n")
        tree = ast.parse(src)
        f, g, h = tree.body
        W.inject(f, "f")
        W.inject(g.value, "lambda")
        W.inject(h.body[0].value.args[0], "genexpr")
        w = W.Witness()
        ns = {W.HIT: w.hook}
        exec(compile(ast.fix_missing_locations(tree), "<inject>", "exec"), ns)
        out = [ns["f"](1), ns["f"](2), *[ns["g"](i) for i in range(3)], ns["h"](4)]
        self.assertEqual(out, [2, 3, 0, 2, 4, 6])
        self.assertEqual(ns["f"].__doc__, "Doc of f.")
        self.assertEqual(w.counts, {"f": 2, "lambda": 3, "genexpr": 4})


class DynamoWitnessTest(unittest.TestCase):

    def test_the_witness_is_a_constant_for_dynamo(self):
        """Input: ops.scale witnessed while torch is loaded; then torch.compile (eager
        backend, fullgraph) of a caller of scale, called 3 times. Expected: the hook is
        marked with assume_constant_result, so Dynamo calls it once while tracing and the
        compiled calls never (1 call, not 3 or 4). Pain point: a recompile failed in Dynamo
        on the traced counter ("modifying buffers ... cudagraph inside the compiler"). Not:
        vLLM's compiler backend or CUDA graphs (a GPU receipt)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        _, _, ops = fresh("dyn")
        j, w = J.Journal(), W.Witness()
        (k,) = install(ops, OPS, [("scale", at(OPS, "def scale"))], j, w, "toy.ops")

        def caller(x):
            return ops.scale(x) + 1
        torch._dynamo.reset()
        c = torch.compile(caller, backend="eager", fullgraph=True)
        outs = [c(torch.ones(2)) for _ in range(3)]
        self.assertEqual(outs[-1].tolist(), [3.0, 3.0])
        self.assertTrue(getattr(w.hook, "_dynamo_marked_constant", False))
        self.assertEqual(w.counts, {k: 1})


class CodeRestartWitnessTest(unittest.TestCase):
    """Tests for code-restart execution witnesses (`instrument_module_functions`,
    `collect_restart_witness`, and `summarize_restart_witness`)."""

    def test_reached_code_restart_edits_are_proven_across_ranks(self):
        """Input: `ops.scale` and `eng.Engine.step` instrumented via
        `instrument_module_functions` on two simulated restart ranks; both targets
        execute during the step loop. Expected: `collect_restart_witness` records exact
        hit counts while preserving `functools.wraps` metadata, and
        `summarize_restart_witness` returns `proven: True` with `finding: None`."""
        _, eng_rank0, ops_rank0 = fresh("restart_rank0")
        _, eng_rank1, ops_rank1 = fresh("restart_rank1")

        targets_rank0 = {
            **W.instrument_module_functions(ops_rank0, ["scale"]),
            **W.instrument_module_functions(eng_rank0, ["Engine.step"]),
        }
        targets_rank1 = {
            **W.instrument_module_functions(ops_rank1, ["scale"]),
            **W.instrument_module_functions(eng_rank1, ["Engine.step"]),
        }
        self.assertEqual(ops_rank0.scale.__name__, "scale")
        self.assertEqual(eng_rank0.Engine.step.__name__, "step")

        engine0 = eng_rank0.Engine()
        engine1 = eng_rank1.Engine()
        for _ in range(3):
            engine0.step(2)
        for _ in range(2):
            engine1.step(2)

        counts_rank0 = W.collect_restart_witness(targets_rank0)
        counts_rank1 = W.collect_restart_witness(targets_rank1)
        self.assertEqual(counts_rank0, {"scale": 3, "Engine.step": 3})
        self.assertEqual(counts_rank1, {"scale": 2, "Engine.step": 2})

        summary = W.summarize_restart_witness(
            {"rank0": counts_rank0, "rank1": counts_rank1}
        )
        # per-rank minimum (audit #21: a sum proved a target that one rank never ran)
        self.assertEqual(
            summary,
            {
                "witness": {"scale": 2, "Engine.step": 2},
                "proven": True,
                "finding": None,
                "per_rank": {"rank0": counts_rank0, "rank1": counts_rank1},
            },
        )

    def test_unreached_code_restart_edits_report_zero_witness(self):
        """Input: `Engine.step` and `Engine.drain` instrumented on a restarted module,
        where `Engine.step` runs twice and `Engine.drain` never runs (plus an all-zero
        case). Expected: `summarize_restart_witness` marks `proven: False` and
        `finding: "zero_witness"` whenever any edited target has 0 calls or total calls
        across edited targets is 0."""
        _, eng_mod, _ = fresh("restart_unreached")
        targets = W.instrument_module_functions(eng_mod, ["Engine.step", "Engine.drain"])

        unreached_summary = W.summarize_restart_witness(W.collect_restart_witness(targets))
        self.assertEqual(
            unreached_summary,
            {
                "witness": {"Engine.step": 0, "Engine.drain": 0},
                "proven": False,
                "finding": "zero_witness",
            },
        )

        engine = eng_mod.Engine()
        engine.step(4)
        engine.step(4)
        partial_counts = W.collect_restart_witness(targets)
        self.assertEqual(partial_counts, {"Engine.step": 2, "Engine.drain": 0})

        partial_summary = W.summarize_restart_witness(partial_counts)
        self.assertEqual(
            partial_summary,
            {
                "witness": {"Engine.step": 2, "Engine.drain": 0},
                "proven": False,
                "finding": "zero_witness",
            },
        )

    def test_a_property_and_the_same_name_in_two_modules(self):
        """Input: (a) `C.val`, a @property, instrumented and read twice; (b) `scale`
        instrumented in two modules, called twice in the first and never in the second, then
        the other way round. Expected: (a) C().val is still 42 (a value, not a bound method)
        and counts 2; (b) the bare name "scale" collects 0 both times (the module that never
        ran it), "module:scale" each module's own count. Pain point: a property became a
        plain method, so `C().val` returned a function; and the second module's wrapper
        replaced the first under the bare name, so the bare name read one module's count and
        could prove an edit the other never ran. Not: an uninstrument API (there is none)."""
        prop_mod = types.ModuleType("kt_restart_prop")
        exec("class C:\n    @property\n    def val(self):\n        return 42\n",
             prop_mod.__dict__)
        targets = W.instrument_module_functions(prop_mod, ["C.val"])
        self.assertEqual((prop_mod.C().val, prop_mod.C().val), (42, 42))
        self.assertEqual(W.collect_restart_witness(targets), {"C.val": 2})
        for first in (0, 1):
            mods = [types.ModuleType(f"kt_restart_same{first}_{index}") for index in (0, 1)]
            for module_obj in mods:
                exec("def scale(x):\n    return x * 2\n", module_obj.__dict__)
                W.instrument_module_functions(module_obj, ["scale"])
            mods[first].scale(1)
            mods[first].scale(2)
            self.assertEqual(W.collect_restart_witness(["scale"]), {"scale": 0}, first)
            self.assertEqual(W.collect_restart_witness([f"{mods[first].__name__}:scale"]),
                             {f"{mods[first].__name__}:scale": 2})

    def test_uninstrument_module_functions_restores_all_targets_and_aliases(self):
        """Input: module with top-level fn, cross-module alias, staticmethod, classmethod,
        @property, and nested class method instrumented (and re-instrumented). Expected:
        partial uninstrument removes only named targets while keeping remaining hooks live; full
        uninstrument restores exact original callables/descriptors, rebinds cross-module aliases,
        removes `__rsibox_hit__` / `__rsibox_restart_counts__`, and cleans global restart
        registries."""
        mod = types.ModuleType("kt_uninst_mod")
        importer = types.ModuleType("kt_uninst_importer")
        sys.modules[mod.__name__] = mod
        sys.modules[importer.__name__] = importer
        self.addCleanup(lambda: (sys.modules.pop(mod.__name__, None),
                                 sys.modules.pop(importer.__name__, None)))
        exec(
            "def scale(x):\n    return x * 2\n\n"
            "class Outer:\n"
            "    @staticmethod\n    def sm(x):\n        return x + 1\n"
            "    @classmethod\n    def cm(cls, x):\n        return x + 2\n"
            "    @property\n    def prop(self):\n        return 99\n"
            "    class Inner:\n        def step(self, x):\n            return x + 3\n",
            mod.__dict__,
        )
        orig_scale = mod.scale
        orig_sm = inspect.getattr_static(mod.Outer, "sm")
        orig_cm = inspect.getattr_static(mod.Outer, "cm")
        orig_prop = inspect.getattr_static(mod.Outer, "prop")
        orig_inner_step = mod.Outer.Inner.step
        importer.scale_alias = orig_scale

        all_targets = ["scale", "Outer.sm", "Outer.cm", "Outer.prop", "Outer.Inner.step"]
        W.instrument_module_functions(mod, all_targets)
        # Re-instrumenting one target must not lose original provenance
        W.instrument_module_functions(mod, ["scale"])
        self.assertIsNot(mod.scale, orig_scale)
        self.assertIs(importer.scale_alias, mod.scale)
        self.assertEqual(importer.scale_alias(5), 10)
        self.assertEqual(mod.Outer.sm(1), 2)
        self.assertEqual(mod.Outer.cm(1), 3)
        self.assertEqual(mod.Outer().prop, 99)
        self.assertEqual(mod.Outer.Inner().step(4), 7)

        # Partial uninstrumentation restores `scale` + alias, keeps `Outer.sm` active
        removed_one = W.uninstrument_module_functions(mod, "scale")
        self.assertEqual(list(removed_one), ["scale"])
        self.assertIs(mod.scale, orig_scale)
        self.assertIs(importer.scale_alias, orig_scale)
        self.assertTrue(hasattr(mod, W.HIT))
        mod.Outer.sm(10)
        self.assertEqual(W.collect_restart_witness(mod), {"Outer.sm": 2, "Outer.cm": 1,
                                                          "Outer.prop": 1, "Outer.Inner.step": 1})

        # Uninstrument remaining targets -> restores all descriptors and cleans module/registries
        removed_rest = W.uninstrument_module_functions(mod)
        self.assertEqual(set(removed_rest), {"Outer.sm", "Outer.cm", "Outer.prop", "Outer.Inner.step"})
        self.assertIs(inspect.getattr_static(mod.Outer, "sm"), orig_sm)
        self.assertIs(inspect.getattr_static(mod.Outer, "cm"), orig_cm)
        self.assertIs(inspect.getattr_static(mod.Outer, "prop"), orig_prop)
        self.assertIs(mod.Outer.Inner.step, orig_inner_step)
        self.assertFalse(hasattr(mod, W.HIT))
        self.assertFalse(hasattr(mod, "__rsibox_restart_counts__"))
        self.assertNotIn(f"{mod.__name__}:scale", W._RESTART_INSTRUMENTED_TARGETS)
        self.assertNotIn(f"{mod.__name__}:Outer.Inner.step", W._RESTART_INSTRUMENTED_TARGETS)
        # Idempotent second call is a clean no-op
        self.assertEqual(W.uninstrument_module_functions(mod), {})

    def test_uninstrument_preserves_preexisting_hit_and_sibling_module_registry(self):
        """Input: two modules both instrument `dup`, and module A had a prior `__rsibox_hit__`.
        Expected: uninstrumenting A restores A's prior `__rsibox_hit__` and keeps B's wrapper
        registered under `dup`."""
        mod_a = types.ModuleType("kt_uninst_a")
        mod_b = types.ModuleType("kt_uninst_b")
        exec("def dup(x):\n    return x\n", mod_a.__dict__)
        exec("def dup(x):\n    return x + 1\n", mod_b.__dict__)
        prior_hit = lambda k: None
        setattr(mod_a, W.HIT, prior_hit)
        W.instrument_module_functions(mod_a, ["dup"])
        wb = W.instrument_module_functions(mod_b, ["dup"])["dup"]
        self.addCleanup(W.uninstrument_module_functions, mod_b)

        W.uninstrument_module_functions(mod_a)
        self.assertIs(getattr(mod_a, W.HIT), prior_hit)
        mod_b.dup(1)
        self.assertEqual(W.collect_restart_witness(["dup"]), {"dup": 1})
        self.assertIs(W._RESTART_INSTRUMENTED_TARGETS.get("dup"), wb)


if __name__ == "__main__":
    unittest.main()
