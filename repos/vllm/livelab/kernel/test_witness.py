"""Golden tests of the witness on the toy engine (CPU, standard library only; one test
imports torch on CPU and is skipped without it).

Run from the repo root:  python3 -m unittest discover -s repos/vllm/livelab/kernel -t repos/vllm/livelab -v
(or python3 -m pytest -q repos/vllm/livelab/kernel)

These tests use the witness without the swap, so `install` below does by hand what the swap
does for a witness: add the counter call to the new body (`witness.inject`),
compile the module text (`inject` leaves the
new name's location to `ast.fix_missing_locations`, as the swap's compile does), put the code on every live holder and bind `__fastrsi_hit__` to the
participant's hook, each through the journal. Toy arithmetic: see test_locate.py.
"""
import ast
import threading
import time
import unittest

from . import findings as F
from . import journal as J
from . import locate as L
from . import witness as W
from .test_locate import ENG, OPS, at, edit, fresh


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
        self.assertIs(eng0.__fastrsi_hit__, W.stub)
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


if __name__ == "__main__":
    unittest.main()
