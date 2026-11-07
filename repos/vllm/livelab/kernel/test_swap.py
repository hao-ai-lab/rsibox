"""Golden tests of the swap (R0: a function-body swap) on the toy engine (CPU, standard
library only).

Run from the repo root:  python3 -m unittest discover -s repos/vllm/livelab/kernel -t repos/vllm/livelab -v
(or python3 -m pytest -q repos/vllm/livelab/kernel)

Toy arithmetic: see test_locate.py (step(3) = 17 at base).
"""
import gc
import threading
import unittest

from . import findings as F
from . import journal as J
from . import locate as L
from . import swap as S
from . import witness as W
from .test_locate import ENG, OPS, at, edit, fresh
from .test_witness import wait_for


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


if __name__ == "__main__":
    unittest.main()
