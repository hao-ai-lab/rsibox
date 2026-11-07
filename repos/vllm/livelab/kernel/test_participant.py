"""Golden tests of the participant side of a transaction (`coordinator.LocalParticipant`) on
the toy engine (CPU, standard library only). Here the test drives the two phases by hand, as
a host does; the host itself (`Coordinator`) is the next kernel PR.

Run from the repo root:  python3 -m unittest discover -s repos/vllm/livelab/kernel -t repos/vllm/livelab -v
(or python3 -m pytest -q repos/vllm/livelab/kernel)
"""
import threading
import unittest

from . import coordinator as C
from . import locate as L
from . import swap as S
from .test_detectors import ESTIMATOR, ESTIMATOR_NEW, temp_modules
from .test_locate import ENG, OPS, at, edit, fresh
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
        journals. Not: the host's all-or-nothing rules (next PR), real RPC."""
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
        plan, with a live estimator. Expected: the participant votes no with one blocking
        new_fields finding, route T1, tagged with the participant's name; nothing is staged,
        and the live step still returns 256. Not: a rebuild or restart that resolves it."""
        (m,) = temp_modules(self, {"kt_part_est": ESTIMATOR})
        est = m.Runner().speculator.acceptance_estimator
        p = C.LocalParticipant("workers")
        r = p.prepare({"id": "nf", "modules": {"kt_part_est": {"source": ESTIMATOR_NEW}}})
        self.assertEqual((r["ok"], [(f["check"], f["route"], f["accepted"], f["participant"])
                                    for f in r["findings"]]),
                         (False, [("new_fields", "T1", False, "workers")]))
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
        self.assertIn(5, ops.make_shift.__code__.co_consts)
        self.assertEqual((ops.scale(3), ops.make_shift(10)(1)), (6, 11))
        self.assertTrue(p.undo("e2")["ok"])                 # make_shift back, scale still v1
        self.assertNotIn(5, ops.make_shift.__code__.co_consts)
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


if __name__ == "__main__":
    unittest.main()
