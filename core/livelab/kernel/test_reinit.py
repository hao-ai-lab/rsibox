"""Golden tests of constructor field re-initialization (reinit.py) through the participant
(`coordinator.LocalParticipant`), on a scheduler-shaped toy (CPU, standard library only).

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
"""
import gc
import itertools
import unittest

from . import coordinator as C
from . import reinit as RI
from .test_detectors import temp_modules

# vLLM #62872's shape (vllm/v1/core/sched/scheduler.py): the constructor turns a set of
# request ids into a dict id -> message, and the two methods that use it change with it.
SCHED = '''class Scheduler:
    def __init__(self, max_num_seqs):
        self.max_num_seqs = max_num_seqs
        self.grammar_compile_error_reqs: set[str] = set()
        self.encoder_cache_mismatch_reqs: set[str] = set()

    def promote(self, req_id, grammar):
        if isinstance(grammar, Exception):
            self.grammar_compile_error_reqs.add(req_id)
            return False
        return True

    def update_from_output(self):
        error_req_ids = set(self.grammar_compile_error_reqs)
        self.grammar_compile_error_reqs.clear()
        return sorted(error_req_ids)


class EngineCore:
    def __init__(self):
        self.scheduler = Scheduler(8)
'''
SCHED_NEW = (SCHED
             .replace("self.grammar_compile_error_reqs: set[str] = set()",
                      "self.grammar_compile_error_reqs: dict[str, str] = {}")
             .replace("            self.grammar_compile_error_reqs.add(req_id)",
                      "            self.grammar_compile_error_reqs[req_id] = (\n"
                      "                f\"Invalid structured outputs: {grammar}\")")
             .replace("        error_req_ids = set(self.grammar_compile_error_reqs)\n"
                      "        self.grammar_compile_error_reqs.clear()\n"
                      "        return sorted(error_req_ids)",
                      "        request_errors = self.grammar_compile_error_reqs\n"
                      "        self.grammar_compile_error_reqs = {}\n"
                      "        return sorted(request_errors.items())"))
_ids = itertools.count()


def setup(test, text=SCHED):
    name = f"reinit_sched_{next(_ids)}"
    (module,) = temp_modules(test, {name: text})
    return name, module, C.LocalParticipant("core")


def plan(name, txn="t", source=SCHED_NEW, **kw):
    return {"id": txn, "modules": {name: {"source": source}}, **kw}


def checks(r):
    return [(f["check"], f["accepted"]) for f in r["findings"]]


class ReinitTest(unittest.TestCase):

    def test_62872_set_to_dict_reinitializes_the_live_scheduler_and_undoes(self):
        """Input: an engine core built at start-up (its Scheduler holds an empty set), then the
        #62872 edit (set -> dict in __init__, promote/update_from_output use the dict).
        Expected: prepare votes yes with one advisory constructor_reinit (1 live instance);
        after commit the live field is an empty dict and a grammar error is reported with its
        message, the constructor's witness counts 1; undo puts the original set object back
        and the base methods run again (`.add` works). Before this rule the plan voted no
        (constructor_edit) and went to a forked restart (16.6 s eager, 51.2 s graph); a plain
        accept would have left the live set, and the new `[...] =` raises TypeError on it.
        Not: other processes' instances (each participant scans its own heap)."""
        name, module, p = setup(self)
        core = module.EngineCore()
        before = core.scheduler.grammar_compile_error_reqs
        r = p.prepare(plan(name))
        self.assertTrue(r["ok"], r)
        self.assertEqual(checks(r), [("constructor_reinit", True)])
        self.assertEqual(r["findings"][0]["reinit"]["instances"], 1)
        self.assertTrue(p.validate("t")["ok"])
        c = p.commit("t")
        self.assertTrue(c["ok"], c)
        self.assertEqual(list(c["reinit"].values())[0]["instances"], 1)
        s = core.scheduler
        self.assertEqual((type(s.grammar_compile_error_reqs), s.grammar_compile_error_reqs),
                         (dict, {}))
        self.assertFalse(s.promote("r1", ValueError("bad grammar")))
        self.assertEqual(s.update_from_output(),
                         [("r1", "Invalid structured outputs: bad grammar")])
        w = p.witness_check("t")
        init_key = next(k for k in w["counts"] if "__init__" in k)
        self.assertEqual(w["counts"][init_key], 1)
        self.assertTrue(p.undo("t")["ok"])
        self.assertIs(s.grammar_compile_error_reqs, before)
        self.assertFalse(s.promote("r2", ValueError("x")))
        self.assertEqual((s.update_from_output(), type(before)), (["r2"], set))
        ab = p.at_base()
        self.assertEqual((ab["ok"], ab["findings"], ab["journal_len"]), (True, [], 0))

    def test_a_live_field_that_holds_state_still_blocks(self):
        """Input: the same edit while the live scheduler's set holds a request id (a grammar
        error not yet reported). Expected: constructor_edit blocks and says the field holds
        state; nothing changes. Replacing it with {} would drop the request."""
        name, module, p = setup(self)
        core = module.EngineCore()
        core.scheduler.grammar_compile_error_reqs.add("r9")
        r = p.prepare(plan(name))
        self.assertFalse(r["ok"])
        self.assertEqual(checks(r), [("constructor_edit", False)])
        self.assertIn("holds state", r["findings"][0]["observed"])
        self.assertEqual(core.scheduler.grammar_compile_error_reqs, {"r9"})

    def test_state_taken_between_prepare_and_the_gap_votes_no_at_validate(self):
        """Input: prepare passes, then the live set takes a request id before the gap.
        Expected: validate votes no with a blocking constructor_edit; abort leaves the set and
        the base code. Proves the gap re-checks, not only prepare."""
        name, module, p = setup(self)
        core = module.EngineCore()
        self.assertTrue(p.prepare(plan(name))["ok"])
        core.scheduler.promote("r3", ValueError("x"))
        v = p.validate("t")
        self.assertFalse(v["ok"])
        self.assertEqual([f["check"] for f in v["findings"]], ["constructor_edit"])
        p.abort("t")
        self.assertEqual(core.scheduler.update_from_output(), ["r3"])

    def test_an_instance_built_after_prepare_is_covered_and_a_frozen_heap_is_seen(self):
        """Input: one Scheduler built, then gc.freeze() (vLLM freezes its start-up heap),
        prepare, then a second Scheduler built before the gap. Expected: prepare finds the
        frozen one (1), validate's scan finds both, commit re-initializes 2."""
        name, module, p = setup(self)
        a = module.Scheduler(1)
        gc.collect()
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        r = p.prepare(plan(name))
        self.assertEqual(r["findings"][0]["reinit"]["instances"], 1)
        b = module.Scheduler(2)
        self.assertTrue(p.validate("t")["ok"])
        c = p.commit("t")
        self.assertEqual(list(c["reinit"].values())[0]["instances"], 2)
        self.assertEqual([type(x.grammar_compile_error_reqs) for x in (a, b)], [dict, dict])
        self.assertTrue(p.undo("t")["ok"])
        self.assertEqual([type(x.grammar_compile_error_reqs) for x in (a, b)], [set, set])

    def test_undo_after_the_new_code_left_state_reports_it(self):
        """Input: commit, then the new code records an error and the step never drains it;
        undo. Expected: the field is restored to the base set anyway, and undo reports
        undo_incomplete naming the field, so the transaction reads partial, not at base."""
        name, module, p = setup(self)
        core = module.EngineCore()
        p.prepare(plan(name))
        p.validate("t")
        self.assertTrue(p.commit("t")["ok"])
        core.scheduler.promote("r4", ValueError("x"))
        u = p.undo("t")
        self.assertFalse(u["ok"])
        self.assertIn("grammar_compile_error_reqs", u["findings"][0]["observed"])
        self.assertEqual(type(core.scheduler.grammar_compile_error_reqs), set)
        self.assertEqual(p.status("t")["state"], "partial")

    def test_a_subclass_constructor_is_covered_unless_it_touches_the_field(self):
        """Input: vLLM's AsyncScheduler shape (its own __init__: super().__init__, then a field
        of its own), live; then a subclass whose __init__ sets the field again. Expected: the
        first is re-initialized (async scheduling is vLLM's default, so the live scheduler is
        this subclass); the second blocks naming the subclass."""
        sub = ("\n\nclass AsyncScheduler(Scheduler):\n"
               "    def __init__(self, *args, **kwargs):\n"
               "        super().__init__(*args, **kwargs)\n"
               "        self.decode_stagger = 1\n")
        name, module, p = setup(self, SCHED + sub)
        live = module.AsyncScheduler(8)
        r = p.prepare(plan(name, source=SCHED_NEW + sub))
        self.assertEqual((r["ok"], checks(r)), (True, [("constructor_reinit", True)]))
        p.validate("t")
        self.assertTrue(p.commit("t")["ok"])
        self.assertEqual(type(live.grammar_compile_error_reqs), dict)
        self.assertTrue(p.undo("t")["ok"])
        bad = sub.replace("self.decode_stagger = 1", "self.grammar_compile_error_reqs = set()")
        name2, module2, p2 = setup(self, SCHED + bad)
        live2 = module2.AsyncScheduler(8)
        r2 = p2.prepare(plan(name2, source=SCHED_NEW + bad))
        self.assertEqual((r2["ok"], checks(r2)), (False, [("constructor_edit", False)]))
        self.assertIn("AsyncScheduler.__init__ sets or reads", r2["findings"][0]["observed"])
        self.assertEqual(type(live2.grammar_compile_error_reqs), set)

    def test_subclass_without_super_init_refuses_added_field(self):
        """Input: a subclass defining its own __init__ without calling super().__init__(), and
        an edit to Scheduler.__init__ that adds a new pure field `self.extra_cache = []`.
        Expected: prepare blocks as constructor_edit naming the missing super().__init__ call,
        so live instances of the subclass never receive an attribute a fresh construction would
        not have (c3_fresh_equivalence)."""
        sub = ("\n\nclass StandaloneScheduler(Scheduler):\n"
               "    def __init__(self):\n"
               "        self.own_state = 1\n")
        added_field = SCHED.replace(
            "        self.encoder_cache_mismatch_reqs: set[str] = set()",
            "        self.encoder_cache_mismatch_reqs: set[str] = set()\n"
            "        self.extra_cache: list = []")
        name, module, p = setup(self, SCHED + sub)
        live = module.StandaloneScheduler()
        r = p.prepare(plan(name, source=added_field + sub))
        self.assertEqual((r["ok"], checks(r)), (False, [("constructor_edit", False)]))
        self.assertIn("StandaloneScheduler.__init__ does not unconditionally call super().__init__",
                      r["findings"][0]["observed"])
        self.assertFalse(hasattr(live, "extra_cache"))

    def test_no_live_instance_needs_only_the_swap(self):
        """Input: the edit with no Scheduler alive. Expected: advisory constructor_reinit with
        0 instances; commit installs; a Scheduler built after it gets a dict."""
        name, module, p = setup(self)
        r = p.prepare(plan(name))
        self.assertEqual((r["ok"], r["findings"][0]["reinit"]["instances"]), (True, 0))
        p.validate("t")
        self.assertTrue(p.commit("t")["ok"])
        self.assertEqual(type(module.Scheduler(1).grammar_compile_error_reqs), dict)
        self.assertTrue(p.undo("t")["ok"])


class PlanTest(unittest.TestCase):
    """reinit.plan on the two ASTs: which constructor edits are eligible."""

    def _plan(self, base_body, new_body, args="self, n"):
        import ast
        b = ast.parse(f"def __init__({args}):\n" + base_body).body[0]
        n = ast.parse(f"def __init__({args}):\n" + new_body).body[0]
        return RI.plan(b, n)

    def test_eligible_and_refused_shapes(self):
        """Each row: (base body, edit body) -> eligible or the reason. The refusals are the
        #61323 shape (a call statement added), a value from a parameter, a removed statement,
        a field read by another statement, and a module name in the value."""
        ok = self._plan("    self.a = set()\n    self.n = n\n",
                        "    self.a = {}\n    self.n = n\n    self.b: list = []\n")
        self.assertEqual(ok, {"fields": [{"field": "a", "old": "set()", "new": "{}"},
                                         {"field": "b", "old": None, "new": "[]"}]})
        rows = [
            ("    self.w = n\n", "    self.w = n\n    set_weight_attrs(self.w, {})\n"),
            ("    self.a = set()\n", "    self.a = set(n)\n"),
            ("    self.a = set()\n    self.b = 1\n", "    self.a = {}\n"),
            ("    self.a = set()\n    self.c = len(self.a)\n",
             "    self.a = {}\n    self.c = len(self.a)\n"),
            ("    self.a = 0\n", "    self.a = DEFAULT\n"),
        ]
        for base, new in rows:
            self.assertIn("why", self._plan(base, new), (base, new))
        import ast
        sig = RI.plan(ast.parse("def __init__(self):\n    self.a = 0\n").body[0],
                      ast.parse("def __init__(self, m):\n    self.a = 1\n").body[0])
        self.assertIn("signature", sig["why"])


if __name__ == "__main__":
    unittest.main()
