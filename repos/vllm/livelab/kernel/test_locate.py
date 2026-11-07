"""Golden tests of findings and locate on the toy engine (CPU, standard library only).

Run from the repo root:  python3 -m unittest discover -s repos/vllm/livelab/kernel -t repos/vllm/livelab -v
(or python3 -m pytest -q repos/vllm/livelab/kernel)

Toy arithmetic: Engine.step(x) = shift(activate(scale(x))) with scale = x*2, activate = x+1,
shift = x+10, so step(3) = 17 at base. Every test loads its own copy of the toy package
(`toy.load_copy`), so tests never share module objects.

The helpers here (`fresh`, `at`, `edit`) are shared with the later kernel tests.
"""
import gc
import itertools
import os
import sys
import unittest

from . import findings as F
from . import locate as L
from . import toy

OPS, ENG = toy.source("ops"), toy.source("engine")
_n = itertools.count()


def fresh(tag, path=None):
    """A new copy of the toy package: (package name, engine module, ops module)."""
    name = f"kt{next(_n)}_{tag}"
    toy.load_copy(name, path)
    return name, sys.modules[name + ".engine"], sys.modules[name + ".ops"]


def at(src, needle):
    """1-based line of the first line containing `needle` (the target's first line)."""
    return next(i for i, line in enumerate(src.splitlines(), 1) if needle in line)


def edit(src, *pairs):
    for old, new in pairs:
        assert old in src, old
        src = src.replace(old, new, 1)
    return src


class LocateTest(unittest.TestCase):

    def test_locate_finds_every_holder_of_a_running_function(self):
        """Proves: on a frozen heap (engines freeze it after startup), locate finds the one live function for each target and names every holder: the module dict and a `from x import f` copy, the decorator wrapper's closure cell, the class dict and a bound method saved as a callback. Not: closures made at run time (no qualname target here)."""
        name, eng, ops = fresh("holders")
        e = eng.Engine()
        gc.freeze()                       # engines freeze the heap after startup (vLLM does)
        self.addCleanup(gc.unfreeze)
        self.assertEqual(e.step(3), 17)
        found = {q: L.locate(m, q, line)
                 for m, q, line in [(name + ".ops", "scale", at(OPS, "def scale")),
                                    (name + ".ops", "activate", at(OPS, "@traced")),
                                    (name + ".engine", "Engine.on_token",
                                     at(ENG, "def on_token"))]}
        self.assertEqual({q: (r["ok"], len(r["functions"])) for q, r in found.items()},
                         {"scale": (True, 1), "activate": (True, 1),
                          "Engine.on_token": (True, 1)})
        self.assertIs(found["scale"]["functions"][0], ops.scale)
        holders = {q: {(h["kind"], h["where"]) for h in r["holders"]} for q, r in found.items()}
        self.assertEqual(holders["scale"], {("module_dict", f"{name}.ops.scale"),
                                            ("module_dict", f"{name}.engine.scale")})
        self.assertIn(("closure_cell", "traced.<locals>.wrapper"), holders["activate"])
        self.assertIn(("class_dict", "Engine.on_token"), holders["Engine.on_token"])
        self.assertIn(("bound_method", "Engine().on_token"), holders["Engine.on_token"])

    def test_a_target_that_does_not_resolve_is_a_finding_not_an_exception(self):
        """Proves: a qualname with no live function and a module that does not import each come back as a not-ok result with one finding keyed by the target, never an exception. Not: what a caller does with it."""
        name, _, _ = fresh("missing")
        r = L.locate(name + ".engine", "Engine.nope", 7)
        self.assertEqual((r["ok"], [f["check"] for f in r["findings"]], r["functions"]),
                         (False, ["no_live_function"], []))
        self.assertEqual(r["findings"][0]["subject"], f"{name}.engine:Engine.nope:7")
        r = L.locate(name + ".no_such_module", "f", 1)
        self.assertEqual((r["ok"], [f["check"] for f in r["findings"]]),
                         (False, ["patch_invalid"]))

    def test_code_identity_and_hash(self):
        """Proves: identity is file/qualname/line and the hash is content only (copies equal, moves equal, edits differ). Not: equal behavior."""
        n0, e0, _ = fresh("id0")
        n1, e1, _ = fresh("id1")
        line = at(ENG, "def step")
        c0 = L.locate(n0 + ".engine", "Engine.step", line)["codes"][0]
        c1 = L.locate(n1 + ".engine", "Engine.step", line)["codes"][0]
        self.assertIsNot(e0.Engine.step.__code__, e1.Engine.step.__code__)
        self.assertEqual((os.path.basename(c0["file"]), c0["qualname"], c0["firstlineno"]),
                         ("engine.py", "Engine.step", 22))
        self.assertEqual(c0["hash"], c1["hash"])

        def code(src, q):
            return L.code_tree(compile(src, "engine.py", "exec"))[0][
                (q, at(src, "def " + q.split(".")[-1]))][0]
        moved = code("\n\n\n" + ENG, "Engine.step")
        self.assertEqual((moved.co_firstlineno, L.content_hash(moved)), (25, c0["hash"]))
        edited = code(edit(ENG, ("self.steps += 1", "self.steps += 2")), "Engine.step")
        self.assertNotEqual(L.content_hash(edited), c0["hash"])


class AcceptRecordTest(unittest.TestCase):
    """Accept by name with a record. Pain point: a standing accept that silently keeps
    covering a check after the checked code changed (DSv4.1 bound_shape_mismatch was accepted
    on every soft restart; the same check must still catch a new stale buffer)."""

    def _f(self, code_hash):
        return {**F.finding("zero_witness", "o", "e"), "participant": "worker",
                "code_hash": code_hash}

    def test_standing_accept_expires_when_code_changes(self):
        rec = {"check": "zero_witness", "scope": "standing", "by": "agent-e5",
               "reason": "false positive on processed weights", "participant": "worker",
               "code_hash": "h1"}
        d = F.decide([self._f("h1")], [rec])
        self.assertTrue(d["proceed"])
        self.assertEqual(d["accepts"], [rec])      # who and why go to the transaction record
        d = F.decide([self._f("h2")], [rec])       # checked code changed: no longer covered
        self.assertFalse(d["proceed"])
        self.assertEqual(d["blocking"], ["zero_witness"])

    def test_accept_record_with_a_subject_covers_only_that_subject(self):
        """Proves: a record naming a subject accepts that subject's finding of the check and leaves the same check on another subject blocking. Not: subject patterns."""
        fs = [{**F.finding("running_frame", "o", "e", subject=k), "participant": "core"}
              for k in ("toy.engine:run_loop:80", "toy.engine:Engine.step:22")]
        rec = {"check": "running_frame", "subject": "toy.engine:run_loop:80", "by": "agent",
               "reason": "the busy loop restarts on the next request"}
        d = F.decide(fs, [rec])
        self.assertEqual((d["proceed"], [f["accepted"] for f in d["findings"]], d["accepts"]),
                         (False, [True, False], [rec]))
        self.assertTrue(F.decide(fs[:1], [rec])["proceed"])

    def test_name_and_txn_record_still_accept(self):
        self.assertTrue(F.decide([self._f("h2")], ["zero_witness"])["proceed"])
        rec = {"check": "zero_witness", "by": "agent", "reason": "rank-specific"}
        d = F.decide([self._f("h2")], [rec])
        self.assertTrue(d["proceed"])
        self.assertEqual(d["accepts"], [rec])
