"""Golden tests of findings and locate on the toy engine (CPU, standard library only).

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
(or python3 -m pytest -q core/livelab/kernel)

Toy arithmetic: Engine.step(x) = shift(activate(scale(x))) with scale = x*2, activate = x+1,
shift = x+10, so step(3) = 17 at base. Every test loads its own copy of the toy package
(`toy.load_copy`), so tests never share module objects.

The helpers here (`fresh`, `at`, `edit`, `loads_constant`) are shared with the later kernel tests.
"""
import dis
import gc
import itertools
import os
import sys
import threading
import types
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


def loads_constant(code, value):
    """Does `code` load the constant `value`? Read from the instructions, not `co_consts`:
    Python 3.14 loads small ints with LOAD_SMALL_INT, which keeps them out of co_consts."""
    return any(instr.opname in ("LOAD_CONST", "LOAD_SMALL_INT") and instr.argval == value
               and type(instr.argval) is type(value) for instr in dis.get_instructions(code))


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


def module_from(test, text):
    """A module made from `text` (no file), registered in sys.modules for the test."""
    name = f"kt_mod{next(_n)}"
    mod = types.ModuleType(name)
    exec(compile(text, name + ".py", "exec"), mod.__dict__)
    sys.modules[name] = mod
    test.addCleanup(sys.modules.pop, name, None)
    return name, mod


class _Proxy:
    """An RPC/lazy proxy shape: slots, and a __getattr__ that raises for any missing name
    (`getattr(p, "__dict__", None)` raises RuntimeError, not AttributeError)."""
    __slots__ = ("fn",)

    def __getattr__(self, name):
        raise RuntimeError(f"proxy has no {name}")


class LocateEdgeCaseTest(unittest.TestCase):
    """Content hashes of edits that keep the bytecode, error replies, lru_cache wrappers, and
    one scan at a time."""

    def test_content_hash_tells_apart_edits_that_keep_the_bytecode(self):
        """Input: (a) a statement moved out of a try block, which on 3.12 keeps co_code and
        changes only the exception table (3.11 and 3.14 also change co_code); (b) `1 if x else 23` against `12 if x else 3`,
        same bytecode, constants (None, 1, 23) and (None, 12, 3). Expected: different
        hashes; the same code moved down 3 lines keeps its hash. Pain point: a standing
        accept or the base_mismatch check could not tell the two versions apart. Not:
        hashes of code compiled by another Python version."""
        def code(src):
            ns = {}
            exec(compile(src, "m.py", "exec"), ns)
            return ns["f"].__code__
        in_try = code("def f(a, b):\n    try:\n        a()\n        b()\n"
                      "    except Exception:\n        return None\n    return 1\n")
        after = code("def f(a, b):\n    try:\n        a()\n    except Exception:\n"
                     "        return None\n    b()\n    return 1\n")
        if sys.version_info[:2] in ((3, 12), (3, 13)):         # 3.11 and 3.14 differ in co_code
            self.assertEqual(in_try.co_code, after.co_code)   # only the exception table differs
        self.assertNotEqual(L.content_hash(in_try), L.content_hash(after))
        self.assertNotEqual(L.content_hash(code("def f(x): return 1 if x else 23\n")),
                            L.content_hash(code("def f(x): return 12 if x else 3\n")))
        moved = code("\n\n\ndef f(a, b):\n    try:\n        a()\n        b()\n"
                     "    except Exception:\n        return None\n    return 1\n")
        self.assertEqual(L.content_hash(moved), L.content_hash(in_try))

    def test_locate_never_raises_and_keeps_its_reply_shape(self):
        """Input: a module that does not import; a first line that is not a number; a live
        function also held by a slotted proxy whose __getattr__ raises. Expected: the first
        two are not-ok replies with codes/functions/holders present (and empty); the third
        is ok and lists the proxy as an "other" holder (a slot, no __dict__). Pain point:
        locate raised out of a call documented "never raises". Not: every proxy type."""
        r = L.locate("kt_no_such_module", "f", 1)
        self.assertEqual((r["ok"], r["codes"], r["functions"], r["holders"]),
                         (False, [], [], []))
        name, mod = module_from(self, "def target(x):\n    return x\n")
        r = L.locate(name, "target", "first")
        self.assertEqual((r["ok"], r["codes"], r["functions"], r["holders"]),
                         (False, [], [], []))
        p = _Proxy()
        p.fn = mod.target
        r = L.locate(name, "target", 1)
        self.assertTrue(r["ok"], r.get("error"))
        self.assertIn({"kind": "other", "where": "_Proxy"}, r["holders"])   # slots: no dict

    def test_an_lru_cache_wrapper_is_found_from_the_module_and_listed_once(self):
        """Input: `@functools.lru_cache def cached` and a function wrapped by a callable
        class through functools.update_wrapper. Expected: module_functions finds both
        originals (no heap scan; the fingerprint and at_base see them), and locate lists the
        lru_cache wrapper as one holder, not twice. Pain point: these targets fell back to a
        heap scan and were missing from the fingerprint. Not: wrappers that keep the
        function under another attribute name."""
        name, mod = module_from(self, (
            "import functools\n\n"
            "class Counted:\n"
            "    def __init__(self, fn):\n"
            "        functools.update_wrapper(self, fn)\n\n"
            "    def __call__(self, *a):\n"
            "        return self.__wrapped__(*a)\n\n"
            "@functools.lru_cache\n"
            "def cached(x):\n"
            "    return x\n\n"
            "@Counted\n"
            "def counted(x):\n"
            "    return x\n"))
        found = {f.__qualname__ for f in L.module_functions(mod)}
        self.assertLessEqual({"cached", "counted"}, found)
        r = L.locate(name, "cached", 10)
        self.assertEqual(r["holders"], [{"kind": "attribute",
                                         "where": "_lru_cache_wrapper().__wrapped__"}])

    def test_one_scan_at_a_time_and_no_collection_while_unfrozen(self):
        """Input: (a) an unfrozen heap with nested `unfrozen()` calls on the same thread;
        (b) a frozen heap with thread A inside `unfrozen()`, thread B entering it, then A
        leaving. Expected: gc is disabled on outermost entry even when the heap was not
        frozen and restored on outermost exit; thread B blocks until thread A exits; B's
        scan never sees the heap frozen again under it (freeze count 0 inside B after A
        left), gc collection is off inside the window and back on after, and the heap is
        frozen again at the end."""
        gc.unfreeze()
        self.assertEqual((gc.get_freeze_count(), gc.isenabled()), (0, True))
        with L.unfrozen():
            self.assertFalse(gc.isenabled())
            with L.unfrozen():
                self.assertFalse(gc.isenabled())
            self.assertFalse(gc.isenabled())
        self.assertTrue(gc.isenabled())

        gc.freeze()
        self.addCleanup(gc.unfreeze)
        b_started, b_entered, release_b, seen = (
            threading.Event(), threading.Event(), threading.Event(), []
        )

        def worker_b():
            b_started.set()
            with L.unfrozen():
                b_entered.set()
                release_b.wait(2)
                seen.append((gc.get_freeze_count(), gc.isenabled()))

        with L.unfrozen():
            with L.unfrozen():
                self.assertEqual((gc.get_freeze_count(), gc.isenabled()), (0, False))
            self.assertEqual((gc.get_freeze_count(), gc.isenabled()), (0, False))
            worker_thread = threading.Thread(target=worker_b)
            worker_thread.start()
            self.assertTrue(b_started.wait(2))
            self.assertFalse(b_entered.wait(0.1))  # B blocks on the lock while A holds unfrozen
        self.assertTrue(b_entered.wait(2))
        release_b.set()
        worker_thread.join(5)
        self.assertEqual(seen, [(0, False)])
        self.assertTrue(gc.isenabled())
        self.assertGreater(gc.get_freeze_count(), 0)


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

    def test_unknown_scope_and_none_standing_keys_do_not_accept(self):
        """Input (#43): accept records with an unknown scope ("standng", "session", None) or a
        standing scope whose participant or code_hash is None (or missing), checked against a
        finding with and without those keys (validate's running_frame carries no code_hash).
        Expected: none of the malformed records match; every finding remains blocking."""
        finding_with_hash = self._f("h1")
        finding_without_hash = {**F.finding("running_frame", "o", "e"), "participant": "worker"}
        finding_bare = F.finding("running_frame", "o", "e")
        for bad_record in (
            {"check": "zero_witness", "scope": "standng", "by": "a", "reason": "r"},
            {"check": "zero_witness", "scope": "session", "by": "a", "reason": "r"},
            {"check": "zero_witness", "scope": None, "by": "a", "reason": "r"},
            {"check": "running_frame", "scope": "standing", "by": "a", "reason": "r",
             "participant": "worker", "code_hash": None},
            {"check": "running_frame", "scope": "standing", "by": "a", "reason": "r",
             "participant": None, "code_hash": None},
            {"check": "running_frame", "scope": "standing", "by": "a", "reason": "r"},
        ):
            for target_finding in (finding_with_hash, finding_without_hash, finding_bare):
                if target_finding["check"] == bad_record["check"]:
                    decision = F.decide([target_finding], [bad_record])
                    self.assertFalse(decision["proceed"], (bad_record, target_finding))
                    self.assertEqual(decision["accepts"], [])
