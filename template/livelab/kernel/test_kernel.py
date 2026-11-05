"""Golden tests of the transaction kernel on the toy engine (CPU, standard library only).

Run from template/livelab:  python3 -m unittest kernel.test_kernel -v

Toy arithmetic: Engine.step(x) = shift(activate(scale(x))) with scale = x*2, activate = x+1,
shift = x+10, so step(3) = 17 at base. Every test loads its own copy of the toy package
(`toy.load_copy`), so participants and tests never share module objects.
"""
import gc
import itertools
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

from . import coordinator as C
from . import findings as F
from . import journal as J
from . import locate as L
from . import receipt as R
from . import swap as S
from . import toy
from . import witness as W

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


def install(module, source, targets, journal, witness, accept=()):
    """stage -> decide -> apply, as a participant does."""
    r = S.stage(module, source, targets)
    d = F.decide(r["findings"], accept)
    if not (r["ok"] and d["proceed"]):
        return r, d, None
    return r, d, S.apply(r["staged"], journal, witness)


def step_target():
    return {"source": None, "base_hash": L.text_hash(ENG),
            "targets": [["Engine.step", at(ENG, "def step")]]}


def text(path):
    with open(path) as f:
        return f.read()


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise AssertionError("timed out")
        time.sleep(0.005)


class KernelTest(unittest.TestCase):

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
        """Proves: undo_to(mark) restores the same code objects and removes added methods, imports and witness keys. Not: data the new code wrote."""
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
        self.assertEqual([x["kind"] for x in j.describe()],
                         ["added", "added", "added", "added", "code", "added"])

        u = j.undo_to(mark)
        self.assertEqual((u["ok"], u["undone"], j.entries), (True, 6, []))
        self.assertIs(eng.Engine.step.__code__, base_code)
        self.assertEqual(L.content_hash(eng.Engine.step.__code__), base_hash)
        self.assertEqual((set(vars(eng)), set(vars(eng.Engine))), (mod_keys, cls_keys))
        self.assertFalse(hasattr(e, "mean_token"))
        self.assertEqual(w.counts, {})
        e.step(1)
        self.assertEqual(e.steps, 5)

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

    def _three(self, tag, quiesce=None, core_path=None):
        out = []
        for pname in ("rank0", "rank1", "core"):
            name, eng, _ = fresh(tag + pname, core_path if pname == "core" else None)
            e = eng.Engine()
            out.append((C.LocalParticipant(pname, alias={"toy": name},
                                           quiesce=(quiesce or {}).get(pname),
                                           scope={"engine": e}), eng, e))
        return out

    def _group(self, ranks, fail_commit=None):
        """One participant made of several ranks (like vLLM's worker group: one collective
        call per method); `fail_commit`: the rank whose commit returns not-ok."""
        def call(method, payload):
            rs = []
            for i, (p, _, _) in enumerate(ranks):
                r = (F.not_ok(f"rank{i} commit failed") if method == "commit" and i == fail_commit
                     else getattr(p, method)(payload))
                rs.append(r)
            return {"ok": all(r["ok"] for r in rs), "ranks": rs,
                    "state": rs[0].get("state"), "error": "; ".join(
                        str(r.get("error")) for r in rs if not r["ok"]) or None}
        return C.CallParticipant("workers", call)

    def test_half_committed_group_is_undone(self):
        """Proves: when one rank of a group participant fails its commit, the ranks of that group that did commit are undone in the same phase 2, so every rank runs its base code. Not: real RPC (TP4 live run shows it on vLLM)."""
        parts = self._three("half")
        ranks = parts[:2]
        base = [(eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__))
                for _, eng, _ in ranks]
        spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        res = C.Coordinator([self._group(ranks, fail_commit=1), parts[2][0]]).run(
            {"modules": {"toy.engine": spec}})
        self.assertEqual(res["outcome"], "rolled_back")
        self.assertEqual([p._state[res["txn"]] for p, _, _ in ranks], ["undone", "aborted"])
        self.assertEqual([(eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__))
                          for _, eng, _ in ranks], base)

    def test_undo_of_an_older_txn_reports_the_later_ones(self):
        """Proves: undoing txn A after txn B committed on top also undoes B (journal order); B's state becomes undone and an undo_cascade finding names it, instead of B staying committed. Not: per-transaction undo out of order."""
        (p, eng, _), = self._three("cascade")[:1]
        a = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        co = C.Coordinator([p])
        ra = co.run({"id": "A", "modules": {"toy.engine": a}})
        b = {**step_target(), "source": edit(ENG, ("return y", "return y * 7"))}
        rb = co.run({"id": "B", "modules": {"toy.engine": b}, "accept": "all"})
        self.assertEqual((ra["outcome"], rb["outcome"]), ("committed", "committed"))
        u = p.undo("A")
        self.assertEqual((u["ok"], u.get("also_undone")), (True, ["B"]))
        self.assertEqual([f["check"] for f in u["findings"]], ["undo_cascade"])
        self.assertEqual(p.status("B")["state"], "undone")

    def test_vote_no_changes_nobody(self):
        """Proves: one participant on another build votes no and every participant keeps its base code objects. Not: a vote over real RPC."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        for f in ("__init__.py", "ops.py", "engine.py"):
            shutil.copy(os.path.join(toy.HERE, f), tmp)
        with open(os.path.join(tmp, "engine.py"), "w") as fh:
            fh.write(edit(ENG, ("make_shift(10)", "make_shift(20)")))
        parts = self._three("vote", core_path=tmp)
        base = [eng.Engine.step.__code__ for _, eng, _ in parts]
        spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        path = os.path.join(tmp, "receipts.jsonl")
        res = C.Coordinator([p for p, _, _ in parts], R.ReceiptWriter(path)).run(
            {"modules": {"toy.engine": spec}})
        self.assertEqual(res["outcome"], "aborted")
        rec = res["receipt"]
        self.assertEqual({n: r["prepare"]["ok"] for n, r in rec["participants"].items()},
                         {"rank0": True, "rank1": True, "core": False})
        self.assertEqual([(f["participant"], f["check"]) for f in rec["findings"]],
                         [("core", "base_mismatch"), ("core", "outside_changes")])
        self.assertEqual([eng.Engine.step.__code__ for _, eng, _ in parts], base)
        self.assertEqual([e.step(3) for _, _, e in parts], [17, 17, 27])
        self.assertEqual((rec["rollback"]["status"], rec["actual_recipes"], rec["edit_to_run_s"]),
                         ("not_needed", [], None))
        self.assertEqual(R.read(path)[0]["outcome"], "aborted")

    def test_commit_failure_undoes_the_committed(self):
        """Proves: a participant lost at commit gets rank0 undone (and itself undone: a no-op here) and core aborted, and the receipt says so. Not: a half-switched participant (next test)."""
        def lost(fn):
            raise RuntimeError("rank1: worker process lost")
        parts = self._three("fail", quiesce={"rank1": lost})
        base = [(eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__))
                for _, eng, _ in parts]
        spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
        res = C.Coordinator([p for p, _, _ in parts]).run({"modules": {"toy.engine": spec}})
        rec = res["receipt"]
        self.assertEqual(res["outcome"], "rolled_back")
        self.assertEqual({n: sorted(r) for n, r in rec["participants"].items()},
                         {"rank0": ["commit", "prepare", "undo", "validate"],
                          "rank1": ["abort", "commit", "prepare", "undo", "validate"],
                          "core": ["abort", "prepare", "validate"]})
        self.assertEqual(rec["participants"]["rank1"]["commit"]["error"],
                         "commit: RuntimeError: rank1: worker process lost")
        self.assertEqual((rec["rollback"]["status"], list(rec["rollback"]["participants"])),
                         ("undone", ["rank1", "rank0"]))
        self.assertIn(("rank1", "participant_failed"),
                      [(f["participant"], f["check"]) for f in rec["findings"]])
        self.assertEqual([(eng.Engine.step.__code__, L.content_hash(eng.Engine.step.__code__))
                          for _, eng, _ in parts], base)
        self.assertEqual([e.step(3) for _, _, e in parts], [17, 17, 17])
        self.assertEqual(parts[0][0].journal.entries, [])

    def test_linked_receipts_and_measured_times(self):
        """Proves: txn and proof records share the id, carry every field, time from measured timestamps, and only append. Not: GPU costs."""
        parts = self._three("receipt")
        _, ceng, ce = parts[2]
        ce.delay = 0.02
        stop = threading.Event()
        th = threading.Thread(target=ceng.run_loop, args=(ce, stop), daemon=True)
        parts[2][0].quiesce = ce.at_boundary
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "receipts.jsonl")
        co = C.Coordinator([p for p, _, _ in parts], R.ReceiptWriter(path),
                           engine={"name": "toy", "version": "0", "commit": None})
        th.start()
        try:
            wait_for(lambda: ce.looping)
            t_edit = time.time()
            spec = {**step_target(), "source": edit(ENG, ("return y", "return y * 100"))}
            res = co.run({"modules": {"toy.engine": spec}, "recipes": ["R0"],
                          "predicted_resources": ["code objects"], "certainty": "L2"},
                         edit_time=t_edit,
                         baseline_restart={"seconds": 31.0, "source": "kb/costs.md R9"})
            first = text(path)
            n = ce.steps
            wait_for(lambda: ce.steps >= n + 2)              # the workload: core's loop steps
            for _, _, e in parts[:2]:
                e.step(1)
            proof = co.prove(res["txn"], golden={"step(3)": "not run"})
            spec2 = {"source": edit(OPS, ("return x + 1", "return x + 2")),
                     "base_hash": L.text_hash(OPS), "targets": [["activate", at(OPS, "@traced")]]}
            res2 = co.run({"modules": {"toy.ops": spec2}})
        finally:
            stop.set()
            th.join()
        self.assertEqual((res["outcome"], res2["outcome"]), ("committed", "committed"))
        self.assertTrue(text(path).startswith(first))
        self.assertEqual([(r["kind"], r["id"]) for r in R.read(path)],
                         [("txn", res["txn"]), ("proof", res["txn"]), ("txn", res2["txn"])])
        rec, prf = R.read(path, res["txn"])
        self.assertEqual(sorted(R.FIELDS["txn"] + ("writer",)), sorted(rec))
        self.assertEqual(sorted(R.FIELDS["proof"] + ("writer",)), sorted(prf))
        tl = rec["timeline"]
        self.assertEqual(tl["edit"], t_edit)
        self.assertEqual(rec["edit_to_run_s"], round(tl["committed"] - tl["edit"], 6))
        self.assertTrue(tl["edit"] <= tl["prepare_start"] <= tl["prepared"] <= tl["gap_start"] <= tl["vote"]
                        <= tl["committed"] <= tl["gap_end"] <= tl["txn_end"])
        self.assertGreater(rec["edit_to_run_s"], 0)
        self.assertEqual(prf["edit_to_finish_s"],
                         round(prf["timeline"]["proved"] - prf["timeline"]["edit"], 6))
        self.assertGreater(prf["edit_to_finish_s"], rec["edit_to_run_s"])
        k = f"toy.engine:Engine.step:{at(ENG, 'def step')}"
        self.assertEqual((prf["witness"]["rank0"][k], prf["witness"]["rank1"][k]), (1, 1))
        self.assertGreaterEqual(prf["witness"]["core"][k], 2)    # its loop stepped >= 2 times
        self.assertEqual((prf["findings"], prf["golden"]), ([], {"step(3)": "not run"}))
        self.assertEqual(rec["baseline_restart_s"], {"seconds": 31.0, "source": "kb/costs.md R9"})
        self.assertEqual((rec["actual_recipes"], rec["commit_hashes"]["agree"], rec["outcome"]),
                         (["R0 swap"], True, "committed"))
        self.assertEqual(rec["engine"], {"name": "toy", "version": "0", "commit": None})
        # agreement compares only targets several participants swapped (seen live: workers
        # swap the sampler, the engine core the scheduler; nothing to compare is not "false")
        self.assertEqual([C._hashes_agree(h) for h in (
            {"workers": {"s:a:1": ["x"]}, "core": {"c:b:2": ["y"]}},
            {"rank0": {"s:a:1": ["x"]}, "rank1": {"s:a:1": ["z"]}})], [None, False])
        self.assertEqual([e.step(3) for _, _, e in parts], [1800, 1800, 1800])

    def test_accept_list(self):
        """Proves: an overridable finding proceeds only once accepted; a non-overridable one is refused even when accepted. Not: who should accept."""
        name, _, ops = fresh("accept")
        p = C.LocalParticipant("rank0", alias={"toy": name})
        new = edit(OPS, ("STEP_SCALE = 2", "STEP_SCALE = 3"),
                   ("return x * STEP_SCALE", "return x * STEP_SCALE + 1"))
        mods = {"toy.ops": {"source": new, "targets": [["scale", at(OPS, "def scale")]]}}
        r = p.prepare({"id": "a", "modules": mods})
        self.assertEqual((r["ok"], [(f["check"], f["accepted"]) for f in r["findings"]]),
                         (False, [("outside_changes", False)]))
        self.assertEqual(ops.scale(3), 6)
        r = p.prepare({"id": "b", "modules": mods, "accept": ["outside_changes"]})
        self.assertEqual((r["ok"], [(f["check"], f["accepted"]) for f in r["findings"]]),
                         (True, [("outside_changes", True)]))
        self.assertTrue(p.commit("b")["ok"])
        self.assertEqual((ops.scale(3), ops.STEP_SCALE), (7, 2))   # body new, constant old

        # on top of what b installed (the participant now stages against b's text)
        bad = {"toy.ops": {"source": edit(new, ("return x + k", "return x + k + bias")),
                           "targets": [["make_shift.<locals>.shift", at(OPS, "def shift")]]}}
        r = p.prepare({"id": "c", "modules": bad, "accept": "all"})
        self.assertEqual((r["ok"], [(f["check"], f["accepted"]) for f in r["findings"]]),
                         (False, [("closure_changed", False)]))

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

    def test_chained_edit_stages_against_the_installed_text(self):
        """Proves: a second edit of the same function, made on top of the first, stages against the text the first installed (no base_mismatch), and undoing both returns the base code object. Not: an edit that moved the target's first line."""
        name, _, ops = fresh("chain")
        p = C.LocalParticipant("rank0", alias={"toy": name})
        base, line = ops.activate.__wrapped__.__code__, at(OPS, "@traced")
        a2 = edit(OPS, ("return x + 1", "return x + 2"))
        a3 = edit(a2, ("return x + 2", "return x + 3"))
        for txn, old, new in (("a", OPS, a2), ("b", a2, a3)):
            self.assertEqual(S.changed_targets(old, new), [["activate", line]])
            r = p.prepare({"id": txn, "modules": {"toy.ops": {
                "source": new, "base_hash": L.text_hash(old), "targets": [["activate", line]]}}})
            self.assertEqual((r["ok"], r["findings"]), (True, []))
            self.assertTrue(p.commit(txn)["ok"])
        self.assertEqual(ops.activate(1), 4)
        self.assertTrue(p.undo("b")["ok"])
        self.assertEqual(ops.activate(1), 3)
        self.assertTrue(p.undo("a")["ok"])
        self.assertIs(ops.activate.__wrapped__.__code__, base)
        self.assertEqual((ops.activate(1), p.status("a")["state"], p.status("b")["state"]),
                         (2, "undone", "undone"))

    def test_removed_definition_is_a_finding_and_undo_restores_it(self):
        """Proves: an edit that only deletes a helper is a removed_definition finding; accepted, the helper is gone live; undo restores it. Not: from-import copies."""
        name, eng, _ = fresh("removed")
        e = eng.Engine()
        e.step(3)
        drain = vars(eng.Engine)["drain"]
        new = edit(ENG, ("    def drain(self):\n        self.tokens.clear()\n\n", ""))
        mods = {"toy.engine": {"source": new, "targets": []}}
        p = C.LocalParticipant("core", alias={"toy": name})
        r = p.prepare({"id": "a", "modules": mods})
        self.assertEqual((r["ok"], [(f["check"], f["subject"], f["accepted"])
                                    for f in r["findings"]]),
                         (False, [("removed_definition", "toy.engine:Engine.drain", False)]))
        e.drain()                                           # not accepted: still there
        self.assertEqual(e.tokens, [])
        self.assertTrue(p.prepare({"id": "b", "modules": mods,
                                   "accept": ["removed_definition"]})["ok"])
        c = p.commit("b")
        self.assertIn({"kind": "removed", "target": "Engine", "name": "drain"}, c["journal"])
        with self.assertRaisesRegex(AttributeError, "drain"):
            e.drain()                                       # the forgotten caller fails live
        self.assertTrue(p.undo("b")["ok"])
        self.assertIs(vars(eng.Engine)["drain"], drain)     # the same function object is back
        e.step(3)
        e.drain()
        self.assertEqual(e.tokens, [])

    def test_repair_cell_sets_missing_field_and_undo_removes_it(self):
        """Proves (R10): a repair cell sets a field the new step needs, step works, and undo removes the field. Not: device memory repairs."""
        name, eng, _ = fresh("repair")
        e = eng.Engine()
        new = edit(ENG, ("        self.steps = 0\n", "        self.steps = 0\n        self.bias = 0\n"),
                   ("y = self.shift(activate(scale(x)))",
                    "y = self.shift(activate(scale(x))) + self.bias"))
        mods = {"toy.engine": {"source": new,
                               "targets": [["Engine.__init__", at(ENG, "def __init__")],
                                           ["Engine.step", at(ENG, "def step")]]}}
        p = C.LocalParticipant("core", alias={"toy": name}, scope={"engine": e})
        self.assertTrue(p.prepare({"id": "swap-only", "modules": mods})["ok"])
        self.assertTrue(p.commit("swap-only")["ok"])
        with self.assertRaisesRegex(AttributeError, "bias"):
            e.step(3)
        self.assertTrue(p.undo("swap-only")["ok"])

        plan = {"id": "swap+repair", "modules": mods,
                "repairs": [{"cell": "engine.bias = 5", "declares": ["engine"]}]}
        self.assertTrue(p.prepare(plan)["ok"])
        c = p.commit("swap+repair")
        self.assertTrue(c["ok"])
        self.assertIn({"kind": "added", "target": "Engine", "name": "bias"}, c["journal"])
        self.assertEqual((e.step(3), vars(e)["bias"]), (22, 5))
        self.assertEqual(eng.Engine().bias, 0)               # new objects get it from __init__
        self.assertTrue(p.undo("swap+repair")["ok"])
        self.assertNotIn("bias", vars(e))
        self.assertEqual(e.step(3), 17)

    def test_repair_that_cannot_be_undone_is_a_finding(self):
        """Proves: in-place or undeclared mutations by a repair cell are repair_not_undoable findings, not exceptions. Not: deeper than one level."""
        name, eng, _ = fresh("repair2")
        e, other = eng.Engine(), eng.Engine()
        p = C.LocalParticipant("core", alias={"toy": name},
                               scope={"engine": e, "other": other})
        cell = "engine.bias = 1\nengine.tokens.append(99)\nother.steps = 7"
        plan = {"id": "r", "repairs": [{"cell": cell, "declares": ["engine"]}]}
        self.assertTrue(p.prepare(plan)["ok"])
        c = p.commit("r")
        self.assertFalse(c["ok"])
        self.assertEqual(sorted((f["check"], f["observed"]) for f in c["findings"]),
                         [("repair_not_undoable", "engine.tokens changed in place"),
                          ("repair_not_undoable", "undeclared other changed: steps")])
        self.assertNotIn("bias", vars(e))                     # declared: undone
        self.assertEqual((e.tokens, other.steps), ([99], 7))  # not undoable: still there
        self.assertTrue(p.prepare({**plan, "id": "r2", "accept": ["repair_not_undoable"]})["ok"])
        self.assertTrue(p.commit("r2")["ok"])
        self.assertEqual(vars(e)["bias"], 1)

    # -- step-gap transactions: ranks that step together --------------------------------------

    def _cluster(self, tag, gap_mode, quiesce=None, slow=None):
        """Three ranks, each its own toy copy, stepped together by one Cluster loop (the engine
        core). gap_mode: participants switch when called (the loop owner calls them in its
        gap); otherwise each switches at its own engine's boundary (the old way). The
        coordinator reaches each rank through CallParticipant over a loopback call that takes
        1 ms, like an RPC: in one process without it, the GIL would make three back-to-back
        commits look atomic and hide the race. slow: {rank: {method: extra seconds}}.
        Returns (cluster, call participants, locals)."""
        engines, ps, lps = {}, [], []
        for pname in ("rank0", "rank1", "rank2"):
            name, eng, _ = fresh(tag + pname)
            e = engines[pname] = eng.Engine()
            q = (quiesce or {}).get(pname) or (None if gap_mode else e.at_boundary)
            lp = C.LocalParticipant(pname, alias={"toy": name}, quiesce=q)

            def rpc(method, payload, lp=lp, extra=(slow or {}).get(pname, {})):
                time.sleep(0.001 + extra.get(method, 0))
                return getattr(lp, method)(payload)
            lps.append(lp)
            ps.append(C.CallParticipant(pname, rpc))
        return eng.Cluster(engines), ps, lps

    def _start(self, cl):
        stop = threading.Event()
        th = threading.Thread(target=cl.run, args=(stop,), daemon=True)
        th.start()

        def halt():
            stop.set()
            th.join()
        self.addCleanup(halt)
        wait_for(lambda: len(cl.log) >= 3)
        return halt

    def _steps(self, cl, n=5):
        k = len(cl.log)
        wait_for(lambda: len(cl.log) >= k + n)

    V1 = {"toy.engine": {"source": edit(ENG, ('self.ran.append("base")', 'self.ran.append("v1")')),
                         "base_hash": L.text_hash(ENG),
                         "targets": [["Engine.step", at(ENG, "def step")]]}}

    def test_step_gap_commit_and_undo_never_mix_versions(self):
        """Proves: with steps running, a step-gap commit then undo switch all 3 ranks between the same two steps, never mid-step. Not: real RPC."""
        cl, ps, lps = self._cluster("gap", gap_mode=True)
        co = C.StepGapCoordinator(ps, cl.at_gap)
        halt = self._start(cl)
        res = co.run({"modules": self.V1})
        self._steps(cl)
        u = co.undo(res["txn"])
        self._steps(cl)
        halt()
        self.assertEqual((res["outcome"], u["ok"]), ("committed", True))
        self.assertEqual([t for t in cl.log if len(set(t)) > 1], [])
        self.assertEqual([k for k, _ in itertools.groupby(cl.log)],
                         [("base",) * 3, ("v1",) * 3, ("base",) * 3])
        self.assertLess(res["receipt"]["gap_s"], 0.1)   # how long the gap stalled the loop

    def test_gap_stays_small_while_prepare_is_slow(self):
        """Proves: a 0.3 s prepare (like a heap scan) runs while steps go on; the loop stalls only for validate+commit. Not: vLLM's real scan cost."""
        cl, ps, _ = self._cluster("slowprep", gap_mode=True, slow={"rank0": {"prepare": 0.3}})
        co = C.StepGapCoordinator(ps, cl.at_gap)
        halt = self._start(cl)
        res = co.run({"modules": self.V1})
        self._steps(cl)
        halt()
        rec, tl = res["receipt"], res["receipt"]["timeline"]
        self.assertEqual(rec["outcome"], "committed")
        self.assertGreaterEqual(rec["prepare_s"], 0.3)
        self.assertLess(rec["gap_s"], 0.05)              # 6 loopback calls of 1 ms, no scan
        self.assertLess(cl.longest_stall(), 0.05)       # measured by the loop itself
        self.assertEqual([t for t in cl.log if len(set(t)) > 1], [])
        k = co.txns[res["txn"]]
        self.assertEqual(sorted(k["res"]["rank0"]), ["commit", "prepare", "validate"])
        self.assertTrue(tl["prepare_start"] < tl["prepared"] <= tl["gap_start"] < tl["gap_end"])

    def test_prepare_never_holds_the_loop_longer_than_the_budget(self):
        """Proves: preparing a method target on a 2.5M-object heap stalls the step loop under 10 ms (namespace locate, no heap scan). Not: nested targets."""
        budget = 0.010
        ballast = [[] for _ in range(2_500_000)]            # a loaded engine's heap
        self.addCleanup(ballast.clear)
        t = time.perf_counter()
        L.all_objects()                                     # what one heap scan holds
        full = time.perf_counter() - t
        if full < 2 * budget:
            self.skipTest(f"one heap scan takes {full:.3f} s here: too fast to tell")
        name, eng, _ = fresh("budget")
        cl = eng.Cluster({"r0": eng.Engine()})
        p = C.LocalParticipant("r0", alias={"toy": name})
        self._start(cl)
        t0 = time.perf_counter()
        r = p.prepare({"id": "b", "modules": self.V1})
        t1 = time.perf_counter()
        self._steps(cl, 3)
        stalls = [b - a for a, b in zip(cl.ends, cl.ends[1:]) if t0 <= b <= t1 + 0.005]
        self.assertTrue(r["ok"])
        self.assertLess(max(stalls, default=0.0), budget, f"one heap scan: {full:.3f} s")
        self.assertEqual(r["timings"]["toy.engine"]["scan"], "namespace")

    def test_validate_catches_a_change_between_prepare_and_gap(self):
        """Proves: a REPL edit of rank1's target after prepare is caught by validate in the gap, and no rank commits. Not: changes to data."""
        cl, ps, lps = self._cluster("validate", gap_mode=True)
        repl = L.code_tree(compile(edit(ENG, ('"base"', '"repl"')), "engine.py", "exec"))[0][
            ("Engine.step", at(ENG, "def step"))][0]
        r1 = cl.engines["rank1"]

        def gap(fn):                              # the REPL edit lands between the phases
            type(r1).step.__code__ = repl
            return cl.at_gap(fn)
        co = C.StepGapCoordinator(ps, gap)
        halt = self._start(cl)
        res = co.run({"modules": self.V1})
        self._steps(cl)
        halt()
        self.assertEqual((res["outcome"], co.status(res["txn"])["state"]), ("aborted", "aborted"))
        self.assertEqual([(f["participant"], f["check"]) for f in res["receipt"]["findings"]],
                         [("rank1", "changed_since_prepare")])
        self.assertEqual([lp.status(res["txn"])["state"] for lp in lps], ["aborted"] * 3)
        self.assertEqual({k for k, _ in itertools.groupby(cl.log)},
                         {("base", "base", "base"), ("base", "repl", "base")})   # never v1

    def test_timed_out_caller_resolves_the_outcome_with_status(self):
        """Proves: a caller whose gap call times out mid-commit gets in-doubt, and status gives the real outcome; an unstarted one is cancelled. Not: host crashes."""
        cl, ps, _ = self._cluster("indoubt", gap_mode=True,
                                  slow={r: {"commit": 0.05} for r in ("rank0", "rank1", "rank2")})
        co = C.StepGapCoordinator(ps, lambda fn: cl.at_gap(fn, timeout=0.05))
        halt = self._start(cl)
        res = co.run({"modules": self.V1})
        self.assertEqual((res["outcome"], res["receipt"]), ("in-doubt", None))
        self.assertEqual(co.status(res["txn"])["state"], "in-doubt")
        wait_for(lambda: co.status(res["txn"])["state"] != "in-doubt")
        st = co.status(res["txn"])
        self._steps(cl)
        halt()
        self.assertEqual((st["state"], st["outcome"]), ("committed", "committed"))
        self.assertEqual(co.txns[res["txn"]]["rec"]["outcome"], "committed")   # written late
        self.assertEqual([t for t in cl.log if len(set(t)) > 1], [])
        self.assertEqual(cl.log[-1], ("v1",) * 3)

        def stalled(fn):                          # the loop never reaches a gap
            raise TimeoutError("no step gap within 0.05 s")
        co2 = C.StepGapCoordinator(ps, stalled)
        res2 = co2.run({"modules": self.V1})
        self.assertEqual((res2["outcome"], co2.status(res2["txn"])["state"]),
                         ("aborted", "aborted"))

    def test_one_by_one_commit_mixes_versions(self):
        """Proves: the old coordinator (each rank at its own boundary) lets a step run ranks on different versions. Not: how often in a real engine."""
        cl, ps, lps = self._cluster("old", gap_mode=False)
        co = C.Coordinator(ps)
        halt = self._start(cl)
        for _ in range(5):                       # GIL timing: it shows on the first try here
            res = co.run({"modules": self.V1})
            self.assertTrue(res["ok"])
            self._steps(cl, 3)
            self.assertTrue(co.undo(res["txn"])["ok"])
            self._steps(cl, 3)
            if any(len(set(t)) > 1 for t in cl.log):
                break
        halt()
        mixed = [t for t in cl.log if len(set(t)) > 1]
        self.assertTrue(mixed)
        self.assertIn(mixed[0], [("v1", "base", "base"), ("v1", "v1", "base"),
                                 ("base", "v1", "v1"), ("base", "base", "v1")])

    def test_step_gap_failed_commit_is_undone_in_the_same_gap(self):
        """Proves: when rank1 fails at commit, rank0's commit is undone inside the same gap, so no step ever ran v1. Not: a rank that hangs."""
        def lost(fn):
            raise RuntimeError("rank1: worker process lost")
        cl, ps, lps = self._cluster("gapfail", gap_mode=True, quiesce={"rank1": lost})
        co = C.StepGapCoordinator(ps, cl.at_gap)
        halt = self._start(cl)
        res = co.run({"modules": self.V1})
        self._steps(cl)
        halt()
        self.assertEqual((res["outcome"], res["receipt"]["rollback"]["status"]),
                         ("rolled_back", "undone"))
        self.assertEqual(set(cl.log), {("base",) * 3})
        self.assertEqual(lps[0].journal.entries, [])

    def test_undo_runs_the_declared_data_reset(self):
        """Proves: code undo leaves a cache entry the new code wrote; the plan's R3 hook clears it, and a rank without the hook gets reset_failed. Not: R2."""
        name0, _, _ = fresh("reset0")
        name1, eng, _ = fresh("reset1")
        e0, e1 = sys.modules[name0 + ".engine"].Engine(), eng.Engine()

        def r3():
            n = len(e0.cache)
            e0.cache.clear()
            return {"cleared": n}
        ps = [C.LocalParticipant("core", alias={"toy": name0}, resets={"R3": r3}),
              C.LocalParticipant("rank0", alias={"toy": name1})]
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "receipts.jsonl")
        co = C.Coordinator(ps, R.ReceiptWriter(path))
        new = edit(ENG, ("self.steps += 1", "self.steps += 1\n        self.cache[x] = y"))
        res = co.run({"undo_resets": ["R3"], "modules": {"toy.engine": {
            "source": new, "targets": [["Engine.step", at(ENG, "def step")]]}}})
        self.assertEqual((e0.step(3), e1.step(3)), (17, 17))
        self.assertEqual((e0.cache, e1.cache), ({3: 17}, {3: 17}))
        u = co.undo(res["txn"])
        self.assertEqual((e0.cache, e1.cache), ({}, {3: 17}))   # rank0 had no R3 hook
        rec = u["receipt"]
        self.assertEqual((rec["status"], rec["resets"]), ("partial", ["R3"]))
        self.assertEqual(rec["participants"]["core"]["reset"]["R3"]["result"], {"cleared": 1})
        self.assertEqual([(f["participant"], f["check"]) for f in rec["findings"]],
                         [("rank0", "reset_failed")])
        e0.step(3)
        self.assertEqual(e0.cache, {})                          # base code writes no cache
        self.assertEqual([r["kind"] for r in R.read(path, res["txn"])], ["txn", "undo"])


if __name__ == "__main__":
    unittest.main()
