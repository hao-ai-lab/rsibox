"""Golden tests of the transaction kernel on the toy engine (CPU, standard library only).

Run from template/livelab:  python3 -m unittest kernel.test_kernel -v

Toy arithmetic: Engine.step(x) = shift(activate(scale(x))) with scale = x*2, activate = x+1,
shift = x+10, so step(3) = 17 at base. Every test loads its own copy of the toy package
(`toy.load_copy`), so participants and tests never share module objects.
"""
import ast
import gc
import importlib
import importlib.util
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
from . import session as SS
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

    def test_a_step_in_flight_at_prepare_is_advisory_and_the_gap_decides(self):
        """Proves: a frame of `Engine.step` on another thread while prepare runs (a step in flight) is an advisory running_frame (accepted, prepare votes yes, no accept in the plan); validate while that step is still running is not ok (running_frame at the quiesce point); once the step returned, validate is ok and the commit runs the new code. Before, stage's one sample vetoed the edit (a MoE forward edit 4 of 5 tries at TP2, eager), though no forward frame was live in the gap. Not: suspended generators (they still block at stage)."""
        name, eng, _ = fresh("inflight")
        e, gate, entered = eng.Engine(), threading.Event(), threading.Event()
        e.callbacks.append(lambda y: (entered.set(), gate.wait(5)))
        th = threading.Thread(target=e.step, args=(1,), daemon=True)
        th.start()
        self.assertTrue(entered.wait(5))
        p = C.LocalParticipant("core", alias={"toy": name})
        plan = {"id": "f", "modules": {"toy.engine": {"source": edit(
            ENG, ('self.ran.append("base")', 'self.ran.append("v1")')),
            "targets": [["Engine.step", at(ENG, "def step")]]}}}
        r = p.prepare(plan)
        self.assertTrue(r["ok"], r["findings"])
        self.assertEqual([(f["check"], f.get("advisory"), f["accepted"]) for f in r["findings"]],
                         [("running_frame", True, True)])
        v = p.validate("f")
        self.assertEqual((v["ok"], [f["check"] for f in v["findings"]]), (False, ["running_frame"]))
        gate.set()
        th.join()
        self.assertTrue(p.validate("f")["ok"])
        self.assertTrue(p.commit("f")["ok"])
        e.callbacks.pop()
        e.step(1)
        self.assertEqual(e.ran[-1], "v1")

    def test_engine_checks_add_findings_at_prepare(self):
        """Proves: an engine check given to LocalParticipant (`checks`) adds its findings to prepare's vote: a `captured_target` for the target votes no, and an accept record with that subject lets it through. Not: the vLLM check itself (adapter tests)."""
        name, eng, _ = fresh("checks")
        seen = []

        def captured(module, items):
            seen.append((module, sorted(items)))
            return [F.finding("captured_target", "runs only in captured graphs", "runs eagerly",
                              subject=k) for k in items]
        p = C.LocalParticipant("core", alias={"toy": name}, checks=[captured])
        mods = {"toy.engine": {"source": edit(ENG, ('self.ran.append("base")',
                                                    'self.ran.append("v1")')),
                               "targets": [["Engine.step", at(ENG, "def step")]]}}
        r = p.prepare({"id": "c1", "modules": mods})
        k = f"toy.engine:Engine.step:{at(ENG, 'def step')}"
        self.assertEqual((r["ok"], [(f["check"], f["subject"]) for f in r["findings"]]),
                         (False, [("captured_target", k)]))
        self.assertEqual(seen, [("toy.engine", [k])])
        r = p.prepare({"id": "c2", "modules": mods,
                       "accept": [{"check": "captured_target", "subject": k}]})
        self.assertTrue(r["ok"], r["findings"])

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
        # one state for the group: the rank that never committed got the group's undo too
        self.assertEqual([p._state[res["txn"]] for p, _, _ in ranks], ["undone", "undone"])
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

        # a standing accept for this participant and this text covers the same finding again;
        # for another participant, or once the text changes, it does not
        name2, _, _ = fresh("accept2")
        q = C.LocalParticipant("rank0", alias={"toy": name2})
        rec = {"check": "outside_changes", "scope": "standing", "by": "t", "reason": "r",
               "participant": "rank0", "code_hash": L.text_hash(new)}
        r = q.prepare({"id": "s1", "modules": mods, "accept": [rec]})
        self.assertTrue(r["ok"], r["findings"])
        q.abort("s1")
        r = C.LocalParticipant("rank1", alias={"toy": name2}).prepare(
            {"id": "s2", "modules": mods, "accept": [rec]})
        self.assertFalse(r["ok"])

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

    def test_editing_a_deleted_definition_is_a_finding_not_a_crash(self):
        """Proves: while an accepted edit has deleted `Engine.drain`, a plan that edits `drain` against the file (explicit target) is a vote no with findings (`base_mismatch`: the live text lacks it; `not_in_patch`: no such definition in the text that runs), not a KeyError out of prepare (the target's key is also an added definition, whose item has no `new_hash`; seen on our 2026-11-05 GPU run). After undo the same plan stages. Not: what an agent should do instead (undo first, or edit the text that runs)."""
        name, eng, _ = fresh("removed2")
        line = next(i for i, x in enumerate(ENG.splitlines(), 1) if "def drain" in x)
        gone = edit(ENG, ("    def drain(self):\n        self.tokens.clear()\n\n", ""))
        p = C.LocalParticipant("core", alias={"toy": name})
        self.assertTrue(p.prepare({"id": "b", "modules": {"toy.engine": {"source": gone}},
                                   "accept": ["removed_definition"]})["ok"])
        self.assertTrue(p.commit("b")["ok"])
        probe = {"toy.engine": {"source": edit(ENG, ("        self.tokens.clear()\n",
                                                     "        del self.tokens[:]\n")),
                                "base_hash": L.text_hash(ENG), "targets": [["Engine.drain", line]]}}
        r = p.prepare({"id": "c", "modules": probe})
        self.assertFalse(r["ok"])
        self.assertEqual(sorted(f["check"] for f in r["findings"]), ["base_mismatch", "not_in_patch"])
        self.assertTrue(p.undo("b")["ok"])
        r = p.prepare({"id": "d", "modules": probe})
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["staged"][f"toy.engine:Engine.drain:{line}"]["changed"])

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

    def test_repair_cell_journals_through_record_attr_and_record_call(self):
        """Proves: a repair cell reaches the participant's journal as `journal`: it replaces an attribute kept by a custom `__setattr__` (as nn.Module keeps submodules) with `record_attr` and re-inits a list in place with `record_call`; the commit is ok with no finding, and undo restores both (the same object, the old contents). Before, a cell had only the scope's names, so `journal` was a NameError (repair_failed) and live tests could not journal a Parameter.data or submodule repair. Not: torch (the journal's own tests cover nn.Module and Parameter.data)."""
        class Mod:                               # keeps attributes in _store, like nn.Module
            def __init__(self):
                object.__setattr__(self, "_store", {})

            def __setattr__(self, k, v):
                self._store[k] = v

            def __getattr__(self, k):
                try:
                    return self._store[k]
                except KeyError:
                    raise AttributeError(k) from None

        class Engine:
            pass
        e, buf = Engine(), Engine()                # buf: a device buffer stand-in
        buf.data = [1, 2, 3]
        e.head, e.cache = Mod(), buf
        e.head.act = act = object()
        p = C.LocalParticipant("core", scope={"engine": e})
        cell = ("journal.record_attr(engine.head, 'act')\n"
                "engine.head.act = 'new act'\n"
                "saved = engine.cache.data[:]\n"
                "journal.record_call(lambda: engine.cache.data.__setitem__(slice(None), saved), "
                "'cache re-init')\n"
                "engine.cache.data[:] = [0, 0, 0]\n")
        plan = {"id": "cell", "repairs": [{"cell": cell, "declares": []}]}
        self.assertTrue(p.prepare(plan)["ok"])
        c = p.commit("cell")
        self.assertTrue(c["ok"], c)
        self.assertEqual(c["findings"], [])
        self.assertEqual((e.head.act, e.cache.data), ("new act", [0, 0, 0]))
        self.assertEqual([j["kind"] for j in c["journal"]], ["attr", "call"])
        self.assertTrue(p.undo("cell")["ok"])
        self.assertIs(e.head.act, act)
        self.assertEqual(e.cache.data, [1, 2, 3])

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

    # -- groups of ranks, phase-2 deadline, compile units ---------------------------------

    def _ranks(self, tag, n=3, quiesce=None):
        """n ranks, each its own toy copy, as one GroupParticipant "workers" (like vLLM's
        worker group: one call reaches every rank). Returns (group, [(local, engine)])."""
        out = []
        for i in range(n):
            name, eng, _ = fresh(f"{tag}r{i}")
            out.append((C.LocalParticipant(f"rank{i}", alias={"toy": name},
                                           quiesce=(quiesce or {}).get(i)), eng.Engine()))
        return C.GroupParticipant("workers", [p for p, _ in out]), out

    def test_group_witness_is_kept_per_rank_and_uneven_is_a_finding(self):
        """Proves: a group's proof keeps each rank's witness count (5, 5, 2), not their sum (12), and a rank that ran the new code less than the others is a witness_uneven finding; equal ranks give none. Not: vLLM (the TP4 receipt summed 4 ranks into one count)."""
        group, ranks = self._ranks("wit")
        co = C.Coordinator([group])
        res = co.run({"modules": {"toy.engine": step_target() | {"source": self.V1[
            "toy.engine"]["source"]}}})
        self.assertEqual(res["outcome"], "committed")
        for (_, e), n in zip(ranks, (5, 5, 2)):         # rank2 ran 2 of the 5 steps
            for _ in range(n):
                e.step(3)
        pr = co.prove(res["txn"])["receipt"]
        k = f"toy.engine:Engine.step:{at(ENG, 'def step')}"
        self.assertEqual(pr["witness"], {"workers/rank0": {k: 5}, "workers/rank1": {k: 5},
                                         "workers/rank2": {k: 2}})
        self.assertEqual([(f["participant"], f["check"], f["subject"], f["observed"])
                          for f in pr["findings"]],
                         [("workers", "witness_uneven", k, "rank 0: 5, rank 1: 5, rank 2: 2")])
        for _ in range(3):
            ranks[2][1].step(3)                         # rank2 catches up: 5, 5, 5
        again = co.prove(res["txn"])
        self.assertEqual((again["ok"], again["receipt"]["findings"]), (True, []))

    def test_group_proof_reports_ranks_agree(self):
        """Proves: the txn receipt carries each rank's code hashes and ranks_agree per group; it is true when the ranks run the same code and false when one rank's function was replaced outside the journal right after commit (agree false too). Not: hashes read later than right after the gap."""
        group, _ = self._ranks("agree")
        rec = C.Coordinator([group]).run({"modules": self.V1})["receipt"]
        ch = rec["commit_hashes"]
        self.assertEqual((rec["outcome"], ch["ranks_agree"], ch["agree"]),
                         ("committed", {"workers": True}, True))
        self.assertEqual(sorted(ch["ranks"]["workers"]), ["0", "1", "2"])

        def tamper(fn):                      # rank 2: a REPL edit lands right after commit
            r = fn()
            mod = sys.modules[p2.alias["toy"] + ".engine"]
            mod.Engine.step.__code__ = mod.Engine.drain.__code__.replace(
                co_name="step", co_qualname="Engine.step")
            return r
        group, ranks = self._ranks("disagree", quiesce={2: tamper})
        p2 = ranks[2][0]
        rec = C.Coordinator([group]).run({"modules": self.V1})["receipt"]
        ch = rec["commit_hashes"]
        self.assertEqual((rec["outcome"], ch["ranks_agree"], ch["agree"]),
                         ("committed", {"workers": False}, False))
        h = ch["ranks"]["workers"]
        self.assertEqual((h["0"] == h["1"], h["1"] == h["2"]), (True, False))

    def test_ranks_that_disagree_after_commit_are_a_finding(self):
        """Proves: when rank 1 of a group runs other code for `Engine.step` right after commit, the txn receipt carries one ranks_disagree finding (participant "workers", the split "ranks 1 vs 0,2" and the function), not overridable, blocking in `decisions`; the outcome stays what happened (committed); ranks that agree give none. Seen live (Qwen3-8B TP2): ranks_agree false with findings [] and blocking [], so only a later proof's witness_uneven showed it. Not: what made the rank differ; ranks that hold different keys by design (that finding is overridable)."""
        group, _ = self._ranks("agreef")
        rec = C.Coordinator([group]).run({"modules": self.V1})["receipt"]
        self.assertNotIn("ranks_disagree", [f["check"] for f in rec["findings"]])

        def tamper(fn):
            r = fn()
            mod = sys.modules[p1.alias["toy"] + ".engine"]
            mod.Engine.step.__code__ = mod.Engine.drain.__code__.replace(
                co_name="step", co_qualname="Engine.step")
            return r
        group, ranks = self._ranks("disagreef", quiesce={1: tamper})
        p1 = ranks[1][0]
        rec = C.Coordinator([group]).run({"modules": self.V1})["receipt"]
        fs = [f for f in rec["findings"] if f["check"] == "ranks_disagree"]
        self.assertEqual([(f["participant"], f["subject"], f["overridable"]) for f in fs],
                         [("workers", "workers", False)])
        self.assertTrue(fs[0]["observed"].startswith("ranks 1 vs 0,2: "), fs[0]["observed"])
        self.assertIn("Engine.step", fs[0]["observed"])
        self.assertEqual(rec["outcome"], "committed")
        self.assertIn("ranks_disagree", rec["decisions"]["blocking"])
        self.assertFalse(F.decide(fs, "all")["proceed"])          # no accept overrides it

    def test_rank_split_names_the_ranks_and_whether_one_function_differs(self):
        """Proves: `_rank_split` groups ranks by their hashes and says whether a function held by several ranks differs (not overridable) or only some ranks hold a key (overridable). Not: the receipt (previous test)."""
        same = {"0": {"m:f:1": ["a"]}, "1": {"m:f:1": ["b"]}, "2": {"m:f:1": ["a"]}}
        only = {"0": {"m:f:1": ["a"], "m:g:9": ["c"]}, "1": {"m:f:1": ["a"]}}
        self.assertEqual(C._rank_split(same), ("ranks 1 vs 0,2: m:f:1", True))
        self.assertEqual(C._rank_split(only), ("ranks 0 vs 1: m:g:9", False))

    def test_hung_commit_ends_the_gap_at_the_deadline(self):
        """Proves: with steps running, a rank that hangs 0.6 s in commit does not stall the loop: the gap ends within the 0.2 s deadline, rank0's commit is undone in it, no step ever runs mixed versions (not even after the late commit arrives: it is refused), the outcome is definite (rolled back, state aborted) and the host and session are poisoned (install refused, undo allowed). Not: a participant stuck inside apply (its late switch is undone by the queued undo, after it ran)."""
        cl, ps, lps = self._cluster("deadline", gap_mode=True, slow={"rank1": {"commit": 0.6}})
        co = C.StepGapCoordinator(ps, cl.at_gap, phase2_deadline_s=0.2)
        sess = SS.Session(co, ["toy.engine"])
        halt = self._start(cl)
        t0 = time.perf_counter()
        res = sess.install({"modules": self.V1})
        took = time.perf_counter() - t0
        st = co.status(res["txn"])
        rec = res["receipt"]
        self.assertEqual((res["outcome"], st["state"], st["poisoned"], st["missed"]),
                         ("rolled_back", "aborted", True, ["rank1"]))
        self.assertLess(rec["gap_s"], 0.2 + 0.05)
        self.assertLess(took, 0.5)
        self.assertEqual(rec["rollback"]["missed_deadline"], ["rank1"])
        self.assertIn(("rank1", "deadline_missed"),
                      [(f.get("participant"), f["check"]) for f in rec["findings"]])
        self.assertEqual(sess.state(), "poisoned")
        again = sess.install({"modules": self.V1})
        self.assertEqual([f["check"] for f in again["findings"]], ["session_poisoned"])
        self.assertEqual(co.run({"modules": self.V1})["outcome"], "refused")
        # the hung commit arrives late; then the undo and abort queued behind it run
        wait_for(lambda: lps[1].status(res["txn"])["state"] != "prepared")
        self._steps(cl)
        halt()
        self.assertLess(cl.longest_stall(), 0.2 + 0.05)  # measured by the loop itself
        self.assertEqual(set(cl.log), {("base",) * 3})    # never v1 anywhere, never mixed
        self.assertEqual([p.status(res["txn"])["state"] for p in lps],
                         ["undone", "undone", "aborted"])   # rank1: its queued undo ran
        self.assertEqual(lps[1].journal.entries, [])     # the late commit applied nothing
        self.assertTrue(sess.uninstall()["ok"])          # undo stays allowed while poisoned
        self.assertEqual(sess.state(), "poisoned")

    def test_undo_after_a_deadline_rollback_leaves_every_rank_undone(self):
        """Proves: after a deadline rollback (rank1 hangs in commit, its late commit applies nothing) and then an undo of the transaction, every rank reports "undone", as the host does; before the fix the late rank said "aborted" and the others "undone" (4-rank live run, T4). Not: a rank whose late commit lands (the queued undo undoes it)."""
        cl, ps, lps = self._cluster("t4undo", gap_mode=True, slow={"rank1": {"commit": 0.4}})
        co = C.StepGapCoordinator(ps, cl.at_gap, phase2_deadline_s=0.1)
        self._start(cl)
        res = co.run({"modules": self.V1})
        self.assertEqual(res["outcome"], "rolled_back")
        wait_for(lambda: lps[1].status(res["txn"])["state"] != "prepared")
        time.sleep(0.05)                                   # the queued undo and abort run
        self.assertTrue(co.undo(res["txn"])["ok"])
        self.assertEqual([p.status(res["txn"])["state"] for p in lps] +
                         [co.status(res["txn"])["state"]], ["undone"] * 4)
        self.assertEqual(lps[1].journal.entries, [])      # the late rank never applied

    TRICKY = '''from __future__ import annotations
import functools
import os.path
try:
    import json
except ImportError:
    json = None


class Base:
    def f(self):
        return 1


class Sub(Base):
    __slots__ = ("__secret",)
    kind = staticmethod(lambda x: x + 1)

    def f(self):
        return super().f() + len(self.__secret) + json.dumps(1).count("1")

    class Inner:
        def g(self, x: int) -> int:
            return os.path.join("a", str(x))


class Box[T]:
    def get(self, x: T) -> T:
        return functools.reduce(lambda a, b: a, [x])


def outer(k):
    def inner(x):
        return x + k
    return inner


@functools.lru_cache(maxsize=None)
def cached(x=lambda: 0):
    return x
'''

    def test_stage_compiles_only_the_changed_definitions(self):
        """Proves: stage compiles one unit per changed definition (Engine.step; the enclosing make_shift for a closure), not the module, and every definition's unit code equals the whole-module compile (super(), name mangling, nested classes, generic classes, closures, decorators, lambdas, __future__ and module imports). Not: the GIL stall it saves (measured live)."""
        _, eng, ops = fresh("units")
        r = S.stage(eng.__name__, self.V1["toy.engine"]["source"], [["Engine.step",
                                                                    at(ENG, "def step")]])
        self.assertEqual((r["ok"], r["timings"]["compiled"]), (True, ["Engine.step"]))
        new_ops = edit(OPS, ("        return x + k", "        return x + k * 1"))
        r = S.stage(ops.__name__, new_ops, [["make_shift.<locals>.shift",
                                             at(OPS, "def shift")]])
        self.assertEqual((r["ok"], r["timings"]["compiled"]), (True, ["make_shift"]))
        old = S.stage(eng.__name__, self.V1["toy.engine"]["source"],
                      [["Engine.step", at(ENG, "def step")]], compile="module")
        self.assertEqual(old["timings"]["compiled"], ["<module>"])
        for src in (ENG, OPS, self.TRICKY):
            tree = ast.parse(src)
            whole = L.code_tree(compile(tree, "<m>", "exec", dont_inherit=True))[0]
            for q, nodes in S.qual_index(tree).items():
                for n in nodes:
                    k = (q, S.first_line(n))
                    unit = S.compile_units(tree, [n], "<m>")[0]
                    self.assertEqual([L.content_hash(c) for c in unit[k]],
                                     [L.content_hash(c) for c in whole[k]], k)


class ProcessAnalyzerTest(unittest.TestCase):
    """Prepare's text half (parse, unit compile, outside-changes compare) in a child process.
    Pain point: on vLLM's scheduler.py the two parses and two `ast.dump` calls held the engine
    core's GIL 14-19 ms at a time, so a step during prepare took 3.4-3.95x the median at TP4."""

    @classmethod
    def setUpClass(cls):
        cls.A = S.ProcessAnalyzer().start()

    @classmethod
    def tearDownClass(cls):
        cls.A.close()

    @staticmethod
    def _summary(r):
        """Everything stage decides: findings, items, and the staged swap (code objects
        compared with ==, which covers bytecode, constants, names and line tables)."""
        out = {"ok": r["ok"], "findings": r["findings"], "items": r["items"]}
        st = r["staged"]
        if st is not None:
            out.update(
                imports={k: id(v) for k, v in st.imports.items()},
                swaps=[(id(fn), c, d, kw) for fn, c, d, kw in st.swaps],
                adds=[(id(o), n, getattr(getattr(x, "__func__", x), "__code__", x))
                      for o, n, x in st.adds],
                removes=[(id(o), n) for o, n in st.removes], keys=st.keys,
                expect=[(id(fn), id(o), h) for fn, o, h in st.expect])
        return out

    def test_process_analysis_stages_the_same_code_and_findings(self):
        """Proves: for 14 edits (method with an assert, closure, closure change, decorated, defaults and defaults that fail, added function, method and import, removed method, outside change, generator expression, targets not in the base, a target nothing runs, whole-module compile, a patch that does not parse), stage with the analysis in a child process returns the same findings, items and staged code objects as in-process, and the child really ran (timings say "process"). Not: other Python versions in parent and child (the child is this interpreter)."""
        name, eng, ops = fresh("analyzer")
        _, _, gone = fresh("analyzer_gone")       # nothing runs scale's base code any more
        gone.scale.__code__ = (lambda x: x).__code__
        E, O = eng.__name__, ops.__name__
        step = [["Engine.step", at(ENG, "def step")]]
        shift = [["make_shift.<locals>.shift", at(OPS, "def shift")]]
        scale = [["scale", at(OPS, "def scale")]]
        stall = [["Cluster.longest_stall.<locals>.<genexpr>", at(ENG, "(b - a for a, b in")]]
        cases = {
            "method": (E, edit(ENG, ('self.ran.append("base")', 'self.ran.append("v1")\n'
                                     '        assert x is not None')), step, {}),
            "closure": (O, edit(OPS, ("return x + k", "return x + k * 1")), shift, {}),
            "closure_changed": (O, edit(OPS, ("return x + k", "return x + k + bias")), shift,
                                {}),
            "decorated": (O, edit(OPS, ("return x + 1", "return x + 2")),
                          [["activate", at(OPS, "@traced")]], {}),
            "defaults": (O, edit(OPS, ("def scale(x):", "def scale(x, m=STEP_SCALE):")),
                         scale, {}),
            "defaults_fail": (O, edit(OPS, ("def scale(x):", "def scale(x, m=NOPE):")), scale,
                              {}),
            "added": (E, edit(ENG, ("import threading\n", "import threading\nimport fractions\n"),
                              ("        self.tokens.clear()\n",
                               "        self.tokens.clear()\n\n    def extra(self):\n"
                               "        return helper(1)\n"),
                              ("class Gap:", "def helper(x):\n    return fractions.Fraction(x)"
                               "\n\n\nclass Gap:"),
                              ('self.ran.append("base")', 'self.ran.append(str(helper(1)))')),
                      step, {}),
            "removed": (E, edit(ENG, ("    def drain(self):\n        self.tokens.clear()\n\n",
                                      "")), [], {}),
            "outside": (O, edit(OPS, ("STEP_SCALE = 2", "STEP_SCALE = 3"),
                                ("return x * STEP_SCALE", "return x * STEP_SCALE * 1")), scale, {}),
            "genexpr": (E, edit(ENG, ("(b - a for a, b in", "(2 * (b - a) for a, b in")), stall,
                        {}),
            "missing": (E, edit(ENG, ('self.ran.append("base")', 'self.ran.append("v1")')),
                        [["Engine.nope", 3], ["Engine.step", 1]] + step, {}),
            "not_live": (gone.__name__, edit(OPS, ("return x * STEP_SCALE",
                                                   "return x * STEP_SCALE * 1")), scale, {}),
            "module": (E, edit(ENG, ('self.ran.append("base")', 'self.ran.append("v1")')), step,
                       {"compile": "module"}),
            "bad_patch": (E, ENG + "\ndef (:\n", step, {}),
        }
        for case, (mod, src, targets, kw) in cases.items():
            here = S.stage(mod, src, targets, **kw)
            there = S.stage(mod, src, targets, analyzer=self.A, **kw)
            if case != "bad_patch":
                self.assertEqual((here["timings"]["analyze"], there["timings"]["analyze"]),
                                 ("inline", "process"), case)
            self.assertEqual(self._summary(there), self._summary(here), case)
        checks = {c: sorted(f["check"] for f in S.stage(m, s, t, analyzer=self.A, **kw)
                            ["findings"]) for c, (m, s, t, kw) in cases.items()}
        self.assertEqual(checks, {   # the cases cover the checks they are named for
            "method": [], "closure": [], "closure_changed": ["closure_changed"],
            "decorated": [], "defaults": [], "defaults_fail": ["defaults_not_evaluated"],
            "added": [], "removed": ["removed_definition"], "outside": ["outside_changes"],
            "genexpr": [], "missing": ["not_in_patch", "not_in_patch"],
            "not_live": ["no_live_function", "outside_changes"], "module": [],
            "bad_patch": ["patch_invalid"]})

    def test_process_analysis_keeps_the_gil_free(self):
        """Proves: staging a one-line edit of a 30,000-line module holds the caller's GIL under 10 ms at a time with the analysis in a child process, while in-process it holds it for one whole parse. Not: vLLM's own numbers (CPU and live runs in the kernel README)."""
        budget = 0.010
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        body = "".join(f"def f{i}(x):\n    y = x + {i}\n    return [y, y * 2, {{'k': y}}]\n\n\n"
                       for i in range(6000))
        with open(os.path.join(tmp, "kt_bigmod.py"), "w") as f:
            f.write(body)
        sys.path.insert(0, tmp)
        self.addCleanup(sys.path.remove, tmp)
        import kt_bigmod
        self.addCleanup(sys.modules.pop, "kt_bigmod", None)
        new = body.replace("y = x + 7\n", "y = x + 8\n", 1)
        targets = [["f7", at(body, "def f7(")]]

        def longest_hold(**kw):
            gc.collect()           # an engine freezes its heap after startup (vLLM does), so a
            gc.freeze()            # collection triggered by earlier tests' garbage is not counted
            self.addCleanup(gc.unfreeze)
            box = {}
            th = threading.Thread(target=lambda: box.update(r=S.stage(
                "kt_bigmod", new, targets, **kw)))
            gaps, last = [], time.perf_counter()
            th.start()
            while th.is_alive():
                time.sleep(0.0005)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now
            th.join()
            self.assertTrue(box["r"]["ok"], box["r"]["findings"])
            self.assertEqual(box["r"]["findings"], [])
            return max(gaps), box["r"]["timings"]

        inline, t_in = longest_hold()
        if t_in["parse_s"] / 2 < 2 * budget:
            self.skipTest(f"one parse takes {t_in['parse_s'] / 2:.3f} s here: too fast to tell")
        there, t_pr = longest_hold(analyzer=self.A)
        self.assertEqual((t_in["analyze"], t_pr["analyze"]), ("inline", "process"))
        self.assertGreater(inline, 2 * budget)
        self.assertLess(there, budget, f"in-process: {inline:.3f} s")


class AnalyzerFallbackTest(unittest.TestCase):
    """Pain point: when the kernel was loaded under another package name (the vLLM adapter
    loads it as `fastrsi_kernel` from a directory named `kernel`), the analyzer child died on
    import and every prepare quietly analyzed in-process, holding the engine's GIL."""

    def test_child_runs_when_the_kernel_has_another_package_name(self):
        """Proves: a kernel loaded as package `fastrsi_kernel_<n>` from its `kernel` directory stages a toy edit with the analysis really in the child (timings "process", no analyze_fallback, no findings). Not: a kernel directory the child cannot read."""
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        pkg = f"fastrsi_kernel_{next(_n)}"
        spec = importlib.util.spec_from_file_location(
            pkg, os.path.join(here, "__init__.py"), submodule_search_locations=[here])
        sys.modules[pkg] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[pkg])
        self.addCleanup(lambda: [sys.modules.pop(k) for k in list(sys.modules)
                                 if k == pkg or k.startswith(pkg + ".")])
        RS = importlib.import_module(pkg + ".swap")
        self.assertNotEqual(RS.__name__, S.__name__)
        A = RS.ProcessAnalyzer().start()
        self.addCleanup(A.close)
        _, eng, _ = fresh("renamed")
        new = edit(ENG, ('self.ran.append("base")', 'self.ran.append("v1")'))
        r = RS.stage(eng.__name__, new, [["Engine.step", at(ENG, "def step")]], analyzer=A)
        self.assertEqual((r["ok"], r["findings"], r["timings"]["analyze"],
                          "analyze_fallback" in r["timings"]), (True, [], "process", False))

    def test_fallback_to_in_process_is_a_finding(self):
        """Proves: an analyzer child that dies at start (here: a stand-in interpreter that prints an import error and exits 1) makes prepare vote no with an analyze_fallback finding that carries the child's reason and the in-process cost; accepting it stages the same edit in-process and commits. Not: a child that hangs (the timeout path raises the same way)."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        fake = os.path.join(tmp, "python")
        with open(fake, "w") as f:
            f.write("#!/bin/sh\necho 'ModuleNotFoundError: No module named fastrsi_kernel' >&2"
                    "\nexit 1\n")
        os.chmod(fake, 0o755)
        A = S.ProcessAnalyzer(python=fake)
        self.addCleanup(A.close)
        name, eng, _ = fresh("fallback")
        p = C.LocalParticipant("rank0", alias={"toy": name}, analyzer=A)
        mods = {"toy.engine": {"source": edit(ENG, ('self.ran.append("base")',
                                                    'self.ran.append("v1")'))}}
        r = p.prepare({"id": "f1", "modules": mods})
        self.assertFalse(r["ok"])
        f, = r["findings"]
        self.assertEqual((f["check"], f["overridable"], f["accepted"]),
                         ("analyze_fallback", True, False))
        self.assertIn("(code 1): ModuleNotFoundError: No module named fastrsi_kernel",
                      f["observed"])
        self.assertRegex(f["observed"], r"holding its GIL for [0-9.]+ ms")
        r = p.prepare({"id": "f2", "modules": mods, "accept": ["analyze_fallback"]})
        self.assertTrue(r["ok"], r["findings"])
        t = r["timings"]["toy.engine"]
        self.assertEqual(t["analyze"], "inline")
        self.assertIn("No module named fastrsi_kernel", t["analyze_fallback"])
        self.assertTrue(p.commit("f2")["ok"])
        e = eng.Engine()
        e.step(3)
        self.assertEqual(e.ran, ["v1"])


class WarmAnalyzerTest(unittest.TestCase):
    """Pain point: the analyzer child was spawned by the first prepare, so on an engine host
    the first edit after engine start paid its 1.5-2.2 s boot (0.2 s edit-to-run became 2 s),
    and an adapter that created its participant lazily could not avoid it."""

    def test_child_boots_at_participant_creation_not_in_the_first_prepare(self):
        """Proves: a LocalParticipant given an unstarted ProcessAnalyzer has a live child before any prepare (spawns 1), a second participant on the same analyzer does not spawn again, and the first prepare spawns nothing and waits for no boot (no analyze_spawned, no analyze_boot_wait_s in its timings; analysis "process"). Not: the boot time on an engine host (the GPU job measures that)."""
        A = S.ProcessAnalyzer()
        self.addCleanup(A.close)
        name, eng, _ = fresh("warm")
        p = C.LocalParticipant("rank0", alias={"toy": name}, analyzer=A)
        st = A.state()
        self.assertEqual((st["alive"], st["spawns"]), (True, 1))
        C.LocalParticipant("rank0b", alias={"toy": name}, analyzer=A)   # idempotent
        self.assertEqual(A.state()["spawns"], 1)
        self.assertTrue(A.wait_ready(30))
        mods = {"toy.engine": {"source": edit(ENG, ('self.ran.append("base")',
                                                    'self.ran.append("v1")'))}}
        r = p.prepare({"id": "w1", "modules": mods})
        self.assertTrue(r["ok"], r["findings"])
        t = r["timings"]["toy.engine"]
        self.assertEqual((t["analyze"], "analyze_spawned" in t, "analyze_boot_wait_s" in t,
                          A.state()["spawns"]), ("process", False, False, 1))

    def test_a_dead_child_is_respawned_by_the_next_prepare_and_said(self):
        """Proves: after the warm child is killed, nothing respawns it until the next prepare (alive False, spawns 1), which respawns it (spawns 2), still analyzes in the child (no analyze_fallback) and says so in its timings (analyze_spawned True). Not: a respawn that fails (test_fallback_to_in_process_is_a_finding covers that)."""
        A = S.ProcessAnalyzer()
        self.addCleanup(A.close)
        name, eng, _ = fresh("respawn")
        p = C.LocalParticipant("rank0", alias={"toy": name}, analyzer=A)
        self.assertTrue(A.wait_ready(30))
        A._p.kill()
        A._p.wait()
        self.assertEqual((A.state()["alive"], A.state()["spawns"]), (False, 1))
        mods = {"toy.engine": {"source": edit(ENG, ('self.ran.append("base")',
                                                    'self.ran.append("v1")'))}}
        r = p.prepare({"id": "r1", "modules": mods})
        self.assertTrue(r["ok"], r["findings"])
        t = r["timings"]["toy.engine"]
        self.assertEqual((t["analyze"], t.get("analyze_spawned"), "analyze_fallback" in t,
                          A.state()["spawns"]), ("process", True, False, 2))


class CrossModuleTest(unittest.TestCase):
    """Pain point (vLLM #51899): one edit adds a name in module A and imports it in module B.
    Staging B failed (A is applied only at commit), so the lab shimmed sys.modules."""

    def test_module_imports_a_name_another_module_of_the_txn_adds(self):
        """Proves: with ops adding `double` and engine importing it (plain import, and as the default of an added function), one prepare stages both in either plan order with nothing visible before commit; the commit makes engine use ops' very object (step(3) = 34); undo removes both. Not: import cycles between the two patches."""
        A_SRC = edit(OPS, ("def make_shift(k):", "def double(x):\n    return 2 * x\n\n\n"
                                                 "def make_shift(k):"))
        B_SRC = edit(ENG, ("from .ops import activate, make_shift, scale",
                           "from .ops import activate, double, make_shift, scale"),
                     ("class Engine:", "def twice(x, f=double):\n    return f(x)\n\n\n"
                                       "class Engine:"),
                     ("y = self.shift(activate(scale(x)))",
                      "y = twice(self.shift(activate(scale(x))))"))
        for order in (("toy.engine", "toy.ops"), ("toy.ops", "toy.engine")):
            name, eng, ops = fresh("xmod")
            p = C.LocalParticipant("rank0", alias={"toy": name})
            src = {"toy.ops": A_SRC, "toy.engine": B_SRC}
            r = p.prepare({"id": "x", "modules": {m: {"source": src[m]} for m in order}})
            self.assertEqual((r["ok"], r["findings"]), (True, []), order)
            self.assertEqual(r["targets"], {"toy.engine": [["Engine.step",
                                                            at(ENG, "def step")]],
                                            "toy.ops": []}, order)
            # staged, not visible: neither module has the new names yet
            self.assertEqual([n for n in ("double", "twice") if n in vars(ops) or n in vars(eng)],
                             [], order)
            self.assertEqual(eng.Engine().step(3), 17)
            self.assertTrue(p.commit("x")["ok"])
            self.assertEqual(eng.Engine().step(3), 34)
            self.assertIs(eng.double, ops.double)
            self.assertIs(eng.twice.__defaults__[0], ops.double)
            self.assertTrue(p.undo("x")["ok"])
            self.assertEqual([n for n in ("double", "twice") if n in vars(ops) or n in vars(eng)],
                             [], order)
            self.assertEqual(eng.Engine().step(3), 17)


class AutoTargetsTest(unittest.TestCase):
    """Pain point: prepare parsed every text twice, once in the caller's planner
    (`changed_targets`) and once in `analyze`. With targets omitted, analyze finds them."""

    @staticmethod
    def _reference(base, new):
        """The changed definitions by ast.dump alone (no text shortcut)."""
        bq, pq = S.qual_index(ast.parse(base)), S.qual_index(ast.parse(new))
        return sorted([[q, S.first_line(b)] for q, bn in bq.items()
                       if len(pq.get(q, [])) == len(bn)
                       for b, p in zip(bn, pq[q]) if ast.dump(b) != ast.dump(p)],
                      key=lambda t: t[1])

    def test_changed_targets_match_the_ast_reference(self):
        """Proves: the text shortcut in `changed_targets` gives the same list as comparing every definition's ast.dump, on edits that trip a naive text compare: a comment-only change (no target), a lambda edit two lambdas share a line with, ordinals shifted by a lambda added above and one removed below, a decorator edit, non-ASCII text before an edited lambda (byte columns). Not: every Python grammar construct."""
        base = ("import functools\n\n\ndef f(x):\n    a = (lambda: 1, lambda: 2)\n"
                "    b = lambda: 3\n    return a\n\n\n"
                "@functools.lru_cache(maxsize=1)\ndef h(x):\n    return [y for y in x]\n\n\n"
                "def u():\n    s = 'é'; return (lambda: s, (z for z in s))\n")
        edits = {
            "comment": edit(base, ("    return a\n", "    return a  # same\n")),
            "same_line": edit(base, ("lambda: 2)", "lambda: 22)")),
            "shifted": edit(base, ("def f(x):\n", "def f(x):\n    q = lambda: 0\n"),
                            ("b = lambda: 3", "b = 3")),
            "decorator": edit(base, ("maxsize=1", "maxsize=2")),
            "utf8": edit(base, ("lambda: s,", "lambda: s + s,"), ("z for z", "z * 2 for z")),
        }
        got = {k: S.changed_targets(base, v) for k, v in edits.items()}
        self.assertEqual(got, {k: self._reference(base, v) for k, v in edits.items()})
        self.assertEqual(got["comment"], [])
        self.assertEqual(got["shifted"], [["f", 4], ["f.<locals>.<lambda>", 5],
                                          ["f.<locals>.<lambda>", 5],
                                          ["f.<locals>.<lambda>", 6]])
        self.assertEqual(got["utf8"], [["u", 15], ["u.<locals>.<lambda>", 16],
                                       ["u.<locals>.<genexpr>", 16]])

    def test_auto_targets_stage_like_explicit_ones(self):
        """Proves: for toy edits (method, closure, decorated, defaults, added and removed definitions, outside change, generator expression, comment only, two methods at once), stage with targets omitted resolves the same targets as `changed_targets` and stages the same findings, items and code objects as with them passed, in-process and in the analyzer child. Not: vLLM modules (checked on CPU against the installed vLLM; numbers in the kernel README)."""
        A = S.ProcessAnalyzer().start()
        self.addCleanup(A.close)
        E, O = toy.source("engine"), toy.source("ops")
        cases = {
            "method": ("engine", edit(E, ('self.ran.append("base")', 'self.ran.append("v1")'))),
            "closure": ("ops", edit(O, ("return x + k", "return x + k * 1"))),
            "decorated": ("ops", edit(O, ("return x + 1", "return x + 2"))),
            "defaults": ("ops", edit(O, ("def scale(x):", "def scale(x, m=STEP_SCALE):"))),
            "added": ("engine", edit(E, ("class Gap:", "def helper(x):\n    return x\n\n\n"
                                         "class Gap:"),
                                     ('self.ran.append("base")',
                                      'self.ran.append(str(helper(1)))'))),
            "removed": ("engine", edit(E, ("    def drain(self):\n        self.tokens.clear()"
                                           "\n\n", ""))),
            "outside": ("ops", edit(O, ("STEP_SCALE = 2", "STEP_SCALE = 3"),
                                    ("return x * STEP_SCALE", "return x * STEP_SCALE * 1"))),
            "genexpr": ("engine", edit(E, ("(b - a for a, b in", "(2 * (b - a) for a, b in"))),
            "comment": ("ops", edit(O, ("def scale(x):", "def scale(x):  # faster"))),
            "two": ("engine", edit(E, ('self.ran.append("base")', 'self.ran.append("v1")'),
                                   ("self.tokens.clear()", "self.tokens.clear()\n"
                                    "        self.steps += 0"))),
        }
        _summary = ProcessAnalyzerTest._summary
        for case, (m, src) in cases.items():
            _, eng, ops = fresh("auto")
            mod = (eng if m == "engine" else ops).__name__
            base = E if m == "engine" else O
            explicit = S.changed_targets(base, src)
            self.assertEqual(explicit, self._reference(base, src), case)
            want = _summary(S.stage(mod, src, explicit))
            for kw in ({}, {"analyzer": A}):
                r = S.stage(mod, src, **kw)
                self.assertEqual(r["targets"], explicit, case)
                self.assertEqual(_summary(r), want, case)
                self.assertIn("targets_s", r["timings"])
        self.assertEqual(S.changed_targets(E, cases["two"][1]),
                         [["Engine.step", at(E, "def step")],
                          ["Engine.drain", at(E, "def drain")]])

    def test_auto_target_only_the_heap_finds_live(self):
        """Proves: a changed lambda inside a closure that only the heap holds (the namespace scan calls it not live) is re-analyzed after the heap scan, so auto stages it like explicit targets do (`analyze_again`), and the closure runs the new lambda. Not: deeper nesting."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        base = ("def make(k):\n    def f(x):\n        return (lambda: x + k)()\n    return f\n\n\n"
                "HOLD = [make(1)]\n")
        new = base.replace("lambda: x + k", "lambda: x + k + 100")
        name = f"kt_heaponly{next(_n)}"
        with open(os.path.join(tmp, name + ".py"), "w") as f:
            f.write(base)
        sys.path.insert(0, tmp)
        self.addCleanup(sys.path.remove, tmp)
        self.addCleanup(sys.modules.pop, name, None)
        mod = importlib.import_module(name)
        _summary = ProcessAnalyzerTest._summary
        explicit = S.changed_targets(base, new)
        self.assertEqual([q for q, _ in explicit], ["make", "make.<locals>.f",
                                                    "make.<locals>.f.<locals>.<lambda>"])
        want = _summary(S.stage(name, new, explicit))
        r = S.stage(name, new)
        self.assertEqual((r["timings"]["scan"], r["timings"].get("analyze_again")),
                         ("heap", True))
        self.assertEqual(_summary(r), want)
        self.assertEqual(r["findings"], [])
        self.assertTrue(S.apply(r["staged"], J.Journal(), W.Witness())["ok"])
        self.assertEqual(mod.HOLD[0](2), 103)


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


class GenexprTargetTest(unittest.TestCase):
    """A changed generator expression is a target in its own right (its code is a constant of
    the enclosing function; no function object holds it). Pain point: the vLLM adapter got
    `not_in_patch` for such targets and fell back to a heap scan."""

    def test_swap_a_generator_expression(self):
        name, eng, _ = fresh("genexpr")
        new = edit(ENG, ("(b - a for a, b in", "(2 * (b - a) for a, b in"))
        line = at(ENG, "(b - a for a, b in")
        c = eng.Cluster({})
        c.ends = [0.0, 1.0, 3.0]
        self.assertEqual(c.longest_stall(), 2.0)
        r, d, a = install(eng.__name__, new, [["Cluster.longest_stall.<locals>.<genexpr>", line]],
                          J.Journal(), W.Witness())
        self.assertTrue(r["ok"], r["findings"])
        self.assertNotIn("not_in_patch", [f["check"] for f in r["findings"]])
        self.assertTrue(d["proceed"], d["blocking"])
        self.assertTrue(a["ok"], a)
        self.assertEqual(c.longest_stall(), 4.0)      # the new expression runs
        self.assertEqual(r["timings"]["scan"], "namespace")   # found via the enclosing method
        self.assertEqual(r["timings"].get("suspended"), False)  # no heap pass for frames either

    def test_second_genexpr_edit_on_top_is_not_a_base_mismatch(self):
        """Pain point: an undo rebuild hit base_mismatch on the kernel's own instrumented
        generator expression (a code nested in a code this participant installed)."""
        name, _, _ = fresh("genexpr2")
        p = C.LocalParticipant("rank0", alias={"toy": name})
        line = at(ENG, "(b - a for a, b in")
        tq = [["Cluster.longest_stall.<locals>.<genexpr>", line]]
        one = edit(ENG, ("(b - a for a, b in", "(2 * (b - a) for a, b in"))
        two = edit(ENG, ("(b - a for a, b in", "(3 * (b - a) for a, b in"))
        r = p.prepare({"id": "g1", "modules": {"toy.engine": {"source": one, "targets": tq}}})
        self.assertTrue(r["ok"], r["findings"])
        self.assertTrue(p.commit("g1")["ok"])
        r = p.prepare({"id": "g2", "modules": {"toy.engine": {"source": two, "targets": tq}}})
        self.assertNotIn("base_mismatch", [f["check"] for f in r["findings"]])
        self.assertTrue(r["ok"], r["findings"])


    def test_stage_findings_carry_the_checked_code_hash(self):
        """A standing accept matches on code_hash; a finding without one could never match."""
        name, eng, _ = fresh("hash")
        line = at(ENG, "def longest_stall")
        t = [["Cluster.longest_stall", line]]
        one = edit(ENG, ("(b - a for a, b in", "(2 * (b - a) for a, b in"))
        two = edit(ENG, ("(b - a for a, b in", "(3 * (b - a) for a, b in"))
        _, _, a = install(eng.__name__, one, t, J.Journal(), W.Witness())
        self.assertTrue(a["ok"], a)
        r = S.stage(eng.__name__, two, t)        # running code is `one`, not the file: base_mismatch
        k = next(iter(r["items"]))
        fs = [f for f in r["findings"] if f["check"] == "base_mismatch"]
        self.assertTrue(fs, r["findings"])
        self.assertEqual(fs[0]["code_hash"], r["items"][k]["new_hash"])


def temp_module(test, tag, text):
    """Write `text` as a new module in a temp directory and import it: (name, module)."""
    tmp = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, tmp, True)
    name = f"kt_{tag}{next(_n)}"
    with open(os.path.join(tmp, name + ".py"), "w") as f:
        f.write(text)
    sys.path.insert(0, tmp)
    test.addCleanup(sys.path.remove, tmp)
    test.addCleanup(sys.modules.pop, name, None)
    return name, importlib.import_module(name)


class RecordAttrTest(unittest.TestCase):
    """Pain point: `record_attr` read and restored only `__dict__` entries, so a repair that
    set a function's `__doc__`, a descriptor (torch `Parameter.data`) or an `nn.Module`
    submodule (kept in `_modules` by its `__setattr__`) was journaled as "added" and undo left
    the new value in place."""

    class Prop:
        def __init__(self):
            self._v = "old"

        @property
        def val(self):
            return self._v

        @val.setter
        def val(self, v):
            self._v = v

    class Side:
        """Stores attributes in a side dict through `__setattr__`, like nn.Module."""

        def __init__(self):
            object.__setattr__(self, "_side", {"sub": "old sub"})

        def __getattr__(self, n):
            try:
                return self.__dict__["_side"][n]
            except KeyError:
                raise AttributeError(n) from None

        def __setattr__(self, n, v):
            self._side[n] = v

        def __delattr__(self, n):
            del self._side[n]

    def test_undo_restores_descriptor_side_dict_and_doc(self):
        """Proves: a property, a name kept by a custom `__setattr__` (replaced and added), a function's `__doc__`, and a plain `__dict__` entry (replaced and added) all come back exactly after `record_attr` + assign + undo. Not: torch (next test)."""
        def fn():
            """old doc"""
        p, sd = self.Prop(), self.Side()

        class Plain:
            pass
        pl = Plain()
        pl.x = old_x = ["x"]
        j = J.Journal()
        for obj, name, value in [(p, "val", "new"), (sd, "sub", "new sub"),
                                 (sd, "extra", "added"), (fn, "__doc__", "new doc"),
                                 (pl, "x", ["new"]), (pl, "y", "added")]:
            j.record_attr(obj, name)
            setattr(obj, name, value)
        self.assertEqual((p.val, sd.sub, sd.extra, fn.__doc__, pl.y),
                         ("new", "new sub", "added", "new doc", "added"))
        u = j.undo_to(0)
        self.assertTrue(u["ok"], u)
        self.assertEqual((p.val, sd.sub, fn.__doc__), ("old", "old sub", "old doc"))
        self.assertEqual(sd._side, {"sub": "old sub"})        # the added name is gone
        self.assertIs(pl.x, old_x)
        self.assertNotIn("y", vars(pl))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_undo_restores_parameter_data_and_module_submodules(self):
        """Proves (CPU torch): `Parameter.data` swapped to another tensor comes back to the old storage and values, a replaced `nn.Module` submodule comes back as the same object, an added one is removed from `_modules`. Not: CUDA tensors, buffers."""
        import torch
        from torch import nn
        m = nn.Module()
        m.lin = old_lin = nn.Linear(2, 2)
        w = m.lin.weight
        ptr, old_vals = w.data_ptr(), w.detach().clone()
        j = J.Journal()
        j.record_attr(w, "data")
        w.data = torch.full((2, 2), 7.0)
        j.record_attr(m, "lin")
        m.lin = nn.Linear(2, 2)
        j.record_attr(m, "extra")
        m.extra = nn.ReLU()
        self.assertTrue(j.undo_to(0)["ok"])
        self.assertEqual(w.data_ptr(), ptr)
        self.assertTrue(torch.equal(w.detach(), old_vals))
        self.assertIs(m.lin, old_lin)
        self.assertEqual(list(m._modules), ["lin"])


class LiveClassBodyTest(unittest.TestCase):
    """Pain point: new defaults and added methods were evaluated in empty scratch classes, so a
    default naming a class-level constant got the module's value of the same name (or a
    NameError), and a method decorated by a helper defined in the class did not stage."""

    BASE = ('SCALE = 1\n\n\nclass K:\n    SCALE = 5\n    ONLY = 9\n\n'
            '    def _twice(fn):\n        def w(*a):\n            return 2 * fn(*a)\n'
            '        return w\n\n'
            '    def f(self, x=1):\n        return x\n\n'
            '    def g(self, y=2):\n        return y\n')

    def test_new_defaults_see_the_live_class_body(self):
        """Proves: `def f(self, x=SCALE)` takes the class's SCALE (5), not the module's (1), and `def g(self, y=ONLY)` (a name only the class has) evaluates instead of a defaults_not_evaluated finding; stage does not touch the live class. Not: defaults that read names defined below them in the class (the live class has every name)."""
        name, mod = temp_module(self, "defaults", self.BASE)
        new = edit(self.BASE, ("x=1", "x=SCALE"), ("y=2", "y=ONLY"))
        before = dict(vars(mod.K))
        r = S.stage(name, new)
        self.assertEqual(dict(vars(mod.K)), before)
        self.assertEqual(r["findings"], [])
        self.assertTrue(S.apply(r["staged"], J.Journal(), W.Witness())["ok"])
        self.assertEqual((mod.K().f(), mod.K().g()), (5, 9))

    def test_added_method_uses_a_decorator_defined_in_the_class(self):
        """Proves: an added method decorated by `_twice`, a function of the class body, stages without a finding, the class is unchanged until apply, and after apply it runs through the decorator (3 * 2 = 6). Not: decorators that mutate class-level objects in place (they run at stage)."""
        name, mod = temp_module(self, "deco", self.BASE)
        new = self.BASE + "\n    @_twice\n    def h(self):\n        return 3\n"
        before = dict(vars(mod.K))
        r = S.stage(name, new)
        self.assertEqual(dict(vars(mod.K)), before)
        self.assertTrue(r["ok"], r["findings"])
        self.assertEqual(r["findings"], [])
        self.assertTrue(S.apply(r["staged"], J.Journal(), W.Witness())["ok"])
        self.assertEqual(mod.K().h(), 6)


class CallerTreesTest(unittest.TestCase):
    """Pain point: a caller with its own planner (constants, fields) parses both texts, then
    stage parses them again. `stage(..., trees=(base, patch))` takes the caller's trees."""

    def test_stage_with_the_callers_trees_parses_nothing_and_stages_the_same(self):
        """Proves: for the toy edits of AutoTargetsTest, stage with the caller's trees calls `ast.parse` zero times, stages the same findings, items and code objects as with explicit targets (targets omitted, and a subset picked with `changed_targets(trees=)`), and gives the trees back unchanged (the same tree staged twice gives the same result). Not: the analyzer child (it parses its own copy)."""
        E, O = toy.source("engine"), toy.source("ops")
        cases = {
            "method": ("engine", edit(E, ('self.ran.append("base")', 'self.ran.append("v1")'),
                                      ("self.tokens.clear()", "self.tokens.clear()\n"
                                       "        self.steps += 0"))),
            "closure": ("ops", edit(O, ("return x + k", "return x + k * 1"))),
            "defaults": ("ops", edit(O, ("def scale(x):", "def scale(x, m=STEP_SCALE):"))),
            "added": ("engine", edit(E, ("class Gap:", "def helper(x):\n    return x\n\n\n"
                                         "class Gap:"),
                                     ('self.ran.append("base")',
                                      'self.ran.append(str(helper(1)))'))),
            "removed": ("engine", edit(E, ("    def drain(self):\n        self.tokens.clear()"
                                           "\n\n", ""))),
            "genexpr": ("engine", edit(E, ("(b - a for a, b in", "(2 * (b - a) for a, b in"))),
        }
        _summary = ProcessAnalyzerTest._summary
        real_parse, calls = ast.parse, []

        def counting(*a, **kw):
            calls.append(1)
            return real_parse(*a, **kw)
        for case, (m, src) in cases.items():
            _, eng, ops = fresh("trees")
            mod = (eng if m == "engine" else ops).__name__
            base = E if m == "engine" else O
            explicit = S.changed_targets(base, src)
            want = _summary(S.stage(mod, src, explicit))
            trees = (ast.parse(base), ast.parse(src))
            dumps = [ast.dump(t, include_attributes=True) for t in trees]
            ast.parse = counting
            try:
                got = [S.stage(mod, src, trees=trees), S.stage(mod, src, trees=trees)]
                sub = S.changed_targets(base, src, trees=trees)[:1]
                one = S.stage(mod, src, sub, trees=trees)
            finally:
                ast.parse = real_parse
            self.assertEqual(calls, [], case)
            for r in got:
                self.assertEqual((r["targets"], r["timings"]["trees"]), (explicit, "caller"))
                self.assertEqual(_summary(r), want, case)
            self.assertEqual(_summary(one), _summary(S.stage(mod, src, sub)), case)
            self.assertEqual([ast.dump(t, include_attributes=True) for t in trees], dumps,
                             case)


class UndoCallTest(unittest.TestCase):
    """Pain point: an in-place re-init (a Triton kernel rebuilt) had no undo entry, so undo
    needed a hand-run `repair()`."""

    def test_record_call_runs_in_reverse_order_and_a_failure_is_a_finding(self):
        """Proves: `record_call` entries run at their place in reverse order with the attribute entries (the first call sees the later attribute already undone), and one that raises is an `undo_incomplete` finding naming its label while the rest still run. Not: calls that take arguments (bind them in a closure)."""
        class O:
            pass
        o, log = O(), []
        j = J.Journal()
        j.set_attr(o, "a", 1)
        j.record_call(lambda: log.append(("k1", vars(o).get("b"))), "rebuild k1")

        def boom():
            raise RuntimeError("kernel cache gone")
        j.set_attr(o, "b", 2)
        j.record_call(boom, "rebuild k2")
        j.record_call(lambda: log.append(("k3", vars(o).get("b"))), "rebuild k3")
        u = j.undo_to(0)
        self.assertEqual(log, [("k3", 2), ("k1", None)])
        self.assertEqual(vars(o), {})
        self.assertFalse(u["ok"])
        self.assertEqual([f["check"] for f in u["findings"]], ["undo_incomplete"])
        self.assertIn("rebuild k2", u["findings"][0]["observed"])
        self.assertIn("kernel cache gone", u["findings"][0]["observed"])


class RuntimeClosureUndoTest(unittest.TestCase):
    """Pain point: after a swap of `make_shift`, closures its new code made while installed
    are in no journal entry; undo restored `make_shift` and left those closures running the
    new inner code for whoever held them."""

    def test_closure_made_after_the_swap_runs_the_old_code_after_undo(self):
        """Proves: a closure made by the swapped `make_shift` after apply (shift(1) = 1 + 1 + 100) runs the base inner code after undo (2), counted in `closures_restored`; the engine's closure made before the swap is restored too. Not: closures held only by a suspended frame's locals of the new code (they are still found: the scan is by referrer)."""
        _, eng, ops = fresh("rtclosure")
        e = eng.Engine()
        new = edit(OPS, ("        return x + k\n", "        return x + k + 100\n"))
        j = J.Journal()
        r, d, a = install(ops.__name__, new, "auto", j, W.Witness())
        self.assertTrue(a and a["ok"], (r["findings"], d))
        held = ops.make_shift(1)
        self.assertEqual((held(1), e.shift(1)), (102, 111))
        u = j.undo_to(0)
        self.assertTrue(u["ok"], u)
        self.assertEqual((held(1), e.shift(1), ops.make_shift(1)(1)), (2, 11, 2))
        self.assertEqual((u["closures_restored"], u["closure_codes_scanned"]), (1, 1))

    def test_undo_skips_the_heap_pass_when_no_closure_outlives_the_call(self):
        """Proves: when the new `make_shift` made no closure that is still alive (one made and dropped, none kept), undo runs no heap pass (`closure_codes_scanned` 0, no `closure_scan_s`) and still restores the code, the engine's pre-swap closure included. On a loaded vLLM engine core the pass took 0.11-0.15 s inside the undo gap for a `Scheduler.schedule` swap that made no closure (our 2026-11-05 GPU run). Not: a reference count that drops elsewhere while a closure is made (the kernel's own holders stay until the undo is over)."""
        _, eng, ops = fresh("rtclosure3")
        e = eng.Engine()
        new = edit(OPS, ("        return x + k\n", "        return x + k + 100\n"))
        j = J.Journal()
        r, d, a = install(ops.__name__, new, "auto", j, W.Witness())
        self.assertTrue(a and a["ok"], (r["findings"], d))
        self.assertEqual((ops.make_shift(1)(1), e.shift(1)), (102, 111))   # made and dropped
        u = j.undo_to(0)
        self.assertTrue(u["ok"], u)
        self.assertEqual((u["closures_restored"], u["closure_codes_scanned"]), (0, 0))
        self.assertNotIn("closure_scan_s", u)
        self.assertEqual((e.shift(1), ops.make_shift(1)(1)), (11, 2))

    def test_closure_with_no_base_version_is_a_finding(self):
        """Proves: a closure from a nested def the edit added (`shift2`) cannot go back to base code; undo restores `make_shift` and reports `closure_not_restored` with the count and the qualname, and is not ok. Not: what the holder should do next (rebuild or drop it)."""
        _, eng, ops = fresh("rtclosure2")
        new = edit(OPS, ("        return x + k\n    return shift\n",
                         "        return x + k\n\n    def shift2(x):\n        return x - k\n"
                         "    return shift2\n"))
        j = J.Journal()
        r, d, a = install(ops.__name__, new, "auto", j, W.Witness())
        self.assertTrue(a and a["ok"], (r["findings"], d))
        held = [ops.make_shift(1), ops.make_shift(2)]
        self.assertEqual([h(5) for h in held], [4, 3])
        u = j.undo_to(0)
        self.assertFalse(u["ok"])
        self.assertEqual([(f["check"], f["subject"]) for f in u["findings"]],
                         [("closure_not_restored", "make_shift.<locals>.shift2")])
        self.assertIn("2 function object(s)", u["findings"][0]["observed"])
        self.assertEqual(ops.make_shift(1)(5), 6)

    def test_a_partial_undo_stays_partial_when_undone_again(self):
        """Proves: after an undo that leaves closures of an added nested def running (`closure_not_restored`, not ok), the participant says "partial", and a second undo is not ok with the same finding; through a coordinator the transaction's state is "partial" too. Before, the second undo found no mark and answered ok with `undone` 0 and state "undone" (our 2026-11-05 GPU run, engine core), while the closure still ran the new code. Not: what to do with the closure (rebuild its holder or respawn)."""
        name, eng, ops = fresh("rtclosure4")
        new = edit(OPS, ("        return x + k\n    return shift\n",
                         "        return x + k\n\n    def shift2(x):\n        return x - k\n"
                         "    return shift2\n"))
        p = C.LocalParticipant("core", alias={"toy": name})
        self.assertTrue(p.prepare({"id": "a", "modules": {"toy.ops": {"source": new}}})["ok"])
        self.assertTrue(p.commit("a")["ok"])
        held = ops.make_shift(1)
        u = p.undo("a")
        self.assertEqual((u["ok"], [f["check"] for f in u["findings"]]),
                         (False, ["closure_not_restored"]))
        self.assertEqual(p.status("a")["state"], "partial")
        u2 = p.undo("a")
        self.assertEqual((u2["ok"], [f["check"] for f in u2["findings"]]),
                         (False, ["closure_not_restored"]))
        self.assertEqual((p.status("a")["state"], held(5), ops.make_shift(1)(5)), ("partial", 4, 6))

        name, eng, ops = fresh("rtclosure5")
        co = C.Coordinator([C.LocalParticipant("core", alias={"toy": name})])
        res = co.run({"modules": {"toy.ops": {"source": new}}})
        self.assertEqual(res["outcome"], "committed")
        held = ops.make_shift(1)
        for _ in range(2):
            u = co.undo(res["txn"])
            self.assertEqual((u["ok"], u["receipt"]["status"]), (False, "partial"))
        self.assertEqual(co.status(res["txn"])["state"], "partial")


if __name__ == "__main__":
    unittest.main()


class CaptureWitnessTest(unittest.TestCase):
    """Pain point: after a CUDA-graph recapture (TP2, Qwen3.5-35B-A3B MoE) the witness read
    13280 calls per rank and no zero_witness, yet every call was made while graphs were
    captured: a replay runs no Python, so the counter proved capture, not a run."""

    def test_capture_calls_are_counted_apart_from_live_calls(self):
        """Input: Engine.step swapped; 3 calls while the engine says it captures (the
        participant's `capturing` probe), 2 more inside `witness.capture()` (a recapture's
        warm-up runs), then the proof. Expected: live 0, capture 5, and a zero_witness routed
        to R1 whose note asks for a device-side probe; the proof receipt says "ran during
        capture 5, ran live 0". After 2 live calls: no finding, "ran during capture 5, ran
        live 2". Not: CUDA itself (vLLM asks torch whether the stream is capturing)."""
        name, eng, _ = fresh("capwit")
        capturing = [False]
        p = C.LocalParticipant("workers", alias={"toy": name}, capturing=lambda: capturing[0])
        co = C.Coordinator([p], phase2_deadline_s=None)
        k = f"toy.engine:Engine.step:{at(ENG, 'def step')}"
        r = co.run({"id": "cw", "modules": {"toy.engine": {
            "source": edit(ENG, ('self.ran.append("base")', 'self.ran.append("v1")')),
            "targets": [["Engine.step", at(ENG, "def step")]]}}})
        self.assertEqual(r["outcome"], "committed")
        e = eng.Engine()
        capturing[0] = True
        for _ in range(3):
            e.step(1)
        capturing[0] = False
        with p.witness.capture():
            e.step(1)
            e.step(1)
        pr = co.prove("cw")["receipt"]
        self.assertEqual((pr["witness"], pr["witness_capture"], pr["ran"]),
                         ({"workers": {k: 0}}, {"workers": {k: 5}},
                          {"workers": {k: "ran during capture 5, ran live 0"}}))
        (f,) = pr["findings"]
        self.assertEqual((f["check"], f.get("route")), ("zero_witness", "R1"))
        self.assertIn("5 calls while graphs were captured", f["observed"])
        self.assertIn("device-side probe", f["note"])
        e.step(1)
        e.step(1)
        pr = co.prove("cw")["receipt"]
        self.assertEqual((pr["findings"], pr["ran"]),
                         ([], {"workers": {k: "ran during capture 5, ran live 2"}}))

    def test_the_witness_is_a_constant_for_dynamo(self):
        """Input: a swap of f(x) = x*2 to x*3 applied while torch is loaded; then
        torch.compile (eager backend, fullgraph) of a caller of f, called 3 times. Expected: the witness hook is marked with
        assume_constant_result, so Dynamo calls it once while tracing and the compiled calls
        never (1 call, not 4). Pain point: a recompile failed in Dynamo on the traced counter
        ("modifying buffers ... cudagraph inside the compiler"). Not: vLLM's compiler backend
        or CUDA graphs (the GPU queue job)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        mod = f"kt{next(_n)}_dyn"
        base = "def f(x):\n    return x * 2\n"
        with open(os.path.join(d, mod + ".py"), "w") as fh:
            fh.write(base)
        sys.path.insert(0, d)
        self.addCleanup(sys.path.remove, d)
        m = importlib.import_module(mod)
        w, j = W.Witness(), J.Journal("t")
        r, dec, a = install(mod, base.replace("x * 2", "x * 3"), [["f", 1]], j, w)
        self.assertTrue(a and a["ok"], (r["findings"], dec))

        def caller(x):
            return m.f(x) + 1
        torch._dynamo.reset()
        g = torch.compile(caller, backend="eager", fullgraph=True)
        x = torch.ones(2)
        outs = [g(x) for _ in range(3)]
        self.assertEqual(outs[-1].tolist(), [4.0, 4.0])
        self.assertEqual(w.counts, {f"{mod}:f:1": 1})
        self.assertTrue(getattr(w.hook, "_dynamo_marked_constant", False))
        self.assertIs(m.__fastrsi_hit__, w.hook)


class RebuildTest(unittest.TestCase):
    """Pain point: a model-code edit in graph mode committed, then the lab had to recapture by
    hand over another RPC: commit and recapture were not one transaction, a failed recapture
    left the edit committed, and undo did not recapture back."""

    def _host(self, name, fail_on=()):
        calls = []

        def r1(txn, why, deadline_at):
            calls.append((txn, why, deadline_at is not None))
            if why in fail_on:
                return F.not_ok(f"capture failed ({why})")
            return F.ok(how="recapture", graphs=3)

        def captured(module, items):
            return [{**F.finding("captured_target", "runs only in graphs", "runs eagerly",
                                 subject=k), "route": "R1"} for k in items]
        p = C.LocalParticipant("workers", alias={"toy": name}, checks=[captured])
        co = C.Coordinator([p], phase2_deadline_s=None, rebuilds={"R1": r1})
        return co, calls

    def _plan(self, tid, **kw):
        return {"id": tid, "accept": ["captured_target"], **kw, "modules": {"toy.engine": {
            "source": edit(ENG, ('self.ran.append("base")', 'self.ran.append("v1")')),
            "targets": [["Engine.step", at(ENG, "def step")]]}}}

    def test_an_accepted_route_rebuilds_after_the_commit_and_after_undo(self):
        """Input: a plan whose captured_target (route R1) is accepted, on a host that
        registers R1. Expected: R1 runs once after the commit (why "commit", with a
        deadline), the txn receipt has rebuild.ok, recapture_s, actual_recipes with "R1
        recapture" and edit_to_run_s >= edit_to_commit_s; undo runs R1 again (why "undo") and
        its receipt has recapture_s. A plan with "rebuild": [] does not rebuild. Not: the
        engine's recapture itself (adapter tests and the GPU job)."""
        name, eng, _ = fresh("r1")
        co, calls = self._host(name)
        r = co.run(self._plan("r1"))
        rec = r["receipt"]
        self.assertEqual((r["outcome"], calls), ("committed", [("r1", "commit", True)]))
        self.assertTrue(rec["rebuild"]["ok"])
        self.assertEqual(rec["rebuild"]["results"]["R1"]["how"], "recapture")
        self.assertIsNotNone(rec["recapture_s"])
        self.assertIn("R1 recapture", rec["actual_recipes"])
        self.assertGreaterEqual(rec["edit_to_run_s"], rec["edit_to_commit_s"])
        e = eng.Engine()
        e.step(1)
        self.assertEqual(e.ran[-1], "v1")
        u = co.undo("r1")
        self.assertEqual((u["ok"], calls[-1]), (True, ("r1", "undo", True)))
        self.assertIsNotNone(u["receipt"]["recapture_s"])
        e.step(1)
        self.assertEqual(e.ran[-1], "base")
        n = len(calls)
        r = co.run(self._plan("r1-none", rebuild=[]))
        self.assertEqual((r["outcome"], len(calls), r["receipt"]["rebuild"]),
                         ("committed", n, None))

    def test_a_failed_rebuild_undoes_the_transaction_and_rebuilds_back(self):
        """Input: the same plan, R1 failing after the commit. Expected: outcome rolled_back,
        a not-overridable rebuild_failed finding, the code back at base, and R1 run again
        from the base code (why "rollback"); a plan naming a route the host lacks aborts
        before phase 2 with rebuild_failed. Not: a rebuild that hangs (its deadline is the
        host's)."""
        name, eng, _ = fresh("r1fail")
        co, calls = self._host(name, fail_on=("commit",))
        r = co.run(self._plan("r1f"))
        rec = r["receipt"]
        self.assertEqual(r["outcome"], "rolled_back")
        self.assertEqual([w for _, w, _ in calls], ["commit", "rollback"])
        f = [x for x in rec["findings"] if x["check"] == "rebuild_failed"]
        self.assertEqual([(x["subject"], x["overridable"]) for x in f], [("commit", False)])
        self.assertEqual((rec["rollback"]["status"], rec["rebuild"]["back"]["ok"]),
                         ("undone", True))
        self.assertEqual(co.status("r1f")["state"], "aborted")
        e = eng.Engine()
        e.step(1)
        self.assertEqual(e.ran[-1], "base")
        r = co.run(self._plan("r9", rebuild=["R9"]))
        self.assertEqual((r["outcome"], [x["check"] for x in r["receipt"]["findings"]
                                         if x["check"] == "rebuild_failed"]),
                         ("aborted", ["rebuild_failed"]))
