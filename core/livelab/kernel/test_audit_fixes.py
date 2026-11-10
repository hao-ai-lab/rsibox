"""Regression tests for the kernel rows of the 2026-11-10 repo audit (#8, #9, #10, #18, #19,
#20, #21, #23). Each test failed on main (f35870f) and passes with the fix; the docstring says
what it saw before. CPU, standard library only (the JIT cases use test_jit's stand-in runtime;
the GPU runs of the same fixes are summarized in the PR description).

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
"""
import os
import shutil
import tempfile
import threading
import time
import unittest

from . import coordinator as C
from . import findings as F
from . import locate as L
from . import receipt as R
from . import witness as W
from .test_detectors import temp_modules
from .test_jit import findings as jit_findings, install as jit_install, lab
from .test_locate import ENG, OPS, at, edit, fresh
from .test_participant import step_target
from .test_witness import wait_for

V1 = {"source": edit(ENG, ("return y", "return y * 100")), "base_hash": L.text_hash(ENG),
      "targets": [["Engine.step", at(ENG, "def step")]]}
SLEEP_CELL = {"cell": "import time\ntime.sleep(0.4)\n", "declares": []}


def timed_quiesce(timeout, delay=0.0):
    """A quiesce point that runs fn on another thread (the engine's loop) and gives up after
    `timeout` s with TimeoutError, as a real quiesce with a bound does; the job keeps going.
    `delay`: the loop reaches its boundary only after that long."""
    threads = []

    def quiesce(fn):
        box, done = {}, threading.Event()

        def loop():
            time.sleep(delay)
            box["r"] = fn()
            done.set()
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        threads.append(th)
        if not done.wait(timeout):
            raise TimeoutError(f"no quiesce point within {timeout} s")
        return box["r"]
    quiesce.threads = threads
    return quiesce


def settle(q):
    for th in q.threads:
        th.join(5)


class CommitUndoRaceTest(unittest.TestCase):
    """#8: the host and every rank agree on a commit whose quiesce timed out mid-apply."""

    def _ranks(self, tag, q1):
        out = []
        for pname, q in (("rank0", None), ("rank1", q1)):
            name, eng, _ = fresh(tag + pname)
            out.append((C.LocalParticipant(pname, alias={"toy": name}, quiesce=q), eng,
                        eng.Engine()))
        return out

    def test_quiesce_timeout_inside_apply_leaves_host_and_rank_agreeing(self):
        """Input: rank1's quiesce gives up after 0.1 s while its commit is inside apply (the
        patch installed, a repair cell still running for 0.4 s). Expected: outcome
        rolled_back, and once rank1's job finishes it runs base code, its state is not
        committed and its journal is empty: host and rank agree. Before: rank1 finished the
        commit after the host's undo said "nothing committed": rank1 committed (step(3) =
        1700), host rolled_back, not poisoned."""
        q = timed_quiesce(0.1)
        ranks = self._ranks("race", q)
        res = C.Coordinator([p for p, _, _ in ranks], phase2_deadline_s=None).run(
            {"modules": {"toy.engine": V1}, "repairs": [SLEEP_CELL]})
        settle(q)
        self.assertEqual(res["outcome"], "rolled_back")
        p1, _, e1 = ranks[1]
        self.assertNotEqual(p1._state[res["txn"]], "committed")
        self.assertEqual((e1.step(3), p1.journal.entries), (17, []))
        self.assertEqual(ranks[0][2].step(3), 17)
        self.assertTrue(p1.at_base()["ok"], p1.at_base())

    def test_commit_whose_caller_gave_up_after_it_finished_reports_committed(self):
        """Input: rank1's quiesce returns the job's result late (the job finished, then the
        caller timed out before reading it). Expected: the commit reply still says what
        happened (ok, committed), so the host never undoes a rank it believes uncommitted
        while the rank stays committed. The participant's state and the reply agree."""
        name, eng, _ = fresh("race2")
        p = C.LocalParticipant("rank1", alias={"toy": name})
        plan = {"id": "t2", "modules": {"toy.engine": V1}}
        self.assertTrue(p.prepare(plan)["ok"])

        def late(fn):
            fn()
            raise TimeoutError("caller gave up")
        p.quiesce = late
        r = p.commit("t2")
        self.assertEqual((r["ok"], p._state["t2"]), (True, "committed"), r)
        self.assertEqual(eng.Engine().step(3), 1700)


class UndoBookkeepingTest(unittest.TestCase):
    """#18: undo order, unknown ids, repeated undo, cascades and a timed-out undo."""

    def test_cascade_reaches_a_transaction_at_the_same_mark(self):
        """Input: txn A is a repair cell that journals nothing, then txn B swaps step(); A and
        B have the same journal mark. Undo of A. Expected: B is undone too (also_undone) and
        not left "committed". Before: `m > mark` skipped B; it stayed committed while its
        code was undone."""
        name, eng, _ = fresh("samemark")
        p = C.LocalParticipant("rank0", alias={"toy": name})
        for txn, plan in (("A", {"repairs": [{"cell": "x = 1", "declares": []}]}),
                          ("B", {"modules": {"toy.engine": V1}})):
            self.assertTrue(p.prepare({"id": txn, **plan})["ok"])
            self.assertTrue(p.commit(txn)["ok"], txn)
        self.assertEqual(p._marks["A"], p._marks["B"])
        u = p.undo("A")
        self.assertEqual((u["ok"], u.get("also_undone"), p._state["B"]),
                         (True, ["B"], "undone"), u)
        self.assertEqual(eng.Engine().step(3), 17)

    def test_host_refuses_unknown_and_repeated_undo(self):
        """Input: undo("never-existed"); a committed transaction undone twice. Expected: both
        refused (not ok, no receipt written), the first undo ok. Before: both returned ok,
        status undone, and appended an undo receipt."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "r.jsonl")
        name, _, _ = fresh("undo2")
        co = C.Coordinator([C.LocalParticipant("rank0", alias={"toy": name})],
                           R.ReceiptWriter(path))
        u0 = co.undo("never-existed")
        self.assertEqual((u0["ok"], u0["status"], u0["receipt"]), (False, "refused", None))
        res = co.run({"modules": {"toy.engine": V1}})
        self.assertEqual(res["outcome"], "committed")
        self.assertTrue(co.undo(res["txn"])["ok"])
        u2 = co.undo(res["txn"])
        self.assertEqual((u2["ok"], u2["status"]), (False, "refused"), u2)
        self.assertEqual([r["kind"] for r in R.read(path)], ["txn", "undo"])

    def test_host_marks_cascaded_transactions_undone(self):
        """Input: txn A then txn B (chained on the same module), undo A. Expected: host
        status(B) is undone, like the participant's. Before: the host said B committed."""
        name, _, _ = fresh("casc")
        co = C.Coordinator([C.LocalParticipant("rank0", alias={"toy": name})])
        a = co.run({"modules": {"toy.engine": V1}})
        b_src = edit(V1["source"], ("return y * 100", "return y * 1000"))
        b = co.run({"modules": {"toy.engine": {"source": b_src,
                                               "targets": [["Engine.step", at(ENG, "def step")]]}}})
        self.assertEqual((a["outcome"], b["outcome"]), ("committed", "committed"), b)
        self.assertTrue(co.undo(a["txn"])["ok"])
        self.assertEqual(co.status(b["txn"])["state"], "undone")

    def test_undo_whose_quiesce_timed_out_never_runs_unrecorded(self):
        """Input: a committed txn; the undo's quiesce gives up before the job starts (the job
        would start 0.3 s later). Expected: the undo is not ok and is cancelled: the code and
        the state stay committed together. Before: the job ran later, undoing the code while
        the state still said committed."""
        name, eng, _ = fresh("undoq")
        p = C.LocalParticipant("rank0", alias={"toy": name})
        self.assertTrue(p.prepare({"id": "t", "modules": {"toy.engine": V1}})["ok"])
        self.assertTrue(p.commit("t")["ok"])
        q = timed_quiesce(0.05, delay=0.3)
        p.quiesce = q
        self.assertFalse(p.undo("t")["ok"])
        settle(q)
        st, out = p._state["t"], eng.Engine().step(3)
        self.assertEqual((st, out), ("committed", 1700))


class DeadlineTest(unittest.TestCase):
    """#19: a phase-2 deadline that has passed is not re-armed."""

    def test_gap_is_bounded_by_the_deadline(self):
        """Input: 6 participants whose validate takes 0.3 s each, phase2_deadline_s=0.4.
        Expected: gap_s at most the deadline plus scheduling slack. Before: each call past
        the deadline got another d/2, gap_s 2.1 s."""
        ps = []
        for i in range(6):
            name, _, _ = fresh(f"dl{i}")
            lp = C.LocalParticipant(f"r{i}", alias={"toy": name})

            def rpc(method, payload, lp=lp):
                if method == "validate":
                    time.sleep(0.3)
                return getattr(lp, method)(payload)
            ps.append(C.CallParticipant(f"r{i}", rpc))
        co = C.Coordinator(ps, phase2_deadline_s=0.4)
        res = co.run({"modules": {"toy.engine": V1}})
        self.assertLess(res["receipt"]["gap_s"], 0.4 + 0.15, res["receipt"]["gap_s"])
        self.assertEqual(res["outcome"], "aborted")


class TimingTest(unittest.TestCase):
    """#20: spans on the monotonic clock; edit_to_run_s is the first execution."""

    def test_spans_ignore_wall_clock_jumps(self):
        """Input: a host clock that jumps 100 s at every read (NTP step, suspend). Expected:
        prepare_s, gap_s, edit_to_commit_s and undo_s stay well under a second. Before: each
        was a multiple of 100 s."""
        jump = [time.time()]

        def clock():
            jump[0] += 100
            return jump[0]
        name, _, _ = fresh("jump")
        co = C.Coordinator([C.LocalParticipant("rank0", alias={"toy": name})], clock=clock)
        rec = co.run({"modules": {"toy.engine": V1}})["receipt"]
        u = co.undo(rec["id"])["receipt"]
        for k in ("prepare_s", "gap_s", "edit_to_commit_s"):
            self.assertLess(rec[k], 1.0, (k, rec[k]))
        self.assertLess(u["undo_s"], 1.0)

    def test_edit_to_run_is_the_first_execution(self):
        """Input: commit, wait 0.3 s, then run one step; prove. Expected: the txn receipt
        does not claim a run (edit_to_run_s null, edit_to_commit_s set); the proof's
        edit_to_run_s is at least 0.3 s and at least edit_to_commit_s. Before: edit_to_run_s
        was edit_to_commit_s, before anything ran."""
        name, eng, _ = fresh("first")
        co = C.Coordinator([C.LocalParticipant("rank0", alias={"toy": name})])
        res = co.run({"modules": {"toy.engine": V1}})
        rec = res["receipt"]
        self.assertIsNone(rec["edit_to_run_s"])
        self.assertIsNotNone(rec["edit_to_commit_s"])
        time.sleep(0.3)
        eng.Engine().step(3)
        pr = co.prove(res["txn"])["receipt"]
        self.assertGreaterEqual(pr["edit_to_run_s"], 0.3)
        self.assertGreaterEqual(pr["edit_to_run_s"], rec["edit_to_commit_s"])


class WitnessAcrossRanksTest(unittest.TestCase):
    """#21: per-rank minimum; a failing capture probe never counts a live call."""

    def test_restart_witness_needs_every_rank(self):
        """Input: rank0 0 calls, rank1 7 calls. Expected: proven False (rank0 never ran it),
        witness the per-rank minimum. Before: summed to 7, proven True."""
        s = W.summarize_restart_witness({"rank0": {"f": 0}, "rank1": {"f": 7}})
        self.assertEqual((s["proven"], s["witness"], s["finding"]),
                         (False, {"f": 0}, "zero_witness"))
        self.assertEqual(s["per_rank"], {"rank0": {"f": 0}, "rank1": {"f": 7}})
        s2 = W.summarize_restart_witness([{"f": 2}, {"f": 5}])
        self.assertEqual((s2["proven"], s2["witness"]), (True, {"f": 2}))

    def test_probe_error_fails_closed(self):
        """Input: a `capturing` probe that raises. Expected: the call is not live (counts 0,
        a probe error recorded), check gives zero_witness. Before: counted live."""
        def probe():
            raise RuntimeError("no stream")
        w = W.Witness(capturing=probe)
        w.hit("k")
        self.assertEqual((w.counts.get("k", 0), w.probe_errors.get("k")), (0, 1))
        self.assertEqual([f["check"] for f in w.check(["k"])["findings"]], ["zero_witness"])


NESTED = '''def make(k):
    def inner(x):
        return x + k
    return inner


F = make(1)


def use(x):
    return F(x)
'''


class NestedSignatureTest(unittest.TestCase):
    """#10: a nested function's new signature or defaults cannot reach live closures."""

    def test_nested_default_change_is_blocking(self):
        """Input: `make.<locals>.inner(x)` -> `inner(x, extra=0)` (a live closure F exists).
        Expected: a blocking finding, nothing installed, F still works. Before: committed
        with no finding; F got the new code with the old (no) defaults: TypeError."""
        (mod,) = temp_modules(self, {"kt_nested_sig": NESTED})
        p = C.LocalParticipant("rank0")
        co = C.Coordinator([p])
        new = NESTED.replace("def inner(x):\n        return x + k",
                             "def inner(x, extra=0):\n        return x + k + extra")
        res = co.run({"modules": {"kt_nested_sig": {"source": new, "targets": [
            ["make.<locals>.inner", at(NESTED, "def inner")]]}}})
        self.assertEqual(res["outcome"], "aborted", res)
        checks = [f["check"] for f in res["receipt"]["findings"]]
        self.assertIn("nested_signature", checks)
        self.assertFalse(F.CATALOG["nested_signature"][4])     # not overridable
        self.assertEqual(mod.use(2), 3)


REGMOD = '''REG = []


def register(fn):
    REG.append(fn.__name__)
    return fn


def f(x):
    return x + 1
'''


class StageSideEffectTest(unittest.TestCase):
    """#23: prepare runs no decorator of an added definition."""

    def test_added_decorator_runs_at_commit_not_at_prepare(self):
        """Input: a patch adding `@register def g` (register appends to REG) and making f
        call g. Expected: REG empty after prepare and after an abort; ['g'] after a commit;
        f(3) == 7 after commit. Before: prepare ran the decorator (REG == ['g']) and abort
        left it."""
        (mod,) = temp_modules(self, {"kt_stage_reg": REGMOD})
        new = REGMOD.replace("def f(x):\n    return x + 1",
                             "@register\ndef g(x):\n    return x * 2\n\n\n"
                             "def f(x):\n    return g(x) + 1")
        p = C.LocalParticipant("rank0")
        plan = {"id": "s1", "modules": {"kt_stage_reg": {"source": new}}}
        self.assertTrue(p.prepare(plan)["ok"], p)
        self.assertEqual(mod.REG, [])
        p.abort("s1")
        self.assertEqual(mod.REG, [])
        self.assertTrue(p.prepare({**plan, "id": "s2"})["ok"])
        self.assertTrue(p.commit("s2")["ok"])
        self.assertEqual((mod.REG, mod.f(3)), (["g"], 7))
        self.assertTrue(p.undo("s2")["ok"])
        self.assertFalse(hasattr(mod, "g"))


KA = '''import {t} as triton


@triton.jit
def helper(v):
    return v + 1


@triton.jit
def kernel_a(x, n):
    return [helper(v) for v in x[:n]]


def launch(x):
    return kernel_a[(1,)](x, len(x))
'''

KB = '''import {t} as triton


@triton.jit
def helper(v):
    return v + 2


@triton.jit
def kernel_b(x, n):
    return [helper(v) for v in x[:n]]


def launch(x):
    return kernel_b[(1,)](x, len(x))
'''


class TritonWitnessIdentityTest(unittest.TestCase):
    """#9: callers by identity; per-kernel launch keys."""

    def test_same_named_kernel_in_another_module_is_not_a_caller(self):
        """Input: modules A and B each define `@triton.jit helper` and a kernel calling its
        own helper; edit A.helper, launch only B.kernel_b. Expected: A.helper's witness 0
        (zero_witness), B's kernels not in the rebuild's callers, B's caches untouched.
        Before: B.kernel_b was taken as a caller by bare name and its launch read
        "ran live 1" for A.helper."""
        names, texts, m, p, sess = lab(self, {"a": KA, "b": KB})
        m["a"].launch([1])
        m["b"].launch([1])
        b_cache = dict(m["b"].kernel_b.device_caches["cpu"][0])
        r = jit_install(sess, names["a"], texts["a"].replace("v + 1", "v + 10"))
        self.assertEqual(r["outcome"], "committed", r)
        rebuild = [f for f in jit_findings(r) if f["check"] == "jit_rebuild"][0]
        self.assertNotIn(names["b"], rebuild["observed"])
        self.assertEqual(dict(m["b"].kernel_b.device_caches["cpu"][0]), b_cache)
        self.assertEqual(m["b"].launch([1]), [3])
        pr = sess.co.prove(r["txn"])
        key = L.key_of(names["a"], "helper", at(texts["a"], "@triton.jit"))
        self.assertEqual(pr["receipt"]["witness"]["rank0"][key], 0)
        self.assertIn("zero_witness", [f["check"] for f in pr["receipt"]["findings"]])
        self.assertTrue(sess.uninstall()["ok"])

    def test_each_launching_kernel_has_its_own_key(self):
        """Input: edit A.helper, launch A.kernel_a twice. Expected: the helper's witness is
        2 and the per-kernel launch counts name A.kernel_a (2), not A.helper (0)."""
        names, texts, m, p, sess = lab(self, {"a": KA, "b": KB})
        r = jit_install(sess, names["a"], texts["a"].replace("v + 1", "v + 10"))
        self.assertEqual(r["outcome"], "committed", r)
        m["a"].launch([1])
        m["a"].launch([1])
        key = L.key_of(names["a"], "helper", at(texts["a"], "@triton.jit"))
        wc = p.witness_check(r["txn"])
        self.assertEqual(wc["counts"][key], 2)
        self.assertEqual(wc["launches"][key], {f"{names['a']}.helper": 0,
                                               f"{names['a']}.kernel_a": 2})
        self.assertTrue(sess.uninstall()["ok"])


if __name__ == "__main__":
    unittest.main()
