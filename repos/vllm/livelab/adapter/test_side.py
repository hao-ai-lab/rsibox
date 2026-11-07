"""CPU golden tests of the side (background prepare, per-rank deadline, version log) and of
graph detection (no vLLM, no GPU). Run as test_host.py says. Workers are stand-ins with only
the config fields graphs.graph_mode reads; Side, WorkerGroup, Host and the kernel are real.
"""
import itertools
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace as NS

from kernel import locate as L
from kernel import toy

from . import worker_ext as WE
from .host import WorkerGroup
from .side import Side
from .test_host import ENG, STEP_LINE, make_host, plan, ran

_n = itertools.count()


def own_plan(tag, faults=None, targets=True):
    """A plan on a fresh toy copy of its own (module `<name>.engine`)."""
    name = f"ads{next(_n)}_{tag}"
    toy.load_copy(name)
    spec = {"on": ["worker"], "base_hash": L.text_hash(ENG),
            "source": ENG.replace('self.ran.append("base")', 'self.ran.append("v1")')}
    if targets:
        spec["targets"] = [["Engine.step", STEP_LINE]]
    return {"id": f"txn-{name}", "faults": faults or {}, "modules": {f"{name}.engine": spec}}


def worker(cudagraph="FULL_AND_PIECEWISE", compile_mode="VLLM_COMPILE", eager=False,
           splitting_ops=()):
    """A stand-in for vLLM's Worker: only the config fields graphs.graph_mode reads."""
    return NS(vllm_config=NS(model_config=NS(enforce_eager=eager), compilation_config=NS(
        cudagraph_mode=NS(name=cudagraph), mode=NS(name=compile_mode),
        splitting_ops=list(splitting_ops), max_cudagraph_capture_size=512)))


def prepared(txn, timeout=5.0):
    """Wait until no background prepare of txn is running in this process."""
    end = time.time() + timeout
    while any(t.name == f"fastrsi-prepare-{txn}" for t in threading.enumerate()):
        if time.time() > end:
            raise AssertionError(f"prepare of {txn} still running")
        time.sleep(0.005)


class PrepareAndDeadlineTest(unittest.TestCase):

    def test_ranks_that_finish_early_keep_their_vote(self):
        """Input: 4 ranks, rank 3 preparing 0.3 s longer. Expected: the group polls every
        rank until all are done and gets four yes votes; a poll after that repeats the result;
        after abort the result is gone ("no prepare started"). Pain point: ranks 0-2 answered
        "no prepare started" on the polls after their first finished one, a false abort (1 of
        33 live TP4 prepares in the prototype). Not: the real collective_rpc."""
        sides = [Side("worker", rank=r) for r in range(4)]
        from .test_host import Executor
        g = WorkerGroup(NS(core=NS(model_executor=Executor(sides)), in_loop=lambda fn: fn()))
        p = own_plan("slow3", {"worker:3": {"method": "prepare", "delay_s": 0.3, "fail": False}})
        t0 = time.perf_counter()
        r = g.prepare(p)
        self.assertEqual([(x["rank"], x["ok"]) for x in r["ranks"]],
                         [(0, True), (1, True), (2, True), (3, True)])
        self.assertGreaterEqual(time.perf_counter() - t0, 0.3)
        self.assertGreater(g.host.core.model_executor.calls.count("prepare_poll"), 2)
        self.assertEqual(g._rpc("prepare_poll", p["id"])["ok"], True)
        self.assertTrue(g.abort(p["id"])["ok"])
        self.assertEqual({x["error"] for x in g._rpc("prepare_poll", p["id"])["ranks"]},
                         {f"no prepare started for {p['id']}"})

    def test_a_no_vote_waits_for_the_rank_still_preparing(self):
        """Input: 2 ranks; rank 0 vetoes its prepare at once, rank 1 takes 0.3 s longer to
        stage. Expected: the group's prepare returns only once rank 1 is done, with both
        final votes (rank 0 no, rank 1 yes, none pending); after the abort rank 1 keeps no
        staged transaction and watches nothing. Fails before the fix: the group returned at
        the first poll with rank 1 still pending (its vote and findings lost), and the abort
        raced rank 1's staging, which then stayed staged and watched. Not: the real
        collective_rpc."""
        sides = [Side("worker", rank=r) for r in range(2)]
        from .test_host import Executor
        g = WorkerGroup(NS(core=NS(model_executor=Executor(sides)), in_loop=lambda fn: fn()))
        p = own_plan("veto", {"worker:0": "prepare",
                              "worker:1": {"method": "prepare", "delay_s": 0.3, "fail": False}})
        t0 = time.perf_counter()
        r = g.prepare(p)
        self.assertGreaterEqual(time.perf_counter() - t0, 0.3)
        self.assertEqual([(x["rank"], x["ok"], x.get("pending")) for x in r["ranks"]],
                         [(0, False, None), (1, True, None)])
        self.assertTrue(g.abort(p["id"])["ok"])
        prepared(p["id"])
        sides[1].tick(True)
        self.assertEqual((p["id"] in sides[1].lp._staged, sides[1].watch), (False, {}))

    def test_an_abort_while_staging_drops_what_the_prepare_staged(self):
        """Input: one rank's background prepare takes 0.3 s; the abort arrives at once (the
        host gave up, or another rank's prepare_start failed). Expected: once the prepare
        finishes, nothing of the transaction is staged, its state is "aborted", and no key is
        watched at the next step start. Fails before the fix: the abort only dropped the job,
        so the late prepare staged the transaction and added its watches afterwards. Not: a
        prepare that never finishes (the host then poisons; test_host)."""
        side = Side("worker", rank=0)
        p = own_plan("late", {"worker:0": {"method": "prepare", "delay_s": 0.3, "fail": False}})
        self.assertTrue(side.dispatch("prepare_start", p)["ok"])
        self.assertTrue(side.dispatch("abort", p["id"])["ok"])
        prepared(p["id"])
        side.tick(True)
        self.assertEqual((p["id"] in side.lp._staged, side.lp._state.get(p["id"]), side.watch),
                         (False, "aborted", {}))

    def test_a_rank_that_hangs_in_commit_is_named_by_the_deadline(self):
        """Input: 2 ranks, rank 1's commit sleeps 0.6 s; phase-2 deadline 0.4 s (commit by
        0.2 s). Expected: the commit reply still has both ranks; rank 1's is `deadline_missed`
        naming "worker rank 1"; the transaction is rolled back, the gap ends within the
        deadline, the host is poisoned, and once rank 1's late commit returns (refused: it
        started after its deadline) both ranks run "base". Pain point: one hung rank holds
        every step, or the RPC times out and its late reply misaligns the next call. Not: a
        real hung GPU."""
        h, names, _ = make_host(self, "hang")
        p = plan("hang", faults={"worker:1": {"method": "commit", "delay_s": 0.6,
                                              "fail": False}})
        r = h.handle({"op": "txn", "plan": p, "phase2_deadline_s": 0.4})
        rec = r["receipt"]
        commit = rec["participants"]["workers"]["commit"]
        self.assertEqual([(x["rank"], x["ok"]) for x in commit["ranks"]], [(0, True), (1, False)])
        self.assertEqual([f["check"] for f in commit["ranks"][1]["findings"]],
                         ["deadline_missed"])
        self.assertIn("worker rank 1", commit["ranks"][1]["error"])
        self.assertEqual(r["outcome"], "rolled_back")
        self.assertLess(rec["gap_s"], 0.5)
        self.assertTrue(h.co.poison)
        time.sleep(0.6)                          # rank 1's late commit returns
        self.assertEqual([ran(n) for n in names], ["base", "base"])


class IdentityTest(unittest.TestCase):

    def test_fingerprint_names_the_same_code_objects_after_undo(self):
        """Input: the host's fingerprint op on the toy engine module, before a commit, after
        it and after undo. Expected: every rank answers with the module's functions (content
        hash, code object id); Engine.step differs after the commit and the whole fingerprint
        is identical to the start after undo. Pain point: the GB200 TP2 run's fingerprint came
        back empty on both ranks (the fault hook hashed the list payload and the reply was
        not-ok), so "identical after undo" compared two empty dicts. Not: real ranks."""
        h, names, _ = make_host(self, "fp")

        def fp():
            r = h.handle({"op": "fingerprint", "modules": ["toy.engine"]})["workers"]
            self.assertTrue(r["ok"], r.get("error"))
            return {x["rank"]: x["code"] for x in r["ranks"]}
        before = fp()
        key = f"toy.engine:Engine.step:{STEP_LINE}"
        self.assertEqual(sorted(before), [0, 1])
        self.assertIn(key, before[0])
        self.assertEqual(h.handle({"op": "txn", "plan": plan("fp")})["outcome"], "committed")
        self.assertNotEqual(fp()[0][key], before[0][key])
        self.assertTrue(h.handle({"op": "undo", "txn": "fp"})["ok"])
        self.assertEqual(fp(), before)


class GraphTest(unittest.TestCase):

    def test_an_edit_inside_a_captured_graph_is_a_finding_and_not_applied(self):
        """Input: the host over 2 ranks whose workers run FULL_AND_PIECEWISE with
        torch.compile; the toy engine plays model code. Expected: the edit of Engine.step is
        reported by every rank as a blocking `captured_target` (route R1, today T1: this
        adapter has no R1), the transaction aborts, and both ranks still run "base". The same
        edit with the toy outside the model-code prefixes commits on both ranks. Pain point: a
        model-code edit that committed with no finding and whose witness stayed 0/0 (graphs
        only, TP2 MoE run). Not: vLLM's real graphs (the GPU receipt)."""
        h, names, sides = make_host(self, "graph")
        for s in sides:
            s.scope["worker"] = worker()
            s.captured_prefixes = ("toy.",)
        r = h.handle({"op": "txn", "plan": plan("g1")})
        fs = [(f["check"], f.get("rank"), f.get("route"), f.get("today"))
              for f in r["receipt"]["findings"]]
        self.assertEqual((r["outcome"], fs), ("aborted", [("captured_target", 0, "R1", "T1"),
                                                          ("captured_target", 1, "R1", "T1")]))
        self.assertIn("no R1 yet", r["receipt"]["findings"][0]["observed"])
        self.assertEqual([ran(n) for n in names], ["base", "base"])
        for s in sides:
            s.captured_prefixes = ("vllm.model_executor.models.",)
        r = h.handle({"op": "txn", "plan": plan("g2")})
        self.assertEqual((r["outcome"], r["receipt"]["findings"]), ("committed", []))
        self.assertEqual([ran(n) for n in names], ["v1", "v1"])

    def test_accepted_captured_target_proves_nothing_and_says_why(self):
        """Input: one graph-mode rank; the captured_target accepted by subject; no step runs.
        Expected: it commits, and the proof's zero_witness is routed R1 / today T1 with the
        recapture action first; on an eager worker the same plan has no finding. Not: whether
        vLLM calls the code (the witness says)."""
        p = own_plan("acc")
        (m,) = p["modules"]
        side = Side("worker", scope={"worker": worker()}, rank=0)
        side.captured_prefixes = (m.split(".")[0] + ".",)
        r = side.dispatch("prepare", p)
        self.assertEqual((r["ok"], [f["check"] for f in r["findings"]]),
                         (False, ["captured_target"]))
        p2 = {**p, "id": p["id"] + "-acc",
              "accept": [{"check": "captured_target", "subject": r["findings"][0]["subject"]}]}
        self.assertTrue(side.dispatch("prepare", p2)["ok"])
        self.assertTrue(side.dispatch("commit", p2["id"])["ok"])
        w = side.dispatch("witness_check", p2["id"])
        self.assertEqual([(f["check"], f.get("route"), f.get("today"),
                           f["actions"][0]["cost"]) for f in w["findings"]],
                         [("zero_witness", "R1", "T1", "recipe R1 / a restart")])
        self.assertTrue(side.dispatch("undo", p2["id"])["ok"])
        eager = Side("worker", scope={"worker": worker(eager=True)}, rank=0)
        eager.captured_prefixes = side.captured_prefixes
        r = eager.dispatch("prepare", own_plan("eager"))
        self.assertEqual((r["ok"], r["findings"]), (True, []))


GDN = '''"""A stand-in for a vLLM layer module: a custom op that splits the graph calls the
layer's core; the MoE block has no split op."""
import functools


def eager_break_during_capture(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


class Attn:
    def forward(self, x):
        out = [0]
        gdn_core(x, out, self)
        return out[0] + self.helper(x)

    def helper(self, x):
        return 0

    def _forward_core(self, x, out):
        out[0] = x * 2 + self.helper(x)


@eager_break_during_capture
def gdn_core(x, out, layer):
    self = layer
    self._forward_core(x, out)


class MoE:
    def forward(self, x):
        return x * 3
'''


class SplitOpTest(unittest.TestCase):
    """Pain point: at TP2 (FULL_AND_PIECEWISE, breakable graphs) captured_target vetoed a GDN
    core that runs in Python on prefill through an eager-break op: accepted by name, it read
    120 calls per rank after 4 prefills."""

    def setUp(self):
        import importlib
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        self.pkg = f"adsplit{next(_n)}"
        os.makedirs(os.path.join(d, self.pkg))
        open(os.path.join(d, self.pkg, "__init__.py"), "w").close()
        with open(os.path.join(d, self.pkg, "gdn.py"), "w") as f:
            f.write(GDN)
        sys.path.insert(0, d)
        self.addCleanup(sys.path.remove, d)
        importlib.import_module(self.pkg + ".gdn")

    def _findings(self, w, qual):
        side = Side("worker", scope={"worker": w}, rank=0)
        side.captured_prefixes = (self.pkg + ".",)
        m = self.pkg + ".gdn"
        cls, fn = qual.split(".")
        lines = GDN.splitlines()
        start = next(i for i, x in enumerate(lines) if x.startswith(f"class {cls}:"))
        line = next(i for i, x in enumerate(lines, 1)
                    if i > start and x.strip().startswith(f"def {fn}("))
        return [(f["check"], f.get("route"), bool(f.get("advisory")), f["observed"])
                for f in side._captured_check(m, [f"{m}:{qual}:{line}"])]

    def test_split_op_target_is_advisory_and_captured_only_stays_blocking(self):
        """Input: breakable graphs (FULL_AND_PIECEWISE, compile NONE). Expected:
        `Attn._forward_core` (reached only through the eager-break op `gdn_core`) is an
        advisory `captured_partly` naming gdn_core; `MoE.forward` (no split op) is a blocking
        `captured_target`; `Attn.helper` (also called from `forward`, which no split op
        reaches) stays blocking and says it could not decide. PIECEWISE: `_forward_core` has
        no finding (Python on every step). torch.compile with the op in splitting_ops:
        advisory again. Not: vLLM's real dispatch."""
        bk = worker("FULL_AND_PIECEWISE", "NONE")
        (core,) = self._findings(bk, "Attn._forward_core")
        self.assertEqual(core[:3], ("captured_partly", "R1", True))
        self.assertIn("gdn_core", core[3])
        (moe,) = self._findings(bk, "MoE.forward")
        self.assertEqual(moe[:3], ("captured_target", "R1", False))
        (helper,) = self._findings(bk, "Attn.helper")
        self.assertEqual(helper[:3], ("captured_target", "R1", False))
        self.assertIn("could not decide", helper[3])
        self.assertEqual(self._findings(worker("PIECEWISE", "NONE"), "Attn._forward_core"), [])
        comp = worker("FULL_AND_PIECEWISE", "VLLM_COMPILE", splitting_ops=["vllm::gdn_core"])
        self.assertEqual(self._findings(comp, "Attn._forward_core")[0][:3],
                         ("captured_partly", "R1", True))


class VersionLogTest(unittest.TestCase):

    def test_the_extension_logs_which_version_each_step_runs(self):
        """Input: a stand-in worker with the real KernelWorkerExtension; a step (execute_model,
        sample_tokens), then a prepare that lands between a step's two calls, a commit, more
        steps. Expected: the rank's `versions` show the key joining at a step start with the
        base tag, then the new tag from the first step after the commit, and no mid-step
        change. Pain point: at TP4 with "auto" targets, a rank logged a mid-step change that
        was only the base code appearing between two calls. Not: vLLM's step hooks."""
        self.addCleanup(WE._ENDPOINT.clear)
        WE._ENDPOINT.clear()

        class Worker(WE.KernelWorkerExtension):
            rank = 0

            def execute_model(self, *a):
                return "out"

            def sample_tokens(self, *a):
                return "tok"
        w = Worker()
        p = own_plan("vlog", targets=False)
        w.fastrsi_kernel_endpoint()             # the side exists and wraps the step calls
        w.execute_model()
        self.assertTrue(w.fastrsi_kernel("prepare", p)["ok"])
        w.sample_tokens()
        w.execute_model()
        w.sample_tokens()
        self.assertTrue(w.fastrsi_kernel("commit", p["id"])["ok"])
        w.execute_model()
        w.sample_tokens()
        v = w.fastrsi_kernel("versions")
        (m,) = p["modules"]
        key = f"{m}:Engine.step"
        self.assertEqual(v["mid_step_changes"], [])
        self.assertEqual([s for s, _ in v["vlog"]], [2, 3])
        base, new = (tags[key] for _, tags in v["vlog"])
        self.assertNotEqual(base, new)
        self.assertEqual(v["watch"], [key])

    def test_an_aborted_transaction_leaves_nothing_watched(self):
        """Input: a rank prepares (auto targets) and then gets the abort of a transaction
        another rank vetoed. Expected: its keys stop being watched (pending ones at once, a
        joined one at the next step start, never mid-step); a committed transaction's keys
        stay. Pain point: at TP2 rank 1 kept watching a function after an aborted
        transaction, so the ranks' version logs differed. Not: the real abort RPC."""
        side = Side("worker", rank=0)
        a, b = own_plan("dropa", targets=False), own_plan("dropb", targets=False)
        self.assertTrue(side.dispatch("prepare", a)["ok"])
        self.assertTrue(side.dispatch("abort", a["id"])["ok"])
        side.tick(True)
        self.assertEqual(side.watch, {})
        self.assertTrue(side.dispatch("prepare", b)["ok"])
        side.tick(True)
        self.assertEqual(len(side.watch), 1)
        self.assertTrue(side.dispatch("abort", b["id"])["ok"])
        side.tick(False)
        side.tick(True)
        self.assertEqual((side.watch, side.mid_step_changes), ({}, []))
        c = own_plan("dropc", targets=False)
        self.assertTrue(side.dispatch("prepare", c)["ok"])
        self.assertTrue(side.dispatch("commit", c["id"])["ok"])
        side.tick(True)
        self.assertEqual(len(side.watch), 1)


if __name__ == "__main__":
    unittest.main()
