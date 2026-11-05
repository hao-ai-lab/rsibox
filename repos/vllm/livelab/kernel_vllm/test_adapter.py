"""CPU tests of the adapter's own logic (no vLLM, no GPU). Run from repos/vllm/livelab with the
kernel on the path:

    PYTHONPATH=../../../template/livelab python3 -m unittest kernel_vllm.test_adapter -v

The worker group's transport is replaced by a loop over in-process `Side`s: one
collective_rpc = one call on every rank, in rank order, as vLLM's executor does. Everything
the tests are named for (Side, WorkerGroup, the kernel participant) is the real code.
"""
import itertools
import sys
import time
import unittest

from kernel import locate as L
from kernel import toy

from .host import WorkerGroup
from .side import Side

ENG = toy.source("engine")
_n = itertools.count()


def plan(tag, faults=None):
    """A worker plan that stages a one-line edit of the toy engine's step (a fresh copy)."""
    name = f"kva{next(_n)}_{tag}"
    toy.load_copy(name)
    line = next(i for i, s in enumerate(ENG.splitlines(), 1) if "def step" in s)
    return {"id": f"txn-{name}", "faults": faults or {},
            "modules": {f"{name}.engine": {
                "on": ["worker"], "base_hash": L.text_hash(ENG),
                "source": ENG.replace('self.ran.append("base")', 'self.ran.append("v1")'),
                "targets": [["Engine.step", line]]}}}


class _Executor:
    def __init__(self, sides):
        self.sides, self.calls = sides, []

    def collective_rpc(self, name, timeout=None, args=()):
        method, payload = args
        self.calls.append(method)
        out = []
        for s in self.sides:
            r = s.dispatch(method, payload)
            r["rank"] = s.rank
            out.append(r)
        return out


class _Host:
    def __init__(self, sides):
        self.core = type("Core", (), {})()
        self.core.model_executor = _Executor(sides)

    def in_loop(self, fn, gap=False, timeout=60.0):
        return fn()


class WorkerPrepareTest(unittest.TestCase):

    def _side(self, rank):
        side = Side("worker", rank=rank)
        if side.lp.analyzer is not None:
            self.addCleanup(side.lp.analyzer.close)
        return side

    def test_ranks_that_finish_early_keep_their_vote(self):
        """Proves: with rank 3 preparing 0.3 s longer than ranks 0-2, the group keeps polling
        every rank and still gets four yes votes (before the fix, ranks 0-2 answered "no
        prepare started" on the polls after their first finished one: a false abort, 1 of 33
        live TP4 prepares). After abort, the result is gone. Not: the real collective_rpc."""
        sides = [self._side(r) for r in range(4)]
        g = WorkerGroup(_Host(sides))
        p = plan("slow3", {"worker:3": {"method": "prepare", "delay_s": 0.3, "fail": False}})
        t0 = time.perf_counter()
        r = g.prepare(p)
        took = time.perf_counter() - t0
        self.assertEqual([(x["rank"], x["ok"], x.get("error")) for x in r["ranks"]],
                         [(0, True, None), (1, True, None), (2, True, None), (3, True, None)])
        self.assertTrue(r["ok"])
        self.assertGreaterEqual(took, 0.3)
        polls = g.host.core.model_executor.calls.count("prepare_poll")
        self.assertGreater(polls, 2)                 # ranks 0-2 were asked again after done
        again = g._rpc("prepare_poll", p["id"])         # repeatable until the vote is over
        self.assertEqual((again["ok"], again.get("pending")), (True, None))
        self.assertTrue(g.abort(p["id"])["ok"])
        after = g._rpc("prepare_poll", p["id"])
        self.assertEqual([x["error"] for x in after["ranks"]],
                         [f"no prepare started for {p['id']}"] * 4)

    def test_side_prepares_with_the_analysis_in_a_child_process(self):
        """Proves: a Side stages through the kernel's ProcessAnalyzer by default, so the
        parses and compare do not hold the engine process's GIL (timings say "process").
        Not: the stall itself (measured live: kernel_vllm/README.md)."""
        side = self._side(0)
        p = plan("analyzer")
        r = side.dispatch("prepare", p)
        self.assertTrue(r["ok"], r.get("error"))
        (t,) = r["timings"].values()
        self.assertEqual(t["analyze"], "process", t)

    def test_side_watches_targets_the_kernel_found(self):
        """Proves: with targets left to the kernel ("auto"), the side still watches the changed
        function (from prepare's reply), so version transitions are recorded. Pain point: a
        plan without targets watched nothing. Not: the live step hook."""
        side = self._side(0)
        p = plan("auto")
        (spec,) = p["modules"].values()
        del spec["targets"]
        r = side.dispatch("prepare", p)
        self.assertTrue(r["ok"], r.get("error"))
        (m,) = p["modules"]
        side.tick(True)                                 # joins the watch set at a step start
        self.assertIn(f"{m}:Engine.step", side.watch)

    def test_a_key_watched_mid_step_is_not_a_mid_step_change(self):
        """Proves: a prepare that lands between a step's two calls (execute_model, then
        sample_tokens) adds its target to the watch set only at the next step start: no
        mid-step change is logged, and the next step's version log shows the key with its base
        tag. Pain point: at TP4 with "auto" targets, rank 2 logged a mid-step change at step 10
        that was only the base code appearing between two ticks. Not: the live step hooks."""
        side = self._side(0)
        p = plan("midstep")
        (spec,) = p["modules"].values()
        del spec["targets"]
        side.tick(True)                                 # step 1: execute_model
        self.assertTrue(side.dispatch("prepare", p)["ok"])
        side.tick(False)                                # step 1: sample_tokens
        side.tick(True)                                 # step 2
        (m,) = p["modules"]
        self.assertEqual(side.mid_step_changes, [])
        self.assertEqual([(s, sorted(v)) for s, v in side.vlog], [(2, [f"{m}:Engine.step"])])


def _worker(cudagraph="FULL_AND_PIECEWISE", compile_mode="VLLM_COMPILE", eager=False):
    """A stand-in for vLLM's Worker: only the config fields the side reads."""
    from types import SimpleNamespace as NS
    return NS(vllm_config=NS(model_config=NS(enforce_eager=eager), compilation_config=NS(
        cudagraph_mode=NS(name=cudagraph), mode=NS(name=compile_mode))))


class GraphModeTest(unittest.TestCase):

    def _side(self, worker, prefix):
        side = Side("worker", scope={"worker": worker}, rank=0)
        side.captured_prefixes = (prefix,)         # the toy copy plays model code
        if side.lp.analyzer is not None:
            self.addCleanup(side.lp.analyzer.close)
        return side

    def test_model_code_in_graph_mode_is_a_captured_target(self):
        """Proves: on a worker whose config runs CUDA graphs (FULL_AND_PIECEWISE, VLLM_COMPILE),
        a target in model code is a `captured_target` finding at prepare (route R1) that votes
        no; accepted by subject it commits, and a proof with no call gives zero_witness routed
        to R1; the same plan on an eager worker gives no finding. Pain point: at TP2 a MoE
        forward edit committed with no finding and its witness stayed 0/0 (graphs only). Not:
        whether vLLM really calls the code (the proof's witness says)."""
        p = plan("graph")
        (m,) = p["modules"]
        prefix = m.split(".")[0] + "."
        side = self._side(_worker(), prefix)
        r = side.dispatch("prepare", p)
        self.assertEqual((r["ok"], [(f["check"], f.get("route")) for f in r["findings"]]),
                         (False, [("captured_target", "R1")]))
        self.assertIn("FULL_AND_PIECEWISE", r["findings"][0]["observed"])
        subject = r["findings"][0]["subject"]
        p2 = {**p, "id": p["id"] + "-acc",
              "accept": [{"check": "captured_target", "subject": subject}]}
        self.assertTrue(side.dispatch("prepare", p2)["ok"])
        self.assertTrue(side.dispatch("commit", p2["id"])["ok"])
        w = side.dispatch("witness_check", p2["id"])
        self.assertEqual([(f["check"], f.get("route"), f["actions"][0]["cost"])
                          for f in w["findings"]], [("zero_witness", "R1", "recipe R1")])
        self.assertTrue(side.dispatch("undo", p2["id"])["ok"])
        eager = self._side(_worker(eager=True), prefix)
        r = eager.dispatch("prepare", plan("eager"))
        self.assertEqual((r["ok"], r["findings"]), (True, []))


def _worker2(cudagraph, compile_mode, splitting_ops=()):
    """`_worker` with splitting_ops and a max capture size (graphs.graph_mode reads them)."""
    w = _worker(cudagraph, compile_mode)
    w.vllm_config.compilation_config.splitting_ops = list(splitting_ops)
    w.vllm_config.compilation_config.max_cudagraph_capture_size = 512
    return w


GDN = '''"""A stand-in for a vLLM layer module: a custom op that splits the graph (decorated as
vLLM's breakable graphs mark it) calls the layer's core; the MoE block has no split op."""
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
        return self.proj(out[0]) + self.helper(x)

    def proj(self, y):
        return y + 1

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


class CapturedSplitTest(unittest.TestCase):
    """Pain point: at TP2 (Qwen3.5-35B-A3B-FP8, FULL_AND_PIECEWISE, breakable graphs)
    captured_target vetoed the GDN core `_forward_core`, which runs in Python on prefill
    through the eager-break op and only inside full graphs on decode: accepted by name, it read
    120 calls per rank after 4 prefills."""

    def setUp(self):
        import importlib
        import os
        import shutil
        import tempfile
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        self.pkg = f"kvsplit{next(_n)}"
        os.makedirs(os.path.join(d, self.pkg))
        open(os.path.join(d, self.pkg, "__init__.py"), "w").close()
        with open(os.path.join(d, self.pkg, "gdn.py"), "w") as f:
            f.write(GDN)
        sys.path.insert(0, d)
        self.addCleanup(sys.path.remove, d)
        importlib.import_module(self.pkg + ".gdn")

    def _findings(self, worker, qual):
        """The worker side's prepare-time check for one target of the stand-in module:
        [(check, route, advisory, observed)]."""
        side = Side("worker", scope={"worker": worker}, rank=0)
        side.captured_prefixes = (self.pkg + ".",)     # plays vLLM model code
        if side.lp.analyzer is not None:
            self.addCleanup(side.lp.analyzer.close)
        m = self.pkg + ".gdn"
        cls, fn = qual.split(".")
        lines = GDN.splitlines()
        start = next(i for i, x in enumerate(lines) if x.startswith(f"class {cls}:"))
        line = next(i for i, x in enumerate(lines, 1)
                    if i > start and x.strip().startswith(f"def {fn}("))
        r = side._captured_check(m, [f"{m}:{qual}:{line}"])
        return [(f["check"], f.get("route"), bool(f.get("advisory")), f["observed"]) for f in r]

    def test_split_op_target_is_advisory_and_captured_only_stays_blocking(self):
        """Input: breakable graphs (FULL_AND_PIECEWISE, compile NONE). Expected:
        `Attn._forward_core` (reached only through the eager-break op `gdn_core`) is an
        advisory `captured_partly` (route R1) saying it runs on prefill now and on decode only
        after R1; `MoE.forward` (no split op) stays a blocking `captured_target`;
        `Attn.helper` (called from `_forward_core` and from `forward`, which no split op
        reaches) stays blocking and says it could not decide. In PIECEWISE mode
        `_forward_core` gets no finding (Python on every step); with torch.compile and the op
        in splitting_ops it is advisory again. Not: vLLM's real dispatch (the GPU queue job
        checks the witness on prefill and, after R1, on decode)."""
        bk = _worker2("FULL_AND_PIECEWISE", "NONE")
        (core,) = self._findings(bk, "Attn._forward_core")
        self.assertEqual(core[:3], ("captured_partly", "R1", True))
        self.assertIn("prefill", core[3])
        self.assertIn("only after R1", core[3])
        self.assertIn("gdn_core", core[3])
        (moe,) = self._findings(bk, "MoE.forward")
        self.assertEqual(moe[:3], ("captured_target", "R1", False))
        (helper,) = self._findings(bk, "Attn.helper")
        self.assertEqual(helper[:3], ("captured_target", "R1", False))
        self.assertIn("could not decide", helper[3])
        self.assertEqual(self._findings(_worker2("PIECEWISE", "NONE"), "Attn._forward_core"),
                         [])
        comp = _worker2("FULL_AND_PIECEWISE", "VLLM_COMPILE", ["vllm::gdn_core"])
        self.assertEqual(self._findings(comp, "Attn._forward_core")[0][:3],
                         ("captured_partly", "R1", True))
        self.assertEqual(self._findings(comp, "MoE.forward")[0][:3],
                         ("captured_target", "R1", False))


class RebuildTest(unittest.TestCase):
    """Pain point: the adapter had no R1. After a graph-mode commit the lab called the live
    lab's recapture over another RPC: commit and recapture were not one transaction, a failed
    recapture left the edit in, and the witness then counted capture calls as runs."""

    def _host(self, tag, ranks=2, worker=None):
        """The real Host (kernel StepGapCoordinator, WorkerGroup, core Side) over in-process
        worker Sides; each side's R1 is a recorder that runs the swapped toy step twice, as a
        capture runs the model code."""
        import shutil
        import tempfile
        import types
        from unittest import mock
        from . import host as H
        calls, sides, names = [], [], []
        for r in range(ranks):
            name = f"kvb{next(_n)}_{tag}{r}"
            toy.load_copy(name)                       # each rank its own module, as a process
            names.append(name)
            s = Side("worker", scope={"worker": worker or object()}, rank=r)
            s.lp.alias = {"toy": name}
            if s.lp.analyzer is not None:
                self.addCleanup(s.lp.analyzer.close)
            if worker is not None:
                s.captured_prefixes = ("toy.",)       # the toy engine plays model code

            def r1(w, r=r, name=name):
                eng = sys.modules[name + ".engine"].Engine()
                eng.step(1)
                eng.step(1)
                calls.append((r, eng.ran[-1]))
                return {"how": "recapture", "graphs": 2}
            s.rebuild_fn = r1
            sides.append(s)
        core = types.SimpleNamespace(model_executor=_Executor(sides))
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        with mock.patch.dict(sys.modules, {"vllm": types.SimpleNamespace(__version__="cpu")}):
            h = H.Host(core, d)
        if h.core_side.lp.analyzer is not None:
            self.addCleanup(h.core_side.lp.analyzer.close)
        return h, names, calls, sides

    def _plan(self, tid, **kw):
        line = next(i for i, x in enumerate(ENG.splitlines(), 1) if "def step" in x)
        return {"id": tid, **kw, "modules": {"toy.engine": {
            "on": ["worker"], "base_hash": L.text_hash(ENG),
            "source": ENG.replace('self.ran.append("base")', 'self.ran.append("v1")'),
            "targets": [["Engine.step", line]]}}}

    def test_recapture_flag_runs_r1_on_every_rank_after_the_commit_and_after_undo(self):
        """Input: a plan with "recapture": true through the host's txn op, two worker ranks.
        Expected: committed; R1 ran once per rank right after the commit and saw the new code
        (v1); the receipt has rebuild.ok, how ["recapture"], recapture_s, "R1 recapture",
        edit_to_run_s >= edit_to_commit_s, steps_during 0. The witness counts R1's calls as
        capture (live 0, zero_witness routed to R1 with a device-probe note) until a live
        call. Undo runs R1 on both ranks from the base code. Not: vLLM's capture itself
        (rebuild.py; the GPU queue job)."""
        h, names, calls, sides = self._host("flag")
        r = h.handle({"op": "txn", "plan": self._plan("fl", recapture=True)})
        rec = r["receipt"]
        self.assertEqual((r["outcome"], calls), ("committed", [(0, "v1"), (1, "v1")]))
        rb = rec["rebuild"]
        self.assertEqual((rb["ok"], rb["results"]["R1"]["how"], rb.get("steps_during")),
                         (True, ["recapture"], 0))
        self.assertIn("paused", rb["while"])
        self.assertIsNotNone(rec["recapture_s"])
        self.assertIn("R1 recapture", rec["actual_recipes"])
        self.assertGreaterEqual(rec["edit_to_run_s"], rec["edit_to_commit_s"])
        pr = h.handle({"op": "prove", "txn": "fl"})["receipt"]
        (k,) = pr["witness"]["workers/rank0"]
        self.assertEqual((pr["witness"]["workers/rank0"][k],
                          pr["witness_capture"]["workers/rank0"][k]), (0, 2))
        self.assertEqual(pr["ran"]["workers/rank1"][k], "ran during capture 2, ran live 0")
        zw = [f for f in pr["findings"] if f["check"] == "zero_witness"]
        self.assertEqual([(f["rank"], f["route"]) for f in zw], [(0, "R1"), (1, "R1")])
        self.assertIn("device-side probe", zw[0]["note"])
        sys.modules[names[0] + ".engine"].Engine().step(1)    # a live run on rank 0
        pr = h.handle({"op": "prove", "txn": "fl"})["receipt"]
        self.assertEqual(pr["ran"]["workers/rank0"][k], "ran during capture 2, ran live 1")
        del calls[:]
        u = h.handle({"op": "undo", "txn": "fl"})
        self.assertEqual((u["ok"], calls), (True, [(0, "base"), (1, "base")]))
        self.assertEqual(u["receipt"]["rebuild"]["why"], "undo")

    def test_failed_r1_rolls_back_and_recaptures_the_base(self):
        """Input: the same plan with R1 failing on rank 1 after the commit (fault hook).
        Expected: outcome rolled_back, a not-overridable rebuild_failed finding, both ranks
        back at the base code (status undone), and R1 run again on both ranks from the base
        code (why rollback). Not: a real capture error."""
        h, names, calls, sides = self._host("fail")
        p = self._plan("ff", recapture=True, faults={"worker:1": "rebuild"})
        r = h.handle({"op": "txn", "plan": p})
        rec = r["receipt"]
        self.assertEqual(r["outcome"], "rolled_back")
        self.assertEqual([(f["subject"], f["overridable"]) for f in rec["findings"]
                          if f["check"] == "rebuild_failed"], [("commit", False)])
        self.assertEqual(calls, [(0, "v1"), (0, "base"), (1, "base")])
        self.assertEqual((rec["rollback"]["status"], rec["rebuild"]["back"]["ok"]),
                         ("undone", True))
        st = h.handle({"op": "pstatus", "txn": "ff"})
        self.assertEqual([x["state"] for x in st["workers"]["ranks"]], ["undone", "undone"])

    def test_an_accepted_captured_target_runs_r1_without_the_flag(self):
        """Input: a graph-mode worker (FULL_AND_PIECEWISE, VLLM_COMPILE), the toy step as model
        code, captured_target accepted, no recapture flag. Expected: R1 runs after the commit
        (auto); with "recapture": false it does not. And rebuild.how picks recompile under
        torch.compile, recapture for pure CUDA graphs, none when eager. Not: the recompile."""
        from . import rebuild as RB
        h, names, calls, _ = self._host("auto", ranks=1, worker=_worker())
        r = h.handle({"op": "txn", "plan": self._plan("au", accept=["captured_target"])})
        self.assertEqual((r["outcome"], calls), ("committed", [(0, "v1")]))
        self.assertTrue(h.handle({"op": "undo", "txn": "au"})["ok"])
        del calls[:]
        r = h.handle({"op": "txn", "plan": self._plan("au2", accept=["captured_target"],
                                                      recapture=False)})
        self.assertEqual((r["outcome"], calls, r["receipt"]["rebuild"]),
                         ("committed", [], None))
        self.assertEqual([RB.how(_worker("FULL_AND_PIECEWISE", "VLLM_COMPILE")),
                          RB.how(_worker("FULL_AND_PIECEWISE", "NONE")),
                          RB.how(_worker("NONE", "NONE")), RB.how(_worker(eager=True))],
                         ["recompile", "recapture", "none", "none"])

class WatchDropTest(unittest.TestCase):

    def test_an_aborted_transaction_leaves_nothing_watched(self):
        """Proves: a rank that prepared a plan (auto targets) and then got the abort of a
        transaction another rank vetoed stops watching its targets: pending keys at once, a key
        already joined at the next step start (never mid-step); a committed transaction's keys
        stay. Pain point: at TP2 rank 1 kept watching Qwen3NextSparseMoeBlock.forward after an
        aborted transaction, so the ranks' version logs differed. Not: the real abort RPC."""
        side = Side("worker", rank=0)
        if side.lp.analyzer is not None:
            self.addCleanup(side.lp.analyzer.close)
        a, b = plan("dropa"), plan("dropb")
        for x in (a, b):
            del next(iter(x["modules"].values()))["targets"]
        self.assertTrue(side.dispatch("prepare", a)["ok"])
        self.assertTrue(side.dispatch("abort", a["id"])["ok"])
        side.tick(True)
        self.assertEqual(side.watch, {})                    # pending: never joined
        self.assertTrue(side.dispatch("prepare", b)["ok"])
        side.tick(True)
        (mb,) = b["modules"]
        self.assertEqual(sorted(side.watch), [f"{mb}:Engine.step"])
        self.assertTrue(side.dispatch("abort", b["id"])["ok"])
        side.tick(False)
        self.assertEqual(side.mid_step_changes, [])
        side.tick(True)
        self.assertEqual(side.watch, {})                    # joined: dropped at a step start
        c = plan("dropc")
        self.assertTrue(side.dispatch("prepare", c)["ok"])
        self.assertTrue(side.dispatch("commit", c["id"])["ok"])
        side.tick(True)
        (mc,) = c["modules"]
        self.assertEqual(sorted(side.watch), [f"{mc}:Engine.step"])


class RanksDisagreeTest(unittest.TestCase):

    def test_host_receipt_names_ranks_that_run_different_code(self):
        """Proves: through the adapter's own pieces (two worker Sides behind a WorkerGroup and
        the engine-core Side, as the host wires them into the kernel's StepGapCoordinator), a
        commit after which rank 1 runs other code for `Engine.step` gives a txn receipt with a
        not-overridable `ranks_disagree` finding for "workers" naming the split, blocking in
        `decisions`; the same plan with both ranks switched gives none. Pain point: at TP2 the
        receipt said ranks_agree false with findings []. Not: real ranks in separate processes
        (needs TP4: the kernel's live receipt there)."""
        from kernel import coordinator as C

        def run(tag, tamper_rank=None):
            sides = []
            for r in range(2):
                name = f"kvr{next(_n)}_{tag}{r}"
                toy.load_copy(name)
                s = Side("worker", rank=r)
                s.lp.alias = {"toy": name}          # each rank its own module, as in a process
                if s.lp.analyzer is not None:
                    self.addCleanup(s.lp.analyzer.close)
                if r == tamper_rank:
                    commit = s.lp.commit

                    def tampered(txn, commit=commit, name=name):
                        out = commit(txn)
                        eng = sys.modules[name + ".engine"]
                        eng.Engine.step.__code__ = eng.Engine.drain.__code__.replace(
                            co_name="step", co_qualname="Engine.step")
                        return out
                    s.lp.commit = tampered
                sides.append(s)
            core = Side("core", scope={})
            if core.lp.analyzer is not None:
                self.addCleanup(core.lp.analyzer.close)
            co = C.StepGapCoordinator(
                [WorkerGroup(_Host(sides)), C.CallParticipant("core", core.dispatch, bounded=True)],
                gap=lambda fn: fn(), phase2_deadline_s=None)
            line = next(i for i, x in enumerate(ENG.splitlines(), 1) if "def step" in x)
            plan = {"modules": {"toy.engine": {
                "on": ["worker"], "base_hash": L.text_hash(ENG),
                "source": ENG.replace('self.ran.append("base")', 'self.ran.append("v1")'),
                "targets": [["Engine.step", line]]}}}
            return co.run(plan)["receipt"]

        ok = run("agree")
        self.assertEqual((ok["outcome"], ok["commit_hashes"]["ranks_agree"], ok["findings"]),
                         ("committed", {"workers": True}, []))
        bad = run("split", tamper_rank=1)
        fs = [f for f in bad["findings"] if f["check"] == "ranks_disagree"]
        self.assertEqual((bad["outcome"], bad["commit_hashes"]["ranks_agree"]),
                         ("committed", {"workers": False}))
        self.assertEqual([(f["participant"], f["overridable"]) for f in fs], [("workers", False)])
        self.assertTrue(fs[0]["observed"].startswith("ranks "), fs[0]["observed"])
        self.assertIn("Engine.step", fs[0]["observed"])
        self.assertIn("ranks_disagree", bad["decisions"]["blocking"])


class RepairCellTest(unittest.TestCase):

    def test_a_repair_cell_runs_on_the_roles_in_its_on_list(self):
        """Proves: a plan with only a repair cell `"on": ["worker"]` commits on a worker side
        (the cell changes the worker's attribute; undo removes it) while the core side abstains.
        Pain point: the side dropped every repair cell (`"repairs": []`), so the worker voted no
        with `empty_change` and no live test could run a cell. Not: the cell's own journal calls
        (kernel tests)."""
        w = type("Worker", (), {})()
        worker = Side("worker", scope={"worker": w}, rank=0)
        core = Side("core", scope={"core": object()})
        for s in (worker, core):
            if s.lp.analyzer is not None:
                self.addCleanup(s.lp.analyzer.close)
        p = {"id": "cell-1", "modules": {},
             "repairs": [{"cell": "worker.fastrsi_fixed = 1", "declares": ["worker"],
                          "on": ["worker"]}]}
        rw, rc = worker.dispatch("prepare", p), core.dispatch("prepare", p)
        self.assertTrue(rw["ok"], rw.get("findings"))
        self.assertEqual((rc["ok"], rc.get("abstain")), (True, True))
        self.assertTrue(worker.dispatch("commit", "cell-1")["ok"])
        self.assertEqual(vars(w), {"fastrsi_fixed": 1})
        self.assertTrue(worker.dispatch("undo", "cell-1")["ok"])
        self.assertEqual(vars(w), {})


if __name__ == "__main__":
    sys.exit(unittest.main())


class WarmStartTest(unittest.TestCase):
    """Pain point: each worker rank created its side (and the side its analyzer child) at the
    first fastrsi_kernel call, which is the first prepare, so the first edit after engine start
    paid the child's spawn (1.5-2.2 s on an engine host, against 0.12 s for later prepares)."""

    def test_engine_start_spawns_every_analyzer_before_the_first_prepare(self):
        """Proves: the patched EngineCoreProc.run_busy_loop (the real _patch_core and Host,
        with a one-rank executor that calls the real KernelWorkerExtension) has the worker
        rank's side and the core side, each with a live analyzer child (spawns 1), before its
        first step; the first worker prepare then spawns nothing and waits for no boot (no
        analyze_spawned, no analyze_boot_wait_s in rank 0's timings; spawns still 1). Not:
        the boot time in a vLLM process (the GPU job measures the first edit live)."""
        import os
        import tempfile
        import types
        from unittest import mock
        from . import host as H
        from . import worker_ext as WE

        self.addCleanup(WE._SIDE.clear)
        WE._SIDE.clear()

        class Worker(WE.KernelWorkerExtension):
            rank = 0

            def execute_model(self, *a):
                return "out"

            def sample_tokens(self, *a):
                return "tok"

        class Executor:
            def __init__(self):
                self.worker = Worker()

            def collective_rpc(self, name, timeout=None, args=()):
                return [getattr(self.worker, name)(*args)]

        seen = {}

        class EngineCoreProc:               # vLLM's class, reduced to what the host patches
            def __init__(self):
                self.model_executor = Executor()

            def _handle_client_request(self, request_type, request):
                pass

            def _process_engine_step(self):
                pass

            def run_busy_loop(self):        # the loop's first step: what exists by now?
                side = WE._SIDE[0] if WE._SIDE else None
                seen["worker"] = side and side.lp.analyzer.state()
                seen["core"] = H._HOST[0].core_side.lp.analyzer.state()

        mod = types.SimpleNamespace(EngineCoreProc=EngineCoreProc)
        H._patch_core(mod)
        tmp = tempfile.mkdtemp()
        with mock.patch.dict(sys.modules, {"vllm": types.SimpleNamespace(__version__="cpu")}), \
                mock.patch.object(H, "DIR", tmp), \
                mock.patch.object(H.Host, "control", lambda self: None), \
                mock.patch.object(H, "_HOST", []):
            core = EngineCoreProc()
            core.run_busy_loop()
            h = H._HOST[0]
            self.addCleanup(h.core_side.lp.analyzer.close)
            self.addCleanup(lambda: WE._SIDE and WE._SIDE[0].lp.analyzer.close())
            w, c = seen["worker"], seen["core"]
            self.assertEqual((w and w["alive"], w and w["spawns"], c["alive"], c["spawns"]),
                             (True, 1, True, 1))
            self.assertTrue(h.analyzers(wait_s=30)["ready"])
            r = H.WorkerGroup(h).prepare(plan("warm"))
            self.assertTrue(r["ok"], r.get("error"))
            (t,) = r["timings"]["rank0"].values()
            self.assertEqual((t["analyze"], "analyze_spawned" in t, "analyze_boot_wait_s" in t,
                              WE._SIDE[0].lp.analyzer.state()["spawns"]),
                             ("process", False, False, 1))
            self.assertEqual(h.started["workers"]["0"]["spawns"], 1)
        import shutil
        shutil.rmtree(tmp, True)
