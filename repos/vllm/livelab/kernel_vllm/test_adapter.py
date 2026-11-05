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
