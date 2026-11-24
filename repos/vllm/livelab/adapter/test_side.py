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
    while any(t.name == f"rsibox-prepare-{txn}" for t in threading.enumerate()):
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

    def test_phase2_lane_binds_worker_cuda_device(self):
        """Input: (a) a worker rank owning `cuda:3`; (b) a worker rank owning un-indexed `"cuda"`
        while the main thread's `torch.cuda.current_device()` is `5`. Expected: bounded `commit`
        and `undo` calls on the lane thread enter `torch.cuda.device("cuda:3")` for (a) and pin
        explicit device `5` on the main thread for (b) so lane threads never touch device 0.
        Fails before the fix: `entered_devices` is empty or passes un-indexed `"cuda"`."""
        import contextlib
        import types
        from unittest import mock

        entered_devices = []
        caller_thread_id = threading.get_ident()

        @contextlib.contextmanager
        def record_device(device):
            entered_devices.append((device, threading.get_ident()))
            yield

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = NS(
            is_initialized=lambda: True,
            is_current_stream_capturing=lambda: False,
            current_device=lambda: 5 if threading.get_ident() == caller_thread_id else 0,
            device=record_device,
        )
        worker_stub = NS(model_runner=NS(device="cuda:3"))
        target_plan = own_plan("lane_dev")
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            side = Side("worker", scope={"worker": worker_stub}, rank=3)
            self.assertTrue(side.dispatch("prepare", target_plan)["ok"])
            deadline_at = time.time() + 5.0
            commit_reply = side.dispatch(
                "commit", {"txn": target_plan["id"], "deadline_at": deadline_at}
            )
            undo_reply = side.dispatch(
                "undo", {"txn": target_plan["id"], "deadline_at": deadline_at}
            )
        self.assertTrue(commit_reply["ok"], commit_reply)
        self.assertTrue(undo_reply["ok"], undo_reply)
        self.assertEqual([device for device, _ in entered_devices], ["cuda:3", "cuda:3"])
        self.assertTrue(all(thread_id != caller_thread_id for _, thread_id in entered_devices))

        entered_devices.clear()
        unindexed_worker = NS(model_runner=NS(device="cuda"))
        unindexed_plan = own_plan("lane_unindexed")
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            unindexed_side = Side("worker", scope={"worker": unindexed_worker}, rank=5)
            self.assertTrue(unindexed_side.dispatch("prepare", unindexed_plan)["ok"])
            deadline_at = time.time() + 5.0
            self.assertTrue(
                unindexed_side.dispatch(
                    "commit", {"txn": unindexed_plan["id"], "deadline_at": deadline_at}
                )["ok"]
            )
            self.assertTrue(
                unindexed_side.dispatch(
                    "undo", {"txn": unindexed_plan["id"], "deadline_at": deadline_at}
                )["ok"]
            )
        self.assertEqual([device for device, _ in entered_devices], [5, 5])
        self.assertTrue(all(thread_id != caller_thread_id for _, thread_id in entered_devices))


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
        reported by every rank as a blocking `captured_target` (route R1; the
        finding names the plan's "rebuild": ["R1"]), the transaction aborts, and both ranks still run "base". The same
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
        self.assertIn('"rebuild": ["R1"]', r["receipt"]["findings"][0]["observed"])
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

    def test_vllm_models_and_kernel_prefixes_block_without_r1_and_commit_with_r1(self):
        """Input: 2-rank graph-mode host using the default `CAPTURED_PREFIXES` and
        `CAPTURED_EXCLUSIONS` (no override), editing vLLM 0.33.0 captured GPU modules
        (`vllm.models.*`, `vllm.model_executor.kernels.*`, `vllm.kernels.*`,
        `vllm.v1.attention.backends.*`, `vllm.v1.attention.ops.*`, `vllm.lora.layers.*`,
        `vllm.lora.ops.*`, `vllm.distributed.device_communicators.*`, exact `vllm._custom_ops`)
        plus host-side exclusions under `vllm.models.*` (`*.model_state`, `*host_table*`,
        `*.quant_config`, `*.mm_preprocess`) and non-captured modules.
        Expected: captured GPU edits abort with blocking `captured_target` (route R1, phrasing
        "captured GPU code") when `rebuild` is omitted, turn advisory and recapture graphs when
        `rebuild=["R1"]` is given, route `zero_witness` to R1 if accepted without R1, commit
        with zero findings on an eager worker, and leave host-side exclusions and non-captured
        modules unblocked. Fails before the fix: `vllm.models.*`/attention/LoRA/comm/`_custom_ops`
        commit silently without `captured_target`, or host-side `model_state`/`engram_host_table`
        are falsely blocked."""
        from .test_recapture import Runner

        lab, rank_packages, sides = make_host(self, "vllm033")
        runners = []
        exact_aliases = (
            "vllm.models.deepseek_v41.nvidia.model",
            "vllm.model_executor.kernels.attention.candidate_blocks",
            "vllm.kernels.triton.activation",
            "vllm.v1.attention.backends.flash_attn",
            "vllm.v1.attention.ops.triton_decode_attention",
            "vllm.lora.layers.row_parallel_linear",
            "vllm.lora.ops.triton_ops.lora_shrink_op",
            "vllm.lora.punica_wrapper.punica_gpu",
            "vllm.distributed.device_communicators.custom_all_reduce",
            "vllm._custom_ops",
            "vllm._custom_ops_extra",
            "vllm.models.deepseek_v41.nvidia.model_state",
            "vllm.models.deepseek_v41.nvidia.ops.engram_host_table",
            "vllm.models.deepseek_v41.quant_config",
            "vllm.models.deepseek_v41.common.mm_preprocess",
            "vllm.distributed.device_communicators.cpu_communicator",
            "vllm.v1.core.sched.scheduler",
        )
        for package_name, side in zip(rank_packages, sides):
            side.scope["worker"] = worker()
            runner = Runner(package_name)
            side.scope["r1_engine"] = runner.engine
            runners.append(runner)
            for module_name in exact_aliases:
                side.lp.alias[module_name] = package_name + ".engine"

        def module_plan(txn_id, module_name, **extra):
            return {
                "id": txn_id,
                **extra,
                "modules": {
                    module_name: {
                        "on": ["worker"],
                        "base_hash": L.text_hash(ENG),
                        "source": ENG.replace(
                            'self.ran.append("base")', 'self.ran.append("v1")'
                        ),
                        "targets": [["Engine.step", STEP_LINE]],
                    }
                },
            }

        captured_modules = (
            "vllm.models.deepseek_v41.nvidia.model",
            "vllm.model_executor.kernels.attention.candidate_blocks",
            "vllm.kernels.triton.activation",
            "vllm.v1.attention.backends.flash_attn",
            "vllm.v1.attention.ops.triton_decode_attention",
            "vllm.lora.layers.row_parallel_linear",
            "vllm.lora.ops.triton_ops.lora_shrink_op",
            "vllm.lora.punica_wrapper.punica_gpu",
            "vllm.distributed.device_communicators.custom_all_reduce",
            "vllm._custom_ops",
        )
        for index, module_name in enumerate(captured_modules):
            refused = lab.handle(
                {"op": "txn", "plan": module_plan(f"v033-block-{index}", module_name)}
            )
            blocking = [
                (finding["check"], finding.get("rank"), finding.get("route"), finding.get("today"))
                for finding in refused["receipt"]["findings"]
            ]
            self.assertEqual(
                (refused["outcome"], blocking),
                ("aborted", [("captured_target", 0, "R1", "T1"), ("captured_target", 1, "R1", "T1")]),
                f"expected blocking captured_target for {module_name}",
            )
            self.assertIn("captured GPU code", refused["receipt"]["findings"][0]["observed"])
            self.assertEqual([runner.step(4) for runner in runners], [("base", "FULL")] * 2)

            recaptured = lab.handle(
                {
                    "op": "txn",
                    "plan": module_plan(
                        f"v033-r1-{index}", module_name, rebuild=["R1"], r1={"sizes": [4]}
                    ),
                }
            )
            findings = recaptured["receipt"]["findings"]
            self.assertEqual(recaptured["outcome"], "committed", findings)
            self.assertEqual(
                {(finding["check"], bool(finding.get("advisory"))) for finding in findings},
                {("captured_target", True)},
            )
            self.assertEqual([runner.step(4) for runner in runners], [("v1", "FULL")] * 2)
            self.assertTrue(lab.handle({"op": "undo", "txn": f"v033-r1-{index}"})["ok"])
            self.assertEqual([runner.step(4) for runner in runners], [("base", "FULL")] * 2)

        # Accepted without R1 still routes zero_witness to R1 via default CAPTURED_PREFIXES
        subject_key = f"vllm.models.deepseek_v41.nvidia.model:Engine.step:{STEP_LINE}"
        accepted = lab.handle(
            {
                "op": "txn",
                "plan": module_plan(
                    "v033-acc",
                    "vllm.models.deepseek_v41.nvidia.model",
                    accept=[{"check": "captured_target", "subject": subject_key}],
                ),
            }
        )
        self.assertEqual(accepted["outcome"], "committed")
        proof = lab.handle({"op": "prove", "txn": "v033-acc"})["receipt"]
        zero_witness_findings = [
            (finding["check"], finding.get("route"), finding.get("today"))
            for finding in proof["findings"]
            if finding["check"] == "zero_witness"
        ]
        self.assertEqual(zero_witness_findings, [("zero_witness", "R1", "T1")] * 2)
        self.assertIn("captured GPU code", proof["findings"][0]["observed"])
        self.assertTrue(lab.handle({"op": "undo", "txn": "v033-acc"})["ok"])

        # Host-side exclusions under vllm.models.* and non-captured modules must NOT block
        unblocked_modules = (
            "vllm.models.deepseek_v41.nvidia.model_state",
            "vllm.models.deepseek_v41.nvidia.ops.engram_host_table",
            "vllm.models.deepseek_v41.quant_config",
            "vllm.models.deepseek_v41.common.mm_preprocess",
            "vllm.distributed.device_communicators.cpu_communicator",
            "vllm._custom_ops_extra",
            "vllm.v1.core.sched.scheduler",
        )
        for index, module_name in enumerate(unblocked_modules):
            unblocked = lab.handle(
                {"op": "txn", "plan": module_plan(f"v033-host-{index}", module_name)}
            )
            self.assertEqual(
                (unblocked["outcome"], unblocked["receipt"]["findings"]),
                ("committed", []),
                f"expected host-side / non-captured module {module_name} not to block",
            )
            self.assertTrue(lab.handle({"op": "undo", "txn": f"v033-host-{index}"})["ok"])

        # Eager worker has no captured_target finding on vllm.models.*
        eager_package = f"ads{next(_n)}_eager_v033"
        toy.load_copy(eager_package)
        eager_side = Side(
            "worker",
            scope={"worker": worker(eager=True)},
            rank=0,
            alias={"vllm.models.deepseek_v41.nvidia.model": eager_package + ".engine"},
        )
        eager_reply = eager_side.dispatch(
            "prepare", module_plan("v033-eager", "vllm.models.deepseek_v41.nvidia.model")
        )
        self.assertEqual((eager_reply["ok"], eager_reply["findings"]), (True, []))


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
        return self.leaf(x)

    def leaf(self, x):
        return 0

    def shared(self, x):
        return x

    def _forward_core(self, x, out):
        out[0] = x * 2 + self.helper(x) + self.shared(x)


@eager_break_during_capture
def gdn_core(x, out, layer):
    self = layer
    self._forward_core(x, out)


class MoE:
    def forward(self, x):
        return self.shared(x) * 3

    def shared(self, x):
        return x
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
        # Class-qualified method resolution: Attn.shared is split-only, MoE.shared is captured
        (attn_shared,) = self._findings(bk, "Attn.shared")
        self.assertEqual(attn_shared[:3], ("captured_partly", "R1", True))
        (moe_shared,) = self._findings(bk, "MoE.shared")
        self.assertEqual(moe_shared[:3], ("captured_target", "R1", False))
        # Transitive outside caller: Attn.forward -> Attn.helper -> Attn.leaf
        (leaf_finding,) = self._findings(bk, "Attn.leaf")
        self.assertEqual(leaf_finding[:3], ("captured_target", "R1", False))
        self.assertIn("Attn.forward", leaf_finding[3])
        # FULL_DECODE_ONLY + compile NONE keeps "unknown" targets blocking
        full_decode_only = Side("worker", scope={"worker": worker("FULL_DECODE_ONLY", "NONE")},
                                rank=0)
        full_decode_only.captured_prefixes = (self.pkg + ".",)
        module_name = self.pkg + ".gdn"
        (unknown_decode_only,) = full_decode_only._captured_check(
            module_name, [f"{module_name}:Attn.forward.<locals>.inner:1"]
        )
        self.assertEqual(
            (unknown_decode_only["check"], unknown_decode_only.get("route"),
             bool(unknown_decode_only.get("advisory"))),
            ("captured_target", "R1", False),
        )
        self.assertIn("could not decide statically", unknown_decode_only["observed"])


BREAKABLE = '''"""A stand-in for vLLM's graph machinery (vllm.compilation.breakable_cudagraph, outside
the model-code prefixes): the split decorator decides at decoration time whether to wrap."""
import functools

ENABLED = False


def eager_break_during_capture(fn):
    if not ENABLED:
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


class Capture:
    def replay(self, x):
        return x + 1


@eager_break_during_capture
def attention(x):
    return x * 2
'''


class SplitDecoratorTest(unittest.TestCase):
    """Pain point (vLLM #59941, Qwen3.5 TP2 graph replay): the edit removed the early
    `return fn` from eager_break_during_capture, in vllm.compilation (not model code), so
    captured_target never looked at it; T0 committed with no finding and the witness read 0
    on every rank, since the decorator ran only when the ops were decorated at import."""

    def setUp(self):
        import importlib
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        self.pkg = f"adbreak{next(_n)}"
        os.makedirs(os.path.join(d, self.pkg))
        open(os.path.join(d, self.pkg, "__init__.py"), "w").close()
        with open(os.path.join(d, self.pkg, "breakable.py"), "w") as f:
            f.write(BREAKABLE)
        sys.path.insert(0, d)
        self.addCleanup(sys.path.remove, d)
        self.mod = self.pkg + ".breakable"
        importlib.import_module(self.mod)

    def _plan(self, tid, old, new, accept=()):
        return {"id": f"{tid}-{self.pkg}", "accept": list(accept), "modules": {self.mod: {
            "base_hash": L.text_hash(BREAKABLE), "source": BREAKABLE.replace(old, new)}}}

    def test_an_edit_of_the_split_decorator_is_refused_on_a_graph_engine(self):
        """Input: a graph-mode rank (FULL_AND_PIECEWISE, compile NONE) with the model-code
        prefixes left as vLLM's; the edit drops `if not ENABLED: return fn` from the
        decorator. Expected: prepare votes no with a blocking captured_target (route R1,
        today T1) on `eager_break_during_capture`, outside model code. Accepted, it commits,
        a step of the decorated op and of Capture.replay runs, and the proof's zero_witness
        on the decorator is routed R1 / today T1 (the check-time backstop). An edit of
        Capture.replay in the same module has no finding, and on an eager rank the decorator
        edit has none either. Fails without the fix: prepare votes yes with no finding.
        Not: vLLM's real graphs (the next GPU replay of 59941)."""
        side = Side("worker", scope={"worker": worker("FULL_AND_PIECEWISE", "NONE")}, rank=0)
        edit = ("    if not ENABLED:\n        return fn\n", "")
        r = side.dispatch("prepare", self._plan("dec", *edit))
        line = BREAKABLE.splitlines().index("def eager_break_during_capture(fn):") + 1
        key = f"{self.mod}:eager_break_during_capture:{line}"
        self.assertEqual((r["ok"], [(f["check"], f["subject"], f.get("route"), f.get("today"))
                                    for f in r["findings"]]),
                         (False, [("captured_target", key, "R1", "T1")]))
        self.assertIn("decorated its ops at import", r["findings"][0]["observed"])

        p2 = self._plan("acc", *edit, accept=[{"check": "captured_target", "subject": key}])
        self.assertTrue(side.dispatch("prepare", p2)["ok"])
        self.assertTrue(side.dispatch("commit", p2["id"])["ok"])
        m = sys.modules[self.mod]
        self.assertEqual((m.attention(3), m.Capture().replay(3)), (6, 4))
        w = side.dispatch("witness_check", p2["id"])
        self.assertEqual([(f["check"], f["subject"], f.get("route"), f.get("today"))
                          for f in w["findings"]], [("zero_witness", key, "R1", "T1")])
        self.assertIn("the split decorator", w["findings"][0]["observed"])
        self.assertTrue(side.dispatch("undo", p2["id"])["ok"])

        r = side.dispatch("prepare", self._plan("replay", "return x + 1", "return x + 2"))
        self.assertEqual((r["ok"], r["findings"]), (True, []))
        side.dispatch("abort", r.get("txn") or f"replay-{self.pkg}")
        eager = Side("worker", scope={"worker": worker(eager=True)}, rank=0)
        r = eager.dispatch("prepare", self._plan("eager", *edit))
        self.assertEqual((r["ok"], r["findings"]), (True, []))
        eager.dispatch("abort", f"eager-{self.pkg}")


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
        w.rsibox_kernel_endpoint()             # the side exists and wraps the step calls
        w.execute_model()
        self.assertTrue(w.rsibox_kernel("prepare", p)["ok"])
        w.sample_tokens()
        w.execute_model()
        w.sample_tokens()
        self.assertTrue(w.rsibox_kernel("commit", p["id"])["ok"])
        w.execute_model()
        w.sample_tokens()
        v = w.rsibox_kernel("versions")
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

    def test_rewatching_a_dropped_target_and_abstain_reset_keep_state_consistent(self):
        """Input: (a) a transaction whose watched target was queued in `_watch_drop` by an
        abort before the next step start, then re-prepared by a second transaction; (b)
        `watch` emptying on a step start and then watching the same target again; (c) `undo`
        arriving while `prepare_start` is still staging; (d) an abstaining `Endpoint` after
        `reset` or re-prepare with work. Expected: (a) the re-watched target stays in `watch`;
        (b) `_cur` resets so `vlog` records the step when watching resumes; (c) the cancelled
        prepare records state `"undone"` and leaves no watches; (d) `_abstain` clears."""
        side = Side("worker", rank=0)
        plan_first = own_plan("rewatch", targets=False)
        self.assertTrue(side.dispatch("prepare", plan_first)["ok"])
        side.tick(True)
        self.assertEqual((len(side.watch), len(side.vlog)), (1, 1))
        self.assertTrue(side.dispatch("abort", plan_first["id"])["ok"])
        # Re-watch the same module/target before the next step start drains _watch_drop
        plan_second = {**plan_first, "id": plan_first["id"] + "-2"}
        self.assertTrue(side.dispatch("prepare", plan_second)["ok"])
        self.assertTrue(side.dispatch("commit", plan_second["id"])["ok"])
        side.tick(True)
        self.assertEqual(len(side.watch), 1)
        self.assertTrue(side.dispatch("undo", plan_second["id"])["ok"])
        # Drop the watch completely and verify _cur resets when watch becomes empty
        side.watch.clear()
        side.tick(True)
        self.assertIsNone(side._cur)
        plan_third = {**plan_first, "id": plan_first["id"] + "-3"}
        self.assertTrue(side.dispatch("prepare", plan_third)["ok"])
        self.assertTrue(side.dispatch("abort", plan_third["id"])["ok"])

        slow_side = Side("worker", rank=0)
        slow_plan = own_plan("undostage", {"worker:0": {"method": "prepare", "delay_s": 0.2,
                                                        "fail": False}})
        self.assertTrue(slow_side.dispatch("prepare_start", slow_plan)["ok"])
        self.assertTrue(slow_side.dispatch("undo", slow_plan["id"])["ok"])
        prepared(slow_plan["id"])
        slow_side.tick(True)
        self.assertEqual(
            (slow_plan["id"] in slow_side.lp._staged,
             slow_side.lp._state.get(slow_plan["id"]),
             slow_side.watch),
            (False, "undone", {}),
        )

        core_ep = WE.Endpoint("core")
        worker_only = own_plan("abstain_reset")
        self.assertTrue(core_ep.dispatch("prepare", worker_only)["abstain"])
        self.assertIn(worker_only["id"], core_ep._abstain)
        (module_name,) = worker_only["modules"]
        both_roles = {
            **worker_only,
            "modules": {
                module_name: {**worker_only["modules"][module_name], "on": ["worker", "core"]}
            },
        }
        self.assertFalse(core_ep.dispatch("prepare", both_roles).get("abstain", False))
        self.assertNotIn(worker_only["id"], core_ep._abstain)
        core_ep.dispatch("abort", worker_only["id"])
        self.assertTrue(core_ep.dispatch("prepare", worker_only)["abstain"])
        core_ep.dispatch("reset", "all")
        self.assertEqual(core_ep._abstain, {})


if __name__ == "__main__":
    unittest.main()
