"""CPU golden tests for O(1) CUDA-graph staleness guard, explicit R1 pool teardown,
vLLM V1/V2 replay hook wiring, and non-blocking background native build subprocesses.

Runs cleanly in CPU-only CI without `torch` or `vllm` installed, exercising the real
`GraphReplayGuard`, `Recapture`/`teardown_graph_pool`, `KernelWorkerExtension`, and
`BackgroundBuildPool` (`subprocess.Popen`) implementations end to end.
"""
import contextlib
import enum
import gc
import os
import stat
import sys
import tempfile
import time
import types
import unittest
import weakref
from types import SimpleNamespace as NS
from unittest import mock

from kernel import locate as L

from . import graph_guard as GG
from . import native as N
from . import recapture as RC
from . import worker_ext as WE
from .test_host import ENG, STEP_LINE, make_host, plan
from .test_recapture import Runner
from .test_side import worker as make_graph_worker


class _TensorLike:
    def __init__(self, shape, dtype="float16", ptr=0x1000):
        self.shape, self.dtype, self._ptr = tuple(shape), dtype, ptr
    def data_ptr(self): return self._ptr


class _TrackedGraph:
    def __init__(self, label, reset_log):
        self.label, self._reset_log = label, reset_log
    def reset(self): self._reset_log.append(self.label)


def _plan_v2(tid, marker="v2", base_marker="v1", **kw):
    src = ENG.replace('self.ran.append("base")', f'self.ran.append("{base_marker}")')
    return {"id": tid, **kw, "modules": {"toy.engine": {
        "on": ["worker"], "base_hash": L.text_hash(src),
        "source": src.replace(f'self.ran.append("{base_marker}")', f'self.ran.append("{marker}")'),
        "targets": [["Engine.step", STEP_LINE]]}}}


class GraphGuardGoldenTests(unittest.TestCase):

    def test_stale_replay_raises_after_code_edit_without_recapture_and_passes_after_recapture(self):
        """Input: graph captured & stamped at epoch 0; live edit bumps `code_epoch` without
        recapture, then R1 recapture stamps new epoch. Expected: stale replay fails closed with
        `StaleGraphReplayError`; recapture passes in O(1); non-R1 undo restores base validity."""
        guard = GG.GraphReplayGuard()
        key = GG.graph_key("FULL", 4)
        guard.stamp(key)
        guard.check_before_replay(key)

        self.assertEqual(guard.bump_code_epoch("commit:txn-1"), 1)
        self.assertFalse(guard.is_fresh(key))
        with self.assertRaises(GG.StaleGraphReplayError) as ctx:
            guard.check_before_replay(key)
        self.assertEqual(ctx.exception.reason, "epoch_mismatch")
        self.assertEqual(ctx.exception.actual_epoch, (0, 0))
        self.assertEqual(ctx.exception.expected_epoch, (1, 0))

        guard.stamp(key)
        t0 = time.perf_counter()
        for _ in range(10000):
            guard.check_before_replay(key)
        self.assertLess(((time.perf_counter() - t0) / 10000.0) * 1e6, 50.0)

        # End-to-end Host/Side: non-R1 commit blocks stale replay; failed undo does NOT mutate
        # epoch; real undo restores base graph validity (with input_sig preserved!).
        h, names, sides = make_host(self, "guard_e2e")
        sig0 = GG.input_signature(_TensorLike((4, 64), "bfloat16", 0x2000))
        for n, s in zip(names, sides):
            r = Runner(n, modes=("FULL",), sizes=(4, 2))
            s.scope["r1_engine"] = r.engine
            s.graph_guard.stamp(GG.graph_key("FULL", 4), sig0)

        self.assertEqual(h.handle({"op": "txn", "plan": plan("g_no_r1")})["outcome"], "committed")
        for s in sides:
            self.assertEqual(s.graph_guard.code_epoch, 1)
            with self.assertRaises(GG.StaleGraphReplayError):
                s.graph_guard.check_before_replay(GG.graph_key("FULL", 4), sig0)
        # Failed undo of unknown txn must NOT mutate code_epoch
        self.assertFalse(h.handle({"op": "undo", "txn": "does_not_exist"})["ok"])
        for s in sides:
            self.assertEqual(s.graph_guard.code_epoch, 1)
        # Real non-R1 undo restores pre-txn epoch -> base graph + input_sig remain valid!
        self.assertTrue(h.handle({"op": "undo", "txn": "g_no_r1"})["ok"])
        for s in sides:
            self.assertEqual(s.graph_guard.code_epoch, 0)
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 4), sig0)

        # Next commit allocates monotonic epoch 2 (no ABA reuse of 1); R1 stamps captured bucket 4
        res_r1 = h.handle({"op": "txn", "plan": plan("g_with_r1", rebuild=["R1"], r1={"sizes": [4]})})
        self.assertEqual(res_r1["outcome"], "committed", res_r1["receipt"]["findings"])
        for s in sides:
            self.assertEqual(s.graph_guard.code_epoch, 2)
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 4))
            with self.assertRaises(GG.StaleGraphReplayError):
                s.graph_guard.check_before_replay(GG.graph_key("FULL", 2))

    def test_vllm_v1_and_v2_real_replay_hooks_startup_stamping_and_signature_drift(self):
        """Input: realistic vLLM V2 `CudaGraphManager.run_fullgraph(desc)` (with `.cg_mode`) and
        V1 `CUDAGraphWrapper.__call__` (via `_all_instances`) with startup graphs already captured,
        plus `strict_unstamped=True` and input signature drift at epoch 0. Expected: `install_replay_guard`
        stamps startup graphs, hooks V2 `run_fullgraph` & V1 `__call__`, enforces `strict_unstamped`
        and signature drift even at epoch (0, 0), and fails closed after a code edit."""
        class CUDAGraphMode(enum.Enum):
            PIECEWISE = 1
            FULL = 2

        class BatchExecutionDescriptor:
            def __init__(self, cg_mode, num_tokens):
                self.cg_mode, self.num_tokens = cg_mode, num_tokens
            def __hash__(self): return hash((self.cg_mode, self.num_tokens))
            def __eq__(self, o): return isinstance(o, BatchExecutionDescriptor) and (self.cg_mode, self.num_tokens) == (o.cg_mode, o.num_tokens)

        class FakeV2Manager:
            def __init__(self):
                self.graphs = {BatchExecutionDescriptor(CUDAGraphMode.FULL, 4): "g_full_4",
                               BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 8): "g_pw_8"}
            def run_fullgraph(self, desc, input_sig=None):
                return f"v2:{desc.cg_mode.name}:{desc.num_tokens}"

        class FakeV1Wrapper:
            _all_instances = []
            def __init__(self, runtime_mode, descs):
                self.runtime_mode = runtime_mode
                self.concrete_cudagraph_entries = {d: f"v1_{d}" for d in descs}
                self.runner = NS(capturing=False)
                self._all_instances.append(self)
            def __call__(self, desc, input_sig=None):
                return f"v1:{self.runtime_mode.name}:{desc}"

        v1_full = FakeV1Wrapper(CUDAGraphMode.FULL, [4])
        v2_mgr = FakeV2Manager()
        d_full = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4)
        d_pw = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 8)
        d_unseen = BatchExecutionDescriptor(CUDAGraphMode.FULL, 99)

        class _FakeWorker(WE.KernelWorkerExtension):
            rank = 0
            def __init__(self): self.model_runner = NS(cudagraph_manager=v2_mgr)
            def execute_model(self, *a, **k): return None
            def sample_tokens(self, *a, **k): return None

        mods = {"vllm.compilation.cuda_graph": NS(CUDAGraphWrapper=FakeV1Wrapper)}
        saved = list(WE._ENDPOINT)
        WE._ENDPOINT.clear()
        try:
            with mock.patch.dict(sys.modules, mods):
                w = _FakeWorker()
                w.rsibox_kernel_endpoint()
                guard = w.rsibox_graph_guard()
                # Startup graphs were automatically stamped with their real cg_mode!
                self.assertIn(GG.graph_key(CUDAGraphMode.FULL, d_full), guard.stamps)
                self.assertIn(GG.graph_key(CUDAGraphMode.PIECEWISE, d_pw), guard.stamps)
                self.assertIn(GG.graph_key(CUDAGraphMode.FULL, 4), guard.stamps)

                # Epoch 0 replays succeed via BOTH V2 run_fullgraph and V1 __call__
                self.assertEqual(v2_mgr.run_fullgraph(d_full), "v2:FULL:4")
                self.assertEqual(v2_mgr.run_fullgraph(d_pw), "v2:PIECEWISE:8")
                self.assertEqual(v1_full(4), "v1:FULL:4")

                # Input signature drift fails closed even at epoch (0, 0)!
                sig0 = GG.input_signature(_TensorLike((4, 64), "torch.bfloat16", 0x1000))
                guard.stamp(GG.graph_key(CUDAGraphMode.FULL, d_full), sig0)
                self.assertEqual(v2_mgr.run_fullgraph(d_full, input_sig=sig0), "v2:FULL:4")
                for bad in (_TensorLike((8, 64), "torch.bfloat16", 0x1000),
                            _TensorLike((4, 64), "torch.float16", 0x1000),
                            _TensorLike((4, 64), "torch.bfloat16", 0x9000)):
                    with self.assertRaises(GG.StaleGraphReplayError) as cm:
                        v2_mgr.run_fullgraph(d_full, input_sig=GG.input_signature(bad))
                    self.assertEqual(cm.exception.reason, "input_sig_mismatch")

                # Code edit without recapture blocks BOTH V2 run_fullgraph and V1 __call__
                guard.bump_code_epoch("live_edit")
                with self.assertRaises(GG.StaleGraphReplayError):
                    v2_mgr.run_fullgraph(d_full)
                with self.assertRaises(GG.StaleGraphReplayError):
                    v1_full(4)

                # strict_unstamped=True fails closed at epoch (0, 0) via attach_wrapper too
                strict_guard = GG.GraphReplayGuard(strict_unstamped=True)
                strict_mgr = FakeV2Manager()
                strict_guard.attach_wrapper(strict_mgr, method="run_fullgraph")
                with self.assertRaises(GG.StaleGraphReplayError) as cm_u:
                    strict_mgr.run_fullgraph(d_unseen)
                self.assertEqual(cm_u.exception.reason, "unstamped")

            # Realistic V2 FULL_AND_PIECEWISE R1: FULL lives in mgr.graphs (BatchExecutionDescriptor),
            # PIECEWISE lives in CUDAGraphWrapper (BatchDescriptor, different key type!) AND a
            # breakable_cg_runner with runtime_mode=None. Verify:
            # (a) startup stamps breakable_cg_runner under "PIECEWISE" (not "None");
            # (b) partial-mode R1 (modes=["FULL"]) does NOT wipe runtime_mode=None breakable wrapper;
            # (c) FULL_AND_PIECEWISE R1 stamps BOTH BatchExecutionDescriptor AND BatchDescriptor so
            #     both V2 run_fullgraph AND V1/breakable __call__ replay cleanly post-R1!
            class BatchDescriptor:
                def __init__(self, num_tokens): self.num_tokens = num_tokens
                def __hash__(self): return hash(("BD", self.num_tokens))
                def __eq__(self, o): return isinstance(o, BatchDescriptor) and self.num_tokens == o.num_tokens

            class _PiecewiseWrapper:
                _all_instances = []
                def __init__(self, runtime_mode):
                    self.runtime_mode = runtime_mode
                    self.concrete_cudagraph_entries = {BatchDescriptor(4): "pw_v0"}
                    self.runner = NS(capturing=False)
                    self._all_instances.append(self)
                def __call__(self, desc, input_sig=None):
                    return self.concrete_cudagraph_entries[desc]

            class _BreakableRunner:
                def __init__(self):
                    self.runtime_mode = None  # vLLM V2 breakable runner has runtime_mode=None
                    self.entries = {BatchDescriptor(4): "br_v0"}
                    self.runner = NS(capturing=False)
                def __call__(self, desc, input_sig=None):
                    return self.entries[desc]

            pw_wrap = _PiecewiseWrapper(CUDAGraphMode.PIECEWISE)
            br_runner = _BreakableRunner()
            d_exec_full = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4)
            d_exec_pw = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 4)
            d_batch = BatchDescriptor(4)

            class _V2HybridMgr:
                def __init__(self):
                    self.graphs = {d_exec_full: "full_v0"}
                    self.breakable_cg_runner = br_runner
                def run_fullgraph(self, desc, input_sig=None):
                    return self.graphs[desc]

            hybrid_mgr = _V2HybridMgr()
            hybrid_worker = NS(model_runner=NS(cudagraph_manager=hybrid_mgr))
            guard2 = GG.GraphReplayGuard()
            with mock.patch.dict(sys.modules, {"vllm.compilation.cuda_graph": NS(CUDAGraphWrapper=_PiecewiseWrapper)}):
                WE.install_replay_guard(hybrid_worker, guard2)
                # (a) breakable_cg_runner stamped under "PIECEWISE", never "None"
                self.assertIn(GG.graph_key("PIECEWISE", d_batch), guard2.stamps)
                self.assertNotIn(GG.graph_key(None, d_batch), guard2.stamps)
                self.assertEqual(pw_wrap(d_batch), "pw_v0")
                self.assertEqual(br_runner(d_batch), "br_v0")
                self.assertEqual(hybrid_mgr.run_fullgraph(d_exec_full), "full_v0")

                # Build Recapture Engine matching V2 FULL + PIECEWISE + runtime_mode=None breakable
                full_w = RC._V2FullWrapper(hybrid_mgr, CUDAGraphMode.FULL)
                keys2 = {CUDAGraphMode.FULL: {d_exec_full}, CUDAGraphMode.PIECEWISE: {d_exec_pw}}
                ver = ["v1"]
                def _cap2(mode, desc):
                    if mode == CUDAGraphMode.FULL:
                        hybrid_mgr.graphs[desc] = f"full_{ver[0]}"
                    else:
                        pw_wrap.concrete_cudagraph_entries[d_batch] = f"pw_{ver[0]}"
                        br_runner.entries[d_batch] = f"br_{ver[0]}"
                eng2 = RC.Engine(lambda: [full_w, pw_wrap, br_runner], keys2, _cap2)
                rec2 = RC.Recapture(guard=guard2)
                j2 = NS(marks=[("t1", 0)], entries=[], record_call=lambda fn, lbl: j2.entries.append((fn, lbl)))
                w2 = NS(capture=contextlib.nullcontext)

                # (b) Partial-mode R1 (modes=["FULL"] only) must NOT wipe runtime_mode=None breakable!
                guard2.bump_code_epoch("t1")
                r_full_only = rec2.run(eng2, "t1", j2, w2, modes=["FULL"])
                self.assertTrue(r_full_only["ok"], r_full_only)
                self.assertEqual(br_runner.entries[d_batch], "br_v0")
                self.assertEqual(hybrid_mgr.run_fullgraph(d_exec_full), "full_v1")

                # (c) Full + Piecewise R1 stamps BOTH BatchExecutionDescriptor AND BatchDescriptor!
                ver[0] = "v2"
                j2.marks.append(("t2", len(j2.entries)))
                guard2.bump_code_epoch("t2")
                r_both = rec2.run(eng2, "t2", j2, w2)
                self.assertTrue(r_both["ok"], r_both)
                self.assertEqual(hybrid_mgr.run_fullgraph(d_exec_full), "full_v2")
                self.assertEqual(pw_wrap(d_batch), "pw_v2")
                self.assertEqual(br_runner(d_batch), "br_v2")
        finally:
            WE._ENDPOINT[:] = saved

    def test_selective_code_epoch_partial_mode_r1_failure_rollback_and_stale_stack_undo(self):
        """Covers all subtle epoch state-machine invariants:
        (1) Unrelated non-captured commit does NOT bump `code_epoch` on a graph-mode worker;
        (2) Partial-mode R1 (`modes=['PIECEWISE']`) bumps ONLY PIECEWISE `pool_epoch`, leaving
            FULL graphs untouched and valid (and empty-selected R1 does not bump `pool_epoch`);
        (3) 2-rank R1 capture failure on rank 1 rolls back cleanly: during failure `restore()`
            keeps pre-R1 stamps (no transient stale bypass!), and after coordinator `undo` BOTH
            ranks match base cleanly with zero divergence or lockout;
        (4) `txn1 (non-R1) -> txn2 (R1) -> undo(txn2)`: restored base graphs preserve their
            epoch-0 stamp and STAY REJECTED under `txn1` (epoch 1), then pass after `undo(txn1)`."""
        # (1) Unrelated non-captured Python commit on graph-mode worker does NOT bump code_epoch
        h0, names0, sides0 = make_host(self, "noncap")
        for n, s in zip(names0, sides0):
            s.scope["worker"] = make_graph_worker()
            s.captured_prefixes = ("vllm.model_executor.models.",)  # toy.engine is NOT captured
            s.graph_guard.stamp(GG.graph_key("FULL", 4))
        self.assertEqual(h0.handle({"op": "txn", "plan": plan("nc1")})["outcome"], "committed")
        for s in sides0:
            self.assertEqual(s.graph_guard.code_epoch, 0)
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 4))

        # (2) Partial-mode R1 bumps ONLY PIECEWISE pool_epoch; FULL untouched; empty selected no-op
        h1, names1, sides1 = make_host(self, "partmode")
        for n, s in zip(names1, sides1):
            r = Runner(n, modes=("PIECEWISE", "FULL"), sizes=(4, 2))
            s.scope["r1_engine"] = r.engine
            s.graph_guard.stamp(GG.graph_key("FULL", 4))
            s.graph_guard.stamp(GG.graph_key("PIECEWISE", 4))
        rep_pw = h1.handle({"op": "txn", "plan": plan("pm1", rebuild=["R1"], r1={"modes": ["PIECEWISE"], "sizes": [4]})})
        self.assertEqual(rep_pw["outcome"], "committed")
        for s in sides1:
            self.assertEqual(s.graph_guard.pool_epoch_for("PIECEWISE"), 1)
            self.assertEqual(s.graph_guard.pool_epoch_for("FULL"), 0)
            s.graph_guard.check_before_replay(GG.graph_key("PIECEWISE", 4))
        self.assertTrue(h1.handle({"op": "undo", "txn": "pm1"})["ok"])
        for s in sides1:
            self.assertEqual(s.graph_guard.pool_epoch_for("PIECEWISE"), 0)
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 4))
            s.graph_guard.check_before_replay(GG.graph_key("PIECEWISE", 4))
        # Teardown with empty `selected` does NOT bump pool_epoch
        g_empty = GG.GraphReplayGuard()
        td_empty = RC.teardown_graph_pool(None, [], [], guard=g_empty)
        self.assertEqual(g_empty.pool_epoch, 0)
        self.assertEqual(td_empty["pool_epoch"], 0)

        # (3) 2-rank R1 capture failure on rank 1 -> no transient bypass + clean rollback on both ranks
        h2, names2, sides2 = make_host(self, "r1fail_guard")
        runners2 = []
        sig_base = GG.input_signature(_TensorLike((4, 16), "float16", 0x5000))
        for n, s in zip(names2, sides2):
            r = Runner(n, modes=("FULL",), sizes=(4, 2))
            s.scope["r1_engine"] = r.engine
            s.graph_guard.stamp(GG.graph_key("FULL", 4), sig_base)
            s.graph_guard.stamp(GG.graph_key("FULL", 2), sig_base)
            runners2.append(r)
        # Intercept rank 1's restore to assert that right after R1 fails (before undo!), base
        # graphs are STILL rejected under live edited code (no transient staleness bypass).
        transient_rejected = []
        orig_rec = sides2[1].recapture.run
        def _spy_run(*a, **k):
            res = orig_rec(*a, **k)
            if not res["ok"]:
                try:
                    sides2[1].graph_guard.check_before_replay(GG.graph_key("FULL", 4), sig_base)
                except GG.StaleGraphReplayError:
                    transient_rejected.append(True)
            return res
        sides2[1].recapture.run = _spy_run
        runners2[1].fail_at = ("FULL", 2)
        res_fail = h2.handle({"op": "txn", "plan": plan("r1boom", rebuild=["R1"])})
        self.assertEqual(res_fail["outcome"], "rolled_back")
        self.assertEqual(transient_rejected, [True])
        # After coordinator undo completes, BOTH ranks match base with sig_base preserved!
        for s in sides2:
            self.assertEqual((s.graph_guard.code_epoch, s.graph_guard.pool_epoch), (0, 0))
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 4), sig_base)
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 2), sig_base)

        # (4) txn1 (non-R1) -> txn2 (R1) -> undo(txn2): stale base graphs STAY REJECTED under txn1!
        h3, names3, sides3 = make_host(self, "stale_stack")
        for n, s in zip(names3, sides3):
            r = Runner(n, modes=("FULL",), sizes=(4, 2))
            s.scope["r1_engine"] = r.engine
            s.graph_guard.stamp(GG.graph_key("FULL", 4), sig_base)
        self.assertEqual(h3.handle({"op": "txn", "plan": plan("st1")})["outcome"], "committed")
        self.assertEqual(h3.handle({"op": "txn", "plan": _plan_v2("st2", rebuild=["R1"])})["outcome"], "committed")
        for s in sides3:
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 4))
        self.assertTrue(h3.handle({"op": "undo", "txn": "st2"})["ok"])
        for s in sides3:
            # Live code is st1 (epoch 1), restored graphs are base (epoch 0) -> MUST fail closed!
            with self.assertRaises(GG.StaleGraphReplayError):
                s.graph_guard.check_before_replay(GG.graph_key("FULL", 4), sig_base)
        self.assertTrue(h3.handle({"op": "undo", "txn": "st1"})["ok"])
        for s in sides3:
            # Back at base (epoch 0) -> restored base graphs + sig_base now pass!
            s.graph_guard.check_before_replay(GG.graph_key("FULL", 4), sig_base)

    def test_repeated_r1_teardown_releases_old_graph_refs_and_increments_pool_epoch(self):
        """Input: repeated R1 recaptures with `release_old=True` holding tracked graph objects.
        Expected: old graph refs become `None` (`weakref.ref`), per-mode `pool_epoch` increments,
        CUDA `synchronize()` + `empty_cache()` run, and VRAM/timing telemetry is recorded."""
        reset_log = []
        cuda_events = {"sync": 0, "empty": 0, "reserved": 512 * 1024 * 1024}
        fake_torch = types.ModuleType("torch")
        def _empty():
            cuda_events["empty"] += 1
            cuda_events["reserved"] = 128 * 1024 * 1024
        fake_torch.cuda = NS(is_available=lambda: True,
                             synchronize=lambda: cuda_events.__setitem__("sync", cuda_events["sync"] + 1),
                             empty_cache=_empty, memory_reserved=lambda: cuda_events["reserved"])

        class _StubJournal:
            def __init__(self): self.marks, self.entries = [], []
            def mark(self, txn): self.marks.append((txn, len(self.entries)))
            def record_call(self, fn, label): self.entries.append((fn, label))

        wrapper, keys, gen = NS(runtime_mode="FULL", concrete_cudagraph_entries={}), {"FULL": {4, 2}}, [0]
        def cap(mode, desc):
            wrapper.concrete_cudagraph_entries[desc] = _TrackedGraph(f"g{gen[0]}_{desc}", reset_log)
        engine = RC.Engine(lambda: [wrapper], keys, cap, memory=lambda: {"reserved": cuda_events["reserved"]})
        for d in (4, 2): cap("FULL", d)
        guard = GG.GraphReplayGuard()
        for d in (4, 2): guard.stamp(GG.graph_key("FULL", d))
        rec, journal, witness = RC.Recapture(guard=guard), _StubJournal(), NS(capture=contextlib.nullcontext)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            for cycle in range(1, 4):
                old_refs = [weakref.ref(obj) for obj in wrapper.concrete_cudagraph_entries.values()]
                cuda_events["reserved"] = 512 * 1024 * 1024
                gen[0], txn = cycle, f"txn-r1-{cycle}"
                journal.mark(txn)
                guard.bump_code_epoch(f"commit:{txn}")
                report = rec.run(engine, txn, journal, witness, release_old=True)
                self.assertTrue(report["ok"], report)
                gc.collect()
                self.assertTrue(all(r() is None for r in old_refs))
                self.assertEqual(guard.pool_epoch_for("FULL"), cycle)
                self.assertEqual((report["freed_graphs_count"], report["teardown_vram_before_mib"],
                                  report["teardown_vram_after_mib"]), (2, 512.0, 128.0))
                self.assertGreaterEqual(report["teardown_s"], 0.0)
                guard.check_before_replay(GG.graph_key("FULL", 4))
        self.assertEqual((cuda_events["sync"], cuda_events["empty"], len(reset_log)), (3, 3, 6))

    def test_background_build_concurrency_sigterm_ignoring_cancel_and_spec_path(self):
        """Input: real background subprocesses via `BackgroundBuildPool.compile_async`.
        Expected: (1) foreground work runs concurrently while build completes; (2) duplicate active
        `job_id` is rejected; (3) `cancel()` kills a grandchild that IGNORES `SIGTERM` even after
        the leader exits (no orphan process); (4) `spec=` path invokes `build_native(spec)` in an
        isolated child with `CUDA_VISIBLE_DEVICES=""` and preserves caller `PYTHONPATH`."""
        with tempfile.TemporaryDirectory() as tmp, N.BackgroundBuildPool() as pool:
            artifact = os.path.join(tmp, "compiled_kernel.so")
            build_script = "import sys, time; time.sleep(0.25); open(sys.argv[1], 'wb').write(b'BUILT_OK')"
            job = pool.compile_async(cmd=[sys.executable, "-c", build_script, artifact], job_id="fast-build")
            self.assertIsNone(job.poll())
            with self.assertRaises(ValueError):
                pool.compile_async(cmd=[sys.executable, "-c", "pass"], job_id="fast-build")
            fg_steps = 0
            while job.poll() is None:
                fg_steps += 1
                time.sleep(0.01)
            self.assertGreater(fg_steps, 3)
            res = job.wait(timeout_s=5.0)
            self.assertTrue(res["ok"] and not res["cancelled"])
            with open(artifact, "rb") as fh:
                self.assertEqual(fh.read(), b"BUILT_OK")

            # Grandchild explicitly IGNORES SIGTERM; leader exits immediately on SIGTERM ->
            # cancel() MUST still SIGKILL the process group so the grandchild cannot survive!
            child_pid_file = os.path.join(tmp, "grandchild.pid")
            slow_script = (
                "import signal, subprocess, sys, time\n"
                "gc_code = 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'\n"
                "p = subprocess.Popen([sys.executable, '-c', gc_code])\n"
                "with open(sys.argv[1], 'w') as fh: fh.write(str(p.pid))\n"
                "time.sleep(60)\n"
            )
            slow = N.compile_async(pool=pool, cmd=[sys.executable, "-c", slow_script, child_pid_file], job_id="slow-ig")
            deadline = time.perf_counter() + 5.0
            while not os.path.isfile(child_pid_file) or os.path.getsize(child_pid_file) == 0:
                self.assertLess(time.perf_counter(), deadline)
                time.sleep(0.01)
            with open(child_pid_file, "r", encoding="utf-8") as fh:
                gc_pid = int(fh.read().strip())
            with self.assertRaises(TimeoutError):
                pool.wait("slow-ig", timeout_s=0.05)
            self.assertTrue(pool.cancel("slow-ig", grace_s=0.1))
            self.assertTrue(slow.cancelled)
            for pid in (slow.pid, gc_pid):
                gone = False
                for _ in range(50):
                    try: os.kill(pid, 0)
                    except ProcessLookupError: gone = True; break
                    time.sleep(0.02)
                self.assertTrue(gone, f"process {pid} survived cancel()")

            # Real `spec=` background build path: runs `build_native(spec)` in child subprocess
            # with CUDA_VISIBLE_DEVICES="" and preserves caller PYTHONPATH!
            csrc = os.path.join(tmp, "csrc")
            os.makedirs(csrc)
            with open(os.path.join(csrc, "k.cu"), "w", encoding="utf-8") as fh:
                fh.write("// kernel\n")
            fake_torch_dir = os.path.join(tmp, "fakepkg")
            os.makedirs(os.path.join(fake_torch_dir, "torch", "include"))
            os.makedirs(os.path.join(fake_torch_dir, "torch", "lib"))
            with open(os.path.join(fake_torch_dir, "torch", "__init__.py"), "w", encoding="utf-8") as fh:
                fh.write(
                    "import os\nassert os.environ.get('CUDA_VISIBLE_DEVICES') == '', os.environ.get('CUDA_VISIBLE_DEVICES')\n"
                )
            fake_nvcc = os.path.join(tmp, "fake_nvcc")
            with open(fake_nvcc, "w", encoding="utf-8") as fh:
                fh.write(
                    f"#!{sys.executable}\nimport sys\n"
                    "out = sys.argv[sys.argv.index('-o') + 1]\n"
                    "with open(out, 'wb') as f: f.write(b'ELF_OK')\n"
                )
            os.chmod(fake_nvcc, os.stat(fake_nvcc).st_mode | stat.S_IEXEC)
            spec = {"csrc_root": csrc, "build_root": os.path.join(tmp, "out"), "files": ["k.cu"],
                    "ops": {"my_op": {"fn": "my_op", "schema": "my_op(Tensor x) -> ()"}}, "arch": "100"}
            spec_job = pool.compile_async(
                spec, job_id="spec-build",
                env={"PYTHONPATH": fake_torch_dir, "RSIBOX_NVCC": fake_nvcc, "RSIBOX_GXX": fake_nvcc},
            )
            spec_res = spec_job.wait(timeout_s=15.0)
            self.assertTrue(spec_res["ok"], spec_res)
            self.assertTrue(os.path.isfile(spec_res["result"]["so"]))


if __name__ == "__main__":
    unittest.main()
