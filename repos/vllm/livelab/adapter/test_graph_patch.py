"""Golden CPU tests for Tier-A surgical CUDA-graph kernel-node pointer swapping (`graph_patch.py`).

Tests the full Tier-A lifecycle with a SWIG-faithful mock CUDA driver (including mutable
`params.func` views), real `kernel.journal` / `kernel.witness` / `kernel.swap` / `kernel.jit`,
real `GraphReplayGuard`, real `Recapture`, and real `Side("worker")`, with zero `torch` or GPU
imports. Each test targets a concrete failure mode and fails without the implementation.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "core" / "livelab"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel import journal as J  # noqa: E402
from kernel import witness as W  # noqa: E402

from adapter import graph_guard as GG  # noqa: E402
from adapter import graph_patch as GP  # noqa: E402
from adapter import recapture as RC  # noqa: E402
from adapter import side as SD  # noqa: E402


# -- SWIG-faithful mock CUDA graph & driver -------------------------------------------

class _SwigFuncView:
    """Simulates SWIG's `params.func` proxy: mutating `params.func` mutates the same C struct
    slot, so holding `old = params.func` without `int(params.func)` loses the old handle."""

    def __init__(self, params_ref: "_SwigKernelParams"):
        self._p = params_ref

    def __int__(self) -> int:
        return self._p._func_val


class _SwigKernelParams:
    def __init__(self, func_int: int, block=(128, 1, 1), grid=(4, 1, 1), shm=0):
        self._func_val = int(func_int)
        self._view = _SwigFuncView(self)
        self.blockDimX, self.blockDimY, self.blockDimZ = block
        self.gridDimX, self.gridDimY, self.gridDimZ = grid
        self.sharedMemBytes = int(shm)

    @property
    def func(self):
        return self._view

    @func.setter
    def func(self, val):
        self._func_val = int(val)


class _MockNode:
    def __init__(self, node_id: int, kind: str, params: _SwigKernelParams | None = None):
        self.node_id = node_id
        self.kind = kind
        self.params = params

    def __int__(self) -> int:
        return self.node_id


class _MockCUDAGraph:
    def __init__(self, graph_id: int, exec_id: int):
        self._g = graph_id
        self._e = exec_id

    def raw_cuda_graph(self) -> int:
        return self._g

    def raw_cuda_graph_exec(self) -> int:
        return self._e


class _MockGraphEntry:
    def __init__(self, cg: _MockCUDAGraph):
        self.cudagraph = cg


class _MockWrapper:
    def __init__(self, mode: str, entries: dict):
        self.runtime_mode = mode
        self.concrete_cudagraph_entries = entries


class _MockDriver:
    """In-memory CUDA graph driver tracking template `CUgraph` nodes separately from live
    `CUgraphExec` kernel params (matching real CUDA semantics where `cuGraphExecKernelNodeSetParams`
    updates `h_exec` without mutating the template `h_graph`)."""

    def __init__(self, graphs_nodes: dict[int, list[_MockNode]], *, fail_on_call: int | None = None):
        self.graphs_nodes = graphs_nodes
        self.exec_params: dict[tuple[int, int], tuple[int, tuple, tuple, int]] = {}
        self.set_calls: list[tuple[int, int, int]] = []
        self._attempts = 0
        self.fail_on_call = fail_on_call
        for nodes in graphs_nodes.values():
            for nd in nodes:
                if nd.kind == "kernel" and nd.params is not None:
                    # initialized lazily per exec on read/write
                    pass

    def bind_exec(self, h_exec: int, h_graph: int) -> None:
        for nd in self.graphs_nodes[h_graph]:
            if nd.kind == "kernel" and nd.params is not None:
                p = nd.params
                self.exec_params[(h_exec, nd.node_id)] = (
                    int(p.func),
                    (p.blockDimX, p.blockDimY, p.blockDimZ),
                    (p.gridDimX, p.gridDimY, p.gridDimZ),
                    p.sharedMemBytes,
                )

    def get_graph_nodes(self, h_graph_int: int):
        return [(nd, int(nd), nd.kind) for nd in self.graphs_nodes[h_graph_int]]

    def get_kernel_node_params(self, node: _MockNode):
        p = node.params
        # Return a fresh SWIG view over the template node's base params (template never changes).
        copy_p = _SwigKernelParams(int(p.func), (p.blockDimX, p.blockDimY, p.blockDimZ),
                                   (p.gridDimX, p.gridDimY, p.gridDimZ), p.sharedMemBytes)
        return (
            copy_p, int(copy_p.func),
            (copy_p.blockDimX, copy_p.blockDimY, copy_p.blockDimZ),
            (copy_p.gridDimX, copy_p.gridDimY, copy_p.gridDimZ),
            copy_p.sharedMemBytes,
        )

    def set_exec_kernel_node_params(self, h_exec_int, node, params, func_int, block, grid, shm):
        self._attempts += 1
        if self.fail_on_call is not None and self._attempts == self.fail_on_call:
            raise GP.CudaDriverError("injected CUDA_ERROR_INVALID_VALUE")
        # Mutate SWIG params in place just like cuda.bindings.driver
        params.func = int(func_int)
        params.blockDimX, params.blockDimY, params.blockDimZ = block
        params.gridDimX, params.gridDimY, params.gridDimZ = grid
        params.sharedMemBytes = int(shm)
        self.exec_params[(int(h_exec_int), int(node))] = (int(func_int), tuple(block), tuple(grid), int(shm))
        self.set_calls.append((int(h_exec_int), int(node), int(func_int)))


# -- Duck-typed Triton JITFunction & CompiledKernel -----------------------------------

class _FakeASTSource:
    def __init__(self, fn, signature, constants, attrs):
        self.fn, self.signature, self.constants, self.attrs = fn, dict(signature), dict(constants), dict(attrs)


class _FakeMeta:
    def __init__(self, **kw):
        defaults = dict(
            warp_size=32, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None,
            debug=False, enable_fp_fusion=True, enable_reflect_ftz=True,
            launch_cooperative_grid=False, launch_pdl=False,
            global_scratch_size=0, profile_scratch_size=0, tmem_size=0,
            tensordesc_meta=[], shared=0, sanitize_overflow=True,
            instrumentation_mode="", target="sm100",
        )
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


class _FakeCompiledKernel:
    def __init__(self, func_id: int, src: _FakeASTSource, meta: _FakeMeta):
        self.function = func_id
        self.src = src
        self.metadata = meta
        self.inited = False

    def _init_handles(self):
        self.inited = True


class JITFunction:
    """Implements all `kernel.jit.NEED` attributes plus `ASTSource` and `compile`."""

    ASTSource = _FakeASTSource

    def __init__(self, fn, src_text: str, *, start_line: int = 1, meta_override=None, next_func_id=None):
        self.fn = fn
        self.__globals__ = fn.__globals__
        self._src = src_text
        self.raw_src = src_text.splitlines(keepends=True)
        self.starting_line_number = start_line
        self.def_file_line_number = start_line + 1
        self.signature = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"}
        self.hash = "h0"
        self.used_global_vals = {}
        self.pre_run_hooks = []
        self.device_caches = {}
        self._meta_override = meta_override or {}
        self._next_func_id = next_func_id or iter(range(2000, 9000))
        self.compile_calls = 0

    def _unsafe_update_src(self, new_src: str):
        self._src = new_src

    def parse(self):
        return ast.parse(self._src)

    def compile(self, src, target=None, options=None):
        self.compile_calls += 1
        meta_kw = dict(options or {})
        meta_kw.update(self._meta_override)
        return _FakeCompiledKernel(next(self._next_func_id), src, _FakeMeta(**meta_kw))


_FakeJIT = JITFunction


def _make_engine(wrapper_entries: dict, modes=("FULL",)):
    wrappers = [_MockWrapper("FULL", wrapper_entries)]
    keys = {"FULL": set(wrapper_entries.keys())}
    capture_calls = []

    def capture(mode, desc):
        capture_calls.append((mode, desc))
        wrapper_entries[desc] = _MockGraphEntry(_MockCUDAGraph(0, 9999))

    eng = RC.Engine(lambda: wrappers, keys, capture, memory=lambda: {"reserved": 100})
    return eng, capture_calls


class TierAPointerSwapTests(unittest.TestCase):
    def _build_kernel_and_graphs(self, *, buckets=(1, 2, 4, 8), meta_override=None, keep_graph=True):
        def my_k(x_ptr, out_ptr, n, BLOCK):
            return None
        my_k.__module__ = "vllm.v1.attention.ops.fake_triton"
        my_k.__qualname__ = "my_k"

        src0 = "def my_k(x_ptr, out_ptr, n, BLOCK):\n    return 1\n"
        jf = _FakeJIT(my_k, src0, meta_override=meta_override)
        old_src = _FakeASTSource(jf, jf.signature, {(3,): 256}, {(0,): [["tt.divisibility", 16]]})
        old_ck = _FakeCompiledKernel(1001, old_src, _FakeMeta(num_warps=4, shared=0))
        jf.device_caches[0] = ({"spec0": old_ck}, {}, "sm100", None, None)

        graphs_nodes = {}
        entries = {}
        drv = _MockDriver(graphs_nodes)
        nid = 10
        for i, b in enumerate(buckets):
            gid = (100 + i) if keep_graph else 0
            eid = 500 + i
            cg = _MockCUDAGraph(gid, eid)
            entries[b] = _MockGraphEntry(cg)
            if gid:
                # 2 matching kernel nodes + 1 unrelated kernel node per bucket
                n1 = _MockNode(nid, "kernel", _SwigKernelParams(1001, (128, 1, 1), (b, 1, 1), 0))
                n2 = _MockNode(nid + 1, "kernel", _SwigKernelParams(7777, (256, 1, 1), (b, 1, 1), 64))
                n3 = _MockNode(nid + 2, "kernel", _SwigKernelParams(1001, (128, 1, 1), (b * 2, 1, 1), 0))
                nid += 3
                graphs_nodes[gid] = [n1, n2, n3]
                drv.bind_exec(eid, gid)
        return jf, old_ck, entries, drv

    def test_swig_safe_pointer_swap_multi_bucket_witness_and_undo(self):
        """Proves Tier-A patches all 4 buckets (8 matching nodes) with 0 dummy forwards,
        credits capture_counts & live replay witness honestly (0 false positives on unpatched
        graphs), and SWIG-safely restores the exact original CUfunction/block/shm on undo."""
        jf, old_ck, entries, drv = self._build_kernel_and_graphs(
            buckets=(1, 2, 4, 8), meta_override={"num_warps": 8, "shared": 4096},
        )
        # Add an extra bucket with ONLY unrelated kernels to verify zero false-positive witness.
        entries[16] = _MockGraphEntry(_MockCUDAGraph(199, 599))
        drv.graphs_nodes[199] = [_MockNode(999, "kernel", _SwigKernelParams(8888))]
        drv.bind_exec(599, 199)

        eng, capture_calls = _make_engine(entries)
        journal, witness = J.Journal("w"), W.Witness()
        guard = GG.GraphReplayGuard()
        for b in entries:
            guard.stamp(GG.graph_key("FULL", b))

        wk = "vllm.v1.attention.ops.fake_triton:my_k:1"
        lk = W.launch_key(wk, "vllm.v1.attention.ops.fake_triton.my_k")
        witness.arm(wk, journal)
        witness.arm_launch(wk, "vllm.v1.attention.ops.fake_triton.my_k", journal)

        jit_snap = {
            "eligible": True, "reason": None, "witness_keys": [wk],
            "specs": [{
                "jf": jf, "witness_key": wk, "kernel_label": "vllm.v1.attention.ops.fake_triton.my_k",
                "device": 0, "cache_key": "spec0", "old_ck": old_ck, "old_func": 1001,
                "target": "sm100", "backend": None, "cache_shell": ({}, "sm100", None, None),
            }],
        }
        # Simulate jit.apply clearing caches & bumping code_epoch.
        mark = journal.mark("t1")
        jf.device_caches.clear()
        guard.bump_code_epoch("commit:t1")

        rec = RC.Recapture(guard=guard)
        res = rec.run(eng, "t1", journal, witness, jit_snap=jit_snap, driver=drv)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["tier"], "tier_a_pointer_swap")
        self.assertEqual(res["patched_nodes"], 8)
        self.assertEqual(res["patched_graphs"], 4)
        self.assertEqual(res["total_graphs"], 5)
        self.assertEqual(res["compiled_specs"], 1)
        self.assertEqual(capture_calls, [])  # ZERO dummy forwards / stream captures!
        self.assertEqual(jf.compile_calls, 1)

        # Verify all 8 matching nodes now point to 2000 with updated blockDimX=256 and shm=4096,
        # while unrelated nodes (7777, 8888) are completely untouched.
        new_func = int(jf.device_caches[0][0]["spec0"].function)
        self.assertEqual(new_func, 2000)
        for i in range(4):
            eid = 500 + i
            n_base = 10 + i * 3
            self.assertEqual(drv.exec_params[(eid, n_base)][0], new_func)
            self.assertEqual(drv.exec_params[(eid, n_base)][1], (256, 1, 1))
            self.assertEqual(drv.exec_params[(eid, n_base)][3], 4096)
            self.assertEqual(drv.exec_params[(eid, n_base + 1)][0], 7777)
            self.assertEqual(drv.exec_params[(eid, n_base + 2)][0], new_func)

        # Two-ledger witness honesty: before any live replay, capture_counts > 0, live counts == 0.
        self.assertEqual(witness.capture_counts.get(wk), 8)
        self.assertEqual(witness.counts.get(wk), 0)

        # Replaying unrelated bucket 16 (no patched nodes) MUST NOT credit live witness.
        guard.check_before_replay(GG.graph_key("FULL", 16))
        self.assertEqual(witness.counts.get(wk), 0)

        # Replaying patched bucket 4 credits live witness honestly.
        guard.check_before_replay(GG.graph_key("FULL", 4))
        self.assertEqual(witness.counts.get(wk), 1)
        self.assertEqual(witness.counts.get(lk), 1)

        # Undo restores exact original func=1001, block=(128,1,1), shm=0 (proving SWIG safety!).
        journal.undo_to(mark)
        guard.restore_code_epoch(0, "undo:t1")
        for i in range(4):
            eid = 500 + i
            n_base = 10 + i * 3
            self.assertEqual(drv.exec_params[(eid, n_base)][0], 1001)
            self.assertEqual(drv.exec_params[(eid, n_base)][1], (128, 1, 1))
            self.assertEqual(drv.exec_params[(eid, n_base)][3], 0)
            self.assertEqual(drv.exec_params[(eid, n_base + 2)][0], 1001)
        # Replay witness binding cleared on undo; base graph still fresh at epoch 0.
        witness.counts[wk] = 0
        guard.check_before_replay(GG.graph_key("FULL", 4))
        self.assertEqual(witness.counts[wk], 0)

    def test_chained_tier_a_edits_and_step_by_step_undo(self):
        """Proves chained edits (v0 -> v1 -> v2) match live exec func even though template
        CUgraph still reports v0, and step-by-step undo restores v1 then v0."""
        jf, old_ck, entries, drv = self._build_kernel_and_graphs(buckets=(1, 2))
        eng, _ = _make_engine(entries)
        journal, witness = J.Journal("w"), W.Witness()
        guard = GG.GraphReplayGuard()
        for b in entries:
            guard.stamp(GG.graph_key("FULL", b))
        wk = "vllm.v1.attention.ops.fake_triton:my_k:1"
        rec = RC.Recapture(guard=guard)

        # Txn 1: 1001 -> 2000
        m1 = journal.mark("t1")
        snap1 = {
            "eligible": True, "reason": None, "witness_keys": [wk],
            "specs": [{
                "jf": jf, "witness_key": wk, "kernel_label": "vllm.v1.attention.ops.fake_triton.my_k",
                "device": 0, "cache_key": "spec0", "old_ck": old_ck, "old_func": 1001,
                "target": "sm100", "backend": None, "cache_shell": ({}, "sm100", None, None),
            }],
        }
        jf.device_caches.clear()
        guard.bump_code_epoch("commit:t1")
        r1 = rec.run(eng, "t1", journal, witness, jit_snap=snap1, driver=drv)
        self.assertTrue(r1["ok"] and r1["tier"] == "tier_a_pointer_swap")
        ck1 = jf.device_caches[0][0]["spec0"]
        self.assertEqual(int(ck1.function), 2000)

        # Txn 2: 2000 -> 2001 (template CUgraph STILL has 1001; _LIVE_NODE_STATE supplies 2000!)
        m2 = journal.mark("t2")
        snap2 = {
            "eligible": True, "reason": None, "witness_keys": [wk],
            "specs": [{
                "jf": jf, "witness_key": wk, "kernel_label": "vllm.v1.attention.ops.fake_triton.my_k",
                "device": 0, "cache_key": "spec0", "old_ck": ck1, "old_func": 2000,
                "target": "sm100", "backend": None, "cache_shell": ({}, "sm100", None, None),
            }],
        }
        jf.device_caches.clear()
        guard.bump_code_epoch("commit:t2")
        r2 = rec.run(eng, "t2", journal, witness, jit_snap=snap2, driver=drv)
        self.assertTrue(r2["ok"] and r2["tier"] == "tier_a_pointer_swap", r2)
        self.assertEqual(drv.exec_params[(500, 10)][0], 2001)

        # Undo t2 -> back to 2000
        journal.undo_to(m2)
        guard.restore_code_epoch(1, "undo:t2")
        self.assertEqual(drv.exec_params[(500, 10)][0], 2000)

        # Undo t1 -> back to 1001
        journal.undo_to(m1)
        guard.restore_code_epoch(0, "undo:t1")
        self.assertEqual(drv.exec_params[(500, 10)][0], 1001)

    def test_atomic_rollback_and_r1_fallback_on_driver_error_and_preconditions(self):
        """Proves partial Tier-A failures roll back already-patched nodes atomically before
        falling back to R1, and verifies every precondition failure triggers clean R1 fallback."""
        wk = "vllm.v1.attention.ops.fake_triton:my_k:1"

        # Case 1: Driver fails on 3rd set_exec_kernel_node_params call -> nodes 1 & 2 roll back to 1001, then R1 runs.
        jf, old_ck, entries, drv = self._build_kernel_and_graphs(buckets=(1, 2))
        drv.fail_on_call = 3
        eng, cap_calls = _make_engine(entries)
        journal, witness, guard = J.Journal("w"), W.Witness(), GG.GraphReplayGuard()
        journal.mark("t_fail")
        snap = {
            "eligible": True, "reason": None, "witness_keys": [wk],
            "specs": [{
                "jf": jf, "witness_key": wk, "kernel_label": "vllm.v1.attention.ops.fake_triton.my_k",
                "device": 0, "cache_key": "spec0", "old_ck": old_ck, "old_func": 1001,
                "target": "sm100", "backend": None, "cache_shell": ({}, "sm100", None, None),
            }],
        }
        res = RC.Recapture(guard=guard).run(eng, "t_fail", journal, witness, jit_snap=snap, driver=drv)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["tier"], "tier_c_r1_recapture")
        self.assertIn("driver_patch_error", res["tier_a_fallback_reason"])
        self.assertEqual(len(cap_calls), 2)
        # First 2 nodes on exec 500 were rolled back atomically to 1001!
        self.assertEqual(drv.exec_params[(500, 10)][0], 1001)
        self.assertEqual(drv.exec_params[(500, 12)][0], 1001)

        # Case 2: keep_graph=False (raw_cuda_graph == 0) -> R1 fallback
        jf2, old_ck2, entries2, drv2 = self._build_kernel_and_graphs(buckets=(1,), keep_graph=False)
        eng2, cap2 = _make_engine(entries2)
        j2 = J.Journal("w"); j2.mark("t2")
        snap2 = {**snap, "specs": [{**snap["specs"][0], "jf": jf2, "old_ck": old_ck2}]}
        res2 = RC.Recapture().run(eng2, "t2", j2, W.Witness(), jit_snap=snap2, driver=drv2)
        self.assertEqual(res2["tier"], "tier_c_r1_recapture")
        self.assertEqual(res2["tier_a_fallback_reason"], "keep_graph_unavailable")
        self.assertEqual(len(cap2), 1)

        # Case 3: ABI mismatch (global_scratch_size > 0) -> R1 fallback and device_caches stays clean
        jf3, old_ck3, entries3, drv3 = self._build_kernel_and_graphs(
            buckets=(1,), meta_override={"global_scratch_size": 1024},
        )
        jf3.device_caches.clear()
        eng3, cap3 = _make_engine(entries3)
        j3 = J.Journal("w"); j3.mark("t3")
        snap3 = {**snap, "specs": [{**snap["specs"][0], "jf": jf3, "old_ck": old_ck3}]}
        res3 = RC.Recapture().run(eng3, "t3", j3, W.Witness(), jit_snap=snap3, driver=drv3)
        self.assertEqual(res3["tier"], "tier_c_r1_recapture")
        self.assertEqual(res3["tier_a_fallback_reason"], "abi_mismatch:global_scratch_required")
        self.assertEqual(jf3.device_caches, {})
        self.assertEqual(len(cap3), 1)

        # Case 4: Non-JIT target -> R1 fallback
        jf4, old_ck4, entries4, drv4 = self._build_kernel_and_graphs(buckets=(1,))
        eng4, cap4 = _make_engine(entries4)
        j4 = J.Journal("w"); j4.mark("t4")
        res4 = RC.Recapture().run(
            eng4, "t4", j4, W.Witness(),
            jit_snap={"eligible": False, "reason": "non_jit_target_present:mod:fwd:1", "specs": []},
            driver=drv4,
        )
        self.assertEqual(res4["tier"], "tier_c_r1_recapture")
        self.assertEqual(res4["tier_a_fallback_reason"], "non_jit_target_present:mod:fwd:1")
        self.assertEqual(len(cap4), 1)

        # Case 5: Nested child graph -> atomic rollback + R1 fallback
        jf5, old_ck5, entries5, drv5 = self._build_kernel_and_graphs(buckets=(1,))
        drv5.graphs_nodes[100].append(_MockNode(999, "child_graph"))
        eng5, cap5 = _make_engine(entries5)
        j5 = J.Journal("w"); j5.mark("t5")
        snap5 = {**snap, "specs": [{**snap["specs"][0], "jf": jf5, "old_ck": old_ck5}]}
        res5 = RC.Recapture().run(eng5, "t5", j5, W.Witness(), jit_snap=snap5, driver=drv5)
        self.assertEqual(res5["tier"], "tier_c_r1_recapture")
        self.assertEqual(res5["tier_a_fallback_reason"], "nested_child_graph_unsupported")
        self.assertEqual(drv5.exec_params[(500, 10)][0], 1001)
        self.assertEqual(len(cap5), 1)

        # Case 6: Driver error during undo fails closed (reports undo_incomplete AND invalidates guard!)
        jf6, old_ck6, entries6, drv6 = self._build_kernel_and_graphs(buckets=(1,))
        eng6, _ = _make_engine(entries6)
        g6 = GG.GraphReplayGuard()
        g6.stamp(GG.graph_key("FULL", 1))
        j6 = J.Journal("w"); m6 = j6.mark("t6")
        snap6 = {**snap, "specs": [{**snap["specs"][0], "jf": jf6, "old_ck": old_ck6}]}
        res6 = RC.Recapture(guard=g6).run(eng6, "t6", j6, W.Witness(), jit_snap=snap6, driver=drv6)
        self.assertEqual(res6["tier"], "tier_a_pointer_swap")
        drv6.fail_on_call = drv6._attempts + 1
        undo_res = j6.undo_to(m6)
        self.assertFalse(undo_res["ok"], undo_res)
        self.assertTrue(
            any(f["check"] == "undo_incomplete" and "CudaDriverError" in str(f) for f in undo_res["findings"]),
            undo_res,
        )
        with self.assertRaises(GG.StaleGraphReplayError):
            g6.check_before_replay(GG.graph_key("FULL", 1))

    def test_p0_p1_hardening_invariants(self):
        """Proves all P0/P1 fail-closed invariants: destroyed-handle undo guard, prior-edit
        staleness fallback, multi-rank TP>1 gate, non-leaf conditional node 0-compile fallback,
        None metadata + backend.parse_options preservation, stacked multi-txn replay witness
        merging, and failed forward rollback fail-closed refusal."""
        import dataclasses
        wk1 = "vllm.v1.attention.ops.fake_triton:my_k:1"
        wk2 = "vllm.v1.attention.ops.fake_triton:my_k2:1"

        # 1. Destroyed CUgraphExec handle on undo (e.g., after release_old=True): never calls driver on dead handle, invalidates guard
        jf1, ck1, entries1, drv1 = self._build_kernel_and_graphs(buckets=(1,))
        eng1, _ = _make_engine(entries1)
        g1 = GG.GraphReplayGuard()
        g1.stamp(GG.graph_key("FULL", 1))
        j1 = J.Journal("w"); m1 = j1.mark("t_destroy")
        snap1 = {"eligible": True, "reason": None, "witness_keys": [wk1], "specs": [{
            "jf": jf1, "witness_key": wk1, "kernel_label": "mod.my_k", "device": 0,
            "cache_key": "s0", "old_ck": ck1, "old_func": 1001, "target": "sm100", "backend": None,
            "cache_shell": ({}, "sm100", None, None),
        }]}
        r1 = RC.Recapture(guard=g1).run(eng1, "t_destroy", j1, W.Witness(), jit_snap=snap1, driver=drv1)
        self.assertEqual(r1["tier"], "tier_a_pointer_swap")
        calls_before_undo = len(drv1.set_calls)
        entries1[1].cudagraph._e = 0  # simulate graph reset() / destruction
        u1 = j1.undo_to(m1)
        self.assertFalse(u1["ok"], u1)
        self.assertEqual(len(drv1.set_calls), calls_before_undo)  # driver NEVER called on destroyed handle!
        with self.assertRaises(GG.StaleGraphReplayError):
            g1.check_before_replay(GG.graph_key("FULL", 1))

        # 2. Prior un-recaptured edit staleness (pre_epoch=1, graph stamped at 0) -> R1 fallback
        jf2, ck2, entries2, drv2 = self._build_kernel_and_graphs(buckets=(1,))
        eng2, cap2 = _make_engine(entries2)
        g2 = GG.GraphReplayGuard()
        g2.stamp(GG.graph_key("FULL", 1))  # stamped at epoch 0
        g2.bump_code_epoch("commit:prior_unrecaptured")  # 0 -> 1
        g2.bump_code_epoch("commit:txn2")  # 1 -> 2
        j2 = J.Journal("w"); j2.mark("txn2")
        snap2 = {**snap1, "specs": [{**snap1["specs"][0], "jf": jf2, "old_ck": ck2}]}
        r2 = RC.Recapture(guard=g2).run(eng2, "txn2", j2, W.Witness(), jit_snap=snap2, pre_epoch=1, driver=drv2)
        self.assertEqual(r2["tier"], "tier_c_r1_recapture")
        self.assertEqual(r2["tier_a_fallback_reason"], "stale_graph_from_prior_edit")
        self.assertEqual(jf2.compile_calls, 0)
        self.assertEqual(len(cap2), 1)

        # 3. Multi-rank world_size=2 -> deterministic R1 fallback (prevents TP>1 NCCL deadlock)
        jf3, ck3, entries3, drv3 = self._build_kernel_and_graphs(buckets=(1,))
        eng3, cap3 = _make_engine(entries3)
        j3 = J.Journal("w"); j3.mark("t_tp2")
        snap3 = {**snap1, "specs": [{**snap1["specs"][0], "jf": jf3, "old_ck": ck3}]}
        r3 = RC.Recapture().run(eng3, "t_tp2", j3, W.Witness(), jit_snap=snap3, world_size=2, driver=drv3)
        self.assertEqual(r3["tier"], "tier_c_r1_recapture")
        self.assertEqual(r3["tier_a_fallback_reason"], "multi_rank_world_size_unsupported")
        self.assertEqual(jf3.compile_calls, 0)

        # 4. Non-leaf conditional node -> R1 fallback with 0 compiles & clean device_caches
        jf4, ck4, entries4, drv4 = self._build_kernel_and_graphs(buckets=(1,))
        jf4.device_caches.clear()
        drv4.graphs_nodes[100].append(_MockNode(998, "unsupported:CU_GRAPH_NODE_TYPE_CONDITIONAL"))
        eng4, cap4 = _make_engine(entries4)
        j4 = J.Journal("w"); j4.mark("t_cond")
        snap4 = {**snap1, "specs": [{**snap1["specs"][0], "jf": jf4, "old_ck": ck4}]}
        r4 = RC.Recapture().run(eng4, "t_cond", j4, W.Witness(), jit_snap=snap4, driver=drv4)
        self.assertEqual(r4["tier"], "tier_c_r1_recapture")
        self.assertIn("unsupported_graph_node", r4["tier_a_fallback_reason"])
        self.assertEqual(jf4.compile_calls, 0)
        self.assertEqual(jf4.device_caches, {})

        # 5. Nullable metadata (tmem_size=None, global_scratch_size=None) + backend.parse_options preservation
        @dataclasses.dataclass
        class _FakeBackendOpts:
            num_warps: int = 4
            default_dot_input_precision: str = "ieee"
            extern_libs: tuple = ()

        class _FakeBackend:
            def parse_options(self, d):
                return _FakeBackendOpts(**d)

        jf5, ck5, entries5, drv5 = self._build_kernel_and_graphs(buckets=(1,))
        ck5.metadata.tmem_size = None
        ck5.metadata.global_scratch_size = None
        ck5.metadata.default_dot_input_precision = "tf32"
        ck5.metadata.extern_libs = (("libdevice", "/path"),)
        captured_opts = {}
        orig_compile = jf5.compile

        def _spy_compile(src, target=None, options=None):
            captured_opts.update(options or {})
            return orig_compile(src, target=target, options=options)

        jf5.compile = _spy_compile
        eng5, _ = _make_engine(entries5)
        j5 = J.Journal("w"); j5.mark("t_opts")
        snap5 = {**snap1, "specs": [{**snap1["specs"][0], "jf": jf5, "old_ck": ck5, "backend": _FakeBackend()}]}
        r5 = RC.Recapture().run(eng5, "t_opts", j5, W.Witness(), jit_snap=snap5, driver=drv5)
        self.assertEqual(r5["tier"], "tier_a_pointer_swap", r5)
        self.assertEqual(captured_opts.get("default_dot_input_precision"), "tf32")
        self.assertEqual(captured_opts.get("extern_libs"), (("libdevice", "/path"),))

        # 6. Stacked Tier-A transactions on two distinct kernels in the same graph merge replay witnesses
        jf6a, ck6a, entries6, drv6 = self._build_kernel_and_graphs(buckets=(1,))
        jf6b, ck6b, _, _ = self._build_kernel_and_graphs(buckets=(1,))
        ck6b.function = 7777  # matches the 2nd kernel node (7777) in bucket 1
        eng6, _ = _make_engine(entries6)
        g6 = GG.GraphReplayGuard()
        g6.stamp(GG.graph_key("FULL", 1))
        j6, w6 = J.Journal("w"), W.Witness()
        w6.arm(wk1, j6); w6.arm(wk2, j6)
        m6a = j6.mark("ta")
        g6.bump_code_epoch("commit:ta")
        ra = RC.Recapture(guard=g6).run(eng6, "ta", j6, w6, jit_snap={
            "eligible": True, "reason": None, "witness_keys": [wk1], "specs": [{
                **snap1["specs"][0], "jf": jf6a, "old_ck": ck6a, "old_func": 1001, "witness_key": wk1, "kernel_label": "mod.k1",
            }]}, pre_epoch=0, driver=drv6)
        self.assertEqual(ra["tier"], "tier_a_pointer_swap", ra)
        m6b = j6.mark("tb")
        g6.bump_code_epoch("commit:tb")
        rb = RC.Recapture(guard=g6).run(eng6, "tb", j6, w6, jit_snap={
            "eligible": True, "reason": None, "witness_keys": [wk2], "specs": [{
                **snap1["specs"][0], "jf": jf6b, "old_ck": ck6b, "old_func": 7777, "witness_key": wk2, "kernel_label": "mod.k2",
            }]}, pre_epoch=1, driver=drv6)
        self.assertEqual(rb["tier"], "tier_a_pointer_swap", rb)
        g6.check_before_replay(GG.graph_key("FULL", 1))
        self.assertEqual(w6.counts.get(wk1), 1)
        self.assertEqual(w6.counts.get(wk2), 1)
        j6.undo_to(m6b)
        g6.restore_code_epoch(1, "undo:tb")
        g6.check_before_replay(GG.graph_key("FULL", 1))
        self.assertEqual(w6.counts.get(wk1), 2)
        self.assertEqual(w6.counts.get(wk2), 1)

        # 7. Failed forward rollback returns ok=False (does NOT silently fall back to R1) & invalidates guard
        jf7, ck7, entries7, drv7 = self._build_kernel_and_graphs(buckets=(1,))
        drv7_orig_set = drv7.set_exec_kernel_node_params
        calls7 = {"n": 0}

        def _fail_patch_and_rollback(*a, **kw):
            calls7["n"] += 1
            if calls7["n"] >= 2:  # 2nd patch fails, AND rollback (3rd call) also fails!
                raise GP.CudaDriverError("persistent driver failure")
            return drv7_orig_set(*a, **kw)

        drv7.set_exec_kernel_node_params = _fail_patch_and_rollback
        eng7, cap7 = _make_engine(entries7)
        g7 = GG.GraphReplayGuard()
        g7.stamp(GG.graph_key("FULL", 1))
        j7 = J.Journal("w"); j7.mark("t_rb_fail")
        snap7 = {**snap1, "specs": [{**snap1["specs"][0], "jf": jf7, "old_ck": ck7}]}
        r7 = RC.Recapture(guard=g7).run(eng7, "t_rb_fail", j7, W.Witness(), jit_snap=snap7, driver=drv7)
        self.assertFalse(r7["ok"], r7)
        self.assertEqual(cap7, [])
        with self.assertRaises(GG.StaleGraphReplayError):
            g7.check_before_replay(GG.graph_key("FULL", 1))
        # Subsequent txn at pre_epoch=0 MUST see pool_epoch==1 staleness and fall back to R1 (P1-3)!
        j7b = J.Journal("w"); j7b.mark("t_after_rb_fail")
        r7b = RC.Recapture(guard=g7).run(eng7, "t_after_rb_fail", j7b, W.Witness(), jit_snap=snap7, pre_epoch=0, driver=drv7)
        self.assertEqual(r7b["tier"], "tier_c_r1_recapture", r7b)
        self.assertEqual(r7b["tier_a_fallback_reason"], "stale_graph_from_prior_edit")

        # 8. Two @triton.jit targets in one txn sharing a captured caller credit BOTH witness keys (P1-2)
        jf8, ck8, entries8, drv8 = self._build_kernel_and_graphs(buckets=(1,))
        fake_st = types.SimpleNamespace(
            items={wk1: object(), wk2: object()}, pending_adds=[],
            jits=[({"affected": [jf8]}, wk1), ({"affected": [jf8]}, wk2)],
        )
        fake_lp = types.SimpleNamespace(_staged={"t_shared": ([fake_st], [], [])})
        snap8 = GP.snapshot_jit_specs(fake_lp, "t_shared")
        self.assertTrue(snap8["eligible"] and len(snap8["specs"]) == 1, snap8)
        self.assertEqual(snap8["specs"][0]["witness_keys"], [wk1, wk2])
        eng8, _ = _make_engine(entries8)
        g8, j8, w8 = GG.GraphReplayGuard(), J.Journal("w"), W.Witness()
        g8.stamp(GG.graph_key("FULL", 1))
        w8.arm(wk1, j8); w8.arm(wk2, j8); j8.mark("t_shared")
        r8 = RC.Recapture(guard=g8).run(eng8, "t_shared", j8, w8, jit_snap=snap8, pre_epoch=0, driver=drv8)
        self.assertEqual(r8["tier"], "tier_a_pointer_swap", r8)
        g8.check_before_replay(GG.graph_key("FULL", 1))
        self.assertEqual(w8.counts.get(wk1), 1)
        self.assertEqual(w8.counts.get(wk2), 1)

        # 9. Direct CudaBindingsDriver test (allowlist leaf / GRAPH / CONDITIONAL + SWIG proxy + tuple CUresult)
        class _FakeResult:
            CUDA_SUCCESS = 0
            CUDA_ERROR_INVALID_VALUE = 1
        class _FakeNodeType:
            CU_GRAPH_NODE_TYPE_KERNEL = 0
            CU_GRAPH_NODE_TYPE_MEMCPY = 1
            CU_GRAPH_NODE_TYPE_GRAPH = 5
            CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
        n_kern, n_mem, n_child, n_cond = _MockNode(1, "k"), _MockNode(2, "m"), _MockNode(3, "g"), _MockNode(4, "c")
        ntype_map = {n_kern: 0, n_mem: 1, n_child: 5, n_cond: 13}
        swig_p = _SwigKernelParams(1001, (128, 1, 1), (4, 1, 1), 64)
        drv_calls = []
        fake_drv_mod = types.SimpleNamespace(
            CUresult=_FakeResult, CUgraphNodeType=_FakeNodeType,
            CUgraph=int, CUgraphExec=int, CUfunction=int,
            cuGraphGetNodes=lambda g, n: (0, None, 4) if n == 0 else (0, [n_kern, n_mem, n_child, n_cond], 4),
            cuGraphNodeGetType=lambda nd: (0, ntype_map[nd]),
            cuGraphKernelNodeGetParams=lambda nd: (0, swig_p),
            cuGraphExecKernelNodeSetParams=lambda he, nd, p: (drv_calls.append((he, int(nd), int(p.func))), (0,))[1],
        )
        cbd = GP.CudaBindingsDriver(fake_drv_mod)
        kinds = [k for _, _, k in cbd.get_graph_nodes(100)]
        self.assertEqual(kinds, ["kernel", "leaf", "child_graph", "unsupported:13"])
        p_obj, orig_f, orig_b, orig_g, orig_s = cbd.get_kernel_node_params(n_kern)
        cbd.set_exec_kernel_node_params(500, n_kern, p_obj, 2000, (256, 1, 1), orig_g, 128)
        self.assertEqual(orig_f, 1001)  # plain int preserved despite SWIG proxy mutation!
        self.assertEqual(drv_calls, [(500, 1, 2000)])

        # 10. Cascaded undo from non-graph txn1 over graph-affecting txn2 restores code_epoch to 0 (P1-1)
        side_c = SD.Side("worker", scope={}, rank=0)
        side_c._txn_graph_affect.add("txn2")
        side_c._txn_pre_epoch["txn2"] = 0
        side_c._txn_jit_snap["txn2"] = {"eligible": True}
        side_c.graph_guard.bump_code_epoch("commit:txn2")
        self.assertEqual(side_c.graph_guard.code_epoch, 1)
        orig_lp_undo = side_c.lp.undo
        side_c.lp.undo = lambda tid: {"ok": True, "txn": tid, "undone": 1, "findings": [], "also_undone": ["txn2"]}
        try:
            ur = side_c.local("undo", "txn1")
            self.assertTrue(ur["ok"], ur)
            self.assertEqual(side_c.graph_guard.code_epoch, 0)
            self.assertNotIn("txn2", side_c._txn_jit_snap)
            self.assertNotIn("txn2", side_c._txn_pre_epoch)
        finally:
            side_c.lp.undo = orig_lp_undo

    def test_transparent_keep_graph_hook_defaults_true_and_instantiates_on_capture_end(self):
        """Proves enable_keep_graph makes default CUDAGraph() retain keep_graph=True and
        eagerly instantiate on capture_end(), while honoring explicit keep_graph kwargs."""
        class FakeCUDAGraph:
            def __new__(cls, keep_graph=False):
                inst = super().__new__(cls)
                inst._keep_graph = keep_graph
                inst._has_graph_exec = False
                inst._raw_g = 0
                inst._raw_e = 0
                inst.instantiate_calls = 0
                return inst

            def capture_end(self):
                self._raw_g = 42 if self._keep_graph else 0
                if not self._keep_graph:
                    self.instantiate()

            def instantiate(self):
                self.instantiate_calls += 1
                self._has_graph_exec = True
                self._raw_e = 99

            def raw_cuda_graph(self):
                return self._raw_g

            def raw_cuda_graph_exec(self):
                return self._raw_e

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(graphs=types.SimpleNamespace(CUDAGraph=FakeCUDAGraph)))
        self.assertTrue(GP.enable_keep_graph(fake_torch))
        try:
            g_default = FakeCUDAGraph()
            self.assertTrue(g_default._keep_graph)
            g_default.capture_end()
            self.assertEqual(g_default.raw_cuda_graph(), 42)
            self.assertEqual(g_default.raw_cuda_graph_exec(), 99)
            self.assertEqual(g_default.instantiate_calls, 1)

            g_explicit_false = FakeCUDAGraph(keep_graph=False)
            self.assertFalse(g_explicit_false._keep_graph)
            g_explicit_false.capture_end()
            self.assertEqual(g_explicit_false.raw_cuda_graph(), 0)
            self.assertEqual(g_explicit_false.instantiate_calls, 1)

            g_explicit_true = FakeCUDAGraph(keep_graph=True)
            self.assertTrue(g_explicit_true._keep_graph)
            g_explicit_true.capture_end()
            self.assertEqual(g_explicit_true.instantiate_calls, 0)
        finally:
            self.assertTrue(GP.disable_keep_graph(fake_torch))
        g_restored = FakeCUDAGraph()
        self.assertFalse(g_restored._keep_graph)

    def test_end_to_end_side_commit_tier_a_prove_and_undo(self):
        """End-to-end through real Side('worker') + swap.stage + swap.apply + jit.apply +
        recapture (Tier-A) + replay witness + witness_check + undo + at_base."""
        src_v0 = textwrap.dedent("""\
            def jit(fn):
                return fn

            @jit
            def add_kernel(x_ptr, out_ptr, n, BLOCK):
                return x_ptr + 1
        """)
        src_v1 = src_v0.replace("return x_ptr + 1", "return x_ptr + 100")

        with tempfile.TemporaryDirectory() as td:
            mod_name = "vllm.v1.attention.ops._test_tier_a_mod"
            mod_path = os.path.join(td, "_test_tier_a_mod.py")
            Path(mod_path).write_text(src_v0, encoding="utf-8")
            mod = types.ModuleType(mod_name)
            mod.__file__ = mod_path
            exec(compile(src_v0, mod_path, "exec", dont_inherit=True), mod.__dict__)
            raw_fn = mod.add_kernel
            jf = _FakeJIT(raw_fn, "def add_kernel(x_ptr, out_ptr, n, BLOCK):\n    return x_ptr + 1\n", start_line=4)
            old_src = _FakeASTSource(jf, jf.signature, {(3,): 256}, {(0,): []})
            old_ck = _FakeCompiledKernel(1001, old_src, _FakeMeta(num_warps=4, shared=0))
            jf.device_caches[0] = ({"s0": old_ck}, {}, "sm100", None, None)
            mod.add_kernel = jf
            sys.modules[mod_name] = mod
            try:
                graphs_nodes = {100: [_MockNode(10, "kernel", _SwigKernelParams(1001))]}
                drv = _MockDriver(graphs_nodes)
                drv.bind_exec(500, 100)
                entries = {1: _MockGraphEntry(_MockCUDAGraph(100, 500))}
                eng, cap_calls = _make_engine(entries)
                side = SD.Side("worker", scope={"r1_engine": eng, "cuda_driver": drv}, rank=0)
                side.graph_guard.stamp(GG.graph_key("FULL", 1))

                plan = {
                    "id": "txn-tier-a",
                    "modules": {mod_name: {"source": src_v1}},
                    "rebuild": ["R1"],
                }
                prep = side.dispatch("prepare", plan)
                self.assertTrue(prep["ok"], prep)
                val = side.dispatch("validate", "txn-tier-a")
                self.assertTrue(val["ok"], val)
                com = side.dispatch("commit", "txn-tier-a")
                self.assertTrue(com["ok"], com)

                rec = side.dispatch("recapture", {"txn": "txn-tier-a"})
                self.assertTrue(rec["ok"], rec)
                self.assertEqual(rec["tier"], "tier_a_pointer_swap")
                self.assertEqual(rec["patched_nodes"], 1)
                self.assertEqual(cap_calls, [])
                self.assertEqual(drv.exec_params[(500, 10)][0], 2000)

                # Before live replay, witness_check reports zero_witness (no false positive!).
                w_before = side.dispatch("witness_check", "txn-tier-a")
                self.assertTrue(any(f["check"] == "zero_witness" for f in w_before["findings"]), w_before)

                # Live guarded replay triggers witness crediting -> witness_check passes!
                side.graph_guard.check_before_replay(GG.graph_key("FULL", 1))
                w_after = side.dispatch("witness_check", "txn-tier-a")
                self.assertEqual(w_after["findings"], [], w_after)

                # Undo restores CUgraphExec to 1001, device_caches to old_ck, and at_base ok.
                und = side.dispatch("undo", "txn-tier-a")
                self.assertTrue(und["ok"], und)
                self.assertEqual(drv.exec_params[(500, 10)][0], 1001)
                self.assertIs(jf.device_caches[0][0]["s0"], old_ck)
                self.assertTrue(side.dispatch("at_base")["ok"])
            finally:
                sys.modules.pop(mod_name, None)


if __name__ == "__main__":
    unittest.main()
