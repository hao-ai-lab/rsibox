"""CPU unit tests for Tier-A surgical in-place CUDA-graph kernel-node patching."""
import types
import unittest

from adapter import (
    GraphReplayGuard,
    KernelSwapSpec,
    collect_triton_func_map,
    disable_keep_graph_hook,
    enable_keep_graph_hook,
    extract_cudagraph,
    graph_key,
    patch_graphs_in_place,
)
from adapter import recapture as RC
from kernel.journal import Journal
from kernel.witness import Witness


class _FakeCUresult:
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1


class _FakeCUgraphNodeType:
    CU_GRAPH_NODE_TYPE_KERNEL = 0
    CU_GRAPH_NODE_TYPE_MEMCPY = 1


class _FakeKernelParams:
    def __init__(self, func_int, grid=(4, 1, 1), block=(256, 1, 1), shared_mem=0):
        self.func = int(func_int)
        self.gridDimX, self.gridDimY, self.gridDimZ = grid
        self.blockDimX, self.blockDimY, self.blockDimZ = block
        self.sharedMemBytes = shared_mem


class _FakeNode:
    def __init__(self, node_type, params=None):
        self.node_type = node_type
        self.params = params


class _FakeCUDAGraph:
    def __init__(self, graph_id, exec_id, nodes):
        self._graph_id = graph_id
        self._exec_id = exec_id
        self.nodes = nodes

    def raw_cuda_graph(self):
        return self._graph_id

    def raw_cuda_graph_exec(self):
        return self._exec_id


class _FakeDriver:
    CUresult = _FakeCUresult
    CUgraphNodeType = _FakeCUgraphNodeType

    def __init__(self, graphs_by_id, *, fail_on_call=None):
        self._graphs = graphs_by_id
        self._fail_on_call = fail_on_call
        self.set_calls = []

    def CUgraph(self, val):
        return int(val)

    def CUgraphExec(self, val):
        return int(val)

    def CUfunction(self, val):
        return int(val)

    def cuGraphGetNodes(self, h_graph, count):
        nodes = self._graphs[int(h_graph)].nodes
        if count == 0:
            return (_FakeCUresult.CUDA_SUCCESS, None, len(nodes))
        return (_FakeCUresult.CUDA_SUCCESS, list(nodes), len(nodes))

    def cuGraphNodeGetType(self, node):
        return (_FakeCUresult.CUDA_SUCCESS, node.node_type)

    def cuGraphKernelNodeGetParams(self, node):
        return (_FakeCUresult.CUDA_SUCCESS, node.params)

    def cuGraphExecKernelNodeSetParams(self, h_exec, node, params):
        call_idx = len(self.set_calls) + 1
        self.set_calls.append((int(h_exec), node, int(params.func)))
        if self._fail_on_call is not None and call_idx == self._fail_on_call:
            return (_FakeCUresult.CUDA_ERROR_INVALID_VALUE,)
        node.params.func = int(params.func)
        return (_FakeCUresult.CUDA_SUCCESS,)


class _FakeWrapper:
    def __init__(self, mode, entries):
        self.runtime_mode = mode
        self.concrete_cudagraph_entries = entries


class _FakeEntry:
    def __init__(self, cg):
        self.cudagraph = cg


class GraphPatchTests(unittest.TestCase):
    def _make_setup(self, *, kept=True, fail_on_call=None):
        n1 = _FakeNode(_FakeCUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL, _FakeKernelParams(1001))
        n2 = _FakeNode(_FakeCUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY)
        n3 = _FakeNode(_FakeCUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL, _FakeKernelParams(1001))
        n4 = _FakeNode(_FakeCUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL, _FakeKernelParams(9999))
        cg1 = _FakeCUDAGraph(10 if kept else 0, 110 if kept else 0, [n1, n2])
        cg2 = _FakeCUDAGraph(20 if kept else 0, 120 if kept else 0, [n3, n4])
        drv = _FakeDriver({10: cg1, 20: cg2}, fail_on_call=fail_on_call)
        wrapper = _FakeWrapper("FULL", {1: _FakeEntry(cg1), 4: _FakeEntry(cg2)})
        captures = []
        syncs = [0]

        def capture(mode, desc):
            captures.append((mode, desc))
            wrapper.concrete_cudagraph_entries[desc] = _FakeEntry(cg1)

        engine = RC.Engine(
            wrappers=lambda: [wrapper],
            keys={"FULL": {1, 4}},
            capture=capture,
            sync=lambda: syncs.__setitem__(0, syncs[0] + 1),
        )
        return engine, drv, (n1, n3, n4), captures, syncs

    def test_surgical_patch_and_journaled_undo_zero_recaptures(self):
        engine, drv, (n1, n3, n4), captures, syncs = self._make_setup()
        guard = GraphReplayGuard(strict_unstamped=True)
        guard.stamp_many([graph_key("FULL", 1), graph_key("FULL", 4)])
        guard.bump_code_epoch("commit:t1")
        self.assertFalse(guard.is_fresh(graph_key("FULL", 1)))

        journal = Journal()
        m1 = journal.mark("t1")
        rc = RC.Recapture(guard=guard)
        res = rc.run(
            engine, "t1", journal, Witness(),
            kernel_swaps={1001: 2002}, tier="auto", drv=drv,
        )
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["tier"], "A_surgical_node_patch")
        self.assertEqual(res["graphs_patched"], 2)
        self.assertEqual(res["nodes_patched"], 2)
        self.assertEqual(res["recaptured"], 0)
        self.assertEqual(captures, [])
        self.assertEqual(n1.params.func, 2002)
        self.assertEqual(n3.params.func, 2002)
        self.assertEqual(n4.params.func, 9999)
        self.assertEqual(guard.pool_epoch, 0)
        self.assertTrue(guard.is_fresh(graph_key("FULL", 1)))
        self.assertTrue(guard.is_fresh(graph_key("FULL", 4)))

        # Undo restores exact old_func_int (1001) and pre-patch guard stamps
        journal.undo_to(m1)
        guard.restore_code_epoch(0, "undo:t1")
        self.assertEqual(n1.params.func, 1001)
        self.assertEqual(n3.params.func, 1001)
        self.assertEqual(n4.params.func, 9999)
        self.assertTrue(guard.is_fresh(graph_key("FULL", 1)))
        self.assertGreaterEqual(syncs[0], 4)

    def test_auto_tier_falls_back_to_r1_when_graph_not_kept(self):
        engine, drv, _nodes, captures, _syncs = self._make_setup(kept=False)
        journal = Journal()
        journal.mark("t1")
        rc = RC.Recapture()
        res = rc.run(
            engine, "t1", journal, Witness(),
            kernel_swaps={1001: 2002}, tier="auto", drv=drv,
        )
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["tier"], "B_selective_recapture")
        self.assertEqual(res["surgical_fallback_reason"], "graph_not_kept")
        self.assertEqual(len(captures), 2)

    def test_strict_surgical_refuses_when_graph_not_kept(self):
        engine, drv, _nodes, captures, _syncs = self._make_setup(kept=False)
        journal = Journal()
        journal.mark("t1")
        rc = RC.Recapture()
        res = rc.run(
            engine, "t1", journal, Witness(),
            kernel_swaps={1001: 2002}, tier="surgical", drv=drv,
        )
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "graph_not_kept")
        self.assertEqual(captures, [])

    def test_partial_driver_failure_rolls_back_patched_nodes(self):
        engine, drv, (n1, n3, _n4), _captures, _syncs = self._make_setup(fail_on_call=2)
        journal = Journal()
        journal.mark("t1")
        rc = RC.Recapture()
        res = rc.patch_nodes(engine, "t1", journal, {1001: 2002}, drv=drv)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "set_params_failed")
        self.assertEqual(res["rolled_back_nodes"], 1)
        self.assertEqual(n1.params.func, 1001)
        self.assertEqual(n3.params.func, 1001)

    def test_incompatible_launch_params_fails_closed_before_mutation(self):
        engine, drv, (n1, n3, _n4), _captures, _syncs = self._make_setup()
        spec = KernelSwapSpec(old_func=1001, new_func=2002, expected_shared_mem=4096)
        res = patch_graphs_in_place(
            graphs=[(("FULL", 1), extract_cudagraph(engine.wrappers()[0].concrete_cudagraph_entries[1]))],
            kernel_swaps=[spec],
            drv=drv,
        )
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "incompatible_launch_params")
        self.assertEqual(drv.set_calls, [])
        self.assertEqual(n1.params.func, 1001)
        self.assertEqual(n3.params.func, 1001)

    def test_zero_matching_nodes_fails_closed(self):
        engine, drv, _nodes, _captures, _syncs = self._make_setup()
        journal = Journal()
        journal.mark("t1")
        res = RC.Recapture().patch_nodes(engine, "t1", journal, {7777: 8888}, drv=drv)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "no_matching_nodes")

    def test_collect_triton_func_map_and_incompatibility_guard(self):
        ck_old = types.SimpleNamespace(function=101, metadata=types.SimpleNamespace(shared=0, num_warps=4, signature="a"))
        ck_new = types.SimpleNamespace(function=202, metadata=types.SimpleNamespace(shared=0, num_warps=4, signature="a"))
        ck_bad = types.SimpleNamespace(function=303, metadata=types.SimpleNamespace(shared=8192, num_warps=4, signature="a"))
        specs, incompat = collect_triton_func_map({0: {"k1": ck_old}}, {0: {"k1": ck_new}}, name="add_k")
        self.assertEqual(incompat, [])
        self.assertEqual(len(specs), 1)
        self.assertEqual((specs[0].old_func, specs[0].new_func, specs[0].expected_shared_mem), (101, 202, 0))

        specs2, incompat2 = collect_triton_func_map({0: {"k1": ck_old}}, {0: {"k1": ck_bad}}, name="add_k")
        self.assertEqual(specs2, [])
        self.assertEqual(len(incompat2), 1)
        self.assertIn("shared memory changed", incompat2[0])

    def test_enable_and_disable_keep_graph_hook(self):
        class _DummyCG:
            def __init__(self, keep_graph=False):
                self.keep_graph = keep_graph

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(CUDAGraph=_DummyCG))
        self.assertFalse(_DummyCG().keep_graph)
        self.assertTrue(enable_keep_graph_hook(fake_torch))
        self.assertTrue(_DummyCG().keep_graph)
        self.assertFalse(_DummyCG(keep_graph=False).keep_graph)
        self.assertTrue(disable_keep_graph_hook(fake_torch))
        self.assertFalse(_DummyCG().keep_graph)


if __name__ == "__main__":
    unittest.main()
