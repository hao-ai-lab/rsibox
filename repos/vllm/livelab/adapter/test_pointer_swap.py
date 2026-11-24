"""CPU golden tests for Tier-A surgical CUDA graph kernel-node pointer swapping.

Runs in CPU-only CI without `torch`, `triton`, or `cuda` installed. Exercises the real
`pointer_swap`, `graph_guard`, `recapture`, `side`, `worker_ext`, and `host` code paths
end to end against a realistic SWIG-view & `kernelParams`-lifetime CUDA driver double.
"""
import ast
import enum
import itertools
import os
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from kernel import jit as kernel_jit
from kernel import journal as kernel_journal
from kernel import locate as kernel_locate
from kernel import witness as kernel_witness

from . import graph_guard as guard_mod
from . import host as host_mod
from . import pointer_swap as swap_mod
from . import recapture as recapture_mod
from . import worker_ext as worker_ext_mod
from .side import Side
from .test_host import Executor

_MOD_COUNTER = itertools.count(1)


class _FakeCUresult(enum.IntEnum):
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_LAUNCH_FAILED = 719


class _FakeNodeType(enum.IntEnum):
    CU_GRAPH_NODE_TYPE_KERNEL = 0
    CU_GRAPH_NODE_TYPE_MEMCPY = 1
    CU_GRAPH_NODE_TYPE_GRAPH = 5
    CU_GRAPH_NODE_TYPE_CONDITIONAL = 13


class _SwigFuncView:
    """Models SWIG's mutable handle view into `CUDA_KERNEL_NODE_PARAMS.func`."""

    def __init__(self, address: int):
        self._address = int(address)

    def __int__(self) -> int:
        return self._address

    def __repr__(self) -> str:
        return f"CUfunction({self._address:#x})"


class _FakeKernelNodeParams:
    def __init__(
        self,
        owner_node: "_FakeNode",
        func_addr: int,
        block: tuple[int, int, int],
        grid: tuple[int, int, int],
        shared_mem: int,
        extra: int = 0,
    ):
        self.owner_node = owner_node
        self.kernel_params_valid = True
        self._func_view = _SwigFuncView(func_addr)
        self.blockDimX, self.blockDimY, self.blockDimZ = block
        self.gridDimX, self.gridDimY, self.gridDimZ = grid
        self.sharedMemBytes = int(shared_mem)
        self.extra = int(extra)

    @property
    def func(self) -> _SwigFuncView:
        return self._func_view

    @func.setter
    def func(self, new_value) -> None:
        self._func_view._address = int(new_value)


class _FakeNode:
    def __init__(
        self,
        node_type: _FakeNodeType,
        func_addr: int = 0,
        block: tuple[int, int, int] = (128, 1, 1),
        grid: tuple[int, int, int] = (4, 1, 1),
        shared_mem: int = 0,
        extra: int = 0,
    ):
        self.node_type = node_type
        self.issued_params: list[_FakeKernelNodeParams] = []
        self.template_state = {
            "func": int(func_addr),
            "block": tuple(block),
            "grid": tuple(grid),
            "shared": int(shared_mem),
            "extra": int(extra),
        }
        self.exec_state = dict(self.template_state)


class _FakeCudaDriver:
    """In-memory CUDA driver double modeling SWIG view mutation AND `kernelParams`
    invalidation when `cuGraphKernelNodeSetParams` modifies a node's template."""

    CUresult = _FakeCUresult
    CUgraphNodeType = _FakeNodeType

    def __init__(self):
        self.graphs: dict[int, list[_FakeNode]] = {}
        self.exec_to_graph: dict[int, int] = {}
        self.func_names: dict[int, bytes] = {}
        self.exec_set_calls = 0
        self.template_set_calls = 0
        self.fail_on_exec_call: int | None = None
        self.fail_on_template_call: int | None = None
        self.fail_exec_on_nodes: set[int] = set()

    def register_graph(self, graph_handle: int, exec_handle: int, nodes: list[_FakeNode]) -> None:
        self.graphs[int(graph_handle)] = list(nodes)
        self.exec_to_graph[int(exec_handle)] = int(graph_handle)

    @staticmethod
    def CUgraph(handle: int) -> int:
        return int(handle)

    @staticmethod
    def CUgraphExec(handle: int) -> int:
        return int(handle)

    @staticmethod
    def CUfunction(handle: int) -> _SwigFuncView:
        return _SwigFuncView(int(handle))

    def cuFuncGetName(self, func_handle: _SwigFuncView):
        raw_name = self.func_names.get(int(func_handle))
        if raw_name is None:
            return (_FakeCUresult.CUDA_ERROR_INVALID_VALUE, None)
        return (_FakeCUresult.CUDA_SUCCESS, raw_name)

    def cuGraphGetNodes(self, graph_handle: int, num_nodes: int = 0):
        nodes = self.graphs.get(int(graph_handle))
        if nodes is None:
            return (_FakeCUresult.CUDA_ERROR_INVALID_VALUE, None, 0)
        if num_nodes == 0:
            return (_FakeCUresult.CUDA_SUCCESS, None, len(nodes))
        return (_FakeCUresult.CUDA_SUCCESS, list(nodes[:num_nodes]), len(nodes))

    def cuGraphNodeGetType(self, node: _FakeNode):
        return (_FakeCUresult.CUDA_SUCCESS, node.node_type)

    def cuGraphKernelNodeGetParams(self, node: _FakeNode):
        state = node.template_state
        params = _FakeKernelNodeParams(
            node, state["func"], state["block"], state["grid"], state["shared"], state["extra"]
        )
        node.issued_params.append(params)
        return (_FakeCUresult.CUDA_SUCCESS, params)

    def cuGraphKernelNodeSetParams(self, node: _FakeNode, params: _FakeKernelNodeParams):
        self.template_set_calls += 1
        if not params.kernel_params_valid:
            return (_FakeCUresult.CUDA_ERROR_INVALID_VALUE,)
        if self.fail_on_template_call == self.template_set_calls:
            return (_FakeCUresult.CUDA_ERROR_INVALID_VALUE,)
        node.template_state = {
            "func": int(params.func),
            "block": (int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ)),
            "grid": (int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ)),
            "shared": int(params.sharedMemBytes),
            "extra": int(params.extra),
        }
        # CUDA invalidates node-owned kernelParams arrays once template params are mutated.
        for issued in node.issued_params:
            issued.kernel_params_valid = False
        return (_FakeCUresult.CUDA_SUCCESS,)

    def cuGraphExecKernelNodeSetParams(self, exec_handle: int, node: _FakeNode, params: _FakeKernelNodeParams):
        self.exec_set_calls += 1
        if not params.kernel_params_valid:
            return (_FakeCUresult.CUDA_ERROR_INVALID_VALUE,)
        if self.fail_on_exec_call == self.exec_set_calls or id(node) in self.fail_exec_on_nodes:
            return (_FakeCUresult.CUDA_ERROR_LAUNCH_FAILED,)
        graph_handle = self.exec_to_graph.get(int(exec_handle))
        if graph_handle is None or node not in self.graphs.get(graph_handle, ()):
            return (_FakeCUresult.CUDA_ERROR_INVALID_VALUE,)
        node.exec_state = {
            "func": int(params.func),
            "block": (int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ)),
            "grid": (int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ)),
            "shared": int(params.sharedMemBytes),
            "extra": int(params.extra),
        }
        return (_FakeCUresult.CUDA_SUCCESS,)


class _FakeCudaGraphObj:
    def __init__(self, graph_handle: int, exec_handle: int, driver: _FakeCudaDriver, nodes: list[_FakeNode]):
        self._graph_handle = graph_handle
        self._exec_handle = exec_handle
        self._driver = driver
        self.reset_called = 0
        if graph_handle != 0:
            driver.register_graph(graph_handle, exec_handle, nodes)

    def raw_cuda_graph(self) -> int:
        return self._graph_handle

    def raw_cuda_graph_exec(self) -> int:
        return self._exec_handle

    def reset(self) -> None:
        self.reset_called += 1

    def replay(self) -> list[int]:
        nodes = self._driver.graphs.get(self._graph_handle, ())
        return [node.exec_state["func"] for node in nodes if node.node_type == _FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL]


class _FakeCompiledKernel:
    def __init__(
        self,
        func_addr: int,
        signature: dict | None = None,
        constants: dict | None = None,
        attrs: dict | None = None,
        metadata_overrides: dict | None = None,
        n_max_threads: int = 1024,
    ):
        self.function = func_addr
        self.n_max_threads = n_max_threads
        self.init_handles_called = 0
        self.src = SimpleNamespace(
            signature=dict(signature or {"x_ptr": "*fp16", "n": "i32"}),
            constants=dict(constants or {"BLOCK_SIZE": 128}),
            attrs=dict(attrs or {}),
        )
        meta_fields = {
            "name": "sample_kernel",
            "num_warps": 4,
            "num_ctas": 1,
            "warp_size": 32,
            "shared": 1024,
            "global_scratch_size": 0,
            "global_scratch_align": 1,
            "profile_scratch_size": 0,
            "launch_cooperative_grid": False,
            "launch_pdl": False,
            "tensordesc_meta": [],
            "tmem_size": 0,
            "instrumentation_mode": "",
            "cluster_dims": (1, 1, 1),
            "maxnreg": None,
            "supported_fp8_dtypes": ["fp8e5", "fp8e4b15"],
            "target": "fake-sm100",
        }
        if metadata_overrides:
            meta_fields.update(metadata_overrides)
        self._meta_dict = meta_fields
        self.metadata = SimpleNamespace(**meta_fields, _asdict=lambda meta_copy=meta_fields: dict(meta_copy))

    def _init_handles(self) -> None:
        self.init_handles_called += 1


class _FakeJITFunction:
    """Duck-typed Triton JITFunction implementing `kernel.jit.NEED` for CPU tests."""

    def __init__(self, python_fn, source_text: str, start_line: int = 1):
        self.fn = python_fn
        self.__globals__ = python_fn.__globals__
        self._src = source_text
        self.raw_src = source_text.splitlines(True)
        self.starting_line_number = start_line
        self.def_file_line_number = start_line
        self.hash = "base_hash"
        self.used_global_vals = {}
        self.device_caches: dict = {}
        self.pre_run_hooks: list = []
        self.signature = {"x_ptr": "*fp16", "n": "i32"}
        self.compile_calls: list[dict] = []

    def _unsafe_update_src(self, new_src: str) -> None:
        self._src = new_src

    def parse(self):
        return ast.parse(textwrap.dedent(self._src))

    @staticmethod
    def ASTSource(jit_fn, signature, constexprs=None, attrs=None):
        return SimpleNamespace(fn=jit_fn, signature=signature, constexprs=constexprs, attrs=attrs)

    def compile(self, ast_source, target=None, options=None):
        self.compile_calls.append({"src": ast_source, "target": target, "options": options})
        hash(options["supported_fp8_dtypes"])
        next_addr = 0x9000 + len(self.compile_calls)
        return _FakeCompiledKernel(next_addr, signature=ast_source.signature, constants=ast_source.constexprs)


def _make_jit_snapshot(jit_fn: _FakeJITFunction, witness_key: str) -> dict:
    staged_item = SimpleNamespace(
        adds=[], pending_adds=[], removes=[], imports={}, jit_adds=[],
        swaps=[(jit_fn.fn, None, None, None)],
        jits=[({"jf": jit_fn, "affected": [jit_fn]}, witness_key)],
    )
    return swap_mod.snapshot_triton_commit([staged_item], [])


def _make_triton_module(case: unittest.TestCase, tag: str, base_src: str) -> tuple[str, types.ModuleType]:
    mod_name = f"ps_triton_{next(_MOD_COUNTER)}_{tag}"
    temp_dir = tempfile.mkdtemp()
    case.addCleanup(shutil.rmtree, temp_dir, True)
    file_path = os.path.join(temp_dir, f"{mod_name}.py")
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(base_src)
    module = types.ModuleType(mod_name)
    module.__file__ = file_path
    module.__dict__["FakeJIT"] = _FakeJITFunction
    exec(compile(base_src, file_path, "exec"), module.__dict__)
    sys.modules[mod_name] = module
    case.addCleanup(sys.modules.pop, mod_name, None)
    return mod_name, module


TRITON_MODULE_SRC = """\
class _FakeTritonNamespace:
    @staticmethod
    def jit(python_fn):
        import inspect
        lines, start = inspect.getsourcelines(python_fn)
        return FakeJIT(python_fn, "".join(lines), start)

triton = _FakeTritonNamespace()

@triton.jit
def rms_norm_kernel(x_ptr, n):
    return x_ptr + n + 1

def host_helper(value):
    return value + 1
"""


class PointerSwapGoldenTests(unittest.TestCase):

    def test_tier_a_multi_bucket_v1_v2_breakable_swig_witness_extend_and_undo(self):
        """Input: 8 batch buckets across V1, V2, Breakable, plus an untouched subgraph, sharing
        2 specializations, under strict `kernelParams`-invalidation + SWIG-view driver rules.
        Expected: exec updated before template (`kernelParams` valid), ONLY 2 compiles, 0 dummy
        forwards, honest two-ledger witness, untouched subgraph non-crediting, `Recapture.extend`
        captures excluded `left` bucket cleanly, and undo restores exact original params and
        releases/restores left entries cleanly."""
        driver = _FakeCudaDriver()

        def sample_kernel():
            return 1

        jit_fn = _FakeJITFunction(sample_kernel, "def sample_kernel():\n    return 1\n")
        spec_small = _FakeCompiledKernel(0x1001, constants={"BLOCK_SIZE": 64})
        spec_large = _FakeCompiledKernel(0x1002, constants={"BLOCK_SIZE": 256})
        jit_fn.device_caches[0] = ({("small",): spec_small, ("large",): spec_large}, {}, "target")

        witness_key = "fake.mod:sample_kernel:1"
        launch_key = kernel_witness.launch_key(witness_key, kernel_jit.label(jit_fn))
        snapshot = _make_jit_snapshot(jit_fn, witness_key)
        self.assertTrue(snapshot["eligible"])

        node_refs: dict[str, list[_FakeNode]] = {}
        next_handle = itertools.count(100)

        def make_graph(label: str, func_addrs: list[int], block_x: int = 128, shared: int = 1024):
            nodes = [
                _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, addr, (block_x, 1, 1), (8, 1, 1), shared)
                for addr in func_addrs
            ]
            nodes.append(_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_MEMCPY))
            node_refs[label] = nodes
            graph_id = next(next_handle)
            return _FakeCudaGraphObj(graph_id, graph_id + 1000, driver, nodes)

        v1_wrapper = SimpleNamespace(
            runtime_mode="FULL",
            concrete_cudagraph_entries={
                1: SimpleNamespace(cudagraph=make_graph("v1_1", [0x1001, 0x5555])),
                2: SimpleNamespace(cudagraph=make_graph("v1_2", [0x1001])),
                4: SimpleNamespace(cudagraph=make_graph("v1_4", [0x1001, 0x1002])),
                8: SimpleNamespace(cudagraph=make_graph("v1_8", [0x1002])),
                128: SimpleNamespace(cudagraph=make_graph("untouched_128", [0x7777])),
            },
        )
        orig_v2_32 = make_graph("v2_32", [0x1002])
        v2_wrapper = SimpleNamespace(
            runtime_mode="FULL",
            entries={16: make_graph("v2_16", [0x1002]), 32: orig_v2_32},
        )
        seg_a, seg_b = make_graph("br_64_a", [0x1001]), make_graph("br_64_b", [0x1002])
        breakable_wrapper = SimpleNamespace(
            runtime_mode="PIECEWISE",
            entries={64: SimpleNamespace(capture=SimpleNamespace(segments=[seg_a.replay, seg_b.replay]))},
        )

        keys = {"FULL": {1, 2, 4, 8, 16, 32, 128}, "PIECEWISE": {64}}
        capture_calls: list = []
        sync_calls = [0]

        def handle_capture(mode, desc):
            capture_calls.append((mode, desc))
            v2_wrapper.entries[desc] = "extended_v2_graph"

        engine = recapture_mod.Engine(
            lambda: [v1_wrapper, v2_wrapper, breakable_wrapper],
            keys,
            handle_capture,
            sync=lambda: sync_calls.__setitem__(0, sync_calls[0] + 1),
        )
        guard = guard_mod.GraphReplayGuard()
        for mode, desc_set in keys.items():
            for desc in desc_set:
                guard.stamp(guard_mod.graph_key(mode, desc))

        journal = kernel_journal.Journal("rank0")
        witness = kernel_witness.Witness()
        recapture = recapture_mod.Recapture(guard=guard)

        mark = journal.mark("txn-1")
        jit_fn.device_caches.clear()
        witness.arm(witness_key, journal)
        witness.arm_launch(witness_key, kernel_jit.label(jit_fn), journal)
        guard.bump_code_epoch("commit:txn-1")

        compile_log: list[int] = []

        def fake_compiler(record):
            compile_log.append(record["old_func_int"])
            new_addr = 0x2001 if record["old_func_int"] == 0x1001 else 0x2002
            return _FakeCompiledKernel(
                new_addr,
                constants=record["old_kernel"].src.constants,
                metadata_overrides={"num_warps": 8, "shared": 2048},
            )

        reply = recapture.run(
            engine, "txn-1", journal, witness,
            sizes=[1, 2, 4, 8, 16, 64, 128],
            jit_snapshot=snapshot,
            pointer_swap_driver=driver,
            pointer_swap_compiler=fake_compiler,
        )
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(reply["tier"], "A")
        self.assertEqual(reply["compiled_specializations"], 2)
        self.assertEqual(len(compile_log), 2)
        self.assertEqual(capture_calls, [])
        self.assertEqual(reply["patched_nodes"], 8)
        self.assertEqual(reply["patched_graphs"], 7)
        self.assertEqual(reply["eager"], {"FULL": [32]})
        self.assertNotIn(32, keys["FULL"])
        # Excluded `left` entry 32 was popped from v2_wrapper so Recapture.extend captures it fresh!
        self.assertNotIn(32, v2_wrapper.entries)

        # Extend captures bucket 32 cleanly after Tier-A partial patch!
        extend_reply = recapture.extend(engine, "txn-1", journal, witness, sizes=[32])
        self.assertTrue(extend_reply["ok"], extend_reply)
        self.assertEqual(capture_calls, [("FULL", 32)])
        self.assertEqual(v2_wrapper.entries[32], "extended_v2_graph")
        self.assertIn(32, keys["FULL"])

        # Honest witness checks before & after replay.
        self.assertGreater(witness.capture_counts[witness_key], 0)
        self.assertEqual(witness.counts[witness_key], 0)
        guard.check_before_replay(guard_mod.graph_key("FULL", 128))
        self.assertEqual(witness.counts[witness_key], 0)
        guard.check_before_replay(guard_mod.graph_key("FULL", 4))
        self.assertGreater(witness.counts[witness_key], 0)
        self.assertGreater(witness.counts[launch_key], 0)

        # Undo restores base pointers, restores orig_v2_32 into v2_wrapper, and is idempotent!
        undo_res = journal.undo_to(mark)
        self.assertTrue(undo_res["ok"], undo_res)
        guard.restore_code_epoch(0, "undo:txn-1")
        self.assertIs(v2_wrapper.entries[32], orig_v2_32)
        self.assertIn(32, keys["FULL"])
        for label, expected_func in (("v1_1", 0x1001), ("v1_8", 0x1002), ("br_64_a", 0x1001), ("br_64_b", 0x1002)):
            node = node_refs[label][0]
            self.assertEqual(node.template_state["func"], expected_func)
            self.assertEqual(node.exec_state["func"], expected_func)
            self.assertEqual(node.exec_state["block"], (128, 1, 1))
            self.assertEqual(node.exec_state["shared"], 1024)

        # Pinning release on _clear_entries cleans up _PINNED_KERNELS.
        self.assertGreater(recapture_mod._clear_entries(v1_wrapper.concrete_cudagraph_entries), 0)

    def test_stacked_tier_a_and_tier_c_witness_preservation_and_undo_failure_fail_closed(self):
        """Input: (1) Stacked `Tier A (txn1) -> Tier A (txn2)` credits BOTH on replay, and
        `undo(txn2)` preserves `txn1`'s replay witness; (2) Stacked `Tier A (txn1) -> Tier C (txn2)`
        rolled back via `undo(txn2)` preserves `txn1`'s replay witness; (3) Mid-undo driver failure
        still restores remaining nodes, invalidates guard fail-closed, and raises aggregated error."""
        driver = _FakeCudaDriver()

        def stacked_kernel():
            return 0

        jit_fn = _FakeJITFunction(stacked_kernel, "def stacked_kernel():\n    return 0\n")
        jit_fn.device_caches[0] = ({("spec",): _FakeCompiledKernel(0x3000)}, {}, "target")

        node_first = _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x3000, (128, 1, 1), (4, 1, 1), 512)
        node_second = _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x3000, (128, 1, 1), (4, 1, 1), 512)
        graph_obj = _FakeCudaGraphObj(200, 1200, driver, [node_first, node_second])
        wrapper = SimpleNamespace(runtime_mode="FULL", entries={4: graph_obj})
        engine = recapture_mod.Engine(
            lambda: [wrapper],
            {"FULL": {4}},
            lambda mode, desc: wrapper.entries.__setitem__(desc, "tier_c_graph"),
        )
        guard = guard_mod.GraphReplayGuard()
        guard.stamp(guard_mod.graph_key("FULL", 4))
        journal = kernel_journal.Journal("rank0")
        witness = kernel_witness.Witness()
        recapture = recapture_mod.Recapture(guard=guard)

        # Tier A txn1: 0x3000 -> 0x3001
        snap1 = _make_jit_snapshot(jit_fn, "mod:stacked_kernel:1")
        mark1 = journal.mark("txn-1")
        jit_fn.device_caches.clear()
        witness.arm("mod:stacked_kernel:1", journal)
        guard.bump_code_epoch("commit:txn-1")
        reply1 = recapture.run(
            engine, "txn-1", journal, witness,
            jit_snapshot=snap1, pointer_swap_driver=driver,
            pointer_swap_compiler=lambda _record: _FakeCompiledKernel(0x3001, metadata_overrides={"shared": 1024}),
        )
        self.assertTrue(reply1["ok"] and reply1["tier"] == "A")

        # Tier A txn2 stacked on top: 0x3001 -> 0x3002
        snap2 = _make_jit_snapshot(jit_fn, "mod:stacked_kernel:2")
        mark2 = journal.mark("txn-2")
        jit_fn.device_caches.clear()
        witness.arm("mod:stacked_kernel:2", journal)
        guard.bump_code_epoch("commit:txn-2")
        reply2 = recapture.run(
            engine, "txn-2", journal, witness,
            jit_snapshot=snap2, pointer_swap_driver=driver,
            pointer_swap_compiler=lambda _record: _FakeCompiledKernel(0x3002, metadata_overrides={"shared": 2048}),
        )
        self.assertTrue(reply2["ok"] and reply2["tier"] == "A")

        # Replaying at txn2 credits BOTH txn1 and txn2 witnesses!
        guard.check_before_replay(guard_mod.graph_key("FULL", 4))
        self.assertGreater(witness.counts["mod:stacked_kernel:1"], 0)
        self.assertGreater(witness.counts["mod:stacked_kernel:2"], 0)

        # Undo txn2 restores 0x3001 AND preserves txn1's replay witness!
        self.assertTrue(journal.undo_to(mark2)["ok"])
        guard.restore_code_epoch(1, "undo:txn-2")
        self.assertEqual(node_first.exec_state["func"], 0x3001)
        prev_txn1_count = witness.counts["mod:stacked_kernel:1"]
        guard.check_before_replay(guard_mod.graph_key("FULL", 4))
        self.assertGreater(witness.counts["mod:stacked_kernel:1"], prev_txn1_count)

        # Tier C txn3 stacked on top of Tier A txn1, then undone -> preserves txn1's replay witness!
        mark3 = journal.mark("txn-3")
        guard.bump_code_epoch("commit:txn-3")
        reply3 = recapture.run(engine, "txn-3", journal, witness, jit_snapshot=None)
        self.assertTrue(reply3["ok"] and reply3["tier"] == "C")
        self.assertTrue(journal.undo_to(mark3)["ok"])
        guard.restore_code_epoch(1, "undo:txn-3")
        self.assertIs(wrapper.entries[4], graph_obj)
        prev_txn1_count = witness.counts["mod:stacked_kernel:1"]
        guard.check_before_replay(guard_mod.graph_key("FULL", 4))
        self.assertGreater(witness.counts["mod:stacked_kernel:1"], prev_txn1_count)

        # Mid-undo driver failure on node_second still restores node_first, invalidates guard, and raises!
        driver.fail_exec_on_nodes.add(id(node_second))
        undo_fail = journal.undo_to(mark1)
        self.assertFalse(undo_fail["ok"])
        self.assertEqual(node_first.exec_state["func"], 0x3000)
        self.assertEqual(node_first.template_state["func"], 0x3000)
        self.assertEqual(node_second.exec_state["func"], 0x3001)
        guard.restore_code_epoch(0, "undo:txn-1")
        with self.assertRaises(guard_mod.StaleGraphReplayError):
            guard.check_before_replay(guard_mod.graph_key("FULL", 4))

    def test_fail_closed_fallbacks_abi_unmatched_node_conditional_lora_and_mid_patch(self):
        """Input: every fail-closed precondition violation (keep_graph=False, ABI mismatches,
        missing src/meta/shared, shared_mem carve tier, unmatched affected node via cuFuncGetName,
        CU_GRAPH_NODE_TYPE_CONDITIONAL, non-1D block dim, params.extra != 0, mid-patch exec/template
        error, LoRA descriptor isolation). Expected: Tier-A fails closed cleanly with exact
        fallback reasons and zero stale cache leaks or descriptor witness aliasing."""

        def dummy_fn():
            return 1

        base_kernel = _FakeCompiledKernel(0x4001)
        for overrides, expected_substr in (
            ({"global_scratch_size": 64}, "global_scratch_size_changed"),
            ({"num_ctas": 2}, "multi_cta_unsupported"),
            ({"tensordesc_meta": ["tma_0"]}, "tensordesc_meta_changed"),
            ({"launch_pdl": True}, "launch_pdl_changed"),
            ({"launch_cooperative_grid": True}, "launch_cooperative_grid_changed"),
            ({"tmem_size": 128}, "tmem_size_changed"),
            ({"profile_scratch_size": 16}, "profile_scratch_nonzero"),
            ({"instrumentation_mode": "consan"}, "instrumentation_mode_changed"),
            ({"cluster_dims": (2, 1, 1)}, "cluster_dims_changed"),
            ({"maxnreg": 128}, "maxnreg_changed"),
            ({"shared": 240 * 1024}, "shared_mem_tier_changed"),
        ):
            ok, reason = swap_mod.check_abi_compatible(base_kernel, _FakeCompiledKernel(0x4002, metadata_overrides=overrides))
            self.assertFalse(ok, f"expected ABI rejection for {overrides}")
            self.assertIn(expected_substr, reason)

        # Missing src / metadata / shared fail closed.
        self.assertEqual(swap_mod.check_abi_compatible(base_kernel, SimpleNamespace(function=0x4002, src=None, metadata=base_kernel.metadata)), (False, "missing_src"))
        self.assertEqual(swap_mod.check_abi_compatible(base_kernel, SimpleNamespace(function=0x4002, src=base_kernel.src, metadata=None)), (False, "missing_metadata"))
        no_shared = _FakeCompiledKernel(0x4002)
        no_shared.metadata.shared = None
        self.assertEqual(swap_mod.check_abi_compatible(base_kernel, no_shared), (False, "missing_shared_metadata"))

        def run_fallback_case(nodes_factory, compiler_fn, driver_setup=None, graph_handle=300, exec_handle=1300):
            driver = _FakeCudaDriver()
            jit_local = _FakeJITFunction(dummy_fn, "def dummy_fn():\n    return 1\n")
            jit_local.device_caches[0] = ({("spec",): _FakeCompiledKernel(0x4001)}, {}, "target")
            snap = _make_jit_snapshot(jit_local, "mod:dummy_fn:1")
            nodes = nodes_factory()
            graph_obj = _FakeCudaGraphObj(graph_handle, exec_handle, driver, nodes)
            wrapper = SimpleNamespace(runtime_mode="FULL", entries={4: graph_obj})
            captured_by_r1: list = []
            engine = recapture_mod.Engine(
                lambda: [wrapper],
                {"FULL": {4}},
                lambda mode, desc: (captured_by_r1.append((mode, desc)), wrapper.entries.__setitem__(desc, "r1_graph")),
            )
            if driver_setup is not None:
                driver_setup(driver)
            guard = guard_mod.GraphReplayGuard()
            journal = kernel_journal.Journal("rank0")
            witness = kernel_witness.Witness()
            recapture = recapture_mod.Recapture(guard=guard)
            journal.mark("txn-fb")
            jit_local.device_caches.clear()
            guard.bump_code_epoch("commit:txn-fb")
            reply = recapture.run(
                engine, "txn-fb", journal, witness,
                jit_snapshot=snap, pointer_swap_driver=driver, pointer_swap_compiler=compiler_fn,
            )
            return reply, nodes, jit_local, captured_by_r1

        # keep_graph=False -> R1 fallback.
        reply, _, _, r1_caps = run_fallback_case(
            lambda: [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001)],
            lambda _record: _FakeCompiledKernel(0x4002),
            graph_handle=0,
        )
        self.assertEqual((reply["tier"], reply["fallback_reason"], r1_caps), ("C", "keep_graph_unavailable", [("FULL", 4)]))

        # CU_GRAPH_NODE_TYPE_GRAPH and CU_GRAPH_NODE_TYPE_CONDITIONAL fail closed.
        reply, _, _, _ = run_fallback_case(
            lambda: [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001), _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_GRAPH)],
            lambda _record: _FakeCompiledKernel(0x4002),
        )
        self.assertEqual((reply["tier"], reply["fallback_reason"]), ("C", "unsupported_child_graph_node"))

        reply, _, _, _ = run_fallback_case(
            lambda: [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001), _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL)],
            lambda _record: _FakeCompiledKernel(0x4002),
        )
        self.assertEqual(reply["tier"], "C")
        self.assertIn("unsupported_node_type", reply["fallback_reason"])

        # Unmatched affected kernel node detected by cuFuncGetName fails closed!
        reply, _, _, _ = run_fallback_case(
            lambda: [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001), _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x8888)],
            lambda _record: _FakeCompiledKernel(0x4002),
            driver_setup=lambda driver_obj: driver_obj.func_names.__setitem__(0x8888, b"dummy_fn"),
        )
        self.assertEqual((reply["tier"], reply["fallback_reason"]), ("C", "unmatched_affected_node"))

        # Non-1D block dim and params.extra != 0 fail closed.
        reply, _, _, _ = run_fallback_case(
            lambda: [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001, block=(128, 2, 1))],
            lambda _record: _FakeCompiledKernel(0x4002),
        )
        self.assertIn("unsupported_non_1d_block_dim", reply["fallback_reason"])

        reply, _, _, _ = run_fallback_case(
            lambda: [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001, extra=0xBEEF)],
            lambda _record: _FakeCompiledKernel(0x4002),
        )
        self.assertEqual(reply["fallback_reason"], "unsupported_kernel_params_extra")

        # Mid-patch exec error on node #2 reverts node #1 cleanly on BOTH exec and template!
        reply, nodes_mid, jit_mid, _ = run_fallback_case(
            lambda: [
                _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001, (128, 1, 1), (4, 1, 1), 512),
                _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001, (128, 1, 1), (4, 1, 1), 512),
            ],
            lambda _record: _FakeCompiledKernel(0x4002, metadata_overrides={"num_warps": 8, "shared": 2048}),
            driver_setup=lambda driver_obj: setattr(driver_obj, "fail_on_exec_call", 2),
        )
        self.assertEqual(reply["tier"], "C")
        self.assertIn("driver_patch_failed", reply["fallback_reason"])
        for node in nodes_mid:
            self.assertEqual((node.template_state["func"], node.exec_state["func"], node.exec_state["shared"]), (0x4001, 0x4001, 512))
        self.assertEqual(kernel_jit.compiled([jit_mid])[kernel_jit.label(jit_mid)], 0)

        # Mid-patch template error on node #1 reverts exec cleanly!
        reply, nodes_tmpl, _, _ = run_fallback_case(
            lambda: [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001, (128, 1, 1), (4, 1, 1), 512)],
            lambda _record: _FakeCompiledKernel(0x4002, metadata_overrides={"num_warps": 8, "shared": 2048}),
            driver_setup=lambda driver_obj: setattr(driver_obj, "fail_on_template_call", 1),
        )
        self.assertEqual(reply["tier"], "C")
        self.assertEqual((nodes_tmpl[0].template_state["func"], nodes_tmpl[0].exec_state["func"]), (0x4001, 0x4001))

        # LoRA vs non-LoRA descriptors sharing num_tokens=4 do NOT alias witnesses!
        class _Desc:
            def __init__(self, num_tokens: int, has_lora: bool):
                self.num_tokens, self.has_lora = num_tokens, has_lora
            def __hash__(self): return hash((self.num_tokens, self.has_lora))
            def __eq__(self, other): return isinstance(other, _Desc) and (self.num_tokens, self.has_lora) == (other.num_tokens, other.has_lora)

        desc_base, desc_lora = _Desc(4, False), _Desc(4, True)
        driver_lora = _FakeCudaDriver()
        jit_lora = _FakeJITFunction(dummy_fn, "def dummy_fn():\n    return 1\n")
        jit_lora.device_caches[0] = ({("spec",): _FakeCompiledKernel(0x4001)}, {}, "target")
        snap_lora = _make_jit_snapshot(jit_lora, "mod:dummy_fn:1")
        graph_base = _FakeCudaGraphObj(501, 1501, driver_lora, [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x4001)])
        graph_lora = _FakeCudaGraphObj(502, 1502, driver_lora, [_FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, 0x7777)])
        wrapper_lora = SimpleNamespace(runtime_mode="FULL", entries={desc_base: graph_base, desc_lora: graph_lora})
        engine_lora = recapture_mod.Engine(lambda: [wrapper_lora], {"FULL": {desc_base, desc_lora}}, lambda *_args: None)
        guard_lora = guard_mod.GraphReplayGuard()
        journal_lora, witness_lora = kernel_journal.Journal("rank0"), kernel_witness.Witness()
        journal_lora.mark("txn-lora")
        jit_lora.device_caches.clear()
        witness_lora.arm("mod:dummy_fn:1", journal_lora)
        guard_lora.bump_code_epoch("commit:txn-lora")
        recapture_lora = recapture_mod.Recapture(guard=guard_lora)
        reply_lora = recapture_lora.run(
            engine_lora, "txn-lora", journal_lora, witness_lora,
            jit_snapshot=snap_lora, pointer_swap_driver=driver_lora,
            pointer_swap_compiler=lambda _record: _FakeCompiledKernel(0x4002),
        )
        self.assertTrue(reply_lora["ok"] and reply_lora["tier"] == "A")
        # Replaying untouched LoRA bucket (same num_tokens=4!) MUST NOT credit base bucket witness!
        guard_lora.check_before_replay(guard_mod.graph_key("FULL", desc_lora))
        self.assertEqual(witness_lora.counts["mod:dummy_fn:1"], 0)
        guard_lora.check_before_replay(guard_mod.graph_key("FULL", desc_base))
        self.assertGreater(witness_lora.counts["mod:dummy_fn:1"], 0)

    def test_keep_graph_hook_lifecycle_v1_eager_mode_gate_and_recompile(self):
        """Input: fake `torch.cuda.CUDAGraph` (including subclass without own `__new__`),
        `worker_ext._hook_v1_callable` with `cudagraph_runtime_mode=NONE`, and
        `recompile_specialization` with JSON-loaded list options."""

        class _FakeTorchCUDAGraph:
            def __new__(cls, keep_graph: bool = False):
                instance = super().__new__(cls)
                instance._keep_graph = keep_graph
                instance._has_graph_exec = False
                instance._raw_graph = 0
                instance._raw_exec = 0
                instance.instantiate_calls = 0
                return instance

            def raw_cuda_graph(self) -> int:
                return self._raw_graph

            def raw_cuda_graph_exec(self) -> int:
                return self._raw_exec

            def instantiate(self) -> None:
                self.instantiate_calls += 1
                self._has_graph_exec = True
                self._raw_exec = 0xE001

            def capture_end(self) -> None:
                self._raw_graph = 0xA001 if self._keep_graph else 0
                if not self._keep_graph:
                    self.instantiate()

        orig_new, orig_capture_end = _FakeTorchCUDAGraph.__new__, _FakeTorchCUDAGraph.capture_end
        self.addCleanup(swap_mod.disable_keep_graph)

        self.assertTrue(swap_mod.enable_keep_graph(_FakeTorchCUDAGraph))
        self.assertTrue(swap_mod.enable_keep_graph(_FakeTorchCUDAGraph))
        auto_graph = _FakeTorchCUDAGraph()
        self.assertTrue(auto_graph._keep_graph)
        auto_graph.capture_end()
        self.assertEqual((auto_graph.raw_cuda_graph(), auto_graph.raw_cuda_graph_exec(), auto_graph.instantiate_calls), (0xA001, 0xE001, 1))
        self.assertFalse(_FakeTorchCUDAGraph(keep_graph=False)._keep_graph)
        self.assertFalse(_FakeTorchCUDAGraph(False)._keep_graph)
        explicit_true = _FakeTorchCUDAGraph(keep_graph=True)
        explicit_true.capture_end()
        self.assertEqual(explicit_true.instantiate_calls, 0)
        self.assertTrue(swap_mod.disable_keep_graph())
        self.assertIs(_FakeTorchCUDAGraph.__new__, orig_new)
        self.assertIs(_FakeTorchCUDAGraph.capture_end, orig_capture_end)

        # V1 callable hook skips replay check when forward_context.cudagraph_runtime_mode is NONE!
        class _V1Wrapper:
            def __init__(self):
                self.runtime_mode = "FULL"
                self.concrete_cudagraph_entries = {4: "graph_4"}
                self.runner = SimpleNamespace(capturing=False)
            def __call__(self, desc):
                return f"ran:{desc}"

        wrapper_v1 = _V1Wrapper()
        guard_v1 = guard_mod.GraphReplayGuard()
        guard_v1.stamp(guard_mod.graph_key("FULL", 4))
        worker_ext_mod._hook_v1_callable(wrapper_v1, guard_v1)
        guard_v1.bump_code_epoch("stale")
        fake_fc_none = SimpleNamespace(get_forward_context=lambda: SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name="NONE"), batch_descriptor=4))
        with mock.patch.dict(sys.modules, {"vllm.forward_context": fake_fc_none}):
            self.assertEqual(wrapper_v1(4), "ran:4")
        fake_fc_full = SimpleNamespace(get_forward_context=lambda: SimpleNamespace(cudagraph_runtime_mode="FULL", batch_descriptor=4))
        with mock.patch.dict(sys.modules, {"vllm.forward_context": fake_fc_full}):
            with self.assertRaises(guard_mod.StaleGraphReplayError):
                wrapper_v1(4)

        # recompile_specialization freezes JSON list options to tuples.
        def norm_fn():
            return 1

        jit_fn = _FakeJITFunction(norm_fn, "def norm_fn():\n    return 1\n")
        record = {"jit_fn": jit_fn, "device_id": 0, "cache_key": ("spec",), "old_kernel": _FakeCompiledKernel(0x5001), "cache_tuple": ("target",)}
        recompiled = swap_mod.recompile_specialization(record)
        self.assertEqual(recompiled.init_handles_called, 1)
        self.assertIsInstance(jit_fn.compile_calls[0]["options"]["supported_fp8_dtypes"], tuple)

    def test_host_side_coordinator_e2e_triton_tier_a_and_python_r1_fallback(self):
        """Input: real `Host`, `RecaptureCoordinator`, `WorkerGroup`, and 2 worker `Side`
        ranks with a Triton JIT module and guarded CUDA graphs. Expected: pure-Triton edit
        takes Tier A across both ranks, proves honestly after live replay, undoes cleanly to
        `at_base`; plain Python edit automatically falls back to R1 (`tier='C'`)."""
        driver = _FakeCudaDriver()
        sides, rank_nodes, rank_graphs, rank_r1_captures = [], [], [], []

        for rank_idx in range(2):
            mod_name, module = _make_triton_module(self, f"e2e_r{rank_idx}", TRITON_MODULE_SRC)
            base_addr = 0x6000 + rank_idx * 0x10
            module.rms_norm_kernel.device_caches[0] = ({("b4",): _FakeCompiledKernel(base_addr)}, {}, "target")
            node = _FakeNode(_FakeNodeType.CU_GRAPH_NODE_TYPE_KERNEL, base_addr)
            graph_obj = _FakeCudaGraphObj(400 + rank_idx, 1400 + rank_idx, driver, [node])
            rank_nodes.append(node)
            rank_graphs.append(graph_obj)

            wrapper = SimpleNamespace(runtime_mode="FULL", entries={4: graph_obj})
            r1_caps: list = []
            rank_r1_captures.append(r1_caps)
            engine = recapture_mod.Engine(
                lambda wrap=wrapper: [wrap],
                {"FULL": {4}},
                lambda mode, desc, cap_log=r1_caps, wrap=wrapper: (cap_log.append((mode, desc)), wrap.entries.__setitem__(desc, "r1_recaptured")),
            )
            side = Side(
                "worker",
                scope={
                    "worker": SimpleNamespace(),
                    "r1_engine": engine,
                    "pointer_swap_driver": driver,
                    "pointer_swap_compiler": lambda _record, offset=rank_idx: _FakeCompiledKernel(0x7000 + offset * 0x10),
                },
                rank=rank_idx,
                alias={"triton_demo": mod_name},
            )
            side.graph_guard.stamp(guard_mod.graph_key("FULL", 4))
            sides.append(side)

        temp_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_root, True)
        core = SimpleNamespace(model_executor=Executor(sides))
        with mock.patch.dict(sys.modules, {"vllm": SimpleNamespace(__version__="cpu")}):
            host = host_mod.Host(core, temp_root)

        triton_plan = {
            "id": "txn-triton-a",
            "rebuild": ["R1"],
            "modules": {
                "triton_demo": {
                    "on": ["worker"],
                    "base_hash": kernel_locate.text_hash(TRITON_MODULE_SRC),
                    "source": TRITON_MODULE_SRC.replace("return x_ptr + n + 1", "return x_ptr + n + 2"),
                }
            },
        }
        res_a = host.handle({"op": "txn", "plan": triton_plan})
        self.assertEqual(res_a["outcome"], "committed", res_a["receipt"]["findings"])
        for rank_str in ("0", "1"):
            rank_info = res_a["receipt"]["rebuild"]["ranks"][rank_str]
            self.assertEqual((rank_info["tier"], rank_info["patched_nodes"], rank_info["compiled_specializations"]), ("A", 1, 1))
        self.assertEqual(rank_r1_captures, [[], []])
        self.assertEqual([node.exec_state["func"] for node in rank_nodes], [0x7000, 0x7010])

        proof_before = host.handle({"op": "prove", "txn": "txn-triton-a"})
        self.assertFalse(proof_before["ok"])
        self.assertIn("zero_witness", {finding["check"] for finding in proof_before["receipt"]["findings"]})

        for side, graph_obj in zip(sides, rank_graphs):
            side.graph_guard.check_before_replay(guard_mod.graph_key("FULL", 4))
            graph_obj.replay()
        proof_after = host.handle({"op": "prove", "txn": "txn-triton-a"})
        self.assertTrue(proof_after["ok"], proof_after["receipt"]["findings"])

        self.assertTrue(host.handle({"op": "undo", "txn": "txn-triton-a"})["ok"])
        self.assertEqual([node.exec_state["func"] for node in rank_nodes], [0x6000, 0x6010])
        self.assertTrue(host.handle({"op": "at_base"})["ok"])

        python_plan = {
            "id": "txn-py-c",
            "rebuild": ["R1"],
            "modules": {
                "triton_demo": {
                    "on": ["worker"],
                    "base_hash": kernel_locate.text_hash(TRITON_MODULE_SRC),
                    "source": TRITON_MODULE_SRC.replace("return value + 1", "return value + 99"),
                }
            },
        }
        res_c = host.handle({"op": "txn", "plan": python_plan})
        self.assertEqual(res_c["outcome"], "committed", res_c["receipt"]["findings"])
        for rank_str in ("0", "1"):
            rank_info = res_c["receipt"]["rebuild"]["ranks"][rank_str]
            self.assertEqual((rank_info["tier"], rank_info["fallback_reason"]), ("C", "not_triton_jit_edit"))
        self.assertEqual(rank_r1_captures, [[("FULL", 4)], [("FULL", 4)]])
        self.assertTrue(host.handle({"op": "undo", "txn": "txn-py-c"})["ok"])
        self.assertTrue(host.handle({"op": "at_base"})["ok"])

    def test_real_pytorch214_pybind_init_runtime_error_and_piecewise_before_full_stamp(self):
        """Regression tests for three bugs caught on live NVIDIA B200 (PyTorch 2.14 + vLLM V2):
        (1) pybind11 inherited `__init__(self, keep_graph=False)` overwriting `__new__`;
        (2) `raw_cuda_graph()` raising `RuntimeError` when `keep_graph=False` failing closed
            (including in `release_graph_kernels`);
        (3) `Recapture._capture` stamping PIECEWISE wrapper entry keys before FULL warmup."""
        from collections import namedtuple

        class _PybindBase:
            def __init__(self, keep_graph: bool = False):
                self._cpp_keep_graph = bool(keep_graph)

        class _PybindCUDAGraph(_PybindBase):
            def __new__(cls, keep_graph: bool = False):
                inst = super().__new__(cls)
                inst._keep_graph = bool(keep_graph)
                inst._has_graph_exec = False
                inst._raw_graph = 0
                inst._raw_exec = 0
                return inst

            def raw_cuda_graph(self) -> int:
                if not self._raw_graph:
                    raise RuntimeError("No cudaGraph_t is available (keep_graph=false)")
                return self._raw_graph

            def raw_cuda_graph_exec(self) -> int:
                if not self._raw_exec:
                    raise RuntimeError("No cudaGraphExec_t is available")
                return self._raw_exec

            def instantiate(self) -> None:
                self._has_graph_exec = True
                self._raw_exec = 0xE123

            def capture_end(self) -> None:
                self._raw_graph = 0xA123 if self._cpp_keep_graph else 0
                if not self._keep_graph:
                    self.instantiate()

        self.addCleanup(swap_mod.disable_keep_graph)
        self.assertNotIn("__init__", _PybindCUDAGraph.__dict__)
        self.assertTrue(swap_mod.enable_keep_graph(_PybindCUDAGraph))
        g_auto = _PybindCUDAGraph()
        self.assertTrue(g_auto._cpp_keep_graph)
        g_auto.capture_end()
        self.assertEqual(g_auto.raw_cuda_graph(), 0xA123)
        self.assertEqual(g_auto.raw_cuda_graph_exec(), 0xE123)

        g_false = _PybindCUDAGraph(keep_graph=False)
        self.assertFalse(g_false._cpp_keep_graph)
        g_false.capture_end()
        with self.assertRaises(RuntimeError):
            g_false.raw_cuda_graph()
        self.assertEqual(swap_mod.release_graph_kernels(g_false), 0)

        wrapper_err = SimpleNamespace(runtime_mode="FULL", entries={1: g_false})
        infos, guard_keys, left_refs, err = swap_mod._partition_wrapper_entries(
            ["FULL"], [(wrapper_err, "entries")], {"FULL": [1]}, {"FULL": []}
        )
        self.assertIsNone(infos)
        self.assertIsNone(guard_keys)
        self.assertIsNone(left_refs)
        self.assertEqual(err, "keep_graph_unavailable")

        self.assertTrue(swap_mod.disable_keep_graph())
        self.assertNotIn("__init__", _PybindCUDAGraph.__dict__)

        _V2Desc = namedtuple("_V2Desc", ("cg_mode", "num_tokens"))
        guard = guard_mod.GraphReplayGuard(strict_unstamped=True)
        guard.bump_code_epoch("commit:txn-r1")
        pw_wrapper = SimpleNamespace(runtime_mode="PIECEWISE", concrete_cudagraph_entries={})
        full_wrapper = SimpleNamespace(runtime_mode="FULL", entries={})
        pw_desc_v2, full_desc_v2 = _V2Desc("PIECEWISE", 4), _V2Desc("FULL", 4)
        pw_wrapper_key = ("pw_batch_desc", 4)

        def capture(mode, desc):
            if mode == "PIECEWISE":
                pw_wrapper.concrete_cudagraph_entries[pw_wrapper_key] = "pw_graph"
            else:
                guard.check_before_replay(guard_mod.graph_key("PIECEWISE", pw_wrapper_key))
                full_wrapper.entries[desc] = "full_graph"

        eng = recapture_mod.Engine(
            lambda: [pw_wrapper, full_wrapper],
            {"PIECEWISE": {pw_desc_v2}, "FULL": {full_desc_v2}},
            capture,
        )
        rec = recapture_mod.Recapture(guard=guard)
        journal = kernel_journal.Journal()
        journal.mark("txn-r1")
        wit = kernel_witness.Witness()
        res = rec.run(eng, "txn-r1", journal, wit, pointer_swap=False)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["tier"], "C")
        self.assertTrue(guard.is_fresh(guard_mod.graph_key("PIECEWISE", pw_wrapper_key)))
        self.assertTrue(guard.is_fresh(guard_mod.graph_key("FULL", full_desc_v2)))


if __name__ == "__main__":
    unittest.main()
