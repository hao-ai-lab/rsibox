"""GPU golden tests of patch 20's graph cache (`_h3_graphs_*` in `minimax_h3.py`).

Run on one CUDA GPU with a FastVideo tree that has patches 01-20 applied first on PYTHONPATH:
    python3 -m unittest repos/fastvideo/patches/h3-hillclimb/test_20_graph_cache.py -v

Skips without CUDA or FastVideo. On a tree with 01-19 only it fails (no `_h3_graphs`): that is
the bug it is for. Each case captures small CUDA graphs into the cache the way the FFN and QKV
paths do (shared pool, side stream, shared input) and checks GPU memory, so it shows what a
second request shape does without loading H3. The whole H3 run is `tools/h3_shapes.py`.
"""
import os
import unittest

try:
    import torch
    CUDA = torch.cuda.is_available()
except ImportError:                                   # pragma: no cover - CPU CI without torch
    torch, CUDA = None, False

H3 = None
if CUDA:
    try:
        from fastvideo.models.dits import minimax_h3 as H3
    except ImportError:
        H3 = None


def capture(graphs, x, owner):
    """One graph the way patch 20 captures: shared input, shared side stream and pool; the
    output (4 inputs' worth, as the QKV graph's q/k/v/gate) stays held by the cache entry."""
    static_input = H3._h3_graphs_buffer(graphs, "hidden", x)
    side = graphs.side
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        static_input.copy_(x)
        torch.cat([static_input * 2] * 4)             # warm-up, as the real path
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=graphs.pool, stream=side):
        out = torch.cat([static_input * 2] * 4)
    torch.cuda.current_stream().wait_stream(side)
    graphs.captures += 1
    graphs.projection[owner] = ("sig", graph, static_input, None, out)
    return graph, static_input, out


@unittest.skipUnless(CUDA and H3 is not None, "needs CUDA and FastVideo with patches 01-20")
class GraphCacheTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(H3, "_h3_graphs"), "patch 20 is not applied (01-19 only?)")
        self.graphs = H3._h3_graphs_new()              # a fresh cache, not the engine's
        self.owners = [torch.nn.Linear(1, 1) for _ in range(8)]   # weak keys, as the blocks
        torch.cuda.empty_cache()

    def x(self, rows):
        return torch.randn(1, rows, 1024, device="cuda", dtype=torch.bfloat16)

    def test_a_second_shape_frees_every_graph_of_the_first_before_it_captures(self):
        """Input: 8 graphs at 4096 rows, then the first call at 8192 rows. Expect: entering the
        new shape resets all 8 (resets 1, nothing held) and memory falls back to the floor
        before any 8192-row graph exists; replaying a fresh 8192-row graph gives 2x its input.
        Proves: no old-shape graph is alive while the new shape captures (patches 15-17 kept
        49 of 50 layers' graphs, and the one being replaced, alive: OOM). Not: H3's memory
        (tools/h3_shapes.py measures 77 -> 122 GiB at 124 -> 243 frames)."""
        policy = H3._h3_graph_policy()
        small, big = self.x(4096), self.x(8192)
        floor = torch.cuda.memory_allocated()                      # the two requests' inputs
        for owner in self.owners:
            H3._h3_graphs_enter(self.graphs, small, policy)
            capture(self.graphs, small, owner)
        held = torch.cuda.memory_allocated() - floor
        self.assertGreater(held, 8 * 4 * small.nbytes)            # 8 graphs' outputs at least
        H3._h3_graphs_enter(self.graphs, big, policy)
        self.assertEqual((self.graphs.resets, len(self.graphs.projection)), (1, 0))
        self.assertEqual(torch.cuda.memory_allocated(), floor)
        graph, static_input, out = capture(self.graphs, big, self.owners[0])
        static_input.copy_(big)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out[:1], big * 2))

    def test_all_graphs_of_a_shape_share_one_input_buffer(self):
        """Input: 8 graphs at one shape. Expect: one input buffer, the same storage in every
        entry. Proves: the inputs cost one layer, not 50 (16 GiB at 124 frames). Not: that
        sharing is exact (the H3 runs compare every video with base)."""
        x = self.x(4096)
        policy = H3._h3_graph_policy()
        for owner in self.owners:
            H3._h3_graphs_enter(self.graphs, x, policy)
            capture(self.graphs, x, owner)
        ptrs = {s[2].data_ptr() for s in self.graphs.projection.values()}
        self.assertEqual(len(ptrs), 1)
        self.assertEqual(list(self.graphs.inputs), ["hidden"])

    def test_a_changed_nvfp4_gemm_setting_resets_the_graphs(self):
        """Input: graphs captured, then FASTVIDEO_NVFP4_MM_BACKEND overridden in the process.
        Expect: the policy differs and the next call resets every graph. Proves: a graph
        captured under one GEMM backend or row split never replays under another (patch 15's
        signature did not include them). Not: that the other backend gives the same bits."""
        x = self.x(4096)
        before = H3._h3_graph_policy()
        H3._h3_graphs_enter(self.graphs, x, before)
        capture(self.graphs, x, self.owners[0])
        with H3.envs.FASTVIDEO_NVFP4_MM_BACKEND.override("cutlass"):
            after = H3._h3_graph_policy()
        self.assertNotEqual(before, after)
        H3._h3_graphs_enter(self.graphs, x, after)
        self.assertEqual((self.graphs.resets, len(self.graphs.projection)), (1, 0))

    def test_graphs_off_and_over_budget_fall_back_to_eager(self):
        """Input: FASTVIDEO_H3_DIT_CUDA_GRAPHS=0, then a headroom larger than the GPU.
        Expect: no policy (the callers return None: eager), and `fits` refuses and counts a
        skip. Proves: the opt-out and the budget refuse before anything is captured. Not:
        the whole-model run under a budget (tools/h3_shapes.py with the headroom set)."""
        x = self.x(4096)
        old = dict(os.environ)
        try:
            os.environ["FASTVIDEO_H3_DIT_CUDA_GRAPHS"] = "0"
            self.assertIsNone(H3._h3_graph_policy())
            os.environ["FASTVIDEO_H3_DIT_CUDA_GRAPHS"] = "1"
            os.environ["FASTVIDEO_H3_GRAPH_HEADROOM_GIB"] = "100000"
            H3._h3_graphs_enter(self.graphs, x, H3._h3_graph_policy())
            self.assertFalse(H3._h3_graphs_fits(self.graphs, x))
            self.assertEqual(self.graphs.skipped, 1)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_taking_over_from_patches_15_17_live_releases_their_graphs(self):
        """Input: graphs held the way patches 15-17 keep them (a WeakKeyDictionary on
        `_h3_ff_graph_forward._graphs_v2`), then the process cache is created, as on the first
        call after patch 20 arrives as a live edit. Expect: the old attribute is gone and its
        memory is free. Proves: a live upgrade does not hold both sets of graphs (holding both
        ran out of memory at 124 frames on 1 GPU, our 2026-11-19 run). Not: a fresh process
        (no such attribute: nothing to release)."""
        import weakref
        x = self.x(4096)
        floor = torch.cuda.memory_allocated()
        legacy = H3._h3_graphs_new()
        H3._h3_graphs_enter(legacy, x, H3._h3_graph_policy())
        capture(legacy, x, self.owners[0])
        old = weakref.WeakKeyDictionary({self.owners[0]: legacy.projection[self.owners[0]]})
        legacy.projection.clear()
        legacy.inputs = {}
        del legacy
        saved = vars(H3._h3_graphs).pop("state", None)
        try:
            H3._h3_ff_graph_forward._graphs_v2 = old
            del old
            self.assertGreater(torch.cuda.memory_allocated(), floor)
            H3._h3_graphs()
            self.assertNotIn("_graphs_v2", vars(H3._h3_ff_graph_forward))
            self.assertEqual(torch.cuda.memory_allocated(), floor)
        finally:
            vars(H3._h3_graphs).pop("state", None)
            if saved is not None:
                H3._h3_graphs.state = saved

    def tearDown(self):
        if hasattr(self, "graphs"):
            H3._h3_graphs_reset(self.graphs)


if __name__ == "__main__":
    unittest.main()
