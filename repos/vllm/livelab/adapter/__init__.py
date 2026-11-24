"""vLLM adapter for the rsibox transaction kernel (`core/livelab/kernel`): an engine-core
host in the busy loop, a worker extension, and a file control channel. See README.md.

Importing the package puts the framework's live-lab folder (RSIBOX_LIVELAB, default
`<repo>/core/livelab`) on sys.path, so `from kernel import ...` resolves in the engine
core and in every worker process that imports `adapter.worker_ext`."""
import os
import sys

LIVELAB = os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)

from .graph_guard import GraphReplayGuard, StaleGraphReplayError, graph_key, input_signature  # noqa: E402
from .graph_patch import (  # noqa: E402
    KernelSwapSpec,
    collect_triton_func_map,
    disable_keep_graph_hook,
    enable_keep_graph_hook,
    extract_cudagraph,
    patch_graphs_in_place,
)
from .native import BackgroundBuildJob, BackgroundBuildPool, compile_async  # noqa: E402
from .recapture import teardown_graph_pool  # noqa: E402
from .snapshot_hooks import (  # noqa: E402
    VllmQuiesceHooks,
    VllmShadowPromoter,
    make_shadow_recovery_driver,
    park_shadow_server,
)

__all__ = [
    "BackgroundBuildJob", "BackgroundBuildPool", "GraphReplayGuard", "KernelSwapSpec", "LIVELAB",
    "StaleGraphReplayError", "VllmQuiesceHooks", "VllmShadowPromoter",
    "collect_triton_func_map", "compile_async", "disable_keep_graph_hook",
    "enable_keep_graph_hook", "extract_cudagraph", "graph_key", "input_signature",
    "make_shadow_recovery_driver", "park_shadow_server", "patch_graphs_in_place",
    "teardown_graph_pool",
]
