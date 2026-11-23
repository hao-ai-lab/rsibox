"""FastVideo live-lab adapter: the kernel inside FastVideo workers (agent) and the lab host.

Importing the package puts the framework's live-lab folder (RSIBOX_LIVELAB, default
`<repo>/core/livelab`) on sys.path, so `from kernel import ...` resolves in the lab host
and in every worker process that imports `adapter.agent`."""
import os
import sys

LIVELAB = os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
