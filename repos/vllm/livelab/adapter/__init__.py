"""vLLM adapter for the FastRSI transaction kernel (`template/livelab/kernel`): an engine-core
host in the busy loop, a worker extension, and a file control channel. See README.md.

Importing the package puts the framework's live-lab folder (FASTRSI_LIVELAB, default
`<repo>/template/livelab`) on sys.path, so `from kernel import ...` resolves in the engine
core and in every worker process that imports `adapter.worker_ext`."""
import os
import sys

LIVELAB = os.path.abspath(os.environ.get("FASTRSI_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "template", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
