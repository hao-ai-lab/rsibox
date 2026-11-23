"""FastVideo restart and cache services; ensures `core/livelab` is importable."""
import os, sys

_LIVELAB = os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if _LIVELAB not in sys.path:
    sys.path.insert(0, _LIVELAB)
