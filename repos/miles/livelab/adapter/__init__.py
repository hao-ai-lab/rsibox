"""Package bootstrap for `repos/miles/livelab/adapter` (`core/livelab` on `sys.path`)."""
import os, sys

_HERE = os.path.dirname(os.path.realpath(__file__))
_CANDIDATES = (os.path.join(_HERE, "..", "..", "..", "..", "core", "livelab"),
               os.path.join(_HERE, "..", "..", "..", "core", "livelab"))
LIVELAB = os.path.abspath(os.environ.get("RSIBOX_LIVELAB")
                          or next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0]))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
