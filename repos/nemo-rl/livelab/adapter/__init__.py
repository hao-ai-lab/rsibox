"""Bootstrap `core/livelab` onto `sys.path` for the NVIDIA NeMo-RL live-lab adapter."""
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
_CANDS = (
    os.path.join(_HERE, "..", "..", "..", "..", "core", "livelab"),
    os.path.join(_HERE, "..", "..", "..", "core", "livelab"),
)
LIVELAB = os.path.abspath(
    os.environ.get("RSIBOX_LIVELAB") or next((c for c in _CANDS if os.path.isdir(c)), _CANDS[0])
)
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
