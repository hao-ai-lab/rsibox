"""NVIDIA Dynamo live-lab adapter package (`core/livelab` path bootstrap)."""
from __future__ import annotations
import os, sys
_LIVELAB_CORE = os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "core", "livelab"))
if os.path.isdir(_LIVELAB_CORE) and _LIVELAB_CORE not in sys.path: sys.path.insert(0, _LIVELAB_CORE)
