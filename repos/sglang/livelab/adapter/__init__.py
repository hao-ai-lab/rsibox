"""SGLang adapter for the FastRSI transaction kernel (`core/livelab/kernel`)."""
import os
import sys

LIVELAB = os.path.abspath(os.environ.get("FASTRSI_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
