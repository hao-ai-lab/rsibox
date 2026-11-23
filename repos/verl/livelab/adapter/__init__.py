"""Onboarding setup: put `<repo>/core/livelab` (`RSIBOX_LIVELAB`) on `sys.path`."""
import os, sys

_HERE = os.path.dirname(os.path.realpath(__file__))
LIVELAB = os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or next(
    (p for p in (os.path.join(_HERE, "..", "..", "..", "..", "core", "livelab"),
                 os.path.join(_HERE, "..", "..", "..", "core", "livelab")) if os.path.isdir(p)),
    os.path.join(_HERE, "..", "..", "..", "..", "core", "livelab")))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
