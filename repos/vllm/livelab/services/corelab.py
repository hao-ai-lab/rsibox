"""Puts the framework's shared live-lab runtime on sys.path: core/livelab, home of `fastrsi`
(process, IPC and import-hook helpers) and `forksafe` (pre-fork CUDA guards). FASTRSI_LIVELAB
overrides the folder. Stdlib only: importing it costs nothing."""
import os
import sys

LIVELAB = os.path.abspath(os.environ.get("FASTRSI_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(1, LIVELAB)
