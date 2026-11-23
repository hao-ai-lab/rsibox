"""rsibox live lab for Megatron-LM. The transaction kernel (`kernel`) and the pre-fork CUDA
guards (`forksafe`) are the framework's, in `core/livelab` (pure Python); importing this
package puts that folder (RSIBOX_LIVELAB, else RSIBOX_KERNEL, default
`<repo>/core/livelab`) on sys.path."""
import os
import sys

LIVELAB = os.path.abspath(
    os.environ.get("RSIBOX_LIVELAB") or os.environ.get("RSIBOX_KERNEL") or os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
