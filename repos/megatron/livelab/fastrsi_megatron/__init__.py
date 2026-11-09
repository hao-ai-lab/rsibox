"""FastRSI live lab for Megatron-LM. The transaction kernel (`kernel`) and the pre-fork CUDA
guards (`forksafe`) are the framework's, in `template/livelab` (pure Python); importing this
package puts that folder (FASTRSI_LIVELAB, else FASTRSI_KERNEL, default
`<repo>/template/livelab`) on sys.path."""
import os
import sys

LIVELAB = os.path.abspath(
    os.environ.get("FASTRSI_LIVELAB") or os.environ.get("FASTRSI_KERNEL") or os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "template", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
