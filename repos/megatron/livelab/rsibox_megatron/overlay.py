"""Start a process with edited modules without touching the source tree (fresh checks).

RSIBOX_OVERLAY=<dir> holds `<dotted.module.name>.py` files. Importing this module installs a
`rsibox.hooks.OverlayFinder` that loads those modules from the overlay text under their
original file names and pins `linecache` (`mtime=None`) so `inspect.getsource`, `@triton.jit`,
and `torch.compile` see the edit, not the file on disk.
"""
from rsibox.hooks import OverlayFinder, install_overlay as install  # noqa: F401

install()
