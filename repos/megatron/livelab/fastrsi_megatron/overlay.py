"""Start a process with edited modules without touching the source tree (fresh checks).

FASTRSI_OVERLAY=<dir> holds `<dotted.module.name>.py` files. Importing this module installs a
finder that loads those modules from the overlay text, with the original file name, so code
objects keep the same `co_filename` and line numbers as the live lab's swapped code. It must be
imported before the overlaid modules are.

The overlay text also goes into linecache under that file name, so inspect.getsource, tracebacks,
@triton.jit and torch.compile see the edit, not the file on disk."""
import importlib.abc
import importlib.machinery
import linecache
import os
import sys


class _Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, text):
        super().__init__(name, path)
        self._text = text

    def get_data(self, path):
        if path == self.path:
            return self._text.encode()
        return super().get_data(path)

    def path_stats(self, path):
        raise OSError("overlay: no bytecode cache")

    def exec_module(self, module):
        # mtime None: linecache.checkcache keeps the entry (it would reload a file entry from disk)
        linecache.cache[self.path] = (len(self._text), None, self._text.splitlines(True), self.path)
        super().exec_module(module)


class OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root
        self.files = {f[:-3]: os.path.join(root, f) for f in os.listdir(root) if f.endswith(".py")}

    def find_spec(self, name, path=None, target=None):
        if name not in self.files:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or not spec.origin:
            return None
        with open(self.files[name]) as f:
            text = f.read()
        spec.loader = _Loader(name, spec.origin, text)
        return spec


def install(root=None):
    """Put an OverlayFinder for `root` (default FASTRSI_OVERLAY) first on sys.meta_path, once per
    root. Returns the finder, or None without an overlay."""
    root = root or os.environ.get("FASTRSI_OVERLAY")
    if not (root and os.path.isdir(root)):
        return None
    for finder in sys.meta_path:
        if isinstance(finder, OverlayFinder) and finder.root == root:
            return finder
    finder = OverlayFinder(root)
    sys.meta_path.insert(0, finder)
    return finder


install()
