"""Start a process with edited modules without touching the source tree (fresh checks).

FASTRSI_OVERLAY=<dir> holds `<dotted.module.name>.py` files. Importing this module installs a
finder that loads those modules from the overlay text, with the original file name, so code
objects keep the same `co_filename` and line numbers as the live lab's swapped code. It must be
imported before the overlaid modules are."""
import importlib.abc
import importlib.machinery
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


class OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
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
    root = root or os.environ.get("FASTRSI_OVERLAY")
    if root and os.path.isdir(root):
        sys.meta_path.insert(0, OverlayFinder(root))


install()
