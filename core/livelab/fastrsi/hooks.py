"""Stack-safe import hooks (`AfterImportHook`) and source overlays (`OverlayFinder`).

Fixes the `RecursionError` trap when multiple `sys.meta_path` hooks coexist (e.g. an
adapter host hook, a live-reconfig agent hook, and an overlay finder): every finder here
delegates ONLY to `sys.meta_path[idx + 1:]` (finders after itself), never back to index 0.
"""
import importlib.abc
import importlib.machinery
import linecache
import os
import sys


def _find_downstream_spec(finder, name, path, target=None):
    """Ask only the finders AFTER `finder` on `sys.meta_path` for `name`'s ModuleSpec."""
    finders = list(sys.meta_path)
    later = finders[finders.index(finder) + 1:] if finder in finders else finders
    for other in later:
        if other is finder or not hasattr(other, "find_spec"):
            continue
        spec = other.find_spec(name, path, target)
        if spec is not None:
            return spec
    return None


class AfterImportHook(importlib.abc.MetaPathFinder):
    """Run `callback(module)` immediately after `module_name` finishes `exec_module`.

    Delegates strictly to downstream finders on `sys.meta_path` so stacked hooks or
    overlays never recurse infinitely and overlays placed downstream still take effect.
    """

    def __init__(self, module_name: str, callback, *, once: bool = True):
        self.module_name = module_name
        self.callback = callback
        self.once = once

    def find_spec(self, name, path=None, target=None):
        if name != self.module_name:
            return None
        spec = _find_downstream_spec(self, name, path, target)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            if self.once:
                sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            self.callback(module)

        spec.loader.exec_module = exec_module
        return spec


def install_import_hook(module_name: str, callback, *, once: bool = True):
    """Invoke `callback(module)` now if `module_name` is already imported, else install a
    stack-safe `AfterImportHook` at `sys.meta_path[0]` (idempotent per `(module_name, callback)`)."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        callback(existing)
        return None
    for finder in sys.meta_path:
        if (
            isinstance(finder, AfterImportHook)
            and finder.module_name == module_name
            and finder.callback == callback
        ):
            return finder
    hook = AfterImportHook(module_name, callback, once=once)
    sys.meta_path.insert(0, hook)
    return hook


class _OverlayLoader(importlib.machinery.SourceFileLoader):
    """Loads overlaid source text under the original module's `co_filename` and pins
    `linecache` (`mtime=None`) so `inspect.getsource`, `@triton.jit`, and Dynamo see it."""

    def __init__(self, name: str, origin: str, text: str):
        super().__init__(name, origin)
        self._text = text

    def get_data(self, path):
        if path == self.path:
            return self._text.encode("utf-8")
        return super().get_data(path)

    def path_stats(self, path):
        raise OSError("overlay: bypass bytecode cache")

    def exec_module(self, module):
        linecache.cache[self.path] = (
            len(self._text),
            None,
            self._text.splitlines(True),
            self.path,
        )
        super().exec_module(module)


class OverlayFinder(importlib.abc.MetaPathFinder):
    """Load selected modules from overlaid source text while keeping their original file paths.

    Accepts either `root` (a directory of `<dotted.module.name>.py` files) and/or an explicit
    `files` dict `{module_name: source_text}`. Delegates downstream on `sys.meta_path` to
    locate each module's original `spec.origin`.
    """

    def __init__(self, root: str | None = None, *, files: dict[str, str] | None = None):
        self.root = os.path.abspath(root) if root else None
        self.disk_files: dict[str, str] = {}
        if self.root and os.path.isdir(self.root):
            self.disk_files = {
                fname[:-3]: os.path.join(self.root, fname)
                for fname in os.listdir(self.root)
                if fname.endswith(".py")
            }
        self.inline_files: dict[str, str] = dict(files or {})

    def _source_for(self, name: str) -> str | None:
        if name in self.inline_files:
            return self.inline_files[name]
        path = self.disk_files.get(name)
        if path is not None:
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        return None

    def find_spec(self, name, path=None, target=None):
        text = self._source_for(name)
        if text is None:
            return None
        spec = _find_downstream_spec(self, name, path, target)
        if spec is None or not spec.origin:
            return None
        spec.loader = _OverlayLoader(name, spec.origin, text)
        return spec


def install_overlay(root: str | None = None, *, files: dict[str, str] | None = None):
    """Install an `OverlayFinder` on `sys.meta_path` for `root` (default `FASTRSI_OVERLAY`)
    and/or inline `files`. Returns the finder, or None if neither has any overlays."""
    resolved_root = root if root is not None else os.environ.get("FASTRSI_OVERLAY")
    if resolved_root and not os.path.isdir(resolved_root):
        resolved_root = None
    if not resolved_root and not files:
        return None
    if resolved_root and not files:
        abs_root = os.path.abspath(resolved_root)
        for finder in sys.meta_path:
            if isinstance(finder, OverlayFinder) and finder.root == abs_root and not finder.inline_files:
                return finder
    finder = OverlayFinder(resolved_root, files=files)
    if not finder.disk_files and not finder.inline_files:
        return None
    sys.meta_path.insert(0, finder)
    return finder
