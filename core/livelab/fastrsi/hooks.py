"""Stack-safe import hooks (`AfterImportHook`) and source overlays (`OverlayFinder`).

Fixes the `RecursionError` trap when multiple `sys.meta_path` hooks coexist (e.g. an
adapter host hook, a live-reconfig agent hook, and an overlay finder): every finder here
delegates ONLY to `sys.meta_path[idx + 1:]` (finders after itself), never back to index 0.
"""
import ast
import contextlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
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


# -- AST import graph and module-refresh purging -----------------------------------------


def _resolve_import(mod: str, is_pkg: bool, node) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    base = node.module or ""
    if node.level:
        parts = mod.split(".") if is_pkg else mod.split(".")[:-1]
        if node.level > 1:
            parts = parts[: max(0, len(parts) - node.level + 1)]
        base = ".".join(parts + ([base] if base else []))
    if not base:
        return []
    return [base] + [f"{base}.{a.name}" for a in node.names if a.name != "*"]


def _binding_imports(tree):
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        todo.extend(ast.iter_child_nodes(node))


def import_graph(prefixes, modules=None, texts=None, cache=None) -> dict[str, set[str]]:
    """Return `{module: {loaded modules imported at import time}}` for loaded modules under `prefixes`."""
    mods = sys.modules if modules is None else modules
    overrides = texts or {}
    graph: dict[str, set[str]] = {}
    for name, mod in list(mods.items()):
        path = getattr(mod, "__file__", None)
        if not (path and path.endswith(".py") and name.split(".")[0] in prefixes):
            continue
        try:
            src = overrides.get(name)
            stamp = None
            if src is None:
                if cache is not None:
                    st = os.stat(path)
                    stamp = (name, st.st_size, st.st_mtime_ns)
                    hit = cache.get(path)
                    if hit and hit[0] == stamp:
                        graph[name] = {d for d in hit[1] if d in mods and d != name}
                        continue
                with open(path, "rb") as handle:
                    src = handle.read()
            is_pkg = os.path.basename(path) == "__init__.py"
            names = [
                dep
                for node in _binding_imports(ast.parse(src, path))
                for dep in _resolve_import(name, is_pkg, node)
            ]
            if cache is not None and stamp is not None:
                cache[path] = (stamp, names)
        except (OSError, SyntaxError, ValueError):
            continue
        graph[name] = {d for d in names if d in mods and d != name}
    return graph


def dependents(changed, graph: dict[str, set[str]]) -> set[str]:
    """Return `changed` plus every module in `graph` that transitively imports one of them."""
    rev: dict[str, set[str]] = {}
    for mod, deps in graph.items():
        for dep in deps:
            rev.setdefault(dep, set()).add(mod)
    out, todo = set(changed), list(changed)
    while todo:
        for mod in rev.get(todo.pop(), ()):
            if mod not in out:
                out.add(mod)
                todo.append(mod)
    return out


def unbind_module(modules, name: str, *, expected=None) -> None:
    """Remove `name`'s attribute binding from its parent package in `modules`."""
    parent_name, _, child = name.rpartition(".")
    parent = modules.get(parent_name)
    if parent is None:
        return
    if expected is not None:
        if getattr(parent, child, None) is expected:
            with contextlib.suppress(AttributeError):
                delattr(parent, child)
    elif child in getattr(parent, "__dict__", {}):
        with contextlib.suppress(AttributeError):
            delattr(parent, child)


def purge_modules(names, modules=None, *, unlink_pyc: bool = False) -> list[str]:
    """Drop `names` from `modules`, unbind from parent packages, optionally remove `.pyc`, and invalidate caches."""
    mods = sys.modules if modules is None else modules
    dropped = list(names)
    for name in dropped:
        old = mods.pop(name, None)
        if old is not None:
            unbind_module(mods, name, expected=old)
        if unlink_pyc:
            mod_file = getattr(old, "__file__", None)
            if mod_file:
                with contextlib.suppress(OSError, ValueError, NotImplementedError):
                    os.unlink(importlib.util.cache_from_source(mod_file))
    if dropped:
        importlib.invalidate_caches()
    return dropped
