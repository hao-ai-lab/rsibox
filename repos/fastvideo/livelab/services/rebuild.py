"""In-place rebuild in a warm process: the T1 restart tier.

A warm worker keeps its CUDA context, its weights on the GPU, its JIT modules (FlashInfer,
Triton) and its compile caches. An edit the R0 swap refuses (a constructor edit, a new field,
a module constant, a new class) does not need a new process if the process can (1) run the
edited modules' new source and (2) rebuild the objects built from them, keeping their tensors.

1. `refresh(files)`: re-import each edited module from its new text, and every loaded module
   that imports it (directly or not, read from the import statements: `import_graph`), in the
   order they were first imported. A failure puts the old modules back. The old module objects
   are returned, so the caller can map old classes to new ones (`class_map`).
2. `swap_classes(objs, cmap)`: an object whose class was refreshed gets the new class
   (`obj.__class__ = new`), its state kept. Method, property and constant edits take effect.
3. `merge_fields(live, base, new)`: what a constructor edit changes. `base` and `new` are the
   same object built by the old and the new constructor with the same arguments, on the meta
   device (no memory, no weights). Three-way merge, per attribute of every submodule:
   - an attribute the new constructor adds: set on the live object;
   - an attribute whose constructed value changed (old ctor gives 1e-6, new gives 1e-5): set;
   - otherwise the live value stays (it may hold state set after construction, such as caches).
   Tensors cannot come from the meta build: a new parameter or buffer, a new submodule that
   holds tensors, or a changed parameter shape is a finding (the edit needs a real load: a
   restart tier), and nothing is changed.

Not covered (findings or a restart): native code (extension `.so`), torch op schemas that a
re-import would register again, code captured in CUDA graphs, and objects held outside the
roots the caller passes (they keep the old class).
"""
import ast
import importlib
import importlib.abc
import importlib.util
import linecache
import os
import sys
import time

_SKIP = {"_parameters", "_buffers", "_modules", "_non_persistent_buffers_set", "training",
         "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
         "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
         "_load_state_dict_post_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
         "_forward_pre_hooks_with_kwargs", "_is_full_backward_hook"}
_PRIMITIVE = (bool, int, float, complex, str, bytes, type(None))


def finding(check, observed, expected):
    return {"check": check, "observed": observed, "expected": expected}


# -- 1. refresh -------------------------------------------------------------------------------
def _resolve(mod, is_pkg, node):
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    base = node.module or ""
    if node.level:
        parts = mod.split(".") if is_pkg else mod.split(".")[:-1]
        if node.level > 1:
            parts = parts[:len(parts) - node.level + 1]
        base = ".".join(parts + ([base] if base else []))
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


_PARSED = {}   # file -> ((size, mtime_ns), names its binding imports may load): parse once per process


def _imports_of(name, f, src=None):
    is_pkg = os.path.basename(f) == "__init__.py"
    if src is None:
        st = os.stat(f)
        stamp = (name, st.st_size, st.st_mtime_ns)
        hit = _PARSED.get(f)
        if hit and hit[0] == stamp:
            return hit[1]
        with open(f, "rb") as fh:
            src = fh.read()
    else:
        stamp = None
    names = [d for node in _binding_imports(ast.parse(src, f)) for d in _resolve(name, is_pkg, node)]
    if stamp is not None:
        _PARSED[f] = (stamp, names)
    return names


def import_graph(prefixes, modules=None, texts=None):
    """{module: loaded modules it imports at import time} for loaded modules under `prefixes`.
    `texts`: {module: source} read instead of the file (an edit not written to disk). Files are
    parsed once per process and again only when their size or mtime changed (a first graph of
    FastH3's 357 modules is most of a rebuild's re-import time)."""
    modules = sys.modules if modules is None else modules
    texts = texts or {}
    graph = {}
    for name, m in list(modules.items()):
        f = getattr(m, "__file__", None)
        if not (f and f.endswith(".py") and name.split(".")[0] in prefixes):
            continue
        try:
            names = _imports_of(name, f, texts.get(name))
        except (OSError, SyntaxError, ValueError):
            continue
        graph[name] = {d for d in names if d in modules and d != name}
    return graph


def dependents(changed, graph):
    rev = {}
    for m, deps in graph.items():
        for d in deps:
            rev.setdefault(d, set()).add(m)
    out, todo = set(changed), list(changed)
    while todo:
        for m in rev.get(todo.pop(), ()):
            if m not in out:
                out.add(m)
                todo.append(m)
    return out


class _TextLoader(importlib.abc.Loader):
    def __init__(self, text, origin):
        self.text, self.origin = text, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.text, self.origin, "exec"), module.__dict__)


class TextFinder(importlib.abc.MetaPathFinder):
    """Serves {module: source text} with the module's real file as its origin (so tracebacks,
    `inspect` and Triton's source readers point at the right file)."""

    def __init__(self, texts, origins):
        self.texts, self.origins = texts, origins

    def find_spec(self, name, path=None, target=None):
        if name not in self.texts:
            return None
        origin = self.origins[name]
        is_pkg = os.path.basename(origin) == "__init__.py"
        spec = importlib.util.spec_from_file_location(
            name, origin, loader=_TextLoader(self.texts[name], origin),
            submodule_search_locations=[os.path.dirname(origin)] if is_pkg else None)
        return spec


def _unbind(modules, n):
    parent, _, child = n.rpartition(".")
    p = modules.get(parent)
    if p is not None and child in getattr(p, "__dict__", {}):
        try:
            delattr(p, child)
        except AttributeError:
            pass


def refresh(files, prefixes, modules=None, max_modules=200):
    """Re-import the edited modules from `files` ({module: new text}) and their importers.
    Returns {"ok", "refreshed": [names], "old": {name: old module}, "findings", "seconds"}."""
    t0 = time.perf_counter()
    modules = sys.modules if modules is None else modules
    missing = [m for m in files if m not in modules]
    if missing:
        return {"ok": False, "refreshed": [], "old": {}, "seconds": 0.0, "findings": [finding(
            "not_loaded", f"{missing} not loaded in this process", "edits of loaded modules")]}
    for m, text in files.items():
        try:
            compile(text, modules[m].__file__, "exec")
        except SyntaxError as e:
            return {"ok": False, "refreshed": [], "old": {}, "seconds": 0.0,
                    "findings": [finding("patch_invalid", f"{m}: {e}", "valid Python")]}
    graph = import_graph(prefixes, modules, texts=files)
    names = dependents(set(files), graph)
    if len(names) > max_modules:
        return {"ok": False, "refreshed": [], "old": {}, "seconds": 0.0, "findings": [finding(
            "refresh_too_wide", f"{len(names)} modules import the edited ones",
            f"at most {max_modules} (an edit in the package core: restart the worker)")]}
    order = [n for n in list(modules) if n in names]          # first-import order
    old = {n: modules[n] for n in order}
    origins = {n: old[n].__file__ for n in files}
    for n in order:
        modules.pop(n)
        _unbind(modules, n)
    importlib.invalidate_caches()
    finder = TextFinder(dict(files), origins)
    sys.meta_path.insert(0, finder)
    try:
        for n in order:
            if n not in modules:
                importlib.import_module(n)
    except BaseException as e:
        for n in order:                                       # put the old modules back
            modules[n] = old[n]
            parent, _, child = n.rpartition(".")
            if parent in modules:
                setattr(modules[parent], child, old[n])
        return {"ok": False, "refreshed": [], "old": {}, "seconds": round(time.perf_counter() - t0, 4),
                "findings": [finding("refresh_failed", f"{type(e).__name__}: {e}",
                                     "the edited modules import")]}
    finally:
        sys.meta_path.remove(finder)
    for n, text in files.items():                             # what inspect/Triton/Dynamo read
        f = origins[n]
        linecache.cache[f] = (len(text), None, text.splitlines(True), f)
    return {"ok": True, "refreshed": order, "old": old, "findings": [],
            "seconds": round(time.perf_counter() - t0, 4)}


def restore(old_modules, modules=None):
    """Put the modules `refresh` replaced back (an edit refused after its refresh)."""
    modules = sys.modules if modules is None else modules
    for n, m in old_modules.items():
        modules[n] = m
        parent, _, child = n.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, m)
        f = getattr(m, "__file__", None)
        if f:
            linecache.cache.pop(f, None)


def class_map(old_modules, modules=None):
    """{old class: new class} for every class defined in a refreshed module (by qualname)."""
    modules = sys.modules if modules is None else modules
    out = {}
    for name, om in old_modules.items():
        nm = modules.get(name)
        if nm is None or nm is om:
            continue
        for v in list(vars(om).values()):
            if isinstance(v, type) and v.__module__ == name:
                new = nm
                try:
                    for part in v.__qualname__.split("."):
                        new = getattr(new, part)
                except AttributeError:
                    continue
                if isinstance(new, type) and new is not v:
                    out[v] = new
    return out


# -- 2. class swap ----------------------------------------------------------------------------
def swap_classes(objs, cmap):
    """Set the new class on every object in `objs` whose class was refreshed. Returns
    {"swapped": n, "findings": [...]}."""
    n, fs = 0, []
    seen = set()
    for o in objs:
        if id(o) in seen:
            continue
        seen.add(id(o))
        new = cmap.get(type(o))
        if new is None:
            continue
        try:
            o.__class__ = new
            n += 1
        except TypeError as e:
            fs.append(finding("class_swap_failed", f"{type(o).__qualname__}: {e}",
                              "a class with the same layout"))
    return {"swapped": n, "findings": fs}


# -- 3. constructor edits ---------------------------------------------------------------------
def _same(a, b):
    """Is the constructed value unchanged between the old and the new constructor?"""
    if type(a) is not type(b):
        return False
    if isinstance(a, _PRIMITIVE) or type(a).__module__ == "torch" and not _is_tensor(a):
        try:
            return bool(a == b)
        except Exception:
            return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    try:
        import enum
        if isinstance(a, enum.Enum):
            return a is b
    except Exception:
        pass
    return True       # same type of object (a config, an impl): the live one stays


def _is_tensor(v):
    try:
        import torch
        return isinstance(v, torch.Tensor)
    except Exception:
        return False


def _holds_tensors(v):
    if _is_tensor(v):
        return True
    params = getattr(v, "parameters", None)
    if callable(params):
        try:
            return any(True for _ in v.parameters()) or any(True for _ in v.buffers())
        except Exception:
            return True
    return False


def merge_fields(live, base, new, dry_run=False):
    """Three-way merge of a constructor edit into the live module tree (see module doc).
    `live`, `base`, `new`: torch modules (base/new on meta). Returns {"ok", "set": [paths],
    "swapped": n, "findings"}; on a finding nothing is changed. `dry_run`: plan only."""
    plan, fs = [], []
    live_mods = dict(live.named_modules(remove_duplicate=False))
    base_mods = dict(base.named_modules(remove_duplicate=False))
    for path, nm in new.named_modules(remove_duplicate=False):
        lm, bm = live_mods.get(path), base_mods.get(path)
        if lm is None:
            if _holds_tensors(nm):
                fs.append(finding("new_submodule", f"{path or '<root>'}: {type(nm).__name__} holds tensors",
                                  "no new weights (a load restart tier)"))
            else:
                plan.append(("module", path, nm))
            continue
        for kind in ("_parameters", "_buffers"):
            lt, nt = getattr(lm, kind), getattr(nm, kind)
            for k, v in nt.items():
                if k not in lt:
                    fs.append(finding("new_tensor", f"{path}.{k}", "no new weights (a load restart tier)"))
                elif v is not None and lt[k] is not None and tuple(v.shape) != tuple(lt[k].shape):
                    fs.append(finding("tensor_shape_changed", f"{path}.{k}: {tuple(lt[k].shape)} -> "
                                      f"{tuple(v.shape)}", "the loaded shapes"))
        if type(lm) is not type(nm):
            plan.append(("class", path, type(nm)))
        bd = bm.__dict__ if bm is not None else {}
        for k, v in nm.__dict__.items():
            if k in _SKIP:
                continue
            if k not in lm.__dict__ or (k in bd and not _same(bd[k], v)):
                if _holds_tensors(v):
                    fs.append(finding("new_tensor_field", f"{path}.{k}", "no tensors built by the edit"))
                else:
                    plan.append(("field", path, (k, v)))
    if fs or dry_run:
        return {"ok": not fs, "planned": len(plan), "set": [], "swapped": 0, "findings": fs}
    done, swapped = [], 0
    for kind, path, v in plan:
        if kind == "module":
            parent, _, child = path.rpartition(".")
            live.get_submodule(parent).add_module(child, v)
            done.append(path)
        elif kind == "class":
            live_mods[path].__class__ = v
            swapped += 1
        else:
            k, val = v
            object.__setattr__(live_mods[path], k, val)
            done.append(f"{path}.{k}" if path else k)
    return {"ok": True, "set": done, "swapped": swapped, "findings": []}
