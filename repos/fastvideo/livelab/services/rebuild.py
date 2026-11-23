"""In-place rebuild in a warm process, without a restart.

A warm worker keeps its CUDA context, its weights on the GPU, its JIT modules (FlashInfer,
Triton) and its compile caches. An edit the live swap refuses (a constructor edit, a new field,
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
   restart), and nothing is changed.

Not covered (findings or a restart): native code (extension `.so`), torch op schemas that a
re-import would register again, code captured in CUDA graphs, and objects held outside the
roots the caller passes (they keep the old class).
"""
import importlib
import importlib.abc
import importlib.util
import linecache
import os
import sys
import time
import types

from rsibox import harness, hooks

_SKIP = {"_parameters", "_buffers", "_modules", "_non_persistent_buffers_set", "training",
         "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
         "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
         "_load_state_dict_post_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
         "_forward_pre_hooks_with_kwargs", "_is_full_backward_hook"}
_PRIMITIVE = (bool, int, float, complex, str, bytes, type(None))

finding = harness.plain_finding
_PARSED = {}   # file -> ((name, size, mtime_ns), names): parse once per process


# -- 1. refresh -------------------------------------------------------------------------------
def import_graph(prefixes, modules=None, texts=None):
    """{module: loaded modules it imports at import time} for loaded modules under `prefixes`."""
    return hooks.import_graph(prefixes, modules=modules, texts=texts, cache=_PARSED)


dependents = hooks.dependents
unbind = hooks.unbind_module


class _TextLoader(importlib.abc.Loader):
    def __init__(self, text, origin):
        self.text, self.origin = text, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.text, self.origin, "exec"), module.__dict__)


class TextFinder(importlib.abc.MetaPathFinder):
    """Serves {module: source text} with the module's real file as its origin."""

    def __init__(self, texts, origins):
        self.texts, self.origins = texts, origins

    def find_spec(self, name, path=None, target=None):
        if name not in self.texts:
            return None
        origin = self.origins[name]
        is_pkg = os.path.basename(origin) == "__init__.py"
        return importlib.util.spec_from_file_location(
            name, origin, loader=_TextLoader(self.texts[name], origin),
            submodule_search_locations=[os.path.dirname(origin)] if is_pkg else None)


def refresh(files, prefixes, modules=None, max_modules=200, kept=None):
    """Re-import the edited modules from `files` ({module: new text}) and their importers.
    `kept`: {module: text} an importer is re-imported from instead of its file (the text an
    earlier rebuild made its base), so a refresh never reverts it to the disk text.
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
    graph = import_graph(prefixes, modules, texts={**(kept or {}), **files})
    names = dependents(set(files), graph)
    if len(names) > max_modules:
        return {"ok": False, "refreshed": [], "old": {}, "seconds": 0.0, "findings": [finding(
            "refresh_too_wide", f"{len(names)} modules import the edited ones",
            f"at most {max_modules} (an edit in the package core: restart the worker)")]}
    order = [n for n in list(modules) if n in names]          # first-import order
    old = {n: modules[n] for n in order}
    texts = {**{n: (kept or {})[n] for n in order if n in (kept or {})}, **files}
    origins = {n: old[n].__file__ for n in texts}
    hooks.purge_modules(order, modules)
    finder = TextFinder(texts, origins)
    sys.meta_path.insert(0, finder)
    try:
        for n in order:
            if n not in modules:
                importlib.import_module(n)
    except BaseException as e:
        restore(old, modules, clear_linecache=False)
        return {"ok": False, "refreshed": [], "old": {}, "seconds": round(time.perf_counter() - t0, 4),
                "findings": [finding("refresh_failed", f"{type(e).__name__}: {e}",
                                     "the edited modules import")]}
    finally:
        sys.meta_path.remove(finder)
    for n, text in texts.items():                             # what inspect/Triton/Dynamo read
        f = origins[n]
        linecache.cache[f] = (len(text), None, text.splitlines(True), f)
    return {"ok": True, "refreshed": order, "old": old, "findings": [],
            "seconds": round(time.perf_counter() - t0, 4)}


def restore(old_modules, modules=None, clear_linecache: bool = True):
    """Put the modules `refresh` replaced back (an edit refused after its refresh)."""
    modules = sys.modules if modules is None else modules
    for n, m in old_modules.items():
        modules[n] = m
        parent, _, child = n.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, m)
        if clear_linecache:
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


# -- 1b. state the old modules still hold --------------------------------------------------------
def _weigh(v, seen, depth=0):
    """(tensor bytes, CUDA bytes, CUDA graphs) reachable from v through containers and plain
    objects (bounded depth)."""
    if id(v) in seen or depth > 4:
        return 0, 0, 0
    seen.add(id(v))
    torch = sys.modules.get("torch")    # a process that never imported torch holds no tensors
    if torch is not None:
        if isinstance(v, torch.Tensor):
            try:
                n = v.untyped_storage().nbytes()
            except Exception:
                n = v.numel() * v.element_size()
            return n, (n if v.is_cuda else 0), 0
        if type(v).__name__ == "CUDAGraph":
            return 0, 0, 1
        if isinstance(v, torch.nn.Module):
            return 0, 0, 0              # live components are not module state of the old code
    if isinstance(v, (str, bytes, int, float, bool, type(None), type)) or callable(v) and not hasattr(v, "__dict__"):
        return 0, 0, 0
    items = ()
    if isinstance(v, dict):
        items = list(v.values())
    elif isinstance(v, (list, tuple, set, frozenset)):
        items = list(v)
    elif hasattr(v, "__dict__") and not isinstance(v, types.ModuleType) and not callable(v):
        items = list(vars(v).values())
    t = c = g = 0
    for x in items:
        a, b, d = _weigh(x, seen, depth + 1)
        t, c, g = t + a, c + b, g + d
    return t, c, g


def held_state(old_modules):
    """State the replaced module objects still hold that the new code will build again: function
    attributes (caches kept on a function, such as per-layer CUDA graphs) and module globals
    that reach tensors or CUDA graphs. [{"where", "kind", "bytes", "cuda_bytes", "graphs"}]."""
    out = []
    for name, om in old_modules.items():
        new = vars(sys.modules.get(name) or types.ModuleType(name))
        for k, v in list(vars(om).items()):
            if k.startswith("__") or isinstance(v, types.ModuleType) or new.get(k) is v:
                continue            # shared with the new module (imported from a kept one)
            fn = getattr(v, "__func__", v)
            if isinstance(fn, types.FunctionType) and fn.__module__ == name:
                for a, av in list(vars(fn).items()):
                    if a.startswith("__"):
                        continue
                    t, c, g = _weigh(av, set())
                    if t or g:
                        out.append({"where": f"{name}.{k}.{a}", "kind": "function_attribute",
                                    "obj": (fn, a), "bytes": t, "cuda_bytes": c, "graphs": g})
            elif not callable(v) or isinstance(v, (dict, list)):
                t, c, g = _weigh(v, set())
                if t or g:
                    out.append({"where": f"{name}.{k}", "kind": "module_global", "obj": (om, k),
                                "bytes": t, "cuda_bytes": c, "graphs": g})
    return out


def release_held(old_modules):
    """Release what `held_state` finds, before the new code's first call: a function attribute
    is deleted, a container global emptied. The old code captured it, so it is never carried
    over. A held object that is neither (a bare tensor or graph global) is a finding, not
    released silently. Then gc and `torch.cuda.empty_cache()`. Returns {"released", "findings",
    "cuda_bytes_released", "graphs_released"}."""
    import gc
    held = held_state(old_modules)
    released, fs = [], []
    for h in held:
        owner, attr = h.pop("obj")
        if h["kind"] == "function_attribute":
            try:
                delattr(owner, attr)
                released.append(h)
            except AttributeError:
                fs.append(finding("held_state_not_released", f"{h['where']}: cannot delete", "released"))
        else:
            v = vars(owner).get(attr)
            if isinstance(v, (dict, list, set)):
                v.clear()
                released.append(h)
            else:
                fs.append(finding("held_state_not_released",
                                  f"{h['where']}: {type(v).__name__} holding {h['bytes']} bytes ({h['cuda_bytes']} on CUDA), "
                                  f"{h['graphs']} CUDA graphs", "state the refreshed code builds again, released"))
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return {"released": released, "findings": fs,
            "cuda_bytes_released": sum(h["cuda_bytes"] for h in released),
            "graphs_released": sum(h["graphs"] for h in released)}


# -- 2. class swap ----------------------------------------------------------------------------
def _set_class(obj, cls):
    obj.__class__ = cls


def swap_classes(objs, cmap, undo=None):
    """Set the new class on every object in `objs` whose class was refreshed. Returns
    {"swapped": n, "findings": [...]}. `undo`: a list that gets one callable per swap, which
    puts the old class back (run them in reverse to roll back)."""
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
            old = type(o)
            o.__class__ = new
            if undo is not None:
                undo.append(lambda obj=o, old=old: _set_class(obj, old))
            n += 1
        except TypeError as e:
            fs.append(finding("class_swap_failed", f"{type(o).__qualname__}: {e}",
                              "a class with the same layout"))
    return {"swapped": n, "findings": fs}


# -- 3. constructor edits ---------------------------------------------------------------------
def _same(a, b, depth=0):
    """Is the constructed value unchanged between the old and the new constructor?"""
    if type(a) is not type(b):
        return False
    if depth > 4:
        return True
    if isinstance(a, _PRIMITIVE) or type(a).__module__ == "torch" and not _is_tensor(a):
        try:
            return bool(a == b)
        except Exception:
            return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(left, right, depth + 1) for left, right in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[key], b[key], depth + 1) for key in a)
    import enum
    if isinstance(a, enum.Enum):
        return a is b
    if _is_tensor(a) or callable(a) or isinstance(a, (type, types.ModuleType)):
        return True   # tensors cannot come from the meta build; code is swapped, not merged
    if hasattr(a, "__dict__"):
        # a plain object (a config): the same when its fields are
        return _same(vars(a), vars(b), depth + 1)
    return True       # no fields to compare (an impl without state): the live one stays


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


def merge_fields(live, base, new, dry_run=False, undo=None):
    """Three-way merge of a constructor edit into the live module tree (see module doc).
    `live`, `base`, `new`: torch modules (base/new on meta). Returns {"ok", "set": [paths],
    "removed": [paths], "swapped": n, "findings"}; on a finding nothing is changed.
    `dry_run`: plan only. `undo`: a list that gets one callable per change, which takes it back
    (run them in reverse to roll back)."""
    plan, fs = [], []
    live_mods = dict(live.named_modules(remove_duplicate=False))
    base_mods = dict(base.named_modules(remove_duplicate=False))
    for path, nm in new.named_modules(remove_duplicate=False):
        lm, bm = live_mods.get(path), base_mods.get(path)
        if lm is None:
            if _holds_tensors(nm):
                fs.append(finding("new_submodule", f"{path or '<root>'}: {type(nm).__name__} holds tensors",
                                  "no new weights (needs a restart that loads them)"))
            else:
                plan.append(("module", path, nm))
            continue
        for kind in ("_parameters", "_buffers"):
            lt, nt = getattr(lm, kind), getattr(nm, kind)
            for k, v in nt.items():
                if k not in lt:
                    fs.append(finding("new_tensor", f"{path}.{k}", "no new weights (needs a restart that loads them)"))
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
        for name in bd:
            # a field the old constructor set and the new one does not: a fresh build lacks it
            if name not in _SKIP and name not in nm.__dict__ and name in lm.__dict__:
                plan.append(("delete", path, name))
    if fs or dry_run:
        return {"ok": not fs, "planned": len(plan), "set": [], "removed": [], "swapped": 0,
                "findings": fs}
    undo = [] if undo is None else undo
    done, removed, swapped = [], [], 0
    for kind, path, v in plan:
        if kind == "module":
            parent, _, child = path.rpartition(".")
            children = live.get_submodule(parent)._modules
            had, prior = child in children, children.get(child)
            live.get_submodule(parent).add_module(child, v)
            undo.append(lambda table=children, key=child, prior=prior, had=had:
                        _put_back(table, key, prior, had))
            done.append(path)
        elif kind == "class":
            old = type(live_mods[path])
            live_mods[path].__class__ = v
            undo.append(lambda obj=live_mods[path], old=old: _set_class(obj, old))
            swapped += 1
        else:
            fields = live_mods[path].__dict__
            name = v if kind == "delete" else v[0]
            had, prior = name in fields, fields.get(name)
            if kind == "delete":
                del fields[name]
                removed.append(f"{path}.{name}" if path else name)
            else:
                object.__setattr__(live_mods[path], name, v[1])
                done.append(f"{path}.{name}" if path else name)
            undo.append(lambda table=fields, key=name, prior=prior, had=had:
                        _put_back(table, key, prior, had))
    return {"ok": True, "set": done, "removed": removed, "swapped": swapped, "findings": []}


def _put_back(table, key, prior, had):
    """Undo of one change to a dict (a module's fields or children): the prior entry, or none."""
    if had:
        table[key] = prior
    else:
        table.pop(key, None)
