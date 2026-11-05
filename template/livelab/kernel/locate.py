"""Locate: from (module, qualname, first line) to the live function objects and their holders.

Code identity is (file, co_qualname, co_firstlineno) plus a content hash. For a decorated
function CPython's co_firstlineno is the line of the first decorator.

`content_hash(code)` is the one hash function for code. It is sha256 over the bytecode, the
names, local/free/cell variable names, argument counts, flags and constants, with nested code
objects hashed recursively and frozenset constants sorted (a .pyc keeps the set order of the
process that wrote it). It leaves out the file, the qualname and line numbers, which form the
identity, so moving a function does not change its hash. First 16 hex digits.
`text_hash(s)` is sha256 of UTF-8 text, 16 hex digits, for sources and plans.

The live functions of a module are found by scanning the gc heap (as
`fastrsi_lab_cov.install_batch` does): every function object whose globals are the module's
dict. Swapping `__code__` on those objects reaches every holder at once: module dicts (also
`from x import f` copies), class dicts, closure cells of decorator wrappers, bound methods
saved as callbacks. `holders(fn)` lists them for the record.
"""
import gc
import hashlib
import importlib
import sys
import types


def _const_repr(c):
    if isinstance(c, frozenset):
        return "frozenset(" + repr(sorted(map(_const_repr, c))) + ")"
    if isinstance(c, tuple):
        return "(" + ",".join(map(_const_repr, c)) + ")"
    return repr(c)


def content_hash(code):
    h = hashlib.sha256(code.co_code)
    for part in (code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
                 (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
                  code.co_flags)):
        h.update(repr(part).encode())
    for c in code.co_consts:
        h.update((content_hash(c) if isinstance(c, types.CodeType) else _const_repr(c)).encode())
    return h.hexdigest()[:16]


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def code_id(code):
    return {"file": code.co_filename, "qualname": code.co_qualname,
            "firstlineno": code.co_firstlineno, "hash": content_hash(code)}


def key_of(module, qualname, firstlineno):
    """The witness and result key of a target."""
    return f"{module}:{qualname}:{firstlineno}"


def code_tree(code, parent=None, out=None):
    """({(qualname, firstlineno): [code]}, {code: enclosing code}) for a code object tree."""
    if out is None:
        out = ({}, {})
    lst = out[0].setdefault((code.co_qualname, code.co_firstlineno), [])
    if code not in lst:
        lst.append(code)
    if parent is not None:
        out[1][code] = parent
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            code_tree(c, code, out)
    return out


class unfrozen:
    """Engines may call gc.freeze() after startup (vLLM does; some interpreters do at start),
    and both gc.get_objects() and gc.get_referrers() skip frozen objects: without this, the
    scan sees almost nothing. Unfreeze for the scan, freeze everything again after (as after
    startup)."""

    def __enter__(self):
        self.frozen = gc.get_freeze_count()
        if self.frozen:
            gc.unfreeze()
        return self

    def __exit__(self, *exc):
        if self.frozen:
            gc.freeze()


def heap_functions():
    """Every function object in the process."""
    with unfrozen():
        return [o for o in gc.get_objects() if type(o) is types.FunctionType]


def index_module(mod, functions=None):
    """For one module: (qualname, line) -> [running code], and code -> [function objects].
    A function belongs to the module whose dict is its globals, so two copies of the same file
    loaded under different names stay apart."""
    g = mod.__dict__
    run, holders = ({}, {}), {}
    for fn in (heap_functions() if functions is None else functions):
        if fn.__globals__ is g:
            holders.setdefault(fn.__code__, []).append(fn)
            code_tree(fn.__code__, None, run)
    return run[0], holders


def _module_of_dict(d):
    for name, m in list(sys.modules.items()):
        if getattr(m, "__dict__", None) is d:
            return name
    return None


def holders(fn, ignore=()):
    """Who references this function object: [{"kind", "where"}]. Kinds: module_dict,
    class_dict, closure_cell (a decorator wrapper or another closure), wrapped_by
    (functools.wraps __wrapped__), bound_method, attribute (an instance attribute),
    container, other."""
    with unfrozen():
        return _holders(fn, {id(x) for x in ignore})


def _holders(fn, ignore):
    out = []
    refs = gc.get_referrers(fn)
    for r in refs:
        if isinstance(r, types.FrameType) or r is refs or id(r) in ignore:
            continue
        if isinstance(r, types.MethodType):
            out.append({"kind": "bound_method", "where": f"{type(r.__self__).__name__}()."
                                                         f"{fn.__name__}"})
        elif isinstance(r, types.CellType):
            for t in gc.get_referrers(r):
                for w in (gc.get_referrers(t) if isinstance(t, tuple) else ()):
                    if isinstance(w, types.FunctionType) and w.__closure__ is t:
                        out.append({"kind": "closure_cell", "where": w.__code__.co_qualname})
        elif isinstance(r, dict):
            name = _module_of_dict(r)
            keys = [k for k, v in r.items() if v is fn]
            if name is not None:
                out += [{"kind": "module_dict", "where": f"{name}.{k}"} for k in keys]
                continue
            owner = None
            for o in gc.get_referrers(r):
                if isinstance(o, type) and any(v is fn for v in vars(o).values()):
                    owner = ("class_dict", o.__qualname__)
                elif isinstance(o, types.FunctionType) and o.__dict__ is r:
                    owner = ("wrapped_by", o.__code__.co_qualname)
                elif not isinstance(o, (type, types.FrameType, list)) and \
                        getattr(o, "__dict__", None) is r:
                    owner = ("attribute", type(o).__name__ + "()")
                if owner:
                    break
            kind, where = owner or ("container", "dict")
            out += [{"kind": kind, "where": f"{where}.{k}"} for k in keys]
        elif isinstance(r, (list, tuple, set)):
            out.append({"kind": "container", "where": type(r).__name__})
        elif any(v is fn for v in getattr(r, "__dict__", {}).values()):
            keys = [k for k, v in vars(r).items() if v is fn]
            out += [{"kind": "attribute", "where": f"{type(r).__name__}().{k}"} for k in keys]
        else:
            out.append({"kind": "other", "where": type(r).__name__})
    return out


def locate(module, qualname, firstlineno):
    """Resolve a target to its running code objects and the function objects that hold them.
    Returns {"ok", "key", "codes": [code_id], "functions": [function objects],
    "holders": [{"kind", "where"}], "findings"}. Never raises."""
    from . import findings as F
    key = key_of(module, qualname, firstlineno)
    try:
        mod = importlib.import_module(module)
    except Exception as e:
        return F.not_ok(f"import {module}: {type(e).__name__}: {e}", [F.finding(
            "patch_invalid", f"import failed: {e}", "an importable module", subject=key)],
            key=key)
    run, held = index_module(mod)
    codes = run.get((qualname, int(firstlineno)), [])
    fns = [f for c in codes for f in held.get(c, [])]
    if not fns:
        return F.not_ok("no live function", [F.finding(
            "no_live_function", f"{len(codes)} code object(s), no function object holds them",
            "a function object running this code", subject=key)], key=key,
            codes=[code_id(c) for c in codes], functions=[], holders=[])
    hs = [h for f in fns for h in holders(f, ignore=[fns, *held.values()])]
    return F.ok(key=key, codes=[code_id(c) for c in codes], functions=fns, holders=hs)
