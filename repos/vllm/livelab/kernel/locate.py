"""Locate: from (module, qualname, first line) to the live function objects and their holders.

Code identity is (file, co_qualname, co_firstlineno) plus a content hash. For a decorated
function CPython's co_firstlineno is the line of the first decorator.

`content_hash(code)` is the one hash function for code. It is sha256 over the bytecode, the
exception table (3.11+: which instructions a try/with covers, by offset), the names,
local/free/cell variable names, argument counts, flags and constants, with nested code
objects hashed recursively and frozenset constants sorted (a .pyc keeps the set order of the
process that wrote it). Each part is tagged and length-prefixed, so (1, 23) and (12, 3) differ.
It leaves out the file, the qualname and line numbers, which form the identity, so moving a
function does not change its hash. First 16 hex digits.
`text_hash(s)` is sha256 of UTF-8 text, 16 hex digits, for sources and plans.

The live functions of a module are found by scanning the gc heap: every function object whose globals are the module's
dict. Swapping `__code__` on those objects reaches every holder at once: module dicts (also
`from x import f` copies), class dicts, closure cells of decorator wrappers, bound methods
saved as callbacks. `holders(fn)` lists them for the record.
"""
import functools
import gc
import hashlib
import importlib
import sys
import threading
import time
import types


def _const_repr(c):
    if isinstance(c, frozenset):
        return "frozenset(" + repr(sorted(map(_const_repr, c))) + ")"
    if isinstance(c, tuple):
        return "(" + ",".join(map(_const_repr, c)) + ")"
    return repr(c)


def content_hash(code):
    h = hashlib.sha256()

    def put(tag, data):
        h.update(tag + len(data).to_bytes(8, "little") + data)
    put(b"c", code.co_code)
    put(b"x", getattr(code, "co_exceptiontable", b""))
    for part in (code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
                 (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
                  code.co_flags)):
        put(b"p", repr(part).encode())
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            put(b"n", content_hash(c).encode())
        else:
            put(b"k", _const_repr(c).encode())
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


_scan_lock = threading.RLock()
_scan = {"depth": 0, "frozen": 0, "gc_on": False}


class unfrozen:
    """Engines may call gc.freeze() after startup (vLLM does; some interpreters do at start),
    and both gc.get_objects() and gc.get_referrers() skip frozen objects: without this, the
    scan sees almost nothing. Unfreeze for the scan, freeze everything again after (as after
    startup). One window at a time in the process (a lock; nested use in one thread keeps
    the outer window): a second thread's scan never has the heap frozen again under it. While
    the heap is unfrozen, automatic collection is off, so no full collection of it can start
    in the window (it would stall the engine). Freezing again also freezes what was made
    after startup: witness.live_codes does not see suspended generators made before a scan
    (#42)."""

    def __enter__(self):
        _scan_lock.acquire()
        try:
            if _scan["depth"] == 0:
                _scan["frozen"] = gc.get_freeze_count()
                if _scan["frozen"]:
                    _scan["gc_on"] = gc.isenabled()
                    gc.disable()
                    gc.unfreeze()
            _scan["depth"] += 1
        except BaseException:
            _scan_lock.release()
            raise
        return self

    def __exit__(self, *exc):
        try:
            _scan["depth"] -= 1
            if _scan["depth"] == 0 and _scan["frozen"]:
                gc.freeze()
                if _scan["gc_on"]:
                    gc.enable()
        finally:
            _scan_lock.release()


SLICE_S = 0.002     # longest stretch a scan keeps the GIL before it sleeps (see `sliced`)


def all_objects():
    """Every gc-tracked object. `gc.get_objects()` is one C call that holds the GIL for its
    whole run (34 ms for 0.86 M objects in a process that imported vLLM's worker; more on a
    loaded engine): the only part of a scan that cannot be sliced. The heap is frozen again
    right after it, so no engine step runs with an unfrozen heap (a full collection of it
    would stall far longer)."""
    with unfrozen():
        return gc.get_objects()


def sliced(seq, budget_s=None):
    """Iterate seq, sleeping briefly after every `budget_s` of work so a busy loop in another
    thread gets the GIL. Without it a Python-level scan hands the GIL back only at the
    interpreter's switch interval (5 ms) each time the loop releases it."""
    budget_s = SLICE_S if budget_s is None else budget_s
    t = time.perf_counter()
    for i, x in enumerate(seq):
        yield x
        if not i & 1023 and time.perf_counter() - t > budget_s:
            time.sleep(0.0002)
            t = time.perf_counter()


def heap_functions():
    """Every function object in the process: one get_objects, then a sliced filter."""
    objs = all_objects()
    try:
        return [o for o in sliced(objs) if type(o) is types.FunctionType]
    finally:
        del objs


def module_functions(mod):
    """Function objects reachable from the module's own namespace, without a heap scan:
    module-level functions, methods of classes defined in the module (also static and class
    methods, properties, nested classes), decorator wrappers' originals (closure cells and
    `__wrapped__`), bound methods. Pure Python over the module, so it never holds the GIL
    long. Also the `.fn` of wrapper objects (Triton's @jit kernels and their autotune /
    heuristics wrappers). Swapping `__code__` on these objects reaches every holder of them (from-import
    copies, bound methods saved as callbacks). It misses closures made at run time and
    functions defined in the module but referenced only elsewhere: `stage` scans the heap
    for nested targets and for targets this does not find."""
    g, name = mod.__dict__, mod.__name__
    out, seen, stack = {}, set(), list(g.values())
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, types.FunctionType):
            if o.__globals__ is g:
                out[id(o)] = o
            for c in o.__closure__ or ():
                try:
                    stack.append(c.cell_contents)
                except ValueError:            # empty cell
                    pass
            if "__wrapped__" in o.__dict__:
                stack.append(o.__dict__["__wrapped__"])
        elif isinstance(o, (staticmethod, classmethod, types.MethodType)):
            stack.append(o.__func__)
        elif isinstance(o, property):
            stack += [f for f in (o.fget, o.fset, o.fdel) if f is not None]
        elif isinstance(o, functools.cached_property):
            stack.append(o.func)
        elif isinstance(o, type) and o.__module__ == name:
            stack += list(vars(o).values())
        elif not isinstance(o, (type, types.ModuleType)):
            # a wrapper object that keeps the function in `.fn` (a Triton @jit kernel, its
            # autotune / heuristics wrappers): found here, not by a heap scan
            # functools.update_wrapper on a non-function (lru_cache, a callable class)
            # keeps the original in `__wrapped__`
            d = _dict_of(o)
            for k in ("fn", "__wrapped__"):
                if d is not None and k in d:
                    stack.append(d[k])
    return list(out.values())


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


def _dict_of(o):
    """o's instance __dict__, or None; never runs o's own __getattr__ / __getattribute__ (an
    RPC or lazy proxy may raise anything from them)."""
    try:
        d = object.__getattribute__(o, "__dict__")
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def _module_of_dict(d):
    for name, m in list(sys.modules.items()):
        if _dict_of(m) is d:
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
    ref_ids = {id(r) for r in refs}
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
                elif not isinstance(o, (type, types.FrameType, list)) and _dict_of(o) is r:
                    owner = ("attribute", type(o).__name__ + "()")
                if owner:
                    break
            kind, where = owner or ("container", "dict")
            out += [{"kind": kind, "where": f"{where}.{k}"} for k in keys]
        elif isinstance(r, (list, tuple, set)):
            out.append({"kind": "container", "where": type(r).__name__})
        elif any(v is fn for v in (_dict_of(r) or {}).values()):
            if id(_dict_of(r)) in ref_ids:
                continue     # its __dict__ is a referrer too, listed above (a C wrapper)
            keys = [k for k, v in _dict_of(r).items() if v is fn]
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
        line = int(firstlineno)
    except Exception as e:
        return F.not_ok(f"import {module}: {type(e).__name__}: {e}", [F.finding(
            "patch_invalid", f"{type(e).__name__}: {e}", "an importable module and a line "
            "number", subject=key)], key=key, codes=[], functions=[], holders=[])
    run, held = index_module(mod)
    codes = run.get((qualname, line), [])
    fns = [f for c in codes for f in held.get(c, [])]
    if not fns:
        return F.not_ok("no live function", [F.finding(
            "no_live_function", f"{len(codes)} code object(s), no function object holds them",
            "a function object running this code", subject=key)], key=key,
            codes=[code_id(c) for c in codes], functions=[], holders=[])
    try:
        hs = [h for f in fns for h in holders(f, ignore=[fns, *held.values()])]
    except Exception as e:        # a referrer this walk does not know: the record says so
        hs = [{"kind": "other", "where": f"holders walk failed: {type(e).__name__}: {e}"}]
    return F.ok(key=key, codes=[code_id(c) for c in codes], functions=fns, holders=hs)
