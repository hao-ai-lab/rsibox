"""JIT targets: a swapped function that the engine runs through a JIT object, not as bytecode.

A Triton `@triton.jit` kernel is a `JITFunction` holding the plain Python function in `.fn`.
Triton compiles from the source TEXT it read when the object was built (`inspect.getsourcelines`)
and caches the compiled kernels per device; it never runs the function's bytecode. So swapping
`fn.__code__` (what the kernel does for every other target) leaves the GPU on the old kernel: a
live run committed such a swap, saw no finding at prepare, and proved it only by a misleading
`zero_witness`.

`holders(mod, fns)` finds the JIT objects in the module namespace (and inside wrappers that keep
them in `.fn`, as `@triton.autotune` and `@triton.heuristics` do) whose `.fn` is one of the
target's function objects. `rebuild(obj, source, filename)` runs the object's own `__init__`
again on the swapped function, reading the new text: the patch is put in `linecache` under the
module's file name only while `__init__` reads it. The object keeps its identity (every holder
sees the new kernel), its compiled-kernel caches are new, and its hash is recomputed, so the
next launch compiles the new source. `dependents(mod, name, skip)` are the module's other JIT
objects whose source calls `name(`: Triton inlines a called `@jit` function when it compiles
the caller, so their hashes and caches are reset too. `snapshot(obj)` returns the function that
puts the object's whole `__dict__` back: undo (journal `record_call`) restores the old compiled
kernels, not a recompile. `count_launches(obj, hook)` adds a pre-run hook (a new list, so the
snapshot keeps the old one): the witness of a JIT target is its launches.

Duck-typed: Triton is never imported here. An object is a JIT object when a class in its MRO is
named JITCallable or JITFunction and comes from a module under `triton`.
"""
import inspect
import linecache
import re
import types

JIT_CLASSES = ("JITCallable", "JITFunction")

# JITFunction.__init__ keyword -> the attribute that holds its value on a built object
_INIT_ATTRS = {"version": "version", "do_not_specialize": "do_not_specialize",
               "do_not_specialize_on_alignment": "do_not_specialize_on_alignment",
               "debug": "debug", "noinline": "noinline", "repr": "_repr",
               "launch_metadata": "launch_metadata"}


def is_jit(o):
    return any(c.__name__ in JIT_CLASSES and (c.__module__ or "").startswith("triton")
               for c in type(o).__mro__)


def kind(o):
    return f"{type(o).__module__}.{type(o).__name__}"


def _objects(mod):
    """The JIT objects of the module namespace: direct values, and the `.fn` chains of
    wrappers (autotune, heuristics), each once."""
    out, seen = [], set()
    for v in list(vars(mod).values()):
        o, depth = v, 0
        while o is not None and depth < 4 and id(o) not in seen \
                and not isinstance(o, types.ModuleType):
            seen.add(id(o))
            if is_jit(o):
                out.append(o)
            o = getattr(o, "__dict__", {}).get("fn") if not isinstance(o, type) else None
            depth += 1
    return out


def holders(mod, fns):
    """JIT objects of `mod` whose `.fn` is one of `fns` (by identity)."""
    ids = {id(f) for f in fns}
    return [o for o in _objects(mod) if id(vars(o).get("fn")) in ids]


def _src(o):
    return vars(o).get("_src") or vars(o).get("src") or ""


def dependents(mod, name, skip=()):
    """The module's other JIT objects whose source calls `name(` (Triton inlines it)."""
    skip = {id(o) for o in skip}
    pat = re.compile(r"(?<![\w.])" + re.escape(name) + r"\s*\(")
    return [o for o in _objects(mod) if id(o) not in skip and pat.search(_src(o))]


def snapshot(o):
    """A function that puts back the object's whole state as it is now."""
    snap = dict(vars(o))

    def restore():
        d = vars(o)
        d.clear()
        d.update(snap)
    return restore


def rebuild(o, source, filename):
    """Run the object's own __init__ again on its (swapped) `.fn`, reading `source` (the
    patched module text) as the file's text. Raises what __init__ raises."""
    fn = vars(o)["fn"]
    params = inspect.signature(type(o).__init__).parameters
    kw = {k: getattr(o, a) for k, a in _INIT_ATTRS.items() if k in params and hasattr(o, a)}
    hooks = list(vars(o).get("pre_run_hooks") or [])
    old = linecache.cache.get(filename)
    lines = source.splitlines(True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    try:
        type(o).__init__(o, fn, **kw)
    finally:
        if old is None:
            linecache.cache.pop(filename, None)
        else:
            linecache.cache[filename] = old
    if "pre_run_hooks" in vars(o):            # __init__ empties it; keep the engine's hooks
        o.pre_run_hooks = hooks


def reset(o):
    """Forget what the object compiled (hash, global-value checks, per-device caches), so its
    next launch compiles again against the functions it calls."""
    d = vars(o)
    if "hash" in d:
        o.hash = None
    if "used_global_vals" in d:
        o.used_global_vals = {}
    if "device_caches" in d and hasattr(o, "create_binder"):
        o.device_caches = type(d["device_caches"])(o.create_binder)


def count_launches(o, hook):
    """Add `hook` to the object's pre-run hooks (called at every launch), as a new list."""
    if "pre_run_hooks" in vars(o):
        o.pre_run_hooks = list(o.pre_run_hooks) + [hook]
        return True
    return False
