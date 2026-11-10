"""JIT rebuild: edit a Triton kernel in place, so the next launch compiles the edited text.

A Triton kernel runs what the JIT compiled from its source text (`_src`), never the Python
function's bytecode, so a code swap alone leaves the GPU on the old kernel. `swap.stage` asks
`plan` for every target held by a JIT object; `swap.apply` calls `apply`. Pure standard
library: Triton's objects are duck-typed by the attributes used here (`NEED`), so CPU CI runs
with no `triton` and real Triton 3.5+ works the same.

What a rebuild must reset (Triton 3.5 / 3.8 `runtime/jit.py`):
- `_src`, `raw_src`, `starting_line_number` of the edited kernel: the text half (`text_of`)
  cuts them from the clean patched text (never from the AST that carries the witness call),
  from the first decorator line, as `inspect.getsourcelines` would, so compile errors point
  at the right lines.
- `hash` (memoised `cache_key`, which folds in every callee's key), `used_global_vals` and
  `device_caches` (compiled kernels keyed by specialization, never by text) of the kernel AND
  of every JIT function that calls it, at any depth, in any loaded module: a caller left with
  its old `hash` loads the old binary from the disk cache, one left with its device cache
  keeps launching it. `Autotuner.cache` of every autotuner over them is cleared too.
- Everything is snapshotted first and journaled (`record_call`): undo puts back the old
  text, hashes, global values, compiled kernels and tuning choices, so the base kernels run
  again with no recompile.

Refused (`plan` gives a reason; swap reports a blocking `jit_target`): a JIT object that
lacks an attribute of `NEED` (Triton's interpreter mode, other JITs), and an edit that changes
the kernel's parameters or decorators, which Triton fixed when it built the objects (`params`,
`constexprs`, `do_not_specialize`; the autotuner's `configs` and `keys`).

Witness: Triton runs `pre_run_hooks` on every launch of a JIT function, and a helper inlined
into a caller runs only the caller's hooks. So the target's witness counter is appended to the
hooks of the kernel and of every caller found above (journaled: undo removes it). A kernel
that never launches, or that only replays inside a graph captured before the edit, reads 0.
"""
import ast
import linecache
import re
import sys
import textwrap
import time
import types

# What a JIT callable must have to be rebuilt in place (Triton's JITFunction does).
NEED = ("_unsafe_update_src", "device_caches", "pre_run_hooks", "parse", "signature",
        "raw_src", "starting_line_number", "used_global_vals")
_DEF_RE = re.compile(r"^def\s+\w+\s*\(", re.MULTILINE)


def _dict_of(o):
    try:
        d = object.__getattribute__(o, "__dict__")
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def unwrap(obj):
    """(the JIT callable whose `.fn` is the Python function, [wrappers above it, outermost
    first]); (None, chain) when `.fn` does not lead to a function within 8 hops."""
    chain = []
    for _ in range(8):
        d = _dict_of(obj)
        fn = d.get("fn") if d is not None else None
        if isinstance(fn, types.FunctionType):
            return obj, chain
        if fn is None:
            return None, chain
        chain.append(obj)
        obj = fn
    return None, chain


def missing(jf):
    """The attributes of `NEED` the JIT callable lacks (empty: rebuildable)."""
    return [a for a in NEED if not hasattr(jf, a)]


def text_of(lines, base, node):
    """Text half, for a decorated target: what the rebuild sets, cut from the patched text
    (`lines`: its lines with ends), and whether the edit changed what Triton fixed at build.
    Returns plain data: {"src", "raw_src", "start", "def_line", "guard"}."""
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    raw = lines[start - 1:node.end_lineno]
    src = textwrap.dedent("".join(raw))
    m = _DEF_RE.search(src)
    guard = None
    if ast.dump(base.args) != ast.dump(node.args) or \
            ast.dump(base.returns or ast.Pass()) != ast.dump(node.returns or ast.Pass()):
        guard = ("the edit changes its parameters (names, defaults or constexpr "
                 "annotations), which Triton fixed when it built the JIT object")
    elif ast.dump(ast.Tuple(base.decorator_list)) != ast.dump(ast.Tuple(node.decorator_list)):
        guard = ("the edit changes its decorators (JIT options, autotune configs or keys, "
                 "heuristics), which Triton fixed when it built the wrapper objects")
    return {"src": src[m.start():] if m else src, "raw_src": raw, "start": start,
            "def_line": node.lineno, "guard": guard}


_kinds = {}


def _jit_like(t):
    """Can an object of type t be a JIT callable or a wrapper of one? (cached per type)"""
    k = _kinds.get(t)
    if k is None:
        try:
            k = not issubclass(t, (type, types.ModuleType, types.FunctionType)) and (
                hasattr(t, "_unsafe_update_src") or hasattr(t, "run"))
        except Exception:                    # a type whose attribute lookup raises
            k = False
        _kinds[t] = k
    return k


def scan(modules=None):
    """Every rebuildable JIT callable the loaded modules hold at top level, and every
    wrapper with a `cache` (autotuner) over one: ({id: callable}, [(wrapper, callable)])."""
    if modules is None:
        module_list = list(sys.modules.values())
    elif isinstance(modules, dict):
        module_list = list(modules.values())
    else:
        module_list = list(modules)
    jits, tuners, seen = {}, [], set()
    for mod in module_list:
        mod_dict = _dict_of(mod)
        if mod_dict is None:
            continue
        for val in list(mod_dict.values()):
            if id(val) in seen or not _jit_like(type(val)):
                continue
            seen.add(id(val))
            jit_fn, chain = unwrap(val)
            if jit_fn is None or missing(jit_fn):
                continue
            jits[id(jit_fn)] = jit_fn
            tuners += [
                (wrapper, jit_fn)
                for wrapper in chain
                if isinstance(getattr(wrapper, "cache", None), dict)
            ]
    return jits, tuners


def _resolve(node, scope):
    if isinstance(node, ast.Name):
        return scope.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, scope)
        return getattr(base, node.attr, None) if base is not None else None
    return None


def calls(caller, callee):
    """Does the caller's text name the callee (through its globals or closure)?"""
    name = callee.fn.__name__
    if name not in caller._src:              # cheap text filter before the parse
        return False
    scope = dict(caller.__globals__)
    fn = caller.fn
    for n, c in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            scope[n] = c.cell_contents
        except ValueError:
            pass
    for n in ast.walk(caller.parse()):
        if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(n.ctx, ast.Load):
            v = _resolve(n, scope)
            if v is callee or (v is not None and unwrap(v)[0] is callee):
                return True
    return False


def callers_of(jit_fn, jits):
    """The JIT callables that call jit_fn, at any depth (breadth first, no repeats)."""
    out, seen, todo = [], {id(jit_fn)}, [jit_fn]
    while todo:
        current = todo.pop(0)
        for candidate in jits.values():
            if id(candidate) not in seen and calls(candidate, current):
                seen.add(id(candidate))
                out.append(candidate)
                todo.append(candidate)
    return out


def _call_site_names(jit_fn):
    """Extract called identifier names from AST Call sites of a JIT callable."""
    called_names = set()
    try:
        parsed_tree = jit_fn.parse()
    except Exception:
        return called_names
    for node in ast.walk(parsed_tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
    return called_names


def discover_triton_callers(edited_jit_names, modules=None):
    """Discover transitive `@triton.jit` callers of `edited_jit_names` across `modules`."""
    if isinstance(edited_jit_names, str) or not hasattr(edited_jit_names, "__iter__"):
        target_items = [edited_jit_names]
    else:
        target_items = list(edited_jit_names)
    jits, _tuners = scan(modules=modules)
    target_names = set()
    seed_jits = []
    seen_seed_ids = set()
    for item in target_items:
        if isinstance(item, str):
            target_names.add(item)
            target_names.add(item.rsplit(".", 1)[-1])
        else:
            unwrapped_jit, _chain = unwrap(item)
            if unwrapped_jit is not None and not missing(unwrapped_jit):
                jits[id(unwrapped_jit)] = unwrapped_jit
                if id(unwrapped_jit) not in seen_seed_ids:
                    seen_seed_ids.add(id(unwrapped_jit))
                    seed_jits.append(unwrapped_jit)
                target_names.add(unwrapped_jit.fn.__name__)
                target_names.add(unwrapped_jit.fn.__qualname__)

    for candidate in list(jits.values()):
        qualname = candidate.fn.__qualname__
        full_name = f"{candidate.__module__}.{qualname}"
        if (
            candidate.fn.__name__ in target_names
            or qualname in target_names
            or full_name in target_names
        ) and id(candidate) not in seen_seed_ids:
            seen_seed_ids.add(id(candidate))
            seed_jits.append(candidate)

    discovered, seen_ids, queue = [], set(seen_seed_ids), list(seed_jits)
    if target_names:
        for candidate in jits.values():
            if id(candidate) not in seen_ids and (
                _call_site_names(candidate) & target_names
            ):
                seen_ids.add(id(candidate))
                discovered.append(candidate)
                queue.append(candidate)

    while queue:
        current = queue.pop(0)
        current_names = {current.fn.__name__, current.fn.__qualname__}
        for candidate in jits.values():
            if id(candidate) in seen_ids:
                continue
            if calls(candidate, current) or (_call_site_names(candidate) & current_names):
                seen_ids.add(id(candidate))
                discovered.append(candidate)
                queue.append(candidate)
    return discovered


def plan(mod, qualname, text):
    """Live half: {"ok": True, "plan": {...}} or {"ok": False, "why": plain words}."""
    start_time = time.perf_counter()
    obj = mod
    for part in qualname.split("."):
        obj_dict = _dict_of(obj) if not isinstance(obj, type) else dict(vars(obj))
        obj = obj_dict.get(part) if obj_dict is not None else None
    jit_fn, chain = unwrap(obj) if obj is not None else (None, [])
    if jit_fn is None:
        return {"ok": False, "why": "no JIT object holding the function was found"}
    lack = missing(jit_fn)
    if lack:
        return {"ok": False, "why": f"its JIT object ({type(jit_fn).__name__}) cannot be rebuilt "
                f"in place (it has no {', '.join(lack)})"}
    if text is None or text["guard"]:
        return {"ok": False, "why": text["guard"] if text else "no text of the kernel"}
    jits, tuners = scan()
    jits[id(jit_fn)] = jit_fn
    callers = callers_of(jit_fn, jits)
    seen_caller_ids = {id(jit_fn), *(id(caller) for caller in callers)}
    for extra_caller in discover_triton_callers([jit_fn, qualname]):
        if id(extra_caller) not in seen_caller_ids:
            seen_caller_ids.add(id(extra_caller))
            callers.append(extra_caller)
    affected = [jit_fn] + callers
    ids = {id(fn_item) for fn_item in affected}
    tuners = [wrapper for wrapper, fn_item in tuners if id(fn_item) in ids]
    tuners += [
        wrapper
        for wrapper in chain
        if isinstance(getattr(wrapper, "cache", None), dict)
        and all(wrapper is not existing for existing in tuners)
    ]
    return {"ok": True, "plan": {
        "jf": jit_fn, "q": qualname, "text": text, "affected": affected, "tuners": tuners,
        "callers": [f"{caller.__module__}.{caller.fn.__qualname__}" for caller in callers],
        "scanned": len(jits), "plan_s": round(time.perf_counter() - start_time, 6)}}


def _caches(device_caches):
    return {
        dev: (dict(val[0]), dict(val[1])) + tuple(val[2:])
        for dev, val in device_caches.items()
    }


def _hook_into(fns, hook, journal, label):
    for fn_item in fns:
        fn_item.pre_run_hooks.append(hook)
        journal.record_call(
            lambda fn_item=fn_item: fn_item.pre_run_hooks.remove(hook),
            f"unhook {label}",
        )


def apply(plan_data, journal, witness, key):
    """Rebuild in place, every change journaled (undo: no recompile). Raises on error; the
    caller (swap.apply) undoes back to its mark."""
    jit_fn, text = plan_data["jf"], plan_data["text"]
    old = (jit_fn._src, list(jit_fn.raw_src), jit_fn.starting_line_number,
           getattr(jit_fn, "def_file_line_number", None))
    states = [
        (
            fn_item,
            fn_item.hash,
            dict(fn_item.used_global_vals),
            _caches(fn_item.device_caches),
            dict(fn_item.cache)
            if isinstance(getattr(fn_item, "cache", None), dict)
            else None,
        )
        for fn_item in plan_data["affected"]
    ]
    tuned = [(wrapper, dict(wrapper.cache)) for wrapper in plan_data["tuners"]]

    def restore():
        _set_text(jit_fn, *old)
        for fn_item, saved_hash, used_vals, dev_cache, direct_cache in states:
            fn_item.hash, fn_item.used_global_vals = saved_hash, dict(used_vals)
            fn_item.device_caches.clear()
            fn_item.device_caches.update(_caches(dev_cache))
            if direct_cache is not None and isinstance(
                getattr(fn_item, "cache", None), dict
            ):
                fn_item.cache.clear()
                fn_item.cache.update(direct_cache)
        for wrapper, cache_copy in tuned:
            wrapper.cache.clear()
            wrapper.cache.update(cache_copy)
    journal.record_call(restore, f"JIT rebuild of {plan_data['q']}")
    _set_text(jit_fn, text["src"], list(text["raw_src"]), text["start"], text["def_line"])
    for fn_item in plan_data["affected"]:
        invalidate(fn_item)
    for wrapper in plan_data["tuners"]:
        wrapper.cache.clear()
    _hook_into(plan_data["affected"], witness.arm(key, journal), journal, key)


def _set_text(jit_fn, src, raw, start, def_line):
    jit_fn._unsafe_update_src(src)
    jit_fn.raw_src, jit_fn.starting_line_number = raw, start
    if def_line is not None and hasattr(jit_fn, "def_file_line_number"):
        jit_fn.def_file_line_number = def_line


def invalidate(jit_fn):
    """All three: Triton's dependency walk reads a callee's memoised hash and global values."""
    jit_fn.hash = None
    jit_fn.used_global_vals.clear()
    jit_fn.device_caches.clear()
    if isinstance(getattr(jit_fn, "cache", None), dict):
        jit_fn.cache.clear()


_DYNAMO_ATTR_PREFIXES = ("_torchdynamo_", "_dynamo_")
_DYNAMO_EXACT_ATTRS = ("_compiled_call_impl", "__dynamo_cache__")
_DYNAMO_SKIP_ATTRS = ("_dynamo_marked_constant",)


def invalidate_compiled_frames(target_functions, journal=None):
    """Refresh `linecache` and selectively clear per-callable TorchDynamo cache attributes
    on `target_functions` without a full-process `torch._dynamo.reset()`.

    Returns a dictionary containing `invalidated`, `cleared_attributes`, and a `restore`
    callable (also recorded on `journal` when given).
    """
    if callable(target_functions) or not hasattr(target_functions, "__iter__"):
        functions = [target_functions]
    else:
        functions = list(target_functions)

    linecache.checkcache()
    saved_entries = []
    cleared_attributes = {}
    invalidated = []

    for func in functions:
        if func is None:
            continue
        code_obj = getattr(func, "__code__", None)
        if code_obj is not None and getattr(code_obj, "co_filename", None):
            linecache.checkcache(code_obj.co_filename)
        func_dict = _dict_of(func)
        if not func_dict:
            continue
        func_name = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
        cleared_for_func = []
        for attr_name in list(func_dict.keys()):
            if attr_name in _DYNAMO_SKIP_ATTRS:
                continue
            if not (
                attr_name in _DYNAMO_EXACT_ATTRS
                or attr_name.startswith(_DYNAMO_ATTR_PREFIXES)
            ):
                continue
            attr_val = func_dict[attr_name]
            cleared_for_func.append(attr_name)
            if isinstance(attr_val, dict):
                saved_entries.append(("dict", func, attr_name, dict(attr_val)))
                attr_val.clear()
            elif isinstance(attr_val, list):
                saved_entries.append(("list", func, attr_name, list(attr_val)))
                attr_val.clear()
            else:
                saved_entries.append(("attr", func, attr_name, attr_val))
                delattr(func, attr_name)
        if cleared_for_func:
            invalidated.append(func_name)
            cleared_attributes[func_name] = cleared_for_func

    def restore():
        for entry_kind, func, attr_name, saved_val in reversed(saved_entries):
            if entry_kind == "dict":
                current_dict = getattr(func, attr_name, None)
                if isinstance(current_dict, dict):
                    current_dict.clear()
                    current_dict.update(saved_val)
                else:
                    setattr(func, attr_name, dict(saved_val))
            elif entry_kind == "list":
                current_list = getattr(func, attr_name, None)
                if isinstance(current_list, list):
                    current_list.clear()
                    current_list.extend(saved_val)
                else:
                    setattr(func, attr_name, list(saved_val))
            else:
                setattr(func, attr_name, saved_val)

    if journal is not None:
        journal.record_call(restore, "restore invalidated compiled frames")

    return {
        "invalidated": invalidated,
        "cleared_attributes": cleared_attributes,
        "restore": restore,
    }


def arm_added(obj, key, plans, journal, witness):
    """A JIT kernel the edit adds: count its launches and those of the rebuilt kernels whose
    new text names it (they inline it). Returns how many JIT functions carry the hook."""
    jf, _ = unwrap(obj)
    if jf is None or missing(jf):
        return 0
    name = jf.fn.__name__
    seen, fns = {id(jf)}, [jf]
    for p in plans:
        if name in re.findall(r"\w+", p["text"]["src"]):
            for f in p["affected"]:
                if id(f) not in seen:
                    seen.add(id(f))
                    fns.append(f)
    _hook_into(fns, witness.arm(key, journal), journal, key)
    return len(fns)


class seeded_lines:
    """Context: linecache holds the patched text for `filename`, so `@triton.jit` on a
    definition the edit adds (`inspect.getsourcelines`) reads its lines, not the file's.
    The previous entry comes back on exit."""

    def __init__(self, filename, source):
        self.f, self.src = filename, source

    def __enter__(self):
        self.old = linecache.cache.get(self.f)
        linecache.cache[self.f] = (len(self.src), None, self.src.splitlines(True), self.f)

    def __exit__(self, *exc):
        if self.old is None:
            linecache.cache.pop(self.f, None)
        else:
            linecache.cache[self.f] = self.old
