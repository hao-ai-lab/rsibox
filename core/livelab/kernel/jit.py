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


def scan():
    """Every rebuildable JIT callable the loaded modules hold at top level, and every
    wrapper with a `cache` (autotuner) over one: ({id: callable}, [(wrapper, callable)])."""
    jits, tuners, seen = {}, [], set()
    for mod in list(sys.modules.values()):
        d = _dict_of(mod)
        if d is None:
            continue
        for v in list(d.values()):
            if id(v) in seen or not _jit_like(type(v)):
                continue
            seen.add(id(v))
            jf, chain = unwrap(v)
            if jf is None or missing(jf):
                continue
            jits[id(jf)] = jf
            tuners += [(w, jf) for w in chain if isinstance(getattr(w, "cache", None), dict)]
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


def callers_of(jf, jits):
    """The JIT callables that call jf, at any depth (breadth first, no repeats)."""
    out, seen, todo = [], {id(jf)}, [jf]
    while todo:
        f = todo.pop(0)
        for c in jits.values():
            if id(c) not in seen and calls(c, f):
                seen.add(id(c))
                out.append(c)
                todo.append(c)
    return out


def plan(mod, q, text):
    """Live half: {"ok": True, "plan": {...}} or {"ok": False, "why": plain words}."""
    t0 = time.perf_counter()
    obj = mod
    for part in q.split("."):
        d = _dict_of(obj) if not isinstance(obj, type) else dict(vars(obj))
        obj = d.get(part) if d is not None else None
    jf, chain = unwrap(obj) if obj is not None else (None, [])
    if jf is None:
        return {"ok": False, "why": "no JIT object holding the function was found"}
    lack = missing(jf)
    if lack:
        return {"ok": False, "why": f"its JIT object ({type(jf).__name__}) cannot be rebuilt "
                f"in place (it has no {', '.join(lack)})"}
    if text is None or text["guard"]:
        return {"ok": False, "why": text["guard"] if text else "no text of the kernel"}
    jits, tuners = scan()
    jits[id(jf)] = jf
    callers = callers_of(jf, jits)
    affected = [jf] + callers
    ids = {id(f) for f in affected}
    tuners = [w for w, f in tuners if id(f) in ids]
    tuners += [w for w in chain if isinstance(getattr(w, "cache", None), dict)
               and all(w is not x for x in tuners)]
    return {"ok": True, "plan": {
        "jf": jf, "q": q, "text": text, "affected": affected, "tuners": tuners,
        "callers": [f"{c.__module__}.{c.fn.__qualname__}" for c in callers],
        "scanned": len(jits), "plan_s": round(time.perf_counter() - t0, 6)}}


def _caches(dc):
    return {d: (dict(v[0]), dict(v[1])) + tuple(v[2:]) for d, v in dc.items()}


def _hook_into(fns, hook, journal, label):
    for f in fns:
        f.pre_run_hooks.append(hook)
        journal.record_call(lambda f=f: f.pre_run_hooks.remove(hook), f"unhook {label}")


def apply(p, journal, witness, key):
    """Rebuild in place, every change journaled (undo: no recompile). Raises on error; the
    caller (swap.apply) undoes back to its mark."""
    jf, text = p["jf"], p["text"]
    old = (jf._src, list(jf.raw_src), jf.starting_line_number,
           getattr(jf, "def_file_line_number", None))
    states = [(f, f.hash, dict(f.used_global_vals), _caches(f.device_caches))
              for f in p["affected"]]
    tuned = [(w, dict(w.cache)) for w in p["tuners"]]

    def restore():
        _set_text(jf, *old)
        for f, h, used, dc in states:
            f.hash, f.used_global_vals = h, dict(used)
            f.device_caches.clear()
            f.device_caches.update(_caches(dc))
        for w, c in tuned:
            w.cache.clear()
            w.cache.update(c)
    journal.record_call(restore, f"JIT rebuild of {p['q']}")
    _set_text(jf, text["src"], list(text["raw_src"]), text["start"], text["def_line"])
    for f in p["affected"]:
        invalidate(f)
    for w in p["tuners"]:
        w.cache.clear()
    _hook_into(p["affected"], witness.arm(key, journal), journal, key)


def _set_text(jf, src, raw, start, def_line):
    jf._unsafe_update_src(src)
    jf.raw_src, jf.starting_line_number = raw, start
    if def_line is not None and hasattr(jf, "def_file_line_number"):
        jf.def_file_line_number = def_line


def invalidate(f):
    """All three: Triton's dependency walk reads a callee's memoised hash and global values."""
    f.hash = None
    f.used_global_vals.clear()
    f.device_caches.clear()


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
