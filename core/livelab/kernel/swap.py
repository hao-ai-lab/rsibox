"""Swap: one code-swap primitive, in two steps.

`stage(module, source, targets)` parses the patched module source, compiles it, and plans
the swap, with no visible change (when the patch adds definitions it binds the no-op witness
stub, as an undo leaves it, so an added decorator can run). Targets omitted ("auto"):
the analysis takes every definition whose AST changed, from its own parse. For each target it
finds the new code object (by qualname and its ordinal
among same-named definitions, so a patch may move lines), adds the witness, and pairs every
running code object of the target and of its enclosing functions with its new version (so
closures made later run the new code too). It runs the checks and returns findings:
`no_live_function`, `not_in_patch`, `base_mismatch`, `outside_changes`, `closure_changed`
(free variables differ), `defaults_not_evaluated`, `running_frame`, `patch_invalid`,
`new_fields`, `jit_rebuild` (advisory: a Triton kernel, rebuilt in place by `jit.py`),
`jit_target` (a Triton kernel that cannot be rebuilt in place, so it blocks) and
`constructor_edit` (an `__init__`, `__new__` or `__post_init__` target: built instances keep
the old state, so it blocks and routes to T1; one that only changes fresh field values carries
a `reinit` plan the participant can carry out on the live instances, reinit.py).
New imports the patch needs are executed into a scratch namespace; new functions, methods and
classes are built in a scratch namespace (methods get the real class for `super()`). New
methods and new defaults of methods are evaluated in a scratch class body that reads the live
class's names (read-only; `_LiveBody`), so `def f(self, x=CONST)` with CONST in the class body,
or a decorator defined in the class, works as in the file; the live class is not touched
before commit.

`apply(staged, journal, witness)` makes it visible. Every change goes through the journal:
added imports and the witness hook (module dict), counter keys, `__code__`, `__defaults__`,
`__kwdefaults__` of every holding function object, added names. On an error it undoes itself.
The caller decides from the findings (findings.decide) whether to apply at all.

`stage` is split in two. `analyze` is the text half: it needs only the two texts, the
targets and which targets are live, and returns plain data and code objects. The rest
(import, locate, base check against running code, exec of imports, added definitions and
defaults, pairing, running frames) is the live half. This file compiles the whole patched
module; compiling only the changed definitions, the analysis in a child process, per-block
witnesses and a backfill of new fields are later PRs (README).
"""
import ast
import collections
import copy
import functools
import importlib
import linecache
import time
import types

from . import findings as F
from . import jit as JIT
from . import journal as J
from . import locate as L
from . import reinit as RI
from . import witness as W

# Methods that build an instance: an edit to one never reaches the instances already built.
CONSTRUCTORS = ("__init__", "__new__", "__post_init__")


def first_line(node):
    """co_firstlineno of the code a def/lambda compiles to (first decorator line if any)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
            and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


_DEF = (ast.FunctionDef, ast.AsyncFunctionDef)


def qual_index(tree):
    """qualname -> def/lambda nodes in source order (how CPython names their code)."""
    idx = {}

    def visit(node, pre):
        if isinstance(node, _DEF):
            q = pre + node.name
            idx.setdefault(q, []).append(node)
            for x in node.decorator_list + node.args.defaults + \
                    [d for d in node.args.kw_defaults if d]:
                visit(x, pre)
            for x in node.body:
                visit(x, q + ".<locals>.")
        elif isinstance(node, ast.ClassDef):
            for x in node.decorator_list + node.bases:
                visit(x, pre)
            for x in node.body:
                visit(x, pre + node.name + ".")
        elif isinstance(node, ast.Lambda):
            q = pre + "<lambda>"
            idx.setdefault(q, []).append(node)
            visit(node.body, q + ".<locals>.")
        elif isinstance(node, ast.GeneratorExp):
            # its own code object, a constant of the enclosing code (list/set/dict
            # comprehensions are inlined since 3.12 and have none). The first iterable is
            # evaluated by the enclosing code, so it is named there. CPython adds ".<locals>"
            # only below a function or lambda: code nested in a generator expression is
            # named "<genexpr>.<genexpr>", "<genexpr>.<lambda>".
            q = pre + "<genexpr>"
            idx.setdefault(q, []).append(node)
            visit(node.generators[0].iter, pre)
            for x in [node.elt] + node.generators[0].ifs + node.generators[1:]:
                visit(x, q + ".")
            for g in node.generators:
                visit(g.target, q + ".")
        else:
            for x in ast.iter_child_nodes(node):
                visit(x, pre)

    for x in tree.body:
        visit(x, "")
    return idx


def _inject(node, key, undo):
    """`witness.inject` on a copy of the node's body list (or expression), logging the old
    one in `undo`: the caller's tree comes back unchanged after `_restore`."""
    f = "elt" if isinstance(node, ast.GeneratorExp) else "body"
    v = getattr(node, f)
    _set(undo, node, **{f: list(v) if isinstance(v, list) else v})
    W.inject(node, key)


def _ordinal(idx, q, line):
    hits = [i for i, n in enumerate(idx.get(q, [])) if first_line(n) == line]
    return hits[0] if len(hits) == 1 else None


def _set(undo, node, **fields):
    """Set node fields, logging the old values in `undo` so `_restore` can put them back."""
    for f, v in fields.items():
        undo.append((node, f, getattr(node, f)))
        setattr(node, f, v)


def _restore(undo):
    while undo:
        node, f, v = undo.pop()
        setattr(node, f, v)


def _blank(tree, targets, added, undo):
    """The module AST with target bodies and signatures, added defs and imports removed: two
    blanked trees differ only if the patch changes something a code swap does not apply.
    Every edit of the tree is logged in `undo` (`_restore` takes them back)."""
    idx = qual_index(tree)
    for q, i in targets:
        n = idx[q][i]
        if isinstance(n, _DEF):          # a code swap installs no annotation: blank them too
            _set(undo, n, body=[ast.Pass()], returns=None,
                 args=ast.arguments([], [], None, [], [], None, []),
                 **({"type_params": []} if hasattr(n, "type_params") else {}))
        elif isinstance(n, ast.GeneratorExp):    # keep the first iterable (enclosing code)
            g = n.generators[0]
            _set(undo, n, elt=ast.Constant(None), generators=[
                ast.comprehension(ast.Name("_", ast.Store()), g.iter, [], 0)])
        else:
            _set(undo, n, body=ast.Constant(None),
                 args=ast.arguments([], [], None, [], [], None, []))

    for n in ast.walk(tree):      # docstrings change no behaviour (see _dump_code)
        b = getattr(n, "body", None)
        if isinstance(n, (*_DEF, ast.ClassDef, ast.Module)) and isinstance(b, list) and _doc(b):
            _set(undo, n, body=b[1:] or [ast.Pass()])

    def strip(body, pre):
        out = []
        for n in body:
            if isinstance(n, (ast.Import, ast.ImportFrom)) and not pre:
                continue
            if isinstance(n, ast.If) and not pre and _type_checking(n.test):
                if not n.orelse:          # `if TYPE_CHECKING:` never runs: like an import
                    continue
                _set(undo, n, body=[ast.Pass()])
            if isinstance(n, (*_DEF, ast.ClassDef)) and pre + n.name in added:
                continue
            if isinstance(n, ast.ClassDef):
                _set(undo, n, body=strip(n.body, pre + n.name + "."))
            out.append(n)
        return out

    _set(undo, tree, body=strip(tree.body, ""))
    return ast.dump(tree)


def _type_checking(test):
    """Is an `if` test `TYPE_CHECKING` or `typing.TYPE_CHECKING` (False at run time)?"""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


class Staged:
    """What `apply` will do. Built by `stage`; nothing in it is visible to the engine yet."""

    def __init__(self, mod, source=None):
        self.mod = mod
        self.source = source
        self.imports = {}        # name -> object to add to the module dict
        self.swaps = []          # (function object, new code, defaults|KEEP, kwdefaults|KEEP)
        self.adds = []           # (owner: module dict or class, name, new object)
        self.keys = []           # witness keys armed by this swap
        self.old_codes = {}      # key -> [code objects the target ran before]
        self.functions = {}      # key -> [function objects that held the target's code]
        self.expect = []         # (function, code it ran at stage time, its hash): validate
        self.removes = []        # (owner: module dict or class, name) the edit deletes
        self.items = {}          # key -> JSON summary
        self.line_map = {}       # (qualname, patch line) -> base line (stage's line_maps)
        self.jits = []           # (jit.plan's plan, witness key): JIT kernels to rebuild
        self.jit_adds = []       # (added JIT object, witness key), built at apply
        self.pending_adds = []   # added definitions to build at apply (analysis records)
        self.add_slots = []      # (owner, name) each added definition goes to


class _Deferred:
    """New defaults that name a definition the edit adds: evaluated at apply, once it is
    built."""

    def __init__(self, q, stub):
        self.q, self.stub = q, stub


_compile = compile          # `stage` has a parameter named compile


LIVE_BODY = "__fastrsi_class_body__"


def _shell(name, body):
    """A scratch class statement whose body reads the live class's names: its metaclass is
    the `LIVE_BODY` name, which the live half binds to a `_LiveBody`."""
    return ast.ClassDef(name, [], [ast.keyword("metaclass", ast.Name(LIVE_BODY, ast.Load()))],
                        body, [], type_params=[])


class _LiveBody:
    """Stands in for the metaclass of a scratch class (`_shell`), so new defaults, decorators
    and methods are evaluated as in the live class body: the body's namespace is a ChainMap
    whose reads see the live class's names, then the patch's new imports (then, as in any
    class body, the module globals and builtins). Writes go to a fresh dict, never to the
    live class, which is not touched before commit. The result is a scratch class holding
    only the new names."""

    def __init__(self, live=None, imports=None, added=None):
        # added: members the patch adds to this class, built before this one (a property's
        # @x.setter reads the x the patch added just above it)
        self.maps = ([added] if added else []) + ([vars(live)] if live is not None else [])
        if imports:
            self.maps.append(imports)

    def __prepare__(self, name, bases, **kw):
        return collections.ChainMap({}, *self.maps)

    def __call__(self, name, bases, ns, **kw):
        return type(name, bases, dict(ns.maps[0]))


def _live_class(mod, path):
    """The live class at dotted `path` in `mod`, or None."""
    obj = mod
    for p in path.split(".") if path else ():
        obj = getattr(obj, p, None)
        if not isinstance(obj, type):
            return None
    return obj if path else None


def _futures(tree):
    """The module's `from __future__` statements: every unit compiled apart carries them, or
    `from __future__ import annotations` stops applying (annotations naming TYPE_CHECKING-only
    imports raise NameError when the unit runs)."""
    return [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]


def _defaults_stub(q, node, futures=()):
    """The def with an empty body, inside a scratch class named like the innermost class of
    the qualname (same name mangling; a class body sees only its own class's names), whose
    body reads the live class (`_shell`), compiled: running it in the module's globals
    evaluates the new defaults as the live class body would."""
    stub = copy.deepcopy(node)
    stub.body, stub.decorator_list = [ast.Pass()], []
    parts = q.split(".")
    body = [stub] if len(parts) == 1 else [_shell(parts[-2], [stub])]
    return _try_compile(ast.fix_missing_locations(ast.Module(list(futures) + body, [])),
                        "<defaults>")


def _defaults(mod, q, stub, imports=None):
    """Evaluate a def's new defaults in a scratch namespace: the module globals, the patch's
    new `imports`, and for a method the live class's names (read-only, `_LiveBody`). Returns
    (defaults, kwdefaults) or raises."""
    parts = q.split(".")
    cls = ".".join(parts[:-1])
    ns = dict(imports or {})
    ns[LIVE_BODY] = _LiveBody(_live_class(mod, cls), imports)
    exec(_raise(stub), mod.__dict__, ns)
    obj = ns[parts[0]] if len(parts) == 1 else ns[parts[-2]].__dict__[parts[-1]]
    return obj.__defaults__, obj.__kwdefaults__


def _member_functions(obj):
    """The plain functions an added class member runs: a function itself, a staticmethod's or
    classmethod's `__func__`, a property's getter, setter and deleter, a cached_property's
    `func`. A member that holds no code (a constant, a nested class, any other descriptor)
    has none."""
    if isinstance(obj, property):
        fns = (obj.fget, obj.fset, obj.fdel)
    elif isinstance(obj, (staticmethod, classmethod)):
        fns = (obj.__func__,)
    elif isinstance(obj, functools.cached_property):
        fns = (obj.func,)
    elif isinstance(getattr(obj, "__wrapped__", None), types.FunctionType):
        fns = (obj.__wrapped__,)
    else:
        fns = (obj,)
    return [f for f in fns if isinstance(f, types.FunctionType)]


def _set_class_cell(obj, cls):
    """Point the `__class__` cell (zero-argument super()) of an added member's functions at
    the live class (they were built in a scratch class). A member that is not a plain function
    (a @property, a staticmethod) has no `__closure__` of its own, so its functions come from
    `_member_functions`."""
    for fn in _member_functions(obj):
        code = fn.__code__
        if fn.__closure__ and "__class__" in code.co_freevars:
            fn.__closure__[code.co_freevars.index("__class__")].cell_contents = cls


def analyze(base_src, source, targets, filename, key_module, live, trees=None):
    """The text half of `stage`: everything that needs only the two texts, the targets and
    which targets have a live function (`live`: {(qualname, line): bool}). It parses both
    texts, matches targets, adds the witness, compiles the module, finds added, removed and
    imported names, and compares the blanked trees. It touches no live object, so it can run
    in another process (`ProcessAnalyzer`). targets "auto" (or None): the changed definitions
    of the two texts, found from the trees parsed here (as `changed_targets`), so the caller
    needs no parse of its own; `live` then holds every live (qualname, line) of the module.
    Returns plain data and code objects (marshal can carry them): {"error"} when a text does
    not parse, else {"targets": [per-target dict in order], "resolved": [[qualname, line]]
    (the targets analyzed), "base_units", "new_units", "new_error", "line_map",
    "added", "removed", "imports", "outside_changes", "timings"}.
    trees: (base tree, patch tree) the caller already parsed from exactly these two texts
    (`ast.parse`), used instead of parsing again; they come back unchanged (every edit the
    analysis makes to them is taken back before it returns)."""
    timings, t0 = {}, time.perf_counter()
    if trees is not None:
        base_tree, tree = trees
        timings["parse_s"] = 0.0
        timings["trees"] = "caller"
    else:
        try:
            base_tree = ast.parse(base_src, filename)
            tree = ast.parse(source, filename)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "timings": timings}
        timings["parse_s"] = round(time.perf_counter() - t0, 6)
    undo = []
    try:
        return _analyze_trees(base_src, source, targets, filename, key_module, live,
                              base_tree, tree, timings, undo)
    finally:
        _restore(undo)


def _analyze_trees(base_src, source, targets, filename, key_module, live,
                   base_tree, tree, timings, undo):
    bq, pq = qual_index(base_tree), qual_index(tree)
    auto = targets is None or targets == "auto"
    if auto:                 # every target found here differs by AST: no dump per target below
        t = time.perf_counter()
        targets = _changed(base_src, bq, source, pq)
        timings["targets_s"] = round(time.perf_counter() - t, 6)
    t = time.perf_counter()
    base_units = [_compile(base_tree, filename, "exec", dont_inherit=True)]
    compile_s = time.perf_counter() - t
    line_map = {}
    for q, nodes in pq.items():
        if len(nodes) == len(bq.get(q, [])):
            for n, b in zip(nodes, bq[q]):
                line_map[(q, first_line(n))] = first_line(b)

    # 1. targets: match base and patch, add the witness (the live checks are the caller's)
    tres, todo, aliases = [], [], []
    lines = source.splitlines(keepends=True)
    for q, line in targets:
        line = int(line)
        k = L.key_of(key_module, q, line)
        i = _ordinal(bq, q, line)
        if i is None:
            tres.append({"status": "no_definition"})
            continue
        if not live.get((q, line)):
            # fallback when no line map led here (see stage's `line_maps`): the one live code
            # with this qualname is taken as it; stage reports it as a finding
            cands = [kl for kl in live if kl[0] == q and live[kl]]
            if len(cands) != 1:
                tres.append({"status": "no_live_function"})
                continue
            aliases.append([q, line, cands[0][1]])
        if len(pq.get(q, [])) != len(bq[q]):
            tres.append({"status": "count", "patch": len(pq.get(q, [])), "base": len(bq[q])})
            continue
        node = pq[q][i]
        changed = auto or ast.dump(bq[q][i]) != ast.dump(node)
        # the text a JIT rebuild compiles: from the clean patch text, before the witness
        jt = JIT.text_of(lines, bq[q][i], node) if isinstance(node, _DEF) and (
            node.decorator_list or bq[q][i].decorator_list) else None
        # a constructor edit that only changes fresh field values: from the clean trees
        ri = RI.plan(bq[q][i], node) if q.rpartition(".")[2] in CONSTRUCTORS else None
        _inject(node, k, undo)
        todo.append((q, i))
        tres.append({"status": "ok", "i": i, "changed": changed, "new_line": first_line(node),
                     "jit": _jit_decorator(bq[q][i]) or _jit_decorator(node), "jit_text": jt,
                     **({"reinit": ri} if ri is not None else {})})

    # 2. added definitions: top-level ones, and methods of classes that exist in the base
    added = []
    base_top = {getattr(n, "name", None) for n in base_tree.body}
    for n in tree.body:
        if isinstance(n, (*_DEF, ast.ClassDef)) and n.name not in base_top:
            added.append((n.name, None, n))
        elif isinstance(n, ast.ClassDef) and n.name in base_top:
            bnames = {getattr(m, "name", None) for b in base_tree.body
                      if isinstance(b, ast.ClassDef) and b.name == n.name for m in b.body}
            added += [(f"{n.name}.{m.name}", n.name, m) for m in n.body
                      if isinstance(m, _DEF) and m.name not in bnames]
    for q, _, n in added:
        if isinstance(n, _DEF) and not _jit_decorator(n):   # a JIT kernel: launch hooks
            _inject(n, L.key_of(key_module, q, first_line(n)), undo)

    # 2b. removed definitions
    removed = removed_definitions(base_tree, tree)
    nested = sorted(_nested_methods(base_tree) ^ _nested_methods(tree))

    # 4. top-level imports, compiled; the caller runs those the module lacks or that changed
    base_imports = {ast.dump(stmt) for stmt in base_tree.body
                    if isinstance(stmt, (ast.Import, ast.ImportFrom))}
    imports = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            names = [(alias.asname or alias.name).split(".")[0] for alias in stmt.names]
            imp = {"names": names, "changed": ast.dump(stmt) not in base_imports,
                   **_try_compile(ast.Module([stmt], []), filename)}
            if isinstance(stmt, ast.ImportFrom):  # resolvable against an overlay (see stage)
                imp["from"] = [stmt.level, stmt.module,
                               [[alias.name, alias.asname] for alias in stmt.names]]
            imports.append(imp)
    t = time.perf_counter()
    new_units, new_error = [], None
    try:
        new_units = [_compile(ast.fix_missing_locations(tree), filename, "exec",
                              dont_inherit=True)]
    except Exception as e:
        new_error = f"{type(e).__name__}: {e}"
    timings["compile_s"] = round(compile_s + time.perf_counter() - t, 6)
    # 5. added definitions, compiled for the caller's scratch namespace
    adds = []
    futures = _futures(tree)       # the module's `from __future__` flags (annotations)
    for q, cls_name, n in added:
        body = [n] if cls_name is None else [_shell(cls_name, [n])]
        adds.append({"q": q, "cls": cls_name, "name": n.name, "def": isinstance(n, _DEF),
                     "line": first_line(n),
                     **_try_compile(ast.fix_missing_locations(ast.Module(futures + body, [])),
                                    filename)})
    # 6. new defaults: a stub the caller evaluates in the module's globals
    for (q, i), r in zip(todo, [r for r in tres if r["status"] == "ok"]):
        bnode, pnode = bq[q][i], pq[q][i]
        if not isinstance(pnode, _DEF) or _defaults_key(bnode) == _defaults_key(pnode):
            continue
        if "<locals>" not in q:
            r["defaults"] = _defaults_stub(q, pnode, futures)
        else:
            # a nested def's defaults live on each closure, made when the enclosing call ran
            # the def statement: a swap gives live closures the new code with the old
            # defaults (TypeError at the next call), so it blocks (audit #10)
            r["nested_signature"] = (f"{ast.unparse(bnode.args)} -> {ast.unparse(pnode.args)}")

    # 6b. fields the new code reads that only the new __init__ sets (`new_fields`)
    meths = {}
    for q, i in todo:
        parts = q.split(".")
        if len(parts) == 2 and isinstance(pq[q][i], _DEF):
            meths.setdefault(parts[0], []).append(pq[q][i])
    for q, cls_name, n in added:
        if cls_name is not None and isinstance(n, _DEF):
            meths.setdefault(cls_name, []).append(n)
    new_fields = _new_fields(base_tree, tree, meths, filename) if meths else []

    # 7. anything else changed? Last, because blanking edits the two trees (logged in undo).
    t = time.perf_counter()
    outside = _blank(base_tree, todo, set(removed), undo) != _blank(
        tree, todo, {q for q, _, _ in added}, undo)
    timings["compare_s"] = round(time.perf_counter() - t, 6)
    return {"targets": tres, "resolved": [[q, int(line)] for q, line in targets],
            "base_units": base_units, "new_units": new_units,
            "new_error": new_error, "line_map": line_map, "added": adds, "removed": removed,
            "imports": imports, "outside_changes": outside, "timings": timings,
            "aliases": aliases, "new_fields": new_fields, "nested_changes": nested}


def _nested_methods(tree):
    """Qualnames of the methods of classes nested in a class (`Outer.Inner.m`): stage adds
    and removes only top-level definitions and methods of top-level classes."""
    out = set()

    def walk(body, prefix, depth):
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, f"{prefix}{node.name}.", depth + 1)
            elif isinstance(node, _DEF) and depth >= 2:
                out.add(prefix + node.name)
    walk(tree.body, "", 0)
    return out


def _defaults_key(node):
    """What the function object's __defaults__ and __kwdefaults__ are made of: the default
    expressions, and the keyword-only names (__kwdefaults__ is keyed by name, so a renamed
    keyword-only parameter with the same default needs new defaults too)."""
    a = node.args
    return ast.dump(ast.Tuple(a.defaults + a.kw_defaults)), [x.arg for x in a.kwonlyargs]


_JIT = ("jit", "autotune", "heuristics")
_JIT_TYPES = ("JITFunction", "Autotuner", "Heuristics", "InterpretedFunction")


def _jit_decorator(node):
    """`triton.jit` (or `triton.autotune(...)`, `triton.heuristics(...)`, also as
    `triton.language.jit`) among the def's decorators, as text; else None. A bare `@jit`
    from-import is left to the live check (`_jit_holder`)."""
    for d in getattr(node, "decorator_list", ()):
        f = d.func if isinstance(d, ast.Call) else d
        if isinstance(f, ast.Attribute) and f.attr in _JIT:
            root = f.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in ("triton", "tl"):
                return ast.unparse(f)
    return None


def _jit_holder(mod, q):
    """The type name of the JIT object (Triton's JITFunction, Autotuner, Heuristics) that the
    live module holds under qualname q and that keeps the target in `.fn`; else None."""
    obj = mod
    for part in q.split("."):
        try:
            d = object.__getattribute__(obj, "__dict__")
        except Exception:
            return None
        obj = d.get(part) if isinstance(d, (dict, types.MappingProxyType)) else None
        if obj is None:
            return None
    outer = obj
    for _ in range(8):                   # autotune(heuristics(jit(fn))): follow .fn
        if isinstance(obj, types.FunctionType):
            break
        t = type(obj)
        if t.__name__ in _JIT_TYPES or t.__module__.split(".")[0] == "triton":
            return type(outer).__name__
        try:
            obj = object.__getattribute__(obj, "__dict__").get("fn")
        except Exception:
            return None
    return None


def _self_name(fn):
    args = fn.args.posonlyargs + fn.args.args
    return args[0].arg if args else None


def _self_attrs(node, me, ctx):
    """Attribute names X of `me.X` in node with the given context (ast.Load / ast.Store).
    `me.X += 1` both reads and writes X (its target has the Store context only)."""
    if me is None:
        return set()

    def mine(n):
        return isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
            and n.value.id == me
    out = {n.attr for n in ast.walk(node) if mine(n) and isinstance(n.ctx, ctx)}
    if ctx is ast.Load:
        out |= {n.target.attr for n in ast.walk(node)
                if isinstance(n, ast.AugAssign) and mine(n.target)}
    return out


def _class_names(cls):
    out = set()
    for n in cls.body:
        if isinstance(n, (*_DEF, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                and n.value is not None:      # a bare `x: int` defines no class attribute
            out.add(n.target.id)
    return out


def _is_dataclass(cls_def):
    for decorator in cls_def.decorator_list:
        func = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (isinstance(func, ast.Name) and func.id == "dataclass") or (
                isinstance(func, ast.Attribute) and func.attr == "dataclass"):
            return True
    return False


def _dataclass_fields(cls_def):
    """Bare annotated field names (`x: T`, no default) of a @dataclass body: the generated
    __init__ sets them on instances, while the class body itself defines no attribute."""
    if not _is_dataclass(cls_def):
        return set()
    return {node.target.id for node in cls_def.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and node.value is None}


def _new_fields(base_tree, tree, meths, filename):
    """[{cls, field, readers, init_line}] for each attribute that a changed or added method
    of a top-level class reads on `self`, that the edited __init__ / __post_init__ or a new
    bare @dataclass field sets, and that no method of the base class sets nor the base class
    body defines (a code swap does not run the patched class body; stage still skips a name
    the live class has)."""
    base_classes = {node.name: node for node in base_tree.body if isinstance(node, ast.ClassDef)}
    patch_classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    out = []
    for cls_name, nodes in meths.items():
        base_cls, patch_cls = base_classes.get(cls_name), patch_classes.get(cls_name)
        if base_cls is None or patch_cls is None:
            continue
        inits = [method for method in patch_cls.body if isinstance(method, _DEF)
                 and method.name in ("__init__", "__post_init__")]
        sets = _dataclass_fields(patch_cls)
        for method in inits:
            sets |= _self_attrs(method, _self_name(method), ast.Store)
        if not sets:
            continue
        old = _class_names(base_cls) | _dataclass_fields(base_cls)
        for method in base_cls.body:
            if isinstance(method, _DEF):
                old |= _self_attrs(method, _self_name(method), ast.Store)
        readers = {}
        for method in nodes:
            if method.name in ("__init__", "__post_init__"):
                continue
            for field in _self_attrs(method, _self_name(method), ast.Load):
                if field in sets and field not in old:
                    readers.setdefault(field, []).append(f"{cls_name}.{method.name}")
        line = first_line(inits[0]) if inits else first_line(patch_cls)
        for field in sorted(readers):
            out.append({"cls": cls_name, "field": field, "readers": sorted(set(readers[field])),
                        "init_line": line})
    return out


def _try_compile(mod, filename):
    try:
        return {"code": _compile(mod, filename, "exec")}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


class _Failed(Exception):
    """An error the analysis reported, with its original "Type: message" text."""

    def __str__(self):
        return self.args[0]


def _raise(entry):
    if entry.get("error") is not None:
        raise _Failed(entry["error"])
    return entry["code"]


def _err(e):
    return str(e) if isinstance(e, _Failed) else f"{type(e).__name__}: {e}"


def stage(module, source, targets="auto", base_hash=None, key_module=None, scan="auto",
          base_source=None, own_codes=(), line_maps=(), trees=None):
    """module: the live module's name; source: the full patched module text; targets:
    [(qualname, firstlineno in the base)], or "auto" (the default; None too): the analysis
    finds the changed definitions itself from its one parse of the two texts (the same list
    as `changed_targets(base text, source)`, so the caller does not parse to plan), and the
    result's `targets` says which; base_hash: text_hash of the source the patch was
    made against (checked against the base text); base_source: the text the live module runs
    now, when an earlier transaction changed it (default: the module's file); own_codes: code
    objects this participant installed itself (with their witness), which match base_source
    by construction and are not reported as `base_mismatch`; key_module: the module name used in
    witness keys (participants that load a module under another name pass the shared one);
    scan: "auto" (heap scan only when the module namespace cannot find a target, suspended
    frames only for generator / coroutine targets) or "always" (both scans, as before v2.2).
    trees: (base tree, patch tree), `ast.parse` of exactly the base text (`base_source`, or
    the module's file) and `source`, for a caller that parsed them already (its own planner):
    the analysis uses them instead of parsing again and leaves them unchanged.
    With `changed_targets(..., trees=)` a caller can also pick a subset of the changed
    definitions and pass them as explicit targets, still without a parse.
    line_maps: when base_source is the text of earlier swaps, their line maps, newest first:
    each {(qualname, line in that swap's patch): line in its base} (the `line_map` of each
    staged swap). A definition no swap replaced still carries the line of the text it was
    compiled from, so a target keyed at base_source's line is found by walking the maps back
    to that text (`timings["line_map_aliases"]`). Without a map that leads there, the one live
    code with the target's qualname is taken as it: a `line_aliased` finding (advisory; the
    content check against the base text still guards it).
    Returns {"ok", "staged", "items", "findings", "timings", "targets"}; ok is False only
    when nothing can be staged (bad patch, import error). Each target's item says whether its definition
    `changed` (AST, before the witness is added). Never raises."""
    kname = key_module or module
    fs, timings, t0 = [], {}, time.perf_counter()
    try:
        mod = importlib.import_module(module)
        if base_source is None:
            with open(mod.__file__) as fh:
                base_src = fh.read()
        else:
            base_src = base_source
    except Exception as e:
        return _invalid(f"{type(e).__name__}: {e}", kname)
    # Locate without a heap scan when the module namespace finds every target (top-level
    # functions and methods); scan (one get_objects + a sliced pass) only for nested targets
    # or targets the namespace does not hold.
    t = time.perf_counter()
    auto = targets is None or targets == "auto"
    run, held = L.index_module(mod, None if scan == "always" else L.module_functions(mod))
    mapped = _follow_line_maps(run, line_maps)
    timings["scan"] = "heap" if scan == "always" else "namespace"
    if not auto and scan == "auto" and any(_needs_heap(run.get((q, int(line)), ()), run, held)
                                           for q, line in targets):
        run, held = L.index_module(mod)
        mapped = _follow_line_maps(run, line_maps)
        timings["scan"] = "heap"
    timings["locate_s"] = round(time.perf_counter() - t, 6)
    # explicit targets that no line map resolved: the qualname fallback (auto: in analyze)
    fallback = [] if auto else _alias_shifted(targets, run)
    # auto: the targets are known only after the analysis, so it gets every live key
    live = _live_keys(held, mapped) if auto else _live_of(targets, run, held)
    req = {"base_src": base_src, "source": source,
           "targets": "auto" if auto else [[q, int(line)] for q, line in targets],
           "filename": mod.__file__, "key_module": kname, "live": live}
    t = time.perf_counter()
    a = analyze(**req, trees=trees)
    if auto and "error" not in a:
        targets = a["resolved"]
        timings["targets_s"] = a["timings"]["targets_s"]
        if scan == "auto" and any(_needs_heap(run.get((q, line), ()), run, held)
                                  for q, line in targets):
            t2 = time.perf_counter()
            run, held = L.index_module(mod)
            mapped = _follow_line_maps(run, line_maps)
            timings["scan"] = "heap"
            timings["locate_s"] = round(timings["locate_s"] + time.perf_counter() - t2, 6)
            heap_live = _live_of(targets, run, held)
            if any(heap_live[k] != bool(live.get(k)) for k in heap_live):
                # the heap found a live function the namespace did not: analyze again with
                # the explicit targets, exactly as a caller passing them would
                req = {**req, "targets": targets, "live": heap_live}
                a = analyze(**req, trees=trees)
                timings["analyze_again"] = True
    timings["analyze_s"] = round(time.perf_counter() - t, 6)
    at = a["timings"]
    if "error" in a:
        return _invalid(a["error"], kname)
    if mapped:
        timings["line_map_aliases"] = mapped
    for q, bl, ll in a.get("aliases") or ():
        if (q, int(bl)) not in run and (q, int(ll)) in run:
            run[(q, int(bl))] = run[(q, int(ll))]
            fallback.append([q, int(bl), int(ll)])
    nodef = {(q, int(line)) for (q, line), r in zip(a["resolved"], a["targets"])
             if r["status"] == "no_definition"}
    fallback = [x for x in fallback if (x[0], x[1]) not in nodef]   # not_in_patch says it
    for q, bl, ll in fallback:
        fs.append({**F.finding(
            "line_aliased", f"no line map leads from line {bl} of the base text to live "
            f"code; the one live {q} (line {ll}) is taken as it",
            "a line map from the earlier swaps to the live code",
            subject=L.key_of(kname, q, bl)), "advisory": True})
    if fallback:
        timings["line_aliases"] = fallback
    timings["parse_s"] = at["parse_s"]
    if "trees" in at:
        timings["trees"] = at["trees"]
    st = Staged(mod)
    st.source = source
    st.line_map = a["line_map"]
    if base_hash is not None and L.text_hash(base_src) != base_hash:
        fs.append(F.finding("base_mismatch", f"base text hash {L.text_hash(base_src)}",
                            f"plan base hash {base_hash}", subject=kname))
    base_idx = {}
    for c in a["base_units"]:
        L.code_tree(c, None, (base_idx, {}))

    # 1. targets: the live checks, in the order of the analysis
    # this participant's own installed code, nested codes included (generator expressions and
    # lambdas inside a swapped function carry its witness, so they differ from the file)
    todo, changed, own = [], {}, {id(c) for c in _nested(own_codes)}
    for (q, line), r in zip(targets, a["targets"]):
        line = int(line)
        k = L.key_of(kname, q, line)
        olds = run.get((q, line), [])
        if r["status"] == "no_definition":
            alts = sorted({ln for (qq, ln) in run if qq == q and ln != line})
            hint = (f"; {q} is keyed at line(s) {alts}: a decorated definition's key is the "
                    "line of its first decorator, not the `def` line" if alts else "")
            fs.append(F.finding("not_in_patch", f"no definition of {q} at line {line} in "
                                f"the base file{hint}",
                                "one definition with this qualname and line", subject=k))
            continue
        if r["status"] == "no_live_function":
            fs.append(F.finding("no_live_function", "no function object runs this code or "
                                "encloses it", "a function object to swap", subject=k))
            continue
        if r["status"] == "count":
            fs.append(F.finding("not_in_patch", f"{r['patch']} definitions named {q} "
                                f"in the patch, {r['base']} in the base",
                                "the same number", subject=k))
            continue
        base_code = base_idx[(q, line)][0]
        for o in olds:
            if id(o) not in own and L.content_hash(o) != L.content_hash(base_code):
                fs.append(F.finding("base_mismatch", f"running code {L.content_hash(o)}",
                                    f"the file's code {L.content_hash(base_code)}", subject=k))
        changed[k] = r["changed"]
        todo.append((q, line, r, k, olds))

    # 2b. removed definitions: a live edit that only adds and swaps leaves them callable, so a
    # forgotten caller works live and fails after a restart (vLLM #61843). A finding; on
    # accept, apply deletes them through the journal.
    for q in a["removed"]:
        cls, _, name = q.rpartition(".")
        owner = mod.__dict__ if not cls else getattr(mod, cls, None)
        present = owner is not None and (name in owner if isinstance(owner, dict)
                                         else name in vars(owner))
        if present:
            fs.append(F.finding("removed_definition", f"{q} is deleted by the edit but defined "
                                "in the live module", f"{q} gone, as after a restart",
                                subject=f"{kname}:{q}"))
            st.removes.append((owner, name))

    try:
        # 4. imports the module lacks or that the edit changed, run in a scratch namespace
        for imp in a["imports"]:
            if imp.get("changed") or any(name not in mod.__dict__ for name in imp["names"]):
                scratch_ns = {}
                exec(_raise(imp), mod.__dict__, scratch_ns)
                st.imports.update({
                    binding_name: bound_value
                    for binding_name, bound_value in scratch_ns.items()
                    if binding_name not in mod.__dict__
                    or mod.__dict__[binding_name] is not bound_value
                })
        if a["new_error"] is not None:
            raise _Failed(a["new_error"])
        new_idx, parent = {}, {}
        for c in a["new_units"]:
            L.code_tree(c, None, (new_idx, parent))
        timings["compile_s"] = at["compile_s"]
        timings["parse_compile_s"] = round(timings["parse_s"] + timings["compile_s"], 6)
        # 5. added definitions: built at apply, not here (audit #23): building runs their
        # decorators and class bodies, whose side effects (a registry, a JIT build) must not
        # happen for a transaction that is then aborted. Here only where each one goes and
        # its witness key.
        top = dict(st.imports)             # new imports (added definitions join at apply)
        for ad in a["added"]:
            if ad.get("error"):
                raise _Failed(ad["error"])
            owner = mod.__dict__ if ad["cls"] is None else getattr(mod, ad["cls"])
            st.pending_adds.append(ad)
            st.add_slots.append((owner, ad["name"]))
            if ad["def"]:
                k = L.key_of(kname, ad["q"], ad["line"])
                st.keys.append(k)
                st.items[k] = {"added": True}
                ad["key"] = k
    except Exception as e:
        fs.append(F.finding("patch_invalid", _err(e),
                            "the patch compiles and its imports run", subject=kname))
        return F.not_ok(_err(e), fs, staged=None, items={})

    # 6. pair old and new code: the target and every enclosing function with live holders
    t = time.perf_counter()
    # A generator expression's generator lives as long as its consumer, which is almost always
    # inside the enclosing call (on a thread stack, checked here). Suspended ones (stored for
    # later) are looked for only with scan="always": in "auto" they finish on the old code.
    susp = scan == "always" or W.suspendable(
        [o for *_, olds in todo for o in olds if o.co_name != "<genexpr>"])
    timings["suspended"] = bool(susp)
    # A frame on a thread stack now is a step in flight (prepare runs while steps go on): it
    # is advisory here, and validate checks the stacks again in the gap, where only a frame
    # live at the quiesce point blocks. A suspended generator / coroutine outlives the gap:
    # that one blocks here.
    on_stack = W.stack_codes()
    # suspended frames only of the target codes (their referrers), never a whole-heap pass
    live = (W.live_codes(suspended=True) if scan == "always" else W.live_codes(
        suspended=True, codes=[o for *_, olds in todo for o in olds])) if susp else set(on_stack)
    timings["live_frames_s"] = round(time.perf_counter() - t, 6)
    plan, line_map = {}, a["line_map"]
    for q, line, r, k, olds in todo:
        new = new_idx[(q, r["new_line"])][0]
        chain, c, bad = [], new, False
        while c is not None and c.co_qualname != "<module>":
            bl = line_map.get((c.co_qualname, c.co_firstlineno))
            for o in run.get((c.co_qualname, bl), []):
                if o.co_freevars != c.co_freevars:
                    fs.append(F.finding("closure_changed", f"free variables {c.co_freevars}",
                                        f"the running closure's {o.co_freevars}", subject=k))
                    bad = True
                chain.append((o, c))
            c = parent.get(c)
        if "nested_signature" in r and olds:
            fs.append(F.finding("nested_signature", f"{q}: parameters {r['nested_signature']}",
                                "the same defaults as the live closures", subject=k))
            bad = True
        if bad:
            continue
        defaults = kwdefaults = J.KEEP
        if "defaults" in r:
            try:
                defaults, kwdefaults = _defaults(mod, q, r["defaults"], top)
            except NameError as e:
                if getattr(e, "name", None) in {n for _, n in st.add_slots}:
                    defaults = _Deferred(q, r["defaults"])   # names an added definition
                else:
                    fs.append(F.finding("defaults_not_evaluated", _err(e),
                                        "the new defaults evaluate", subject=k))
            except Exception as e:
                fs.append(F.finding("defaults_not_evaluated", _err(e),
                                    "the new defaults evaluate", subject=k))
        for o, c in chain:
            plan.setdefault(o, (c, J.KEEP, J.KEEP))
        for o in olds:
            plan[o] = (plan[o][0], defaults, kwdefaults)
            if o in live and o not in on_stack:
                fs.append(F.finding("running_frame", "a suspended frame of the old code "
                                    "(generator or coroutine) will resume on it",
                                    "no running frame", subject=k))
            elif o in on_stack:
                fs.append({**F.finding("running_frame", "a frame of the old code was on a "
                                       "thread stack during prepare (a step in flight); "
                                       "validate checks again in the gap",
                                       "no running frame", subject=k), "advisory": True})
        st.keys.append(k)
        st.old_codes[k] = list(olds)
        st.functions[k] = [fn for o in olds for fn in held.get(o, [])]
        st.items[k] = {"changed": changed[k],
                       "base_hash": L.content_hash(olds[0]) if olds else None,
                       "new_hash": L.content_hash(new),
                       "functions": sum(len(held.get(o, [])) for o in olds),
                       "via": sum(len(held.get(o, [])) for o, _ in chain)}
    for o, (c, d, kw) in plan.items():
        for fn in held.get(o, []):
            st.swaps.append((fn, c, d, kw))
            st.expect.append((fn, o, L.content_hash(o)))

    # 6a. Triton kernels: the engine runs what the JIT compiled from the text, not the
    # function's bytecode. jit.plan rebuilds it in place (advisory jit_rebuild) or says why
    # it cannot: then the swap would commit as a GPU no-op, so it blocks (jit_target).
    for q, line, r, k, olds in todo:
        why = r.get("jit") or _jit_holder(mod, q)
        if why:
            p = JIT.plan(mod, q, r.get("jit_text"))
            if p["ok"] and k in st.items:     # not when a check above dropped the swap
                pl = p["plan"]
                st.jits.append((pl, k))
                st.items[k]["jit"] = {"callers": pl["callers"], "autotuners": len(pl["tuners"]),
                                      "scanned": pl["scanned"], "plan_s": pl["plan_s"]}
                fs.append({**F.finding(
                    "jit_rebuild", f"{q} ({why}) is rebuilt in place from the edited text; "
                    f"{len(pl['callers'])} JIT caller(s) reset: {pl['callers'][:5]}; "
                    f"{len(pl['tuners'])} autotuner cache(s) cleared",
                    "the next launch compiles the edited kernel", subject=k), "advisory": True})
                continue
            fs.append({**F.finding(
                "jit_target", f"{q} is a JIT kernel ({why}) and {p['why']}: a code swap "
                "changes the Python function, the launches keep the compiled kernel",
                "a target the engine runs as Python bytecode", subject=k), "route": "T1"})

    # 6b. fields only the new __init__ sets, read by new code: live instances lack them.
    # No backfill here: the finding blocks and routes to T1 (rebuild the objects or restart).
    for nf in a.get("new_fields") or ():
        cls = getattr(mod, nf["cls"], None)
        x = nf["field"]
        attr = f"_{nf['cls'].lstrip('_')}{x}" if x.startswith("__") and not x.endswith("__") \
            else x
        if not isinstance(cls, type) or hasattr(cls, attr):
            continue                       # a class-level default covers old instances
        f = F.finding(
            "new_fields", f"{', '.join(nf['readers'])} read self.{x}, which only the edited "
            f"{nf['cls']}.__init__ sets; instances built before the swap lack it",
            f"every live {nf['cls']} has {x}", subject=f"{kname}:{nf['cls']}.{x}")
        f["actions"] = [act for act in f["actions"] if "backfill" not in act["action"]]
        fs.append({**f, "route": "T1"})

    # 6c. constructors: the swap reaches only instances built after it; every instance built
    # before keeps what the old constructor set (vLLM #61323 edits RMSNorm.__init__: the model's
    # norm layers were built at start-up, so a commit changes nothing they hold, witness 0).
    # Stage does not look at instances: any constructor target blocks here. When the edit only
    # changes fresh field values (vLLM #62872), the finding carries a `reinit` plan, and the
    # participant re-initializes those fields on every live instance or keeps it blocking
    # (reinit.py).
    for q, line, r, k, olds in todo:
        cls, _, meth = q.rpartition(".")
        if meth in CONSTRUCTORS and cls and not cls.endswith("<locals>"):
            ri = r.get("reinit") or {}
            f = {**F.finding(
                "constructor_edit", f"{q} changed: instances of {cls} built before the swap "
                "keep the state the old constructor gave them; only new instances run it"
                + ("" if ri.get("fields") else
                   f" (not re-initializable in place: {ri.get('why', 'not a constructor')})"),
                "a target that runs again on the objects the engine already holds",
                subject=k), "route": "T1"}
            if ri.get("fields") and k in st.items:
                f["reinit"] = {"cls": cls, "meth": meth, "fields": ri["fields"]}
            fs.append(f)

    # 7. anything else changed? (compared by the analysis)
    if a.get("nested_changes"):        # a swap cannot add or remove them: not overridable
        fs.append({**F.finding(
            "outside_changes", f"the edit adds or removes methods of a nested class: "
            f"{', '.join(a['nested_changes'][:5])}; the live class would not get them (a new "
            "call to one raises AttributeError)", "only top-level definitions and methods of "
            "top-level classes added or removed", subject=kname), "overridable": False})
    if a["outside_changes"]:
        fs.append(F.finding("outside_changes", "the module differs outside the target bodies "
                            "(beyond added definitions and imports)",
                            "only target bodies, added definitions and imports differ",
                            subject=kname))
    timings["compare_s"] = at["compare_s"]
    timings["stage_s"] = round(time.perf_counter() - t0, 6)
    targets = [[q, int(line)] for q, line in targets]
    for f in fs:        # what a standing accept matches on: the checked code
        item = st.items.get(f.get("subject")) if isinstance(f.get("subject"), str) else None
        if item and "new_hash" in item:
            f["code_hash"] = item["new_hash"]
        elif item:                           # an added definition: no staged code to name
            continue
        elif f.get("subject") == kname:      # about the whole module: its new text
            f["code_hash"] = L.text_hash(source)
    return F.ok(staged=st, items=st.items, findings=fs, timings=timings, targets=targets)


def _live_of(targets, run, held):
    """{(qualname, line): does a function object run this code, or enclose it at any depth?}
    (as `stage` pairs it: through every enclosing code up to one a function holds)"""
    live = {}
    for q, line in targets:
        live[(q, int(line))] = any(_held_up(o, run, held) for o in run.get((q, int(line)), []))
    return live


def _held_up(code, run, held):
    """Is `code`, or a code that encloses it at any depth, run by a function object?"""
    seen, stack = set(), [code]
    while stack:
        c = stack.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        if held.get(c):
            return True
        stack += _enclosing(c, run)
    return False


def _alias_shifted(targets, run):
    """Fallback for explicit targets whose (qualname, base line) runs no code while exactly
    one running code has that qualname: alias the target to it in `run` (the base_mismatch
    content check still compares it with the base text's code). Returns [[qualname, base
    line, live line]]."""
    out = []
    for q, line in targets:
        line = int(line)
        if run.get((q, line)):
            continue
        cands = [k for k in run if k[0] == q and run[k]]
        if len(cands) == 1:
            run[(q, line)] = run[cands[0]]
            out.append([q, line, cands[0][1]])
    return out


def _follow_line_maps(run, line_maps):
    """Add to `run` the keys of base_source's text that earlier swaps' line maps (newest
    first) lead to live code: a definition no swap replaced still carries the line of the
    text it was compiled from. The codes of every step back are merged into the key: a
    swapped enclosing function makes new closures at the new line while the closures it
    made before still run the code at the old line, and both must be swapped. Returns
    [[qualname, base_source line, live line]] for each step that added code."""
    out = []
    if not line_maps:
        return out
    for q, line in list(line_maps[0]):
        cur = line
        for m in line_maps:
            cur = m.get((q, cur))
            if cur is None:
                break
            have = run.get((q, line), [])
            more = [c for c in run.get((q, cur), ()) if not any(c is x for x in have)]
            if more:
                run[(q, line)] = list(have) + more
                out.append([q, line, cur])
    return out


def _live_keys(held, mapped=()):
    """`_live_of` for every code of the module at once: a held code and every code nested
    in it; `mapped` ([qualname, line, live line]) adds the line-mapped keys."""
    live = {}
    for c in _nested(list(held)):
        live[(c.co_qualname, c.co_firstlineno)] = True
    for q, line, ll in mapped:
        if live.get((q, ll)):
            live[(q, line)] = True
    return live


def _invalid(msg, kname):
    return F.not_ok(msg, [F.finding("patch_invalid", msg, "an importable module and a patch "
                                    "that parses", subject=kname)], staged=None, items={})


def validate(st, frames=True):
    """Phase 2, inside the gap: is what `stage` saw still true? Every function to swap must
    still run the exact code object staged against (identity and hash), and every name to add
    must still be absent. With `frames`, no thread stack may hold a frame of an old code now
    (`running_frame`: the quiesce point is where it matters; prepare saw steps in flight).
    Cheap: no heap scan, no compile. Never raises."""
    fs = []
    if frames:
        on_stack = W.stack_codes()
        for k, olds in st.old_codes.items():
            if any(o in on_stack for o in olds):
                fs.append(F.finding("running_frame", "a frame of the old code is running at "
                                    "the quiesce point", "no running frame", subject=k))
    for fn, old, h in st.expect:
        now = fn.__code__
        if now is not old:
            fs.append(F.finding("changed_since_prepare",
                                f"{fn.__qualname__} runs code {L.content_hash(now)}",
                                f"the staged base {h}", subject=fn.__qualname__))
    for owner, name in st.add_slots:
        present = name in owner if isinstance(owner, dict) else name in vars(owner)
        if present:
            fs.append(F.finding("changed_since_prepare", f"{name} was added meanwhile",
                                f"{name} absent, as at prepare", subject=name))
    for owner, name in st.removes:
        present = name in owner if isinstance(owner, dict) else name in vars(owner)
        if not present:
            fs.append(F.finding("changed_since_prepare", f"{name} was deleted meanwhile",
                                f"{name} present, as at prepare", subject=name))
    return F.ok(checked=len(st.expect)) if not fs else F.not_ok("changed since prepare", fs)


def stage_and_apply(module, source, journal, witness, targets="auto", accept=(), **stage_kw):
    """For callers that stage and apply in one call (no two-phase host): stage -> decide ->
    validate -> apply. Since stage reports a frame on a thread stack only as advisory (prepare
    sees steps in flight), the blocking frame check is `validate`, and it must run at the
    apply point: a caller that applies right after stage without it loses the loop-frame
    blocker and swaps code a running loop never leaves. Here validate runs right before
    apply; its running_frame goes through `accept` (as in the gap), changed_since_prepare
    always blocks. Returns {"ok", "stage", "validate", "decision", "apply", "findings"}; ok is
    True only when applied. Never raises."""
    r = stage(module, source, targets, **stage_kw)
    out = {"ok": False, "stage": r, "validate": None, "apply": None}
    d = F.decide(r["findings"], accept)
    out["decision"], out["findings"] = d, d["findings"]
    if not (r["ok"] and d["proceed"]):
        return out
    v = validate(r["staged"])
    frames = F.decide([f for f in v["findings"] if f["check"] == "running_frame"], accept)
    rest = [f for f in v["findings"] if f["check"] != "running_frame"]
    out["validate"] = v
    out["findings"] = d["findings"] + rest + frames["findings"]
    if rest or not frames["proceed"]:
        return out
    a = apply(r["staged"], journal, witness)
    out["apply"], out["ok"] = a, a["ok"]
    out["findings"] += a.get("findings", [])
    return out


def changed_targets(base_src, new_src, trees=None):
    """Planning helper: [[qualname, first line in base_src]] of every definition (function,
    method, nested function, lambda, generator expression) whose AST differs, for qualnames
    defined as many times in both texts. An enclosing function of a changed one is listed too.
    Added and removed definitions are not targets: `stage` finds them. Raises SyntaxError on a
    bad text. `stage(..., targets="auto")` finds the same list from its own parse. trees:
    (base tree, new tree), the caller's `ast.parse` of the two texts, used instead of
    parsing (not changed)."""
    bt, nt = trees if trees is not None else (ast.parse(base_src), ast.parse(new_src))
    return _changed(base_src, qual_index(bt), new_src, qual_index(nt))


def _segment(lines, node):
    """The node's source bytes (from its first decorator), from UTF-8 lines (ast columns
    are byte offsets)."""
    if isinstance(node, (*_DEF, ast.ClassDef)) and node.decorator_list:
        start = node.decorator_list[0]
    else:
        start = node
    sl, sc, el, ec = start.lineno, start.col_offset, node.end_lineno, node.end_col_offset
    if sl == el:
        return lines[sl - 1][sc:ec]
    return b"".join([lines[sl - 1][sc:], *lines[sl:el - 1], lines[el - 1][:ec]])


def _changed(base_src, bq, new_src, pq):
    """`changed_targets` from qual_index of both texts. Equal source text means equal AST, so
    only definitions whose text differs (edited, or with edited comments or layout) pay an
    `ast.dump` compare; on a one-function edit of a large module that is a handful of dumps
    instead of one per definition."""
    bl = base_src.encode().splitlines(keepends=True)
    pl = new_src.encode().splitlines(keepends=True)
    out = []
    for q, bn in bq.items():
        pn = pq.get(q, [])
        if len(pn) == len(bn):
            out += [[q, first_line(b)] for b, p in zip(bn, pn)
                    if _segment(bl, b) != _segment(pl, p) and _dump_code(b) != _dump_code(p)]
    return sorted(out, key=lambda t: t[1])


def _doc(body):
    """The docstring statement heading `body`, or None."""
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0]
    return None


def _dump_code(node):
    """ast.dump of a definition without its docstrings (its own and those of the functions
    and classes nested in it): an edit of a docstring alone changes no behaviour, so it is
    not a target, nor is the function that encloses it (a live run swapped one and then
    reported a false zero_witness)."""
    undo = []
    for n in ast.walk(node):
        b = getattr(n, "body", None)
        if isinstance(n, (*_DEF, ast.ClassDef)) and isinstance(b, list) and _doc(b):
            _set(undo, n, body=b[1:] or [ast.Pass()])
    try:
        return ast.dump(node)
    finally:
        _restore(undo)


def removed_definitions(base_tree, tree):
    """Qualnames the edit deletes: top-level functions and classes, and methods of top-level
    classes that stay (a deleted class is reported once, not per method)."""
    def defs(t):
        out = set()
        for n in t.body:
            if isinstance(n, (*_DEF, ast.ClassDef)):
                out.add(n.name)
            if isinstance(n, ast.ClassDef):
                out |= {f"{n.name}.{m.name}" for m in n.body if isinstance(m, _DEF)}
        return out
    gone = defs(base_tree) - defs(tree)
    return sorted(q for q in gone if "." not in q or q.split(".")[0] not in gone)


def _nested(codes):
    out, stack = [], list(codes)
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(x for x in c.co_consts if hasattr(x, "co_code"))
    return out


def _needs_heap(olds, run, held):
    """Locate needs a heap scan unless a namespace-held function runs the code, or the code is
    a generator expression (no function object ever holds one) inside such a function, at any
    depth of generator expressions. Nested defs and lambdas create function objects at run
    time that others may hold, so they still need the scan."""
    if not olds:
        return True
    for o in olds:
        c = o
        while not held.get(c):
            if c.co_name != "<genexpr>":
                return True
            up = _enclosing(c, run)
            if len(up) != 1:
                return True
            c = up[0]
    return False


def _enclosing(code, run):
    """Running code objects that contain `code` in their constants (enclosing functions)."""
    return [o for codes in run.values() for o in codes
            if any(c is code for c in o.co_consts)]


def _nested_in_order(code):
    """Nested code objects of `code`, depth-first in constant order (not `code` itself)."""
    out = []
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out.append(c)
            out += _nested_in_order(c)
    return out


def closure_pairs(st):
    """{id(new nested code): (new code, old code or None)} for every code `apply` installs:
    the nested codes from which the new code makes function objects at run time (closures,
    lambdas; not generator expressions, whose function never outlives the call). A new
    nested code is paired with the old one of the same qualname and ordinal
    when both have as many of that qualname; else (a nested def the edit adds) with None.
    Added functions' nested codes have no old version."""
    pairs = {}

    def pair(old, new):
        olds = {}
        for c in _nested_in_order(old) if old is not None else ():
            olds.setdefault(c.co_qualname, []).append(c)
        news = [c for c in _nested_in_order(new) if c.co_name != "<genexpr>"]
        count, seen = {}, {}
        for c in news:
            count[c.co_qualname] = count.get(c.co_qualname, 0) + 1
        for c in news:
            q = c.co_qualname
            i = seen[q] = seen.get(q, -1) + 1
            same = olds.get(q, [])
            o = same[i] if len(same) == count[q] else None
            if o is not c:
                pairs.setdefault(id(c), (c, o))

    for fn, code, _, _ in st.swaps:
        pair(fn.__code__, code)
    for _, _, obj in st.adds:
        if isinstance(obj, type):
            for member in vars(obj).values():
                for fn_obj in _member_functions(member):
                    pair(None, fn_obj.__code__)
        else:
            for fn_obj in _member_functions(obj):
                pair(None, fn_obj.__code__)
    return pairs


def _retarget_code_filename(code, filename):
    """Return `code` (and all nested code objects) with `co_filename` pinned to `filename`."""
    consts = tuple(
        _retarget_code_filename(c, filename) if isinstance(c, types.CodeType) else c
        for c in code.co_consts
    )
    return code.replace(co_filename=filename, co_consts=consts)


def _retarget_nested_filename(code, filename):
    """Retarget only nested code objects inside `code` to `filename`, keeping `code`'s own
    `co_filename` intact (preserving `locate.code_id` and `file_namespace` matching) so a
    lazily-built nested `@triton.jit` kernel keeps reading the exact transaction source even
    after later swaps shift lines in `mod.__file__`."""
    if not any(isinstance(c, types.CodeType) for c in code.co_consts):
        return code
    consts = tuple(
        _retarget_code_filename(c, filename) if isinstance(c, types.CodeType) else c
        for c in code.co_consts
    )
    return code.replace(co_consts=consts)


def _retarget_added_nested(obj, filename):
    if isinstance(obj, type):
        for member in vars(obj).values():
            for fn_obj in _member_functions(member):
                fn_obj.__code__ = _retarget_nested_filename(fn_obj.__code__, filename)
    else:
        for fn_obj in _member_functions(obj):
            fn_obj.__code__ = _retarget_nested_filename(fn_obj.__code__, filename)


def _build_adds(st, journal, ver_file=None):
    """Build the definitions the edit adds (stage only planned them), in a scratch namespace:
    their decorators and class bodies run now, at the commit. Returns the namespace of new
    imports and top-level definitions (for deferred defaults)."""
    mod, top = st.mod, dict(st.imports)
    st.adds, st.jit_adds = [], []
    if not st.pending_adds:
        return top
    # an added decorator runs here and may call the new body, which starts with the
    # witness call: bind the no-op stub where the module lacks the name (journaled)
    # (the stub is what undo leaves anyway; at_base ignores the name)
    mod.__dict__.setdefault(W.HIT, W.stub)
    built = {}                         # class -> {name: member added so far}
    for ad in st.pending_adds:
        ns = dict(top)
        if ad["cls"] is not None:      # decorators and defaults see the live class body
            ns[LIVE_BODY] = _LiveBody(getattr(mod, ad["cls"]), top, built.get(ad["cls"]))
        with JIT.seeded_lines(mod.__file__, st.source):    # @triton.jit reads its lines
            exec(_raise(ad), mod.__dict__, ns)
        cls_name, name = ad["cls"], ad["name"]
        if cls_name is None:
            if ver_file:
                _retarget_added_nested(ns[name], ver_file)
            top[name] = ns[name]       # a later added definition may use it
            st.adds.append((mod.__dict__, name, ns[name]))
            if ad["def"] and JIT.unwrap(ns[name])[0] is not None:
                st.jit_adds.append((ns[name], ad["key"]))
        else:
            obj, cls = ns[cls_name].__dict__[name], getattr(mod, cls_name)
            if ver_file:
                _retarget_added_nested(obj, ver_file)
            built.setdefault(cls_name, {})[name] = obj
            _set_class_cell(obj, cls)
            st.adds.append((cls, name, obj))
    return top


def apply(st, journal, witness):
    """Make a staged swap visible, every change through the journal. Never raises: on an
    error it undoes its own entries and returns not-ok. First it journals the nested-code
    pairs (`closure_pairs`), so undo, which runs last for it, also takes back closures the new
    code made while installed."""
    start = len(journal.entries)
    g = st.mod.__dict__
    try:
        mod_file = getattr(st.mod, "__file__", None)
        ver_file = None
        if mod_file and getattr(st, "source", None) is not None and hasattr(journal, "on_undo"):
            lines = st.source.splitlines(True)
            ver_file = f"<fastrsi-live:{mod_file}#{L.text_hash(st.source)}>"
            prev_linecache = linecache.cache.get(mod_file)
            prev_ver = linecache.cache.get(ver_file)
            linecache.cache[mod_file] = (len(st.source), None, lines, mod_file)
            linecache.cache[ver_file] = (len(st.source), None, lines, ver_file)

            def _restore_linecache(
                target_file=mod_file,
                previous=prev_linecache,
                version_file=ver_file,
                previous_ver=prev_ver,
            ):
                if previous is None:
                    linecache.cache.pop(target_file, None)
                else:
                    linecache.cache[target_file] = previous
                if previous_ver is None:
                    linecache.cache.pop(version_file, None)
                else:
                    linecache.cache[version_file] = previous_ver

            journal.on_undo(_restore_linecache)
        if ver_file:
            st.swaps = [
                (fn, _retarget_nested_filename(code, ver_file), d, kw)
                for fn, code, d, kw in st.swaps
            ]
        top = _build_adds(st, journal, ver_file=ver_file)
        journal.record_closures(closure_pairs(st), mod=st.mod)
        for name, value in st.imports.items():
            journal.set_item(g, name, value)
        hook = getattr(witness, "hook", None) or witness.hit
        if g.get(W.HIT) is not hook:
            W.trace_safe(hook)         # torch.compile calls it while tracing, never traces it
            W.trace_safe(W.stub)
            # undo binds the no-op stub, never deletes the name: code compiled or captured
            # while the edit was installed may still call it (NameError killed an engine)
            journal.set_item(g, W.HIT, hook, absent=W.stub)
        for k in st.keys:
            journal.set_item(witness.counts, k, 0)
            witness.capture_counts[k] = 0   # read only for committed keys: not journaled
        for k in st.keys:
            if hasattr(witness, "reset"):
                witness.reset(k)
        for fn, code, d, kw in st.swaps:
            if isinstance(d, _Deferred):
                d, kw = _defaults(st.mod, d.q, d.stub, top)
            journal.set_code(fn, code, d, kw)
        for owner, name, obj in st.adds:
            if isinstance(owner, dict):
                journal.set_item(owner, name, obj)
            else:
                journal.set_attr(owner, name, obj)
        for p, k in st.jits:               # after the code swap: undo restores it first
            JIT.apply(p, journal, witness, k)
        for obj, k in st.jit_adds:
            JIT.arm_added(obj, k, [p for p, _ in st.jits], journal, witness,
                          added=[other for other, _ in st.jit_adds if other is not obj])
        for owner, name in st.removes:     # only reached when removed_definition was accepted
            journal.delete(owner, name)
    except Exception as e:
        u = journal.undo_to(start)
        return F.not_ok(f"apply: {type(e).__name__}: {e}", u["findings"], undo=u)
    return F.ok(changes=len(journal.entries) - start)
