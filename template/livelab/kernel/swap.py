"""Swap: one code-swap primitive, in two steps.

`stage(module, source, targets)` parses the patched module source, compiles only the
definitions it swaps, and plans the swap, with no visible change. Targets omitted ("auto"):
the analysis takes every definition whose AST changed, from its own parse. For each target it
finds the new code object (by qualname and its ordinal
among same-named definitions, so a patch may move lines), adds the witness, and pairs every
running code object of the target and of its enclosing functions with its new version (so
closures made later run the new code too). It runs the checks and returns findings:
`no_live_function`, `not_in_patch`, `base_mismatch`, `outside_changes`, `closure_changed`
(free variables differ), `defaults_not_evaluated`, `running_frame`, `patch_invalid`.
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

Compile units. Prepare runs in the engine's own process while it steps, and `compile` is one C
call that holds the GIL: compiling a 3,800-line module stalls a step by its whole compile time.
So `stage` compiles one unit per changed definition (`compile_units`): the module- or
class-level statement that holds it, inside empty shells of its enclosing classes (same names,
lines and type parameters), with the module's `from __future__` imports. Qualnames, private
name mangling, `__class__` cells and closures come out as in the whole module, so the code
objects are the same (content hash) and the free-variable check compares the same things.
`compile="module"` compiles the whole module instead (the old path, for a before/after).

Analysis off the process. Even per definition, the two parses and the outside-changes compare
(two `ast.dump` calls) are single C calls of 10-20 ms each on a 3,800-line module, and a
thread does not help: it holds the same GIL. So `stage` is split in two. `analyze` is the
text half: it needs only the two texts, the targets and which targets are live, and returns
plain data and code objects. The rest (import, locate, base check against running code,
exec of imports, added definitions and defaults, pairing, running frames) stays here.
`analyzer=ProcessAnalyzer()` runs the text half in a long-lived child process of this
interpreter (code objects travel by `marshal`), so this process holds the GIL only for the
live half; `analyzer=None` runs it here. Both give the same staged code and findings.
"""
import ast
import collections
import copy
import importlib
import importlib.util
import marshal
import os
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types

from . import findings as F
from . import jit as JIT
from . import journal as J
from . import locate as L
from . import witness as W


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
            # evaluated by the enclosing code, so it is named there.
            q = pre + "<genexpr>"
            idx.setdefault(q, []).append(node)
            visit(node.generators[0].iter, pre)
            for x in [node.elt] + node.generators[0].ifs + node.generators[1:]:
                visit(x, q + ".<locals>.")
            for g in node.generators:
                visit(g.target, q + ".<locals>.")
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
        if isinstance(n, _DEF):
            _set(undo, n, body=[ast.Pass()],
                 args=ast.arguments([], [], None, [], [], None, []))
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
            if isinstance(n, (*_DEF, ast.ClassDef)) and pre + n.name in added:
                continue
            if isinstance(n, ast.ClassDef):
                _set(undo, n, body=strip(n.body, pre + n.name + "."))
            out.append(n)
        return out

    _set(undo, tree, body=strip(tree.body, ""))
    return ast.dump(tree)


def _holder(body, node):
    """The statement of `body` that contains `node` (by identity), or None."""
    line = node.lineno
    for st in body:
        if first_line(st) <= line <= st.end_lineno and any(n is node for n in ast.walk(st)):
            return st
    return None


def compile_unit(tree, node):
    """(shells, statement): the module- or class-level statement holding `node`, and the
    classes around it, outermost first. A node in a class header (decorator, base) makes the
    whole class the unit."""
    body, shells = tree.body, []
    while True:
        st = _holder(body, node)
        if st is None:
            raise LookupError(f"line {node.lineno}: definition not found in the tree")
        if st is node or not isinstance(st, ast.ClassDef):
            return shells, st
        inner = _holder(st.body, node)
        if inner is None:                        # in the class header: compile the class
            return shells, st
        shells.append(st)
        body = st.body


def _module_imports(tree):
    """Every import statement of the module scope (also inside top-level if / try / with /
    loops), except `from __future__`. CPython 3.12 compiles `name.attr(...)` differently
    when `name` is bound by an import in the module (no method-call form), so a unit
    without them would give other bytecode."""
    out = []

    def visit(body):
        for n in body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                if not (isinstance(n, ast.ImportFrom) and n.module == "__future__"):
                    out.append(n)
            elif not isinstance(n, (*_DEF, ast.ClassDef)):
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(n, field, None) or [])
                for h in getattr(n, "handlers", None) or []:
                    visit(h.body)
                for c in getattr(n, "cases", None) or []:
                    visit(c.body)
    visit(tree.body)
    return out


def _unit_codes(tree, nodes, filename):
    """The compiled module code of each unit that holds one of `nodes`, and the unit names."""
    futures = [n for n in tree.body
               if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    codes, seen, names = [], set(), []
    imports = _module_imports(tree)
    for node in nodes:
        shells, st = compile_unit(tree, node)
        if id(st) in seen:
            continue
        seen.add(id(st))
        body = [st]
        for cls in reversed(shells):
            body = [ast.copy_location(ast.ClassDef(cls.name, [], [], body, [],
                                                   type_params=list(cls.type_params)), cls)]
        mod = ast.fix_missing_locations(ast.Module(futures + imports + body, []))
        codes.append(_compile(mod, filename, "exec", dont_inherit=True))
        names.append(".".join([c.name for c in shells] +
                              [getattr(st, "name", None) or type(st).__name__]))
    return codes, names


def compile_units(tree, nodes, filename):
    """Compile only the units that hold `nodes` (see the module docstring). Returns
    ({(qualname, firstlineno): [code]}, {code: enclosing code}, [unit names])."""
    codes, names = _unit_codes(tree, nodes, filename)
    out = ({}, {})
    for c in codes:
        L.code_tree(c, None, out)
    return out[0], out[1], names


class Staged:
    """What `apply` will do. Built by `stage`; nothing in it is visible to the engine yet."""

    def __init__(self, mod):
        self.mod = mod
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
        self.jits = []           # (key, [JIT objects holding it], [JIT objects calling it])
        self.source = None       # the patched module text (a JIT rebuild reads it)
        self.new_fields = []     # fields only the new __init__ sets (analysis + live check)


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

    def __init__(self, live=None, imports=None):
        self.maps = [vars(live)] if live is not None else []
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


def _defaults_stub(q, node):
    """The def with an empty body, inside a scratch class named like the innermost class of
    the qualname (same name mangling; a class body sees only its own class's names), whose
    body reads the live class (`_shell`), compiled: running it in the module's globals
    evaluates the new defaults as the live class body would."""
    stub = copy.deepcopy(node)
    stub.body, stub.decorator_list = [ast.Pass()], []
    parts = q.split(".")
    body = [stub] if len(parts) == 1 else [_shell(parts[-2], [stub])]
    return _try_compile(ast.fix_missing_locations(ast.Module(body, [])), "<defaults>")


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


def _set_class_cell(fn, cls):
    if fn.__closure__ and "__class__" in fn.__code__.co_freevars:
        fn.__closure__[fn.__code__.co_freevars.index("__class__")].cell_contents = cls


def analyze(base_src, source, targets, filename, key_module, live, compile="definitions",
            trees=None):
    """The text half of `stage`: everything that needs only the two texts, the targets and
    which targets have a live function (`live`: {(qualname, line): bool}). It parses both
    texts, matches targets, adds the witness, compiles the units, finds added, removed and
    imported names, and compares the blanked trees. It touches no live object, so it can run
    in another process (`ProcessAnalyzer`). targets "auto" (or None): the changed definitions
    of the two texts, found from the trees parsed here (as `changed_targets`), so the caller
    needs no parse of its own; `live` then holds every live (qualname, line) of the module.
    Returns plain data and code objects (marshal can carry them): {"error"} when a text does
    not parse, else {"targets": [per-target dict in order], "resolved": [[qualname, line]]
    (the targets analyzed), "base_units", "new_units", "compiled", "new_error", "line_map",
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
        return _analyze_trees(base_src, source, targets, filename, key_module, live, compile,
                              base_tree, tree, timings, undo)
    finally:
        _restore(undo)


def _analyze_trees(base_src, source, targets, filename, key_module, live, compile,
                   base_tree, tree, timings, undo):
    bq, pq = qual_index(base_tree), qual_index(tree)
    auto = targets is None or targets == "auto"
    if auto:                 # every target found here differs by AST: no dump per target below
        t = time.perf_counter()
        targets = _changed(base_src, bq, source, pq)
        timings["targets_s"] = round(time.perf_counter() - t, 6)
    whole = compile == "module"
    t = time.perf_counter()
    if whole:
        base_units = [_compile(base_tree, filename, "exec", dont_inherit=True)]
    else:
        nodes = [bq[q][i] for q, line in targets
                 for i in [_ordinal(bq, q, int(line))] if i is not None]
        base_units, _ = _unit_codes(base_tree, nodes, filename)
    compile_s = time.perf_counter() - t
    line_map = {}
    for q, nodes in pq.items():
        if len(nodes) == len(bq.get(q, [])):
            for n, b in zip(nodes, bq[q]):
                line_map[(q, first_line(n))] = first_line(b)

    # 1. targets: match base and patch, add the witness (the live checks are the caller's)
    tres, todo, aliases = [], [], []
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
        _inject(node, k, undo)
        todo.append((q, i))
        tres.append({"status": "ok", "i": i, "changed": changed, "new_line": first_line(node)})

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
        if isinstance(n, _DEF):
            _inject(n, L.key_of(key_module, q, first_line(n)), undo)

    # 2b. removed definitions
    removed = removed_definitions(base_tree, tree)

    # 4. top-level imports, compiled; the caller runs those the module lacks
    imports = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            names = [(a.asname or a.name).split(".")[0] for a in n.names]
            imp = {"names": names, **_try_compile(ast.Module([n], []), filename)}
            if isinstance(n, ast.ImportFrom):     # resolvable against an overlay (see stage)
                imp["from"] = [n.level, n.module, [[a.name, a.asname] for a in n.names]]
            imports.append(imp)
    t = time.perf_counter()
    new_units, compiled, new_error = [], None, None
    try:
        if whole:
            new_units = [_compile(ast.fix_missing_locations(tree), filename, "exec",
                                  dont_inherit=True)]
            compiled = ["<module>"]
        else:
            new_units, compiled = _unit_codes(tree, [pq[q][i] for q, i in todo], filename)
    except Exception as e:
        new_error = f"{type(e).__name__}: {e}"
    timings["compile_s"] = round(compile_s + time.perf_counter() - t, 6)
    if compiled is not None:
        timings["compiled"] = compiled
    # 5. added definitions, compiled for the caller's scratch namespace
    adds = []
    for q, cls_name, n in added:
        body = [n] if cls_name is None else [_shell(cls_name, [n])]
        adds.append({"q": q, "cls": cls_name, "name": n.name, "def": isinstance(n, _DEF),
                     "line": first_line(n),
                     **_try_compile(ast.fix_missing_locations(ast.Module(body, [])), filename)})
    # 6. new defaults: a stub the caller evaluates in the module's globals
    for (q, i), r in zip(todo, [r for r in tres if r["status"] == "ok"]):
        bnode, pnode = bq[q][i], pq[q][i]
        if "<locals>" not in q and isinstance(pnode, _DEF) and (
                ast.dump(ast.Tuple(bnode.args.defaults + bnode.args.kw_defaults))
                != ast.dump(ast.Tuple(pnode.args.defaults + pnode.args.kw_defaults))):
            r["defaults"] = _defaults_stub(q, pnode)

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
            "aliases": aliases, "new_fields": new_fields}


def _self_name(fn):
    args = fn.args.posonlyargs + fn.args.args
    return args[0].arg if args else None


def _self_attrs(node, me, ctx):
    """Attribute names X of `me.X` in node with the given context (ast.Load / ast.Store)."""
    if me is None:
        return set()
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == me and isinstance(n.ctx, ctx)}


def _class_names(cls):
    out = set()
    for n in cls.body:
        if isinstance(n, (*_DEF, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out


def _field_stmt(st, me):
    """X when `st` is a top-level `me.X = value` (or annotated) statement, else None."""
    t = st.targets[0] if isinstance(st, ast.Assign) and len(st.targets) == 1 else \
        st.target if isinstance(st, ast.AnnAssign) and st.value is not None else None
    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == me:
        return t.attr
    return None


def _loads(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _backfill_code(cls_name, init, i, filename):
    """The new __init__'s statement i (`self.X = ...`) and the earlier top-level statements
    that bind the local names it reads, compiled as `__fastrsi_backfill__(self, *, params)`
    inside a scratch class named like the class (same name mangling). Params: the __init__
    parameters the statements read, with the __init__'s defaults."""
    me = _self_name(init)
    body, need = [init.body[i]], _loads(init.body[i].value) - {me}
    for st in reversed(init.body[:i]):
        if isinstance(st, (ast.Assign, ast.AnnAssign)):
            tg = st.targets if isinstance(st, ast.Assign) else [st.target]
            bound = {n.id for t in tg for n in ast.walk(t) if isinstance(n, ast.Name)}
            if bound & need and st.value is not None:
                body.insert(0, st)
                need = (need - bound) | (_loads(st.value) - {me})
    a = init.args
    pos = a.posonlyargs + a.args
    defaults = dict(zip([x.arg for x in pos[len(pos) - len(a.defaults):]], a.defaults))
    defaults.update({x.arg: d for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None})
    names = [x.arg for x in pos[1:] + a.kwonlyargs]
    params = [n for n in names if n in need]
    fn = ast.FunctionDef(
        "__fastrsi_backfill__", ast.arguments(
            [], [ast.arg(me)], None, [ast.arg(p) for p in params],
            [copy.deepcopy(defaults[p]) if p in defaults else None for p in params], None, []),
        [copy.deepcopy(x) for x in body] + [ast.Return(ast.Constant(None))], [], None,
        type_params=[])
    ast.copy_location(fn, init)
    cls = ast.ClassDef(cls_name, [], [], [fn], [], type_params=[])
    ast.copy_location(cls, init)
    return {"params": params, "defaults": [p for p in params if p in defaults],
            "statements": [ast.unparse(x) for x in body],
            **_try_compile(ast.fix_missing_locations(ast.Module([cls], [])), filename)}


def _new_fields(base_tree, tree, meths, filename):
    """[{cls, field, readers, init_line, backfill}] for each attribute that a changed or added
    method of a top-level class reads on `self`, that the edited __init__ sets, and that no
    method of the base class sets nor any class body defines. backfill: the compiled
    statements (`_backfill_code`), or {"error"} when the field is not set by a top-level
    statement of the new __init__."""
    bcls = {n.name: n for n in base_tree.body if isinstance(n, ast.ClassDef)}
    pcls = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    out = []
    for c, nodes in meths.items():
        b, p = bcls.get(c), pcls.get(c)
        init = next((m for m in (p.body if p else ()) if isinstance(m, _DEF)
                     and m.name == "__init__"), None)
        if b is None or init is None:
            continue
        old = set()
        for m in b.body:
            if isinstance(m, _DEF):
                old |= _self_attrs(m, _self_name(m), ast.Store)
        old |= _class_names(b) | _class_names(p)
        me = _self_name(init)
        sets = _self_attrs(init, me, ast.Store)
        readers = {}
        for m in nodes:
            if m.name == "__init__":
                continue
            for x in _self_attrs(m, _self_name(m), ast.Load):
                if x in sets and x not in old:
                    readers.setdefault(x, []).append(f"{c}.{m.name}")
        for x in sorted(readers):
            i = next((j for j, st in enumerate(init.body) if _field_stmt(st, me) == x), None)
            out.append({"cls": c, "field": x, "readers": sorted(set(readers[x])),
                        "init_line": first_line(init),
                        "backfill": {"error": f"{c}.__init__ sets {x} inside a block, not in a "
                                     "top-level statement"} if i is None
                        else _backfill_code(c, init, i, filename)})
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
          base_source=None, own_codes=(), compile="definitions", analyzer=None, overlay=None,
          line_maps=(),
          trees=None):
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
    compile: "definitions" (compile only the units holding the targets) or "module" (the
    whole base and patched modules, as before).
    analyzer: runs `analyze` (the text half: parse, compile, compare); None runs it here,
    a `ProcessAnalyzer` in a child process, so this process holds the GIL only for the live
    half. Both give the same staged code and findings. When the analyzer fails, the analysis
    runs here and an `analyze_fallback` finding says why and what it cost.
    overlay: {module name: {name: object}}, names that modules staged earlier in the same
    transaction add (`staged_names`): a `from m import name` of this patch takes the staged
    object when the live module m does not have the name yet. Nothing is made visible.
    trees: (base tree, patch tree), `ast.parse` of exactly the base text (`base_source`, or
    the module's file) and `source`, for a caller that parsed them already (its own planner):
    the analysis uses them instead of parsing again and leaves them unchanged. Used when the
    analysis runs in this process (analyzer None, or a fallback); a `ProcessAnalyzer` child
    parses its own copy (trees cannot cross the pipe, and that parse holds no GIL here).
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
           "filename": mod.__file__, "key_module": kname, "live": live, "compile": compile}
    t = time.perf_counter()
    a = _analyze(req, analyzer, timings, fs, kname, trees)
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
                a = _analyze(req, analyzer, timings, fs, kname, trees)
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
    st.line_map = a["line_map"]
    st.source = source
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
            fs.append(F.finding("not_in_patch", "no definition at this line in the base file",
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
        # 4. imports the module lacks, run in a scratch namespace (a name another module of
        # this transaction adds comes from the overlay: that module is not applied yet)
        for imp in a["imports"]:
            if any(x not in mod.__dict__ for x in imp["names"]):
                ns = _from_overlay(mod, imp, overlay)
                if ns is None:
                    ns = {}
                    exec(_raise(imp), mod.__dict__, ns)
                st.imports.update({x: v for x, v in ns.items() if x not in mod.__dict__})
        if a["new_error"] is not None:
            raise _Failed(a["new_error"])
        new_idx, parent = {}, {}
        for c in a["new_units"]:
            L.code_tree(c, None, (new_idx, parent))
        timings["compiled"] = at["compiled"]
        timings["compile_s"] = at["compile_s"]
        timings["parse_compile_s"] = round(timings["parse_s"] + timings["compile_s"], 6)
        # 5. added definitions, built in a scratch namespace
        for ad in a["added"]:
            ns = dict(st.imports)
            if ad["cls"] is not None:      # decorators and defaults see the live class body
                ns[LIVE_BODY] = _LiveBody(getattr(mod, ad["cls"]), st.imports)
            exec(_raise(ad), mod.__dict__, ns)
            q, cls_name, name = ad["q"], ad["cls"], ad["name"]
            if cls_name is None:
                st.adds.append((mod.__dict__, name, ns[name]))
            else:
                obj, cls = ns[cls_name].__dict__[name], getattr(mod, cls_name)
                _set_class_cell(getattr(obj, "__func__", obj), cls)
                st.adds.append((cls, name, obj))
            if ad["def"]:
                k = L.key_of(kname, q, ad["line"])
                st.keys.append(k)
                st.items[k] = {"added": True}
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
    live = W.live_codes(suspended=True) if susp else set(on_stack)
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
        if bad:
            continue
        defaults = kwdefaults = J.KEEP
        if "defaults" in r:
            try:
                defaults, kwdefaults = _defaults(mod, q, r["defaults"], st.imports)
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
        jits = JIT.holders(mod, st.functions[k]) if st.functions[k] else []
        if jits:
            # the engine runs what the JIT compiled from the text, not this bytecode: apply
            # rebuilds the JIT object in place (jit.py) and counts its launches
            deps = JIT.dependents(mod, q.rsplit(".", 1)[-1], skip=jits)
            st.jits.append((k, jits, deps))
            st.items[k]["jit"] = {"holders": [JIT.kind(o) for o in jits],
                                  "dependents": len(deps), "witness": "launches"}
            fs.append({**F.finding(
                "jit_rebuild", f"{len(jits)} JIT object(s) ({JIT.kind(jits[0])}) run this "
                f"function; {len(deps)} other JIT object(s) of the module call it",
                "a function the engine runs as Python bytecode", subject=k), "advisory": True})
    for o, (c, d, kw) in plan.items():
        for fn in held.get(o, []):
            st.swaps.append((fn, c, d, kw))
            st.expect.append((fn, o, L.content_hash(o)))

    # 6b. fields only the new __init__ sets, read by new code: live instances lack them
    for nf in a.get("new_fields") or ():
        cls = getattr(mod, nf["cls"], None)
        x = nf["field"]
        attr = f"_{nf['cls'].lstrip('_')}{x}" if x.startswith("__") and not x.endswith("__") \
            else x
        if not isinstance(cls, type) or hasattr(cls, attr):
            continue                       # a class-level default covers old instances
        bf = nf["backfill"]
        how = (f"backfill offered: {'; '.join(bf['statements'])}" + (
            f" (with {', '.join(bf['params'])} resolved from each instance or its owners)"
            if bf["params"] else "")) if "code" in bf else \
            f"no backfill: {bf.get('error')}"
        fs.append(F.finding(
            "new_fields", f"{', '.join(nf['readers'])} read self.{x}, which only the edited "
            f"{nf['cls']}.__init__ sets; instances built before the swap lack it; {how}",
            f"every live {nf['cls']} has {x}", subject=f"{kname}:{nf['cls']}.{x}"))
        st.new_fields.append({**nf, "cls_obj": cls, "attr": attr,
                              "subject": f"{kname}:{nf['cls']}.{x}"})

    # 7. anything else changed? (compared by the analysis)
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
    """{(qualname, line): does a function object run this code, or directly enclose it?}"""
    live = {}
    for q, line in targets:
        olds = run.get((q, int(line)), [])
        live[(q, int(line))] = any(held.get(o) for o in olds) or any(
            held.get(p) for o in olds for p in _enclosing(o, run))
    return live


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
    text it was compiled from. Returns [[qualname, base_source line, live line]]."""
    out = []
    if not line_maps:
        return out
    for q, line in list(line_maps[0]):
        if run.get((q, line)):
            continue
        cur = line
        for m in line_maps:
            cur = m.get((q, cur))
            if cur is None:
                break
            if run.get((q, cur)):
                run[(q, line)] = run[(q, cur)]
                out.append([q, line, cur])
                break
    return out


def _live_keys(held, mapped=()):
    """`_live_of` for every code of the module at once: a held code and the codes it
    directly contains; `mapped` ([qualname, line, live line]) adds the line-mapped keys."""
    live = {}
    for o in held:
        live[(o.co_qualname, o.co_firstlineno)] = True
        for c in o.co_consts:
            if hasattr(c, "co_code"):
                live[(c.co_qualname, c.co_firstlineno)] = True
    for q, line, ll in mapped:
        if live.get((q, ll)):
            live[(q, line)] = True
    return live


def _analyze(req, analyzer, timings, fs, kname, trees=None):
    """Run the analysis with `analyzer`; if it fails, here, with an `analyze_fallback`
    finding that says why and what it cost. `trees` (the caller's parse) are used only
    here, in this process."""
    if analyzer is not None:
        info = {}
        try:
            a = analyzer(req, info=info) if isinstance(analyzer, ProcessAnalyzer) \
                else analyzer(req)
            timings["analyze"] = "process"
            return a
        except Exception as e:           # the child is gone: analyze here, and say so
            timings["analyze_fallback"] = why = f"{type(e).__name__}: {e}"
        finally:                         # a spawn or a boot wait inside this prepare
            if info.get("spawned"):
                timings["analyze_spawned"] = True
            if info.get("boot_wait_s"):
                timings["analyze_boot_wait_s"] = info["boot_wait_s"]
        t = time.perf_counter()
        a = analyze(**req, trees=trees)
        timings["analyze"] = "inline"
        cost = time.perf_counter() - t
        fs.append(F.finding(
            "analyze_fallback", f"the analyzer failed ({why}); analyzed in this process "
            f"instead, holding its GIL for {cost * 1e3:.1f} ms of parse, compile and "
            "compare", "the analysis in the analyzer's child process", subject=kname))
        return a
    timings["analyze"] = "inline"
    return analyze(**req, trees=trees)


def staged_names(st):
    """{module name: {name: object}} a staged module adds at module level (definitions and
    imports): the `overlay` for the next module of the same transaction."""
    names = dict(st.imports)
    names.update({n: o for owner, n, o in st.adds if owner is st.mod.__dict__})
    return {st.mod.__name__: names}


def _from_overlay(mod, imp, overlay):
    """The names a `from m import ...` binds, when m is in the overlay and the overlay has
    one of the names; else None (run the import as usual)."""
    if not overlay or not imp.get("from") or imp.get("error") is not None:
        return None
    level, name, pairs = imp["from"]
    if any(n == "*" for n, _ in pairs):
        return None
    try:
        target = importlib.util.resolve_name("." * level + (name or ""), mod.__package__)
    except Exception:
        return None
    staged = overlay.get(target)
    if not staged or not any(n in staged for n, _ in pairs):
        return None
    tmod = importlib.import_module(target)
    ns = {}
    for n, asname in pairs:
        if n in staged:
            ns[asname or n] = staged[n]
        elif hasattr(tmod, n):
            ns[asname or n] = getattr(tmod, n)
        else:
            ns[asname or n] = importlib.import_module(f"{target}.{n}")
    return ns


def _invalid(msg, kname):
    return F.not_ok(msg, [F.finding("patch_invalid", msg, "an importable module and a patch "
                                    "that parses", subject=kname)], staged=None, items={})


_CHILD = r"""
import importlib, importlib.util, marshal, os, struct, sys, traceback
# the kernel by file location, under a private name: the parent may have loaded it under any
# package name (or from a directory of another name), which the child cannot import by name
_d = sys.argv[1]
_spec = importlib.util.spec_from_file_location(
    "_fastrsi_analyzer_kernel", os.path.join(_d, "__init__.py"), submodule_search_locations=[_d])
_pkg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _pkg
_spec.loader.exec_module(_pkg)
S = importlib.import_module(_spec.name + ".swap")
inp, out = sys.stdin.buffer, sys.stdout.buffer
out.write(b"R")            # ready: the kernel is imported (the parent times the boot)
out.flush()
def read(n):
    b = bytearray()
    while len(b) < n:
        x = inp.read(n - len(b))
        if not x:
            raise EOFError
        b += x
    return bytes(b)
while True:
    try:
        req = marshal.loads(read(struct.unpack("<Q", read(8))[0]))
    except EOFError:
        break
    try:
        rep = S.analyze(**req)
    except BaseException:
        rep = {"crash": traceback.format_exc()[-2000:]}
    b = marshal.dumps(rep)
    out.write(struct.pack("<Q", len(b)) + b)
    out.flush()
"""


class ProcessAnalyzer:
    """Runs `analyze` in one long-lived child Python process (this interpreter, same
    optimization level, `-I`), so the parses, compiles and `ast.dump` compares of a prepare
    do not hold this process's GIL: the engine loop keeps stepping. Requests and replies go
    over pipes as `marshal` data (code objects included). Thread-safe (one request at a time).
    The child exits when this process closes the pipe. The child loads the kernel from this
    file's directory under a private name, whatever package name the parent loaded it as.

    Warm start: `start()` spawns the child and returns at once (idempotent: a live child is
    kept); the child boots (interpreter, kernel import) in the background and writes one
    ready byte. A `LocalParticipant` given an analyzer calls `start()` when it is created, so
    the first prepare does not pay the spawn (1.5-2.2 s on an engine host). `wait_ready(s)`
    blocks until the child is ready; `state()` says whether it is, without blocking. A child
    that died is respawned lazily, by the next call; `last_call` says whether that call
    spawned it and how long it waited for the boot (stage puts both in its timings). A failed
    call (child gone, or no answer within `timeout_s`: the child is killed) raises, with the
    child's last stderr line when it died; `stage` then analyzes in-process and reports an
    `analyze_fallback` finding (and the reason in its timings)."""

    def __init__(self, python=None, timeout_s=60.0):
        self.python = python or sys.executable
        self.timeout_s = timeout_s     # a child that does not answer by then is killed
        self._p = None
        self._err = None
        self._ready = False            # the child's ready byte was read
        self._t_spawn = None
        self.boot_s = None             # spawn to ready byte, when this process saw it
        self.spawns = 0                # children spawned (the first, then respawns)
        self.last_call = {}            # {"spawned": bool, "boot_wait_s": float}
        self._lock = threading.Lock()

    def _spawn(self):
        """Spawn the child unless a live one exists. Returns True when it spawned."""
        if self._p is not None and self._p.poll() is None:
            return False
        if self._p is not None or self._err is not None:
            self._kill()               # a dead child: close its pipes and stderr file
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        # stderr to a file, not a pipe: a child that warns a lot (SyntaxWarning while
        # compiling a patch) must never block on a full pipe; its tail explains a death
        self._err = tempfile.TemporaryFile()
        self._ready, self.boot_s = False, None
        self._t_spawn = time.monotonic()
        self._p = subprocess.Popen(
            [self.python, "-I", *["-O"] * sys.flags.optimize, "-c", _CHILD, kernel_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, close_fds=True)
        self.spawns += 1
        return True

    def _await_ready(self, p, end):
        """Read the child's ready byte (by monotonic time `end`). Returns seconds waited."""
        if self._ready:
            return 0.0
        t = time.monotonic()
        if self._read(p, 1, end) != b"R":
            raise RuntimeError("the analyzer child did not start its protocol")
        self._ready = True
        self.boot_s = time.monotonic() - self._t_spawn
        return time.monotonic() - t

    def _stderr_tail(self):
        try:
            self._err.seek(0, os.SEEK_END)
            self._err.seek(max(0, self._err.tell() - 2000))
            lines = self._err.read().decode(errors="replace").strip().splitlines()
            return lines[-1] if lines else ""
        except Exception:
            return ""

    def start(self):
        """Spawn the child now, if none is alive; does not wait for its boot. Returns True
        when this call spawned a child, False when a live one was kept (callers record it)."""
        with self._lock:
            return self._spawn()

    def wait_ready(self, timeout_s=None):
        """Start the child if needed and wait until it is ready (at most `timeout_s`, default
        the call timeout). True when ready; False when the child died or is still booting."""
        with self._lock:
            self._spawn()
            end = time.monotonic() + (self.timeout_s if timeout_s is None else timeout_s)
            try:
                self._await_ready(self._p, end)
            except (EOFError, TimeoutError, RuntimeError, OSError):
                return False
            return True

    def state(self):
        """{"alive", "ready", "spawns", "boot_s", "pid"} without blocking (a ready byte that
        already arrived is read)."""
        with self._lock:
            p = self._p
            alive = p is not None and p.poll() is None
            if alive and not self._ready:
                try:
                    self._await_ready(p, time.monotonic())
                except (EOFError, TimeoutError, RuntimeError, OSError):
                    pass
            return {"alive": alive, "ready": alive and self._ready, "spawns": self.spawns,
                    "boot_s": None if self.boot_s is None else round(self.boot_s, 6),
                    "pid": p.pid if alive else None}

    def __call__(self, req, info=None):
        """The analysis of `req` in the child. `info` (a dict), if given, gets "spawned"
        (this call started the child: a cold or respawned analyzer) and "boot_wait_s" (time
        this call waited for the child's boot), filled even when the call raises."""
        info = {} if info is None else info
        with self._lock:
            spawned = self._spawn()
            p = self._p
            info.update(spawned=spawned, boot_wait_s=0.0)
            self.last_call = info
            try:
                end = time.monotonic() + self.timeout_s
                info["boot_wait_s"] = round(self._await_ready(p, end), 6)
                b = marshal.dumps(req)
                p.stdin.write(struct.pack("<Q", len(b)) + b)
                p.stdin.flush()
                n = struct.unpack("<Q", self._read(p, 8, end))[0]
                body = self._read(p, n, end)
            except (EOFError, BrokenPipeError) as e:    # the child died: say what it said
                try:
                    p.wait(1)
                except Exception:
                    pass
                tail = self._stderr_tail()
                self._kill()
                raise EOFError(f"the analyzer child exited (code {p.returncode})"
                               + (f": {tail}" if tail else "")) from e
            except Exception:
                self._kill()
                raise
        rep = marshal.loads(body)
        if "crash" in rep:
            raise RuntimeError(f"analyzer: {rep['crash']}")
        return rep

    @staticmethod
    def _read(p, n, end):
        fd, out = p.stdout.fileno(), bytearray()
        while len(out) < n:
            left = end - time.monotonic()
            if not select.select([fd], [], [], max(0.0, left))[0]:
                raise TimeoutError("the analyzer child did not answer in time")
            b = os.read(fd, n - len(out))
            if not b:
                raise EOFError("the analyzer child exited")
            out += b
        return bytes(out)

    def _kill(self):
        p, self._p = self._p, None
        self._ready = False
        if p is not None:
            if p.poll() is None:
                p.kill()
            p.wait()
            for f in (p.stdin, p.stdout):
                try:
                    f.close()
                except Exception:
                    pass
        if self._err is not None:
            self._err.close()
            self._err = None

    def close(self):
        with self._lock:
            p, self._p = self._p, None
            self._ready = False
        if p is not None:
            p.stdin.close()
            p.wait()
            p.stdout.close()
        if self._err is not None:
            self._err.close()
            self._err = None


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
    for owner, name, _ in st.adds:
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
    """ast.dump of a definition without its own docstring: an edit of the docstring alone
    changes no behaviour, so it is not a target (a live run swapped one and then reported a
    false zero_witness)."""
    body = getattr(node, "body", None)
    d = _doc(body) if isinstance(body, list) else None
    if d is None:
        return ast.dump(node)
    node.body = body[1:] or [ast.Pass()]
    try:
        return ast.dump(node)
    finally:
        node.body = body


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
        f = getattr(obj, "__func__", obj)
        if isinstance(f, types.FunctionType):
            pair(None, f.__code__)
    return pairs


def _launch_hook(witness, key):
    """A JIT pre-run hook: one witness call of `key` per launch."""
    def hook(*_a, **_k):
        witness.hit(key)
    return hook


def apply(st, journal, witness):
    """Make a staged swap visible, every change through the journal. Never raises: on an
    error it undoes its own entries and returns not-ok. First it journals the nested-code
    pairs (`closure_pairs`), so undo, which runs last for it, also takes back closures the new
    code made while installed."""
    start = len(journal.entries)
    g = st.mod.__dict__
    try:
        journal.record_closures(closure_pairs(st))
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
        for fn, code, d, kw in st.swaps:
            journal.set_code(fn, code, d, kw)
        for owner, name, obj in st.adds:
            if isinstance(owner, dict):
                journal.set_item(owner, name, obj)
            else:
                journal.set_attr(owner, name, obj)
        for owner, name in st.removes:     # only reached when removed_definition was accepted
            journal.delete(owner, name)
        for k, jits, deps in st.jits:      # JIT targets: rebuild from the new text, in place
            for o in jits:
                journal.record_call(JIT.snapshot(o), f"jit rebuild {k}")
                JIT.rebuild(o, st.source, st.mod.__file__)
                JIT.count_launches(o, _launch_hook(witness, k), k)
            for o in deps:                 # they inline the target when they compile
                journal.record_call(JIT.snapshot(o), f"jit reset (calls {k})")
                JIT.reset(o)
                JIT.count_launches(o, _launch_hook(witness, k), k)
    except Exception as e:
        u = journal.undo_to(start)
        return F.not_ok(f"apply: {type(e).__name__}: {e}", u["findings"], undo=u)
    return F.ok(changes=len(journal.entries) - start)
