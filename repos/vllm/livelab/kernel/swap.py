"""Swap, part 1: the change analysis (the text half of `stage`).

`analyze(base_src, source, targets, filename, key_module, live)` needs only the two module
texts, the targets and which targets have a live function; it touches no live object. It
parses both texts, takes every definition whose AST changed (targets "auto", the same list as
`changed_targets`) or matches the explicit [(qualname, first line)] by qualname and ordinal
(so a patch may move lines), adds the witness call to each new body, compiles the patched
module, and finds added, removed and imported names. It compares the two trees with target
bodies, added definitions, imports and docstrings blanked: any other difference is
`outside_changes` (a code swap would not apply it). It returns plain data and code objects.
A docstring edit alone is not a target: it changes no behaviour.

`stage` (the live half: locate, base check against running code, scratch builds, pairing,
running frames), `validate` and `apply` come in the next PR, on top of this one.
"""
import ast
import copy
import time

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


_compile = compile          # `stage` has a parameter named compile


LIVE_BODY = "__fastrsi_class_body__"


def _shell(name, body):
    """A scratch class statement whose body reads the live class's names: its metaclass is
    the `LIVE_BODY` name, which the live half binds to a `_LiveBody`."""
    return ast.ClassDef(name, [], [ast.keyword("metaclass", ast.Name(LIVE_BODY, ast.Load()))],
                        body, [], type_params=[])


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
    tres, todo = [], []
    for q, line in targets:
        line = int(line)
        k = L.key_of(key_module, q, line)
        i = _ordinal(bq, q, line)
        if i is None:
            tres.append({"status": "no_definition"})
            continue
        if not live.get((q, line)):
            tres.append({"status": "no_live_function"})
            continue
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
        if "<locals>" not in q and isinstance(pnode, _DEF) and (
                ast.dump(ast.Tuple(bnode.args.defaults + bnode.args.kw_defaults))
                != ast.dump(ast.Tuple(pnode.args.defaults + pnode.args.kw_defaults))):
            r["defaults"] = _defaults_stub(q, pnode, futures)

    # 7. anything else changed? Last, because blanking edits the two trees (logged in undo).
    t = time.perf_counter()
    outside = _blank(base_tree, todo, set(removed), undo) != _blank(
        tree, todo, {q for q, _, _ in added}, undo)
    timings["compare_s"] = round(time.perf_counter() - t, 6)
    return {"targets": tres, "resolved": [[q, int(line)] for q, line in targets],
            "base_units": base_units, "new_units": new_units,
            "new_error": new_error, "line_map": line_map, "added": adds, "removed": removed,
            "imports": imports, "outside_changes": outside, "timings": timings}


def _try_compile(mod, filename):
    try:
        return {"code": _compile(mod, filename, "exec")}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


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


