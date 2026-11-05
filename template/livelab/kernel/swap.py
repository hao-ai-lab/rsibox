"""Swap: one code-swap primitive, in two steps.

`stage(module, source, targets)` parses the patched module source, compiles only the
definitions it swaps, and plans the swap, with no visible change: for each target it finds the new code object (by qualname and its ordinal
among same-named definitions, so a patch may move lines), adds the witness, and pairs every
running code object of the target and of its enclosing functions with its new version (so
closures made later run the new code too). It runs the checks and returns findings:
`no_live_function`, `not_in_patch`, `base_mismatch`, `outside_changes`, `closure_changed`
(free variables differ), `defaults_not_evaluated`, `running_frame`, `patch_invalid`.
New imports the patch needs are executed into a scratch namespace; new functions, methods and
classes are built in a scratch namespace (methods get the real class for `super()`).

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
import copy
import importlib
import marshal
import os
import select
import struct
import subprocess
import sys
import threading
import time

from . import findings as F
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


def _ordinal(idx, q, line):
    hits = [i for i, n in enumerate(idx.get(q, [])) if first_line(n) == line]
    return hits[0] if len(hits) == 1 else None


def _blank(tree, targets, added):
    """The module AST with target bodies and signatures, added defs and imports removed: two
    blanked trees differ only if the patch changes something a code swap does not apply."""
    idx = qual_index(tree)
    for q, i in targets:
        n = idx[q][i]
        if isinstance(n, _DEF):
            n.body, n.args = [ast.Pass()], ast.arguments([], [], None, [], [], None, [])
        elif isinstance(n, ast.GeneratorExp):    # keep the first iterable (enclosing code)
            g = n.generators[0]
            n.elt, n.generators = ast.Constant(None), [
                ast.comprehension(ast.Name("_", ast.Store()), g.iter, [], 0)]
        else:
            n.body, n.args = ast.Constant(None), ast.arguments([], [], None, [], [], None, [])

    def strip(body, pre):
        out = []
        for n in body:
            if isinstance(n, (ast.Import, ast.ImportFrom)) and not pre:
                continue
            if isinstance(n, (*_DEF, ast.ClassDef)) and pre + n.name in added:
                continue
            if isinstance(n, ast.ClassDef):
                n.body = strip(n.body, pre + n.name + ".")
            out.append(n)
        return out

    tree.body = strip(tree.body, "")
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


_compile = compile          # `stage` has a parameter named compile


def _defaults_stub(q, node):
    """The def with an empty body inside scratch classes for the qualname's class part,
    compiled: running it in the module's globals evaluates the new defaults."""
    stub = copy.deepcopy(node)
    stub.body, stub.decorator_list = [ast.Pass()], []
    body = [stub]
    for cls in reversed(q.split(".")[:-1]):
        body = [ast.ClassDef(cls, [], [], body, [], type_params=[])]
    return _try_compile(ast.fix_missing_locations(ast.Module(body, [])), "<defaults>")


def _defaults(mod, q, stub):
    """Evaluate a def's new defaults in a scratch namespace (module globals). Returns
    (defaults, kwdefaults) or raises."""
    ns = {}
    exec(_raise(stub), mod.__dict__, ns)
    parts = q.split(".")
    obj = ns[parts[0]]
    for p in parts[1:]:
        obj = obj.__dict__[p]
    return obj.__defaults__, obj.__kwdefaults__


def _set_class_cell(fn, cls):
    if fn.__closure__ and "__class__" in fn.__code__.co_freevars:
        fn.__closure__[fn.__code__.co_freevars.index("__class__")].cell_contents = cls


def analyze(base_src, source, targets, filename, key_module, live, compile="definitions"):
    """The text half of `stage`: everything that needs only the two texts, the targets and
    which targets have a live function (`live`: {(qualname, line): bool}). It parses both
    texts, matches targets, adds the witness, compiles the units, finds added, removed and
    imported names, and compares the blanked trees. It touches no live object, so it can run
    in another process (`ProcessAnalyzer`). Returns plain data and code objects (marshal can
    carry them): {"error"} when a text does not parse, else
    {"targets": [per-target dict in order], "base_units", "new_units", "compiled",
     "new_error", "line_map", "added", "removed", "imports", "outside_changes", "timings"}."""
    timings, t0 = {}, time.perf_counter()
    try:
        base_tree = ast.parse(base_src, filename)
        tree = ast.parse(source, filename)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "timings": timings}
    timings["parse_s"] = round(time.perf_counter() - t0, 6)
    bq, pq = qual_index(base_tree), qual_index(tree)
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
        changed = ast.dump(bq[q][i]) != ast.dump(node)
        W.inject(node, k)
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
            W.inject(n, L.key_of(key_module, q, first_line(n)))

    # 2b. removed definitions
    removed = removed_definitions(base_tree, tree)

    # 4. top-level imports, compiled; the caller runs those the module lacks
    imports = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            names = [(a.asname or a.name).split(".")[0] for a in n.names]
            imports.append({"names": names, **_try_compile(ast.Module([n], []), filename)})
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
        body = [n] if cls_name is None else [ast.ClassDef(cls_name, [], [], [n], [],
                                                          type_params=[])]
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

    # 7. anything else changed? Last, because blanking edits the two trees in place.
    t = time.perf_counter()
    outside = _blank(base_tree, todo, set(removed)) != _blank(tree, todo, {q for q, _, _ in added})
    timings["compare_s"] = round(time.perf_counter() - t, 6)
    return {"targets": tres, "base_units": base_units, "new_units": new_units,
            "new_error": new_error, "line_map": line_map, "added": adds, "removed": removed,
            "imports": imports, "outside_changes": outside, "timings": timings}


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


def stage(module, source, targets, base_hash=None, key_module=None, scan="auto",
          base_source=None, own_codes=(), compile="definitions", analyzer=None):
    """module: the live module's name; source: the full patched module text; targets:
    [(qualname, firstlineno in the base)]; base_hash: text_hash of the source the patch was
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
    half. Both give the same staged code and findings.
    Returns {"ok", "staged", "items", "findings", "timings"}; ok is False only when nothing
    can be staged (bad patch, import error). Each target's item says whether its definition
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
    run, held = L.index_module(mod, None if scan == "always" else L.module_functions(mod))
    timings["scan"] = "heap" if scan == "always" else "namespace"
    if scan == "auto" and any(_needs_heap(run.get((q, int(line)), ()), run, held)
                              for q, line in targets):
        run, held = L.index_module(mod)
        timings["scan"] = "heap"
    timings["locate_s"] = round(time.perf_counter() - t, 6)
    live = {}
    for q, line in targets:
        olds = run.get((q, int(line)), [])
        live[(q, int(line))] = any(held.get(o) for o in olds) or any(
            held.get(p) for o in olds for p in _enclosing(o, run))
    req = {"base_src": base_src, "source": source,
           "targets": [[q, int(line)] for q, line in targets], "filename": mod.__file__,
           "key_module": kname, "live": live, "compile": compile}
    t = time.perf_counter()
    a = None
    if analyzer is not None:
        try:
            a = analyzer(req)
            timings["analyze"] = "process"
        except Exception as e:           # the child is gone: analyze here, and say so
            timings["analyze_fallback"] = f"{type(e).__name__}: {e}"
    if a is None:
        a = analyze(**req)
        timings["analyze"] = "inline"
    timings["analyze_s"] = round(time.perf_counter() - t, 6)
    at = a["timings"]
    if "error" in a:
        return _invalid(a["error"], kname)
    timings["parse_s"] = at["parse_s"]
    st = Staged(mod)
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
        # 4. imports the module lacks, run in a scratch namespace
        for imp in a["imports"]:
            if any(x not in mod.__dict__ for x in imp["names"]):
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
            ns = {}
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
    live = W.live_codes(suspended=susp)
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
                defaults, kwdefaults = _defaults(mod, q, r["defaults"])
            except Exception as e:
                fs.append(F.finding("defaults_not_evaluated", _err(e),
                                    "the new defaults evaluate", subject=k))
        for o, c in chain:
            plan.setdefault(o, (c, J.KEEP, J.KEEP))
        for o in olds:
            plan[o] = (plan[o][0], defaults, kwdefaults)
            if o in live:
                fs.append(F.finding("running_frame", "a frame of the old code is running now",
                                    "no running frame", subject=k))
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

    # 7. anything else changed? (compared by the analysis)
    if a["outside_changes"]:
        fs.append(F.finding("outside_changes", "the module differs outside the target bodies "
                            "(beyond added definitions and imports)",
                            "only target bodies, added definitions and imports differ",
                            subject=kname))
    timings["compare_s"] = at["compare_s"]
    timings["stage_s"] = round(time.perf_counter() - t0, 6)
    for f in fs:        # what a standing accept matches on: the checked code
        item = st.items.get(f.get("subject")) if isinstance(f.get("subject"), str) else None
        if item:
            f["code_hash"] = item["new_hash"]
        elif f.get("subject") == kname:      # about the whole module: its new text
            f["code_hash"] = L.text_hash(source)
    return F.ok(staged=st, items=st.items, findings=fs, timings=timings)


def _invalid(msg, kname):
    return F.not_ok(msg, [F.finding("patch_invalid", msg, "an importable module and a patch "
                                    "that parses", subject=kname)], staged=None, items={})


_CHILD = r"""
import importlib, marshal, struct, sys, traceback
sys.path.insert(0, sys.argv[1])
S = importlib.import_module(sys.argv[2] + ".swap")
inp, out = sys.stdin.buffer, sys.stdout.buffer
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
    The child exits when this process closes the pipe. `start()` spawns it ahead of the first
    prepare; a call also spawns it when needed. A failed call (child gone, or no answer within
    `timeout_s`: the child is killed) raises; `stage` then analyzes in-process and records
    `analyze_fallback` in its timings."""

    def __init__(self, python=None, timeout_s=60.0):
        self.python = python or sys.executable
        self.timeout_s = timeout_s     # a child that does not answer by then is killed
        self._p = None
        self._lock = threading.Lock()

    def _spawn(self):
        if self._p is not None and self._p.poll() is None:
            return
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pkg = __name__.rpartition(".")[0]
        self._p = subprocess.Popen(
            [self.python, "-I", *["-O"] * sys.flags.optimize, "-c", _CHILD, root, pkg],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)

    def start(self):
        with self._lock:
            self._spawn()
        return self

    def __call__(self, req):
        with self._lock:
            self._spawn()
            p = self._p
            try:
                b = marshal.dumps(req)
                p.stdin.write(struct.pack("<Q", len(b)) + b)
                p.stdin.flush()
                end = time.monotonic() + self.timeout_s
                n = struct.unpack("<Q", self._read(p, 8, end))[0]
                body = self._read(p, n, end)
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
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                raise TimeoutError("the analyzer child did not answer in time")
            b = os.read(fd, n - len(out))
            if not b:
                raise EOFError("the analyzer child exited")
            out += b
        return bytes(out)

    def _kill(self):
        p, self._p = self._p, None
        if p is not None:
            p.kill()
            p.wait()

    def close(self):
        with self._lock:
            p, self._p = self._p, None
        if p is not None:
            p.stdin.close()
            p.wait()


def validate(st):
    """Phase 2, inside the gap: is what `stage` saw still true? Every function to swap must
    still run the exact code object staged against (identity and hash), and every name to add
    must still be absent. Cheap: no heap scan, no compile. Never raises."""
    fs = []
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


def changed_targets(base_src, new_src):
    """Planning helper: [[qualname, first line in base_src]] of every definition (function,
    method, nested function, lambda) whose AST differs, for qualnames defined as many times in
    both texts. An enclosing function of a changed one is listed too. Added and removed
    definitions are not targets: `stage` finds them. Raises SyntaxError on a bad text."""
    bq, pq = qual_index(ast.parse(base_src)), qual_index(ast.parse(new_src))
    out = []
    for q, bn in bq.items():
        pn = pq.get(q, [])
        if len(pn) == len(bn):
            out += [[q, first_line(b)] for b, p in zip(bn, pn) if ast.dump(b) != ast.dump(p)]
    return sorted(out, key=lambda t: t[1])


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


def apply(st, journal, witness):
    """Make a staged swap visible, every change through the journal. Never raises: on an
    error it undoes its own entries and returns not-ok."""
    start = len(journal.entries)
    g = st.mod.__dict__
    try:
        for name, value in st.imports.items():
            journal.set_item(g, name, value)
        if getattr(g.get(W.HIT), "__self__", None) is not witness:
            journal.set_item(g, W.HIT, witness.hit)
        for k in st.keys:
            journal.set_item(witness.counts, k, 0)
        for fn, code, d, kw in st.swaps:
            journal.set_code(fn, code, d, kw)
        for owner, name, obj in st.adds:
            if isinstance(owner, dict):
                journal.set_item(owner, name, obj)
            else:
                journal.set_attr(owner, name, obj)
        for owner, name in st.removes:     # only reached when removed_definition was accepted
            journal.delete(owner, name)
    except Exception as e:
        u = journal.undo_to(start)
        return F.not_ok(f"apply: {type(e).__name__}: {e}", u["findings"], undo=u)
    return F.ok(changes=len(journal.entries) - start)
