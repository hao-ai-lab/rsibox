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
"""
import ast
import copy
import importlib
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


def compile_units(tree, nodes, filename):
    """Compile only the units that hold `nodes` (see the module docstring). Returns
    ({(qualname, firstlineno): [code]}, {code: enclosing code}, [unit names])."""
    futures = [n for n in tree.body
               if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    out, seen, names = ({}, {}), set(), []
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
        L.code_tree(compile(mod, filename, "exec", dont_inherit=True), None, out)
        names.append(".".join([c.name for c in shells] +
                              [getattr(st, "name", None) or type(st).__name__]))
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


def _defaults(mod, q, node):
    """Evaluate a def's new defaults in a scratch namespace (module globals, scratch classes
    for the qualname's class part). Returns (defaults, kwdefaults) or raises."""
    stub = copy.deepcopy(node)
    stub.body, stub.decorator_list = [ast.Pass()], []
    parts = q.split(".")
    body = [stub]
    for cls in reversed(parts[:-1]):
        body = [ast.ClassDef(cls, [], [], body, [], type_params=[])]
    ns = {}
    exec(_compile(ast.fix_missing_locations(ast.Module(body, [])), "<defaults>", "exec"),
         mod.__dict__, ns)
    obj = ns[parts[0]]
    for p in parts[1:]:
        obj = obj.__dict__[p]
    return obj.__defaults__, obj.__kwdefaults__


def _set_class_cell(fn, cls):
    if fn.__closure__ and "__class__" in fn.__code__.co_freevars:
        fn.__closure__[fn.__code__.co_freevars.index("__class__")].cell_contents = cls


def stage(module, source, targets, base_hash=None, key_module=None, scan="auto",
          base_source=None, own_codes=(), compile="definitions"):
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
        base_tree = ast.parse(base_src, mod.__file__)
        tree = ast.parse(source, mod.__file__)
    except Exception as e:
        return F.not_ok(f"{type(e).__name__}: {e}", [F.finding(
            "patch_invalid", f"{type(e).__name__}: {e}", "an importable module and a patch "
            "that parses", subject=kname)], staged=None, items={})
    timings["parse_s"] = round(time.perf_counter() - t0, 6)
    st = Staged(mod)
    if base_hash is not None and L.text_hash(base_src) != base_hash:
        fs.append(F.finding("base_mismatch", f"base text hash {L.text_hash(base_src)}",
                            f"plan base hash {base_hash}", subject=kname))
    # Locate without a heap scan when the module namespace finds every target (top-level
    # functions and methods); scan (one get_objects + a sliced pass) only for nested targets
    # or targets the namespace does not hold.
    t = time.perf_counter()
    run, held = L.index_module(mod, None if scan == "always" else L.module_functions(mod))
    timings["scan"] = "heap" if scan == "always" else "namespace"
    if scan == "auto" and any("<locals>" in q or not any(
            held.get(o) for o in run.get((q, int(line)), ())) for q, line in targets):
        run, held = L.index_module(mod)
        timings["scan"] = "heap"
    timings["locate_s"] = round(time.perf_counter() - t, 6)
    bq, pq = qual_index(base_tree), qual_index(tree)
    whole = compile == "module"
    t = time.perf_counter()
    if whole:
        base_idx = L.code_tree(_compile(base_tree, mod.__file__, "exec", dont_inherit=True))[0]
    else:
        nodes = [bq[q][i] for q, line in targets
                 for i in [_ordinal(bq, q, int(line))] if i is not None]
        base_idx, _, _ = compile_units(base_tree, nodes, mod.__file__)
    compile_s = time.perf_counter() - t
    line_map = {}
    for q, nodes in pq.items():
        if len(nodes) == len(bq.get(q, [])):
            for n, b in zip(nodes, bq[q]):
                line_map[(q, first_line(n))] = first_line(b)

    # 1. targets: find them in base, running code and patch; add the witness
    todo, changed, own = [], {}, {id(c) for c in own_codes}
    for q, line in targets:
        line = int(line)
        k = L.key_of(kname, q, line)
        i = _ordinal(bq, q, line)
        olds = run.get((q, line), [])
        if i is None:
            fs.append(F.finding("not_in_patch", "no definition at this line in the base file",
                                "one definition with this qualname and line", subject=k))
            continue
        if not any(held.get(o) for o in olds) and not any(
                held.get(p) for o in olds for p in _enclosing(o, run)):
            fs.append(F.finding("no_live_function", "no function object runs this code or "
                                "encloses it", "a function object to swap", subject=k))
            continue
        if len(pq.get(q, [])) != len(bq[q]):
            fs.append(F.finding("not_in_patch", f"{len(pq.get(q, []))} definitions named {q} "
                                f"in the patch, {len(bq[q])} in the base",
                                "the same number", subject=k))
            continue
        base_code = base_idx[(q, line)][0]
        for o in olds:
            if id(o) not in own and L.content_hash(o) != L.content_hash(base_code):
                fs.append(F.finding("base_mismatch", f"running code {L.content_hash(o)}",
                                    f"the file's code {L.content_hash(base_code)}", subject=k))
        node = pq[q][i]
        changed[k] = ast.dump(bq[q][i]) != ast.dump(node)
        W.inject(node, k)
        todo.append((q, line, i, k, olds))

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
            W.inject(n, L.key_of(kname, q, first_line(n)))

    # 2b. removed definitions: a live edit that only adds and swaps leaves them callable, so a
    # forgotten caller works live and fails after a restart (vLLM #61843). A finding; on
    # accept, apply deletes them through the journal.
    removed = removed_definitions(base_tree, tree)
    for q in removed:
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
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [(a.asname or a.name).split(".")[0] for a in n.names]
                if any(x not in mod.__dict__ for x in names):
                    ns = {}
                    exec(_compile(ast.Module([n], []), mod.__file__, "exec"), mod.__dict__, ns)
                    st.imports.update({x: v for x, v in ns.items() if x not in mod.__dict__})
        t = time.perf_counter()
        if whole:
            new_idx, parent = L.code_tree(_compile(ast.fix_missing_locations(tree),
                                                   mod.__file__, "exec", dont_inherit=True))
            timings["compiled"] = ["<module>"]
        else:
            new_idx, parent, timings["compiled"] = compile_units(
                tree, [pq[q][i] for q, _, i, _, _ in todo], mod.__file__)
        timings["compile_s"] = round(compile_s + time.perf_counter() - t, 6)
        timings["parse_compile_s"] = round(timings["parse_s"] + timings["compile_s"], 6)
        # 5. added definitions, built in a scratch namespace
        for q, cls_name, n in added:
            body = [n] if cls_name is None else [ast.ClassDef(cls_name, [], [], [n], [],
                                                              type_params=[])]
            ns = {}
            exec(_compile(ast.fix_missing_locations(ast.Module(body, [])), mod.__file__,
                          "exec"), mod.__dict__, ns)
            if cls_name is None:
                st.adds.append((mod.__dict__, n.name, ns[n.name]))
            else:
                obj, cls = ns[cls_name].__dict__[n.name], getattr(mod, cls_name)
                _set_class_cell(getattr(obj, "__func__", obj), cls)
                st.adds.append((cls, n.name, obj))
            if isinstance(n, _DEF):
                k = L.key_of(kname, q, first_line(n))
                st.keys.append(k)
                st.items[k] = {"added": True}
    except Exception as e:
        fs.append(F.finding("patch_invalid", f"{type(e).__name__}: {e}",
                            "the patch compiles and its imports run", subject=kname))
        return F.not_ok(f"{type(e).__name__}: {e}", fs, staged=None, items={})

    # 6. pair old and new code: the target and every enclosing function with live holders
    t = time.perf_counter()
    live = W.live_codes(suspended=scan == "always" or W.suspendable(
        [o for *_, olds in todo for o in olds]))
    timings["live_frames_s"] = round(time.perf_counter() - t, 6)
    plan = {}
    for q, line, i, k, olds in todo:
        new = new_idx[(q, first_line(pq[q][i]))][0]
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
        bnode, pnode = bq[q][i], pq[q][i]
        if "<locals>" not in q and isinstance(pnode, _DEF) and (
                ast.dump(ast.Tuple(bnode.args.defaults + bnode.args.kw_defaults))
                != ast.dump(ast.Tuple(pnode.args.defaults + pnode.args.kw_defaults))):
            try:
                defaults, kwdefaults = _defaults(mod, q, pnode)
            except Exception as e:
                fs.append(F.finding("defaults_not_evaluated", f"{type(e).__name__}: {e}",
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

    # 7. anything else changed? Last, because blanking edits the two trees in place (they are
    # not parsed again).
    t = time.perf_counter()
    tq = [(q, i) for q, _, i, _, _ in todo]
    if _blank(base_tree, tq, set(removed)) != _blank(tree, tq, {q for q, _, _ in added}):
        fs.append(F.finding("outside_changes", "the module differs outside the target bodies "
                            "(beyond added definitions and imports)",
                            "only target bodies, added definitions and imports differ",
                            subject=kname))
    timings["compare_s"] = round(time.perf_counter() - t, 6)
    timings["stage_s"] = round(time.perf_counter() - t0, 6)
    for f in fs:        # what a standing accept matches on: the checked code
        item = st.items.get(f.get("subject")) if isinstance(f.get("subject"), str) else None
        if item:
            f["code_hash"] = item["new_hash"]
    return F.ok(staged=st, items=st.items, findings=fs, timings=timings)


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
