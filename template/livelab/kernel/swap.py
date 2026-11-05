"""Swap: one code-swap primitive, in two steps.

`stage(module, source, targets)` compiles the patched module source and plans the swap, with
no visible change: for each target it finds the new code object (by qualname and its ordinal
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
"""
import ast
import copy
import importlib

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
        self.items = {}          # key -> JSON summary


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
    exec(compile(ast.fix_missing_locations(ast.Module(body, [])), "<defaults>", "exec"),
         mod.__dict__, ns)
    obj = ns[parts[0]]
    for p in parts[1:]:
        obj = obj.__dict__[p]
    return obj.__defaults__, obj.__kwdefaults__


def _set_class_cell(fn, cls):
    if fn.__closure__ and "__class__" in fn.__code__.co_freevars:
        fn.__closure__[fn.__code__.co_freevars.index("__class__")].cell_contents = cls


def stage(module, source, targets, base_hash=None, key_module=None):
    """module: the live module's name; source: the full patched module text; targets:
    [(qualname, firstlineno in the base file)]; base_hash: text_hash of the source the patch
    was made against (checked against this module's file); key_module: the module name used in
    witness keys (participants that load a module under another name pass the shared one).
    Returns {"ok", "staged", "items", "findings"}; ok is False only when nothing can be staged
    (bad patch, import error). Never raises."""
    kname = key_module or module
    fs = []
    try:
        mod = importlib.import_module(module)
        with open(mod.__file__) as fh:
            base_src = fh.read()
        base_tree = ast.parse(base_src, mod.__file__)
        tree = ast.parse(source, mod.__file__)
    except Exception as e:
        return F.not_ok(f"{type(e).__name__}: {e}", [F.finding(
            "patch_invalid", f"{type(e).__name__}: {e}", "an importable module and a patch "
            "that parses", subject=kname)], staged=None, items={})
    st = Staged(mod)
    if base_hash is not None and L.text_hash(base_src) != base_hash:
        fs.append(F.finding("base_mismatch", f"module file hash {L.text_hash(base_src)}",
                            f"plan base hash {base_hash}", subject=kname))
    base_idx = L.code_tree(compile(base_tree, mod.__file__, "exec", dont_inherit=True))[0]
    run, held = L.index_module(mod)
    bq, pq = qual_index(base_tree), qual_index(tree)
    line_map = {}
    for q, nodes in pq.items():
        if len(nodes) == len(bq.get(q, [])):
            for n, b in zip(nodes, bq[q]):
                line_map[(q, first_line(n))] = first_line(b)

    # 1. targets: find them in base, running code and patch; add the witness
    todo = []
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
            if L.content_hash(o) != L.content_hash(base_code):
                fs.append(F.finding("base_mismatch", f"running code {L.content_hash(o)}",
                                    f"the file's code {L.content_hash(base_code)}", subject=k))
        node = pq[q][i]
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

    # 3. anything else changed?
    tq = [(q, i) for q, _, i, _, _ in todo]
    if _blank(ast.parse(base_src), tq, set()) != _blank(ast.parse(source), tq,
                                                        {q for q, _, _ in added}):
        fs.append(F.finding("outside_changes", "the module differs outside the target bodies "
                            "(beyond added definitions and imports)",
                            "only target bodies, added definitions and imports differ",
                            subject=kname))

    try:
        # 4. imports the module lacks, run in a scratch namespace
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [(a.asname or a.name).split(".")[0] for a in n.names]
                if any(x not in mod.__dict__ for x in names):
                    ns = {}
                    exec(compile(ast.Module([n], []), mod.__file__, "exec"), mod.__dict__, ns)
                    st.imports.update({x: v for x, v in ns.items() if x not in mod.__dict__})
        new_idx, parent = L.code_tree(compile(ast.fix_missing_locations(tree), mod.__file__,
                                              "exec", dont_inherit=True))
        # 5. added definitions, built in a scratch namespace
        for q, cls_name, n in added:
            body = [n] if cls_name is None else [ast.ClassDef(cls_name, [], [], [n], [],
                                                              type_params=[])]
            ns = {}
            exec(compile(ast.fix_missing_locations(ast.Module(body, [])), mod.__file__,
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
    live = W.live_codes()
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
        st.items[k] = {"base_hash": L.content_hash(olds[0]) if olds else None,
                       "new_hash": L.content_hash(new),
                       "functions": sum(len(held.get(o, [])) for o in olds),
                       "via": sum(len(held.get(o, [])) for o, _ in chain)}
    for o, (c, d, kw) in plan.items():
        for fn in held.get(o, []):
            st.swaps.append((fn, c, d, kw))
    return F.ok(staged=st, items=st.items, findings=fs)


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
    except Exception as e:
        u = journal.undo_to(start)
        return F.not_ok(f"apply: {type(e).__name__}: {e}", u["findings"], undo=u)
    return F.ok(changes=len(journal.entries) - start)
