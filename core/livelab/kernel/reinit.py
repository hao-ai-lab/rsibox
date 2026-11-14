"""Constructor field re-initialization: the in-place rebuild of a constructor edit that only
changes what fresh value a field starts with.

A swapped `__init__` reaches only instances built after the swap; the engine built its
long-lived objects (the scheduler, managers, model layers) at start-up, so they keep what the
old constructor set (`constructor_edit`, vLLM #61323). Most such edits need a restart of the
box that holds the objects. One shape does not: vLLM #62872 changes

    self.grammar_compile_error_reqs: set[str] = set()      # base
    self.grammar_compile_error_reqs: dict[str, str] = {}   # edit

and the edited methods index the field as a dict. Swapping the methods alone leaves the live
scheduler's set in place (the new `[...] = ` on it raises on the first grammar error), and
undo leaves whatever dict the new code put there (the base `.add` then raises): the commit
read as done (0.38 s) but neither direction was safe. A forked restart of the whole engine
fixed it at 16.6 s eager / 51.2 s with CUDA graphs.

The rule (`plan` at stage, `resolve` at prepare and again at validate, `apply` at commit):

- **Eligible edit** (`plan`, from the two ASTs alone). The constructor's signature and
  decorators are unchanged, and its body differs from the base only in statements
  `self.F = E` (or `self.F: T = E`) whose old and new values are *pure fresh values*: a
  literal, a display of pure values (`[]`, `{}`, `(1, 2)`), or a call of a builtin
  container or scalar type on pure values (`set()`, `dict()`, `list()`, ...). A statement
  may also be added (a new field with a pure value). F is assigned once in each body, and no
  other statement of the new body reads `self.F`. Anything else (a statement removed, a
  call, a name, a parameter) is not eligible and the edit blocks as before.
- **Every live instance, found** (`resolve`): one heap scan (`locate.all_objects`, frozen
  heap included) for objects whose type is the class or a subclass. A subclass that defines
  its own constructor is refused: it may set the field again. For each instance the field
  must still hold exactly the old fresh value (same type, equal: `set() == set()`); an
  instance whose field holds state (`{"r1"}`) is refused, since replacing it would drop that
  state. An added field must be absent. With no live instance, nothing live keeps old state
  and only the swap is needed. A subclass with its own constructor (vLLM's AsyncScheduler:
  `super().__init__(*args, **kwargs)`, then fields of its own) is covered when its source
  neither sets nor reads the field and uses no `setattr` / `__dict__`; else refused.
- **Applied in the commit's gap** (`apply`), after the code swap, through the journal: each
  instance's field gets a fresh value of the new expression, exactly what the new constructor
  would set. The validate step (also in the gap, just before) scans again, so an instance
  built between prepare and the gap is covered. The constructor's witness counts one call per
  instance re-initialized: its changed statements ran on that object.
- **Undo** puts the object the field held before the commit back (journal), after checking
  the field still holds only a fresh new value; if the new code left state there (a request
  id), undo still restores and reports it (`undo_incomplete`), so the transaction reads
  partial, not at base.

Cost: two heap scans (prepare, off the loop; validate, in the gap) plus one assignment per
instance. `resolve` reports `scan_s`.
"""
import ast
import builtins
import difflib
import inspect
import textwrap
import time

from . import locate as L

ELIGIBLE = ("__init__", "__post_init__")      # __new__ builds the object: always blocks
PURE_TYPES = ("set", "frozenset", "dict", "list", "tuple", "int", "float", "str", "bool",
              "bytes")
_DEF = (ast.FunctionDef, ast.AsyncFunctionDef)


def _pure(node):
    """A pure fresh value: evaluating it twice gives equal, independent objects."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_pure(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _pure(k) for k in node.keys) and \
            all(_pure(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Name) and node.func.id in PURE_TYPES and \
            all(_pure(a) for a in node.args) and all(
                k.arg is not None and _pure(k.value) for k in node.keywords)
    return False


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _field_assign(stmt, me):
    """(field, value node) of `me.F = E` / `me.F: T = E`, else None."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        target, value = stmt.target, stmt.value
    else:
        return None
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
            and target.value.id == me:
        return target.attr, value
    return None


def _stores(body, me, field):
    return sum(1 for s in body for n in ast.walk(s)
               if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and n.value.id == me and n.attr == field and isinstance(n.ctx, ast.Store))


def _reads(stmts, me, field):
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                    and n.value.id == me and n.attr == field and isinstance(n.ctx, ast.Load):
                return True
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute) and \
                    isinstance(n.target.value, ast.Name) and n.target.value.id == me and \
                    n.target.attr == field:
                return True
    return False


def plan(base_def, patch_def):
    """{"fields": [{"field", "old", "new"}]} when the constructor edit only changes (or adds)
    pure fresh field values; else {"why": reason}. `old` is None for an added field. Call it
    on the two definitions before any witness is injected."""
    if not isinstance(base_def, _DEF) or not isinstance(patch_def, _DEF):
        return {"why": "not a function definition"}
    if base_def.name not in ELIGIBLE:
        return {"why": f"{base_def.name} builds the object itself"}
    if ast.dump(base_def.args) != ast.dump(patch_def.args) or \
            [ast.dump(d) for d in base_def.decorator_list] != \
            [ast.dump(d) for d in patch_def.decorator_list]:
        return {"why": "the signature or decorators changed"}
    args = patch_def.args.posonlyargs + patch_def.args.args
    me = args[0].arg if args else None
    if me is None:
        return {"why": "no self parameter"}
    ob, nb = base_def.body, patch_def.body
    od, nd = [ast.dump(s) for s in ob], [ast.dump(s) for s in nb]
    fields, touched = [], set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, od, nd, autojunk=False) \
            .get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete" or (tag == "replace" and i2 - i1 != j2 - j1):
            return {"why": "a statement of the base constructor is removed or rewritten"}
        olds = ob[i1:i2] if tag == "replace" else [None] * (j2 - j1)
        for o, n in zip(olds, nb[j1:j2]):
            nf = _field_assign(n, me)
            if nf is None or not _pure(nf[1]):
                return {"why": f"line {n.lineno}: not `{me}.<field> = <literal or empty "
                               "container>`"}
            if o is None:
                if _stores(ob, me, nf[0]):
                    return {"why": f"{me}.{nf[0]} is also set elsewhere in the base"}
                old = None
            else:
                of = _field_assign(o, me)
                if of is None or of[0] != nf[0] or not _pure(of[1]):
                    return {"why": f"line {n.lineno}: the base statement is not a pure "
                                   f"assignment of {me}.{nf[0]}"}
                if _stores(ob, me, of[0]) != 1:
                    return {"why": f"{me}.{of[0]} is set more than once in the base"}
                old = ast.unparse(of[1])
            if _stores(nb, me, nf[0]) != 1:
                return {"why": f"{me}.{nf[0]} is set more than once in the edit"}
            fields.append({"field": nf[0], "old": old, "new": ast.unparse(nf[1])})
            touched.add(nf[0])
    if not fields:
        return {"why": "no field value changed"}
    for f in touched:
        rest = [s for s in nb if _field_assign(s, me) is None or _field_assign(s, me)[0] != f]
        if _reads(rest, me, f):
            return {"why": f"another statement of the new constructor reads {me}.{f}"}
    return {"fields": fields}


def _value(src, glb):
    """Evaluate a pure expression with the real builtins, refusing a module that shadows one
    of the names it uses (the constructor would call the module's, not the builtin)."""
    tree = ast.parse(src, mode="eval")
    shadowed = sorted(n for n in _names(tree) if n in (glb or {}))
    if shadowed:
        raise ValueError(f"the module defines {shadowed}, which the value calls")
    env = {n: getattr(builtins, n) for n in PURE_TYPES}
    return eval(compile(tree, "<reinit>", "eval"), {"__builtins__": env})


def _same(live, fresh):
    return type(live) is type(fresh) and live == fresh


def _subclasses(cls):
    out, todo = set(), [cls]
    while todo:
        c = todo.pop()
        if c not in out:
            out.add(c)
            todo += c.__subclasses__()
    return out


def _subclass_why(c, meth, names):
    """None when the subclass's own constructor cannot touch the fields; else why."""
    fn = vars(c)[meth]
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError) as e:
        return f"{c.__qualname__} defines its own {meth} and its source is unreadable ({e})"
    node = tree.body[0]
    if not isinstance(node, _DEF):
        return f"{c.__qualname__}.{meth} is not a plain function"
    args = node.args.posonlyargs + node.args.args
    me = args[0].arg if args else None
    if {"setattr", "__dict__", "vars"} & ({n.id for n in ast.walk(node)
                                           if isinstance(n, ast.Name)}
                                          | {n.attr for n in ast.walk(node)
                                             if isinstance(n, ast.Attribute)}):
        return f"{c.__qualname__}.{meth} sets attributes indirectly (setattr / __dict__)"
    for x in names:
        if _stores(node.body, me, x) or _reads(node.body, me, x):
            return f"{c.__qualname__}.{meth} sets or reads {me}.{x} itself"
    return None


def resolve(cls, meth, fields, glb):
    """Find every live instance of cls and check each field holds the old fresh value.
    Returns {"ok", "instances": [obj], "why", "scan_s", "count"}."""
    t = time.perf_counter()
    try:
        news = {f["field"]: _value(f["new"], glb) for f in fields}
        olds = {f["field"]: _value(f["old"], glb) for f in fields if f["old"] is not None}
    except Exception as e:
        return {"ok": False, "why": f"{type(e).__name__}: {e}", "instances": [],
                "scan_s": 0.0, "count": 0}
    del news
    kinds = _subclasses(cls)
    objs = L.all_objects()
    try:
        found = [o for o in L.sliced(objs) if type(o) in kinds]
    finally:
        del objs
    scan_s = round(time.perf_counter() - t, 6)

    def no(why):
        return {"ok": False, "why": why, "instances": [], "scan_s": scan_s,
                "count": len(found)}
    subs = {}
    for o in found:
        for c in type(o).__mro__:
            if c is cls:
                break
            if meth in vars(c):
                if c not in subs:
                    subs[c] = _subclass_why(c, meth, [f["field"] for f in fields])
                if subs[c]:
                    return no(subs[c])
        d = getattr(o, "__dict__", None)
        if not isinstance(d, dict):
            return no(f"a {type(o).__qualname__} has no instance __dict__")
        for f in fields:
            x = f["field"]
            if any(x in vars(c) and hasattr(type(vars(c)[x]), "__set__")
                   for c in type(o).__mro__):
                return no(f"{x} is a descriptor on {type(o).__qualname__}")
            if f["old"] is None:
                if x in d:
                    return no(f"a live {type(o).__qualname__} already has {x}")
            elif x not in d:
                return no(f"a live {type(o).__qualname__} has no {x}")
            elif not _same(d[x], olds[x]):
                return no(f"a live {type(o).__qualname__}.{x} holds state "
                          f"({type(d[x]).__name__} of {len(d[x]) if hasattr(d[x], '__len__') else '?'}"
                          f", not a fresh {f['old']}): replacing it would drop that state")
    return {"ok": True, "instances": found, "why": None, "scan_s": scan_s,
            "count": len(found)}


def apply(journal, instances, fields, glb, label):
    """Give each instance's fields a fresh new value (journaled). Undo checks first that the
    fields still hold only fresh new values; else it restores anyway and reports it."""
    for o in instances:
        for f in fields:
            journal.set_attr(o, f["field"], _value(f["new"], glb))

    def check_before_restore():
        held = []
        for o in instances:
            for f in fields:
                v = vars(o).get(f["field"])
                if not _same(v, _value(f["new"], glb)):
                    held.append(f"{type(o).__qualname__}.{f['field']}")
        if held:
            raise ValueError(f"{label}: the new code left state in {held}; undo restored the "
                             "base value and dropped it")
    journal.record_call(check_before_restore, f"reinit check {label}")
    return len(instances)
