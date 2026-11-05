"""Journal: one ordered undo log per participant, with transaction marks.

Every change the kernel makes goes through a Journal method that records the inverse first.
Entries are typed:

| kind     | change                                   | inverse                         |
|----------|------------------------------------------|---------------------------------|
| `code`   | fn.__code__, __defaults__, __kwdefaults__ | the same three, old objects     |
| `attr`   | setattr on an attribute that existed      | setattr back to the old object  |
| `added`  | an attribute or dict key that did not exist | delete it                     |
| `object` | container[key] = value on an existing key | put the old object back         |

`mark()` returns the log length; `undo_to(mark)` pops entries back to it, newest first, so the
base comes back exactly: the same code objects, the same attribute objects, added names gone.

Repair cells (recipe R10, "the live lab is a notebook"): a cell that fixes state by hand is a
change too. Either call `record_attr(obj, name)` before assigning, or run the cell inside
`with journal.repair({"engine": engine}) as r:`. The context snapshots the declared objects'
attributes and journals every attribute the cell set, added or deleted on them. What it cannot
undo (an attribute changed on an object the cell did not declare, or a list/dict/set value
changed in place) becomes a `repair_not_undoable` finding in `r.findings`, not an exception.
"""
import types

from . import findings as F

MISSING = object()   # "this attribute or key did not exist"
KEEP = object()      # set_code: leave defaults / kwdefaults as they are


def _own(obj, name):
    """The object's own value for `name` (not inherited), or MISSING."""
    try:
        return vars(obj).get(name, MISSING)
    except TypeError:                       # no __dict__ (slots)
        return getattr(obj, name, MISSING)


def _label(obj):
    if isinstance(obj, (types.FunctionType, type)):
        return obj.__qualname__
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    return type(obj).__name__


class Entry:
    __slots__ = ("kind", "target", "name", "old")

    def __init__(self, kind, target, name, old):
        self.kind, self.target, self.name, self.old = kind, target, name, old

    def invert(self):
        t, n, old = self.target, self.name, self.old
        if self.kind == "code":
            t.__code__, t.__defaults__, t.__kwdefaults__ = old
        elif self.kind == "attr":
            setattr(t, n, old)
        elif self.kind == "object":
            t[n] = old
        elif self.kind == "added":
            if isinstance(t, (dict, list)):
                if n in t:
                    del t[n]
            elif _own(t, n) is not MISSING:
                delattr(t, n)

    def describe(self):
        return {"kind": self.kind, "target": _label(self.target), "name": str(self.name)}


class Journal:
    def __init__(self, name="local"):
        self.name = name
        self.entries = []
        self.marks = []          # [(label, position)]

    # -- marks ---------------------------------------------------------------------------
    def mark(self, label=""):
        pos = len(self.entries)
        self.marks.append((label, pos))
        return pos

    def undo_to(self, mark):
        """Replay inverses back to `mark`, newest first. Never raises: an inverse that fails is
        reported as an `undo_incomplete` finding and the rest still run."""
        errors, n = [], 0
        while len(self.entries) > mark:
            e = self.entries.pop()
            try:
                e.invert()
                n += 1
            except Exception as ex:          # keep going: restore as much as possible
                errors.append(f"{e.describe()}: {type(ex).__name__}: {ex}")
        self.marks = [(lab, p) for lab, p in self.marks if p <= mark]
        if errors:
            return F.not_ok("undo incomplete", [F.finding(
                "undo_incomplete", "; ".join(errors), "every inverse succeeds")], undone=n)
        return F.ok(undone=n)

    def describe(self, since=0):
        return [e.describe() for e in self.entries[since:]]

    # -- typed changes -------------------------------------------------------------------
    def set_code(self, fn, code, defaults=KEEP, kwdefaults=KEEP):
        old = (fn.__code__, fn.__defaults__, fn.__kwdefaults__)
        self.entries.append(Entry("code", fn, fn.__qualname__, old))
        fn.__code__ = code
        if defaults is not KEEP:
            fn.__defaults__ = defaults
        if kwdefaults is not KEEP:
            fn.__kwdefaults__ = kwdefaults

    def record_attr(self, obj, name):
        """Call before assigning obj.name by hand (a repair cell)."""
        old = _own(obj, name)
        self.entries.append(Entry("added", obj, name, None) if old is MISSING
                            else Entry("attr", obj, name, old))

    def set_attr(self, obj, name, value):
        self.record_attr(obj, name)
        setattr(obj, name, value)

    def record_item(self, container, key):
        if key in container:
            self.entries.append(Entry("object", container, key, container[key]))
        else:
            self.entries.append(Entry("added", container, key, None))

    def set_item(self, container, key, value):
        self.record_item(container, key)
        container[key] = value

    # -- repair cells (R10) --------------------------------------------------------------
    def repair(self, declared, audit=None):
        """`declared`: {name: obj} the cell may change (or a list of objects). `audit`:
        {name: obj} the cell can reach but did not declare; changes there are findings.
        The declared objects' instance attributes are audited one level down too."""
        return _Repair(self, declared, audit or {})


_PLAIN = (types.ModuleType, type, types.FunctionType, types.MethodType,
          types.BuiltinFunctionType)


def _content(v):
    """Fingerprint of a mutable container's contents (by element identity)."""
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(map(id, v)))
    if isinstance(v, dict):
        return ("map", frozenset((k, id(x)) for k, x in v.items()))
    if isinstance(v, (set, frozenset)):
        return ("set", frozenset(map(id, v)))
    if isinstance(v, bytearray):
        return ("bytes", bytes(v))
    return None


def _snap(obj):
    try:
        d = dict(vars(obj))
    except TypeError:
        return None
    return d, {k: _content(v) for k, v in d.items()}


class _Repair:
    def __init__(self, journal, declared, audit):
        if not isinstance(declared, dict):
            declared = {_label(o): o for o in declared}
        self.journal, self.declared, self.audit = journal, declared, dict(audit)
        self.findings = []
        self.changed = []

    def __enter__(self):
        mine = {id(o) for o in self.declared.values()}
        self.before = {n: (o, _snap(o)) for n, o in self.declared.items()}
        watch = dict(self.audit)
        for n, o in self.declared.items():      # one level down: objects the cell can reach
            snap = self.before[n][1]
            for k, v in (snap[0].items() if snap else ()):
                if id(v) not in mine and not isinstance(v, _PLAIN) and _snap(v) is not None:
                    watch.setdefault(f"{n}.{k}", v)
        self.watched = {n: (o, _snap(o)) for n, o in watch.items() if id(o) not in mine}
        return self

    def __exit__(self, et, ev, tb):
        for n, (o, snap) in self.before.items():
            if snap is None:
                continue
            old, content = snap
            now = _snap(o)[0]
            for k in sorted(set(old) | set(now)):
                a, b = old.get(k, MISSING), now.get(k, MISSING)
                if a is b:
                    if k in old and content[k] is not None and _content(b) != content[k]:
                        self.findings.append(F.finding(
                            "repair_not_undoable", f"{n}.{k} changed in place",
                            f"{n}.{k} rebound to a new object, or left alone", subject=f"{n}.{k}"))
                    continue
                if a is MISSING:
                    self.journal.entries.append(Entry("added", o, k, None))
                else:
                    self.journal.entries.append(Entry("attr", o, k, a))
                self.changed.append(f"{n}.{k}")
        for n, (o, snap) in self.watched.items():
            now = _snap(o)
            if snap is None or now is None:
                continue
            diff = sorted(k for k in set(snap[0]) | set(now[0])
                          if snap[0].get(k, MISSING) is not now[0].get(k, MISSING)
                          or (k in snap[0] and snap[1][k] is not None
                              and _content(now[0][k]) != snap[1][k]))
            if diff:
                self.findings.append(F.finding(
                    "repair_not_undoable", f"undeclared {n} changed: {', '.join(diff)}",
                    f"only declared objects ({', '.join(self.declared)}) change", subject=n))
        return False
