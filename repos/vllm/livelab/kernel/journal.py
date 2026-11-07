"""Journal: one ordered undo log per participant, with transaction marks.

Every change the kernel makes goes through a Journal method that records the inverse first.
Entries are typed:

| kind     | change                                   | inverse                         |
|----------|------------------------------------------|---------------------------------|
| `code`   | fn.__code__, __defaults__, __kwdefaults__ | the same three, old objects     |
| `attr`   | setattr on an attribute that existed      | setattr back to the old object  |
| `added`  | an attribute or dict key that did not exist | delete it                     |
| `call`   | an in-place change only a function can undo (`record_call`) | call that function |
| `closures` | functions the new code makes at run time (`record_closures`) | old code back |
| `object` | container[key] = value on an existing key | put the old object back         |
| `removed`| delete an attribute or dict key (a definition the edit deletes) | put it back |

`mark()` returns the log length; `undo_to(mark)` pops entries back to it, newest first, so the
base comes back exactly: the same code objects, the same attribute objects, added names gone.

Attributes. `record_attr(obj, name)` reads and restores the instance `__dict__` entry when
that is exact: the object has a `__dict__`, its type keeps the default `__setattr__` and
`__delattr__`, and no data descriptor of the type owns the name. Otherwise it records
`getattr(obj, name)` and undo restores it with `setattr` (or `delattr` when the attribute was
absent): a function's `__doc__`, descriptors such as a torch `Parameter.data`, an `nn.Module`
submodule (kept in `_modules` by the module's `__setattr__`), `__slots__`. That path restores
the value `getattr` saw, through the same `setattr` the change went through.

In-place re-inits (a Triton kernel rebuilt, a cache re-created in place) record their own
inverse: `record_call(undo_fn, label)` runs `undo_fn()` at its place in the reverse order; if
it raises, the undo reports `undo_incomplete` with the label and goes on.

Closures made at run time. A swapped function's new code can create function objects while it
is installed (a closure it returns, a callback it registers); they are in no journal entry and
would keep running the new inner code after undo. `apply` records the (new nested code, old
nested code) pairs of every code it installs (`record_closures`), first, so the entry is undone
last. Only at undo, and only for the new nested codes that gained a reference since apply (a
function object made from a code holds one; `sys.getrefcount` against the count taken when
the entry was recorded), one heap pass finds the functions that run them (by identity;
`gc.get_referrers` of those codes) and gives each its old code. When no code gained a
reference, nothing was made that outlives the call and the pass is skipped: on a loaded vLLM
engine core the pass costs 0.1-0.15 s in the undo gap. A reference count that went down as
another came up could hide a function; the kernel's own holders of these codes (the entry, the
staged swap that installed them) stay until the undo is over.
Generator expressions are left out: no function object of theirs outlives the call. A function
whose new code has no old counterpart (a nested definition the edit added, or other free
variables) cannot go back: it is a `closure_not_restored` finding with the count, and the undo
is not ok.

Repair cells (recipe R10, "the live lab is a notebook"): a cell that fixes state by hand is a
change too. Either call `record_attr(obj, name)` before assigning, or run the cell inside
`with journal.repair({"engine": engine}) as r:`. The context snapshots the declared objects'
attributes and journals every attribute the cell set, added or deleted on them. What it cannot
undo (an attribute changed on an object the cell did not declare, or a list/dict/set value
changed in place) becomes a `repair_not_undoable` finding in `r.findings`, not an exception.
"""
import gc
import sys
import time
import types

from . import findings as F
from . import locate as L

MISSING = object()   # "this attribute or key did not exist"
KEEP = object()      # set_code: leave defaults / kwdefaults as they are


def _own(obj, name):
    """The object's own value for `name` (not inherited), or MISSING."""
    try:
        return vars(obj).get(name, MISSING)
    except TypeError:                       # no __dict__ (slots)
        return getattr(obj, name, MISSING)


_SETATTR = {object.__setattr__, type.__setattr__, types.ModuleType.__setattr__}
_DELATTR = {object.__delattr__, type.__delattr__, types.ModuleType.__delattr__}


def _in_dict(obj, name):
    """Is obj.name a plain `__dict__` entry, read and written straight through the instance
    dict? Not when the type overrides `__setattr__` / `__delattr__` (nn.Module), when a data
    descriptor of the type owns the name (`__doc__` of a function, a property, torch's
    `Tensor.data`, `__slots__`), or when the object has no `__dict__`."""
    t = type(obj)
    if t.__setattr__ not in _SETATTR or t.__delattr__ not in _DELATTR:
        return False
    try:
        vars(obj)
    except TypeError:
        return False
    for k in t.__mro__:
        d = k.__dict__.get(name, MISSING)
        if d is not MISSING:
            dt = type(d)
            return not (hasattr(dt, "__set__") or hasattr(dt, "__delete__"))
    return True


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
        """Undo this change. Returns None, or (for `closures`) a dict: what it restored and
        the findings of what it could not."""
        t, n, old = self.target, self.name, self.old
        if self.kind == "call":
            t()
        elif self.kind == "closures":
            return _restore_closures(old)
        elif self.kind == "getattr_added":   # absent before, set through setattr
            try:
                getattr(t, n)
            except AttributeError:
                return None
            delattr(t, n)
        elif self.kind == "code":
            t.__code__, t.__defaults__, t.__kwdefaults__ = old
        elif self.kind == "attr":
            setattr(t, n, old)
        elif self.kind == "object":
            t[n] = old
        elif self.kind == "removed":
            if isinstance(t, dict):
                t[n] = old
            else:
                setattr(t, n, old)
        elif self.kind == "added":
            if isinstance(t, (dict, list)):
                if n in t:
                    del t[n]
            elif _own(t, n) is not MISSING:
                delattr(t, n)

    def describe(self):
        if self.kind == "call":
            return {"kind": "call", "target": str(self.name), "name": str(self.name)}
        if self.kind == "closures":
            return {"kind": "closures", "target": "nested codes", "name": str(len(self.old[0]))}
        return {"kind": "added" if self.kind == "getattr_added" else self.kind,
                "target": _label(self.target), "name": str(self.name)}


def _refs(pairs):
    """{id(new code): its reference count}, taken the same way at record and at undo."""
    return {k: sys.getrefcount(p[0]) for k, p in pairs.items()}


def _restore_closures(entry):
    """entry: (pairs, refs at record). pairs: {id(new nested code): (new code, old code or
    None)}. Only the new codes that gained a reference since the entry was recorded can have
    a function object; when none did, no pass runs. Else one heap pass (`gc.get_referrers`
    of those codes, frozen objects included): every function whose `__code__` is one of them
    gets its old code back when the free variables match; the rest are counted in a
    `closure_not_restored` finding."""
    pairs, base = entry
    now = _refs(pairs)
    pairs = {k: p for k, p in pairs.items() if now[k] != base[k]}
    if not pairs:
        return {"closures_restored": 0, "closure_codes_scanned": 0}
    t0 = time.perf_counter()
    restored, kept = 0, {}
    with L.unfrozen():
        refs = gc.get_referrers(*[new for new, _ in pairs.values()])
    try:
        for o in refs:
            if type(o) is not types.FunctionType:
                continue
            p = pairs.get(id(o.__code__))
            if p is None or p[0] is not o.__code__:
                continue
            new, old = p
            if old is not None and old.co_freevars == new.co_freevars:
                o.__code__ = old
                restored += 1
            else:
                kept[new.co_qualname] = kept.get(new.co_qualname, 0) + 1
    finally:
        del refs
    out = {"closures_restored": restored, "closure_codes_scanned": len(pairs),
           "closure_scan_s": round(time.perf_counter() - t0, 6)}
    if kept:
        out["findings"] = [F.finding(
            "closure_not_restored",
            f"{n} function object(s) made by the undone code at run time still run its "
            f"{q}: it has no base version with the same free variables", "every function "
            "the undone code made runs base code again", subject=q) for q, n in kept.items()]
    return out


class Journal:
    def __init__(self, name="local"):
        self.name = name
        self.entries = []
        self.marks = []          # [(label, position)]
        self.routes = []         # rebuild routes asked for by cells (`need_rebuild`)

    # -- marks ---------------------------------------------------------------------------
    def mark(self, label=""):
        pos = len(self.entries)
        self.marks.append((label, pos))
        return pos

    def undo_to(self, mark):
        """Replay inverses back to `mark`, newest first. Never raises: an inverse that fails is
        reported as an `undo_incomplete` finding and the rest still run."""
        errors, n, fs, info = [], 0, [], {}
        while len(self.entries) > mark:
            e = self.entries.pop()
            try:
                r = e.invert()
                n += 1
            except Exception as ex:          # keep going: restore as much as possible
                errors.append(f"{e.describe()}: {type(ex).__name__}: {ex}")
                continue
            if r:
                fs += r.pop("findings", [])
                for k, v in r.items():
                    info[k] = round(info.get(k, 0) + v, 6)
        self.marks = [(lab, p) for lab, p in self.marks if p <= mark]
        if errors:
            fs.insert(0, F.finding("undo_incomplete", "; ".join(errors),
                                   "every inverse succeeds"))
        if fs:
            return F.not_ok("undo incomplete", fs, undone=n, **info)
        return F.ok(undone=n, **info)

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
        """Call before assigning obj.name by hand (a repair cell). A plain `__dict__` entry
        is read from the dict; any other attribute (descriptor, custom `__setattr__`,
        `__doc__`, slots) with `getattr`, and undo puts it back with `setattr`, or `delattr`
        when it was absent (see the module docstring)."""
        if _in_dict(obj, name):
            old = vars(obj).get(name, MISSING)
            self.entries.append(Entry("added", obj, name, None) if old is MISSING
                                else Entry("attr", obj, name, old))
            return
        old = getattr(obj, name, MISSING)
        self.entries.append(Entry("getattr_added", obj, name, None) if old is MISSING
                            else Entry("attr", obj, name, old))

    def record_call(self, undo_fn, label, after_code=False):
        """Journal a change only a function can undo (a Triton kernel rebuilt in place):
        undo calls `undo_fn()` at this entry's place, newest first with the others. If it
        raises, the undo reports `undo_incomplete` naming `label` and goes on.
        after_code=True: the entry goes to the start of the current transaction (the last
        mark), so undo calls it after every other inverse of the transaction, once the code
        is back at base. For engine state that must be rebuilt FROM the base code: a cell
        that released captured graphs records "recapture" this way, so a rollback recaptures
        the base code's graphs, not the edit's."""
        e = Entry("call", undo_fn, label, None)
        if after_code and self.marks:
            self.entries.insert(self.marks[-1][1], e)
        else:
            self.entries.append(e)

    def need_rebuild(self, route):
        """A repair cell changed engine state that only a rebuild can restore (it released
        and recaptured graphs: R1). The participant reports the route with its commit, for a
        host that runs it again from the base code after an undo or a rollback, OUTSIDE the
        phase-2 gap (a recompile takes seconds; in the gap it missed the deadline and killed a
        live engine). Coordinator does not run it yet (kernel README, Not built). Not an inverse: nothing runs at undo_to."""
        self.routes.append(route)

    def record_closures(self, pairs):
        """Journal {id(new nested code): (new code, old code or None)} for the codes a swap
        installs, with each new code's reference count now; undo gives every function made
        from a new code its old one (one heap pass, only at undo, only for the codes that
        gained a reference). Nothing is recorded when `pairs` is empty."""
        if pairs:
            pairs = dict(pairs)
            self.entries.append(Entry("closures", None, "closures", (pairs, _refs(pairs))))

    def set_attr(self, obj, name, value):
        self.record_attr(obj, name)
        setattr(obj, name, value)

    def record_item(self, container, key):
        if key in container:
            self.entries.append(Entry("object", container, key, container[key]))
        else:
            self.entries.append(Entry("added", container, key, None))

    def set_item(self, container, key, value, absent=MISSING):
        """container[key] = value. absent: what undo puts there when the key did not exist
        (default: undo deletes it). The witness global uses it: undo binds a no-op stub, since
        compiled code may still name it."""
        if absent is not MISSING and key not in container:
            self.entries.append(Entry("object", container, key, absent))
        else:
            self.record_item(container, key)
        container[key] = value

    def delete(self, owner, name):
        """Delete owner[name] (a dict) or owner.name (its own attribute); undo puts the same
        object back."""
        if isinstance(owner, dict):
            self.entries.append(Entry("removed", owner, name, owner[name]))
            del owner[name]
        else:
            self.entries.append(Entry("removed", owner, name, vars(owner)[name]))
            delattr(owner, name)

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
