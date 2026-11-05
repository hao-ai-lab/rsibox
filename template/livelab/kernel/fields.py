"""New fields: live instances that lack an attribute only the edited `__init__` sets.

A swap replaces code, not objects. When the edited `__init__` adds `self.X = ...` and a changed
method reads `self.X`, every instance built before the swap lacks X, and the first call of the
new method raises AttributeError (vLLM #63819 killed an engine this way). `swap.analyze` finds
those fields from the two texts (`new_fields`) and compiles the new `__init__`'s statement that
sets each one, with the earlier statements that bind the locals it reads; `swap.stage` drops
fields the live class already has as a class attribute and reports the rest as a `new_fields`
finding (blocking unless accepted).

This module is the live half, in the participant:

- `find_instances(roots, classes)`: the live instances, found by walking attributes, list and
  dict items from the participant's scope (the engine core, the worker: they own the scheduler,
  the runner, the speculator...), breadth first and bounded. Not by a heap scan: engines freeze
  the gc heap (vLLM calls gc.freeze()), and a scan holds the GIL for its whole first call.
- `resolve(param, chain)`: the value of an `__init__` parameter the statement reads, taken from
  the instance itself or the nearest owner on the path that has an attribute of that name (or
  `_name`): `max_num_reqs` of an estimator comes from the speculator that built it.
- `backfill(nf, found, journal)`: runs the compiled statements on each instance that lacks the
  field, after `journal.record_attr`, so undo removes the fields again.
"""
import collections
import time
import types

from .journal import MISSING

_LEAF = (str, bytes, bytearray, int, float, complex, bool, type(None), range, memoryview,
         types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType,
         types.MethodType, types.CodeType, types.FrameType, set, frozenset)


def _children(o, max_items):
    if isinstance(o, dict):
        for i, (k, v) in enumerate(list(o.items())[:max_items]):
            yield (f"[{k!r}]" if isinstance(k, (str, int)) else f"[{i}]"), v
        return
    if isinstance(o, (list, tuple, collections.deque)):
        for i, v in enumerate(list(o)[:max_items]):
            yield f"[{i}]", v
        return
    try:
        d = object.__getattribute__(o, "__dict__")
    except Exception:
        return
    if isinstance(d, dict):
        for k, v in list(d.items()):
            yield f".{k}", v


def find_instances(roots, classes, max_objects=200_000, max_items=256, max_depth=12):
    """Breadth-first from roots ({name: object}) over attributes and container items:
    ({class: [(instance, path, owners)]}, stats). owners: the objects on the path, root first,
    the instance excluded. Each object is visited once; containers contribute at most
    `max_items` items; the walk stops after `max_objects` objects (stats["truncated"])."""
    classes = tuple(classes)
    found = {c: [] for c in classes}
    t0 = time.perf_counter()
    seen, n = set(), 0
    q = collections.deque((o, name, ()) for name, o in roots.items())
    truncated = False
    while q:
        o, path, owners = q.popleft()
        if id(o) in seen or isinstance(o, _LEAF):
            continue
        seen.add(id(o))
        n += 1
        if n > max_objects:
            truncated = True
            break
        for c in classes:
            if isinstance(o, c):
                found[c].append((o, path, owners))
        if len(owners) >= max_depth:
            continue
        for name, v in _children(o, max_items):
            if id(v) not in seen and not isinstance(v, _LEAF):
                q.append((v, path + name, owners + (o,)))
    return found, {"objects": n, "truncated": truncated,
                   "seconds": round(time.perf_counter() - t0, 6)}


def resolve(param, inst, owners):
    """(value, where) of `param` from the instance or its nearest owner, else (MISSING, None)."""
    for i, o in enumerate((inst,) + tuple(reversed(owners))):
        try:
            d = object.__getattribute__(o, "__dict__")
        except Exception:
            continue
        for name in (param, "_" + param):
            if isinstance(d, dict) and name in d:
                return d[name], ("self" if i == 0 else f"owner -{i}") + f".{name}"
    return MISSING, None


def plan_for(nf, inst, owners):
    """(kwargs, where, missing) for one instance: the backfill's parameters resolved."""
    kw, where, missing = {}, {}, []
    bf = nf["backfill"]
    for p in bf.get("params", ()):
        v, w = resolve(p, inst, owners)
        if v is MISSING:
            if p not in bf.get("defaults", ()):
                missing.append(p)
            continue
        kw[p], where[p] = v, w
    return kw, where, missing


def compiled(nf, mod):
    """The backfill function of a new field (built in a scratch namespace)."""
    ns = {}
    exec(nf["backfill"]["code"], mod.__dict__, ns)
    return ns[nf["cls"]].__dict__["__fastrsi_backfill__"]


def survey(new_fields, roots):
    """At prepare: for every new field, the live instances lacking it and whether a backfill
    can run on each. {subject: {"instances", "lacking", "paths", "params", "missing",
    "backfill"}} plus the walk stats."""
    classes = list({nf["cls_obj"] for nf in new_fields})
    found, stats = find_instances(roots, classes) if roots else ({}, {"objects": 0})
    out = {}
    for nf in new_fields:
        lack = [(o, p, ow) for o, p, ow in found.get(nf["cls_obj"], [])
                if nf["attr"] not in getattr(o, "__dict__", {})]
        info = {"instances": len(found.get(nf["cls_obj"], [])), "lacking": len(lack),
                "paths": [p for _, p, _ in lack[:5]], "params": {}, "missing": [],
                "backfill": "code" in nf["backfill"]}
        for o, p, ow in lack:
            _, where, missing = plan_for(nf, o, ow)
            info["params"][p] = where
            info["missing"] += [f"{p}: {m}" for m in missing]
        info["backfill"] = info["backfill"] and not info["missing"]
        out[nf["subject"]] = info
    return out, stats


def backfill(new_fields, mod_of, roots, journal):
    """At commit (after the swap): set each new field on every live instance that lacks it,
    journaled. mod_of(nf) -> the live module. Returns {"set": [[path, field]], "seconds",
    "walk"}; raises on the first statement that fails (the caller undoes)."""
    t0 = time.perf_counter()
    classes = list({nf["cls_obj"] for nf in new_fields})
    found, stats = find_instances(roots, classes)
    done = []
    for nf in new_fields:
        fn = compiled(nf, mod_of(nf))
        for o, p, ow in found.get(nf["cls_obj"], []):
            if nf["attr"] in getattr(o, "__dict__", {}):
                continue
            kw, _, missing = plan_for(nf, o, ow)
            if missing:
                raise LookupError(f"{p}: cannot resolve {missing} for {nf['field']}")
            journal.record_attr(o, nf["attr"])
            fn(o, **kw)
            done.append([p, nf["field"]])
    return {"set": done, "seconds": round(time.perf_counter() - t0, 6), "walk": stats}
