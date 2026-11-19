"""Reach the kernel's participants inside Ray actors.

A Ray job keeps code in places the kernel cannot see by module name:

- `@ray.remote` replaces a class in its module with an ActorClass handle, and each actor process
  runs its own deserialized copy of the class. The copy's methods read a globals dict of their
  own, not the module's, so the kernel (which finds a module's functions by their globals)
  swaps the module's class, which no actor runs: the commit succeeds and nothing changes.
- An engine may load a file without a `sys.modules` entry (verl loads a custom reward file
  that way and keeps only the function).

`actor_class` unwraps a handle. `Namespace` is a module stand-in over a live globals dict:
`actor_view` builds one over the actor copy's globals, `file_namespace` one over a loaded file's.
`serve` runs in the actor (through `__ray_call__`): it maps each plan module to what this process
runs, then calls one `LocalParticipant`. A plan module that defines the actor's class gets a
second entry `<module>#actor` over the copy; its patch is the copy's text with only that class
taken from the edit, so module-level functions stay with the module. A process that loaded none
of the plan's modules abstains (`absent`, `abstain`). `RayActorParticipant` is the host's side.
Workers must import this file as `ray_participants` with `kernel` beside it (PYTHONPATH).
"""
import ast, hashlib, importlib, os, sys, types, uuid

try:
    from kernel import coordinator as C, findings as F, locate as L
except ImportError:
    from .kernel import coordinator as C, findings as F, locate as L

ACTOR = "#actor"
_LOCAL = {}       # participant name -> LocalParticipant, in this process
_ABSENT = {}      # participant name -> {txn: state} of transactions this process abstained from
_STATE = {"commit": "committed", "abort": "aborted", "undo": "undone"}


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def actor_class(value, host=None):
    """The class behind a `@ray.remote` handle; the class `host` runs when it is a copy of it.
    Anything else comes back unchanged."""
    try:
        metadata = vars(value).get("__ray_metadata__")
    except TypeError:
        return value
    cls = getattr(getattr(metadata, "modified_class", None), "__ray_actor_class__", None)
    if cls is None:
        return value
    own = getattr(type(host), "__ray_actor_class__", None)
    same = own is not None and (own.__module__, own.__qualname__) == (cls.__module__, cls.__qualname__)
    return own if same else cls


class Namespace:
    """A module stand-in whose `__dict__` is a live globals dict, so the kernel finds the
    functions that read it (`fn.__globals__ is vars(namespace)`)."""

    def __init__(self, globals_dict):
        self.__dict__ = globals_dict


def _register(name, namespace):
    if sys.modules.get(name) is not namespace:
        sys.modules[name] = namespace
    return name, namespace


def actor_view(host):
    """(module name, alias, Namespace) over the globals of the class copy `host` runs, or None
    when the copy reads its module's own globals (no view needed) or `host` is no actor."""
    own = getattr(type(host), "__ray_actor_class__", None)
    own_mod = getattr(own, "__module__", None)
    if own is None or not own_mod:
        return None
    if own_mod not in sys.modules:
        try:
            importlib.import_module(own_mod)
        except ImportError:
            return None
    module = sys.modules.get(own_mod)
    if module is None:
        return None
    for value in vars(own).values():
        function = getattr(value, "__func__", value)
        if isinstance(function, types.FunctionType):
            if function.__globals__ is vars(module):
                return None
            function.__globals__.setdefault(own.__name__, own)   # module_functions walks it
            alias, namespace = _register(
                "fastrsi_actor_" + _digest(f"{module.__name__}:{id(own)}"),
                Namespace(function.__globals__))
            return module.__name__, alias, namespace
    return None


def file_namespace(path):
    """(module name, module) for the file at `path` as this process loaded it: its
    `sys.modules` entry, else a Namespace over its functions' globals registered under
    `file_module(path)`; (None, None) when this process never loaded it. Raises if the file's
    functions read several globals dicts (loaded twice): one swap would miss the others."""
    path = os.path.realpath(path)
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if isinstance(file, str) and _real(file) == path:
            return name, module
    base, dicts = os.path.basename(path), {}
    for obj in L.all_objects():
        # type() rather than isinstance(): dead weakref proxies raise on isinstance
        if type(obj) is types.FunctionType and obj.__code__.co_filename.endswith(base) \
                and os.path.realpath(obj.__code__.co_filename) == path:
            dicts[id(obj.__globals__)] = obj.__globals__
    if len(dicts) > 1:
        raise RuntimeError(f"{path} is loaded {len(dicts)} times in this process")
    if not dicts:
        return None, None
    globals_dict = next(iter(dicts.values()))
    globals_dict.setdefault("__file__", path)
    return _register(file_module(path), Namespace(globals_dict))


_REAL = {}   # module __file__ -> realpath: a module's file never moves, and realpath on a network file system costs a stat per path component


def _real(file):
    try:
        return _REAL[file]
    except KeyError:
        return _REAL.setdefault(file, os.path.realpath(file))


def file_module(path):
    """The plan module name of a file loaded without a `sys.modules` entry."""
    return "fastrsi_file_" + _digest(os.path.realpath(path))


def _class_lines(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return first - 1, node.end_lineno
    return None


def splice_class(base, edited, name):
    """`base` with the top-level class `name` (decorators included) taken from `edited`;
    None when either text lacks it."""
    old, new = _class_lines(base, name), _class_lines(edited, name)
    if old is None or new is None:
        return None
    lines, edited_lines = base.splitlines(True), edited.splitlines(True)
    return "".join(lines[:old[0]] + edited_lines[new[0]:new[1]] + lines[old[1]:])


def actor_plan(local, host, plan):
    """The plan as this process runs it: file modules mapped to what it loaded, modules it never
    imported dropped (it does not import them), the `#actor` entry added for the actor's class."""
    modules, view = {}, actor_view(host)
    if view is not None:
        local.alias[view[0] + ACTOR] = view[1]
    for name, spec in (plan.get("modules") or {}).items():
        if spec.get("file"):
            actual, module = file_namespace(spec["file"])
            if module is None:
                continue
            local.alias[name] = actual
        elif name not in sys.modules:
            continue
        modules[name] = spec
        if view is not None and view[0] == name:
            entry = name + ACTOR
            base = local._live_source(entry)
            if base is None:
                with open(sys.modules[name].__file__) as handle:
                    base = handle.read()
            patch = splice_class(base, spec["source"], type(host).__ray_actor_class__.__name__)
            if patch is not None and patch != base:
                modules[entry] = {"source": patch, "targets": "auto", "scan": spec.get("scan", "auto")}
    return {**plan, "modules": modules}


def _dead(key):
    """Is `key` a function of a `@ray.remote` class body in its module? No process runs it."""
    if not isinstance(key, str) or key.count(":") < 2:
        return False
    module_name, qualname, _ = key.rsplit(":", 2)
    module = sys.modules.get(module_name)
    value = vars(module).get(qualname.split(".")[0]) if module is not None and ACTOR not in module_name else None
    return value is not None and actor_class(value) is not value


def _fingerprint_modules(local, host, modules):
    """Subset of `modules` loaded in this actor, plus `<module>#actor` for its actor copy."""
    view = actor_view(host)
    if view is not None:
        local.alias[view[0] + ACTOR] = view[1]
    present = []
    for item in modules:
        is_path = isinstance(item, str) and (item.endswith(".py") or os.sep in item)
        name = file_module(item) if is_path else (item if isinstance(item, str) else item["name"])
        file_path = item if is_path else (None if isinstance(item, str) else item.get("file"))
        if file_path:
            actual, module = file_namespace(file_path)
            if module is None:
                continue
            local.alias[name] = actual
        elif local._actual(name) not in sys.modules:
            continue
        if name not in present:
            present.append(name)
        if view is not None and view[0] == name and name + ACTOR not in present:
            present.append(name + ACTOR)
    return present


def serve(host, name, method, payload):
    """One participant call in this process; `host` is the actor (None outside one)."""
    actor_view(host)  # ensure a pure-class @ray.remote actor imports its defining module
    if method == "loaded":
        loaded = sorted(n for n, s in (payload.get("modules") or {}).items()
                        if (file_namespace(s["file"])[1] is not None if s.get("file") else n in sys.modules))
        return F.ok(loaded=loaded)
    local = _LOCAL.get(name) or _LOCAL.setdefault(name, C.LocalParticipant(name))
    absent = _ABSENT.setdefault(name, {})
    if method == "prepare":
        plan = actor_plan(local, host, payload)
        if not plan["modules"] and not plan.get("repairs"):
            txn = plan.get("id") or f"txn-{uuid.uuid4().hex[:8]}"
            absent[txn] = "prepared"
            return F.ok(staged={}, txn=txn, absent=True, abstain=True, findings=[])
        payload = plan
    elif method in ("validate", "commit", "abort", "undo", "status", "proof", "witness_check"):
        txn, _ = C.txn_ref(payload)
        if txn in absent:
            absent[txn] = _STATE.get(method, absent[txn])
            return F.ok(txn=txn, state=absent[txn], absent=True, abstain=True, findings=[],
                        counts={}, hashes={}, witness={})
    elif method == "fingerprint":
        payload = _fingerprint_modules(local, host, payload)
    reply = getattr(local, method)(payload)
    if method == "witness_check":
        for key in [k for k in list(reply.get("counts", ())) if _dead(k)]:
            reply["counts"].pop(key, None)
            (reply.get("capture_counts") or {}).pop(key, None)
            (reply.get("ran") or {}).pop(key, None)
        reply["findings"] = [f for f in reply.get("findings", [])
                             if not (f["check"] == "zero_witness" and _dead(f.get("subject")))]
    return reply


def _ray_get(ref, timeout_s):
    import ray
    return ray.get(ref, timeout=timeout_s)


class RayActorParticipant(C.CallParticipant):
    """The host's side of a participant in a Ray actor: every call is
    `get(handle.__ray_call__.remote(serve, name, method, payload))`."""

    def __init__(self, name, handle, get=None, timeout_s=None, bounded=False):
        fetch = get or _ray_get
        super().__init__(name, lambda m, p: fetch(handle.__ray_call__.remote(serve, name, m, p), timeout_s), bounded)

    def loaded(self, plan):
        return self._do("loaded", plan)


def unloaded(participants, plan):
    """Plan modules that no participant runs. Ask before `Coordinator.run`: a plan whose
    module every participant abstains from would commit and change nothing."""
    seen = set()
    for participant in participants:
        seen.update(participant.loaded(plan).get("loaded", ()))
    return sorted(set(plan.get("modules") or {}) - seen)
