"""Refresh edited modules in a process forked from the zygote, before it builds anything.

The zygote imports vLLM once; every server it forks inherits those modules. When a source file
changed after the zygote imported it, the forked child would run the old code. Instead of
refusing (or restarting the zygote, ~20 s of imports), the child drops the edited modules and
every module that imports them, directly or not, from `sys.modules`. The server then imports
them again from the new source as it starts. This is safe only in the child before it builds
anything: no config, model or graph exists yet that could still point at the old classes.

What cannot be refreshed this way (the caller reports these as a finding):
- a changed file that is not Python source (a C++ extension `.so` cannot be unloaded);
- a changed module outside the packages the import graph covers (`prefixes`).

A module that registers a torch op runs `Library.define` again when re-imported, which torch
refuses for a name that exists. `op_redefine_guard()` lets an identical schema pass (the new
implementation still replaces the old one: `Library.impl` overrides) and raises, naming the
op, when the schema itself changed. Opaque types (torch.compile's custom classes, such as vLLM's
`LayerName`) are registered by name in C++ and by class in Python: the guard lets the C++ name
registration repeat, and the Python tables then map the new class.

vLLM also keeps name registries of its own in Python: `CustomOp.register("rms_norm")` asserts the
name is new in `op_registry` (and `register_oot` in `op_registry_oot`), so re-importing an edited
layer module fails with "Duplicate op name" (upstream 61323, layernorm.py). Those registries live
in modules a refresh keeps, so `forget_registrations()` drops the entries whose class came from a
refreshed module before the re-import, the state a fresh process has before it imports the
module. Entries from modules that were not refreshed stay, so a real duplicate still fails.
"""
import ast, contextlib, os, re, sys, time


def _resolve(mod, is_pkg, node):
    """Absolute module names an import statement in module `mod` may load."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    base = node.module or ""
    if node.level:
        parts = mod.split(".") if is_pkg else mod.split(".")[:-1]
        if node.level > 1:
            parts = parts[:len(parts) - node.level + 1]
        base = ".".join(parts + ([base] if base else []))
    # "from pkg import name" loads pkg, and pkg.name when name is a submodule
    return [base] + [f"{base}.{a.name}" for a in node.names if a.name != "*"]


def _binding_imports(tree):
    """Import statements that run at import time and bind names in the module or a class body
    (module level, including inside if/try/with and class bodies). Imports inside a function run
    on each call and look the module up in sys.modules then, so they see a refreshed module."""
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        todo.extend(ast.iter_child_nodes(node))


def import_graph(prefixes=("vllm",), modules=None):
    """{module: set of loaded modules it imports at import time}, read from the source of every
    loaded module under `prefixes`."""
    modules = sys.modules if modules is None else modules
    graph = {}
    for name, m in list(modules.items()):
        f = getattr(m, "__file__", None)
        if not (f and f.endswith(".py") and name.split(".")[0] in prefixes):
            continue
        try:
            tree = ast.parse(open(f, "rb").read(), f)
        except (OSError, SyntaxError, ValueError):
            continue
        is_pkg = os.path.basename(f) == "__init__.py"
        deps = set()
        for node in _binding_imports(tree):
            deps.update(d for d in _resolve(name, is_pkg, node) if d in modules and d != name)
        graph[name] = deps
    return graph


def dependents(changed, graph):
    """`changed` plus every module that imports one of them, directly or through others."""
    rev = {}
    for m, deps in graph.items():
        for d in deps:
            rev.setdefault(d, set()).add(m)
    out, todo = set(changed), list(changed)
    while todo:
        for m in rev.get(todo.pop(), ()):
            if m not in out:
                out.add(m)
                todo.append(m)
    return out


def plan(changed_files, graph, modules=None, whole_above=100):
    """What a fork must refresh for these changed files:
    {"modules": modules to drop, "changed": changed modules, "scope": "dependents" | "all",
     "blocked": [{"file", "why"}] for files a refresh cannot apply}.

    Scope "dependents" drops the changed modules and their importers. When those exceed
    `whole_above` modules the edit reached the package's core, where modules register into
    package-wide registries (vLLM's IR op table, CustomOp, the model registry) that a partial
    re-import would fill twice: scope "all" then drops every module the graph covers, which
    costs about one import of the package itself (third-party libraries stay imported)."""
    modules = sys.modules if modules is None else modules
    by_file = {}
    for name, m in list(modules.items()):
        f = getattr(m, "__file__", None)
        if f:
            by_file.setdefault(f, []).append(name)
    changed, blocked = set(), []
    for f in changed_files:
        names = by_file.get(f, [])
        if not f.endswith(".py"):
            blocked.append({"file": f, "why": "not Python source (an extension module cannot be "
                                              "unloaded from a process)"})
        elif not names or not all(n in graph for n in names):
            blocked.append({"file": f, "why": "outside the packages the import graph covers"})
        else:
            changed.update(names)
    drop, scope = dependents(changed, graph), "dependents"
    if len(drop) > whole_above:
        drop, scope = set(graph), "all"
    return {"modules": sorted(drop), "changed": sorted(changed), "scope": scope,
            "blocked": blocked}


def purge(names, modules=None):
    """Drop modules so the next import reads their source again. Also removes each from its
    parent package's attributes, where `import a.b` would otherwise still find the old one."""
    modules = sys.modules if modules is None else modules
    for n in names:
        modules.pop(n, None)
        parent, _, child = n.rpartition(".")
        p = modules.get(parent)
        if p is not None and child in getattr(p, "__dict__", {}):
            try:
                delattr(p, child)
            except AttributeError:
                pass
    import importlib
    importlib.invalidate_caches()


def _norm_schema(schema):
    """Arguments and returns of a schema, without the op name and with opaque class types
    (dotted names, which the C++ schema shows as PyObject) unified."""
    body = schema.split("(", 1)[-1]
    return re.sub(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)+", "PyObject", body).replace(" ", "")


@contextlib.contextmanager
def op_redefine_guard(report=None):
    """While re-importing refreshed modules: `Library.define` of an op that already exists with
    the same schema is skipped (its implementation is still replaced by `Library.impl`, which
    overrides); a changed schema raises with the op's name, since a running process cannot
    change an op's signature. `report(event)` gets one dict per skipped define, replaced
    implementation and re-registered opaque type. Use it through guard_for()."""
    try:
        from torch.library import Library
        import torch
    except ImportError:
        yield
        return
    orig = Library.define
    skipped = set()

    def _emit(event):
        if report is not None:
            report(event)

    def define(self, schema, *args, **kw):
        name = schema.split("(", 1)[0].strip()
        op, _, overload = name.partition(".")
        qual = f"{self.ns}::{op}"
        try:
            old = str(torch._C._get_schema(qual, overload))
        except RuntimeError:
            return orig(self, schema, *args, **kw)
        try:   # torch's own printing, so defaults compare alike ("0.0" prints as "0.")
            new = str(torch._C.parse_schema(f"{self.ns}::{schema}"))
        except RuntimeError:
            new = schema
        if _norm_schema(old) != _norm_schema(new):
            raise RuntimeError(f"refresh: op {qual} changed its schema ({old} -> {schema}); "
                               "a process cannot change an op's signature: restart the zygote")
        skipped.add((self.ns, op))
        _emit({"event": "zygote_op_redefined", "op": qual + (f".{overload}" if overload else ""),
               "schema": old})
        return name

    orig_impl = Library.impl
    import inspect
    can_override = "allow_override" in inspect.signature(orig_impl).parameters

    def impl(self, op_name, *args, **kw):
        # Override only ops whose define was skipped above: those are the refreshed modules'
        # re-registrations. Any other duplicate keeps torch's own behavior. (torch >= 2.9 has
        # the flag; its default is False before 2.14 and True after.)
        op = op_name.split(".", 1)[0] if isinstance(op_name, str) else None
        if (self.ns, op) in skipped:
            if can_override:
                kw["allow_override"] = True
            _emit({"event": "zygote_op_impl_replaced", "op": f"{self.ns}::{op_name}",
                   "dispatch_key": kw.get("dispatch_key", args[1] if len(args) > 1 else "")})
        return orig_impl(self, op_name, *args, **kw)
    orig_opaque = getattr(torch._C, "_register_opaque_type", None)

    def register_opaque(name, *args, **kw):
        try:
            return orig_opaque(name, *args, **kw)
        except RuntimeError as e:
            if "already registered" not in str(e):
                raise
            _emit({"event": "zygote_opaque_reregistered", "type": name})
    Library.define, Library.impl = define, impl
    if orig_opaque is not None:
        torch._C._register_opaque_type = register_opaque
    try:
        yield
    finally:
        Library.define, Library.impl = orig, orig_impl
        if orig_opaque is not None:
            torch._C._register_opaque_type = orig_opaque


# Python name registries that refuse a name twice, by module: {module: (dict attributes)}.
NAME_REGISTRIES = {
    "vllm.model_executor.custom_op": ("op_registry", "op_registry_oot"),
    "vllm.model_executor.hw_agnostic.custom_op": ("op_registry", "op_registry_oot"),
}


def forget_registrations(refresh_names, registries=None, modules=None, report=None):
    """Drop registry entries whose registered class (or function) was defined in a refreshed
    module, so the module's re-import can register the name again with its new class. A
    registry module that was itself refreshed starts empty anyway and is skipped. Returns the
    dropped [(registry, name)]."""
    registries = NAME_REGISTRIES if registries is None else registries
    modules = sys.modules if modules is None else modules
    refreshed, dropped = set(refresh_names), []
    for mod, attrs in registries.items():
        m = modules.get(mod)
        if m is None or mod in refreshed:
            continue
        for attr in attrs:
            reg = getattr(m, attr, None)
            if not isinstance(reg, dict):
                continue
            for name, obj in list(reg.items()):
                if getattr(obj, "__module__", None) in refreshed:
                    del reg[name]
                    dropped.append((f"{mod}.{attr}", name))
                    if report is not None:
                        report({"event": "zygote_registry_forgot", "registry": f"{mod}.{attr}",
                                "name": name, "was": f"{obj.__module__}.{obj.__qualname__}"})
    return dropped


@contextlib.contextmanager
def _refresh_guard(refresh_names, report=None):
    forget_registrations(refresh_names, report=report)
    with op_redefine_guard(report):
        yield


def guard_for(refresh_names, report=None):
    """The re-registration guard for a forked child: active only when the fork refreshed
    modules. It forgets the refreshed modules' entries in vLLM's name registries, then guards
    torch op definitions (op_redefine_guard). A fork without a refresh (and the engine core and
    workers it forks later) keeps torch's and vLLM's own duplicate-registration errors."""
    if not refresh_names:
        return contextlib.nullcontext()
    return _refresh_guard(refresh_names, report)


def print_event(event):
    """Report one guard event as a JSON line in the server log."""
    import json
    print(json.dumps(event), flush=True)


def refresh(names, modules=None):
    """In the forked child: drop `names`; returns {"modules": n, "seconds"}. The re-imports
    happen when the server imports them (inside op_redefine_guard, see zygote._child)."""
    t = time.time()
    purge(names, modules)
    return {"modules": len(names), "purge_seconds": round(time.time() - t, 4)}
