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
import contextlib, json, os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import corelab  # noqa: F401,E402
from rsibox import hooks  # noqa: E402
from rsibox.op_guard import op_redefine_guard  # noqa: E402


def import_graph(prefixes=("vllm",), modules=None):
    """{module: set of loaded modules it imports at import time}, read from the source of every
    loaded module under `prefixes`."""
    return hooks.import_graph(prefixes, modules=modules)


dependents = hooks.dependents


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
        if (f := getattr(m, "__file__", None)):
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
    return {"modules": sorted(drop), "changed": sorted(changed), "scope": scope, "blocked": blocked}


def purge(names, modules=None):
    """Drop modules so the next import reads their source again, unbinding parent attributes."""
    hooks.purge_modules(names, modules=modules)


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
    print(json.dumps(event), flush=True)


def refresh(names, modules=None):
    """In the forked child: drop `names`; returns {"modules": n, "purge_seconds"}. The re-imports
    happen when the server imports them (inside op_redefine_guard, see zygote._child)."""
    t = time.time()
    purge(names, modules)
    return {"modules": len(names), "purge_seconds": round(time.time() - t, 4)}
