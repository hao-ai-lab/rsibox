"""The rule set: how one function maps to a function group and an install class, and how a
lab's capabilities are read from its files. Standard library only; reads text, imports no engine.

Framework rules live here (they hold for any engine). An engine adds a rules file (JSON) with
its source roots, path and decorator declarations, and the lab files that prove each capability.
"""
import ast
import fnmatch
import hashlib
import json
import re

RULESET_VERSION = "1.1"

# Function groups of core/agents/livelab_overview.md, and the install class of each
# (core/livelab/INTERFACE.md, "How install refreshes").
GROUP_INSTALL = {
    "OP": "swap",                # swap the function body in place
    "GRAPH": "swap_recapture",   # swap, then recapture or recompile the graphs that hold it
    "STATE": "rebuild",          # rebuild in place the objects built from it
    "RESOURCE": "rebuild",       # allocate new, copy, switch, free
    "RANKS": "respawn",          # process groups every rank agrees on: a new worker group
    "NATIVE": "restart",         # C++/CUDA source: no lab builds a versioned library yet
    "restart": "restart",        # startup logic, launch flags, environment variables
}
INSTALLS = ("swap", "swap_recapture", "rebuild", "respawn", "restart")
CAPABILITIES = ("swap", "swap_recapture", "jit_rebuild", "constructor_reinit", "rebuild", "respawn")
LIVE = ("swap", "swap_recapture", "rebuild")   # installed into the warm engine, no new process

CONSTRUCTORS = ("__init__", "__new__", "__post_init__")       # kernel: constructor_edit
REINIT_ELIGIBLE = ("__init__", "__post_init__")               # kernel/reinit.py: fresh field values
PURE_TYPES = ("set", "frozenset", "dict", "list", "tuple", "int", "float", "str", "bool", "bytes")
TRITON = ("triton.jit", "triton.autotune", "triton.heuristics")  # kernel/jit.py rebuilds these
COMPILED = ("torch.compile", "torch.jit.script")
NATIVE_EXT = (".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp")
LAUNCH_CALLS = ("add_argument", "ArgumentParser", "add_argument_group")
RESOURCE_NAME = re.compile(
    r"(^|_)(alloc|allocate|preallocate|reserve|mmap|pin_memory|load_weights|load_model|"
    r"load_state_dict|load_checkpoint)(_|$)")
STATE_NAME = re.compile(
    r"^_*(build|setup|set_up|init|initialize|create|make|construct|register|configure|"
    r"prepare|warmup|warm_up|compile|capture|from_config|from_pretrained)(_|$)")

# rule id -> (basis, what it reads). Only "syntax" rules are firm.
BASIS = {
    "native_file": "syntax", "jit_kernel": "syntax", "dsl_kernel": "syntax",
    "constructor": "syntax", "constructor_reinit": "syntax", "compiled": "syntax",
    "module_level": "syntax", "path": "declared", "launch_flags": "name",
    "name_resource": "name", "name_state": "name", "default_op": "default",
}


def _is_pure_fresh(node):
    """True when evaluating `node` twice produces equal, independent fresh objects (reinit.py)."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_pure_fresh(elem) for elem in node.elts)
    if isinstance(node, ast.Dict):
        return (all(key is not None and _is_pure_fresh(key) for key in node.keys)
                and all(_is_pure_fresh(val) for val in node.values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, ast.Call):
        return (isinstance(node.func, ast.Name) and node.func.id in PURE_TYPES
                and all(_is_pure_fresh(arg) for arg in node.args)
                and all(kw.arg is not None and _is_pure_fresh(kw.value) for kw in node.keywords))
    return False


def _field_assign(statement, receiver):
    """(field_name, value_node) for `receiver.F = E` or `receiver.F: T = E`, else None."""
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target, value = statement.targets[0], statement.value
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        target, value = statement.target, statement.value
    else:
        return None
    if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
            and target.value.id == receiver):
        return target.attr, value
    return None


def _field_stores(body, receiver, field_name):
    return sum(
        1 for statement in body for node in ast.walk(statement)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == receiver and node.attr == field_name
        and isinstance(node.ctx, ast.Store)
    )


def _field_reads(statements, receiver, field_name):
    for statement in statements:
        for node in ast.walk(statement):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == receiver and node.attr == field_name
                    and isinstance(node.ctx, ast.Load)):
                return True
            if (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute)
                    and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == receiver and node.target.attr == field_name):
                return True
    return False


def has_reinit_fields(function_node):
    """True when an `__init__` or `__post_init__` defines at least one single-store, unread-in-init
    pure fresh field assignment (`self.F = <pure>`), matching `kernel/reinit.py` eligibility."""
    if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if function_node.name not in REINIT_ELIGIBLE:
        return False
    arguments = function_node.args.posonlyargs + function_node.args.args
    if not arguments:
        return False
    receiver = arguments[0].arg
    for statement in function_node.body:
        assigned = _field_assign(statement, receiver)
        if assigned is None or not _is_pure_fresh(assigned[1]):
            continue
        field_name = assigned[0]
        if _field_stores(function_node.body, receiver, field_name) != 1:
            continue
        other_statements = [other for other in function_node.body if other is not statement]
        if not _field_reads(other_statements, receiver, field_name):
            return True
    return False


def load(path):
    """An engine's rules file, with defaults filled in and a digest of its text."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    r = json.loads(text)
    for key in ("engine", "rules_version", "source_roots"):
        if key not in r:
            raise ValueError(f"{path}: missing {key!r}")
    r.setdefault("native_roots", r["source_roots"])
    r.setdefault("exclude", [])
    r.setdefault("paths", [])
    r.setdefault("decorators", {})
    r.setdefault("install_overrides", {})
    r.setdefault("capabilities", {})
    r.setdefault("upstream", {})
    for p in r["paths"]:
        if p["group"] not in GROUP_INSTALL:
            raise ValueError(f"{path}: unknown group {p['group']!r} for {p['glob']!r}")
    r["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return r


def match(path, globs):
    """fnmatch on the posix path relative to the upstream root; `*` also crosses `/`."""
    return any(fnmatch.fnmatchcase(path, g) for g in globs)


def _named(decorators, names):
    return any(d == n or d.endswith("." + n) for d in decorators for n in names)


def classify(rules, path, name, decorators, calls, reinit_eligible=False):
    """(group, install, capability, rule) for one function. First match wins:
    a restart path, then what the text itself shows, then declared paths, names, the default."""
    declared = next((p for p in rules["paths"] if match(path, [p["glob"]])), None)
    extra = rules["decorators"]
    if declared and declared["group"] == "restart":
        hit = ("restart", "path")
    elif _named(decorators, TRITON):
        # The kernel rebuilds a Triton kernel in place from its new text (jit_rebuild).
        return "NATIVE", "swap_recapture", "jit_rebuild", "jit_kernel"
    elif _named(decorators, extra.get("dsl_kernel", ())):
        # Another JIT DSL: the swap would change Python the launches never run (jit_target).
        return "NATIVE", "respawn", "respawn", "dsl_kernel"
    elif name in CONSTRUCTORS and reinit_eligible:
        install = rules["install_overrides"].get("constructor_reinit", "rebuild")
        return "STATE", install, (None if install == "restart" else "constructor_reinit"), "constructor_reinit"
    elif name in CONSTRUCTORS:
        hit = ("STATE", "constructor")
    elif _named(decorators, COMPILED + tuple(extra.get("compiled", ()))):
        hit = ("GRAPH", "compiled")
    elif declared:
        hit = (declared["group"], "path")
    elif name == "main" or any(c in LAUNCH_CALLS for c in calls):
        hit = ("restart", "launch_flags")
    elif RESOURCE_NAME.search(name):
        hit = ("RESOURCE", "name_resource")
    elif STATE_NAME.search(name):
        hit = ("STATE", "name_state")
    else:
        hit = ("OP", "default_op")
    group, rule = hit
    install = rules["install_overrides"].get(rule, GROUP_INSTALL[group])
    return group, install, (None if install == "restart" else install), rule


def module_level(rules, path):
    """Code outside every function (imports, constants, class bodies): the kernel's swap
    refuses it (outside_changes); a process that imports the module again runs it."""
    if any(p["group"] == "restart" and match(path, [p["glob"]]) for p in rules["paths"]):
        return "restart", "restart", None, "path"
    install = rules["install_overrides"].get("module_level", "respawn")
    return "STATE", install, install, "module_level"


def capabilities(rules, tree):
    """{capability: bool} for the lab files of `tree` (an object with `read(path) -> text or
    None`). A capability holds when every requirement of one of its alternatives is met: the
    file exists and, with "contains", its text matches that regular expression."""
    def met(req):
        text = tree.read(req["path"])
        return text is not None and ("contains" not in req or re.search(req["contains"], text) is not None)
    return {name: any(all(met(r) for r in alt) for alt in cap.get("evidence", []))
            for name, cap in rules["capabilities"].items()}


def reached(rules, capability, path):
    """Does the lab's path for `capability` reach this upstream file (its `reach` globs, minus
    `unreached`)? A capability the rules do not name reaches nothing."""
    cap = rules["capabilities"].get(capability)
    if cap is None:
        return False
    return match(path, cap.get("reach", ["*"])) and not match(path, cap.get("unreached", []))
