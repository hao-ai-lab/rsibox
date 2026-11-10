"""Where vLLM runs a model-code target: in Python on every step, only inside captured graphs or
compiled code, or in Python on some steps and in graphs on others. The worker side's
`captured_target` check at prepare (side.Side._captured_check) asks this module.

vLLM splits its graphs at some ops and runs those ops eagerly between graph pieces:

- with torch.compile (compile mode VLLM_COMPILE), the ops in
  `compilation_config.splitting_ops` ("vllm::<name>": a custom op whose Python body is a
  module-level function `<name>`);
- with breakable CUDA graphs (compile mode NONE, a PIECEWISE cudagraph mode), the module-level
  functions decorated with `eager_break_during_capture`: the capture ends its segment, runs the
  op eagerly, and replays it eagerly every step.

A FULL graph captures those ops too. So, by cudagraph mode:

| mode               | prefill / mixed steps        | decode-only steps   |
|--------------------|------------------------------|---------------------|
| PIECEWISE          | split ops in Python          | split ops in Python |
| FULL_AND_PIECEWISE | split ops in Python          | full graph          |
| FULL_DECODE_ONLY   | no graph (eager, if compile NONE) | full graph     |
| FULL               | full graph                   | full graph          |

Whether a target is reached through a split op is read from its module's source: the split op
functions of that module are the roots, and calls `self.<name>(...)` or `<name>(...)` lead to
the module's functions and methods of that name. The target is "split" only if a root reaches
it and every caller of its name in the module is reached from a root too. A target the module
does not define at top level or in a top-level class, or one also called from code no root
reaches, is "unknown": the check stays blocking and says why. Calls from other modules are not
seen (static, this module only). Steps above the largest capture size run with no graph; the
table does not count them.

The split decorator itself is graph machinery, wherever it is defined: its body runs once per
op, when vLLM decorates the op at import, and decides there whether and how the op breaks the
graphs; its wrapper's capture path runs only while a graph is captured. The decorated ops and
the captured graphs keep the old decision, so a swap of it changes nothing the engine runs
(vLLM #59941 removed an early `return fn` from the decorator's body: live code swap committed, witness 0 on
every rank). `decoration_time` names such targets, in any module.
"""
import ast
import hashlib

SPLIT_DECORATOR = "eager_break_during_capture"
PIECEWISE = ("PIECEWISE", "FULL_AND_PIECEWISE")


def graph_mode(worker):
    """{"cudagraph_mode", "compile_mode", "splitting_ops", "max_capture"} when this worker runs
    captured graphs or compiled model code, from its vLLM config; None when it runs eager (or
    has no config)."""
    cfg = getattr(worker, "vllm_config", None)
    cc = getattr(cfg, "compilation_config", None)
    if cc is None or getattr(getattr(cfg, "model_config", None), "enforce_eager", False):
        return None
    cg = getattr(getattr(cc, "cudagraph_mode", None), "name", None) or "NONE"
    mode = getattr(getattr(cc, "mode", None), "name", None) or "NONE"
    if cg == "NONE" and mode == "NONE":
        return None
    return {"cudagraph_mode": cg, "compile_mode": mode,
            "splitting_ops": list(getattr(cc, "splitting_ops", None) or []),
            "max_capture": getattr(cc, "max_cudagraph_capture_size", None)}


def is_eager_break(fn):
    """Is fn the wrapper `eager_break_during_capture` puts around a custom op's body?"""
    code = getattr(fn, "__code__", None)
    return (code is not None and hasattr(fn, "__wrapped__")
            and code.co_qualname.startswith(SPLIT_DECORATOR + "."))


def split_roots(mod, g):
    """Names of the module-level functions of `mod` at which the engine splits its graphs, for
    graph mode g (see the module doc)."""
    names = set()
    if g["compile_mode"] != "NONE":
        names |= {op.split("::")[-1] for op in g["splitting_ops"]}
    elif g["cudagraph_mode"] in PIECEWISE:      # breakable graphs
        names |= {n for n, v in vars(mod).items()
                  if is_eager_break(v) and getattr(v, "__module__", None) == mod.__name__}
    return sorted(n for n in names if callable(getattr(mod, n, None)))


def _defs(tree):
    """{qualname: def node} for module-level functions and methods of top-level classes."""
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = n
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{n.name}.{m.name}"] = m
    return out


def _calls(node):
    """(bare_names, self_attrs) called inside `node` as `<name>(...)` or `self.<name>(...)`."""
    bare_names, self_attrs = set(), set()
    for call_node in ast.walk(node):
        if not isinstance(call_node, ast.Call):
            continue
        func = call_node.func
        if isinstance(func, ast.Name):
            bare_names.add(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id == "self":
            self_attrs.add(func.attr)
    return bare_names, self_attrs


_TREES = {}


def _index(source):
    key = hashlib.sha256(source.encode()).hexdigest()
    hit = _TREES.get(key)
    if hit is None:
        defs = _defs(ast.parse(source))
        raw_calls = {qual: _calls(node) for qual, node in defs.items()}
        by_name = {}
        for qual in defs:
            by_name.setdefault(qual.rsplit(".", 1)[-1], []).append(qual)
        calls = {}
        for caller_qual, (bare_names, self_attrs) in raw_calls.items():
            targets = set()
            for bare_name in bare_names:
                if bare_name in defs:
                    targets.add(bare_name)
                else:
                    targets.update(by_name.get(bare_name, ()))
            class_prefix = caller_qual.split(".", 1)[0] + "." if "." in caller_qual else None
            for attr_name in self_attrs:
                if class_prefix and class_prefix + attr_name in defs:
                    targets.add(class_prefix + attr_name)
                else:
                    targets.update(by_name.get(attr_name, ()))
            calls[caller_qual] = targets
        hit = _TREES[key] = (defs, calls, by_name)
        while len(_TREES) > 32:
            _TREES.pop(next(iter(_TREES)))
    return hit


def reach(source, roots, qual):
    """("split", detail) when the split ops `roots` (module-level function names) reach `qual`
    and nothing else in the module calls it; ("captured", detail) when no root reaches it;
    ("unknown", reason) when this cannot be decided statically."""
    defs, calls, _by_name = _index(source)
    if qual not in defs:
        return "unknown", (f"{qual} is not a module-level function or a method of a "
                           "top-level class")

    def closure(starts, stop=()):
        seen, todo = set(), [root for root in starts if root in defs and root not in stop]
        while todo:
            current = todo.pop()
            if current in seen or current in stop:
                continue
            seen.add(current)
            for target_qual in calls[current]:
                if target_qual not in stop:
                    todo.append(target_qual)
        return seen
    via = [root for root in roots if qual in closure([root])]
    if not via:
        return "captured", (f"no op that splits the graph in this module ({roots or 'none'}) "
                            f"calls {qual}")
    reached = closure(roots)
    split_stop = set(roots) - {qual}
    outside = sorted(
        caller_qual for caller_qual in defs
        if caller_qual != qual and caller_qual not in reached
        and qual in closure([caller_qual], stop=split_stop)
    )
    if outside:
        return "unknown", (f"{qual} is reached through {via}, but also called from {outside}, "
                           "which no split op reaches (they may run inside the graphs)")
    return "split", f"{qual} is reached only through the split op(s) {via}"


def decoration_time(qual):
    """Why the target `qual` (a qualname) runs only when vLLM decorates its ops or captures its
    graphs, or None: the split decorator, or code nested in it."""
    if qual == SPLIT_DECORATOR or qual.startswith(SPLIT_DECORATOR + ".<locals>."):
        return (f"{qual} is the decorator that makes ops break the graphs: it ran when vLLM "
                "decorated its ops at import, and its wrapper's capture path runs only while "
                "graphs are captured; the decorated ops and the captured graphs keep the old "
                "code's decision")
    return None


def where(g, kind):
    """Where the engine runs a target of reach kind `kind` ("split", "captured", "unknown")
    in graph mode g: {"prefill": bool, "decode": bool} (True: in Python)."""
    if kind == "unknown":
        return {"prefill": False, "decode": False}
    cudagraph_mode = g["cudagraph_mode"]
    compiled = g["compile_mode"] != "NONE"
    split = kind == "split"
    prefill = (split and (cudagraph_mode in PIECEWISE or cudagraph_mode == "NONE")) or (
        cudagraph_mode == "FULL_DECODE_ONLY" and (split or not compiled))
    decode = split and cudagraph_mode in ("PIECEWISE", "NONE")
    return {"prefill": prefill, "decode": decode}
