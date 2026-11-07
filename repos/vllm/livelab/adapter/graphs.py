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
(vLLM #59941 removed an early `return fn` from the decorator's body: T0 committed, witness 0 on
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
    """Names this def calls as `self.<name>(...)` or `<name>(...)`."""
    out = set()
    for c in ast.walk(node):
        if not isinstance(c, ast.Call):
            continue
        f = c.func
        if isinstance(f, ast.Name):
            out.add(f.id)
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                and f.value.id == "self":
            out.add(f.attr)
    return out


_TREES = {}


def _index(source):
    key = hashlib.sha256(source.encode()).hexdigest()
    hit = _TREES.get(key)
    if hit is None:
        defs = _defs(ast.parse(source))
        calls = {q: _calls(n) for q, n in defs.items()}
        by_name = {}
        for q in defs:
            by_name.setdefault(q.rsplit(".", 1)[-1], []).append(q)
        hit = _TREES[key] = (defs, calls, by_name)
        while len(_TREES) > 32:
            _TREES.pop(next(iter(_TREES)))
    return hit


def reach(source, roots, qual):
    """("split", detail) when the split ops `roots` (module-level function names) reach `qual`
    and every other caller of it in the module is itself reached from a split op; ("captured", detail) when no root reaches it;
    ("unknown", reason) when this cannot be decided statically."""
    defs, calls, by_name = _index(source)
    if qual not in defs:
        return "unknown", (f"{qual} is not a module-level function or a method of a "
                           "top-level class")

    def closure(starts):
        seen, todo = set(), [r for r in starts if r in defs]
        while todo:
            q = todo.pop()
            if q in seen:
                continue
            seen.add(q)
            for name in calls[q]:
                todo += by_name.get(name, [])
        return seen
    via = [r for r in roots if qual in closure([r])]
    if not via:
        return "captured", (f"no op that splits the graph in this module ({roots or 'none'}) "
                            f"calls {qual}")
    reached = closure(roots)
    name = qual.rsplit(".", 1)[-1]
    outside = sorted(q for q in defs if q != qual and name in calls[q] and q not in reached)
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
    cg, compiled = g["cudagraph_mode"], g["compile_mode"] != "NONE"
    split = kind == "split"
    prefill = (split and (cg in PIECEWISE or cg == "NONE")) or (
        cg == "FULL_DECODE_ONLY" and (split or not compiled))
    decode = split and cg in ("PIECEWISE", "NONE")
    return {"prefill": prefill, "decode": decode}
