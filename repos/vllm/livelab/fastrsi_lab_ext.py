"""FastRSI live lab for vLLM: install, uninstall and witness, run inside every worker.

Load it with `LLM(..., worker_extension_cls="fastrsi_lab_ext.LabExtension")` and call the
methods with `llm.collective_rpc("fastrsi_install", args=(...))`, so every rank applies the
same change in the same step. Standard library only.

Install rule for OP edits (see repos/vllm/kb/mechanisms.md): swap the code of the innermost
function under any decorators. Decorators such as `@eager_break_during_capture` close over
that inner function, and the registered custom op calls the outer wrapper, so swapping the
wrapper or rebinding the module attribute would change nothing.
"""
import ast
import hashlib
import importlib
import os

_SAVED = {}   # (module, name) -> (code, defaults, kwdefaults) of the base function
HITS = "__fastrsi_hits__"


def _innermost(fn):
    seen = 0
    while hasattr(fn, "__wrapped__") and seen < 16:
        fn, seen = fn.__wrapped__, seen + 1
    return fn


def _code_hash(code):
    return hashlib.sha256(code.co_code + repr(code.co_consts).encode()).hexdigest()[:12]


def _rank():
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))


class LabExtension:
    def fastrsi_install(self, module_name, func_name, patched_source):
        """Compile `func_name` from the patched module source and swap it in."""
        mod = importlib.import_module(module_name)
        tree = ast.parse(patched_source, filename=mod.__file__)
        # Bring in names the patched module imports and this module does not have yet.
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [(a.asname or a.name).split(".")[0] for a in node.names]
                if any(n not in mod.__dict__ for n in names):
                    exec(compile(ast.Module([node], []), mod.__file__, "exec"), mod.__dict__)
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
        if len(defs) != 1:
            raise ValueError(f"{func_name}: found {len(defs)} definitions in the patched source")
        fdef = defs[0]
        fdef.decorator_list = []
        # Witness: count calls of the new code, per process.
        key = ast.Constant(f"{module_name}.{func_name}")
        hits = ast.Name(HITS, ast.Load())
        fdef.body.insert(0, ast.Assign(
            targets=[ast.Subscript(ast.Name(HITS, ast.Load()), key, ast.Store())],
            value=ast.BinOp(ast.Call(ast.Attribute(hits, "get", ast.Load()),
                                     [key, ast.Constant(0)], []), ast.Add(), ast.Constant(1))))
        mod.__dict__.setdefault(HITS, {})
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module([fdef], [])), mod.__file__, "exec"),
             mod.__dict__, ns)
        new = ns[func_name]
        target = _innermost(getattr(mod, func_name))
        if target.__code__.co_freevars != new.__code__.co_freevars:
            raise ValueError(f"{func_name}: closure variables changed; restart needed")
        k = (module_name, func_name)
        _SAVED.setdefault(k, (target.__code__, target.__defaults__, target.__kwdefaults__))
        target.__code__ = new.__code__
        target.__defaults__, target.__kwdefaults__ = new.__defaults__, new.__kwdefaults__
        mod.__dict__[HITS][f"{module_name}.{func_name}"] = 0
        return {"rank": _rank(), "pid": os.getpid(), "installed": f"{module_name}.{func_name}",
                "base_hash": _code_hash(_SAVED[k][0]), "new_hash": _code_hash(new.__code__)}

    def fastrsi_uninstall(self, module_name, func_name):
        mod = importlib.import_module(module_name)
        k = (module_name, func_name)
        if k not in _SAVED:
            return {"rank": _rank(), "uninstalled": None}
        target = _innermost(getattr(mod, func_name))
        target.__code__, target.__defaults__, target.__kwdefaults__ = _SAVED.pop(k)
        return {"rank": _rank(), "uninstalled": f"{module_name}.{func_name}",
                "now_hash": _code_hash(target.__code__)}

    def fastrsi_witness(self, module_name):
        mod = importlib.import_module(module_name)
        return {"rank": _rank(), "pid": os.getpid(), "hits": dict(mod.__dict__.get(HITS, {}))}
