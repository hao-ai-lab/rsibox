"""FastRSI live lab for vLLM: install, uninstall and witness, run inside every worker.

Load it with `LLM(..., worker_extension_cls="fastrsi_lab_ext.LabExtension")` and call the
methods with `llm.collective_rpc("fastrsi_install", args=(...))`, so every rank applies the
same change in the same step. Standard library only.

How install works: it replaces the code of the innermost function under any decorators.
Decorators such as `@eager_break_during_capture` keep a reference to that inner function, and
vLLM's registered op calls the outer wrapper, so replacing only the module attribute would not
change what the engine runs. A patch that changes anything besides this one function and its
imports is refused.
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
    """Fingerprint of a code object: bytecode, names and constants, nested code included."""
    h = hashlib.sha256(code.co_code + repr(code.co_names).encode())
    for c in code.co_consts:
        h.update(_code_hash(c).encode() if hasattr(c, "co_code") else repr(c).encode())
    return h.hexdigest()[:12]


def _rank():
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))


class LabExtension:
    def fastrsi_install(self, module_name, func_name, patched_source):
        """Compile `func_name` from the patched module source and swap it in."""
        mod = importlib.import_module(module_name)
        tree = ast.parse(patched_source, filename=mod.__file__)
        # Fail closed: this installs one function. If the patch also changes any other top-level
        # definition or assignment, refuse instead of silently ignoring it.
        base = ast.parse(open(mod.__file__).read())
        def others(t):
            return {ast.dump(n) for n in t.body
                    if not isinstance(n, (ast.Import, ast.ImportFrom))
                    and getattr(n, "name", None) != func_name}
        extra = others(tree) - others(base)
        if extra:
            raise ValueError(f"{func_name}: the patch changes {len(extra)} other top-level "
                             "statement(s); install them too or restart")
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

    def fastrsi_capture(self, sizes_json):
        """Capture CUDA graphs for exactly these batch sizes (tokens per step), now.

        Start the engine with few or no capture sizes, then capture only what the experiment
        runs: e.g. "[1, 128, 2048]". vLLM's capture_model() captures the sizes listed in
        compilation_config.cudagraph_capture_sizes at call time. Runs between engine steps
        (collective_rpc), on every rank. Returns seconds taken and the sizes now captured.
        """
        import json, time
        runner = self.model_runner
        cc = runner.vllm_config.compilation_config
        before = list(cc.cudagraph_capture_sizes or [])
        sizes = sorted({int(s) for s in json.loads(sizes_json)} | set(before))
        if max(sizes) > (cc.max_cudagraph_capture_size or 0):
            raise ValueError(f"size {max(sizes)} is above the engine's max capture size "
                             f"{cc.max_cudagraph_capture_size}; buffers are sized at start")
        cc.cudagraph_capture_sizes = sizes
        # The graph manager plans its captures once at init (_init_candidates). Re-plan for the
        # union of old and new sizes (so dispatch knows them all), then capture only the graphs
        # that do not exist yet.
        mgr = runner.cudagraph_manager
        mgr._candidates, mgr._capture_descs = {}, {}
        mgr._init_candidates()
        for mode, descs in mgr._capture_descs.items():
            mgr._capture_descs[mode] = [d for d in descs if d not in mgr.graphs]
        mgr._graphs_captured = False
        t = time.time()
        runner.capture_model()
        return {"rank": _rank(), "captured": sizes, "previous_sizes": before,
                "seconds": round(time.time() - t, 3)}

    def fastrsi_witness(self, module_name):
        mod = importlib.import_module(module_name)
        return {"rank": _rank(), "pid": os.getpid(), "hits": dict(mod.__dict__.get(HITS, {}))}
