"""Loaded by every Python process whose PYTHONPATH starts with this folder (engine and workers).

- FASTRSI_GRAPHMEM=1: wrap the model runner's graph-memory profiling (graphmem.py) when vLLM
  imports it.
- FASTRSI_HOOKS=mod1,mod2: also import these modules first (machine-specific hooks kept outside
  the repo, e.g. a small-/dev/shm workaround).
"""
import importlib, importlib.abc, importlib.machinery, os, sys

for _m in filter(None, os.environ.get("FASTRSI_HOOKS", "").split(",")):
    importlib.import_module(_m)


class _AfterImport(importlib.abc.MetaPathFinder):
    """Run a callback right after a target module finishes importing."""

    def __init__(self, targets, callback):
        self.targets, self.callback = set(targets), callback

    def find_spec(self, name, path, target=None):
        if name not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        orig = spec.loader.exec_module

        def exec_module(module):
            orig(module)
            self.targets.discard(name)      # only once the module has really loaded
            self.callback(module)
        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("FASTRSI_GRAPHMEM") == "1":
    sys.path.insert(0, os.path.dirname(__file__))
    import graphmem  # noqa: E402
    sys.meta_path.insert(0, _AfterImport(graphmem.RUNNER_MODULES, graphmem.on_import))
