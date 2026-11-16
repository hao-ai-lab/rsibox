"""Ray worker setup hook: register the `fastrsi` trainer mode when verl loads."""
import importlib, importlib.abc, importlib.util, sys

TARGET = "verl.trainer.ppo.v1"


class _AfterImport(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._busy = False

    def find_spec(self, name, path=None, target=None):
        if self._busy or name != TARGET:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_register(loaded_module):
            exec_module(loaded_module)
            sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _AfterImport)]
            importlib.import_module("adapter.gate").register()

        spec.loader.exec_module = exec_and_register
        return spec


def setup():
    if TARGET in sys.modules:
        importlib.import_module("adapter.gate").register()
    elif not any(isinstance(f, _AfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterImport())
