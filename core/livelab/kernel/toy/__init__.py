"""A toy engine for the kernel tests: no ML, standard library only.

- `ops`: plain functions, a decorated one (the wrapper holds the original in a closure cell),
  a closure factory, and a module constant.
- `engine`: imports ops by name (`from .ops import ...` copies), builds an `Engine` whose
  `__init__` sets fields and saves a bound method as a callback, a `step`, and `run_loop`, a
  busy loop whose frame runs for the engine's life and drains work queued for its step
  boundary (the quiesce point).

`load_copy(name, path=None)` loads the package under another name, so several participants
("rank0", "rank1", "core") each get their own module objects from the same files (or, with
`path`, from another directory: a participant running a different build).
"""
import importlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_copy(name, path=None):
    path = path or HERE
    for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    importlib.import_module(name + ".engine")
    return pkg


def source(module):
    """Text of one of the toy's files: source("engine")."""
    with open(os.path.join(HERE, module + ".py")) as f:
        return f.read()
