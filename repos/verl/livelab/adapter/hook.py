"""Ray worker setup hook: register the `rsibox` trainer mode when verl loads."""
import importlib

import adapter  # noqa: F401  (ensures core/livelab is on sys.path)
from rsibox import hooks

TARGET = "verl.trainer.ppo.v1"


def _register(_module):
    importlib.import_module("adapter.gate").register()


def setup():
    hooks.install_import_hook(TARGET, _register, once=True)
