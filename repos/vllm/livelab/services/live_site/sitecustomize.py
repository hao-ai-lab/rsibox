"""Loads the live reconfiguration agent (../live_reconfig.py) in every engine process when
RSIBOX_LIVE_RECONFIG_DIR is set, then runs the next `sitecustomize` on sys.path (Python imports
only the first one it finds, so this one must not hide another, such as the lab's startup hooks).
engine.py puts this folder first on the engine's PYTHONPATH."""
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("RSIBOX_LIVE_RECONFIG_DIR"):
    sys.path.insert(0, os.path.dirname(_here))
    try:
        import live_reconfig
        live_reconfig.install()
    except Exception as e:  # noqa: BLE001 - never stop the engine from starting
        print(f"RSIBOX_LIVE_RECONFIG: not installed: {e!r}", file=sys.stderr, flush=True)
    finally:
        sys.path.remove(os.path.dirname(_here))

for _entry in sys.path:
    _cand = os.path.join(_entry or os.getcwd(), "sitecustomize.py")
    if os.path.abspath(os.path.dirname(_cand)) == _here or not os.path.isfile(_cand):
        continue
    _spec = importlib.util.spec_from_file_location("_rsibox_next_sitecustomize", _cand)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    break
