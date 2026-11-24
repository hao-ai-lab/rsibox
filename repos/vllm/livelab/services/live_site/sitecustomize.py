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

if os.environ.get("RSIBOX_GMS_DEVICE_UUID_MAP"):
    try:
        import json as _json
        import tempfile as _tempfile
        from gpu_memory_service.common import utils as _gms_utils

        _raw_map = _json.loads(os.environ["RSIBOX_GMS_DEVICE_UUID_MAP"])
        _int_map = {}
        if isinstance(_raw_map, dict):
            for _k, _v in _raw_map.items():
                if isinstance(_k, bool) or not isinstance(_v, str) or not _v.strip():
                    continue
                _idx = int(_k)
                if _idx >= 0:
                    _int_map[_idx] = _v.strip()
        if _int_map:
            _cache = getattr(_gms_utils, "_uuid_cache", None)
            _prev_map = getattr(_gms_utils, "_rsibox_device_uuid_map", None)
            if isinstance(_cache, dict):
                if isinstance(_prev_map, dict):
                    for _old_k in _prev_map:
                        _cache.pop(_old_k, None)
                _cache.update(_int_map)
            _orig_get = getattr(_gms_utils, "_rsibox_orig_get_socket_path", _gms_utils.get_socket_path)
            _orig_inv = getattr(_gms_utils, "_rsibox_orig_invalidate_uuid_cache", _gms_utils.invalidate_uuid_cache)
            _gms_utils._rsibox_orig_get_socket_path = _orig_get
            _gms_utils._rsibox_orig_invalidate_uuid_cache = _orig_inv
            _gms_utils._rsibox_device_uuid_map = dict(_int_map)

            def _rsibox_get_socket_path(device, tag="weights"):
                _m = getattr(_gms_utils, "_rsibox_device_uuid_map", None) or {}
                _u = _m.get(device)
                if _u is not None:
                    _d = os.environ.get("GMS_SOCKET_DIR") or _tempfile.gettempdir()
                    return os.path.join(_d, f"gms_{_u}_{tag}.sock")
                return _orig_get(device, tag)

            def _rsibox_invalidate():
                _orig_inv()
                _c = getattr(_gms_utils, "_uuid_cache", None)
                _m = getattr(_gms_utils, "_rsibox_device_uuid_map", None)
                if isinstance(_c, dict) and isinstance(_m, dict):
                    _c.update(_m)

            _gms_utils.get_socket_path = _rsibox_get_socket_path
            _gms_utils.invalidate_uuid_cache = _rsibox_invalidate
    except Exception:  # noqa: BLE001 - never stop the engine from starting
        pass

for _entry in sys.path:
    _cand = os.path.join(_entry or os.getcwd(), "sitecustomize.py")
    if os.path.abspath(os.path.dirname(_cand)) == _here or not os.path.isfile(_cand):
        continue
    _spec = importlib.util.spec_from_file_location("_rsibox_next_sitecustomize", _cand)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    break
