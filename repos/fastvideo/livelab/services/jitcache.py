"""Path-stable JIT caches for a fresh FastVideo process: the warm restart with
path-stable caches.

A fresh FastH3 process compiles nothing when every JIT cache it reads is warm *at the same
path*. Measured (kb/costs.md, "Why a restart costs what it costs"):

- FlashInfer's JIT modules (NVFP4 `fp4_quantization`, `fp4_gemm_cutlass`) rebuild when the
  cache root moves: FlashInfer always runs ninja, and ninja rebuilds when a command line
  changed, and the command lines hold the absolute cache path. A byte-identical copy of a
  warm cache at a new path cost 159 s of rebuild. Triton's and Inductor's caches are keyed
  by content and survive a move.
- Triton autotuning (VSA `_attn_fwd_sparse`: 12 benchmark rounds, about 2 s) is not cached
  unless `TRITON_CACHE_AUTOTUNING=1`.
- Inductor's `torch_key()` hashes every torch source file (0.8-2.6 s per process) before its
  first cache lookup, and keeps the result in memory only. `preset_torch_key` stores it on
  disk under a stamp of torch's install record, and presets it in the next process.

`env_for(root)` gives the variables that pin every cache under one root. The root must keep
its real path across restarts and nodes: `stage_root(copy, root)` puts a shipped copy back at
the path it was built at (a symlink does not do: FlashInfer resolves it). `check_root(root)`
records where a root was first used and reports a finding when it is used from another path
(the 153-159 s rebuild), before the process pays it.
"""
import hashlib
import json
import os
import time

ROOTS = {"FLASHINFER_WORKSPACE_BASE": "flashinfer", "TRITON_CACHE_DIR": "triton",
         "TORCHINDUCTOR_CACHE_DIR": "inductor", "XDG_CACHE_HOME": "xdg"}
MARKER = ".fastrsi-jit-root"


def stage_root(src, dst):
    """Put a JIT cache root built at `dst` back at `dst` from a copy `src` (another node, an
    archive, a staging area): a recursive copy that keeps every mtime, so ninja finds its outputs
    up to date. FlashInfer names each output by its *resolved* absolute path
    (`flashinfer/jit/cpp_ext.py`, `(output_dir / obj_name).resolve()`), so a symlink to the copy
    does not keep the path: the copy must live at the real path it was built at."""
    import shutil
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    if os.path.lexists(dst):
        raise FileExistsError(f"{dst} exists: stage into an empty path")
    shutil.copytree(src, dst, symlinks=True, copy_function=shutil.copy2)
    return dst


def env_for(root):
    """Environment that puts every JIT cache under `root` (absolute, symlinks resolved: what
    FlashInfer keys its builds on), autotuning included."""
    root = os.path.realpath(root)
    env = {k: os.path.join(root, sub) for k, sub in ROOTS.items()}
    env["TRITON_CACHE_AUTOTUNING"] = "1"
    env["FASTRSI_TORCH_KEY_CACHE"] = os.path.join(root, "torch_key.json")
    return env


def check_root(root):
    """Record the absolute path a cache root was first used from. Returns a list of findings:
    `jit_root_moved` when the root's marker names another path (a copied or moved cache)."""
    root = os.path.realpath(root)
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, MARKER)
    try:
        with open(marker) as f:
            first = json.load(f).get("path")
    except (OSError, ValueError):
        with open(marker, "w") as f:
            json.dump({"path": root, "t": time.time()}, f)
        return []
    if first != root:
        return [{"check": "jit_root_moved", "observed": f"cache root built at {first}, used at {root}",
                 "expected": "a JIT cache root used from the path it was built at (FlashInfer "
                             "rebuilds its NVFP4 modules after a move: 159 s measured)"}]
    return []


def _stamp(top):
    """What identifies the installed torch without reading it: size and mtime of its install
    record (pip rewrites it on every install) and of the files that define the key. A walk
    over every source file cost 1.5-4.0 s on a network filesystem (2187 files), more than the
    hash it replaces; a hand edit of torch's sources is not seen: delete the cache file."""
    import glob
    site = os.path.dirname(top)
    files = sorted(glob.glob(os.path.join(site, "torch-*.dist-info", "RECORD"))) + [
        os.path.join(top, "version.py"), os.path.join(top, "_inductor", "codecache.py")]
    h = hashlib.sha256()
    n = 0
    for f in files:
        try:
            st = os.stat(f)
        except OSError:
            continue
        h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        n += 1
    return h.hexdigest(), n


def preset_torch_key(path=None):
    """Preset Inductor's `torch_key()` from `path` when the torch sources are unchanged, or
    compute it and store it there. Returns {"hit", "seconds", "files"}; never raises."""
    path = path or os.environ.get("FASTRSI_TORCH_KEY_CACHE")
    if not path:
        return {"hit": False, "seconds": 0.0, "skipped": "no cache path"}
    t0 = time.perf_counter()
    try:
        import torch
        from torch._inductor import codecache
        top = os.path.dirname(torch.__file__)
        stamp, n = _stamp(top)
        stamp = f"{torch.__version__}:{top}:{stamp}"
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = {}
        if rec.get("stamp") == stamp:
            try:
                codecache.torch_key.set(bytes.fromhex(rec["key"]))
            except AssertionError:          # already computed in this process
                pass
            return {"hit": True, "files": n, "seconds": round(time.perf_counter() - t0, 4)}
        key = codecache.torch_key()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"stamp": stamp, "key": key.hex()}, f)
        os.replace(tmp, path)
        return {"hit": False, "files": n, "seconds": round(time.perf_counter() - t0, 4)}
    except Exception as e:
        return {"hit": False, "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.perf_counter() - t0, 4)}
