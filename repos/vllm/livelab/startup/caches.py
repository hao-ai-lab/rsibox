"""Point every compiler and autotune cache at one folder, so later starts with the same config
reuse compiled GPU kernels and tuning results instead of rebuilding them (compile once).

`snapshot()` counts files per cache before and after a start. New files mean that start had to
compile or tune something, a cache miss.

Env var names marked (verify) are from library docs/source and are checked by `snapshot`
showing growth in the expected folder on a cold start.
"""
import json, os, sys
from pathlib import Path

LAYOUT = {
    "VLLM_CACHE_ROOT": "vllm",                 # torch.compile artifacts, FlashInfer autotune winners, DeepGEMM
    "TRITON_CACHE_DIR": "triton",              # Triton kernels
    "TILELANG_CACHE_DIR": "tilelang",          # TileLang kernels (verify)
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",  # FlashInfer JIT modules (verify)
    "TORCHINDUCTOR_CACHE_DIR": "inductor",      # Inductor
    "CUDA_CACHE_PATH": "cuda",                  # driver PTX JIT cache
}


def env(root):
    root = Path(root)
    return {k: str(root / sub) for k, sub in LAYOUT.items()}


def snapshot(root):
    """Files and bytes per cache folder."""
    out = {}
    for k, sub in LAYOUT.items():
        p = Path(root) / sub
        n = size = 0
        for f in (p.rglob("*") if p.exists() else []):
            try:
                if f.is_file():
                    n, size = n + 1, size + f.stat().st_size
            except FileNotFoundError:   # a temp file removed while we walk
                pass
        out[sub] = {"files": n, "bytes": size}
    return out


def diff(before, after):
    """Which caches grew during a start (a miss) and by how much."""
    empty = {"files": 0, "bytes": 0}
    return {k: {"new_files": after[k]["files"] - before.get(k, empty)["files"],
                "new_bytes": after[k]["bytes"] - before.get(k, empty)["bytes"]}
            for k in after if after[k] != before.get(k)}


if __name__ == "__main__":
    print(json.dumps(snapshot(sys.argv[1]), indent=1))
