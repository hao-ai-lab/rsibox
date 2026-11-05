"""Reuse the CUDA graph memory size measured on an earlier start with the same config.

At start-up vLLM captures every CUDA graph once into a scratch pool only to measure how much
memory graphs need, then sizes the KV cache, then captures for real. The measured number depends on the config,
the capture sizes and modes, and the rank. This hook stores it after a cold boot and returns it
on the next boot with the same key instead of capturing.

Enable with FASTRSI_GRAPHMEM=1 (and FASTRSI_GRAPHMEM_DIR=<dir>); installed by sitecustomize.py
when vLLM's worker module is imported. Every reuse is logged ("fastrsi graphmem: reused"), so a
startup trace shows it. Port of the internal fastboot graph-memory piece (vLLM 0.31, one rank)
to multi-rank.

Risk: an edit that changes graph memory but not the config reuses a stale number; the KV cache
is then sized slightly wrong. Key includes the vLLM version and FASTRSI_GRAPHMEM_SALT (set it to
the patch hash when the edit touches captured code).
"""
import functools, hashlib, json, os
from pathlib import Path

RUNNER_MODULES = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")


def _key(runner):
    import torch.distributed as dist
    import vllm
    cfg = runner.vllm_config
    cc = cfg.compilation_config
    ident = {"vllm": vllm.__version__, "config": cfg.compute_hash(),
             "sizes": list(getattr(cc, "cudagraph_capture_sizes", None) or []),
             "mode": str(getattr(cc, "cudagraph_mode", "")),
             "rank": dist.get_rank() if dist.is_initialized() else 0,
             "salt": os.environ.get("FASTRSI_GRAPHMEM_SALT", "")}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:20], ident


def wrap(cls):
    orig = getattr(cls, "profile_cudagraph_memory", None)
    if orig is None or getattr(orig, "_fastrsi", False):
        return

    @functools.wraps(orig)
    def profile_cudagraph_memory(self, *a, **k):
        from vllm.logger import init_logger
        log = init_logger("vllm.fastrsi.graphmem")   # must sit under vllm.*, or vLLM drops INFO lines
        key, ident = _key(self)
        path = Path(os.environ.get("FASTRSI_GRAPHMEM_DIR", "/tmp/fastrsi-graphmem")) / f"{key}.json"
        if path.exists():
            nbytes = json.loads(path.read_text())["bytes"]
            log.info("fastrsi graphmem: reused %d bytes (key %s, rank %s)", nbytes, key, ident["rank"])
            return nbytes
        nbytes = orig(self, *a, **k)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"bytes": int(nbytes), "ident": ident}))
        tmp.replace(path)
        log.info("fastrsi graphmem: measured %d bytes, saved (key %s, rank %s)", nbytes, key, ident["rank"])
        return nbytes

    profile_cudagraph_memory._fastrsi = True
    cls.profile_cudagraph_memory = profile_cudagraph_memory


def on_import(module):
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and "profile_cudagraph_memory" in vars(obj):
            wrap(obj)
