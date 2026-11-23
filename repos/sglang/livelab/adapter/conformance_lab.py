"""Real-engine conformance wrapper (`LabUnderTest`) around `sglang.Engine`."""
import argparse, asyncio, atexit, json, os
from adapter.fixture.harness import sglang_env_config, sglang_patches
from adapter.lab import SglangLab
from kernel import receipt as R

SAMPLER, TP_WORKER, _LOOPS = "sglang.srt.layers.sampler", "sglang.srt.managers.tp_worker", []

class SglangConformanceLab(SglangLab):
    name, _shared = "sglang", None

    @staticmethod
    def launch_environment(argv):
        """Translate `rsibox conformance sglang --real ...` flags into `RSIBOX_SGLANG_*` env vars."""
        clean = [tok for tok in argv if tok != "--"]
        ap = argparse.ArgumentParser(prog="rsibox conformance sglang --real", allow_abbrev=False)
        ap.add_argument("--model", default=None, help="model path or HuggingFace repo id")
        ap.add_argument("--tp", type=int, default=None, help="tensor-parallel size")
        ap.add_argument("--mem-fraction", type=float, default=None, help="static GPU memory fraction")
        ap.add_argument("--max-bs", type=int, default=None, help="CUDA graph max batch size")
        ap.add_argument("--port", type=int, default=None, help="engine port merged into extra kwargs")
        ap.add_argument("--work", default=None, help="scratch directory for temporary files (TMPDIR)")
        ap.add_argument("--disable-cuda-graph", action="store_true", help="disable CUDA graphs")
        ap.add_argument("--fork-subprocs", action="store_true",
                        help="fork SGLang scheduler/detokenizer subprocesses when host CUDA is clean")
        ap.add_argument("--weight-cache-dir", default=None,
                        help="connect SGLang to a live weight-cache daemon in this socket directory")
        ap.add_argument("--extra", default=None, help="JSON object of extra sglang.Engine kwargs")
        ap.add_argument("--check-timeout", type=float, default=None, help="workload check timeout (s)")
        ap.add_argument("--rpc-timeout", type=float, default=None, help="control RPC timeout (s)")
        ns = ap.parse_args(clean)
        spec = (("model", "RSIBOX_SGLANG_MODEL", False), ("tp", "RSIBOX_SGLANG_TP", False),
                ("mem_fraction", "RSIBOX_SGLANG_MEM_FRACTION", False), ("max_bs", "RSIBOX_SGLANG_MAX_BS", False),
                ("work", "TMPDIR", True), ("check_timeout", "RSIBOX_SGLANG_CHECK_TIMEOUT", False),
                ("rpc_timeout", "RSIBOX_SGLANG_RPC_TIMEOUT", False))
        env = {key: (os.path.abspath(str(val)) if is_p else str(val))
               for attr, key, is_p in spec if (val := getattr(ns, attr)) is not None}
        if ns.fork_subprocs:
            env["RSIBOX_SGLANG_FORK_SUBPROCS"] = "1"
        if ns.weight_cache_dir:
            env["RSIBOX_SGLANG_WEIGHT_CACHE_DIR"] = os.path.abspath(str(ns.weight_cache_dir))
        extra = {}
        if ns.extra is not None:
            try:
                parsed = json.loads(ns.extra)
            except ValueError as exc:
                ap.error(f"--extra must be valid JSON object: {exc}")
            if not isinstance(parsed, dict):
                ap.error("--extra must be a JSON object")
            extra.update(parsed)
        if ns.port is not None:
            extra["port"] = ns.port
        if ns.disable_cuda_graph:
            extra["disable_cuda_graph"] = True
        if ns.extra is not None or extra:
            env["RSIBOX_SGLANG_EXTRA_ARGS"] = json.dumps(extra)
        return env

    def __init__(self, receipts=None, rpc_timeout=None, check_timeout=None, fork_subprocs=None, weight_cache_dir=None):
        rpc_timeout = float(rpc_timeout if rpc_timeout is not None else os.environ.get("RSIBOX_SGLANG_RPC_TIMEOUT", "180.0"))
        check_timeout = float(check_timeout if check_timeout is not None else os.environ.get("RSIBOX_SGLANG_CHECK_TIMEOUT", "15.0"))
        if fork_subprocs is None:
            fork_subprocs = os.environ.get("RSIBOX_SGLANG_FORK_SUBPROCS", "").strip().lower() in ("1", "true", "yes")
        else:
            fork_subprocs = bool(fork_subprocs)
        if weight_cache_dir is None:
            raw_wc = os.environ.get("RSIBOX_SGLANG_WEIGHT_CACHE_DIR", "").strip()
        else:
            raw_wc = os.fspath(weight_cache_dir).strip()
        weight_cache_dir = os.path.abspath(raw_wc) if raw_wc else None
        shared = SglangConformanceLab._shared
        if (shared is not None and getattr(shared, "engine", None) is not None
                and not shared.dirty and shared.status().get("state") == "base"
                and shared.fork_subprocs == fork_subprocs
                and getattr(shared, "weight_cache_dir", None) == weight_cache_dir):
            self.__dict__.update(shared.__dict__)
            self.writer = self.co.writer = R.ReceiptWriter(receipts) if receipts else None
            self.reset("state")
            return
        if shared is not None:
            SglangConformanceLab._shared = None
            shared._shutdown_engine()
        if fork_subprocs:
            SglangLab._preload_fork_subprocs()
        asyncio.events._set_running_loop(None); _LOOPS.append(loop := asyncio.new_event_loop()); asyncio.set_event_loop(loop)
        cfg = sglang_env_config()
        kwargs = {"fork_subprocs": fork_subprocs}
        if weight_cache_dir is not None:
            from services import weights as wc
            cfg = wc.client_config(cfg)
            kwargs["weight_cache_dir"] = weight_cache_dir
        super().__init__(cfg, (SAMPLER, TP_WORKER), receipts, rpc_timeout, check_timeout, **kwargs)
        SglangConformanceLab._shared = self
    def close(self):
        pass
    def reset(self, kind="state"):
        if kind == "respawn": asyncio.events._set_running_loop(None); _LOOPS.append(loop := asyncio.new_event_loop()); asyncio.set_event_loop(loop)
        reply = super().reset(kind)
        if kind == "respawn": SglangConformanceLab._shared = self
        return reply
    def inject(self, fault, **kwargs):
        if fault == "tamper":
            kwargs.update({"module": kwargs.get("module", SAMPLER), "function": kwargs.get("function", "Sampler.forward")})
        return super().inject(fault, **kwargs)
    def patches(self):
        return sglang_patches(self._base, SAMPLER, TP_WORKER)

atexit.register(lambda: SglangConformanceLab._shared and SglangConformanceLab._shared._shutdown_engine())
