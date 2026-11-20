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
        """Translate `fastrsi conformance sglang --real ...` flags into `FASTRSI_SGLANG_*` env vars."""
        clean = [tok for tok in argv if tok != "--"]
        ap = argparse.ArgumentParser(prog="fastrsi conformance sglang --real", allow_abbrev=False)
        ap.add_argument("--model", default=None, help="model path or HuggingFace repo id")
        ap.add_argument("--tp", type=int, default=None, help="tensor-parallel size")
        ap.add_argument("--mem-fraction", type=float, default=None, help="static GPU memory fraction")
        ap.add_argument("--max-bs", type=int, default=None, help="CUDA graph max batch size")
        ap.add_argument("--port", type=int, default=None, help="engine port merged into extra kwargs")
        ap.add_argument("--work", default=None, help="scratch directory for temporary files (TMPDIR)")
        ap.add_argument("--disable-cuda-graph", action="store_true", help="disable CUDA graphs")
        ap.add_argument("--extra", default=None, help="JSON object of extra sglang.Engine kwargs")
        ap.add_argument("--check-timeout", type=float, default=None, help="workload check timeout (s)")
        ap.add_argument("--rpc-timeout", type=float, default=None, help="control RPC timeout (s)")
        ns = ap.parse_args(clean)
        env = {}
        if ns.model:
            env["FASTRSI_SGLANG_MODEL"] = ns.model
        if ns.tp is not None:
            env["FASTRSI_SGLANG_TP"] = str(ns.tp)
        if ns.mem_fraction is not None:
            env["FASTRSI_SGLANG_MEM_FRACTION"] = str(ns.mem_fraction)
        if ns.max_bs is not None:
            env["FASTRSI_SGLANG_MAX_BS"] = str(ns.max_bs)
        if ns.work:
            env["TMPDIR"] = os.path.abspath(ns.work)
        if ns.check_timeout is not None:
            env["FASTRSI_SGLANG_CHECK_TIMEOUT"] = str(ns.check_timeout)
        if ns.rpc_timeout is not None:
            env["FASTRSI_SGLANG_RPC_TIMEOUT"] = str(ns.rpc_timeout)
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
            env["FASTRSI_SGLANG_EXTRA_ARGS"] = json.dumps(extra)
        return env

    def __init__(self, receipts=None, rpc_timeout=None, check_timeout=None):
        rpc_timeout = float(rpc_timeout if rpc_timeout is not None else os.environ.get("FASTRSI_SGLANG_RPC_TIMEOUT", "180.0"))
        check_timeout = float(check_timeout if check_timeout is not None else os.environ.get("FASTRSI_SGLANG_CHECK_TIMEOUT", "15.0"))
        shared = SglangConformanceLab._shared
        if shared is not None and not shared.dirty and shared.status().get("state") == "base":
            self.__dict__.update(shared.__dict__)
            self.writer = self.co.writer = R.ReceiptWriter(receipts) if receipts else None
            self.reset("state")
            return
        if shared is not None: shared._shutdown_engine()
        asyncio.events._set_running_loop(None); _LOOPS.append(loop := asyncio.new_event_loop()); asyncio.set_event_loop(loop)
        super().__init__(sglang_env_config(), (SAMPLER, TP_WORKER), receipts, rpc_timeout, check_timeout)
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
