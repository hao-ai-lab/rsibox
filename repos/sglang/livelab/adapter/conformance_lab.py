"""Real-engine conformance wrapper (`LabUnderTest`) around `sglang.Engine`."""
import asyncio, atexit
from adapter.fixture.harness import sglang_env_config, sglang_patches
from adapter.lab import SglangLab
from kernel import receipt as R

SAMPLER, TP_WORKER, _LOOPS = "sglang.srt.layers.sampler", "sglang.srt.managers.tp_worker", []

class SglangConformanceLab(SglangLab):
    name, _shared = "sglang", None
    def __init__(self, receipts=None, rpc_timeout=180.0, check_timeout=15.0):
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
