"""Worker side: a vLLM worker-extension class (`--worker-extension-cls
kernel_vllm.worker_ext.KernelWorkerExtension`). The engine-core host calls it with
`model_executor.collective_rpc("fastrsi_kernel", args=(method, payload))`, from inside its busy
loop, so every rank handles the same call between the same two steps."""
from .side import Side

_SIDE = []


class KernelWorkerExtension:
    def fastrsi_kernel(self, method, payload=None):
        """The kernel participant protocol on this rank. Returns an ok-dict, never raises."""
        if not _SIDE:
            _SIDE.append(Side("worker", scope={"worker": self}))
        r = _SIDE[0].dispatch(method, payload)
        r["rank"] = getattr(self, "rank", None)
        return r
