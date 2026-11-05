"""Worker side: a vLLM worker-extension class (`--worker-extension-cls
kernel_vllm.worker_ext.KernelWorkerExtension`). The engine-core host calls it with
`model_executor.collective_rpc("fastrsi_kernel", args=(method, payload))`, from inside its busy
loop, so every rank handles the same call between the same two steps.

On its first call the extension wraps this rank's `execute_model` and `sample_tokens` (instance
attributes) to keep the side's per-step version log: `execute_model` starts a step, and
`sample_tokens` checks the version did not change inside the step."""
from .side import Side

_SIDE = []


def _wrap(side, fn, first):
    def wrapped(*a, **k):
        side.tick(first)
        return fn(*a, **k)
    wrapped.__wrapped__ = fn
    return wrapped


class KernelWorkerExtension:
    def fastrsi_kernel(self, method, payload=None):
        """The kernel participant protocol on this rank. Returns an ok-dict, never raises."""
        if not _SIDE:
            side = Side("worker", scope={"worker": self}, rank=getattr(self, "rank", None))
            _SIDE.append(side)
            try:
                self.execute_model = _wrap(side, self.execute_model, True)
                self.sample_tokens = _wrap(side, self.sample_tokens, False)
            except Exception:              # the version log is optional; the protocol is not
                pass
        r = _SIDE[0].dispatch(method, payload)
        r["rank"] = getattr(self, "rank", None)
        return r
