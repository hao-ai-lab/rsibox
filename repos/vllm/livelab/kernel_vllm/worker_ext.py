"""Worker side: a vLLM worker-extension class (`--worker-extension-cls
kernel_vllm.worker_ext.KernelWorkerExtension`). The engine-core host calls it with
`model_executor.collective_rpc("fastrsi_kernel", args=(method, payload))`, from inside its busy
loop, so every rank handles the same call between the same two steps.

The host sends one `warm` call to every rank when the engine core starts (kernel_vllm.host),
outside any step gap, so each rank creates its side, and the side its analyzer child, at engine
start: the first real prepare spawns nothing (a cold spawn cost the first edit 1.5-2.2 s).

When it creates the side, the extension wraps this rank's `execute_model` and `sample_tokens`
(instance attributes) to keep the side's per-step version log: `execute_model` starts a step,
and `sample_tokens` checks the version did not change inside the step."""
from .side import Side

_SIDE = []


def _wrap(side, fn, first):
    def wrapped(*a, **k):
        side.tick(first)
        return fn(*a, **k)
    wrapped.__wrapped__ = fn
    return wrapped


class KernelWorkerExtension:
    def fastrsi_kernel_side(self):
        """This rank's side, created on first use (the host's `warm` call at engine start)."""
        if not _SIDE:
            side = Side("worker", scope={"worker": self}, rank=getattr(self, "rank", None))
            _SIDE.append(side)
            try:
                self.execute_model = _wrap(side, self.execute_model, True)
                self.sample_tokens = _wrap(side, self.sample_tokens, False)
            except Exception:              # the version log is optional; the protocol is not
                pass
        return _SIDE[0]

    def fastrsi_kernel(self, method, payload=None):
        """The kernel participant protocol on this rank. Returns an ok-dict, never raises."""
        try:
            side = self.fastrsi_kernel_side()
        except Exception as e:
            return {"ok": False, "error": f"side: {type(e).__name__}: {e}",
                    "rank": getattr(self, "rank", None)}
        r = side.dispatch(method, payload)
        r["rank"] = getattr(self, "rank", None)
        return r
