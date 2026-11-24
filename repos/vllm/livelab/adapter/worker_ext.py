"""The participant protocol in one vLLM process, and the worker extension that serves it.

`Endpoint(role)` wraps one kernel `LocalParticipant` (a worker rank, or the engine core) and
answers the participant protocol by name: `dispatch(method, payload)` returns an ok-dict and
never raises. A vLLM collective_rpc whose method raises on one rank leaves the other ranks'
replies queued in the executor, and every later call then reads the reply of the call before
it (the prototype hit it): a crash here is a not-ok reply, tagged with this process's role,
rank and pid.

- A plan module runs only on the roles in its `"on"` list (default: every role), so one plan
  can swap a worker function and an engine-core function. Repair cells too (`{"cell",
  "declares", "on"}`): a cell runs at commit with the role's scope (`worker`: this rank's
  worker; `core`: the EngineCore) and the kernel's `journal`.
- A role the plan gives no module and no cell abstains: it votes yes, and its commit, abort and
  undo change nothing. Without that, the kernel calls its empty sub-plan an `empty_change` and
  vetoes every one-role edit. A plan that changes nothing anywhere still goes to the kernel.

`KernelWorkerExtension` is vLLM's `--worker-extension-cls` (each rank serves a `Side`, side.py, built
on this `Endpoint`): the engine-core host (host.py) calls
`collective_rpc("rsibox_kernel", args=(method, payload))` from inside its busy loop, so every
rank handles the same call between the same two steps. The rank's endpoint is made at its first
call. Lab use only: it runs code sent to it.
"""
import os
import traceback

from rsibox.lab import ENDPOINT_METHODS as METHODS, RoleEndpoint as Endpoint  # noqa: F401
from kernel import findings as F


_ENDPOINT = []


def _wrap(side, fn, first):
    def wrapped(*a, **k):
        side.tick(first)
        return fn(*a, **k)
    wrapped.__wrapped__ = fn
    return wrapped


def install_replay_guard(worker, guard):
    """Attach `guard` onto replay-capable graph wrappers on `worker.model_runner` (idempotent)."""
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return 0
    attached = 0
    candidates = []
    mgr = getattr(runner, "cudagraph_manager", None)
    if mgr is not None:
        candidates.append(mgr)
        br = getattr(mgr, "breakable_cg_runner", None)
        if br is not None:
            candidates.append(br)
    for attr in ("cudagraph_wrappers", "graph_wrappers"):
        extra = getattr(runner, attr, None)
        if extra:
            candidates.extend(extra)
    for obj in candidates:
        if callable(getattr(obj, "replay", None)):
            guard.attach_wrapper(obj)
            attached += 1
    return attached


class KernelWorkerExtension:
    def rsibox_kernel_endpoint(self):
        """This rank's side (side.py), made at the first call. It wraps this rank's
        execute_model (a step starts) and sample_tokens (the same step, later) to keep the
        side's per-step version log and installs zero-overhead CUDA-graph replay guards."""
        if not _ENDPOINT:
            from .side import Side
            side = Side("worker", scope={"worker": self}, rank=getattr(self, "rank", None))
            _ENDPOINT.append(side)
            try:
                self.execute_model = _wrap(side, self.execute_model, True)
                self.sample_tokens = _wrap(side, self.sample_tokens, False)
            except Exception:              # the version log is optional; the protocol is not
                pass
            try:
                install_replay_guard(self, side.graph_guard)
            except Exception:
                pass
        return _ENDPOINT[0]

    def rsibox_graph_guard(self):
        """Return this rank's `GraphReplayGuard`."""
        return self.rsibox_kernel_endpoint().graph_guard

    def rsibox_check_graph_replay(self, key, input_sig=None):
        """O(1) fail-closed check before replaying `key` (zero overhead at epoch 0)."""
        guard = self.rsibox_kernel_endpoint().graph_guard
        guard.check_before_replay(key, input_sig)

    def rsibox_kernel(self, method, payload=None):
        """The kernel participant protocol on this rank. Returns an ok-dict, never raises."""
        try:
            endpoint = self.rsibox_kernel_endpoint()
        except BaseException as error:
            return F.not_ok(
                f"endpoint: {type(error).__name__}: {error}",
                traceback=traceback.format_exc()[-1500:],
                rank=getattr(self, "rank", None),
                pid=os.getpid(),
            )
        return endpoint.dispatch(method, payload)
