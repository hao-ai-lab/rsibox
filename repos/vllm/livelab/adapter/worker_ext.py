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
import sys
import traceback

from rsibox.lab import ENDPOINT_METHODS as METHODS, RoleEndpoint as Endpoint  # noqa: F401
from kernel import findings as F
from . import graph_patch as _GP

if os.environ.get("RSIBOX_KERNEL_DIR"):
    _GP.enable_keep_graph()


_ENDPOINT = []


def _wrap(side, fn, first):
    def wrapped(*a, **k):
        side.tick(first)
        return fn(*a, **k)
    wrapped.__wrapped__ = fn
    return wrapped


def _v1_entries(wrapper):
    return next((d for a in ("concrete_cudagraph_entries", "entries") if isinstance(d := getattr(wrapper, a, None), dict)), None)

def _stamp_entries(guard, entries, default_mode):
    from .graph_guard import graph_key
    for desc in (entries or ()):
        guard.stamp(graph_key(getattr(desc, "cg_mode", None) or default_mode, desc))

def _v1_call_desc(entries, mode, args, kwargs):
    if (d := kwargs.get("batch_descriptor")) is not None or (d := kwargs.get("desc")) is not None: return d
    if len(args) >= 2 and args[0] == mode: return args[1]
    if args and isinstance(entries, dict) and args[0] in entries: return args[0]
    get_fc = getattr(sys.modules.get("vllm.forward_context"), "get_forward_context", None)
    if callable(get_fc):
        try: return getattr(get_fc(), "batch_descriptor", None)
        except Exception: pass
    return args[0] if args else None

def _hook_v1_callable(wrapper, guard):
    from .graph_guard import graph_key
    from .side import stream_capturing
    cls = type(wrapper)
    if not getattr(cls, "_rsibox_call_hooked", False):
        orig_call = cls.__call__
        def guarded_call(self, *args, **kwargs):
            if (g := getattr(self, "_rsibox_guard", None)) is not None and not stream_capturing() and not getattr(getattr(self, "runner", None), "capturing", False):
                entries, mode = _v1_entries(self), getattr(self, "runtime_mode", None) or "PIECEWISE"
                desc = _v1_call_desc(entries, mode, args, kwargs)
                if desc is not None and isinstance(entries, dict) and desc in entries:
                    g.check_before_replay(graph_key(getattr(desc, "cg_mode", None) or mode, desc), kwargs.get("input_sig"))
            return orig_call(self, *args, **kwargs)
        guarded_call.__wrapped__, cls.__call__, cls._rsibox_call_hooked = orig_call, guarded_call, True
    wrapper._rsibox_guard = guard

def install_replay_guard(worker, guard):
    """Hook real vLLM V1 (`__call__`) and V2 (`run_fullgraph`) replay paths and stamp startup graphs."""
    runner, attached = getattr(worker, "model_runner", None), 0
    v1 = [w for m, c in (("vllm.compilation.cuda_graph", "CUDAGraphWrapper"), ("vllm.compilation.breakable_cudagraph", "BreakableCUDAGraphWrapper"))
          if (cls := getattr(sys.modules.get(m), c, None)) is not None for w in list(getattr(cls, "_all_instances", ()))]
    if runner is not None:
        if (mgr := getattr(runner, "cudagraph_manager", None)) is not None:
            for meth in ("run_fullgraph", "replay"):
                if callable(getattr(mgr, meth, None)): guard.attach_wrapper(mgr, method=meth); attached += 1
            _stamp_entries(guard, getattr(mgr, "graphs", None), "FULL")
            if (br := getattr(mgr, "breakable_cg_runner", None)) is not None: v1.append(br)
        for attr in ("cudagraph_wrappers", "graph_wrappers"): v1.extend(list(getattr(runner, attr, None) or ()))
    for w in {id(x): x for x in v1}.values():
        if callable(w): _hook_v1_callable(w, guard); attached += 1
        elif callable(getattr(w, "replay", None)): guard.attach_wrapper(w, method="replay"); attached += 1
        _stamp_entries(guard, _v1_entries(w), getattr(w, "runtime_mode", None) or "PIECEWISE")
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
