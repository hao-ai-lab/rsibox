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
`collective_rpc("fastrsi_kernel", args=(method, payload))` from inside its busy loop, so every
rank handles the same call between the same two steps. The rank's endpoint is made at its first
call. Lab use only: it runs code sent to it.
"""
import os
import time
import traceback

from kernel import coordinator as C
from kernel import findings as F

METHODS = ("prepare", "validate", "commit", "abort", "undo", "status", "proof",
           "witness_check", "reset", "at_base", "fingerprint")


class Endpoint:
    def __init__(self, role, scope=None, rank=None, **participant):
        self.role, self.rank = role, rank
        self.scope = dict(scope or {})
        # participant: LocalParticipant keywords an engine side adds (checks, capturing, ...)
        self.lp = C.LocalParticipant(role, scope=self.scope, **participant)
        self._abstain = {}     # txn -> state, for plans that give this role nothing

    def who(self):
        return self.role if self.rank is None else f"{self.role} rank {self.rank}"

    def subplan(self, plan):
        """The modules and repair cells of `plan` that run on this role."""
        mods = {m: s for m, s in (plan.get("modules") or {}).items()
                if self.role in s.get("on", [self.role])}
        reps = [r for r in plan.get("repairs") or [] if self.role in r.get("on", [self.role])]
        return {**plan, "modules": mods, "repairs": reps}

    def prepare(self, plan):
        """Prepare this role's sub-plan, or abstain when the plan changes only other roles."""
        sub = self.subplan(plan)
        if sub["modules"] or sub["repairs"] or not (plan.get("modules") or plan.get("repairs")):
            return self.lp.prepare(sub)
        self._abstain[plan["id"]] = "prepared"
        return F.ok(staged={}, abstain=True, note=f"no module or cell for role {self.role}")

    def local(self, method, payload):
        """The kernel participant's method, or the abstaining role's answer."""
        txn, _ = C.txn_ref(payload)
        if not isinstance(txn, str) or txn not in self._abstain:
            return getattr(self.lp, method)(payload)
        nxt = {"commit": "committed", "abort": "aborted", "undo": "undone"}.get(method)
        if nxt and not (method == "abort" and self._abstain[txn] == "committed"):
            self._abstain[txn] = nxt
        if method == "status":
            return F.ok(txn=txn, state=self._abstain[txn], abstain=True)
        if method == "proof":
            return F.ok(hashes={}, witness={}, abstain=True)
        if method == "witness_check":
            return F.ok(counts={}, findings=[], abstain=True)
        return F.ok(abstain=True)

    def dispatch(self, method, payload=None):
        t0 = time.perf_counter()
        try:
            r = self._dispatch(method, payload)
        except BaseException as e:   # never raise across the RPC, not even SystemExit
            r = F.not_ok(f"{method} on {self.who()}: {type(e).__name__}: {e}",
                         traceback=traceback.format_exc()[-1500:])
        if not isinstance(r, dict) or "ok" not in r:
            r = F.not_ok(f"{method}: bad reply {r!r}")
        r.setdefault("pid", os.getpid())
        r.setdefault("role", self.role)
        if self.rank is not None:
            r["rank"] = self.rank
        r["side_s"] = round(time.perf_counter() - t0, 6)
        return r

    def _dispatch(self, method, payload):
        if method not in METHODS:
            return F.not_ok(f"unknown method {method}")
        if method == "prepare":
            return self.prepare(payload)
        if method in ("reset", "fingerprint"):
            return getattr(self.lp, method)(payload)
        return self.local(method, payload)


_ENDPOINT = []


def _wrap(side, fn, first):
    def wrapped(*a, **k):
        side.tick(first)
        return fn(*a, **k)
    wrapped.__wrapped__ = fn
    return wrapped


class KernelWorkerExtension:
    def fastrsi_kernel_endpoint(self):
        """This rank's side (side.py), made at the first call. It wraps this rank's
        execute_model (a step starts) and sample_tokens (the same step, later) to keep the
        side's per-step version log."""
        if not _ENDPOINT:
            from .side import Side
            side = Side("worker", scope={"worker": self}, rank=getattr(self, "rank", None))
            _ENDPOINT.append(side)
            try:
                self.execute_model = _wrap(side, self.execute_model, True)
                self.sample_tokens = _wrap(side, self.sample_tokens, False)
            except Exception:              # the version log is optional; the protocol is not
                pass
        return _ENDPOINT[0]

    def fastrsi_kernel(self, method, payload=None):
        """The kernel participant protocol on this rank. Returns an ok-dict, never raises."""
        try:
            ep = self.fastrsi_kernel_endpoint()
        except BaseException as e:
            return {"ok": False, "error": f"endpoint: {type(e).__name__}: {e}",
                    "rank": getattr(self, "rank", None), "pid": os.getpid()}
        return ep.dispatch(method, payload)
