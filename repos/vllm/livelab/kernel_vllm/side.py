"""One participant side of the kernel inside a vLLM process: a worker rank or the engine core.

`Side(role)` wraps a kernel LocalParticipant and answers the participant protocol by name
(`dispatch(method, payload)`), always with an ok-dict, never raising: a vLLM collective_rpc
whose method raises on one rank leaves the other ranks' replies queued, and every later reply
is then read by the wrong call (kb/lessons.md).

- Phase 1 runs in a background thread of this process (`prepare_start`, then `prepare_poll`),
  so the engine keeps stepping while the side locates, compiles and stages. The heap scan
  still holds the GIL while it runs, so steps of this process slow down meanwhile.
- A plan module runs only on the roles in its `"on"` list (default: every role), so one plan
  can swap a worker function and an engine-core function.
- `plan["faults"] = {role: method}` makes that method return not-ok on that role: a test hook
  for forced failures, named in the reply and the receipt.
"""
import os
import threading
import time
import traceback

from kernel import coordinator as C
from kernel import findings as F

METHODS = ("prepare", "prepare_start", "prepare_poll", "validate", "commit", "abort", "undo",
           "status", "proof", "witness_check", "reset")


class Side:
    def __init__(self, role, scope=None, resets=None):
        self.role = role
        self.lp = C.LocalParticipant(role, scope=scope, resets=resets)
        self._jobs = {}        # txn -> {"thread", "result", "t0"}
        self._faults = {}      # txn -> method that must fail on this role

    def _subplan(self, plan):
        mods = {m: s for m, s in plan.get("modules", {}).items()
                if self.role in s.get("on", [self.role])}
        fault = (plan.get("faults") or {}).get(self.role)
        if fault:
            self._faults[plan["id"]] = fault
        return {**plan, "modules": mods, "repairs": []}

    def _fault(self, method, txn):
        if self._faults.get(txn) == method:
            return F.not_ok(f"injected fault: {self.role} {method}", [{**F.finding(
                "participant_failed", f"injected fault at {method} (test hook)",
                f"{method} ok"), "participant": self.role}])
        return None

    def dispatch(self, method, payload):
        t0 = time.perf_counter()
        try:
            r = self._dispatch(method, payload)
        except Exception as e:                       # never raise across the RPC
            r = F.not_ok(f"{method}: {type(e).__name__}: {e}",
                         traceback=traceback.format_exc()[-1500:])
        if not isinstance(r, dict) or "ok" not in r:
            r = F.not_ok(f"{method}: bad reply {r!r}")
        r.setdefault("pid", os.getpid())
        r.setdefault("role", self.role)
        r["side_s"] = round(time.perf_counter() - t0, 6)
        return r

    def _dispatch(self, method, payload):
        if method not in METHODS:
            return F.not_ok(f"unknown method {method}")
        if method == "prepare":
            plan = self._subplan(payload)
            return self._fault("prepare", plan["id"]) or self.lp.prepare(plan)
        if method == "prepare_start":
            plan = self._subplan(payload)
            job = {"t0": time.perf_counter(), "result": None}

            def run():
                r = self._fault("prepare", plan["id"]) or self.lp.prepare(plan)
                r["prepare_s"] = round(time.perf_counter() - job["t0"], 6)
                job["result"] = r
            job["thread"] = threading.Thread(target=run, daemon=True,
                                             name=f"fastrsi-prepare-{plan['id']}")
            self._jobs[plan["id"]] = job
            job["thread"].start()
            return F.ok(started=True)
        if method == "prepare_poll":
            job = self._jobs.get(payload)
            if job is None:
                return F.not_ok(f"no prepare started for {payload}")
            if job["result"] is None:
                return F.ok(pending=True)
            return self._jobs.pop(payload)["result"]
        if method == "reset":
            return self.lp.reset(payload)
        return self._fault(method, payload) or getattr(self.lp, method)(payload)
