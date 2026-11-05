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
- `plan["faults"] = {role or "role:rank": method or {"method", "delay_s", "fail"}}` makes that
  method return not-ok on that role (or on one rank of it), or delay its reply by `delay_s`
  first (`fail` false: delay only): a test hook for forced failures and slow ranks, named in
  the reply and the receipt.
- A per-step version log (`tick`): at every step the side records which code version each
  watched function runs (the targets of every plan it prepared), as transitions
  `[step, {key: tag}]`, and counts steps whose version changed between the step's two calls
  (execute_model and sample_tokens on a worker). Comparing the logs of all ranks shows whether
  any step ran mixed versions (`versions`).
"""
import importlib
import os
import sys
import threading
import time
import traceback

from kernel import coordinator as C
from kernel import findings as F

METHODS = ("prepare", "prepare_start", "prepare_poll", "validate", "commit", "abort", "undo",
           "status", "proof", "witness_check", "reset", "versions")


class Side:
    def __init__(self, role, scope=None, resets=None, rank=None):
        self.role, self.rank = role, rank
        self.lp = C.LocalParticipant(role, scope=scope, resets=resets)
        self._jobs = {}        # txn -> {"thread", "result", "t0"}
        self._faults = {}      # txn -> {"method", "delay_s", "fail"} for this role / rank
        self.watch = {}        # key -> function object whose __code__ is the live version
        self._tags = {}        # id(code) -> (code, tag); the code ref keeps the id unique
        self.steps = 0
        self.vlog = []         # [[step, {key: tag}]] at every version change
        self.mid_step_changes = []
        self._cur = None

    # -- per-step version log -------------------------------------------------------------
    def _tag(self, code):
        e = self._tags.get(id(code))
        if e is None:
            from kernel import locate as L
            e = self._tags[id(code)] = (code, L.content_hash(code)[:10])
        return e[1]

    def _watch_plan(self, plan):
        """Watch every target of the plan this side swaps (module namespace lookup)."""
        for m, spec in plan.get("modules", {}).items():
            mod = sys.modules.get(m) or importlib.import_module(m)
            for qual, _line in spec.get("targets", []):
                obj = mod
                try:
                    for part in qual.split("."):
                        obj = getattr(obj, part)
                except AttributeError:
                    continue
                if hasattr(obj, "__code__"):
                    self.watch.setdefault(f"{m}:{qual}", obj)

    def tick(self, first=True):
        """Called at the start of a step (first=True) and again later in the same step
        (first=False, e.g. sample_tokens): records version changes per step."""
        if not self.watch:
            if first:
                self.steps += 1
            return
        cur = {k: self._tag(f.__code__) for k, f in list(self.watch.items())}
        if first:
            self.steps += 1
            if cur != self._cur:
                self.vlog.append([self.steps, cur])
                self._cur = cur
        elif cur != self._cur:
            self.mid_step_changes.append([self.steps, cur])

    def versions(self, _payload=None):
        return F.ok(steps=self.steps, vlog=self.vlog, mid_step_changes=self.mid_step_changes,
                    watch=sorted(self.watch))

    def _subplan(self, plan):
        mods = {m: s for m, s in plan.get("modules", {}).items()
                if self.role in s.get("on", [self.role])}
        faults = plan.get("faults") or {}
        fault = faults.get(f"{self.role}:{self.rank}") or faults.get(self.role)
        if fault:
            if isinstance(fault, str):
                fault = {"method": fault}
            self._faults[plan["id"]] = {"delay_s": 0, "fail": True, **fault}
        return {**plan, "modules": mods, "repairs": []}

    def _fault(self, method, txn):
        f = self._faults.get(txn)
        if not f or f["method"] != method:
            return None
        if f["delay_s"]:
            time.sleep(f["delay_s"])
        if not f["fail"]:
            return None
        who = self.role if self.rank is None else f"{self.role} rank {self.rank}"
        return F.not_ok(f"injected fault: {who} {method}", [{**F.finding(
            "participant_failed", f"injected fault at {method} on {who} (test hook)",
            f"{method} ok"), "participant": self.role, "rank": self.rank}])

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
            self._watch_plan(plan)
            return self._fault("prepare", plan["id"]) or self.lp.prepare(plan)
        if method == "prepare_start":
            plan = self._subplan(payload)
            self._watch_plan(plan)
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
        if method == "versions":
            return self.versions()
        return self._fault(method, payload) or getattr(self.lp, method)(payload)
