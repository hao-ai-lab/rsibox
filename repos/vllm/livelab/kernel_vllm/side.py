"""One participant side of the kernel inside a vLLM process: a worker rank or the engine core.

`Side(role)` wraps a kernel LocalParticipant and answers the participant protocol by name
(`dispatch(method, payload)`), always with an ok-dict, never raising: a vLLM collective_rpc
whose method raises on one rank leaves the other ranks' replies queued, and every later reply
is then read by the wrong call (kb/lessons.md).

- Phase 1 runs in a background thread of this process (`prepare_start`, then `prepare_poll`),
  so the engine keeps stepping while the side locates, compiles and stages. The parses,
  compiles and the outside-changes compare run in a child process (kernel `ProcessAnalyzer`,
  started with the side; FASTRSI_KERNEL_ANALYZE=inline keeps them in this process), so they
  do not hold this process's GIL. A heap scan (nested targets only) still does.
- `prepare_poll` can be repeated: a finished rank keeps answering with its result until the
  transaction is committed, aborted or undone. The worker group polls every rank with one
  collective_rpc while any rank is pending, so a rank that finished early is asked again.
- A plan module runs only on the roles in its `"on"` list (default: every role), so one plan
  can swap a worker function and an engine-core function. A role the plan gives no module
  abstains: it votes yes, and its commit, abort and undo change nothing (the kernel would
  call its empty sub-plan an `empty_change` and veto the transaction).
- `plan["faults"] = {role or "role:rank": method or {"method", "delay_s", "fail"}}` makes that
  method return not-ok on that role (or on one rank of it), or delay its reply by `delay_s`
  first (`fail` false: delay only): a test hook for forced failures and slow ranks, named in
  the reply and the receipt.
- Phase-2 calls with a deadline (`{"txn", "deadline_at"}`: validate, commit, abort, undo) run
  on this side's lane thread (kernel `Lane`, calls in order) and the side answers by the
  deadline: `deadline_missed` if the call is still running. So the process keeps stepping
  while a commit hangs, and the RPC reply never comes late. The late call keeps the lane; a
  commit that starts after its deadline applies nothing (kernel LocalParticipant), and the
  calls queued behind it (the host's undo) run after it.
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
from kernel import swap as S

METHODS = ("prepare", "prepare_start", "prepare_poll", "validate", "commit", "abort", "undo",
           "status", "proof", "witness_check", "reset", "versions")


class Side:
    def __init__(self, role, scope=None, resets=None, rank=None):
        self.role, self.rank = role, rank
        analyzer = None
        if os.environ.get("FASTRSI_KERNEL_ANALYZE", "process") == "process":
            analyzer = S.ProcessAnalyzer().start()
        self.lp = C.LocalParticipant(role, scope=scope, resets=resets, analyzer=analyzer)
        self._jobs = {}        # txn -> {"thread", "result", "t0"}, until commit/abort/undo
        self._faults = {}      # txn -> {"method", "delay_s", "fail"} for this role / rank
        self._lane = None      # phase-2 calls with a deadline (kernel Lane)
        self._abstain = {}     # txn -> state, for plans that give this role no module
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

    def _prepare(self, plan, full):
        """Prepare this role's sub-plan; abstain when the plan changes only other roles (a plan
        that changes nothing anywhere still goes to the kernel: empty_change)."""
        if plan["modules"] or not full.get("modules"):
            return self.lp.prepare(plan)
        self._abstain[plan["id"]] = "prepared"
        return F.ok(staged={}, abstain=True, note=f"no module for role {self.role}")

    def _local(self, method, payload):
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
            return F.ok(counts={}, abstain=True)
        return F.ok(abstain=True)

    def _fault(self, method, txn):
        txn, _ = C.txn_ref(txn)
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
            return self._fault("prepare", plan["id"]) or self._prepare(plan, payload)
        if method == "prepare_start":
            plan = self._subplan(payload)
            self._watch_plan(plan)
            job = {"t0": time.perf_counter(), "result": None}

            def run():
                r = self._fault("prepare", plan["id"]) or self._prepare(plan, payload)
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
            return job["result"]           # kept: other ranks may still be pending
        if method == "reset":
            return self.lp.reset(payload)
        if method == "versions":
            return self.versions()
        txn, deadline_at = C.txn_ref(payload)
        if method in ("commit", "abort", "undo") and isinstance(txn, str):
            self._jobs.pop(txn, None)          # the vote is over: drop the prepare result
        if deadline_at is None or method not in C.PHASE2:
            return self._fault(method, payload) or self._local(method, payload)
        if self._lane is None:
            self._lane = C.Lane(f"{self.role}{'' if self.rank is None else self.rank}")
        done, r = self._lane.call(
            lambda: self._fault(method, payload) or self._local(method, payload),
            deadline_at)
        if done:
            return r
        who = self.role if self.rank is None else f"{self.role} rank {self.rank}"
        return F.not_ok(f"{method}: no answer by the deadline on {who}", [{**F.finding(
            "deadline_missed", f"{method} still running on {who} at the deadline (or queued "
            "behind a call that is)", f"{method} answers by the deadline"),
            "participant": self.role, "rank": self.rank}])
