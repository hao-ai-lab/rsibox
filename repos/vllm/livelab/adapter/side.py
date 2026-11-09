"""One side of a transaction inside a vLLM process (a worker rank or the engine core): the
`Endpoint` of worker_ext.py plus what a live engine needs from each side.

- **Background prepare.** `prepare_start` stages in a thread of this process and returns at
  once; `prepare_poll` answers `pending` until the result is in, then repeats it until the
  transaction is committed, aborted or undone (a rank that finished early is asked again while
  others are pending, and must not answer "no prepare started"). The host polls every rank with
  one collective_rpc, so the loop holds each poll for milliseconds, not for the whole stage.
  An abort (or undo) that arrives while this side is still staging cancels it: a prepare
  that has not started never stages, and one that is running drops what it staged (and its
  watches) as soon as it finishes, so nothing of an aborted transaction stays on any rank.
- **Per-rank deadline.** Phase-2 calls with a deadline (`{"txn", "deadline_at"}`: validate,
  commit, abort, undo) run on this side's own lane thread (kernel `Lane`), and the side
  answers by the deadline: a call still running is `deadline_missed` naming this rank. So
  every rank replies to every collective_rpc by the deadline, the reply says which rank is
  late, and the RPC itself never times out. The late call keeps the lane; a commit that starts
  after its deadline applies nothing (kernel), and the host's undo queues behind it.
- **Graph detection** (graphs.py). On a worker whose config runs CUDA graphs or torch.compile,
  the split decorator itself (`eager_break_during_capture`, in any module, and code nested in
  it) is a blocking `captured_target` at prepare: it runs only when ops are decorated (import)
  or graphs captured (vLLM #59941). A target in model code (`vllm.model_executor.models` /
  `.layers`) is checked at prepare:
  reached only through an op that splits the graph (Python on prefill, graphs on decode) is an
  advisory `captured_partly`; captured only, or undecided, is a blocking `captured_target`.
  Both route R1. A plan with `"rebuild": ["R1"]` asks for the recapture (recapture.py, run by
  the host in the commit's gap): its graph findings turn advisory and say so. Without it a
  blocking finding votes no, so the edit is reported and not applied. A `zero_witness` on
  model code or on the split decorator in graph mode is routed the same way at proof. Calls
  made while the stream captures a graph (or inside an R1) count apart from live calls.
- **R1** (`recapture`, `r1_extend`, `graphs`): the worker's side of recapture.py, on this
  rank's model runner (`scope["r1_engine"]` overrides it in tests).
- **Version log** (`tick`, `versions`). At every step the side records which code version each
  watched function runs (the targets of every plan it prepared), as `[step, {key: tag}]` at
  each change, and counts steps whose version changed between the step's two calls
  (execute_model and sample_tokens). Keys join only at a step start. Comparing the ranks' logs
  shows whether any step ran mixed versions.
- **Fault hook** (tests): `plan["faults"] = {role or "role:rank": method or {"method",
  "delay_s", "fail"}}` makes that method fail on that role (or rank), or delays it.
"""
import importlib
import os
import sys
import threading
import time

from kernel import coordinator as C
from kernel import findings as F
from kernel import locate as L

from . import graphs as G
from . import recapture as RC
from .worker_ext import Endpoint

CAPTURED_PREFIXES = ("vllm.model_executor.models.", "vllm.model_executor.layers.")
R1_NOT_HERE = ("add \"rebuild\": [\"R1\"] to the plan to recapture the graphs in the commit's "
               "gap, or restart with the edit, or run eager")
R1_ASKED = "the plan asks for R1: the host recaptures the graphs in the commit's gap"
R1 = {"action": "recapture the graphs with the commit (plan \"rebuild\": [\"R1\"]), or "
                "restart with the edit or run eager", "cost": "recipe R1 / a restart"}


def wants_r1(plan):
    return "R1" in ((plan or {}).get("rebuild") or ())
# globals torch.compile (Dynamo) adds to a module it traces: at_base lists them apart
DYNAMO_GLOBALS = ("__compiled_fn", "__resume_at", "__builtins_dict__", "__subclass_",
                  "__import_", "___")


def dynamo_name(name):
    return name.startswith(DYNAMO_GLOBALS)


def stream_capturing():
    """Is this thread's current CUDA stream recording a graph? (the worker witness's probe)"""
    torch = sys.modules.get("torch")
    return bool(torch is not None and torch.cuda.is_initialized()
                and torch.cuda.is_current_stream_capturing())


class Side(Endpoint):
    def __init__(self, role, scope=None, rank=None, **participant):
        worker = role == "worker"
        super().__init__(role, scope=scope, rank=rank, checks=[self._captured_check],
                         engine_names=dynamo_name,
                         capturing=stream_capturing if worker else None, **participant)
        self.captured_prefixes = CAPTURED_PREFIXES
        self._jobs = {}        # txn -> {"result", "t0", "lock", "cancelled"}, until the vote ends
        self._faults = {}      # txn -> {"method", "delay_s", "fail"} for this role / rank
        self._lane = None
        self.watch, self._watch_new = {}, {}   # key -> function; new keys join at a step start
        self._watch_lock = threading.Lock()
        self._watch_txn, self._watch_drop = {}, set()
        self._tags = {}        # id(code) -> (code, tag); the code ref keeps the id unique
        self.steps, self.vlog, self.mid_step_changes, self._cur = 0, [], [], None
        self._plan = threading.local()    # the plan this thread prepares (graph checks read it)
        self.r1 = RC.Recapture()

    # -- graph detection ----------------------------------------------------------------
    def _graph(self):
        w = self.scope.get("worker")
        return G.graph_mode(w) if w is not None else None

    def _captured_check(self, module, items):
        """Kernel check at prepare, for targets in model code while the worker runs graphs or
        compiled code (see the module doc)."""
        g = self._graph()
        if g is None:
            return []
        mode = f"cudagraph_mode {g['cudagraph_mode']}, compile mode {g['compile_mode']}"
        if wants_r1(getattr(self._plan, "plan", None)):
            return [{**f, "advisory": True, "observed": f["observed"].replace(
                R1_NOT_HERE, R1_ASKED).replace("so this stays blocking", "so it blocks "
                                               "without R1")}
                    for f in self._graph_findings(module, items, g, mode)]
        return self._graph_findings(module, items, g, mode)

    def _graph_findings(self, module, items, g, mode):
        out, model_items = [], []
        for k in items:                 # the split decorator: in any module
            detail = G.decoration_time(k.split(":")[1])
            if detail is None:
                model_items.append(k)
                continue
            out.append({**F.finding(
                "captured_target", f"the engine runs {mode}; {detail}; {R1_NOT_HERE}",
                "a target the engine runs in Python", subject=k),
                "route": "R1", "today": "T1"})
        items = model_items
        if not items or not module.startswith(self.captured_prefixes):
            return out
        try:
            act = self.lp._actual(module)
            mod = sys.modules.get(act) or importlib.import_module(act)
            src = self.lp._live_source(module)
            if src is None:
                with open(mod.__file__) as fh:
                    src = fh.read()
            roots, err = G.split_roots(mod, g), None
        except Exception as e:
            src, roots, err = None, [], f"{type(e).__name__}: {e}"
        for k in items:
            qual = k.split(":")[1]
            kind, detail = G.reach(src, roots, qual) if src is not None else \
                ("unknown", f"the module source could not be read ({err})")
            where = G.where(g, kind)
            if where["prefill"] and where["decode"]:
                continue
            if where["prefill"]:
                out.append({**F.finding(
                    "captured_partly", f"the engine runs {mode}; {detail}: the new code runs "
                    "in Python on prefill and mixed steps now, and on decode steps (full "
                    f"graphs) only after R1; {R1_NOT_HERE}",
                    "a target the engine runs in Python on every step", subject=k),
                    "route": "R1", "today": "T1", "advisory": True})
                continue
            why = (f"; could not decide statically, so this stays blocking: {detail}"
                   if kind == "unknown" else f"; {detail}")
            out.append({**F.finding(
                "captured_target", f"the engine runs {mode}, and {module} is model code{why}; "
                f"{R1_NOT_HERE}", "a target the engine runs in Python", subject=k),
                "route": "R1", "today": "T1"})
        return out

    def _route(self, r):
        """Proof time: a zero_witness on model code, or on the split decorator, while the
        engine runs graphs says why."""
        for f in (r.get("findings") or []) if isinstance(r, dict) else []:
            mod, _, rest = str(f.get("subject", "")).partition(":")
            what = "model code" if mod.startswith(self.captured_prefixes) else \
                "the split decorator" if G.decoration_time(rest.rpartition(":")[0]) else None
            if f.get("check") == "zero_witness" and what and self._graph() is not None:
                f.update(route="R1", today="T1")
                f["observed"] += f" ({what}; the engine runs captured graphs; {R1_NOT_HERE})"
                f["actions"] = [R1] + list(f.get("actions", []))
        return r

    # -- version log ----------------------------------------------------------------------
    def _tag(self, code):
        e = self._tags.get(id(code))
        if e is None:
            e = self._tags[id(code)] = (code, L.content_hash(code)[:10])
        return e[1]

    def _watch_targets(self, targets, txn):
        for m, ts in (targets or {}).items():
            act = self.lp._actual(m)
            mod = sys.modules.get(act) or importlib.import_module(act)
            for qual, _line in ts if isinstance(ts, list) else ():
                obj = mod
                try:
                    for part in qual.split("."):
                        obj = getattr(obj, part)
                except AttributeError:
                    continue
                k = f"{m}:{qual}"
                if hasattr(obj, "__code__") and k not in self.watch:
                    with self._watch_lock:
                        if k not in self._watch_new:
                            self._watch_new[k] = obj
                            self._watch_txn.setdefault(txn, set()).add(k)

    def _drop_watch(self, txn):
        """The keys a vetoed transaction added stop being watched (at the next step start)."""
        keys = self._watch_txn.pop(txn, set())
        with self._watch_lock:
            for k in keys:
                if self._watch_new.pop(k, None) is None:
                    self._watch_drop.add(k)

    def tick(self, first=True):
        """At a step start (first=True) and later in the same step (first=False)."""
        if first and (self._watch_new or self._watch_drop):
            with self._watch_lock:
                new, self._watch_new = self._watch_new, {}
                drop, self._watch_drop = self._watch_drop, set()
            for k, f in new.items():
                self.watch.setdefault(k, f)
            for k in drop:
                self.watch.pop(k, None)
        if first:
            self.steps += 1
        if not self.watch:
            return
        cur = {k: self._tag(f.__code__) for k, f in list(self.watch.items())}
        if first and cur != self._cur:
            self.vlog.append([self.steps, cur])
            self._cur = cur
        elif not first and cur != self._cur:
            self.mid_step_changes.append([self.steps, cur])

    def versions(self, _payload=None):
        return F.ok(steps=self.steps, vlog=self.vlog, mid_step_changes=self.mid_step_changes,
                    watch=sorted(self.watch), watch_pending=sorted(self._watch_new))

    # -- protocol -------------------------------------------------------------------------
    def _fault(self, method, txn):
        txn, _ = C.txn_ref(txn)
        f = self._faults.get(txn) if isinstance(txn, str) else None   # fingerprint: a list
        if not f or f["method"] != method:
            return None
        if f["delay_s"]:
            time.sleep(f["delay_s"])
        if not f["fail"]:
            return None
        return F.not_ok(f"injected fault: {self.who()} {method}", [{**F.finding(
            "participant_failed", f"injected fault at {method} on {self.who()} (test hook)",
            f"{method} ok"), "participant": self.role, "rank": self.rank}])

    def _prepare_once(self, plan):
        txn = plan["id"]
        fault = (plan.get("faults") or {}).get(f"{self.role}:{self.rank}") or \
            (plan.get("faults") or {}).get(self.role)
        if fault:
            fault = {"method": fault} if isinstance(fault, str) else fault
            self._faults[txn] = {"delay_s": 0, "fail": True, **fault}
        self._plan.plan = plan
        try:
            r = self._fault("prepare", txn) or self.prepare(plan)
        finally:
            self._plan.plan = None
        if isinstance(r, dict) and r.get("ok"):
            self._watch_targets(r.get("targets"), txn)
        else:
            self._drop_watch(txn)
        return r

    # -- R1 -------------------------------------------------------------------------------
    def _engine(self):
        if "r1_engine" in self.scope:
            return self.scope["r1_engine"]
        w = self.scope.get("worker")
        return RC.vllm_engine(w) if w is not None and self._graph() is not None else None

    def _r1(self, method, p):
        if self.role != "worker":
            return F.ok(abstain=True, note="R1 runs on worker ranks")
        try:
            eng = self._engine()
        except Exception as e:
            return F.not_ok(f"R1: {type(e).__name__}: {e}", [F.finding(
                "rebuild_failed", f"{type(e).__name__}: {e}", "R1 can reach this rank's graphs")])
        if method == "graphs":
            return F.ok(**RC.graphs(eng))
        if method == "r1_extend":
            return self.r1.extend(eng, p["txn"], self.lp.journal, self.lp.witness, p.get("sizes"))
        if self.lp._state.get(p["txn"]) != "committed":
            return F.not_ok(f"{p['txn']} is not committed on {self.who()}")
        return self.r1.run(eng, p["txn"], self.lp.journal, self.lp.witness,
                           p.get("modes"), p.get("sizes"))

    def _dispatch(self, method, payload):
        if method in ("recapture", "r1_extend", "graphs"):
            return self._r1(method, payload or {})
        if method == "versions":
            return self.versions()
        if method == "prepare":
            return self._prepare_once(payload)
        if method == "prepare_start":
            txn = payload["id"]
            job = self._jobs[txn] = {"t0": time.perf_counter(), "result": None,
                                     "lock": threading.Lock(), "cancelled": False}

            def run():
                with job["lock"]:
                    cancelled = job["cancelled"]
                r = F.not_ok("prepare cancelled before it started") if cancelled else \
                    self._prepare_once(payload)
                r["prepare_s"] = round(time.perf_counter() - job["t0"], 6)
                with job["lock"]:
                    job["result"] = r
                    cancelled = job["cancelled"]
                if cancelled:        # aborted while staging: drop what this prepare staged
                    Endpoint._dispatch(self, "abort", txn)
                    self._drop_watch(txn)
            threading.Thread(target=run, daemon=True,
                             name=f"fastrsi-prepare-{payload['id']}").start()
            return F.ok(started=True)
        if method == "prepare_poll":
            job = self._jobs.get(payload)
            if job is None:
                return F.not_ok(f"no prepare started for {payload}")
            return F.ok(pending=True) if job["result"] is None else dict(job["result"])
        txn, deadline_at = C.txn_ref(payload)
        if method in ("commit", "abort", "undo") and isinstance(txn, str):
            job = self._jobs.pop(txn, None)    # the vote is over
            if job is not None and method != "commit":
                with job["lock"]:              # still staging: it cleans up when done
                    job["cancelled"] = job["result"] is None
            if method == "abort":
                self._drop_watch(txn)          # vetoed here or on another rank
            else:
                self._watch_txn.pop(txn, None)  # committed: its keys stay watched
        if method == "witness_check":
            return self._route(super()._dispatch(method, payload))
        if deadline_at is None or method not in C.PHASE2:
            return self._fault(method, payload) or super()._dispatch(method, payload)
        if self._lane is None:
            self._lane = C.Lane(self.who().replace(" rank ", ""))
        def call():
            try:
                return self._fault(method, payload) or super(Side, self)._dispatch(method,
                                                                                   payload)
            except Exception as e:                 # named here, not by the lane
                return F.not_ok(f"{method} on {self.who()}: {type(e).__name__}: {e}")
        done, r = self._lane.call(call, deadline_at)
        if done:
            return r
        return F.not_ok(f"{method}: no answer by the deadline on {self.who()}", [{**F.finding(
            "deadline_missed", f"{method} still running on {self.who()} at the deadline (or "
            "queued behind a call that is)", f"{method} answers by the deadline"),
            "participant": self.role, "rank": self.rank}])
