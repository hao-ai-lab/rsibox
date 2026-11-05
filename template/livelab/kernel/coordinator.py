"""Coordinator: two-phase transactions over participants (worker ranks, engine core, API server).

    prepare (every participant stages beside the old state, runs its checks, votes)
      -> unanimous yes: commit on each
      -> otherwise:     abort on all (drop staged state; nothing was visible)
    a commit that fails or crashes -> undo on the participants that already committed

`StepGapCoordinator` runs all of that inside ONE gap between two steps of the loop owner, so
participants that step together never run a step on different versions. `Coordinator` calls
participants one after another, each switching at its own quiesce point: between two commits
a step can see mixed versions (the toy test shows it). Receipts are linked records sharing the
transaction id: `run` appends `txn`, `prove` appends `proof` (witnesses after the workload,
edit_to_finish_s), `undo` appends `undo` (code undo, then the plan's `undo_resets`).

Participant protocol (every call returns an ok-dict and never raises; a crash or a reply that
is not an ok-dict counts as not-ok):

    prepare(plan) -> {"ok", "staged", "findings"}     ok is the vote
    commit(txn)   -> {"ok", ...}                     at the participant's quiesce point
    abort(txn)    -> {"ok"}                          drop staged state
    undo(txn)     -> {"ok", ...}                     journal.undo_to(the txn's mark)
    proof(txn)    -> {"ok", "hashes", "witness"}     code hashes right after commit
    witness_check(txn) -> {"ok", "counts", "findings"}  after the workload
    reset(kind)   -> {"ok", ...}                     data reset hook: "R2", "R3", ...

A plan is plain data, so it can cross a process boundary:

    {"id": str (optional),
     "modules": {module: {"source": patched text, "base_hash": text_hash of the base,
                          "targets": [[qualname, first line in the base]]}},
     "repairs": [{"cell": python source, "declares": [names in the participant's scope]}],
     "accept": [check names] or "all",
     "undo_resets": ["R2" | "R3" | ...],   resets the undo runs after the code undo
     "recipes": [...], "predicted_resources": [...], "certainty": str}

`LocalParticipant` runs the kernel in this process. `CallParticipant` is the shape the engine
side implements: one `call(method, payload) -> dict` per participant (for vLLM: collective_rpc
to a worker rank, or the engine-core input-queue channel), which on the far side dispatches
to a LocalParticipant.
"""
import json
import time
import uuid

from . import findings as F
from . import journal as J
from . import locate as L
from . import receipt as R
from . import swap as S
from . import witness as W


class Participant:
    name = "participant"

    def prepare(self, plan):
        raise NotImplementedError

    def commit(self, txn):
        raise NotImplementedError

    def abort(self, txn):
        raise NotImplementedError

    def undo(self, txn):
        raise NotImplementedError

    def proof(self, txn):
        return F.ok(hashes={}, witness={})

    def witness_check(self, txn):
        return F.ok(counts={})

    def reset(self, kind):
        return F.not_ok(f"no {kind} reset", [F.finding(
            "reset_failed", f"no {kind} hook", f"a {kind} hook", subject=kind)])


class CallParticipant(Participant):
    """A participant in another process, reached through `call(method, payload) -> dict`."""

    def __init__(self, name, call):
        self.name, self._call = name, call

    def _do(self, method, payload):
        try:
            r = self._call(method, payload)
        except Exception as e:
            return F.not_ok(f"{method}: {type(e).__name__}: {e}")
        return r if isinstance(r, dict) and "ok" in r else F.not_ok(f"{method}: bad reply {r!r}")

    def prepare(self, plan):
        return self._do("prepare", plan)

    def commit(self, txn):
        return self._do("commit", txn)

    def abort(self, txn):
        return self._do("abort", txn)

    def undo(self, txn):
        return self._do("undo", txn)

    def proof(self, txn):
        return self._do("proof", txn)

    def witness_check(self, txn):
        return self._do("witness_check", txn)

    def reset(self, kind):
        return self._do("reset", kind)


class LocalParticipant(Participant):
    """The kernel in this process. `alias`: {plan module prefix: this process's prefix} (a
    participant that loaded the package under another name). `quiesce(fn)`: runs fn at this
    participant's step boundary and returns its result (default: now, in the caller's thread).
    `scope`: names a repair cell can use. `resets`: {"R3": callable, ...}, data resets an
    undo can ask for (each returns a JSON-able summary or None)."""

    def __init__(self, name, alias=None, quiesce=None, scope=None, resets=None):
        self.name = name
        self.resets = dict(resets or {})
        self.alias = dict(alias or {})
        self.quiesce = quiesce or (lambda fn: fn())
        self.scope = dict(scope or {})
        self.journal = J.Journal(name)
        self.witness = W.Witness()
        self._staged, self._marks, self._done = {}, {}, {}

    def _actual(self, module):
        for k, v in self.alias.items():
            if module == k or module.startswith(k + "."):
                return v + module[len(k):]
        return module

    def prepare(self, plan):
        txn, accept = plan["id"], plan.get("accept", ())
        fs, staged, items, ok = [], [], {}, True
        for m, spec in plan.get("modules", {}).items():
            r = S.stage(self._actual(m), spec["source"], spec["targets"],
                        spec.get("base_hash"), key_module=m)
            fs += r["findings"]
            ok = ok and r["ok"]
            if r["staged"] is not None:
                staged.append(r["staged"])
            items.update(r["items"])
        cells = []
        for rep in plan.get("repairs", []):
            missing = [n for n in rep["declares"] if n not in self.scope]
            try:
                code = compile(rep["cell"], f"<repair {txn}>", "exec")
            except SyntaxError as e:
                missing.append(f"syntax: {e}")
                code = None
            if missing:
                fs.append(F.finding("patch_invalid", f"repair cell: {missing}",
                                    "a cell that compiles and declares names in scope"))
                ok = False
            cells.append((code, list(rep["declares"])))
        d = F.decide(fs, accept)
        out_fs = [{**f, "participant": self.name} for f in d["findings"]]
        if not (ok and d["proceed"]):
            return F.not_ok(f"vote no: {d['blocking'] or 'stage failed'}", out_fs, staged=items)
        self._staged[txn] = (staged, cells, accept)
        return F.ok(staged=items, findings=out_fs)

    def commit(self, txn):
        if txn not in self._staged:
            return F.not_ok(f"nothing staged for {txn}")
        staged, cells, accept = self._staged.pop(txn)
        gone = []                     # set if the quiesce call gave up: the job must not run

        def do():
            if gone:
                return F.not_ok("commit cancelled before it ran")
            mark = self.journal.mark(txn)
            for st in staged:
                r = S.apply(st, self.journal, self.witness)
                if not r["ok"]:
                    self.journal.undo_to(mark)
                    return r
            fs = []
            for code, declares in cells:
                decl = {n: self.scope[n] for n in declares}
                audit = {n: o for n, o in self.scope.items() if n not in decl}
                try:
                    with self.journal.repair(decl, audit) as rep:
                        exec(code, dict(self.scope))
                except Exception as e:
                    u = self.journal.undo_to(mark)
                    return F.not_ok("repair failed", [F.finding(
                        "repair_failed", f"{type(e).__name__}: {e}", "the cell runs")]
                        + u["findings"])
                fs += rep.findings
            d = F.decide(fs, accept)
            fs = [{**f, "participant": self.name} for f in d["findings"]]
            if not d["proceed"]:
                u = self.journal.undo_to(mark)
                return F.not_ok(f"commit refused: {d['blocking']}", fs + u["findings"])
            self._marks[txn] = mark
            self._done[txn] = staged
            return F.ok(changes=len(self.journal.entries) - mark, findings=fs,
                        journal=self.journal.describe(mark))

        try:
            return self.quiesce(do)
        except Exception as e:
            gone.append(True)
            return F.not_ok(f"commit: {type(e).__name__}: {e}")

    def abort(self, txn):
        self._staged.pop(txn, None)
        return F.ok()

    def undo(self, txn):
        if txn not in self._marks:
            return F.ok(undone=0, note="nothing committed")
        mark = self._marks[txn]
        try:
            r = self.quiesce(lambda: self.journal.undo_to(mark))
        except Exception as e:
            return F.not_ok(f"undo: {type(e).__name__}: {e}")
        if r["ok"]:
            del self._marks[txn]
            self._done.pop(txn, None)
        return r

    def proof(self, txn):
        hashes, keys = {}, []
        for st in self._done.get(txn, []):
            keys += st.keys
            for k, fns in st.functions.items():
                hashes[k] = sorted({L.content_hash(f.__code__) for f in fns})
        return F.ok(hashes=hashes, witness={k: self.witness.counts.get(k, 0) for k in keys})

    def reset(self, kind):
        hook = self.resets.get(kind)
        if hook is None:
            return super().reset(kind)
        try:
            return F.ok(kind=kind, result=self.quiesce(hook))
        except Exception as e:
            return F.not_ok(f"reset {kind}", [F.finding(
                "reset_failed", f"{type(e).__name__}: {e}", f"the {kind} hook runs",
                subject=kind)])

    def witness_check(self, txn):
        """After the workload ran: did every swapped target run? Findings, never a pass."""
        keys, old = [], {}
        for st in self._done.get(txn, []):
            keys += [k for k in st.keys if k in st.old_codes]
            old.update(st.old_codes)
        return self.witness.check(keys, old)


def _call(p, method, arg):
    try:
        r = getattr(p, method)(arg)
    except Exception as e:
        return F.not_ok(f"{method} crashed: {type(e).__name__}: {e}", [
            {**F.finding("participant_failed", f"{type(e).__name__}: {e}",
                         f"{method} returns an ok-dict"), "participant": p.name}])
    if not isinstance(r, dict) or "ok" not in r:
        return F.not_ok(f"{method}: bad reply {r!r}")
    return r


class Coordinator:
    """Two-phase transactions, one participant call after another. Each participant switches
    at its own quiesce point, so between two commits a step can run with participants on
    different versions. Use it only when participants do not step together (or for one);
    for ranks that step together use StepGapCoordinator."""

    def __init__(self, participants, writer=None, engine=None, interface_version="0",
                 clock=time.time):
        self.participants = list(participants)
        self.writer = writer
        self.engine = engine or {"name": "unknown", "version": None, "commit": None}
        self.interface_version = interface_version
        self.clock = clock
        self.txns = {}                     # txn id -> {"plan", "edit"} for prove and undo

    def _in_gap(self, fn):
        """Run fn where the transaction must happen. Here: right now, in this thread."""
        return fn()

    def _guarded(self, fn):
        """Run fn through _in_gap. If the gap call gives up (timeout, loop gone), a job still
        queued must not run later: it is cancelled. Returns (result, error)."""
        gone = []

        def job():
            return None if gone else fn()
        try:
            return self._in_gap(job), None
        except Exception as e:
            gone.append(True)
            return None, f"{type(e).__name__}: {e}"

    def _write(self, rec):
        return self.writer.append(rec) if self.writer else {"ok": False, "error": "no writer"}

    # -- the transaction -----------------------------------------------------------------
    def _phases(self, plan, tl):
        """prepare -> vote -> commit, or abort; undo the committed ones if a commit fails."""
        now, txn = self.clock, plan["id"]
        res = {p.name: {} for p in self.participants}
        tl["prepare_start"] = now()
        for p in self.participants:
            res[p.name]["prepare"] = _call(p, "prepare", plan)
        tl["vote"] = now()
        rollback = {"status": "not_needed", "participants": {}}
        out = {"res": res, "rollback": rollback, "extra": [], "hashes": {}}
        if not all(res[p.name]["prepare"]["ok"] for p in self.participants):
            for p in self.participants:
                res[p.name]["abort"] = _call(p, "abort", txn)
            out["outcome"] = "aborted"
            return out
        committed, failed = [], None
        for p in self.participants:
            r = res[p.name]["commit"] = _call(p, "commit", txn)
            if not r["ok"]:
                failed = p
                break
            committed.append(p)
        if failed is None:
            tl["committed"] = now()
            out["outcome"] = "committed"
            for p in self.participants:
                out["hashes"][p.name] = _call(p, "proof", txn).get("hashes")
            return out
        out["outcome"] = "rolled_back"
        out["extra"].append({**F.finding("participant_failed", res[failed.name]["commit"]
                                         .get("error"), "commit ok on every participant"),
                             "participant": failed.name})
        for p in reversed(committed):
            rollback["participants"][p.name] = res[p.name]["undo"] = _call(p, "undo", txn)
        for p in self.participants:
            if p not in committed:
                res[p.name]["abort"] = _call(p, "abort", txn)
        rollback["status"] = ("undone" if all(u["ok"] for u in rollback["participants"].values())
                              else "partial")
        tl["rolled_back"] = now()
        return out

    def run(self, plan, edit_time=None, baseline_restart=None):
        """One transaction; appends a `txn` receipt. edit_time: when the edit was made (unix
        seconds; default now). baseline_restart: {"seconds", "source"} of the warm restart
        this replaces. Returns {"ok", "outcome", "txn", "receipt", "written"}; never raises."""
        now = self.clock
        tl = {"edit": edit_time if edit_time is not None else now()}
        txn = plan.get("id") or f"txn-{uuid.uuid4().hex[:8]}"
        plan = {**plan, "id": txn}
        tl["queued"] = now()
        out, err = self._guarded(lambda: self._phases(plan, tl))
        if out is None:
            out = {"res": {}, "rollback": {"status": "not_needed", "participants": {}},
                   "hashes": {}, "outcome": "aborted",
                   "extra": [F.finding("participant_failed", f"the transaction never ran: {err}",
                                       "a step gap within the timeout")]}
        tl["txn_end"] = now()
        if out["outcome"] == "committed":
            self.txns[txn] = {"plan": plan, "edit": tl["edit"]}
        fs = [f for r in out["res"].values() for x in r.values()
              for f in x.get("findings", [])] + out["extra"]
        modules = plan.get("modules", {})
        hs = [json.dumps(h, sort_keys=True) for h in out["hashes"].values()]
        rec = {
            "schema": R.SCHEMA, "kind": "txn", "id": txn,
            "interface_version": self.interface_version, "engine": self.engine,
            "base_hash": L.text_hash(json.dumps(
                {m: s.get("base_hash") for m, s in modules.items()}, sort_keys=True)),
            "patch_hash": L.text_hash(json.dumps(
                {"modules": {m: L.text_hash(str(s.get("source", "")))
                             for m, s in modules.items()},
                 "repairs": plan.get("repairs", [])}, sort_keys=True)),
            "plan": {"targets": {m: s.get("targets", []) for m, s in modules.items()},
                     "predicted_resources": plan.get("predicted_resources", []),
                     "recipes": plan.get("recipes", []), "certainty": plan.get("certainty"),
                     "accept": plan.get("accept", []),
                     "undo_resets": plan.get("undo_resets", [])},
            "actual_recipes": ((["R0 swap"] if modules else [])
                               + (["R10 repair"] if plan.get("repairs") else []))
            if out["outcome"] == "committed" else [],
            "participants": out["res"],
            "commit_hashes": {"participants": out["hashes"],
                              "agree": (len(set(hs)) == 1) if hs else None},
            "findings": fs,
            "decisions": {"accepted": sorted({f["check"] for f in fs if f.get("accepted")}),
                          "blocking": sorted({f["check"] for f in fs
                                              if not f.get("accepted")})},
            "outcome": out["outcome"], "timeline": tl,
            "edit_to_run_s": (round(tl["committed"] - tl["edit"], 6)
                              if "committed" in tl else None),
            "phases_s": round(tl["txn_end"] - tl["queued"], 6),
            "baseline_restart_s": baseline_restart,
            "rollback": out["rollback"],
        }
        return {"ok": out["outcome"] == "committed", "outcome": out["outcome"], "txn": txn,
                "receipt": rec, "written": self._write(rec)}

    def prove(self, txn, golden=None):
        """After the workload ran: collect witness counts and findings from every participant;
        append a `proof` receipt. golden: the caller's golden-check results, recorded as is."""
        now = self.clock
        res = {p.name: _call(p, "witness_check", txn) for p in self.participants}
        fs = [{**f, "participant": n} for n, r in res.items() for f in r.get("findings", [])]
        edit = self.txns.get(txn, {}).get("edit")
        t = now()
        rec = {"schema": R.SCHEMA, "kind": "proof", "id": txn,
               "witness": {n: r.get("counts", {}) for n, r in res.items()},
               "findings": fs, "golden": golden,
               "timeline": {"edit": edit, "proved": t},
               "edit_to_finish_s": round(t - edit, 6) if edit is not None else None}
        return {"ok": all(r["ok"] for r in res.values()) and not fs, "receipt": rec,
                "written": self._write(rec)}

    def _undo_phases(self, txn, resets, tl):
        res = {p.name: {} for p in self.participants}
        for p in reversed(self.participants):
            res[p.name]["undo"] = _call(p, "undo", txn)
        tl["code_undone"] = self.clock()
        for kind in resets:
            for p in self.participants:
                res[p.name].setdefault("reset", {})[kind] = _call(p, "reset", kind)
        return res

    def undo(self, txn):
        """Undo a committed transaction on every participant (newest first), then run the
        plan's `undo_resets` (R2, R3, ...) on each: code undo cannot un-write data the new code
        produced. Appends an `undo` receipt."""
        now = self.clock
        resets = self.txns.get(txn, {}).get("plan", {}).get("undo_resets", [])
        tl = {"queued": now()}
        res, err = self._guarded(lambda: self._undo_phases(txn, resets, tl))
        tl["undo_end"] = now()
        res = res or {}
        calls = [x for r in res.values() for x in [r.get("undo")] + list(r.get("reset", {})
                                                                              .values()) if x]
        fs = [{**f, "participant": n} for n, r in res.items()
              for x in [r.get("undo")] + list(r.get("reset", {}).values()) if x
              for f in x.get("findings", [])]
        ok = err is None and all(x["ok"] for x in calls)
        rec = {"schema": R.SCHEMA, "kind": "undo", "id": txn, "participants": res,
               "resets": list(resets), "status": "undone" if ok else "partial",
               "findings": fs + ([F.finding("undo_incomplete", f"undo never ran: {err}",
                                            "a step gap within the timeout")] if err else []),
               "timeline": tl, "undo_s": round(tl["undo_end"] - tl["queued"], 6)}
        if ok:
            self.txns.pop(txn, None)
        return {"ok": ok, "participants": res, "receipt": rec, "written": self._write(rec)}


class StepGapCoordinator(Coordinator):
    """The coordinator for participants that step together (worker ranks + engine core): the
    loop owner runs the whole transaction in ONE gap between two steps: prepare on all,
    vote, commit on all, and undo after a failed commit. So no step ever sees participants on
    different versions. `gap(fn)` queues fn to the loop owner and returns its result (for
    vLLM: the engine-core busy loop; the participants are then called directly from it, e.g.
    by collective_rpc, which completes on every worker before the next step).
    Participants must not quiesce on their own: build them with the default `quiesce`.
    The gap stalls the engine for the whole transaction: `phases_s` in the receipt."""

    def __init__(self, participants, gap, **kw):
        super().__init__(participants, **kw)
        self.gap = gap

    def _in_gap(self, fn):
        return self.gap(fn)
