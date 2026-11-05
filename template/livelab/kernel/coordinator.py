"""Coordinator: two-phase transactions over participants (worker ranks, engine core, API server).

    phase 1, while steps go on: prepare on every participant (stage beside the old state,
      run checks, vote); a no vote aborts here
    phase 2, in the gap: validate on every participant (the staged plan is still valid),
      then commit on all, or abort on all; a commit that fails undoes the committed ones

`StepGapCoordinator` runs phase 2 inside ONE gap between two steps of the loop owner, so
participants that step together never run a step on different versions, and the loop stalls
only for phase 2. `Coordinator` runs phase 2 right away, each participant switching at its
own quiesce point: between two commits a step can see mixed versions (the toy test shows it).
The host answers `status(txn)`: prepared, committed, aborted, undone or in-doubt. Receipts are
linked records sharing the transaction id: `run` appends `txn`, `prove` appends `proof`
(witnesses after the workload, edit_to_finish_s), `undo` appends `undo` (code undo, then the
plan's `undo_resets`).

Participant protocol (every call returns an ok-dict and never raises; a crash or a reply that
is not an ok-dict counts as not-ok):

    prepare(plan) -> {"ok", "staged", "findings"}     phase 1; ok is the vote
    validate(txn) -> {"ok", "findings"}              phase 2: staged identities still hold
    status(txn)   -> {"ok", "state"}                 prepared, committed, aborted, undone
    commit(txn)  -> {"ok", ...}                     at the participant's quiesce point
    abort(txn)    -> {"ok"}                          drop staged state
    undo(txn)     -> {"ok", ...}                     journal.undo_to(the txn's mark)
    proof(txn)    -> {"ok", "hashes", "witness"}     code hashes right after commit
    witness_check(txn) -> {"ok", "counts", "findings"}  after the workload
    reset(kind)   -> {"ok", ...}                     data reset hook: "R2", "R3", ...
    fingerprint(modules) -> {"ok", "code"}           {key: [content hash, code object id]}

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
import sys
import threading
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

    def validate(self, txn):
        return F.ok()

    def status(self, txn):
        return F.ok(txn=txn, state="unknown")

    def fingerprint(self, modules):
        return F.not_ok("no fingerprint")


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

    def validate(self, txn):
        return self._do("validate", txn)

    def status(self, txn):
        return self._do("status", txn)

    def fingerprint(self, modules):
        return self._do("fingerprint", modules)


class LocalParticipant(Participant):
    """The kernel in this process. `alias`: {plan module prefix: this process's prefix} (a
    participant that loaded the package under another name). `quiesce(fn)`: runs fn at this
    participant's step boundary and returns its result (default: now, in the caller's thread).
    `scope`: names a repair cell can use. `resets`: {"R3": callable, ...}, data resets an
    undo can ask for (each returns a JSON-able summary or None).

    Chained edits: the participant remembers the source text each committed transaction
    installed and the code objects it swapped in. A later plan for the same module is staged
    against that text (not the file), and the participant's own code objects are not
    `base_mismatch`. Undo of a transaction also forgets every later one (the journal undoes
    them too). A plan that would change nothing is an `empty_change` finding."""

    def __init__(self, name, alias=None, quiesce=None, scope=None, resets=None):
        self.name = name
        self.resets = dict(resets or {})
        self.alias = dict(alias or {})
        self.quiesce = quiesce or (lambda fn: fn())
        self.scope = dict(scope or {})
        self.journal = J.Journal(name)
        self.witness = W.Witness()
        self._staged, self._marks, self._done = {}, {}, {}
        self._state = {}              # txn -> prepared | committed | aborted | undone
        self._sources = {}            # committed txn -> {plan module: source it installed}
        self._codes = {}              # committed txn -> [code objects it swapped in]
        self._pending = {}            # prepared txn -> {plan module: source}

    def _live_source(self, module):
        """The text the live module runs: the latest committed transaction's, else None."""
        for txn in reversed(list(self._sources)):
            if module in self._sources[txn]:
                return self._sources[txn][module]
        return None

    def fingerprint(self, modules):
        """{key: [content hash, id of the code object]} for every function the modules' own
        namespaces reach (no heap scan). Equal fingerprints mean the same code objects run."""
        out = {}
        try:
            for m in modules:
                mod = sys.modules.get(self._actual(m))
                if mod is None:
                    return F.not_ok(f"module {m} not loaded")
                for fn in L.module_functions(mod):
                    c = fn.__code__
                    out[L.key_of(m, c.co_qualname, c.co_firstlineno)] = [L.content_hash(c),
                                                                         id(c)]
        except Exception as e:
            return F.not_ok(f"fingerprint: {type(e).__name__}: {e}")
        return F.ok(code=out)

    def status(self, txn):
        return F.ok(txn=txn, state=self._state.get(txn, "unknown"))

    def validate(self, txn):
        """Phase 2, in the gap, before the vote: is the staged plan still valid?"""
        if txn not in self._staged:
            return F.not_ok(f"nothing staged for {txn}")
        staged, cells, _ = self._staged[txn]
        fs = []
        for st in staged:
            fs += S.validate(st)["findings"]
        fs = [{**f, "participant": self.name} for f in fs]
        return F.ok(checked=sum(len(st.expect) for st in staged)) if not fs else \
            F.not_ok("changed since prepare", fs)

    def _actual(self, module):
        for k, v in self.alias.items():
            if module == k or module.startswith(k + "."):
                return v + module[len(k):]
        return module

    def prepare(self, plan):
        txn, accept = plan["id"], plan.get("accept", ())
        fs, staged, items, ok, timings = [], [], {}, True, {}
        own = [c for cs in self._codes.values() for c in cs]
        for m, spec in plan.get("modules", {}).items():
            r = S.stage(self._actual(m), spec["source"], spec["targets"],
                        spec.get("base_hash"), key_module=m, scan=spec.get("scan", "auto"),
                        base_source=self._live_source(m), own_codes=own)
            fs += r["findings"]
            ok = ok and r["ok"]
            if r["staged"] is not None:
                staged.append(r["staged"])
            items.update(r["items"])
            timings[m] = r.get("timings", {})
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
        if ok and not cells and all(f["overridable"] for f in fs) and not any(
                it.get("changed") or it.get("added") for it in items.values()) \
                and not any(st.removes for st in staged):
            fs.append(F.finding("empty_change", f"{len(items)} target(s), none changed; no "
                                "definition added or deleted; no repair cell",
                                "at least one change to install"))
        d = F.decide(fs, accept)
        out_fs = [{**f, "participant": self.name} for f in d["findings"]]
        if not (ok and d["proceed"]):
            self._state[txn] = "aborted"
            return F.not_ok(f"vote no: {d['blocking'] or 'stage failed'}", out_fs, staged=items)
        self._staged[txn] = (staged, cells, accept)
        self._pending[txn] = {m: spec["source"] for m, spec in plan.get("modules", {}).items()}
        self._state[txn] = "prepared"
        return F.ok(staged=items, findings=out_fs, timings=timings)

    def commit(self, txn):
        if txn not in self._staged:
            return F.not_ok(f"nothing staged for {txn}")
        staged, cells, accept = self._staged.pop(txn)
        self._state[txn] = "aborted"  # until the commit below succeeds
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
            self._sources[txn] = self._pending.pop(txn, {})
            self._codes[txn] = [c for st in staged for _, c, _, _ in st.swaps]
            self._state[txn] = "committed"
            return F.ok(changes=len(self.journal.entries) - mark, findings=fs,
                        journal=self.journal.describe(mark))

        try:
            return self.quiesce(do)
        except Exception as e:
            gone.append(True)
            return F.not_ok(f"commit: {type(e).__name__}: {e}")

    def abort(self, txn):
        self._pending.pop(txn, None)
        if self._staged.pop(txn, None) is not None or self._state.get(txn) == "prepared":
            self._state[txn] = "aborted"
        return F.ok()

    def undo(self, txn):
        if txn not in self._marks:
            return F.ok(undone=0, note="nothing committed")
        mark = self._marks[txn]
        try:
            r = self.quiesce(lambda: self.journal.undo_to(mark))
        except Exception as e:
            return F.not_ok(f"undo: {type(e).__name__}: {e}")
        if r["ok"]:                    # the journal undid every later transaction too
            for t in [t for t, m in self._marks.items() if m >= mark]:
                del self._marks[t]
                for d in (self._done, self._sources, self._codes):
                    d.pop(t, None)
                self._state[t] = "undone"
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


def _hashes_agree(hashes):
    """Do participants that swapped the same target run the same code? Compares each witness
    key across the participants that report it (participants may swap different modules, e.g.
    workers the sampler and the engine core the scheduler). None when no key is shared."""
    by_key = {}
    for h in hashes.values():
        for k, v in (h or {}).items():
            by_key.setdefault(k, []).append(json.dumps(v, sort_keys=True))
    shared = [vs for vs in by_key.values() if len(vs) > 1]
    return all(len(set(vs)) == 1 for vs in shared) if shared else None


class Coordinator:
    """The transaction host. Two phases:

    1. `prepare` on every participant, in the caller's thread while steps go on: locate,
       compile, stage, check. Nothing is visible. A no vote aborts here.
    2. Inside `_in_gap`: `validate` on every participant (is the staged plan still valid?),
       then commit on all, or abort on all; a commit that fails undoes the committed ones.

    The plain Coordinator runs phase 2 right away and each participant switches at its own
    quiesce point: between two commits a step can see mixed versions. StepGapCoordinator runs
    phase 2 in one gap between two steps of the loop owner.

    The host records every transaction's state; `status(txn)` answers prepared, committed,
    aborted, undone, or in-doubt (phase 2 started but has not finished). A caller whose gap
    call timed out resolves the outcome with `status`, never by assuming failure; a phase 2
    that had not started when the caller gave up is cancelled (aborted)."""

    def __init__(self, participants, writer=None, engine=None, interface_version="0",
                 clock=time.time):
        self.participants = list(participants)
        self.writer = writer
        self.engine = engine or {"name": "unknown", "version": None, "commit": None}
        self.interface_version = interface_version
        self.clock = clock
        self.txns = {}                     # txn id -> host record (see _record)
        self._lock = threading.Lock()

    def _in_gap(self, fn):
        """Run fn where phase 2 must happen. Here: right now, in this thread."""
        return fn()

    def _write(self, rec):
        return self.writer.append(rec) if self.writer else {"ok": False, "error": "no writer"}

    # -- host state ----------------------------------------------------------------------
    def status(self, txn):
        """{"ok", "txn", "state", "outcome", "written"}. Once phase 2 has finished, the first
        status call after a timed-out run writes the transaction's receipt."""
        with self._lock:
            t = self.txns.get(txn)
            if t is None:
                return F.not_ok(f"unknown transaction {txn}", txn=txn, state="unknown")
            state = t["state"]
            if state == "prepared" and t["started"] and not t["done"]:
                state = "in-doubt"
            pending = t["done"] and not t["receipt"]
        if pending:
            self._finish(txn)
        return F.ok(txn=txn, state=state, outcome=t.get("outcome"), written=t["receipt"])

    # -- the transaction -----------------------------------------------------------------
    def _phase2(self, txn, tl):
        """validate -> commit, or abort; undo the committed ones if a commit fails. Inside
        the gap. Returns the phase-2 result; never raises."""
        now, t = self.clock, self.txns[txn]
        res = t["res"]
        tl["gap_start"] = now()
        for p in self.participants:
            res[p.name]["validate"] = _call(p, "validate", txn)
        tl["vote"] = now()
        rollback = {"status": "not_needed", "participants": {}}
        out = {"rollback": rollback, "extra": [], "hashes": {}}
        if not all(res[p.name]["validate"]["ok"] for p in self.participants):
            for p in self.participants:
                res[p.name]["abort"] = _call(p, "abort", txn)
            out["outcome"] = "aborted"
        else:
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
            else:
                out["outcome"] = "rolled_back"
                out["extra"].append({**F.finding(
                    "participant_failed", res[failed.name]["commit"].get("error"),
                    "commit ok on every participant"), "participant": failed.name})
                for p in reversed(committed):
                    rollback["participants"][p.name] = res[p.name]["undo"] = \
                        _call(p, "undo", txn)
                for p in self.participants:
                    if p not in committed:
                        res[p.name]["abort"] = _call(p, "abort", txn)
                rollback["status"] = ("undone" if all(
                    u["ok"] for u in rollback["participants"].values()) else "partial")
        tl["gap_end"] = now()
        return out

    def run(self, plan, edit_time=None, baseline_restart=None):
        """One transaction; appends a `txn` receipt. edit_time: when the edit was made (unix
        seconds; default now). baseline_restart: {"seconds", "source"} of the warm restart
        this replaces. Returns {"ok", "outcome", "txn", "receipt", "written"}, where outcome is
        committed, aborted, rolled_back, or in-doubt (resolve it with `status`)."""
        now = self.clock
        tl = {"edit": edit_time if edit_time is not None else now()}
        txn = plan.get("id") or f"txn-{uuid.uuid4().hex[:8]}"
        plan = {**plan, "id": txn}
        t = self.txns[txn] = {"plan": plan, "edit": tl["edit"], "tl": tl, "state": "prepared",
                              "res": {p.name: {} for p in self.participants},
                              "started": False, "cancelled": False, "done": False,
                              "receipt": False, "baseline": baseline_restart}
        # phase 1: outside the gap, steps go on
        tl["prepare_start"] = now()
        for p in self.participants:
            t["res"][p.name]["prepare"] = _call(p, "prepare", plan)
        tl["prepared"] = now()
        if not all(t["res"][p.name]["prepare"]["ok"] for p in self.participants):
            for p in self.participants:
                t["res"][p.name]["abort"] = _call(p, "abort", txn)
            self._done(txn, {"outcome": "aborted", "extra": [], "hashes": {},
                             "rollback": {"status": "not_needed", "participants": {}}})
            return self._finish(txn)

        # phase 2: inside the gap
        def job():
            with self._lock:
                if t["cancelled"]:
                    return None
                t["started"] = True
            out = self._phase2(txn, tl)
            self._done(txn, out)
            return out
        try:
            self._in_gap(job)
        except Exception as e:
            with self._lock:
                if not t["started"]:
                    t["cancelled"] = True
            if t["cancelled"]:
                for p in self.participants:
                    t["res"][p.name]["abort"] = _call(p, "abort", txn)
                self._done(txn, {"outcome": "aborted", "hashes": {},
                                 "rollback": {"status": "not_needed", "participants": {}},
                                 "extra": [F.finding("participant_failed",
                                                     f"phase 2 never started: {type(e).__name__}:"
                                                     f" {e}", "a step gap within the timeout")]})
                return self._finish(txn)
            if not t["done"]:
                return {"ok": False, "outcome": "in-doubt", "txn": txn, "receipt": None,
                        "written": {"ok": False, "error": "in doubt: ask status(txn)"},
                        "error": f"{type(e).__name__}: {e}"}
        return self._finish(txn)

    def _done(self, txn, out):
        t = self.txns[txn]
        with self._lock:
            t["tl"]["txn_end"] = self.clock()
            t["out"] = out
            t["outcome"] = out["outcome"]
            t["state"] = "committed" if out["outcome"] == "committed" else "aborted"
            t["done"] = True

    def _finish(self, txn):
        """Build and append the `txn` receipt once phase 2 is over (or phase 1 aborted)."""
        t = self.txns[txn]
        with self._lock:
            if t["receipt"]:
                return {"ok": t["outcome"] == "committed", "outcome": t["outcome"], "txn": txn,
                        "receipt": t["rec"], "written": {"ok": True, "note": "already written"}}
            t["receipt"] = True
        plan, tl, out, res = t["plan"], t["tl"], t["out"], t["res"]
        if out["outcome"] == "committed":       # code hashes, read after the gap
            for p in self.participants:
                out["hashes"][p.name] = _call(p, "proof", txn).get("hashes")
        fs = [f for r in res.values() for x in r.values()
              for f in x.get("findings", [])] + out["extra"]
        modules = plan.get("modules", {})
        agree = _hashes_agree(out["hashes"])

        def span(a, b):
            return round(tl[b] - tl[a], 6) if a in tl and b in tl else None
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
            "participants": res,
            "commit_hashes": {"participants": out["hashes"],
                              "agree": agree},
            "findings": fs,
            "decisions": {"accepted": sorted({f["check"] for f in fs if f.get("accepted")}),
                          "blocking": sorted({f["check"] for f in fs
                                              if not f.get("accepted")})},
            "outcome": out["outcome"], "timeline": tl,
            "edit_to_run_s": span("edit", "committed"),
            "prepare_s": span("prepare_start", "prepared"),
            "gap_s": span("gap_start", "gap_end"),
            "baseline_restart_s": t["baseline"],
            "rollback": out["rollback"],
        }
        t["rec"] = rec
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
        produced. In the gap. Appends an `undo` receipt."""
        now = self.clock
        t = self.txns.get(txn, {})
        resets = t.get("plan", {}).get("undo_resets", [])
        tl = {"queued": now()}
        try:
            res, err = self._in_gap(lambda: self._undo_phases(txn, resets, tl)), None
        except Exception as e:
            res, err = None, f"{type(e).__name__}: {e}"
        tl["undo_end"] = now()
        res = res or {}
        calls = [x for r in res.values()
                 for x in [r.get("undo")] + list(r.get("reset", {}).values()) if x]
        fs = [{**f, "participant": n} for n, r in res.items()
              for x in [r.get("undo")] + list(r.get("reset", {}).values()) if x
              for f in x.get("findings", [])]
        ok = err is None and all(x["ok"] for x in calls)
        rec = {"schema": R.SCHEMA, "kind": "undo", "id": txn, "participants": res,
               "resets": list(resets), "status": "undone" if ok else "partial",
               "findings": fs + ([F.finding("undo_incomplete", f"undo did not finish: {err}",
                                            "a step gap within the timeout")] if err else []),
               "timeline": tl, "undo_s": round(tl["undo_end"] - tl["queued"], 6)}
        if ok and t:
            with self._lock:
                t["state"] = "undone"
        return {"ok": ok, "participants": res, "receipt": rec, "written": self._write(rec)}


class StepGapCoordinator(Coordinator):
    """The host for participants that step together (worker ranks + engine core), living in
    the loop owner: phase 1 (prepare) runs in the caller's thread while steps go on; phase 2
    (validate, commit or abort, undo of a failed commit) runs in ONE gap between two steps.
    So no step ever sees participants on different versions, and the loop stalls only for
    phase 2 (`gap_s` in the receipt). `gap(fn)` queues fn to the loop owner and returns its
    result (for vLLM: the engine-core busy loop, which calls the workers by collective_rpc;
    that completes on every worker before the next step). Participants must not quiesce on
    their own: build them with the default `quiesce`."""

    def __init__(self, participants, gap, **kw):
        super().__init__(participants, **kw)
        self.gap = gap

    def _in_gap(self, fn):
        return self.gap(fn)
