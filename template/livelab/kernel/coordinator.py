"""Coordinator: two-phase transactions over participants (worker ranks, engine core, API server).

    prepare (every participant stages beside the old state, runs its checks, votes)
      -> unanimous yes: commit on each at its quiesce point
      -> otherwise:     abort on all (drop staged state; nothing was visible)
    a commit that fails or crashes -> undo on the participants that already committed

Participant protocol (every call returns an ok-dict and never raises; a crash or a reply that
is not an ok-dict counts as not-ok):

    prepare(plan) -> {"ok", "staged", "findings"}     ok is the vote
    commit(txn)   -> {"ok", ...}                     at the participant's quiesce point
    abort(txn)    -> {"ok"}                          drop staged state
    undo(txn)     -> {"ok", ...}                     journal.undo_to(the txn's mark)
    proof(txn)    -> {"ok", "hashes", "witness"}     optional

A plan is plain data, so it can cross a process boundary:

    {"id": str (optional),
     "modules": {module: {"source": patched text, "base_hash": text_hash of the base,
                          "targets": [[qualname, first line in the base]]}},
     "repairs": [{"cell": python source, "declares": [names in the participant's scope]}],
     "accept": [check names] or "all",
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


class LocalParticipant(Participant):
    """The kernel in this process. `alias`: {plan module prefix: this process's prefix} (a
    participant that loaded the package under another name). `quiesce(fn)`: runs fn at this
    participant's step boundary and returns its result (default: now, in the caller's thread).
    `scope`: names a repair cell can use."""

    def __init__(self, name, alias=None, quiesce=None, scope=None):
        self.name = name
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
    def __init__(self, participants, writer=None, engine=None, interface_version="0",
                 clock=time.time):
        self.participants = list(participants)
        self.writer = writer
        self.engine = engine or {"name": "unknown", "version": None, "commit": None}
        self.interface_version = interface_version
        self.clock = clock

    def run(self, plan, edit_time=None, baseline_restart=None):
        """One transaction. edit_time: when the edit was made (unix seconds; default now).
        baseline_restart: {"seconds", "source"} of the warm restart this replaces.
        Returns {"ok", "outcome", "txn", "receipt", "written"}; never raises."""
        now = self.clock
        tl = {"edit": edit_time if edit_time is not None else now()}
        txn = plan.get("id") or f"txn-{uuid.uuid4().hex[:8]}"
        plan = {**plan, "id": txn}
        res = {p.name: {} for p in self.participants}
        tl["prepare_start"] = now()
        for p in self.participants:
            res[p.name]["prepare"] = _call(p, "prepare", plan)
        tl["vote"] = now()
        yes = all(res[p.name]["prepare"]["ok"] for p in self.participants)
        undo = {"status": "not_needed", "participants": {}}
        proofs, extra = {"participants": {}, "hashes_agree": None}, []
        if not yes:
            outcome = "aborted"
            for p in self.participants:
                res[p.name]["abort"] = _call(p, "abort", txn)
        else:
            committed, failed = [], None
            for p in self.participants:
                r = res[p.name]["commit"] = _call(p, "commit", txn)
                if not r["ok"]:
                    failed = p
                    break
                committed.append(p)
            if failed is None:
                outcome = "committed"
                tl["committed"] = now()
                for p in self.participants:
                    proofs["participants"][p.name] = _call(p, "proof", txn)
                hs = [json.dumps(v.get("hashes"), sort_keys=True)
                      for v in proofs["participants"].values()]
                proofs["hashes_agree"] = len(set(hs)) == 1
            else:
                outcome = "rolled_back"
                extra.append({**F.finding("participant_failed", res[failed.name]["commit"]
                                          .get("error"), "commit ok on every participant"),
                              "participant": failed.name})
                for p in reversed(committed):
                    undo["participants"][p.name] = res[p.name]["undo"] = _call(p, "undo", txn)
                for p in self.participants:
                    if p not in committed:
                        res[p.name]["abort"] = _call(p, "abort", txn)
                undo["status"] = ("undone" if all(u["ok"] for u in undo["participants"].values())
                                  else "partial")
        fs = [f for p in self.participants for r in res[p.name].values()
              for f in r.get("findings", [])] + extra
        tl["finished"] = now()
        modules = plan.get("modules", {})
        rec = {
            "schema": R.SCHEMA, "id": txn, "interface_version": self.interface_version,
            "engine": self.engine,
            "base_hash": L.text_hash(json.dumps(
                {m: s.get("base_hash") for m, s in modules.items()}, sort_keys=True)),
            "patch_hash": L.text_hash(json.dumps(
                {"modules": {m: L.text_hash(str(s.get("source", ""))) for m, s in modules.items()},
                 "repairs": plan.get("repairs", [])}, sort_keys=True)),
            "plan": {"targets": {m: s.get("targets", []) for m, s in modules.items()},
                     "predicted_resources": plan.get("predicted_resources", []),
                     "recipes": plan.get("recipes", []), "certainty": plan.get("certainty"),
                     "accept": plan.get("accept", [])},
            "actual_recipes": ((["R0 swap"] if modules else [])
                               + (["R10 repair"] if plan.get("repairs") else []))
            if outcome == "committed" else [],
            "participants": res, "proofs": proofs, "findings": fs,
            "decisions": {"accepted": sorted({f["check"] for f in fs if f.get("accepted")}),
                          "blocking": sorted({f["check"] for f in fs
                                              if not f.get("accepted")})},
            "outcome": outcome, "timeline": tl,
            "edit_to_run_s": (round(tl["committed"] - tl["edit"], 6)
                              if "committed" in tl else None),
            "edit_to_finish_s": round(tl["finished"] - tl["edit"], 6),
            "baseline_restart_s": baseline_restart,
            "undo": undo,
        }
        written = self.writer.append(rec) if self.writer else {"ok": False, "error": "no writer"}
        return {"ok": outcome == "committed", "outcome": outcome, "txn": txn, "receipt": rec,
                "written": written}

    def undo(self, txn):
        """Undo a committed transaction on every participant, newest participant first."""
        out = {p.name: _call(p, "undo", txn) for p in reversed(self.participants)}
        return {"ok": all(r["ok"] for r in out.values()), "participants": out}
