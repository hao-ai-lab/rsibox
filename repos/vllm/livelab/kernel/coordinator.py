"""Coordinator, part 1: the participant side of two-phase transactions over participants
(worker ranks, engine core, API server).

    phase 1, while steps go on: prepare (stage beside the old state, run checks, vote)
    phase 2, in the gap: validate (the staged plan is still valid), then commit or
      abort; undo takes a committed transaction back

The host that drives them (`Coordinator`, `StepGapCoordinator`: votes, the step gap, the
phase-2 deadline, group participants, receipts) is the next PR, on top of this one.

Participant protocol (every call returns an ok-dict and never raises; a crash or a reply that
is not an ok-dict counts as not-ok):

    prepare(plan) -> {"ok", "staged", "findings"}     phase 1; ok is the vote
    validate(ref) -> {"ok", "findings"}              phase 2: staged identities still hold
    status(txn)   -> {"ok", "state"}                 prepared, committed, aborted, undone
                                                     (undone once undo ran, committed or not)
    commit(ref)   -> {"ok", ...}                     at the participant's quiesce point
    abort(ref)    -> {"ok"}                          drop staged state
    undo(ref)     -> {"ok", ...}                     journal.undo_to(the txn's mark)
    proof(txn)    -> {"ok", "hashes", "witness"}     code hashes right after commit
    witness_check(txn) -> {"ok", "counts", "findings"}  after the workload
    reset(kind)   -> {"ok", ...}                     data reset hook: "R2", "R3", ...
    fingerprint(modules) -> {"ok", "code"}           {key: [content hash, code object id]}

`ref` is a transaction id, or `{"txn": id, "deadline_at": unix seconds}` when the host gives
phase 2 a deadline (`phase2_deadline_s`). A participant must not switch after `deadline_at`
(LocalParticipant refuses a commit that starts late). Participants share the host's clock (one
node; across nodes the clock skew shifts the cut-off by the skew).

A plan is plain data, so it can cross a process boundary:

    {"id": str (optional),
     "modules": {module: {"source": patched text, "base_hash": text_hash of the base,
                          "targets": [[qualname, first line in the base]] or "auto"
                                     (default: the definitions that changed, found by stage),
                          "scan": "auto" | "always",
                          "trees": (base AST, patch AST)}},  in-process only (not plain
                                     data): the caller's parse of the text the participant
                                     runs and of `source`, so stage does not parse again
     "repairs": [{"cell": python source, "declares": [names in the participant's scope]}],
                 a cell runs at commit with the scope's names and `journal` (this
                 participant's): `journal.record_attr(obj, name)` before assigning what the
                 declared-object audit cannot undo (an nn.Module submodule, Parameter.data),
                 `journal.record_call(undo_fn, label)` for an in-place re-init
     "accept": [check names] or "all",
     "undo_resets": ["R2" | "R3" | ...],   resets the undo runs after the code undo
     "recipes": [...], "predicted_resources": [...], "certainty": str}

`LocalParticipant` runs the kernel in this process. `CallParticipant` is the shape the engine
side implements: one `call(method, payload) -> dict` per participant (for vLLM: collective_rpc
to a worker rank, or the engine-core input-queue channel), which on the far side dispatches
to a LocalParticipant.

Not in v1 (see the kernel README): the analyzer child, cross-module overlays, the
docstring-only label, per-block witness keys, compile units, and the new-field
backfill. A `new_fields` finding (route T1) blocks like any other.
"""
import json
import sys
import time
import types

from . import findings as F
from . import journal as J
from . import locate as L
from . import swap as S
from . import witness as W


def txn_ref(ref):
    """(txn id, deadline_at or None) from a phase-2 argument: an id or {"txn", "deadline_at"}."""
    if isinstance(ref, dict):
        return ref.get("txn"), ref.get("deadline_at")
    return ref, None


PHASE2 = ("validate", "commit", "abort", "undo")


class Participant:
    name = "participant"
    bounded = False      # True: answers every phase-2 call by its deadline_at on its own

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

    def at_base(self, _=None):
        return F.not_ok("no at_base")


class CallParticipant(Participant):
    """A participant in another process, reached through `call(method, payload) -> dict`."""

    def __init__(self, name, call, bounded=False):
        self.name, self._call, self.bounded = name, call, bounded

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

    def at_base(self, _=None):
        return self._do("at_base", None)


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
    them too). A plan that would change nothing is an `empty_change` finding.

    A plan module without `targets` lets stage find them; the reply's `targets` says which it
    staged."""

    def __init__(self, name, alias=None, quiesce=None, scope=None, resets=None,
                 checks=None, capturing=None, engine_names=None):
        self.name = name
        # at_base: module names the engine adds itself (a compiler's generated globals), not
        # left by an edit: fn(name) -> True, listed apart (`engine_names`)
        self.engine_names = engine_names
        self.checks = list(checks or ())   # engine checks: fn(module, items) -> [findings]
        self.resets = dict(resets or {})
        self.alias = dict(alias or {})
        self.quiesce = quiesce or (lambda fn: fn())
        self.scope = dict(scope or {})
        self.journal = J.Journal(name)
        self.witness = W.Witness(capturing)   # capturing: fn() -> True while a graph records
        self._staged, self._marks, self._done = {}, {}, {}
        self._state = {}              # txn -> prepared | committed | aborted | undone | partial
        self._partial = {}            # txn -> findings of an undo that could not restore all
        self._sources = {}            # committed txn -> {plan module: source it installed}
        self._codes = {}              # committed txn -> [code objects it swapped in]
        self._pending = {}            # prepared txn -> {plan module: source}
        self._maps = {}               # committed txn -> {plan module: line map of its swap}
        self._pending_maps = {}       # prepared txn -> the same, until commit
        self._cell_keys = {}          # committed txn -> witness keys its cells armed
        self._base = {}               # plan module -> (digest, names) when first touched
        self._scope_keys = None       # scope objects' attribute names at the first prepare

    def _live_source(self, module):
        """The text the live module runs: the latest committed transaction's, else None."""
        for txn in reversed(list(self._sources)):
            if module in self._sources[txn]:
                return self._sources[txn][module]
        return None

    def _line_maps(self, module):
        """The line maps of the committed swaps of this module, newest first: a later plan's
        targets are keyed at the installed text's lines, and the definitions no swap replaced
        still carry the lines of the text they were compiled from (swap.stage line_maps)."""
        return [self._maps[t][module] for t in reversed(list(self._maps))
                if module in self._maps[t]]

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
        txn, _ = txn_ref(txn)
        return F.ok(txn=txn, state=self._state.get(txn, "unknown"))

    def validate(self, txn):
        """Phase 2, in the gap, before the vote: is the staged plan still valid?"""
        txn, _ = txn_ref(txn)
        if txn not in self._staged:
            return F.not_ok(f"nothing staged for {txn}")
        staged, cells, accept = self._staged[txn]
        fs = []
        for st in staged:
            fs += S.validate(st)["findings"]
        fs = [{**f, "participant": self.name} for f in fs]
        # changed_since_prepare always votes no; a frame live at the quiesce point goes
        # through the plan's accept, as it did at stage before
        frames = F.decide([f for f in fs if f["check"] == "running_frame"], accept)
        rest = [f for f in fs if f["check"] != "running_frame"]
        if not rest and frames["proceed"]:
            return F.ok(checked=sum(len(st.expect) for st in staged), findings=frames["findings"])
        return F.not_ok("changed since prepare" if rest else "running frame at the gap",
                        rest + frames["findings"])

    def _actual(self, module):
        for k, v in self.alias.items():
            if module == k or module.startswith(k + "."):
                return v + module[len(k):]
        return module

    def _stage(self, m, spec, own):
        return S.stage(self._actual(m), spec["source"], spec.get("targets", "auto"),
                       spec.get("base_hash"), key_module=m, scan=spec.get("scan", "auto"),
                       base_source=self._live_source(m), own_codes=own,
                       line_maps=self._line_maps(m), trees=spec.get("trees"))

    def _remember_base(self, plan):
        """At the first prepare that touches a module: its code digest and namespace names;
        at the first prepare at all: the scope objects' attribute names (and one level down),
        so `at_base` can say what is still not at the start state. Cheap: the modules'
        namespaces and the scope's attribute dicts, no heap scan."""
        for m in plan.get("modules", {}):
            if m not in self._base:
                mod = sys.modules.get(self._actual(m))
                fp = self.fingerprint([m])
                if mod is not None and fp["ok"]:
                    self._base[m] = (L.text_hash(json.dumps(fp["code"], sort_keys=True)),
                                     set(vars(mod)))
        if self._scope_keys is None:
            self._scope_keys = _attr_names(self.scope)

    def at_base(self, _=None):
        """Is this participant back at its start state? {ok, pending_txns, journal_len,
        modules: {module: [base digest, live digest]}, extra_names (module names added and
        left), extra_attrs (scope attributes added and left), poisoned (an undo that could
        not restore all)}. The witness global stays by design (a no-op stub after undo) and
        is not an extra. For a warm engine pool: apply, undo, then check before the next job."""
        pending = sorted(t for t, st in self._state.items()
                         if st in ("prepared", "committed", "partial"))
        mods, extra_names, engine = {}, {}, {}
        for m, (digest, names) in self._base.items():
            fp = self.fingerprint([m])
            live = L.text_hash(json.dumps(fp["code"], sort_keys=True)) if fp["ok"] else None
            mods[m] = [digest, live]
            mod = sys.modules.get(self._actual(m))
            # the witness global stays by design, and so do aliases a tracer made of it
            extra = sorted(n for n in (set(vars(mod)) - names if mod is not None else ())
                           if W.HIT not in n)
            eng = [n for n in extra if self.engine_names and self.engine_names(n)]
            if eng:
                engine[m] = eng
            extra = [n for n in extra if n not in eng]
            if extra:
                extra_names[m] = extra
        extra_attrs = {}
        if self._scope_keys is not None:
            now = _attr_names(self.scope)
            for path, keys in now.items():
                new = sorted(keys - self._scope_keys.get(path, keys))
                if new:
                    extra_attrs[path] = new
        poisoned = bool(self._partial)
        ok = (not pending and not self.journal.entries and not extra_names
              and not extra_attrs and not poisoned and all(a == b for a, b in mods.values()))
        return F.ok(**{"ok": ok}, pending_txns=pending, journal_len=len(self.journal.entries),
                    modules=mods, extra_names=extra_names, extra_attrs=extra_attrs,
                    engine_names=engine, poisoned=poisoned)

    def prepare(self, plan):
        self._remember_base(plan)
        txn, accept = plan["id"], plan.get("accept", ())
        fs, staged, items, ok, timings, targets = [], [], {}, True, {}, {}
        own = [c for cs in self._codes.values() for c in cs]
        res = {m: self._stage(m, spec, own) for m, spec in plan.get("modules", {}).items()}
        for m in plan.get("modules", {}):
            r = res[m]
            fs += r["findings"]
            ok = ok and r["ok"]
            if r["staged"] is not None:
                staged.append(r["staged"])
            items.update(r["items"])
            timings[m] = r.get("timings", {})
            if "targets" in r:
                targets[m] = r["targets"]
            for check in self.checks:         # the engine's own static checks of the targets
                try:
                    fs += check(m, r["items"]) or []
                except Exception as e:
                    fs.append({**F.finding("patch_invalid", f"engine check {check!r}: "
                                           f"{type(e).__name__}: {e}", "the check runs",
                                           subject=m), "advisory": True})
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
            fs.append(F.finding("empty_change", f"{len(items)} target(s), none changed; "
                                "no definition added or deleted; no repair cell",
                                "at least one change to install"))
        # tag before deciding: a standing accept matches on participant and code_hash
        d = F.decide([{**f, "participant": self.name} for f in fs], accept)
        out_fs = d["findings"]
        if not (ok and d["proceed"]):
            self._state[txn] = "aborted"
            return F.not_ok(f"vote no: {d['blocking'] or 'stage failed'}", out_fs, staged=items)
        self._staged[txn] = (staged, cells, accept)
        self._pending[txn] = {m: spec["source"] for m, spec in plan.get("modules", {}).items()}
        self._pending_maps[txn] = {m: r["staged"].line_map for m, r in res.items()
                                   if r["staged"] is not None}
        self._state[txn] = "prepared"
        return F.ok(staged=items, findings=out_fs, timings=timings, targets=targets)

    def commit(self, txn):
        """Apply the staged transaction at the quiesce point. With a deadline (`txn` is
        {"txn", "deadline_at"}), a commit that would start after deadline_at does nothing:
        the host already counted this participant as failed and undid the others."""
        txn, deadline_at = txn_ref(txn)
        if txn not in self._staged:
            return F.not_ok(f"nothing staged for {txn}")
        staged, cells, accept = self._staged.pop(txn)
        self._state[txn] = "aborted"  # until the commit below succeeds
        gone = []                     # set if the quiesce call gave up: the job must not run

        def do():
            if gone:
                return F.not_ok("commit cancelled before it ran")
            if deadline_at is not None and time.time() > deadline_at:
                return F.not_ok("commit started after the phase-2 deadline; nothing applied",
                                [{**F.finding("deadline_missed", "commit reached the "
                                              f"participant {time.time() - deadline_at:.3f} s "
                                              "after the deadline", "commit by the deadline"),
                                  "participant": self.name}])
            mark = self.journal.mark(txn)
            for st in staged:
                r = S.apply(st, self.journal, self.witness)
                if not r["ok"]:
                    self.journal.undo_to(mark)
                    return r
            fs = []
            armed, n_routes = [], len(self.journal.routes)

            def routes():
                rs = sorted(set(self.journal.routes[n_routes:]))
                del self.journal.routes[n_routes:]
                return rs

            def arm(key):
                """A witness counter for what this cell changes (see witness.arm)."""
                armed.append(key)
                return self.witness.arm(key, self.journal)
            for code, declares in cells:
                decl = {n: self.scope[n] for n in declares}
                audit = {n: o for n, o in self.scope.items() if n not in decl}
                rep = None
                try:
                    with self.journal.repair(decl, audit) as rep:
                        exec(code, {**self.scope, "journal": self.journal, "arm": arm,
                                    "witness": self.witness})
                except Exception as e:
                    # what the cell changed outside the journal stays: say so (the audit's
                    # findings), so a rolled-back cell is never read as a clean rollback
                    left = list(rep.findings) if rep is not None else []
                    u = self.journal.undo_to(mark)
                    return F.not_ok("repair failed", [F.finding(
                        "repair_failed", f"{type(e).__name__}: {e}", "the cell runs")]
                        + left + u["findings"], rebuild_routes=routes())
                fs += rep.findings
            d = F.decide([{**f, "participant": self.name} for f in fs], accept)
            fs = d["findings"]
            rebuild_routes = routes()
            if not d["proceed"]:
                u = self.journal.undo_to(mark)
                return F.not_ok(f"commit refused: {d['blocking']}", fs + u["findings"],
                                rebuild_routes=rebuild_routes)
            self._marks[txn] = mark
            self._done[txn] = staged
            self._sources[txn] = self._pending.pop(txn, {})
            self._maps[txn] = self._pending_maps.pop(txn, {})
            self._codes[txn] = [c for st in staged for _, c, _, _ in st.swaps]
            self._cell_keys[txn] = armed
            self._state[txn] = "committed"
            return F.ok(changes=len(self.journal.entries) - mark, findings=fs,
                        journal=self.journal.describe(mark),
                        **({"rebuild_routes": rebuild_routes} if rebuild_routes else {}))

        try:
            return self.quiesce(do)
        except Exception as e:
            gone.append(True)
            return F.not_ok(f"commit: {type(e).__name__}: {e}")

    def abort(self, txn):
        txn, _ = txn_ref(txn)
        self._pending.pop(txn, None)
        self._pending_maps.pop(txn, None)
        if self._staged.pop(txn, None) is not None or self._state.get(txn) == "prepared":
            self._state[txn] = "aborted"
        return F.ok()

    def undo(self, txn):
        txn, _ = txn_ref(txn)
        if txn in self._partial:
            # An earlier undo ran the whole journal back but could not restore everything
            # (e.g. closure_not_restored). Nothing is left to undo, and saying "undone" now
            # would hide what is still running: repeat the findings.
            return F.not_ok("undo already ran and was partial", self._partial[txn], undone=0)
        if txn not in self._marks:
            # Nothing of txn is installed here (never committed, or a late commit after a
            # deadline applied nothing). Drop what is staged, so a commit arriving after this
            # applies nothing, and record the undo: every participant (every rank) of an undone
            # transaction then says "undone", not some "aborted" and some "undone".
            if txn in self._state:
                self._staged.pop(txn, None)
                self._pending.pop(txn, None)
                self._pending_maps.pop(txn, None)
                self._state[txn] = "undone"
            return F.ok(undone=0, note="nothing committed")
        mark = self._marks[txn]
        # The journal undoes back to the mark, so it also undoes every transaction committed
        # after this one. Say so and record their state: they must not stay "committed".
        later = sorted((t for t, m in self._marks.items() if m > mark),
                       key=lambda t: self._marks[t])
        try:
            r = self.quiesce(lambda: self.journal.undo_to(mark))
        except Exception as e:
            return F.not_ok(f"undo: {type(e).__name__}: {e}")
        # The journal is back at the mark either way (a failed inverse is skipped and
        # reported), and it undid every later transaction too.
        for t in [txn] + later:
            self._marks.pop(t, None)
            for d in (self._done, self._sources, self._codes, self._maps, self._cell_keys):
                d.pop(t, None)
            self._state[t] = "undone" if r["ok"] else "partial"
        if not r["ok"]:
            self._partial[txn] = r.get("findings", [])
        if later:
            r = {**r, "also_undone": later, "findings": r.get("findings", []) + [{
                **F.finding("undo_cascade",
                            f"undoing {txn} also undid later transactions {later}",
                            f"only {txn} undone"), "participant": self.name}]}
        return r

    def proof(self, txn):
        hashes, keys = {}, list(self._cell_keys.get(txn, []))
        for st in self._done.get(txn, []):
            keys += st.keys
            for k, fns in st.functions.items():
                hashes[k] = sorted({L.content_hash(f.__code__) for f in fns})
        return F.ok(hashes=hashes, witness={k: self.witness.counts.get(k, 0) for k in keys},
                    witness_capture={k: self.witness.capture_counts.get(k, 0) for k in keys})

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
            keys += st.keys                # added definitions too: they get a witness
            old.update(st.old_codes)
        keys += self._cell_keys.get(txn, [])      # counters the repair cells armed
        r = self.witness.check(keys, old)
        if not keys and self._state.get(txn) == "committed":
            r["findings"].append(F.finding(
                "no_witness_possible", "the transaction swapped no code and its cells armed "
                "no counter: an empty witness proves nothing", "at least one witness key"))
        return r


def _attr_names(scope):
    """{path: set of attribute names} of the scope's objects and of the objects their
    attributes hold (one level down, as a repair cell's audit watches)."""
    out = {}
    for n, o in scope.items():
        d = getattr(o, "__dict__", None)
        if not isinstance(d, dict):
            continue
        out[n] = set(d)
        for k, v in list(d.items()):
            dv = getattr(v, "__dict__", None)
            if isinstance(dv, dict) and not isinstance(v, (type, types.ModuleType,
                                                           types.FunctionType)):
                out[f"{n}.{k}"] = set(dv)
    return out
