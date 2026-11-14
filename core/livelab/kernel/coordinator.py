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

Phase-2 deadline. The gap is a global barrier: a participant that hangs in commit stalls every
step. With a deadline D, validate and commit must answer by gap start + D/2; the rollback has
until gap start + D, so the gap ends within D. A participant that misses its deadline is
treated as failed (`deadline_missed`), the others are undone in the gap, and the host is
poisoned: its late call may still be running, so only undo or respawn are allowed until
`recovered()`. The host calls an unbounded participant on a `Lane` (one thread per
participant, calls in order) and stops waiting at the deadline; later calls queue behind the
late one, so an undo runs after a late commit. A `bounded` participant (its transport has a
timeout and it bounds its own calls, as the vLLM worker group does) is called directly.

A group participant (several ranks reached by one call, e.g. one collective_rpc) answers with
the per-rank replies in `ranks` (`merge_ranks`): witness counts and code hashes are kept per
rank in the receipts, and ranks that differ are findings (`witness_uneven`) or
`commit_hashes.ranks_agree: false`.

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

Not in v1 (later PRs; see the kernel README): rebuilds after the commit (R1 recapture, R6
weight reload), the analyzer child, cross-module overlays, the docstring-only label, per-block
witness keys, compile units, and the new-field backfill. A `new_fields` finding (route T1)
blocks like any other.
"""
import json
import math
import queue
import sys
import threading
import time
import types
import uuid

from . import findings as F
from . import jit as JIT
from . import journal as J
from . import locate as L
from . import receipt as R
from . import reinit as RI
from . import swap as S
from . import witness as W


def txn_ref(ref):
    """(txn id, deadline_at or None) from a phase-2 argument: an id or {"txn", "deadline_at"}."""
    if isinstance(ref, dict):
        return ref.get("txn"), ref.get("deadline_at")
    return ref, None


PHASE2 = ("validate", "commit", "abort", "undo")
UNDO_WAIT_S = 30.0   # the longest an undo without a deadline waits for an in-flight commit


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


def _imported(mod, name):
    """Is mod.name a module that the import system holds under its own name (an import)?"""
    v = vars(mod).get(name)
    return isinstance(v, types.ModuleType) and sys.modules.get(v.__name__) is v


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
        self._ours = {}               # plan module -> names a transaction or cell added to it
        self._scope_keys = None       # scope objects' attribute names at the first prepare
        self._seq = {}                # committed txn -> commit order (undo cascades by it)
        self._next_seq = 0
        self._lock = threading.Lock()  # the commit/undo decision of an in-flight job
        self._inflight = {}           # txn -> {"done": Event, "gone": bool} while it commits
        self._reinit = {}             # prepared txn -> [constructor field reinit (reinit.py)]

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
        rest += self._recheck_reinit(self._reinit.get(txn, ()))
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

    def _note_added(self, mark):
        """Remember the names the journal entries since `mark` added to a plan module's
        namespace, so at_base can tell them from names the engine added on its own."""
        dicts = {}
        for m in self._base:
            mod = sys.modules.get(self._actual(m))
            if mod is not None:
                dicts[id(vars(mod))] = dicts[id(mod)] = m
        for e in self.journal.entries[mark:]:
            m = dicts.get(id(e.target))
            if m is not None and e.kind in ("added", "getattr_added"):
                self._ours.setdefault(m, set()).add(e.name)

    def at_base(self, _=None):
        """Is this participant back at its start state? {ok, pending_txns, journal_len,
        modules: {module: [base digest, live digest]}, extra_names (module names added and
        left), extra_attrs (scope attributes added and left), poisoned (an undo that could
        not restore all)}. The witness global stays by design (a no-op stub after undo) and
        is not an extra. A name the engine added since base by importing a module lazily (a
        module value that is the imported module, under a name no transaction or cell of
        this participant added: vLLM's LazyLoader binds `llguidance.hf` at first use) is
        listed apart in `lazy_imports` and does not make at_base false. For a warm engine
        pool: apply, undo, then check before the next job."""
        pending = sorted(t for t, st in self._state.items()
                         if st in ("prepared", "committed", "partial"))
        mods, extra_names, engine, lazy = {}, {}, {}, {}
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
            imp = [n for n in extra if n not in self._ours.get(m, ()) and _imported(mod, n)]
            if imp:
                lazy[m] = imp
            extra = [n for n in extra if n not in imp]
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
                    engine_names=engine, lazy_imports=lazy, poisoned=poisoned)

    def prepare(self, plan):
        """Phase 1. The plan's "id" is optional: without one the participant names the
        transaction, and the reply's `txn` says which. An id that is prepared, committed or
        partial here is refused (re-preparing it would drop the earlier one's undo mark).
        A plan that is not as the schema documents is a no vote, never an exception."""
        txn = None
        try:
            txn = (plan.get("id") if isinstance(plan, dict) else None) \
                or f"txn-{uuid.uuid4().hex[:8]}"
            if self._state.get(txn) in ("prepared", "committed", "partial"):
                return F.not_ok(f"transaction {txn} is {self._state[txn]} here", [{
                    **F.finding("patch_invalid", f"transaction id {txn} is "
                                f"{self._state[txn]} here", "a new transaction id",
                                subject=txn), "participant": self.name}], staged={}, txn=txn)
            return {**self._prepare(plan, txn), "txn": txn}
        except Exception as e:
            if txn is not None:
                self._state[txn] = "aborted"
                self._staged.pop(txn, None)
                self._pending.pop(txn, None)
                self._pending_maps.pop(txn, None)
            return F.not_ok(f"prepare: {type(e).__name__}: {e}", [{
                **F.finding("patch_invalid", f"{type(e).__name__}: {e}",
                            "a plan as the coordinator's plan schema documents"),
                "participant": self.name}], staged={}, txn=txn)

    def _prepare(self, plan, txn):
        self._remember_base(plan)
        accept = plan.get("accept", ())
        fs, staged, items, ok, timings, targets = [], [], {}, True, {}, {}
        own = [c for cs in self._codes.values() for c in cs]
        res = {m: self._stage(m, spec, own) for m, spec in plan.get("modules", {}).items()}
        reinits = []
        for m in plan.get("modules", {}):
            r = res[m]
            fs += [self._resolve_reinit(m, f, reinits) if f.get("reinit") else f
                   for f in r["findings"]]
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
                and not any(st.removes or st.imports for st in staged):
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
        if reinits:
            self._reinit[txn] = reinits
        self._pending[txn] = {m: spec["source"] for m, spec in plan.get("modules", {}).items()}
        self._pending_maps[txn] = {m: r["staged"].line_map for m, r in res.items()
                                   if r["staged"] is not None}
        self._state[txn] = "prepared"
        return F.ok(staged=items, findings=out_fs, timings=timings, targets=targets)

    def commit(self, txn):
        """Apply the staged transaction at the quiesce point. With a deadline (`txn` is
        {"txn", "deadline_at"}), a commit that would start after deadline_at does nothing:
        the host already counted this participant as failed and undid the others.

        Two-phase decision (audit #8): the job makes the patch visible, then, before it
        returns from the quiesce point (no step has run the patch yet), it decides under a
        lock whether the commit stands. It stands only while the caller still waits for it
        and the deadline has not passed; otherwise the job undoes its own entries and the
        transaction ends aborted (or undone, when the host's undo already came). The caller
        that gives up (the quiesce call raised) marks the job gone under the same lock: if
        the job had already decided, the reply says what it decided, so the host and this
        participant never disagree."""
        txn, deadline_at = txn_ref(txn)
        if txn not in self._staged:
            return F.not_ok(f"nothing staged for {txn}")
        staged, cells, accept = self._staged.pop(txn)
        self._state[txn] = "aborted"  # until the commit below succeeds
        job = self._inflight[txn] = {"done": threading.Event(), "gone": False,
                                     "started": False, "result": None}
        t_call = time.monotonic()

        def late(now=None):
            return deadline_at is not None and (now or time.time()) > deadline_at

        def do():
            t_start = time.monotonic()
            with self._lock:
                if job["gone"]:
                    return F.not_ok("commit cancelled before it ran")
                job["started"] = True
            try:
                r = body(t_start)
            finally:
                job["done"].set()
            return r

        def body(t_start):
            if late():
                return F.not_ok("commit started after the phase-2 deadline; nothing applied",
                                [{**F.finding("deadline_missed", "commit reached the "
                                              f"participant {time.time() - deadline_at:.3f} s "
                                              "after the deadline", "commit by the deadline"),
                                  "participant": self.name}])
            mark = self.journal.mark(txn)
            for st in staged:
                r = S.apply(st, self.journal, self.witness)
                if not r["ok"]:
                    self._note_added(mark)
                    u = self.journal.undo_to(mark)
                    self._rolled_back(txn, u, [])
                    return {**r, "findings": r.get("findings", []) + u["findings"]}
            reinit_fs, reinit_done = self._apply_reinit(txn)
            if reinit_fs:
                self._note_added(mark)
                u = self.journal.undo_to(mark)
                self._rolled_back(txn, u, [])
                return F.not_ok("commit refused: constructor_edit", reinit_fs + u["findings"])
            t_applied = time.monotonic()
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
                    self._note_added(mark)
                    u = self.journal.undo_to(mark)
                    self._rolled_back(txn, u, fs + left)
                    return F.not_ok("repair failed", [F.finding(
                        "repair_failed", f"{type(e).__name__}: {e}", "the cell runs")]
                        + left + u["findings"], rebuild_routes=routes())
                fs += rep.findings
            d = F.decide([{**f, "participant": self.name} for f in fs], accept)
            fs = d["findings"]
            rebuild_routes = routes()
            self._note_added(mark)
            if not d["proceed"]:
                u = self.journal.undo_to(mark)
                self._rolled_back(txn, u, fs)
                return F.not_ok(f"commit refused: {d['blocking']}", fs + u["findings"],
                                rebuild_routes=rebuild_routes)
            # the decision: still wanted (caller waiting, deadline not passed)? Under the
            # lock, so the caller's give-up and this decision are ordered.
            with self._lock:
                abandon = ("the caller gave up (quiesce timed out) while the commit applied"
                           if job["gone"] else
                           "the commit finished after the phase-2 deadline" if late() else None)
                if abandon is None:
                    self._marks[txn] = mark
                    self._seq[txn] = self._next_seq
                    self._next_seq += 1
                    self._state[txn] = "committed"
            if abandon is not None:
                u = self.journal.undo_to(mark)
                self._rolled_back(txn, u, [])
                with self._lock:
                    if self._state.get(txn) not in ("undone", "partial"):
                        self._state[txn] = "aborted"
                return F.not_ok(f"commit abandoned: {abandon}; rolled back here", [{
                    **F.finding("deadline_missed", abandon, "commit decided while the host "
                                "waits for it"), "participant": self.name}] + u["findings"],
                    abandoned=True)
            self._done[txn] = staged
            self._sources[txn] = self._pending.pop(txn, {})
            self._maps[txn] = self._pending_maps.pop(txn, {})
            # own code: what the swaps put in, and the code of every definition the edit added
            # (a later edit of an added function must not read as base_mismatch)
            self._codes[txn] = ([c for st in staged for _, c, _, _ in st.swaps]
                                + [c for st in staged for _, _, obj in st.adds
                                   for c in _codes_of(obj)])
            self._cell_keys[txn] = armed
            t_end = time.monotonic()
            return F.ok(changes=len(self.journal.entries) - mark, findings=fs,
                        journal=self.journal.describe(mark),
                        timings={"quiesce_wait_s": round(t_start - t_call, 6),
                                 "apply_s": round(t_applied - t_start, 6),
                                 "cells_s": round(t_end - t_applied, 6),
                                 "commit_s": round(t_end - t_start, 6),
                                 "committed_mono": t_end},
                        **({"rebuild_routes": rebuild_routes} if rebuild_routes else {}),
                        **({"reinit": reinit_done} if reinit_done else {}))

        try:
            r = self.quiesce(do)
            with self._lock:
                self._inflight.pop(txn, None)
            return r
        except Exception as e:
            with self._lock:
                job["gone"] = True
                committed = self._state.get(txn) == "committed"
                running = job["started"] and not job["done"].is_set()
                if not running:
                    self._inflight.pop(txn, None)
            if committed:
                # the job decided to commit before the caller gave up: say so
                return F.ok(note=f"committed; the quiesce call then raised {type(e).__name__}: "
                            f"{e}", late_reply=True)
            # not committed, and it never will be: a job still running rolls itself back
            # before it leaves the quiesce point (see above); undo waits for it
            return F.not_ok(f"commit: {type(e).__name__}: {e}", in_flight=running)

    # -- constructor field reinit (reinit.py) ---------------------------------------------
    def _resolve_reinit(self, m, f, reinits):
        """Prepare: a constructor_edit with a reinit plan becomes an advisory
        constructor_reinit when every live instance can take the new fresh values; else it
        stays a blocking constructor_edit that says why."""
        plan = f["reinit"]
        mod = sys.modules.get(self._actual(m))
        cls = mod
        for part in plan["cls"].split("."):
            cls = getattr(cls, part, None)
        if not isinstance(cls, type):
            return {**f, "observed": f["observed"] + f" (not re-initializable in place: "
                    f"{plan['cls']} is not a live class)"}
        r = RI.resolve(cls, plan["meth"], plan["fields"], vars(mod))
        if not r["ok"]:
            return {**f, "observed": f["observed"] + f" (not re-initializable in place: "
                    f"{r['why']})"}
        reinits.append({"key": f["subject"], "cls": cls, "meth": plan["meth"],
                        "fields": plan["fields"], "glb": vars(mod), "found": None,
                        "label": f"{plan['cls']}.{plan['meth']}"})
        what = ", ".join(f"{x['field']}: {x['old']} -> {x['new']}" for x in plan["fields"])
        nf = F.finding("constructor_reinit", f"{plan['cls']}.{plan['meth']} changes only fresh "
                       f"field values ({what}); {r['count']} live instance(s) hold the old "
                       f"fresh values and get the new ones in the commit's gap "
                       f"(heap scan {r['scan_s']} s)",
                       "every live instance re-initialized, or none live",
                       subject=f["subject"])
        return {**nf, "advisory": True, "reinit": {**plan, "instances": r["count"],
                                                   "scan_s": r["scan_s"]}}

    def _recheck_reinit(self, entries):
        """Validate (in the gap): scan again, so an instance built since prepare is covered
        and an instance whose field took state since is refused."""
        out = []
        for e in entries:
            r = RI.resolve(e["cls"], e["meth"], e["fields"], e["glb"])
            e["found"] = r["instances"] if r["ok"] else None
            e["scan_s"] = r["scan_s"]
            if not r["ok"]:
                out.append({**F.finding(
                    "constructor_edit", f"{e['label']}: re-initialization in place refused at "
                    f"the gap: {r['why']}", "every live instance holds the old fresh value",
                    subject=e["key"]), "route": "T1", "participant": self.name})
        return out

    def _apply_reinit(self, txn):
        """Commit, after the swaps (journaled). Returns (blocking findings, summary)."""
        entries = self._reinit.pop(txn, ())
        if entries and any(e["found"] is None for e in entries):
            fs = self._recheck_reinit(entries)     # a commit without a validate first
            if fs:
                return fs, None
        done = {}
        for e in entries:
            n = RI.apply(self.journal, e["found"], e["fields"], e["glb"], e["label"])
            for _ in range(n):
                self.witness.hit(e["key"])
            done[e["key"]] = {"instances": n, "fields": [x["field"] for x in e["fields"]],
                              "scan_s": e.get("scan_s")}
        return [], done

    def _rolled_back(self, txn, u, cell_findings):
        """A commit rolled back (journal.undo_to its mark). When that could not restore
        everything (an inverse failed, or a repair cell changed what the journal cannot
        undo: repair_not_undoable, accepted or not), the transaction is "partial", as after
        a partial undo: at_base is poisoned and undo repeats the findings."""
        lost = [{**f, "participant": self.name} for f in cell_findings
                if f["check"] == "repair_not_undoable"]
        if not u["ok"]:
            lost += [{**f, "participant": self.name} for f in u.get("findings", [])]
        if lost:
            self._state[txn] = "partial"
            self._partial[txn] = lost

    def abort(self, txn):
        txn, _ = txn_ref(txn)
        self._pending.pop(txn, None)
        self._reinit.pop(txn, None)
        self._pending_maps.pop(txn, None)
        if self._staged.pop(txn, None) is not None or self._state.get(txn) == "prepared":
            self._state[txn] = "aborted"
        return F.ok()

    def undo(self, txn):
        txn, deadline_at = txn_ref(txn)
        job = self._inflight.get(txn)
        if job is not None and not job["done"].is_set():
            # a commit whose caller gave up is still applying (audit #8): it rolls itself
            # back before it leaves the quiesce point; wait for that, so this reply is true
            wait = UNDO_WAIT_S if deadline_at is None else max(0.0, deadline_at - time.time())
            if not job["done"].wait(wait):
                return F.not_ok(f"undo: the commit of {txn} is still applying", [{
                    **F.finding("deadline_missed", f"the commit of {txn} was still applying "
                                f"{wait:.3f} s later (it rolls itself back when it ends)",
                                "an undo that can say what this participant runs"),
                    "participant": self.name}], in_flight=True)
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
            with self._lock:
                if txn in self._state:
                    self._staged.pop(txn, None)
                    self._reinit.pop(txn, None)
                    self._pending.pop(txn, None)
                    self._pending_maps.pop(txn, None)
                    self._state[txn] = "undone"
            return F.ok(undone=0, note="nothing committed")
        mark = self._marks[txn]
        gone = []                     # set if the quiesce call gave up before the job ran

        def job_undo():
            with self._lock:
                if gone:
                    return None               # cancelled: never runs unrecorded (audit #18)
                gone.append("started")
            # The journal undoes back to the mark, so it also undoes every transaction
            # committed after this one (by commit order: two may share a mark). Say so and
            # record their state: they must not stay "committed". The bookkeeping runs here,
            # in the job, so it happens even when the caller has given up.
            seq = self._seq.get(txn, -1)
            later = sorted((t for t, n in self._seq.items() if n > seq and t in self._marks),
                           key=lambda t: self._seq[t])
            r = self.journal.undo_to(mark)
            for t in [txn] + later:
                self._marks.pop(t, None)
                self._seq.pop(t, None)
                for d in (self._done, self._sources, self._codes, self._maps, self._cell_keys):
                    d.pop(t, None)
                self._state[t] = "undone" if r["ok"] else "partial"
            if not r["ok"]:
                for t in [txn] + later:      # which txn's entry failed, the journal does not say
                    self._partial[t] = r.get("findings", [])
            if later:
                r = {**r, "also_undone": later, "findings": r.get("findings", []) + [{
                    **F.finding("undo_cascade",
                                f"undoing {txn} also undid later transactions {later}",
                                f"only {txn} undone"), "participant": self.name}]}
            return r
        try:
            r = self.quiesce(job_undo)
        except Exception as e:
            with self._lock:
                started = bool(gone)
                if not started:
                    gone.append("cancelled")
            return F.not_ok(f"undo: {type(e).__name__}: {e}" + (
                "; the undo had started and finishes on its own (ask status)" if started else
                "; cancelled: nothing undone"), in_flight=started)
        return r if r is not None else F.not_ok("undo cancelled")

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
        state = self._state.get(txn, "unknown")
        if state != "committed":
            return F.not_ok(f"{txn} is {state} here", [{**F.finding(
                "no_witness_possible", f"transaction {txn} is {state} on this participant",
                "a transaction committed here", subject=txn), "participant": self.name}],
                counts={}, capture_counts={})
        r = self.witness.check(keys, old)
        # Triton: which kernel launched for each edited kernel, and whether a binary of the
        # new text exists (apply emptied the caches): a hook that fired with no compiled
        # kernel never reached a launch of the new binary, so it proves nothing (audit #9)
        launches = {}
        for st in self._done.get(txn, []):
            for pl, k in st.jits:
                per = self.witness.launches(k) if hasattr(self.witness, "launches") else {}
                launches[k] = per
                built = JIT.compiled(pl["affected"])
                if r["counts"].get(k) and not any(n and built.get(lab) for lab, n in per.items()):
                    r["findings"].append(F.finding(
                        "zero_witness", f"{r['counts'][k]} launch hook call(s), but no kernel "
                        f"that launched holds a binary compiled from the edited text {built}",
                        "a launch of a kernel compiled after the edit", subject=k))
        if launches:
            r["launches"] = launches
        if not keys:
            r["findings"].append(F.finding(
                "no_witness_possible", "the transaction swapped no code and its cells armed "
                "no counter: an empty witness proves nothing", "at least one witness key"))
        r["findings"] = [{**f, "participant": self.name} for f in r["findings"]]
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


def _codes_of(obj, depth=0):
    """Code objects an added object runs: a function, a static/class method's function, a
    property/cached_property, an added class's members, a decorator's __wrapped__, or a JIT
    wrapper's `fn` (Triton). Empty for anything else."""
    if depth > 4 or obj is None:
        return []
    if isinstance(obj, type):
        out = []
        for member in vars(obj).values():
            out += _codes_of(member, depth + 1)
        return out
    code = getattr(obj, "__code__", None)
    out = [code] if isinstance(code, types.CodeType) else []
    for attr in ("__func__", "__wrapped__", "fn", "fget", "fset", "fdel", "func"):
        inner = getattr(obj, attr, None)
        if inner is not None and inner is not obj:
            out += _codes_of(inner, depth + 1)
    return out


def _call(p, method, arg):
    try:
        r = getattr(p, method)(arg)
    except Exception as e:
        return F.not_ok(f"{method} crashed: {type(e).__name__}: {e}", [
            {**F.finding("participant_failed", f"{type(e).__name__}: {e}",
                         f"{method} returns an ok-dict"), "participant": p.name}])
    if not isinstance(r, dict) or "ok" not in r:
        return F.not_ok(f"{method}: bad reply {r!r}")
    return expect_ranks(p, method, r)


def expect_ranks(p, method, r):
    """A group participant that declares the ranks it must reach (`expected`: rank ids, e.g.
    every rank of every data-parallel group, from the engine's parallel config) and whose
    reply lacks some: the reply becomes not-ok with a blocking `participant_missing` naming
    them. At DP>1 a vLLM client returned only DP rank 0's replies, and witness, receipt and
    undo covered 1 of 4 groups with no word."""
    exp = getattr(p, "expected", None)
    if exp is None:
        return r
    got = {str(item["rank"]) for item in (r.get("ranks") or [r])
           if item.get("rank") is not None}
    missing = [str(expected_rank) for expected_rank in exp if str(expected_rank) not in got]
    if not missing or r.get("missing") == missing:
        return r
    f = {**F.finding("participant_missing", f"{method}: no reply from {', '.join(missing)} "
                     f"(replies from {', '.join(sorted(got)) or 'none'})",
                     f"a reply from every expected rank: {', '.join(map(str, exp))}",
                     subject=p.name), "participant": p.name, "missing": missing}
    return {**r, "ok": False, "missing": missing,
            "error": (f"{r.get('error')}; " if r.get("error") else "")
            + f"no reply from {missing}", "findings": list(r.get("findings") or []) + [f]}


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


def _rank_split(rank_hashes):
    """For ranks whose code hashes differ: ("ranks 0 vs 1,2: k1, ...", same_key) where
    same_key says whether a function key held by several ranks has different hashes (as
    opposed to only some ranks holding a key). Ranks are grouped by identical hashes."""
    groups = {}
    for r, h in rank_hashes.items():
        groups.setdefault(json.dumps(h, sort_keys=True), []).append(str(r))
    keys = set().union(*[set(h or {}) for h in rank_hashes.values()])
    differ, same_key = [], False
    for k in sorted(keys):
        vals = [json.dumps((h or {}).get(k), sort_keys=True) for h in rank_hashes.values()]
        held = [v for v, h in zip(vals, rank_hashes.values()) if k in (h or {})]
        if len(set(vals)) > 1:
            differ.append(k)
            same_key = same_key or len(set(held)) > 1
    split = " vs ".join(",".join(rs) for rs in sorted(groups.values(), key=len))
    return f"ranks {split}: {', '.join(differ)[:300]}", same_key


def _ranks_agree(rank_hashes):
    """Do the ranks of one group run the same code? None for a group of one (nothing to
    compare)."""
    if len(rank_hashes) < 2:
        return None
    return len({json.dumps(h, sort_keys=True) for h in rank_hashes.values()}) == 1


# Ranks of one group run the same steps and switch in the same gap, and a group reads all its
# counters in one call (for vLLM: one collective_rpc between the same two steps), so code that
# runs on every rank gives equal counts. A read that is not step-aligned can be off by one
# step's calls, which is under 1% once the workload ran 100 steps after the commit. The
# tolerance is 1% of the largest count, at least 1 call; a rank that missed part of the
# workload (or never ran the edit) is far outside it.
WITNESS_TOLERANCE = 0.01


def witness_uneven(name, ranks, tolerance=WITNESS_TOLERANCE):
    """Findings for the keys whose per-rank counts differ beyond the tolerance.
    ranks: {rank label: {key: count}}."""
    out = []
    keys = sorted({k for c in ranks.values() for k in c})
    for k in keys:
        by = {r: c.get(k, 0) for r, c in ranks.items()}
        hi, lo = max(by.values()), min(by.values())
        if hi - lo > max(1, math.ceil(tolerance * hi)):
            out.append({**F.finding(
                "witness_uneven", ", ".join(f"rank {r}: {n}" for r, n in by.items()),
                f"equal counts within {max(1, math.ceil(tolerance * hi))} call(s)",
                subject=k), "participant": name})
    return out


def merge_ranks(method, replies):
    """One reply for a group participant from its ranks' replies (each an ok-dict with its
    `rank`). Keeps every rank's reply in `ranks`; findings carry their rank. witness_check:
    `counts` is the count every rank reached (the minimum), the per-rank counts are in
    `ranks`. status: one state, or "mixed". fingerprint: `code` holds every rank's keys,
    each prefixed with its rank (`rank0/<key>`), since code object ids differ per process."""
    rs = [r if isinstance(r, dict) and "ok" in r else F.not_ok(f"bad reply {r!r}")
          for r in replies]
    for i, r in enumerate(rs):
        r.setdefault("rank", i)
    out = {"ok": all(r["ok"] for r in rs), "ranks": rs,
           "findings": [{**f, "rank": r["rank"]} for r in rs for f in r.get("findings", [])]}
    if not out["ok"]:
        out["error"] = "; ".join(f"rank {r['rank']}: {r.get('error')}" for r in rs
                                 if not r["ok"])
    if method == "status":
        states = {r.get("state") for r in rs}
        out["state"] = states.pop() if len(states) == 1 else "mixed"
    elif method == "witness_check":
        for f in ("counts", "capture_counts"):
            keys = {k for r in rs for k in r.get(f, {})}
            out[f] = {k: min(r.get(f, {}).get(k, 0) for r in rs) for k in keys}
    elif method == "proof":
        out["hashes"] = rs[0].get("hashes") if rs else {}
    elif method == "fingerprint":
        out["code"] = {f"rank{r['rank']}/{k}": v for r in rs for k, v in
                       (r.get("code") or {}).items()}
    return out


class GroupParticipant(Participant):
    """Several ranks as one participant: every call reaches each member in turn and the
    replies merge into one (`merge_ranks`), as one collective_rpc reaches every worker rank."""

    def __init__(self, name, members):
        self.name, self.members = name, list(members)

    def _all(self, method, arg):
        return merge_ranks(method, [{**_call(m, method, arg), "rank": i}
                                    for i, m in enumerate(self.members)])

    def prepare(self, plan):
        return self._all("prepare", plan)

    def validate(self, txn):
        return self._all("validate", txn)

    def commit(self, txn):
        return self._all("commit", txn)

    def abort(self, txn):
        return self._all("abort", txn)

    def undo(self, txn):
        return self._all("undo", txn)

    def status(self, txn):
        return self._all("status", txn)

    def proof(self, txn):
        return self._all("proof", txn)

    def witness_check(self, txn):
        return self._all("witness_check", txn)

    def reset(self, kind):
        return self._all("reset", kind)

    def at_base(self, _=None):
        return self._all("at_base", None)

    def fingerprint(self, modules):
        return self._all("fingerprint", modules)


class Lane:
    """One thread that runs the host's calls to one participant, in order. `call` waits until
    the deadline at most. A call that misses it keeps running; later calls queue behind it
    and are not waited for (so an undo runs after the late commit returns)."""

    def __init__(self, name):
        self._q = queue.SimpleQueue()
        self._lock = threading.Lock()
        self.pending, self.late = 0, False
        threading.Thread(target=self._run, daemon=True, name=f"fastrsi-lane-{name}").start()

    def _run(self):
        while True:
            job = self._q.get()
            job()
            with self._lock:
                self.pending -= 1
                if not self.pending:
                    self.late = False

    def call(self, fn, deadline_at):
        """(True, value) if fn finished by deadline_at; (False, None) if it did not, or if it
        only queued behind a late call."""
        box, done = {}, threading.Event()

        def job():
            try:
                box["value"] = fn()
            except BaseException as e:            # never kill the lane
                box["value"] = F.not_ok(f"{type(e).__name__}: {e}")
            done.set()
        with self._lock:
            behind = self.late
            self.pending += 1
        self._q.put(job)
        if behind:
            return False, None
        if done.wait(max(0.0, deadline_at - time.time())):
            return True, box["value"]
        with self._lock:
            if not done.is_set():
                self.late = True
        return (True, box["value"]) if done.is_set() else (False, None)


class Timeline(dict):
    """A receipt timeline: unix seconds (comparable across hosts, as before), and beside
    each stamp the monotonic clock (`mono`), which every span uses (audit #20: a wall-clock
    step made gap_s or undo_s wrong). A subclass that stamps `tl[k] = self.clock()` gets
    the monotonic stamp too."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.mono = {key: time.monotonic() for key in self}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.mono[key] = time.monotonic()

    def pop(self, key, *default):
        self.mono.pop(key, None)
        return super().pop(key, *default)

    def at(self, key, wall, mono):
        """Stamp key with a time taken earlier (wall, mono)."""
        super().__setitem__(key, wall)
        self.mono[key] = mono

    def span(self, a, b):
        if a in self.mono and b in self.mono:
            return round(self.mono[b] - self.mono[a], 6)
        return round(self[b] - self[a], 6) if a in self and b in self else None

    def relative(self, origin="edit"):
        """{stamp: monotonic seconds since origin} for the receipt."""
        o = self.mono.get(origin)
        return {k: round(v - o, 6) for k, v in self.mono.items()} if o is not None else {}


def _edit_stamp(edit_time, clock):
    """(wall, mono) of the edit: edit_time is unix seconds (when the agent made the edit);
    its monotonic time is now minus the wall clock's age of it. None: now (the start of
    run(); the receipt says so in `edit_source`)."""
    mono, wall = time.monotonic(), clock()
    if edit_time is None:
        return wall, mono
    return edit_time, mono - max(0.0, time.time() - edit_time)


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
    that had not started when the caller gave up is cancelled (aborted).

    phase2_deadline_s: the most phase 2 (and an undo) may hold the gap; None waits for every
    participant however long it takes (the gap then lasts as long as the slowest one). A
    participant that misses it poisons the host (`poison`): `run` is refused, `undo` is
    allowed, `recovered()` clears it after the lab respawned its participants."""

    def __init__(self, participants, writer=None, engine=None, interface_version="0",
                 clock=time.time, phase2_deadline_s=5.0):
        self.participants = list(participants)
        self.writer = writer
        self.engine = engine or {"name": "unknown", "version": None, "commit": None}
        self.interface_version = interface_version
        self.clock = clock
        self.phase2_deadline_s = phase2_deadline_s
        self.poison = []                   # deadline_missed findings; see recovered()
        self.txns = {}                     # txn id -> host record (see _record)
        self._lanes = {}
        self._lock = threading.Lock()
        self._in_rollback_gate = False

    def recovered(self):
        """After the lab respawned its participants: clear the poison, the lanes (a lane
        still stuck in a late call belonged to the old participant) and pending rollbacks."""
        with self._lock:
            self.poison, self._lanes = [], {}
            for txn_record in self.txns.values():
                txn_record["pending_rollback"] = []
        return F.ok()

    def poll_rollback(self):
        """Poll participants with unconfirmed phase-2 or undo rollbacks; return True while
        any late participant's queued undo has not yet confirmed state=='undone'."""
        if not self.poison:
            return False
        with self._lock:
            pending_items = [(txn_id, txn_record, list(txn_record.get("pending_rollback") or ()))
                             for txn_id, txn_record in self.txns.items()
                             if txn_record.get("pending_rollback")]
        still_pending = False
        for txn_id, txn_record, participants_to_check in pending_items:
            remaining = []
            for participant in participants_to_check:
                lane = self._lanes.get(participant.name)
                if lane is not None and lane.pending > 0:
                    remaining.append(participant)
                    continue
                status_reply = _call(participant, "status", txn_id)
                if status_reply.get("ok") and status_reply.get("state") == "undone":
                    continue
                remaining.append(participant)
            with self._lock:
                if not self.poison:
                    txn_record["pending_rollback"] = []
                    continue
                txn_record["pending_rollback"] = remaining
                if not remaining:
                    if "out" in txn_record and "rollback" in txn_record["out"]:
                        txn_record["out"]["rollback"]["status"] = "undone"
                    if txn_record.get("rec") and "rollback" in txn_record["rec"]:
                        txn_record["rec"]["rollback"]["status"] = "undone"
                else:
                    still_pending = True
        return still_pending

    def has_pending_rollback(self):
        """True while a phase-2 deadline miss left a participant's rollback unconfirmed."""
        if self._in_rollback_gate:
            with self._lock:
                return bool(self.poison) and any(
                    bool(txn_record.get("pending_rollback"))
                    for txn_record in self.txns.values()
                )
        return self.poll_rollback()

    def _bounded(self, p, method, arg, deadline_at, missed):
        """Call p with a deadline (unix seconds; None: no deadline). Phase-2 methods get
        {"txn", "deadline_at"}. A participant that does not answer in time is reported with a
        `deadline_missed` finding and added to `missed` (participant names)."""
        if deadline_at is None:
            return _call(p, method, arg)
        ref = {"txn": arg, "deadline_at": deadline_at} if method in PHASE2 else arg
        if method in ("validate", "abort") and p.name not in missed and \
                time.time() >= deadline_at:
            # The deadline passed before this participant's turn (audit #19: no re-armed
            # deadline). Not its fault, and nothing of it changes: validate is not asked (the
            # vote is lost, so the transaction aborts), and abort is sent without a wait
            # (dropping staged state is safe whenever it lands; no commit can follow).
            if method == "validate":
                return F.not_ok("validate not asked: the phase-2 deadline passed before this "
                                "participant's turn", not_asked=True)
            if getattr(p, "bounded", False):
                return _call(p, method, ref)
            lane = self._lanes.get(p.name) or self._lanes.setdefault(p.name, Lane(p.name))
            done, r = lane.call(lambda: _call(p, method, ref), deadline_at)
            return r if done else F.ok(queued=True, note="abort sent after the deadline; "
                                       "not waited for")
        if getattr(p, "bounded", False):
            r = _call(p, method, ref)
            late = any(f.get("check") == "deadline_missed" for f in r.get("findings", []))
        else:
            lane = self._lanes.get(p.name)
            if lane is None:
                lane = self._lanes[p.name] = Lane(p.name)
            done, r = lane.call(lambda: _call(p, method, ref), deadline_at)
            late = not done
            if late:
                why = ("queued behind a call that missed the deadline" if p.name in missed
                       else f"no answer {self.phase2_deadline_s} s into the gap")
                r = F.not_ok(f"{method}: {why}", [{**F.finding(
                    "deadline_missed", f"{method}: {why}", f"{method} answers by the deadline"),
                    "participant": p.name}], missed=True)
        if late and p.name not in missed:
            missed.add(p.name)
            self.poison.append({**F.finding(
                "deadline_missed", f"{p.name} missed the deadline in {method}",
                f"{method} answers within the phase-2 deadline"), "participant": p.name})
        return r

    def _in_gap(self, fn):
        """Run fn where phase 2 must happen. Here: right now, in this thread."""
        return fn()

    def _write(self, rec):
        return self.writer.append(rec) if self.writer else {"ok": False, "error": "no writer"}

    # -- host state ----------------------------------------------------------------------
    def status(self, txn=None):
        """{"ok", "txn", "state", "outcome", "written"}. Once phase 2 has finished, the first
        status call after a timed-out run writes the transaction's receipt. Without a txn:
        the host's own state ("idle", "in-doubt" while a phase 2 has not finished,
        "poisoned"), its last transactions (newest last, at most 10) and at_base."""
        if txn is None:
            return self._overall()
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
        rollback_pending = self.has_pending_rollback()
        return F.ok(txn=txn, state=state, outcome=t.get("outcome"), written=t["receipt"],
                    poisoned=bool(self.poison), rollback_pending=rollback_pending,
                    missed=sorted(t.get("missed", ())))

    def _overall(self):
        rollback_pending = self.has_pending_rollback()
        with self._lock:
            last = []
            for t, rec in list(self.txns.items())[-10:]:
                st = rec["state"]
                if st == "prepared" and rec["started"] and not rec["done"]:
                    st = "in-doubt"
                last.append({"txn": t, "state": st, "outcome": rec.get("outcome")})
        state = "poisoned" if self.poison else (
            "in-doubt" if any(x["state"] == "in-doubt" for x in last) else "idle")
        try:
            ab = self.at_base()
        except Exception as e:
            ab = F.not_ok(f"at_base: {type(e).__name__}: {e}")
        return F.ok(txn=None, state=state, txns=len(self.txns), last=last,
                    at_base=ab.get("ok", False), at_base_detail=ab,
                    poisoned=bool(self.poison), rollback_pending=rollback_pending)

    # -- the transaction -----------------------------------------------------------------
    def _phase2(self, txn, tl):
        """validate -> commit, or abort; undo the committed ones if a commit fails. Inside
        the gap. Returns the phase-2 result; never raises."""
        now, t = self.clock, self.txns[txn]
        res = t["res"]
        tl["gap_start"] = now()
        d = self.phase2_deadline_s
        # validate and commit by half the deadline, the rollback by all of it: the gap ends
        # within the deadline even when a commit misses its half
        # (deadlines are on the shared wall clock, which participants also read; a deadline
        # that has passed is not re-armed: later calls get no wait and count as missed)
        g = time.time()
        commit_at, end_at = (None, None) if d is None else (g + d / 2, g + d)
        missed = t["missed"] = set()

        def call(p, method, deadline_at):
            return self._bounded(p, method, txn, deadline_at, missed)
        for p in self.participants:
            validate_deadline = commit_at   # never re-armed once passed (audit #19)
            res[p.name]["validate"] = call(p, "validate", validate_deadline)
        tl["vote"] = now()
        rollback = {"status": "not_needed", "participants": {}}
        out = {"rollback": rollback, "extra": [], "hashes": {}}
        if not all(res[p.name]["validate"]["ok"] for p in self.participants):
            for p in self.participants:
                abort_deadline = end_at   # never re-armed once passed (audit #19)
                res[p.name]["abort"] = call(p, "abort", abort_deadline)
            out["outcome"] = "aborted"
        else:
            committed, failed = [], None
            for p in self.participants:
                r = res[p.name]["commit"] = call(p, "commit", commit_at)
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
                # The failed participant may have committed in part (a worker group whose
                # ranks commit in one call: some ranks did, one did not), so it is undone too;
                # undo is a no-op where nothing was committed.
                # A participant that missed the deadline gets its undo queued behind the late
                # commit (not waited for), so it runs if that commit lands after all.
                rollback_targets = [failed] + list(reversed(committed))
                for p in rollback_targets:
                    undo_deadline = end_at   # never re-armed once passed (audit #19)
                    rollback["participants"][p.name] = res[p.name]["undo"] = \
                        call(p, "undo", undo_deadline)
                for p in self.participants:
                    if p not in committed:
                        abort_deadline = end_at   # never re-armed once passed (audit #19)
                        res[p.name]["abort"] = call(p, "abort", abort_deadline)
                for name, u in rollback["participants"].items():
                    if any(x.get("in_flight") for x in (u.get("ranks") or [u])) and \
                            name not in missed:
                        # a commit still applying there: the host cannot say what it runs
                        missed.add(name)
                        self.poison.append({**F.finding(
                            "deadline_missed", f"{name}: a commit was still applying at undo",
                            "every commit decided within the gap"), "participant": name})
                rollback["status"] = ("undone" if all(
                    u["ok"] for u in rollback["participants"].values()) else "partial")
                if rollback["status"] != "undone":
                    with self._lock:
                        t["pending_rollback"] = [
                            participant for participant in rollback_targets
                            if not rollback["participants"][participant.name]["ok"]
                        ]
        tl["gap_end"] = now()
        return out

    def run(self, plan, edit_time=None, baseline_restart=None):
        """One transaction; appends a `txn` receipt. edit_time: when the edit was made (unix
        seconds; default now). baseline_restart: {"seconds", "source"} of the warm restart
        this replaces. Returns {"ok", "outcome", "txn", "receipt", "written"}, where outcome is
        committed, aborted, rolled_back, or in-doubt (resolve it with `status`)."""
        now = self.clock
        tl = Timeline()
        tl.at("edit", *_edit_stamp(edit_time, now))
        txn = plan.get("id") or f"txn-{uuid.uuid4().hex[:8]}"
        plan = {**plan, "id": txn}
        if self.poison:              # a late participant may still switch: undo or respawn
            return {"ok": False, "outcome": "refused", "txn": txn, "receipt": None,
                    "written": {"ok": False, "error": "host poisoned: no receipt"},
                    "error": "host poisoned: undo or respawn first",
                    "findings": [F.finding("session_poisoned", "; ".join(
                        f"{f['participant']}: {f['check']}" for f in self.poison),
                        "a host whose participants all met their deadlines")]}
        t = self.txns[txn] = {"plan": plan, "edit": tl["edit"], "edit_mono": tl.mono["edit"],
                              "edit_source": "run_start" if edit_time is None else "caller",
                              "tl": tl, "state": "prepared",
                              "res": {p.name: {} for p in self.participants},
                              "started": False, "cancelled": False, "done": False,
                              "receipt": False, "baseline": baseline_restart,
                              "fin": threading.Lock()}
        # phase 1: outside the gap, steps go on
        tl["prepare_start"] = now()
        for p in self.participants:
            t["res"][p.name]["prepare"] = _call(p, "prepare", plan)
        tl["prepared"] = now()
        if not all(t["res"][p.name]["prepare"]["ok"] for p in self.participants):
            for p in self.participants:
                t["res"][p.name]["abort"] = _call(p, "abort", txn)
            self._done(txn, {"outcome": "aborted", "hashes": {}, "extra": [],
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
        ok = t["outcome"] == "committed"
        # One finisher at a time. `receipt` turns True only once the append succeeded, so a
        # failed append is retried by the next status(txn), and nobody reads a missing `rec`.
        if not t["fin"].acquire(blocking=False):
            return {"ok": ok, "outcome": t["outcome"], "txn": txn, "receipt": None,
                    "written": {"ok": False, "error": "receipt being written: ask status(txn)"}}
        try:
            with self._lock:
                if t["receipt"]:
                    return {"ok": ok, "outcome": t["outcome"], "txn": txn, "receipt": t["rec"],
                            "written": {"ok": True, "note": "already written"}}
                rec = t.get("rec")
            if rec is None:
                rec = self._build(txn)
                with self._lock:
                    t["rec"] = rec
            w = self._write(rec)
            if w.get("ok"):
                with self._lock:
                    t["receipt"] = True
            return {"ok": ok, "outcome": t["outcome"], "txn": txn, "receipt": rec, "written": w}
        finally:
            t["fin"].release()

    def _build(self, txn):
        """The `txn` receipt record; reads the code hashes of a committed transaction."""
        t = self.txns[txn]
        plan, tl, out, res = t["plan"], t["tl"], t["out"], t["res"]
        rank_hashes, ranks_agree = {}, {}
        if out["outcome"] == "committed":       # code hashes, read after the gap
            for p in self.participants:
                # kept in the receipt: its findings (participant_missing, participant_failed)
                # count like those of any other call
                r = res[p.name]["proof"] = _call(p, "proof", txn)
                out["hashes"][p.name] = r.get("hashes")
                if r.get("ranks"):              # a group: every rank's hashes, compared
                    rank_hashes[p.name] = {str(x.get("rank")): x.get("hashes")
                                           for x in r["ranks"]}
                    ranks_agree[p.name] = _ranks_agree(rank_hashes[p.name])
                    if ranks_agree[p.name] is False:     # never only a flag: a finding
                        split, same_key = _rank_split(rank_hashes[p.name])
                        out["extra"].append({**F.finding(
                            "ranks_disagree", split,
                            "every rank of the group runs the same code", subject=p.name),
                            "participant": p.name,
                            # the same function differs: no accept can make that safe
                            **({"overridable": False} if same_key else {})})
                if not r["ok"]:        # a failed proof, or one lacking ranks, shows no agreement
                    ranks_agree[p.name] = False
                    if not r.get("findings"):
                        out["extra"].append({**F.finding(
                            "participant_failed", f"proof: {r.get('error')}",
                            "proof answers with every rank's code hashes"),
                            "participant": p.name})
        fs = [f for r in res.values() for x in r.values()
              for f in x.get("findings", [])] + out["extra"]
        modules = plan.get("modules", {})
        # agree: False if any comparison differs (across participants on a shared target, or
        # across the ranks of one group); None when nothing could be compared
        checks = [a for a in [_hashes_agree(out["hashes"])] + list(ranks_agree.values())
                  if a is not None]
        agree = all(checks) if checks else None

        def span(a, b):
            if isinstance(tl, Timeline):
                return tl.span(a, b)
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
            "plan": {"targets": {m: s.get("targets", "auto") for m, s in modules.items()},
                     "predicted_resources": plan.get("predicted_resources", []),
                     "recipes": plan.get("recipes", []), "certainty": plan.get("certainty"),
                     "accept": plan.get("accept", []),
                     "undo_resets": plan.get("undo_resets", [])},
            "actual_recipes": ((["R0 swap"] if modules else [])
                               + (["R10 repair"] if plan.get("repairs") else []))
            if out["outcome"] == "committed" else [],
            "participants": res,
            "commit_hashes": {"participants": out["hashes"], "ranks": rank_hashes,
                              "ranks_agree": ranks_agree, "agree": agree},
            "findings": fs,
            "decisions": {"accepted": sorted({f["check"] for f in fs if f.get("accepted")}),
                          "blocking": sorted({f["check"] for f in fs
                                              if not f.get("accepted")})},
            "outcome": out["outcome"], "timeline": dict(tl),
            "timeline_mono": tl.relative() if isinstance(tl, Timeline) else {},
            "edit_source": t.get("edit_source"),
            # the first execution of the edit is seen only after the commit, by the witness:
            # the `proof` receipt has it (audit #20); here nothing has run yet
            "edit_to_run_s": None,
            "edit_to_commit_s": span("edit", "committed"),
            "edit_to_rebuilt_s": span("edit", "rebuilt") if t.get("rebuilt") else None,
            "rebuild": out.get("rebuild"),
            "recapture_s": ((out.get("rebuild") or {}).get("seconds") or {}).get("R1"),
            "prepare_s": span("prepare_start", "prepared"),
            "gap_s": span("gap_start", "gap_end"),
            "baseline_restart_s": t["baseline"],
            # the phase-2 deadline and who missed it (they poison the host)
            "rollback": {**out["rollback"], "deadline_s": self.phase2_deadline_s,
                         "missed_deadline": sorted(t.get("missed", ())),
                         "poisoned": bool(t.get("missed"))},
        }
        return rec

    def prove(self, txn, golden=None, tolerance=WITNESS_TOLERANCE):
        """After the workload ran: collect witness counts and findings from every participant;
        append a `proof` receipt. golden: the caller's golden-check results, recorded as is.
        Witness counts are kept per rank: a group participant's ranks appear as
        `<participant>/rank<r>`, and ranks whose counts differ beyond `tolerance` (a fraction
        of the largest count, at least 1 call; see WITNESS_TOLERANCE) are `witness_uneven`."""
        now = self.clock
        res = {p.name: _call(p, "witness_check", txn) for p in self.participants}
        fs = [{**f, "participant": n} for n, r in res.items() for f in r.get("findings", [])]
        witness, capture = {}, {}
        for n, r in res.items():
            if r.get("ranks"):
                ranks = {str(x.get("rank")): x.get("counts", {}) for x in r["ranks"]}
                witness.update({f"{n}/rank{k}": c for k, c in ranks.items()})
                capture.update({f"{n}/rank{x.get('rank')}": x.get("capture_counts", {})
                                for x in r["ranks"]})
                fs += witness_uneven(n, ranks, tolerance)
            else:
                witness[n] = r.get("counts", {})
                capture[n] = r.get("capture_counts", {})
        edit = self.txns.get(txn, {}).get("edit")
        edit_mono = self.txns.get(txn, {}).get("edit_mono")
        t = now()
        t_mono = time.monotonic()
        # first execution: per participant (per rank) the first live witness hit of the edit
        # (CLOCK_MONOTONIC, shared by the processes of one host); the edit has run everywhere
        # at the latest of them (audit #20). None when some participant shows no hit.
        firsts = {}
        for n, r in res.items():
            for x in (r.get("ranks") or [r]):
                label = f"{n}/rank{x.get('rank')}" if r.get("ranks") else n
                hits = list((x.get("first_hit") or {}).values())
                firsts[label] = min(hits) if hits else None
        run_at = (max(firsts.values()) if firsts and all(v is not None for v in firsts.values())
                  else None)
        rec = {"schema": R.SCHEMA, "kind": "proof", "id": txn,
               "witness": witness,
               # calls made while graphs were captured, apart from the live ones above: a
               # replay runs no Python, so only live calls (or a device probe) prove a run
               "witness_capture": capture,
               "ran": {w: {k: W.ran(capture.get(w, {}).get(k, 0), n) for k, n in c.items()}
                       for w, c in witness.items()},
               "findings": fs, "golden": golden,
               "launches": {n: r.get("launches") for n, r in res.items() if r.get("launches")},
               "timeline": {"edit": edit, "proved": t},
               "first_run_s": {k: (round(v - edit_mono, 6) if v is not None and
                                   edit_mono is not None else None) for k, v in firsts.items()},
               "edit_to_run_s": (round(run_at - edit_mono, 6)
                                 if run_at is not None and edit_mono is not None else None),
               "edit_to_finish_s": (round(t_mono - edit_mono, 6) if edit_mono is not None
                                    else round(t - edit, 6) if edit is not None else None)}
        return {"ok": all(r["ok"] for r in res.values()) and not fs, "receipt": rec,
                "written": self._write(rec)}

    def at_base(self):
        """Every participant (every rank of a group) back at its start state? Cheap: no heap
        scan, only the modules a transaction touched (LocalParticipant.at_base). Returns
        {ok, participants: {name: reply}, poisoned, missing}."""
        res = {p.name: _call(p, "at_base", None) for p in self.participants}
        ok = all(r["ok"] for r in res.values()) and not self.poison
        return F.ok(**{"ok": ok}, participants=res, poisoned=bool(self.poison),
                    findings=[f for r in res.values() for f in r.get("findings", [])])

    def _undo_phases(self, txn, resets, tl):
        res = {p.name: {} for p in self.participants}
        d, missed = self.phase2_deadline_s, set()
        end_at = None if d is None else time.time() + d
        for p in reversed(self.participants):
            undo_deadline = end_at   # never re-armed once passed (audit #19)
            res[p.name]["undo"] = self._bounded(p, "undo", txn, undo_deadline, missed)
        tl["code_undone"] = self.clock()
        for kind in resets:
            for p in self.participants:
                reset_deadline = end_at   # never re-armed once passed (audit #19)
                res[p.name].setdefault("reset", {})[kind] = self._bounded(
                    p, "reset", kind, reset_deadline, missed)
        txn_record = self.txns.get(txn)
        if txn_record is not None and missed:
            with self._lock:
                txn_record["pending_rollback"] = [
                    participant for participant in reversed(self.participants)
                    if not res[participant.name]["undo"]["ok"]
                ]
        return res

    def undo(self, txn):
        """Undo a committed transaction on every participant (newest first), then run the
        plan's `undo_resets` (R2, R3, ...) on each: code undo cannot un-write data the new code
        produced. In the gap. Appends an `undo` receipt (its `rebuild` stays None in v1)."""
        now = self.clock
        t = self.txns.get(txn, {})
        with self._lock:
            state = t.get("state") if t else None
        # refuse what cannot be undone (audit #18): an unknown id, an undone one, one
        # that never committed. No receipt: nothing was asked of any participant.
        why = (f"unknown transaction {txn}" if not t else
               f"{txn} is already undone" if state == "undone" else
               f"{txn} did not commit ({t.get('outcome')})" if state == "aborted" and
               not t.get("pending_rollback") else None)
        if why:
            return {"ok": False, "status": "refused", "error": why, "participants": {},
                    "receipt": None, "written": {"ok": False, "error": "refused: no receipt"}}
        resets = t.get("plan", {}).get("undo_resets", [])
        tl = Timeline(queued=now())
        rb = {}
        undo_state = {"started": False, "cancelled": False, "done": False}

        def undo_job():
            with self._lock:
                if undo_state["cancelled"]:
                    return None
                undo_state["started"] = True
            phase_res = self._undo_phases(txn, resets, tl)
            with self._lock:
                undo_state["done"] = True
            return phase_res

        try:
            res, err = self._in_gap(undo_job), None
        except Exception as e:
            with self._lock:
                if not undo_state["started"]:
                    undo_state["cancelled"] = True
            res, err = None, f"{type(e).__name__}: {e}"
            if undo_state["started"] and not undo_state["done"]:
                self.poison.append({**F.finding(
                    "deadline_missed", f"undo of {txn} started in the gap but did not finish: {err}",
                    "undo finishes within the gap timeout"), "participant": "host"})
                if t:
                    with self._lock:
                        t["pending_rollback"] = list(reversed(self.participants))
        tl["undo_end"] = now()
        res = res or {}
        calls = [x for r in res.values()
                 for x in [r.get("undo")] + list(r.get("reset", {}).values()) if x]
        fs = [{**f, "participant": n} for n, r in res.items()
              for x in [r.get("undo")] + list(r.get("reset", {}).values()) if x
              for f in x.get("findings", [])]
        ok = err is None and bool(calls) and all(x["ok"] for x in calls) and rb.get("ok", True)
        status_label = ("undone" if ok else
                        "in-doubt" if (undo_state["started"] and not undo_state["done"]) else
                        "cancelled" if undo_state["cancelled"] else "partial")
        rec = {"schema": R.SCHEMA, "kind": "undo", "id": txn, "participants": res,
               "resets": list(resets), "status": "undone" if ok else "partial",
               "findings": fs + ([F.finding("undo_incomplete", f"undo did not finish: {err}",
                                            "a step gap within the timeout")] if err else []),
               "rebuild": rb or None,
               "recapture_s": (rb.get("seconds") or {}).get("R1"),
               "timeline": dict(tl), "undo_s": tl.span("queued", "undo_end")}
        if t and rb.get("ok"):
            t["rebuilt"] = []
        if t and not undo_state["cancelled"]:
            # partial: an undo ran (or started) but could not restore everything; a cancelled
            # undo never ran in the gap so the transaction's prior state stays intact.
            with self._lock:
                t["state"] = "undone" if ok else (
                    "in-doubt" if status_label == "in-doubt" else "partial"
                )
                # transactions the participants undid with it (journal cascade): the host
                # says so too, never "committed" for code that is gone
                for x in calls:
                    for r in (x.get("ranks") or [x]):
                        for later in r.get("also_undone") or ():
                            if later in self.txns and ok:
                                self.txns[later]["state"] = "undone"
        return {"ok": ok, "status": status_label, "participants": res, "receipt": rec,
                "written": self._write(rec)}


class StepGapCoordinator(Coordinator):
    """The host for participants that step together (worker ranks + engine core), living in
    the loop owner: phase 1 (prepare) runs in the caller's thread while steps go on; phase 2
    (validate, commit or abort, undo of a failed commit) runs in ONE gap between two steps.
    So while every participant answers by the phase-2 deadline, no step sees participants on
    different versions. One that misses it poisons the host and the gap ends anyway: until the
    lab respawns, steps may run mixed (a late commit lands before its queued undo runs).
    The loop stalls only for
    phase 2 (`gap_s` in the receipt). `gap(fn)` queues fn to the loop owner and returns its
    result (for vLLM: the engine-core busy loop, which calls the workers by collective_rpc;
    that completes on every worker before the next step). Participants must not quiesce on
    their own: build them with the default `quiesce`."""

    def __init__(self, participants, gap, **kw):
        super().__init__(participants, **kw)
        self.gap = gap

    def _in_gap(self, fn):
        return self.gap(fn)
