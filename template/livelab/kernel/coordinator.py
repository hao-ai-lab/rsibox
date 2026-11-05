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

Rebuild after the commit (a route such as R1: make the engine's captured or compiled code run
the edit). The host registers `rebuilds={"R1": fn(txn, why, deadline_at) -> ok-dict}` (why:
"commit", "rollback", "undo"; the engine picks how, e.g. vLLM recompiles under torch.compile and
recaptures in pure CUDA-graph modes, and says which in the reply). A transaction rebuilds when
its plan says so (`"rebuild": ["R1"]`; `[]`: never) or, with no `rebuild` key, when prepare
returned an accepted, non-advisory finding whose `route` the host can rebuild (vLLM: an
accepted `captured_target`). The rebuild is a post-commit phase of the same transaction: it
runs right after phase 2, outside the gap and not under the phase-2 deadline (a recompile takes
tens of seconds), under its own deadline (`rebuild_deadline_s`), as one job of the loop owner
(`rebuild_job(fn)`; StepGapCoordinator: the host's loop, so the engine is paused while it
runs). Steps that start between the commit and the rebuild run the new Python code on eager
paths and the old graphs elsewhere; the receipt says how many (`steps(a, b)`, a host count of
steps started between two unix times) and what the engine did meanwhile (`rebuild_while`). If
the rebuild fails, the transaction is undone (in a gap) and every route is rebuilt again from
the base code (`rebuild_failed`, outcome rolled_back). The txn receipt has `rebuild` (routes,
results, seconds, steps) and `recapture_s` (the R1 seconds); `edit_to_run_s` then ends when
the rebuild did (`edit_to_commit_s`: at the commit). Undo of a transaction that rebuilt (or of
one whose undo cascades to one) rebuilds again after the code undo.

A plan is plain data, so it can cross a process boundary:

    {"id": str (optional),
     "modules": {module: {"source": patched text, "base_hash": text_hash of the base,
                          "targets": [[qualname, first line in the base]] or "auto"
                                     (default: the definitions that changed, found by stage),
                          "scan": "auto" | "always", "compile": "definitions" | "module",
                          "trees": (base AST, patch AST)}},  in-process only (not plain
                                     data): the caller's parse of the text the participant
                                     runs and of `source`, so stage does not parse again
     "repairs": [{"cell": python source, "declares": [names in the participant's scope]}],
                 a cell runs at commit with the scope's names and `journal` (this
                 participant's): `journal.record_attr(obj, name)` before assigning what the
                 declared-object audit cannot undo (an nn.Module submodule, Parameter.data),
                 `journal.record_call(undo_fn, label)` for an in-place re-init
     "accept": [check names] or "all",
     "rebuild": ["R1", ...],   routes to rebuild right after the commit (default: from the
                               accepted findings' routes; [] turns that off)
     "undo_resets": ["R2" | "R3" | ...],   resets the undo runs after the code undo
     "recipes": [...], "predicted_resources": [...], "certainty": str}

`LocalParticipant` runs the kernel in this process. `CallParticipant` is the shape the engine
side implements: one `call(method, payload) -> dict` per participant (for vLLM: collective_rpc
to a worker rank, or the engine-core input-queue channel), which on the far side dispatches
to a LocalParticipant.
"""
import json
import math
import queue
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


def txn_ref(ref):
    """(txn id, deadline_at or None) from a phase-2 argument: an id or {"txn", "deadline_at"}."""
    if isinstance(ref, dict):
        return ref.get("txn"), ref.get("deadline_at")
    return ref, None


PHASE2 = ("validate", "commit", "abort", "undo")
REBUILD_RECIPES = {"R1": "R1 recapture"}      # receipt names of the rebuild routes


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


class LocalParticipant(Participant):
    """The kernel in this process. `alias`: {plan module prefix: this process's prefix} (a
    participant that loaded the package under another name). `quiesce(fn)`: runs fn at this
    participant's step boundary and returns its result (default: now, in the caller's thread).
    `scope`: names a repair cell can use. `resets`: {"R3": callable, ...}, data resets an
    undo can ask for (each returns a JSON-able summary or None). `analyzer`: where `stage`
    parses, compiles and compares the texts (`swap.ProcessAnalyzer()`: a child process, so
    prepare holds this process's GIL only briefly; default: here). The participant starts the
    analyzer when it is created (`start()`, idempotent), so the child boots before the first
    prepare instead of inside it; a child that dies later is respawned by the next prepare,
    which says so (`timings[module]["analyze_spawned"]`, and `analyze_fallback` if it fails).
    A plan module with `"analyze": "inline"` is analyzed here anyway (for a before/after).

    Chained edits: the participant remembers the source text each committed transaction
    installed and the code objects it swapped in. A later plan for the same module is staged
    against that text (not the file), and the participant's own code objects are not
    `base_mismatch`. Undo of a transaction also forgets every later one (the journal undoes
    them too). A plan that would change nothing is an `empty_change` finding.

    Several modules in one plan are staged so that each sees the module-level names the
    others' patches add (`swap.staged_names` as stage's `overlay`), in any plan order; nothing
    is visible before commit. A plan module without `targets` lets stage find them; the reply's
    `targets` says which it staged."""

    def __init__(self, name, alias=None, quiesce=None, scope=None, resets=None, analyzer=None,
                 checks=None, capturing=None):
        self.name = name
        self.checks = list(checks or ())   # engine checks: fn(module, items) -> [findings]
        self.analyzer = analyzer      # swap.ProcessAnalyzer: parse and compare off this process
        self.analyzer_start = None    # why the analyzer did not start here (then: at first use)
        if hasattr(analyzer, "start"):
            # warm start: the child boots now, not inside the first prepare (idempotent; a
            # child that dies later is respawned lazily, by the next prepare)
            try:
                analyzer.start()
            except Exception as e:
                self.analyzer_start = f"{type(e).__name__}: {e}"
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

    def _stage(self, m, spec, own, overlay):
        return S.stage(self._actual(m), spec["source"], spec.get("targets", "auto"),
                       spec.get("base_hash"), key_module=m, scan=spec.get("scan", "auto"),
                       base_source=self._live_source(m), own_codes=own,
                       line_maps=self._line_maps(m),
                       compile=spec.get("compile", "definitions"),
                       analyzer=None if spec.get("analyze") == "inline" else self.analyzer,
                       overlay=overlay, trees=spec.get("trees"))

    def prepare(self, plan):
        txn, accept = plan["id"], plan.get("accept", ())
        fs, staged, items, ok, timings, targets = [], [], {}, True, {}, {}
        own = [c for cs in self._codes.values() for c in cs]
        # The modules of one transaction see each other's staged names (`overlay`): module B's
        # `from a import new` finds what A's patch adds, though A is applied only at commit.
        # A module that fails is staged again once more modules have staged, until no more
        # succeed, so the plan's module order does not matter.
        res, overlay, todo = {}, {}, list(plan.get("modules", {}))
        while todo:
            again = []
            for m in todo:
                r = res[m] = self._stage(m, plan["modules"][m], own, overlay)
                if r["staged"] is not None:
                    overlay.update(S.staged_names(r["staged"]))
                elif len(todo) > 1:
                    again.append(m)
            if len(again) == len(todo):
                break
            todo = again
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
            fs.append(F.finding("empty_change", f"{len(items)} target(s), none changed; no "
                                "definition added or deleted; no repair cell",
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
            for code, declares in cells:
                decl = {n: self.scope[n] for n in declares}
                audit = {n: o for n, o in self.scope.items() if n not in decl}
                try:
                    with self.journal.repair(decl, audit) as rep:
                        exec(code, {**self.scope, "journal": self.journal})
                except Exception as e:
                    u = self.journal.undo_to(mark)
                    return F.not_ok("repair failed", [F.finding(
                        "repair_failed", f"{type(e).__name__}: {e}", "the cell runs")]
                        + u["findings"])
                fs += rep.findings
            d = F.decide([{**f, "participant": self.name} for f in fs], accept)
            fs = d["findings"]
            if not d["proceed"]:
                u = self.journal.undo_to(mark)
                return F.not_ok(f"commit refused: {d['blocking']}", fs + u["findings"])
            self._marks[txn] = mark
            self._done[txn] = staged
            self._sources[txn] = self._pending.pop(txn, {})
            self._maps[txn] = self._pending_maps.pop(txn, {})
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
            for d in (self._done, self._sources, self._codes, self._maps):
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
        hashes, keys = {}, []
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
    `ranks`. status: one state, or "mixed"."""
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
                 clock=time.time, phase2_deadline_s=5.0, rebuilds=None,
                 rebuild_deadline_s=600.0, rebuild_job=None, rebuild_while=None, steps=None):
        self.participants = list(participants)
        # post-commit rebuilds (module doc): route -> fn(txn, why, deadline_at) -> ok-dict
        self.rebuilds = dict(rebuilds or {})
        self.rebuild_deadline_s = rebuild_deadline_s
        self.rebuild_job = rebuild_job or (lambda fn: fn())  # where a rebuild runs
        self.rebuild_while = rebuild_while or "the caller's thread runs the rebuild"
        self.steps = steps            # fn(t0, t1) -> steps started between two unix times
        self.writer = writer
        self.engine = engine or {"name": "unknown", "version": None, "commit": None}
        self.interface_version = interface_version
        self.clock = clock
        self.phase2_deadline_s = phase2_deadline_s
        self.poison = []                   # deadline_missed findings; see recovered()
        self.txns = {}                     # txn id -> host record (see _record)
        self._lanes = {}
        self._lock = threading.Lock()

    def recovered(self):
        """After the lab respawned its participants: clear the poison and the lanes (a lane
        still stuck in a late call belonged to the old participant)."""
        self.poison, self._lanes = [], {}
        return F.ok()

    def _bounded(self, p, method, arg, deadline_at, missed):
        """Call p with a deadline (unix seconds; None: no deadline). Phase-2 methods get
        {"txn", "deadline_at"}. A participant that does not answer in time is reported with a
        `deadline_missed` finding and added to `missed` (participant names)."""
        if deadline_at is None:
            return _call(p, method, arg)
        ref = {"txn": arg, "deadline_at": deadline_at} if method in PHASE2 else arg
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
        return F.ok(txn=txn, state=state, outcome=t.get("outcome"), written=t["receipt"],
                    poisoned=bool(self.poison), missed=sorted(t.get("missed", ())))

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
        # (deadlines are on the shared wall clock, which participants also read)
        g = time.time()
        commit_at, end_at = (None, None) if d is None else (g + d / 2, g + d)
        missed = t["missed"] = set()

        def call(p, method, deadline_at):
            return self._bounded(p, method, txn, deadline_at, missed)
        for p in self.participants:
            res[p.name]["validate"] = call(p, "validate", commit_at)
        tl["vote"] = now()
        rollback = {"status": "not_needed", "participants": {}}
        out = {"rollback": rollback, "extra": [], "hashes": {}}
        if not all(res[p.name]["validate"]["ok"] for p in self.participants):
            for p in self.participants:
                res[p.name]["abort"] = call(p, "abort", end_at)
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
                for p in [failed] + list(reversed(committed)):
                    rollback["participants"][p.name] = res[p.name]["undo"] = \
                        call(p, "undo", end_at)
                for p in self.participants:
                    if p not in committed:
                        res[p.name]["abort"] = call(p, "abort", end_at)
                rollback["status"] = ("undone" if all(
                    u["ok"] for u in rollback["participants"].values()) else "partial")
        tl["gap_end"] = now()
        return out

    def _routes(self, plan, res):
        """The routes to rebuild after the commit: the plan's `rebuild`, else the routes of
        accepted, non-advisory prepare findings that a registered rebuild serves."""
        if plan.get("rebuild") is not None:
            return list(plan["rebuild"])
        return sorted({f["route"] for r in res.values()
                       for f in (r.get("prepare") or {}).get("findings", [])
                       if f.get("accepted") and not f.get("advisory")
                       and f.get("route") in self.rebuilds})

    def _rebuild(self, route, txn, why, deadline_at):
        """(ok-dict, seconds) of one rebuild; never raises."""
        t0 = time.perf_counter()
        try:
            r = self.rebuilds[route](txn, why, deadline_at)
        except Exception as e:
            r = F.not_ok(f"{route}: {type(e).__name__}: {e}")
        if not isinstance(r, dict) or "ok" not in r:
            r = F.not_ok(f"{route}: bad reply {r!r}")
        return r, round(time.perf_counter() - t0, 6)

    def _rebuild_all(self, routes, txn, why, since=None):
        """Every route in order, as one job of the loop owner, stopping at the first that
        fails. since: unix time of the code change (for the steps between it and the job).
        Returns {"routes", "why", "ok", "results", "seconds", "while", "steps_between",
        "steps_during", "start", "end"}."""
        rb = {"routes": list(routes), "why": why, "ok": True, "results": {}, "seconds": {},
              "while": self.rebuild_while}
        d = self.rebuild_deadline_s
        deadline_at = None if d is None else time.time() + d

        def job():
            rb["start"] = time.time()
            for route in routes:
                r, sec = self._rebuild(route, txn, why, deadline_at)
                rb["results"][route], rb["seconds"][route] = r, sec
                if not r["ok"]:
                    rb["ok"] = False
                    break
            rb["end"] = time.time()
        try:
            self.rebuild_job(job)
        except Exception as e:          # the job did not run, or not by the deadline
            rb["ok"] = False
            rb["error"] = f"{type(e).__name__}: {e}"
            self.poison.append({**F.finding(
                "deadline_missed", f"rebuild {routes} ({why}): {rb['error']}",
                f"the rebuild answers within {d} s"), "participant": "rebuild"})
        rb.setdefault("end", time.time())
        if self.steps is not None:     # what the engine did around the rebuild
            try:
                if since is not None and "start" in rb:
                    rb["steps_between"] = self.steps(since, rb["start"])
                if "start" in rb:
                    rb["steps_during"] = self.steps(rb["start"], rb["end"])
            except Exception:
                pass
        return rb

    @staticmethod
    def _rebuild_error(rb):
        bad = [f"{k}: {r.get('error')}" for k, r in rb["results"].items() if not r["ok"]]
        return "; ".join(bad) or rb.get("error") or "no result"

    def _after_commit(self, txn):
        """The post-commit phase (outside the gap): rebuild; on failure undo every participant
        in a gap and rebuild every route again from the base code."""
        t, now = self.txns[txn], self.clock
        out, tl = t["out"], t["tl"]
        routes = t["rebuild"]
        tl["rebuild_start"] = now()
        rb = out["rebuild"] = self._rebuild_all(routes, txn, "commit", since=tl.get("gap_end"))
        tl["rebuilt"] = now()
        if rb["ok"]:
            t["rebuilt"] = list(routes)
            return
        out["outcome"] = "rolled_back"
        out["extra"].append({**F.finding(
            "rebuild_failed", f"after the commit: {self._rebuild_error(rb)}",
            f"{routes} ok on every participant"), "subject": "commit"})
        rollback = out["rollback"]
        d = self.phase2_deadline_s

        def undo_all():
            end_at = None if d is None else time.time() + d
            for p in reversed(self.participants):
                rollback["participants"][p.name] = t["res"][p.name]["undo"] = self._bounded(
                    p, "undo", txn, end_at, t["missed"])
        try:
            self._in_gap(undo_all)
        except Exception as e:
            rollback["error"] = f"undo: {type(e).__name__}: {e}"
        tl["code_undone"] = now()
        back = rb["back"] = self._rebuild_all(routes, txn, "rollback", since=time.time())
        if not back["ok"]:
            out["extra"].append({**F.finding(
                "rebuild_failed", f"rebuilding from the base code after the undo: "
                f"{self._rebuild_error(back)}", "the base code's graphs again"),
                "subject": "rollback"})
        rollback["status"] = ("undone" if back["ok"] and "error" not in rollback and all(
            u["ok"] for u in rollback["participants"].values()) else "partial")
        tl["rolled_back"] = now()
        with self._lock:
            t["outcome"], t["state"] = "rolled_back", "aborted"

    def run(self, plan, edit_time=None, baseline_restart=None):
        """One transaction; appends a `txn` receipt. edit_time: when the edit was made (unix
        seconds; default now). baseline_restart: {"seconds", "source"} of the warm restart
        this replaces. Returns {"ok", "outcome", "txn", "receipt", "written"}, where outcome is
        committed, aborted, rolled_back, or in-doubt (resolve it with `status`)."""
        now = self.clock
        tl = {"edit": edit_time if edit_time is not None else now()}
        txn = plan.get("id") or f"txn-{uuid.uuid4().hex[:8]}"
        plan = {**plan, "id": txn}
        if self.poison:              # a late participant may still switch: undo or respawn
            return {"ok": False, "outcome": "refused", "txn": txn, "receipt": None,
                    "written": {"ok": False, "error": "host poisoned: no receipt"},
                    "error": "host poisoned: undo or respawn first",
                    "findings": [F.finding("session_poisoned", "; ".join(
                        f"{f['participant']}: {f['check']}" for f in self.poison),
                        "a host whose participants all met their deadlines")]}
        t = self.txns[txn] = {"plan": plan, "edit": tl["edit"], "tl": tl, "state": "prepared",
                              "res": {p.name: {} for p in self.participants},
                              "started": False, "cancelled": False, "done": False,
                              "receipt": False, "baseline": baseline_restart}
        # phase 1: outside the gap, steps go on
        tl["prepare_start"] = now()
        for p in self.participants:
            t["res"][p.name]["prepare"] = _call(p, "prepare", plan)
        tl["prepared"] = now()
        t["rebuild"] = self._routes(plan, t["res"])
        missing = [r for r in t["rebuild"] if r not in self.rebuilds]
        if missing or not all(t["res"][p.name]["prepare"]["ok"] for p in self.participants):
            for p in self.participants:
                t["res"][p.name]["abort"] = _call(p, "abort", txn)
            self._done(txn, {"outcome": "aborted", "hashes": {},
                             "extra": [{**F.finding("rebuild_failed", f"the plan asks for "
                                                    f"{r}, and this host has no rebuild for it",
                                                    f"one of {sorted(self.rebuilds)}"),
                                        "subject": r} for r in missing],
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
        if t["outcome"] == "committed" and t["rebuild"]:
            self._after_commit(txn)       # post-commit phase, outside the gap
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
        rank_hashes, ranks_agree = {}, {}
        if out["outcome"] == "committed":       # code hashes, read after the gap
            for p in self.participants:
                r = _call(p, "proof", txn)
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
        fs = [f for r in res.values() for x in r.values()
              for f in x.get("findings", [])] + out["extra"]
        modules = plan.get("modules", {})
        # agree: False if any comparison differs (across participants on a shared target, or
        # across the ranks of one group); None when nothing could be compared
        checks = [a for a in [_hashes_agree(out["hashes"])] + list(ranks_agree.values())
                  if a is not None]
        agree = all(checks) if checks else None

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
            "plan": {"targets": {m: s.get("targets", "auto") for m, s in modules.items()},
                     "predicted_resources": plan.get("predicted_resources", []),
                     "recipes": plan.get("recipes", []), "certainty": plan.get("certainty"),
                     "accept": plan.get("accept", []),
                     "undo_resets": plan.get("undo_resets", [])},
            "actual_recipes": ((["R0 swap"] if modules else [])
                               + (["R10 repair"] if plan.get("repairs") else [])
                               + [REBUILD_RECIPES.get(r, r) for r in t.get("rebuilt", [])])
            if out["outcome"] == "committed" else [],
            "participants": res,
            "commit_hashes": {"participants": out["hashes"], "ranks": rank_hashes,
                              "ranks_agree": ranks_agree, "agree": agree},
            "findings": fs,
            "decisions": {"accepted": sorted({f["check"] for f in fs if f.get("accepted")}),
                          "blocking": sorted({f["check"] for f in fs
                                              if not f.get("accepted")})},
            "outcome": out["outcome"], "timeline": tl,
            # with a rebuild, the edit runs in every step only once the rebuild is done
            "edit_to_run_s": span("edit", "rebuilt" if t.get("rebuilt") else "committed"),
            "edit_to_commit_s": span("edit", "committed"),
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
        t["rec"] = rec
        return {"ok": out["outcome"] == "committed", "outcome": out["outcome"], "txn": txn,
                "receipt": rec, "written": self._write(rec)}

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
        t = now()
        rec = {"schema": R.SCHEMA, "kind": "proof", "id": txn,
               "witness": witness,
               # calls made while graphs were captured, apart from the live ones above: a
               # replay runs no Python, so only live calls (or a device probe) prove a run
               "witness_capture": capture,
               "ran": {w: {k: W.ran(capture.get(w, {}).get(k, 0), n) for k, n in c.items()}
                       for w, c in witness.items()},
               "findings": fs, "golden": golden,
               "timeline": {"edit": edit, "proved": t},
               "edit_to_finish_s": round(t - edit, 6) if edit is not None else None}
        return {"ok": all(r["ok"] for r in res.values()) and not fs, "receipt": rec,
                "written": self._write(rec)}

    def _undo_phases(self, txn, resets, tl):
        res = {p.name: {} for p in self.participants}
        d, missed = self.phase2_deadline_s, set()
        end_at = None if d is None else time.time() + d
        for p in reversed(self.participants):
            res[p.name]["undo"] = self._bounded(p, "undo", txn, end_at, missed)
        tl["code_undone"] = self.clock()
        for kind in resets:
            for p in self.participants:
                res[p.name].setdefault("reset", {})[kind] = self._bounded(
                    p, "reset", kind, end_at, missed)
        return res

    def _undo_routes(self, txn, res):
        """The routes to rebuild after an undo: this transaction's, and those of the later
        transactions its undo cascaded to (`also_undone`, on any rank)."""
        also = set()
        for r in res.values():
            u = r.get("undo") or {}
            for x in [u] + list(u.get("ranks") or []):
                also.update(x.get("also_undone") or [])
        routes = list(self.txns.get(txn, {}).get("rebuilt", []))
        for a in sorted(also):
            routes += [x for x in self.txns.get(a, {}).get("rebuilt", []) if x not in routes]
        return routes

    def undo(self, txn):
        """Undo a committed transaction on every participant (newest first), then run the
        plan's `undo_resets` (R2, R3, ...) on each: code undo cannot un-write data the new code
        produced. In the gap. Then, outside the gap, the rebuilds the transaction ran after its
        commit (R1: the graphs of the base code again). Appends an `undo` receipt."""
        now = self.clock
        t = self.txns.get(txn, {})
        resets = t.get("plan", {}).get("undo_resets", [])
        tl = {"queued": now()}
        rb = {}
        try:
            res, err = self._in_gap(lambda: self._undo_phases(txn, resets, tl)), None
        except Exception as e:
            res, err = None, f"{type(e).__name__}: {e}"
        routes = self._undo_routes(txn, res or {}) if err is None else []
        if routes:                     # rebuild from the base code, outside the gap
            tl["rebuild_start"] = now()
            rb = self._rebuild_all(routes, txn, "undo", since=tl.get("code_undone"))
            tl["rebuilt"] = now()
        tl["undo_end"] = now()
        res = res or {}
        calls = [x for r in res.values()
                 for x in [r.get("undo")] + list(r.get("reset", {}).values()) if x]
        fs = [{**f, "participant": n} for n, r in res.items()
              for x in [r.get("undo")] + list(r.get("reset", {}).values()) if x
              for f in x.get("findings", [])]
        if rb and not rb["ok"]:
            fs.append({**F.finding("rebuild_failed", "rebuilding after the undo: "
                                   f"{self._rebuild_error(rb)}", "the base code's graphs again"),
                       "subject": "undo"})
        ok = err is None and all(x["ok"] for x in calls) and rb.get("ok", True)
        rec = {"schema": R.SCHEMA, "kind": "undo", "id": txn, "participants": res,
               "resets": list(resets), "status": "undone" if ok else "partial",
               "findings": fs + ([F.finding("undo_incomplete", f"undo did not finish: {err}",
                                            "a step gap within the timeout")] if err else []),
               "rebuild": rb or None,
               "recapture_s": (rb.get("seconds") or {}).get("R1"),
               "timeline": tl, "undo_s": round(tl["undo_end"] - tl["queued"], 6)}
        if t and rb.get("ok"):
            t["rebuilt"] = []
        if t:                          # partial: an undo ran but could not restore everything
            with self._lock:
                t["state"] = "undone" if ok else "partial"
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
        # a rebuild is one job of the loop owner (not inside phase 2's gap): no step runs
        # while it runs; steps may run between the commit and the rebuild
        kw.setdefault("rebuild_job", gap)
        kw.setdefault("rebuild_while", "paused: the rebuild runs as one job of the engine loop, "
                      "so no step runs during it; steps between the commit and the rebuild "
                      "run the new Python code and the old graphs")
        super().__init__(participants, **kw)
        self.gap = gap

    def _in_gap(self, fn):
        return self.gap(fn)
