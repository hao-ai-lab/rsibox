"""Session: the host's state machine for one live lab: base, edited or poisoned.

The coordinator runs single transactions. The session strings them together and keeps what
the interface's `status()` needs:

- a stack of committed transactions, newest last; `uninstall` undoes the newest (or every
  transaction down to a given one, newest first);
- the code fingerprint every participant should show now (`fingerprint(modules)`: content
  hash and code object id of every function the modules' namespaces reach). It is recorded
  at the start and after every commit, and checked after every abort, rollback and undo,
  before every `install` and on every `status()`;
- poison: findings that put the session in a state the host cannot prove. Set by a
  fingerprint that differs from the expected one (`tampered`: code changed outside the
  journal, or a participant that does not answer), by an undo that failed, by a transaction
  left in doubt, by a commit after which participants or ranks run different code
  (`ranks_disagree`), by a participant that missed the phase-2 deadline (the coordinator's
  `poison`), or by the lab itself (`fault`: a crash or a hang it saw). A poisoned session
  refuses `install` with `session_poisoned` and allows only `uninstall` (undo; the session
  stays poisoned) and `recovered()` (after the lab respawned its worker group).

Never raises to the caller: every call returns an ok-dict.
"""
from . import coordinator as C
from . import findings as F
from . import identity as ID
from . import locate as L
from . import swap as S
import ast
import json
import sys


def _targeted_modules(plan, name, unit):
    """Modules in `plan` targeting `unit` (respecting `'on'` role/name filters, else all)."""
    if not isinstance(plan, dict):
        return []
    role = unit.get("role")
    rank = unit.get("rank")
    targeted = []
    for mod_name, spec in (plan.get("modules") or {}).items():
        on = spec.get("on") if isinstance(spec, dict) else None
        if on is None:
            targeted.append(mod_name)
        elif role is not None:
            if role in on:
                targeted.append(mod_name)
        elif name in on or (rank is not None and str(rank) in on):
            targeted.append(mod_name)
    return targeted


def _fp_has_module(part_fp, mod_name, rank=None):
    """True when `part_fp` records any live function for `mod_name` (or `<mod_name>#actor`)."""
    if not part_fp:
        return False
    prefixes = (f"{mod_name}:", f"{mod_name}#actor:")
    for k in part_fp:
        if rank is not None:
            r_prefix = f"rank{rank}/"
            if k.startswith(r_prefix) and k[len(r_prefix):].startswith(prefixes):
                return True
        elif k.startswith(prefixes) or (k.startswith("rank") and "/" in k and
                                        k.split("/", 1)[1].startswith(prefixes)):
            return True
    return False


def _read_txn_receipt(co, res):
    """Return `res["receipt"]`, waiting on a concurrent receipt finisher if needed."""
    receipt = res.get("receipt")
    if receipt is not None or not res.get("txn"):
        return receipt
    record = getattr(co, "txns", {}).get(res["txn"]) or {}
    finisher = record.get("fin")
    if finisher is not None and finisher.acquire(timeout=30):
        finisher.release()
    return record.get("rec")


def _split_commit(receipt, plan=None, expected_fp=None):
    """Why a committed transaction cannot be recorded as the code every participant runs, or
    None. Never excused by an accept: a rank that did not switch must not become the expected
    state. Reasons: no receipt to read; commit_hashes.agree False (participants or ranks
    differ, or a proof did not answer); a participant falsely returning abstain=True or empty
    staged={} without abstaining for a module targeting it; or a participant (a rank, in a
    group) that staged items at prepare and whose proof names none of them."""
    if not receipt:
        return "no txn receipt to read the commit hashes from"
    if (receipt.get("commit_hashes") or {}).get("agree") is False:
        splits = [finding.get("observed", "") for finding in receipt.get("findings", [])
                  if finding.get("check") == "ranks_disagree"]
        why = "; ".join(splits) or "a proof did not answer"
        return f"code hashes differ after commit: {why}"[:500]
    for name, reply in (receipt.get("participants") or {}).items():
        prepared = reply.get("prepare") or {}
        committed = reply.get("commit") or {}
        proof = reply.get("proof") or {}
        proof_by_rank = {str(unit.get("rank")): unit for unit in proof.get("ranks") or ()}
        commit_by_rank = {str(unit.get("rank")): unit for unit in committed.get("ranks") or ()}
        part_fp = (expected_fp or {}).get(name) or {}
        for unit in prepared.get("ranks") or [prepared]:
            rank = unit.get("rank") if prepared.get("ranks") else None
            where = name if rank is None else f"{name} rank {rank}"
            targeted_mods = _targeted_modules(plan, name, unit) if plan is not None else []
            if unit.get("abstain"):
                if targeted_mods:
                    has_loaded = any(_fp_has_module(part_fp, mod, rank=rank)
                                     for mod in targeted_mods)
                    if not unit.get("absent") or has_loaded:
                        return (f"{where} falsely abstained from targeted module(s) "
                                f"{', '.join(targeted_mods)}")
                continue
            if not unit.get("staged"):
                if targeted_mods:
                    commit_unit = committed if rank is None else commit_by_rank.get(str(rank), {})
                    prep_checks = {f.get("check") for f in (unit.get("findings") or ())}
                    if ("removed_definition" not in prep_checks
                            and (commit_unit.get("changes") or 0) <= 0):
                        return (f"{where} returned empty staged={{}} without abstaining for "
                                f"targeted module(s) {', '.join(targeted_mods)}")
                continue
            shown = proof if rank is None else proof_by_rank.get(str(rank), {})
            # the proof names what the participant committed: code hashes of the swapped
            # functions, and witness keys of every staged item (added definitions too)
            committed_keys = set(shown.get("hashes") or {}) | set(shown.get("witness") or {})
            if not committed_keys & set(unit["staged"]):
                return (f"{where} staged {len(unit['staged'])} item(s) but its proof names "
                        "none of them (no code hash, no witness key): it did not switch")
    return None


class Session:
    def __init__(self, coordinator, modules, base_sources=None):
        self.co = coordinator
        self.modules = list(modules)
        self.stack = []          # committed txn ids, oldest first
        self.expected = []       # fingerprint expected at each stack depth: [base, after t1, ...]
        self.poison = []         # findings
        self._recovery_required = False
        self._recovery_reasons = []
        self._base_sources = dict(base_sources or {})
        self._source_stack = [dict(self._base_sources)]
        self._last_fp = None
        fp = self._fingerprint()
        if fp is None:
            self.fault("tampered", "a participant gave no fingerprint at start",
                       "every participant answers")
        self.expected = [fp]

    # -- fingerprints --------------------------------------------------------------------
    def _fingerprint(self):
        """{participant: {key: [hash, id]}}, or None when a participant does not answer."""
        if getattr(self.co, "parallel_fanout", False):
            replies = C._fanout(
                self.co.participants,
                lambda p: C._call(p, "fingerprint", self.modules),
            )
        else:
            replies = [C._call(p, "fingerprint", self.modules) for p in self.co.participants]
        out = {}
        for p, r in zip(self.co.participants, replies):
            if not r["ok"]:
                self._last_fp = None
                return None
            out[p.name] = r["code"]
        self._last_fp = out
        return out

    @staticmethod
    def digest(fp):
        return None if fp is None else L.text_hash(json.dumps(fp, sort_keys=True))

    def verify(self):
        """Do participants run exactly the code the session expects? Poisons if not."""
        fp = self._fingerprint()
        want = self.expected[-1] if self.expected else None
        if fp is None:
            self.fault("tampered", "a participant did not answer the fingerprint call",
                       "every participant answers")
        elif fp != want:
            diff = sorted({f"{p}:{k}" for p in set(fp) | set(want or {})
                           for k in set(fp.get(p, {})) | set((want or {}).get(p, {}))
                           if fp.get(p, {}).get(k) != (want or {}).get(p, {}).get(k)})
            self.fault("tampered", f"{len(diff)} function(s) differ: {diff[:5]}",
                       "the code the last commit or undo left")
        if fp is None:      # no fingerprint proves nothing, even when none was expected
            return F.not_ok("no fingerprint", self.poison, fingerprint=None)
        return F.ok(fingerprint=self.digest(fp)) if fp == want else \
            F.not_ok("fingerprint differs", self.poison, fingerprint=self.digest(fp))

    # -- state ---------------------------------------------------------------------------
    def fault(self, check, observed, expected, subject=None):
        """Record a fault the lab saw (crash, hang, lost participant). Poisons the session."""
        f = F.finding(check, observed, expected, subject=subject)
        self.poison.append(f)
        return f

    def poisoned(self):
        """Every poison finding: the session's own and the coordinator's (deadline missed)."""
        return self.poison + list(getattr(self.co, "poison", []))

    def recovery_required(self):
        """True when a failed rollback, undo, or unconfirmed abort requires a clean respawn."""
        if self._recovery_required:
            return True
        co_check = getattr(self.co, "is_recovery_required", None)
        return bool(co_check()) if callable(co_check) else False

    def _latch_recovery(self, reason):
        self._recovery_required = True
        if reason and reason not in self._recovery_reasons:
            self._recovery_reasons.append(reason)

    def _alias_map(self):
        for p in getattr(self.co, "participants", ()):
            if isinstance(p, C.LocalParticipant) and p.alias:
                return p.alias
        return None

    def identity(self, fp_raw=None):
        """Deterministic SHA-256 identity of active module sources, txn chain, and state."""
        fp = fp_raw if fp_raw is not None else (self.expected[-1] if self.expected else None)
        eng = getattr(self.co, "engine", None)
        return ID.compute_identity(
            module_sources=self._source_stack[-1], txns=self.stack, state=self.state(),
            modules=self.modules, fingerprint=fp, incarnation=getattr(self.co, "incarnation", 1),
            recovery_required=self.recovery_required(), metadata={"engine": eng} if eng else {},
            alias=self._alias_map())

    def state(self):
        return "poisoned" if (self.poisoned() or self._recovery_required) else (
            "edited" if self.stack else "base")

    def status(self):
        """One fingerprint round: `verify` already took it, so its digest is reused instead of
        asking every participant again (on a 21-actor Ray job a round costs about 10 s)."""
        v = self.verify() if not self.poisoned() else None
        fp_raw = self._last_fp if v is not None else self._fingerprint()
        fp = v.get("fingerprint") if v is not None else self.digest(fp_raw)
        rec_req = self.recovery_required()
        return F.ok(state=self.state(), txns=list(self.stack), poison=self.poisoned(),
                    fingerprint=fp, recovery_required=rec_req,
                    incarnation=getattr(self.co, "incarnation", 1),
                    identity=self.identity(fp_raw=fp_raw))

    def _refuse(self, what):
        rec_req = self.recovery_required()
        if what == "uninstall":
            why = "; ".join(self._recovery_reasons) or "; ".join(
                f["check"] for f in self.poisoned()) or "rollback or undo failed"
            return F.not_ok(
                "uninstall refused: recovery required (respawn)",
                [F.finding("recovery_required", why,
                           "clean respawn via reset('respawn') / recovered()")],
                outcome="refused", state="poisoned", recovery_required=True, undone=[])
        why = "; ".join(f["check"] for f in self.poisoned()) or (
            "recovery_required" if rec_req else "poisoned")
        if rec_req and "recovery_required" not in why:
            why = f"{why}; recovery_required"
        return F.not_ok(f"{what} refused: session poisoned", [F.finding(
            "session_poisoned", why,
            "a session in a known state")], outcome="refused", recovery_required=rec_req)

    # -- non-mutating dry-run planner ----------------------------------------------------
    def _plan_live_for(self, mod_name, base_src, mod_obj):
        if mod_obj is not None:
            run, held = L.index_module(mod_obj, L.module_functions(mod_obj))
            return {kl: True for kl, cs in run.items() if any(held.get(c) for c in cs)}
        live, top_live, prefixes = {}, set(), (f"{mod_name}:", f"{mod_name}#actor:")
        for entries in (self.expected[-1] if self.expected else {}).values():
            for key in (entries or ()):
                bare = key.split("/", 1)[1] if key.startswith("rank") and "/" in key else key
                for pref in prefixes:
                    if bare.startswith(pref):
                        q, sep, ln = bare[len(pref):].rpartition(":")
                        if sep and ln.isdigit():
                            live[(q, int(ln))] = True
                            top_live.add(q.split(".<locals>.", 1)[0])
        try:
            for q, nodes in S.qual_index(ast.parse(base_src, f"<{mod_name}>")).items():
                if not live or q.split(".<locals>.", 1)[0] in top_live:
                    for n in nodes:
                        live.setdefault((q, S.first_line(n)), True)
        except SyntaxError:
            pass
        return live

    def _plan_base_source(self, mod_name, spec, extra_sources):
        mod_obj = sys.modules.get(ID._resolve_alias(mod_name, self._alias_map()))
        for cand in (extra_sources.get(mod_name) if isinstance(extra_sources, dict) else None,
                     self._source_stack[-1].get(mod_name),
                     *(p._live_source(mod_name) for p in getattr(self.co, "participants", ()) if isinstance(p, C.LocalParticipant)),
                     spec.get("base_source") if isinstance(spec, dict) else None):
            if isinstance(cand, str):
                return cand, mod_obj
        mod_file = getattr(mod_obj, "__file__", None) if mod_obj is not None else None
        if isinstance(mod_file, str):
            try:
                with open(mod_file, encoding="utf-8") as fh:
                    return fh.read(), mod_obj
            except (OSError, UnicodeDecodeError):
                pass
        return None, mod_obj

    def plan(self, patch, base_sources=None):
        """Non-mutating dry-run planner: inspect `patch` without mutating participants or journals."""
        ident = self.identity()

        def refused(affected, findings):
            return {"ok": False, "affected_modules": affected, "targets": {}, "mechanisms": ["refused"],
                    "restart_boundary": "refused", "refusal_findings": findings, "findings": findings, "identity": ident}

        if not isinstance(patch, dict):
            return refused([], [F.finding("patch_invalid", "patch must be a dict", "a patch dict")])
        key_name = "modules" if "modules" in patch else ("files" if "files" in patch else None)
        if key_name is not None and not isinstance(patch[key_name], dict):
            return refused([], [F.finding("patch_invalid", f"patch[{key_name!r}] must be a dict", "a dict")])
        modules = ({} if key_name is None else patch["modules"] if key_name == "modules"
                   else {m: {"source": src} for m, src in patch["files"].items()})
        repairs, accept, affected = patch.get("repairs", ()), patch.get("accept", ()), sorted(str(m) for m in modules)
        if not isinstance(repairs, (list, tuple)) or (accept != "all" and not isinstance(accept, (list, tuple, set))):
            return refused(affected, [F.finding("patch_invalid", "invalid repairs or accept", "list/tuple")])
        if self.poisoned() or self.recovery_required():
            return refused(affected, self._refuse("plan").get("findings", []))
        if not modules and not repairs:
            return refused([], [F.finding("empty_change", "patch has no modules or repair cells", "at least one change")])
        fs, mechs, targets_out, any_change = [], [], {}, bool(repairs)
        for mod_name, spec in modules.items():
            if not isinstance(mod_name, str) or not isinstance(spec, dict) or not isinstance(spec.get("source"), str):
                fs.append(F.finding("patch_invalid", f"{mod_name}: missing source", "string source", subject=str(mod_name)))
                continue
            base_src, mod_obj = self._plan_base_source(mod_name, spec, base_sources)
            if base_src is None:
                fs.append(F.finding("patch_invalid", f"{mod_name}: base source unavailable", "known module", subject=mod_name))
                continue
            if spec.get("base_hash") and spec["base_hash"] != L.text_hash(base_src):
                fs.append(F.finding("base_mismatch", f"{mod_name}: base_hash differs", spec["base_hash"], subject=mod_name))
            try:
                ana = S.analyze(base_src, spec["source"], spec.get("targets", "auto"),
                                getattr(mod_obj, "__file__", None) or f"<{mod_name}>",
                                mod_name, self._plan_live_for(mod_name, base_src, mod_obj))
            except Exception as exc:
                fs.append(F.finding("patch_invalid", f"{type(exc).__name__}: {exc}", "valid targets/source", subject=mod_name))
                continue
            if "error" in ana or ana.get("new_error"):
                fs.append(F.finding("patch_invalid", ana.get("error") or ana["new_error"], "valid Python", subject=mod_name))
                continue
            targets_out[mod_name] = ana["resolved"]
            base_codes = L.code_tree(ana["base_units"][0])[0] if ana["base_units"] else {}
            new_codes = L.code_tree(ana["new_units"][0])[0] if ana["new_units"] else {}
            any_change = any_change or bool(ana["added"] or ana["removed"] or any(i.get("changed") for i in ana["imports"]))
            for (q, line), tr in zip(ana["resolved"], ana["targets"]):
                key, target_status = L.key_of(mod_name, q, line), tr["status"]
                if target_status in ("no_definition", "count", "no_live_function"):
                    fs.append(F.finding("no_live_function" if target_status == "no_live_function" else "not_in_patch",
                                        f"{key}: {target_status}", "live matching definition", subject=key))
                    continue
                any_change = any_change or bool(tr.get("changed"))
                bc, nc = (base_codes.get((q, line)) or [None])[0], (new_codes.get((q, tr["new_line"])) or [None])[0]
                if bc is not None and nc is not None and bc.co_freevars != nc.co_freevars:
                    fs.append(F.finding("closure_changed", f"{bc.co_freevars} -> {nc.co_freevars}", f"same freevars {bc.co_freevars}", subject=key))
                if tr.get("nested_signature"):
                    fs.append(F.finding("nested_signature", tr["nested_signature"], "unchanged signature", subject=key))
                jit_kind = tr.get("jit") or (S._jit_holder(mod_obj, q) if mod_obj else None)
                if jit_kind:
                    jt = tr.get("jit_text")
                    if jt and not jt.get("guard"):
                        mechs.append("triton_rejit")
                        fs.append({**F.finding("jit_rebuild", f"{key}: {jit_kind}", "in-place Triton rebuild", subject=key), "advisory": True})
                    else:
                        fs.append(F.finding("jit_target", (jt or {}).get("guard") or f"{key}: {jit_kind}", "rebuildable JIT kernel", subject=key))
                elif q.rpartition(".")[2] in S.CONSTRUCTORS:
                    if tr.get("reinit"):
                        mechs.append("constructor_reinit")
                        fs.append({**F.finding("constructor_reinit", key, "re-init in place", subject=key), "advisory": True})
                    else:
                        fs.append(F.finding("constructor_edit", key, "non-constructor target", subject=key))
                elif tr.get("changed"):
                    mechs.append("in_place_python")
            if ana["added"]:
                mechs.append("in_place_python")
            if ana["removed"]:
                fs.append(F.finding("removed_definition", ", ".join(ana["removed"]), "explicit accept", subject=mod_name))
            fs += [F.finding("new_fields", f"{nf['cls']}.{nf['field']}", "existing field", subject=mod_name) for nf in ana["new_fields"]]
            if ana["outside_changes"]:
                fs.append(F.finding("outside_changes", f"{mod_name}: changes outside targets", "target bodies only", subject=mod_name))
        for rep in repairs:
            try:
                if not isinstance(rep, dict) or not isinstance(rep.get("cell"), str):
                    raise TypeError("repair cell must be a dict with string 'cell'")
                compile(rep["cell"], "<plan-repair>", "exec")
                mechs.append("repair_cell")
            except Exception as e:
                fs.append(F.finding("patch_invalid", f"repair cell: {type(e).__name__}: {e}", "valid repair cell"))
        if not any_change and all(f["overridable"] for f in fs):
            fs.append(F.finding("empty_change", "no definition changed, added, or removed", "at least one change"))
        try:
            dec = F.decide(fs, accept)
        except Exception as exc:
            return refused(affected, [F.finding("patch_invalid", f"invalid accept: {exc}", "valid accept list")])
        ok = bool(dec["proceed"])
        uniq = list(dict.fromkeys(mechs)) if ok else ["refused"]
        boundary = ("refused" if not ok else "triton_rejit" if "triton_rejit" in uniq else
                    "constructor_reinit" if "constructor_reinit" in uniq else
                    "repair_cell" if "repair_cell" in uniq and "in_place_python" not in uniq else "in_place_python")
        return {"ok": ok, "affected_modules": affected, "targets": targets_out, "mechanisms": uniq,
                "restart_boundary": boundary, "refusal_findings": [f for f in dec["findings"] if not f.get("accepted")],
                "findings": dec["findings"], "identity": ident}

    # -- verbs ---------------------------------------------------------------------------
    def install(self, plan, **kw):
        """One transaction. Returns the coordinator's result plus `state`."""
        if self.poisoned() or self.recovery_required():
            return self._refuse("install")
        # code changed outside the journal since the last commit or undo: refuse before the
        # transaction, or its commit would record the tampered code as expected
        if not self.verify()["ok"]:
            return self._refuse("install")
        res = self.co.run(plan, **kw)
        outcome = res["outcome"]
        if outcome == "in-doubt":
            status_info = self.co.status(res["txn"])
            outcome = status_info.get("outcome") if status_info.get("state") != "in-doubt" else None
            if outcome is None:
                self.fault("participant_failed", f"transaction {res['txn']} still in doubt",
                           "a known outcome", subject=res["txn"])
                return {**res, "state": self.state()}
        if outcome == "committed":
            before_fp = self.expected[-1] if self.expected else None
            fp = self._fingerprint()
            if fp is None:
                self.fault("tampered", "a participant did not answer after commit",
                           "every participant answers")
            # an in-doubt result carries no receipt, nor does one whose receipt another
            # thread (a status call) is finishing: wait for that finisher, then read the
            # receipt the coordinator kept
            receipt = _read_txn_receipt(self.co, res)
            split = _split_commit(receipt, plan=plan, expected_fp=before_fp)
            if split:
                self.fault("ranks_disagree", split, "every participant and every rank runs "
                           "the code the commit installed", subject=res["txn"])
            self.stack.append(res["txn"])
            self.expected.append(fp)
            next_src = dict(self._source_stack[-1])
            for m, spec in (plan.get("modules") or {}).items():
                if isinstance(spec, dict) and isinstance(spec.get("source"), str):
                    next_src[m] = spec["source"]
            self._source_stack.append(next_src)
        else:                                    # aborted or rolled back: base must hold
            if outcome == "rolled_back":
                receipt = _read_txn_receipt(self.co, res)
                rb = (receipt or {}).get("rollback") or {}
                rollback_status = rb.get("status")
                if rollback_status != "undone":
                    already = any(f.get("check") == "undo_incomplete" for f in self.poisoned())
                    why = ("rollback receipt missing" if receipt is None else
                           f"rollback status was {rollback_status!r}")
                    if not already:
                        self.fault("undo_incomplete",
                                   f"rollback of a failed commit was partial ({why})",
                                   "every committed participant undone")
                    rb_parts = rb.get("participants") or {}
                    hard_fail = (receipt is None or not rb_parts or any(
                        not u.get("ok") and not u.get("missed") for u in rb_parts.values()
                    ))
                    if hard_fail:
                        self._latch_recovery(f"commit rollback failed ({why})")
            self.verify()
        return {**res, "outcome": outcome, "state": self.state(),
                "recovery_required": self.recovery_required()}

    def uninstall(self, txn=None):
        """Undo `txn` and every later transaction (newest first); default: the newest.
        Refused fail-closed when `recovery_required` is latched (failed rollback/undo/abort);
        otherwise allowed while poisoned (e.g. tampered code) before respawn."""
        if self.recovery_required():
            return self._refuse("uninstall")
        if not self.stack:
            return F.ok(undone=[], state=self.state(), recovery_required=False)
        txn = txn or self.stack[-1]
        if txn not in self.stack:
            return F.not_ok(f"{txn} is not committed in this session")
        # check before the first undo: undoing a transaction writes its base code back, which
        # would overwrite (and hide) a hand edit of a function it had installed
        self.verify()
        undone, results = [], []
        while self.stack and txn in self.stack:
            t = self.stack[-1]
            u = self.co.undo(t)
            results.append(u)
            if not u["ok"]:
                status_label = u.get("status") or (u.get("receipt") or {}).get("status", "partial")
                if status_label != "cancelled":
                    self._latch_recovery(f"undo of {t}: {status_label}")
                self.fault("undo_incomplete", f"undo of {t}: {status_label}",
                           "undone on every participant", subject=t)
                return F.not_ok("undo failed", self.poisoned(), undone=undone,
                                results=results, state=self.state(),
                                recovery_required=self.recovery_required())
            self.stack.pop()
            self.expected.pop()
            if len(self._source_stack) > 1:
                self._source_stack.pop()
            undone.append(t)
        v = self.verify()
        return {**v, "undone": undone, "results": results, "state": self.state(),
                "recovery_required": self.recovery_required()}

    def recovered(self, coordinator=None, base_sources=None):
        """After the lab respawned its participants: a new base, no poison, empty stack."""
        old_inc = getattr(self.co, "incarnation", 1)
        if coordinator is not None and coordinator is not self.co:
            self.co = coordinator
            if hasattr(self.co, "incarnation"):
                lock = getattr(self.co, "_lock", None)
                if lock is not None:
                    with lock:
                        self.co.incarnation = max(self.co.incarnation, old_inc + 1)
                else:
                    self.co.incarnation = max(self.co.incarnation, old_inc + 1)
        elif hasattr(self.co, "recovered"):
            self.co.recovered()
        self.stack, self.poison, self._recovery_required, self._recovery_reasons = [], [], False, []
        if base_sources is not None:
            self._base_sources = dict(base_sources)
        self._source_stack = [dict(self._base_sources)]
        fp = self._fingerprint()
        self.expected = [fp]
        if fp is None:
            self.fault("tampered", "a participant gave no fingerprint after respawn",
                       "every participant answers")
        return F.ok(state=self.state(), recovery_required=self.recovery_required(),
                    incarnation=getattr(self.co, "incarnation", 1), identity=self.identity(fp_raw=fp))
