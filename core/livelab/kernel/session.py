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
from . import locate as L
import json


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
    def __init__(self, coordinator, modules):
        self.co = coordinator
        self.modules = list(modules)
        self.stack = []          # committed txn ids, oldest first
        self.expected = []       # fingerprint expected at each stack depth: [base, after t1, ...]
        self.poison = []         # findings
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
                return None
            out[p.name] = r["code"]
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

    def state(self):
        return "poisoned" if self.poisoned() else "edited" if self.stack else "base"

    def status(self):
        """One fingerprint round: `verify` already took it, so its digest is reused instead of
        asking every participant again (on a 21-actor Ray job a round costs about 10 s)."""
        v = self.verify() if not self.poisoned() else None
        fp = v.get("fingerprint") if v is not None else self.digest(self._fingerprint())
        return F.ok(state=self.state(), txns=list(self.stack), poison=self.poisoned(), fingerprint=fp)

    def _refuse(self, what):
        return F.not_ok(f"{what} refused: session poisoned", [F.finding(
            "session_poisoned", "; ".join(f["check"] for f in self.poisoned()),
            "a session in a known state")], outcome="refused")

    # -- verbs ---------------------------------------------------------------------------
    def install(self, plan, **kw):
        """One transaction. Returns the coordinator's result plus `state`."""
        if self.poisoned():
            return self._refuse("install")
        # code changed outside the journal since the last commit or undo: refuse before the
        # transaction, or its commit would record the tampered code as expected
        if not self.verify()["ok"]:
            return self._refuse("install")
        res = self.co.run(plan, **kw)
        outcome = res["outcome"]
        if outcome == "in-doubt":
            st = self.co.status(res["txn"])
            outcome = st.get("outcome") if st.get("state") != "in-doubt" else None
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
        else:                                    # aborted or rolled back: base must hold
            if outcome == "rolled_back":
                receipt = _read_txn_receipt(self.co, res)
                rollback_status = ((receipt or {}).get("rollback") or {}).get("status")
                if rollback_status != "undone":
                    already = any(f.get("check") == "undo_incomplete" for f in self.poisoned())
                    if not already:
                        why = ("rollback receipt missing" if receipt is None else
                               f"rollback status was {rollback_status!r}")
                        self.fault("undo_incomplete",
                                   f"rollback of a failed commit was partial ({why})",
                                   "every committed participant undone")
            self.verify()
        return {**res, "outcome": outcome, "state": self.state()}

    def uninstall(self, txn=None):
        """Undo `txn` and every later transaction (newest first); default: the newest. Allowed
        while poisoned (undo is how a poisoned lab backs out before a respawn); the session
        stays poisoned."""
        if not self.stack:
            return F.ok(undone=[], state=self.state())
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
                status_label = (u.get("receipt") or {}).get("status", u.get("status", "partial"))
                self.fault("undo_incomplete", f"undo of {t}: {status_label}",
                           "undone on every participant", subject=t)
                return F.not_ok("undo failed", self.poisoned(), undone=undone,
                                results=results, state=self.state())
            self.stack.pop()
            self.expected.pop()
            undone.append(t)
        v = self.verify()
        return {**v, "undone": undone, "results": results, "state": self.state()}

    def recovered(self, coordinator=None):
        """After the lab respawned its participants: a new base, no poison, empty stack."""
        if coordinator is not None:
            self.co = coordinator
        elif hasattr(self.co, "recovered"):
            self.co.recovered()
        self.stack, self.poison = [], []
        fp = self._fingerprint()
        self.expected = [fp]
        if fp is None:
            self.fault("tampered", "a participant gave no fingerprint after respawn",
                       "every participant answers")
        return F.ok(state=self.state())
