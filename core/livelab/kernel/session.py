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


def _split_commit(receipt):
    """Why a committed transaction cannot be recorded as the code every participant runs, or
    None. Never excused by an accept: a rank that did not switch must not become the expected
    state. Reasons: no receipt to read; commit_hashes.agree False (participants or ranks
    differ, or a proof did not answer); or a participant (a rank, in a group) that staged
    targets at prepare and shows no code hash in its proof. A role that abstained, or staged
    nothing (a plan of repair cells only), may show none."""
    if not receipt:
        return "no txn receipt to read the commit hashes from"
    if (receipt.get("commit_hashes") or {}).get("agree") is False:
        splits = [finding.get("observed", "") for finding in receipt.get("findings", [])
                  if finding.get("check") == "ranks_disagree"]
        why = "; ".join(splits) or "a proof did not answer"
        return f"code hashes differ after commit: {why}"[:500]
    for name, reply in (receipt.get("participants") or {}).items():
        prepared, proof = reply.get("prepare") or {}, reply.get("proof") or {}
        proof_by_rank = {str(unit.get("rank")): unit for unit in proof.get("ranks") or ()}
        for unit in prepared.get("ranks") or [prepared]:
            if unit.get("abstain") or not unit.get("staged"):
                continue
            rank = unit.get("rank") if prepared.get("ranks") else None
            shown = proof if rank is None else proof_by_rank.get(str(rank), {})
            if not shown.get("hashes"):
                where = name if rank is None else f"{name} rank {rank}"
                return (f"{where} staged {len(unit['staged'])} target(s) but its proof shows "
                        "no code hash: it did not switch")
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
        out = {}
        for p in self.co.participants:
            r = C._call(p, "fingerprint", self.modules)
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
        if not self.poisoned():
            self.verify()
        return F.ok(state=self.state(), txns=list(self.stack), poison=self.poisoned(),
                    fingerprint=self.digest(self._fingerprint()))

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
            fp = self._fingerprint()
            if fp is None:
                self.fault("tampered", "a participant did not answer after commit",
                           "every participant answers")
            # an in-doubt result carries no receipt; the coordinator keeps the one it built
            receipt = res.get("receipt") or (
                getattr(self.co, "txns", {}).get(res["txn"]) or {}).get("rec")
            split = _split_commit(receipt)
            if split:
                self.fault("ranks_disagree", split, "every participant and every rank runs "
                           "the code the commit installed", subject=res["txn"])
            self.stack.append(res["txn"])
            self.expected.append(fp)
        else:                                    # aborted or rolled back: base must hold
            if outcome == "rolled_back" and res["receipt"]["rollback"]["status"] != "undone":
                self.fault("undo_incomplete", "rollback of a failed commit was partial",
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
                self.fault("undo_incomplete", f"undo of {t}: {u['receipt']['status']}",
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
