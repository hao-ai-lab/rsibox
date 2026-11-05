"""Findings: what a check saw, why it matters, what to do. Checks advise; the caller decides.

One schema for every check in the kernel:

    {"check": str,             # name, also the key in the accept list
     "observed": str,          # what the check saw
     "expected": str,          # what it needed to see
     "why": str,               # why the difference matters
     "actions": [{"action": str, "cost": str}],
     "risk_if_ignored": str,
     "lessons": [str],         # related rows of repos/<engine>/kb/lessons.md
     "overridable": bool}      # False: physical, accepting it cannot make the change happen

plus two optional annotations: "subject" (what was checked, e.g. a witness key) and
"accepted" (set by `decide`). Nothing in the kernel raises to its caller for a check: every
call returns an ok-dict `{"ok": bool, ..., "findings": [...]}`.

One accept mechanism: a list (or "all"). `decide(findings, accept)` says whether to proceed:
every finding must be accepted, and only overridable ones can be. An item of the list is a
check name, or an accept record that says who accepted it and why, for the transaction record:

    {"check": str, "by": str, "reason": str,
     "scope": "txn" | "standing",   # "txn": this transaction only (the default)
     "subject": str,                # optional, any scope: only the findings of this check
                                    # about this subject (e.g. one witness key), not all
     "participant": str, "code_hash": str}   # standing only: matches a finding only while the
                                             # finding's participant and code_hash still equal
                                             # these, so it expires when the checked code changes

A check accepted on every run is a calibration bug in the check, not a standing accept.
"""

RESTART = {"action": "warm restart of the engine", "cost": "recipe R9 (see kb/costs.md)"}
REBUILD = {"action": "soft restart: rebuild the objects from the edited source",
           "cost": "recipe R5 (see kb/costs.md)"}
ACCEPT = {"action": "accept this finding and go on", "cost": "none"}

# check -> (why, actions, risk if ignored, lessons, overridable)
CATALOG = {
    "patch_invalid": (
        "the patched source does not parse or compile, or its module cannot be imported",
        [{"action": "fix the patch", "cost": "seconds"}],
        "nothing is installed", [], False),
    "no_live_function": (
        "no function object in this process runs this code (module or class body, or a "
        "function nothing references any more), so there is nothing to swap",
        [REBUILD, RESTART], "the edit is not applied anywhere", ["Keeping state across edits"],
        False),
    "not_in_patch": (
        "the target is missing from the patched source, or ambiguous (a same-named definition "
        "was added or removed)",
        [{"action": "check the qualname and first line, or target the new name",
          "cost": "seconds"}],
        "nothing is installed for this target", [], False),
    "base_mismatch": (
        "the code this participant runs is not the base the patch was made against; "
        "installing replaces it with the patch's version",
        [{"action": "diff the running code against the plan's base and rebase the patch",
          "cost": "seconds"}, ACCEPT, RESTART],
        "changes that exist only in the running code are dropped, or this participant ends "
        "up different from the others", ["Proving an edit applied"], True),
    "outside_changes": (
        "the patch also changes code outside the target bodies (decorators, module or class "
        "statements, other functions); a code swap applies only the target bodies",
        [{"action": "add the other changed functions as targets", "cost": "none"}, REBUILD,
         ACCEPT],
        "the engine runs new target bodies with the old decorators and module state",
        ["Keeping state across edits"], True),
    "closure_changed": (
        "the new code captures different variables than the running closure; CPython cannot "
        "give a closure code with other free variables",
        [{"action": "edit and target the enclosing function so new closures are made",
          "cost": "none"}, REBUILD, RESTART],
        "nothing is installed for this target", ["Keeping state across edits"], False),
    "defaults_not_evaluated": (
        "the new default values could not be evaluated outside the class body; the old "
        "defaults are kept",
        [{"action": "move the default into the body (None sentinel)", "cost": "none"}, ACCEPT],
        "calls that rely on a changed default get the old value", [], True),
    "running_frame": (
        "a frame of this code is running now (a loop entered before the swap, or a "
        "suspended coroutine); it keeps executing the old bytecode until it returns",
        [{"action": "restart the loop's thread or process", "cost": "a restart today"},
         RESTART, ACCEPT],
        "the edit looks installed but the running loop never runs it",
        ["Proving an edit applied"], True),
    "zero_witness": (
        "the swapped code has not run since the swap",
        [{"action": "send work that takes this path", "cost": "seconds"},
         {"action": "if it runs only at startup: soft restart", "cost": "recipe R5"},
         {"action": "if it runs inside captured graphs: recapture", "cost": "recipe R1"}],
        "results are judged on code that never ran", ["Proving an edit applied"], True),
    "witness_uneven": (
        "ranks of one group (they run the same steps and switched in the same gap) counted "
        "different numbers of calls of the new code: a rank took another path, or did not "
        "run the swapped code for part of the workload",
        [{"action": "compare the ranks' per-step version logs and the code each rank runs",
          "cost": "seconds"},
         {"action": "accept if the code is rank-specific (runs on some ranks only)",
          "cost": "none"}],
        "a summed or single count hides a rank that never ran the edit",
        ["Proving an edit applied"], True),
    "deadline_missed": (
        "a participant did not answer a phase-2 call within the deadline; it is treated as "
        "failed and the others are undone in the gap, but its own call may still be running",
        [{"action": "respawn the worker group (reset kind respawn)", "cost": "respawn"},
         {"action": "undo the transaction, then respawn", "cost": "undo + respawn"}],
        "the late participant may still switch after the gap and run code the others do not",
        [], False),
    "repair_not_undoable": (
        "the repair cell changed state it did not declare, or changed a value in place; "
        "undo restores only the declared attributes",
        [{"action": "declare the objects the cell touches and run it again", "cost": "ms"},
         ACCEPT, RESTART],
        "undo leaves the undeclared change in place: the base is not restored exactly",
        ["Keeping state across edits"], True),
    "repair_failed": (
        "the repair cell raised; what it set before raising is undone",
        [{"action": "fix the cell", "cost": "seconds"}],
        "nothing is repaired", [], False),
    "participant_failed": (
        "a participant crashed or returned not-ok; the transaction cannot be all-or-nothing "
        "without undoing the others",
        [{"action": "inspect the participant (REPL) and retry", "cost": "seconds"}, RESTART],
        "participants run different code: every collective of the next step is suspect",
        [], False),
    "undo_cascade": (
        "the journal undoes back to a transaction's mark, so undoing an older transaction "
        "also undoes every transaction committed after it on that participant",
        [{"action": "accept: the later transactions are now undone (their state says so)",
          "cost": "none"},
         {"action": "re-run the later transactions after this undo", "cost": "seconds"}],
        "a later transaction looks committed while its code is gone",
        [], True),
    "removed_definition": (
        "the edit deletes a function, method or class that the live module still defines; a "
        "swap that only adds and replaces leaves it callable, so a forgotten caller keeps "
        "working live and fails after a restart",
        [{"action": "accept: delete it from the live module (undo puts it back)", "cost": "ms"},
         {"action": "leave it (dead code runs nowhere unless a caller was forgotten)",
          "cost": "none, but live results can hide the forgotten caller"}],
        "a live run passes while the restarted engine fails", [], True),
    "changed_since_prepare": (
        "the code a staged swap was built against changed between prepare and the gap "
        "(another transaction or a REPL edit); committing would overwrite that change",
        [{"action": "prepare again against the current code", "cost": "one prepare"}],
        "the other change is lost silently, and undo would restore a base that no longer "
        "was the base", ["Proving an edit applied"], False),
    "reset_failed": (
        "the data reset the undo plan declares did not run; undo restored the code, but data "
        "the new code wrote (requests, cache entries) stays",
        [{"action": "run the reset by hand (REPL) or add the hook", "cost": "ms to seconds"},
         RESTART],
        "later results read data written by code that is no longer installed", [], False),
    "empty_change": (
        "the plan changes no definition, adds or deletes none and runs no repair cell, so "
        "committing it would report success while the engine runs the same code",
        [{"action": "check that the edit reached the planner (file saved, right module)",
          "cost": "seconds"},
         {"action": "if the edit is outside function bodies: soft or warm restart",
          "cost": "recipe R5 or R9"}],
        "a silent no-op install reads as a success", ["Proving an edit applied"], False),
    "tampered": (
        "a participant runs code that no committed transaction installed (a REPL or another "
        "tool changed it outside the journal), or it did not answer",
        [{"action": "inspect the participant (REPL) and respawn its worker group",
          "cost": "respawn"}, RESTART],
        "undo cannot restore a base it never recorded; later results are not comparable",
        [], False),
    "session_poisoned": (
        "an earlier fault left the session in a state the host cannot prove (crash, hang, "
        "lost participant, failed undo, code changed outside the journal)",
        [{"action": "respawn the worker group (reset kind respawn)", "cost": "respawn"},
         RESTART],
        "a change installed now would be measured against an unknown base", [], False),
    "analyze_fallback": (
        "prepare's text half (parse, compile, compare) was meant to run in the analyzer's "
        "child process but ran in this one: the child could not start, died or did not "
        "answer. The staged code is the same; the cost is that this process held its GIL "
        "for the whole analysis, so the engine loop could not step meanwhile",
        [{"action": "read the reason (the child's last stderr line) and fix the analyzer, "
          "e.g. a kernel the child cannot load", "cost": "seconds"},
         {"action": "accept: analyze in-process", "cost": "the GIL time in `observed`, on "
          "every prepare until fixed"}],
        "every prepare stalls the engine loop for its parse and compare, silently "
        "(tens of ms per module on a 3,800-line file)", [], True),
    "undo_incomplete": (
        "an inverse failed while undoing; the participant is not back at its base",
        [RESTART], "the base is not restored; later results are not comparable", [], False),
}


def finding(check, observed, expected, subject=None):
    """One finding of a catalogued check."""
    why, actions, risk, lessons, overridable = CATALOG[check]
    f = {"check": check, "observed": str(observed), "expected": str(expected), "why": why,
         "actions": [dict(a) for a in actions], "risk_if_ignored": risk,
         "lessons": list(lessons), "overridable": overridable}
    if subject is not None:
        f["subject"] = subject
    return f


def _matches(rec, f):
    if rec.get("check") != f["check"]:
        return False
    if "subject" in rec and f.get("subject") != rec["subject"]:
        return False
    if rec.get("scope", "txn") == "standing":
        for k in ("participant", "code_hash"):
            if k not in rec or f.get(k) != rec[k]:
                return False
    return True


def decide(findings, accept=()):
    """Proceed only if every finding is accepted. A finding is accepted when it is overridable
    and `accept` is "all", names its check, or holds an accept record matching it. Returns
    {"proceed": bool, "accepted": [check], "blocking": [check], "accepts": [record used],
     "findings": [annotated]}. Records for name-only accepts have by/reason None."""
    recs = None if accept == "all" else [
        a if isinstance(a, dict) else {"check": a, "scope": "txn", "by": None, "reason": None}
        for a in (accept or ())]
    out, accepted, blocking, used = [], [], [], []
    for f in findings:
        rec = None if recs is None else next((r for r in recs if _matches(r, f)), None)
        ok = f["overridable"] and (recs is None or rec is not None)
        out.append({**f, "accepted": ok})
        (accepted if ok else blocking).append(f["check"])
        if ok and rec is not None and rec not in used:
            used.append(rec)
    return {"proceed": not blocking, "accepted": sorted(set(accepted)),
            "blocking": sorted(set(blocking)), "accepts": used, "findings": out}


def ok(**kw):
    return {"ok": True, "findings": [], **kw}


def not_ok(error, findings=(), **kw):
    return {"ok": False, "error": str(error), "findings": list(findings), **kw}
