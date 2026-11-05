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

One accept mechanism: a list of check names (or "all"). `decide(findings, accept)` says
whether to proceed: every finding must be accepted, and only overridable ones can be.
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


def decide(findings, accept=()):
    """Proceed only if every finding is accepted. A finding is accepted when its check is in
    `accept` (or accept == "all") and it is overridable. Returns
    {"proceed": bool, "accepted": [check], "blocking": [check], "findings": [annotated]}."""
    names = None if accept == "all" else set(accept or ())
    out, accepted, blocking = [], [], []
    for f in findings:
        ok = f["overridable"] and (names is None or f["check"] in names)
        out.append({**f, "accepted": ok})
        (accepted if ok else blocking).append(f["check"])
    return {"proceed": not blocking, "accepted": sorted(set(accepted)),
            "blocking": sorted(set(blocking)), "findings": out}


def ok(**kw):
    return {"ok": True, "findings": [], **kw}


def not_ok(error, findings=(), **kw):
    return {"ok": False, "error": str(error), "findings": list(findings), **kw}
