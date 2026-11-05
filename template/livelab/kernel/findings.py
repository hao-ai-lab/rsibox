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
    "line_aliased": (
        "an earlier swap changed the module's line numbers and no recorded line map leads from "
        "this target's line in the running text to live code, so the one live function with "
        "the target's qualname was taken as it; the content check against the base text still "
        "compares the code",
        [{"action": "check the swapped function is the intended one (the finding's subject "
                    "and observed lines)", "cost": "seconds"}, ACCEPT],
        "a qualname that names another definition than the target's would be swapped",
        ["Proving an edit applied"], True),
    "running_frame": (
        "a frame of this code is running now (a loop entered before the swap, or a "
        "suspended coroutine); it keeps executing the old bytecode until it returns",
        [{"action": "restart the loop's thread or process", "cost": "a restart today"},
         RESTART, ACCEPT],
        "the edit looks installed but the running loop never runs it",
        ["Proving an edit applied"], True),
    "captured_target": (
        "the engine runs this code only inside captured graphs or compiled code (the engine "
        "says so from its config, the target's module and the ops that split its graphs): a "
        "swap changes the Python function, but the captured or compiled steps never call it, "
        "so the edit commits and never runs",
        [{"action": "recapture the graphs with the commit (plan \"rebuild\": [\"R1\"]; an "
                    "accepted finding with route R1 does it on a host that can)",
          "cost": "recipe R1"},
         {"action": "run the engine eager for this experiment", "cost": "a restart"},
         {"action": "accept if the code runs outside the graphs (the finding says when the "
                    "engine could not decide); the proof's witness will tell", "cost": "none"}],
        "an edit that never runs reads as an edit that changed nothing",
        ["Proving an edit applied"], True),
    "captured_partly": (
        "the engine runs this code in Python on some steps (prefill and mixed batches: through "
        "an op that splits the graph, or with no graph) and only inside captured graphs on "
        "others (decode batches in full graphs): the new code runs on the first kind now, and "
        "on the second only after a recapture",
        [{"action": "recapture with the commit (plan \"rebuild\": [\"R1\"]) to run it on every "
                    "step", "cost": "recipe R1"},
         {"action": "go on: prefill steps run the new code now, decode steps the old one",
          "cost": "none"}],
        "a decode-heavy result mixes old and new code", ["Proving an edit applied"], True),
    "rebuild_failed": (
        "the rebuild the transaction needs right after its commit (R1: capture the graphs "
        "again) failed, or the host cannot run it: graphs would replay the old code, or be "
        "missing. The transaction is undone and the rebuild runs again from the base code",
        [{"action": "read the rebuild's error (per rank), fix it, run the transaction again",
          "cost": "seconds"}, RESTART],
        "graphs and Python run different code: results and step times are not comparable",
        [], False),
    "zero_witness": (
        "the swapped code has not run since the swap",
        [{"action": "send work that takes this path", "cost": "seconds"},
         {"action": "if it runs only at startup: soft restart", "cost": "recipe R5"},
         {"action": "if it runs inside captured graphs: recapture", "cost": "recipe R1"}],
        "results are judged on code that never ran", ["Proving an edit applied"], True),
    "ranks_disagree": (
        "the ranks of one group run different code right after the commit: their code hashes "
        "differ for the same function (something changed a rank outside this transaction, or a "
        "rank did not switch), or only some ranks hold a function. Not overridable when the "
        "same function differs",
        [{"action": "read commit_hashes.ranks to find the rank that differs; undo this "
                    "transaction, or undo the other change on that rank", "cost": "seconds"},
         {"action": "respawn the worker group", "cost": "respawn"},
         {"action": "accept only when the ranks hold different functions by design (no "
                    "function differs)", "cost": "none"}],
        "ranks of one tensor-parallel group compute different results or issue different "
        "collectives: wrong outputs or a hang",
        ["Proving an edit applied"], True),
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
    "closure_not_restored": (
        "undo restored the swapped functions, but function objects their new code created "
        "while installed (a closure it returned, a callback it registered) run a nested "
        "code with no base version to go back to: a nested definition the edit added, or "
        "one with other free variables",
        [{"action": "find the holders (REPL: functions whose __qualname__ is the subject) and "
          "rebuild or drop them", "cost": "seconds"}, RESTART],
        "whoever holds those closures keeps running the undone edit's code",
        ["Keeping state across edits"], False),
    "undo_incomplete": (
        "an inverse failed while undoing; the participant is not back at its base",
        [RESTART], "the base is not restored; later results are not comparable", [], False),
    "jit_rebuild": (
        "the target is the Python function inside a JIT object (a Triton @jit kernel): the "
        "engine runs what the JIT compiled from the source text, never the function's "
        "bytecode, so a code swap alone leaves the GPU on the old kernel. The swap rebuilds "
        "the JIT object in place from the new text (journaled: undo restores the old compiled "
        "kernels), resets the module's JIT objects that call it, and counts launches as the "
        "witness",
        [{"action": "go on: the next launch compiles the new kernel", "cost": "one JIT "
          "compile per specialization (seconds), on the first launch after the commit"}],
        "none if the rebuild runs; kernels in other modules that inline this one keep the old "
        "compiled code", ["Proving an edit applied"], True),
    "new_fields": (
        "the new code of a method reads an attribute that only the edited __init__ sets; "
        "every live instance was built by the old __init__ and lacks it, so the first call of "
        "the new code raises AttributeError (a live run killed the engine this way)",
        [{"action": "accept with a backfill (plan \"backfill\": true): run the new "
          "__init__'s assignment of each field on the live instances found from the "
          "participant's scope (journaled: undo removes the fields)", "cost": "ms"},
         {"action": "a repair cell that sets the fields (journal.record_attr)", "cost": "ms"},
         RESTART],
        "the engine dies at the first step that runs the new code",
        ["Keeping state across edits"], True),
    "participant_missing": (
        "a participant the host expects (a rank of a worker group, a data-parallel engine) "
        "sent no reply: the transaction, its witness, receipt and undo would cover only the "
        "ranks that answered, while the others run whatever they ran before",
        [{"action": "reach every rank (every data-parallel group's workers), or run the "
          "engine with one data-parallel group", "cost": "a different launch"}, RESTART],
        "some ranks run the edit and others do not, and the receipt says nothing about them",
        [], False),
    "no_witness_possible": (
        "the transaction swapped no code, so no counter can prove it ran: it ran only repair "
        "cells, and none armed a witness",
        [{"action": "arm a counter from the cell (`hook = arm(key)`, call hook() where the "
          "changed thing runs, e.g. a Triton pre_run_hook)", "cost": "none"},
         {"action": "prove it by an output change of the workload", "cost": "seconds"}],
        "a cell that changed nothing reads as a pass", ["Proving an edit applied"], True),
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
        # advisory: reported, never blocks (e.g. a frame seen while steps ran; the gap decides)
        ok = bool(f.get("advisory")) or (f["overridable"] and (recs is None or rec is not None))
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
