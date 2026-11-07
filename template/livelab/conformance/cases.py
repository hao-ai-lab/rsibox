"""The conformance cases c0-c10. Each drives a lab only through `LabUnderTest` (lab.py).

A case is a function `case(lab) -> [evidence lines]` that raises `CaseFailed` when the lab
does not show the property. Each docstring says what passing proves and what it does not.
"""
import time

REFUSALS = {                       # refusal class -> the code the lab must report
    "syntax": "patch_invalid",
    "closure": "closure_changed",
    "module_state": "outside_changes",
    "empty": "empty_change",
}


class CaseFailed(AssertionError):
    pass


def expect(cond, what, got=None):
    if not cond:
        raise CaseFailed(what + ("" if got is None else f"; got {got!r}"))


def _short(out):
    """One rank's output when every rank agrees (the usual case), else all of them."""
    if isinstance(out, dict) and out and len({str(v) for v in out.values()}) == 1:
        return f"{next(iter(out.values()))} x{len(out)} ranks"
    return str(out)[:80]


def _calls(witness):
    """{key: [calls per participant]}"""
    out = {}
    for w in witness.values():
        for k, n in w.items():
            out.setdefault(k, []).append(n)
    return out


def _base(lab):
    st = lab.status()
    expect(st["ok"] and st["state"] == "base", "lab starts at base", st.get("state"))
    c = lab.check()
    expect(c["ok"], "base check runs", c.get("error"))
    return c["output"], st["fingerprint"]


def _at_base(lab, out0, fp0, when):
    st = lab.status()
    expect(st["state"] == "base", f"status base {when}", (st["state"], st.get("poison")))
    expect(st["fingerprint"] == fp0, f"code identity equals base {when}",
           (st["fingerprint"], fp0))
    c = lab.check()
    expect(c["ok"] and c["output"] == out0, f"output equals base {when}", _short(c["output"]))


def _installed(lab, patch, what):
    r = lab.install(patch)
    expect(r["ok"] and r["outcome"] == "committed", f"{what} installs",
           (r["outcome"], r["codes"]))
    return r


def c0_repeatable(lab):
    """Passes only if: the workload gives the same output when nothing changes (3 checks,
    one after reset). Proves: comparisons later in the suite are meaningful. Not: that the
    output is right."""
    out0, fp0 = _base(lab)
    outs = [out0]
    for i in range(2):
        if i == 1:
            r = lab.reset()
            expect(r["ok"], "reset succeeds", r.get("error"))
        c = lab.check()
        expect(c["ok"], "check runs", c.get("error"))
        outs.append(c["output"])
    expect(all(o == out0 for o in outs), "3 checks give one output", [_short(o) for o in outs])
    expect(lab.status()["fingerprint"] == fp0, "checks change no code")
    return [f"3 checks (one after reset {r['reset']}) gave {_short(out0)}"]


def c1_noop(lab):
    """Passes only if: an edit that changes code but keeps behavior installs, runs (witness),
    and leaves output bit-exact. Proves: the install itself perturbs nothing. Not: that the
    edit really keeps behavior on other inputs."""
    out0, fp0 = _base(lab)
    r = _installed(lab, lab.patches()["noop"], "no-op edit")
    st = lab.status()
    expect(st["state"] == "edited" and st["fingerprint"] != fp0,
           "status edited and code identity changed", (st["state"], st["fingerprint"]))
    c = lab.check()
    expect(c["proven"] is True, "the no-op code ran during the check", c["witness"])
    expect(c["output"] == out0, "output bit-exact vs base", _short(c["output"]))
    expect(lab.uninstall()["ok"], "uninstall succeeds")
    return [f"installed by {r['mechanisms']}; calls {_calls(c['witness'])}; output unchanged"]


def c2_visible(lab):
    """Passes only if: a visible edit changes the output and its witness fires on every
    participant. Proves: install reaches the running code. Not: that the new output is
    the one the edit intended (c3 does)."""
    out0, _ = _base(lab)
    _installed(lab, lab.patches()["visible"], "visible edit")
    c = lab.check()
    expect(c["output"] != out0, "output differs from base", _short(c["output"]))
    expect(c["proven"] is True, "proven: every live edit ran", c["findings"])
    counts = [n for w in c["witness"].values() for n in w.values()]
    expect(len(c["witness"]) >= 2 and counts and min(counts) > 0,
           "witness > 0 on every participant", c["witness"])
    expect(lab.uninstall()["ok"], "uninstall succeeds")
    return [f"output {_short(out0)} -> {_short(c['output'])}",
            f"calls per participant {_calls(c['witness'])}"]


def c3_fresh_equivalence(lab):
    """Passes only if: the live output after an edit (and after a chain of 3) equals a new
    process started from the edited files, and the fresh base equals the live base.
    Proves: a live edit runs what a restart would. Not: equality on other workloads."""
    out0, _ = _base(lab)
    fb = lab.fresh_check(None)
    expect(fb["ok"] and fb["output"] == out0, "fresh base equals live base", fb)
    P = lab.patches()
    _installed(lab, P["visible"], "visible edit")
    live = lab.check()["output"]
    fresh = lab.fresh_check(P["visible"])
    expect(live != out0, "the edit is visible", _short(live))
    expect(fresh["ok"] and fresh["output"] == live, "fresh equals live (visible)",
           (_short(fresh["output"]), _short(live)))
    lab.uninstall()
    merged = {}
    for p in P["chain"]:
        _installed(lab, p, "chain edit")
        merged.update(p["files"])
    live_c = lab.check()["output"]
    fresh_c = lab.fresh_check({"files": merged})
    expect(fresh_c["ok"] and fresh_c["output"] == live_c, "fresh equals live (chain of 3)",
           (_short(fresh_c["output"]), _short(live_c)))
    lab.uninstall(lab.status()["txns"][0])
    return [f"visible: live == fresh == {_short(live)}",
            f"chain of 3: live == fresh == {_short(live_c)}"]


def c4_uninstall(lab):
    """Passes only if: after install and uninstall, status is base, the code identity
    (content hash and object identity of every engine function) equals base, and output
    equals base. Proves: undo is exact for code. Not: data the edit wrote (undo_resets)."""
    out0, fp0 = _base(lab)
    _installed(lab, lab.patches()["visible"], "visible edit")
    expect(lab.check()["output"] != out0, "the edit is visible")
    u = lab.uninstall()
    expect(u["ok"] and u["state"] == "base", "uninstall returns to base", u)
    _at_base(lab, out0, fp0, "after uninstall")
    c = lab.check()
    expect(c["proven"] is None and c["witness"] == {}, "no edit live, no witness", c["witness"])
    return [f"uninstall {u['undone']} in {u['cost_s']} s; identity {fp0} restored"]


def c5_failed_install(lab):
    """Passes only if: a fault in the middle of installing an edit of >= 3 functions in
    >= 2 modules (one participant fails after 2 swaps; another's commit fails) leaves
    status base, base code identity and base output, and says why. Proves: no partial
    install survives a fault. Not: a participant that dies for good (c8)."""
    out0, fp0 = _base(lab)
    multi = lab.patches()["multi"]
    lines = []
    for fault, kw in (("fail_apply", {"after": 2}), ("fail_commit", {})):
        expect(lab.inject(fault, **kw)["ok"], f"inject {fault}")
        r = lab.install(multi)
        expect(not r["ok"] and r["outcome"] in ("rolled_back", "aborted"),
               f"{fault}: install fails", (r["outcome"], r["codes"]))
        expect("participant_failed" in r["codes"], f"{fault}: says a participant failed",
               r["codes"])
        _at_base(lab, out0, fp0, f"after {fault}")
        lines.append(f"{fault}: {r['outcome']}, codes {r['codes']}, base proven")
    r = _installed(lab, multi, "the same edit without a fault")
    c = lab.check()
    keys = {k for w in c["witness"].values() for k in w}
    expect(len(keys) >= 3 and len({k.split(":")[0] for k in keys}) >= 2 and c["proven"],
           "the edit really spans >= 3 functions in >= 2 modules, all ran", sorted(keys))
    expect(c["output"] != out0, "and is visible")
    lab.uninstall()
    lines.append(f"control: no fault, committed {len(keys)} targets, output changed")
    return lines


def c6_refusal(lab):
    """Passes only if: each unsupported edit (syntax error, changed closure, module state,
    an edit that changes no code) is refused with its code, and status, code identity and
    output stay base. Proves: refusals are safe and say why. Not: every unsupported class."""
    out0, fp0 = _base(lab)
    lines = []
    for cls, code in REFUSALS.items():
        r = lab.install(lab.patches()["refused"][cls])
        expect(not r["ok"] and r["outcome"] in ("aborted", "refused"),
               f"{cls}: refused", (r["outcome"], r["codes"]))
        expect(code in r["codes"], f"{cls}: code {code}", r["codes"])
        if cls == "empty":
            expect(r["codes"] == ["empty_change"], "empty: only empty_change", r["codes"])
        _at_base(lab, out0, fp0, f"after {cls}")
        lines.append(f"{cls}: {r['codes']}")
    return lines


def c7_chain(lab):
    """Passes only if: 3 edits installed on top of each other (two on the same function)
    each change the output and run, and undoing them one by one returns each earlier
    output exactly, ending at base output and code identity. Proves: stacked edits and
    their undo compose. Not: edits that move a target's first line."""
    out0, fp0 = _base(lab)
    outs = [out0]
    for i, p in enumerate(lab.patches()["chain"]):
        _installed(lab, p, f"chain edit {i + 1}")
        c = lab.check()
        expect(c["proven"] is True and c["output"] != outs[-1],
               f"chain edit {i + 1} runs and changes output", (c["proven"], _short(c["output"])))
        outs.append(c["output"])
    for i in range(len(outs) - 1, 0, -1):
        u = lab.uninstall()
        expect(u["ok"], f"uninstall edit {i}", u)
        c = lab.check()
        expect(c["output"] == outs[i - 1], f"after undoing edit {i}: output of step {i - 1}",
               _short(c["output"]))
    _at_base(lab, out0, fp0, "after the chain is undone")
    return [" -> ".join(str(o[next(iter(o))][:3])[:-1] + ", ...]" for o in outs)
            + " -> undone one by one to base"]


def c8_crash_and_hang(lab):
    """Passes only if: a crash, a hang, a lost participant and code changed outside the
    journal each leave status poisoned, a poisoned lab refuses install with
    session_poisoned, and a respawn returns to base output. Proves: faults are never
    silent. Not: recovery of real GPU state (sticky CUDA errors)."""
    out0, _ = _base(lab)
    visible = lab.patches()["visible"]
    lines = []
    for fault in ("crash", "hang", "lost", "tamper"):
        expect(lab.inject(fault)["ok"], f"inject {fault}")
        c = lab.check()
        st = lab.status()
        expect(st["state"] == "poisoned", f"{fault}: status poisoned",
               (st["state"], c.get("error")))
        r = lab.install(visible)
        expect(not r["ok"] and r["codes"] == ["session_poisoned"],
               f"{fault}: install refused while poisoned", r["codes"])
        rs = lab.reset("respawn")
        expect(rs["ok"], f"{fault}: respawn", rs)
        st2 = lab.status()
        c2 = lab.check()
        expect(st2["state"] == "base" and c2["output"] == out0,
               f"{fault}: base after respawn", (st2["state"], _short(c2["output"])))
        lines.append(f"{fault}: poisoned by {[f['check'] for f in st['poison']]}, "
                     f"respawn {rs['cost_s']} s")
    return lines


def c9_soak(lab, n=20):
    """Passes only if: 20 cycles of install, check, uninstall, check give the edited output
    and then the base output exactly every time, and end at base code identity. Proves: no
    drift or leak across repeated edits. Not: thousands of cycles or memory growth."""
    out0, fp0 = _base(lab)
    visible = lab.patches()["visible"]
    first, ti, tu = None, [], []
    for i in range(n):
        r = _installed(lab, visible, f"cycle {i}")
        ti.append(r["cost_s"])
        c = lab.check()
        first = first or c["output"]
        expect(c["proven"] and c["output"] == first != out0, f"cycle {i}: edited output",
               _short(c["output"]))
        u = lab.uninstall()
        tu.append(u["cost_s"])
        expect(u["ok"] and lab.check()["output"] == out0, f"cycle {i}: base output")
    _at_base(lab, out0, fp0, f"after {n} cycles")
    return [f"{n} cycles; install median {sorted(ti)[n // 2]:.4f} s, "
            f"uninstall median {sorted(tu)[n // 2]:.4f} s"]


def c10_not_proven(lab):
    """Passes only if: an edit the workload never runs as Python (startup-only code) installs
    but its check is not proven (proven False, a zero_witness finding), while a fresh
    process shows the edit would change the output. Proves: an unrun edit never reads as a
    pass. Not: graph-captured code (no graphs on the toy)."""
    out0, _ = _base(lab)
    P = lab.patches()
    _installed(lab, P["unreached"], "startup-only edit")
    c = lab.check()
    checks = [f["check"] for f in c["findings"]]
    expect(c["proven"] is False, "proven is False, not True or None", c["proven"])
    expect("zero_witness" in checks, "a zero_witness finding", checks)
    fresh = lab.fresh_check(P["unreached"])
    expect(fresh["ok"] and fresh["output"] != c["output"],
           "a fresh process shows the edit matters", _short(fresh["output"]))
    lab.uninstall()
    return [f"proven {c['proven']}, findings {checks}",
            f"live {_short(c['output'])} (base: {c['output'] == out0}); "
            f"fresh {_short(fresh['output'])}"]


CASES = {f.__name__: f for f in (
    c0_repeatable, c1_noop, c2_visible, c3_fresh_equivalence, c4_uninstall,
    c5_failed_install, c6_refusal, c7_chain, c8_crash_and_hang, c9_soak, c10_not_proven)}

# Cases a lab is known to fail, with the reason: {lab name: {case: reason}}. A listed case
# that passes is reported as an unexpected pass. Empty: the toy lab passes every case.
XFAIL = {"toy": {}}


def run(name, make_lab):
    """Run one case on a new lab. Returns {"case", "result", "seconds", "evidence", "error",
    "xfail_reason"}."""
    lab, t0 = make_lab(), time.perf_counter()
    reason = XFAIL.get(lab.name, {}).get(name)
    try:
        ev, err = CASES[name](lab), None
    except Exception as e:            # CaseFailed, or a lab that raised
        ev, err = [], f"{type(e).__name__}: {e}"
    finally:
        lab.close()
    if reason:
        result = "xfail" if err else "XPASS"
    else:
        result = "pass" if err is None else "FAIL"
    return {"case": name, "result": result, "seconds": round(time.perf_counter() - t0, 3),
            "evidence": ev, "error": err, "xfail_reason": reason}
