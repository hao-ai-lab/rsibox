"""Fault injection into one participant process (conformance harness only, never on a user's job).

`inject(name, fault, **kw)` runs *inside* the target process (a Ray actor through `__ray_call__`,
or the driver itself). `STATE` is that process's fault flags; the gate and the toy cluster read it
at their step boundary (`crash`, `hang_s`) and RPC entry (`lost`).
"""
import os

from fastrsi import harness

STATE = {}


def inject(name, fault, **kw):
    if fault in ("crash", "lost"):
        STATE[fault] = True
        return {"ok": True, "fault": fault, "participant": name, "pid": os.getpid()}
    if fault == "hang":
        STATE["hang_s"] = float(kw.get("hang_s", 5.0))
        return {"ok": True, "fault": fault, "participant": name, "pid": os.getpid()}
    if fault in ("fail_apply", "fail_commit", "tamper"):
        import ray_participants as RP
        from kernel import coordinator as C

        local = RP._LOCAL.get(name) or RP._LOCAL.setdefault(name, C.LocalParticipant(name))
        res = harness.inject_participant_fault(local, fault, **kw)
        return {**res, "pid": os.getpid()} if res.get("ok") else res
    return {"ok": False, "error": f"unknown fault {fault}"}
