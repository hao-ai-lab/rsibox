"""Fault injection into one participant process (conformance harness only, never on a user's job).

`inject(name, fault, **kw)` runs *inside* the target process (a Ray actor through `__ray_call__`,
or the driver itself). `STATE` is that process's fault flags; the gate and the toy cluster read it
at their step boundary (`crash`, `hang_s`) and RPC entry (`lost`).
"""
import os, sys

STATE = {}


def inject(name, fault, **kw):
    import ray_participants as RP
    from kernel import coordinator as C
    local = RP._LOCAL.get(name) or RP._LOCAL.setdefault(name, C.LocalParticipant(name))
    if fault == "fail_apply":
        journal, after, cnt, orig = local.journal, int(kw.get("after", 1)), [0], local.journal.set_code
        def set_code(*a, **k):
            cnt[0] += 1
            if cnt[0] > after:
                del journal.set_code
                raise RuntimeError(f"{name}: injected fault after {after}")
            return orig(*a, **k)
        journal.set_code = set_code
    elif fault == "fail_commit":
        orig_q = local.quiesce
        def failing_q(_fn):
            local.quiesce = orig_q
            raise RuntimeError(f"{name}: injected commit failure")
        local.quiesce = failing_q
    elif fault in ("crash", "lost"):
        STATE[fault] = True
    elif fault == "hang":
        STATE["hang_s"] = float(kw.get("hang_s", 5.0))
    elif fault == "tamper":
        fn = getattr(sys.modules[kw["module"]], kw["function"])
        fn.__code__ = fn.__code__.replace(co_name=fn.__code__.co_name)   # same text, new code object: outside the journal
    else:
        return {"ok": False, "error": f"unknown fault {fault}"}
    return {"ok": True, "fault": fault, "participant": name, "pid": os.getpid()}
