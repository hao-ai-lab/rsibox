"""Conformance fault injection on a scheduler rank (test harness only; never production)."""
import os
import time

from rsibox import harness as H


def _install_forward_fault(scheduler, fault, hang_s, rank_name):
    runner = scheduler.tp_worker.model_runner
    attr = "sample" if hasattr(runner, "sample") else "forward"
    original_fn = getattr(runner, attr)

    def wrapped_fn(*args, **kwargs):
        setattr(runner, attr, original_fn)
        if fault == "crash":
            raise RuntimeError(f"{rank_name}: injected crash in {attr}")
        time.sleep(hang_s)
        return original_fn(*args, **kwargs)

    setattr(runner, attr, wrapped_fn)


def inject(scheduler, local_participant, state_dict, fault, **kwargs):
    if fault in ("fail_apply", "fail_commit", "tamper"):
        reply = H.inject_participant_fault(local_participant, fault, **kwargs)
        return {**reply, "pid": os.getpid()} if reply.get("ok") else reply
    if fault == "fail_recapture":
        state_dict["faults"]["fail_recapture"] = True
    elif fault in ("crash", "hang"):
        _install_forward_fault(scheduler, fault, float(kwargs.get("hang_s", 5.0)), local_participant.name)
    elif fault == "lost":
        state_dict["faults"]["lost"] = True
    else:
        return {"ok": False, "error": f"unknown fault {fault}"}
    return {"ok": True, "fault": fault, "participant": local_participant.name, "pid": os.getpid()}
