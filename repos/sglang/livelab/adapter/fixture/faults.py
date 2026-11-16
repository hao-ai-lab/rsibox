"""Conformance fault injection on a scheduler rank (test harness only; never production)."""
import os
import sys
import time


def _install_forward_fault(scheduler, state_dict, fault, hang_s):
    runner = scheduler.tp_worker.model_runner
    rank_num = scheduler.tp_rank if hasattr(scheduler, "tp_rank") else runner.tp_rank
    rank_name = f"rank{rank_num}"
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
    if fault == "fail_apply":
        journal, after, counter = local_participant.journal, kwargs.get("after", 1), [0]
        original_set_code = journal.set_code

        def failing_set_code(*args, **kw):
            counter[0] += 1
            if counter[0] > after:
                del journal.set_code
                raise RuntimeError(f"{local_participant.name}: injected fault after {after} swap(s)")
            return original_set_code(*args, **kw)

        journal.set_code = failing_set_code
    elif fault == "fail_commit":
        original_quiesce = local_participant.quiesce

        def failing_quiesce(_fn):
            local_participant.quiesce = original_quiesce
            raise RuntimeError(f"{local_participant.name}: injected commit failure")

        local_participant.quiesce = failing_quiesce
    elif fault == "fail_recapture":
        state_dict["faults"]["fail_recapture"] = True
    elif fault in ("crash", "hang"):
        _install_forward_fault(scheduler, state_dict, fault, float(kwargs.get("hang_s", 5.0)))
    elif fault == "lost":
        state_dict["faults"]["lost"] = True
    elif fault == "tamper":
        target_obj = sys.modules[kwargs["module"]]
        for attr_part in kwargs["function"].split("."):
            target_obj = getattr(target_obj, attr_part)
        target_obj.__code__ = target_obj.__code__.replace(co_name=target_obj.__code__.co_name)
    else:
        return {"ok": False, "error": f"unknown fault {fault}"}
    return {"ok": True, "fault": fault, "participant": local_participant.name, "pid": os.getpid()}
