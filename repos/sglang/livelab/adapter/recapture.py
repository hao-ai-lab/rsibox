"""Selective decode-graph recapture (recipe R1) on one SGLang scheduler rank."""
import time
from kernel import findings as F

def _bs_of_key(key):
    if isinstance(getattr(key, "size", None), int): return key.size
    tail = key if isinstance(key, int) else str(key).rsplit("_", 1)[-1]
    return tail if isinstance(tail, int) else (int(tail) if tail.isdigit() else None)

def _graph_dicts(runner):
    return (runner.graphs, runner.output_buffers) if hasattr(runner, "graphs") else (runner.backend._graphs, runner.backend._outputs)

def _sync(runner):
    fn = getattr(getattr(runner, "device_module", None), "synchronize", None)
    if callable(fn): fn()

def _capture_chosen(runner, chosen, merge_with, has_max, restore_fn, witness):
    started = time.perf_counter()
    try:
        if chosen:
            runner.capture_bs = list(chosen)
            with witness.capture():
                runner.capture()
        runner.capture_bs = sorted(set(merge_with) | set(chosen))
        if has_max: runner.max_bs = max(runner.capture_bs, default=0)
        _sync(runner)
    except Exception as exc:
        restore_fn()
        _sync(runner)
        finding = F.finding("rebuild_failed", f"{chosen}: {type(exc).__name__}: {exc}", "every size captured")
        return F.not_ok(f"recapture: {type(exc).__name__}: {exc}", [finding], captured=[])
    return F.ok(seconds=round(time.perf_counter() - started, 6))

class Recapture:
    def __init__(self):
        self.live = {}
    def run(self, runner, txn, journal, witness, sizes=None):
        if runner is None or not getattr(runner, "capture_bs", None): return F.ok(skipped="this worker runs no decode graphs", captured=[], seconds=0.0)
        if txn in self.live or not journal.marks or journal.marks[-1][0] != txn: return F.not_ok(f"cannot recapture {txn}: already recaptured or not newest")
        gmap, bmap = _graph_dicts(runner)
        old_bs, old_g, old_b = list(runner.capture_bs), dict(gmap), dict(bmap)
        has_max, old_max = hasattr(runner, "max_bs"), getattr(runner, "max_bs", 0)
        wanted = None if sizes is None else {int(s) for s in sizes}
        chosen, left = [b for b in old_bs if wanted is None or b in wanted], [b for b in old_bs if wanted and b not in wanted]
        def restore():
            runner.capture_bs = list(old_bs); gmap.clear(); gmap.update(old_g); bmap.clear(); bmap.update(old_b)
            if has_max: runner.max_bs = old_max
        gmap.clear(); bmap.clear()
        res = _capture_chosen(runner, chosen, (), has_max, restore, witness)
        if not res["ok"]: return res
        journal.record_call(restore, f"R1 decode graphs of {txn}")
        self.live[txn] = {"old_bs": old_bs, "has_max": has_max, "left": left}
        journal.record_call(lambda: self.live.pop(txn, None), f"R1 state of {txn}")
        return F.ok(captured=list(chosen), eager=list(left), graphs=len(gmap), seconds=res["seconds"])
    def extend(self, runner, txn, journal, witness, sizes=None, extend_id=None):
        entry = self.live.get(txn)
        if entry is None or runner is None or not journal.marks or journal.marks[-1][0] != txn: return F.not_ok(f"cannot extend recapture for {txn}")
        entry.pop("extend", None)
        gmap, bmap = _graph_dicts(runner)
        wanted = None if sizes is None else {int(s) for s in sizes}
        chosen = [bs for bs in entry["left"] if wanted is None or bs in wanted]
        prev = (list(runner.capture_bs), dict(gmap), dict(bmap), getattr(runner, "max_bs", 0))
        def restore_prev():
            runner.capture_bs = prev[0]; gmap.clear(); gmap.update(prev[1]); bmap.clear(); bmap.update(prev[2])
            if entry["has_max"]: runner.max_bs = prev[3]
        res = _capture_chosen(runner, chosen, prev[0], entry["has_max"], restore_prev, witness)
        if not res["ok"]: return res
        entry["left"] = [bs for bs in entry["left"] if bs not in chosen]
        def unextend():
            for key in [k for k in list(gmap) if _bs_of_key(k) in set(chosen)]:
                gmap.pop(key, None); bmap.pop(key, None)
            runner.capture_bs = [bs for bs in runner.capture_bs if bs not in chosen]
            if entry["has_max"]: runner.max_bs = max(runner.capture_bs, default=0)
            entry["left"] = [bs for bs in entry["old_bs"] if bs not in runner.capture_bs]
        journal.record_call(unextend, f"R1 extend of {txn}")
        if extend_id is not None: entry["extend"] = (extend_id, journal.entries[-1], unextend)
        return F.ok(captured=list(chosen), eager=list(entry["left"]), seconds=res["seconds"])
    def drop_extend(self, txn, journal, extend_id):
        last = (self.live.get(txn) or {}).get("extend")
        if last is None or last[0] != extend_id: return F.ok(dropped=False)
        _, jentry, unextend = self.live[txn].pop("extend")
        if not journal.entries or journal.entries[-1] is not jentry: return F.not_ok(f"extend {extend_id} of {txn} is no longer the newest entry")
        journal.entries.pop(); unextend()
        return F.ok(dropped=True)

def recapture(state_holder, runner, txn, journal, witness, sizes=None):
    return state_holder.run(runner, txn, journal, witness, sizes=sizes)
