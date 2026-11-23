"""The toy live lab: the kernel's toy engine behind the interface verbs, on CPU.

Two worker ranks, each with its own copy of the toy package, step together in a `Cluster`
loop thread (the engine core, loaded as a third copy). The host is a `StepGapCoordinator`
(phase 2 of every transaction runs in one gap between two steps) wrapped in a kernel
`Session` (base / edited / poisoned). Ranks are reached through `CallParticipant` over a
loopback call, as an engine reaches its workers over RPC.

There is no planner: a plan omits the targets, and each participant's prepare finds the
definitions whose AST changed against the text the module runs now from its one parse of the
two texts (`swap.stage(..., targets="auto")`).
The workload runs inside the loop, between steps: every rank steps once per input.
"""
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from kernel import coordinator as C
from kernel import locate as L
from kernel import receipt as R
from kernel import session as SS
from kernel import toy

LIVELAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ("toy.ops", "toy.engine")
WORKLOAD = {"inputs": [0, 1, 2, 3, 5, 8]}
_n = itertools.count()


def run_workload(engines, inputs):
    """The toy workload: every rank steps once per input. Shared with the fresh process."""
    return {r: [e.step(x) for x in inputs] for r, e in engines.items()}


def _edit(src, old, new):
    assert src.count(old) == 1, old
    return src.replace(old, new)


FRESH = """
import json, sys
livelab, path, inputs, ranks = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
sys.path.insert(0, livelab)
from kernel import toy
toy.load_copy("fresh", path)
from kernel.toy_lab import run_workload
E = sys.modules["fresh.engine"]
print(json.dumps(run_workload({r: E.Engine() for r in ranks}, inputs)))
"""


class ToyLab:
    name = "toy"

    def __init__(self, ranks=2, receipts=None, check_timeout=0.5, hang_s=1.0):
        self.ranks = [f"rank{i}" for i in range(ranks)]
        self.writer = R.ReceiptWriter(receipts) if receipts else None
        self.check_timeout, self.hang_s = check_timeout, hang_s
        self.base_text = {m: toy.source(m.split(".", 1)[1]) for m in MODULES}
        self._spawn()
        self.session = SS.Session(self.co, MODULES)

    # -- process layout ------------------------------------------------------------------
    def _spawn(self):
        tag = f"tl{next(_n)}"
        self.faults = {}
        self.live_text = [dict(self.base_text)]      # one entry per session stack depth
        self.engines, self.locals, self.pkgs, ps = {}, {}, {}, []
        for r in self.ranks:
            pkg = self.pkgs[r] = f"{tag}_{r}"
            toy.load_copy(pkg)
            e = self.engines[r] = sys.modules[pkg + ".engine"].Engine()
            lp = self.locals[r] = C.LocalParticipant(
                r, alias={"toy": pkg}, scope={"engine": e},
                resets={"R2": e.drain, "R3": e.cache.clear})
            ps.append(C.CallParticipant(r, self._rpc(r, lp)))
        toy.load_copy(f"{tag}_core")
        self.cluster = sys.modules[f"{tag}_core.engine"].Cluster(self.engines)
        self.co = C.StepGapCoordinator(
            ps, self.cluster.at_gap, writer=self.writer, interface_version="1-draft",
            engine={"name": "toy", "version": "0", "commit": None})
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self.cluster.run, args=(self._stop,),
                                        daemon=True, name=f"{tag}-loop")
        self._thread.start()
        t = time.time()
        while len(self.cluster.log) < 2 and time.time() - t < 5:
            time.sleep(0.001)

    def _rpc(self, rank, lp):
        def call(method, payload):
            if self.faults.get("lost") == rank:
                raise ConnectionError(f"{rank}: process lost")
            return getattr(lp, method)(payload)
        return call

    def _halt(self):
        self._stop.set()
        self._thread.join(self.hang_s + 5)

    def close(self):
        self._halt()

    # -- verbs ---------------------------------------------------------------------------
    def install(self, patch):
        t0 = time.perf_counter()
        base = self.live_text[-1]
        modules = {}
        for m, new in patch["files"].items():    # targets: "auto" (stage finds them)
            if m not in base:
                return {"ok": False, "id": None, "outcome": "refused",
                        "codes": ["module_unknown"], "mechanisms": [],
                        "findings": [{"check": "module_unknown", "observed": m,
                                      "expected": sorted(base)}],
                        "receipt": None, "cost_s": round(time.perf_counter() - t0, 6)}
            modules[m] = {"source": new, "base_hash": L.text_hash(base[m])}
        plan = {"modules": modules, "accept": patch.get("accept", []),
                "undo_resets": patch.get("undo_resets", []), "recipes": ["R0 swap"]}
        res = self.session.install(plan, edit_time=patch.get("edit_time"))
        outcome, rec = res.get("outcome", "refused"), res.get("receipt")
        if outcome == "committed":
            self.live_text.append({**base, **patch["files"]})
        codes = (rec["decisions"]["blocking"] if rec else
                 sorted({f["check"] for f in res.get("findings", [])}))
        return {"ok": outcome == "committed", "id": res.get("txn"), "outcome": outcome,
                "codes": [] if outcome == "committed" else codes,
                "mechanisms": rec["actual_recipes"] if rec else [],
                "findings": rec["findings"] if rec else res.get("findings", []),
                "receipt": rec, "cost_s": round(time.perf_counter() - t0, 6)}

    def check(self, workload=None):
        t0 = time.perf_counter()
        inputs = (workload or WORKLOAD)["inputs"]
        for p in self.co.participants:                    # every rank must answer
            r = C._call(p, "status", "ping")
            if not r["ok"]:
                self.session.fault("participant_failed", f"{p.name}: {r.get('error')}",
                                   "every rank answers", subject=p.name)
                return {"ok": False, "error": r.get("error"), "output": None, "proven": None,
                        "witness": {}, "findings": list(self.session.poison),
                        "cost_s": round(time.perf_counter() - t0, 6)}
        faults = self.faults

        def job():
            if faults.pop("crash", None):
                raise RuntimeError(f"{self.ranks[-1]}: worker died in step")
            if faults.pop("hang", None):
                time.sleep(self.hang_s)
            return run_workload(self.engines, inputs)
        try:
            out = json.loads(json.dumps(self.cluster.at_gap(job, self.check_timeout)))
        except Exception as e:
            self.session.fault("participant_failed", f"check: {type(e).__name__}: {e}",
                               f"the workload finishes within {self.check_timeout} s")
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "output": None,
                    "proven": None, "witness": {}, "findings": list(self.session.poison),
                    "cost_s": round(time.perf_counter() - t0, 6)}
        golden = {"output_hash": L.text_hash(json.dumps(out, sort_keys=True))}
        proofs = [self.co.prove(t, golden=golden) for t in self.session.stack]
        witness = {}
        for p in proofs:
            for part, counts in p["receipt"]["witness"].items():
                witness.setdefault(part, {}).update(counts)
        return {"ok": True, "output": out,
                "proven": all(p["ok"] for p in proofs) if proofs else None,
                "witness": witness,
                "findings": [f for p in proofs for f in p["receipt"]["findings"]],
                "cost_s": round(time.perf_counter() - t0, 6)}

    def reset(self, kind="state"):
        t0 = time.perf_counter()
        if kind == "respawn":
            self._halt()
            self._spawn()
            r = self.session.recovered(self.co)
            return {"ok": r["ok"], "reset": ["worker group", "session"],
                    "cost_s": round(time.perf_counter() - t0, 6)}

        def job():
            for e in self.engines.values():
                e.drain()
                e.cache.clear()
        try:
            self.cluster.at_gap(job, self.check_timeout)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "reset": [],
                    "cost_s": round(time.perf_counter() - t0, 6)}
        return {"ok": True, "reset": ["tokens", "cache"],
                "cost_s": round(time.perf_counter() - t0, 6)}

    def uninstall(self, id=None):
        t0 = time.perf_counter()
        r = self.session.uninstall(id)
        n = len(r.get("undone", []))
        if n:
            del self.live_text[-n:]
        return {"ok": r["ok"], "undone": r.get("undone", []), "state": self.session.state(),
                "findings": r.get("findings", []), "cost_s": round(time.perf_counter() - t0, 6)}

    def status(self):
        return self.session.status()

    # -- harness -------------------------------------------------------------------------
    def patches(self):
        ops, eng = self.base_text["toy.ops"], self.base_text["toy.engine"]
        a2 = _edit(ops, "    return x + 1", "    return x + 2")
        a3 = _edit(a2, "    return x + 2", "    return x + 3")

        def p(**files):
            return {"files": {"toy." + k: v for k, v in files.items()}}
        return {
            "noop": p(ops=_edit(ops, "    return x * STEP_SCALE",
                                "    y = x * STEP_SCALE\n    return y")),
            "visible": p(ops=a2),
            "chain": [p(ops=a2), p(ops=a3), p(engine=_edit(eng, "        return y\n",
                                                           "        return y * 10\n"))],
            "multi": p(ops=_edit(a2, "    return x * STEP_SCALE",
                                 "    return x * STEP_SCALE + 1"),
                       engine=_edit(eng, "        return y\n", "        return y - 1\n")),
            # Engine.__init__ edit: stage blocks it (constructor_edit); accepted here so c10
            # can show that an installed edit the workload never runs is not proven.
            "unreached": {**p(engine=_edit(eng, "make_shift(10)", "make_shift(11)")),
                          "accept": ["constructor_edit"]},
            "refused": {
                "syntax": p(ops=_edit(ops, "    return x + 1", "    return x +")),
                "closure": p(ops=_edit(ops, "        return x + k",
                                       "        return x + k + bias")),
                "module_state": p(ops=_edit(ops, "STEP_SCALE = 2", "STEP_SCALE = 3")),
                "empty": p(ops=_edit(ops, "def scale(x):", "def scale(x):  # faster")),
            },
        }

    def fresh_check(self, patch, workload=None):
        inputs = (workload or WORKLOAD)["inputs"]
        tmp = tempfile.mkdtemp(prefix="toylab-fresh-")
        try:
            for f in ("__init__.py", "ops.py", "engine.py"):
                shutil.copy(os.path.join(toy.HERE, f), tmp)
            for m, text in ((patch or {}).get("files") or {}).items():
                with open(os.path.join(tmp, m.split(".", 1)[1] + ".py"), "w") as fh:
                    fh.write(text)
            r = subprocess.run([sys.executable, "-I", "-c", FRESH, LIVELAB, tmp,
                                json.dumps(inputs), json.dumps(self.ranks)],
                               capture_output=True, text=True, timeout=60)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if r.returncode:
            stderr_lines = r.stderr.strip().splitlines()
            error_msg = stderr_lines[-1] if stderr_lines else "fresh_check failed"
            return {"ok": False, "error": error_msg, "output": None}
        return {"ok": True, "output": json.loads(r.stdout)}

    def inject(self, fault, **kw):
        rank = kw.get("participant", self.ranks[-1])
        if fault == "fail_apply":
            j, after, n = self.locals[rank].journal, kw.get("after", 1), [0]
            orig = j.set_code

            def set_code(*a, **k):
                n[0] += 1
                if n[0] > after:
                    del j.set_code                  # one shot
                    raise RuntimeError(f"{rank}: injected fault after {after} swap(s)")
                return orig(*a, **k)
            j.set_code = set_code
        elif fault == "fail_commit":
            lp = self.locals[rank]
            q = lp.quiesce

            def lost(fn):
                lp.quiesce = q
                raise RuntimeError(f"{rank}: injected commit failure")
            lp.quiesce = lost
        elif fault in ("crash", "hang"):
            self.faults[fault] = True
        elif fault == "lost":
            self.faults["lost"] = rank
        elif fault == "tamper":
            fn = sys.modules[self.pkgs[rank] + ".ops"].scale
            fn.__code__ = (lambda x: x * 3).__code__        # a REPL edit, not journaled
        else:
            return {"ok": False, "error": f"unknown fault {fault}"}
        return {"ok": True, "fault": fault, "participant": rank}
