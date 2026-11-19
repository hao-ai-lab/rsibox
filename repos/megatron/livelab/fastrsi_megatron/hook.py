"""The Megatron live lab: the kernel hosted in the gap before a train step.

`install_hook(training_module)` wraps `megatron.training.training.train_step` (the name
`train()` calls). At the first call, on every rank, the lab:

1. freezes the batches the checks will feed (from the real data iterator),
2. takes the start state S0 (state.py), runs `warmup` real train steps, puts S0 back and zeroes
   the optimizer state the warm-up created, then keeps that as S (the start of every check),
3. serves the gap: rank 0 takes client requests (gap.Server) and runs them; the other ranks
   handle rank 0's rounds (gap.Ranks) until `resume` or `exit`.

Rank 0 hosts the kernel's StepGapCoordinator and Session. The ranks are one participant
(`RanksParticipant`): each kernel call is one gloo round, every rank answers from its own
LocalParticipant, and the replies merge per rank (`merge_ranks`), so a rank that does not
answer is a blocking `participant_missing`. Phase 1 and phase 2 both run in this gap: no train
step runs while the lab serves, so no step can see two code versions.

Verbs (rank 0, one JSON request each; see client.py): install, check, reset, uninstall, status,
digest, repl, inject, resume, exit.

Robustness: a request that is not a JSON object with a string `verb` gets an error reply; a client
that times out or disconnects before its reply costs only that reply (logged, the lab serves on;
the other ranks never see it). `reset` and `check` fail closed when a rank's restore refuses
(state.Snapshot.restore). `digest` returns each rank's sha256 of every piece of state, so a reset
round trip is shown bit for bit. `resume` puts S back on every rank (unless `state: "keep"`) and
the job's next train steps get the frozen batches first, then the real iterator: the job continues
as if the lab had not run. Fresh mode (FASTRSI_FRESH=<output path>): steps 1-2, one check,
the output written as JSON, then exit; the conformance harness compares it to a live check.
"""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
import traceback

import torch
import torch.distributed as dist

from fastrsi import diffs, harness, ipc
from kernel import coordinator as C
from kernel import findings as F
from kernel import locate as L
from kernel import receipt as R
from kernel import session as SS

from . import gap as G
from . import state as ST

INTERFACE = "1-draft"
# exit code of a rank other than 0 whose send or receive of a lab round failed: never a clean exit
LOST_ROUND_EXIT = 19
# Modules whose code identity the session fingerprints (status, tamper checks). A patch's own
# modules are added when it installs.
MODULES = ("megatron.core.transformer.mlp", "megatron.core.transformer.attention",
           "megatron.core.transformer.transformer_layer",
           "megatron.core.models.gpt.gpt_model", "megatron.training.training")


def _now():
    return time.perf_counter()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class RanksParticipant(C.CallParticipant):
    """Every rank as one participant: one kernel call is one gloo round."""

    def __init__(self, lab):
        super().__init__("ranks", self._all, bounded=True)
        self.lab = lab
        self.expected = [str(r) for r in range(lab.ranks.world)]

    def _all(self, method, arg):
        rs = self.lab.round({"op": "pcall", "method": method, "arg": arg})
        if method == "fingerprint":
            ok = all(r.get("ok") for r in rs)
            return {"ok": ok, "code": {str(r.get("rank")): r.get("code") for r in rs},
                    "ranks": rs, **({} if ok else {"error": "a rank gave no fingerprint"})}
        return C.merge_ranks(method, rs)


def unknown_state_finding(unknown):
    """Optimizer state the first step created that the lab cannot reset (not a fresh Adam's zeros,
    e.g. the precision-aware optimizer's master copies), as a finding in the kernel's shape. The
    check is Megatron-only, so it is built here, not from the kernel's catalog. The lab still
    starts; checks then begin from the warm-up's values of these tensors, which the loss gate
    shows."""
    return {"check": "optimizer_state_unknown", "observed": f"state keys not reset: {unknown[:5]}",
            "expected": f"only {sorted(ST.ZERO_KEYS)} keys are created by the first optimizer step",
            "why": "a check would start from the warm-up's values of these tensors",
            "actions": [{"do": "report", "detail": "say which optimizer option created them"}],
            "risk_if_ignored": "every check starts from a different optimizer state than the base",
            "lessons": ["Resetting the optimizer for a check"], "overridable": True, "n_unknown": len(unknown)}


class Lab:
    def __init__(self, train_step, a, k):
        self.ts, self.a, self.k = train_step, list(a), dict(k)
        self.model, self.optimizer, self.scheduler = self.a[2], self.a[3], self.a[4]
        self.ranks = G.Ranks()
        self.rank = self.ranks.rank
        self.steps_max = int(os.environ.get("FASTRSI_CHECK_STEPS_MAX", "8"))
        self.default_steps = int(os.environ.get("FASTRSI_CHECK_STEPS", "4"))
        self.warmup = int(os.environ.get("FASTRSI_WARMUP", "2"))
        self.faults = {}
        self.timings = {}
        self.local = C.LocalParticipant(
            f"rank{self.rank}",
            scope={"model": self.model, "optimizer": self.optimizer},
            resets={"R2": self._r2, "R3": self._r3})
        self._setup()
        self.host = None
        self.replayed = None

    # -- state ---------------------------------------------------------------------------
    def _mb(self):
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        return get_num_microbatches()

    def _setup(self):
        t0 = _now()
        n = max(self.steps_max, self.warmup) * self._mb()
        self.batches = ST.FrozenBatches(self.a[1], n)
        t1 = _now()
        s0 = ST.Snapshot(self.model, self.optimizer, self.scheduler)
        t2 = _now()
        warm = self._steps(self.warmup)
        from . import zygote
        zygote.quiesce_compile_pools_after_warmup()
        t3 = _now()
        r = s0.restore(allow_extra=True)        # the warm-up created Adam's state: zeroed next
        if not r["ok"]:
            raise RuntimeError(f"lab setup: the start state could not be put back: {r}")
        zeroed, unknown = ST.zero_new_state(self.optimizer, set(s0.host))
        del s0
        self.snap = ST.Snapshot(self.model, self.optimizer, self.scheduler)
        t4 = _now()
        self.setup_findings = []
        if unknown:
            self.setup_findings.append(unknown_state_finding(unknown))
        self.timings.update(freeze_s=round(t1 - t0, 3), snapshot_s=round(t2 - t1, 3),
                            warmup_s=round(t3 - t2, 3), warmup_steps=warm["step_s"],
                            restore_zero_snapshot_s=round(t4 - t3, 3),
                            snapshot_gib=round(self.snap.bytes / 2**30, 2),
                            zeroed=len(zeroed), restore=r)

    def _r2(self):
        return self._restore()

    def _r3(self):
        torch.cuda.empty_cache()
        return {"empty_cache": True}

    def _restore(self):
        return self.snap.restore()

    def _steps(self, n):
        """n real train steps on the frozen batches. Per-step loss, grad norm and seconds."""
        a = list(self.a)
        a[1] = self.batches.iterator()
        loss, gn, sec = [], [], []
        it0 = self.k.get("iteration") or 0
        for i in range(n):
            if self.faults.pop("crash", None):
                os._exit(17)                              # a worker dies in a step
            if self.faults.pop("hang", None):
                time.sleep(float(os.environ.get("FASTRSI_HANG_S", "3600")))
            _sync()
            t0 = _now()
            r = self.ts(*a, **{**self.k, "iteration": it0 + i})
            _sync()
            sec.append(round(_now() - t0, 5))
            ld = r[0] if isinstance(r[0], dict) else {}
            v = ld.get("lm loss")
            if torch.is_tensor(v):
                v = v.float().mean().item() if v.numel() > 1 else v.item()
            loss.append(v)
            g = r[5]          # (loss_dict, skipped, should_checkpoint, should_exit, exit_code, grad_norm, ...)
            gn.append(float(g) if g is not None else None)
        return {"loss": loss, "grad_norm": gn, "step_s": sec}

    # -- rounds --------------------------------------------------------------------------
    def round(self, msg):
        """Rank 0: one round. A round that fails (a rank died or did not answer in time) poisons
        the session, once, and raises G.RoundFailed; every later round raises it at once."""
        try:
            return self.ranks.round(msg, self.handle)
        except G.RoundFailed as failure:
            if self.host is not None and not self.host.poisoned():
                self.host.fault("participant_failed", f"round failed: {failure}",
                                "every rank answers each round")
            raise

    def handle(self, msg):
        """Every rank: one request of a round. Never raises."""
        op = msg.get("op")
        try:
            if op == "pcall":
                if self.faults.get("lost"):
                    os._exit(18)                          # this rank stops answering
                r = C._call(self.local, msg["method"], msg["arg"])
                return {**r, "rank": self.rank}
            if op == "check":
                t0 = _now()
                rst = self._restore()
                if not rst["ok"]:
                    return {"ok": False, "rank": self.rank, "restore": rst,
                            "error": f"restore refused: {rst['error']}"}
                out = self._steps(msg["steps"])
                return {"ok": True, "rank": self.rank, **out, "restore": rst,
                        "seconds": round(_now() - t0, 4)}
            if op in ("reset", "prepare_resume"):
                rst = self._restore()
                return {"ok": rst["ok"], "rank": self.rank, "restore": rst,
                        **({} if rst["ok"] else {"error": f"restore refused: {rst['error']}"})}
            if op == "digest":
                t0 = _now()
                d = ST.digest(self.model, self.optimizer, self.scheduler)
                return {"ok": True, "rank": self.rank, "digest": d,
                        "seconds": round(_now() - t0, 4)}
            if op == "repl":
                return {**self._repl(msg["code"]), "rank": self.rank}
            if op == "inject":
                return {**self._inject(msg), "rank": self.rank}
            if op == "resume":
                self.resumed = msg.get("state", "start")
            if op in ("resume", "exit", "noop"):
                return {"ok": True, "rank": self.rank}
            return {"ok": False, "rank": self.rank, "error": f"unknown op {op}"}
        except Exception as e:
            return {"ok": False, "rank": self.rank, "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()[-2000:]}

    def _repl(self, code):
        from megatron.training import get_args
        ns = {"lab": self, "model": self.model, "optimizer": self.optimizer,
              "scheduler": self.scheduler, "torch": torch, "args": get_args(),
              "rank": self.rank}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<fastrsi_repl>", "exec"), ns)
            return {"ok": True, "out": buf.getvalue()[-8000:]}
        except Exception as e:
            return {"ok": False, "out": buf.getvalue()[-4000:],
                    "error": f"{type(e).__name__}: {e}"}

    def _inject(self, msg):
        if msg.get("target") not in (None, self.rank):
            return {"ok": True, "skipped": True}
        fault = msg["fault"]
        if fault in ("fail_apply", "fail_commit", "tamper"):
            kw = {k: v for k, v in msg.items() if k not in ("op", "fault", "target")}
            reply = harness.inject_participant_fault(self.local, fault, **kw)
            return {"ok": bool(reply.get("ok")), "fault": fault,
                    **({"error": reply.get("error")} if not reply.get("ok") else {})}
        if fault in ("crash", "hang", "lost"):
            self.faults[fault] = True
            return {"ok": True, "fault": fault}
        return {"ok": False, "error": f"unknown fault {fault}"}

    # -- host (rank 0) -------------------------------------------------------------------
    def _host(self):
        if self.host is None:
            writer = R.ReceiptWriter(os.environ["FASTRSI_RECEIPTS"]) \
                if os.environ.get("FASTRSI_RECEIPTS") else None
            co = C.StepGapCoordinator(
                [RanksParticipant(self)], gap=lambda fn: fn(), writer=writer,
                interface_version=INTERFACE, phase2_deadline_s=None,
                engine={"name": "megatron-lm", "version": _version(), "commit": None})
            mods = sorted(set(MODULES) | set(os.environ.get("FASTRSI_MODULES", "").split(","))
                          - {""})
            self.live_text = [{}]
            self.host = SS.Session(co, [m for m in mods if m in sys.modules])
        return self.host

    @staticmethod
    def _track(session, names):
        """Add the modules a patch edits to the session's fingerprint (status and tamper checks
        then cover them), at every stack depth: no committed transaction changed them yet, so
        their code now is their code at each depth. A change made to them before this call is
        taken as their base."""
        new = [name for name in dict.fromkeys(names) if name not in session.modules]
        if not new:
            return
        codes = {}
        for participant in session.co.participants:
            answer = C._call(participant, "fingerprint", new)
            if not answer["ok"]:
                return                          # install's own verify reports the participant
            codes[participant.name] = answer["code"]
        for expected in session.expected:
            for name, code in codes.items():
                merged = dict(expected.get(name) or {})
                for key, value in code.items():  # {rank: {function: identity}} or {function: ...}
                    both = isinstance(value, dict) and isinstance(merged.get(key), dict)
                    merged[key] = {**merged[key], **value} if both else value
                expected[name] = merged
        session.modules += new

    def _text(self, m):
        for d in reversed(self.live_text):
            if m in d:
                return d[m]
        return L.module_source(m) if hasattr(L, "module_source") else \
            Path(sys.modules[m].__file__).read_text()

    def serve(self):
        """Every rank: serve the gap until resume or exit."""
        if self.rank != 0:
            while True:
                try:
                    msg = self.ranks.bcast()
                    self.ranks.gather(self.handle(msg))
                except Exception as error:      # rank 0 gave up on the round, or is gone
                    print(f"[fastrsi] rank {self.rank}: lost the lab's round with rank 0 "
                          f"({type(error).__name__}: {str(error)[:200]}); exiting with code "
                          f"{LOST_ROUND_EXIT}", file=sys.stderr, flush=True)
                    os._exit(LOST_ROUND_EXIT)
                if msg.get("op") in ("resume", "exit"):
                    return msg["op"]
        path = os.environ.get("FASTRSI_SOCK") or ipc.short_sock_path("fastrsi-mg")
        server = G.Server(path)
        if os.environ.get("FASTRSI_MAILBOX"):
            G.Mailbox(os.environ["FASTRSI_MAILBOX"], server.q)
        with open(os.environ.get("FASTRSI_READY", os.devnull), "w") as f:
            json.dump({"sock": path, "pid": os.getpid(), "timings": self.timings,
                       "mono": time.monotonic(), "time": time.time()}, f)
        print(f"[fastrsi] lab ready on {path}: {json.dumps(self.timings)}", flush=True)
        while True:
            req, reply = server.next()
            t0 = _now()
            verb = req.get("verb") if isinstance(req, dict) else None
            try:
                if not isinstance(verb, str):
                    out = {"ok": False, "error": "bad request: a JSON object with a string "
                           f"'verb' is required, got {str(req)[:200]}"}
                else:
                    out = self.verb(verb, req)
            except Exception as e:
                out = {"ok": False, "error": f"{type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-3000:]}
            out["cost_s"] = round(_now() - t0, 6)
            try:
                sent = reply(out)
            except OSError as e:                # a reply function that raises: same as gone
                sent = False
                print(f"[fastrsi] reply failed: {e}", file=sys.stderr, flush=True)
            if sent is False:
                print(f"[fastrsi] client gone before the reply to {verb!r} "
                      f"({out['cost_s']} s); the lab serves on", flush=True)
            if verb == "exit" or (verb == "resume" and out.get("ok")):
                server.close()
                return verb

    def verb(self, verb, req):
        s = self._host()
        if verb == "install":
            patch = req.get("patch")
            files = patch.get("files") if isinstance(patch, dict) else None
            repairs = patch.get("repairs", []) if isinstance(patch, dict) else []
            why, modules, missing = None, {}, []
            if isinstance(patch, dict) and patch.get("diff") is not None:
                diff_text = patch["diff"]
                if not isinstance(diff_text, str):
                    why = "patch `diff` must be a unified diff string"
                else:
                    diff_targets = [diffs.module_of(p) for p in diffs.paths(diff_text)]
                    missing = sorted(name for name in diff_targets if name not in sys.modules)
                    if missing:
                        why = f"modules not loaded in the job: {missing!r}"
                    else:
                        try:
                            files = diffs.apply(diff_text, self._text)
                        except ValueError as error:
                            why = str(error)
            if not why:
                missing = sorted(name for name in files if name not in sys.modules) if isinstance(files, dict) else []
                if not isinstance(files, dict):
                    why = "a patch is an object with `files`: {module: source} or `diff`: unified diff"
                elif missing:
                    why = f"modules not loaded in the job: {missing!r}"
                else:
                    for name, new in files.items():
                        try:
                            modules[name] = {"source": new, "base_hash": L.text_hash(self._text(name))}
                        except (AttributeError, OSError, TypeError, ValueError) as error:
                            why = f"module {name!r} has no readable source file ({type(error).__name__})"
                            break
            if why:
                return {"ok": False, "id": None, "outcome": "refused", "codes": ["patch_invalid"],
                        "mechanisms": [], "receipt": None, "not_loaded": missing, "findings": [F.finding(
                            "patch_invalid", why, "modules the running job imported from source")]}
            if not s.poisoned():
                self._track(s, list(files))
            plan = {"modules": modules, "accept": patch.get("accept", []),
                    "undo_resets": patch.get("undo_resets", []), "repairs": repairs,
                    "recipes": (["R0 swap"] if files else []) + (["R10 repair"] if repairs else [])}
            res = s.install(plan, edit_time=patch.get("edit_time"))
            outcome, rec = res.get("outcome", "refused"), res.get("receipt")
            if outcome == "committed":
                self.live_text.append(dict(files))
            codes = (rec["decisions"]["blocking"] if rec else
                     sorted({f["check"] for f in res.get("findings", [])}))
            return {"ok": outcome == "committed", "id": res.get("txn"), "outcome": outcome,
                    "codes": [] if outcome == "committed" else codes,
                    "mechanisms": rec["actual_recipes"] if rec else [],
                    "findings": rec["findings"] if rec else res.get("findings", []),
                    "receipt": rec}
        if verb == "check":
            steps = min(int(req.get("steps") or self.default_steps), self.steps_max)
            rs = self.round({"op": "check", "steps": steps})
            if not all(r.get("ok") for r in rs):
                err = "; ".join(f"rank {r.get('rank')}: {r.get('error')}" for r in rs
                                if not r.get("ok"))
                s.fault("participant_failed", f"check: {err}", "every rank finishes the check")
                return {"ok": False, "error": err, "output": None, "proven": None,
                        "witness": {}, "findings": list(s.poison)}
            out = {"loss": rs[0]["loss"], "grad_norm": rs[0]["grad_norm"]}
            golden = {"output_hash": L.text_hash(json.dumps(out, sort_keys=True))}
            proofs = [s.co.prove(t, golden=golden) for t in s.stack]
            witness = {}
            for p in proofs:
                for part, counts in p["receipt"]["witness"].items():
                    witness.setdefault(part, {}).update(counts)
            return {"ok": True, "output": out,
                    "step_s": {r["rank"]: r["step_s"] for r in rs},
                    "restore_s": rs[0]["restore"]["seconds"],
                    "proven": all(p["ok"] for p in proofs) if proofs else None,
                    "witness": witness,
                    "findings": [f for p in proofs for f in p["receipt"]["findings"]]}
        if verb == "reset":
            kind = req.get("kind", "state")
            if kind != "state":
                return {"ok": False, "error": "respawn restarts the worker group: the "
                        "harness does it (client.respawn)", "reset": []}
            rs = self.round({"op": "reset"})
            ok = all(r.get("ok") for r in rs)
            done = set.intersection(*[set(r["restore"].get("restored", [])) for r in rs]) \
                if ok else set()
            restored = [c for c in rs[0]["restore"].get("restored", []) if c in done]
            out = {"ok": ok, "reset": restored + (["data"] if ok else []),
                   "restore": rs[0].get("restore")}
            if not ok:
                out["error"] = "; ".join(f"rank {r.get('rank')}: {r.get('error')}" for r in rs
                                         if not r.get("ok"))
                out["ranks"] = [r.get("restore") for r in rs]
            return out
        if verb == "digest":
            t0 = _now()
            rs = self.round({"op": "digest"})
            ok = all(r.get("ok") for r in rs)
            ds = [r.get("digest") for r in rs]
            import hashlib
            # no digest when a rank gave none: two failed digests must not compare equal
            return {"ok": ok, "ranks": ds,
                    "digest": hashlib.sha256(json.dumps(ds, sort_keys=True).encode()).hexdigest()
                    if ok else None,
                    "rank_s": [r.get("seconds") for r in rs], "seconds": round(_now() - t0, 4),
                    **({} if ok else {"error": "; ".join(str(r.get("error")) for r in rs
                                                         if not r.get("ok"))})}
        if verb == "uninstall":
            r = s.uninstall(req.get("id"))
            n = len(r.get("undone", []))
            if n:
                del self.live_text[-n:]
            return {"ok": r["ok"], "undone": r.get("undone", []), "state": s.state(),
                    "findings": r.get("findings", [])}
        if verb == "status":
            return {**s.status(), "timings": self.timings,
                    "setup_findings": self.setup_findings}
        if verb == "repl":
            return {"ok": True, "ranks": self.round({"op": "repl", "code": req["code"]})}
        if verb == "inject":
            try:
                target = target_rank(req.get("target"), self.ranks.world)
            except ValueError as error:
                return {"ok": False, "error": str(error), "ranks": []}
            replies = self.round({"op": "inject", "target": target,
                                  **{key: value for key, value in req.items()
                                     if key not in ("verb", "target")}})
            return {"ok": all(reply["ok"] for reply in replies), "ranks": replies}
        if verb == "resume" and req.get("state", "start") == "start":
            rs = self.round({"op": "prepare_resume"})
            if not all(r.get("ok") for r in rs):
                return {"ok": False, "error": "resume refused, the lab serves on: " + "; ".join(
                    f"rank {r.get('rank')}: {r.get('error')}" for r in rs if not r.get("ok"))}
        if verb in ("resume", "exit"):
            self.round({"op": verb, "state": req.get("state", "start")})
            return {"ok": True, **({"state": req.get("state", "start")} if verb == "resume"
                                   else {})}
        return {"ok": False, "error": f"unknown verb {verb}"}

    def replay(self, it):
        """The iterator the job's train steps get after resume: the frozen batches, then `it`
        (the batches the lab took from it come first again), so the job continues as if the lab
        had not run. With resume state "keep", `it` itself."""
        if self.batches.items is None or getattr(self, "resumed", "start") != "start":
            return it
        if self.replayed is None or self.replayed[0] is not it:
            base = _Replay(self.batches.items, it)
            w = self.batches.wrapper
            self.replayed = (it, w(base) if w is not None and w.__name__ == "RerunDataIterator"
                             else base)
        return self.replayed[1]


ZERO = ST.ZERO_KEYS


class _Replay:
    """The frozen batches, then `it` (an iterator that may lack __iter__, as Megatron's
    RerunDataIterator does)."""

    def __init__(self, items, it):
        self.items, self.it = list(items), it

    def __iter__(self):
        return self

    def __next__(self):
        if self.items:
            return self.items.pop(0)
        return next(self.it)


def target_rank(target, world):
    """An inject target -> the rank it names, or None for every rank. Takes a rank number or a
    participant name ("rank1", "1"); raises ValueError for anything that names no rank, so a
    fault is never skipped on every rank while the reply says ok."""
    if target is None:
        return None
    if isinstance(target, int) and not isinstance(target, bool):
        text = str(target)
    else:
        text = target if isinstance(target, str) else ""
    digits = text[len("rank"):] if text.startswith("rank") else text
    if not (digits.isascii() and digits.isdigit() and digits == str(int(digits))
            and int(digits) < world):
        raise ValueError(f"inject target {target!r} names no rank: give 0..{world - 1} "
                         f"or rank0..rank{world - 1}")
    return int(digits)


def _version():
    try:
        import megatron.core
        return getattr(megatron.core, "__version__", None)
    except Exception:
        return None


def install_hook(tr):
    """Wrap tr.train_step: the first call opens the lab in the step gap."""
    ts = tr.train_step
    raw = getattr(ts, "__wrapped__", ts)
    box = {"lab": None}

    def train_step(*a, **k):
        if box["lab"] is None:
            lab = box["lab"] = Lab(raw, a, k)
            fresh = os.environ.get("FASTRSI_FRESH")
            if fresh:
                out = lab.handle({"op": "check", "steps": lab.default_steps})
                if lab.rank == 0:
                    with open(fresh, "w") as f:
                        json.dump({"output": {"loss": out.get("loss"),
                                              "grad_norm": out.get("grad_norm")},
                                   "ok": out.get("ok"), "error": out.get("error"),
                                   "timings": lab.timings}, f)
                dist.barrier()
                os._exit(0)
            if lab.serve() == "exit":
                dist.barrier(group=lab.ranks.group)
                os._exit(0)
        lab = box["lab"]
        a = (a[0], lab.replay(a[1])) + tuple(a[2:]) if len(a) > 1 else a
        return ts(*a, **k)
    train_step.__wrapped__ = raw
    tr.train_step = train_step
