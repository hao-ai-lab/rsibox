"""Engine-core host: kernel transactions inside vLLM's EngineCore busy loop (live code swap, and
selective recapture of CUDA graphs in the commit's gap).

Enable with FASTRSI_KERNEL_DIR=<dir> in the engine's environment, import this module early in
every engine process (a sitecustomize does it), and add `--worker-extension-cls
adapter.worker_ext.KernelWorkerExtension`. Lab use only: it runs code sent to it.

How a transaction runs (kernel `StepGapCoordinator`):

- Control: a file channel. A client writes <dir>/ctl.json `{"seq", "op", ...}`; a host thread
  in the engine-core process runs it and writes <dir>/reply/<seq>.json. Ops: `txn {plan,
  edit_time, baseline_restart}`, `prove {txn, golden}`, `undo {txn}`, `status {txn}` (without
  txn: the host's own state), `pstatus {txn}` (every rank's own state), `fingerprint
  {modules}` (per rank: content hash and code object id of every function), `versions` (per
  rank: the per-step version log), `at_base`, `ping`, `graphs` (per rank: the descriptors
  each mode can replay and every graph wrapper's entry count), `recapture_extend {txn, sizes}`
  (capture more of the descriptors a recapture left eager, in a gap).
- Recapture (`RecaptureCoordinator`): a plan with `"rebuild": ["recapture"]` (or `"rebuild": ["R1"]`,
  optional `"recapture"` or `"r1"`: `{"modes": [mode names], "sizes": [num_tokens]}`) is recaptured
  on every rank right after its commit, in the same gap, so no step replays a graph of the old code
  next to Python running the new one. The receipt's `rebuild` and `recapture_s` say what it captured
  and how long it took; the edit runs on every step from `rebuilt` on (edit_to_run_s). A recapture that
  fails on any rank is `rebuild_failed`: the transaction is undone on every participant in the same gap
  (each rank's recapture puts its old graphs back first), outcome rolled_back. Undo of a recaptured
  transaction puts the base graphs back on every rank (journal; no capture).
  Receipts also go to <dir>/receipts.jsonl (kernel `ReceiptWriter`).
- Two participants: the workers (every rank behind one collective_rpc, `WorkerGroup`) and the
  engine core (a `Side` in this process).
- Every call to the workers is `model_executor.collective_rpc` issued FROM the busy loop (a
  job put on the engine core's input queue), never from another thread: the executor's
  message queue is not thread-safe, and from the loop every rank handles the call between the
  same two steps. Phase 1 runs in a thread on every side (side.py): the loop is held only for
  `prepare_start` and the `prepare_poll`s, every 20 ms, while the engine steps between them.
- Phase 2 (validate, commit or abort, undo of a failed commit) is one job that runs only when
  no batch is in flight. While it waits, the loop schedules nothing new ("pause keep":
  `scheduler.has_requests` is shadowed to False while `batch_queue` drains).
- Phase 2 has a deadline (FASTRSI_KERNEL_PHASE2_DEADLINE, default 5 s; `phase2_deadline_s` per
  op; 0 turns it off). Every side answers its phase-2 calls by the deadline from its own lane
  (`deadline_missed` naming the late rank), and collective_rpc gets `timeout` = the time left
  + FASTRSI_KERNEL_RPC_GRACE (default 2 s) as a backstop for a process that cannot answer at
  all. A timeout leaves the late reply in the executor's
  response queue, where the next call would read it, so after one the worker group refuses
  every call (`deadline_missed`; respawn the engine) and the kernel poisons the host. The
  same holds for a call made from outside the loop (prepare, status, ...) whose job the loop
  started but did not finish while the caller waited (`in_loop`): it may still apply.
- The worker group expects a reply from every rank this engine core reaches (TP x PP); a
  missing one is a blocking `participant_missing`. At data parallel >1 vLLM runs one engine
  core per DP group and this host reaches only its own, so the other groups are expected too
  and every transaction is refused naming them (DP>1 unsupported, said plainly). Only DP rank
  0 serves the channel.
- The import finder that patches vllm.v1.engine.core asks the other finders on sys.meta_path
  for the module (an overlay first), so import this module after the overlay's finder.

The host records every step's duration, so a reply says how long steps stalled.
"""
import collections
import importlib.abc
import json
import os
import statistics
import sys
import threading
import time
import traceback

from kernel import coordinator as C
from kernel import findings as F
from kernel import receipt as R

from .side import Side

TAG = "FASTRSI_KERNEL"
DIR = os.environ.get("FASTRSI_KERNEL_DIR")
GAP_TIMEOUT_S = float(os.environ.get("FASTRSI_KERNEL_GAP_TIMEOUT", "30"))
PHASE2_DEADLINE_S = float(os.environ.get("FASTRSI_KERNEL_PHASE2_DEADLINE", "5")) or None
RPC_GRACE_S = float(os.environ.get("FASTRSI_KERNEL_RPC_GRACE", "2"))
PREPARE_TIMEOUT_S = 300.0
_HOST = []


class _Job:
    """A callable for the busy loop. gap=True: run only with no batch in flight."""

    def __init__(self, fn, gap):
        self.fn, self.gap = fn, gap
        self.done, self.started = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.cancelled = False         # the caller stopped waiting before it started
        self.value = self.error = None

    def run(self):
        with self.lock:
            if self.cancelled:
                self.done.set()
                return
            self.started.set()
        try:
            self.value = self.fn()
        except BaseException as e:            # handed back to the waiting caller
            self.error = e
        self.done.set()


class LateJob(TimeoutError):
    """The loop started the job but it did not finish in time: it may still be running."""


class WorkerGroup(C.Participant):
    """All worker ranks as one participant: each call is one collective_rpc from the loop, and
    the reply keeps every rank's own reply (kernel `merge_ranks`). Bounded by the RPC timeout
    (see the module doc)."""
    name = "workers"
    bounded = True

    def __init__(self, host, expected=None):
        self.host = host
        self.broken = None       # set when a collective_rpc timed out: replies are misaligned
        self.expected = expected  # rank ids every reply must carry (kernel expect_ranks)

    def _rpc(self, method, payload):
        if self.broken:
            return F.not_ok(f"worker group unusable: {self.broken}", [{**F.finding(
                "deadline_missed", self.broken, "a worker group whose replies line up"),
                "participant": self.name}])
        ex = self.host.core.model_executor
        _, deadline_at = C.txn_ref(payload)
        timeout = None if deadline_at is None else \
            max(0.0, deadline_at - time.time()) + RPC_GRACE_S

        def call():                       # in the loop; its TimeoutError is the RPC's own
            try:
                return ex.collective_rpc("fastrsi_kernel", timeout=timeout,
                                         args=(method, payload))
            except TimeoutError as e:
                return e
        try:
            rs = self.host.in_loop(call)
        except LateJob as e:              # started and still running: it may yet apply
            return self._break(method, deadline_at, e, (
                f"collective_rpc {method}: {e}; it may still apply: respawn"),
                f"{method} finishes in the loop while its caller waits")
        except Exception as e:            # the loop never started the job: it never runs
            return F.not_ok(f"collective_rpc {method}: {type(e).__name__}: {e}")
        if isinstance(rs, TimeoutError):
            return self._break(method, deadline_at, rs, (
                f"collective_rpc {method} timed out after {timeout} s; a late reply stays "
                "in the executor's queue: respawn"), f"{method} answers within {timeout} s")
        out = C.merge_ranks(method, rs)
        if any(r.get("pending") for r in out["ranks"]):
            out["pending"] = True
        return out

    def _break(self, method, deadline_at, err, why, expected):
        """A call the group cannot account for: refuse every later call and poison the host.
        A phase-2 call with a deadline is poisoned by the kernel (`deadline_missed` in the
        reply of a bounded participant); any other call (prepare, status, ...) here."""
        self.broken = why
        f = {**F.finding("deadline_missed", why, expected), "participant": self.name}
        if deadline_at is None:
            self.host.co.poison.append(f)
        return F.not_ok(f"collective_rpc {method}: {type(err).__name__}: {err}", [f])

    def prepare(self, plan):
        """Every rank stages in its own thread (side.py); the loop is held only for the start
        call and the polls, while the engine keeps stepping between them."""
        r = self._rpc("prepare_start", plan)
        if not r["ok"]:
            return r
        t = time.perf_counter()
        while True:
            time.sleep(0.02)
            r = self._rpc("prepare_poll", plan["id"])
            if not r.get("pending"):     # a no vote waits for the ranks still preparing
                return r
            if time.perf_counter() - t > PREPARE_TIMEOUT_S:
                return F.not_ok(f"workers did not prepare within {PREPARE_TIMEOUT_S} s")

    def versions(self, _=None):
        return self._rpc("versions", None)

    def validate(self, txn):
        return self._rpc("validate", txn)

    def commit(self, txn):
        return self._rpc("commit", txn)

    def abort(self, txn):
        return self._rpc("abort", txn)

    def undo(self, txn):
        return self._rpc("undo", txn)

    def status(self, txn):
        return self._rpc("status", txn)

    def proof(self, txn):
        return self._rpc("proof", txn)

    def witness_check(self, txn):
        return self._rpc("witness_check", txn)

    def reset(self, kind):
        return self._rpc("reset", kind)

    def fingerprint(self, modules):
        return self._rpc("fingerprint", modules)

    def at_base(self, _=None):
        return self._rpc("at_base", None)

    def recapture(self, payload):
        return self._rpc("recapture", payload)

    def recapture_extend(self, payload):
        return self._rpc("recapture_extend", payload)

    def r1_extend(self, payload):
        return self.recapture_extend(payload)

    def graphs(self, _=None):
        return self._rpc("graphs", None)


def parallel(core):
    """{"world", "dp", "dp_rank", "expected", "note"} from the engine's parallel config: the
    worker ranks this engine core's collective_rpc reaches (TP x PP) and, at data parallel >1,
    the other DP groups as `dp<d>`, which this host cannot reach. No config: no expectation."""
    cfg = getattr(core, "vllm_config", None) or getattr(
        getattr(core, "model_executor", None), "vllm_config", None)
    pc = getattr(cfg, "parallel_config", None)
    if pc is None:
        return {"world": None, "dp": None, "dp_rank": None, "expected": None, "note": None}
    world = int(getattr(pc, "world_size", 1) or 1)
    dp = int(getattr(pc, "data_parallel_size", 1) or 1)
    rank = int(getattr(pc, "data_parallel_rank", 0) or 0)
    expected = list(range(world)) + [f"dp{d}" for d in range(dp) if d != rank]
    note = None if dp == 1 else (
        f"data parallel {dp}: this host (DP rank {rank}) reaches only its own group's "
        f"{world} worker(s); the other groups are not participants, so transactions are "
        "refused (participant_missing). DP>1 is not supported by the adapter yet")
    return {"world": world, "dp": dp, "dp_rank": rank, "expected": expected, "note": note}


class RecaptureCoordinator(C.StepGapCoordinator):
    """StepGapCoordinator plus selective CUDA graph recapture after a commit, in the same gap
    (see the module doc). `rebuild(txn, spec)` reaches the worker ranks."""

    def __init__(self, *a, rebuild=None, **k):
        super().__init__(*a, **k)
        self.rebuild = rebuild

    def _phase2(self, txn, tl):
        out = super()._phase2(txn, tl)
        plan = self.txns[txn]["plan"]
        rebuild_keys = plan.get("rebuild") or ()
        if out["outcome"] != "committed" or not ("recapture" in rebuild_keys or "R1" in rebuild_keys):
            return out
        spec = dict(plan.get("recapture") or plan.get("r1") or {})
        t0 = self.clock()
        r = self.rebuild(txn, spec)
        tl["rebuilt"] = self.clock()
        ranks = r.get("ranks") or []
        dur = round(tl["rebuilt"] - t0, 6)
        route_name = "recapture" if "recapture" in rebuild_keys else "R1"
        out["rebuild"] = {"ok": bool(r.get("ok")), "routes": [route_name],
                          "seconds": {route_name: dur, "R1": dur, "recapture": dur}, "spec": spec,
                          "ranks": {str(x.get("rank")): {k: x.get(k) for k in (
                              "ok", "error", "seconds", "captured", "modes", "wrappers",
                              "eager", "memory", "skipped")} for x in ranks}}
        if r.get("ok"):
            self.txns[txn]["rebuilt"] = [route_name]
            return out
        fs = [f for x in ranks for f in x.get("findings", [])] or [F.finding(
            "rebuild_failed", str(r.get("error")), "recapture on every rank")]
        out["extra"] += [{**f, "participant": "workers"} for f in fs]
        rb = out["rollback"] = {"status": "undone", "participants": {}}
        res = self.txns[txn]["res"]
        for p in reversed(self.participants):
            rb["participants"][p.name] = res[p.name]["undo"] = self._bounded(
                p, "undo", txn, None, set())
        if not all(u["ok"] for u in rb["participants"].values()):
            rb["status"] = "partial"
        out["outcome"] = "rolled_back"
        tl.pop("committed", None)
        tl["gap_end"] = self.clock()
        return out


R1Coordinator = RecaptureCoordinator


class Host:
    def __init__(self, core, root):
        import vllm
        self.core, self.root = core, root
        self.loop_thread = threading.current_thread()
        self.pending = []                   # phase-2 jobs waiting for in-flight batches
        self.held_steps = 0
        self.gap_timeout = GAP_TIMEOUT_S
        self.steps = collections.deque(maxlen=100000)   # (start perf_counter, seconds)
        self.core_side = Side("core", scope={"core": core})
        self.dp = parallel(core)
        self.workers = WorkerGroup(self, self.dp["expected"])
        self.co = RecaptureCoordinator(
            [self.workers, C.CallParticipant("core", self.core_side.dispatch, bounded=True)],
            gap=lambda fn: self.in_loop(fn, gap=True, timeout=self.gap_timeout),
            phase2_deadline_s=PHASE2_DEADLINE_S,
            writer=R.ReceiptWriter(os.path.join(root, "receipts.jsonl")),
            engine={"name": "vllm", "version": vllm.__version__, "commit": None},
            rebuild=lambda txn, spec: self.workers.recapture({"txn": txn, **spec}))

    def in_loop(self, fn, gap=False, timeout=60.0):
        """Run fn on the busy-loop thread (between steps; with gap=True, also with no batch
        in flight) and return its result. The loop must start the job within `timeout` (else
        it is cancelled and never runs); once started it may run `timeout` more seconds."""
        if threading.current_thread() is self.loop_thread:
            return fn()
        job = _Job(fn, gap)
        self.core.input_queue.put_nowait((TAG, job))
        if not job.started.wait(timeout):
            with job.lock:
                if not job.started.is_set():
                    job.cancelled = True
                    raise TimeoutError(f"the engine loop did not run the job within {timeout} s")
        if not job.done.wait(timeout):
            raise LateJob(f"the engine loop started the job but it did not finish within "
                          f"{timeout} s")
        if job.error is not None:
            raise job.error
        return job.value

    def step_stats(self, t0, t1):
        d = [s for start, s in list(self.steps) if t0 <= start <= t1]
        if not d:
            return {"steps": 0}
        return {"steps": len(d), "median_s": round(statistics.median(d), 6),
                "max_s": round(max(d), 6)}

    # -- control channel -----------------------------------------------------------------
    def _op(self, op, cmd):
        if op == "txn":
            return self.co.run(cmd["plan"], edit_time=cmd.get("edit_time"),
                               baseline_restart=cmd.get("baseline_restart"))
        if op == "prove":
            return self.co.prove(cmd["txn"], golden=cmd.get("golden"))
        if op == "undo":
            return self.co.undo(cmd["txn"])
        if op == "status":                   # no txn: the host's overall status
            return self.co.status(cmd.get("txn"))
        if op == "pstatus":                  # each participant's own state (per rank)
            if cmd.get("txn") is None:
                return F.not_ok("pstatus needs a txn (status without one is the host's)")
            return F.ok(workers=self.workers.status(cmd["txn"]),
                        core=self.core_side.dispatch("status", cmd["txn"]))
        if op == "fingerprint":
            mods = list(cmd.get("modules") or [])
            return F.ok(workers=self.workers.fingerprint(mods),
                        core=self.core_side.dispatch("fingerprint", mods))
        if op == "versions":                 # per-step version logs, per rank and core
            return F.ok(workers=self.workers.versions(), core=self.core_side.versions())
        if op == "at_base":                  # back at the start state? (warm engine pool)
            return self.co.at_base()
        if op == "graphs":
            return F.ok(workers=self.workers.graphs())
        if op in ("recapture_extend", "r1_extend"): # in a gap: no batch replays half-captured graphs
            t0 = time.perf_counter()
            r = self.in_loop(lambda: self.workers.recapture_extend(
                {"txn": cmd["txn"], "sizes": cmd.get("sizes")}), gap=True,
                timeout=self.gap_timeout)
            return {**r, "seconds": round(time.perf_counter() - t0, 6)}
        if op == "ping":
            return F.ok(pid=os.getpid(), poison=self.co.poison, parallel=self.dp)
        return F.not_ok(f"unknown op {op!r}")

    def handle(self, cmd):
        op = cmd.get("op")
        self.gap_timeout = float(cmd.get("gap_timeout_s") or GAP_TIMEOUT_S)
        self.co.phase2_deadline_s = (cmd["phase2_deadline_s"] or None
                                     if "phase2_deadline_s" in cmd else PHASE2_DEADLINE_S)
        t0, held0 = time.perf_counter(), self.held_steps
        try:
            r = self._op(op, cmd)
        except Exception as e:
            r = F.not_ok(f"{op}: {type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        t1 = time.perf_counter()
        r["host"] = {"op": op, "seconds": round(t1 - t0, 6), "held_steps": self.held_steps - held0,
                     "steps_during": self.step_stats(t0, t1),
                     "steps_before": self.step_stats(t0 - 5.0, t0)}
        tl = (r.get("receipt") or {}).get("timeline") or {}
        off = time.time() - time.perf_counter()               # unix -> perf_counter
        if "prepare_start" in tl and "prepared" in tl:
            r["host"]["steps_prepare"] = self.step_stats(tl["prepare_start"] - off,
                                                         tl["prepared"] - off)
        if "gap_start" in tl and "gap_end" in tl:             # did any step start in the gap?
            a, b = tl["gap_start"] - off, tl["gap_end"] - off
            r["host"]["steps_started_in_gap"] = sum(1 for st, _ in list(self.steps)
                                                    if a <= st <= b)
        return r

    def control(self):
        ctl = os.path.join(self.root, "ctl.json")
        os.makedirs(os.path.join(self.root, "reply"), exist_ok=True)
        last = 0
        while True:
            try:
                with open(ctl) as f:
                    cmd = json.load(f)
            except (OSError, ValueError):
                cmd = None
            if cmd and cmd.get("seq", 0) > last:
                last = cmd["seq"]
                r = self.handle(cmd)
                tmp = os.path.join(self.root, "reply", f".{last}.tmp")
                with open(tmp, "w") as f:
                    json.dump(r, f, default=str)
                os.replace(tmp, os.path.join(self.root, "reply", f"{last}.json"))
            time.sleep(0.01)


def _patch_core(mod):
    P = mod.EngineCoreProc
    orig_handle, orig_step, orig_loop = (P._handle_client_request, P._process_engine_step,
                                         P.run_busy_loop)

    def _handle_client_request(self, request_type, request):
        if request_type != TAG:
            return orig_handle(self, request_type, request)
        if request.gap and self.batch_queue:     # a batch is in flight: after it lands
            _HOST[0].pending.append(request)
        else:
            request.run()

    def _process_engine_step(self):
        h = _HOST[0] if _HOST else None
        hold = bool(h and h.pending and self.batch_queue)
        if hold:                                 # pause keep: schedule nothing new
            self.scheduler.has_requests = lambda: False
            h.held_steps += 1
        t = time.perf_counter()
        try:
            r = orig_step(self)
        finally:
            if hold:
                del self.scheduler.has_requests
        if h is not None:
            h.steps.append((t, time.perf_counter() - t))
            while h.pending and not self.batch_queue:
                h.pending.pop(0).run()
        return r

    def run_busy_loop(self):
        pc = getattr(getattr(self, "vllm_config", None), "parallel_config", None)
        if DIR and not _HOST and int(getattr(pc, "data_parallel_rank", 0) or 0) != 0:
            print(f"{TAG}: DP rank {pc.data_parallel_rank}: no host (DP rank 0 serves the "
                  "channel)", file=sys.stderr, flush=True)
            return orig_loop(self)
        if DIR and not _HOST:
            h = Host(self, DIR)
            _HOST.append(h)
            threading.Thread(target=h.control, daemon=True, name="fastrsi-kernel-host").start()
        return orig_loop(self)

    P._handle_client_request = _handle_client_request
    P._process_engine_step = _process_engine_step
    P.run_busy_loop = run_busy_loop


class _After(importlib.abc.MetaPathFinder):
    """Patch vllm.v1.engine.core right after it is imported, from wherever the other import
    finders load it: it asks the finders after it on sys.meta_path, in order (an overlay
    finder first, then the installed package), instead of PathFinder directly, which would
    bypass the overlay and report a false base_mismatch."""

    def find_spec(self, name, path, target=None):
        if name != "vllm.v1.engine.core":
            return None
        spec = None
        for f in list(sys.meta_path):
            if f is self or isinstance(f, _After) or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        orig = spec.loader.exec_module

        def exec_module(module):
            orig(module)
            _patch_core(module)
        spec.loader.exec_module = exec_module
        return spec


if DIR and not any(isinstance(f, _After) for f in sys.meta_path):
    if "vllm.v1.engine.core" in sys.modules:     # imported before this module: patch it now
        _patch_core(sys.modules["vllm.v1.engine.core"])
    else:
        sys.meta_path.insert(0, _After())
