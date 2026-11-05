"""Engine-core host: kernel transactions inside vLLM's EngineCore busy loop.

Enable with FASTRSI_KERNEL_DIR=<dir> in the engine's environment and import this module early
in every engine process (a sitecustomize does it); add `--worker-extension-cls
kernel_vllm.worker_ext.KernelWorkerExtension`. Lab use only: it runs code sent to it.

How a transaction runs (kernel StepGapCoordinator, loop.md 4b):

- Control: a file channel. A client writes <dir>/ctl.json `{"seq", "op", ...}`; a host thread
  in the engine-core process picks it up, runs it, and writes <dir>/reply/<seq>.json. Ops:
  `txn {plan, edit_time, baseline_restart}`, `prove {txn, golden}`, `undo {txn}`,
  `status {txn}`, `ping`. Replies carry the kernel's receipt records; the receipts are also
  appended to <dir>/receipts.jsonl by the kernel's writer.
- Phase 1 (prepare) while the engine steps: the engine core prepares in the host thread; the
  workers prepare in a background thread each (`prepare_start`, then `prepare_poll`).
- Every call to the workers is `model_executor.collective_rpc` issued FROM the busy loop
  (a job put on the engine core's input queue as a custom item), never from another thread:
  the executor's message queue is not thread-safe, and from the loop every rank handles the
  call between the same two steps.
- Phase 2 has a deadline (kernel `phase2_deadline_s`; FASTRSI_KERNEL_PHASE2_DEADLINE, default
  5 s, or `phase2_deadline_s` per op; 0 turns it off). Each side runs its phase-2 calls on its
  own lane thread and answers by the deadline (`deadline_missed` if the call is still
  running), so a rank that hangs in commit does not hold the engine; the kernel then undoes
  the others in the gap and poisons the host (txn refused; undo allowed). collective_rpc gets
  `timeout` = the time left + FASTRSI_KERNEL_RPC_GRACE (default 2 s) as a backstop for a
  worker process that cannot answer at all. That timeout leaves the late reply in the
  executor's response queue, where the next call would read it, so after it the worker group
  refuses every call (respawn the engine).
- Phase 2 (validate, commit or abort, undo of a failed commit) is one job that runs only when
  no batch is in flight. While it waits, the loop schedules nothing new ("pause keep":
  `scheduler.has_requests` is shadowed to False while `batch_queue` drains), as in the
  engine-core transaction prototype.

The host records every step's duration, so a reply can say how long steps stalled.
"""
import collections
import importlib.abc
import importlib.machinery
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
        self.done = threading.Event()
        self.value = self.error = None
        self.t_queued = time.perf_counter()
        self.t_start = self.t_end = None

    def run(self):
        self.t_start = time.perf_counter()
        try:
            self.value = self.fn()
        except BaseException as e:            # handed back to the waiting caller
            self.error = e
        self.t_end = time.perf_counter()
        self.done.set()


def _merge(method, rs):
    """One reply for the worker group from the per-rank replies of a collective_rpc: the
    kernel's group reply (every rank's reply in `ranks`, so receipts keep per-rank witness
    counts and code hashes), plus prepare_poll's pending flag and timings."""
    out = C.merge_ranks(method, rs)
    rs = out["ranks"]
    if any(r.get("pending") for r in rs):
        out["pending"] = True
    if method == "prepare_poll" and not out.get("pending"):
        out["staged"] = rs[0].get("staged") if rs else {}
        out["timings"] = {f"rank{r.get('rank')}": r.get("timings") for r in rs}
        out["prepare_s"] = max((r.get("prepare_s") or 0) for r in rs) if rs else None
    return out


class WorkerGroup(C.Participant):
    """All worker ranks as one participant: each call is one collective_rpc from the loop.
    Bounded: the ranks answer phase-2 calls by their deadline, and collective_rpc gets a
    timeout as a backstop (see the module docstring)."""
    name = "workers"
    bounded = True

    def __init__(self, host):
        self.host = host
        self.broken = None       # set when a collective_rpc timed out: replies are misaligned

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
        except Exception as e:            # includes the loop not running the job in time
            return F.not_ok(f"collective_rpc {method}: {type(e).__name__}: {e}")
        if isinstance(rs, TimeoutError):
            self.broken = (f"collective_rpc {method} timed out after {timeout} s; a late "
                           "reply stays in the executor's queue: respawn")
            return F.not_ok(f"collective_rpc {method}: TimeoutError: {rs}", [{**F.finding(
                "deadline_missed", self.broken, f"{method} answers within {timeout} s"),
                "participant": self.name}])
        return _merge(method, rs)

    def prepare(self, plan):
        r = self._rpc("prepare_start", plan)
        if not r["ok"]:
            return r
        t = time.perf_counter()
        while True:
            time.sleep(0.02)
            r = self._rpc("prepare_poll", plan["id"])
            if not r["ok"] or not r.get("pending"):
                return r
            if time.perf_counter() - t > PREPARE_TIMEOUT_S:
                return F.not_ok(f"workers did not prepare within {PREPARE_TIMEOUT_S} s")

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

    def versions(self, _=None):
        return self._rpc("versions", None)


class Host:
    def __init__(self, core, root):
        import vllm
        self.core, self.root = core, root
        self.loop_thread = threading.current_thread()
        self.pending = []                   # phase-2 jobs waiting for in-flight batches
        self.held_steps = 0
        self.gap_timeout = GAP_TIMEOUT_S        # per op: cmd["gap_timeout_s"]
        self.steps = collections.deque(maxlen=100000)   # (start perf_counter, seconds)
        self.core_side = Side("core", scope={"core": core})
        self.co = C.StepGapCoordinator(
            [WorkerGroup(self), C.CallParticipant("core", self.core_side.dispatch,
                                                  bounded=True)],
            gap=lambda fn: self.in_loop(fn, gap=True, timeout=self.gap_timeout),
            phase2_deadline_s=PHASE2_DEADLINE_S,
            writer=R.ReceiptWriter(os.path.join(root, "receipts.jsonl")),
            engine={"name": "vllm", "version": vllm.__version__, "commit": None})

    def in_loop(self, fn, gap=False, timeout=60.0):
        """Run fn on the busy-loop thread (between steps; with gap=True, also with no batch
        in flight) and return its result."""
        if threading.current_thread() is self.loop_thread:
            return fn()
        job = _Job(fn, gap)
        self.core.input_queue.put_nowait((TAG, job))
        if not job.done.wait(timeout):
            raise TimeoutError(f"the engine loop did not run the job within {timeout} s")
        if job.error is not None:
            raise job.error
        return job.value

    def step_stats(self, t0, t1):
        d = [s for start, s in self.steps if t0 <= start <= t1]
        if not d:
            return {"steps": 0}
        return {"steps": len(d), "median_s": round(statistics.median(d), 6),
                "max_s": round(max(d), 6)}

    # -- control channel -----------------------------------------------------------------
    def handle(self, cmd):
        op = cmd.get("op")
        self.gap_timeout = float(cmd.get("gap_timeout_s") or GAP_TIMEOUT_S)
        if "phase2_deadline_s" in cmd:        # per op; null or 0 turns the deadline off
            self.co.phase2_deadline_s = cmd["phase2_deadline_s"] or None
        else:
            self.co.phase2_deadline_s = PHASE2_DEADLINE_S
        t0 = time.perf_counter()
        held0 = self.held_steps
        try:
            if op == "txn":
                r = self.co.run(cmd["plan"], edit_time=cmd.get("edit_time"),
                                baseline_restart=cmd.get("baseline_restart"))
            elif op == "prove":
                r = self.co.prove(cmd["txn"], golden=cmd.get("golden"))
            elif op == "undo":
                r = self.co.undo(cmd["txn"])
            elif op == "status":
                r = self.co.status(cmd["txn"])
            elif op == "pstatus":             # each participant's own state (per rank)
                r = F.ok(workers=self.co.participants[0].status(cmd["txn"]),
                         core=self.core_side.dispatch("status", cmd["txn"]))
            elif op == "versions":            # per-rank and core version logs
                r = F.ok(workers=self.co.participants[0].versions(),
                         core=self.core_side.versions())
            elif op == "ping":
                r = F.ok(pid=os.getpid(), poison=self.co.poison)
            else:
                r = F.not_ok(f"unknown op {op!r}")
        except Exception as e:
            r = F.not_ok(f"{op}: {type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        t1 = time.perf_counter()
        r["host"] = {"op": op, "seconds": round(t1 - t0, 6), "held_steps": self.held_steps - held0,
                     "steps_during": self.step_stats(t0, t1),
                     "steps_before": self.step_stats(t0 - 5.0, t0)}
        tl = (r.get("receipt") or {}).get("timeline") or {}
        if "prepare_start" in tl and "prepared" in tl:       # unix times -> perf_counter
            off = time.time() - time.perf_counter()
            r["host"]["steps_prepare"] = self.step_stats(tl["prepare_start"] - off,
                                                         tl["prepared"] - off)
        if "gap_start" in tl and "gap_end" in tl:            # did any step start in the gap?
            off = time.time() - time.perf_counter()
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
        if h is not None:
            h.core_side.tick(True)
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
        if DIR and not _HOST:
            h = Host(self, DIR)
            _HOST.append(h)
            threading.Thread(target=h.control, daemon=True, name="fastrsi-kernel-host").start()
        return orig_loop(self)

    P._handle_client_request = _handle_client_request
    P._process_engine_step = _process_engine_step
    P.run_busy_loop = run_busy_loop


class _After(importlib.abc.MetaPathFinder):
    """Patch vllm.v1.engine.core right after it is imported."""

    def find_spec(self, name, path, target=None):
        if name != "vllm.v1.engine.core":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module):
            orig(module)
            _patch_core(module)
        spec.loader.exec_module = exec_module
        return spec


if DIR and not any(isinstance(f, _After) for f in sys.meta_path):
    sys.meta_path.insert(0, _After())
