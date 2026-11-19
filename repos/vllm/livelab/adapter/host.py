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
  (capture more of the descriptors a recapture left eager, in a gap; if any rank fails, the
  ranks that captured drop it in the same gap; refused while the host is poisoned).
- Recapture (`RecaptureCoordinator`): a plan with `"rebuild": ["recapture"]` (or `"rebuild": ["R1"]`,
  optional `"recapture"` or `"r1"`: `{"modes": [mode names], "sizes": [num_tokens]}`) is recaptured
  on every rank right after its commit, in the same gap, so no step replays a graph of the old code
  next to Python running the new one. The receipt's `rebuild` and `recapture_s` say what it captured
  and how long it took; the edit runs on every step from `rebuilt` on (edit_to_run_s). A recapture that
  fails on any rank is `rebuild_failed`: the transaction is undone on every participant in the same gap
  (each rank's recapture puts its old graphs back first), outcome rolled_back; that undo has the
  phase-2 deadline, and a participant that misses it poisons the host. Undo of a recaptured
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
- The import finder that patches vllm.v1.engine.core asks the finders after it on
  sys.meta_path for the module (an overlay first), so import this module after the overlay's
  finder.

The host records every step's duration, so a reply says how long steps stalled.
"""
import collections
import itertools
import os
import statistics
import sys
import threading
import time
import traceback

from fastrsi import hooks, lab as lab_base
from kernel import coordinator as C
from kernel import findings as F
from kernel import receipt as R
from services import core_hook

from .side import Side

TAG = "FASTRSI_KERNEL"
DIR = os.environ.get("FASTRSI_KERNEL_DIR")
GAP_TIMEOUT_S = float(os.environ.get("FASTRSI_KERNEL_GAP_TIMEOUT", "30"))
PHASE2_DEADLINE_S = float(os.environ.get("FASTRSI_KERNEL_PHASE2_DEADLINE", "5")) or None
RPC_GRACE_S = float(os.environ.get("FASTRSI_KERNEL_RPC_GRACE", "2"))
PREPARE_TIMEOUT_S = 300.0
_HOST = []
_Job = core_hook.LoopJob
LateJob = core_hook.LateJob


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
        return C.expect_ranks(self, method, out)

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

    def recapture_extend_drop(self, payload):
        return self._rpc("recapture_extend_drop", payload)

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


class RecaptureCoordinator(lab_base.RecaptureCoordinator):
    """`fastrsi.lab.RecaptureCoordinator` plus vLLM's in-loop pending-rollback gate."""

    def _queue_rollback_gate(self):
        gap_owner = getattr(self.gap, "__self__", None)
        gap_obj = getattr(gap_owner, "gap", None)
        queue_list = getattr(gap_obj, "_queue", None)
        lock_obj = getattr(gap_obj, "_lock", None)
        if queue_list is None or lock_obj is None:
            return

        def hold_until_rollback_done():
            self._in_rollback_gate = True
            try:
                while getattr(gap_obj, "active", True) and self.poll_rollback():
                    while True:
                        with lock_obj:
                            if not queue_list:
                                break
                            queued_job = queue_list.pop(0)
                        queued_job()
                    time.sleep(0.005)
            finally:
                self._in_rollback_gate = False

        with lock_obj:
            if getattr(gap_obj, "active", True):
                queue_list.append(hold_until_rollback_done)

    def _in_gap(self, fn):
        def wrapped():
            try:
                return fn()
            finally:
                if not self._in_rollback_gate and self.has_pending_rollback():
                    self._queue_rollback_gate()
        return super()._in_gap(wrapped)


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
        self._extends = itertools.count(1)
        self.dp = parallel(core)
        self.workers = WorkerGroup(self, self.dp["expected"])
        self.co = RecaptureCoordinator(
            [self.workers, C.CallParticipant("core", self.core_side.dispatch, bounded=True)],
            gap=self._in_gap,
            phase2_deadline_s=PHASE2_DEADLINE_S,
            writer=R.ReceiptWriter(os.path.join(root, "receipts.jsonl")),
            engine={"name": "vllm", "version": vllm.__version__, "commit": None},
            rebuild=lambda txn, spec: self.workers.recapture({"txn": txn, **spec}))

    def _in_gap(self, fn):
        try:
            return self.in_loop(fn, gap=True, timeout=self.gap_timeout)
        except LateJob as error:
            self.workers._break(
                "phase2", None, error,
                f"phase-2 gap job: {error}; it may still apply: respawn",
                "phase-2 gap job finishes in the loop while its caller waits",
            )
            raise

    def _extend(self, txn, sizes):
        """recapture_extend on every rank, in one gap job. If any rank fails, the ranks that
        captured drop that extend in the same gap, so every rank keeps replaying the same
        descriptors; a drop that fails poisons the host (ranks may replay different graphs)."""
        payload = {"txn": txn, "sizes": sizes, "extend_id": f"{txn}/extend{next(self._extends)}"}
        reply = self.workers.recapture_extend(payload)
        if reply.get("ok"):
            return reply
        drop = self.workers.recapture_extend_drop(payload)
        reply = {**reply, "dropped": drop}
        if not drop.get("ok"):
            finding = {**F.finding(
                "rebuild_failed", f"recapture_extend of {txn} failed on a rank and the other "
                f"ranks could not drop it: {drop.get('error')}",
                "every rank replays the same descriptors"), "participant": "workers"}
            self.co.poison.append(finding)
            reply["findings"] = list(reply.get("findings") or []) + [finding]
        return reply

    def in_loop(self, fn, gap=False, timeout=60.0):
        """Run fn on the busy-loop thread (between steps; with gap=True, also with no batch
        in flight) and return its result. The loop must start the job within `timeout` (else
        it is cancelled and never runs); once started it may run `timeout` more seconds."""
        return core_hook.in_loop(
            self.core, TAG, fn, gap=gap, timeout=timeout, loop_thread=self.loop_thread
        )

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
        if op == "recovered":
            return self.co.recovered()
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
            if self.co.poison:
                return F.not_ok("host poisoned: undo or respawn first", poison=self.co.poison)
            t0 = time.perf_counter()
            reply = self._in_gap(lambda: self._extend(cmd["txn"], cmd.get("sizes")))
            return {**reply, "seconds": round(time.perf_counter() - t0, 6)}
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

def _cur():
    return _HOST[0] if _HOST else None


def _drain(host, core):
    host.pending = [j for j in host.pending if not getattr(j, "cancelled", False)]
    while host.pending and not core.batch_queue:
        host.pending.pop(0).run()


def _on_request(core, request):
    if request.gap and core.batch_queue:         # a batch is in flight: after it lands
        _HOST[0].pending.append(request)
    else:
        request.run()


def _pre_step(core):
    host = _cur()
    if host is None:
        return True
    co = getattr(host, "co", None)
    host.loop_thread = threading.current_thread()
    _drain(host, core)
    if not core.batch_queue and co is not None and co.poll_rollback():
        host.held_steps += 1
        time.sleep(0.005)
        return False
    return True


def _should_hold(core):
    host = _cur()
    co = getattr(host, "co", None)
    return bool(host and (host.pending or (co is not None and co.has_pending_rollback()))
                and core.batch_queue)


def _before_step(_core, _hold):
    side = getattr(_cur(), "core_side", None)
    if side is not None:
        side.tick(True)


def _after_step(core, step_start, _result):
    host = _cur()
    if host is None:
        return
    if getattr(host, "core_side", None) is not None:
        host.core_side.tick(False)
    host.steps.append((step_start, time.perf_counter() - step_start))
    _drain(host, core)


def _on_loop_start(core):
    root = DIR or os.environ.get("FASTRSI_KERNEL_DIR")
    pc = getattr(getattr(core, "vllm_config", None), "parallel_config", None)
    if root and not _HOST and int(getattr(pc, "data_parallel_rank", 0) or 0) != 0:
        print(f"{TAG}: DP rank {pc.data_parallel_rank}: no host (DP rank 0 serves the "
              "channel)", file=sys.stderr, flush=True)
        return None
    if root and not _HOST:
        h = Host(core, root)
        _HOST.append(h)
        return (h.root, h.handle, 0.01)
    return None


core_hook.register(core_hook.CorePlugin(
    name="kernel_host", tag=TAG, on_loop_start=_on_loop_start, on_request=_on_request,
    pre_step=_pre_step, should_hold=_should_hold, on_hold=lambda _: setattr(_HOST[0], "held_steps", _HOST[0].held_steps + 1),
    before_step=_before_step, after_step=_after_step,
))

_patch_core = core_hook._patch_core


class _After(hooks.AfterImportHook):
    """Stack-safe import hook patching `vllm.v1.engine.core` via downstream finders only."""

    def __init__(self):
        super().__init__(core_hook.TARGET_MODULE, _patch_core, once=False)


if DIR:
    core_hook.install()
