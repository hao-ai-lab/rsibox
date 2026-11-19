"""Live scheduler reconfiguration inside a running vLLM engine core: the consumer that
`engine.py reconfigure` talks to for the flags vLLM can change without a restart.

Two halves, one file:

- **Agent** (engine-core process). Loaded by `live_site/sitecustomize.py` when
  FASTRSI_LIVE_RECONFIG_DIR=<dir> is in the engine's environment (engine.py sets both). It
  patches `vllm.v1.engine.core.EngineCoreProc` as that module is imported:
  - `run_busy_loop` starts a control thread and writes `<dir>/ready.json` {pid, api_pid, boot,
    live_fields, dp_size}. Only DP rank 0 of a DP=1 engine serves the channel.
  - The control thread polls `<dir>/ctl.json` {"seq", "op": "set", "values": {...}}. Each
    command runs as a job on the engine core's own input queue, so it runs on the busy-loop
    thread between two engine steps (also when the engine is idle: the loop blocks on that
    queue). The reply goes to `<dir>/reply/<seq>.json` with the values read back from the
    scheduler before and after.
  - Every step first lowers `scheduler.max_num_running_reqs` toward its target, never below
    the number of running requests (vLLM's admission check is `len(running) == max`, and it
    asserts `len(running) <= max` after scheduling): a lower max_num_seqs drains, it never
    preempts or crashes.
- **Client** (`request()`, used by engine.py): write ctl.json, wait for the reply.

What can be changed live (LIVE_FIELDS), and only within what the engine was built for:
- `max_num_seqs` -> `scheduler.max_num_running_reqs`, 1 <= value <= the boot value (the
  model runner's input batch and CUDA graphs are sized by the boot value).
- `max_num_batched_tokens` -> `scheduler.max_num_scheduled_tokens`, value <= the boot value
  (runner buffers are sized by it); below max_model_len only with chunked prefill on.
- `scheduling_policy` rebuilds both waiting queues for fcfs or priority.
Everything else (max_model_len, prefix caching, eager/graph mode, ...) has
no live consumer: engine.py restarts the engine for it and says so.

Lab use only. No vLLM files are edited; the patch is applied at import.
"""
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
import corelab  # noqa: F401,E402  (core/livelab on sys.path)
from fastrsi import hooks, ipc  # noqa: E402
from services import core_hook  # noqa: E402

TAG = "FASTRSI_LIVE_RECONFIG"
ENV = "FASTRSI_LIVE_RECONFIG_DIR"
POLICY_FILE = "policy.json"
LIVE_FIELDS = ("max_num_seqs", "max_num_batched_tokens", "scheduling_policy")
_AGENT = []
_POLICY_CACHE = {"path": None, "mtime_ns": None, "policy": None}
_write_json = ipc.atomic_write_json
_read_json = ipc.read_json


# -- validation (shared by agent and client) ----------------------------------------------

def check_values(values, boot):
    """Problems with applying `values` live to an engine whose boot settings are `boot`
    ({max_num_seqs, max_num_batched_tokens, max_model_len, chunked_prefill}); [] when ok."""
    problems = []
    for name, value in values.items():
        if name not in LIVE_FIELDS:
            problems.append(f"{name}: no live consumer (live: {', '.join(LIVE_FIELDS)})")
            continue
        if name == "scheduling_policy":
            if value not in ("fcfs", "priority"):
                problems.append(f"scheduling_policy={value!r}: expected fcfs or priority")
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            problems.append(f"{name}={value!r}: must be a positive integer")
            continue
        cap = boot.get(name)
        if cap is not None and value > cap:
            problems.append(f"{name}={value} above the boot value {cap}: the runner's buffers "
                            "and CUDA graphs are sized by the boot value; restart")
    tokens = values.get("max_num_batched_tokens")
    if (isinstance(tokens, int) and not boot.get("chunked_prefill")
            and boot.get("max_model_len") and tokens < boot["max_model_len"]):
        problems.append(f"max_num_batched_tokens={tokens} below max_model_len "
                        f"{boot['max_model_len']} without chunked prefill: long prompts would "
                        "never be scheduled")
    return problems


# -- agent (engine-core process) -----------------------------------------------------------

def _switch_policy(scheduler, value):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    policy = SchedulingPolicy(value)
    if policy == scheduler.policy:
        return
    # Stage all replacements before publishing any on the engine-loop thread.
    queue_attrs = [
        attr for attr in ("waiting", "kv_holding_waiting")
        if getattr(scheduler, attr, None) is not None
    ]
    rebuilt = []
    for attr in queue_attrs:
        old = getattr(scheduler, attr)
        queue = create_request_queue(policy)
        for request in sorted(old, key=lambda r: r.arrival_time):
            queue.add_request(request)
        rebuilt.append((attr, queue))
    for attr, queue in rebuilt:
        setattr(scheduler, attr, queue)
    scheduler.policy = policy
    if getattr(scheduler, "scheduler_config", None) is not None:
        scheduler.scheduler_config.policy = value


def _sync_api_policy(obj):
    """In the parent API server process: keep `scheduler_config.policy` synced with the engine
    core's live policy (`<root>/policy.json`) so `/v1/completions` priority validation matches."""
    root = os.environ.get(ENV)
    if not root:
        return
    path = os.path.join(root, POLICY_FILE)
    try:
        st = os.stat(path)
        sig = (st.st_mtime_ns, st.st_ino, st.st_size)
    except OSError:
        return
    if _POLICY_CACHE["path"] != path or _POLICY_CACHE["mtime_ns"] != sig:
        data = _read_json(path)
        if not isinstance(data, dict) or data.get("scheduling_policy") not in ("fcfs", "priority"):
            return
        _POLICY_CACHE.update(path=path, mtime_ns=sig, policy=data["scheduling_policy"])
    policy = _POLICY_CACHE["policy"]
    sub = (obj, getattr(obj, "input_processor", None), getattr(obj, "processor", None))
    for holder in (*sub, *(getattr(x, "vllm_config", None) for x in sub)):
        sc = getattr(holder, "scheduler_config", None)
        if sc is not None and getattr(sc, "policy", None) != policy:
            sc.policy = policy


_Job = core_hook.LoopJob


class Agent:
    def __init__(self, core, root):
        self.core, self.root = core, root
        sched = core.scheduler
        cfg = core.vllm_config
        sc = cfg.scheduler_config
        self.boot = {
            "max_num_seqs": int(sched.max_num_running_reqs),
            "max_num_batched_tokens": int(sched.max_num_scheduled_tokens),
            "max_model_len": int(cfg.model_config.max_model_len),
            "chunked_prefill": bool(getattr(sc, "enable_chunked_prefill", False)),
        }
        self.target_seqs = self.boot["max_num_seqs"]
        self.steps = 0
        self.lock = threading.Lock()

    def scheduler_view(self):
        s = self.core.scheduler
        return {"max_num_running_reqs": int(s.max_num_running_reqs),
                "max_num_scheduled_tokens": int(s.max_num_scheduled_tokens),
                "target_max_num_seqs": int(self.target_seqs),
                "scheduling_policy": s.policy.value,
                "running": len(s.running), "waiting": len(s.waiting), "steps": self.steps}

    def tick(self):
        """Before every step, on the loop thread: drain toward the max_num_seqs target."""
        self.steps += 1
        s = self.core.scheduler
        want = max(self.target_seqs, len(s.running))
        if s.max_num_running_reqs != want and want <= self.boot["max_num_seqs"]:
            s.max_num_running_reqs = want

    def apply(self, values):
        """On the loop thread, between two steps."""
        t0 = time.perf_counter()
        before = self.scheduler_view()
        problems = check_values(values, self.boot)
        if problems:
            return {"ok": False, "error": "; ".join(problems), "before": before,
                    "after": before, "boot": self.boot}
        s = self.core.scheduler
        if "scheduling_policy" in values:
            _switch_policy(s, values["scheduling_policy"])
            if self.root:
                _write_json(os.path.join(self.root, POLICY_FILE),
                            {"scheduling_policy": values["scheduling_policy"]})
        if "max_num_batched_tokens" in values:
            s.max_num_scheduled_tokens = int(values["max_num_batched_tokens"])
        if "max_num_seqs" in values:
            self.target_seqs = int(values["max_num_seqs"])
            s.max_num_running_reqs = max(self.target_seqs, len(s.running))
        after = self.scheduler_view()
        return {"ok": True, "applied": dict(values), "before": before, "after": after,
                "draining": after["max_num_running_reqs"] != after["target_max_num_seqs"],
                "boot": self.boot, "apply_s": round(time.perf_counter() - t0, 6),
                "pid": os.getpid()}

    def in_loop(self, fn, timeout=30.0):
        return core_hook.in_loop(self.core, TAG, fn, gap=False, timeout=timeout)

    def handle(self, cmd):
        op = cmd.get("op")
        t0 = time.perf_counter()
        try:
            if op == "set":
                r = self.in_loop(lambda: self.apply(dict(cmd.get("values") or {})))
            elif op == "status":
                r = {"ok": True, **self.in_loop(self.scheduler_view), "boot": self.boot}
            else:
                r = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:          # noqa: BLE001
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        r["seq"], r["roundtrip_in_engine_s"] = cmd.get("seq"), round(time.perf_counter() - t0, 6)
        return r


def _before_step(core, hold):
    if not hold and _AGENT and _AGENT[0].core is core:
        _AGENT[0].tick()


def _on_loop_start(core):
    root = os.environ.get(ENV)
    pc = getattr(getattr(core, "vllm_config", None), "parallel_config", None)
    dp_size = int(getattr(pc, "data_parallel_size", 1) or 1)
    if root and not _AGENT and dp_size == 1:
        try:
            agent = Agent(core, root)
            os.makedirs(root, exist_ok=True)
            _AGENT.append(agent)
            _write_json(os.path.join(root, "ready.json"), {
                "pid": os.getpid(), "api_pid": os.getppid(), "boot": agent.boot,
                "live_fields": list(LIVE_FIELDS), "dp_size": dp_size, "t": time.time()})
            return (root, agent.handle, 0.005)
        except Exception as e:          # noqa: BLE001 - the engine runs without the agent
            print(f"{TAG}: agent not started: {e!r}", file=sys.stderr, flush=True)
    elif root and dp_size != 1:
        print(f"{TAG}: data_parallel_size={dp_size}: no live agent (DP=1 only)",
              file=sys.stderr, flush=True)
    return None


def _on_patch(proc_cls):
    proc_cls._fastrsi_live_reconfig = True


def _register_core_plugin():
    core_hook.register(core_hook.CorePlugin(
        name="live_reconfig",
        tag=TAG,
        on_loop_start=_on_loop_start,
        before_step=_before_step,
        on_patch=_on_patch,
    ))


def _patch_core(mod):
    _register_core_plugin()
    core_hook._patch_core(mod)


def _patch_api_entry(mod, cls_names, method_name):
    for cls_name in cls_names:
        cls = getattr(mod, cls_name, None)
        if cls is None or getattr(cls, "_fastrsi_live_policy", False):
            continue
        orig = getattr(cls, method_name, None)
        if orig is None:
            continue

        def wrapped(self, *a, _orig=orig, **k):
            _sync_api_policy(self)
            return _orig(self, *a, **k)

        wrapped.__wrapped__ = orig
        setattr(cls, method_name, wrapped)
        cls._fastrsi_live_policy = True


def _patch_input_processor(mod):
    _patch_api_entry(mod, ("InputProcessor", "Processor"), "process_inputs")


def _patch_async_llm(mod):
    _patch_api_entry(mod, ("AsyncLLM",), "add_request")


def _patch_http_priority(mod):
    """Validate the resolved header/body priority before vLLM starts an HTTP stream."""
    cls = getattr(mod, "GenerateBaseServing", None)
    resolve = getattr(cls, "_get_priority", None)
    if resolve is None or getattr(resolve, "_fastrsi_live_policy", False):
        return

    def get_priority(self, request, raw_request):
        priority = resolve(request, raw_request)
        if priority:
            client = self.engine_client
            _sync_api_policy(client)
            if client.vllm_config.scheduler_config.policy != "priority":
                from vllm.exceptions import VLLMValidationError
                raise VLLMValidationError("Nonzero priority requires priority scheduling.",
                                          parameter="priority", value=priority)
        return priority

    get_priority._fastrsi_live_policy = True
    cls._get_priority = get_priority


_IMPORT_HOOKS = (
    ("vllm.entrypoints.generate.base.serving", _patch_http_priority),
    (core_hook.TARGET_MODULE, core_hook._patch_core),
    ("vllm.v1.engine.input_processor", _patch_input_processor),
    ("vllm.v1.engine.processor", _patch_input_processor),
    ("vllm.v1.engine.async_llm", _patch_async_llm),
)


def install():
    """Called by live_site/sitecustomize.py when ENV is set."""
    _register_core_plugin()
    for mod_name, patcher in _IMPORT_HOOKS:
        hooks.install_import_hook(mod_name, patcher, once=False)


# -- client (engine.py) --------------------------------------------------------------------

def read_ready(root):
    return ipc.read_json(os.path.join(root, "ready.json"))


def request(root, op, timeout=15.0, **payload):
    """Send one command; return the agent's reply, or {"ok": False, "error"} on timeout."""
    r = ipc.file_channel_call(root, {"op": op, **payload}, timeout=timeout, poll_s=0.002)
    r["client_roundtrip_s"] = r.get("client_round_trip_s", 0.0)
    if r.get("lost"):
        return {"ok": False, "error": f"no reply from the live agent in {timeout} s",
                "client_roundtrip_s": r["client_roundtrip_s"]}
    return r
