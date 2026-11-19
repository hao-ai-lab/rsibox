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
import importlib.abc
import json
import os
import sys
import threading
import time
import traceback

TAG = "FASTRSI_LIVE_RECONFIG"
ENV = "FASTRSI_LIVE_RECONFIG_DIR"
POLICY_FILE = "policy.json"
LIVE_FIELDS = ("max_num_seqs", "max_num_batched_tokens", "scheduling_policy")
_AGENT = []
_POLICY_CACHE = {"path": None, "mtime_ns": None, "policy": None}


def _write_json(path, payload):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, default=str)
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


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
    for holder in (obj, getattr(obj, "vllm_config", None), getattr(obj, "input_processor", None)):
        sc = getattr(holder, "scheduler_config", None)
        if sc is not None and getattr(sc, "policy", None) != policy:
            sc.policy = policy


class _Job:
    def __init__(self, fn):
        self.fn, self.done, self.value, self.error = fn, threading.Event(), None, None

    def run(self):
        try:
            self.value = self.fn()
        except BaseException as e:      # noqa: BLE001 - reported to the client, loop survives
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        finally:
            self.done.set()


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
        job = _Job(fn)
        self.core.input_queue.put_nowait((TAG, job))
        if not job.done.wait(timeout):
            raise TimeoutError(f"engine loop did not run the job within {timeout} s")
        if job.error:
            raise RuntimeError(job.error)
        return job.value

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

    def control(self):
        ctl = os.path.join(self.root, "ctl.json")
        reply_dir = os.path.join(self.root, "reply")
        os.makedirs(reply_dir, exist_ok=True)
        last = 0
        while True:
            try:
                with open(ctl) as f:
                    cmd = json.load(f)
            except (OSError, ValueError):
                cmd = None
            if cmd and int(cmd.get("seq", 0)) > last:
                last = int(cmd["seq"])
                _write_json(os.path.join(reply_dir, f"{last}.json"), self.handle(cmd))
            time.sleep(0.005)


def _patch_core(mod):
    P = mod.EngineCoreProc
    if getattr(P, "_fastrsi_live_reconfig", False):
        return
    orig_handle, orig_step, orig_loop = (P._handle_client_request, P._process_engine_step,
                                         P.run_busy_loop)

    def _handle_client_request(self, request_type, request):
        if request_type == TAG:
            return request.run()
        return orig_handle(self, request_type, request)

    def _process_engine_step(self):
        if _AGENT and _AGENT[0].core is self:
            _AGENT[0].tick()
        return orig_step(self)

    def run_busy_loop(self):
        root = os.environ.get(ENV)
        pc = getattr(getattr(self, "vllm_config", None), "parallel_config", None)
        dp_size = int(getattr(pc, "data_parallel_size", 1) or 1)
        if root and not _AGENT and dp_size == 1:
            try:
                agent = Agent(self, root)
                os.makedirs(root, exist_ok=True)
                _AGENT.append(agent)
                threading.Thread(target=agent.control, daemon=True,
                                 name="fastrsi-live-reconfig").start()
                _write_json(os.path.join(root, "ready.json"), {
                    "pid": os.getpid(), "api_pid": os.getppid(), "boot": agent.boot,
                    "live_fields": list(LIVE_FIELDS), "dp_size": dp_size, "t": time.time()})
            except Exception as e:      # noqa: BLE001 - the engine runs without the agent
                print(f"{TAG}: agent not started: {e!r}", file=sys.stderr, flush=True)
        elif root and dp_size != 1:
            print(f"{TAG}: data_parallel_size={dp_size}: no live agent (DP=1 only)",
                  file=sys.stderr, flush=True)
        return orig_loop(self)

    P._handle_client_request = _handle_client_request
    P._process_engine_step = _process_engine_step
    P.run_busy_loop = run_busy_loop
    P._fastrsi_live_reconfig = True


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


class _After(importlib.abc.MetaPathFinder):
    """Patch target vLLM modules right after they are imported, via the finders after this one
    (an overlay finder first, then the installed package). Only the finders after it: a hook
    before it that asks every finder would otherwise call back into it without end."""

    TARGETS = {
        "vllm.v1.engine.core": _patch_core,
        "vllm.v1.engine.input_processor": _patch_input_processor,
        "vllm.v1.engine.processor": _patch_input_processor,
        "vllm.v1.engine.async_llm": _patch_async_llm,
    }

    def find_spec(self, name, path, target=None):
        patcher = self.TARGETS.get(name)
        if patcher is None:
            return None
        spec = None
        finders = list(sys.meta_path)
        later = finders[finders.index(self) + 1:] if self in finders else finders
        for finder in later:
            if isinstance(finder, _After) or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        orig = spec.loader.exec_module

        def exec_module(module):
            orig(module)
            patcher(module)
        spec.loader.exec_module = exec_module
        return spec


def install():
    """Called by live_site/sitecustomize.py when ENV is set."""
    for mod_name, patcher in _After.TARGETS.items():
        if mod_name in sys.modules:
            patcher(sys.modules[mod_name])
    if not any(isinstance(f, _After) for f in sys.meta_path):
        sys.meta_path.insert(0, _After())


# -- client (engine.py) --------------------------------------------------------------------

def read_ready(root):
    try:
        with open(os.path.join(root, "ready.json")) as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return None


def request(root, op, timeout=15.0, **payload):
    """Send one command; return the agent's reply, or {"ok": False, "error"} on timeout."""
    reply_dir = os.path.join(root, "reply")
    seq = max([int(n.split(".")[0]) for n in os.listdir(reply_dir) if n.endswith(".json")]
              + [0]) + 1 if os.path.isdir(reply_dir) else 1
    seq = max(seq, int(time.time() * 1000))        # monotonic across clients
    t0 = time.perf_counter()
    _write_json(os.path.join(root, "ctl.json"), {"seq": seq, "op": op, **payload})
    path = os.path.join(reply_dir, f"{seq}.json")
    while time.perf_counter() - t0 < timeout:
        try:
            with open(path) as f:
                r = json.load(f)
            r["client_roundtrip_s"] = round(time.perf_counter() - t0, 6)
            return r
        except (OSError, ValueError):
            time.sleep(0.002)
    return {"ok": False, "error": f"no reply from the live agent in {timeout} s",
            "client_roundtrip_s": round(time.perf_counter() - t0, 6)}
