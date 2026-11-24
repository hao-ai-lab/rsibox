"""Golden cases for live_reconfig.py's import hook (CPU only, no vLLM: a minimal
vllm.v1.engine.core package is written to a temp dir and imported for real).

Run: python3 -m pytest repos/vllm/livelab/services/test_live_reconfig.py -v
"""
import subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# A second import hook placed before live_reconfig's on sys.meta_path, written the way the
# adapter's hook (adapter/host.py `_After`) was: it asks every other finder on sys.meta_path,
# including the ones before it. The child imports vllm.v1.engine.core through both hooks.
CHILD = r'''
import importlib.abc, sys
sys.path[:0] = [{services!r}, {pkgs!r}]
import live_reconfig
live_reconfig.install()

class OtherHook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "vllm.v1.engine.core":
            return None
        for finder in list(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                return spec
        return None

sys.meta_path.insert(0, OtherHook())
import vllm.v1.engine.core as core
print("patched", getattr(core.EngineCoreProc, "_rsibox_live_reconfig", False))
'''


def test_import_hook_asks_only_the_finders_after_it():
    """Input: live_reconfig's hook at sys.meta_path[1] and another hook at [0] that asks all
    other finders (live_reconfig's included), as when the adapter's sitecustomize runs after
    live_site's. Action: import vllm.v1.engine.core. Expect: the import succeeds and
    EngineCoreProc carries live_reconfig's patch. Before: live_reconfig's hook asked every
    finder on sys.meta_path, including the one before it, and the two hooks called each other
    until RecursionError. Does not prove: the adapter's own hook (adapter/host.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        engine_dir = Path(tmp, "vllm", "v1", "engine")
        engine_dir.mkdir(parents=True)
        for package in (Path(tmp, "vllm"), Path(tmp, "vllm", "v1"), engine_dir):
            (package / "__init__.py").write_text("")
        (engine_dir / "core.py").write_text(
            "class EngineCoreProc:\n"
            "    def _handle_client_request(self, *a): pass\n"
            "    def _process_engine_step(self): return False\n"
            "    def run_busy_loop(self): pass\n")
        child = subprocess.run(
            [sys.executable, "-c", CHILD.format(services=str(HERE), pkgs=tmp)],
            capture_output=True, text=True, timeout=60)
        assert child.returncode == 0, child.stderr[-2000:]
        assert child.stdout.strip() == "patched True", child.stdout


def test_policy_switch_preserves_both_queues_and_changes_dequeue_order():
    """Lower numeric priority wins, then FCFS restores arrival order in both queues."""
    import unittest
    from types import SimpleNamespace

    try:
        from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
        from vllm.v1.request import Request
    except ModuleNotFoundError as err:
        if err.name == "vllm":
            raise unittest.SkipTest("needs installed vLLM request queues")
        raise
    import live_reconfig

    def request(name, arrival, priority):
        req = Request.__new__(Request)
        req.request_id, req.arrival_time, req.priority = name, arrival, priority
        return req

    config = SimpleNamespace(policy="fcfs", enable_chunked_prefill=True)
    scheduler = SimpleNamespace(policy=SchedulingPolicy.FCFS, scheduler_config=config,
                                running=[], max_num_running_reqs=2, max_num_scheduled_tokens=16)
    queues = ("waiting", "kv_holding_waiting")
    requests = {}
    for name in queues:
        queue = create_request_queue(scheduler.policy)
        for args in [("early", 1, 5), ("urgent", 2, -1)]:
            req = request(name + args[0], *args[1:])
            requests[req.request_id] = req
            queue.add_request(req)
        setattr(scheduler, name, queue)
    agent = live_reconfig.Agent(SimpleNamespace(scheduler=scheduler, vllm_config=SimpleNamespace(
        scheduler_config=config, model_config=SimpleNamespace(max_model_len=16))), None)
    for policy, order in [("priority", ["urgent", "early"]), ("fcfs", ["early", "urgent"])]:
        ack = agent.apply({"scheduling_policy": policy})
        assert ack["ok"], ack
        assert ack["after"]["scheduling_policy"] == config.policy == policy
        for name in queues:
            actual = list(getattr(scheduler, name))
            assert actual == [requests[name + suffix] for suffix in order]
            assert all(req is requests[req.request_id] for req in actual)
    before = agent.scheduler_view()
    assert not agent.apply({"max_num_seqs": 1, "scheduling_policy": "fifo"})["ok"]
    assert agent.scheduler_view() == before


def test_policy_switch_handles_missing_kv_holding_waiting_and_syncs_api_server(tmp_path, monkeypatch):
    """Input: (a) a scheduler lacking `kv_holding_waiting` (like FakeCore / older vLLM builds);
    (b) an API server `InputProcessor` / `AsyncLLM` booted under `fcfs` that rejects `priority != 0`.
    Expected: `Agent.apply({"scheduling_policy": "priority"})` succeeds without `AttributeError`,
    writes `<root>/policy.json`, and the hooked API server accepts `priority=-1`; switching back
    to `fcfs` updates the API server again so `priority=-1` is rejected."""
    import enum
    import types
    import pytest
    sys.path.insert(0, str(HERE))
    import live_reconfig

    class FakePolicy(enum.Enum):
        FCFS = "fcfs"
        PRIORITY = "priority"

    class FakeQueue(list):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def add_request(self, req):
            self.append(req)
            if self.policy == FakePolicy.PRIORITY:
                self.sort(key=lambda r: (r.priority, r.arrival_time))
            else:
                self.sort(key=lambda r: r.arrival_time)

    rq_mod = types.ModuleType("vllm.v1.core.sched.request_queue")
    rq_mod.SchedulingPolicy = FakePolicy
    rq_mod.create_request_queue = lambda p: FakeQueue(p)
    for mod_name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.request_queue", rq_mod)

    # Child EngineCoreProc scheduler (no kv_holding_waiting attribute!)
    core_cfg = types.SimpleNamespace(policy="fcfs", enable_chunked_prefill=True)
    waiting = FakeQueue(FakePolicy.FCFS)
    waiting.add_request(types.SimpleNamespace(request_id="early", arrival_time=1, priority=5))
    waiting.add_request(types.SimpleNamespace(request_id="urgent", arrival_time=2, priority=-1))
    sched = types.SimpleNamespace(
        policy=FakePolicy.FCFS, scheduler_config=core_cfg, running=[], waiting=waiting,
        max_num_running_reqs=4, max_num_scheduled_tokens=64,
    )
    core = types.SimpleNamespace(
        scheduler=sched,
        vllm_config=types.SimpleNamespace(
            scheduler_config=core_cfg, model_config=types.SimpleNamespace(max_model_len=64)
        ),
    )

    # Parent API server process with its own separate VllmConfig (booted under fcfs)
    api_sc = types.SimpleNamespace(policy="fcfs")
    api_vllm_cfg = types.SimpleNamespace(scheduler_config=api_sc)
    ip_mod = types.ModuleType("vllm.v1.engine.input_processor")

    class FakeInputProcessor:
        def __init__(self):
            self.vllm_config = api_vllm_cfg

        def process_inputs(self, request_id, priority=0):
            if priority != 0 and self.vllm_config.scheduler_config.policy != "priority":
                raise ValueError("Priority scheduling is not enabled.")
            return (request_id, priority)

    ip_mod.InputProcessor = FakeInputProcessor
    live_reconfig._patch_input_processor(ip_mod)
    ip = FakeInputProcessor()

    monkeypatch.setenv(live_reconfig.ENV, str(tmp_path))
    live_reconfig._POLICY_CACHE.update(path=None, mtime_ns=None, policy=None)
    agent = live_reconfig.Agent(core, str(tmp_path))

    with pytest.raises(ValueError, match="Priority scheduling is not enabled"):
        ip.process_inputs("r0", priority=-1)

    ack = agent.apply({"scheduling_policy": "priority"})
    assert ack["ok"] and ack["after"]["scheduling_policy"] == "priority"
    assert [r.request_id for r in sched.waiting] == ["urgent", "early"]
    assert ip.process_inputs("r1", priority=-1) == ("r1", -1)
    assert api_sc.policy == "priority"

    ack_back = agent.apply({"scheduling_policy": "fcfs"})
    assert ack_back["ok"] and ack_back["after"]["scheduling_policy"] == "fcfs"
    assert [r.request_id for r in sched.waiting] == ["early", "urgent"]
    with pytest.raises(ValueError, match="Priority scheduling is not enabled"):
        ip.process_inputs("r2", priority=-1)
    assert api_sc.policy == "fcfs"


def test_serving_priority_uses_effective_header_and_live_policy(tmp_path, monkeypatch):
    """Exercise vLLM's HTTP priority resolver before a stream can send its headers."""
    import pytest
    from types import SimpleNamespace
    import live_reconfig

    serving = pytest.importorskip("vllm.entrypoints.generate.base.serving")
    from starlette.requests import Request
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.exceptions import VLLMValidationError

    # Restore the module's real class after exercising the production import hooks.
    class Handler(serving.GenerateBaseServing):
        pass

    monkeypatch.setattr(serving, "GenerateBaseServing", Handler)
    monkeypatch.setenv(live_reconfig.ENV, str(tmp_path))
    for module, patch in live_reconfig._IMPORT_HOOKS:
        if module == serving.__name__:
            patch(serving)
    handler = object.__new__(Handler)
    config = SimpleNamespace(policy="fcfs")
    handler.engine_client = SimpleNamespace(vllm_config=SimpleNamespace(scheduler_config=config))
    request = CompletionRequest(prompt="Hello", model="m", priority=-1)
    override = Request({"type": "http", "headers": [(b"x-vllm-priority", b"0")]})
    with pytest.raises(VLLMValidationError, match="priority"):
        handler._get_priority(request, None)
    assert handler._get_priority(request, override) == 0
    for policy in ("priority", "fcfs"):
        live_reconfig._write_json(str(tmp_path / "policy.json"), {"scheduling_policy": policy})
        if policy == "priority":
            assert handler._get_priority(request, None) == -1
        else:
            with pytest.raises(VLLMValidationError, match="priority"):
                handler._get_priority(request, None)


_COMPOSED_CHILD = r'''
import json, os, queue, sys, threading, types

sys.path[:0] = [{livelab!r}, {services!r}, {pkgs!r}]
vllm_stub = types.ModuleType("vllm")
vllm_stub.__version__ = "0.0-test"
vllm_stub.__path__ = [os.path.join({pkgs!r}, "vllm")]
sys.modules["vllm"] = vllm_stub

mode = {mode!r}
order = {order!r}
if mode == "kernel_only":
    import adapter.host as host_mod
    lr_mod = None
elif mode == "reconfig_only":
    import live_reconfig as lr_mod
    lr_mod.install()
    host_mod = None
elif order == "reconfig_first":
    import live_reconfig as lr_mod
    lr_mod.install()
    import adapter.host as host_mod
else:
    import adapter.host as host_mod
    import live_reconfig as lr_mod
    lr_mod.install()

from rsibox import hooks
core_hooks = [f for f in sys.meta_path
              if isinstance(f, hooks.AfterImportHook) and f.module_name == "vllm.v1.engine.core"]
assert len(core_hooks) == 1, f"expected 1 core import hook, got {{len(core_hooks)}}"

import vllm.v1.engine.core as core_mod
P = core_mod.EngineCoreProc
before_ids = (id(P._handle_client_request), id(P._process_engine_step), id(P.run_busy_loop))
if lr_mod is not None:
    lr_mod._patch_core(core_mod)
if host_mod is not None:
    host_mod._patch_core(core_mod)
after_ids = (id(P._handle_client_request), id(P._process_engine_step), id(P.run_busy_loop))
assert before_ids == after_ids, f"double patched: {{before_ids}} != {{after_ids}}"

stop = threading.Event()
core = P(stop)
loop_thread = threading.Thread(target=core.run_busy_loop, daemon=True)
loop_thread.start()
assert core.started.wait(5.0), "busy loop did not start"

ctl_threads = [t.name for t in threading.enumerate()
               if t.name in ("rsibox-vllm-core-ctl", "rsibox-kernel-host", "rsibox-live-reconfig")]
assert ctl_threads == ["rsibox-vllm-core-ctl"], f"unexpected ctl threads: {{ctl_threads}}"

out = {{"channels": len(core._rsibox_channel_mux._channels),
       "patched_core": getattr(P, "_rsibox_core_hook", False),
       "patched_lr": getattr(P, "_rsibox_live_reconfig", False)}}

if os.environ.get("RSIBOX_KERNEL_DIR"):
    from adapter import client
    ping = client.send(os.environ["RSIBOX_KERNEL_DIR"], {{"op": "ping"}}, timeout=5.0)
    assert ping.get("ok"), ping
    out["ping_ok"] = True
if os.environ.get("RSIBOX_LIVE_RECONFIG_DIR"):
    import live_reconfig
    r_set = live_reconfig.request(os.environ["RSIBOX_LIVE_RECONFIG_DIR"], "set",
                                  timeout=5.0, values={{"max_num_seqs": 4}})
    assert r_set.get("ok") and r_set["after"]["target_max_num_seqs"] == 4, r_set
    r_st = live_reconfig.request(os.environ["RSIBOX_LIVE_RECONFIG_DIR"], "status", timeout=5.0)
    assert r_st.get("ok") and r_st["target_max_num_seqs"] == 4, r_st
    assert live_reconfig.read_ready(os.environ["RSIBOX_LIVE_RECONFIG_DIR"]) is not None
    out["reconfig_ok"] = True

stop.set()
core.input_queue.put_nowait(("STOP", None))
loop_thread.join(timeout=5.0)
assert not loop_thread.is_alive()
print(json.dumps(out))
'''


def test_core_hook_all_env_modes_and_import_orders(tmp_path):
    """Input: real `vllm.v1.engine.core` import + `run_busy_loop` across kernel-only,
    reconfig-only, and BOTH-active (in both import orders). Expected: strictly 1 import hook,
    strictly 1 patch of `EngineCoreProc` (method identity invariant), strictly 1 control thread
    (`rsibox-vllm-core-ctl`), and every active channel answers."""
    import json
    import os

    pkgs = tmp_path / "pkgs"
    eng = pkgs / "vllm" / "v1" / "engine"
    eng.mkdir(parents=True)
    for pkg in (pkgs / "vllm", pkgs / "vllm" / "v1", eng):
        (pkg / "__init__.py").write_text("")
    (eng / "core.py").write_text(
        "import queue, threading, types\n"
        "class EngineCoreProc:\n"
        "    def __init__(self, stop):\n"
        "        self._stop = stop\n"
        "        self.started = threading.Event()\n"
        "        self.input_queue = queue.Queue()\n"
        "        self.batch_queue = []\n"
        "        self.model_executor = types.SimpleNamespace(collective_rpc=lambda *a, **k: [])\n"
        "        self.scheduler = types.SimpleNamespace(\n"
        "            max_num_running_reqs=8, max_num_scheduled_tokens=64,\n"
        "            running=[], waiting=[], policy=types.SimpleNamespace(value='fcfs'),\n"
        "            has_requests=lambda: True)\n"
        "        self.vllm_config = types.SimpleNamespace(\n"
        "            scheduler_config=types.SimpleNamespace(enable_chunked_prefill=True),\n"
        "            model_config=types.SimpleNamespace(max_model_len=64),\n"
        "            parallel_config=types.SimpleNamespace(\n"
        "                world_size=1, data_parallel_size=1, data_parallel_rank=0))\n"
        "    def _handle_client_request(self, t, r): pass\n"
        "    def _process_engine_step(self): return False\n"
        "    def run_busy_loop(self):\n"
        "        self.started.set()\n"
        "        while not self._stop.is_set():\n"
        "            t, r = self.input_queue.get()\n"
        "            if t == 'STOP':\n"
        "                break\n"
        "            self._handle_client_request(t, r)\n"
        "            self._process_engine_step()\n"
    )
    livelab = str(HERE.parent)
    cases = [
        ("kernel_only", None, 1, False),
        ("reconfig_only", None, 1, True),
        ("both", "reconfig_first", 2, True),
        ("both", "host_first", 2, True),
    ]
    for idx, (mode, order, expected_channels, expect_lr) in enumerate(cases):
        env = {k: v for k, v in os.environ.items()
               if k not in ("RSIBOX_KERNEL_DIR", "RSIBOX_LIVE_RECONFIG_DIR")}
        if mode in ("kernel_only", "both"):
            kdir = tmp_path / f"k-{idx}"
            kdir.mkdir()
            env["RSIBOX_KERNEL_DIR"] = str(kdir)
        if mode in ("reconfig_only", "both"):
            ldir = tmp_path / f"l-{idx}"
            ldir.mkdir()
            env["RSIBOX_LIVE_RECONFIG_DIR"] = str(ldir)
        code = _COMPOSED_CHILD.format(
            livelab=livelab, services=str(HERE), pkgs=str(pkgs), mode=mode, order=order
        )
        res = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", "-c", code],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert res.returncode == 0, f"{mode}/{order}: {res.stderr[-2000:]}"
        data = json.loads(res.stdout.strip().splitlines()[-1])
        assert data["channels"] == expected_channels, (mode, order, data)
        assert data["patched_core"] is True
        assert data["patched_lr"] is expect_lr
        if mode in ("kernel_only", "both"):
            assert data.get("ping_ok") is True
        if mode in ("reconfig_only", "both"):
            assert data.get("reconfig_ok") is True


def test_core_hook_pause_keep_and_rollback_hold_isolate_live_reconfig(monkeypatch):
    """When host holds a step (pause-keep or pending rollback), live_reconfig's tick is skipped
    so scheduler limits and step counters are not mutated mid-gap; normal steps run both."""
    import types
    livelab = str(HERE.parent)
    if livelab not in sys.path:
        sys.path.insert(0, livelab)
    import adapter.host as host_mod
    import live_reconfig

    events = []

    class Scheduler:
        def __init__(self):
            self.max_num_running_reqs = 8
            self.max_num_scheduled_tokens = 64
            self.running = []
            self.waiting = []
            self.policy = types.SimpleNamespace(value="fcfs")

        def has_requests(self):
            return True

    class EngineCoreProc:
        def __init__(self):
            self.batch_queue = ["b0"]
            self.scheduler = Scheduler()
            self.vllm_config = types.SimpleNamespace(
                scheduler_config=types.SimpleNamespace(enable_chunked_prefill=True),
                model_config=types.SimpleNamespace(max_model_len=64),
            )

        def _handle_client_request(self, t, r):
            pass

        def _process_engine_step(self):
            events.append(("step", self.scheduler.has_requests(),
                           self.scheduler.max_num_running_reqs))
            self.batch_queue.clear()
            return True

        def run_busy_loop(self):
            pass

    mod = types.SimpleNamespace(EngineCoreProc=EngineCoreProc)
    live_reconfig._patch_core(mod)
    host_mod._patch_core(mod)

    core = EngineCoreProc()
    agent = live_reconfig.Agent(core, None)
    agent.target_seqs = 4
    rollback_flag = [False]
    fake_co = types.SimpleNamespace(
        poll_rollback=lambda: rollback_flag[0],
        has_pending_rollback=lambda: rollback_flag[0],
    )
    ticks = []
    fake_host = types.SimpleNamespace(
        pending=[host_mod._Job(lambda: events.append("gap_job"), gap=True)],
        held_steps=0, steps=[], co=fake_co, loop_thread=None,
        core_side=types.SimpleNamespace(tick=lambda entering: ticks.append(entering)),
    )
    monkeypatch.setattr(host_mod, "_HOST", [fake_host])
    monkeypatch.setattr(live_reconfig, "_AGENT", [agent])

    # 1. Pause-keep while batch_queue drains: agent.tick() is skipped, then gap_job runs.
    assert core._process_engine_step() is True
    assert events == [("step", False, 8), "gap_job"]
    assert agent.steps == 0 and core.scheduler.max_num_running_reqs == 8
    assert fake_host.held_steps == 1

    # 2. Pending rollback with empty batch_queue: step short-circuits, agent.tick() skipped.
    rollback_flag[0] = True
    assert core._process_engine_step() is False
    assert agent.steps == 0 and core.scheduler.max_num_running_reqs == 8
    assert fake_host.held_steps == 2

    # 3. Normal step: both live_reconfig.tick() and host hooks run.
    rollback_flag[0] = False
    assert core._process_engine_step() is True
    assert agent.steps == 1 and core.scheduler.max_num_running_reqs == 4
    assert len(fake_host.steps) == 2 and ticks == [True, False, True, False]


def test_core_hook_refuses_duplicate_control_dir(tmp_path, capsys):
    """Two plugins pointing at the same control dir: second is refused with a stderr diagnostic."""
    livelab = str(HERE.parent)
    if livelab not in sys.path:
        sys.path.insert(0, livelab)
    from rsibox import ipc
    from services import core_hook

    seen = []
    same = str(tmp_path / "shared")
    mux = core_hook._ChannelMux([
        ("live_reconfig", same, lambda cmd: seen.append(("lr", cmd)) or {"ok": True, "who": "lr"}, 0.005),
        ("kernel_host", same, lambda cmd: seen.append(("kh", cmd)) or {"ok": True, "who": "kh"}, 0.01),
    ])
    err = capsys.readouterr().err
    assert "refusing duplicate control dir" in err
    assert len(mux._channels) == 1
    assert mux._channels[0]["name"] == "live_reconfig"
    ipc.atomic_write_json(tmp_path / "shared" / "ctl.json", {"seq": 1, "op": "status"})
    mux.poll_once()
    reply = ipc.read_json(tmp_path / "shared" / "reply" / "1.json")
    assert reply == {"ok": True, "who": "lr"}
    assert len(seen) == 1


def _start_local_live_agent(tmp_path, monkeypatch, *, api_pid, boot_seqs=128,
                            boot_tokens=4096, max_model_len=4096, chunked_prefill=True):
    """Run a real live_reconfig.Agent + core_hook._ChannelMux on local background threads."""
    import enum
    import os
    import queue
    import threading
    import time
    import types

    livelab = str(HERE.parent)
    if livelab not in sys.path:
        sys.path.insert(0, livelab)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import live_reconfig
    from rsibox import ipc
    from services import core_hook

    class LocalPolicy(enum.Enum):
        FCFS = "fcfs"
        PRIORITY = "priority"

    class LocalQueue(list):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def add_request(self, item):
            self.append(item)

    request_queue_module = types.ModuleType("vllm.v1.core.sched.request_queue")
    request_queue_module.SchedulingPolicy = LocalPolicy
    request_queue_module.create_request_queue = lambda policy: LocalQueue(policy)
    for package_name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        monkeypatch.setitem(sys.modules, package_name, types.ModuleType(package_name))
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.request_queue", request_queue_module)

    live_dir = tmp_path / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    scheduler_config = types.SimpleNamespace(
        policy="fcfs", enable_chunked_prefill=chunked_prefill
    )
    scheduler = types.SimpleNamespace(
        policy=LocalPolicy.FCFS,
        scheduler_config=scheduler_config,
        running=[],
        waiting=LocalQueue(LocalPolicy.FCFS),
        max_num_running_reqs=boot_seqs,
        max_num_scheduled_tokens=boot_tokens,
    )
    stop_event = threading.Event()
    input_queue = queue.Queue()
    core = types.SimpleNamespace(
        scheduler=scheduler,
        input_queue=input_queue,
        vllm_config=types.SimpleNamespace(
            scheduler_config=scheduler_config,
            model_config=types.SimpleNamespace(max_model_len=max_model_len),
        ),
    )
    agent = live_reconfig.Agent(core, str(live_dir))
    ipc.atomic_write_json(
        live_dir / "ready.json",
        {
            "pid": os.getpid(),
            "api_pid": api_pid,
            "boot": agent.boot,
            "live_fields": list(live_reconfig.LIVE_FIELDS),
            "dp_size": 1,
            "t": time.time(),
        },
    )
    channel_mux = core_hook._ChannelMux(
        [("live_reconfig", str(live_dir), agent.handle, 0.002)]
    )

    def busy_loop():
        while not stop_event.is_set():
            tag, payload = input_queue.get()
            if tag == "STOP":
                break
            if tag == live_reconfig.TAG:
                payload.run()

    loop_thread = threading.Thread(target=busy_loop, daemon=True)
    loop_thread.start()
    ctl_thread = channel_mux.start()

    def shutdown():
        stop_event.set()
        channel_mux.stop()
        input_queue.put_nowait(("STOP", None))
        loop_thread.join(timeout=5.0)
        ctl_thread.join(timeout=5.0)

    return str(live_dir), agent, scheduler, shutdown


def _seed_engine_state(cache_root, log_path, api_pid, live_dir, *,
                       boot_seqs=128, boot_tokens=4096, port=8123):
    import engine
    import test_engine

    base_vargv = [
        "Qwen/Qwen3-8B", "--port", str(port),
        "--max-num-seqs", str(boot_seqs),
        "--max-num-batched-tokens", str(boot_tokens),
    ]
    old_parsed = {**test_engine._stub_parser()(base_vargv), "via": "direct"}
    start_args, _ = engine.parse([
        "start", "--cache-root", str(cache_root), "--log", str(log_path), *base_vargv
    ])
    start_args.live_dir = live_dir
    command = engine.serve_cmd(start_args, base_vargv, old_parsed)
    engine.save_engine_state(start_args, api_pid, port, base_vargv, command, old_parsed)
    return base_vargv, old_parsed


def test_reconfigure_live_fast_path_bypasses_vllm_parser_and_updates_state(tmp_path, monkeypatch):
    """Input: a running local live_reconfig file-channel agent + persisted engine state file,
    with `vllm_args.parse`, `engine.read_vllm_args`, and `engine.engine_env` wired to fail if
    called. Action: call `engine.reconfigure` with (1) `--max-num-seqs 64`,
    (2) `--max-num-seqs=32 --max_num_batched_tokens=2048 --scheduling-policy=priority`, and
    (3) repeated `--max-num-seqs 32` (noop). Expect: all succeed quickly without ever calling
    vLLM's parser, real parse seconds are recorded, the scheduler mutates in-place,
    `policy.json` and `engines/<pid>.json` are updated consistently with the slow-path contract,
    and a follow-up non-live `override_finding` accepts the fast-path persisted state cleanly."""
    import json
    import os
    import time
    import pytest

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import engine
    import vllm_args

    api_pid = os.getpid()
    cache_root = tmp_path / "cache"
    log_path = tmp_path / "serve.log"
    live_dir, agent, scheduler, shutdown = _start_local_live_agent(
        tmp_path, monkeypatch, api_pid=api_pid, boot_seqs=128, boot_tokens=4096
    )
    try:
        _seed_engine_state(cache_root, log_path, api_pid, live_dir)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("slow parser/env path must not run on live fast path")

        monkeypatch.setattr(vllm_args, "parse", fail_if_called)
        monkeypatch.setattr(engine, "read_vllm_args", fail_if_called)
        monkeypatch.setattr(engine, "engine_env", fail_if_called)

        raw_argv_first = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "64",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv_first)
        started_at = time.perf_counter()
        report_first = engine.reconfigure(launcher_args, patch_vargv, raw_argv_first)
        elapsed_first = time.perf_counter() - started_at

        assert elapsed_first <= 3.0, f"fast path took {elapsed_first:.3f}s"
        assert report_first["reconfig"]["path"] == "in_process_live"
        assert report_first["reconfig"]["restarted"] is False
        assert report_first["vllm_args"]["via"] == "live_fast_path"
        assert isinstance(report_first["vllm_args"]["seconds"]["parse"], float)
        assert report_first["vllm_args"]["seconds"]["parse"] >= 0.0
        assert scheduler.max_num_running_reqs == 64
        assert agent.target_seqs == 64

        raw_argv_second = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--",
            "--max-num-seqs=32", "--max_num_batched_tokens=2048", "--scheduling-policy=priority",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv_second)
        started_at = time.perf_counter()
        report_second = engine.reconfigure(launcher_args, patch_vargv, raw_argv_second)
        elapsed_second = time.perf_counter() - started_at

        assert elapsed_second <= 3.0, f"fast path took {elapsed_second:.3f}s"
        assert report_second["reconfig"]["path"] == "in_process_live"
        assert scheduler.max_num_running_reqs == 32
        assert scheduler.max_num_scheduled_tokens == 2048
        assert scheduler.policy.value == "priority"
        assert json.loads((Path(live_dir) / "policy.json").read_text()) == {
            "scheduling_policy": "priority"
        }

        persisted = engine.load_engine_state(cache_root, api_pid)
        assert persisted["vargv"] == [
            "Qwen/Qwen3-8B", "--port", "8123",
            "--max-num-seqs=32", "--max_num_batched_tokens=2048", "--scheduling-policy=priority",
        ]
        assert persisted["vllm_args"]["args"]["max_num_seqs"] == 32
        assert persisted["vllm_args"]["args"]["max_num_batched_tokens"] == 2048
        assert persisted["vllm_args"]["args"]["scheduling_policy"] == "priority"
        assert {"max_num_seqs", "max_num_batched_tokens", "scheduling_policy"} <= set(
            persisted["vllm_args"]["explicit"]
        )
        assert persisted["vllm_args"]["explicit_values"]["scheduling_policy"] == "priority"

        # A follow-up slow-path override against persisted state must not trigger override_finding.
        followup_new = {
            **persisted["vllm_args"],
            "args": {**persisted["vllm_args"]["args"], "max_model_len": 2048},
            "explicit": [*persisted["vllm_args"]["explicit"], "max_model_len"],
            "explicit_values": {**persisted["vllm_args"]["explicit_values"], "max_model_len": 2048},
        }
        assert vllm_args.override_finding(["--max-model-len", "2048"], persisted["vllm_args"], followup_new) is None

        # Setting an already-applied value is a fast-path noop (exit 0, parser still not called).
        raw_argv_noop = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "32",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv_noop)
        with pytest.raises(SystemExit) as exit_info:
            engine.reconfigure(launcher_args, patch_vargv, raw_argv_noop)
        assert exit_info.value.code == 0
    finally:
        shutdown()


def test_live_fast_path_parse_and_synthesize_match_full_parser_parity():
    """Golden parity test: `synthesize_live_parsed(old, *parse_live_patch(patch))` matches
    full parser output (`args`, `set(explicit)`, `explicit_values`, `arities`) across `=`
    syntax, underscore spellings, duplicate flags, and mixed base/patch forms, while rejecting
    overlong >4300-digit integers cleanly without raising `ValueError`."""
    import argparse

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import test_engine
    import vllm_args

    class _FlexibleStubParser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            normalized = []
            for token in list(args or ()):
                if token.startswith("--") and token != "--":
                    if "=" in token:
                        option, value = token.split("=", 1)
                        token = "--" + option[2:].replace("_", "-") + "=" + value
                    else:
                        token = "--" + token[2:].replace("_", "-")
                normalized.append(token)
            return super().parse_args(normalized, namespace=namespace)

    stub_namespace = {}
    exec(test_engine.STUB_PARSER["vllm/entrypoints/common/cli_args.py"].split("\n", 2)[2], stub_namespace)
    parser = stub_namespace["make_arg_parser"](_FlexibleStubParser(prog="vllm serve"))
    parser.add_argument("--scheduling-policy", default="fcfs", choices=["fcfs", "priority"])
    stub_parse = lambda argv: vllm_args.parse(argv, parser)

    cases = (
        (
            ["Qwen/Qwen3-8B", "--port", "8123", "--max-num-seqs", "128"],
            ["--max-num-seqs=32"],
        ),
        (
            ["Qwen/Qwen3-8B", "--port", "8123", "--max-num-seqs=128", "--max-num-batched-tokens", "4096"],
            ["--max_num_batched_tokens=2048", "--scheduling_policy", "priority"],
        ),
        (
            ["Qwen/Qwen3-8B", "--port", "8123", "--max-num-seqs", "128"],
            ["--max-num-seqs", "64", "--max-num-seqs=32"],
        ),
        (
            ["Qwen/Qwen3-8B", "--port=8123", "--max-num-seqs=128", "--max-num-batched-tokens=4096"],
            ["--max-num-seqs", "32", "--max-num-batched-tokens=2048", "--scheduling-policy=priority"],
        ),
    )
    for base_argv, patch_argv in cases:
        old_parsed = stub_parse(base_argv)
        live_values, live_arities = vllm_args.parse_live_patch(patch_argv)
        assert live_values is not None, patch_argv
        synthesized = vllm_args.synthesize_live_parsed(old_parsed, live_values, live_arities)
        merged_argv = vllm_args.override(base_argv, patch_argv)
        full_parsed = stub_parse(merged_argv)
        assert full_parsed["ok"], (merged_argv, full_parsed)
        assert synthesized["args"] == full_parsed["args"], (base_argv, patch_argv)
        assert set(synthesized["explicit"]) == set(full_parsed["explicit"]), (base_argv, patch_argv)
        assert synthesized["explicit_values"] == full_parsed["explicit_values"], (base_argv, patch_argv)
        assert synthesized["arities"] == full_parsed["arities"], (base_argv, patch_argv)
        assert synthesized["seconds"]["parse"] >= 0.0
        assert vllm_args.override_finding(patch_argv, old_parsed, synthesized) is None

    # >4300-digit integer must fail closed without raising ValueError.
    overlong_values, overlong_arities = vllm_args.parse_live_patch(["--max-num-seqs", "9" * 5000])
    assert overlong_values is None and overlong_arities is None


def test_reconfigure_live_fast_path_falls_back_on_non_live_invalid_or_missing_ready(
    tmp_path, monkeypatch, capsys
):
    """Input: a live agent with boot max_num_seqs=128 and stubbed slow parser. Action: invoke
    `engine.reconfigure` for (1) non-live `--max-model-len 2048`, (2) mixed live + non-live,
    (3) `--max-num-seqs 99999` above boot limit, (4) invalid `--max-num-seqs 0`, (5) live agent
    returning `ok=False`, (6) missing `ready.json`, and (7) launcher-backend change (`--kvplan`),
    `--reconfigure-restart`, `--zygote-check`, mismatched `api_pid`, dead `core_pid`, corrupt
    non-int `boot`, and legacy state missing `arities`/`explicit_values`. Expect: every case
    falls back cleanly to `read_vllm_args` without crashing, reports exact findings or restart
    reasons, leaves the live scheduler untouched on invalid/non-live inputs, and never retries
    the live request on failure."""
    import json
    import os
    import pytest

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import engine
    import live_reconfig
    import test_engine
    from rsibox import ipc

    api_pid = os.getpid()
    cache_root = tmp_path / "cache"
    log_path = tmp_path / "serve.log"
    live_dir, _agent, scheduler, shutdown = _start_local_live_agent(
        tmp_path, monkeypatch, api_pid=api_pid, boot_seqs=128, boot_tokens=4096
    )
    stub_parse = test_engine._stub_parser()
    parser_calls = []

    def tracking_read_vllm_args(_launcher_args, target_vargv, _env):
        parser_calls.append(list(target_vargv))
        return {**stub_parse(target_vargv), "via": "direct"}

    try:
        _seed_engine_state(cache_root, log_path, api_pid, live_dir)
        monkeypatch.setattr(engine, "read_vllm_args", tracking_read_vllm_args)
        monkeypatch.setattr(engine, "stop_engine", lambda *_args, **_kwargs: {"stopped": True})
        monkeypatch.setattr(
            engine,
            "_launch_process",
            lambda _launcher_args, new_vargv, new_parsed, _process_env, _preflight, extra_report=None: {
                "pid": api_pid + 1, "port": new_parsed["args"]["port"], "vargv": new_vargv,
                "vllm_args": new_parsed, **(extra_report or {}),
            },
        )

        # (1) Non-live flag and (2) mixed live + non-live flag both fall back to full path.
        for patch_flags in (["--max-model-len", "2048"], ["--max-num-seqs", "64", "--max-model-len", "2048"]):
            parser_calls.clear()
            raw_argv = [
                "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
                "--log", str(log_path), "--", *patch_flags,
            ]
            launcher_args, patch_vargv = engine.parse(raw_argv)
            report = engine.reconfigure(launcher_args, patch_vargv, raw_argv)
            assert len(parser_calls) == 1, patch_flags
            assert report["reconfig"]["path"] == "restart"
            assert scheduler.max_num_running_reqs == 128

        # (3) Live field exceeding boot limit (99999 > 128) falls back and restarts with boot reason.
        parser_calls.clear()
        raw_argv = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "99999",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv)
        report = engine.reconfigure(launcher_args, patch_vargv, raw_argv)
        assert len(parser_calls) == 1
        assert report["reconfig"]["path"] == "restart"
        assert "boot value 128" in report["reconfig"]["restart_reason"]
        assert scheduler.max_num_running_reqs == 128

        # (4) Invalid value (--max-num-seqs 0) falls back and exits 2 with exact vllm_value finding.
        parser_calls.clear()
        capsys.readouterr()
        raw_argv = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "0",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv)
        with pytest.raises(SystemExit) as exit_info:
            engine.reconfigure(launcher_args, patch_vargv, raw_argv)
        assert exit_info.value.code == 2
        assert len(parser_calls) == 1
        finding_payload = json.loads(capsys.readouterr().out.strip())
        assert finding_payload["findings"][0]["check"] == "vllm_value"
        assert scheduler.max_num_running_reqs == 128

        # (5) Live agent refusal during fast-path apply falls back to restart without retrying request.
        live_request_calls = []
        original_request = live_reconfig.request

        def refusing_request(*args, **kwargs):
            live_request_calls.append((args, kwargs))
            return {"ok": False, "error": "simulated refusal"}

        monkeypatch.setattr(live_reconfig, "request", refusing_request)
        parser_calls.clear()
        raw_argv = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "64",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv)
        report = engine.reconfigure(launcher_args, patch_vargv, raw_argv)
        assert len(live_request_calls) == 1, "must not retry live agent on fallback"
        assert len(parser_calls) == 1
        assert report["reconfig"]["path"] == "restart"
        assert "simulated refusal" in report["reconfig"]["restart_reason"]
        monkeypatch.setattr(live_reconfig, "request", original_request)

        # (6) Launcher-backend change (--kvplan), --reconfigure-restart, and --zygote-check bypass fast path.
        for extra_launcher_flags in (
            ["--kvplan"],
            ["--reconfigure-restart"],
        ):
            parser_calls.clear()
            raw_argv = [
                "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
                "--log", str(log_path), *extra_launcher_flags, "--", "--max-num-seqs", "64",
            ]
            launcher_args, patch_vargv = engine.parse(raw_argv)
            report = engine.reconfigure(launcher_args, patch_vargv, raw_argv)
            assert len(parser_calls) == 1, extra_launcher_flags
            assert report["reconfig"]["path"] == "restart", extra_launcher_flags
            assert scheduler.max_num_running_reqs == 128

        parser_calls.clear()
        capsys.readouterr()
        raw_argv = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--zygote-check", "--", "--max-num-seqs", "64",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv)
        with pytest.raises(SystemExit) as exit_info:
            engine.reconfigure(launcher_args, patch_vargv, raw_argv)
        assert exit_info.value.code == 0
        assert len(parser_calls) == 1
        assert scheduler.max_num_running_reqs == 128

        # (7) Mismatched api_pid, dead core_pid, corrupt non-int boot, and missing ready.json fall back cleanly.
        ready_path = Path(live_dir) / "ready.json"
        original_ready = json.loads(ready_path.read_text())
        for mutated_ready, expect_reason_substr in (
            ({**original_ready, "api_pid": api_pid + 999}, "belongs to api pid"),
            ({**original_ready, "pid": 99999999}, "is gone"),
            ({**original_ready, "boot": {**original_ready["boot"], "max_num_seqs": "128"}}, None),
        ):
            ipc.atomic_write_json(ready_path, mutated_ready)
            parser_calls.clear()
            raw_argv = [
                "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
                "--log", str(log_path), "--", "--max-num-seqs", "64",
            ]
            launcher_args, patch_vargv = engine.parse(raw_argv)
            if expect_reason_substr is None:
                # Corrupt non-int boot fails closed in fast path without crashing; restore valid boot for slow path check.
                report_fast, failure_fast, reconciled_fast = engine._try_fast_live_reconfigure(
                    launcher_args, engine.load_engine_state(cache_root, api_pid), patch_vargv, raw_argv
                )
                assert (report_fast, failure_fast, reconciled_fast) == (None, None, None)
            else:
                report = engine.reconfigure(launcher_args, patch_vargv, raw_argv)
                assert len(parser_calls) == 1, mutated_ready
                assert report["reconfig"]["path"] == "restart"
                assert expect_reason_substr in report["reconfig"]["restart_reason"]

        # Legacy state missing arities / explicit_values fails closed in fast path.
        ipc.atomic_write_json(ready_path, original_ready)
        legacy_state = engine.load_engine_state(cache_root, api_pid)
        legacy_state["vllm_args"].pop("arities", None)
        legacy_state["vllm_args"].pop("explicit_values", None)
        launcher_args, patch_vargv = engine.parse([
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "64",
        ])
        assert engine._try_fast_live_reconfigure(
            launcher_args, legacy_state, patch_vargv, []
        ) == (None, None, None)

        ready_path.unlink()
        parser_calls.clear()
        raw_argv = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "64",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv)
        report = engine.reconfigure(launcher_args, patch_vargv, raw_argv)
        assert len(parser_calls) == 1
        assert report["reconfig"]["path"] == "restart"
        assert "no live agent" in report["reconfig"]["restart_reason"]
    finally:
        shutdown()


def test_reconfigure_fast_live_failure_reconciles_drift_and_preserves_fallback_reason(
    tmp_path, monkeypatch, capsys
):
    """P1 regression: (a) fast-path preflight finding aborts cleanly before sending any live
    command; (b) `lost`/timeout with late pickup reconciles actual scheduler state into
    `engines/<pid>.json` and preserves `live_fallback_reason` + `live_reconciled_fields` on a
    subsequent exit-2 `read_vllm_args` refusal; (c) read-back mismatch reconciles state and
    preserves `live_fallback_reason` + `live_reconciled_fields` on a subsequent `zygote.check`
    refusal."""
    import json
    import os
    import pytest

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import engine
    import live_reconfig
    import test_engine

    api_pid = os.getpid()
    cache_root = tmp_path / "cache"
    log_path = tmp_path / "serve.log"
    live_dir, agent, scheduler, shutdown = _start_local_live_agent(
        tmp_path, monkeypatch, api_pid=api_pid, boot_seqs=128, boot_tokens=4096
    )
    stub_parse = test_engine._stub_parser()
    try:
        # (a) Preflight failure prevents fast path from touching the live agent at all.
        _seed_engine_state(cache_root, log_path, api_pid, live_dir)
        original_request = live_reconfig.request
        live_calls = []

        def spy_request(*args, **kwargs):
            live_calls.append((args, kwargs))
            return original_request(*args, **kwargs)

        monkeypatch.setattr(live_reconfig, "request", spy_request)
        monkeypatch.setattr(
            engine,
            "preflight",
            lambda *_args, **_kwargs: (
                [{"check": "weights_daemon", "observed": "dead", "expected": "up"}],
                {"weights_probe": {"ok": False}},
            ),
        )
        monkeypatch.setattr(
            engine,
            "read_vllm_args",
            lambda _launcher_args, argv, _process_env: {**stub_parse(argv), "via": "direct"},
        )
        capsys.readouterr()
        raw_argv = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "64",
        ]
        launcher_args, patch_vargv = engine.parse(raw_argv)
        with pytest.raises(SystemExit) as exit_info:
            engine.reconfigure(launcher_args, patch_vargv, raw_argv)
        assert exit_info.value.code == 2
        assert live_calls == [], "preflight failure must abort before sending any live command"
        preflight_payload = json.loads(capsys.readouterr().out.strip())
        assert preflight_payload["findings"][0]["check"] == "weights_daemon"
        assert "live_fallback_reason" not in preflight_payload

        # Restore clean preflight for (b) and (c).
        monkeypatch.setattr(engine, "preflight", lambda *_args, **_kwargs: ([], {}))

        # (b) Timeout/lost with late pickup + fallback read_vllm_args refusal (vllm not importable).
        _seed_engine_state(cache_root, log_path, api_pid, live_dir)

        def lost_with_late_pickup(root, op, timeout=15.0, **payload):
            if op == "set":
                # Simulate the engine core applying the command after the client timed out.
                agent.apply(payload["values"])
                return {"ok": False, "error": f"no reply from the live agent in {timeout} s"}
            return original_request(root, op, timeout=timeout, **payload)

        monkeypatch.setattr(live_reconfig, "request", lost_with_late_pickup)
        monkeypatch.setattr(
            engine,
            "read_vllm_args",
            lambda *_args, **_kwargs: {
                "ok": False,
                "error": "vLLM's parser is not importable by python",
                "via": "direct",
            },
        )
        capsys.readouterr()
        launcher_args, patch_vargv = engine.parse(raw_argv)
        with pytest.raises(SystemExit) as exit_info:
            engine.reconfigure(launcher_args, patch_vargv, raw_argv)
        assert exit_info.value.code == 2
        lost_payload = json.loads(capsys.readouterr().out.strip())
        assert "no reply" in lost_payload.get("live_fallback_reason", ""), lost_payload
        assert lost_payload.get("live_reconciled_fields") == {"max_num_seqs": 64}, lost_payload
        reconciled_state = engine.load_engine_state(cache_root, api_pid)
        assert reconciled_state["vllm_args"]["args"]["max_num_seqs"] == 64

        # Subsequent reconfigure back to 128 must NOT treat 128 as a noop (state was reconciled to 64).
        monkeypatch.setattr(live_reconfig, "request", original_request)
        restore_argv = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--", "--max-num-seqs", "128",
        ]
        launcher_args, patch_vargv = engine.parse(restore_argv)
        restore_report = engine.reconfigure(launcher_args, patch_vargv, restore_argv)
        assert restore_report["reconfig"]["path"] == "in_process_live"
        assert scheduler.max_num_running_reqs == 128

        # (c) Read-back mismatch + fallback zygote.check refusal preserves live_fallback_reason.
        _seed_engine_state(cache_root, log_path, api_pid, live_dir)

        def mismatch_request(root, op, timeout=15.0, **payload):
            if op == "set":
                agent.apply({"max_num_seqs": 48})
                return {"ok": True, "after": agent.scheduler_view()}
            return original_request(root, op, timeout=timeout, **payload)

        monkeypatch.setattr(live_reconfig, "request", mismatch_request)
        monkeypatch.setattr(
            engine,
            "read_vllm_args",
            lambda _launcher_args, argv, _process_env: {**stub_parse(argv), "via": "direct"},
        )
        monkeypatch.setattr(
            engine.zygote,
            "check",
            lambda *_args, **_kwargs: {
                "ok": False, "findings": [{"check": "zygote_stale"}], "error": "stale",
            },
        )
        monkeypatch.setattr(engine, "engine_env", lambda *_args, **_kwargs: dict(os.environ))
        capsys.readouterr()
        zygote_raw_argv = [
            "reconfigure", "--pid", str(api_pid), "--cache-root", str(cache_root),
            "--log", str(log_path), "--zygote", str(tmp_path / "z.sock"),
            "--", "--max-num-seqs", "64",
        ]
        launcher_args, patch_vargv = engine.parse(zygote_raw_argv)
        with pytest.raises(SystemExit) as exit_info:
            engine.reconfigure(launcher_args, patch_vargv, zygote_raw_argv)
        assert exit_info.value.code == 2
        zygote_payload = json.loads(capsys.readouterr().out.strip())
        assert "scheduler reads back" in zygote_payload.get("live_fallback_reason", ""), zygote_payload
        assert zygote_payload.get("live_reconciled_fields") == {"max_num_seqs": 48}, zygote_payload
        assert engine.load_engine_state(cache_root, api_pid)["vllm_args"]["args"]["max_num_seqs"] == 48
    finally:
        shutdown()
