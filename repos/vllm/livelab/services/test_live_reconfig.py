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
