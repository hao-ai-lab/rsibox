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
print("patched", getattr(core.EngineCoreProc, "_fastrsi_live_reconfig", False))
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
