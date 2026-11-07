"""Golden cases for the KV plan holder (kvplan.py). CPU only: kvplan.wrap() runs for real around
a fake worker whose determine_available_memory stands in for the profile run (it counts calls
and sets the attributes vLLM's sets); vLLM's version and the GPU's name come from stubs.

Run: python3 -m pytest repos/vllm/livelab/services/test_kvplan.py -v
"""
import json, logging, os, subprocess, sys, tempfile, types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kvplan  # noqa: E402
from test_engine import _free_port, _script, _stop, stub_env  # noqa: E402

GiB = 1 << 30


class FakeWorker:
    """The attributes kvplan reads from vLLM's Worker, and a profile run that counts itself."""
    profiles = 0

    def __init__(self, config="c1", free=150 * GiB, rank=0):
        self.vllm_config = types.SimpleNamespace(
            compute_hash=lambda: config,
            load_config=types.SimpleNamespace(load_format="auto"))
        self.cache_config = types.SimpleNamespace(kv_cache_memory_bytes=None)
        self.parallel_config = types.SimpleNamespace(world_size=1)
        self.init_snapshot = types.SimpleNamespace(free_memory=free)
        self.rank, self.device, self.requested_memory = rank, "cuda:0", 110 * GiB

    def determine_available_memory(self):
        FakeWorker.profiles += 1
        self.peak_activation_memory = 3 * GiB
        self.available_kv_cache_memory_bytes = 90 * GiB
        return 90 * GiB


Planned = type("Planned", (FakeWorker,), {
    "determine_available_memory": kvplan.wrap(FakeWorker.determine_available_memory)})


@pytest.fixture
def plan_dir(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.0.test"))
    monkeypatch.setattr(kvplan, "gpu", lambda w: ("Fake GB200", 189 * GiB))
    monkeypatch.setenv("FASTRSI_KVPLAN_DIR", str(tmp_path))
    FakeWorker.profiles = 0
    return tmp_path


def _events(caplog, finding):
    tag = kvplan.TAG + (" finding: " if finding else ": ")
    return [json.loads(r.getMessage().split(tag, 1)[1]) for r in caplog.records
            if r.getMessage().startswith(tag)]


def test_plan_round_trip_skips_the_profile_run(plan_dir, caplog):
    """Input: a start measures (profile run), then a second start with the same config.
    Expect: one plan file holding the ident and 90 GiB; the second start returns the same
    90 GiB and the attributes later steps read, without a profile run (profiles stays 1),
    and logs a "reused" line. Not: that vLLM's real KV size equals the reused one (GPU)."""
    caplog.set_level(logging.INFO, logger="vllm.fastrsi.kvplan")
    assert Planned().determine_available_memory() == 90 * GiB
    files = list(plan_dir.glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec["result"] == 90 * GiB and rec["ident"]["config"] == "c1"
    w = Planned()
    assert w.determine_available_memory() == 90 * GiB
    assert FakeWorker.profiles == 1
    assert w.peak_activation_memory == 3 * GiB
    assert [e["event"] for e in _events(caplog, False)] == ["measured", "reused"]


def test_plan_of_another_config_is_a_finding_and_the_profile_runs(plan_dir, caplog):
    """Input: a plan measured for config c1; a start with config c2, then one with c1 but less
    free GPU memory than when measured. Expect: c2 profiles, logs a kvplan_mismatch finding
    naming the differing field (config: c1 -> c2) and records its own plan; the low-memory c1
    start profiles too, with a kvplan_free_memory finding. Before: a miss profiled with no
    word on why, so a config drift looked like a broken cache."""
    caplog.set_level(logging.INFO, logger="vllm.fastrsi.kvplan")
    Planned("c1").determine_available_memory()
    assert Planned("c2").determine_available_memory() == 90 * GiB
    assert FakeWorker.profiles == 2
    (f,) = _events(caplog, True)
    assert f["event"] == "kvplan_mismatch" and f["differs"] == {"config": ["c1", "c2"]}
    assert len(list(plan_dir.glob("*.json"))) == 2
    Planned("c1", free=100 * GiB).determine_available_memory()
    assert FakeWorker.profiles == 3
    assert _events(caplog, True)[-1]["event"] == "kvplan_free_memory"


# A stand-in for `vllm serve` that records argv and the KV plan env, logs one plan line as a
# worker would, and serves /health.
FAKE_VLLM_KV = r'''#!{py}
import http.server, json, os, sys
argv = sys.argv[1:]
open(os.environ["FAKE_OUT"], "w").write(json.dumps({{"argv": argv, "env": {{
    k: os.environ.get(k) for k in ("PYTHONPATH", "FASTRSI_KVPLAN_DIR")}}}}))
print('(EngineCore pid=1) INFO fastrsi kvplan: {{"event": "reused", "seconds": 0.07}}', flush=True)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(argv[argv.index("--port") + 1])), H).serve_forever()
'''


def test_kvplan_flag_installs_the_worker_class_and_reports_reuse():
    """Input: `engine.py start --kvplan`. Expect: the server gets --worker-cls
    kvplan.KVPlanWorker, this folder first on PYTHONPATH (so every worker can import it) and
    FASTRSI_KVPLAN_DIR under --cache-root; the report counts the reused line with its seconds.
    Not: that vLLM imports the class (GPU receipt, cases c and d)."""
    with tempfile.TemporaryDirectory() as d:
        fake = _script(d, "vllm", FAKE_VLLM_KV)
        out = os.path.join(d, "ran.json")
        r = subprocess.run([sys.executable, str(HERE / "engine.py"), "start", "m",
                            "--port", str(_free_port()), "--cache-root", os.path.join(d, "c"),
                            "--log", os.path.join(d, "serve.log"), "--vllm", fake,
                            "--timeout", "30", "--kvplan"], capture_output=True, text=True,
                           timeout=60, env=stub_env(d, FAKE_OUT=out))
        rep = json.loads(r.stdout)
        _stop(rep["pid"])
        assert r.returncode == 0, r.stdout + r.stderr
        ran = json.loads(Path(out).read_text())
        argv = ran["argv"]
        assert argv[argv.index("--worker-cls") + 1] == "kvplan.KVPlanWorker"
        assert ran["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(HERE)
        assert ran["env"]["FASTRSI_KVPLAN_DIR"] == str(Path(d, "c", "kvplan").resolve())
        assert rep["kvplan"]["reused"] == 1 and rep["kvplan"]["seconds"] == [0.07]


def test_kvplan_with_another_worker_cls_or_a_swallowed_model_is_a_finding():
    """Input: (1) `--kvplan --worker-cls my.Worker m`; (2) `--kvplan m` with nothing else, where
    the optional DIR of --kvplan takes `m`. Expect: exit 2 and nothing ran; (1) a
    kvplan_worker_cls finding naming --kvplan and `--worker-cls my.Worker` (before: the argv's
    class won and every start profiled, reported only as a missing reuse); (2) a vllm_model
    finding (vLLM would have served its default model)."""
    with tempfile.TemporaryDirectory() as d:
        fake = _script(d, "vllm", FAKE_VLLM_KV)
        out = os.path.join(d, "ran.json")
        seen = []
        for vargv in (["--worker-cls", "my.Worker", "m"], ["m"]):
            r = subprocess.run([sys.executable, str(HERE / "engine.py"), "start", "--cache-root",
                                os.path.join(d, "c"), "--log", os.path.join(d, "serve.log"),
                                "--vllm", fake, "--kvplan", *vargv], capture_output=True,
                               text=True, timeout=60, env=stub_env(d, FAKE_OUT=out))
            assert r.returncode == 2, r.stdout + r.stderr
            seen.append(json.loads(r.stdout)["findings"][0])
        assert seen[0]["check"] == "kvplan_worker_cls"
        assert "--kvplan" in seen[0]["observed"] and "my.Worker" in seen[0]["observed"]
        assert seen[1]["check"] == "vllm_model"
        assert not os.path.exists(out)
