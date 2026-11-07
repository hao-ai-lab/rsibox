"""Golden cases for the weights holder (weights.py) as engine.py uses it. CPU only: a fake
`vllm` stands in for the server, a fake `nvidia-smi` names the GPUs, and a plain Unix socket
listener stands in for the weight cache daemon.

Run: python3 -m pytest repos/vllm/livelab/services/test_weights.py -v
"""
import json, os, shutil, socket, subprocess, sys, time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from weights import socket_name  # noqa: E402
from test_engine import FAKE_VLLM, _free_port, _script, _short_dir, _stop, stub_env  # noqa: E402

ENGINE = str(HERE / "engine.py")

# nvidia-smi --query-gpu=uuid ... -i 1,2  ->  one uuid per index, like the real one.
FAKE_SMI = r'''#!{py}
import sys
ids = sys.argv[sys.argv.index("-i") + 1].split(",") if "-i" in sys.argv else ["0"]
print("\n".join(f"GPU-fake-{{i}}" for i in ids))
'''


@pytest.fixture
def d():
    path = _short_dir()     # AF_UNIX paths are short: not under pytest's tmp_path
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _listen(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(path); s.listen(4)
    return s


def _start(d, sock_dir, port, extra=()):
    smi_dir = os.path.join(d, "bin"); os.makedirs(smi_dir, exist_ok=True)
    _script(smi_dir, "nvidia-smi", FAKE_SMI)
    fake = _script(d, "vllm", FAKE_VLLM)
    out = os.path.join(d, "ran.json")
    env = stub_env(d, FAKE_OUT=out, CUDA_VISIBLE_DEVICES="1",
                   PATH=smi_dir + os.pathsep + os.environ["PATH"])
    t = time.time()
    r = subprocess.run([sys.executable, ENGINE, "start", "m", "--port", str(port),
                        "--cache-root", os.path.join(d, "c"), "--log",
                        os.path.join(d, "serve.log"), "--vllm", fake, "--timeout", "30",
                        "--weights-socket-dir", sock_dir, *extra],
                       capture_output=True, text=True, timeout=60, env=env)
    return r, time.time() - t, out


def test_ipc_cache_without_a_daemon_is_a_finding_and_nothing_starts(d):
    """Input: `engine.py start --load-format ipc_cache` on GPU 1 with an empty socket folder
    and, in a second start, a daemon socket for GPU 0 only. Expect: both exit 2 within a few
    seconds with a weights_daemon finding naming GPU 1's socket, and the fake vllm never ran.
    Before: the launcher started the server, and a real one waited state_timeout_s (300 s) for
    the daemon and then failed. Not: vLLM's real socket name (GPU receipt, case b)."""
    sock_dir = os.path.join(d, "s"); os.makedirs(sock_dir)
    want = os.path.join(sock_dir, socket_name("GPU-fake-1"))
    r, secs, out = _start(d, sock_dir, _free_port(), ["--load-format", "ipc_cache"])
    assert r.returncode == 2, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert rep["findings"][0]["check"] == "weights_daemon"
    assert rep["weights_probe"]["missing"] == [want]
    assert secs < 10 and not os.path.exists(out)
    other = _listen(os.path.join(sock_dir, socket_name("GPU-fake-0")))   # wrong GPU
    try:
        r, secs, out = _start(d, sock_dir, _free_port(), ["--weights"])
    finally:
        other.close()
    assert r.returncode == 2, r.stdout + r.stderr
    assert json.loads(r.stdout)["weights_probe"]["missing"] == [want]
    assert secs < 10 and not os.path.exists(out)


def test_weights_with_a_daemon_maps_them_without_disk_fallback(d):
    """Input: a listener on GPU 1's daemon socket; `engine.py start --weights`. Expect: the
    server starts with --load-format ipc_cache and a loader config naming that socket folder
    with fallback off (a mismatched daemon fails the load instead of reading the checkpoint
    silently); the report carries the probe. Not: a real daemon's protocol (GPU receipt)."""
    sock_dir = os.path.join(d, "s"); os.makedirs(sock_dir)
    daemon = _listen(os.path.join(sock_dir, socket_name("GPU-fake-1")))
    try:
        r, _, out = _start(d, sock_dir, _free_port(), ["--weights"])
    finally:
        daemon.close()
    rep = json.loads(r.stdout)
    _stop(rep["pid"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert rep["weights_probe"]["ok"] is True and rep["weights_probe"]["matched_gpus"] is True
    argv = json.loads(Path(out).read_text())["argv"]
    assert argv[argv.index("--load-format") + 1] == "ipc_cache"
    cfg = json.loads(argv[argv.index("--model-loader-extra-config") + 1])
    assert cfg == {"socket_dir": sock_dir, "mode": "zero_copy", "fallback": False}


def test_weights_with_an_explicit_other_load_format_is_a_finding(d):
    """Input: a listener on GPU 1's daemon socket; `engine.py start --weights m --load-format
    auto` (auto is vLLM's default, but given). Expect: exit 2, a weights_load_format finding
    naming --weights and `--load-format auto`; the fake vllm never ran. Before: the launcher
    put its own `--load-format ipc_cache` first and the argv's `auto` won (last flag wins), so
    --weights silently loaded from disk."""
    sock_dir = os.path.join(d, "s"); os.makedirs(sock_dir)
    daemon = _listen(os.path.join(sock_dir, socket_name("GPU-fake-1")))
    try:
        r, _, out = _start(d, sock_dir, _free_port(), ["--weights", "--load-format", "auto"])
    finally:
        daemon.close()
    assert r.returncode == 2, r.stdout + r.stderr
    f = json.loads(r.stdout)["findings"]
    assert [x["check"] for x in f] == ["weights_load_format"]
    assert "--weights" in f[0]["observed"] and "--load-format auto" in f[0]["observed"]
    assert not os.path.exists(out)
