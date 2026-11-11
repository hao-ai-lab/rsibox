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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
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


def _refused(r):
    """Assert the start exited 2 with nothing started; stop the server if one did start."""
    if r.returncode == 0:
        _stop(json.loads(r.stdout)["pid"])
    assert r.returncode == 2, r.stdout + r.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
def test_weights_with_an_explicit_config_that_keeps_the_disk_fallback_is_a_finding(d):
    """Input: a listener on GPU 1's daemon socket; `engine.py start --weights m
    --model-loader-extra-config C` with C = {"socket_dir": S} (fallback unset; vLLM's
    default is true) and C = {"socket_dir": S, "fallback": true}. Expect: exit 2, a
    weights_fallback finding naming the value; the fake vllm never ran. With "fallback": false
    it starts. Before: the launcher passed the config through and the engine read the
    checkpoint from disk whenever the mapping failed, though --weights promises no fallback."""
    sock_dir = os.path.join(d, "s"); os.makedirs(sock_dir)
    daemon = _listen(os.path.join(sock_dir, socket_name("GPU-fake-1")))
    try:
        seen = []
        for cfg in ({"socket_dir": sock_dir}, {"socket_dir": sock_dir, "fallback": True}):
            r, _, out = _start(d, sock_dir, _free_port(), [
                "--weights", "--model-loader-extra-config", json.dumps(cfg)])
            _refused(r)
            f = json.loads(r.stdout)["findings"]
            assert [x["check"] for x in f] == ["weights_fallback"], f
            seen.append(f[0]["observed"])
            assert not os.path.exists(out)
        assert "unset (vLLM's default: true)" in seen[0] and seen[1].endswith("fallback true")
        r, _, out = _start(d, sock_dir, _free_port(), [
            "--weights", "--model-loader-extra-config",
            json.dumps({"socket_dir": sock_dir, "fallback": False})])
    finally:
        daemon.close()
    _stop(json.loads(r.stdout)["pid"])
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_probe_checks_the_folder_vllm_dials(d):
    """Input: a daemon socket for GPU 1 in folder A only; `engine.py start --weights
    --weights-socket-dir A m --model-loader-extra-config {"socket_dir": B, "fallback": false}`.
    Expect: exit 2, a weights_socket_dir finding naming A and B; nothing started. Without
    --weights-socket-dir the probe checks B (what vLLM dials): a weights_daemon finding naming
    B's socket. Before: the probe checked A, passed, and vLLM waited 300 s on B."""
    a_dir, b_dir = os.path.join(d, "a"), os.path.join(d, "b")
    os.makedirs(a_dir); os.makedirs(b_dir)
    cfg = json.dumps({"socket_dir": b_dir, "fallback": False})
    daemon = _listen(os.path.join(a_dir, socket_name("GPU-fake-1")))
    try:
        r, _, out = _start(d, a_dir, _free_port(), ["--weights", "--model-loader-extra-config",
                                                    cfg])
        r2, _, _ = _start(d, "", _free_port(), ["--weights", "--model-loader-extra-config", cfg])
    finally:
        daemon.close()
    _refused(r)
    f = json.loads(r.stdout)["findings"]
    assert [x["check"] for x in f] == ["weights_socket_dir"], f
    assert a_dir in f[0]["observed"] and b_dir in f[0]["observed"]
    assert not os.path.exists(out)
    _refused(r2)
    assert json.loads(r2.stdout)["weights_probe"]["missing"] == [
        os.path.join(b_dir, socket_name("GPU-fake-1"))]


def test_a_probe_that_cannot_name_the_gpus_does_not_pass(d):
    """Input: CUDA_VISIBLE_DEVICES=1 but `-tp 2` (nvidia-smi names one GPU, not two), and a
    daemon socket for GPU 1 answering. Expect: exit 2, a weights_daemon finding that says the
    GPUs could not be named and that 1 socket of unknown GPUs answers; matched_gpus false;
    nothing started. Before: any one answering socket passed a TP2 probe."""
    sock_dir = os.path.join(d, "s"); os.makedirs(sock_dir)
    daemon = _listen(os.path.join(sock_dir, socket_name("GPU-fake-1")))
    try:
        r, _, out = _start(d, sock_dir, _free_port(), ["--weights", "-tp", "2"])
    finally:
        daemon.close()
    _refused(r)
    rep = json.loads(r.stdout)
    assert rep["weights_probe"]["ok"] is False and rep["weights_probe"]["matched_gpus"] is False
    assert "cannot name the 2 GPU(s)" in rep["findings"][0]["observed"]
    assert "1 daemon socket(s) of unknown GPUs" in rep["findings"][0]["observed"]
    assert not os.path.exists(out)


# A `vllm preload` that never gets ready: it logs a traceback (FAKE_FAIL=1) or stays silent,
# and keeps running either way, as a daemon stuck after an error would.
FAKE_PRELOAD = r'''#!{py}
import os, sys, time
if os.environ.get("FAKE_FAIL") == "1":
    print("Traceback (most recent call last):", flush=True)
time.sleep(120)
'''


def test_a_daemon_that_does_not_get_ready_is_stopped(d):
    """Input: `weights.py start` with a fake `vllm preload` that logs a traceback and keeps
    running, then one that stays silent past --timeout 2. Expect: ready false, stopped true,
    and the daemon's pid is gone right after (it would hold GPU memory). Before: start()
    returned and the daemon kept running in its own session."""
    fake = _script(d, "vllm", FAKE_PRELOAD)
    for fail, timeout in (("1", "60"), ("0", "2")):
        r = subprocess.run([sys.executable, str(HERE / "weights.py"), "start", "--model", "m",
                            "--socket-dir", os.path.join(d, "s"), "--log",
                            os.path.join(d, "preload.log"), "--vllm", fake,
                            "--timeout", timeout], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "FAKE_FAIL": fail})
        rep = json.loads(r.stdout)
        try:
            assert rep["ready"] is False and rep.get("stopped") is True, rep
            assert not os.path.exists(f"/proc/{rep['pid']}"), rep
        finally:
            _stop(rep["pid"])


def test_stop_terminates_surviving_child_after_session_leader_exits(d):
    """Input: a session leader started with `start_new_session=True` that spawns a child in the
    same process group and then exits immediately while the child stays alive. Expect:
    `weights.stop(proc)` still signals the process group by `proc.pid`, terminates the child,
    and returns True."""
    import weights

    child_pid_file = os.path.join(d, "child.pid")
    leader_code = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({child_pid_file!r}, 'w').write(str(child.pid))\n"
    )
    leader_proc = subprocess.Popen(
        [sys.executable, "-c", leader_code], start_new_session=True
    )
    leader_proc.wait(timeout=10)
    assert leader_proc.poll() is not None
    deadline = time.time() + 5.0
    while not os.path.exists(child_pid_file) and time.time() < deadline:
        time.sleep(0.02)
    child_pid = int(Path(child_pid_file).read_text().strip())
    try:
        assert weights.stop(leader_proc, grace=2) is True
        assert weights._exited(child_pid)
    finally:
        _stop(leader_proc.pid)
        _stop(child_pid)


def test_a_socket_dir_too_long_for_af_unix_is_refused_before_the_daemon_loads(d):
    """Input: `weights.py start --socket-dir D` with D 60 bytes long, so D plus vLLM's 55-byte
    socket name (`vllm_weight_cache_<md5 of the GPU uuid>.sock`) exceeds AF_UNIX's 107 bytes;
    the same D for weight_scaling_colocate's Holder.start. Expect: weights.py exits 2 at once
    with a `weights_socket_path` finding and the fake `vllm preload` never runs; Holder.start
    raises before it spawns anything. Before: the daemon loaded the weights for minutes and
    then failed at bind(). Does not prove: vLLM's own name (taken from vLLM's
    weight_cache/protocol.py get_socket_path, not imported here)."""
    long_dir = os.path.join(d, "s" * (60 - len(d) - 1))
    assert len(os.fsencode(os.path.abspath(long_dir))) == 60
    marker = os.path.join(d, "preload-ran")
    fake = _script(d, "vllm", "#!{py}\nopen(%r, 'w').write('ran')\n" % marker)
    t = time.time()
    r = subprocess.run([sys.executable, str(HERE / "weights.py"), "start", "--model", "m",
                        "--socket-dir", long_dir, "--log", os.path.join(d, "preload.log"),
                        "--vllm", fake, "--timeout", "30"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert rep["ready"] is False and rep["pid"] is None
    assert rep["finding"]["check"] == "weights_socket_path" and "107" in rep["finding"]["observed"]
    assert time.time() - t < 10 and not os.path.exists(marker)
    import weight_scaling_colocate
    holder = weight_scaling_colocate.Holder(0, long_dir, os.path.join(d, "holder.log"))
    with pytest.raises(RuntimeError, match="too long"):
        holder.start("m", sys.executable)
    assert holder.proc is None and not os.path.exists(os.path.join(d, "holder.log"))
