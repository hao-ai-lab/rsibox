"""A zygote whose socket path is too long for AF_UNIX must be reported at once, not after a
10-minute wait: the zygote died at bind() after its imports, stayed a zombie child of the
launcher, and wait_ready() saw /proc/<pid> and kept polling."""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zygote


def test_long_socket_path_is_reported_before_launch():
    """Input: a 134-character socket path, then a short one. Expected: the long one is an
    error naming "too long", the short one None. Pain point: a zygote that died at bind() after
    its imports, found only by a 10-minute wait. Not: the bind itself."""
    long_path = "/tmp/" + "x" * 120 + "/zyg.sock"
    err = zygote.sock_path_error(long_path)
    assert err and "too long" in err
    assert zygote.sock_path_error("/dev/shm/run/zyg.sock") is None


def test_wait_ready_sees_an_unreaped_dead_child():
    """Input: a child that exited but is not reaped (/proc/<pid> still exists). Expected:
    wait_ready returns {"ready": False, "error": "zygote exited"} in under 5 s. Pain point: a
    10-minute wait on a zombie zygote. Not: a zygote that is alive but never binds."""
    p = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
    time.sleep(0.5)                  # exited but not reaped: a zombie, /proc/<pid> still exists
    assert os.path.exists(f"/proc/{p.pid}")
    t = time.time()
    r = zygote.wait_ready("/tmp/nonexistent-zyg.sock", p.pid, timeout=20)
    assert r == {"ready": False, "error": "zygote exited"}
    assert time.time() - t < 5
    p.wait()
