"""Weights holder: keep a model's weights in GPU memory across engine restarts.

`vllm preload` (the weight cache daemon) loads and post-processes the weights once and serves
them over a Unix socket, one socket per GPU. An engine started with `--load-format ipc_cache`
maps the daemon's GPU memory through CUDA IPC (zero copy) instead of reading the checkpoint.
Start the daemon once; it outlives every engine.

Usage:
  python3 weights.py start --model M --log preload.log [--socket-dir D] [--tp N]
      [--max-model-len L] [--timeout S] [-- extra vllm preload args]
  python3 weights.py probe [--socket-dir D] [--tp N]   # would an ipc_cache start find a daemon?

Both print one JSON line. `start` prints {"pid", "cmd", "ready", "seconds"} (and "error",
"stopped" or "finding" when not ready); pid leads its own session (setsid): stop the daemon
by that pid, never by the caller's group.

Contract:
- socket_paths() is the socket vLLM's loader dials for each GPU the engine uses: the daemon
  must serve that exact path, or the engine waits for it (state_timeout_s, 300 s by default)
  and then fails.
- probe() connects to each socket (no request: the daemon handles a closed connection) and
  returns the ones that did not answer. engine.py runs it before every ipc_cache start and
  starts nothing when one is missing (a finding instead of a 300 s wait). When it cannot name
  the engine's GPUs (no nvidia-smi, or CUDA_VISIBLE_DEVICES lists fewer GPUs than tp) it
  cannot tell whether a socket is this engine's, so it fails too, saying so.
- start() refuses (exit 2, a `weights_socket_path` finding, nothing started) a socket dir
  whose daemon socket path would exceed AF_UNIX's 107 bytes. The name is vLLM's, unchanged:
  md5 of the uuid gives 55 bytes, the sha256 fallback on FIPS hosts 87, and a draft model's
  socket 6 more ("_draft"); the check counts the longest.
- start() stops the daemon it started (its session) when the daemon does not get ready: a
  failed or timed-out daemon never stays behind holding GPU memory.
- loader_config() turns the disk fallback off: an engine that asked for resident weights
  either maps them or fails at load; it never reads the checkpoint silently.

Writes only --log (and the daemon's sockets in --socket-dir).
"""
import argparse, hashlib, json, os, re, signal, socket, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corelab  # noqa: F401,E402  (core/livelab on sys.path)
from fastrsi import proc  # noqa: E402

READY = re.compile(r"Weight cache daemon READY")
MAX_SOCK_PATH = 107     # Linux sun_path is 108 bytes with the trailing NUL
FAILED = ("Traceback (most recent call last)", "exited during startup")


def default_socket_dir():
    """vLLM's default when no socket dir is given (weight_cache/protocol.py get_socket_dir)."""
    uid = os.getuid() if hasattr(os, "getuid") else -1
    return os.path.join(tempfile.gettempdir(), f"vllm_weight_cache_{uid}")


def socket_name(gpu_uuid):
    """vLLM's per-GPU socket name: md5 of the NVML uuid (safe_hash; sha256 on FIPS systems)."""
    try:
        h = hashlib.md5(gpu_uuid.encode())
    except ValueError:      # FIPS: md5 unavailable, vLLM falls back to sha256 too
        h = hashlib.sha256(gpu_uuid.encode())
    return f"vllm_weight_cache_{h.hexdigest()}.sock"


def long_socket_path_finding(socket_dir):
    """A `weights_socket_path` finding when the daemon's sockets in socket_dir would be longer
    than AF_UNIX allows, else None. Every GPU's socket name has the same length (a hex digest),
    so this needs no GPU uuid. The longest name counts: a draft model's daemon group adds
    "_draft" (vLLM's format_socket_role_suffix). The daemon binds only after loading the
    weights, so a too-long path fails minutes in; this check costs microseconds."""
    name = socket_name("GPU-length-only").replace(".sock", "_draft.sock")
    path = os.path.join(os.path.abspath(socket_dir), name)
    path_bytes = len(os.fsencode(path))
    if path_bytes <= MAX_SOCK_PATH:
        return None
    return {"check": "weights_socket_path",
            "observed": f"daemon socket path is {path_bytes} bytes > {MAX_SOCK_PATH} (AF_UNIX): {path}",
            "expected": f"--socket-dir short enough that the socket path fits {MAX_SOCK_PATH} "
                        "bytes",
            "why": "the daemon loads the weights, then fails to bind its socket",
            "actions": ["use a short --socket-dir, such as /dev/shm/<run>/w"]}


def gpu_uuids(visible, tp):
    """NVML uuids of the GPUs a TP-`tp` engine uses: the first `tp` entries of
    CUDA_VISIBLE_DEVICES (vLLM maps them to NVML indices the same way). None if nvidia-smi
    cannot answer."""
    ids = [x for x in (visible or "").split(",") if x] or [str(i) for i in range(tp)]
    q = ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", ",".join(ids[:tp])]
    try:
        out = subprocess.check_output(q, text=True, timeout=30, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    uuids = out.split()
    return uuids if len(uuids) == tp else None


def socket_paths(socket_dir, visible, tp):
    """The socket each rank's loader dials, or None when the GPU uuids are unknown."""
    uuids = gpu_uuids(visible, tp)
    if uuids is None:
        return None
    return [os.path.join(socket_dir, socket_name(u)) for u in uuids]


def _answers(path, timeout=2.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        return True
    except OSError:
        return False
    finally:
        s.close()


def probe(socket_dir, visible=None, tp=1):
    """{"ok", "socket_dir", "missing", "matched_gpus", "seconds"}. With GPU uuids known, every
    rank's socket must answer. Without them (no nvidia-smi, or fewer visible GPUs than tp) the
    probe cannot match a socket to a GPU: ok is False, matched_gpus False, "error" says why and
    "answering" lists the daemon sockets in the folder that answer (any GPU's)."""
    t0 = time.time()
    paths = socket_paths(socket_dir, visible, tp)
    if paths is None:
        try:
            found = sorted(os.path.join(socket_dir, n) for n in os.listdir(socket_dir)
                           if n.startswith("vllm_weight_cache_") and n.endswith(".sock"))
        except OSError:
            found = []
        return {"ok": False, "socket_dir": socket_dir, "missing": [], "matched_gpus": False,
                "answering": [p for p in found if _answers(p)],
                "error": f"cannot name the {tp} GPU(s) of this engine (nvidia-smi failed, or "
                         f"CUDA_VISIBLE_DEVICES={visible!r} lists fewer than {tp})",
                "seconds": round(time.time() - t0, 3)}
    missing = [p for p in paths if not _answers(p)]
    return {"ok": not missing, "socket_dir": socket_dir, "missing": missing,
            "matched_gpus": True, "seconds": round(time.time() - t0, 3)}


def daemon_finding(pr):
    """The finding for a failed probe (checks advise; the caller decides)."""
    observed = (f"{pr['error']}; {len(pr.get('answering', []))} daemon socket(s) of unknown "
                f"GPUs answer in {pr['socket_dir']}" if pr.get("error") else
                f"no weight cache daemon answers at {', '.join(pr['missing'])}")
    return {"check": "weights_daemon", "observed": observed,
            "expected": "one daemon socket per GPU of this engine, answering",
            "why": "an ipc_cache start waits up to state_timeout_s (300 s) for the daemon and "
                   "then fails; a run waited the full 300 s for a daemon that never started",
            "actions": ["start the daemon: weights.py start --socket-dir "
                        f"{pr['socket_dir']} (cost: one disk load, once)",
                        "start from disk instead: drop --weights / --load-format ipc_cache",
                        "--weights-accept-missing: start anyway (the engine waits for a daemon "
                        "started later, up to 300 s)"],
            "risk": "a start that waits 300 s and fails"}


def loader_config(socket_dir, mode="zero_copy"):
    """--model-loader-extra-config for an ipc_cache engine: this daemon, and no disk fallback."""
    return {"socket_dir": socket_dir, "mode": mode, "fallback": False}


def daemon_cmd(model, socket_dir, tp=1, max_model_len=None, vllm="vllm", extra=()):
    cmd = [vllm, "preload", "--model", model, "--tensor-parallel-size", str(tp),
           "--weight-cache-socket-dir", socket_dir]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    return cmd + list(extra)


def wait_ready(log, pid, timeout=1800, poll=0.5):
    """{"ready": True, "seconds"} once the daemon logs READY; {"ready": False, "error"} as soon
    as it exits or logs a failure, or at the timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        text = open(log, errors="replace").read() if os.path.exists(log) else ""
        if READY.search(text):
            return {"ready": True, "seconds": round(time.time() - t0, 2)}
        if any(s in text for s in FAILED) or _exited(pid):
            return {"ready": False, "error": "daemon exited or failed; see its log",
                    "seconds": round(time.time() - t0, 2)}
        time.sleep(poll)
    return {"ready": False, "error": "timeout", "seconds": round(time.time() - t0, 2)}


def _exited(pid):
    """True when pid is gone or a zombie (our unreaped child)."""
    return not proc.pid_alive(pid)


def _pg_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_session(pid, grace=10.0, reap=None):
    """Stop the process group `pid` leads (a daemon started with start_new_session, here or by
    reshard_live.py) and the processes below it: SIGTERM, then SIGKILL after `grace` s, each
    followed by a wait of up to `grace` s for the group to empty. `reap` (the caller's
    Popen.poll) collects a zombie leader, which would keep the group alive. Returns True when
    something of it was running. A live pid that leads no group of its own is another process
    and is never signalled; a leader that exited keeps its group id reserved while members
    remain (Linux reuses no live group id), so the group is still stopped by it."""
    try:
        if not _exited(pid) and os.getpgid(pid) != pid:
            return False
    except ProcessLookupError:
        pass
    if not _pg_alive(pid):
        return False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        proc.kill_tree(pid, sig, include_pg=True)
        deadline = time.time() + grace
        while time.time() < deadline:
            if reap is not None:
                reap()
            if not _pg_alive(pid):
                return True
            time.sleep(0.02)
    return True


def stop(p, grace=10):
    """Stop the daemon's session (SIGTERM, SIGKILL after `grace` s); True if it was running.
    Only a session this module started: its pid leads it (start_new_session)."""
    return stop_session(p.pid, grace, reap=p.poll)


def start(a, extra):
    cmd = daemon_cmd(a.model, a.socket_dir, a.tp, a.max_model_len, a.vllm, extra)
    finding = long_socket_path_finding(a.socket_dir)
    if finding:
        return {"pid": None, "cmd": cmd, "ready": False, "error": finding["observed"],
                "seconds": 0.0, "finding": finding}
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    with open(a.log, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    r = {"pid": p.pid, "cmd": cmd, **wait_ready(a.log, p.pid, a.timeout)}
    if not r["ready"]:
        r["stopped"] = stop(p)
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["start", "probe"])
    p.add_argument("--model"); p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--socket-dir", default=default_socket_dir())
    p.add_argument("--log"); p.add_argument("--vllm", default="vllm")
    p.add_argument("--timeout", type=float, default=1800)
    a, extra = p.parse_known_args(argv)
    extra = [x for x in extra if x != "--"]
    if a.cmd == "probe":
        r = probe(a.socket_dir, os.environ.get("CUDA_VISIBLE_DEVICES"), a.tp)
        if not r["ok"]:
            r["finding"] = daemon_finding(r)
        print(json.dumps(r))
        return 0 if r["ok"] else 2
    r = start(a, extra)
    print(json.dumps(r))
    return 0 if r["ready"] else (2 if "finding" in r else 1)


if __name__ == "__main__":
    sys.exit(main())
