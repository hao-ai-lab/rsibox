"""Weights holder: keep a model's weights in GPU memory across engine restarts.

`vllm preload` (the weight cache daemon) loads and post-processes the weights once and serves
them over a Unix socket, one socket per GPU. An engine started with `--load-format ipc_cache`
maps the daemon's GPU memory through CUDA IPC (zero copy) instead of reading the checkpoint.
Start the daemon once; it outlives every engine.

Usage:
  python3 weights.py start --model M --socket-dir D --log preload.log [--tp N]
      [--max-model-len L] [--timeout S] [-- extra vllm preload args]
  python3 weights.py probe --socket-dir D [--tp N]   # would an ipc_cache start find a daemon?

Both print one JSON line. `start` prints {"pid", "cmd", "ready", "seconds"} (and "error",
"stopped" when not ready); pid leads its own session (setsid): stop the daemon by that pid,
never by the caller's group.

Contract:
- socket_paths() is the socket vLLM's loader dials for each GPU the engine uses: the daemon
  must serve that exact path, or the engine waits for it (state_timeout_s, 300 s by default)
  and then fails.
- probe() connects to each socket (no request: the daemon handles a closed connection) and
  returns the ones that did not answer. engine.py runs it before every ipc_cache start and
  starts nothing when one is missing (a finding instead of a 300 s wait). When it cannot name
  the engine's GPUs (no nvidia-smi, or CUDA_VISIBLE_DEVICES lists fewer GPUs than tp) it
  cannot tell whether a socket is this engine's, so it fails too, saying so.
- start() stops the daemon it started (its session) when the daemon does not get ready: a
  failed or timed-out daemon never stays behind holding GPU memory.
- loader_config() turns the disk fallback off: an engine that asked for resident weights
  either maps them or fails at load; it never reads the checkpoint silently.

Writes only --log (and the daemon's sockets in --socket-dir).
"""
import argparse, hashlib, json, os, re, signal, socket, subprocess, sys, tempfile, time

READY = re.compile(r"Weight cache daemon READY")
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
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def stop(p, grace=10):
    """Stop the daemon's session (SIGTERM, SIGKILL after `grace` s); True if it was running.
    Only a session this module started: its pid leads it (start_new_session)."""
    if p.poll() is not None:
        return False
    try:
        if os.getpgid(p.pid) != p.pid:
            return False
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(grace)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(grace)
    except ProcessLookupError:
        pass
    return True


def start(a, extra):
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    cmd = daemon_cmd(a.model, a.socket_dir, a.tp, a.max_model_len, a.vllm, extra)
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
    return 0 if r["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
