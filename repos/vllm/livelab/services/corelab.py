"""Puts the framework's shared live-lab runtime on sys.path: core/livelab, home of `rsibox`
(process, IPC and import-hook helpers) and `forksafe` (pre-fork CUDA guards). RSIBOX_LIVELAB
overrides the folder. Stdlib only: importing it costs nothing."""
import contextlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

LIVELAB = os.path.abspath(os.environ.get("RSIBOX_LIVELAB") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "core", "livelab"))
if LIVELAB not in sys.path:
    sys.path.insert(1, LIVELAB)


def read_text(path, *, missing_ok=True):
    """Read UTF-8 text lossily; return '' when `missing_ok` and `path` is absent."""
    if not path or (missing_ok and not os.path.exists(path)):
        return ""
    with open(path, errors="replace") as fh:
        return fh.read()


def last_json_line(text, default=None):
    """Parse the last line starting with '{' from `text`, else return `{}` (or `default`)."""
    line = next((x for x in (text or "").splitlines()[::-1] if x.startswith("{")), None)
    return json.loads(line) if line else ({} if default is None else default)


def spawn_logged(cmd, log_path, env=None):
    """Spawn `cmd` in a new session with stdout+stderr redirected to `log_path`."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        return subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )


def http_healthy(port, timeout=0.5, path="/health"):
    """True when GET `http://127.0.0.1:<port><path>` answers HTTP 200."""
    try:
        with contextlib.closing(http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)) as conn:
            conn.request("GET", path)
            return conn.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False


def http_post_json(port, path, body, timeout=120.0):
    """POST JSON `body` to `http://127.0.0.1:<port><path>` and decode the 200 JSON reply."""
    with contextlib.closing(http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)) as conn:
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"POST {path} on :{port}: {resp.status} {data[:500]!r}")
    return json.loads(data) if data else {}


def weight_cache_get_state(socket_path, cache_key, label="holder"):
    """Send a `get_state` RPC to a vLLM weight-cache daemon over AF_UNIX."""
    from vllm.model_executor.model_loader.weight_cache import protocol as P
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(socket_path)
        P.send_msg(conn, {"cmd": "get_state", "cache_key": cache_key})
        reply = P.recv_msg(conn)
    if reply.get("status") != "ok":
        raise RuntimeError(f"{label} at {socket_path} answered {reply}")
    return reply["entries"]
