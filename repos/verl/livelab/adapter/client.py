"""Gate protocol client for a live gated verl job (`adapter/gate.py`).

Shared by `VerlConformanceLab` (`adapter/conformance_lab.py`) and `tools.cli`
(`repos/verl/tools/cli.py`) so `fastrsi conformance verl --real` works with only
`repos/verl/livelab` + `core/livelab` on `PYTHONPATH`.
"""
import json
import os
import socket
import time
from pathlib import Path

import adapter  # noqa: F401  (ensures core/livelab is on sys.path)
from fastrsi import ipc, proc


class GateError(RuntimeError):
    """The gated verl job is not answering or has exited."""


def _job_died(gate_dir) -> bool:
    """True only when the driver PID recorded in gate.json is gone on *this* host; a client on
    another host (a Slurm login node reading a shared gate dir) cannot see the PID and says no."""
    info = ipc.read_json(Path(gate_dir) / "gate.json")
    if not isinstance(info, dict):
        return False
    pid, host = info.get("pid"), info.get("host")
    if isinstance(pid, int) and not isinstance(pid, bool) and (host is None or host == socket.gethostname()):
        return not proc.pid_alive(pid)
    return False


def _next_request(gate_path, settle_s: float = 10.0) -> int:
    """The request number `ready.json` names. Right after a response the gate may not have
    advanced `ready.json` yet (two files on a shared file system), so a `req-<n>.json` that
    already exists is waited out for `settle_s` before it counts as another client's request."""
    gate_path = Path(gate_path)
    deadline = time.monotonic() + settle_s
    while True:
        data = ipc.read_json(gate_path / "ready.json")
        req_num = data.get("n") if isinstance(data, dict) else None
        if req_num is not None and not (gate_path / f"req-{req_num}.json").exists():
            return req_num
        if time.monotonic() > deadline:
            raise GateError(f"{gate_path}: request {req_num} is already pending")
        time.sleep(0.05)


def request(
    gate_dir,
    files=(),
    steps: int = 1,
    stop: bool = False,
    timeout: float = 1800.0,
    live_settings=None,
    verb: str = "install",
    extra=None,
) -> dict:
    gate_path = Path(gate_dir)
    if not (gate_path / "ready.json").exists():
        raise GateError(f"{gate_path}: no ready.json; is the job running with this gate?")
    req_num = _next_request(gate_path)
    payload = (
        {"stop": True}
        if stop
        else {
            "verb": verb,
            "files": [os.path.realpath(f) for f in files],
            "steps": steps,
            **(extra or {}),
            **({"set": {k: str(v) for k, v in live_settings.items()}} if live_settings else {}),
        }
    )
    payload["t_client"] = time.time()
    ipc.atomic_write_json(gate_path / f"req-{req_num}.json", payload)
    resp, deadline = gate_path / f"resp-{req_num}.json", time.monotonic() + timeout
    while not resp.exists():
        if (gate_path / "stopped.json").exists() and not stop:
            raise GateError(f"{gate_path}: job stopped while serving request {req_num}")
        if _job_died(gate_path) and not stop:
            raise GateError(f"{gate_path}: job process died while serving request {req_num}")
        if time.monotonic() > deadline:
            raise GateError(f"{gate_path}: timed out after {timeout}s")
        time.sleep(0.05)
    return json.loads(resp.read_text(encoding="utf-8"))


def summary(receipt: dict) -> dict:
    keep = (
        "ok",
        "outcome",
        "verdict_s",
        "install_s",
        "processes",
        "proven",
        "witness",
        "set",
        "metrics",
        "error",
        "codes",
        "restart_required",
    )
    return {k: receipt[k] for k in keep if k in receipt}
