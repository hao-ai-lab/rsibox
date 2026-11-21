"""Launchers: start the box daemon somewhere and wait for its box file.

A launcher never speaks a verb. `plan(spec)` returns what would run; `up(spec)` runs it,
waits for `<state>/box.json`, and records how it was started in `<state>/launch.json`;
`down(state)` closes the box and removes what `up` started.

`spec` keys: `launcher` (local | docker | gvisor | slurm), `lab`, `real`, `state`, and the
optional `python`, `repo`, `env` ({name: value}), `bind`, `port`, `busy_wait`, `image`,
`gpus`, `runtime`, `mounts`, `docker_args`, `nested_runsc`, `runsc`,
`allow_unsupported_driver`, `sbatch_args`, `setup`.
"""
import json
import os
import shlex
import subprocess
import sys
import time

from . import client as C
from .server import BOX_FILE

LAUNCH_FILE = "launch.json"
LIVELAB = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REPO_IN_CONTAINER, STATE_IN_CONTAINER = "/fastrsi", "/rsibox-state"
_DAEMONS = []


def repo_root(spec):
    return os.path.abspath(spec.get("repo") or os.path.dirname(os.path.dirname(LIVELAB)))


def serve_argv(spec, python, state, bind=None):
    argv = [python, "-u", "-m", "rsibox", "serve", "--lab", spec["lab"], "--state", state,
            "--bind", bind or spec.get("bind") or "127.0.0.1", "--port", str(spec.get("port") or 0)]
    if spec.get("real"):
        argv.append("--real")
    if spec.get("busy_wait") is not None:
        argv += ["--busy-wait", str(spec["busy_wait"])]
    return argv


def _gvisor_wrap(spec, python, inner, state, runsc):
    wrap = [python, "-m", "rsibox.gvisor", "--runsc", runsc, "--bind", state]
    if spec.get("allow_unsupported_driver"):
        wrap.append("--allow-unsupported-driver")
    return [*wrap, "--", *inner]


def plan(spec):
    """What `up` would run: `{"argv", "env", "script"}` (script: the sbatch file, else None)."""
    kind, state = spec["launcher"], os.path.abspath(spec["state"])
    env, script = dict(spec.get("env") or {}), None
    if kind in ("local", "gvisor", "slurm"):
        python = spec.get("python") or sys.executable
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(repo_root(spec), "core", "livelab")] + [p for p in [env.get("PYTHONPATH")] if p])
    if kind == "local":
        argv = serve_argv(spec, python, state)
    elif kind == "gvisor":
        argv = _gvisor_wrap(spec, python, serve_argv(spec, python, state), state, spec.get("runsc") or "runsc")
    elif kind == "docker":
        python = spec.get("python") or "python3"
        env["PYTHONPATH"] = os.pathsep.join(
            [f"{REPO_IN_CONTAINER}/core/livelab"] + [p for p in [env.get("PYTHONPATH")] if p])
        inner = serve_argv(spec, python, STATE_IN_CONTAINER)
        argv = ["docker", "run", "-d", "--name", container_name(state), "--network", "host",
                "-v", f"{repo_root(spec)}:{REPO_IN_CONTAINER}:ro", "-v", f"{state}:{STATE_IN_CONTAINER}",
                "-w", f"{REPO_IN_CONTAINER}/core/livelab"]
        if spec.get("runtime"):
            argv += ["--runtime", spec["runtime"]]
        if spec.get("gpus"):
            argv += ["--gpus", spec["gpus"]]
        if spec.get("nested_runsc"):
            argv.append("--privileged")
            inner = _gvisor_wrap(spec, python, inner, STATE_IN_CONTAINER, spec["nested_runsc"])
        for mount in spec.get("mounts") or ():
            argv += ["-v", mount]
        for key, val in sorted(env.items()):
            argv += ["-e", f"{key}={val}"]
        argv += [*(spec.get("docker_args") or ()), "--entrypoint", inner[0], spec["image"], *inner[1:]]
        env = {}
    elif kind == "slurm":
        lines = ["#!/bin/bash", "#SBATCH --job-name=rsibox", f"#SBATCH --output={state}/slurm-%j.log"]
        lines += [f"#SBATCH {arg}" for arg in spec.get("sbatch_args") or ()]
        lines += ["set -eu"] + ([spec["setup"]] if spec.get("setup") else [])
        lines += [f"export {key}={shlex.quote(str(val))}" for key, val in sorted(env.items())]
        lines += ["exec " + shlex.join(serve_argv(spec, python, state, bind=spec.get("bind") or "0.0.0.0"))]
        script, argv, env = "\n".join(lines) + "\n", ["sbatch", "--parsable", os.path.join(state, "rsibox.sbatch")], {}
    else:
        raise ValueError(f"unknown launcher {kind!r} (local, docker, gvisor, slurm)")
    return {"argv": argv, "env": env, "script": script}


def container_name(state):
    return "rsibox-" + "".join(c if c.isalnum() else "-" for c in os.path.basename(state.rstrip("/")))[-40:]


def up(spec, wait_s=120.0):
    """Start the daemon and wait for its box file. Returns the launch record."""
    kind, state = spec["launcher"], os.path.abspath(spec["state"])
    os.makedirs(state, exist_ok=True)
    box_file = os.path.join(state, BOX_FILE)
    if os.path.exists(box_file):
        os.remove(box_file)
    what = plan(spec)
    record = {"launcher": kind, "state": state, "box_file": box_file, "argv": what["argv"]}
    if kind in ("local", "gvisor"):
        with open(os.path.join(state, "daemon.log"), "ab") as log:
            proc = subprocess.Popen(what["argv"], env={**os.environ, **what["env"]}, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        _DAEMONS.append(proc)                    # the daemon outlives this call by design
        record["pid"] = proc.pid
        alive = lambda: proc.poll() is None
    else:
        if what["script"]:
            with open(what["argv"][-1], "w", encoding="utf-8") as handle:
                handle.write(what["script"])
        done = subprocess.run(what["argv"], capture_output=True, text=True)
        if done.returncode != 0:
            return {**record, "ok": False, "error": (done.stderr or done.stdout).strip()[-800:]}
        record["container" if kind == "docker" else "job"] = done.stdout.strip().split(";")[0]
        alive = (lambda: _container_running(record["container"])) if kind == "docker" else (lambda: True)
    with open(os.path.join(state, LAUNCH_FILE), "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    deadline = time.monotonic() + wait_s
    while not os.path.exists(box_file):
        if not alive():
            return {**record, "ok": False, "error": f"the daemon exited before writing {box_file}; see the log in {state}"}
        if time.monotonic() > deadline:
            return {**record, "ok": False, "error": f"no box file after {wait_s} s (queued job, or a slow start)"}
        time.sleep(0.1)
    if kind == "docker" and not os.access(box_file, os.R_OK):
        # the daemon ran as another user (root in the container): take our own copy of the file
        got = subprocess.run(["docker", "exec", record["container"], "cat", f"{STATE_IN_CONTAINER}/{BOX_FILE}"],
                             capture_output=True, text=True)
        if got.returncode != 0:
            return {**record, "ok": False, "error": f"cannot read {box_file}: {got.stderr.strip()[-300:]}"}
        fd = os.open(box_file + ".host", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(got.stdout)
        os.replace(box_file + ".host", box_file)
    return {**record, "ok": True}


def _container_running(name):
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True)
    return out.stdout.strip() == "true"


def down(state):
    """Close the box, then remove what `up` started. Each step reports what it did."""
    state, done = os.path.abspath(state), {}
    try:
        with open(os.path.join(state, LAUNCH_FILE), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        record = {}
    try:
        done["close"] = C.Client.from_box_file(os.path.join(state, BOX_FILE), timeout=120).close()
    except (OSError, ValueError, KeyError) as exc:
        done["close"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if record.get("pid"):
        try:
            os.killpg(record["pid"], 15)
            done["signal"] = "SIGTERM"
        except (ProcessLookupError, PermissionError):
            done["signal"] = "already gone"
        for proc in [p for p in _DAEMONS if p.pid == record["pid"]]:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                done["signal"] = "SIGTERM sent, still running after 30 s"
            _DAEMONS.remove(proc)
    if record.get("container"):
        done["docker_rm"] = subprocess.run(["docker", "rm", "-f", record["container"]],
                                           capture_output=True, text=True).returncode
    if record.get("job"):
        done["scancel"] = subprocess.run(["scancel", record["job"]], capture_output=True, text=True).returncode
    return {"ok": bool(done["close"].get("ok")) or any(k in done for k in ("signal", "docker_rm", "scancel")),
            **done}
