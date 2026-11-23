#!/usr/bin/env python3
"""Host and probe SGLang's native IPC weight-cache daemon without importing torch or sglang.

SGLang's `sglang.srt.weight_cache.daemon` binds one `AF_UNIX` socket and writes one `.ready`
marker file (`pid=<int>`) per GPU, named by two env templates containing `{device_uuid}`:
- `SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE`
- `SGLANG_WEIGHT_CACHE_READY_TEMPLATE`

A worker started with `weight_cache_mode="client"` and those same two env templates maps
unquantized/FP8 checkpoint weights over CUDA IPC in seconds rather than reading disk or
safely failing before engine launch if any target GPU socket or `.ready` PID is missing.
"""
import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time

try:
    from rsibox import harness, ipc, proc
except ImportError:  # pragma: no cover - direct CLI invocation without PYTHONPATH
    _CORE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "core", "livelab")
    )
    if os.path.isdir(_CORE) and _CORE not in sys.path:
        sys.path.insert(0, _CORE)
    from rsibox import harness, ipc, proc

_SAMPLE_UUID = "0" * 36
SOCKET_ENV = "SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"
READY_ENV = "SGLANG_WEIGHT_CACHE_READY_TEMPLATE"


def normalize_uuid(uuid: str) -> str:
    """Normalize an `nvidia-smi` GPU UUID (`GPU-<uuid>`) to SGLang's `<uuid>` form."""
    s = str(uuid).strip()
    return s[4:] if s.startswith("GPU-") else s


def socket_template(sock_dir: str) -> str:
    """Return the `SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE` under `sock_dir`."""
    return os.path.join(os.path.abspath(os.fspath(sock_dir)), "wc-{device_uuid}.sock")


def ready_template(sock_dir: str) -> str:
    """Return the `SGLANG_WEIGHT_CACHE_READY_TEMPLATE` under `sock_dir`."""
    return os.path.join(os.path.abspath(os.fspath(sock_dir)), "wc-{device_uuid}.ready")


def long_socket_path_finding(sock_dir: str, uuids=None) -> dict | None:
    """Return a plain finding if any formatted `AF_UNIX` socket path exceeds `MAX_SOCK_PATH`."""
    sample_uuids = [normalize_uuid(u) for u in uuids if str(u).strip()] if uuids else [_SAMPLE_UUID]
    tmpl = socket_template(sock_dir)
    for uuid in sample_uuids:
        path = tmpl.format(device_uuid=uuid)
        err = ipc.sock_path_error(path)
        if err:
            return harness.plain_finding(
                "weights_socket_path_too_long",
                err,
                f"an AF_UNIX socket path under {ipc.MAX_SOCK_PATH} bytes (pass a short --sock-dir like /tmp/sgl-wc)",
                sock_dir=os.path.abspath(os.fspath(sock_dir)),
                socket_path=path,
                max_sock_path=ipc.MAX_SOCK_PATH,
            )
    return None


def _parse_visible_devices(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def gpu_uuids(gpu_ids=None, *, visible_devices: str | None = None) -> list[str]:
    """Resolve target GPU UUIDs from `gpu_ids`, `visible_devices`, or `CUDA_VISIBLE_DEVICES`."""
    if gpu_ids is None:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
        items = _parse_visible_devices(raw)
        if items is not None and not items:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is empty; no visible GPUs")
    elif isinstance(gpu_ids, str):
        items = _parse_visible_devices(gpu_ids)
        if not items:
            raise RuntimeError("no visible GPU IDs supplied")
    else:
        items = [str(item).strip() for item in gpu_ids if str(item).strip()]
        if not items:
            raise RuntimeError("no visible GPU IDs supplied")

    if items and all(item.startswith(("GPU-", "MIG-")) for item in items):
        uuids = [normalize_uuid(item) for item in items if normalize_uuid(item)]
        if not uuids:
            raise RuntimeError("no valid GPU UUIDs supplied")
        return uuids

    cmd = ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    if items:
        cmd.extend(["-i", ",".join(items)])
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=15)
    uuids = [normalize_uuid(line) for line in out.splitlines() if line.strip()]
    if not uuids:
        raise RuntimeError(f"nvidia-smi returned no GPU UUIDs for {' '.join(cmd)}")
    return uuids


def socket_paths(sock_dir: str, uuids) -> list[str]:
    """Expand per-GPU Unix socket paths under `sock_dir`."""
    tmpl = socket_template(sock_dir)
    return [tmpl.format(device_uuid=normalize_uuid(u)) for u in uuids]


def ready_paths(sock_dir: str, uuids) -> list[str]:
    """Expand per-GPU `.ready` marker paths under `sock_dir`."""
    tmpl = ready_template(sock_dir)
    return [tmpl.format(device_uuid=normalize_uuid(u)) for u in uuids]


def _read_ready_pid(ready_path: str) -> int | None:
    try:
        with open(ready_path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("pid="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    stripped = text.strip()
    if stripped.isdigit():
        return int(stripped)
    return None


def _unix_socket_connectable(sock_path: str, timeout: float = 0.5) -> bool:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(sock_path)
        return True
    except OSError:
        return False
    finally:
        with contextlib.suppress(OSError):
            sock.close()


def probe(sock_dir: str, uuids=None, tp_size: int | None = None) -> dict:
    """Inspect per-GPU `.ready` PIDs and connectable `AF_UNIX` sockets under `sock_dir`."""
    abs_dir = os.path.abspath(os.fspath(sock_dir))
    tmpl_sock = socket_template(abs_dir)
    tmpl_ready = ready_template(abs_dir)
    error = None
    if uuids is None:
        try:
            target_uuids = gpu_uuids()
            if tp_size is not None:
                tp = int(tp_size)
                if tp < 1:
                    raise ValueError(f"tp_size must be >= 1, got {tp_size!r}")
                if len(target_uuids) < tp:
                    raise RuntimeError(
                        f"tp_size={tp} requires {tp} visible GPUs, but only {len(target_uuids)} resolved"
                    )
                target_uuids = target_uuids[:tp]
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            target_uuids = []
            error = f"{type(exc).__name__}: {exc}"
    else:
        target_uuids = [normalize_uuid(u) for u in uuids if normalize_uuid(u)]
        if not target_uuids:
            error = "no GPU UUIDs supplied"

    path_finding = long_socket_path_finding(abs_dir, target_uuids or None)
    socks = socket_paths(abs_dir, target_uuids)
    readies = ready_paths(abs_dir, target_uuids)
    pids = {}
    missing_ready = []
    stale_pids = []
    unconnectable_sockets = []

    for uuid, sock_p, ready_p in zip(target_uuids, socks, readies):
        pid = _read_ready_pid(ready_p)
        if pid is None:
            missing_ready.append(ready_p)
        elif not proc.pid_alive(pid):
            stale_pids.append({"uuid": uuid, "pid": pid, "ready": ready_p})
        else:
            pids[uuid] = pid

        if not _unix_socket_connectable(sock_p):
            unconnectable_sockets.append(sock_p)

    matched = (
        bool(target_uuids)
        and error is None
        and path_finding is None
        and not missing_ready
        and not stale_pids
        and not unconnectable_sockets
    )
    return {
        "ok": matched,
        "matched_gpus": matched,
        "sock_dir": abs_dir,
        "socket_template": tmpl_sock,
        "ready_template": tmpl_ready,
        "uuids": target_uuids,
        "sockets": socks,
        "ready_files": readies,
        "pids": pids,
        "missing_ready": missing_ready,
        "stale_pids": stale_pids,
        "unconnectable_sockets": unconnectable_sockets,
        "missing": unconnectable_sockets,
        "path_finding": path_finding,
        "error": error,
    }


def daemon_finding(probe_result: dict, sock_dir: str | None = None) -> dict | None:
    """Return None when `probe_result` is live for all GPUs, else a plain finding."""
    if probe_result.get("ok"):
        return None
    if probe_result.get("path_finding") is not None:
        return probe_result["path_finding"]

    target_dir = os.path.abspath(os.fspath(sock_dir or probe_result.get("sock_dir") or "."))
    details = []
    if probe_result.get("error"):
        details.append(probe_result["error"])
    if probe_result.get("missing_ready"):
        details.append(f"missing ready files: {probe_result['missing_ready']}")
    if probe_result.get("stale_pids"):
        details.append(f"stale ready pids: {probe_result['stale_pids']}")
    if probe_result.get("unconnectable_sockets"):
        details.append(f"unconnectable sockets: {probe_result['unconnectable_sockets']}")
    observed = "; ".join(details) if details else f"no ready weight-cache daemon under {target_dir}"
    return harness.plain_finding(
        "weights_daemon",
        observed,
        f"live sglang.srt.weight_cache.daemon sockets and .ready PIDs under {target_dir}",
        sock_dir=target_dir,
        weights_probe=probe_result,
    )


def client_env(sock_dir: str) -> dict[str, str]:
    """Environment overrides binding SGLang daemon/client to `sock_dir`."""
    return {
        SOCKET_ENV: socket_template(sock_dir),
        READY_ENV: ready_template(sock_dir),
    }


def client_config(cfg: dict | None) -> dict:
    """Return a shallow copy of SGLang ServerArgs kwargs pinned to `weight_cache_mode='client'`."""
    out = dict(cfg or {})
    out["weight_cache_mode"] = "client"
    return out


def daemon_cmd(model: str, sock_dir: str, *, tp_size: int = 1, python: str | None = None, extra=()) -> list[str]:
    """Build the argv for `sglang.srt.weight_cache.daemon` (socket dir flows through `client_env`)."""
    _ = sock_dir  # validated by start() via long_socket_path_finding and client_env()
    return [
        python or sys.executable,
        "-m",
        "sglang.srt.weight_cache.daemon",
        "--model-path",
        str(model),
        "--tp-size",
        str(int(tp_size)),
        *(str(x) for x in extra),
    ]


def wait_ready(sock_dir: str, uuids, timeout: float = 180.0, poll_interval: float = 0.25, popen=None) -> dict:
    """Poll `probe(sock_dir, uuids)` until all GPUs report live PIDs and sockets."""
    deadline = time.monotonic() + float(timeout)
    last = None
    while True:
        last = probe(sock_dir, uuids=uuids)
        if last.get("ok"):
            return last
        if popen is not None:
            rc = popen.poll()
            if rc is not None:
                raise RuntimeError(
                    f"weight_cache.daemon exited with status {rc} before ready: "
                    f"{json.dumps(last, sort_keys=True)}"
                )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"weight_cache.daemon not ready within {timeout} s: "
                f"{json.dumps(last, sort_keys=True)}"
            )
        time.sleep(poll_interval)


def _unlink_stale(paths) -> None:
    for path in paths:
        with contextlib.suppress(OSError):
            os.unlink(path)


def start(
    model: str,
    sock_dir: str,
    *,
    tp_size: int = 1,
    uuids=None,
    python: str | None = None,
    extra=(),
    env=None,
    log_path: str | None = None,
    timeout: float = 180.0,
    poll_interval: float = 0.25,
) -> dict:
    """Launch `sglang.srt.weight_cache.daemon` and block until every target GPU socket is live."""
    abs_dir = os.path.abspath(os.fspath(sock_dir))
    tp = int(tp_size)
    if tp < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size!r}")
    if uuids is None:
        vis_override = env["CUDA_VISIBLE_DEVICES"] if env and "CUDA_VISIBLE_DEVICES" in env else None
        all_uuids = gpu_uuids(visible_devices=vis_override)
        if len(all_uuids) < tp:
            raise RuntimeError(
                f"tp_size={tp} requires {tp} visible GPUs, but only {len(all_uuids)} resolved"
            )
        target_uuids = all_uuids[:tp]
    else:
        target_uuids = [normalize_uuid(u) for u in uuids if normalize_uuid(u)]
        if not target_uuids:
            raise ValueError("no valid GPU UUIDs supplied")
    path_finding = long_socket_path_finding(abs_dir, target_uuids)
    if path_finding is not None:
        raise ValueError(path_finding["observed"])

    os.makedirs(abs_dir, exist_ok=True)
    _unlink_stale(socket_paths(abs_dir, target_uuids))
    _unlink_stale(ready_paths(abs_dir, target_uuids))

    cenv = client_env(abs_dir)
    proc_env = {**os.environ, **cenv, **(env or {})}
    cmd = daemon_cmd(model, abs_dir, tp_size=tp_size, python=python, extra=extra)

    if log_path:
        log_dir = os.path.dirname(os.path.abspath(log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "ab") as log_fh:
            child = subprocess.Popen(
                cmd,
                env=proc_env,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    else:
        child = subprocess.Popen(
            cmd,
            env=proc_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    try:
        res = wait_ready(
            abs_dir,
            target_uuids,
            timeout=timeout,
            poll_interval=poll_interval,
            popen=child,
        )
    except BaseException:
        stop(child.pid, grace=2.0, reap=child.poll)
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            child.wait(timeout=2.0)
        raise

    return {
        "pid": child.pid,
        "process": child,
        "sock_dir": abs_dir,
        "uuids": target_uuids,
        "cmd": cmd,
        "env": cenv,
        "probe": res,
    }


def stop(pid: int, grace: float = 5.0, *, reap=None) -> bool:
    """Stop a weight-cache daemon session cleanly via SIGTERM -> SIGKILL."""
    return proc.stop_session(int(pid), float(grace), reap=reap)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SGLang native IPC weight-cache daemon helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="start sglang.srt.weight_cache.daemon and wait until ready")
    p_start.add_argument("model", help="model path or HF repo id")
    p_start.add_argument("--sock-dir", required=True, help="short directory for per-GPU sockets and .ready files")
    p_start.add_argument("--tp-size", type=int, default=1)
    p_start.add_argument("--python", default=None)
    p_start.add_argument("--uuids", default=None, help="comma-separated GPU UUID override")
    p_start.add_argument("--timeout", type=float, default=180.0)
    p_start.add_argument("--log", default=None)

    p_probe = sub.add_parser("probe", help="check live ready PIDs and sockets in --sock-dir")
    p_probe.add_argument("--sock-dir", required=True)
    p_probe.add_argument("--tp-size", type=int, default=None)
    p_probe.add_argument("--uuids", default=None, help="comma-separated GPU UUID override")
    return ap


def main(argv=None) -> int:
    ap = _build_parser()
    ns, unknown = ap.parse_known_args(argv)
    if ns.cmd != "start" and unknown:
        ap.error(f"unrecognized arguments: {' '.join(unknown)}")
    uuids = (
        [normalize_uuid(u) for u in ns.uuids.split(",") if normalize_uuid(u)]
        if ns.uuids is not None
        else None
    )

    if ns.cmd == "probe":
        res = probe(ns.sock_dir, uuids=uuids, tp_size=ns.tp_size)
        finding = daemon_finding(res, ns.sock_dir)
        payload = {"weights_probe": res, "findings": [finding] if finding else []}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if res.get("ok") else 2

    if ns.cmd == "start":
        extra = unknown[1:] if unknown and unknown[0] == "--" else list(unknown)
        res = start(
            ns.model,
            ns.sock_dir,
            tp_size=ns.tp_size,
            uuids=uuids,
            python=ns.python,
            extra=extra,
            log_path=ns.log,
            timeout=ns.timeout,
        )
        child = res.pop("process", None)
        if child is not None:
            # Daemon intentionally outlives this CLI process.
            child._child_created = False
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
