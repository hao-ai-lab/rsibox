"""Cross-platform process-tree supervision and TCP port helpers (pure Python stdlib).

Unifies `/proc/<pid>/stat` parsing (with macOS/BSD `ps` fallback), zombie-aware liveness
checks, child-first tree termination, subreaper child reaping, and port probing shared
across engine launchers, conformance harnesses, and zygotes.
"""
import contextlib
import os
import signal
import socket
import subprocess
import time


def stat_ppid(stat_text: str) -> int:
    """Parent PID from `/proc/<pid>/stat` (`pid (comm) state ppid ...`).

    Splits after the last `')'` so process names containing spaces or parentheses parse cleanly.
    """
    return int(stat_text.rsplit(")", 1)[1].split()[1])


def stat_state(stat_text: str) -> str:
    """Single-letter Linux process state (`R`, `S`, `Z`, ...) from `/proc/<pid>/stat`."""
    return stat_text.rsplit(")", 1)[1].split()[0]


def pid_alive(pid: int, *, proc_root: str = "/proc") -> bool:
    """True when `pid` exists and is not a zombie (`Z`)."""
    if not pid or pid <= 0:
        return False
    if proc_root and os.path.isdir(proc_root):
        try:
            with open(os.path.join(proc_root, str(pid), "stat"), encoding="utf-8") as handle:
                return stat_state(handle.read()) != "Z"
        except (OSError, IndexError, ValueError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def ppid_map(*, proc_root: str = "/proc") -> dict[int, int]:
    """Return `{pid: ppid}` for all visible processes (Linux `/proc`, else POSIX `ps`)."""
    out: dict[int, int] = {}
    if proc_root and os.path.isdir(proc_root):
        for entry in os.listdir(proc_root):
            if not entry.isdigit():
                continue
            try:
                with open(os.path.join(proc_root, entry, "stat"), encoding="utf-8") as handle:
                    out[int(entry)] = stat_ppid(handle.read())
            except (OSError, IndexError, ValueError):
                pass
        return out
    try:
        proc = subprocess.Popen(
            ["ps", "-A", "-o", "pid=,ppid="], stdout=subprocess.PIPE, text=True
        )
        stdout, _ = proc.communicate(timeout=10)
    except (OSError, subprocess.SubprocessError):
        return out
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            pid_val = int(parts[0])
            if pid_val != proc.pid:
                out[pid_val] = int(parts[1])
    return out


def direct_children(parent_pid: int | None = None, *, proc_root: str = "/proc") -> list[int]:
    """Sorted PIDs whose parent is `parent_pid` (default: current process)."""
    target = os.getpid() if parent_pid is None else parent_pid
    return sorted(pid for pid, ppid in ppid_map(proc_root=proc_root).items() if ppid == target)


def descendants(root_pid: int, *, proc_root: str = "/proc") -> list[int]:
    """All live descendant PIDs of `root_pid` in breadth-first (parent-before-child) order."""
    kids: dict[int, list[int]] = {}
    for child, parent in ppid_map(proc_root=proc_root).items():
        kids.setdefault(parent, []).append(child)
    found: list[int] = []
    todo = [root_pid]
    seen = {root_pid}
    while todo:
        current = todo.pop(0)
        for child in sorted(kids.get(current, ())):
            if child not in seen:
                seen.add(child)
                found.append(child)
                todo.append(child)
    return found


def kill_tree(
    pid: int,
    sig: int = signal.SIGTERM,
    *,
    include_pg: bool = True,
    proc_root: str = "/proc",
) -> list[int]:
    """Signal `pid`'s descendants (children first), `pid` itself, and optionally its process
    group (safe when `pid` leads its own session via `setsid`). Returns signaled PIDs."""
    targets = list(reversed(descendants(pid, proc_root=proc_root))) + [pid]
    signaled: list[int] = []
    for target in targets:
        try:
            os.kill(target, sig)
            signaled.append(target)
        except (ProcessLookupError, PermissionError):
            pass
    if include_pg:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pid, sig)
    return signaled


def stop_tree(
    pid: int,
    *,
    grace_s: float = 8.0,
    kill_grace_s: float = 5.0,
    include_pg: bool = False,
    proc_root: str = "/proc",
) -> bool:
    """Gracefully stop `pid` and all descendants (`SIGTERM`, then `SIGKILL` after `grace_s`).

    Returns True when every process in the tree has exited.
    """
    if not pid_alive(pid, proc_root=proc_root):
        return True
    tree = [pid] + descendants(pid, proc_root=proc_root)
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, kill_grace_s)):
        if sig == signal.SIGTERM:
            for target in list(reversed(tree)):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(target, sig)
            if include_pg:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(pid, sig)
        else:
            all_targets = list(dict.fromkeys(
                list(reversed(tree + descendants(pid, proc_root=proc_root)))
            ))
            for target in all_targets:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(target, sig)
            if include_pg:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(pid, sig)
            tree = all_targets
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            # Reap if any target is our direct child so zombies disappear promptly.
            with contextlib.suppress(ChildProcessError):
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            if not any(pid_alive(t, proc_root=proc_root) for t in tree):
                return True
            time.sleep(0.03)
    return not any(pid_alive(t, proc_root=proc_root) for t in tree)


def reap_all(
    timeout: float = 10.0,
    *,
    keep: tuple[int, ...] | set[int] = (),
    proc_root: str = "/proc",
) -> list[int]:
    """Kill and reap all direct children of this process except PIDs in `keep`."""
    keep_set = set(keep)
    reaped: list[int] = []
    killed: set[int] = set()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        left = [c for c in direct_children(proc_root=proc_root) if c not in keep_set]
        for child in left:
            if child not in killed:
                kill_tree(child, signal.SIGKILL, include_pg=True, proc_root=proc_root)
                killed.add(child)
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        if pid:
            reaped.append(pid)
        elif not left:
            return reaped
        else:
            time.sleep(0.02)
    return reaped


def free_port(host: str = "127.0.0.1") -> int:
    """Bind an ephemeral TCP port on `host` and return the port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def port_in_use(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """True when a listener accepts TCP connections on `host:port`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        try:
            return probe.connect_ex((host, int(port))) == 0
        except OSError:
            return False
