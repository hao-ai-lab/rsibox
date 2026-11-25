"""Stdlib JSON IPC transports shared across live-lab hosts, clients, and zygotes.

Unifies the three transport shapes used across rsibox engines:
- **Unix domain sockets** (`UnixJsonServer`, `unix_json_call`, `unix_json_stream`):
  newline-delimited JSON over `AF_UNIX` with platform-aware `MAX_SOCK_PATH` validation
  (`sock_path_error`, `short_sock_path`).
- **File control channels** (`FileChannelServer`, `file_channel_call`): atomic `ctl.json`
  plus `reply/<seq>.json` polling with PID/thread-scoped temp files.
- **NFS-safe directory mailboxes** (`MailboxServer`, `mailbox_call`): atomic `req-<id>.json`
  claimed via `.taken` rename and answered via `resp-<id>.json` with `os.listdir` cache
  revalidation.
"""
import contextlib
import errno
import json
import math
import os
import queue
import selectors
import socket
import sys
import tempfile
import threading
import time
import uuid

# sun_path is 104 bytes on macOS/BSD and 108 bytes on Linux (including trailing NUL).
MAX_SOCK_PATH = 103 if sys.platform == "darwin" else 107


def sock_path_error(sock_path: str) -> str | None:
    """Why an `AF_UNIX` socket cannot be bound at `sock_path`, or None when it fits."""
    abs_path = os.path.abspath(sock_path)
    length = len(os.fsencode(abs_path))
    if length > MAX_SOCK_PATH:
        return (
            f"socket path too long for AF_UNIX ({length} bytes > {MAX_SOCK_PATH}): {abs_path}; "
            "use a short path such as /tmp/rsibox-<id>.sock"
        )
    return None


def short_sock_path(prefix: str = "rsibox", suffix: str = ".sock", root: str | None = None) -> str:
    """Return a unique, guaranteed-short `AF_UNIX` socket path under `/tmp` (or `root`)."""
    base = root or ("/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir())
    path = os.path.join(base, f"{prefix}-{os.getpid()}-{uuid.uuid4().hex[:8]}{suffix}")
    err = sock_path_error(path)
    if err:
        raise ValueError(err)
    return path


def atomic_write_json(path: str | os.PathLike, payload, *, indent: int | None = None) -> None:
    """Atomically write JSON to `path` via a PID+tid-scoped sibling temp file and `os.replace`."""
    target = os.fspath(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=str)
    os.replace(tmp, target)


def read_json(path: str | os.PathLike, default=None):
    """Read JSON from `path`, returning `default` on missing/unreadable/malformed files."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


# -- Unix domain socket JSON-line server and client --------------------------------------


class UnixJsonServer:
    """Newline-delimited JSON server over an `AF_UNIX` stream socket.

    Two serving modes:
    - **Callback mode** (`handler` given): each valid JSON object calls `handler(req) -> dict`
      on the connection thread and writes the JSON reply line back.
    - **Queue mode** (`handler=None`): queues `(req, reply_fn)` tuples for a main-thread loop
      to drain via `next(timeout=...)` (e.g. a step-gap loop). Calling `reply_fn(obj)` sends
      the reply and returns `False` without raising if the client already disconnected.
    """

    def __init__(
        self,
        sock_path: str,
        handler=None,
        *,
        backlog: int = 16,
        mode: int = 0o600,
        read_timeout_s: float | None = 30.0,
        start: bool = True,
    ):
        err = sock_path_error(sock_path)
        if err:
            raise ValueError(err)
        self.path = os.path.abspath(sock_path)
        self._handler = handler
        self._read_timeout_s = read_timeout_s
        self.q: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        if mode is not None:
            with contextlib.suppress(OSError):
                os.chmod(self.path, mode)
        self.sock.listen(backlog)
        self._thread = None
        if start:
            self.start()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._accept_loop, daemon=True, name="rsibox-unix-server"
            )
            self._thread.start()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve_one, args=(conn,), daemon=True, name="rsibox-unix-conn"
            ).start()

    def _serve_one(self, conn: socket.socket) -> None:
        done = threading.Event()
        try:
            conn.settimeout(self._read_timeout_s)
            stream = conn.makefile("rwb")
        except OSError:
            with contextlib.suppress(OSError):
                conn.close()
            return

        def close_conn():
            for resource in (stream, conn):
                with contextlib.suppress(OSError):
                    resource.close()

        def reply(obj) -> bool:
            try:
                stream.write((json.dumps(obj, default=str) + "\n").encode("utf-8"))
                stream.flush()
                return True
            except (OSError, ValueError) as exc:
                print(
                    f"[rsibox-ipc] client gone before reply ({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            finally:
                close_conn()
                done.set()

        try:
            line = stream.readline()
            conn.settimeout(None)
        except OSError:
            close_conn()
            return
        if not line:
            close_conn()
            return
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError(f"a request must be a JSON object, not {type(req).__name__}")
        except ValueError as exc:
            reply({"ok": False, "error": f"bad request: {exc}"})
            return

        if self._handler is not None:
            try:
                out = self._handler(req)
            except Exception as exc:
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            reply(out)
            return

        self.q.put((req, reply))
        done.wait()

    def next(self, timeout: float | None = None):
        """Pop the next `(req, reply_fn)` tuple in queue mode."""
        return self.q.get(timeout=timeout)

    def close(self) -> None:
        self._closed.set()
        with contextlib.suppress(OSError):
            self.sock.close()
        with contextlib.suppress(OSError):
            os.unlink(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


_MAX_WAIT_SLICE_S = 3600.0


def _validate_liveness_args(alive_fn, poll_s: float) -> float:
    if alive_fn is not None and not callable(alive_fn):
        raise TypeError(f"alive_fn must be callable or None, got {type(alive_fn).__name__}")
    if isinstance(poll_s, bool) or not isinstance(poll_s, (int, float)):
        raise ValueError(f"poll_s must be a finite number >= 0, got {poll_s!r}")
    try:
        poll_f = float(poll_s)
    except OverflowError as exc:
        raise ValueError(f"poll_s must be finite, got {poll_s!r}") from exc
    if not math.isfinite(poll_f) or poll_f < 0.0:
        raise ValueError(f"poll_s must be a finite number >= 0, got {poll_s!r}")
    return poll_f


def _validate_unix_timeout(timeout: float | None, label: str) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"timeout must be a number or None, got {type(timeout).__name__}")
    try:
        timeout_f = float(timeout)
    except OverflowError as exc:
        raise ValueError(f"timeout must be finite or inf, got {timeout!r}") from exc
    if math.isnan(timeout_f):
        raise TimeoutError(f"no answer to {label} in {timeout} s")
    if timeout_f < 0.0:
        raise ValueError(f"timeout must be >= 0, got {timeout!r}")
    if math.isinf(timeout_f):
        return None
    return timeout_f


def _wait_socket_event(
    sel: selectors.BaseSelector,
    *,
    label: str,
    timeout_orig: float | None,
    deadline: float | None,
    poll_s: float,
    alive_fn,
    phase: str,
) -> bool:
    """Wait until registered socket event is ready, polling `alive_fn` and enforcing `deadline`.

    Returns True when the socket became ready; raises TimeoutError or ConnectionError otherwise.
    """
    while True:
        remaining = None if deadline is None else (deadline - time.monotonic())
        if remaining is not None and not (remaining > 0.0):
            if sel.select(0.0):
                return True
            raise TimeoutError(f"no answer to {label} in {timeout_orig} s")
        if alive_fn is not None:
            wait_s = poll_s if remaining is None else min(poll_s, remaining)
        else:
            wait_s = _MAX_WAIT_SLICE_S if remaining is None else remaining
        wait_s = min(wait_s, _MAX_WAIT_SLICE_S)
        if sel.select(wait_s):
            return True
        if alive_fn is not None and not alive_fn():
            if sel.select(0.0):
                return True
            raise ConnectionError(f"peer process exited before {phase} {label}")


def _connect_and_send_unix(
    conn: socket.socket,
    sock_path: str,
    payload: bytes,
    *,
    label: str,
    timeout_orig: float | None,
    deadline: float | None,
    poll_s: float,
    alive_fn,
) -> None:
    """Non-blocking AF_UNIX connect + chunked send polling `alive_fn` via DefaultSelector."""
    conn.setblocking(False)
    with selectors.DefaultSelector() as sel:
        sel.register(conn, selectors.EVENT_WRITE)
        try:
            conn.connect(sock_path)
        except BlockingIOError:
            _wait_socket_event(
                sel, label=label, timeout_orig=timeout_orig, deadline=deadline,
                poll_s=poll_s, alive_fn=alive_fn, phase="sending",
            )
            so_err = conn.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if so_err:
                raise OSError(so_err, os.strerror(so_err))
        except OSError as exc:
            if exc.errno in (errno.EINPROGRESS, errno.EAGAIN, errno.EWOULDBLOCK):
                _wait_socket_event(
                    sel, label=label, timeout_orig=timeout_orig, deadline=deadline,
                    poll_s=poll_s, alive_fn=alive_fn, phase="sending",
                )
                so_err = conn.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if so_err:
                    raise OSError(so_err, os.strerror(so_err))
            else:
                raise

        view = memoryview(payload)
        sent = 0
        total = len(payload)
        while sent < total:
            try:
                n = conn.send(view[sent:])
                if n == 0:
                    raise ConnectionError(f"server closed the connection during {label}")
                sent += n
                continue
            except BlockingIOError:
                pass
            _wait_socket_event(
                sel, label=label, timeout_orig=timeout_orig, deadline=deadline,
                poll_s=poll_s, alive_fn=alive_fn, phase="sending",
            )


def _iter_unix_json_lines(
    conn: socket.socket,
    *,
    label: str,
    timeout_orig: float | None,
    timeout_s: float | None,
    deadline: float | None,
    poll_s: float,
    alive_fn,
    reset_deadline_on_line: bool = False,
):
    """Yield complete newline-delimited byte lines from `conn` using DefaultSelector."""
    conn.setblocking(False)
    buf = bytearray()
    with selectors.DefaultSelector() as sel:
        sel.register(conn, selectors.EVENT_READ)
        while True:
            nl = buf.find(b"\n")
            if nl != -1:
                line = bytes(buf[:nl])
                del buf[: nl + 1]
                if reset_deadline_on_line and timeout_s is not None:
                    deadline = time.monotonic() + timeout_s
                yield line
                continue

            remaining = None if deadline is None else (deadline - time.monotonic())
            if remaining is not None and not (remaining > 0.0):
                if sel.select(0.0):
                    try:
                        chunk = conn.recv(65536)
                    except BlockingIOError:
                        chunk = None
                    if chunk:
                        buf.extend(chunk)
                        continue
                    if chunk == b"":
                        if buf:
                            line = bytes(buf)
                            buf.clear()
                            yield line
                        return
                raise TimeoutError(f"no answer to {label} in {timeout_orig} s")

            if alive_fn is not None:
                wait_s = poll_s if remaining is None else min(poll_s, remaining)
            else:
                wait_s = _MAX_WAIT_SLICE_S if remaining is None else remaining
            wait_s = min(wait_s, _MAX_WAIT_SLICE_S)

            if sel.select(wait_s):
                try:
                    chunk = conn.recv(65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    if buf:
                        line = bytes(buf)
                        buf.clear()
                        yield line
                    return
                buf.extend(chunk)
                continue

            if alive_fn is not None and not alive_fn():
                eof = False
                while sel.select(0.0):
                    try:
                        chunk = conn.recv(65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        eof = True
                        break
                    buf.extend(chunk)
                    if b"\n" in buf:
                        break
                nl = buf.find(b"\n")
                if nl != -1:
                    line = bytes(buf[:nl])
                    del buf[: nl + 1]
                    if reset_deadline_on_line and timeout_s is not None:
                        deadline = time.monotonic() + timeout_s
                    yield line
                    continue
                if eof:
                    if buf:
                        line = bytes(buf)
                        buf.clear()
                        yield line
                    return
                raise ConnectionError(f"peer process exited before answering {label}")


def unix_json_call(
    sock_path: str,
    req: dict,
    timeout: float | None = 60.0,
    *,
    poll_s: float = 0.05,
    alive_fn=None,
) -> dict:
    """Send one JSON request line to `sock_path` and return the decoded JSON reply object.

    Optional `alive_fn` (`() -> bool`) is checked before connecting and every `poll_s` across
    connect, send, and receive so a SIGKILLed peer fails fast with `ConnectionError` even when
    a forked grandchild still holds the inherited socket FD open.
    """
    poll_f = _validate_liveness_args(alive_fn, poll_s)
    label = req.get("verb") or req.get("op") or req.get("cmd") or req.get("method") or "request"
    timeout_s = _validate_unix_timeout(timeout, label)
    if alive_fn is not None and not alive_fn():
        raise ConnectionError(f"peer process exited before sending {label}")
    payload = (json.dumps(req, default=str) + "\n").encode("utf-8")
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    use_selector = alive_fn is not None or timeout_s == 0.0 or (timeout_s is not None and timeout_s > _MAX_WAIT_SLICE_S)
    try:
        if not use_selector:
            conn.settimeout(timeout_s)
            conn.connect(sock_path)
            conn.sendall(payload)
            with conn.makefile("rb") as stream:
                line = stream.readline()
        else:
            deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
            _connect_and_send_unix(
                conn, sock_path, payload, label=label, timeout_orig=timeout,
                deadline=deadline, poll_s=poll_f, alive_fn=alive_fn,
            )
            line = next(
                _iter_unix_json_lines(
                    conn, label=label, timeout_orig=timeout, timeout_s=timeout_s,
                    deadline=deadline, poll_s=poll_f, alive_fn=alive_fn,
                ),
                b"",
            )
    except socket.timeout as exc:
        raise TimeoutError(f"no answer to {label} in {timeout} s") from exc
    finally:
        with contextlib.suppress(OSError):
            conn.close()
    if not line:
        raise ConnectionError(f"server closed the connection during {label}")
    return json.loads(line)


def unix_json_stream(
    sock_path: str,
    req: dict,
    *,
    timeout: float | None = None,
    stop_events: tuple[str, ...] = ("exited", "refused", "error", "stopped", "pong"),
    poll_s: float = 0.05,
    alive_fn=None,
):
    """Send one JSON request line and yield streamed JSON event objects until a terminal event.

    `timeout`, when set, acts as a per-event inactivity timeout across both `alive_fn=None`
    and `alive_fn!=None` paths. Argument validation and pre-connect liveness checks run eagerly.
    """
    poll_f = _validate_liveness_args(alive_fn, poll_s)
    label = req.get("verb") or req.get("op") or req.get("cmd") or req.get("method") or "request"
    timeout_s = _validate_unix_timeout(timeout, label)
    if alive_fn is not None and not alive_fn():
        raise ConnectionError(f"peer process exited before sending {label}")
    payload = (json.dumps(req, default=str) + "\n").encode("utf-8")
    use_selector = alive_fn is not None or timeout_s == 0.0 or (timeout_s is not None and timeout_s > _MAX_WAIT_SLICE_S)

    def _stream():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            if not use_selector:
                conn.settimeout(timeout_s)
                try:
                    conn.connect(sock_path)
                    conn.sendall(payload)
                    with conn.makefile("rb") as stream:
                        for line in stream:
                            event = json.loads(line)
                            yield event
                            if isinstance(event, dict) and event.get("event") in stop_events:
                                break
                except socket.timeout as exc:
                    raise TimeoutError(f"no answer to {label} in {timeout} s") from exc
            else:
                deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
                _connect_and_send_unix(
                    conn, sock_path, payload, label=label, timeout_orig=timeout,
                    deadline=deadline, poll_s=poll_f, alive_fn=alive_fn,
                )
                read_deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
                for line in _iter_unix_json_lines(
                    conn, label=label, timeout_orig=timeout, timeout_s=timeout_s,
                    deadline=read_deadline, poll_s=poll_f, alive_fn=alive_fn,
                    reset_deadline_on_line=True,
                ):
                    event = json.loads(line)
                    yield event
                    if isinstance(event, dict) and event.get("event") in stop_events:
                        break

    return _stream()


# -- File control channel & NFS-safe directory mailbox -----------------------------------


class _PollingServer:
    """Shared daemon-thread lifecycle for polling IPC servers."""

    _thread_name = "rsibox-poll"

    def _init_polling(self, poll_s: float, start: bool) -> None:
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread = None
        if start:
            self.start()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name=self._thread_name)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.stop()


class FileChannelServer(_PollingServer):
    """Polling server over `<root>/ctl.json` and `<root>/reply/<seq>.json`."""

    _thread_name = "rsibox-file-channel"

    def __init__(
        self,
        root: str | os.PathLike,
        handler,
        *,
        poll_s: float = 0.005,
        ctl_name: str = "ctl.json",
        reply_dir: str = "reply",
        start: bool = True,
    ):
        self.root = os.fspath(root)
        self._handler = handler
        self.ctl_path = os.path.join(self.root, ctl_name)
        self.reply_dir = os.path.join(self.root, reply_dir)
        os.makedirs(self.reply_dir, exist_ok=True)
        self._last = 0
        self._init_polling(poll_s, start)

    def _run(self) -> None:
        while not self._stop.is_set():
            cmd = read_json(self.ctl_path)
            if isinstance(cmd, dict):
                try:
                    seq = int(cmd.get("seq", 0))
                except (TypeError, ValueError):
                    seq = 0
                if seq > self._last:
                    self._last = seq
                    try:
                        reply = self._handler(cmd)
                    except Exception as exc:
                        reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    atomic_write_json(os.path.join(self.reply_dir, f"{seq}.json"), reply)
            self._stop.wait(self.poll_s)


def file_channel_call(
    root: str | os.PathLike,
    cmd: dict,
    timeout: float = 300.0,
    *,
    poll_s: float = 0.005,
    ctl_name: str = "ctl.json",
    reply_dir: str = "reply",
    min_seq: int | None = None,
    alive_fn=None,
) -> dict:
    """Send one command over `<root>/ctl.json` and wait for `<root>/reply/<seq>.json`.

    Optional `min_seq` guarantees `seq > min_seq` (used after CRIU/snapshot restore).
    Optional `alive_fn` (`() -> bool`) is checked on every poll tick so a crashed/SIGKILLed
    peer fails fast with `lost=True` instead of stalling for the full `timeout`.
    """
    if min_seq is not None:
        if isinstance(min_seq, bool) or not isinstance(min_seq, int):
            raise TypeError(f"min_seq must be an int or None, got {type(min_seq).__name__}")
        if min_seq < 0:
            raise ValueError(f"min_seq must be >= 0, got {min_seq}")
    if alive_fn is not None and not callable(alive_fn):
        raise TypeError(f"alive_fn must be callable or None, got {type(alive_fn).__name__}")
    if isinstance(poll_s, bool) or not isinstance(poll_s, (int, float)) or not (0.0 <= poll_s < float("inf")):
        raise ValueError(f"poll_s must be a finite number >= 0, got {poll_s!r}")
    root_str = os.fspath(root)
    ctl_path = os.path.join(root_str, ctl_name)
    rdir = os.path.join(root_str, reply_dir)
    prev = read_json(ctl_path)
    raw_prev_seq = prev.get("seq") if isinstance(prev, dict) else None
    prev_seq = raw_prev_seq if isinstance(raw_prev_seq, int) and not isinstance(raw_prev_seq, bool) else 0
    existing = [
        int(name[:-5])
        for name in (os.listdir(rdir) if os.path.isdir(rdir) else ())
        if name.endswith(".json") and name[:-5].isdigit()
    ]
    floor_seq = max(prev_seq, max(existing, default=0), min_seq if min_seq is not None else 0)
    seq = floor_seq + 1
    t0 = time.perf_counter()
    if alive_fn is not None and not alive_fn():
        return {
            "ok": False,
            "error": "peer process exited before send",
            "seq": seq,
            "lost": True,
            "client_round_trip_s": round(time.perf_counter() - t0, 6),
        }
    payload = {**cmd, "seq": seq}
    atomic_write_json(ctl_path, payload)
    reply_path = os.path.join(rdir, f"{seq}.json")
    deadline = t0 + timeout
    while True:
        reply = read_json(reply_path)
        if isinstance(reply, dict):
            reply["client_round_trip_s"] = round(time.perf_counter() - t0, 6)
            return reply
        if alive_fn is not None and not alive_fn():
            final_reply = read_json(reply_path)
            if isinstance(final_reply, dict):
                final_reply["client_round_trip_s"] = round(time.perf_counter() - t0, 6)
                return final_reply
            return {
                "ok": False,
                "error": "peer process exited before reply",
                "seq": seq,
                "lost": True,
                "client_round_trip_s": round(time.perf_counter() - t0, 6),
            }
        remaining = deadline - time.perf_counter()
        if not (remaining > 0.0):
            break
        time.sleep(min(poll_s, remaining))
    return {
        "ok": False,
        "error": f"no reply in {timeout} s",
        "seq": seq,
        "lost": True,
        "client_round_trip_s": round(time.perf_counter() - t0, 6),
    }


# -- NFS-safe directory mailbox (`req-<id>.json` -> `resp-<id>.json`) --------------------


class MailboxServer(_PollingServer):
    """NFS-safe directory mailbox server (`req-<id>.json` -> `.taken` -> `resp-<id>.json`).

    Accepts either a `queue.Queue` (queuing `(req, reply_fn)` like `UnixJsonServer`) or a
    callable `handler(req) -> dict`.
    """

    _thread_name = "rsibox-mailbox"

    def __init__(
        self,
        root: str | os.PathLike,
        sink,
        *,
        poll_s: float = 0.02,
        start: bool = True,
    ):
        self.root = os.fspath(root)
        self._sink = sink
        os.makedirs(self.root, exist_ok=True)
        self._init_polling(poll_s, start)

    def _replier(self, rid: str):
        def reply(obj) -> bool:
            try:
                atomic_write_json(os.path.join(self.root, f"resp-{rid}.json"), obj)
                return True
            except OSError as exc:
                print(
                    f"[rsibox-ipc] mailbox reply {rid} not written ({exc})",
                    file=sys.stderr,
                    flush=True,
                )
                return False
        return reply

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                names = sorted(
                    n for n in os.listdir(self.root)
                    if n.startswith("req-") and n.endswith(".json")
                )
            except OSError:
                names = []
            for name in names:
                rid = name[4:-5]
                src = os.path.join(self.root, name)
                taken = os.path.join(self.root, name + ".taken")
                try:
                    os.rename(src, taken)
                except OSError:
                    continue
                try:
                    with open(taken, encoding="utf-8") as handle:
                        req = json.load(handle)
                except (OSError, ValueError) as exc:
                    req = {"verb": "bad", "error": str(exc)}
                replier = self._replier(rid)
                if callable(self._sink):
                    try:
                        out = self._sink(req)
                    except Exception as exc:
                        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    replier(out)
                else:
                    self._sink.put((req, replier))
            self._stop.wait(self.poll_s)


def mailbox_call(
    root: str | os.PathLike,
    req: dict,
    timeout: float = 600.0,
    *,
    poll_s: float = 0.02,
    alive_fn=None,
) -> dict:
    """Send one JSON request through a `MailboxServer` directory and wait for its reply."""
    if alive_fn is not None and not callable(alive_fn):
        raise TypeError(f"alive_fn must be callable or None, got {type(alive_fn).__name__}")
    if isinstance(poll_s, bool) or not isinstance(poll_s, (int, float)) or not (0.0 <= poll_s < float("inf")):
        raise ValueError(f"poll_s must be a finite number >= 0, got {poll_s!r}")
    label = req.get("verb") or req.get("op") or "request"
    if alive_fn is not None and not alive_fn():
        raise ConnectionError(f"peer process exited before sending {label}")
    root_str = os.fspath(root)
    os.makedirs(root_str, exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    req_path = os.path.join(root_str, f"req-{rid}.json")
    resp_path = os.path.join(root_str, f"resp-{rid}.json")
    atomic_write_json(req_path, req)
    deadline = time.monotonic() + timeout
    while True:
        with contextlib.suppress(OSError):
            os.listdir(root_str)  # revalidate directory attribute cache on NFS
        reply = read_json(resp_path)
        if reply is not None:
            with contextlib.suppress(OSError):
                os.unlink(resp_path)
            return reply
        if alive_fn is not None and not alive_fn():
            with contextlib.suppress(OSError):
                os.listdir(root_str)  # revalidate NFS dentry cache before final reply check
            final_reply = read_json(resp_path)
            if final_reply is not None:
                with contextlib.suppress(OSError):
                    os.unlink(resp_path)
                return final_reply
            with contextlib.suppress(OSError):
                os.unlink(req_path)
            raise ConnectionError(f"peer process exited before answering {label}")
        remaining = deadline - time.monotonic()
        if not (remaining > 0.0):
            break
        time.sleep(min(poll_s, remaining))
    raise TimeoutError(f"no answer to {label} in {timeout} s")
