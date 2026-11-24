"""Cross-process Unix-domain-socket daemon (`PoolServer`) and client (`PoolClient`) for `SandboxPool`.

Multiplexes independent OS processes over one warm `SandboxPool` using persistent
newline-delimited JSON connections over `AF_UNIX`. Every lease carries a fenced token bound
to its connection; abrupt client disconnect/crash triggers immediate server-side cleanup.
"""
from __future__ import annotations

import argparse, contextlib, json, operator, os, secrets, signal, socket, sys, threading
from typing import Any

from . import box as _box
from .ipc import sock_path_error
from .pool import (
    DEFAULT_PRIORITY, PoolClosedError, PoolError, PoolSlot, PoolTimeoutError,
    SandboxPool, SlotLease, _format_error, _validate_timeout,
)

__all__ = ["PoolClient", "PoolServer", "RemoteLease", "cli_main"]
_ERROR_MAP: dict[str, type[Exception]] = {
    "PoolTimeoutError": PoolTimeoutError, "PoolClosedError": PoolClosedError,
    "PoolError": PoolError, "ValueError": ValueError, "TypeError": TypeError,
}

def _write_json_line(stream: Any, payload: dict[str, Any]) -> None:
    stream.write((json.dumps(payload, default=str) + "\n").encode("utf-8"))
    stream.flush()

def _read_json_line(stream: Any) -> dict[str, Any] | None:
    raw = stream.readline()
    if not raw:
        return None
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object, got {type(obj).__name__}")
    return obj

def _raise_if_error(reply: dict[str, Any]) -> dict[str, Any]:
    error_type = reply.get("error_type")
    if not reply.get("ok") and isinstance(error_type, str) and error_type in _ERROR_MAP:
        raise _ERROR_MAP[error_type](str(reply.get("error") or error_type))
    return reply

def _close_resources(*resources: Any) -> None:
    for res in resources:
        if res is not None:
            with contextlib.suppress(OSError):
                res.close()

def _read_lab_source(lab: Any, module: str) -> dict[str, Any]:
    if not isinstance(module, str) or not module:
        return {"ok": False, "error": "module is required"}
    text: Any = None
    for attr in ("source", "_live_source", "_base_source"):
        if text is None and callable(fn := getattr(lab, attr, None)):
            got = fn(module)
            if isinstance(got, dict): return got
            text = got
        if text is None and attr == "_live_source" and (stack := getattr(lab, "live_text", None)):
            text = stack[-1].get(module)
    if text is None:
        text = (getattr(lab, "base_text", None) or {}).get(module)
    if not isinstance(text, str):
        return {"ok": False, "module": module, "error": "the lab does not serve this module's text"}
    return {"ok": True, "module": module, "text": text, "hash": _box.text_hash(text)}

def _dispatch_verb(lease: SlotLease, op: str, req: dict[str, Any]) -> dict[str, Any]:
    lab = lease.lab
    if op == "install":
        patch = req.get("patch")
        return lab.install(patch) if isinstance(patch, dict) else {"ok": False, "error": "install requires a patch dict"}
    if op == "check": return lab.check(req.get("workload"))
    if op == "reset": return lab.reset(req.get("kind", "state"))
    if op == "uninstall": return lab.uninstall(req.get("id"))
    if op == "status": return lab.status()
    if op == "source": return _read_lab_source(lab, req.get("module", ""))
    if op == "receipts":
        return fn() if callable(fn := getattr(lab, "receipts", None)) else {"ok": False, "error": "lab does not expose receipts()"}
    return {"ok": False, "error": f"unknown op {op!r}"}

class PoolServer:
    """Multi-threaded `AF_UNIX` JSON-line RPC daemon exposing a `SandboxPool` across OS processes."""

    def __init__(self, pool: SandboxPool, sock_path: str, *, backlog: int = 64,
                 mode: int = 0o600, start: bool = True, close_pool: bool = False) -> None:
        if (err := sock_path_error(sock_path)): raise ValueError(err)
        self.pool, self.sock_path, self._close_pool = pool, os.path.abspath(sock_path), close_pool
        self._closed, self._state_lock = threading.Event(), threading.Lock()
        self._active_leases: dict[str, SlotLease] = {}
        self._conns: set[socket.socket] = set()
        self._threads: set[threading.Thread] = set()
        self._accept_thread: threading.Thread | None = None
        if (parent := os.path.dirname(self.sock_path)): os.makedirs(parent, exist_ok=True)
        with contextlib.suppress(FileNotFoundError): os.unlink(self.sock_path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.bind(self.sock_path)
            if mode is not None:
                with contextlib.suppress(OSError): os.chmod(self.sock_path, mode)
            self._sock.listen(backlog)
            self._sock.settimeout(0.1)
        except BaseException:
            self._sock.close()
            raise
        if start: self.start()

    def start(self) -> PoolServer:
        if self._accept_thread is None:
            self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="rsibox-pool-accept")
            self._accept_thread.start()
        return self

    def stats(self) -> dict[str, Any]:
        metrics = self.pool.metrics()
        with self._state_lock: n_leases, n_conns = len(self._active_leases), len(self._conns)
        return {**metrics, "active_leases": n_leases, "active_connections": n_conns, "sock_path": self.sock_path}

    def health(self) -> dict[str, Any]:
        info = self.pool.health()
        with self._state_lock: n_leases, n_conns = len(self._active_leases), len(self._conns)
        return {**info, "ok": bool(info["ok"]) and not self._closed.is_set(),
                "active_leases": n_leases, "active_connections": n_conns}

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try: conn, _ = self._sock.accept()
            except socket.timeout: continue
            except OSError: return
            if self._closed.is_set():
                _close_resources(conn)
                return
            thread = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True, name="rsibox-pool-conn")
            with self._state_lock:
                self._conns.add(conn)
                self._threads.add(thread)
            thread.start()

    def _release_conn_lease(self, conn_leases: dict[str, SlotLease], token: str, reason: str) -> None:
        lease = conn_leases.pop(token, None)
        with self._state_lock: self._active_leases.pop(token, None)
        if lease is not None:
            with contextlib.suppress(Exception): lease.release(error=reason)

    def _serve_conn(self, conn: socket.socket) -> None:
        conn_leases: dict[str, SlotLease] = {}
        stream = None
        try:
            stream = conn.makefile("rwb")
            while not self._closed.is_set():
                try: req = _read_json_line(stream)
                except ValueError as exc:
                    _write_json_line(stream, {"ok": False, "error": f"bad request: {exc}", "error_type": "ValueError"})
                    continue
                except OSError: break
                if req is None: break
                reply, new_token = self._handle_request(req, conn_leases)
                try: _write_json_line(stream, reply)
                except OSError:
                    if new_token is not None:
                        self._release_conn_lease(conn_leases, new_token, "client disconnected during acquire reply")
                    break
        finally:
            for token in list(conn_leases):
                self._release_conn_lease(conn_leases, token, "client disconnected")
            _close_resources(stream, conn)
            with self._state_lock:
                self._conns.discard(conn)
                self._threads.discard(threading.current_thread())

    def _handle_request(self, req: dict[str, Any], conn_leases: dict[str, SlotLease]) -> tuple[dict[str, Any], str | None]:
        op = req.get("op")
        if op in ("stats", "metrics"): return {"ok": True, **self.stats()}, None
        if op == "health": return self.health(), None
        if op == "acquire":
            try:
                lease = self.pool.acquire(
                    req.get("agent_id", ""), role=req.get("role"), idea_id=req.get("idea_id"),
                    timeout_s=req.get("timeout_s"), priority=req.get("priority", DEFAULT_PRIORITY),
                )
            except (PoolTimeoutError, PoolClosedError, PoolError, ValueError, TypeError) as exc:
                return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}, None
            token = secrets.token_urlsafe(18)
            conn_leases[token] = lease
            with self._state_lock: self._active_leases[token] = lease
            return {
                "ok": True, "token": token, "slot_id": lease.slot_id, "slot_role": lease.slot.role,
                "role": lease.role, "agent_id": lease.agent_id, "idea_id": lease.idea_id,
                "priority": lease.priority, "acquired_at": lease.acquired_at, "queue_wait_s": lease.queue_wait_s,
            }, token
        token = req.get("token")
        lease = conn_leases.get(token) if isinstance(token, str) else None
        if lease is None:
            return {"ok": False, "error": "invalid or released lease token", "error_type": "PoolError"}, None
        if op == "release":
            conn_leases.pop(token, None)
            with self._state_lock: self._active_leases.pop(token, None)
            try: return {"ok": True, "record": lease.release(error=req.get("error"))}, None
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__}, None
        try: reply = _dispatch_verb(lease, str(op), req)
        except PoolError as exc: return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}, None
        except Exception as exc: return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None
        return (reply if isinstance(reply, dict) else {"ok": False, "error": f"bad reply {type(reply).__name__}"}), None

    def wait(self) -> None:
        while not self._closed.is_set(): self._closed.wait(timeout=0.2)

    def close(self, *, close_pool: bool | None = None) -> None:
        should_close_pool = self._close_pool if close_pool is None else close_pool
        self._closed.set()
        with contextlib.suppress(OSError): self._sock.shutdown(socket.SHUT_RDWR)
        _close_resources(self._sock)
        with self._state_lock: conns, threads = list(self._conns), list(self._threads)
        for conn in conns:
            with contextlib.suppress(OSError): conn.shutdown(socket.SHUT_RDWR)
            _close_resources(conn)
        if should_close_pool: self.pool.close()
        cur = threading.current_thread()
        if self._accept_thread is not None and self._accept_thread is not cur:
            self._accept_thread.join(timeout=2.0)
        for thread in threads:
            if thread is not cur: thread.join(timeout=5.0)
        with contextlib.suppress(OSError): os.unlink(self.sock_path)

    def __enter__(self) -> PoolServer: return self
    def __exit__(self, *_exc: Any) -> bool:
        self.close()
        return False

class RemoteLease:
    """Client-side fenced lease over a persistent `PoolServer` connection implementing the 5-verb Lab API."""

    slot_id, slot_role = property(operator.attrgetter("_slot_id")), property(operator.attrgetter("_slot_role"))
    role, agent_id = property(operator.attrgetter("_role")), property(operator.attrgetter("_agent_id"))
    idea_id, priority = property(operator.attrgetter("_idea_id")), property(operator.attrgetter("_priority"))
    token, acquired_at = property(operator.attrgetter("_token")), property(operator.attrgetter("_acquired_at"))
    queue_wait_s, released = property(operator.attrgetter("_queue_wait_s")), property(operator.attrgetter("_released"))
    lab = property(lambda self: self)

    def __init__(self, conn: socket.socket, stream: Any, meta: dict[str, Any]) -> None:
        self._conn, self._stream = conn, stream
        self._token, self._slot_id, self._slot_role = meta["token"], meta["slot_id"], meta.get("slot_role")
        self._role, self._agent_id, self._idea_id = meta.get("role"), meta["agent_id"], meta.get("idea_id")
        self._priority = meta.get("priority", DEFAULT_PRIORITY)
        self._acquired_at, self._queue_wait_s = float(meta.get("acquired_at", 0.0)), float(meta.get("queue_wait_s", 0.0))
        self._released, self._trace_record, self._lock = False, None, threading.Lock()

    def _call(self, op: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            if self._released: raise PoolError(f"lease for slot {self._slot_id!r} has already been released")
            try:
                _write_json_line(self._stream, {"op": op, "token": self._token, **payload})
                reply = _read_json_line(self._stream)
            except OSError as exc:
                raise PoolError(f"pool daemon connection lost during {op}: {exc}") from exc
            if reply is None: raise PoolError(f"pool daemon closed connection during {op}")
            return _raise_if_error(reply)

    def install(self, patch: dict[str, Any]) -> dict[str, Any]: return self._call("install", patch=patch)
    def check(self, workload: dict[str, Any] | None = None) -> dict[str, Any]: return self._call("check", workload=workload)
    def reset(self, kind: str = "state") -> dict[str, Any]: return self._call("reset", kind=kind)
    def uninstall(self, id: str | None = None) -> dict[str, Any]: return self._call("uninstall", id=id)
    def status(self) -> dict[str, Any]: return self._call("status")
    def source(self, module: str) -> dict[str, Any]: return self._call("source", module=module)
    def receipts(self) -> dict[str, Any]: return self._call("receipts")
    def close(self) -> None:
        if self._released: raise PoolError(f"lease for slot {self._slot_id!r} has already been released")
        raise PoolError("cannot close a pooled lab via RemoteLease; call release()")

    def release(self, error: str | BaseException | None = None) -> dict[str, Any]:
        with self._lock:
            if self._released: return dict(self._trace_record or {})
            self._released = True
            try:
                _write_json_line(self._stream, {"op": "release", "token": self._token, "error": _format_error(error)})
                reply = _read_json_line(self._stream)
                if isinstance(reply, dict) and reply.get("ok"): self._trace_record = dict(reply.get("record") or {})
                elif isinstance(reply, dict): _raise_if_error(reply)
            except OSError: pass
            finally: _close_resources(self._stream, self._conn)
            return dict(self._trace_record or {})

    def __enter__(self) -> RemoteLease: return self
    def __exit__(self, _exc_type: Any, exc_value: BaseException | None, _tb: Any) -> bool:
        self.release(error=exc_value)
        return False

class PoolClient:
    """Cross-process client talking to `PoolServer` over an `AF_UNIX` JSON-line socket."""

    def __init__(self, sock_path: str, *, timeout_s: float | None = 60.0) -> None:
        self.sock_path, self.timeout_s = os.path.abspath(sock_path), timeout_s

    def _connect(self, timeout_s: float | None) -> tuple[socket.socket, Any]:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(timeout_s)
        try:
            conn.connect(self.sock_path)
            return conn, conn.makefile("rwb")
        except BaseException:
            _close_resources(conn)
            raise

    def _one_shot(self, payload: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        conn, stream = self._connect(self.timeout_s if timeout_s is None else timeout_s)
        try:
            _write_json_line(stream, payload)
            if (reply := _read_json_line(stream)) is None: raise PoolError("pool daemon closed connection")
            return _raise_if_error(reply)
        finally: _close_resources(stream, conn)

    def acquire(self, agent_id: str, *, role: str | None = None, idea_id: str | None = None,
                priority: str = DEFAULT_PRIORITY, timeout_s: float | None = None) -> RemoteLease:
        norm_t = _validate_timeout(timeout_s)
        sock_t = None if (norm_t is None or norm_t > 3600.0) else max(2.0, norm_t + 5.0)
        conn, stream = self._connect(sock_t)
        try:
            _write_json_line(stream, {"op": "acquire", "agent_id": agent_id, "role": role,
                                      "idea_id": idea_id, "priority": priority, "timeout_s": norm_t})
            if (reply := _read_json_line(stream)) is None:
                raise PoolError("pool daemon closed connection during acquire")
            _raise_if_error(reply)
            if not reply.get("ok"): raise PoolError(str(reply.get("error") or "acquire failed"))
            conn.settimeout(self.timeout_s)
            return RemoteLease(conn, stream, reply)
        except BaseException:
            _close_resources(stream, conn)
            raise

    lease = acquire
    def stats(self, *, timeout_s: float | None = None) -> dict[str, Any]: return self._one_shot({"op": "stats"}, timeout_s=timeout_s)
    def health(self, *, timeout_s: float | None = None) -> dict[str, Any]: return self._one_shot({"op": "health"}, timeout_s=timeout_s)

def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for `rsibox pool serve|stats|health`."""
    ap = argparse.ArgumentParser(prog="rsibox pool", description="Multi-agent sandbox pool daemon.")
    sub = ap.add_subparsers(dest="pool_cmd", required=True)
    default_sock = os.environ.get("RSIBOX_POOL_SOCK")
    p = sub.add_parser("serve", help="run a cross-process Unix-socket SandboxPool daemon")
    p.add_argument("--sock", default=default_sock, required=not default_sock)
    p.add_argument("--lab", required=True, help="module:Class, or a name `rsibox list` knows")
    p.add_argument("--real", action="store_true")
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--role", action="append", default=[], help="slot role (repeat per slot or once for all)")
    p.add_argument("--trace", default=None, help="JSONL hash-chained lease trace path")
    p.add_argument("--max-priority-streak", type=int, default=2)
    for cmd_name in ("stats", "health"):
        q = sub.add_parser(cmd_name, help=f"query pool daemon {cmd_name}")
        q.add_argument("--sock", default=default_sock, required=not default_sock)
        q.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args(argv)
    if a.pool_cmd in ("stats", "health"):
        try: reply = getattr(PoolClient(a.sock, timeout_s=a.timeout), a.pool_cmd)()
        except (OSError, PoolError) as exc:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=1))
            return 1
        print(json.dumps(reply, indent=1, default=str))
        return 0 if reply.get("ok") else 1
    if a.slots < 1:
        print("rsibox pool serve: --slots must be >= 1", file=sys.stderr)
        return 2
    from . import cli as _cli
    try:
        spec = _cli._resolve(a.lab, a.real)
        lab_cls = _box.load_lab(spec)
    except Exception as exc:
        print(f"rsibox pool serve: cannot load lab {a.lab!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    slots: list[PoolSlot] = []
    try:
        for idx in range(a.slots):
            role = a.role[0] if len(a.role) == 1 else (a.role[idx] if idx < len(a.role) else None)
            slots.append(PoolSlot(f"slot-{idx}", lab_cls(), role=role))
    except Exception as exc:
        for slot in slots:
            with contextlib.suppress(Exception): slot.lab.close()
        print(f"rsibox pool serve: failed to start slot: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    pool = SandboxPool(slots, trace_path=a.trace, close_labs=True, max_priority_streak=a.max_priority_streak)
    server = PoolServer(pool, a.sock, close_pool=True)
    signal.signal(signal.SIGTERM, lambda *_: server._closed.set())
    signal.signal(signal.SIGINT, lambda *_: server._closed.set())
    print(json.dumps({"ok": True, "sock": server.sock_path, "slots": len(slots), "lab": spec}), flush=True)
    server.wait()
    server.close(close_pool=True)
    return 0
