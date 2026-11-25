"""S2: Shared Zygote & Pre-Fork Daemon Service (pure Python stdlib + `forksafe`).

Coordinates pre-fork CUDA guard installation, optional module preloading, heap GC
freezing, and `AF_UNIX` + `SCM_RIGHTS` (`socket.send_fds` / `socket.recv_fds`)
child spawning across all ten RSIBox engines without importing `torch` or engine code.
"""
from __future__ import annotations

import contextlib, gc, importlib, json, os, selectors, signal, socket, struct, sys, tempfile, threading, time, uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import forksafe

MAX_SOCK_PATH = 103 if sys.platform == "darwin" else 107
_FRAME_HDR = struct.Struct("!I")
_MAX_FRAME_BYTES = 1024 * 1024
_MAX_FDS_PER_SPAWN = 64


class ZygoteError(RuntimeError): """Base error raised by `ZygoteService` on lifecycle, socket, or spawn failures."""
class ForkPoisonedError(ZygoteError): """Raised fail-closed when CUDA driver or torch runtime is initialized pre-fork."""
class ZygoteClosedError(ZygoteError): """Raised when operating on a closed or torn-down `ZygoteService`."""
class SpawnError(ZygoteError): """Raised when spawning, passing descriptors, or waiting on a child fails."""


@dataclass(frozen=True)
class ZygoteProfile:
    """Engine-scoped pre-fork guard, preload, and environment configuration."""
    engine: str
    preload_modules: tuple[str, ...] = ()
    env_defaults: dict[str, str] = field(default_factory=dict)
    pin_triton: bool = True
    pin_cute_dsl: bool = True
    guard_cuda_python: bool = True
    install_properties: bool = True
    freeze_gc: bool = True


@dataclass(frozen=True)
class SpawnedChild:
    """Metadata and post-fork guard/affinity reports for a forked worker child."""
    pid: int
    role: str
    engine: str
    incarnation: int
    child_guard_report: dict[str, Any] = field(default_factory=dict)
    affinity_report: dict[str, Any] = field(default_factory=dict)
    exit_code: int | None = None


@dataclass(frozen=True)
class ZygoteState:
    """Immutable snapshot of `ZygoteService` lifecycle, poison status, and children."""
    engine: str
    started: bool
    guards_armed: bool
    closed: bool
    poisoned: bool
    poison_reasons: tuple[str, ...]
    socket_path: str | None
    incarnation: int
    active_children: tuple[SpawnedChild, ...]
    reaped_exit_codes: dict[int, int]
    guard_report: dict[str, Any]
    preload_report: dict[str, Any]
    gc_report: dict[str, Any]


_BASE_ENV: dict[str, str] = {"OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
_PROFILES: tuple[ZygoteProfile, ...] = tuple(
    ZygoteProfile(eng, mods, {**_BASE_ENV, **extra})
    for eng, mods, extra in (
        ("vllm", ("torch", "vllm"), {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}),
        ("sglang", ("torch", "sglang.srt.server_args"), {}),
        ("trt-llm", ("torch", "tensorrt_llm"), {}),
        ("megatron", ("torch", "megatron.core"), {"TORCHINDUCTOR_COMPILE_THREADS": "1"}),
        ("torchtitan", ("torch", "torchtitan"), {"TORCHINDUCTOR_COMPILE_THREADS": "1"}),
        ("verl", ("torch", "verl"), {}),
        ("fastvideo", ("torch", "fastvideo"), {}),
        ("transformers", ("torch", "transformers"), {}),
        ("miles", ("torch", "miles"), {}),
        ("nemo-rl", ("torch", "nemo_rl"), {}),
    )
)
_PROFILE_MAP = {p.engine: p for p in _PROFILES}
_ENGINE_ALIASES: dict[str, str] = {
    "trtllm": "trt-llm", "tensorrt-llm": "trt-llm", "tensorrtllm": "trt-llm",
    "megatron-lm": "megatron", "megatronlm": "megatron", "rsibox-megatron": "megatron",
    "torch-titan": "torchtitan", "fast-video": "fastvideo",
    "hf-transformers": "transformers", "huggingface-transformers": "transformers", "nemorl": "nemo-rl",
}
_ERROR_MAP: dict[str, type[ZygoteError]] = {
    "ForkPoisonedError": ForkPoisonedError, "ZygoteClosedError": ZygoteClosedError,
    "SpawnError": SpawnError, "ZygoteError": ZygoteError,
}


def _copy_profile(profile: ZygoteProfile) -> ZygoteProfile:
    return replace(profile, env_defaults=dict(profile.env_defaults))


def profile_for_engine(engine: str) -> ZygoteProfile:
    """Return the canonical `ZygoteProfile` for any of the 10 RSIBox engines or aliases."""
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError(f"engine must be a non-empty string, got {engine!r}")
    token = engine.strip().lower().replace("_", "-").replace(" ", "-")
    canonical = token if token in _PROFILE_MAP else (_ENGINE_ALIASES.get(token) or _ENGINE_ALIASES.get(token.replace("-", "")))
    if canonical is None:
        raise ValueError(f"unknown RSIBox engine: {engine!r}")
    return _copy_profile(_PROFILE_MAP[canonical])


def list_profiles() -> tuple[ZygoteProfile, ...]:
    """Return copies of all 10 canonical RSIBox engine `ZygoteProfile` definitions."""
    return tuple(_copy_profile(p) for p in _PROFILES)


def _close_fds(fds: Iterable[int]) -> None:
    for fd in fds:
        if fd >= 0:
            with contextlib.suppress(OSError): os.close(fd)


def _check_sock_len(path: Path) -> Path:
    encoded = os.fsencode(os.path.abspath(os.fspath(path)))
    if len(encoded) > MAX_SOCK_PATH:
        raise ZygoteError(f"AF_UNIX socket path too long ({len(encoded)} > {MAX_SOCK_PATH}): {path}")
    return path


def _validate_fds(pass_fds: Sequence[int]) -> list[int]:
    if isinstance(pass_fds, (str, bytes, bytearray)):
        raise TypeError("pass_fds must be a sequence of integer file descriptors")
    fds = list(pass_fds)
    for fd in fds:
        if isinstance(fd, bool) or not isinstance(fd, int) or fd < 0:
            raise ValueError(f"pass_fds entries must be non-negative ints, got {fd!r}")
    if len(fds) > _MAX_FDS_PER_SPAWN:
        raise SpawnError(f"too many pass_fds ({len(fds)} > {_MAX_FDS_PER_SPAWN})")
    return fds


def _recv_framed_fds(conn: socket.socket, max_fds: int = _MAX_FDS_PER_SPAWN) -> tuple[dict[str, Any], list[int]]:
    chunk, fds, flags, _ = socket.recv_fds(conn, 65536, max_fds)
    try:
        if (flags & getattr(socket, "MSG_CTRUNC", 0)) or not chunk:
            raise SpawnError("truncated SCM_RIGHTS control data or empty request frame")
        buf = bytearray(chunk)
        def _fill(need: int) -> None:
            while len(buf) < need:
                if not (more := conn.recv(min(65536, need - len(buf)))):
                    raise SpawnError("unexpected EOF reading framed payload")
                buf.extend(more)
        _fill(_FRAME_HDR.size)
        (body_len,) = _FRAME_HDR.unpack_from(buf, 0)
        if body_len <= 0 or body_len > _MAX_FRAME_BYTES:
            raise SpawnError(f"invalid request frame length: {body_len}")
        _fill(_FRAME_HDR.size + body_len)
        payload = json.loads(bytes(buf[_FRAME_HDR.size : _FRAME_HDR.size + body_len]).decode("utf-8"))
        if not isinstance(payload, dict): raise SpawnError("spawn request payload must be a JSON object")
        return payload, fds
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _close_fds(fds); raise SpawnError(f"malformed JSON spawn request: {exc}") from exc
    except BaseException:
        _close_fds(fds); raise


def _send_json_frame(sock: socket.socket, payload: dict[str, Any], fds: Sequence[int] = ()) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    frame = _FRAME_HDR.pack(len(body)) + body
    if fds:
        sent = socket.send_fds(sock, [frame], list(fds))
        if sent < len(frame): sock.sendall(frame[sent:])
    else:
        sock.sendall(frame)


def _poison_check() -> list[str]:
    reasons = list(forksafe.poison_reasons())
    if not reasons and forksafe.fork_poisoned(): reasons.append("fork_poisoned() reported True")
    return reasons


class ZygoteService:
    """Thread-safe, fail-closed S2 zygote coordinator with `AF_UNIX` + `SCM_RIGHTS` fork IPC."""

    def __init__(
        self, profile: ZygoteProfile, socket_path: str | Path | None = None, *,
        cuda_module_fn: Callable[[], Any] | None = None,
        worker_handler: Callable[[dict[str, Any], list[int]], int] | None = None,
        max_children: int = 64, importer: Callable[[str], Any] | None = None,
    ) -> None:
        if not isinstance(profile, ZygoteProfile): raise TypeError(f"profile must be a ZygoteProfile, not {type(profile).__name__}")
        if isinstance(max_children, bool) or not isinstance(max_children, int) or max_children < 1:
            raise ValueError(f"max_children must be an int >= 1, got {max_children!r}")
        self._profile, self._explicit_socket_path = _copy_profile(profile), (Path(socket_path) if socket_path is not None else None)
        self._cuda_module_fn, self._worker_handler, self._default_importer = cuda_module_fn, worker_handler, importer
        self._max_children, self._lock, self._incarnation = max_children, threading.RLock(), 0
        self._closed = self._torn_down = self._started = self._guards_armed = self._gc_frozen = self._owns_listener = False
        self._socket_path: Path | None = None
        self._children: dict[int, SpawnedChild] = {}
        self._active_pids: set[int] = set()
        self._exits: dict[int, int] = {}
        self._pending_handlers: dict[str, Callable[[dict[str, Any], list[int]], int]] = {}
        self._cuda_mod: Any = None
        self._restore_guard: Callable[[], Any] | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._listener: socket.socket | None = None
        self._wakeup_r: int | None = None
        self._wakeup_w: int | None = None
        self._server_thread: threading.Thread | None = None
        self._guard_report: dict[str, Any] = {}
        self._preload_report: dict[str, Any] = {}
        self._gc_report: dict[str, Any] = {}
        self._stop_event = threading.Event()

    @property
    def profile(self) -> ZygoteProfile: return _copy_profile(self._profile)
    @property
    def socket_path(self) -> Path | None:
        with self._lock: return self._socket_path
    def __enter__(self) -> ZygoteService:
        self.start(); return self
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: self.close()
    def _require_open(self) -> None:
        if self._closed: raise ZygoteClosedError("ZygoteService is closed")
    def _assert_not_poisoned(self) -> None:
        if reasons := _poison_check(): raise ForkPoisonedError(f"zygote fork poisoned: {'; '.join(reasons)}")

    def arm_guards(
        self, *, capability: Any = None, nvml: Any = None, probe: Any = None,
        preload: bool = True, importer: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        """Verify zero CUDA poisoning, apply env defaults, install `forksafe` guards, and optionally preload."""
        with self._lock:
            self._require_open(); self._assert_not_poisoned()
            for key, val in self._profile.env_defaults.items(): os.environ.setdefault(key, val)
            cuda_mod = self._cuda_module_fn() if self._cuda_module_fn is not None else None
            import_fn = importer or self._default_importer
            use_forksafe_torch = preload and cuda_mod is None and import_fn is None and "torch" in self._profile.preload_modules
            guard_report, restore_fn = forksafe.install_all_guards(
                cuda=cuda_mod, capability=capability, nvml=nvml, properties=self._profile.install_properties,
                probe=probe, triton=self._profile.pin_triton, cuda_python=self._profile.guard_cuda_python,
                cute_dsl=self._profile.pin_cute_dsl, import_torch=use_forksafe_torch,
            )
            self._cuda_mod, self._restore_guard, self._guard_report = cuda_mod, restore_fn, dict(guard_report)
            preload_report: dict[str, Any] = {}
            if preload and self._profile.preload_modules:
                loader = import_fn or importlib.import_module
                for mod_name in self._profile.preload_modules:
                    if mod_name == "torch" and use_forksafe_torch and "torch" in sys.modules:
                        preload_report[mod_name] = {"loaded": True, "via": "install_all_guards"}; continue
                    try: loader(mod_name)
                    except ImportError as exc: preload_report[mod_name] = {"loaded": False, "skipped": repr(exc)[:200]}; continue
                    preload_report[mod_name] = {"loaded": True}
                    if forksafe.cuda_initialized(): raise ForkPoisonedError(f"preloading module {mod_name!r} initialized CUDA")
            self._preload_report = preload_report
            self._gc_report = forksafe.freeze_gc_before_fork() if self._profile.freeze_gc else {}
            self._gc_frozen, self._guards_armed = self._profile.freeze_gc, True
            return {"guards": dict(self._guard_report), "preload": dict(self._preload_report), "gc": dict(self._gc_report)}

    def start(
        self, *, listener: socket.socket | None = None, capability: Any = None, nvml: Any = None,
        probe: Any = None, preload: bool = True, importer: Callable[[str], Any] | None = None,
    ) -> Path:
        """Arm pre-fork guards (if needed) and start the `AF_UNIX` `SCM_RIGHTS` server loop."""
        with self._lock:
            self._require_open()
            if self._started:
                self._assert_not_poisoned(); assert self._socket_path is not None; return self._socket_path
            if not self._guards_armed:
                self.arm_guards(capability=capability, nvml=nvml, probe=probe, preload=preload, importer=importer)
            else:
                self._assert_not_poisoned()
            tdir: tempfile.TemporaryDirectory[str] | None = None
            srv_sock: socket.socket | None = None
            owns_listener = listener is None
            try:
                if listener is not None:
                    raw = listener.getsockname()
                    sock_path = _check_sock_len(Path(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)))
                    srv_sock = listener
                else:
                    if self._explicit_socket_path is not None:
                        sock_path = _check_sock_len(self._explicit_socket_path)
                        sock_path.parent.mkdir(parents=True, exist_ok=True)
                        with contextlib.suppress(FileNotFoundError): sock_path.unlink()
                    else:
                        tdir = tempfile.TemporaryDirectory(prefix="zyg_")
                        sock_path = _check_sock_len(Path(tdir.name) / "z.sock")
                    srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    srv_sock.bind(os.fspath(sock_path)); os.chmod(sock_path, 0o600); srv_sock.listen(16)
                wake_r, wake_w = os.pipe(); os.set_blocking(wake_r, False); os.set_blocking(wake_w, False)
            except BaseException:
                if owns_listener and srv_sock is not None: srv_sock.close()
                if tdir is not None: tdir.cleanup()
                raise
            self._temp_dir, self._listener, self._socket_path, self._owns_listener = tdir, srv_sock, sock_path, owns_listener
            self._wakeup_r, self._wakeup_w = wake_r, wake_w
            self._stop_event.clear(); ready = threading.Event()
            thread = threading.Thread(target=self._serve_loop, args=(srv_sock, wake_r, ready), name=f"zygote-{self._profile.engine}", daemon=True)
            self._server_thread, self._started = thread, True
            thread.start()
        ready.wait(timeout=5.0)
        return sock_path

    def _serve_loop(self, listener: socket.socket, wake_r: int, ready: threading.Event) -> None:
        sel = selectors.DefaultSelector()
        try:
            listener.setblocking(False); sel.register(listener, selectors.EVENT_READ, "accept"); sel.register(wake_r, selectors.EVENT_READ, "wake")
            ready.set()
            while not self._stop_event.is_set():
                for key, _ in sel.select(timeout=0.5):
                    if key.data == "wake":
                        with contextlib.suppress(OSError): os.read(wake_r, 1024)
                        continue
                    if self._stop_event.is_set(): break
                    try: conn, _ = listener.accept()
                    except BlockingIOError: continue
                    except OSError: break
                    with conn:
                        conn.setblocking(True); received_fds: list[int] = []
                        try:
                            payload, received_fds = _recv_framed_fds(conn)
                            self._process_spawn_request(conn, payload, received_fds)
                        except Exception as exc:
                            with contextlib.suppress(OSError):
                                _send_json_frame(conn, {"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
                        finally: _close_fds(received_fds)
        finally:
            ready.set(); sel.close()

    def _process_spawn_request(self, conn: socket.socket, payload: dict[str, Any], received_fds: list[int]) -> None:
        role = str(payload.get("role") or "").strip()
        if not role: raise SpawnError("spawn request missing non-empty 'role'")
        raw_env = payload.get("env") or {}
        if not isinstance(raw_env, dict): raise SpawnError("spawn request 'env' must be a JSON object")
        env_overrides, req_id = {str(k): str(v) for k, v in raw_env.items()}, payload.get("req_id")
        with self._lock:
            self._require_open(); self._reap_finished_locked()
            if len(self._active_pids) >= self._max_children: raise SpawnError(f"max_children={self._max_children} reached")
            self._assert_not_poisoned(); self._incarnation += 1
            incarnation, engine = self._incarnation, self._profile.engine
            handler = (self._pending_handlers.get(req_id) if isinstance(req_id, str) else None) or self._worker_handler
            cuda_mod, clear_env = self._cuda_mod, bool(self._guard_report.get("nvml_env_set", False))
            parent_fds = (self._listener.fileno() if self._listener is not None else -1, conn.fileno(),
                          self._wakeup_r if self._wakeup_r is not None else -1, self._wakeup_w if self._wakeup_w is not None else -1)
        child_payload: dict[str, Any] = {"role": role, "engine": engine, "incarnation": incarnation, "env": env_overrides}
        report_r, report_w = os.pipe()
        try: pid = os.fork()
        except BaseException: _close_fds((report_r, report_w)); raise
        if pid == 0:
            _close_fds((report_r, *parent_fds))
            _run_child(report_w, env_overrides, cuda_mod, clear_env, child_payload, received_fds, handler)
            os._exit(1)
        _close_fds((report_w,))
        try:
            _close_fds(received_fds); received_fds.clear()
            chunks: list[bytes] = []
            while chunk := os.read(report_r, 65536): chunks.append(chunk)
            raw_report = b"".join(chunks)
        finally: _close_fds((report_r,))
        if not raw_report:
            with contextlib.suppress(ChildProcessError):
                _, status = os.waitpid(pid, 0)
                with self._lock: self._record_exit_locked(pid, status)
            raise SpawnError(f"child pid={pid} exited before sending guard report")
        rep = json.loads(raw_report.decode("utf-8"))
        child = SpawnedChild(pid=pid, role=role, engine=engine, incarnation=incarnation,
                             child_guard_report=rep.get("child_guard_report", {}), affinity_report=rep.get("affinity_report", {}))
        with self._lock: self._children[pid] = child; self._active_pids.add(pid)
        _send_json_frame(conn, {"ok": True, "child": {
            "pid": child.pid, "role": child.role, "engine": child.engine, "incarnation": child.incarnation,
            "child_guard_report": child.child_guard_report, "affinity_report": child.affinity_report}})

    def spawn_child(
        self, role: str, *, env: dict[str, str] | None = None, pass_fds: Sequence[int] = (),
        handler: Callable[[dict[str, Any], list[int]], int] | None = None, wait: bool = False, timeout_s: float = 10.0,
    ) -> SpawnedChild:
        """Spawn a worker child via `AF_UNIX` + `SCM_RIGHTS` FD passing and `os.fork()`."""
        if not isinstance(role, str) or not role.strip(): raise ValueError(f"role must be a non-empty string, got {role!r}")
        if env is not None and not isinstance(env, dict): raise TypeError(f"env must be a dict or None, not {type(env).__name__}")
        if timeout_s <= 0: raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
        fds, env_clean = _validate_fds(pass_fds), {str(k): str(v) for k, v in (env or {}).items()}
        with self._lock:
            self._require_open()
            if not self._started or self._socket_path is None: raise ZygoteError("ZygoteService is not started; call start() first")
            self._reap_finished_locked()
            if len(self._active_pids) >= self._max_children: raise SpawnError(f"max_children={self._max_children} reached")
            self._assert_not_poisoned()
            sock_path, req_id = self._socket_path, uuid.uuid4().hex
            if handler is not None: self._pending_handlers[req_id] = handler
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(timeout_s); client.connect(os.fspath(sock_path))
                _send_json_frame(client, {"role": role.strip(), "env": env_clean, "req_id": req_id}, fds)
                reply, reply_fds = _recv_framed_fds(client, max_fds=1); _close_fds(reply_fds)
        except OSError as exc:
            with self._lock: closed = self._closed
            raise (ZygoteClosedError("ZygoteService closed during spawn_child") if closed else SpawnError(f"AF_UNIX spawn IPC failed: {exc}")) from exc
        finally:
            with self._lock: self._pending_handlers.pop(req_id, None)
        if not reply.get("ok"):
            raise _ERROR_MAP.get(str(reply.get("error_type") or ""), SpawnError)(str(reply.get("error") or "spawn failed"))
        cd = reply["child"]
        child = SpawnedChild(pid=int(cd["pid"]), role=str(cd["role"]), engine=str(cd["engine"]), incarnation=int(cd["incarnation"]),
                             child_guard_report=dict(cd.get("child_guard_report", {})), affinity_report=dict(cd.get("affinity_report", {})))
        return replace(child, exit_code=self.wait_child(child.pid, timeout_s=timeout_s)) if wait else child

    def _record_exit_locked(self, pid: int, status: int) -> int:
        code = os.waitstatus_to_exitcode(status)
        self._exits[pid] = code; self._active_pids.discard(pid)
        if pid in self._children: self._children[pid] = replace(self._children[pid], exit_code=code)
        return code

    def _reap_pid_locked(self, pid: int) -> int | None:
        if pid in self._exits: self._active_pids.discard(pid); return self._exits[pid]
        try: waited_pid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError: self._active_pids.discard(pid); return None
        return None if waited_pid == 0 else self._record_exit_locked(pid, status)

    def _reap_finished_locked(self) -> dict[int, int]:
        for pid in list(self._active_pids): self._reap_pid_locked(pid)
        return dict(self._exits)

    def reap_finished(self) -> dict[int, int]:
        """Non-blockingly reap any exited children and return `{pid: exit_code}`."""
        with self._lock: return self._reap_finished_locked()

    def wait_child(self, pid: int, *, timeout_s: float = 10.0) -> int:
        """Wait for a managed child PID up to `timeout_s` and return its exit code."""
        if timeout_s <= 0: raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
        deadline, sleep_s = time.monotonic() + timeout_s, 0.001
        while True:
            with self._lock:
                if pid not in self._children and pid not in self._exits: raise SpawnError(f"unknown child pid={pid}")
                if (code := self._reap_pid_locked(pid)) is not None: return code
            if time.monotonic() >= deadline: raise SpawnError(f"timed out waiting for child pid={pid} after {timeout_s}s")
            time.sleep(min(sleep_s, max(0.0005, deadline - time.monotonic()))); sleep_s = min(0.02, sleep_s * 1.5)

    def status(self) -> ZygoteState:
        """Return an immutable `ZygoteState` snapshot of service health and children."""
        with self._lock:
            self._reap_finished_locked(); reasons = tuple(_poison_check())
            active = tuple(self._children[pid] for pid in sorted(self._active_pids) if pid in self._children)
            return ZygoteState(
                engine=self._profile.engine, started=self._started, guards_armed=self._guards_armed, closed=self._closed,
                poisoned=bool(reasons), poison_reasons=reasons, socket_path=os.fspath(self._socket_path) if self._socket_path else None,
                incarnation=self._incarnation, active_children=active, reaped_exit_codes=dict(self._exits),
                guard_report=dict(self._guard_report), preload_report=dict(self._preload_report), gc_report=dict(self._gc_report),
            )

    def teardown(self, *, grace_s: float = 2.0) -> dict[int, int]:
        """Stop the accept loop, close owned sockets/FDs, reap children, and restore guards."""
        with self._lock:
            if self._torn_down: return dict(self._exits)
            self._closed = self._torn_down = True; self._stop_event.set()
            wake_w, wake_r, self._wakeup_w, self._wakeup_r = self._wakeup_w, self._wakeup_r, None, None
            thread, self._server_thread = self._server_thread, None
            listener, owns_listener, self._listener = self._listener, self._owns_listener, None
            sock_path, temp_dir, self._temp_dir = self._socket_path, self._temp_dir, None
            restore_guard, self._restore_guard = self._restore_guard, None
            cuda_mod, clear_env = self._cuda_mod, bool(self._guard_report.get("nvml_env_set", False))
            gc_frozen, self._gc_frozen = self._gc_frozen, False; self._pending_handlers.clear()
        if wake_w is not None:
            with contextlib.suppress(OSError): os.write(wake_w, b"x")
        if thread is not None and thread.is_alive(): thread.join(timeout=max(2.0, grace_s))
        _close_fds(fd for fd in (wake_w, wake_r) if fd is not None)
        if listener is not None and owns_listener:
            with contextlib.suppress(OSError): listener.close()
            if sock_path is not None:
                with contextlib.suppress(OSError): sock_path.unlink()
        if temp_dir is not None:
            with contextlib.suppress(OSError): temp_dir.cleanup()
        self._terminate_remaining_children(grace_s=grace_s)
        for fn in (restore_guard, forksafe.unguard_cuda_python, lambda: forksafe.unpatch(cuda_mod, clear_env=clear_env),
                   gc.unfreeze if gc_frozen and hasattr(gc, "unfreeze") else None):
            if fn is not None:
                with contextlib.suppress(Exception): fn()
        with self._lock: return dict(self._exits)

    def _terminate_remaining_children(self, *, grace_s: float) -> None:
        with self._lock: self._reap_finished_locked(); remaining = sorted(self._active_pids)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with self._lock: alive = [pid for pid in remaining if pid in self._active_pids]
            if not alive: return
            for pid in alive:
                with contextlib.suppress(OSError): os.kill(pid, sig)
            deadline = time.monotonic() + (max(0.05, grace_s) if sig == signal.SIGTERM else 1.0)
            while time.monotonic() < deadline:
                with self._lock:
                    self._reap_finished_locked()
                    if not any(pid in self._active_pids for pid in remaining): return
                time.sleep(0.01)
        for pid in remaining:
            with self._lock:
                if pid not in self._active_pids: continue
            with contextlib.suppress(ChildProcessError):
                _, status = os.waitpid(pid, 0)
                with self._lock: self._record_exit_locked(pid, status)

    def close(self) -> None:
        """Close the service idempotently, releasing all sockets, FDs, and child processes."""
        self.teardown()


def _run_child(
    report_w: int, env_overrides: dict[str, str], cuda_mod: Any, clear_env: bool,
    child_payload: dict[str, Any], received_fds: list[int], handler: Callable[[dict[str, Any], list[int]], int] | None,
) -> None:
    exit_code = 0
    try:
        if env_overrides: os.environ.update(env_overrides)
        child_guard_report = forksafe.unpatch_child(cuda_mod, clear_env=clear_env)
        affinity_report = forksafe.apply_child_runtime_affinity(env_overrides)
        report_bytes = json.dumps({"child_guard_report": child_guard_report, "affinity_report": affinity_report}, separators=(",", ":")).encode("utf-8")
        view = memoryview(report_bytes)
        while view: view = view[os.write(report_w, view) :]
        _close_fds((report_w,)); report_w = -1
        child_payload.update({"pid": os.getpid(), "child_guard_report": child_guard_report, "affinity_report": affinity_report})
        if handler is not None:
            res = handler(child_payload, list(received_fds))
            exit_code = 0 if res is None else int(res)
    except SystemExit as exc:
        exit_code = 0 if exc.code is None else (int(exc.code) if isinstance(exc.code, int) else 1)
    except BaseException:
        exit_code = 1
    finally:
        _close_fds((report_w, *received_fds))
        os._exit(exit_code)
