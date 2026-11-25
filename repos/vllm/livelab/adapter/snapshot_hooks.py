"""vLLM snapshot quiesce hooks and Standby Shadow promotion (stdlib only).

Implements `VllmQuiesceHooks` (`rsibox.warm.QuiesceHooks` for `ai-dynamo/snapshot` via
vLLM dev routes `/pause`, `/sleep`, `/release_kv_cache_memory`, `/is_sleeping`, `/wake_up`,
`/resume`) and `park_shadow_server` / `VllmShadowPromoter` / `make_shadow_recovery_driver`
(`rsibox.warm.StandbyShadowDriver` failover for Dynamo GMS Scratch-KV shadow servers).
"""
from __future__ import annotations

import collections
import contextlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Sequence

from rsibox.warm import Mutability, QuiesceReceipt, StandbyShadowDriver

from . import client

DEFAULT_WORKER_MODULES: tuple[str, ...] = (
    "vllm.v1.worker.gpu.sample.sampler",
    "vllm.model_executor.layers.logits_processor",
)
VALID_PAUSE_MODES: tuple[str, ...] = ("keep", "abort", "wait")
VALID_SLEEP_LEVELS: tuple[int, ...] = (0, 1, 2)


def _resolve_base_url(target: Any) -> str:
    if isinstance(target, str):
        stripped = target.strip().rstrip("/")
        if stripped:
            return stripped
        raise ValueError("base_url must not be empty")
    port = target if isinstance(target, int) and not isinstance(target, bool) else getattr(target, "port", None)
    if isinstance(port, int) and not isinstance(port, bool) and port > 0:
        return f"http://127.0.0.1:{port}"
    raise ValueError(f"cannot resolve HTTP base URL from {target!r}")


def _validate_sleep_args(sleep_level: int, pause_mode: str) -> None:
    if sleep_level not in VALID_SLEEP_LEVELS:
        raise ValueError(f"sleep_level must be in {VALID_SLEEP_LEVELS}, got {sleep_level!r}")
    if pause_mode not in VALID_PAUSE_MODES:
        raise ValueError(f"pause_mode must be in {VALID_PAUSE_MODES}, got {pause_mode!r}")


def _http_json_request(
    base_url: str, method: str, path: str, *,
    query: Sequence[tuple[str, str]] | None = None,
    body: dict[str, Any] | None = None, timeout: float = 30.0,
) -> tuple[bool, int | None, dict[str, Any] | None, str | None]:
    """Send an HTTP request and decode a JSON object reply without raising."""
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"
    headers: dict[str, str] = {}
    data: bytes | None = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode("utf-8"), "application/json"
    elif method.upper() in ("POST", "PUT", "PATCH"):
        data = b""
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code, raw_bytes = int(getattr(response, "status", 200)), response.read()
    except urllib.error.HTTPError as error:
        raw_error, parsed_error = "", None
        with contextlib.suppress(Exception):
            raw_error = error.read().decode("utf-8", errors="replace").strip()
            decoded = json.loads(raw_error) if raw_error else None
            parsed_error = decoded if isinstance(decoded, dict) else None
        detail = (
            str(parsed_error.get("error") or parsed_error.get("detail") or raw_error)
            if parsed_error is not None else (raw_error or str(error.reason))
        )
        return False, int(error.code), parsed_error, f"HTTP {error.code} on {method} {path}: {detail}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return False, None, None, f"{type(error).__name__} on {method} {path}: {error}"
    if not raw_bytes:
        return True, status_code, {}, None
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except ValueError as error:
        return False, status_code, None, f"invalid JSON from {method} {path}: {error}"
    if not isinstance(parsed, dict):
        return False, status_code, None, f"expected JSON object from {method} {path}, got {type(parsed).__name__}"
    return True, status_code, parsed, None


def _query_is_sleeping(base_url: str, timeout: float) -> tuple[bool, bool | None, str | None]:
    ok, _status, payload, error = _http_json_request(base_url, "GET", "/is_sleeping", timeout=timeout)
    if not ok or payload is None:
        return False, None, error or "GET /is_sleeping failed"
    sleeping = payload.get("is_sleeping")
    if not isinstance(sleeping, bool):
        return False, None, f"GET /is_sleeping returned non-boolean payload: {payload!r}"
    return True, sleeping, None


def _verify_host_channel(
    control_call: Callable[[dict[str, Any], float], dict[str, Any]],
    modules: Sequence[str], timeout: float,
) -> tuple[bool, str | None]:
    """Verify `adapter.host` answers `ping` cleanly (not poisoned) and worker `fingerprint` succeeds."""
    def invoke(operation: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            reply = control_call(payload, timeout)
        except Exception as error:
            return None, f"adapter host {operation} raised {type(error).__name__}: {error}"
        if not isinstance(reply, dict) or not reply.get("ok"):
            detail = reply.get("error") if isinstance(reply, dict) else repr(reply)
            return None, f"adapter host {operation} failed: {detail}"
        return reply, None

    ping_reply, ping_error = invoke("ping", {"op": "ping"})
    if ping_error is not None or ping_reply is None:
        return False, ping_error
    if ping_reply.get("poison"):
        return False, f"adapter host is poisoned: {ping_reply['poison']}"
    fingerprint_reply, fingerprint_error = invoke(
        "fingerprint", {"op": "fingerprint", "modules": list(modules)}
    )
    if fingerprint_error is not None or fingerprint_reply is None:
        return False, fingerprint_error
    workers_reply = fingerprint_reply.get("workers")
    if not isinstance(workers_reply, dict) or not workers_reply.get("ok"):
        detail = workers_reply.get("error") if isinstance(workers_reply, dict) else repr(workers_reply)
        return False, f"worker fingerprint failed: {detail}"
    return True, None


def _wake_resume_and_verify(
    base_url: str, control_call: Callable[[dict[str, Any], float], dict[str, Any]],
    modules: Sequence[str], *, wake_query: Sequence[tuple[str, str]] | None = None,
    timeout: float = 30.0, label: str = "post-restore",
) -> dict[str, Any]:
    """Fail-closed wake_up -> resume -> is_sleeping==False -> host ping/fingerprint sequence."""
    def fail(message: str, **timings: float) -> dict[str, Any]:
        return {"ok": False, "error": f"{label} {message}", **timings, "host_ping_ok": False}

    wake_start = time.monotonic()
    wake_ok, _status, wake_payload, wake_error = _http_json_request(
        base_url, "POST", "/wake_up", query=wake_query, timeout=timeout
    )
    wake_up_seconds = round(time.monotonic() - wake_start, 6)
    if not wake_ok or (isinstance(wake_payload, dict) and wake_payload.get("status") == "sleeping"):
        return fail(f"wake_up failed: {wake_error or wake_payload!r}", wake_up_s=wake_up_seconds)

    resume_start = time.monotonic()
    resume_ok, _status, _payload, resume_error = _http_json_request(base_url, "POST", "/resume", timeout=timeout)
    resume_seconds = round(time.monotonic() - resume_start, 6)
    if not resume_ok:
        return fail(f"resume failed: {resume_error}", wake_up_s=wake_up_seconds, resume_s=resume_seconds)

    check_ok, is_sleeping, check_error = _query_is_sleeping(base_url, timeout)
    if not check_ok or is_sleeping is not False:
        reason = check_error or "GET /is_sleeping returned True after wake_up and resume"
        return fail(f"awake check failed: {reason}", wake_up_s=wake_up_seconds, resume_s=resume_seconds)

    host_ok, host_error = _verify_host_channel(control_call, modules, timeout)
    if not host_ok:
        return fail(f"host check failed: {host_error}", wake_up_s=wake_up_seconds, resume_s=resume_seconds)
    return {"ok": True, "wake_up_s": wake_up_seconds, "resume_s": resume_seconds, "host_ping_ok": True}


class VllmQuiesceHooks:
    """`rsibox.warm.QuiesceHooks` implementation for a live vLLM server with dev endpoints."""

    mutability: Mutability = "readonly_weights"

    def __init__(
        self, *, lab: Any = None, base_url: str | None = None, port: int | None = None,
        kernel_dir: str | None = None, model: str = "m", sleep_level: int = 1,
        pause_mode: str = "keep", release_kv_only: bool = False, weights_backend: str | None = None,
        warmup_prompt: str = "The capital of France is", warmup_max_tokens: int = 2,
        timeout: float = 30.0, control_caller: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
        modules: Sequence[str] | None = None,
    ) -> None:
        _validate_sleep_args(sleep_level, pause_mode)
        self._lab, self._base_url, self._port, self._kernel_dir = lab, base_url, port, kernel_dir
        self._model, self._sleep_level, self._pause_mode = str(model), int(sleep_level), str(pause_mode)
        self._release_kv_only, self._weights_backend = bool(release_kv_only), weights_backend
        self._warmup_prompt, self._warmup_max_tokens = str(warmup_prompt), int(warmup_max_tokens)
        self._timeout, self._control_caller = float(timeout), control_caller
        self._modules = tuple(modules) if modules is not None else DEFAULT_WORKER_MODULES

    def _current_base_url(self) -> str:
        for candidate in (
            self._base_url, self._port,
            getattr(self._lab, "server", None) if self._lab is not None else None, self._lab,
        ):
            if candidate is not None:
                return _resolve_base_url(candidate)
        raise ValueError("VllmQuiesceHooks requires lab, base_url, or port")

    def _current_kernel_dir(self) -> str | None:
        if self._kernel_dir is not None:
            return self._kernel_dir
        server = getattr(self._lab, "server", None) if self._lab is not None else None
        return getattr(server, "kernel_dir", None) if server is not None else None

    def _check_lab_health(self) -> str | None:
        if self._lab is None:
            return None
        server = getattr(self._lab, "server", None)
        if server is not None and hasattr(server, "alive") and not server.alive():
            return "vLLM server process is not alive"
        session = getattr(self._lab, "session", None)
        if session is not None and session.poisoned():
            return f"vLLM lab session is poisoned: {session.poisoned()}"
        return None

    def _make_control_call(
        self, *, min_seq: int | None = None
    ) -> Callable[[dict[str, Any], float], dict[str, Any]]:
        if self._control_caller is not None:
            return self._control_caller
        kernel_dir = self._current_kernel_dir()
        if kernel_dir:
            first_call = True

            def call_file_channel(command: dict[str, Any], timeout: float) -> dict[str, Any]:
                nonlocal first_call
                seq_floor = min_seq if first_call else None
                first_call = False
                return client.send(kernel_dir, command, timeout=timeout, min_seq=seq_floor)

            return call_file_channel
        if self._lab is not None and hasattr(self._lab, "_ctl"):
            return lambda command, timeout: self._lab._ctl(command, timeout=timeout)
        return lambda _command, _timeout: {"ok": False, "error": "no adapter control channel configured"}

    def warmup_for_snapshot(self) -> None:
        """Run one deterministic completion and verify `adapter.host` ping + worker fingerprint."""
        health_error = self._check_lab_health()
        if health_error is not None:
            raise RuntimeError(health_error)
        ok, _status, payload, error = _http_json_request(
            self._current_base_url(), "POST", "/v1/completions",
            body={"model": self._model, "prompt": self._warmup_prompt,
                  "max_tokens": self._warmup_max_tokens, "temperature": 0.0, "seed": 0},
            timeout=self._timeout,
        )
        if not ok or payload is None or not payload.get("choices"):
            raise RuntimeError(f"snapshot warmup /v1/completions failed: {error or payload!r}")
        host_ok, host_error = _verify_host_channel(self._make_control_call(), self._modules, self._timeout)
        if not host_ok:
            raise RuntimeError(f"snapshot warmup host check failed: {host_error}")

    def _rollback_resume(self, base_url: str) -> None:
        with contextlib.suppress(Exception):
            _http_json_request(base_url, "POST", "/resume", timeout=min(self._timeout, 10.0))

    def quiesce_for_snapshot(self) -> QuiesceReceipt:
        """Pause generation, park GPU memory via `/sleep` or `/release_kv_cache_memory`, and verify."""
        health_error = self._check_lab_health()
        if health_error is not None:
            return QuiesceReceipt(ok=False, mutability=self.mutability, error=health_error)
        try:
            base_url = self._current_base_url()
        except ValueError as error:
            return QuiesceReceipt(ok=False, mutability=self.mutability, error=str(error))

        kernel_dir = self._current_kernel_dir()
        pre_snapshot_seq = client.current_seq(kernel_dir) if kernel_dir else 0
        pause_ok, _status, pause_payload, pause_error = _http_json_request(
            base_url, "POST", "/pause",
            query=[("mode", self._pause_mode), ("clear_cache", "false")], timeout=self._timeout,
        )
        if not pause_ok or (isinstance(pause_payload, dict) and pause_payload.get("status") not in (None, "paused")):
            detail = pause_error or f"unexpected /pause response: {pause_payload!r}"
            return QuiesceReceipt(ok=False, mutability=self.mutability, error=f"pause failed: {detail}")

        if self._release_kv_only:
            park_ok, _status, _payload, park_error = _http_json_request(
                base_url, "POST", "/release_kv_cache_memory", timeout=self._timeout
            )
            released, park_mode = ("kv_cache",), "release_kv_cache_memory"
        else:
            park_ok, _status, _payload, park_error = _http_json_request(
                base_url, "POST", "/sleep",
                query=[("level", str(self._sleep_level)), ("mode", self._pause_mode)], timeout=self._timeout,
            )
            released = ("weights", "kv_cache") if self._sleep_level >= 1 else ()
            park_mode = f"sleep_level_{self._sleep_level}"

        if not park_ok:
            self._rollback_resume(base_url)
            return QuiesceReceipt(ok=False, mutability=self.mutability,
                                  error=f"{park_mode} failed (rollback /resume attempted): {park_error}")

        check_ok, is_sleeping, check_error = _query_is_sleeping(base_url, self._timeout)
        if not check_ok or is_sleeping is not True:
            self._rollback_resume(base_url)
            reason = check_error or "GET /is_sleeping returned False after quiesce"
            return QuiesceReceipt(ok=False, mutability=self.mutability,
                                  error=f"quiesce verification failed: {reason}")

        weights_backend = (
            self._weights_backend if self._weights_backend is not None
            else str(getattr(self._lab, "weights_backend", "") or "")
        )
        token = json.dumps({
            "ctl_seq": pre_snapshot_seq, "mode": park_mode,
            "sleep_level": self._sleep_level, "weights_backend": weights_backend or "none",
        }, sort_keys=True)
        return QuiesceReceipt(ok=True, released=released, mutability=self.mutability, token=token)

    def resume_after_restore(self, receipt: QuiesceReceipt) -> dict[str, Any]:
        """Wake memory, resume scheduler, verify awake state, and confirm `adapter.host` is healthy."""
        if not receipt.ok:
            return {"ok": False, "error": f"cannot resume from failed quiesce receipt: {receipt.error}",
                    "host_ping_ok": False}
        try:
            base_url = self._current_base_url()
        except ValueError as error:
            return {"ok": False, "error": str(error), "host_ping_ok": False}
        token_info: dict[str, Any] = {}
        if receipt.token:
            with contextlib.suppress(ValueError):
                decoded = json.loads(receipt.token)
                token_info = decoded if isinstance(decoded, dict) else {}
        raw_ctl_seq = token_info.get("ctl_seq")
        min_seq = int(raw_ctl_seq) if isinstance(raw_ctl_seq, int) and raw_ctl_seq > 0 else None
        # Pass no `tags` for full wake (GMSWorker.wake_up expects None -> weights + kv_cache);
        # pass `?tags=kv_cache` when only KV cache was discarded via `/release_kv_cache_memory`.
        wake_query = [("tags", "kv_cache")] if receipt.released == ("kv_cache",) else None
        outcome = _wake_resume_and_verify(
            base_url, self._make_control_call(min_seq=min_seq), self._modules,
            wake_query=wake_query, timeout=self._timeout, label="post-restore",
        )
        if outcome.get("ok"):
            server = getattr(self._lab, "server", None) if self._lab is not None else None
            if callable(getattr(server, "refresh_start_ticks", None)):
                server.refresh_start_ticks()
            outcome["resumed"] = list(receipt.released)
        return outcome


def park_shadow_server(
    server: Any, *, sleep_level: int = 1, pause_mode: str = "keep", timeout: float = 30.0
) -> dict[str, Any]:
    """Park a standby shadow server via `POST /sleep?level=1&mode=keep` and verify `is_sleeping`."""
    _validate_sleep_args(sleep_level, pause_mode)
    if hasattr(server, "alive") and not server.alive():
        return {"ok": False, "error": "shadow server is not alive"}
    try:
        base_url = _resolve_base_url(server)
    except ValueError as error:
        return {"ok": False, "error": str(error)}
    sleep_start = time.monotonic()
    sleep_ok, _status, _payload, sleep_error = _http_json_request(
        base_url, "POST", "/sleep",
        query=[("level", str(sleep_level)), ("mode", pause_mode)], timeout=timeout,
    )
    sleep_seconds = round(time.monotonic() - sleep_start, 6)
    if not sleep_ok:
        return {"ok": False, "error": f"shadow sleep failed: {sleep_error}", "sleep_s": sleep_seconds}
    check_ok, is_sleeping, check_error = _query_is_sleeping(base_url, timeout)
    if not check_ok or is_sleeping is not True:
        reason = check_error or "shadow GET /is_sleeping returned False after /sleep"
        return {"ok": False, "error": reason, "sleep_s": sleep_seconds, "is_sleeping": is_sleeping}
    return {"ok": True, "port": getattr(server, "port", server if isinstance(server, int) else None),
            "sleep_level": sleep_level, "sleep_s": sleep_seconds, "is_sleeping": True}


class VllmShadowPromoter:
    """Promote a parked standby Scratch-KV shadow server onto a `VllmLab` on failover."""

    def __init__(
        self, shadow_server: Any = None, *,
        shadow_provider: Callable[[Any, Any], Any] | None = None,
        replenish_fn: Callable[[Any, Any], Any] | None = None,
        timeout: float = 30.0, modules: Sequence[str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._shadows: collections.deque[Any] = collections.deque()
        if shadow_server is not None:
            self._shadows.append(shadow_server)
        self._shadow_provider, self._replenish_fn = shadow_provider, replenish_fn
        self._timeout = float(timeout)
        self._modules = tuple(modules) if modules is not None else DEFAULT_WORKER_MODULES

    def ready_count(self) -> int:
        with self._lock:
            return sum(
                1 for srv in self._shadows
                if srv is not None and (not hasattr(srv, "alive") or srv.alive())
            )

    def register_shadow(self, shadow_server: Any) -> None:
        if shadow_server is None:
            raise ValueError("shadow_server cannot be None")
        with self._lock:
            if shadow_server not in self._shadows:
                self._shadows.append(shadow_server)

    def park_shadow(
        self, shadow_server: Any, *, sleep_level: int = 1, pause_mode: str = "keep"
    ) -> dict[str, Any]:
        result = park_shadow_server(
            shadow_server, sleep_level=sleep_level, pause_mode=pause_mode, timeout=self._timeout
        )
        if result.get("ok"):
            with self._lock:
                if shadow_server not in self._shadows:
                    self._shadows.append(shadow_server)
        return result

    def _shadow_control_call(
        self, shadow_server: Any
    ) -> Callable[[dict[str, Any], float], dict[str, Any]]:
        if hasattr(shadow_server, "ctl") and callable(shadow_server.ctl):
            return lambda command, timeout: shadow_server.ctl(command, timeout=timeout)
        kernel_dir = getattr(shadow_server, "kernel_dir", None)
        if kernel_dir:
            return lambda command, timeout: client.send(kernel_dir, command, timeout=timeout)
        return lambda _command, _timeout: {"ok": False, "error": "shadow server has no kernel_dir or ctl"}

    def promote(self, lab: Any, weight_store: Any = None) -> dict[str, Any]:
        """Wake and verify the next alive parked shadow, then swap it into `lab`."""
        shadow_server = None
        saw_dead = False
        with self._lock:
            while self._shadows:
                candidate = self._shadows.popleft()
                if candidate is None:
                    continue
                if hasattr(candidate, "alive") and not candidate.alive():
                    saw_dead = True
                    continue
                shadow_server = candidate
                break
        if shadow_server is None and self._shadow_provider is not None:
            shadow_server = self._shadow_provider(lab, weight_store)
        if shadow_server is None:
            error_msg = (
                "standby shadow server is not alive"
                if saw_dead
                else "no parked standby shadow server available"
            )
            return {"ok": False, "error": error_msg}
        if hasattr(shadow_server, "alive") and not shadow_server.alive():
            return {"ok": False, "error": "standby shadow server is not alive"}
        try:
            base_url = _resolve_base_url(shadow_server)
        except ValueError as error:
            return {"ok": False, "error": str(error)}

        outcome = _wake_resume_and_verify(
            base_url, self._shadow_control_call(shadow_server), self._modules,
            wake_query=None, timeout=self._timeout, label="shadow",
        )
        if not outcome.get("ok"):
            return outcome

        old_server = getattr(lab, "server", None)
        lab.server, lab.port = shadow_server, shadow_server.port
        lab_class = type(lab)
        if hasattr(lab_class, "_replace_shared_server"):
            lab_class._replace_shared_server(lab, old_server, shadow_server)
        elif (getattr(lab_class, "_shared", None) is old_server and old_server is not None) or getattr(
            lab, "_was_shared", False
        ):
            lab_class._shared = shadow_server
            if hasattr(lab, "_was_shared"):
                lab._was_shared = False

        from .conformance_lab import HostBridge

        lab.bridge = HostBridge(lab._ctl)
        recovered = lab.session.recovered(lab.bridge)
        if not recovered.get("ok") or lab.session.poisoned():
            return {**outcome, "ok": False, "promoted_port": shadow_server.port,
                    "error": recovered.get("error") or "session recovery failed after shadow promotion"}
        return {**outcome, "reset": ["shadow promoted", "session"],
                "promoted_port": shadow_server.port, "promoted_pid": getattr(shadow_server, "pid", None)}

    def recovery_driver(
        self, *,
        replenish_wait_timeout_s: float = 0.0,
        fallback_driver: Any = None,
        ready_count_fn: Callable[[], int] | None = None,
    ) -> StandbyShadowDriver:
        effective_ready_count_fn = (
            ready_count_fn if ready_count_fn is not None
            else (self.ready_count if self._shadow_provider is None else None)
        )
        driver = StandbyShadowDriver(
            self.promote,
            replenish_fn=self._replenish_fn,
            replenish_wait_timeout_s=replenish_wait_timeout_s,
            fallback_driver=fallback_driver,
            ready_count_fn=effective_ready_count_fn,
        )
        driver.promoter = self  # type: ignore[attr-defined]
        return driver


def make_shadow_recovery_driver(
    shadow_server: Any = None, *,
    shadow_provider: Callable[[Any, Any], Any] | None = None,
    replenish_fn: Callable[[Any, Any], Any] | None = None,
    timeout: float = 30.0, modules: Sequence[str] | None = None,
    replenish_wait_timeout_s: float = 0.0,
    fallback_driver: Any = None,
    ready_count_fn: Callable[[], int] | None = None,
) -> StandbyShadowDriver:
    """Create a `StandbyShadowDriver` configured with `VllmShadowPromoter`."""
    return VllmShadowPromoter(
        shadow_server=shadow_server, shadow_provider=shadow_provider,
        replenish_fn=replenish_fn, timeout=timeout, modules=modules,
    ).recovery_driver(
        replenish_wait_timeout_s=replenish_wait_timeout_s,
        fallback_driver=fallback_driver,
        ready_count_fn=ready_count_fn,
    )
