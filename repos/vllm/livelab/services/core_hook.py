"""Shared vLLM EngineCoreProc hook, loop-job primitive, and single control-channel mux.

Closes STL-2: `adapter/host.py` and `services/live_reconfig.py` register plugins here so
`vllm.v1.engine.core.EngineCoreProc` is patched once via one `AfterImportHook` and served by
at most one control-channel polling thread.
"""
from dataclasses import dataclass
import os
import sys
import threading
import time
from typing import Callable

try:
    from fastrsi import hooks, ipc
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import corelab  # noqa: F401,E402
    from fastrsi import hooks, ipc  # noqa: E402

TARGET_MODULE = "vllm.v1.engine.core"
_PLUGINS: dict[str, "CorePlugin"] = {}


class LoopJob:
    """Callable submitted to `EngineCoreProc.input_queue`; `gap=True` waits for empty batch_queue."""

    def __init__(self, fn: Callable, gap: bool = False):
        self.fn, self.gap = fn, gap
        self.done, self.started = threading.Event(), threading.Event()
        self.lock, self.cancelled, self.value, self.error = threading.Lock(), False, None, None

    def run(self):
        with self.lock:
            if self.cancelled:
                self.done.set()
                return
            self.started.set()
        try:
            self.value = self.fn()
        except BaseException as exc:  # noqa: BLE001 - propagated to caller
            self.error = exc
        self.done.set()


class LateJob(TimeoutError):
    """The engine loop started the job, but it did not finish before the caller's deadline."""


def in_loop(core, tag: str, fn: Callable, *, gap: bool = False, timeout: float = 60.0,
            loop_thread: threading.Thread | None = None):
    """Run `fn` on the engine core's busy-loop thread and return its result."""
    if loop_thread is not None and threading.current_thread() is loop_thread:
        return fn()
    job = LoopJob(fn, gap=gap)
    core.input_queue.put_nowait((tag, job))
    if not job.started.wait(timeout):
        with job.lock:
            if not job.started.is_set():
                job.cancelled = True
                raise TimeoutError(f"the engine loop did not run the job within {timeout} s")
    if not job.done.wait(timeout):
        raise LateJob(f"the engine loop started the job but it did not finish within {timeout} s")
    if job.error is not None:
        raise job.error
    return job.value


@dataclass(frozen=True)
class CorePlugin:
    """One consumer attached to the single `EngineCoreProc` hook."""
    name: str
    tag: str
    on_loop_start: Callable
    on_request: Callable | None = None
    pre_step: Callable | None = None
    should_hold: Callable | None = None
    on_hold: Callable | None = None
    before_step: Callable | None = None
    after_step: Callable | None = None
    on_patch: Callable | None = None


class _ChannelMux:
    """Single daemon thread polling one or more `ctl.json` -> `reply/<seq>.json` channels."""

    def __init__(self, specs):
        self._channels, seen, periods = [], {}, []
        for name, root, handler, poll_s in specs:
            abs_root = os.path.abspath(root)
            if abs_root in seen:
                print(f"FASTRSI_CORE_HOOK: refusing duplicate control dir {abs_root!r} for "
                      f"{name!r} (already owned by {seen[abs_root]!r})", file=sys.stderr, flush=True)
                continue
            seen[abs_root] = name
            reply_dir = os.path.join(abs_root, "reply")
            os.makedirs(reply_dir, exist_ok=True)
            self._channels.append({
                "name": name, "root": abs_root, "ctl_path": os.path.join(abs_root, "ctl.json"),
                "reply_dir": reply_dir, "handler": handler, "last": 0,
            })
            periods.append(poll_s)
        self.poll_s = min(periods, default=0.005)
        self._stop, self._thread = threading.Event(), None

    def poll_once(self) -> None:
        for ch in self._channels:
            cmd = ipc.read_json(ch["ctl_path"])
            if not isinstance(cmd, dict):
                continue
            try:
                seq = int(cmd.get("seq", 0))
            except (TypeError, ValueError):
                seq = 0
            if seq <= ch["last"]:
                continue
            ch["last"] = seq
            try:
                reply = ch["handler"](cmd)
            except Exception as exc:  # noqa: BLE001
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            ipc.atomic_write_json(os.path.join(ch["reply_dir"], f"{seq}.json"), reply)

    def start(self):
        if self._channels and self._thread is None:
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="fastrsi-vllm-core-ctl"
            )
            self._thread.start()
        return self._thread

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.poll_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _notify_patch(proc_cls) -> None:
    for plugin in list(_PLUGINS.values()):
        if plugin.on_patch is not None:
            plugin.on_patch(proc_cls)


def register(plugin: CorePlugin) -> None:
    """Register (or replace) a plugin by name; notifies `on_patch` if core is already patched."""
    _PLUGINS[plugin.name] = plugin
    mod = sys.modules.get(TARGET_MODULE)
    proc_cls = getattr(mod, "EngineCoreProc", None) if mod is not None else None
    if proc_cls is not None and getattr(proc_cls, "_fastrsi_core_hook", False):
        if plugin.on_patch is not None:
            plugin.on_patch(proc_cls)


def _patch_core(mod) -> None:
    """Idempotently wrap `EngineCoreProc` once across all registered plugins."""
    P = mod.EngineCoreProc
    if getattr(P, "_fastrsi_core_hook", False):
        _notify_patch(P)
        return
    orig_handle, orig_step, orig_loop = (
        P._handle_client_request, P._process_engine_step, P.run_busy_loop,
    )

    def _handle_client_request(self, request_type, request):
        for plugin in list(_PLUGINS.values()):
            if request_type == plugin.tag:
                return plugin.on_request(self, request) if plugin.on_request else request.run()
        return orig_handle(self, request_type, request)

    def _process_engine_step(self):
        plugins = list(_PLUGINS.values())
        for plugin in plugins:
            if plugin.pre_step is not None and plugin.pre_step(self) is False:
                return False
        holders = [p for p in plugins if p.should_hold is not None and p.should_hold(self)]
        hold = bool(holders)
        if hold:
            self.scheduler.has_requests = lambda: False
            for plugin in holders:
                if plugin.on_hold is not None:
                    plugin.on_hold(self)
        for plugin in plugins:
            if plugin.before_step is not None:
                plugin.before_step(self, hold)
        step_start = time.perf_counter()
        try:
            result = orig_step(self)
        finally:
            if hold:
                del self.scheduler.has_requests
        for plugin in plugins:
            if plugin.after_step is not None:
                plugin.after_step(self, step_start, result)
        return result

    def run_busy_loop(self):
        specs = []
        for plugin in list(_PLUGINS.values()):
            channel = plugin.on_loop_start(self)
            if channel is not None:
                specs.append((plugin.name, *channel))
        mux = _ChannelMux(specs)
        self._fastrsi_channel_mux = mux
        mux.start()
        try:
            return orig_loop(self)
        finally:
            mux.stop()

    P._handle_client_request = _handle_client_request
    P._process_engine_step = _process_engine_step
    P.run_busy_loop = run_busy_loop
    P._fastrsi_core_hook = True
    _notify_patch(P)


def install():
    """Install the single stack-safe `AfterImportHook` for `vllm.v1.engine.core`."""
    return hooks.install_import_hook(TARGET_MODULE, _patch_core, once=False)
