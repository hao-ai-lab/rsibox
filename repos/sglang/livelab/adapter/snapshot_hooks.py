"""SGLang `QuiesceHooks` implementation for `ai-dynamo/snapshot` checkpoint/restore."""
from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from collections.abc import Sequence
from typing import Any

from rsibox.warm import Mutability, QuiesceReceipt

DEFAULT_WARMUP_WORKLOAD: dict[str, Any] = {
    "prompts": ["Warmup."],
    "max_new_tokens": 2,
    "temperature": 0.0,
}


async def _await_result(awaitable: Any) -> Any:
    return await awaitable


def _close_unawaited(value: Any) -> None:
    if inspect.iscoroutine(value):
        value.close()


def _resolve_maybe_awaitable(engine: Any, value: Any, *, timeout: float | None = None) -> Any:
    """Resolve a synchronous return value or an awaitable using SGLang's event loop."""
    if not inspect.isawaitable(value):
        return value

    event_loop = getattr(engine, "loop", None)
    if not (isinstance(event_loop, asyncio.AbstractEventLoop) and not event_loop.is_closed()):
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        fallback_loop = getattr(tokenizer_manager, "event_loop", None)
        if isinstance(fallback_loop, asyncio.AbstractEventLoop) and not fallback_loop.is_closed():
            event_loop = fallback_loop
        else:
            event_loop = None

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if event_loop is not None:
        if event_loop.is_running():
            if running_loop is event_loop:
                _close_unawaited(value)
                raise RuntimeError(
                    "cannot synchronously await SGLang coroutine on its own running event loop"
                )
            coroutine = value if inspect.iscoroutine(value) else _await_result(value)
            future = asyncio.run_coroutine_threadsafe(coroutine, event_loop)
            try:
                return future.result(timeout=timeout)
            except BaseException:
                future.cancel()
                raise
        if running_loop is not None:
            _close_unawaited(value)
            raise RuntimeError(
                "cannot run SGLang event loop while another event loop is running on this thread"
            )
        return event_loop.run_until_complete(value)

    if running_loop is not None:
        _close_unawaited(value)
        raise RuntimeError(
            "cannot run SGLang coroutine synchronously while an event loop is running on this thread"
        )
    coroutine = value if inspect.iscoroutine(value) else _await_result(value)
    return asyncio.run(coroutine)


def _check_call_outcome(result: Any, action_name: str) -> None:
    """Fail closed when an engine control call reports an explicit negative status."""
    if result is False:
        raise RuntimeError(f"{action_name} returned False")
    if isinstance(result, dict):
        if result.get("ok") is False:
            message = result.get("error") or f"{action_name} returned ok=False"
            raise RuntimeError(str(message))
        if result.get("success") is False:
            message = (
                result.get("message")
                or result.get("error")
                or f"{action_name} returned success=False"
            )
            raise RuntimeError(str(message))
    elif getattr(result, "ok", None) is False:
        message = (
            getattr(result, "error", None)
            or getattr(result, "message", None)
            or f"{action_name} returned ok=False"
        )
        raise RuntimeError(str(message))
    elif getattr(result, "success", None) is False:
        message = (
            getattr(result, "message", None)
            or getattr(result, "error", None)
            or f"{action_name} returned success=False"
        )
        raise RuntimeError(str(message))


def _invoke_with_tags(method: Any, tags: list[str]) -> Any:
    """Invoke SGLang memory-occupation methods supporting keyword or positional `tags`."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(tags=tags)

    tag_parameter = signature.parameters.get("tags")
    if tag_parameter is not None and tag_parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
        return method(tags=tags)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return method(tags=tags)
    return method(tags)


class SglangQuiesceHooks:
    """`rsibox.warm.QuiesceHooks` adapter for a live `SglangLab`."""

    mutability: Mutability = "readonly_weights"

    def __init__(
        self,
        lab: Any,
        *,
        release_tags: Sequence[str] | None = None,
        pause_mode: str = "retract",
        warmup_workload: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(release_tags, (str, bytes)):
            raise TypeError("release_tags must be a sequence of tag names, not a string")
        self.lab = lab
        self.pause_mode = str(pause_mode)
        self.warmup_workload = dict(warmup_workload) if warmup_workload is not None else None
        if release_tags is not None:
            self.release_tags: tuple[str, ...] = tuple(str(tag) for tag in release_tags)
        elif getattr(getattr(lab, "weight_store", None), "name", None) == "dynamo_gms":
            self.release_tags = ("weights", "kv_cache")
        else:
            self.release_tags = ("kv_cache",)
        self._validate_release_tags(self.release_tags)

    def _validate_release_tags(self, tags: Sequence[str] | None) -> None:
        if (
            tags
            and getattr(self.lab, "allow_nonpauseable_cuda_graphs", False)
            and getattr(getattr(self.lab, "weight_store", None), "name", None) == "dynamo_gms"
            and "cuda_graph" in tags
        ):
            raise ValueError(
                "cannot include 'cuda_graph' in release_tags when SglangLab uses "
                "allow_nonpauseable_cuda_graphs=True under Dynamo GMS; release only "
                "('weights', 'kv_cache')"
            )

    def _rpc_timeout(self) -> float | None:
        timeout = getattr(self.lab, "rpc_timeout", None)
        return float(timeout) if isinstance(timeout, (int, float)) and timeout > 0 else None

    def _import_io_module(self) -> Any:
        return importlib.import_module(self.lab.io_module)

    def _call_pause_generation(self, engine: Any) -> None:
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        if tokenizer_manager is None or not hasattr(tokenizer_manager, "pause_generation"):
            raise RuntimeError("sglang engine does not expose tokenizer_manager.pause_generation")
        io_module = self._import_io_module()
        request = io_module.PauseGenerationReqInput(mode=self.pause_mode)
        raw_result = tokenizer_manager.pause_generation(request)
        resolved = _resolve_maybe_awaitable(engine, raw_result, timeout=self._rpc_timeout())
        _check_call_outcome(resolved, "pause_generation")

    def _call_continue_generation(self, engine: Any) -> None:
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        if tokenizer_manager is None or not hasattr(tokenizer_manager, "continue_generation"):
            raise RuntimeError("sglang engine does not expose tokenizer_manager.continue_generation")
        io_module = self._import_io_module()
        request = io_module.ContinueGenerationReqInput()
        raw_result = tokenizer_manager.continue_generation(request)
        resolved = _resolve_maybe_awaitable(engine, raw_result, timeout=self._rpc_timeout())
        _check_call_outcome(resolved, "continue_generation")

    def warmup_for_snapshot(self) -> None:
        """Run one deterministic generation step and verify RSIBox rank channels are live."""
        if getattr(self.lab, "engine", None) is None:
            raise RuntimeError("sglang engine is not running")
        if getattr(self.lab, "dirty", False):
            raise RuntimeError("sglang lab control channel is dirty; respawn required")

        workload = (
            dict(self.warmup_workload)
            if self.warmup_workload is not None
            else dict(DEFAULT_WARMUP_WORKLOAD)
        )
        self.lab.run_generate(workload)
        probe_ok, probe_error = self.lab._probe_participants()
        if not probe_ok:
            raise RuntimeError(f"participant probe failed during warmup: {probe_error}")

    def quiesce_for_snapshot(self) -> QuiesceReceipt:
        """Pause generation (`mode='retract'` by default) and release disposable GPU memory."""
        engine = getattr(self.lab, "engine", None)
        if engine is None:
            return QuiesceReceipt(
                ok=False,
                mutability=self.mutability,
                error="sglang engine is not running",
            )
        if getattr(self.lab, "dirty", False):
            return QuiesceReceipt(
                ok=False,
                mutability=self.mutability,
                error="sglang lab control channel is dirty; respawn required",
            )

        tags = self.release_tags
        try:
            self._validate_release_tags(tags)
        except ValueError as error:
            return QuiesceReceipt(
                ok=False,
                mutability=self.mutability,
                error=f"{type(error).__name__}: {error}",
            )
        paused = False
        try:
            self._call_pause_generation(engine)
            paused = True
            if tags:
                if not hasattr(engine, "release_memory_occupation"):
                    raise RuntimeError("sglang engine does not expose release_memory_occupation")
                raw_result = _invoke_with_tags(engine.release_memory_occupation, list(tags))
                resolved = _resolve_maybe_awaitable(
                    engine, raw_result, timeout=self._rpc_timeout()
                )
                _check_call_outcome(resolved, "release_memory_occupation")
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            if paused:
                try:
                    self._call_continue_generation(engine)
                except Exception as rollback_error:
                    error_message = (
                        f"{error_message} (best-effort continue_generation rollback failed: "
                        f"{type(rollback_error).__name__}: {rollback_error})"
                    )
            return QuiesceReceipt(
                ok=False,
                mutability=self.mutability,
                error=error_message,
            )

        token = f"sglang:{self.pause_mode}:{','.join(tags)}"
        return QuiesceReceipt(
            ok=True,
            released=tags,
            mutability=self.mutability,
            token=token,
        )

    def resume_after_restore(self, receipt: QuiesceReceipt) -> dict[str, Any]:
        """Re-map released GPU memory, unpause generation, and verify RSIBox rank channels."""
        engine = getattr(self.lab, "engine", None)
        if engine is None:
            return {"ok": False, "error": "sglang engine is not running"}
        if getattr(self.lab, "dirty", False):
            return {"ok": False, "error": "sglang lab control channel is dirty; respawn required"}

        if receipt is not None and receipt.released is not None:
            tags = list(receipt.released)
        else:
            tags = list(self.release_tags)
        try:
            self._validate_release_tags(tags)
        except ValueError as error:
            return {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "resumed": [],
                "resume_memory_s": 0.0,
                "continue_s": 0.0,
                "participants_ok": False,
            }
        resume_memory_seconds = 0.0
        continue_seconds = 0.0

        try:
            if tags:
                if not hasattr(engine, "resume_memory_occupation"):
                    raise RuntimeError("sglang engine does not expose resume_memory_occupation")
                resume_start = time.perf_counter()
                raw_result = _invoke_with_tags(engine.resume_memory_occupation, tags)
                resolved = _resolve_maybe_awaitable(
                    engine, raw_result, timeout=self._rpc_timeout()
                )
                _check_call_outcome(resolved, "resume_memory_occupation")
                resume_memory_seconds = time.perf_counter() - resume_start
        except Exception as error:
            return {
                "ok": False,
                "error": f"resume_memory_occupation failed: {type(error).__name__}: {error}",
                "resumed": [],
                "resume_memory_s": round(resume_memory_seconds, 6),
                "continue_s": 0.0,
                "participants_ok": False,
            }

        try:
            continue_start = time.perf_counter()
            self._call_continue_generation(engine)
            continue_seconds = time.perf_counter() - continue_start
        except Exception as error:
            return {
                "ok": False,
                "error": f"continue_generation failed: {type(error).__name__}: {error}",
                "resumed": tags,
                "resume_memory_s": round(resume_memory_seconds, 6),
                "continue_s": round(continue_seconds, 6),
                "participants_ok": False,
            }

        try:
            probe_ok, probe_error = self.lab._probe_participants()
        except Exception as error:
            probe_ok = False
            probe_error = f"{type(error).__name__}: {error}"

        if not probe_ok:
            return {
                "ok": False,
                "error": f"participant probe failed after restore: {probe_error}",
                "resumed": tags,
                "resume_memory_s": round(resume_memory_seconds, 6),
                "continue_s": round(continue_seconds, 6),
                "participants_ok": False,
            }

        return {
            "ok": True,
            "resumed": tags,
            "resume_memory_s": round(resume_memory_seconds, 6),
            "continue_s": round(continue_seconds, 6),
            "participants_ok": True,
        }
