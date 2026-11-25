"""Standby Scratch-KV shadow failover for `TrtllmLab` (`rsibox.warm.StandbyShadowDriver`).

Parks standby `TrtllmLab` / `ToyTrtllmLab` worker(s) via `TrtllmQuiesceHooks` (releasing
disposable `kv_cache` GPU memory while keeping weights and engine state warm). On primary
crash/poison (`lab.reset(kind="respawn")`), `TrtllmShadowPromoter` skips any dead or
incompatible head shadows, tears down the primary's isolated process-group workers only
after acquiring a validated candidate, resumes the parked shadow's KV cache, transfers
worker/MPI ownership cleanly onto `lab`, rebuilds the `RecaptureCoordinator` and `Session`,
and recovers live operation in milliseconds.
"""
from __future__ import annotations

import collections
import contextlib
import os
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from kernel import coordinator, session as kernel_session
from rsibox.lab import RecaptureCoordinator
from rsibox.warm import AsyncShadowReplenisher, QuiesceReceipt, StandbyShadowDriver
from rsibox.warm.recovery import _resolve_async_replenisher

from .snapshot_hooks import TrtllmQuiesceHooks

DEFAULT_SHADOW_RELEASE_TAGS: tuple[str, ...] = ("kv_cache",)

__all__ = [
    "DEFAULT_SHADOW_RELEASE_TAGS",
    "ParkedTrtllmShadow",
    "TrtllmShadowPromoter",
    "park_shadow_lab",
]


@dataclass(frozen=True)
class ParkedTrtllmShadow:
    """A parked standby `TrtllmLab` paired with its `TrtllmQuiesceHooks` and receipt."""

    lab: Any
    hooks: TrtllmQuiesceHooks
    receipt: QuiesceReceipt


def _worker_alive(worker: Any) -> bool:
    proc = getattr(worker, "process", None)
    if proc is not None and hasattr(proc, "poll"):
        try:
            return proc.poll() is None
        except Exception:
            return False
    # MPI `_RankClient` has no direct `.process`; verify its IPC endpoint still exists.
    endpoint = getattr(worker, "endpoint", None)
    if isinstance(endpoint, str) and endpoint:
        return os.path.exists(endpoint)
    return True


def _resolve_shadow_ranks(lab: Any) -> int:
    raw = getattr(lab, "rank_count", None)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    workers = getattr(lab, "workers", None)
    return len(workers) if isinstance(workers, (list, tuple)) else 0


def _check_clean_shadow_lab(shadow_lab: Any) -> str | None:
    """Return an error string if `shadow_lab` is not live, clean, and free of active edits."""
    workers = getattr(shadow_lab, "workers", None)
    if not isinstance(workers, (list, tuple)) or not workers:
        return "standby shadow lab has no running workers"
    driver = getattr(shadow_lab, "_driver", None)
    if driver is not None and not _worker_alive(driver):
        return "standby shadow lab MPI driver process died"
    for worker in workers:
        if not _worker_alive(worker):
            return f"standby shadow lab worker {getattr(worker, 'name', '?')} died"
    if getattr(shadow_lab, "_dirty", False) or getattr(shadow_lab, "dirty", False):
        return "standby shadow lab control channel is dirty"
    session = getattr(shadow_lab, "session", None)
    if session is not None:
        poisoned = session.poisoned() if callable(getattr(session, "poisoned", None)) else None
        if poisoned:
            return f"standby shadow lab session is poisoned ({len(poisoned)} finding(s))"
        stack = getattr(session, "stack", None)
        if stack:
            return f"standby shadow lab has {len(stack)} active live edit(s); undo before parking/promoting"
    live_text = getattr(shadow_lab, "live_text", None)
    if isinstance(live_text, list) and len(live_text) > 1:
        return "standby shadow lab has uncommitted live_text edits; undo before parking/promoting"
    return None


def _shadow_compatibility_error(primary: Any, shadow: Any) -> str | None:
    """Return an error string if `shadow` is incompatible with `primary`."""
    if _resolve_shadow_ranks(shadow) != _resolve_shadow_ranks(primary):
        return f"shadow rank_count mismatch ({_resolve_shadow_ranks(shadow)} != {_resolve_shadow_ranks(primary)})"
    p_mode, s_mode = getattr(primary, "engine_mode", None), getattr(shadow, "engine_mode", None)
    if p_mode is not None and s_mode is not None and p_mode != s_mode:
        return f"shadow engine_mode mismatch ({s_mode!r} != {p_mode!r})"
    p_pkg, s_pkg = getattr(primary, "overlay_package", None), getattr(shadow, "overlay_package", None)
    if p_pkg is not None and s_pkg is not None and p_pkg != s_pkg:
        return f"shadow overlay_package mismatch ({s_pkg!r} != {p_pkg!r})"
    p_cfg = getattr(primary, "engine_config", None)
    s_cfg = getattr(shadow, "engine_config", None)
    if isinstance(p_cfg, dict) and isinstance(s_cfg, dict):
        if p_cfg.get("model") != s_cfg.get("model"):
            return f"shadow model mismatch ({s_cfg.get('model')!r} != {p_cfg.get('model')!r})"
        if (p_cfg.get("extra_kwargs") or {}) != (s_cfg.get("extra_kwargs") or {}):
            return "shadow extra_kwargs mismatch"
    p_mods = getattr(primary, "_tracked_modules", None)
    s_mods = getattr(shadow, "_tracked_modules", None)
    if p_mods is not None and s_mods is not None and tuple(p_mods) != tuple(s_mods):
        return f"shadow tracked modules mismatch ({tuple(s_mods)!r} != {tuple(p_mods)!r})"
    return None


def _close_shadow_quietly(shadow_lab: Any) -> None:
    if shadow_lab is None:
        return
    with contextlib.suppress(Exception):
        shadow_lab.close()


def _coerce_parked_shadow(item: Any) -> ParkedTrtllmShadow | None:
    if item is None:
        return None
    if isinstance(item, ParkedTrtllmShadow):
        entry = item
    elif isinstance(item, tuple) and len(item) == 3:
        shadow_lab, hooks, receipt = item
        entry = ParkedTrtllmShadow(lab=shadow_lab, hooks=hooks, receipt=receipt)
    else:
        raise TypeError(
            "shadow entry must be a ParkedTrtllmShadow or a (shadow_lab, hooks, receipt) tuple"
        )
    if not hasattr(entry.hooks, "resume_after_restore") or not callable(entry.hooks.resume_after_restore):
        raise TypeError("shadow hooks must implement resume_after_restore(receipt)")
    if getattr(entry.hooks, "lab", entry.lab) is not entry.lab:
        raise ValueError("shadow hooks.lab does not match shadow_lab")
    if not isinstance(entry.receipt, QuiesceReceipt):
        raise TypeError(f"shadow receipt must be a QuiesceReceipt, got {type(entry.receipt).__name__}")
    return entry


def park_shadow_lab(
    shadow_lab: Any,
    *,
    release_tags: Sequence[str] = DEFAULT_SHADOW_RELEASE_TAGS,
) -> dict[str, Any]:
    """Quiesce `shadow_lab` (releasing disposable KV cache) into a parked standby entry."""
    if isinstance(release_tags, (str, bytes)):
        return {"ok": False, "error": "release_tags must be a sequence of tag names, not a string"}

    shadow_lock = getattr(shadow_lab, "lock", None)
    ctx = shadow_lock if shadow_lock is not None else contextlib.nullcontext()
    with ctx:
        clean_err = _check_clean_shadow_lab(shadow_lab)
        if clean_err is not None:
            return {"ok": False, "error": clean_err}

        t0 = time.perf_counter()
        try:
            hooks = TrtllmQuiesceHooks(shadow_lab, release_tags=tuple(release_tags))
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "park_s": round(time.perf_counter() - t0, 6),
            }

        receipt = hooks.quiesce_for_snapshot()
        park_s = round(time.perf_counter() - t0, 6)
        if not receipt.ok:
            return {
                "ok": False,
                "error": receipt.error or "shadow quiesce failed",
                "park_s": park_s,
                "receipt": receipt,
            }
        return {
            "ok": True,
            "shadow": ParkedTrtllmShadow(lab=shadow_lab, hooks=hooks, receipt=receipt),
            "hooks": hooks,
            "receipt": receipt,
            "released": list(receipt.released),
            "park_s": park_s,
        }


def _transfer_engine_ownership(dst_lab: Any, src_lab: Any) -> None:
    """Move live TRT-LLM worker/MPI ownership cleanly from `src_lab` onto `dst_lab`."""
    dst_lab.workers = list(getattr(src_lab, "workers", ()) or ())
    dst_lab._driver = getattr(src_lab, "_driver", None)
    dst_lab._mpi_hook_dir = getattr(src_lab, "_mpi_hook_dir", None)
    dst_lab._mpi_sock_dir = getattr(src_lab, "_mpi_sock_dir", None)
    dst_lab.rank_count = _resolve_shadow_ranks(src_lab)
    if hasattr(src_lab, "transport"):
        dst_lab.transport = src_lab.transport
    dst_lab.base_text = dict(getattr(src_lab, "base_text", {}) or {})
    src_modules = getattr(src_lab, "modules", None)
    if src_modules:
        dst_lab.modules = list(src_modules)
    src_tracked = getattr(src_lab, "_tracked_modules", None)
    if src_tracked is not None:
        dst_lab._tracked_modules = tuple(src_tracked)
    dst_lab._dirty = False

    # Detach shell shadow lab so closing/GC'ing it cannot kill promoted workers or remove MPI dirs.
    src_lab.workers = []
    src_lab._driver = None
    src_lab._mpi_hook_dir = None
    src_lab._mpi_sock_dir = None
    src_lab._dirty = True


def _rebuild_coordinator(lab: Any) -> RecaptureCoordinator:
    participants = [
        coordinator.CallParticipant(worker.name, worker.call, bounded=True)
        for worker in lab.workers
    ]
    return RecaptureCoordinator(
        participants,
        gap=lab._step_gap,
        writer=getattr(lab, "writer", None),
        rebuild=lab._rebuild_all_ranks,
    )


class _TrtllmShadowDriver(StandbyShadowDriver):
    """`StandbyShadowDriver` that validates/promotes inside `TrtllmShadowPromoter` before stopping primary."""

    def __init__(
        self,
        promoter: TrtllmShadowPromoter,
        *,
        replenish_fn: Callable[[Any, Any], Any] | None = None,
        replenish_wait_timeout_s: float = 0.0,
        fallback_driver: Any = None,
        ready_count_fn: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(
            promoter.promote,
            replenish_fn=replenish_fn,
            replenish_wait_timeout_s=replenish_wait_timeout_s,
            fallback_driver=fallback_driver,
            ready_count_fn=ready_count_fn,
        )
        self.promoter = promoter

    def _run_fallback(
        self, lab: Any, store: Any, *, reason: str, trigger_replenish: bool = False
    ) -> dict[str, Any]:
        assert self._fallback_driver is not None
        fallback_result = self._fallback_driver.respawn(lab, store, reason=reason)
        fallback_result["fallback_from"] = self.name
        if trigger_replenish and not self.promoter.is_closed() and fallback_result.get("ok"):
            rep_val, rep_err = self._safe_replenish(lab, store)
            if rep_err is not None:
                fallback_result["replenish_error"] = rep_err
            elif rep_val is not None:
                fallback_result["replenish"] = rep_val
        return fallback_result

    def _safe_replenish(self, lab: Any, store: Any) -> tuple[Any, str | None]:
        if self._replenish_fn is None or self.promoter.is_closed():
            return None, None
        try:
            return self._replenish_fn(lab, store), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _closed_response(self, lab: Any, store: Any, *, reason: str, started_at: float) -> dict[str, Any]:
        if self._fallback_driver is not None:
            return self._run_fallback(lab, store, reason=reason, trigger_replenish=False)
        return {
            "ok": False,
            "reset": [],
            "driver": self.name,
            "recovery_mode": self.name,
            "reason": reason,
            "error": "TrtllmShadowPromoter is closed",
            "replenish": "closed",
            "cost_s": round(time.perf_counter() - started_at, 6),
        }

    def _guard_async_replenisher(
        self, async_replenisher: Any, lab: Any, store: Any, *, reason: str, started_at: float
    ) -> dict[str, Any] | None:
        if isinstance(async_replenisher, AsyncShadowReplenisher):
            self.promoter._track_replenisher(async_replenisher)
        if self.promoter.is_closed():
            return self._closed_response(lab, store, reason=reason, started_at=started_at)
        state = async_replenisher.status()
        is_closed = bool(state.get("closed"))
        has_ready, probe_error = (None, None) if is_closed else self._probe_ready_standby()
        if has_ready is True and not is_closed and not self.promoter.is_closed():
            return None

        if (
            not is_closed
            and not self.promoter.is_closed()
            and probe_error is None
            and (state.get("in_flight") or state.get("pending"))
            and self._replenish_wait_timeout_s > 0.0
        ):
            if isinstance(async_replenisher, AsyncShadowReplenisher):
                async_replenisher.wait(
                    timeout_s=self._replenish_wait_timeout_s,
                    until=lambda: self.promoter.is_closed() or (
                        self._ready_count_fn is not None and self._probe_ready_standby()[0] is True
                    ),
                )
            else:
                async_replenisher.wait(timeout_s=self._replenish_wait_timeout_s)
            if self.promoter.is_closed():
                return self._closed_response(lab, store, reason=reason, started_at=started_at)
            state = async_replenisher.status()
            is_closed = bool(state.get("closed"))
            has_ready, probe_error = (None, None) if is_closed else self._probe_ready_standby()
            if has_ready is True and not is_closed:
                return None

        if self.promoter.is_closed():
            return self._closed_response(lab, store, reason=reason, started_at=started_at)
        still_busy = bool(state.get("in_flight") or state.get("pending"))
        last_failed = state.get("attempts", 0) > 0 and (
            state.get("last_ok") is False or state.get("last_error") is not None
        )
        if not is_closed and not still_busy and not last_failed and has_ready is not False:
            return None

        if self._fallback_driver is not None:
            should_rep = (last_failed or (has_ready is False and probe_error is None)) and not still_busy and not is_closed
            return self._run_fallback(lab, store, reason=reason, trigger_replenish=should_rep)

        rep_err: str | None = None
        if is_closed:
            err_text, rep_status = "standby_not_ready: replenisher is closed", "closed"
        elif probe_error is not None:
            err_text, rep_status = f"standby_not_ready: ready_count_fn invalid ({probe_error})", None
        elif still_busy:
            err_text, rep_status = "standby_not_ready: replenishment still in flight", "in_flight"
        elif last_failed:
            err_text = f"standby_not_ready: last replenishment failed ({state.get('last_error')})"
            rep_status, rep_err = self._safe_replenish(lab, store)
        else:
            err_text = "standby_not_ready: no parked standby shadow lab available"
            rep_status, rep_err = self._safe_replenish(lab, store)
        result: dict[str, Any] = {
            "ok": False,
            "reset": [],
            "driver": self.name,
            "recovery_mode": self.name,
            "reason": reason,
            "error": err_text,
            "replenish": rep_status,
            "replenisher_status": state,
            "cost_s": round(time.perf_counter() - started_at, 6),
        }
        if rep_err is not None:
            result["replenish_error"] = rep_err
        return result

    def respawn(self, lab: Any, store: Any = None, *, reason: str = "reset_respawn") -> dict[str, Any]:
        t0 = time.perf_counter()
        if self.promoter.is_closed():
            return self._closed_response(lab, store, reason=reason, started_at=t0)

        async_replenisher = _resolve_async_replenisher(self._replenish_fn)
        if async_replenisher is not None:
            guard_result = self._guard_async_replenisher(
                async_replenisher, lab, store, reason=reason, started_at=t0
            )
            if guard_result is not None:
                return guard_result
        else:
            has_ready, probe_error = self._probe_ready_standby()
            if has_ready is False:
                if self._fallback_driver is not None:
                    return self._run_fallback(lab, store, reason=reason, trigger_replenish=(probe_error is None))
                err_text = (
                    f"standby_not_ready: ready_count_fn invalid ({probe_error})"
                    if probe_error is not None
                    else "standby_not_ready: no parked standby shadow lab available"
                )
                return {
                    "ok": False,
                    "reset": [],
                    "driver": self.name,
                    "recovery_mode": self.name,
                    "reason": reason,
                    "error": err_text,
                    "cost_s": round(time.perf_counter() - t0, 6),
                }

        promoted = self.promoter.promote(lab, store) or {}
        promote_cost_s = round(time.perf_counter() - t0, 6)
        session = getattr(lab, "session", None)
        poisoned = bool(session.poisoned()) if session is not None else False
        ok = bool(promoted.get("ok", False)) and not poisoned

        if not ok and self._fallback_driver is not None:
            fb = self._run_fallback(lab, store, reason=reason, trigger_replenish=True)
            fb["promote_error"] = promoted.get("error")
            return fb

        rep_outcome, rep_err = (self._safe_replenish(lab, store) if ok else (None, None))
        reserved = {"ok", "reset", "driver", "reason", "cost_s", "replenish"}
        extra = {k: v for k, v in promoted.items() if k not in reserved}
        if ok and async_replenisher is not None and rep_err is None and rep_outcome is not None:
            extra["replenish"] = rep_outcome
        if rep_err is not None:
            extra["replenish_error"] = rep_err
        return {
            "ok": ok,
            "reset": list(promoted.get("reset", ["shadow promoted", "session"])) if ok else [],
            "driver": self.name,
            "reason": reason,
            "cost_s": promote_cost_s,
            **extra,
        }


class TrtllmShadowPromoter:
    """Promote a parked standby Scratch-KV `TrtllmLab` onto a primary `TrtllmLab` on failover."""

    def __init__(
        self,
        shadow: ParkedTrtllmShadow | tuple[Any, TrtllmQuiesceHooks, QuiesceReceipt] | None = None,
        *,
        shadow_provider: Callable[[Any, Any], Any] | None = None,
        replenish_fn: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        if shadow_provider is not None and not callable(shadow_provider):
            raise TypeError(f"shadow_provider must be callable or None, got {type(shadow_provider).__name__}")
        if replenish_fn is not None and not callable(replenish_fn):
            raise TypeError(f"replenish_fn must be callable or None, got {type(replenish_fn).__name__}")
        self._lock = threading.Lock()
        self._provider_lock = threading.Lock()
        self._closed = False
        self._shadows: collections.deque[ParkedTrtllmShadow] = collections.deque()
        self._attached_replenishers: weakref.WeakSet[Any] = weakref.WeakSet()
        if shadow is not None:
            self.register_shadow(shadow)
        self._shadow_provider = shadow_provider
        self._replenish_fn = replenish_fn

    def _track_replenisher(self, replenisher: Any) -> None:
        if replenisher is None:
            return
        with self._lock:
            self._attached_replenishers.add(replenisher)
            already_closed = self._closed
        if already_closed:
            self._wake_replenisher(replenisher)

    @staticmethod
    def _wake_replenisher(replenisher: Any) -> None:
        wake_fn = getattr(replenisher, "wake", None)
        if callable(wake_fn):
            with contextlib.suppress(Exception):
                wake_fn()

    def _entry_is_clean_standby(self, entry: ParkedTrtllmShadow) -> bool:
        return (
            getattr(entry.hooks, "lab", entry.lab) is entry.lab
            and bool(entry.receipt.ok)
            and _check_clean_shadow_lab(entry.lab) is None
            and _resolve_shadow_ranks(entry.lab) > 0
        )

    def pending_count(self) -> int:
        with self._lock:
            if self._closed:
                return 0
            return sum(1 for entry in self._shadows if self._entry_is_clean_standby(entry))

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def register_shadow(
        self,
        shadow: ParkedTrtllmShadow | tuple[Any, TrtllmQuiesceHooks, QuiesceReceipt],
    ) -> ParkedTrtllmShadow:
        entry = _coerce_parked_shadow(shadow)
        if entry is None:
            raise ValueError("shadow cannot be None")
        if not entry.receipt.ok:
            raise ValueError(
                f"cannot register shadow with unsuccessful quiesce receipt: {entry.receipt.error}"
            )
        clean_err = _check_clean_shadow_lab(entry.lab)
        if clean_err is not None:
            raise ValueError(f"cannot register shadow: {clean_err}")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot register shadow on a closed TrtllmShadowPromoter")
            if any(existing.lab is entry.lab for existing in self._shadows):
                raise ValueError("standby shadow lab is already registered in promoter queue")
            self._shadows.append(entry)
        return entry

    def park_shadow(
        self,
        shadow_lab: Any,
        *,
        release_tags: Sequence[str] = DEFAULT_SHADOW_RELEASE_TAGS,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return {"ok": False, "error": "TrtllmShadowPromoter is closed"}
            if any(existing.lab is shadow_lab for existing in self._shadows):
                return {"ok": False, "error": "standby shadow lab is already registered in promoter queue"}
        parked = park_shadow_lab(shadow_lab, release_tags=release_tags)
        if not parked.get("ok"):
            return parked
        entry = parked["shadow"]
        with self._lock:
            if self._closed:
                _close_shadow_quietly(shadow_lab)
                return {
                    "ok": False,
                    "error": "TrtllmShadowPromoter was closed while parking shadow",
                    "park_s": parked.get("park_s", 0.0),
                }
            if any(existing.lab is shadow_lab for existing in self._shadows):
                return {
                    "ok": False,
                    "error": "standby shadow lab is already registered in promoter queue",
                    "park_s": parked.get("park_s", 0.0),
                }
            self._shadows.append(entry)
        return parked

    def _pop_next_candidate(self, lab: Any, store: Any) -> tuple[ParkedTrtllmShadow | None, int, str | None]:
        skipped_dead = 0
        last_mismatch: str | None = None
        while True:
            with self._lock:
                if self._closed:
                    return None, skipped_dead, "TrtllmShadowPromoter is closed"
                entry = self._shadows.popleft() if self._shadows else None
            if entry is None:
                break
            if entry.lab is lab:
                skipped_dead += 1
                last_mismatch = "standby shadow cannot be the primary lab itself"
                continue
            if not self._entry_is_clean_standby(entry):
                skipped_dead += 1
                _close_shadow_quietly(entry.lab)
                continue
            compat_err = _shadow_compatibility_error(lab, entry.lab)
            if compat_err is not None:
                skipped_dead += 1
                last_mismatch = compat_err
                _close_shadow_quietly(entry.lab)
                continue
            return entry, skipped_dead, None

        if self._shadow_provider is not None:
            with self._provider_lock:
                if self.is_closed():
                    return None, skipped_dead, "TrtllmShadowPromoter is closed"
                try:
                    provided = _coerce_parked_shadow(self._shadow_provider(lab, store))
                except Exception as exc:
                    return None, skipped_dead, f"shadow_provider failed: {type(exc).__name__}: {exc}"
                if provided is None:
                    return None, skipped_dead, last_mismatch or "no parked standby shadow lab available"
                if self.is_closed():
                    _close_shadow_quietly(provided.lab)
                    return None, skipped_dead, "TrtllmShadowPromoter is closed"
                if provided.lab is lab:
                    return None, skipped_dead, "shadow_provider returned the primary lab itself"
                if not self._entry_is_clean_standby(provided):
                    _close_shadow_quietly(provided.lab)
                    return None, skipped_dead, "shadow_provider returned an unhealthy standby shadow lab"
                compat_err = _shadow_compatibility_error(lab, provided.lab)
                if compat_err is not None:
                    _close_shadow_quietly(provided.lab)
                    return None, skipped_dead, compat_err
                return provided, skipped_dead, None

        return None, skipped_dead, last_mismatch or "no parked standby shadow lab available"

    def promote(self, lab: Any, store: Any = None) -> dict[str, Any]:
        """Promote the next clean, compatible standby `TrtllmLab` onto `lab`."""
        t0 = time.perf_counter()
        if self.is_closed():
            return {"ok": False, "error": "TrtllmShadowPromoter is closed"}

        expected_ranks = _resolve_shadow_ranks(lab)
        if expected_ranks <= 0:
            return {"ok": False, "error": "primary lab has invalid rank_count"}

        lab_lock = getattr(lab, "lock", None)
        with (lab_lock if lab_lock is not None else contextlib.nullcontext()):
            primary_stopped = False
            stop_primary_s = 0.0
            total_skipped = 0
            last_error: str | None = None

            while True:
                candidate, skipped, pop_err = self._pop_next_candidate(lab, store)
                total_skipped += skipped
                if candidate is None:
                    if primary_stopped:
                        lab._dirty = True
                        sess = getattr(lab, "session", None)
                        if sess is not None and not sess.poisoned() and hasattr(sess, "fault"):
                            sess.fault(
                                "participant_failed",
                                last_error or pop_err or "shadow promotion failed after stopping primary",
                                "standby shadow promoted cleanly",
                                subject="shadow_promote",
                            )
                    err_out = (
                        last_error
                        if (last_error is not None and pop_err in (None, "no parked standby shadow lab available"))
                        else (pop_err or last_error or "no parked standby shadow lab available")
                    )
                    return {
                        "ok": False,
                        "error": err_out,
                        "skipped_dead_shadows": total_skipped,
                        "primary_stopped": primary_stopped,
                        "promote_s": round(time.perf_counter() - t0, 6),
                    }

                if not primary_stopped:
                    t_stop = time.perf_counter()
                    with contextlib.suppress(Exception):
                        lab._stop_engine()
                    stop_primary_s = round(time.perf_counter() - t_stop, 6)
                    primary_stopped = True

                cand_lock = getattr(candidate.lab, "lock", None)
                with (cand_lock if cand_lock is not None else contextlib.nullcontext()):
                    t_resume = time.perf_counter()
                    try:
                        resume_reply = candidate.hooks.resume_after_restore(candidate.receipt)
                    except Exception as exc:
                        resume_reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    resume_s = round(time.perf_counter() - t_resume, 6)
                    if not isinstance(resume_reply, dict) or not resume_reply.get("ok"):
                        last_error = (
                            str(resume_reply.get("error"))
                            if isinstance(resume_reply, dict) and resume_reply.get("error")
                            else "shadow resume_after_restore failed"
                        )
                        total_skipped += 1
                        _close_shadow_quietly(candidate.lab)
                        continue

                    clean_err = _check_clean_shadow_lab(candidate.lab)
                    if clean_err is not None:
                        last_error = clean_err
                        total_skipped += 1
                        _close_shadow_quietly(candidate.lab)
                        continue

                    _transfer_engine_ownership(lab, candidate.lab)

                co = _rebuild_coordinator(lab)
                modules = list(getattr(lab, "modules", ()) or getattr(lab, "_tracked_modules", ()))
                lab.co = co
                lab.modules = modules
                if getattr(lab, "session", None) is not None and hasattr(lab.session, "recovered"):
                    lab.session.recovered(co)
                else:
                    lab.session = kernel_session.Session(co, modules)

                if lab.session is not None and lab.session.poisoned():
                    last_error = f"promoted shadow session poisoned after recovery: {lab.session.poisoned()}"
                    total_skipped += 1
                    lab._dirty = True
                    with contextlib.suppress(Exception):
                        lab._stop_engine()
                    _close_shadow_quietly(candidate.lab)
                    continue

                lab.live_text = [dict(getattr(lab, "base_text", {}) or {})]
                lab.last_check = None
                lab._dirty = False

                _close_shadow_quietly(candidate.lab)

                return {
                    "ok": True,
                    "reset": ["shadow promoted", "session"],
                    "stop_primary_s": stop_primary_s,
                    "resume_s": resume_s,
                    "promote_s": round(time.perf_counter() - t0, 6),
                    "resumed": list(resume_reply.get("resumed") or ()),
                    "skipped_dead_shadows": total_skipped,
                    "pending_shadows": self.pending_count(),
                }

    def as_recovery_driver(
        self,
        *,
        replenish_fn: Callable[[Any, Any], Any] | None = None,
        replenish_wait_timeout_s: float = 0.0,
        fallback_driver: Any = None,
    ) -> StandbyShadowDriver:
        """Build a promoter-aware `StandbyShadowDriver` (deferring primary stop until candidate validation)."""
        effective_replenish = replenish_fn if replenish_fn is not None else self._replenish_fn
        async_rep = _resolve_async_replenisher(effective_replenish)
        if async_rep is not None:
            self._track_replenisher(async_rep)
        ready_fn = self.pending_count if self._shadow_provider is None else None
        return _TrtllmShadowDriver(
            self,
            replenish_fn=effective_replenish,
            replenish_wait_timeout_s=replenish_wait_timeout_s,
            fallback_driver=fallback_driver,
            ready_count_fn=ready_fn,
        )

    def close(self) -> None:
        """Close the promoter and tear down all remaining parked standby shadow labs."""
        with self._lock:
            self._closed = True
            remaining = list(self._shadows)
            self._shadows.clear()
            replenishers = list(self._attached_replenishers)
        for rep in replenishers:
            self._wake_replenisher(rep)
        for entry in remaining:
            _close_shadow_quietly(entry.lab)

    def __enter__(self) -> TrtllmShadowPromoter:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.close()
        return False
