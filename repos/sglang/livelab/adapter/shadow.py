"""Standby Scratch-KV shadow failover for `SglangLab` (`rsibox.warm.StandbyShadowDriver`).

Parks a standby `SglangLab` sharing read-only weights (e.g. via `DynamoGmsStore`) by
pausing generation and releasing disposable `kv_cache` GPU memory while keeping weights
and captured CUDA graphs warm. On primary crash/poison (`lab.reset(kind="respawn")`),
`SglangShadowPromoter` performs targeted primary engine teardown (avoiding upstream
`sglang.Engine.shutdown()`'s blanket `kill_process_tree(os.getpid())`, which would kill
co-process parked shadow workers), remaps KV cache, unpauses generation, verifies all TP
rank channels, transfers engine ownership cleanly onto `lab`, and recovers the session.
"""
from __future__ import annotations

import atexit
import collections
import contextlib
import importlib.metadata
import os
import signal
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from kernel import coordinator as C, session as SS
from rsibox.lab import RecaptureCoordinator
from rsibox.warm import QuiesceReceipt, StandbyShadowDriver

from .snapshot_hooks import SglangQuiesceHooks

DEFAULT_SHADOW_RELEASE_TAGS: tuple[str, ...] = ("kv_cache",)
_TARGETED_STOP_TERM_TIMEOUT_S = 2.0
_TARGETED_STOP_KILL_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ParkedSglangShadow:
    """A parked standby `SglangLab` paired with its `SglangQuiesceHooks` and receipt."""

    lab: Any
    hooks: SglangQuiesceHooks
    receipt: QuiesceReceipt


def _check_clean_shadow_lab(shadow_lab: Any) -> str | None:
    """Return an error message if `shadow_lab` is not live, clean, and free of active edits."""
    if getattr(shadow_lab, "engine", None) is None:
        return "standby shadow lab engine is not running"
    if getattr(shadow_lab, "dirty", False):
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


def _resolve_tp_size(lab: Any) -> int | None:
    cfg = getattr(lab, "config", None)
    if not isinstance(cfg, dict):
        return None
    if "tp_size" not in cfg:
        return 1
    raw_tp = cfg["tp_size"]
    if isinstance(raw_tp, bool) or not isinstance(raw_tp, int) or raw_tp <= 0:
        return -1
    return raw_tp


def _resolve_shadow_ranks(shadow_lab: Any) -> int:
    raw = getattr(shadow_lab, "ranks", None)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    return raw


def _terminate_child_processes(procs: Sequence[Any]) -> None:
    """Terminate and reap a specific set of `multiprocessing.Process`-like children."""
    live = [p for p in procs if p is not None and getattr(p, "pid", None)]
    for sig_method, sig_num, timeout_s in (
        ("terminate", signal.SIGTERM, _TARGETED_STOP_TERM_TIMEOUT_S),
        ("kill", signal.SIGKILL, _TARGETED_STOP_KILL_TIMEOUT_S),
    ):
        for proc in live:
            with contextlib.suppress(Exception):
                if hasattr(proc, "is_alive") and not proc.is_alive():
                    continue
                if hasattr(proc, sig_method):
                    getattr(proc, sig_method)()
                else:
                    os.kill(int(proc.pid), sig_num)
        deadline = time.monotonic() + timeout_s
        for proc in live:
            with contextlib.suppress(Exception):
                if hasattr(proc, "join"):
                    proc.join(timeout=max(0.0, deadline - time.monotonic()))


def _shutdown_targeted_engine(engine: Any, tm: Any, watchdog: Any, owned_procs: Sequence[Any]) -> None:
    with contextlib.suppress(Exception):
        atexit.unregister(engine.shutdown)
    with contextlib.suppress(Exception):
        shm = getattr(engine, "_multi_tokenizer_shm", None)
        if shm is not None:
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()
            engine._multi_tokenizer_shm = None
    for obj, method in (
        (getattr(engine, "_scheduler_init_result", None), "stop_grpc_server"),
        (watchdog, "stop"),
    ):
        if obj is not None and hasattr(obj, method):
            with contextlib.suppress(Exception):
                getattr(obj, method)()
    with contextlib.suppress(Exception):
        send_to_rpc = getattr(engine, "send_to_rpc", None)
        if send_to_rpc is not None:
            send_to_rpc.close(linger=0)
            engine.send_to_rpc = None
    with contextlib.suppress(Exception):
        daemon_procs = getattr(engine, "_weight_cache_daemon_procs", None)
        if daemon_procs and hasattr(engine, "_terminate_weight_cache_daemons"):
            engine._terminate_weight_cache_daemons(daemon_procs)
            engine._weight_cache_daemon_procs = []
    _terminate_child_processes(owned_procs)
    for attr in ("mm_processor", "cuda_vmm_feature_transport"):
        sub = getattr(tm, attr, None)
        if sub is not None and hasattr(sub, "shutdown"):
            with contextlib.suppress(Exception):
                sub.shutdown()


def stop_sglang_engine_targeted(lab: Any, *, preserve_process_env: bool = True) -> bool:
    """Stop `lab`'s engine without killing co-process parked shadows.

    Upstream `sglang.Engine.shutdown()` calls `kill_process_tree(os.getpid(), include_parent=False)`,
    terminating sibling co-process shadow workers. When `tokenizer_manager._subprocess_watchdog._processes`
    is present, this terminates only `lab`'s own processes and unregisters `atexit(engine.shutdown)`.
    """
    cleaned = False
    rank_conns = getattr(lab, "rank_conns", None)
    if isinstance(rank_conns, dict) and rank_conns:
        for conn in list(rank_conns.values()):
            with contextlib.suppress(OSError):
                conn.close()
        rank_conns.clear()
        cleaned = True

    sock_path = getattr(lab, "sock_path", None)
    if sock_path:
        with contextlib.suppress(OSError):
            os.unlink(sock_path)
        lab.sock_path = None
        cleaned = True

    engine = getattr(lab, "engine", None)
    if engine is not None:
        cleaned = True
        tm = getattr(engine, "tokenizer_manager", None)
        watchdog = getattr(tm, "_subprocess_watchdog", None) if tm is not None else None
        owned_procs = getattr(watchdog, "_processes", None) if watchdog is not None else None
        if isinstance(owned_procs, (list, tuple)) and owned_procs:
            _shutdown_targeted_engine(engine, tm, watchdog, owned_procs)
        else:
            with contextlib.suppress(Exception):
                atexit.unregister(engine.shutdown)
            with contextlib.suppress(Exception):
                engine.shutdown()
        lab.engine = None

    lab.ranks = 0
    if hasattr(lab, "rank_locks"):
        lab.rank_locks = {}
    lab.dirty = True

    if not preserve_process_env:
        fork_restore = getattr(lab, "_fork_subprocs_restore", None)
        if fork_restore is not None:
            with contextlib.suppress(Exception):
                fork_restore()
            lab._fork_subprocs_restore = None
        if hasattr(lab, "_restore_weight_cache_env"):
            with contextlib.suppress(Exception):
                lab._restore_weight_cache_env()

    return cleaned


def park_shadow_lab(
    shadow_lab: Any,
    *,
    release_tags: Sequence[str] = DEFAULT_SHADOW_RELEASE_TAGS,
    pause_mode: str = "retract",
) -> dict[str, Any]:
    """Pause `shadow_lab` and release disposable memory (`kv_cache` by default) for standby."""
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
            hooks = SglangQuiesceHooks(
                shadow_lab,
                release_tags=tuple(release_tags),
                pause_mode=pause_mode,
            )
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
            "shadow": ParkedSglangShadow(lab=shadow_lab, hooks=hooks, receipt=receipt),
            "hooks": hooks,
            "receipt": receipt,
            "released": list(receipt.released),
            "park_s": park_s,
        }


def _coerce_parked_shadow(item: Any) -> ParkedSglangShadow | None:
    if item is None:
        return None
    if isinstance(item, ParkedSglangShadow):
        entry = item
    elif isinstance(item, tuple) and len(item) == 3:
        shadow_lab, hooks, receipt = item
        entry = ParkedSglangShadow(lab=shadow_lab, hooks=hooks, receipt=receipt)
    else:
        raise TypeError(
            "shadow entry must be a ParkedSglangShadow or a (shadow_lab, hooks, receipt) tuple"
        )
    if not hasattr(entry.hooks, "resume_after_restore") or not callable(entry.hooks.resume_after_restore):
        raise TypeError("shadow hooks must implement resume_after_restore(receipt)")
    if getattr(entry.hooks, "lab", entry.lab) is not entry.lab:
        raise ValueError("shadow hooks.lab does not match shadow_lab")
    if not isinstance(entry.receipt, QuiesceReceipt):
        raise TypeError(f"shadow receipt must be a QuiesceReceipt, got {type(entry.receipt).__name__}")
    return entry


def _sglang_version(lab: Any) -> str:
    old_co = getattr(getattr(lab, "session", None), "co", None)
    old_ver = (
        getattr(old_co, "engine", {}).get("version")
        if isinstance(getattr(old_co, "engine", None), dict)
        else None
    )
    if old_ver:
        return str(old_ver)
    with contextlib.suppress(Exception):
        return importlib.metadata.version("sglang")
    return "0"


def _transfer_engine_ownership(dst_lab: Any, src_lab: Any) -> None:
    """Move live SGLang engine & rank-channel ownership from `src_lab` onto `dst_lab`."""
    dst_lab.engine = src_lab.engine
    dst_lab.sock_path = getattr(src_lab, "sock_path", None)
    dst_lab.ranks = _resolve_shadow_ranks(src_lab)
    dst_lab.rank_conns = dict(getattr(src_lab, "rank_conns", {}) or {})
    dst_lab.rank_locks = dict(getattr(src_lab, "rank_locks", {}) or {})
    dst_lab.dirty = False
    dst_lab.base_text = dict(getattr(src_lab, "base_text", {}) or {})
    src_modules = getattr(src_lab, "modules", None)
    if src_modules:
        dst_lab.modules = list(src_modules)
    if getattr(dst_lab, "_fork_subprocs_restore", None) is None:
        dst_lab._fork_subprocs_restore = getattr(src_lab, "_fork_subprocs_restore", None)
    if getattr(dst_lab, "_weight_cache_env_restore", None) is None:
        dst_lab._weight_cache_env_restore = getattr(src_lab, "_weight_cache_env_restore", None)
    if hasattr(src_lab, "fork_subprocs"):
        dst_lab.fork_subprocs = bool(src_lab.fork_subprocs)
    if hasattr(src_lab, "fork_subprocs_fallback"):
        dst_lab.fork_subprocs_fallback = list(getattr(src_lab, "fork_subprocs_fallback", ()) or ())
    if hasattr(src_lab, "allow_nonpauseable_cuda_graphs"):
        dst_lab.allow_nonpauseable_cuda_graphs = bool(src_lab.allow_nonpauseable_cuda_graphs)
    elif hasattr(dst_lab, "allow_nonpauseable_cuda_graphs"):
        dst_lab.allow_nonpauseable_cuda_graphs = False

    # Detach from the shell shadow lab so closing/GC'ing it cannot tear down the promoted engine
    # or mutate process-global env/shim state still owned by `dst_lab`.
    src_lab.engine = None
    src_lab.sock_path = None
    src_lab.ranks = 0
    src_lab.rank_conns = {}
    src_lab.rank_locks = {}
    src_lab.dirty = True
    src_lab._fork_subprocs_restore = None
    src_lab._weight_cache_env_restore = None


def _rebuild_coordinator(lab: Any) -> RecaptureCoordinator:
    participants = [
        C.CallParticipant(f"rank{r}", lambda m, p, r=r: lab.rank_call(r, m, p))
        for r in range(lab.ranks)
    ]
    return RecaptureCoordinator(
        participants,
        gap=lab.at_gap,
        writer=getattr(lab, "writer", None),
        interface_version="1-draft",
        engine={"name": getattr(lab, "name", "sglang"), "version": _sglang_version(lab), "commit": None},
        phase2_deadline_s=getattr(lab, "rpc_timeout", 30.0),
        rebuild=lab._rebuild_all_ranks,
        post_commit_reset="state",
    )


class _SglangShadowDriver(StandbyShadowDriver):
    """`StandbyShadowDriver` using targeted SGLang primary stop to protect co-process shadows."""

    def __init__(
        self,
        promoter: SglangShadowPromoter,
        *,
        replenish_fn: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        super().__init__(promoter.promote, replenish_fn=replenish_fn)
        self.promoter = promoter

    def respawn(
        self,
        lab: Any,
        store: Any = None,
        *,
        reason: str = "reset_respawn",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        # Validate candidate availability & stop primary cleanly using targeted shutdown
        # inside promoter.promote so we never kill co-process parked shadow workers.
        promoted = self._promote_fn(lab, store) or {}
        promote_cost_s = round(time.perf_counter() - t0, 6)
        session = getattr(lab, "session", None)
        poisoned = bool(session.poisoned()) if session is not None else False
        ok = bool(promoted.get("ok", False)) and not poisoned
        if ok and self._replenish_fn is not None:
            self._replenish_fn(lab, store)
        reserved = {"ok", "reset", "driver", "reason", "cost_s"}
        return {
            "ok": ok,
            "reset": list(promoted.get("reset", ["shadow promoted", "session"])) if ok else [],
            "driver": self.name,
            "reason": reason,
            "cost_s": promote_cost_s,
            **{k: v for k, v in promoted.items() if k not in reserved},
        }


class SglangShadowPromoter:
    """Promote a parked standby Scratch-KV `SglangLab` onto a primary `SglangLab` on failover."""

    def __init__(
        self,
        shadow: ParkedSglangShadow | tuple[Any, SglangQuiesceHooks, QuiesceReceipt] | None = None,
        *,
        shadow_provider: Callable[[Any, Any], Any] | None = None,
        replenish_fn: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._provider_lock = threading.Lock()
        self._closed = False
        self._shadows: collections.deque[ParkedSglangShadow] = collections.deque()
        if shadow is not None:
            self.register_shadow(shadow)
        self._shadow_provider = shadow_provider
        self._replenish_fn = replenish_fn

    def pending_count(self) -> int:
        with self._lock:
            return len(self._shadows)

    def register_shadow(
        self,
        shadow: ParkedSglangShadow | tuple[Any, SglangQuiesceHooks, QuiesceReceipt],
    ) -> ParkedSglangShadow:
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
                raise RuntimeError("cannot register shadow on a closed SglangShadowPromoter")
            if any(existing.lab is entry.lab for existing in self._shadows):
                raise ValueError("standby shadow lab is already registered in promoter queue")
            self._shadows.append(entry)
        return entry

    def park_shadow(
        self,
        shadow_lab: Any,
        *,
        release_tags: Sequence[str] = DEFAULT_SHADOW_RELEASE_TAGS,
        pause_mode: str = "retract",
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return {"ok": False, "error": "cannot park shadow on a closed SglangShadowPromoter"}
            if any(existing.lab is shadow_lab for existing in self._shadows):
                return {"ok": False, "error": "standby shadow lab is already parked in promoter queue"}
        result = park_shadow_lab(
            shadow_lab,
            release_tags=release_tags,
            pause_mode=pause_mode,
        )
        if result.get("ok"):
            with self._lock:
                if self._closed:
                    stop_sglang_engine_targeted(shadow_lab, preserve_process_env=False)
                    return {"ok": False, "error": "promoter was closed while parking shadow"}
                if any(existing.lab is shadow_lab for existing in self._shadows):
                    return {"ok": False, "error": "standby shadow lab is already parked in promoter queue"}
                self._shadows.append(result["shadow"])
        return result

    def _validate_candidate(self, lab: Any, entry: ParkedSglangShadow) -> tuple[str | None, bool]:
        """Validate `entry` for `lab`. Returns `(error, keep_in_queue)`."""
        shadow_lab, receipt = entry.lab, entry.receipt
        if shadow_lab is lab:
            return "standby shadow lab must be distinct from primary lab", True
        if getattr(entry.hooks, "lab", shadow_lab) is not shadow_lab:
            return "standby shadow hooks.lab does not match shadow_lab", False
        if not receipt.ok:
            return f"standby shadow receipt is not ok: {receipt.error}", False
        clean_err = _check_clean_shadow_lab(shadow_lab)
        if clean_err is not None:
            return clean_err, False

        expected_tp = _resolve_tp_size(lab)
        if expected_tp == -1:
            return "primary lab config has invalid tp_size", True
        shadow_ranks = _resolve_shadow_ranks(shadow_lab)
        if shadow_ranks <= 0:
            return "standby shadow lab has no active TP ranks", False
        if expected_tp is not None and shadow_ranks != expected_tp:
            return (
                f"standby shadow tp_size mismatch: shadow has {shadow_ranks}, primary expects {expected_tp}",
                True,
            )

        for attr in ("io_module", "scheduler_module"):
            lab_val = getattr(lab, attr, None)
            sh_val = getattr(shadow_lab, attr, None)
            if lab_val is not None and sh_val is not None and lab_val != sh_val:
                return f"standby shadow {attr} mismatch: {sh_val!r} != {lab_val!r}", True

        lab_cfg = getattr(lab, "config", None)
        sh_cfg = getattr(shadow_lab, "config", None)
        if isinstance(lab_cfg, dict) and isinstance(sh_cfg, dict):
            lab_model = lab_cfg.get("model_path")
            sh_model = sh_cfg.get("model_path")
            if lab_model and sh_model and lab_model != sh_model:
                return f"standby shadow model_path mismatch: {sh_model!r} != {lab_model!r}", True

        lab_mods = set(getattr(lab, "modules", ()) or ())
        sh_mods = set(getattr(shadow_lab, "modules", ()) or ())
        if lab_mods and sh_mods and not lab_mods.issubset(sh_mods):
            missing = sorted(lab_mods - sh_mods)
            return f"standby shadow missing tracked modules: {missing}", True

        return None, False

    def _acquire_shadow(self, lab: Any, weight_store: Any) -> tuple[ParkedSglangShadow | None, str | None]:
        """Atomically validate and pop the next compatible queued shadow, or invoke `shadow_provider`."""
        with self._lock:
            if self._shadows:
                candidate = self._shadows[0]
                err, keep = self._validate_candidate(lab, candidate)
                if err is not None:
                    if not keep:
                        self._shadows.popleft()
                    return None, err
                return self._shadows.popleft(), None

        if self._shadow_provider is None:
            return None, "no parked standby shadow lab available"
        with self._provider_lock:
            try:
                provided = _coerce_parked_shadow(self._shadow_provider(lab, weight_store))
            except Exception as exc:
                return None, f"shadow_provider failed: {type(exc).__name__}: {exc}"
        if provided is None:
            return None, "no parked standby shadow lab available"
        err, _keep = self._validate_candidate(lab, provided)
        if err is not None:
            return None, err
        return provided, None

    def promote(self, lab: Any, weight_store: Any = None) -> dict[str, Any]:
        """Stop primary cleanly, wake and verify the next parked standby `SglangLab`,
        and transfer ownership onto `lab`."""
        t_promote = time.perf_counter()
        entry, acquire_err = self._acquire_shadow(lab, weight_store)
        if entry is None:
            return {"ok": False, "error": acquire_err or "no parked standby shadow lab available"}

        shadow_lab, hooks, receipt = entry.lab, entry.hooks, entry.receipt

        # Stop primary BEFORE resuming shadow so primary releases exclusive GMS rw locks (kv_cache)
        # without killing the co-process parked shadow or tearing down shared process env.
        if getattr(lab, "engine", None) is not None or getattr(lab, "rank_conns", None):
            stop_sglang_engine_targeted(lab, preserve_process_env=True)
        else:
            lab.dirty = True

        shadow_lock = getattr(shadow_lab, "lock", None)
        ctx = shadow_lock if shadow_lock is not None else contextlib.nullcontext()
        with ctx:
            try:
                outcome = hooks.resume_after_restore(receipt)
            except Exception as exc:
                stop_sglang_engine_targeted(shadow_lab, preserve_process_env=True)
                lab.dirty = True
                return {
                    "ok": False,
                    "error": f"resume_after_restore raised {type(exc).__name__}: {exc}",
                }

            if not isinstance(outcome, dict) or not outcome.get("ok"):
                stop_sglang_engine_targeted(shadow_lab, preserve_process_env=True)
                lab.dirty = True
                err = (
                    outcome.get("error")
                    if isinstance(outcome, dict)
                    else f"unexpected resume_after_restore result: {type(outcome).__name__}"
                )
                return {
                    **(outcome if isinstance(outcome, dict) else {}),
                    "ok": False,
                    "error": err or "shadow resume failed",
                }

            _transfer_engine_ownership(lab, shadow_lab)

        coordinator = _rebuild_coordinator(lab)
        lab.co = coordinator
        lab.live_text = [dict(lab.base_text)]

        if getattr(lab, "session", None) is None:
            lab.session = SS.Session(coordinator, list(getattr(lab, "modules", ()) or ()))
            recovered = {"ok": not lab.session.poisoned()}
        else:
            recovered = lab.session.recovered(coordinator)

        poison_findings = list(lab.session.poisoned())
        if not recovered.get("ok") or poison_findings:
            stop_sglang_engine_targeted(lab, preserve_process_env=True)
            detail = (
                recovered.get("error")
                or (poison_findings[0].get("observed") if poison_findings else None)
                or "session recovery failed after shadow promotion"
            )
            return {
                **outcome,
                "ok": False,
                "promoted_ranks": 0,
                "findings": poison_findings,
                "error": str(detail),
            }

        promote_s = round(time.perf_counter() - t_promote, 6)
        if isinstance(getattr(lab, "costs", None), dict):
            lab.costs["promote_s"] = promote_s
        return {
            **outcome,
            "ok": True,
            "reset": ["shadow promoted", "session"],
            "promoted_ranks": lab.ranks,
            "promote_s": promote_s,
        }

    def close_parked(self) -> int:
        """Close every parked shadow still queued and reject future parks; returns count closed."""
        with self._lock:
            self._closed = True
            remaining = list(self._shadows)
            self._shadows.clear()
        closed = 0
        for entry in remaining:
            try:
                if stop_sglang_engine_targeted(entry.lab, preserve_process_env=False):
                    closed += 1
            except Exception:
                pass
        return closed

    def recovery_driver(self) -> StandbyShadowDriver:
        return _SglangShadowDriver(self, replenish_fn=self._replenish_fn)


def make_shadow_recovery_driver(
    shadow: ParkedSglangShadow | tuple[Any, SglangQuiesceHooks, QuiesceReceipt] | None = None,
    *,
    shadow_provider: Callable[[Any, Any], Any] | None = None,
    replenish_fn: Callable[[Any, Any], Any] | None = None,
) -> StandbyShadowDriver:
    """Create a `StandbyShadowDriver` configured with `SglangShadowPromoter`."""
    return SglangShadowPromoter(
        shadow=shadow,
        shadow_provider=shadow_provider,
        replenish_fn=replenish_fn,
    ).recovery_driver()
