"""TensorRT-LLM `QuiesceHooks` implementation for `ai-dynamo/snapshot` checkpoint/restore."""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from rsibox.warm import Mutability, QuiesceReceipt


class TrtllmQuiesceHooks:
    """`rsibox.warm.QuiesceHooks` adapter for a live `TrtllmLab` or `ToyTrtllmLab`."""

    mutability: Mutability = "readonly_weights"

    def __init__(
        self,
        lab: Any,
        *,
        release_tags: Sequence[str] | None = None,
        warmup_workload: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(release_tags, (str, bytes)):
            raise TypeError("release_tags must be a sequence of tag names, not a string")
        self.lab = lab
        self.release_tags = (
            tuple(str(tag) for tag in release_tags) if release_tags is not None else ("kv_cache",)
        )
        self.warmup_workload = dict(warmup_workload) if warmup_workload is not None else None

    def _preflight_error(self) -> str | None:
        if not getattr(self.lab, "workers", None):
            return "trtllm workers are not running"
        if getattr(self.lab, "_dirty", False):
            return "trtllm lab control channel is dirty; respawn required"
        session = getattr(self.lab, "session", None)
        if session is not None and session.poisoned():
            return "trtllm lab session is poisoned; respawn required"
        if hasattr(self.lab, "status"):
            lab_status = self.lab.status()
            if not lab_status.get("ok") or lab_status.get("state") == "poisoned":
                return f"trtllm lab status is {lab_status.get('state', 'unhealthy')}; respawn required"
        return None

    def _quiesce_timeout(self) -> float:
        timeout = getattr(self.lab, "quiesce_timeout", None)
        if isinstance(timeout, (int, float)) and timeout > 0:
            return float(timeout)
        return float(getattr(self.lab, "rpc_timeout", 30.0))

    def _poison_lab(self, reason: str) -> None:
        self.lab._dirty = True
        session = getattr(self.lab, "session", None)
        if session is not None and hasattr(session, "fault"):
            session.fault(
                "participant_failed",
                reason,
                "every worker restores clean memory state",
                subject="quiesce_rollback",
            )

    def warmup_for_snapshot(self) -> None:
        """Run one deterministic generation step under `lab.lock` and probe all participants."""
        with self.lab.lock:
            preflight_error = self._preflight_error()
            if preflight_error:
                raise RuntimeError(preflight_error)
            workload = (
                dict(self.warmup_workload)
                if self.warmup_workload is not None
                else self.lab.default_warmup_workload()
            )
            self.lab._execute_workload(workload)
            probe_ok, probe_error = self.lab._probe_participants()
            if not probe_ok:
                raise RuntimeError(f"participant probe failed during warmup: {probe_error}")

    def quiesce_for_snapshot(self) -> QuiesceReceipt:
        """Release disposable GPU KV-cache pools (and optional graphs/VM tags) before snapshot."""
        with self.lab.lock:
            preflight_error = self._preflight_error()
            if preflight_error:
                return QuiesceReceipt(ok=False, mutability=self.mutability, error=preflight_error)

            tags = list(self.release_tags)
            timeout_seconds = self._quiesce_timeout()
            quiesced_workers: list[tuple[Any, list[str]]] = []
            released_union: list[str] = []
            failing_worker: Any = None
            failing_reply: dict[str, Any] | None = None
            transport_failed = False

            try:
                for worker in self.lab.workers:
                    failing_worker = worker
                    failing_reply = None
                    reply = worker.call("quiesce_memory", {"tags": tags}, timeout=timeout_seconds)
                    failing_reply = reply
                    if not reply.get("ok"):
                        transport_failed = bool(reply.get("transport_error"))
                        raise RuntimeError(f"{worker.name}: {reply.get('error', 'quiesce_memory failed')}")
                    worker_released = list(reply.get("released") or ())
                    if tags and not worker_released:
                        raise RuntimeError(f"{worker.name}: quiesce_memory released no tags")
                    quiesced_workers.append((worker, worker_released))
                    for released_tag in worker_released:
                        if released_tag not in released_union:
                            released_union.append(released_tag)
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
                rollback_failed = False
                rollback_targets: list[tuple[Any, list[str]]] = list(reversed(quiesced_workers))
                if failing_worker is not None and failing_reply is not None:
                    partial_tags = list(failing_reply.get("partial_released") or ())
                    if partial_tags:
                        rollback_targets.insert(0, (failing_worker, partial_tags))
                    elif transport_failed:
                        rollback_targets.insert(0, (failing_worker, tags))

                for worker, rollback_tags in rollback_targets:
                    rollback_reply = worker.call("resume_memory", {"tags": rollback_tags}, timeout=timeout_seconds)
                    if not rollback_reply.get("ok"):
                        rollback_failed = True
                        error_message += (
                            f" (rollback on {worker.name} failed: "
                            f"{rollback_reply.get('error', 'resume_memory failed')})"
                        )
                if rollback_failed or transport_failed:
                    self._poison_lab(error_message)
                return QuiesceReceipt(ok=False, mutability=self.mutability, error=error_message)

            if tags and not released_union:
                return QuiesceReceipt(
                    ok=False,
                    mutability=self.mutability,
                    error="quiesce_for_snapshot released no memory tags",
                )
            released_tuple = tuple(released_union)
            return QuiesceReceipt(
                ok=True,
                released=released_tuple,
                mutability=self.mutability,
                token=f"trtllm:{','.join(released_tuple)}",
            )

    def resume_after_restore(self, receipt: QuiesceReceipt) -> dict[str, Any]:
        """Re-allocate released KV-cache pools after restore and verify every participant."""
        with self.lab.lock:
            if receipt is None or not receipt.ok:
                return {
                    "ok": False,
                    "error": "cannot resume after restore from a missing or failed QuiesceReceipt",
                    "resumed": [],
                    "participants_ok": False,
                }
            preflight_error = self._preflight_error()
            if preflight_error:
                return {"ok": False, "error": preflight_error, "resumed": [], "participants_ok": False}

            tags = list(receipt.released) if receipt.released else list(self.release_tags)
            timeout_seconds = self._quiesce_timeout()
            start_time = time.perf_counter()
            rank_details: dict[str, Any] = {}
            resumed_any_worker = False
            try:
                for worker in self.lab.workers:
                    reply = worker.call("resume_memory", {"tags": tags}, timeout=timeout_seconds)
                    if not reply.get("ok"):
                        error_message = f"{worker.name}: {reply.get('error', 'resume_memory failed')}"
                        if resumed_any_worker or not reply.get("rejected"):
                            self._poison_lab(f"resume_memory failed: {error_message}")
                        raise RuntimeError(error_message)
                    resumed_any_worker = True
                    rank_details[worker.name] = {
                        "resumed": list(reply.get("resumed") or ()),
                        "vram_before_mib": reply.get("vram_before_mib"),
                        "vram_after_mib": reply.get("vram_after_mib"),
                        "seconds": reply.get("seconds"),
                    }
            except Exception as error:
                return {
                    "ok": False,
                    "error": f"resume_memory failed: {type(error).__name__}: {error}",
                    "resumed": [],
                    "resume_memory_s": round(time.perf_counter() - start_time, 6),
                    "participants_ok": False,
                    "ranks": rank_details,
                }

            resume_seconds = round(time.perf_counter() - start_time, 6)
            probe_ok, probe_error = self.lab._probe_participants()
            if not probe_ok:
                self._poison_lab(f"participant probe failed after restore: {probe_error}")
                return {
                    "ok": False,
                    "error": f"participant probe failed after restore: {probe_error}",
                    "resumed": tags,
                    "resume_memory_s": resume_seconds,
                    "participants_ok": False,
                    "ranks": rank_details,
                }
            return {
                "ok": True,
                "resumed": tags,
                "resume_memory_s": resume_seconds,
                "participants_ok": True,
                "ranks": rank_details,
            }
