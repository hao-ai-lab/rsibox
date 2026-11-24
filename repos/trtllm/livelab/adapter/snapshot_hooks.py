"""TensorRT-LLM `QuiesceHooks` implementation for `ai-dynamo/snapshot` checkpoint/restore."""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any
from rsibox.warm import Mutability, QuiesceReceipt

class TrtllmQuiesceHooks:
    """`rsibox.warm.QuiesceHooks` adapter for a live `TrtllmLab` or `ToyTrtllmLab`."""

    mutability: Mutability = "readonly_weights"

    def __init__(self, lab: Any, *, release_tags: Sequence[str] | None = None,
                 warmup_workload: dict[str, Any] | None = None) -> None:
        if isinstance(release_tags, (str, bytes)):
            raise TypeError("release_tags must be a sequence of tag names, not a string")
        self.lab = lab
        self.release_tags = tuple(str(tag) for tag in release_tags) if release_tags is not None else ("kv_cache",)
        self.warmup_workload = dict(warmup_workload) if warmup_workload is not None else None

    def _preflight_error(self) -> str | None:
        if not getattr(self.lab, "workers", None):
            return "trtllm workers are not running"
        if getattr(self.lab, "_dirty", False):
            return "trtllm lab control channel is dirty; respawn required"
        session = getattr(self.lab, "session", None)
        return "trtllm lab session is poisoned; respawn required" if (session is not None and session.poisoned()) else None

    def warmup_for_snapshot(self) -> None:
        if preflight_error := self._preflight_error():
            raise RuntimeError(preflight_error)
        workload = dict(self.warmup_workload) if self.warmup_workload is not None else self.lab.default_warmup_workload()
        self.lab._run_workload(workload)
        probe_ok, probe_error = self.lab._probe_participants()
        if not probe_ok:
            raise RuntimeError(f"participant probe failed during warmup: {probe_error}")

    def quiesce_for_snapshot(self) -> QuiesceReceipt:
        if preflight_error := self._preflight_error():
            return QuiesceReceipt(ok=False, mutability=self.mutability, error=preflight_error)
        tags, quiesced_workers, released_union = list(self.release_tags), [], []
        try:
            for worker in self.lab.workers:
                reply = worker.call("quiesce_memory", {"tags": tags})
                if not reply.get("ok"):
                    raise RuntimeError(f"{worker.name}: {reply.get('error', 'quiesce_memory failed')}")
                quiesced_workers.append(worker)
                for released_tag in reply.get("released") or ():
                    if released_tag not in released_union:
                        released_union.append(released_tag)
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            for worker in reversed(quiesced_workers):
                try:
                    worker.call("resume_memory", {"tags": tags})
                except Exception as rollback_error:
                    error_message += f" (rollback on {worker.name} failed: {type(rollback_error).__name__}: {rollback_error})"
            return QuiesceReceipt(ok=False, mutability=self.mutability, error=error_message)
        released_tuple = tuple(released_union) if released_union else tuple(tags)
        return QuiesceReceipt(ok=True, released=released_tuple, mutability=self.mutability, token=f"trtllm:{','.join(released_tuple)}")

    def resume_after_restore(self, receipt: QuiesceReceipt) -> dict[str, Any]:
        if preflight_error := self._preflight_error():
            return {"ok": False, "error": preflight_error, "participants_ok": False}
        tags = list(receipt.released) if (receipt is not None and receipt.released) else list(self.release_tags)
        start_time, rank_details = time.perf_counter(), {}
        try:
            for worker in self.lab.workers:
                reply = worker.call("resume_memory", {"tags": tags})
                if not reply.get("ok"):
                    raise RuntimeError(f"{worker.name}: {reply.get('error', 'resume_memory failed')}")
                rank_details[worker.name] = {key: reply.get(key) for key in ("resumed", "vram_before_mib", "vram_after_mib", "seconds")}
        except Exception as error:
            return {"ok": False, "error": f"resume_memory failed: {type(error).__name__}: {error}", "resumed": [],
                    "resume_memory_s": round(time.perf_counter() - start_time, 6), "participants_ok": False, "ranks": rank_details}
        resume_seconds = round(time.perf_counter() - start_time, 6)
        probe_ok, probe_error = self.lab._probe_participants()
        return {"ok": probe_ok, **({} if probe_ok else {"error": f"participant probe failed after restore: {probe_error}"}),
                "resumed": tags, "resume_memory_s": resume_seconds, "participants_ok": probe_ok, "ranks": rank_details}
