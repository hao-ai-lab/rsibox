"""HuggingFace Transformers `QuiesceHooks` for `ai-dynamo/snapshot` checkpoint/restore."""
from __future__ import annotations

import time
from typing import Any

from rsibox.warm import Mutability, QuiesceReceipt

DEFAULT_WARMUP_WORKLOAD: dict[str, Any] = {"prompts": ["Warmup."], "max_new_tokens": 2, "temperature": 0.0}


def _require_ok(reply: Any, action_name: str) -> dict[str, Any]:
    if not isinstance(reply, dict) or reply.get("ok") is not True:
        detail = reply.get("error") if isinstance(reply, dict) else f"bad reply {reply!r}"
        raise RuntimeError(str(detail or f"{action_name} returned ok=False"))
    return reply


class TransformersQuiesceHooks:
    """`rsibox.warm.QuiesceHooks` adapter for a live `TransformersLab`."""

    mutability: Mutability = "readonly_weights"

    def __init__(self, lab: Any, *, offload_to_cpu: bool = True, pin_memory: bool = True, warmup_workload: dict[str, Any] | None = None) -> None:
        self.lab, self.offload_to_cpu, self.pin_memory = lab, bool(offload_to_cpu), bool(pin_memory)
        self.warmup_workload = dict(warmup_workload) if warmup_workload is not None else None

    def _preflight_error(self) -> str | None:
        if not self.lab.is_running():
            return "transformers worker is not running"
        return "transformers lab control channel is dirty; respawn required" if getattr(self.lab, "dirty", False) else None

    def warmup_for_snapshot(self) -> None:
        if error_text := self._preflight_error():
            raise RuntimeError(error_text)
        self.lab.run_generate(dict(self.warmup_workload if self.warmup_workload is not None else self.lab.default_warmup_workload()))
        probe_ok, probe_error = self.lab._probe_participants()
        if not probe_ok:
            raise RuntimeError(f"participant probe failed during warmup: {probe_error}")

    def quiesce_for_snapshot(self) -> QuiesceReceipt:
        if error_text := self._preflight_error():
            return QuiesceReceipt(ok=False, mutability=self.mutability, error=error_text)
        try:
            reply = _require_ok(self.lab.quiesce_worker(offload_to_cpu=self.offload_to_cpu, pin_memory=self.pin_memory), "worker quiesce")
            probe_ok, probe_error = self.lab._probe_participants()
            if not probe_ok:
                raise RuntimeError(f"participant probe failed after quiesce: {probe_error}")
        except Exception as error:
            return QuiesceReceipt(ok=False, mutability=self.mutability, error=f"{type(error).__name__}: {error}")
        released = tuple(str(tag) for tag in (reply.get("released") or ()))
        mode_tag = "cpu_offload" if self.offload_to_cpu else "cache_clear"
        return QuiesceReceipt(ok=True, released=released, mutability=self.mutability, token=f"transformers:{mode_tag}:{','.join(released)}")

    def resume_after_restore(self, receipt: QuiesceReceipt) -> dict[str, Any]:
        if error_text := self._preflight_error():
            return {"ok": False, "error": error_text, "participants_ok": False}
        released = tuple(receipt.released) if receipt is not None and receipt.released else ()
        resume_start, resumed_tags = time.perf_counter(), []
        try:
            reply = _require_ok(self.lab.resume_worker(released=released), "worker resume")
            resume_seconds = time.perf_counter() - resume_start
            resumed_tags = list(reply.get("resumed") or released)
            probe_ok, probe_error = self.lab._probe_participants()
            if not probe_ok:
                raise RuntimeError(f"participant probe failed after restore: {probe_error}")
            status_reply = self.lab.status()
            if not status_reply.get("ok") or status_reply.get("state") == "poisoned" or not status_reply.get("fingerprint"):
                raise RuntimeError(f"invalid status after restore: {status_reply.get('poison') or 'missing fingerprint'}")
        except Exception as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}", "resumed": resumed_tags, "resume_s": round(time.perf_counter() - resume_start, 6), "participants_ok": False}
        return {"ok": True, "resumed": resumed_tags, "device": reply.get("device"), "resume_s": round(resume_seconds, 6), "participants_ok": True, "fingerprint": status_reply["fingerprint"]}
