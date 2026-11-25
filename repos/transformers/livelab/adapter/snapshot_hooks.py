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

    def __init__(
        self, lab: Any, *, offload_to_cpu: bool = True, pin_memory: bool = True,
        warmup_workload: dict[str, Any] | None = None,
    ) -> None:
        self.lab, self.offload_to_cpu, self.pin_memory = lab, bool(offload_to_cpu), bool(pin_memory)
        self.warmup_workload = dict(warmup_workload) if warmup_workload is not None else None

    def _preflight_error(self, *, check_status: bool = True) -> str | None:
        if not self.lab.is_running():
            return "transformers worker is not running"
        if getattr(self.lab, "dirty", False):
            return "transformers lab control channel is dirty; respawn required"
        session = getattr(self.lab, "session", None)
        if session is not None and session.poisoned():
            return f"transformers lab session is poisoned: {session.poisoned()}"
        if check_status:
            status_reply = self.lab.status()
            if not status_reply.get("ok") or status_reply.get("state") == "poisoned":
                return f"transformers lab status is poisoned: {status_reply.get('poison')}"
        return None

    def warmup_for_snapshot(self) -> None:
        with self.lab.lock:
            if error_text := self._preflight_error(check_status=True):
                raise RuntimeError(error_text)
            workload = self.warmup_workload if self.warmup_workload is not None else self.lab.default_warmup_workload()
            self.lab.run_generate(dict(workload))
            probe_ok, probe_error = self.lab._probe_participants()
            if not probe_ok:
                raise RuntimeError(f"participant probe failed during warmup: {probe_error}")

    def quiesce_for_snapshot(self) -> QuiesceReceipt:
        with self.lab.lock:
            if error_text := self._preflight_error(check_status=True):
                return QuiesceReceipt(ok=False, mutability=self.mutability, error=error_text)
            released, quiesced = (), False
            try:
                reply = _require_ok(
                    self.lab.quiesce_worker(offload_to_cpu=self.offload_to_cpu, pin_memory=self.pin_memory), "quiesce"
                )
                released, quiesced = tuple(str(tag) for tag in (reply.get("released") or ())), True
                probe_ok, probe_error = self.lab._probe_participants()
                if not probe_ok:
                    raise RuntimeError(f"participant probe failed after quiesce: {probe_error}")
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                if quiesced:
                    try:
                        _require_ok(self.lab.resume_worker(released=released or ("weights",)), "rollback resume")
                    except Exception as rollback_error:
                        self.lab.dirty = True
                        message = f"{message} (rollback failed: {rollback_error})"
                return QuiesceReceipt(ok=False, mutability=self.mutability, error=message)
            mode_tag = "cpu_offload" if self.offload_to_cpu else "cache_clear"
            token = f"transformers:{mode_tag}:{','.join(released)}"
            return QuiesceReceipt(ok=True, released=released, mutability=self.mutability, token=token)

    def resume_after_restore(self, receipt: QuiesceReceipt) -> dict[str, Any]:
        if receipt is not None and not receipt.ok:
            return {"ok": False, "error": receipt.error or "failed QuiesceReceipt", "participants_ok": False}
        with self.lab.lock:
            if error_text := self._preflight_error(check_status=False):
                return {"ok": False, "error": error_text, "participants_ok": False}
            released = tuple(receipt.released) if receipt is not None and receipt.released else ()
            resume_start, resumed_tags = time.perf_counter(), []
            try:
                reply = _require_ok(self.lab.resume_worker(released=released), "worker resume")
                resume_seconds = time.perf_counter() - resume_start
                resumed_tags = list(reply.get("resumed") or ())
                probe_ok, probe_error = self.lab._probe_participants()
                if not probe_ok:
                    raise RuntimeError(f"participant probe failed after restore: {probe_error}")
                status_reply = self.lab.status()
                if not status_reply.get("ok") or status_reply.get("state") == "poisoned" or not status_reply.get("fingerprint"):
                    raise RuntimeError(f"invalid status after restore: {status_reply.get('poison') or 'no fingerprint'}")
            except Exception as error:
                elapsed = round(time.perf_counter() - resume_start, 6)
                return {
                    "ok": False, "error": f"{type(error).__name__}: {error}",
                    "resumed": resumed_tags, "resume_s": elapsed, "participants_ok": False,
                }
            result: dict[str, Any] = {
                "ok": True, "resumed": resumed_tags, "device": reply.get("device"),
                "resume_s": round(resume_seconds, 6), "participants_ok": True, "fingerprint": status_reply["fingerprint"],
            }
            if reply.get("devices") is not None:
                result["devices"] = reply["devices"]
            if reply.get("hf_device_map") is not None:
                result["hf_device_map"] = reply["hf_device_map"]
            return result
