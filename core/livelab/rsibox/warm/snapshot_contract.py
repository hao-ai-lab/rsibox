"""Generic `ai-dynamo/snapshot` Workload Contract driver for any RSIBox engine (stdlib only).

Implements the exact sentinel-file lifecycle documented in
`ai-dynamo/snapshot/docs/reference/workload-contract.md`:

- `SNAPSHOT_CONTROL_DIR`: shared `emptyDir` mounted by the Snapshot node agent.
- `ready-for-snapshot`: written by the workload ONLY after warmup + quiesce + memory park hold.
- `restore-complete`: written by the Snapshot agent after CRIU + `cuda-checkpoint` restore.
- `<engine>-restore-ready`: written by the workload after `resume_after_restore()` succeeds.

Because `ai-dynamo/snapshot` is completely framework-agnostic at the container boundary,
any RSIBox engine (vLLM, SGLang, FastVideo, Megatron-LM, veRL, Miles) that implements
`QuiesceHooks` can participate in Kubernetes `PodSnapshot` / `SnapshotJob` without
modifying `ai-dynamo/snapshot` or hard-binding RSIBox to Kubernetes.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recovery import QuiesceHooks
from .types import QuiesceReceipt
from .weight_store import WeightStore

ENV_SNAPSHOT_CONTROL_DIR = "SNAPSHOT_CONTROL_DIR"
SENTINEL_READY = "ready-for-snapshot"
SENTINEL_RESTORE_COMPLETE = "restore-complete"


@dataclass(frozen=True)
class SnapshotLifecycleResult:
    """Structured outcome of one `run_snapshot_lifecycle()` pass."""

    ok: bool
    engine: str
    control_dir: str
    quiesce: QuiesceReceipt
    warmup_s: float
    quiesce_s: float
    wait_restore_s: float
    resume_s: float
    restore_ready_path: str | None = None
    resume_details: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "engine": self.engine,
            "control_dir": self.control_dir,
            "quiesce": self.quiesce.to_dict(),
            "warmup_s": round(self.warmup_s, 4),
            "quiesce_s": round(self.quiesce_s, 4),
            "wait_restore_s": round(self.wait_restore_s, 4),
            "resume_s": round(self.resume_s, 4),
        }
        if self.restore_ready_path:
            out["restore_ready_path"] = self.restore_ready_path
        if self.resume_details:
            out["resume_details"] = self.resume_details
        if self.error:
            out["error"] = self.error
        return out


def _atomic_touch(path: Path, payload: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _wait_for_sentinel(path: Path, *, timeout_s: float | None, poll_s: float) -> bool:
    deadline = None if timeout_s is None else (time.monotonic() + float(timeout_s))
    while True:
        if path.exists():
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


def run_snapshot_lifecycle(
    hooks: QuiesceHooks,
    *,
    engine_name: str,
    control_dir: str | os.PathLike[str] | None = None,
    skip_warmup: bool = False,
    wait_timeout_s: float | None = None,
    poll_s: float = 0.05,
) -> SnapshotLifecycleResult:
    """Execute the `ai-dynamo/snapshot` Workload Contract around `hooks`.

    Follows the contract's load-bearing invariants:
    1. Clear stale `ready-for-snapshot` and `<engine>-restore-ready` before warming up.
    2. Warm up so lazy CUDA contexts, autotuning, and CUDA graphs enter the checkpoint.
    3. Quiesce and park disposable memory; NEVER write `ready-for-snapshot` if quiesce fails.
    4. Atomically write `ready-for-snapshot`, then block until `restore-complete` appears.
    5. Resume (`resume_after_restore`) and atomically write `<engine>-restore-ready`.
    """
    raw_dir = control_dir if control_dir is not None else os.environ.get(ENV_SNAPSHOT_CONTROL_DIR)
    if not raw_dir:
        raise ValueError(
            f"snapshot control_dir not provided and {ENV_SNAPSHOT_CONTROL_DIR} is unset"
        )
    ctrl = Path(raw_dir)
    ctrl.mkdir(parents=True, exist_ok=True)
    ready_file = ctrl / SENTINEL_READY
    restore_complete_file = ctrl / SENTINEL_RESTORE_COMPLETE
    restore_ready_file = ctrl / f"{engine_name}-restore-ready"

    # Step 1 (MUST): clear stale sentinels before initializing/warming up.
    ready_file.unlink(missing_ok=True)
    restore_ready_file.unlink(missing_ok=True)

    # Step 2 (SHOULD): run at least one real step so CUDA graphs / JIT enter the snapshot.
    t0 = time.monotonic()
    if not skip_warmup:
        hooks.warmup_for_snapshot()
    warmup_s = time.monotonic() - t0

    # Step 3 (MUST/SHOULD): quiesce & park disposable GPU memory; fail closed on error.
    t1 = time.monotonic()
    receipt = hooks.quiesce_for_snapshot()
    quiesce_s = time.monotonic() - t1
    if not receipt.ok:
        return SnapshotLifecycleResult(
            ok=False,
            engine=engine_name,
            control_dir=str(ctrl),
            quiesce=receipt,
            warmup_s=warmup_s,
            quiesce_s=quiesce_s,
            wait_restore_s=0.0,
            resume_s=0.0,
            error=receipt.error or "quiesce_for_snapshot failed",
        )

    # Step 4 (MUST): write `ready-for-snapshot` only after quiesce succeeded, then wait.
    _atomic_touch(ready_file)
    t2 = time.monotonic()
    restored = _wait_for_sentinel(
        restore_complete_file, timeout_s=wait_timeout_s, poll_s=poll_s
    )
    wait_restore_s = time.monotonic() - t2
    if not restored:
        return SnapshotLifecycleResult(
            ok=False,
            engine=engine_name,
            control_dir=str(ctrl),
            quiesce=receipt,
            warmup_s=warmup_s,
            quiesce_s=quiesce_s,
            wait_restore_s=wait_restore_s,
            resume_s=0.0,
            error=f"timed out after {wait_timeout_s}s waiting for {restore_complete_file}",
        )

    # Step 5: resume after CRIU + cuda-checkpoint restore and signal readiness.
    t3 = time.monotonic()
    resume_details = hooks.resume_after_restore(receipt) or {}
    resume_s = time.monotonic() - t3
    if resume_details.get("ok") is False:
        return SnapshotLifecycleResult(
            ok=False,
            engine=engine_name,
            control_dir=str(ctrl),
            quiesce=receipt,
            warmup_s=warmup_s,
            quiesce_s=quiesce_s,
            wait_restore_s=wait_restore_s,
            resume_s=resume_s,
            resume_details=resume_details,
            error=str(resume_details.get("error") or "resume_after_restore failed"),
        )

    _atomic_touch(restore_ready_file)
    return SnapshotLifecycleResult(
        ok=True,
        engine=engine_name,
        control_dir=str(ctrl),
        quiesce=receipt,
        warmup_s=warmup_s,
        quiesce_s=quiesce_s,
        wait_restore_s=wait_restore_s,
        resume_s=resume_s,
        restore_ready_path=str(restore_ready_file),
        resume_details=resume_details,
    )


class DynamoK8sSnapshotDriver:
    """`RecoveryDriver` adapter exposing the `ai-dynamo/snapshot` contract for any `BaseLab`."""

    name = "dynamo_k8s_snapshot"

    def __init__(
        self,
        *,
        engine_name: str,
        control_dir: str | None = None,
        wait_timeout_s: float | None = None,
        poll_s: float = 0.05,
    ):
        self.engine_name = engine_name
        self.control_dir = control_dir
        self.wait_timeout_s = wait_timeout_s
        self.poll_s = poll_s

    def checkpoint_and_wait(
        self,
        hooks: QuiesceHooks,
        *,
        skip_warmup: bool = False,
    ) -> SnapshotLifecycleResult:
        """Run the full `ready-for-snapshot` -> `restore-complete` -> restore-ready handshake."""
        return run_snapshot_lifecycle(
            hooks,
            engine_name=self.engine_name,
            control_dir=self.control_dir,
            skip_warmup=skip_warmup,
            wait_timeout_s=self.wait_timeout_s,
            poll_s=self.poll_s,
        )

    def respawn(
        self,
        lab: Any,
        store: WeightStore | None = None,
        *,
        reason: str = "reset_respawn",
    ) -> dict[str, Any]:
        """Fall back to local respawn when called inside an already-restored pod."""
        from .recovery import DefaultRespawnDriver

        out = DefaultRespawnDriver().respawn(lab, store, reason=reason)
        out["driver"] = self.name
        return out
