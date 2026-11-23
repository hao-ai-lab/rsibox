"""Shared data types for RSIBox warm-state foundations and snapshot lifecycles.

Pure Python stdlib (zero torch, engine, or Dynamo imports). Every engine adapter,
`WeightStore`, and `RecoveryDriver` exchanges these plain dataclasses so callers
(`BaseLab`, conformance suites, RL verifiers) never depend on engine-specific flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Mutability = Literal["readonly_weights", "mutable_step_state"]


@dataclass(frozen=True)
class LaunchDelta:
    """Minimal CLI/env additions a `WeightStore` requests from an engine launcher.

    Launchers (such as `repos/vllm/livelab/services/engine.py`) merge `args` and `env`
    only after checking `findings` and `requires_worker_cls` against explicit user flags.
    """

    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    requires_worker_cls: str | None = None
    findings: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.findings


@dataclass(frozen=True)
class StoreProbe:
    """Result of probing whether a warm `WeightStore` is ready for the target GPUs."""

    ok: bool
    backend: str
    sockets: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    stale_keys: tuple[str, ...] = ()
    matched_gpus: bool = True
    seconds: float = 0.0
    error: str | None = None
    finding: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "backend": self.backend,
            "sockets": list(self.sockets),
            "missing": list(self.missing),
            "stale_keys": list(self.stale_keys),
            "matched_gpus": self.matched_gpus,
            "seconds": round(self.seconds, 4),
        }
        if self.error:
            out["error"] = self.error
        if self.finding:
            out["finding"] = self.finding
        return out


@dataclass(frozen=True)
class QuiesceReceipt:
    """Receipt emitted when an engine reaches a checkpoint-safe quiesced state."""

    ok: bool
    released: tuple[str, ...] = ()
    mutability: Mutability = "readonly_weights"
    token: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "released": list(self.released),
            "mutability": self.mutability,
        }
        if self.token is not None:
            out["token"] = self.token
        if self.error is not None:
            out["error"] = self.error
        return out


def stamp_diff(old: dict[str, Any] | None, new: dict[str, Any] | None) -> tuple[str, ...]:
    """Compare two weight-store stamps (`files`, `weights`, `env`, `extra`).

    Returns an empty tuple when the cached GPU weights remain valid for the current
    source tree, checkpoint files, and environment; otherwise lists every changed key.
    """
    if old is None or new is None:
        return () if old == new else ("stamp_missing",)
    diff: list[str] = []
    for section in ("files", "weights", "env"):
        a = old.get(section) or {}
        b = new.get(section) or {}
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                diff.append(f"{section}:{key}")
    if old.get("extra") != new.get("extra"):
        diff.append("extra")
    return tuple(diff)
