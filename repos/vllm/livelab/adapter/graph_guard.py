"""O(1) runtime CUDA-graph staleness guard for vLLM live-lab workers.

Tracks monotonic `code_epoch` (bumped on live edits/rollbacks) and `pool_epoch` (bumped on
graph-pool teardown/recapture), plus per-graph `(code_epoch, pool_epoch, input_signature)`
stamps. `check_before_replay(key, input_sig=None)` verifies freshness in O(1) nanoseconds and
fails closed with `StaleGraphReplayError` on epoch or input-signature drift (no `torch` import).
"""
from typing import Any


class StaleGraphReplayError(RuntimeError):
    """Fail-closed error raised when replaying a stale or signature-mismatched CUDA graph."""

    def __init__(self, message: str, *, graph_key: Any = None, expected_epoch=None,
                 actual_epoch=None, expected_sig=None, actual_sig=None, reason: str = "stale"):
        super().__init__(message)
        self.graph_key, self.reason = graph_key, reason
        self.expected_epoch, self.actual_epoch = expected_epoch, actual_epoch
        self.expected_sig, self.actual_sig = expected_sig, actual_sig


def graph_key(mode: Any, desc: Any) -> tuple[str, Any]:
    """Canonical hashable `(mode_name, descriptor)` key for a captured graph bucket."""
    mode_name = getattr(mode, "name", None) or str(mode)
    try:
        hash(desc)
        return (mode_name, desc)
    except TypeError:
        return (mode_name, getattr(desc, "num_tokens", id(desc)))


def input_signature(*items: Any) -> tuple[Any, ...]:
    """Immutable `(shape, dtype, data_ptr)` signature tuple for tensors or stand-ins."""
    out = []
    for x in items:
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            ptr = getattr(x, "data_ptr", None)
            out.append((tuple(int(d) for d in x.shape), str(x.dtype), int(ptr()) if callable(ptr) else None))
        else:
            out.append(x)
    return tuple(out)


class GraphReplayGuard:
    """Per-rank O(1) staleness guard; `_epoch` is a shared tuple for pointer-fast equality."""

    __slots__ = ("_code_epoch", "_pool_epoch", "_epoch", "_stamps",
                 "_strict_unstamped", "_stale_rejections", "_last_reason")

    def __init__(self, *, strict_unstamped: bool = False):
        self._code_epoch = self._pool_epoch = self._stale_rejections = 0
        self._epoch: tuple[int, int] = (0, 0)
        self._stamps: dict[Any, tuple[tuple[int, int], Any]] = {}
        self._strict_unstamped, self._last_reason = bool(strict_unstamped), ""

    @property
    def code_epoch(self) -> int: return self._code_epoch
    @property
    def pool_epoch(self) -> int: return self._pool_epoch
    @property
    def epoch(self) -> tuple[int, int]: return self._epoch
    @property
    def stamps(self) -> dict[Any, tuple[int, int, Any]]:
        return {k: (ep[0], ep[1], sig) for k, (ep, sig) in self._stamps.items()}

    def bump_code_epoch(self, reason: str = "") -> int:
        self._code_epoch += 1
        self._epoch, self._last_reason = (self._code_epoch, self._pool_epoch), reason
        return self._code_epoch

    def bump_pool_epoch(self, reason: str = "") -> int:
        self._pool_epoch += 1
        self._epoch, self._last_reason = (self._code_epoch, self._pool_epoch), reason
        return self._pool_epoch

    def stamp(self, key: Any, input_sig: Any = None) -> tuple[int, int, Any]:
        ep = self._epoch
        self._stamps[key] = (ep, input_sig)
        return (ep[0], ep[1], input_sig)

    def stamp_many(self, keys, input_sig: Any = None) -> int:
        ep, stamps, count = self._epoch, self._stamps, 0
        for key in keys:
            stamps[key] = (ep, input_sig)
            count += 1
        return count

    def invalidate(self, key: Any = None) -> int:
        if key is None:
            n = len(self._stamps)
            self._stamps.clear()
            return n
        return 1 if self._stamps.pop(key, None) is not None else 0

    def invalidate_modes(self, mode_names) -> int:
        targets = set(mode_names)
        drop = [k for k in self._stamps if isinstance(k, tuple) and k and k[0] in targets]
        for k in drop:
            self._stamps.pop(k, None)
        return len(drop)

    def check_before_replay(self, key: Any, input_sig: Any = None) -> None:
        """O(1) staleness check; raises `StaleGraphReplayError` on epoch or input drift."""
        stamp = self._stamps.get(key)
        if stamp is not None:
            s_epoch, s_sig = stamp
            if (s_epoch is self._epoch or s_epoch == self._epoch) and (
                input_sig is None or (s_sig is not None and s_sig == input_sig)
            ):
                return
            self._stale_rejections += 1
            if s_epoch != self._epoch:
                raise StaleGraphReplayError(
                    f"stale CUDA graph replay for {key!r}: captured at {s_epoch}, live is {self._epoch}",
                    graph_key=key, expected_epoch=self._epoch, actual_epoch=s_epoch, reason="epoch_mismatch",
                )
            raise StaleGraphReplayError(
                f"CUDA graph input signature drift for {key!r}: captured {s_sig!r}, got {input_sig!r}",
                graph_key=key, expected_epoch=self._epoch, actual_epoch=s_epoch,
                expected_sig=s_sig, actual_sig=input_sig, reason="input_sig_mismatch",
            )
        if not self._strict_unstamped and self._code_epoch == 0 and self._pool_epoch == 0 and input_sig is None:
            return
        self._stale_rejections += 1
        raise StaleGraphReplayError(
            f"graph {key!r} has no capture stamp at {self._epoch}",
            graph_key=key, expected_epoch=self._epoch, actual_epoch=None,
            actual_sig=input_sig, reason="unstamped",
        )

    def is_fresh(self, key: Any, input_sig: Any = None) -> bool:
        stamp = self._stamps.get(key)
        if stamp is not None:
            s_epoch, s_sig = stamp
            return (s_epoch is self._epoch or s_epoch == self._epoch) and (
                input_sig is None or (s_sig is not None and s_sig == input_sig)
            )
        return not self._strict_unstamped and self._code_epoch == 0 and self._pool_epoch == 0 and input_sig is None

    def wrap_replay(self, key: Any, replay_fn, *, sig_fn=None):
        """Wrap `replay_fn` so `check_before_replay(key, ...)` guards every replay."""
        guard = self
        def guarded(*args, **kwargs):
            guard.check_before_replay(key, sig_fn(*args, **kwargs) if sig_fn is not None else None)
            return replay_fn(*args, **kwargs)
        guarded.__wrapped__ = replay_fn
        return guarded

    def attach_wrapper(self, wrapper, *, mode: Any = None, key_fn=None):
        """Attach an idempotent replay guard to `wrapper.replay` (zero overhead at epoch 0)."""
        if getattr(wrapper, "_rsibox_guard", None) is self or not callable(getattr(wrapper, "replay", None)):
            return wrapper
        replay_fn, guard = wrapper.replay, self
        mode_obj = mode if mode is not None else getattr(wrapper, "runtime_mode", "FULL")
        def guarded_replay(*args, **kwargs):
            if guard._code_epoch or guard._pool_epoch or guard._stamps:
                k = key_fn(*args, **kwargs) if key_fn else graph_key(mode_obj, args[0] if args else kwargs.get("desc"))
                guard.check_before_replay(k, kwargs.get("input_sig"))
            return replay_fn(*args, **kwargs)
        guarded_replay.__wrapped__ = replay_fn
        wrapper.replay = guarded_replay
        wrapper._rsibox_guard = self
        return wrapper

    def stats(self) -> dict[str, Any]:
        return {"code_epoch": self._code_epoch, "pool_epoch": self._pool_epoch,
                "stamped_graphs": len(self._stamps), "stale_rejections": self._stale_rejections}
