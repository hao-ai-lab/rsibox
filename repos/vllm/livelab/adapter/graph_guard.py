"""O(1) runtime CUDA-graph staleness guard for vLLM live-lab workers.

Tracks a monotonic `code_epoch` (ABA-free on undo) and per-mode `pool_epoch` counters,
plus per-graph `(code_epoch, mode_pool_epoch, input_sig)` stamps. `check_before_replay`
verifies freshness in O(1) and fails closed with `StaleGraphReplayError` (no `torch` import).
"""
import threading
from typing import Any


class StaleGraphReplayError(RuntimeError):
    """Fail-closed error raised when replaying a stale or signature-mismatched CUDA graph."""

    def __init__(self, message: str, *, graph_key: Any = None, expected_epoch=None,
                 actual_epoch=None, expected_sig=None, actual_sig=None, reason: str = "stale"):
        super().__init__(message)
        self.graph_key, self.reason = graph_key, reason
        self.expected_epoch, self.actual_epoch = expected_epoch, actual_epoch
        self.expected_sig, self.actual_sig = expected_sig, actual_sig


def _mode_name(mode: Any) -> str:
    return getattr(mode, "name", None) or str(mode)


def _key_mode(key: Any) -> str:
    return key[0] if isinstance(key, tuple) and key and isinstance(key[0], str) else "*"


def graph_key(mode: Any, desc: Any) -> tuple[str, Any]:
    """Canonical hashable `(mode_name, descriptor)` key (never falls back to unsafe `id()`)."""
    m = _mode_name(mode)
    try:
        hash(desc)
        return (m, desc)
    except TypeError:
        attrs = tuple((a, getattr(desc, a)) for a in ("num_tokens", "num_reqs", "uniform", "has_lora") if hasattr(desc, a))
        return (m, attrs or repr(desc))


def input_signature(*items: Any) -> tuple[Any, ...]:
    """Immutable `(shape, dtype, data_ptr)` signature tuple for tensors or stand-ins."""
    return tuple(
        (tuple(int(d) for d in x.shape), str(x.dtype),
         int(x.data_ptr()) if callable(getattr(x, "data_ptr", None)) else None)
        if hasattr(x, "shape") and hasattr(x, "dtype") else x
        for x in items
    )


class GraphReplayGuard:
    """Thread-safe per-rank O(1) staleness guard for CUDA-graph replay."""

    __slots__ = ("_code_epoch", "_next_code_epoch", "_pool_epochs", "_stamps",
                 "_strict_unstamped", "_stale_rejections", "_last_reason", "_lock")

    def __init__(self, *, strict_unstamped: bool = False):
        self._code_epoch, self._next_code_epoch, self._stale_rejections = 0, 1, 0
        self._pool_epochs: dict[str, int] = {}
        self._stamps: dict[Any, tuple[int, int, Any]] = {}
        self._strict_unstamped, self._last_reason, self._lock = bool(strict_unstamped), "", threading.RLock()

    @property
    def code_epoch(self) -> int: return self._code_epoch
    @property
    def pool_epoch(self) -> int: return sum(self._pool_epochs.values())
    @property
    def pool_epochs(self) -> dict[str, int]:
        with self._lock: return dict(self._pool_epochs)
    @property
    def epoch(self) -> tuple[int, int]: return (self._code_epoch, self.pool_epoch)
    @property
    def stamps(self) -> dict[Any, tuple[int, int, Any]]:
        with self._lock: return dict(self._stamps)

    def pool_epoch_for(self, mode: Any) -> int:
        return self._pool_epochs.get(_mode_name(mode), 0)

    def bump_code_epoch(self, reason: str = "") -> int:
        with self._lock:
            self._code_epoch, self._next_code_epoch, self._last_reason = self._next_code_epoch, self._next_code_epoch + 1, reason
            return self._code_epoch

    def restore_code_epoch(self, epoch: int, reason: str = "") -> int:
        with self._lock:
            self._code_epoch, self._last_reason = int(epoch), reason
            self._next_code_epoch = max(self._next_code_epoch, self._code_epoch + 1)
            return self._code_epoch

    def bump_pool_epoch(self, reason: str = "", *, modes=None) -> int:
        with self._lock:
            names = ([_mode_name(m) for m in modes] if modes is not None
                     else list(set(self._pool_epochs) | {_key_mode(k) for k in self._stamps} or {"*"}))
            for m in names:
                self._pool_epochs[m] = self._pool_epochs.get(m, 0) + 1
            if names:
                self._last_reason = reason
            return self.pool_epoch

    def snapshot_modes(self, modes) -> tuple[dict[str, int], dict[Any, tuple[int, int, Any]]]:
        names = {_mode_name(m) for m in modes}
        with self._lock:
            return ({m: self._pool_epochs.get(m, 0) for m in names},
                    {k: v for k, v in self._stamps.items() if _key_mode(k) in names})

    def restore_modes(self, pool_epochs: dict[str, int], stamps: dict[Any, tuple[int, int, Any]]) -> None:
        with self._lock:
            self._invalidate_modes_unlocked(pool_epochs.keys())
            for m, ep in pool_epochs.items():
                if ep: self._pool_epochs[m] = ep
                else: self._pool_epochs.pop(m, None)
            self._stamps.update(stamps)

    def stamp(self, key: Any, input_sig: Any = None) -> tuple[int, int, Any]:
        with self._lock:
            val = (self._code_epoch, self._pool_epochs.get(_key_mode(key), 0), input_sig)
            self._stamps[key] = val
            return val

    def stamp_many(self, keys, input_sig: Any = None) -> int:
        with self._lock:
            ce, pe, stamps, n = self._code_epoch, self._pool_epochs, self._stamps, 0
            for k in keys:
                stamps[k] = (ce, pe.get(_key_mode(k), 0), input_sig); n += 1
            return n

    def invalidate(self, key: Any = None) -> int:
        with self._lock:
            if key is None:
                n = len(self._stamps); self._stamps.clear(); return n
            return int(self._stamps.pop(key, None) is not None)

    def _invalidate_modes_unlocked(self, mode_names) -> int:
        targets = {_mode_name(m) for m in mode_names}
        drop = [k for k in self._stamps if _key_mode(k) in targets]
        for k in drop: self._stamps.pop(k, None)
        return len(drop)

    def invalidate_modes(self, mode_names) -> int:
        with self._lock: return self._invalidate_modes_unlocked(mode_names)

    def _mismatch_unlocked(self, key: Any, input_sig: Any):
        live, stamp = (self._code_epoch, self._pool_epochs.get(_key_mode(key), 0)), self._stamps.get(key)
        if stamp is not None:
            s_code, s_pool, s_sig = stamp
            if s_code == live[0] and s_pool == live[1] and (input_sig is None or (s_sig is not None and s_sig == input_sig)):
                return None
            if (s_code, s_pool) != live:
                return StaleGraphReplayError(f"stale CUDA graph replay for {key!r}: captured {(s_code, s_pool)}, live {live}",
                                             graph_key=key, expected_epoch=live, actual_epoch=(s_code, s_pool), reason="epoch_mismatch")
            return StaleGraphReplayError(f"CUDA graph input signature drift for {key!r}: captured {s_sig!r}, got {input_sig!r}",
                                         graph_key=key, expected_epoch=live, actual_epoch=(s_code, s_pool),
                                         expected_sig=s_sig, actual_sig=input_sig, reason="input_sig_mismatch")
        if not self._strict_unstamped and live == (0, 0) and input_sig is None:
            return None
        return StaleGraphReplayError(f"graph {key!r} has no capture stamp at {live}",
                                     graph_key=key, expected_epoch=live, actual_epoch=None, actual_sig=input_sig, reason="unstamped")

    def check_before_replay(self, key: Any, input_sig: Any = None) -> None:
        with self._lock:
            if (err := self._mismatch_unlocked(key, input_sig)) is not None:
                self._stale_rejections += 1; raise err

    def is_fresh(self, key: Any, input_sig: Any = None) -> bool:
        with self._lock: return self._mismatch_unlocked(key, input_sig) is None

    def wrap_replay(self, key: Any, replay_fn, *, sig_fn=None):
        guard = self
        def guarded(*args, **kwargs):
            guard.check_before_replay(key, sig_fn(*args, **kwargs) if sig_fn is not None else None)
            return replay_fn(*args, **kwargs)
        guarded.__wrapped__ = replay_fn
        return guarded

    def attach_wrapper(self, wrapper, *, mode: Any = None, method: str = "replay", key_fn=None):
        attr = f"_rsibox_guard_{method}"
        if getattr(wrapper, attr, None) is self or not callable(getattr(wrapper, method, None)):
            return wrapper
        orig, guard = getattr(wrapper, method), self
        def guarded(*args, **kwargs):
            desc = args[0] if args else kwargs.get("desc", kwargs.get("batch_descriptor"))
            m = getattr(desc, "cg_mode", None) or (mode if mode is not None else getattr(wrapper, "runtime_mode", "FULL"))
            guard.check_before_replay(key_fn(*args, **kwargs) if key_fn else graph_key(m, desc), kwargs.get("input_sig"))
            return orig(*args, **kwargs)
        guarded.__wrapped__ = orig
        setattr(wrapper, method, guarded); setattr(wrapper, attr, self)
        return wrapper

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"code_epoch": self._code_epoch, "pool_epoch": sum(self._pool_epochs.values()),
                    "pool_epochs": dict(self._pool_epochs), "stamped_graphs": len(self._stamps),
                    "stale_rejections": self._stale_rejections}
