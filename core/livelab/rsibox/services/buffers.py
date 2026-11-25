"""S5: Indexed Prefetch & Versioned RL Replay Buffers (pure Python stdlib).

Provides two thread-safe, context-managed volatile/stateful buffer services:
- `IndexedPrefetchService`: mmap-backed random/cursor dataset reader backed by a
  headered binary offset + SHA-256-prefix table and atomic cursor checkpoints.
- `VersionedReplayBufferService`: bounded, `code_version`-fenced RL rollout
  replay buffer supporting stale filtering, version promotion, and epoch resets.
"""
from __future__ import annotations

from collections import deque
import contextlib, hashlib, json, mmap, os, random, struct, tempfile, threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

_INDEX_MAGIC = b"RSIIDX01"
_HEADER_STRUCT = struct.Struct("<8sQQ")
_ENTRY_STRUCT = struct.Struct("<QQ8s")
_CHECKPOINT_SCHEMA = 1
_SCAN_CHUNK_BYTES = 65536


class BufferServiceError(RuntimeError): """Base error raised by `rsibox.services.buffers`."""
class BufferClosedError(BufferServiceError): """Raised fail-closed when operating on a closed buffer service."""
class IndexCorruptError(BufferServiceError): """Raised fail-closed when a binary index, payload, or checkpoint is corrupt."""
class CursorOutOfRangeError(BufferServiceError): """Raised when a cursor or record index is negative or out of bounds."""
class EmptyBufferError(BufferServiceError): """Raised when sampling from a replay buffer with insufficient eligible entries."""
class StaleVersionError(BufferServiceError): """Raised fail-closed when pushing or sampling mismatched `code_version` entries."""


@dataclass(frozen=True)
class PrefetchItem:
    """Immutable dataset record returned by `IndexedPrefetchService`."""
    index: int; offset: int; length: int; payload: bytes


@dataclass(frozen=True)
class VersionedEntry:
    """Immutable rollout entry stored in `VersionedReplayBufferService`."""
    seq: int; code_version: str; payload: Any; metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BufferResetReceipt:
    """Structured outcome returned by `advance_code_version()` and `reset_state()`."""
    drained: int; retained: int; active_code_version: str; epoch: int


def _check_str(value: object, name: str) -> str:
    if not isinstance(value, str): raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if "\x00" in value or not value.strip(): raise ValueError(f"{name} must be a non-empty str without NUL, got {value!r}")
    return value


def _check_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool): raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _check_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int): raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _validate_path(value: object, name: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be path-like, got {type(value).__name__}")
    return Path(_check_str(os.fspath(value), name))


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as fh:
            tmp_path = Path(fh.name)
            fh.write(data)
            fh.flush()
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError): tmp_path.unlink()


def _reject_nonfinite(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _scan_delimited_file(data_path: Path, delimiter: bytes) -> tuple[list[tuple[int, int, bytes]], int]:
    entries: list[tuple[int, int, bytes]] = []
    dlen = len(delimiter)
    with open(data_path, "rb") as df:
        data_size = os.fstat(df.fileno()).st_size
        if data_size == 0:
            return entries, 0
        buf, buf_offset, rec_start, hasher = b"", 0, 0, hashlib.sha256()
        while chunk := df.read(_SCAN_CHUNK_BYTES):
            buf += chunk
            search_from = 0
            while (idx := buf.find(delimiter, search_from)) != -1:
                end_in_file = buf_offset + idx
                hasher.update(buf[search_from:idx])
                entries.append((rec_start, end_in_file - rec_start, hasher.digest()[:8]))
                hasher = hashlib.sha256()
                rec_start = end_in_file + dlen
                search_from = idx + dlen
            safe_flush = max(search_from, len(buf) - (dlen - 1))
            if safe_flush > search_from:
                hasher.update(buf[search_from:safe_flush])
            buf_offset += safe_flush
            buf = buf[safe_flush:]
        if buf:
            hasher.update(buf)
        if rec_start < data_size:
            entries.append((rec_start, data_size - rec_start, hasher.digest()[:8]))
    return entries, data_size


def _write_records_atomic(data_path: Path, records: Iterable[bytes], delimiter: bytes) -> tuple[list[tuple[int, int, bytes]], int]:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[int, int, bytes]] = []
    tmp_path: Path | None = None
    offset = 0
    try:
        with tempfile.NamedTemporaryFile(dir=data_path.parent, prefix=f".{data_path.name}.", suffix=".tmp", delete=False) as fh:
            tmp_path = Path(fh.name)
            for idx, item in enumerate(records):
                if not isinstance(item, (bytes, bytearray, memoryview)):
                    raise TypeError(f"records[{idx}] must be bytes-like, got {type(item).__name__}")
                raw = bytes(item)
                fh.write(raw)
                if delimiter:
                    fh.write(delimiter)
                entries.append((offset, len(raw), hashlib.sha256(raw).digest()[:8]))
                offset += len(raw) + len(delimiter)
            fh.flush()
        os.replace(tmp_path, data_path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError): tmp_path.unlink()
    return entries, offset


class IndexedPrefetchService:
    """Thread-safe, mmap-backed random/cursor dataset reader with binary offset index."""

    def __init__(self, data_path: str | os.PathLike[str], index_path: str | os.PathLike[str], *, start_cursor: int = 0) -> None:
        self._lock = threading.RLock()
        self._file: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None
        self._closed = True
        self._data_path = _validate_path(data_path, "data_path")
        self._index_path = _validate_path(index_path, "index_path")
        cursor = _check_int(start_cursor, "start_cursor")
        if cursor < 0:
            raise CursorOutOfRangeError(f"start_cursor must be >= 0, got {cursor}")

        try:
            self._file = open(self._data_path, "rb")
            self._data_size = os.fstat(self._file.fileno()).st_size
            if self._data_size > 0:
                self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            idx_bytes = self._index_path.read_bytes()
            self._entries = self._parse_index(idx_bytes, self._data_size)
            self._index_sha256 = hashlib.sha256(idx_bytes).hexdigest()
            if cursor > len(self._entries):
                raise CursorOutOfRangeError(f"start_cursor {cursor} exceeds record count {len(self._entries)}")
            self._cursor = cursor
            self._closed = False
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(cls, data_path: str | os.PathLike[str], index_path: str | os.PathLike[str], *, start_cursor: int = 0) -> IndexedPrefetchService:
        """Open and validate an `IndexedPrefetchService` over `data_path` and `index_path`."""
        return cls(data_path, index_path, start_cursor=start_cursor)

    @classmethod
    def build_index(
        cls, data_path: str | os.PathLike[str], index_path: str | os.PathLike[str],
        delimiter: bytes = b"\n", *, records: Iterable[bytes] | None = None,
    ) -> int:
        """Build an atomic binary offset index for `data_path`, optionally writing `records` first."""
        dpath, ipath = _validate_path(data_path, "data_path"), _validate_path(index_path, "index_path")
        if not isinstance(delimiter, bytes):
            raise TypeError(f"delimiter must be bytes, got {type(delimiter).__name__}")
        if records is None:
            if not delimiter:
                raise ValueError("delimiter must be non-empty bytes when indexing an existing data file")
            entries, data_size = _scan_delimited_file(dpath, delimiter)
        else:
            entries, data_size = _write_records_atomic(dpath, records, delimiter)

        buf = bytearray(_HEADER_STRUCT.pack(_INDEX_MAGIC, len(entries), data_size))
        for offset, length, digest8 in entries:
            buf.extend(_ENTRY_STRUCT.pack(offset, length, digest8))
        _atomic_write_bytes(ipath, bytes(buf))
        return len(entries)

    @staticmethod
    def _parse_index(idx_bytes: bytes, data_size: int) -> list[tuple[int, int, bytes]]:
        if len(idx_bytes) < _HEADER_STRUCT.size:
            raise IndexCorruptError(f"index too short: {len(idx_bytes)} bytes")
        magic, count, indexed_size = _HEADER_STRUCT.unpack_from(idx_bytes, 0)
        if magic != _INDEX_MAGIC:
            raise IndexCorruptError(f"invalid index magic: {magic!r}")
        if indexed_size != data_size:
            raise IndexCorruptError(f"data size mismatch: index expects {indexed_size}, file has {data_size}")
        body_size = len(idx_bytes) - _HEADER_STRUCT.size
        if body_size % _ENTRY_STRUCT.size != 0 or body_size // _ENTRY_STRUCT.size != count:
            raise IndexCorruptError(f"index length mismatch for {count} entries: {len(idx_bytes)} bytes")

        entries: list[tuple[int, int, bytes]] = []
        prev_end, pos = 0, _HEADER_STRUCT.size
        for idx in range(count):
            offset, length, digest8 = _ENTRY_STRUCT.unpack_from(idx_bytes, pos)
            pos += _ENTRY_STRUCT.size
            if offset < prev_end or offset > data_size or length > data_size - offset:
                raise IndexCorruptError(f"invalid entry[{idx}] offset={offset} length={length} for data_size={data_size}")
            entries.append((offset, length, digest8))
            prev_end = offset + length
        return entries

    @property
    def closed(self) -> bool:
        with self._lock: return self._closed

    def __enter__(self) -> IndexedPrefetchService:
        with self._lock: self._require_open_locked()
        return self

    def __exit__(self, *_exc: object) -> None: self.close()

    def _require_open_locked(self) -> None:
        if self._closed: raise BufferClosedError("IndexedPrefetchService is closed")

    def __len__(self) -> int:
        with self._lock: self._require_open_locked(); return len(self._entries)

    def cursor(self) -> int:
        """Return the current sequential cursor position."""
        with self._lock: self._require_open_locked(); return self._cursor

    def _read_item_locked(self, index: int) -> PrefetchItem:
        offset, length, digest8 = self._entries[index]
        if length == 0:
            payload = b""
        else:
            if self._mmap is None or offset + length > len(self._mmap):
                raise IndexCorruptError(f"mmap out of bounds for index={index} ({offset}+{length})")
            payload = self._mmap[offset : offset + length]
        if len(payload) != length or hashlib.sha256(payload).digest()[:8] != digest8:
            raise IndexCorruptError(f"payload checksum/length mismatch at index={index}")
        return PrefetchItem(index=index, offset=offset, length=length, payload=payload)

    def get(self, index: int) -> PrefetchItem:
        """Read and verify record `index` from the mmap region."""
        idx = _check_int(index, "index")
        with self._lock:
            self._require_open_locked()
            if idx < 0 or idx >= len(self._entries):
                raise CursorOutOfRangeError(f"index {idx} out of range [0, {len(self._entries)})")
            return self._read_item_locked(idx)

    def seek(self, cursor: int) -> int:
        """Move the sequential cursor to `cursor` (inclusive of EOF `len(self)`)."""
        pos = _check_int(cursor, "cursor")
        with self._lock:
            self._require_open_locked()
            if pos < 0 or pos > len(self._entries):
                raise CursorOutOfRangeError(f"cursor {pos} out of range [0, {len(self._entries)}]")
            self._cursor = pos
            return self._cursor

    def next_batch(self, batch_size: int) -> list[PrefetchItem]:
        """Read up to `batch_size` items starting at `cursor()` and advance the cursor."""
        size = _check_int(batch_size, "batch_size")
        if size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {size}")
        with self._lock:
            self._require_open_locked()
            end = min(self._cursor + size, len(self._entries))
            batch = [self._read_item_locked(idx) for idx in range(self._cursor, end)]
            self._cursor = end
            return batch

    def checkpoint_cursor(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        """Atomically persist the current cursor bound to the index digest."""
        cpath = _validate_path(path, "path")
        with self._lock:
            self._require_open_locked()
            doc: dict[str, Any] = {
                "schema": _CHECKPOINT_SCHEMA, "cursor": self._cursor,
                "count": len(self._entries), "data_size": self._data_size, "index_sha256": self._index_sha256,
            }
        _atomic_write_bytes(cpath, json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        return dict(doc)

    def restore_cursor(self, path: str | os.PathLike[str]) -> int:
        """Validate and restore a previously checkpointed cursor position."""
        cpath = _validate_path(path, "path")
        with self._lock: self._require_open_locked()
        try: doc = json.loads(cpath.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise IndexCorruptError(f"corrupt cursor checkpoint {cpath}: {exc}") from exc
        if not isinstance(doc, dict):
            raise IndexCorruptError(f"cursor checkpoint must be a JSON object: {cpath}")
        schema, cursor = doc.get("schema"), doc.get("cursor")
        if isinstance(schema, bool) or schema != _CHECKPOINT_SCHEMA:
            raise IndexCorruptError(f"invalid cursor checkpoint schema in {cpath}")
        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise IndexCorruptError(f"invalid cursor value in {cpath}: {cursor!r}")
        with self._lock:
            self._require_open_locked()
            if (doc.get("count") != len(self._entries) or doc.get("data_size") != self._data_size
                    or doc.get("index_sha256") != self._index_sha256):
                raise IndexCorruptError(f"cursor checkpoint does not match active index: {cpath}")
            if cursor < 0 or cursor > len(self._entries):
                raise CursorOutOfRangeError(f"restored cursor {cursor} out of range [0, {len(self._entries)}]")
            self._cursor = cursor
            return self._cursor

    def close(self) -> None:
        """Idempotently close the mmap and underlying file descriptor."""
        with self._lock:
            mm, fh = self._mmap, self._file
            self._mmap, self._file, self._closed = None, None, True
            try:
                if mm is not None: mm.close()
            finally:
                if fh is not None: fh.close()


class VersionedReplayBufferService:
    """Thread-safe, `code_version`-fenced bounded RL replay/rollout buffer."""

    def __init__(
        self, capacity: int, *, current_code_version: str = "v0",
        drain_on_version_mismatch: bool = False, reject_stale_push: bool = False,
    ) -> None:
        cap = _check_int(capacity, "capacity")
        if cap <= 0:
            raise ValueError(f"capacity must be >= 1, got {cap}")
        self._capacity = cap
        self._current_code_version = _check_str(current_code_version, "current_code_version")
        self._drain_on_version_mismatch = _check_bool(drain_on_version_mismatch, "drain_on_version_mismatch")
        self._reject_stale_push = _check_bool(reject_stale_push, "reject_stale_push")
        self._lock = threading.RLock()
        self._entries: deque[VersionedEntry] = deque(maxlen=cap)
        self._seq = self._epoch = self._evicted = self._drained = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock: return self._closed

    @property
    def current_code_version(self) -> str:
        with self._lock: self._require_open_locked(); return self._current_code_version

    def __enter__(self) -> VersionedReplayBufferService:
        with self._lock: self._require_open_locked()
        return self

    def __exit__(self, *_exc: object) -> None: self.close()

    def _require_open_locked(self) -> None:
        if self._closed: raise BufferClosedError("VersionedReplayBufferService is closed")

    def __len__(self) -> int:
        with self._lock: self._require_open_locked(); return len(self._entries)

    def _drain_not_matching_locked(self, version: str) -> int:
        kept = [e for e in self._entries if e.code_version == version]
        drained = len(self._entries) - len(kept)
        self._entries = deque(kept, maxlen=self._capacity)
        self._drained += drained
        return drained

    def push(self, payload: Any, *, code_version: str | None = None, metadata: Mapping[str, Any] | None = None) -> VersionedEntry:
        """Append a rollout entry tagged with `code_version`, evicting oldest at capacity."""
        explicit_ver = _check_str(code_version, "code_version") if code_version is not None else None
        if metadata is None: meta_copy: dict[str, Any] = {}
        elif isinstance(metadata, Mapping): meta_copy = dict(metadata)
        else: raise TypeError(f"metadata must be a mapping or None, got {type(metadata).__name__}")

        with self._lock:
            self._require_open_locked()
            ver = self._current_code_version if explicit_ver is None else explicit_ver
            if ver != self._current_code_version:
                if self._reject_stale_push:
                    raise StaleVersionError(f"pushed code_version {ver!r} does not match active {self._current_code_version!r}")
                if self._drain_on_version_mismatch:
                    self._drain_not_matching_locked(self._current_code_version)
            if len(self._entries) == self._capacity:
                self._evicted += 1
            self._seq += 1
            entry = VersionedEntry(seq=self._seq, code_version=ver, payload=payload, metadata=meta_copy)
            self._entries.append(entry)
            return VersionedEntry(seq=entry.seq, code_version=entry.code_version, payload=entry.payload, metadata=dict(entry.metadata))

    def sample(
        self, batch_size: int, *, code_version: str | None = None,
        allow_stale: bool = False, exact: bool = False, rng: random.Random | None = None,
    ) -> list[VersionedEntry]:
        """Sample up to `batch_size` entries filtered by `code_version` unless `allow_stale=True`."""
        size = _check_int(batch_size, "batch_size")
        if size <= 0: raise ValueError(f"batch_size must be >= 1, got {size}")
        allow_stale_flag, exact_flag = _check_bool(allow_stale, "allow_stale"), _check_bool(exact, "exact")
        target_ver = _check_str(code_version, "code_version") if code_version is not None else None
        if rng is not None and (isinstance(rng, bool) or not isinstance(rng, random.Random)):
            raise TypeError(f"rng must be a random.Random or None, got {type(rng).__name__}")

        with self._lock:
            self._require_open_locked()
            if not allow_stale_flag:
                if target_ver is not None and target_ver != self._current_code_version:
                    raise StaleVersionError(f"requested code_version {target_ver!r} does not match active {self._current_code_version!r}")
                ver_filter: str | None = self._current_code_version
            else:
                ver_filter = target_ver
            eligible = [e for e in self._entries if ver_filter is None or e.code_version == ver_filter]
            if not eligible or (exact_flag and len(eligible) < size):
                raise EmptyBufferError(f"insufficient eligible entries (have {len(eligible)}, need {size}, version={ver_filter!r})")
            sampler = rng if rng is not None else random
            chosen = sampler.sample(eligible, min(size, len(eligible)))
            return [VersionedEntry(seq=e.seq, code_version=e.code_version, payload=e.payload, metadata=dict(e.metadata)) for e in chosen]

    def advance_code_version(self, new_version: str, *, drain_stale: bool = True) -> BufferResetReceipt:
        """Promote active `code_version`, bump epoch, and optionally drain stale entries."""
        ver, drain_flag = _check_str(new_version, "new_version"), _check_bool(drain_stale, "drain_stale")
        with self._lock:
            self._require_open_locked()
            self._epoch += 1
            self._current_code_version = ver
            drained = self._drain_not_matching_locked(ver) if drain_flag else 0
            return BufferResetReceipt(drained=drained, retained=len(self._entries), active_code_version=ver, epoch=self._epoch)

    def reset_state(self) -> BufferResetReceipt:
        """Drain all buffered entries on `reset('state')` and bump the buffer epoch."""
        with self._lock:
            self._require_open_locked()
            drained = len(self._entries)
            self._entries.clear()
            self._epoch += 1
            self._drained += drained
            return BufferResetReceipt(drained=drained, retained=0, active_code_version=self._current_code_version, epoch=self._epoch)

    def stats(self) -> dict[str, Any]:
        """Return buffer occupancy, version histogram, epoch, and eviction/drain counters."""
        with self._lock:
            counts: dict[str, int] = {}
            for entry in self._entries:
                counts[entry.code_version] = counts.get(entry.code_version, 0) + 1
            return {
                "capacity": self._capacity, "size": len(self._entries), "current_code_version": self._current_code_version,
                "epoch": self._epoch, "total_pushed": self._seq, "evicted": self._evicted,
                "drained": self._drained, "closed": self._closed, "version_counts": counts,
            }

    def close(self) -> None:
        """Idempotently close the replay buffer and clear retained entries."""
        with self._lock:
            self._entries.clear()
            self._closed = True


__all__ = [
    "BufferClosedError", "BufferResetReceipt", "BufferServiceError", "CursorOutOfRangeError",
    "EmptyBufferError", "IndexedPrefetchService", "IndexCorruptError", "PrefetchItem",
    "StaleVersionError", "VersionedEntry", "VersionedReplayBufferService",
]
