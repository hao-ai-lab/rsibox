"""S3: Shared Compile & Graph Cache Service (pure Python stdlib).

Fail-closed, cross-process and cross-container artifact cache for Triton
kernels, TorchInductor artifacts, FlashInfer kernels, and CUDA graph plans.
Entries are bound to an `EnvFingerprint` SHA-256 digest, checksummed via
payload SHA-256, and synchronized with per-entry `fcntl.flock` + `os.replace`.
"""
from __future__ import annotations

import contextlib, fcntl, hashlib, json, math, os, tempfile, threading, time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping

_SCHEMA_VERSION = 1


class CompileCacheError(RuntimeError): """Base error raised by `CompileCacheService` on cache or validation failures."""
class CacheClosedError(CompileCacheError): """Raised when operating on a closed or torn-down `CompileCacheService`."""
class FingerprintMismatchError(CompileCacheError): """Raised fail-closed when a cached entry's `EnvFingerprint` mismatches."""
class CacheCorruptError(CompileCacheError): """Raised fail-closed when a cache manifest or payload is corrupt or tampered."""


class CacheArtifactKind(str, Enum):
    """Canonical compiled artifact categories managed by `CompileCacheService`."""
    TRITON_KERNEL = "triton_kernel"
    TORCH_INDUCTOR = "torch_inductor"
    FLASHINFER = "flashinfer"
    CUDA_GRAPH_PLAN = "cuda_graph_plan"
    CUSTOM = "custom"


def _check_str(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if "\x00" in value or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a {'str' if allow_empty else 'non-empty str'} without NUL, got {value!r}")
    return value


def _validate_segment(value: object, name: str) -> str:
    text = _check_str(value, name)
    if "/" in text or "\\" in text or ".." in text or text == ".":
        raise ValueError(f"{name} must not contain '/', '\\\\', '..', or '.', got {value!r}")
    return text


def _normalize_extra(extra: object) -> tuple[tuple[str, str], ...]:
    if isinstance(extra, Mapping):
        items = list(extra.items())
    elif isinstance(extra, (tuple, list)):
        items = list(extra)
    else:
        raise TypeError(f"extra must be a sequence of (str, str) pairs or dict, got {type(extra).__name__}")
    out: list[tuple[str, str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"extra[{idx}] must be a (key, value) pair, got {item!r}")
        out.append((_check_str(item[0], f"extra[{idx}][0]"), _check_str(item[1], f"extra[{idx}][1]", allow_empty=True)))
    return tuple(sorted(out))


@dataclass(frozen=True)
class EnvFingerprint:
    """Immutable environment fingerprint binding cached artifacts to GPU/runtime ABI."""
    gpu_arch: str
    cuda_version: str
    triton_version: str = ""
    torch_version: str = ""
    driver_version: str = ""
    engine: str = "generic"
    extra: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for attr, allow_empty in (
            ("gpu_arch", False), ("cuda_version", False), ("triton_version", True),
            ("torch_version", True), ("driver_version", True), ("engine", False),
        ):
            object.__setattr__(self, attr, _check_str(getattr(self, attr), attr, allow_empty=allow_empty))
        object.__setattr__(self, "extra", _normalize_extra(self.extra))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "gpu_arch": self.gpu_arch, "cuda_version": self.cuda_version, "triton_version": self.triton_version,
            "torch_version": self.torch_version, "driver_version": self.driver_version, "engine": self.engine,
            "extra": [[k, v] for k, v in sorted(self.extra)],
        }

    @classmethod
    def from_dict(cls, data: object) -> EnvFingerprint:
        """Construct and validate an `EnvFingerprint` from a dict, failing closed."""
        if not isinstance(data, dict):
            raise TypeError(f"fingerprint data must be a dict, got {type(data).__name__}")
        for req in ("gpu_arch", "cuda_version"):
            if req not in data:
                raise ValueError(f"fingerprint missing required field {req!r}")
        raw_extra = data.get("extra", ())
        if isinstance(raw_extra, (str, bytes, bytearray)):
            raise TypeError(f"extra must be a sequence of pairs or dict, not {type(raw_extra).__name__}")
        return cls(
            gpu_arch=data["gpu_arch"], cuda_version=data["cuda_version"],
            triton_version=data.get("triton_version", ""), torch_version=data.get("torch_version", ""),
            driver_version=data.get("driver_version", ""), engine=data.get("engine", "generic"),
            extra=_normalize_extra(raw_extra),
        )

    def digest(self) -> str:
        """Return deterministic SHA-256 hex digest over canonical fingerprint JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """Immutable descriptor for a cached compile or CUDA-graph artifact."""
    namespace: str; key: str; kind: CacheArtifactKind; fingerprint_digest: str
    fingerprint: EnvFingerprint; payload_path: Path; size_bytes: int; sha256: str; created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize entry metadata (without host-specific paths) for sidecar JSON."""
        return {
            "schema": _SCHEMA_VERSION, "namespace": self.namespace, "key": self.key,
            "kind": self.kind.value, "fingerprint_digest": self.fingerprint_digest,
            "fingerprint": self.fingerprint.to_dict(), "size_bytes": self.size_bytes,
            "sha256": self.sha256, "created_at": self.created_at, "metadata": dict(self.metadata),
        }


def _slug_hash(token: str) -> str:
    slug = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in token)[:48].strip("._-") or "entry"
    return f"{slug}-{hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]}"


@contextlib.contextmanager
def _flock_file(lock_path: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _stage_temp_bytes(directory: Path, prefix: str, data: bytes, *, fsync: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=prefix, suffix=".tmp", delete=False) as fh:
            tmp_path = Path(fh.name)
            fh.write(data)
            fh.flush()
            if fsync:
                with contextlib.suppress(OSError): os.fsync(fh.fileno())
        return tmp_path
    except BaseException:
        if tmp_path is not None:
            with contextlib.suppress(OSError): tmp_path.unlink()
        raise

def _unlink_pair(manifest_path: Path, payload_path: Path) -> bool:
    removed = False
    for path in (manifest_path, payload_path):
        if path.exists(): path.unlink(); removed = True
    return removed


_HEX64_CHARS = frozenset("0123456789abcdef")


def _reject_nonfinite(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _parse_manifest(manifest_path: Path, payload_path: Path, expected_ns: str | None, expected_key: str | None) -> CacheEntry:
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CacheCorruptError(f"corrupt manifest {manifest_path}: {exc}") from exc
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or isinstance(schema, bool) or schema != _SCHEMA_VERSION:
        raise CacheCorruptError(f"invalid manifest header in {manifest_path}")
    try:
        ns, key = _validate_segment(doc.get("namespace"), "namespace"), _validate_segment(doc.get("key"), "key")
        if (expected_ns is not None and ns != expected_ns) or (expected_key is not None and key != expected_key):
            raise CacheCorruptError(f"manifest identity mismatch in {manifest_path}")
        kind, fp = CacheArtifactKind(doc.get("kind")), EnvFingerprint.from_dict(doc.get("fingerprint"))
        fp_digest, sha256 = _check_str(doc.get("fingerprint_digest"), "fingerprint_digest"), _check_str(doc.get("sha256"), "sha256")
        size_bytes, created_at, metadata = doc.get("size_bytes"), doc.get("created_at"), doc.get("metadata")
        if (
            fp_digest != fp.digest() or len(sha256) != 64 or not _HEX64_CHARS.issuperset(sha256)
            or isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0
            or isinstance(created_at, bool) or not isinstance(created_at, (int, float))
            or not math.isfinite(created_at) or not isinstance(metadata, dict)
        ):
            raise CacheCorruptError(f"invalid manifest fields in {manifest_path}")
    except CacheCorruptError:
        raise
    except (TypeError, ValueError) as exc:
        raise CacheCorruptError(f"corrupt manifest fields in {manifest_path}: {exc}") from exc
    return CacheEntry(
        namespace=ns, key=key, kind=kind, fingerprint_digest=fp_digest, fingerprint=fp,
        payload_path=payload_path, size_bytes=size_bytes, sha256=sha256,
        created_at=float(created_at), metadata=dict(metadata),
    )


class CompileCacheService:
    """Thread-safe, cross-process & cross-container compile/graph artifact cache."""

    def __init__(self, cache_dir: str | os.PathLike[str], *, fsync: bool = True) -> None:
        if isinstance(cache_dir, bool) or not isinstance(cache_dir, (str, os.PathLike)):
            raise TypeError(f"cache_dir must be path-like, got {type(cache_dir).__name__}")
        self._cache_dir = Path(_check_str(os.fspath(cache_dir), "cache_dir")).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._fsync, self._lock, self._closed = bool(fsync), threading.RLock(), False
        self._stats = {k: 0 for k in ("hits", "misses", "puts", "evictions", "corrupt_errors", "fingerprint_mismatches", "invalidations")}

    @property
    def cache_dir(self) -> Path: return self._cache_dir

    @property
    def closed(self) -> bool:
        with self._lock: return self._closed

    def __enter__(self) -> CompileCacheService:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None: self.close()

    def _require_open(self) -> None:
        with self._lock:
            if self._closed: raise CacheClosedError("CompileCacheService is closed")

    def _inc(self, name: str, delta: int = 1) -> None:
        with self._lock: self._stats[name] += delta

    def _entry_paths(self, namespace: str, key: str) -> tuple[Path, Path, Path, Path]:
        ns_dir, stem = self._cache_dir / _slug_hash(namespace), _slug_hash(key)
        return ns_dir, ns_dir / f"{stem}.bin", ns_dir / f"{stem}.json", ns_dir / f"{stem}.lock"

    @staticmethod
    def _validate_fp(fp: object, name: str) -> EnvFingerprint:
        if not isinstance(fp, EnvFingerprint): raise TypeError(f"{name} must be an EnvFingerprint, got {type(fp).__name__}")
        return fp

    def put(
        self, namespace: str, key: str, data: bytes, *, fingerprint: EnvFingerprint,
        kind: CacheArtifactKind | str = CacheArtifactKind.CUSTOM,
        metadata: Mapping[str, Any] | None = None, overwrite: bool = True,
    ) -> CacheEntry:
        """Atomically store payload bytes and SHA-256 sidecar manifest under exclusive flock."""
        self._require_open()
        ns, clean_key = _validate_segment(namespace, "namespace"), _validate_segment(key, "key")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
        raw = bytes(data)
        self._validate_fp(fingerprint, "fingerprint")
        artifact_kind = CacheArtifactKind(kind)
        if metadata is None: meta_copy: dict[str, Any] = {}
        elif isinstance(metadata, Mapping): meta_copy = json.loads(json.dumps(dict(metadata), allow_nan=False))
        else: raise TypeError(f"metadata must be a mapping or None, got {type(metadata).__name__}")

        ns_dir, payload_path, manifest_path, lock_path = self._entry_paths(ns, clean_key)
        entry = CacheEntry(
            namespace=ns, key=clean_key, kind=artifact_kind, fingerprint_digest=fingerprint.digest(),
            fingerprint=fingerprint, payload_path=payload_path, size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(), created_at=time.time(), metadata=meta_copy,
        )
        manifest_bytes = json.dumps(entry.to_manifest_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

        with _flock_file(lock_path, exclusive=True):
            self._require_open()
            if not overwrite and (manifest_path.exists() or payload_path.exists()):
                raise CompileCacheError(f"cache entry already exists for {ns}/{clean_key} with overwrite=False")
            tmp_payload: Path | None = None
            tmp_manifest: Path | None = None
            try:
                tmp_payload = _stage_temp_bytes(ns_dir, f".{payload_path.stem}.", raw, fsync=self._fsync)
                tmp_manifest = _stage_temp_bytes(ns_dir, f".{manifest_path.stem}.", manifest_bytes, fsync=self._fsync)
                os.replace(tmp_payload, payload_path); tmp_payload = None
                os.replace(tmp_manifest, manifest_path); tmp_manifest = None
            finally:
                for leftover in (tmp_payload, tmp_manifest):
                    if leftover is not None:
                        with contextlib.suppress(OSError): leftover.unlink()
        self._inc("puts")
        return replace(entry, metadata=dict(entry.metadata))

    def _read_and_verify_locked(
        self, ns: str, clean_key: str, payload_path: Path, manifest_path: Path, *, verify_checksum: bool,
    ) -> tuple[CacheEntry, bytes] | None:
        has_m, has_p = manifest_path.exists(), payload_path.exists()
        if not has_m and not has_p: return None
        if not has_m or not has_p: raise CacheCorruptError(f"incomplete cache entry for {ns}/{clean_key}")
        entry = _parse_manifest(manifest_path, payload_path, ns, clean_key)
        try: payload = payload_path.read_bytes()
        except OSError as exc: raise CacheCorruptError(f"failed reading payload {payload_path}: {exc}") from exc
        if len(payload) != entry.size_bytes:
            raise CacheCorruptError(f"payload size mismatch for {ns}/{clean_key}: {len(payload)} != {entry.size_bytes}")
        if verify_checksum and hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise CacheCorruptError(f"payload sha256 mismatch for {ns}/{clean_key}")
        return entry, payload

    def get(
        self, namespace: str, key: str, *, expected_fingerprint: EnvFingerprint,
        verify_checksum: bool = True, strict: bool = True,
    ) -> tuple[CacheEntry, bytes] | None:
        """Read and verify a cached artifact under shared flock, failing closed on drift or corruption."""
        self._require_open()
        ns, clean_key = _validate_segment(namespace, "namespace"), _validate_segment(key, "key")
        self._validate_fp(expected_fingerprint, "expected_fingerprint")
        ns_dir, payload_path, manifest_path, lock_path = self._entry_paths(ns, clean_key)
        if not ns_dir.is_dir() or (not manifest_path.exists() and not payload_path.exists()):
            self._inc("misses"); return None

        corrupt_exc: CacheCorruptError | None = None
        result: tuple[CacheEntry, bytes] | None = None
        with _flock_file(lock_path, exclusive=False):
            self._require_open()
            try: result = self._read_and_verify_locked(ns, clean_key, payload_path, manifest_path, verify_checksum=verify_checksum)
            except CacheCorruptError as exc: corrupt_exc = exc

        if corrupt_exc is not None:
            self._inc("corrupt_errors")
            if strict: raise corrupt_exc
            with _flock_file(lock_path, exclusive=True):
                self._require_open()
                try: result = self._read_and_verify_locked(ns, clean_key, payload_path, manifest_path, verify_checksum=verify_checksum)
                except CacheCorruptError:
                    with contextlib.suppress(OSError):
                        if _unlink_pair(manifest_path, payload_path): self._inc("evictions")
                    self._inc("misses"); return None

        if result is None:
            self._inc("misses"); return None
        entry, payload = result
        if entry.fingerprint_digest != expected_fingerprint.digest():
            self._inc("fingerprint_mismatches")
            if strict: raise FingerprintMismatchError(f"fingerprint mismatch for {ns}/{clean_key}")
            self._inc("misses"); return None
        self._inc("hits")
        return entry, payload

    def invalidate(self, namespace: str, key: str | None = None) -> int:
        """Remove a specific key or all entries in `namespace` safely under exclusive locks."""
        self._require_open()
        ns = _validate_segment(namespace, "namespace")
        clean_key = _validate_segment(key, "key") if key is not None else None
        ns_dir = self._cache_dir / _slug_hash(ns)
        if not ns_dir.is_dir(): return 0
        if clean_key is not None:
            _, payload_path, manifest_path, lock_path = self._entry_paths(ns, clean_key)
            if not manifest_path.exists() and not payload_path.exists(): return 0
            with _flock_file(lock_path, exclusive=True):
                self._require_open()
                removed = 1 if _unlink_pair(manifest_path, payload_path) else 0
            self._inc("invalidations", removed); return removed

        removed_count = 0
        with _flock_file(ns_dir / "_namespace.lock", exclusive=True):
            self._require_open()
            for stem in sorted({p.stem for p in ns_dir.iterdir() if p.suffix in (".json", ".bin")}):
                with _flock_file(ns_dir / f"{stem}.lock", exclusive=True):
                    self._require_open()
                    if _unlink_pair(ns_dir / f"{stem}.json", ns_dir / f"{stem}.bin"): removed_count += 1
        self._inc("invalidations", removed_count)
        return removed_count

    def list_entries(self, namespace: str | None = None) -> tuple[CacheEntry, ...]:
        """Return valid cached `CacheEntry` descriptors, optionally filtered by `namespace`."""
        self._require_open()
        ns_dirs = [self._cache_dir / _slug_hash(_validate_segment(namespace, "namespace"))] if namespace is not None else [p for p in sorted(self._cache_dir.iterdir()) if p.is_dir()]
        entries: list[CacheEntry] = []
        for ns_dir in ns_dirs:
            if not ns_dir.is_dir(): continue
            for manifest_path in sorted(ns_dir.glob("*.json")):
                stem = manifest_path.stem
                payload_path = ns_dir / f"{stem}.bin"
                with _flock_file(ns_dir / f"{stem}.lock", exclusive=False):
                    self._require_open()
                    if manifest_path.exists() and payload_path.exists():
                        with contextlib.suppress(CacheCorruptError):
                            entries.append(_parse_manifest(manifest_path, payload_path, namespace, None))
        entries.sort(key=lambda e: (e.namespace, e.key))
        return tuple(entries)

    def stats(self) -> dict[str, Any]:
        """Return in-process hit/miss counters and live on-disk entry/byte counts."""
        with self._lock: closed, counters = self._closed, dict(self._stats)
        entry_count, total_bytes = 0, 0
        if self._cache_dir.is_dir():
            for manifest_path in self._cache_dir.glob("*/*.json"):
                payload_path = manifest_path.with_suffix(".bin")
                if payload_path.is_file():
                    entry_count += 1
                    with contextlib.suppress(OSError): total_bytes += payload_path.stat().st_size
        return {"cache_dir": str(self._cache_dir), "closed": closed, "entries": entry_count, "total_bytes": total_bytes, **counters}

    def teardown(self) -> dict[str, Any]:
        """Mark the service closed idempotently without deleting shared cache artifacts."""
        with self._lock: already, self._closed = self._closed, True
        return {**self.stats(), "already_closed": already}

    def close(self) -> None:
        """Close the service idempotently."""
        self.teardown()
