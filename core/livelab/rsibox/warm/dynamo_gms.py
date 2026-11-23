"""Optional NVIDIA Dynamo GPU Memory Service (GMS) `WeightStore` backend (stdlib only).

Integrates `ai-dynamo/dynamo`'s standalone `gpu_memory_service` V0 (CUDA VMM out-of-process
weight ownership and cross-process RO sharing) into RSIBox without hard-binding RSIBox to
Dynamo, PyTorch, or Kubernetes.

Socket layout mirrors `gpu_memory_service/common/utils.py`:
- Directory: `GMS_SOCKET_DIR` (defaults to `/tmp` when available to stay under `AF_UNIX` limits).
- Per-GPU sockets: `gms_{gpu_uuid}_{tag}.sock` for `tag in ("weights", "kv_cache")`.

Key guarantees:
- Pure Python stdlib at import time; never imports `torch`, `msgspec`, or `gpu_memory_service`.
- Enforces `AF_UNIX` `MAX_SOCK_PATH` on both sample and resolved UUID paths before starting.
- Probes GMS sockets with the lock-free MsgPack `GetRuntimeStateRequest` frame so real GMS
  servers reply cleanly with zero `Handshake recv error` logs and expose `committed` state
  (failing fast when `read_only=True` against an uncommitted `EMPTY` GMS layout).
- Strips `CUDA_VISIBLE_DEVICES` when spawning per-GPU V0 GMS servers by physical NVML index,
  spawns servers only for GPUs with missing sockets, reaps owned `Popen` children cleanly
  without zombie stalls, and fails closed when bare-metal `CUDA_VISIBLE_DEVICES` remaps
  logical ordinals away from NVML physical indices (which GMS V0's `get_socket_path` ignores).
- Preserves RSIBox's staleness stamp guard (failing closed when `expected_stamp` is supplied
  and the stamp file is missing, malformed, or drifted).
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Sequence

try:
    from .. import harness, ipc, proc
except ImportError:  # pragma: no cover - fallback if imported outside package root
    from rsibox import harness, ipc, proc  # type: ignore[no-redef]

from .types import LaunchDelta, StoreProbe, stamp_diff
from .weight_store import register_weight_store

GMS_TAGS: tuple[str, ...] = ("weights", "kv_cache")
GMS_VLLM_LOAD_FORMAT = "gms"
GMS_VLLM_WORKER_CLS = "gpu_memory_service.integrations.vllm.worker.GMSWorker"
GMS_SGLANG_LOAD_FORMAT = "gms"
DEFAULT_RO_CONNECT_TIMEOUT_MS = 10_000
_SAMPLE_UUID = "GPU-00000000-0000-0000-0000-000000000000"

# Exact length-prefixed MsgPack wire frame for `GetRuntimeStateRequest()`
# (`gpu_memory_service.common.protocol.wire._frame_message(GetRuntimeStateRequest())`).
# Handled pre-lock by `GMSRPCServer._do_handshake()` with zero server error logs.
_GET_RUNTIME_STATE_FRAME = b"\x00\x00\x00 \x81\xa4type\xb9get_runtime_state_request"


def default_socket_dir() -> str:
    """Default GMS socket directory (prefers `/tmp` to stay within `AF_UNIX` path caps)."""
    env_dir = os.environ.get("GMS_SOCKET_DIR")
    if env_dir:
        return env_dir
    return "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()


def socket_name(gpu_uuid: str, tag: str = "weights") -> str:
    """Canonical GMS Unix socket basename for `(gpu_uuid, tag)`."""
    return f"gms_{gpu_uuid}_{tag}.sock"


def long_socket_path_finding(
    socket_dir: str,
    tags: Sequence[str] = GMS_TAGS,
    uuids: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Return a `gms_socket_path` finding if any GMS socket path exceeds `AF_UNIX` limits."""
    base = os.path.abspath(socket_dir)
    check_uuids = tuple(uuids) if uuids else (_SAMPLE_UUID,)
    for uuid in check_uuids:
        for tag in tags:
            cand = os.path.join(base, socket_name(uuid, tag))
            err = ipc.sock_path_error(cand)
            if err:
                return harness.plain_finding(
                    "gms_socket_path",
                    err,
                    f"a GMS socket dir short enough that `{socket_name('<uuid>', tag)}` fits {ipc.MAX_SOCK_PATH} bytes",
                    why="GMS server binds per-GPU Unix sockets named by GPU UUID and tag; paths above AF_UNIX limits fail at bind time",
                    actions=["use a short --gms-socket-dir such as /tmp/rsibox-gms"],
                )
    return None


def _query_nvml_gpus() -> list[tuple[int, str]] | None:
    """Query `(physical_nvml_index, uuid)` for every GPU visible to `nvidia-smi`."""
    cmd = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=15, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    pairs: list[tuple[int, str]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1]:
            pairs.append((int(parts[0]), parts[1]))
    return pairs or None


def resolve_target_gpus(
    gpus: Sequence[str] = (),
    *,
    visible_devices: str | None = None,
    tp: int | None = None,
    nvml_query: Callable[[], list[tuple[int, str]] | None] = _query_nvml_gpus,
) -> list[tuple[int, str]] | None:
    """Resolve target GPUs into `(physical_nvml_index, gpu_uuid)` pairs.

    Correctly distinguishes an explicit empty `CUDA_VISIBLE_DEVICES=""` (0 visible GPUs -> None)
    from unset `CUDA_VISIBLE_DEVICES` (all host GPUs). Supports both integer indices and
    `GPU-...` / `MIG-...` UUIDs.
    """
    explicit_items: list[str] | None = None
    if gpus:
        explicit_items = [str(g).strip() for g in gpus if str(g).strip()]
        if not explicit_items:
            return None
    else:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
        if raw is not None:
            if not raw.strip():
                return None  # CUDA_VISIBLE_DEVICES="" means zero visible GPUs
            explicit_items = [x.strip() for x in raw.split(",") if x.strip()]

    nvml = nvml_query()
    if explicit_items is None:
        if not nvml:
            return None
        resolved = list(nvml)
    else:
        by_idx = {str(idx): (idx, uuid) for idx, uuid in (nvml or ())}
        by_uuid = {uuid: (idx, uuid) for idx, uuid in (nvml or ())}
        resolved = []
        for pos, item in enumerate(explicit_items):
            if item in by_idx:
                resolved.append(by_idx[item])
            elif item in by_uuid:
                resolved.append(by_uuid[item])
            elif item.startswith(("GPU-", "MIG-")) and nvml is None:
                # Allow UUID-only probing in CPU tests without nvidia-smi.
                resolved.append((pos, item))
            else:
                return None

    if tp is not None:
        if tp <= 0 or len(resolved) < tp:
            return None
        resolved = resolved[:tp]
    return resolved or None


def resolve_gpu_uuids(
    gpus: Sequence[str] = (),
    *,
    visible_devices: str | None = None,
    tp: int | None = None,
) -> list[str] | None:
    """Resolve NVML GPU UUIDs (`GPU-...`) for target GPUs (or None if unavailable)."""
    targets = resolve_target_gpus(gpus, visible_devices=visible_devices, tp=tp)
    return [uuid for _, uuid in targets] if targets else None


def _visible_devices_remap_finding(
    targets: Sequence[tuple[int, str]],
    visible_devices: str | None,
) -> dict[str, Any] | None:
    """Detect bare-metal `CUDA_VISIBLE_DEVICES` ordinal != physical NVML index under GMS V0.

    In upstream `gpu_memory_service` V0 (`common/utils.py:get_socket_path`), workers pass their
    logical CUDA device ordinal (`0..tp-1`) to `pynvml.nvmlDeviceGetHandleByIndex(device)`,
    which ignores `CUDA_VISIBLE_DEVICES` on bare-metal hosts. If `CUDA_VISIBLE_DEVICES=4,5`,
    logical device `0` dials physical GPU 0's socket instead of GPU 4's.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if not raw or not raw.strip():
        return None
    mismatches = [
        f"logical cuda:{logical} -> physical GPU {phys} ({uuid})"
        for logical, (phys, uuid) in enumerate(targets)
        if phys != logical
    ]
    if not mismatches:
        return None
    return harness.plain_finding(
        "gms_v0_visible_devices_mismatch",
        f"bare-metal CUDA_VISIBLE_DEVICES={raw!r} remaps {', '.join(mismatches)}",
        "container-isolated GPU indices starting at 0, or physical GPUs 0..tp-1 under GMS V0",
        why="upstream gpu_memory_service V0 get_socket_path(local_rank) queries pynvml by "
            "logical ordinal (ignoring CUDA_VISIBLE_DEVICES on bare-metal), so workers would dial "
            "physical GPU 0..tp-1 sockets instead of the visible GPUs",
        actions=[
            "run inside a container with `--gpus device=...` (so NVML renumbers visible GPUs to 0..tp-1) and pass allow_remapped_visible_devices=True",
            "or use physical GPUs starting at 0 with GMS V0",
            "or switch --weights-backend to ipc_cache/none on bare-metal non-zero GPUs",
        ],
        risk="workers silently attach to the wrong physical GPU's GMS socket or hang",
    )


# -- Pure-stdlib MsgPack mini-decoder for `GetRuntimeStateResponse` ----------------------


def _slice_exact(buf: bytes, pos: int, n: int) -> tuple[bytes, int]:
    end = pos + n
    if n < 0 or end > len(buf):
        raise ValueError("truncated msgpack payload")
    return buf[pos:end], end


def _decode_msgpack(buf: bytes, pos: int = 0) -> tuple[Any, int]:
    """Minimal pure-stdlib MsgPack decoder sufficient for GMS runtime-state maps."""
    if pos >= len(buf):
        raise ValueError("truncated msgpack payload")
    b = buf[pos]
    pos += 1
    if b <= 0x7F:
        return b, pos
    if b >= 0xE0:
        return b - 256, pos
    if 0xA0 <= b <= 0xBF:
        raw, pos = _slice_exact(buf, pos, b & 0x1F)
        return raw.decode("utf-8", errors="replace"), pos
    if 0x80 <= b <= 0x8F:
        return _decode_msgpack_map(buf, pos, b & 0x0F)
    if 0x90 <= b <= 0x9F:
        return _decode_msgpack_array(buf, pos, b & 0x0F)
    if b == 0xC0:
        return None, pos
    if b == 0xC2:
        return False, pos
    if b == 0xC3:
        return True, pos
    if b in (0xC4, 0xC5, 0xC6):
        fmt = {0xC4: "!B", 0xC5: "!H", 0xC6: "!I"}[b]
        hdr, pos = _slice_exact(buf, pos, struct.calcsize(fmt))
        (n,) = struct.unpack(fmt, hdr)
        return _slice_exact(buf, pos, n)
    if b in (0xC7, 0xC8, 0xC9):
        fmt = {0xC7: "!B", 0xC8: "!H", 0xC9: "!I"}[b]
        hdr, pos = _slice_exact(buf, pos, struct.calcsize(fmt))
        (n,) = struct.unpack(fmt, hdr)
        ext_type_b, pos = _slice_exact(buf, pos, 1)
        data, pos = _slice_exact(buf, pos, n)
        return (struct.unpack("!b", ext_type_b)[0], data), pos
    if b == 0xCA:
        raw, pos = _slice_exact(buf, pos, 4)
        return struct.unpack("!f", raw)[0], pos
    if b == 0xCB:
        raw, pos = _slice_exact(buf, pos, 8)
        return struct.unpack("!d", raw)[0], pos
    if b in (0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD3):
        fmt = {
            0xCC: "!B", 0xCD: "!H", 0xCE: "!I", 0xCF: "!Q",
            0xD0: "!b", 0xD1: "!h", 0xD2: "!i", 0xD3: "!q",
        }[b]
        raw, pos = _slice_exact(buf, pos, struct.calcsize(fmt))
        return struct.unpack(fmt, raw)[0], pos
    if b in (0xD4, 0xD5, 0xD6, 0xD7, 0xD8):
        n = {0xD4: 1, 0xD5: 2, 0xD6: 4, 0xD7: 8, 0xD8: 16}[b]
        ext_type_b, pos = _slice_exact(buf, pos, 1)
        data, pos = _slice_exact(buf, pos, n)
        return (struct.unpack("!b", ext_type_b)[0], data), pos
    if b in (0xD9, 0xDA, 0xDB):
        fmt = {0xD9: "!B", 0xDA: "!H", 0xDB: "!I"}[b]
        hdr, pos = _slice_exact(buf, pos, struct.calcsize(fmt))
        (n,) = struct.unpack(fmt, hdr)
        raw, pos = _slice_exact(buf, pos, n)
        return raw.decode("utf-8", errors="replace"), pos
    if b in (0xDC, 0xDD):
        fmt = "!H" if b == 0xDC else "!I"
        hdr, pos = _slice_exact(buf, pos, struct.calcsize(fmt))
        (n,) = struct.unpack(fmt, hdr)
        return _decode_msgpack_array(buf, pos, n)
    if b in (0xDE, 0xDF):
        fmt = "!H" if b == 0xDE else "!I"
        hdr, pos = _slice_exact(buf, pos, struct.calcsize(fmt))
        (n,) = struct.unpack(fmt, hdr)
        return _decode_msgpack_map(buf, pos, n)
    raise ValueError(f"unsupported msgpack byte 0x{b:02x}")


def _decode_msgpack_array(buf: bytes, pos: int, count: int) -> tuple[list[Any], int]:
    out = []
    for _ in range(count):
        val, pos = _decode_msgpack(buf, pos)
        out.append(val)
    return out, pos


def _decode_msgpack_map(buf: bytes, pos: int, count: int) -> tuple[dict[Any, Any], int]:
    out: dict[Any, Any] = {}
    for _ in range(count):
        k, pos = _decode_msgpack(buf, pos)
        v, pos = _decode_msgpack(buf, pos)
        out[k] = v
    return out, pos


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < n:
        part = sock.recv(n - len(chunks))
        if not part:
            return None
        chunks.extend(part)
    return bytes(chunks)


def _probe_gms_socket(path: str, timeout_s: float = 1.0) -> tuple[bool, dict[str, Any] | None]:
    """Probe a GMS Unix socket using lock-free `GetRuntimeStateRequest`.

    Returns `(True, state_dict)` against a real `GMSRPCServer` (zero server error logs),
    `(True, None)` if a minimal listener accepts and closes without speaking GMS wire protocol,
    and `(False, None)` if the socket does not exist or refuses connections.
    """
    if not os.path.exists(path):
        return False, None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            sock.connect(path)
        except OSError:
            return False, None
        try:
            sock.sendall(_GET_RUNTIME_STATE_FRAME)
            hdr = _recv_exact(sock, 4)
            if not hdr:
                return True, None
            (length,) = struct.unpack("!I", hdr)
            if length <= 0 or length > (1 << 20):
                return True, None
            payload = _recv_exact(sock, length)
            if not payload:
                return True, None
            parsed, _ = _decode_msgpack(payload, 0)
            return True, parsed if isinstance(parsed, dict) else None
        except (TimeoutError, socket.timeout):
            # Hung / SIGSTOP'd listener accepts in kernel backlog but never responds.
            return False, None
        except (OSError, ValueError, IndexError, struct.error):
            return True, None


class DynamoGmsStore:
    """`WeightStore` backed by NVIDIA Dynamo's `gpu_memory_service` (GMS V0).

    Supports:
    - `engine="vllm"`: injects `--load-format gms`, `--worker-cls ...GMSWorker`,
      optional `--model-loader-extra-config` (`gms_read_only`, `gms_ro_connect_timeout_ms`),
      and `GMS_SOCKET_DIR` / `DYN_GMS_SCRATCH_KV_ENABLED`.
    - `engine="sglang"`: injects `GMS_SOCKET_DIR` and GMS config consumed via
      `gpu_memory_service.integrations.sglang.setup_gms()` + `declare_resolution()`.
    - `engine="torch_module"` / `"fastvideo"`: injects `GMS_SOCKET_DIR` and `RSIBOX_GMS_MODE`
      for `gpu_memory_service.client.torch.module.materialize_module_from_gms`.
    """

    name = "dynamo_gms"

    def __init__(
        self,
        *,
        socket_dir: str | None = None,
        tags: Sequence[str] = GMS_TAGS,
        read_only: bool = False,
        ro_connect_timeout_ms: int | None = None,
        scratch_kv: bool = False,
        allow_remapped_visible_devices: bool = False,
        device_type: str = "cuda",
        python: str | None = None,
        extra_pythonpath: Sequence[str] = (),
        log_dir: str | None = None,
        stamp_path: str | None = None,
        uuid_resolver: Callable[..., list[str] | None] | None = None,
        target_resolver: Callable[..., list[tuple[int, str]] | None] | None = None,
    ):
        self.socket_dir = os.path.abspath(socket_dir or default_socket_dir())
        self.tags = tuple(tags) if tags else GMS_TAGS
        self.read_only = bool(read_only)
        self.ro_connect_timeout_ms = (
            ro_connect_timeout_ms
            if ro_connect_timeout_ms is not None
            else (DEFAULT_RO_CONNECT_TIMEOUT_MS if self.read_only else None)
        )
        self.scratch_kv = bool(scratch_kv)
        self.allow_remapped_visible_devices = bool(allow_remapped_visible_devices)
        self.device_type = str(device_type)
        self.python = python or sys.executable
        self.extra_pythonpath = tuple(str(p) for p in extra_pythonpath if p)
        self.log_dir = log_dir
        self.stamp_path = stamp_path or os.path.join(self.socket_dir, "rsibox_gms_stamp.json")
        self._uuid_resolver = uuid_resolver
        self._target_resolver = target_resolver
        self._owned_procs: list[subprocess.Popen] = []

    def _resolve_targets(
        self,
        *,
        gpus: Sequence[str] = (),
        visible_devices: str | None = None,
        tp: int | None = None,
    ) -> list[tuple[int, str]] | None:
        if self._target_resolver is not None:
            return self._target_resolver(gpus, visible_devices=visible_devices, tp=tp)
        if self._uuid_resolver is not None:
            uuids = self._uuid_resolver(gpus, visible_devices=visible_devices, tp=tp)
            return [(i, u) for i, u in enumerate(uuids)] if uuids else None
        # Allow unittest.mock.patch("rsibox.warm.dynamo_gms.resolve_gpu_uuids", ...)
        if resolve_gpu_uuids is not _ORIG_RESOLVE_GPU_UUIDS:
            uuids = resolve_gpu_uuids(gpus, visible_devices=visible_devices, tp=tp)
            return [(i, u) for i, u in enumerate(uuids)] if uuids else None
        return resolve_target_gpus(gpus, visible_devices=visible_devices, tp=tp)

    def socket_paths(
        self,
        *,
        gpus: Sequence[str] = (),
        visible_devices: str | None = None,
        tp: int | None = None,
    ) -> list[str] | None:
        """Expected per-(GPU, tag) GMS Unix socket paths, or None if GPUs cannot be identified."""
        targets = self._resolve_targets(gpus=gpus, visible_devices=visible_devices, tp=tp)
        if not targets:
            return None
        return [
            os.path.join(self.socket_dir, socket_name(uuid, tag))
            for _, uuid in targets
            for tag in self.tags
        ]

    def write_stamp(self, stamp: dict[str, Any]) -> None:
        """Persist the loader/checkpoint/env stamp alongside the GMS sockets."""
        os.makedirs(os.path.dirname(self.stamp_path) or ".", exist_ok=True)
        ipc.atomic_write_json(self.stamp_path, stamp)

    def read_stamp(self) -> dict[str, Any] | None:
        """Read the persisted stamp if present, else None."""
        data = ipc.read_json(self.stamp_path, default=None)
        return data if isinstance(data, dict) else None

    def probe(
        self,
        *,
        gpus: Sequence[str] = (),
        visible_devices: str | None = None,
        tp: int | None = None,
        expected_stamp: dict[str, Any] | None = None,
        connect_timeout_s: float = 1.0,
    ) -> StoreProbe:
        """Probe every target `(gpu, tag)` GMS socket, RO commit state, and optional stamp."""
        t0 = time.monotonic()
        path_finding = long_socket_path_finding(self.socket_dir, self.tags)
        if path_finding is not None:
            return StoreProbe(
                ok=False,
                backend=self.name,
                seconds=time.monotonic() - t0,
                error=path_finding["observed"],
                finding=path_finding,
            )

        targets = self._resolve_targets(gpus=gpus, visible_devices=visible_devices, tp=tp)
        if not targets:
            err = (
                f"cannot resolve GPU UUIDs for Dynamo GMS probe "
                f"(gpus={list(gpus)!r}, tp={tp!r}, visible={visible_devices!r})"
            )
            finding = harness.plain_finding(
                "dynamo_gms_daemon",
                err,
                "accessible nvidia-smi or explicit GPU UUIDs matching tp",
                why="cannot match GMS per-GPU Unix sockets without GPU UUIDs",
                actions=[
                    "set CUDA_VISIBLE_DEVICES to the target GPU(s)",
                    "or pass --weights-backend none / ipc_cache",
                ],
            )
            return StoreProbe(
                ok=False,
                backend=self.name,
                matched_gpus=False,
                seconds=time.monotonic() - t0,
                error=err,
                finding=finding,
            )

        resolved_path_finding = long_socket_path_finding(
            self.socket_dir, self.tags, [u for _, u in targets]
        )
        if resolved_path_finding is not None:
            return StoreProbe(
                ok=False,
                backend=self.name,
                seconds=time.monotonic() - t0,
                error=resolved_path_finding["observed"],
                finding=resolved_path_finding,
            )

        if not self.allow_remapped_visible_devices:
            remap_finding = _visible_devices_remap_finding(targets, visible_devices)
            if remap_finding is not None:
                return StoreProbe(
                    ok=False,
                    backend=self.name,
                    seconds=time.monotonic() - t0,
                    error=remap_finding["observed"],
                    finding=remap_finding,
                )

        tagged_paths = [
            (tag, os.path.join(self.socket_dir, socket_name(uuid, tag)))
            for _, uuid in targets
            for tag in self.tags
        ]
        paths = [path for _, path in tagged_paths]
        missing: list[str] = []
        uncommitted_weights: list[str] = []
        for tag, path in tagged_paths:
            alive, state = _probe_gms_socket(path, timeout_s=connect_timeout_s)
            if not alive:
                missing.append(path)
                continue
            if self.read_only and tag == "weights":
                if (
                    not isinstance(state, dict)
                    or state.get("type") != "get_runtime_state_response"
                    or state.get("committed") is not True
                ):
                    uncommitted_weights.append(path)

        if missing:
            finding = harness.plain_finding(
                "dynamo_gms_daemon",
                f"Dynamo GMS socket(s) not answering: {', '.join(missing)}",
                f"one answering GMS server per GPU for tags {list(self.tags)} under {self.socket_dir}",
                why="an engine launched with --load-format gms will hang or fail if the GMS server sockets are absent",
                actions=[
                    f"start GMS server: GMS_SOCKET_DIR={self.socket_dir} python -m gpu_memory_service --device <id>",
                    "or call DynamoGmsStore.ensure_ready()",
                    "or drop --weights-backend dynamo_gms",
                ],
                risk="engine hangs waiting for GMS socket",
            )
            return StoreProbe(
                ok=False,
                backend=self.name,
                sockets=tuple(paths),
                missing=tuple(missing),
                seconds=time.monotonic() - t0,
                error=finding["observed"],
                finding=finding,
            )

        if uncommitted_weights:
            finding = harness.plain_finding(
                "dynamo_gms_not_committed",
                f"GMS weights socket(s) are EMPTY / uncommitted: {', '.join(uncommitted_weights)}",
                "committed GMS weights layout before launching a read-only (`--gms-read-only`) worker",
                why="a read-only GMS client cannot load from disk and will block or time out waiting for a writer to commit",
                actions=[
                    "boot a writer without --gms-read-only first so it loads and commits weights into GMS",
                    "or drop --gms-read-only to use RW_OR_RO mode",
                ],
                risk="worker hangs or times out in GMSWorker.init_device",
            )
            return StoreProbe(
                ok=False,
                backend=self.name,
                sockets=tuple(paths),
                seconds=time.monotonic() - t0,
                error=finding["observed"],
                finding=finding,
            )

        if expected_stamp is not None:
            recorded = self.read_stamp()
            stale = stamp_diff(recorded, expected_stamp)
            if stale:
                finding = harness.plain_finding(
                    "dynamo_gms_stale",
                    f"GMS cached weights stamp mismatch ({', '.join(stale)})",
                    "GMS weights committed under the current loader/model/env stamp",
                    why="missing or drifted stamp means resident GMS weights cannot be trusted for the current loader/checkpoint/env",
                    actions=[
                        "restart GMS server or clear committed layout and reload with an RW writer",
                        "record the current stamp via DynamoGmsStore.write_stamp() after committing",
                    ],
                )
                return StoreProbe(
                    ok=False,
                    backend=self.name,
                    sockets=tuple(paths),
                    stale_keys=stale,
                    seconds=time.monotonic() - t0,
                    error=finding["observed"],
                    finding=finding,
                )

        return StoreProbe(
            ok=True,
            backend=self.name,
            sockets=tuple(paths),
            seconds=time.monotonic() - t0,
        )

    def _gms_server_env(self) -> dict[str, str]:
        """Child environment for `python -m gpu_memory_service --device <phys_idx>`.

        Strips `CUDA_VISIBLE_DEVICES` (so CUDA driver device ordinals match physical NVML
        indices) and `DYN_GMS_USE_V1` (so V0 CLI `--device-type` / `--tag` flags are accepted).
        """
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("DYN_GMS_USE_V1", None)
        env["GMS_SOCKET_DIR"] = self.socket_dir
        if self.extra_pythonpath:
            existing = env.get("PYTHONPATH", "")
            parts = [*([existing] if existing else []), *self.extra_pythonpath]
            env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def ensure_ready(
        self,
        *,
        gpus: Sequence[str] = (),
        visible_devices: str | None = None,
        tp: int | None = None,
        timeout_s: float = 30.0,
    ) -> StoreProbe:
        """Reuse already-running GMS servers, and spawn local GMS servers only for missing GPUs."""
        t0 = time.monotonic()
        first = self.probe(gpus=gpus, visible_devices=visible_devices, tp=tp)
        if first.ok or not first.matched_gpus:
            return first
        if first.finding and first.finding.get("check") != "dynamo_gms_daemon":
            return first

        targets = self._resolve_targets(gpus=gpus, visible_devices=visible_devices, tp=tp)
        if not targets:
            return first

        missing_set = set(first.missing)
        missing_tags_by_dev: dict[int, list[str]] = {}
        for phys_idx, uuid in targets:
            dev_missing = [
                tag
                for tag in self.tags
                if os.path.join(self.socket_dir, socket_name(uuid, tag)) in missing_set
            ]
            if not dev_missing:
                continue
            bucket = missing_tags_by_dev.setdefault(phys_idx, [])
            for tag in dev_missing:
                if tag not in bucket:
                    bucket.append(tag)
        if not missing_tags_by_dev:
            return first

        os.makedirs(self.socket_dir, exist_ok=True)
        log_root = self.log_dir or self.socket_dir
        os.makedirs(log_root, exist_ok=True)
        env = self._gms_server_env()

        started: list[tuple[int, subprocess.Popen]] = []
        try:
            for dev, dev_tags in missing_tags_by_dev.items():
                cmd = [
                    self.python,
                    "-m",
                    "gpu_memory_service",
                    "--device",
                    str(dev),
                    "--device-type",
                    self.device_type,
                ]
                for tag in dev_tags:
                    cmd.extend(["--tag", tag])
                log_path = os.path.join(log_root, f"gms-server-gpu{dev}.log")
                with open(log_path, "ab") as log_fh:
                    child = subprocess.Popen(
                        cmd,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                started.append((dev, child))
                self._owned_procs.append(child)

            deadline = time.monotonic() + float(timeout_s)
            while time.monotonic() < deadline:
                for dev, child in started:
                    rc = child.poll()
                    if rc is not None:
                        raise RuntimeError(
                            f"GMS server for GPU {dev} exited early with code {rc} "
                            f"(see {os.path.join(log_root, f'gms-server-gpu{dev}.log')})"
                        )
                pr = self.probe(gpus=gpus, visible_devices=visible_devices, tp=tp, connect_timeout_s=0.3)
                if pr.ok or (pr.finding and pr.finding.get("check") == "dynamo_gms_not_committed"):
                    # In read_only mode, sockets are alive even before a writer commits.
                    return StoreProbe(
                        ok=pr.ok,
                        backend=self.name,
                        sockets=pr.sockets,
                        seconds=time.monotonic() - t0,
                        error=pr.error,
                        finding=pr.finding,
                    )
                time.sleep(0.2)

            raise TimeoutError(f"GMS server sockets did not become ready within {timeout_s:.1f}s")
        except BaseException as exc:
            # Never leave half-started GMS daemons or unreaped zombies holding GPU resources.
            for _, child in started:
                self._stop_child(child, grace=3.0)
                if child in self._owned_procs:
                    self._owned_procs.remove(child)
            if not isinstance(exc, Exception):
                raise
            finding = harness.plain_finding(
                "dynamo_gms_start_failed",
                str(exc),
                "GMS server binds all target (gpu, tag) sockets",
                why="failed or timed-out GMS server cannot serve weights to workers",
                actions=[
                    "verify `gpu_memory_service` is installed in the engine python environment",
                    f"inspect logs under {log_root}",
                ],
            )
            return StoreProbe(
                ok=False,
                backend=self.name,
                seconds=time.monotonic() - t0,
                error=str(exc),
                finding=finding,
            )

    def extra_loader_config(self) -> dict[str, Any]:
        """GMS keys for `--model-loader-extra-config`."""
        cfg: dict[str, Any] = {}
        if self.read_only:
            cfg["gms_read_only"] = True
        if self.ro_connect_timeout_ms is not None:
            cfg["gms_ro_connect_timeout_ms"] = int(self.ro_connect_timeout_ms)
        return cfg

    @staticmethod
    def _parse_extra_config(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
        """Parse an explicit `model_loader_extra_config` (None, dict, or JSON object str)."""
        if raw is None:
            return {}, None
        if isinstance(raw, dict):
            return dict(raw), None
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return {}, None
            try:
                parsed = json.loads(s)
            except ValueError as exc:
                return None, f"invalid JSON in model_loader_extra_config: {exc}"
            if not isinstance(parsed, dict):
                return None, f"model_loader_extra_config must be a JSON object, got {type(parsed).__name__}"
            return parsed, None
        return None, f"model_loader_extra_config must be a JSON object or dict, got {type(raw).__name__}"

    def merged_loader_config(self, raw_explicit: Any = None) -> dict[str, Any]:
        """Merge non-conflicting user `model_loader_extra_config` keys with required GMS keys."""
        parsed, err = self._parse_extra_config(raw_explicit)
        if err is not None or parsed is None:
            return self.extra_loader_config()
        return {**parsed, **self.extra_loader_config()}

    def _check_explicit_extra_config(
        self,
        explicit_flags: set[str] | frozenset[str],
        vals: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fail closed when explicit `model_loader_extra_config` is malformed or contradicts GMS keys."""
        if "model_loader_extra_config" not in explicit_flags:
            return None
        cfg, err = self._parse_extra_config(vals.get("model_loader_extra_config"))
        if err is not None or cfg is None:
            return harness.plain_finding(
                "gms_loader_extra_config",
                err or "invalid --model-loader-extra-config",
                "a JSON object for --model-loader-extra-config",
                why="Dynamo GMS requires a valid JSON object to merge gms_read_only / timeout keys",
                actions=["pass a valid JSON object to --model-loader-extra-config", "or omit it"],
            )
        req = self.extra_loader_config()
        if not req:
            return None
        mismatches = [
            f"{k}={cfg[k]!r} (expected {v!r})"
            for k, v in req.items()
            if k in cfg and cfg[k] != v
        ]
        if not mismatches:
            return None
        return harness.plain_finding(
            "gms_loader_extra_config",
            f"explicit --model-loader-extra-config conflicts with GMS settings: {', '.join(mismatches)}",
            f"compatible {json.dumps(req)} values in --model-loader-extra-config, or omit conflicting gms_* keys",
            why="contradicting --gms-read-only or timeout settings would silently launch a writer or hang indefinitely",
            actions=[
                "update or drop the conflicting gms_* keys in --model-loader-extra-config",
                "or omit --gms-read-only",
            ],
        )

    def launch_delta(
        self,
        *,
        engine: str,
        explicit_flags: set[str] | frozenset[str] = frozenset(),
        explicit_values: dict[str, Any] | None = None,
        gpus: Sequence[str] = (),
    ) -> LaunchDelta:
        """Build engine-specific CLI args, env vars, and conflict findings for Dynamo GMS."""
        vals = dict(explicit_values or {})
        findings: list[dict[str, Any]] = []
        path_finding = long_socket_path_finding(self.socket_dir, self.tags)
        if path_finding is not None:
            findings.append(path_finding)

        env: dict[str, str] = {"GMS_SOCKET_DIR": self.socket_dir}
        if self.scratch_kv:
            env["DYN_GMS_SCRATCH_KV_ENABLED"] = "1"
        if self.extra_pythonpath:
            env["RSIBOX_GMS_PYTHONPATH"] = os.pathsep.join(self.extra_pythonpath)

        args: list[str] = []
        requires_worker_cls: str | None = None

        if engine == "vllm":
            requires_worker_cls = GMS_VLLM_WORKER_CLS
            if "load_format" in explicit_flags and vals.get("load_format") != GMS_VLLM_LOAD_FORMAT:
                findings.append(
                    harness.plain_finding(
                        "gms_load_format",
                        f"--weights-backend dynamo_gms and explicit --load-format {vals.get('load_format')!r}",
                        f"--load-format {GMS_VLLM_LOAD_FORMAT}, or omit --load-format",
                        why="Dynamo GMS requires vLLM's GMSModelLoader (`--load-format gms`)",
                        actions=["drop --load-format", "or switch --weights-backend to none/ipc_cache"],
                    )
                )
            elif "load_format" not in explicit_flags:
                args.extend(["--load-format", GMS_VLLM_LOAD_FORMAT])

            if "worker_cls" in explicit_flags and vals.get("worker_cls") != GMS_VLLM_WORKER_CLS:
                findings.append(
                    harness.plain_finding(
                        "gms_worker_cls",
                        f"--weights-backend dynamo_gms and explicit --worker-cls {vals.get('worker_cls')!r}",
                        f"--worker-cls {GMS_VLLM_WORKER_CLS}, or omit --worker-cls",
                        why="Dynamo GMS registers its CUDA VMM allocator and loader via GMSWorker; "
                            "note that RSIBox's live-edit KernelWorkerExtension uses --worker-extension-cls and composes cleanly",
                        actions=["drop --worker-cls (and --kvplan if set)", "or switch --weights-backend"],
                    )
                )
            elif "worker_cls" not in explicit_flags:
                args.extend(["--worker-cls", GMS_VLLM_WORKER_CLS])

            extra_conflict = self._check_explicit_extra_config(explicit_flags, vals)
            if extra_conflict is not None:
                findings.append(extra_conflict)
            else:
                extra_cfg = self.extra_loader_config()
                if extra_cfg:
                    raw_explicit = (
                        vals.get("model_loader_extra_config")
                        if "model_loader_extra_config" in explicit_flags
                        else None
                    )
                    parsed_explicit, _ = self._parse_extra_config(raw_explicit)
                    merged = {**(parsed_explicit or {}), **extra_cfg}
                    if "model_loader_extra_config" not in explicit_flags or merged != (parsed_explicit or {}):
                        args.extend(["--model-loader-extra-config", json.dumps(merged)])

        elif engine == "sglang":
            if "load_format" in explicit_flags and vals.get("load_format") not in (GMS_SGLANG_LOAD_FORMAT, None):
                findings.append(
                    harness.plain_finding(
                        "gms_load_format",
                        f"--weights-backend dynamo_gms and explicit load_format={vals.get('load_format')!r}",
                        "omit load_format (SglangLab wires GMSModelLoader via setup_gms + declare_resolution)",
                        why="stock SGLang rejects string load_format='gms' unless bound through gpu_memory_service.integrations.sglang.setup_gms",
                        actions=["omit load_format from SglangLab config", "or use weight_store=None"],
                    )
                )
            extra_conflict = self._check_explicit_extra_config(explicit_flags, vals)
            if extra_conflict is not None:
                findings.append(extra_conflict)
            else:
                extra_cfg = self.extra_loader_config()
                if extra_cfg:
                    raw_explicit = (
                        vals.get("model_loader_extra_config")
                        if "model_loader_extra_config" in explicit_flags
                        else None
                    )
                    parsed_explicit, _ = self._parse_extra_config(raw_explicit)
                    merged = {**(parsed_explicit or {}), **extra_cfg}
                    if "model_loader_extra_config" not in explicit_flags or merged != (parsed_explicit or {}):
                        args.extend(["--model-loader-extra-config", json.dumps(merged)])

        elif engine in ("torch_module", "fastvideo"):
            env["RSIBOX_GMS_MODE"] = "ro" if self.read_only else "rw_or_ro"

        else:
            findings.append(
                harness.plain_finding(
                    "gms_unsupported_engine",
                    f"Dynamo GMS launch_delta requested for unsupported engine {engine!r}",
                    "one of: vllm, sglang, torch_module, fastvideo",
                    why="mutable training engines (e.g. Megatron) require step-snapshot or writable lease semantics rather than read-only GMS weights",
                    actions=["use RSIBox's default warm backend for this engine"],
                )
            )

        return LaunchDelta(
            args=tuple(args),
            env=env,
            requires_worker_cls=requires_worker_cls,
            findings=tuple(findings),
        )

    @staticmethod
    def _stop_child(child: subprocess.Popen, grace: float = 5.0) -> None:
        proc.stop_session(child.pid, grace=grace, reap=child.poll)
        try:
            child.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2.0)

    def stop(self) -> None:
        """Stop and reap every GMS server process spawned by this instance (leaves external sidecars alone)."""
        procs = list(self._owned_procs)
        self._owned_procs.clear()
        for child in procs:
            self._stop_child(child, grace=5.0)


_ORIG_RESOLVE_GPU_UUIDS = resolve_gpu_uuids
register_weight_store("dynamo_gms", lambda **kw: DynamoGmsStore(**kw))
