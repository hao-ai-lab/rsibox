"""Optional NVIDIA Dynamo GPU Memory Service (GMS) `WeightStore` backend (stdlib only).

Integrates `ai-dynamo/dynamo`'s standalone `gpu_memory_service` (CUDA VMM out-of-process
weight ownership and cross-process RO sharing) into RSIBox without hard-binding RSIBox to
Dynamo, PyTorch, or Kubernetes.

Socket layout mirrors `gpu_memory_service/common/utils.py`:
- Directory: `GMS_SOCKET_DIR` (or `tempfile.gettempdir()` when unset).
- Per-GPU sockets: `gms_{gpu_uuid}_{tag}.sock` for `tag in ("weights", "kv_cache")`.

Key guarantees:
- Pure Python stdlib at import time; never imports `torch` or `gpu_memory_service`.
- Enforces `AF_UNIX` `MAX_SOCK_PATH` before spawning any GMS server.
- Supports both externally managed GMS servers (K8s sidecar / host daemon) and
  RSIBox-managed local GMS servers (`ensure_ready()` / `stop()`), cleaning up only
  daemons started by this store instance.
- Preserves RSIBox's staleness stamp guard (optional JSON stamp file) so edits to a
  model loader or quantization path refuse stale GMS weights rather than silently reading them.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from rsibox import harness, ipc, proc
from .types import LaunchDelta, StoreProbe, stamp_diff
from .weight_store import register_weight_store

GMS_TAGS: tuple[str, ...] = ("weights", "kv_cache")
GMS_VLLM_LOAD_FORMAT = "gms"
GMS_VLLM_WORKER_CLS = "gpu_memory_service.integrations.vllm.worker.GMSWorker"
GMS_SGLANG_LOAD_FORMAT = "gms"
_SAMPLE_UUID = "GPU-00000000-0000-0000-0000-000000000000"


def default_socket_dir() -> str:
    """Default GMS socket directory matching `gpu_memory_service.common.utils`."""
    return os.environ.get("GMS_SOCKET_DIR") or tempfile.gettempdir()


def socket_name(gpu_uuid: str, tag: str = "weights") -> str:
    """Canonical GMS Unix socket basename for `(gpu_uuid, tag)`."""
    return f"gms_{gpu_uuid}_{tag}.sock"


def long_socket_path_finding(
    socket_dir: str,
    tags: Sequence[str] = GMS_TAGS,
) -> dict[str, Any] | None:
    """Return a `gms_socket_path` finding if any GMS socket path exceeds `AF_UNIX` limits."""
    base = os.path.abspath(socket_dir)
    for tag in tags:
        cand = os.path.join(base, socket_name(_SAMPLE_UUID, tag))
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


def resolve_gpu_uuids(
    gpus: Sequence[str] = (),
    *,
    visible_devices: str | None = None,
    tp: int | None = None,
) -> list[str] | None:
    """Resolve NVML GPU UUIDs (`GPU-...`) for target GPUs.

    Accepts explicit `GPU-...` / `MIG-...` UUIDs directly, or queries `nvidia-smi` for
    device indices from `gpus` or `visible_devices` (`CUDA_VISIBLE_DEVICES`).
    Returns `None` if `nvidia-smi` fails or fewer than `tp` GPUs are visible.
    """
    items = [str(g).strip() for g in gpus if str(g).strip()]
    if not items:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
        if raw is not None and raw.strip():
            items = [x.strip() for x in raw.split(",") if x.strip()]
        elif tp is not None and tp > 0:
            items = [str(i) for i in range(tp)]
    if tp is not None and items and len(items) < tp:
        return None
    if tp is not None and items:
        items = items[:tp]

    if items and all(item.startswith(("GPU-", "MIG-")) for item in items):
        return items

    cmd = ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    if items:
        cmd.extend(["-i", ",".join(items)])
    try:
        out = subprocess.check_output(cmd, text=True, timeout=15, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    uuids = [line.strip() for line in out.splitlines() if line.strip()]
    if not uuids:
        return None
    if tp is not None and len(uuids) < tp:
        return None
    return uuids[:tp] if tp is not None else uuids


def _resolve_device_indices(gpus: Sequence[str] = (), visible_devices: str | None = None) -> list[int]:
    items = [str(g).strip() for g in gpus if str(g).strip()]
    if not items:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
        if raw:
            items = [x.strip() for x in raw.split(",") if x.strip()]
    indices: list[int] = []
    for item in items:
        if item.isdigit():
            indices.append(int(item))
    return indices or [0]


def _socket_answers(path: str, timeout_s: float = 1.0) -> bool:
    """True when an `AF_UNIX` socket at `path` accepts a connection (non-locking pre-handshake)."""
    if not os.path.exists(path):
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            sock.connect(path)
            return True
        except OSError:
            return False


class DynamoGmsStore:
    """`WeightStore` backed by NVIDIA Dynamo's `gpu_memory_service` (GMS).

    Supports:
    - `engine="vllm"`: injects `--load-format gms`, `--worker-cls ...:GMSWorker`,
      optional `--model-loader-extra-config` (`gms_read_only`, `gms_ro_connect_timeout_ms`),
      and `GMS_SOCKET_DIR` / `DYN_GMS_SCRATCH_KV_ENABLED`.
    - `engine="sglang"`: injects `--load-format gms`, `GMS_SOCKET_DIR`, and RO config.
    - `engine="torch_module"` (e.g. FastVideo / generic PyTorch): injects `GMS_SOCKET_DIR`
      and `RSIBOX_GMS_MODE` so modules can bind via `gpu_memory_service.client.torch.module`.
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
        use_v1: bool = False,
        device_type: str = "cuda",
        python: str | None = None,
        extra_pythonpath: Sequence[str] = (),
        log_dir: str | None = None,
        stamp_path: str | None = None,
        uuid_resolver: Callable[..., list[str] | None] | None = None,
    ):
        self.socket_dir = os.path.abspath(socket_dir or default_socket_dir())
        self.tags = tuple(tags) if tags else GMS_TAGS
        self.read_only = bool(read_only)
        self.ro_connect_timeout_ms = ro_connect_timeout_ms
        self.scratch_kv = bool(scratch_kv)
        self.use_v1 = bool(use_v1)
        self.device_type = str(device_type)
        self.python = python or sys.executable
        self.extra_pythonpath = tuple(str(p) for p in extra_pythonpath if p)
        self.log_dir = log_dir
        self.stamp_path = stamp_path or os.path.join(self.socket_dir, "rsibox_gms_stamp.json")
        self._uuid_resolver = uuid_resolver
        self._owned_pids: list[int] = []

    def socket_paths(
        self,
        *,
        gpus: Sequence[str] = (),
        visible_devices: str | None = None,
        tp: int | None = None,
    ) -> list[str] | None:
        """Expected per-(GPU, tag) GMS Unix socket paths, or None if GPUs cannot be identified."""
        resolver = self._uuid_resolver or resolve_gpu_uuids
        uuids = resolver(gpus, visible_devices=visible_devices, tp=tp)
        if not uuids:
            return None
        return [
            os.path.join(self.socket_dir, socket_name(uuid, tag))
            for uuid in uuids
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
        """Probe whether every target `(gpu, tag)` GMS socket answers and matches any stamp."""
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

        paths = self.socket_paths(gpus=gpus, visible_devices=visible_devices, tp=tp)
        if paths is None:
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

        missing = tuple(p for p in paths if not _socket_answers(p, timeout_s=connect_timeout_s))
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
                missing=missing,
                seconds=time.monotonic() - t0,
                error=finding["observed"],
                finding=finding,
            )

        stale: tuple[str, ...] = ()
        if expected_stamp is not None:
            recorded = self.read_stamp()
            if recorded is not None:
                stale = stamp_diff(recorded, expected_stamp)
                if stale:
                    finding = harness.plain_finding(
                        "dynamo_gms_stale",
                        f"GMS cached weights stamp differs on: {', '.join(stale)}",
                        "GMS weights committed under the current loader/model/env stamp",
                        why="edits to model loader, quantization, or checkpoint invalidates resident GMS weights",
                        actions=[
                            "restart GMS server or clear committed layout before reloading",
                            "or launch without --gms-read-only so a fresh RW writer republishes",
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
            stale_keys=stale,
            seconds=time.monotonic() - t0,
        )

    def _gms_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["GMS_SOCKET_DIR"] = self.socket_dir
        if self.use_v1:
            env["DYN_GMS_USE_V1"] = "true"
        if self.extra_pythonpath:
            existing = env.get("PYTHONPATH", "")
            parts = [*self.extra_pythonpath, *([existing] if existing else [])]
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
        """Reuse an already-running GMS server, or launch one local GMS server per target GPU."""
        t0 = time.monotonic()
        first = self.probe(gpus=gpus, visible_devices=visible_devices, tp=tp)
        if first.ok or not first.matched_gpus or (first.finding and first.finding.get("check") == "gms_socket_path"):
            return first

        os.makedirs(self.socket_dir, exist_ok=True)
        log_root = self.log_dir or self.socket_dir
        os.makedirs(log_root, exist_ok=True)
        env = self._gms_env()
        devices = _resolve_device_indices(gpus, visible_devices=visible_devices)
        if tp is not None and len(devices) > tp:
            devices = devices[:tp]

        started: list[tuple[int, subprocess.Popen]] = []
        try:
            for dev in devices:
                cmd = [
                    self.python,
                    "-m",
                    "gpu_memory_service",
                    "--device",
                    str(dev),
                    "--device-type",
                    self.device_type,
                ]
                for tag in self.tags:
                    cmd.extend(["--tag", tag])
                log_path = os.path.join(log_root, f"gms-server-gpu{dev}.log")
                log_fh = open(log_path, "ab")
                try:
                    child = subprocess.Popen(
                        cmd,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    log_fh.close()
                started.append((dev, child))
                self._owned_pids.append(child.pid)

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
                if pr.ok:
                    return StoreProbe(
                        ok=True,
                        backend=self.name,
                        sockets=pr.sockets,
                        seconds=time.monotonic() - t0,
                    )
                time.sleep(0.2)

            raise TimeoutError(f"GMS server sockets did not become ready within {timeout_s:.1f}s")
        except Exception as exc:
            # Never leave half-started GMS daemons holding GPU resources.
            for _, child in started:
                proc.stop_session(child.pid, grace=3.0)
                if child.pid in self._owned_pids:
                    self._owned_pids.remove(child.pid)
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
        if self.use_v1:
            env["DYN_GMS_USE_V1"] = "true"
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

            extra_cfg = self.extra_loader_config()
            if extra_cfg and "model_loader_extra_config" not in explicit_flags:
                args.extend(["--model-loader-extra-config", json.dumps(extra_cfg)])

        elif engine == "sglang":
            if "load_format" in explicit_flags and vals.get("load_format") != GMS_SGLANG_LOAD_FORMAT:
                findings.append(
                    harness.plain_finding(
                        "gms_load_format",
                        f"--weights-backend dynamo_gms and explicit --load-format {vals.get('load_format')!r}",
                        f"--load-format {GMS_SGLANG_LOAD_FORMAT}, or omit --load-format",
                        why="Dynamo GMS integrates with SGLang via `--load-format gms`",
                        actions=["drop --load-format", "or switch --weights-backend"],
                    )
                )
            elif "load_format" not in explicit_flags:
                args.extend(["--load-format", GMS_SGLANG_LOAD_FORMAT])
            extra_cfg = self.extra_loader_config()
            if extra_cfg and "model_loader_extra_config" not in explicit_flags:
                args.extend(["--model-loader-extra-config", json.dumps(extra_cfg)])

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

    def stop(self) -> None:
        """Stop every GMS server process spawned by this instance (leaves external sidecars alone)."""
        pids = list(self._owned_pids)
        self._owned_pids.clear()
        for pid in pids:
            proc.stop_session(pid, grace=5.0)


register_weight_store("dynamo_gms", lambda **kw: DynamoGmsStore(**kw))
