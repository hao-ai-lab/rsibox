"""Stateful KV cache snapshot, thrashing steady-state replay, and deterministic
speculative acceptance mocking for the vLLM live lab.

Status (read before using it):
- Runner level only. `KVStateSnapshotter` saves and restores the state of one V1 or V2
  `GPUModelRunner` (V1 `input_batch` + `requests` + `block_table` + runner step buffers; V2
  `req_states` + `block_tables` + `sampler` + `model_state`; plus the KV blocks the next
  `horizon_steps` touch, and every RNG the sampler reads). It does NOT cover the scheduler's
  side (EngineCore block pool, request queues, num_computed_tokens of scheduler Requests): a
  restore inside a serving engine whose scheduler then keeps going desynchronizes the two. The
  one sound use is the one gpu_check_snapshot_replay.py makes: restore, then re-execute the
  recorded SchedulerOutputs of the horizon on the runner, so the scheduler is never involved.
  It is not wired into any measured path for that reason.
- KV blocks are saved per KV-cache group: each layer's tensor is matched to its group through
  `runner.kv_cache_config.kv_cache_groups[g].layer_names` and the attention layer's bound
  `kv_cache` (static forward context). A multi-group runner without that mapping is an error,
  never a guess.
- Times (`snapshot.timing`, `restore()`'s return) are wall time on the monotonic clock with
  `torch.cuda.synchronize()` before every reading when CUDA is in use.
- `SpeculativeAcceptanceMocker` fakes the acceptance decision. It is refused inside
  `scored_run()` and when FASTRSI_SCORED=1, and a scored run refuses to start or end while one
  is installed: mocked acceptance can never produce a scored number.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import math
import os
import random
import time
from typing import Any, Callable, Sequence

try:
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]

PLACEHOLDER_TOKEN_ID = -1


class SnapshotError(ValueError):
    """The snapshot cannot be taken or restored faithfully (nothing is guessed)."""


def _cuda_in_use() -> bool:
    return bool(_torch is not None and _torch.cuda.is_available() and _torch.cuda.is_initialized())


def _sync() -> None:
    if _cuda_in_use():
        _torch.cuda.synchronize()


def _now_ms() -> float:
    """Monotonic clock in ms, after the GPU finished the work queued so far."""
    _sync()
    return time.perf_counter() * 1000.0


@dataclass(frozen=True)
class SnapshotFinding:
    """Structured finding emitted during snapshot capture, synthesis, or replay."""
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True)
class ThrashingRegimeSpec:
    """Specification for a synthetic long-context decode + chunked-prefill batch."""
    num_decode_reqs: int = 4
    decode_context_len: int = 131_072
    num_prefill_reqs: int = 1
    prefill_chunk_len: int = 8_192
    prefill_prior_context_len: int = 0
    num_spec_tokens: int = 0
    horizon_steps: int = 20
    block_sizes: tuple[int, ...] = (16,)
    vocab_size: int = 32_000
    seed: int = 42


@dataclass
class SimpleCachedRequest:
    """Stand-in for vLLM's ``CachedRequestState`` during synthetic thrashing replay."""
    req_id: str
    prompt_token_ids: list[int]
    num_computed_tokens: int
    output_token_ids: list[int]
    block_ids: tuple[list[int], ...]
    prev_num_draft_len: int | None = 0
    mrope_positions: Any = None
    mrope_position_delta: int | None = None


@dataclass
class SimpleSamplerOutput:
    """Stand-in for vLLM V1 ``SamplerOutput``."""
    sampled_token_ids: Any
    logprobs_tensors: Any = None


@dataclass
class KVStateSnapshot:
    """Captured steady-state snapshot of a vLLM model runner."""
    created_at_ns: int
    kv_mode: str
    horizon_steps: int
    max_step_tokens: int
    num_reqs: int
    total_scheduled_tokens: int
    req_ids: list[str | None]
    req_id_to_index: dict[str, int]
    requests: dict[str, dict[str, Any]]
    input_slices: dict[str, Any]
    block_tables: list[dict[str, Any]]
    runner_buffers: dict[str, Any]
    kv_backups: list[dict[str, Any]] = field(default_factory=list)
    dirty_block_ids_per_group: list[list[int]] = field(default_factory=list)
    rng_state: dict[str, Any] = field(default_factory=dict)
    findings: list[SnapshotFinding] = field(default_factory=list)
    capture_ms: float = 0.0
    timing: dict[str, float] = field(default_factory=dict)
    layer_groups: dict[str, int] = field(default_factory=dict)

    @property
    def total_backed_up_bytes(self) -> int:
        return sum(int(item["nbytes"]) for item in self.kv_backups)

    def summary(self) -> dict[str, Any]:
        return {
            "kv_mode": self.kv_mode, "num_reqs": self.num_reqs, "total_scheduled_tokens": self.total_scheduled_tokens,
            "horizon_steps": self.horizon_steps, "max_step_tokens": self.max_step_tokens,
            "dirty_blocks_per_group": [len(grp) for grp in self.dirty_block_ids_per_group],
            "backed_up_kv_bytes": self.total_backed_up_bytes, "capture_ms": round(self.capture_ms, 3),
            "timing_ms": {k: round(v, 4) for k, v in self.timing.items()},
            "findings": [item.to_dict() for item in self.findings],
        }


def _is_torch(obj: Any) -> bool:
    return _torch is not None and isinstance(obj, _torch.Tensor)


def _clone_slice(buf: Any, rows: int, cols: int | None = None, pin: bool = False) -> Any:
    if buf is None:
        return None
    if _is_torch(buf):
        sliced = buf[:rows, :cols] if (cols is not None and buf.ndim >= 2) else buf[:rows]
        cloned = sliced.detach().cpu().clone()
        if pin and _torch.cuda.is_available():
            try:
                cloned = cloned.pin_memory()
            except RuntimeError:
                pass
        return cloned
    if hasattr(buf, "clone_slice"):
        return buf.clone_slice(rows, cols)
    if hasattr(buf, "copy") and callable(buf.copy):
        return (buf[:rows, :cols] if (cols is not None and getattr(buf, "ndim", 1) >= 2) else buf[:rows]).copy()
    if isinstance(buf, list):
        return [list(row[:cols]) for row in buf[:rows]] if (cols is not None and buf and isinstance(buf[0], list)) else copy.deepcopy(buf[:rows])
    return copy.deepcopy(buf)


def _restore_slice(dst: Any, src: Any, rows: int, cols: int | None = None) -> None:
    if dst is None or src is None:
        return
    if _is_torch(dst) and _is_torch(src):
        target = dst[:rows, :cols] if (cols is not None and dst.ndim >= 2) else dst[:rows]
        target.copy_(src, non_blocking=(dst.device.type == "cuda"))
    elif hasattr(dst, "restore_slice"):
        dst.restore_slice(src, rows, cols)
    elif isinstance(dst, list) and isinstance(src, list):
        if cols is not None and dst and isinstance(dst[0], list):
            for row_idx in range(min(rows, len(src), len(dst))):
                dst[row_idx][:cols] = list(src[row_idx][:cols])
        else:
            dst[:rows] = copy.deepcopy(src[:rows])
    elif cols is not None and getattr(dst, "ndim", 1) >= 2:
        dst[:rows, :cols] = src
    else:
        dst[:rows] = src


def _extract_pair(buf: Any) -> tuple[Any, Any]:
    if buf is None:
        return None, None
    if hasattr(buf, "cpu") and hasattr(buf, "gpu"):
        return buf.cpu, buf.gpu
    return (None, buf) if (_is_torch(buf) and buf.device.type != "cpu") else (buf, None)


def _snapshot_pair(buf: Any, rows: int, cols: int | None = None, pin: bool = False) -> dict[str, Any] | None:
    cpu_buf, gpu_buf = _extract_pair(buf)
    if cpu_buf is None and gpu_buf is None:
        return None
    return {"rows": rows, "cols": cols, "cpu": _clone_slice(cpu_buf, rows, cols, pin=pin), "gpu": _clone_slice(gpu_buf, rows, cols, pin=pin)}


def _restore_pair(buf: Any, snap: dict[str, Any] | None) -> None:
    if buf is None or snap is None:
        return
    cpu_buf, gpu_buf = _extract_pair(buf)
    rows, cols = snap["rows"], snap["cols"]
    if cpu_buf is not None and snap["cpu"] is not None:
        _restore_slice(cpu_buf, snap["cpu"], rows, cols)
    if gpu_buf is not None and (snap["gpu"] is not None or snap["cpu"] is not None):
        _restore_slice(gpu_buf, snap["gpu"] if snap["gpu"] is not None else snap["cpu"], rows, cols)


def _write_pair_1d(buf: Any, values: Sequence[int]) -> None:
    cpu_b, gpu_b = _extract_pair(buf)
    _write_1d_ints(cpu_b, values)
    _write_1d_ints(gpu_b, values)


def _is_v2_runner(runner: Any) -> bool:
    return (
        getattr(runner, "input_batch", None) is None
        and getattr(runner, "req_states", None) is not None
        and getattr(runner, "block_tables", None) is not None
    )


def _require_input_batch(runner: Any) -> Any:
    if (batch := getattr(runner, "input_batch", None)) is None:
        raise ValueError("Unsupported kv_mode='tail' or missing runner.input_batch / runner.req_states")
    return batch


def _require_runner_kind(runner: Any) -> str:
    if getattr(runner, "input_batch", None) is not None:
        return "v1"
    if _is_v2_runner(runner):
        return "v2"
    raise ValueError("Unsupported kv_mode or missing runner.input_batch / runner.req_states (need V1 input_batch or V2 req_states + block_tables)")


def _unwrap_gpu(obj: Any) -> Any:
    return obj.gpu if hasattr(obj, "gpu") else obj


def _read_scalar(buf: Any, idx: int | tuple[int, ...], default: Any = 0) -> Any:
    if buf is None:
        return default
    val = buf[idx]
    if _is_torch(val):
        return val.detach().cpu().item()
    return val.item() if hasattr(val, "item") else val


def _write_scalar(buf: Any, idx: int | tuple[int, ...], val: Any) -> None:
    if buf is None:
        return
    buf[idx] = val


def _clone_row(buf: Any, row_idx: int, cols: int | None = None, *, pin: bool = False) -> Any:
    if buf is None:
        return None
    if _is_torch(buf):
        row = buf[row_idx, :cols] if (cols is not None and buf.ndim >= 2) else buf[row_idx]
        cloned = row.detach().cpu().clone()
        if pin and cloned.numel() > 0 and _torch.cuda.is_available():
            try:
                cloned = cloned.pin_memory()
            except RuntimeError:
                pass
        return cloned
    if _np is not None and isinstance(buf, _np.ndarray):
        row = buf[row_idx, :cols] if (cols is not None and buf.ndim >= 2) else buf[row_idx]
        return row.copy()
    if hasattr(buf, "read_row"):
        return buf.read_row(row_idx, cols if cols is not None else _shape_of(buf)[1])
    if isinstance(buf, list):
        row = buf[row_idx]
        return list(row[:cols]) if (cols is not None and isinstance(row, list)) else copy.deepcopy(row)
    return copy.deepcopy(buf[row_idx, :cols] if cols is not None else buf[row_idx])


def _restore_row(dst: Any, row_idx: int, src: Any, cols: int | None = None) -> None:
    if dst is None or src is None:
        return
    if _is_torch(dst) and _is_torch(src):
        target = dst[row_idx, :cols] if (cols is not None and dst.ndim >= 2) else dst[row_idx]
        if target.numel() > 0:
            target.copy_(src, non_blocking=(dst.device.type == "cuda"))
    elif _np is not None and isinstance(dst, _np.ndarray):
        if cols is not None and dst.ndim >= 2:
            dst[row_idx, :cols] = src
        else:
            dst[row_idx] = src
    elif hasattr(dst, "write_row"):
        dst.write_row(row_idx, 0, src)
    elif isinstance(dst, list):
        if cols is not None and isinstance(dst[row_idx], list):
            dst[row_idx][:cols] = list(src[:cols])
        else:
            dst[row_idx] = copy.deepcopy(src)
    elif cols is not None and getattr(dst, "ndim", 1) >= 2:
        dst[row_idx, :cols] = src
    else:
        dst[row_idx] = src


def _clone_col(buf: Any, col_idx: int, *, pin: bool = False) -> Any:
    if buf is None:
        return None
    if _is_torch(buf):
        col = buf[:, col_idx]
        cloned = col.detach().cpu().clone()
        if pin and cloned.numel() > 0 and _torch.cuda.is_available():
            try:
                cloned = cloned.pin_memory()
            except RuntimeError:
                pass
        return cloned
    if _np is not None and isinstance(buf, _np.ndarray):
        return buf[:, col_idx].copy()
    if hasattr(buf, "clone_col"):
        return buf.clone_col(col_idx)
    return [row[col_idx] for row in buf]


def _restore_col(dst: Any, col_idx: int, src: Any) -> None:
    if dst is None or src is None:
        return
    if _is_torch(dst) and _is_torch(src):
        target = dst[:, col_idx]
        if target.numel() > 0:
            target.copy_(src, non_blocking=(dst.device.type == "cuda"))
    elif _np is not None and isinstance(dst, _np.ndarray):
        dst[:, col_idx] = src
    elif hasattr(dst, "restore_col"):
        dst.restore_col(col_idx, src)
    else:
        for r_idx, val in enumerate(src):
            dst[r_idx][col_idx] = val


def _sync_uva(obj: Any) -> None:
    if hasattr(obj, "copy_to_uva"):
        obj.copy_to_uva()


def _clear_staged(obj: Any) -> None:
    if hasattr(obj, "clear_staged_writes"):
        obj.clear_staged_writes()


def _v2_active_slots(req_states: Any) -> tuple[list[int], dict[str, int], dict[int, str], list[int]]:
    req_id_to_index = {str(k): int(v) for k, v in (getattr(req_states, "req_id_to_index", {}) or {}).items()}
    raw_idx_map = getattr(req_states, "index_to_req_id", None)
    if raw_idx_map:
        index_to_req_id = {int(k): str(v) for k, v in raw_idx_map.items()}
    else:
        index_to_req_id = {v: k for k, v in req_id_to_index.items()}
    free_indices = [int(i) for i in (getattr(req_states, "free_indices", None) or ())]
    active_slots = sorted(set(req_id_to_index.values()) | set(index_to_req_id))
    return active_slots, req_id_to_index, index_to_req_id, free_indices


def _v2_capture_uva_or_np(buf: Any, slots: list[int]) -> dict[int, Any] | None:
    if buf is None:
        return None
    arr = getattr(buf, "np", buf)
    return {s: _read_scalar(arr, s) for s in slots}


def _v2_restore_uva_or_np(buf: Any, values: dict[int, Any] | None) -> None:
    if buf is None or values is None:
        return
    arr = getattr(buf, "np", buf)
    for s, val in values.items():
        _write_scalar(arr, s, val)
    _sync_uva(buf)


def _iter_block_tables(multi_bt: Any) -> list[Any]:
    if multi_bt is None:
        return []
    return list(multi_bt.block_tables) if hasattr(multi_bt, "block_tables") else (list(multi_bt) if isinstance(multi_bt, (list, tuple)) else [multi_bt])


def _is_recurrent_table(table: Any) -> bool:
    mode = getattr(table, "slot_mapping_mode", None)
    return True if (mode is not None and getattr(mode, "name", None) == "NONE") else bool(getattr(table, "is_recurrent", False))


def _iter_kv_tensors(kv_caches: Any) -> list[tuple[Any, Any]]:
    if kv_caches is None:
        return []
    items: list[tuple[Any, Any]] = []
    for key, val in (kv_caches.items() if isinstance(kv_caches, dict) else enumerate(kv_caches)):
        items.extend(((key, sub_idx), sub_val) for sub_idx, sub_val in enumerate(val)) if (isinstance(val, (list, tuple)) and not hasattr(val, "shape")) else items.append((key, val))
    return items


def kv_layer_groups(runner: Any) -> list[tuple[Any, Any, int]]:
    """[(key, kv tensor, KV-cache group index)] for every KV tensor of the runner.

    vLLM's runner.kv_caches is ordered by layer index, NOT by group, so a list index is never a
    group index. The group of a layer comes from runner.kv_cache_config.kv_cache_groups[g]
    .layer_names, and its tensor from the attention layer the runner bound it to
    (compilation_config.static_forward_context[name].kv_cache[0]). A runner with one block
    table group needs no mapping: every tensor is group 0. A multi-group runner without the
    mapping raises SnapshotError."""
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is not None:
        num_groups = len(_iter_block_tables(getattr(input_batch, "block_table", None)))
    else:
        bt = getattr(runner, "block_tables", None)
        num_groups = int(getattr(bt, "num_kv_cache_groups", len(_iter_block_tables(bt))))
    cfg = getattr(runner, "kv_cache_config", None)
    groups = getattr(cfg, "kv_cache_groups", None)
    if groups:
        comp = getattr(runner, "compilation_config", None)
        if comp is None:
            comp = getattr(getattr(runner, "vllm_config", None), "compilation_config", None)
        ctx = getattr(comp, "static_forward_context", None) or {}
        out: list[tuple[Any, Any, int]] = []
        for g_idx, group in enumerate(groups):
            for name in getattr(group, "layer_names", []):
                layer = ctx.get(name)
                kv = getattr(layer, "kv_cache", None)
                if kv is None:
                    raise SnapshotError(f"layer {name} (KV group {g_idx}) has no bound kv_cache")
                kv = kv[0] if isinstance(kv, (list, tuple)) and kv and not _is_torch(kv) else kv
                if isinstance(kv, (list, tuple)) and not hasattr(kv, "shape"):   # mamba state
                    out.extend(((name, i), sub, g_idx) for i, sub in enumerate(kv))
                else:
                    out.append((name, kv, g_idx))
        return out
    items = _iter_kv_tensors(getattr(runner, "kv_caches", None))
    if num_groups <= 1:
        return [(key, val, 0) for key, val in items]
    raise SnapshotError(f"runner has {num_groups} KV-cache groups but no kv_cache_config to map "
                        "its layers to them; refusing to guess (a layer index is not a group)")


def _shape_of(tensor: Any) -> tuple[int, ...]:
    if hasattr(tensor, "shape"):
        return tuple(int(dim) for dim in tensor.shape)
    return ((len(tensor), len(tensor[0])) if (tensor and isinstance(tensor[0], list)) else (len(tensor),)) if isinstance(tensor, list) else ()


def _infer_block_axis(kv_tensor: Any) -> int:
    shape = _shape_of(kv_tensor)
    return 1 if (len(shape) >= 3 and shape[0] == 2 and shape[1] != 2) else 0


def _read_1d_ints(buf: Any, count: int) -> list[int]:
    if buf is None or count <= 0:
        return []
    if _is_torch(buf):
        return [int(val) for val in buf[:count].detach().cpu().tolist()]
    if hasattr(buf, "to_list"):
        return [int(val) for val in buf.to_list()[:count]]
    return [int(val) for val in (buf[:count].tolist() if hasattr(buf, "tolist") else list(buf)[:count])]


def _write_1d_ints(buf: Any, values: Sequence[int]) -> None:
    if buf is None or not values:
        return
    if _is_torch(buf):
        buf[:len(values)].copy_(_torch.tensor(list(values), dtype=buf.dtype, device=buf.device))
    elif hasattr(buf, "write_1d"):
        buf.write_1d(values)
    else:
        for idx, val in enumerate(values):
            buf[idx] = int(val)


def _read_2d_row(buf: Any, row_idx: int, cols: int) -> list[int]:
    if _is_torch(buf):
        return [int(val) for val in buf[row_idx, :cols].detach().cpu().tolist()]
    return [int(val) for val in (buf.read_row(row_idx, cols) if hasattr(buf, "read_row") else (buf[row_idx, :cols].tolist() if hasattr(buf, "tolist") else buf[row_idx][:cols]))]


def _write_2d_row(buf: Any, row_idx: int, start_col: int, values: Sequence[Any]) -> None:
    if not values:
        return
    if _is_torch(buf):
        buf[row_idx, start_col : start_col + len(values)].copy_(_torch.tensor(list(values), dtype=buf.dtype, device=buf.device))
    elif hasattr(buf, "write_row"):
        buf.write_row(row_idx, start_col, values)
    else:
        buf[row_idx][start_col : start_col + len(values)] = list(values)


def _block_index(kv_tensor: Any, block_ids: Sequence[int]) -> Any:
    return _torch.tensor(block_ids, dtype=_torch.long, device=kv_tensor.device)


def _backup_kv_blocks(kv_tensor: Any, block_axis: int, block_ids: list[int], pin: bool) -> dict[str, Any]:
    if _is_torch(kv_tensor):
        selected = _torch.index_select(kv_tensor.detach(), dim=block_axis, index=_block_index(kv_tensor, block_ids))
        pages = _clone_slice(selected, selected.shape[0], pin=pin)
        return {"block_axis": block_axis, "block_ids": list(block_ids), "pages": pages, "nbytes": int(pages.numel() * pages.element_size())}
    if hasattr(kv_tensor, "backup_blocks"):
        pages, nbytes = kv_tensor.backup_blocks(block_axis, block_ids)
        return {"block_axis": block_axis, "block_ids": list(block_ids), "pages": pages, "nbytes": nbytes}
    pages_list = [copy.deepcopy(kv_tensor[bid] if block_axis == 0 else [kv_tensor[0][bid], kv_tensor[1][bid]]) for bid in block_ids]
    return {"block_axis": block_axis, "block_ids": list(block_ids), "pages": pages_list, "nbytes": len(block_ids) * 64}


def _restore_kv_blocks(kv_tensor: Any, backup: dict[str, Any]) -> None:
    block_axis, block_ids, pages = int(backup["block_axis"]), list(backup["block_ids"]), backup["pages"]
    if _is_torch(kv_tensor):
        kv_tensor.detach().index_copy_(block_axis, _block_index(kv_tensor, block_ids), pages.to(device=kv_tensor.device, non_blocking=True))
    elif hasattr(kv_tensor, "restore_blocks"):
        kv_tensor.restore_blocks(block_axis, block_ids, pages)


def _fill_kv_blocks(kv_tensor: Any, block_axis: int, block_ids: list[int], fill_val: float) -> None:
    if not block_ids:
        return
    if _is_torch(kv_tensor):
        shape = list(kv_tensor.shape)
        shape[block_axis] = len(block_ids)
        kv_tensor.detach().index_copy_(block_axis, _block_index(kv_tensor, block_ids), _torch.full(shape, fill_val, dtype=kv_tensor.dtype, device=kv_tensor.device))
    elif hasattr(kv_tensor, "fill_blocks"):
        kv_tensor.fill_blocks(block_axis, block_ids, fill_val)


def _iter_valid_kv_blocks(runner: Any, per_group_ids: list[list[int]]):
    """Yield `(key, kv_tensor, group_idx, axis, num_phys, requested_ids, valid_ids)`."""
    for key, kv_t, g_idx in kv_layer_groups(runner):
        grp = per_group_ids[g_idx] if g_idx < len(per_group_ids) else []
        axis = _infer_block_axis(kv_t)
        num_phys = _shape_of(kv_t)[axis]
        yield key, kv_t, g_idx, axis, num_phys, grp, [bid for bid in grp if 0 <= bid < num_phys]


_BATCH_INT_FIELDS = ("num_tokens", "num_tokens_no_spec", "num_prompt_tokens", "num_computed_tokens_cpu", "num_accepted_tokens_cpu")
_RUNNER_PAIR_BUFFERS = ("seq_lens", "query_start_loc", "positions", "num_computed_tokens", "num_accepted_tokens")


class KVStateSnapshotter:
    """Captures, synthesizes, and replays steady-state KV and batch states."""
    VALID_KV_MODES = ("metadata_only", "tail", "full")

    def __init__(self, kv_mode: str = "tail", horizon_steps: int = 20, max_step_tokens: int = 8, pin_memory: bool = True) -> None:
        if kv_mode not in self.VALID_KV_MODES or horizon_steps < 1 or max_step_tokens < 1:
            raise ValueError(f"Invalid snapshotter config: kv_mode={kv_mode!r}, horizon_steps={horizon_steps}, max_step_tokens={max_step_tokens}")
        self.kv_mode, self.horizon_steps, self.max_step_tokens, self.pin_memory = kv_mode, int(horizon_steps), int(max_step_tokens), bool(pin_memory)
        self._last_snapshot: KVStateSnapshot | None = None

    @property
    def last_snapshot(self) -> KVStateSnapshot | None:
        return self._last_snapshot

    def _compute_v2_dirty_block_ids(
        self, runner: Any, eff_mode: str, horizon_tokens: int, step_toks: int
    ) -> list[list[int]]:
        req_states, bt_mgr = runner.req_states, runner.block_tables
        active_slots, _, _, _ = _v2_active_slots(req_states)
        comp_gpu_buf = _unwrap_gpu(getattr(req_states, "num_computed_tokens", None))
        comp_np_buf = getattr(req_states, "num_computed_tokens_np", None)
        tot_gpu_buf = _unwrap_gpu(getattr(req_states, "total_len", None))
        tables = list(getattr(bt_mgr, "block_tables", []))
        block_sizes = list(getattr(bt_mgr, "block_sizes", []))
        slot_enabled = list(getattr(bt_mgr, "_slot_mapping_enabled", []))
        num_blocks_buf = getattr(bt_mgr, "num_blocks", None)
        num_blocks_arr = getattr(num_blocks_buf, "np", num_blocks_buf)
        dirty_per_group: list[list[int]] = []
        for g_idx, bt_wrapper in enumerate(tables):
            bt_buf = _unwrap_gpu(bt_wrapper)
            if bt_buf is None:
                dirty_per_group.append([])
                continue
            block_size = max(1, int(block_sizes[g_idx]) if g_idx < len(block_sizes) else 16)
            is_recurrent = _is_recurrent_table(bt_wrapper) or (
                g_idx < len(slot_enabled) and not bool(slot_enabled[g_idx])
            )
            bt_shape, dirty_set = _shape_of(bt_buf), set()
            for slot in active_slots:
                if slot < 0 or slot >= bt_shape[0]:
                    continue
                row_blocks = int(_read_scalar(num_blocks_arr, (g_idx, slot), bt_shape[1])) if num_blocks_arr is not None else bt_shape[1]
                if row_blocks <= 0:
                    continue
                if eff_mode == "full" or is_recurrent:
                    start_blk, end_blk = 0, min(row_blocks, bt_shape[1])
                else:
                    comp_np = int(_read_scalar(comp_np_buf, slot, 0)) if comp_np_buf is not None else None
                    comp_gpu = int(_read_scalar(comp_gpu_buf, slot, 0)) if comp_gpu_buf is not None else (comp_np or 0)
                    min_comp = min(comp_gpu, comp_np) if comp_np is not None else comp_gpu
                    tot_tok = int(_read_scalar(tot_gpu_buf, slot, comp_gpu + step_toks)) if tot_gpu_buf is not None else (comp_gpu + step_toks)
                    max_pos = max(comp_gpu, comp_np or 0, tot_tok)
                    start_blk = max(0, max(0, min_comp - 1) // block_size)
                    end_blk = min(row_blocks, bt_shape[1], (max_pos + horizon_tokens + block_size - 1) // block_size + 1)
                dirty_set.update(int(phys_id) for phys_id in _read_2d_row(bt_buf, slot, end_blk)[start_blk:end_blk] if phys_id >= 0)
            dirty_per_group.append(sorted(dirty_set))
        return dirty_per_group

    def compute_dirty_block_ids(self, runner: Any, horizon_steps: int | None = None, max_step_tokens: int | None = None, mode: str | None = None) -> list[list[int]]:
        eff_mode, input_batch = mode or self.kv_mode, getattr(runner, "input_batch", None)
        if eff_mode == "metadata_only":
            return []
        steps = horizon_steps if horizon_steps is not None else self.horizon_steps
        step_toks = max_step_tokens if max_step_tokens is not None else self.max_step_tokens
        horizon_tokens = steps * step_toks
        if input_batch is None:
            return self._compute_v2_dirty_block_ids(runner, eff_mode, horizon_tokens, step_toks) if _is_v2_runner(runner) else []
        num_reqs = int(getattr(input_batch, "num_reqs", 0))
        comp_list = _read_1d_ints(getattr(input_batch, "num_computed_tokens_cpu", None), num_reqs)
        tot_list = _read_1d_ints(getattr(input_batch, "num_tokens_no_spec", None), num_reqs)
        dirty_per_group: list[list[int]] = []
        for table in _iter_block_tables(getattr(input_batch, "block_table", None)):
            block_size, is_recurrent = max(1, int(getattr(table, "block_size", 16))), _is_recurrent_table(table)
            bt_cpu, _ = _extract_pair(getattr(table, "block_table", table))
            row_counts = _read_1d_ints(getattr(table, "num_blocks_per_row", None), num_reqs)
            if bt_cpu is None:
                dirty_per_group.append([])
                continue
            bt_shape, dirty_set = _shape_of(bt_cpu), set()
            for req_idx in range(min(num_reqs, bt_shape[0])):
                row_blocks = row_counts[req_idx] if req_idx < len(row_counts) else bt_shape[1]
                if row_blocks <= 0:
                    continue
                if eff_mode == "full" or is_recurrent:
                    start_blk, end_blk = 0, row_blocks
                else:
                    comp_tok = comp_list[req_idx] if req_idx < len(comp_list) else 0
                    tot_tok = tot_list[req_idx] if req_idx < len(tot_list) else comp_tok + step_toks
                    start_blk = max(0, max(0, comp_tok - 1) // block_size)
                    end_blk = min(row_blocks, (max(comp_tok, tot_tok) + horizon_tokens + block_size - 1) // block_size + 1)
                dirty_set.update(int(phys_id) for phys_id in _read_2d_row(bt_cpu, req_idx, end_blk)[start_blk:end_blk] if phys_id >= 0)
            dirty_per_group.append(sorted(dirty_set))
        return dirty_per_group

    def _capture_v2_metadata(self, runner: Any, eff_steps: int, eff_step_toks: int) -> dict[str, Any]:
        req_states, bt_mgr = runner.req_states, runner.block_tables
        active_slots, req_id_to_index, index_to_req_id, free_indices = _v2_active_slots(req_states)
        req_ids = [index_to_req_id.get(s, "") for s in active_slots]
        num_reqs, pin = len(active_slots), self.pin_memory
        comp_gpu_buf = _unwrap_gpu(getattr(req_states, "num_computed_tokens", None))
        tot_gpu_buf = _unwrap_gpu(getattr(req_states, "total_len", None))
        comp_np_buf = getattr(req_states, "num_computed_tokens_np", None)
        comp_prefill_buf = getattr(req_states, "num_computed_prefill_tokens", None)
        max_seq_buf = getattr(req_states, "max_seq_len", None)
        prompt_len_buf = getattr(req_states, "prompt_len", None)
        prefill_len_buf = getattr(req_states, "prefill_len", None)
        all_tok_buf = _unwrap_gpu(getattr(req_states, "all_token_ids", None))
        max_model_len = int(getattr(req_states, "max_model_len", _shape_of(all_tok_buf)[1] if all_tok_buf is not None else 0))

        requests_snap = {}
        for req_id, slot in req_id_to_index.items():
            requests_snap[req_id] = {
                "slot": slot,
                "num_computed_tokens_np": int(_read_scalar(comp_np_buf, slot, 0)) if comp_np_buf is not None else 0,
                "num_computed_prefill_tokens": int(_read_scalar(comp_prefill_buf, slot, 0)) if comp_prefill_buf is not None else 0,
                "max_seq_len": int(_read_scalar(max_seq_buf, slot, 0)) if max_seq_buf is not None else 0,
                "prompt_len": int(_read_scalar(getattr(prompt_len_buf, "np", prompt_len_buf), slot, 0)) if prompt_len_buf is not None else 0,
                "prefill_len": int(_read_scalar(getattr(prefill_len_buf, "np", prefill_len_buf), slot, 0)) if prefill_len_buf is not None else 0,
            }

        last_sampled = getattr(req_states, "last_sampled_tokens", None)
        draft_toks = getattr(req_states, "draft_tokens", None)
        has_draft_cols = draft_toks is not None and len(_shape_of(draft_toks)) >= 2 and _shape_of(draft_toks)[1] > 0
        next_prefill = getattr(req_states, "next_prefill_tokens", None)
        slot_tensors: dict[int, dict[str, Any]] = {}
        for slot in active_slots:
            t_len = int(_read_scalar(tot_gpu_buf, slot, 0)) if tot_gpu_buf is not None else 0
            c_gpu = int(_read_scalar(comp_gpu_buf, slot, 0)) if comp_gpu_buf is not None else 0
            c_np = int(_read_scalar(comp_np_buf, slot, 0)) if comp_np_buf is not None else 0
            cols = min(max_model_len, max(1, max(t_len, c_gpu, c_np) + eff_steps * eff_step_toks + 4)) if all_tok_buf is not None else 0
            slot_tensors[slot] = {
                "total_len": t_len,
                "num_computed_tokens": c_gpu,
                "last_sampled_tokens": _clone_row(last_sampled, slot, pin=pin),
                "draft_tokens": _clone_row(draft_toks, slot, pin=pin) if has_draft_cols else None,
                "next_prefill_tokens": _clone_col(next_prefill, slot, pin=pin),
                "all_token_ids_cols": cols,
                "all_token_ids": _clone_row(all_tok_buf, slot, cols, pin=pin) if all_tok_buf is not None and cols > 0 else None,
            }

        num_blocks_buf = getattr(bt_mgr, "num_blocks", None)
        num_blocks_arr = getattr(num_blocks_buf, "np", num_blocks_buf)
        block_sizes = list(getattr(bt_mgr, "block_sizes", []))
        bt_snapshots: list[dict[str, Any]] = []
        for g_idx, bt_wrapper in enumerate(getattr(bt_mgr, "block_tables", [])):
            bt_buf = _unwrap_gpu(bt_wrapper)
            bsize = max(1, int(block_sizes[g_idx]) if g_idx < len(block_sizes) else 16)
            max_cols = _shape_of(bt_buf)[1] if bt_buf is not None else 0
            per_slot_nb = {s: int(_read_scalar(num_blocks_arr, (g_idx, s), 0)) if num_blocks_arr is not None else max_cols for s in active_slots}
            rows_snap = {}
            for s in active_slots:
                if bt_buf is not None and max_cols > 0:
                    cols = min(max_cols, max(1, per_slot_nb[s] + math.ceil((eff_steps * eff_step_toks) / bsize) + 1))
                    rows_snap[s] = (_clone_row(bt_buf, s, cols, pin=pin), cols)
            bt_snapshots.append({"group_idx": g_idx, "v2": True, "num_blocks_by_slot": per_slot_nb, "rows": rows_snap})

        sampler_snap: dict[str, Any] = {}
        sampler = getattr(runner, "sampler", None)
        if sampler is not None:
            ss = getattr(sampler, "sampling_states", None)
            if ss is not None:
                sampler_snap["sampling_states"] = {
                    "uva": {name: _v2_capture_uva_or_np(getattr(ss, name, None), active_slots)
                            for name in ("temperature", "top_k", "top_p", "min_p", "seeds")},
                    "seeds_set": {s: bool(_read_scalar(ss.seeds_set, s, False)) for s in active_slots} if getattr(ss, "seeds_set", None) is not None else None,
                    "num_logprobs": {s: int(_read_scalar(ss.num_logprobs, s, -1)) for s in active_slots} if getattr(ss, "num_logprobs", None) is not None else None,
                }
            ps = getattr(sampler, "penalties_state", None)
            if ps is not None:
                use_pen = {s: bool(_read_scalar(ps.use_penalty, s, False)) for s in active_slots} if getattr(ps, "use_penalty", None) is not None else {}
                obc = getattr(ps, "output_bin_counts", None)
                sampler_snap["penalties_state"] = {
                    "uva": {name: _v2_capture_uva_or_np(getattr(ps, name, None), active_slots)
                            for name in ("repetition_penalty", "frequency_penalty", "presence_penalty")},
                    "use_penalty": use_pen,
                    "output_bin_counts": {s: _clone_row(obc, s, pin=pin) for s, used in use_pen.items() if used} if obc is not None else {},
                }
            tbs = getattr(sampler, "thinking_budget_state", None)
            if tbs is not None and getattr(tbs, "enabled", False):
                sampler_snap["thinking_budget_state"] = {
                    "use_thinking_budget": {s: bool(_read_scalar(tbs.use_thinking_budget, s, False)) for s in active_slots},
                    "thinking_token_budget": _v2_capture_uva_or_np(getattr(tbs, "thinking_token_budget", None), active_slots),
                    "cached_last_start": {s: _read_scalar(tbs.cached_last_start, s, -1) for s in active_slots},
                    "cached_last_end": {s: _read_scalar(tbs.cached_last_end, s, -1) for s in active_slots},
                    "cached_scan_pos": {s: _read_scalar(tbs.cached_scan_pos, s, 0) for s in active_slots},
                }

        model_state_snap: dict[str, Any] = {}
        ms = getattr(runner, "model_state", None)
        if ms is not None:
            if getattr(ms, "num_accepted_tokens_gpu", None) is not None:
                model_state_snap["num_accepted_tokens_gpu"] = {s: _read_scalar(ms.num_accepted_tokens_gpu, s, 1) for s in active_slots}
            if getattr(ms, "_mamba_state_idx_gpu", None) is not None:
                model_state_snap["_mamba_state_idx_gpu"] = {s: _read_scalar(ms._mamba_state_idx_gpu, s, 0) for s in active_slots}

        return {
            "num_reqs": num_reqs,
            "total_scheduled_tokens": max(0, num_reqs * eff_step_toks),
            "req_ids": req_ids,
            "req_id_to_index": req_id_to_index,
            "requests": requests_snap,
            "input_slices": {
                "runner_kind": "v2",
                "active_slots": active_slots,
                "index_to_req_id": index_to_req_id,
                "free_indices": free_indices,
                "slot_tensors": slot_tensors,
            },
            "block_tables": bt_snapshots,
            "runner_buffers": {
                "runner_kind": "v2",
                "num_dirty_rows": int(getattr(bt_mgr, "_num_dirty_rows", 0)),
                "sampler": sampler_snap,
                "model_state": model_state_snap,
            },
        }

    def capture(self, runner: Any, *, kv_mode: str | None = None, horizon_steps: int | None = None, max_step_tokens: int | None = None, journal: Any | None = None) -> KVStateSnapshot:
        t0, eff_mode = _now_ms(), kv_mode or self.kv_mode
        if eff_mode not in self.VALID_KV_MODES:
            raise ValueError(f"Unsupported kv_mode={eff_mode!r}; expected one of {sorted(self.VALID_KV_MODES)}")
        runner_kind = _require_runner_kind(runner)
        eff_steps = horizon_steps if horizon_steps is not None else self.horizon_steps
        eff_step_toks = max_step_tokens if max_step_tokens is not None else self.max_step_tokens
        if runner_kind == "v2":
            meta = self._capture_v2_metadata(runner, eff_steps, eff_step_toks)
            num_reqs, total_sched = meta["num_reqs"], meta["total_scheduled_tokens"]
            req_ids, req_id_to_index = meta["req_ids"], meta["req_id_to_index"]
            requests_snap, input_slices = meta["requests"], meta["input_slices"]
            bt_snapshots, runner_buffers = meta["block_tables"], meta["runner_buffers"]
        else:
            input_batch = runner.input_batch
            num_reqs = int(getattr(input_batch, "num_reqs", 0))
            req_ids = list(getattr(input_batch, "_req_ids", getattr(input_batch, "req_ids", [])))
            req_id_to_index = dict(getattr(input_batch, "req_id_to_index", {}))
            requests_snap = {
                req_id: {"num_computed_tokens": int(getattr(req_state, "num_computed_tokens", 0)), "output_token_ids": list(getattr(req_state, "output_token_ids", [])), "block_ids": tuple(list(grp) for grp in getattr(req_state, "block_ids", ())), "prev_num_draft_len": getattr(req_state, "prev_num_draft_len", None), "mrope_position_delta": getattr(req_state, "mrope_position_delta", None)}
                for req_id, req_state in (getattr(runner, "requests", {}) or {}).items()
            }
            int_fields = {f: _read_1d_ints(getattr(input_batch, f, None), num_reqs) for f in _BATCH_INT_FIELDS}
            max_col = max(int_fields["num_tokens"] + int_fields["num_tokens_no_spec"] + [1]) + eff_steps * eff_step_toks
            token_ids_cpu, is_tok_cpu = getattr(input_batch, "token_ids_cpu", None), getattr(input_batch, "is_token_ids", None)
            tok_cols = min(_shape_of(token_ids_cpu)[1], max_col) if token_ids_cpu is not None else 0
            input_slices = {
                **int_fields, "tok_cols": tok_cols,
                "token_ids_cpu": _clone_slice(token_ids_cpu, num_reqs, tok_cols) if num_reqs > 0 else None,
                "is_token_ids": _clone_slice(is_tok_cpu, num_reqs, tok_cols) if num_reqs > 0 else None,
                "spec_token_ids": [list(row) for row in getattr(input_batch, "spec_token_ids", [])[:num_reqs]],
                "req_output_token_ids": [list(row) if row is not None else None for row in getattr(input_batch, "req_output_token_ids", [])[:num_reqs]],
                "prev_req_id_to_index": copy.deepcopy(getattr(input_batch, "prev_req_id_to_index", None)),
            }
            qsl_cpu, _ = _extract_pair(getattr(runner, "query_start_loc", None))
            qsl_vals = _read_1d_ints(qsl_cpu, num_reqs + 1)
            total_sched = qsl_vals[num_reqs] if len(qsl_vals) > num_reqs else max(1, num_reqs * eff_step_toks)
            bt_snapshots = []
            for group_idx, table in enumerate(_iter_block_tables(getattr(input_batch, "block_table", None))):
                block_size = max(1, int(getattr(table, "block_size", 16)))
                row_counts = _read_1d_ints(getattr(table, "num_blocks_per_row", None), num_reqs)
                bt_cpu, _ = _extract_pair(getattr(table, "block_table", table))
                max_blks = min(_shape_of(bt_cpu)[1], max(row_counts + [1]) + math.ceil((eff_steps * eff_step_toks) / block_size))
                bt_snapshots.append({"group_idx": group_idx, "num_blocks_per_row": row_counts, "block_table": _snapshot_pair(getattr(table, "block_table", table), num_reqs, max_blks, pin=self.pin_memory), "slot_mapping": _snapshot_pair(getattr(table, "slot_mapping", None), max(1, total_sched), pin=self.pin_memory)})
            pair_sizes = {"seq_lens": max(1, num_reqs), "query_start_loc": max(1, num_reqs + 1), "positions": max(1, total_sched), "num_computed_tokens": max(1, num_reqs), "num_accepted_tokens": max(1, num_reqs)}
            runner_buffers = {
                **{name: _snapshot_pair(getattr(runner, name, None), rows, pin=self.pin_memory) for name, rows in pair_sizes.items()},
                "optimistic_seq_lens_cpu": _clone_slice(getattr(runner, "optimistic_seq_lens_cpu", None), max(1, num_reqs), pin=self.pin_memory),
            }
        dirty_per_group = self.compute_dirty_block_ids(runner, eff_steps, eff_step_toks, eff_mode)
        t_meta = _now_ms()
        findings, kv_backups, layer_groups = [], [], {}
        if eff_mode != "metadata_only":
            for key, kv_t, g_idx, axis, num_phys, grp_dirty, valid_ids in _iter_valid_kv_blocks(runner, dirty_per_group):
                layer_groups[str(key)] = g_idx
                if g_idx >= len(dirty_per_group):
                    raise SnapshotError(f"layer {key} is in KV group {g_idx}, but the runner has {len(dirty_per_group)} block tables")
                if not grp_dirty:
                    continue
                if len(valid_ids) < len(grp_dirty):
                    findings.append(SnapshotFinding("snapshot_block_id_out_of_range", "warning", f"Layer {key}: block IDs exceed pool ({num_phys}).", {"layer": str(key)}))
                if valid_ids:
                    kv_backups.append({**_backup_kv_blocks(kv_t, axis, valid_ids, pin=self.pin_memory), "key": key, "group": g_idx})
        t_kv = _now_ms()
        rng_state = capture_rng_state(getattr(runner, "input_batch", None))
        t_end = _now_ms()
        snapshot = KVStateSnapshot(
            created_at_ns=time.time_ns(), kv_mode=eff_mode, horizon_steps=eff_steps, max_step_tokens=eff_step_toks,
            num_reqs=num_reqs, total_scheduled_tokens=total_sched, req_ids=req_ids, req_id_to_index=req_id_to_index,
            requests=requests_snap, input_slices=input_slices, block_tables=bt_snapshots, runner_buffers=runner_buffers,
            kv_backups=kv_backups, dirty_block_ids_per_group=dirty_per_group, rng_state=rng_state, findings=findings,
            capture_ms=t_end - t0,
            timing={"capture_metadata_ms": t_meta - t0, "capture_kv_d2h_ms": t_kv - t_meta, "capture_rng_ms": t_end - t_kv, "capture_total_ms": t_end - t0},
            layer_groups=layer_groups,
        )
        self._last_snapshot = snapshot
        if journal is not None and hasattr(journal, "record_call"):
            journal.record_call(lambda: self.restore(runner, snapshot))
        return snapshot

    def _restore_v2_state(self, runner: Any, snap: KVStateSnapshot, kv_map: dict[Any, Any] | None) -> None:
        if not _is_v2_runner(runner):
            raise ValueError("snapshot was captured from a V2 runner, but runner has no req_states/block_tables")
        req_states, bt_mgr = runner.req_states, runner.block_tables
        cur_map = getattr(req_states, "req_id_to_index", {}) or {}
        missing = sorted(rid for rid, slot in snap.req_id_to_index.items() if cur_map.get(rid) != slot)
        if missing:
            raise SnapshotError(f"requests {missing} left or moved slot in the runner since the snapshot; their state is gone")
        if kv_map is not None:
            for backup in snap.kv_backups:
                if backup["key"] not in kv_map:
                    raise SnapshotError(f"KV tensor {backup['key']} of the snapshot is not in the runner")

        slices = snap.input_slices
        extra_slots = [int(slot) for rid, slot in cur_map.items() if rid not in snap.req_id_to_index]

        # Clear post-snapshot requests before restoring maps.
        comp_np_buf = getattr(req_states, "num_computed_tokens_np", None)
        comp_prefill_buf = getattr(req_states, "num_computed_prefill_tokens", None)
        max_seq_buf = getattr(req_states, "max_seq_len", None)
        prompt_len_buf = getattr(req_states, "prompt_len", None)
        prefill_len_buf = getattr(req_states, "prefill_len", None)
        tot_st = getattr(req_states, "total_len", None)
        comp_st = getattr(req_states, "num_computed_tokens", None)
        all_tok_st = getattr(req_states, "all_token_ids", None)
        for st in (tot_st, comp_st, all_tok_st):
            _clear_staged(st)
        tot_gpu_buf = _unwrap_gpu(tot_st)
        comp_gpu_buf = _unwrap_gpu(comp_st)
        all_tok_buf = _unwrap_gpu(all_tok_st)
        num_blocks_buf = getattr(bt_mgr, "num_blocks", None)
        num_blocks_arr = getattr(num_blocks_buf, "np", num_blocks_buf)
        sampler = getattr(runner, "sampler", None)
        for slot in extra_slots:
            for arr in (comp_np_buf, comp_prefill_buf, max_seq_buf, tot_gpu_buf, comp_gpu_buf):
                _write_scalar(arr, slot, 0)
            _write_scalar(getattr(prompt_len_buf, "np", prompt_len_buf), slot, 0)
            _write_scalar(getattr(prefill_len_buf, "np", prefill_len_buf), slot, 0)
            if num_blocks_arr is not None:
                for g_idx in range(len(getattr(bt_mgr, "block_tables", []))):
                    _write_scalar(num_blocks_arr, (g_idx, slot), 0)
            if sampler is not None:
                ss = getattr(sampler, "sampling_states", None)
                if ss is not None:
                    if getattr(ss, "seeds_set", None) is not None:
                        _write_scalar(ss.seeds_set, slot, False)
                    if getattr(ss, "num_logprobs", None) is not None:
                        _write_scalar(ss.num_logprobs, slot, -1)
                ps = getattr(sampler, "penalties_state", None)
                if ps is not None and getattr(ps, "use_penalty", None) is not None:
                    _write_scalar(ps.use_penalty, slot, False)
                tbs = getattr(sampler, "thinking_budget_state", None)
                if tbs is not None and getattr(tbs, "enabled", False):
                    _write_scalar(tbs.use_thinking_budget, slot, False)

        if hasattr(req_states, "req_id_to_index"):
            req_states.req_id_to_index.clear()
            req_states.req_id_to_index.update(snap.req_id_to_index)
        if hasattr(req_states, "index_to_req_id"):
            req_states.index_to_req_id.clear()
            req_states.index_to_req_id.update(slices["index_to_req_id"])
        if hasattr(req_states, "free_indices"):
            req_states.free_indices[:] = list(slices["free_indices"])

        for rsnap in snap.requests.values():
            slot = int(rsnap["slot"])
            _write_scalar(comp_np_buf, slot, rsnap["num_computed_tokens_np"])
            _write_scalar(comp_prefill_buf, slot, rsnap["num_computed_prefill_tokens"])
            _write_scalar(max_seq_buf, slot, rsnap["max_seq_len"])
            _write_scalar(getattr(prompt_len_buf, "np", prompt_len_buf), slot, rsnap["prompt_len"])
            _write_scalar(getattr(prefill_len_buf, "np", prefill_len_buf), slot, rsnap["prefill_len"])
        _sync_uva(prompt_len_buf)
        _sync_uva(prefill_len_buf)

        last_sampled = getattr(req_states, "last_sampled_tokens", None)
        draft_toks = getattr(req_states, "draft_tokens", None)
        next_prefill = getattr(req_states, "next_prefill_tokens", None)
        for slot, st_snap in slices["slot_tensors"].items():
            s = int(slot)
            _write_scalar(tot_gpu_buf, s, st_snap["total_len"])
            _write_scalar(comp_gpu_buf, s, st_snap["num_computed_tokens"])
            _restore_row(last_sampled, s, st_snap["last_sampled_tokens"])
            if st_snap["draft_tokens"] is not None:
                _restore_row(draft_toks, s, st_snap["draft_tokens"])
            _restore_col(next_prefill, s, st_snap["next_prefill_tokens"])
            if st_snap["all_token_ids"] is not None and st_snap["all_token_ids_cols"] > 0:
                _restore_row(all_tok_buf, s, st_snap["all_token_ids"], st_snap["all_token_ids_cols"])

        bt_list = list(getattr(bt_mgr, "block_tables", []))
        for btsnap in snap.block_tables:
            g_idx = int(btsnap["group_idx"])
            if g_idx >= len(bt_list):
                continue
            bt_wrapper = bt_list[g_idx]
            _clear_staged(bt_wrapper)
            bt_buf = _unwrap_gpu(bt_wrapper)
            if num_blocks_arr is not None:
                for slot, nb in btsnap["num_blocks_by_slot"].items():
                    _write_scalar(num_blocks_arr, (g_idx, int(slot)), nb)
            for slot, (row_data, cols) in btsnap["rows"].items():
                _restore_row(bt_buf, int(slot), row_data, cols)
        _sync_uva(num_blocks_buf)

        rbufs = snap.runner_buffers
        if hasattr(bt_mgr, "_num_dirty_rows"):
            bt_mgr._num_dirty_rows = 0

        ssnap = rbufs.get("sampler") or {}
        if sampler is not None and ssnap:
            ss_snap = ssnap.get("sampling_states")
            ss = getattr(sampler, "sampling_states", None)
            if ss is not None and ss_snap is not None:
                for name, vals in ss_snap["uva"].items():
                    _v2_restore_uva_or_np(getattr(ss, name, None), vals)
                if ss_snap["seeds_set"] is not None and getattr(ss, "seeds_set", None) is not None:
                    for s, val in ss_snap["seeds_set"].items():
                        _write_scalar(ss.seeds_set, int(s), val)
                if ss_snap["num_logprobs"] is not None and getattr(ss, "num_logprobs", None) is not None:
                    for s, val in ss_snap["num_logprobs"].items():
                        _write_scalar(ss.num_logprobs, int(s), val)
            ps_snap = ssnap.get("penalties_state")
            ps = getattr(sampler, "penalties_state", None)
            if ps is not None and ps_snap is not None:
                for name, vals in ps_snap["uva"].items():
                    _v2_restore_uva_or_np(getattr(ps, name, None), vals)
                if getattr(ps, "use_penalty", None) is not None:
                    for s, val in ps_snap["use_penalty"].items():
                        _write_scalar(ps.use_penalty, int(s), val)
                obc = getattr(ps, "output_bin_counts", None)
                if obc is not None:
                    for s, row in ps_snap["output_bin_counts"].items():
                        _restore_row(obc, int(s), row)
            tbs_snap = ssnap.get("thinking_budget_state")
            tbs = getattr(sampler, "thinking_budget_state", None)
            if tbs is not None and getattr(tbs, "enabled", False) and tbs_snap is not None:
                for s, val in tbs_snap["use_thinking_budget"].items():
                    _write_scalar(tbs.use_thinking_budget, int(s), val)
                _v2_restore_uva_or_np(getattr(tbs, "thinking_token_budget", None), tbs_snap["thinking_token_budget"])
                for attr in ("cached_last_start", "cached_last_end", "cached_scan_pos"):
                    dst = getattr(tbs, attr, None)
                    for s, val in tbs_snap[attr].items():
                        _write_scalar(dst, int(s), val)

        ms_snap = rbufs.get("model_state") or {}
        ms = getattr(runner, "model_state", None)
        if ms is not None and ms_snap:
            for attr in ("num_accepted_tokens_gpu", "_mamba_state_idx_gpu"):
                vals = ms_snap.get(attr)
                dst = getattr(ms, attr, None)
                if vals is not None and dst is not None:
                    for s, val in vals.items():
                        _write_scalar(dst, int(s), val)

        if hasattr(runner, "execute_model_state"):
            runner.execute_model_state = None

    def restore(self, runner: Any, snapshot: KVStateSnapshot | None = None) -> float:
        """Restore `snapshot`; returns its wall time in ms (CUDA-synchronized). The per-phase
        times land in `snapshot.timing` (restore_*_ms). A request of the snapshot that left the
        runner since, or a seeded row whose generator is gone, cannot be restored: SnapshotError,
        before anything is changed. Requests that joined since are dropped with their rows."""
        t0, snap = _now_ms(), snapshot or self._last_snapshot
        if snap is None:
            raise ValueError("Valid snapshot required for restore()")
        is_v2 = snap.input_slices.get("runner_kind") == "v2"
        kv_map = {key: kv_t for key, kv_t, _ in kv_layer_groups(runner)} if snap.kv_backups else None
        if is_v2:
            self._restore_v2_state(runner, snap, kv_map)
        else:
            input_batch, num_reqs, runner_requests = _require_input_batch(runner), snap.num_reqs, getattr(runner, "requests", None)
            if isinstance(runner_requests, dict):
                missing = sorted(req_id for req_id in snap.requests if req_id not in runner_requests)
                if missing:
                    raise SnapshotError(f"requests {missing} left the runner since the snapshot; their state is gone")
            gens, saved_gens = getattr(input_batch, "generators", None), snap.rng_state.get("generators")
            if isinstance(gens, dict) and saved_gens is not None and set(saved_gens) - set(gens):
                raise SnapshotError(f"seeded rows {sorted(set(saved_gens) - set(gens))} of the snapshot have no generator now; their state is gone")
            if isinstance(runner_requests, dict):
                for key in [key for key in runner_requests if key not in snap.requests]:
                    del runner_requests[key]
                for req_id, rsnap in snap.requests.items():
                    req_obj = runner_requests.get(req_id)
                    if req_obj is not None:
                        req_obj.num_computed_tokens = rsnap["num_computed_tokens"]
                        req_obj.output_token_ids[:] = list(rsnap["output_token_ids"])
                        for grp_idx, saved_blks in enumerate(rsnap["block_ids"]):
                            if grp_idx < len(req_obj.block_ids):
                                req_obj.block_ids[grp_idx][:] = list(saved_blks)
                        req_obj.prev_num_draft_len, req_obj.mrope_position_delta = rsnap["prev_num_draft_len"], rsnap["mrope_position_delta"]
            if hasattr(input_batch, "_req_ids"):
                input_batch._req_ids[:] = list(snap.req_ids)
            if hasattr(input_batch, "req_id_to_index"):
                input_batch.req_id_to_index.clear()
                input_batch.req_id_to_index.update(snap.req_id_to_index)
            slices, tok_cols = snap.input_slices, int(snap.input_slices["tok_cols"])
            _restore_slice(getattr(input_batch, "token_ids_cpu", None), slices["token_ids_cpu"], num_reqs, tok_cols)
            _restore_slice(getattr(input_batch, "is_token_ids", None), slices["is_token_ids"], num_reqs, tok_cols)
            for field_name in _BATCH_INT_FIELDS:
                _write_1d_ints(getattr(input_batch, field_name, None), slices[field_name])
            for row_idx, spec_row in enumerate(slices["spec_token_ids"]):
                if row_idx < len(input_batch.spec_token_ids):
                    input_batch.spec_token_ids[row_idx] = list(spec_row)
            # rows at and above the snapshot's num_reqs held requests that joined later: empty them
            for row_idx in range(num_reqs, len(input_batch.spec_token_ids)):
                input_batch.spec_token_ids[row_idx] = []
            for row_idx in range(num_reqs, len(input_batch.req_output_token_ids)):
                input_batch.req_output_token_ids[row_idx] = None
            for row_idx, out_row in enumerate(slices["req_output_token_ids"]):
                if row_idx < len(input_batch.req_output_token_ids):
                    # vLLM aliases this row with requests[req_id].output_token_ids (one list): keep
                    # the alias, or later appends reach only one of the two.
                    req_id = snap.req_ids[row_idx] if row_idx < len(snap.req_ids) else None
                    req_obj = runner_requests.get(req_id) if isinstance(runner_requests, dict) and req_id else None
                    if req_obj is not None and out_row is not None:
                        input_batch.req_output_token_ids[row_idx] = req_obj.output_token_ids
                    else:
                        input_batch.req_output_token_ids[row_idx] = list(out_row) if out_row is not None else None
            input_batch.prev_req_id_to_index = copy.deepcopy(slices["prev_req_id_to_index"])
            raw_tables = _iter_block_tables(getattr(input_batch, "block_table", None))
            for btsnap in snap.block_tables:
                grp_idx = int(btsnap["group_idx"])
                if grp_idx < len(raw_tables):
                    table = raw_tables[grp_idx]
                    _write_1d_ints(getattr(table, "num_blocks_per_row", None), btsnap["num_blocks_per_row"])
                    _restore_pair(getattr(table, "block_table", table), btsnap["block_table"])
                    _restore_pair(getattr(table, "slot_mapping", None), btsnap["slot_mapping"])
            rbufs = snap.runner_buffers
            for buf_name in _RUNNER_PAIR_BUFFERS:
                _restore_pair(getattr(runner, buf_name, None), rbufs[buf_name])
            _restore_slice(getattr(runner, "optimistic_seq_lens_cpu", None), rbufs["optimistic_seq_lens_cpu"], max(1, num_reqs))
            if hasattr(runner, "execute_model_state"):
                runner.execute_model_state = None
        t_meta = _now_ms()
        if snap.kv_backups:
            assert kv_map is not None
            for backup in snap.kv_backups:
                if backup["key"] not in kv_map:
                    raise SnapshotError(f"KV tensor {backup['key']} of the snapshot is not in the runner")
                _restore_kv_blocks(kv_map[backup["key"]], backup)
        t_kv = _now_ms()
        restore_rng_state(snap.rng_state, None if is_v2 else getattr(runner, "input_batch", None))
        t_end = _now_ms()
        snap.timing.update({"restore_metadata_ms": t_meta - t0, "restore_kv_h2d_ms": t_kv - t_meta, "restore_rng_ms": t_end - t_kv, "restore_total_ms": t_end - t0})
        return t_end - t0

    @contextmanager
    def frozen(self, runner: Any, **capture_kwargs: Any):
        snap = self.capture(runner, **capture_kwargs)
        try:
            yield snap
        finally:
            self.restore(runner, snap)

    def synthesize_thrashing_state(self, runner: Any, spec: ThrashingRegimeSpec | None = None, *, fill_tail_kv: bool = False) -> list[SnapshotFinding]:
        cfg, findings = spec or ThrashingRegimeSpec(), []
        py_rng, input_batch = random.Random(cfg.seed), getattr(runner, "input_batch", None)
        total_reqs = cfg.num_decode_reqs + cfg.num_prefill_reqs
        if input_batch is None or total_reqs > int(getattr(input_batch, "max_num_reqs", total_reqs)):
            raise ValueError("Missing input_batch or total_reqs exceeds max_num_reqs")
        max_model_len = int(getattr(runner, "max_model_len", getattr(input_batch, "max_model_len", 262_144)))
        decode_step_tokens = 1 + max(0, cfg.num_spec_tokens)
        needed_lens = [1] + ([cfg.decode_context_len + cfg.horizon_steps * decode_step_tokens] if cfg.num_decode_reqs > 0 else []) + ([cfg.prefill_prior_context_len + cfg.prefill_chunk_len + cfg.horizon_steps * decode_step_tokens] if cfg.num_prefill_reqs > 0 else [])
        if max(needed_lens) > max_model_len:
            raise ValueError(f"Required sequence length {max(needed_lens)} exceeds runner max_model_len={max_model_len}")
        kv_tensors = _iter_kv_tensors(getattr(runner, "kv_caches", None))
        phys_pool_limit = _shape_of(kv_tensors[0][1])[_infer_block_axis(kv_tensors[0][1])] if kv_tensors else None
        req_specs = [
            {"req_id": f"thrash-decode-{idx}", "num_computed": int(cfg.decode_context_len), "num_scheduled": decode_step_tokens, "total_tokens": int(cfg.decode_context_len) + decode_step_tokens, "num_prompt": int(cfg.decode_context_len), "spec_tokens": [py_rng.randint(10, cfg.vocab_size - 1) for _ in range(cfg.num_spec_tokens)]}
            for idx in range(cfg.num_decode_reqs)
        ] + [
            {"req_id": f"thrash-prefill-{idx}", "num_computed": int(cfg.prefill_prior_context_len), "num_scheduled": int(cfg.prefill_chunk_len), "total_tokens": int(cfg.prefill_prior_context_len) + int(cfg.prefill_chunk_len), "num_prompt": int(cfg.prefill_prior_context_len) + int(cfg.prefill_chunk_len), "spec_tokens": []}
            for idx in range(cfg.num_prefill_reqs)
        ]
        group_blocks: list[list[list[int]]] = [[] for _ in range(total_reqs)]
        for grp_idx, table in enumerate(_iter_block_tables(getattr(input_batch, "block_table", None))):
            block_size = max(1, int(getattr(table, "block_size", cfg.block_sizes[min(grp_idx, len(cfg.block_sizes) - 1)])))
            bt_cpu, _ = _extract_pair(getattr(table, "block_table", table))
            max_cols = _shape_of(bt_cpu)[1] if bt_cpu is not None else 16_384
            num_blks_list = [min(max_cols, max(1, math.ceil((item["total_tokens"] + cfg.horizon_steps * decode_step_tokens) / block_size))) for item in req_specs]
            tail_blks_list = [min(n_blks, max(2, math.ceil(((1 + cfg.horizon_steps) * decode_step_tokens) / block_size) + 1)) for n_blks in num_blks_list]
            total_needed, total_tail, next_phys, tail_cursor = sum(num_blks_list), sum(tail_blks_list), 1, 0
            for req_idx, rspec in enumerate(req_specs):
                n_blks, t_blks = num_blks_list[req_idx], tail_blks_list[req_idx]
                if phys_pool_limit is not None and total_needed + 1 > phys_pool_limit:
                    if total_tail + 2 > phys_pool_limit:
                        raise ValueError(f"Physical KV pool has only {phys_pool_limit} blocks; need {total_tail + 2} for unique tails")
                    prefix_cnt, ceiling = n_blks - t_blks, max(3, phys_pool_limit - total_tail)
                    prefix_ids = ([1] + [2 + ((col + req_idx * 17) % max(1, ceiling - 2)) for col in range(prefix_cnt - 1)]) if (req_idx == 0 and prefix_cnt > 0) else [2 + ((col + req_idx * 17) % max(1, ceiling - 2)) for col in range(prefix_cnt)]
                    blocks = prefix_ids + [ceiling + tail_cursor + col for col in range(t_blks)]
                    tail_cursor += t_blks
                    findings.append(SnapshotFinding("snapshot_prefix_blocks_aliased_to_fit_pool", "info", f"Aliased {prefix_cnt} blocks for {rspec['req_id']}.", {"req_id": rspec["req_id"]}))
                else:
                    blocks = list(range(next_phys, next_phys + n_blks))
                    next_phys += n_blks
                group_blocks[req_idx].append(blocks)
        input_batch._req_ids[:] = [item["req_id"] for item in req_specs] + [None] * max(0, len(input_batch._req_ids) - total_reqs)
        input_batch.req_id_to_index.clear()
        for idx, item in enumerate(req_specs):
            input_batch.req_id_to_index[item["req_id"]] = idx
        runner_requests = getattr(runner, "requests", None)
        if isinstance(runner_requests, dict):
            runner_requests.clear()
        _write_1d_ints(getattr(input_batch, "num_computed_tokens_cpu", None), [item["num_computed"] for item in req_specs])
        _write_1d_ints(getattr(input_batch, "num_tokens_no_spec", None), [item["total_tokens"] - len(item["spec_tokens"]) for item in req_specs])
        _write_1d_ints(getattr(input_batch, "num_tokens", None), [item["total_tokens"] for item in req_specs])
        _write_1d_ints(getattr(input_batch, "num_prompt_tokens", None), [item["num_prompt"] for item in req_specs])
        _write_1d_ints(getattr(input_batch, "num_accepted_tokens_cpu", None), [1] * total_reqs)
        tok_cpu, is_tok = getattr(input_batch, "token_ids_cpu", None), getattr(input_batch, "is_token_ids", None)
        for idx, rspec in enumerate(req_specs):
            input_batch.spec_token_ids[idx], out_toks = list(rspec["spec_tokens"]), [py_rng.randint(10, cfg.vocab_size - 1)]
            input_batch.req_output_token_ids[idx] = out_toks
            tail_start, tail_end = max(0, rspec["num_computed"] - 16), rspec["total_tokens"]
            if tok_cpu is not None and tail_end > tail_start:
                _write_2d_row(tok_cpu, idx, tail_start, [py_rng.randint(10, cfg.vocab_size - 1) for _ in range(tail_end - tail_start)])
            if is_tok is not None and tail_end > tail_start:
                _write_2d_row(is_tok, idx, tail_start, [True] * (tail_end - tail_start))
            if isinstance(runner_requests, dict):
                runner_requests[rspec["req_id"]] = SimpleCachedRequest(rspec["req_id"], [1] * min(32, rspec["num_prompt"]), rspec["num_computed"], list(out_toks), tuple(list(grp) for grp in group_blocks[idx]), len(rspec["spec_tokens"]))
        self._rebuild_step_metadata(runner, req_specs, group_blocks)
        if fill_tail_kv:
            dirty_groups = self.compute_dirty_block_ids(runner, cfg.horizon_steps, decode_step_tokens, "tail")
            for _, kv_t, _, axis, _, _, valid_ids in _iter_valid_kv_blocks(runner, dirty_groups):
                _fill_kv_blocks(kv_t, axis, valid_ids, 0.01)
        return findings

    def _rebuild_step_metadata(self, runner: Any, req_specs: list[dict[str, Any]], group_blocks: list[list[list[int]]] | None = None) -> None:
        input_batch, num_reqs, qsl_list, seq_lens_list, positions_list = runner.input_batch, len(req_specs), [0], [], []
        for rspec in req_specs:
            comp, sched = int(rspec["num_computed"]), int(rspec["num_scheduled"])
            qsl_list.append(qsl_list[-1] + sched)
            seq_lens_list.append(comp + sched)
            positions_list.extend(range(comp, comp + sched))
        for pair_buf, vals in ((getattr(runner, "query_start_loc", None), qsl_list), (getattr(runner, "seq_lens", None), seq_lens_list), (getattr(runner, "positions", None), positions_list)):
            _write_pair_1d(pair_buf, vals)
        _write_1d_ints(getattr(runner, "optimistic_seq_lens_cpu", None), seq_lens_list)
        for grp_idx, table in enumerate(_iter_block_tables(getattr(input_batch, "block_table", None))):
            block_size = max(1, int(getattr(table, "block_size", 16)))
            bt_cpu, bt_gpu = _extract_pair(getattr(table, "block_table", table))
            if bt_cpu is None:
                continue
            row_counts, slots_list = _read_1d_ints(getattr(table, "num_blocks_per_row", None), num_reqs), []
            for req_idx, rspec in enumerate(req_specs):
                if group_blocks is not None:
                    blks = group_blocks[req_idx][grp_idx]
                    n_blks, row_counts[req_idx : req_idx + 1] = len(blks), [len(blks)]
                    _write_2d_row(bt_cpu, req_idx, 0, blks)
                    if bt_gpu is not None:
                        _write_2d_row(bt_gpu, req_idx, 0, blks)
                else:
                    n_blks = row_counts[req_idx]
                row_vals = _read_2d_row(bt_cpu, req_idx, n_blks)
                comp, sched = int(rspec["num_computed"]), int(rspec["num_scheduled"])
                slots_list.extend(int(row_vals[min(pos // block_size, max(0, n_blks - 1))]) * block_size + (pos % block_size) for pos in range(comp, comp + sched))
            if group_blocks is not None:
                _write_1d_ints(getattr(table, "num_blocks_per_row", None), row_counts)
            if not _is_recurrent_table(table):
                _write_pair_1d(getattr(table, "slot_mapping", None), slots_list)

    def advance_steady_state_step(self, runner: Any, *, accepted_tokens_per_req: int | Sequence[int] = 1, prefill_chunk_len: int | None = None, mutate_tail_kv: bool = True) -> dict[str, Any]:
        input_batch, num_reqs = runner.input_batch, int(getattr(runner.input_batch, "num_reqs", 0))
        req_ids = list(getattr(input_batch, "_req_ids", getattr(input_batch, "req_ids", [])))[:num_reqs]
        runner_requests = getattr(runner, "requests", {}) or {}
        acc_list = [max(1, accepted_tokens_per_req)] * num_reqs if isinstance(accepted_tokens_per_req, int) else [max(1, int(val)) for val in accepted_tokens_per_req]
        qsl_cpu, _ = _extract_pair(getattr(runner, "query_start_loc", None))
        qsl_vals = _read_1d_ints(qsl_cpu, num_reqs + 1)
        prev_sched = [max(1, qsl_vals[idx + 1] - qsl_vals[idx]) for idx in range(num_reqs)] if len(qsl_vals) > num_reqs else [1] * num_reqs
        old_comps = _read_1d_ints(getattr(input_batch, "num_computed_tokens_cpu", None), num_reqs)
        req_specs, new_comps, new_no_specs, new_tots = [], [], [], []
        for idx in range(num_reqs):
            req_id = req_ids[idx] or f"req-{idx}"
            num_spec = len(input_batch.spec_token_ids[idx]) if idx < len(input_batch.spec_token_ids) else 0
            # a prefill chunk computes all its scheduled tokens; a decode step its accepted ones
            new_comp = old_comps[idx] + (prev_sched[idx] if "prefill" in str(req_id) else min(prev_sched[idx], acc_list[idx]))
            next_sched = (prefill_chunk_len or prev_sched[idx]) if "prefill" in str(req_id) else (1 + num_spec)
            new_no_spec, new_tot, new_tok = new_comp + (next_sched if "prefill" in str(req_id) else 1), new_comp + next_sched, 100 + ((new_comp + idx) % 30_000)
            new_comps.append(new_comp)
            new_no_specs.append(new_no_spec)
            new_tots.append(new_tot)
            tok_cpu = getattr(input_batch, "token_ids_cpu", None)
            if tok_cpu is not None and new_comp < _shape_of(tok_cpu)[1]:
                _write_2d_row(tok_cpu, idx, new_comp, [new_tok])
            if idx < len(input_batch.req_output_token_ids) and isinstance(input_batch.req_output_token_ids[idx], list):
                input_batch.req_output_token_ids[idx].append(new_tok)
            req_obj = runner_requests.get(req_id)
            if req_obj is not None:
                req_obj.num_computed_tokens = new_comp
                req_obj.output_token_ids.append(new_tok)
            req_specs.append({"req_id": req_id, "num_computed": new_comp, "num_scheduled": next_sched, "total_tokens": new_tot})
        for field_name, vals in (("num_computed_tokens_cpu", new_comps), ("num_tokens_no_spec", new_no_specs), ("num_tokens", new_tots), ("num_accepted_tokens_cpu", acc_list)):
            _write_1d_ints(getattr(input_batch, field_name, None), vals)
        self._rebuild_step_metadata(runner, req_specs, None)
        raw_tables, kv_tensors, total_sched = _iter_block_tables(getattr(input_batch, "block_table", None)), _iter_kv_tensors(getattr(runner, "kv_caches", None)), sum(item["num_scheduled"] for item in req_specs)
        if mutate_tail_kv and raw_tables and kv_tensors and total_sched > 0:
            fill_val, touched_per_group = float((req_specs[0]["num_computed"] % 97) + 1), []
            for table in raw_tables:
                block_size = max(1, int(getattr(table, "block_size", 16)))
                sm_cpu, _ = _extract_pair(getattr(table, "slot_mapping", None))
                touched_per_group.append(sorted({slot // block_size for slot in _read_1d_ints(sm_cpu, total_sched)}) if sm_cpu is not None else [])
            for _, kv_t, _, axis, _, _, valid_ids in _iter_valid_kv_blocks(runner, touched_per_group):
                _fill_kv_blocks(kv_t, axis, valid_ids, fill_val)
        return {"num_reqs": num_reqs, "total_scheduled_tokens": total_sched, "seq_lens": [item["num_computed"] + item["num_scheduled"] for item in req_specs]}

    def replay_steps(self, runner: Any, *, num_steps: int = 10, step_fn: Callable[[Any, int], Any] | None = None, snapshot: KVStateSnapshot | None = None, restore_before: bool = True, restore_after: bool = True) -> dict[str, Any]:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        snap = snapshot or self._last_snapshot or self.capture(runner)
        restore_before_ms = self.restore(runner, snap) if restore_before else 0.0
        start_replay = _now_ms()
        records = [step_fn(runner, step_idx) if step_fn is not None else self.advance_steady_state_step(runner) for step_idx in range(num_steps)]
        replay_ms = _now_ms() - start_replay
        restore_after_ms = self.restore(runner, snap) if restore_after else 0.0
        return {"num_steps": num_steps, "restore_before_ms": restore_before_ms, "replay_ms": replay_ms, "per_step_ms": replay_ms / num_steps, "restore_after_ms": restore_after_ms, "total_ms": restore_before_ms + replay_ms + restore_after_ms, "steps": records}


def capture_rng_state(input_batch: Any = None) -> dict[str, Any]:
    """Every RNG vLLM's sampling can read: Python, numpy, torch CPU, torch CUDA (every device),
    and the per-request torch.Generators of seeded requests (input_batch.generators)."""
    state: dict[str, Any] = {"py_random": random.getstate()}
    if _np is not None:
        state["numpy"] = _np.random.get_state()
    if _torch is not None:
        state["torch_cpu"] = _torch.get_rng_state().clone()
        if _cuda_in_use():
            state["torch_cuda"] = [s.clone() for s in _torch.cuda.get_rng_state_all()]
        gens = getattr(input_batch, "generators", None)
        if isinstance(gens, dict):
            state["generators"] = {idx: gen.get_state().clone() for idx, gen in gens.items()}
    return state


def restore_rng_state(state: dict[str, Any], input_batch: Any = None) -> None:
    if "py_random" in state:
        random.setstate(state["py_random"])
    if _np is not None and "numpy" in state:
        _np.random.set_state(state["numpy"])
    if _torch is None:
        return
    if "torch_cpu" in state:
        _torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        _torch.cuda.set_rng_state_all(state["torch_cuda"])
    gens = getattr(input_batch, "generators", None)
    if "generators" in state and isinstance(gens, dict):
        if set(state["generators"]) - set(gens):
            raise SnapshotError(f"seeded requests changed rows: {sorted(state['generators'])} -> {sorted(gens)}")
        for idx in set(gens) - set(state["generators"]):   # rows of requests that joined later
            del gens[idx]
        for idx, gen_state in state["generators"].items():
            gens[idx].set_state(gen_state)


class MockedInScoredRunError(RuntimeError):
    """A SpeculativeAcceptanceMocker and a scored run met: the score would be fake."""


_ACTIVE_MOCKERS: set[int] = set()
_SCORED_DEPTH = [0]


def _scored() -> bool:
    return _SCORED_DEPTH[0] > 0 or os.environ.get("FASTRSI_SCORED") == "1"


def assert_unmocked() -> None:
    """Call before reporting any scored number: raises if acceptance is mocked anywhere here."""
    if _ACTIVE_MOCKERS:
        raise MockedInScoredRunError(f"{len(_ACTIVE_MOCKERS)} SpeculativeAcceptanceMocker installed: acceptance is fake")


@contextmanager
def scored_run():
    """Mark a scored check. Refuses to start or finish with a mocker installed, and no mocker
    can be installed inside it."""
    assert_unmocked()
    _SCORED_DEPTH[0] += 1
    try:
        yield
    finally:
        _SCORED_DEPTH[0] -= 1
    assert_unmocked()


SpecAcceptanceSpec = int | float | Sequence[int] | Sequence[Sequence[int]]


class SpeculativeAcceptanceMocker:
    """Deterministic override/mocker for speculative decoding draft acceptance."""
    def __init__(self, accepted_tokens_per_step: SpecAcceptanceSpec = 0, *, schedule_axis: str = "step", align_inputs: bool = True, call_underlying: bool = True, bonus_token_id: int | None = None) -> None:
        if schedule_axis not in ("step", "request") or isinstance(accepted_tokens_per_step, bool) or (isinstance(accepted_tokens_per_step, int) and accepted_tokens_per_step < 0) or (isinstance(accepted_tokens_per_step, float) and not (0.0 <= accepted_tokens_per_step <= 1.0)) or (isinstance(accepted_tokens_per_step, Sequence) and len(accepted_tokens_per_step) == 0):
            raise ValueError(f"Invalid mocker config: accepted_tokens_per_step={accepted_tokens_per_step!r}, schedule_axis={schedule_axis!r}")
        self.accepted_tokens_per_step, self.schedule_axis, self.align_inputs, self.call_underlying, self.bonus_token_id = accepted_tokens_per_step, schedule_axis, bool(align_inputs), bool(call_underlying), bonus_token_id
        self._step_index = 0
        self._bresenham_acc: dict[int, float] = {}
        self._history_accepted: list[list[int]] = []
        self._history_drafted: list[list[int]] = []
        self._installed_target: Any = None
        self._installed_owner: Any = None
        self._orig_forward: Any = None
        self._orig_call: Any = None

    def reset_stats(self) -> None:
        self._step_index = 0
        self._bresenham_acc.clear()
        self._history_accepted.clear()
        self._history_drafted.clear()

    @property
    def total_accepted_draft_tokens(self) -> int:
        return sum(sum(row) for row in self._history_accepted)

    @property
    def total_drafted_tokens(self) -> int:
        return sum(sum(row) for row in self._history_drafted)

    @property
    def realized_alpha(self) -> float:
        tot = self.total_drafted_tokens
        return (self.total_accepted_draft_tokens / tot) if tot > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {"steps": self._step_index, "total_accepted_draft_tokens": self.total_accepted_draft_tokens, "total_drafted_tokens": self.total_drafted_tokens, "realized_alpha": round(self.realized_alpha, 6), "accepted_per_step": [list(row) for row in self._history_accepted], "drafted_per_step": [list(row) for row in self._history_drafted]}

    def resolve_step_acceptance(self, num_draft_tokens_per_req: Sequence[int], *, advance_step: bool = True) -> list[int]:
        spec, step_idx, accepted = self.accepted_tokens_per_step, self._step_index, []
        for req_idx, draft_cnt in enumerate(num_draft_tokens_per_req):
            num_draft = max(0, int(draft_cnt))
            if num_draft == 0:
                accepted.append(0)
                continue
            if isinstance(spec, int):
                raw_k = spec
            elif isinstance(spec, float):
                acc = self._bresenham_acc.get(req_idx, 0.0) + spec * num_draft
                raw_k = int(math.floor(acc + 1e-9))
                if advance_step:
                    self._bresenham_acc[req_idx] = acc - raw_k
            elif isinstance(spec, Sequence):
                first = spec[0]
                row = spec[step_idx % len(spec)] if isinstance(first, Sequence) else None
                raw_k = int(row[req_idx % len(row)]) if row is not None else int(spec[(req_idx if self.schedule_axis == "request" else step_idx) % len(spec)])
            else:
                raw_k = 0
            accepted.append(min(num_draft, max(0, raw_k)))
        if advance_step:
            self._step_index += 1
            self._history_accepted.append(list(accepted))
            self._history_drafted.append([max(0, int(val)) for val in num_draft_tokens_per_req])
        return accepted

    @staticmethod
    def _extract_draft_layout(metadata: Any, input_batch: Any = None, batch_fallback: int = 1) -> tuple[list[int], int, Any, Any]:
        if metadata is not None:
            num_draft = getattr(metadata, "num_draft_tokens", None)
            num_draft_list = [max(0, int(val)) for val in num_draft] if num_draft is not None else [0] * batch_fallback
            return num_draft_list, int(getattr(metadata, "max_spec_len", max(num_draft_list, default=0))), getattr(metadata, "draft_token_ids", None), getattr(metadata, "bonus_token_ids", None)
        if input_batch is not None:
            num_reqs = int(getattr(input_batch, "num_reqs", batch_fallback))
            num_draft_list = [len(row) for row in getattr(input_batch, "spec_token_ids", [])[:num_reqs]]
            return num_draft_list, int(getattr(input_batch, "num_speculative_steps", max(num_draft_list, default=0))), getattr(input_batch, "draft_tokens", None), None
        return [0] * batch_fallback, 0, None, None

    def _align_draft_tokens(self, metadata: Any, logits: Any, num_draft_list: list[int], accepted_list: list[int]) -> Any:
        draft_ids = getattr(metadata, "draft_token_ids", None) if metadata is not None else None
        if draft_ids is None or logits is None:
            return None
        total_draft = sum(num_draft_list)
        saved = _clone_slice(draft_ids, total_draft)
        if _is_torch(logits):
            argmax_list, vocab_size = [int(val) for val in logits.argmax(dim=-1).detach().cpu().tolist()], max(2, int(logits.shape[-1]))
        elif hasattr(logits, "argmax_list"):
            argmax_list, vocab_size = logits.argmax_list(), max(2, getattr(logits, "vocab_size", 64))
        else:
            return saved
        new_draft, draft_off, logit_off = _read_1d_ints(draft_ids, total_draft), 0, 0
        for num_d, acc_k in zip(num_draft_list, accepted_list):
            for pos in range(acc_k):
                new_draft[draft_off + pos] = argmax_list[logit_off + pos]
            if acc_k < num_d:
                new_draft[draft_off + acc_k] = (argmax_list[logit_off + acc_k] + 1) % vocab_size
            draft_off, logit_off = draft_off + num_d, logit_off + num_d + 1
        _write_1d_ints(draft_ids, new_draft)
        return saved

    def build_output_token_ids(self, num_draft_list: list[int], accepted_list: list[int], max_spec_len: int, *, draft_token_ids: Any = None, bonus_token_ids: Any = None, logits: Any = None, prefer_torch: bool = False) -> Any:
        cols, flat_draft, bonus_list = max(1, int(max_spec_len) + 1), _read_1d_ints(draft_token_ids, sum(num_draft_list)), _read_1d_ints(bonus_token_ids, len(num_draft_list))
        rows, draft_off = [], 0
        for req_idx, (num_d, acc_k) in enumerate(zip(num_draft_list, accepted_list)):
            row = [PLACEHOLDER_TOKEN_ID] * cols
            for pos in range(acc_k):
                row[pos] = flat_draft[draft_off + pos] if (draft_off + pos) < len(flat_draft) else (100 + pos)
            row[acc_k] = int(self.bonus_token_id) if self.bonus_token_id is not None else (bonus_list[req_idx] if (acc_k == num_d and req_idx < len(bonus_list)) else (200 + req_idx))
            rows.append(row)
            draft_off += num_d
        if _torch is not None and (prefer_torch or _is_torch(draft_token_ids) or _is_torch(logits)):
            return _torch.tensor(rows, dtype=_torch.int32, device=(logits.device if _is_torch(logits) else "cpu"))
        return _MatrixIntBuffer(rows)

    def run_v1_forward(self, metadata: Any, draft_probs: Any, logits: Any, sampling_metadata: Any = None, *, underlying_forward: Any = None) -> Any:
        num_draft_list, max_spec, draft_ids, bonus_ids = self._extract_draft_layout(metadata)
        accepted_list = self.resolve_step_acceptance(num_draft_list, advance_step=True)
        saved_draft = self._align_draft_tokens(metadata, logits, num_draft_list, accepted_list) if (self.align_inputs and self.call_underlying and underlying_forward is not None) else None
        raw_out = None
        try:
            if self.call_underlying and underlying_forward is not None:
                raw_out = underlying_forward(metadata, draft_probs, logits, sampling_metadata)
        finally:
            if saved_draft is not None and metadata is not None:
                _restore_slice(metadata.draft_token_ids, saved_draft, sum(num_draft_list))
        enforced = self.build_output_token_ids(num_draft_list, accepted_list, max_spec, draft_token_ids=draft_ids, bonus_token_ids=bonus_ids, logits=logits)
        if raw_out is not None and hasattr(raw_out, "sampled_token_ids"):
            raw_out.sampled_token_ids = enforced
            return raw_out
        return SimpleSamplerOutput(sampled_token_ids=enforced)

    def run_v2_call(self, sampler_obj: Any, logits: Any, input_batch: Any, *args: Any, underlying_call: Any = None, **kwargs: Any) -> Any:
        num_draft_list, max_spec, draft_ids, bonus_ids = self._extract_draft_layout(None, input_batch=input_batch)
        accepted_list = self.resolve_step_acceptance(num_draft_list, advance_step=True)
        if self.call_underlying and underlying_call is not None:
            underlying_call(logits, input_batch, *args, **kwargs)
        enforced = self.build_output_token_ids(num_draft_list, accepted_list, max_spec, draft_token_ids=draft_ids, bonus_token_ids=bonus_ids, logits=logits)
        _write_1d_ints(getattr(sampler_obj, "num_sampled", None), [acc_k + 1 for acc_k in accepted_list])
        _write_1d_ints(getattr(sampler_obj, "num_rejected", None), [max(0, num_d - acc_k) for num_d, acc_k in zip(num_draft_list, accepted_list)])
        return enforced

    def install(self, target: Any, *, journal: Any | None = None) -> SpeculativeAcceptanceMocker:
        if _scored():
            raise MockedInScoredRunError("SpeculativeAcceptanceMocker cannot be installed in a scored run (scored_run() or FASTRSI_SCORED=1)")
        if self._installed_target is not None:
            self.uninstall()
        sampler = getattr(target, "rejection_sampler", target)
        if sampler is None:
            raise ValueError("Target has no rejection_sampler to mock")
        self._installed_target, self._installed_owner = sampler, (target if target is not sampler else None)
        if hasattr(sampler, "forward") and callable(sampler.forward):
            orig_fwd = sampler.forward
            self._orig_forward = orig_fwd
            sampler.forward = lambda metadata, draft_probs=None, logits=None, sampling_metadata=None: self.run_v1_forward(metadata, draft_probs, logits, sampling_metadata, underlying_forward=orig_fwd)
        else:
            orig_call = sampler.__call__
            self._orig_call = orig_call
            mock_fn = lambda logits, input_batch, *args, **kwargs: self.run_v2_call(sampler, logits, input_batch, *args, underlying_call=orig_call, **kwargs)
            sampler.forward = mock_fn
            if hasattr(target, "rejection_sampler") and target is not sampler:
                target.rejection_sampler = _CallableSamplerProxy(sampler, mock_fn)
        _ACTIVE_MOCKERS.add(id(self))
        if journal is not None and hasattr(journal, "record_call"):
            journal.record_call(self.uninstall)
        return self

    def uninstall(self) -> None:
        sampler = self._installed_target
        _ACTIVE_MOCKERS.discard(id(self))
        if sampler is None:
            return
        if self._installed_owner is not None and hasattr(self._installed_owner, "rejection_sampler"):
            self._installed_owner.rejection_sampler = sampler
        if self._orig_forward is not None:
            sampler.forward = self._orig_forward
        elif "forward" in getattr(sampler, "__dict__", {}):
            del sampler.__dict__["forward"]
        self._installed_target = self._installed_owner = self._orig_forward = self._orig_call = None

    @contextmanager
    def applied(self, target: Any, *, journal: Any | None = None):
        self.install(target, journal=journal)
        try:
            yield self
        finally:
            self.uninstall()


class _CallableSamplerProxy:
    def __init__(self, underlying: Any, call_fn: Callable[..., Any]) -> None:
        self._underlying, self._call_fn = underlying, call_fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_fn(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


class _MatrixIntBuffer:
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = [list(row) for row in rows]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._rows), len(self._rows[0]) if self._rows else 0)

    def count_non_placeholder(self) -> list[int]:
        return [sum(1 for val in row if val != PLACEHOLDER_TOKEN_ID) for row in self._rows]

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, tuple):
            row_idx, col_sel = item
            row = self._rows[row_idx]
            return _VectorIntView(row[col_sel]) if isinstance(col_sel, slice) else _ScalarIntView(row[col_sel])
        return _VectorIntView(self._rows[item])


class _VectorIntView:
    def __init__(self, values: list[int]) -> None:
        self._values = list(values)

    def tolist(self) -> list[int]:
        return list(self._values)


class _ScalarIntView:
    def __init__(self, value: int) -> None:
        self._value = int(value)

    def item(self) -> int:
        return self._value
