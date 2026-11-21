"""Live parallelism resharder (2D DP<->TP and MoE TP<->EP) for vLLM WeightHolders.

Reconfigures rank-local sharded model weights in memory without re-reading
safetensors checkpoints from disk across any 2D parallelism factor pair on a
fixed ``world_size``:
- ``(tp_src, dp_src) <-> (tp_dst, dp_dst)`` (e.g., ``TP4,DP1 <-> TP2,DP2 <-> TP1,DP4``
  on 4 GPUs, or ``TP2,DP1 <-> TP1,DP2`` on 2 GPUs): handles fused
  ``MergedColumnParallelLinear`` (``gate_up_proj``), ``QKVParallelLinear``
  (``qkv_proj``, including replicated KV heads when ``total_num_kv_heads < tp_size``),
  ``RowParallelLinear``, ``VocabParallelEmbedding``, and ``FusedMoE`` weights.
- ``MoE (moe_tp_src, ep_src) <-> (moe_tp_dst, ep_dst)`` (e.g., ``TP2 <-> EP2`` on
  2 GPUs, or ``(TP4,EP1) <-> (TP2,EP2) <-> (TP1,EP4)`` on 4 GPUs): performs a
  layer-by-layer transpose of ``FusedMoE`` ``w13_weight``
  (``[E/ep, 2*(I/moe_tp), H]``) and ``w2_weight`` (``[E/ep, H, I/moe_tp]``) under
  ``"linear"`` or ``"round_robin"`` expert placement while bounding peak scratch
  memory to a single tensor layer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import time
from typing import Any, Sequence

try:
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]

TILE_REPACKED_QUANT_FORMATS = frozenset({
    "marlin", "gptq_marlin", "awq_marlin", "machete", "compressed-tensors",
})
VALID_SHARD_KINDS = frozenset({
    "replicated", "col_parallel", "merged_col_parallel", "qkv_parallel",
    "row_parallel", "vocab_parallel", "moe_w13", "moe_w2",
})
VALID_EXPERT_PLACEMENTS = frozenset({"linear", "round_robin"})
_SIMPLE_TP_DIM = {"col_parallel": 0, "vocab_parallel": 0, "row_parallel": 1, "moe_w2": 2}


@dataclass(frozen=True)
class ReshardFinding:
    """Structured finding emitted during parallelism validation or resharding."""
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True)
class ShardSpec:
    """Describes how a parameter is partitioned across tensor/expert parallel ranks."""
    kind: str = "replicated"
    partition_dim: int = 0
    sub_sizes: tuple[int, ...] = ()
    total_num_heads: int | None = None
    total_num_kv_heads: int | None = None
    head_size: int | None = None
    num_experts: int | None = None
    expert_placement: str = "linear"
    quant_format: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in VALID_SHARD_KINDS:
            raise ValueError(f"Unsupported ShardSpec kind={self.kind!r}; expected {sorted(VALID_SHARD_KINDS)}")
        if self.expert_placement not in VALID_EXPERT_PLACEMENTS:
            raise ValueError(f"Unsupported expert_placement={self.expert_placement!r}")


@dataclass
class RankWeightHolder:
    """In-memory weight shard dictionary for one worker rank."""
    rank: int
    tp_size: int
    dp_size: int = 1
    ep_size: int = 1
    tensors: dict[str, Any] = field(default_factory=dict)
    quant_format: str | None = None
    moe_tp_size: int | None = None

    @property
    def effective_moe_tp_size(self) -> int:
        if self.moe_tp_size is not None:
            return int(self.moe_tp_size)
        world_size = max(1, int(self.tp_size) * int(self.dp_size))
        return max(1, world_size // max(1, int(self.ep_size)))

    @property
    def dp_rank(self) -> int:
        return self.rank // max(1, int(self.tp_size))

    @property
    def tp_rank(self) -> int:
        return self.rank % max(1, int(self.tp_size))

    @property
    def ep_rank(self) -> int:
        return self.rank // max(1, self.effective_moe_tp_size)

    @property
    def moe_tp_rank(self) -> int:
        return self.rank % max(1, self.effective_moe_tp_size)

    def total_bytes(self) -> int:
        return sum(_tensor_nbytes(val) for val in self.tensors.values())

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: _tensor_shape(val) for name, val in self.tensors.items()}


@dataclass
class ReshardReport:
    """Execution telemetry and structured findings from a reshard operation."""
    source_mode: str
    target_mode: str
    world_size: int
    num_tensors_resharded: int
    total_bytes_transferred: int
    peak_scratch_bytes: int
    elapsed_ms: float
    findings: list[ReshardFinding] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return not any(item.severity == "error" for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.is_ok,
            "source_mode": self.source_mode,
            "target_mode": self.target_mode,
            "world_size": self.world_size,
            "num_tensors_resharded": self.num_tensors_resharded,
            "total_bytes_transferred": self.total_bytes_transferred,
            "peak_scratch_bytes": self.peak_scratch_bytes,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "findings": [item.to_dict() for item in self.findings],
        }


class SimpleTensor:
    """Pure-Python 1D/2D/3D numeric tensor for stdlib-only CPU CI and tests."""

    def __init__(self, data: Any, dtype_bytes: int = 2) -> None:
        self.data = copy.deepcopy(data)
        self.dtype_bytes = int(dtype_bytes)

    @property
    def shape(self) -> tuple[int, ...]:
        dims: list[int] = []
        curr = self.data
        while isinstance(curr, list):
            dims.append(len(curr))
            curr = curr[0] if curr else None
        return tuple(dims)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod * self.dtype_bytes

    def clone(self) -> SimpleTensor:
        return SimpleTensor(self.data, dtype_bytes=self.dtype_bytes)

    def tolist(self) -> Any:
        return copy.deepcopy(self.data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SimpleTensor):
            return self.data == other.data
        return False


def _is_torch(obj: Any) -> bool:
    return _torch is not None and isinstance(obj, _torch.Tensor)


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    if hasattr(tensor, "shape"):
        return tuple(int(dim) for dim in tensor.shape)
    return SimpleTensor(tensor).shape


def _tensor_nbytes(tensor: Any) -> int:
    if _is_torch(tensor):
        return int(tensor.numel() * tensor.element_size())
    if isinstance(tensor, SimpleTensor):
        return tensor.nbytes
    return SimpleTensor(tensor).nbytes


def _clone_tensor(tensor: Any) -> Any:
    if _is_torch(tensor):
        return tensor.detach().clone()
    if isinstance(tensor, SimpleTensor):
        return tensor.clone()
    return copy.deepcopy(tensor)


def _slice_raw(raw: Any, dim: int, start: int, end: int) -> Any:
    if dim == 0:
        return copy.deepcopy(raw[start:end])
    return [_slice_raw(sub, dim - 1, start, end) for sub in raw]


def _concat_raw(raws: Sequence[Any], dim: int) -> Any:
    if dim == 0:
        return [copy.deepcopy(elem) for arr in raws for elem in arr]
    return [_concat_raw([arr[i] for arr in raws], dim - 1) for i in range(len(raws[0]))]


def _slice_dim(tensor: Any, dim: int, start: int, end: int) -> Any:
    if _is_torch(tensor):
        return tensor.narrow(dim, start, end - start).clone()
    if dim not in (0, 1, 2):
        raise ValueError(f"Unsupported slice dim={dim}")
    is_simple = isinstance(tensor, SimpleTensor)
    sliced = _slice_raw(tensor.data if is_simple else tensor, dim, start, end)
    return SimpleTensor(sliced, dtype_bytes=tensor.dtype_bytes) if is_simple else sliced


def _concat_dim(tensors: Sequence[Any], dim: int) -> Any:
    if not tensors:
        raise ValueError("Cannot concatenate empty sequence of tensors")
    first = tensors[0]
    if _is_torch(first):
        return _torch.cat(list(tensors), dim=dim)
    if dim not in (0, 1, 2):
        raise ValueError(f"Unsupported concat dim={dim}")
    is_simple = isinstance(first, SimpleTensor)
    raws = [item.data if isinstance(item, SimpleTensor) else item for item in tensors]
    merged = _concat_raw(raws, dim)
    return SimpleTensor(merged, dtype_bytes=first.dtype_bytes) if is_simple else merged


def _gather_fused(tensors: Sequence[Any], local_subs: Sequence[int], dim: int) -> Any:
    per_rank = [_split_sizes_dim(t, local_subs, dim) for t in tensors]
    return _concat_dim([_concat_dim([r[i] for r in per_rank], dim=dim) for i in range(len(local_subs))], dim=dim)


def _scatter_fused(full_tensor: Any, sub_sizes: Sequence[int], target_tp: int, dim: int) -> list[Any]:
    sliced = [_split_even_dim(sub, target_tp, dim) for sub in _split_sizes_dim(full_tensor, sub_sizes, dim)]
    return [_concat_dim([s[r] for s in sliced], dim=dim) for r in range(target_tp)]


def _split_sizes_dim(tensor: Any, sizes: Sequence[int], dim: int) -> list[Any]:
    chunks: list[Any] = []
    offset = 0
    for chunk_size in sizes:
        chunks.append(_slice_dim(tensor, dim, offset, offset + int(chunk_size)))
        offset += int(chunk_size)
    return chunks


def _split_even_dim(tensor: Any, parts: int, dim: int) -> list[Any]:
    total = _tensor_shape(tensor)[dim]
    if total % parts != 0:
        raise ValueError(f"Dimension {dim} of size {total} is not divisible by {parts}")
    step = total // parts
    return [_slice_dim(tensor, dim, idx * step, (idx + 1) * step) for idx in range(parts)]


def resolve_qkv_sub_sizes(full_dim0: int | None, spec: ShardSpec, tp_size: int = 1) -> tuple[int, int, int]:
    """Return full ``(q_size, k_size, v_size)`` along dim 0 for ``QKVParallelLinear``."""
    if len(spec.sub_sizes) == 3:
        return int(spec.sub_sizes[0]), int(spec.sub_sizes[1]), int(spec.sub_sizes[2])
    if spec.total_num_heads is not None and spec.total_num_kv_heads is not None and spec.head_size is not None:
        q_size = int(spec.total_num_heads) * int(spec.head_size)
        kv_size = int(spec.total_num_kv_heads) * int(spec.head_size)
        return q_size, kv_size, kv_size
    if full_dim0 is not None and full_dim0 % 3 == 0:
        third = full_dim0 // 3
        return third, third, third
    raise ValueError(f"Cannot infer QKV sub_sizes from spec={spec} and full_dim0={full_dim0} (tp_size={tp_size})")


def map_global_expert_to_ep(global_expert_id: int, num_experts: int, ep_size: int, placement: str = "linear") -> tuple[int, int]:
    """Return ``(ep_rank, local_expert_slot)`` for a global expert index."""
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
    experts_per_rank = num_experts // ep_size
    if placement == "linear":
        return global_expert_id // experts_per_rank, global_expert_id % experts_per_rank
    if placement == "round_robin":
        return global_expert_id % ep_size, global_expert_id // ep_size
    raise ValueError(f"Unsupported expert placement={placement!r}")


class ParallelismResharder:
    """Re-shards rank-local ``RankWeightHolder`` dictionaries across 2D parallelism modes."""

    def validate_quant_format(
        self, shard_specs: dict[str, ShardSpec], holders: Sequence[RankWeightHolder] = (),
        *, has_canonical_weights: bool = False,
    ) -> list[ReshardFinding]:
        observed = {s.quant_format.lower() for s in shard_specs.values() if s.quant_format} | {
            h.quant_format.lower() for h in holders if h.quant_format
        }
        blocked = sorted(observed & TILE_REPACKED_QUANT_FORMATS)
        if not blocked or has_canonical_weights:
            return []
        return [ReshardFinding(
            code="reshard_quantized_requires_canonical", severity="error",
            message=(
                f"Tile-repacked quantization format(s) {blocked} permute weight tiles "
                "after loading (`process_weights_after_loading`) and cannot be "
                "directly all-gathered; supply canonical pre-repack weights."
            ),
            details={"blocked_formats": blocked},
        )]

    def estimate_tp_shard_bytes(
        self, rank0_holder: RankWeightHolder, shard_specs: dict[str, ShardSpec], target_tp_size: int = 1,
    ) -> int:
        """Estimate per-GPU weight bytes when sharded across ``target_tp_size`` TP ranks."""
        src_tp, dst_tp, total_bytes = max(1, int(rank0_holder.tp_size)), max(1, int(target_tp_size)), 0
        for name, tensor in rank0_holder.tensors.items():
            spec, nbytes = shard_specs.get(name, ShardSpec("replicated")), _tensor_nbytes(tensor)
            if spec.kind == "replicated":
                total_bytes += nbytes
            elif spec.kind == "qkv_parallel" and spec.total_num_kv_heads is not None:
                q_full, k_full, _ = resolve_qkv_sub_sizes(None, spec, src_tp)
                src_rows = (q_full // src_tp) + 2 * (k_full // min(src_tp, spec.total_num_kv_heads))
                dst_rows = (q_full // dst_tp) + 2 * (k_full // min(dst_tp, spec.total_num_kv_heads))
                total_bytes += int(round(nbytes * (dst_rows / max(1, src_rows))))
            else:
                total_bytes += int(round(nbytes * (src_tp / dst_tp)))
        return total_bytes

    def estimate_dp_replica_bytes(self, rank0_holder: RankWeightHolder, shard_specs: dict[str, ShardSpec]) -> int:
        return self.estimate_tp_shard_bytes(rank0_holder, shard_specs, target_tp_size=1)

    @staticmethod
    def _gather_tp_parameter(name: str, tp_group_holders: Sequence[RankWeightHolder], spec: ShardSpec) -> Any:
        tp_size = len(tp_group_holders)
        tensors = [h.tensors[name] for h in tp_group_holders]
        first = tensors[0]
        if spec.kind == "replicated" or tp_size == 1:
            return _clone_tensor(first)
        if spec.kind in _SIMPLE_TP_DIM:
            return _concat_dim(tensors, dim=_SIMPLE_TP_DIM[spec.kind])
        if spec.kind == "merged_col_parallel":
            shard_dim0 = _tensor_shape(first)[0]
            full_subs = spec.sub_sizes or ((shard_dim0 * tp_size) // 2, (shard_dim0 * tp_size) // 2)
            return _gather_fused(tensors, tuple(size // tp_size for size in full_subs), dim=0)
        if spec.kind == "moe_w13":
            half = _tensor_shape(first)[1] // 2
            return _gather_fused(tensors, (half, half), dim=1)
        if spec.kind == "qkv_parallel":
            num_kv = spec.total_num_kv_heads if spec.total_num_kv_heads is not None else tp_size
            shard_dim0 = _tensor_shape(first)[0]
            if len(spec.sub_sizes) == 3 or (spec.total_num_heads is not None and spec.head_size is not None):
                q_full, k_full, v_full = resolve_qkv_sub_sizes(None, spec, tp_size)
            else:
                q_full = k_full = v_full = (shard_dim0 * tp_size) // 3
            kv_div = min(tp_size, num_kv)
            per_rank_qkv = [_split_sizes_dim(t, (q_full // tp_size, k_full // kv_div, v_full // kv_div), 0) for t in tensors]
            rep_factor = max(1, tp_size // num_kv) if num_kv < tp_size else 1
            return _concat_dim([
                _concat_dim([per_rank_qkv[r][0] for r in range(tp_size)], dim=0),
                _concat_dim([per_rank_qkv[r][1] for r in range(0, tp_size, rep_factor)], dim=0),
                _concat_dim([per_rank_qkv[r][2] for r in range(0, tp_size, rep_factor)], dim=0),
            ], dim=0)
        return _clone_tensor(first)

    @staticmethod
    def _scatter_tp_parameter(full_tensor: Any, target_tp_size: int, spec: ShardSpec) -> tuple[list[Any], ReshardFinding | None]:
        if spec.kind == "replicated" or target_tp_size == 1:
            return [_clone_tensor(full_tensor) for _ in range(target_tp_size)], None
        if spec.kind in _SIMPLE_TP_DIM:
            return _split_even_dim(full_tensor, target_tp_size, _SIMPLE_TP_DIM[spec.kind]), None
        if spec.kind == "merged_col_parallel":
            sub_sizes = spec.sub_sizes or (_tensor_shape(full_tensor)[0] // 2, _tensor_shape(full_tensor)[0] // 2)
            return _scatter_fused(full_tensor, sub_sizes, target_tp_size, dim=0), None
        if spec.kind == "moe_w13":
            half = _tensor_shape(full_tensor)[1] // 2
            return _scatter_fused(full_tensor, (half, half), target_tp_size, dim=1), None
        if spec.kind == "qkv_parallel":
            q_size, k_size, v_size = resolve_qkv_sub_sizes(_tensor_shape(full_tensor)[0], spec, target_tp_size)
            q_full, k_full, v_full = _split_sizes_dim(full_tensor, (q_size, k_size, v_size), 0)
            q_parts = _split_even_dim(q_full, target_tp_size, 0)
            num_kv = spec.total_num_kv_heads if spec.total_num_kv_heads is not None else target_tp_size
            replicated = num_kv < target_tp_size
            if replicated and target_tp_size % num_kv != 0:
                raise ValueError(f"target_tp_size={target_tp_size} must be divisible by total_num_kv_heads={num_kv}")
            rep_factor = (target_tp_size // num_kv) if replicated else 1
            kv_parts = num_kv if replicated else target_tp_size
            k_parts, v_parts = _split_even_dim(k_full, kv_parts, 0), _split_even_dim(v_full, kv_parts, 0)
            shards = [
                _concat_dim([
                    q_parts[r],
                    _clone_tensor(k_parts[r // rep_factor]) if replicated else k_parts[r],
                    _clone_tensor(v_parts[r // rep_factor]) if replicated else v_parts[r],
                ], dim=0)
                for r in range(target_tp_size)
            ]
            finding = ReshardFinding(
                code="reshard_kv_heads_replicated", severity="info",
                message=f"KV heads ({num_kv}) < tp_size ({target_tp_size}); replicated with factor {rep_factor}.",
                details={"total_num_kv_heads": num_kv, "tp_size": target_tp_size, "replication_factor": rep_factor},
            ) if replicated else None
            return shards, finding
        return [_clone_tensor(full_tensor) for _ in range(target_tp_size)], None

    def reshard_tp_dp(
        self, source_weights: dict[str, Any] | Sequence[RankWeightHolder], shard_specs: dict[str, ShardSpec],
        *, target_tp_size: int, target_dp_size: int | None = None,
        vram_budget_bytes: int | None = None, canonical_weights: dict[str, Any] | None = None,
    ) -> tuple[list[RankWeightHolder], ReshardReport]:
        """Re-shard between ANY ``(tp_src, dp_src)`` and ``(tp_dst, dp_dst)`` mesh on ``world_size``."""
        start_time = time.perf_counter()
        if target_tp_size < 1:
            raise ValueError(f"target_tp_size must be >= 1, got {target_tp_size}")
        if isinstance(source_weights, dict):
            eff_dp_dst = int(target_dp_size) if target_dp_size is not None else 1
            world_size, src_tp_size = target_tp_size * eff_dp_dst, 1
            src_dp_size = world_size
            holder_list = [RankWeightHolder(rank=0, tp_size=1, dp_size=src_dp_size, ep_size=1, tensors=source_weights)]
            has_canonical, quant_fmt = True, None
        else:
            holder_list = list(source_weights)
            if not holder_list:
                raise ValueError("source_weights sequence cannot be empty")
            world_size = len(holder_list)
            src_tp_size = max(1, int(holder_list[0].tp_size))
            src_dp_size = max(1, world_size // src_tp_size)
            if src_tp_size * src_dp_size != world_size or len(holder_list) < src_tp_size:
                raise ValueError(f"Invalid source mesh: len(holders)={world_size} incompatible with src_tp_size={src_tp_size}")
            eff_dp_dst = int(target_dp_size) if target_dp_size is not None else (world_size // target_tp_size)
            if target_tp_size * eff_dp_dst != world_size:
                raise ValueError(f"target_tp_size ({target_tp_size}) * target_dp_size ({eff_dp_dst}) != world_size ({world_size})")
            has_canonical, quant_fmt = canonical_weights is not None, holder_list[0].quant_format

        findings = self.validate_quant_format(shard_specs, holder_list, has_canonical_weights=has_canonical)
        # The TP gather below assumes MoE tensors are sharded like the dense ones (moe_tp == tp,
        # ep == 1). Expert-sharded holders would be concatenated across experts or lose all
        # experts but rank 0's, so refuse them.
        if not isinstance(source_weights, dict):
            src_ep_size, src_moe_tp_size = max(1, int(holder_list[0].ep_size)), holder_list[0].effective_moe_tp_size
            moe_names = [
                n for n in holder_list[0].tensors
                if shard_specs.get(n, ShardSpec("replicated")).kind in ("moe_w13", "moe_w2")
                and not (canonical_weights is not None and n in canonical_weights)
            ]
            if moe_names and (src_ep_size != 1 or src_moe_tp_size != src_tp_size):
                findings.append(ReshardFinding(
                    code="reshard_moe_mesh_unsupported", severity="error",
                    message=(f"reshard_tp_dp cannot move MoE tensors held as moe_tp={src_moe_tp_size}, "
                             f"ep={src_ep_size} (tp={src_tp_size}); transpose them with "
                             f"reshard_moe_tp_ep to moe_tp={src_tp_size}, ep=1 first."),
                    details={"source_tp_size": src_tp_size, "source_ep_size": src_ep_size,
                             "source_moe_tp_size": src_moe_tp_size, "moe_tensors": moe_names},
                ))
        required_bytes = self.estimate_tp_shard_bytes(holder_list[0], shard_specs, target_tp_size=target_tp_size)
        if vram_budget_bytes is not None and required_bytes > vram_budget_bytes:
            findings.append(ReshardFinding(
                code="reshard_dp_vram_exceeded", severity="error",
                message=(f"Reconfiguring TP{src_tp_size}_DP{src_dp_size}->TP{target_tp_size}_DP{eff_dp_dst} "
                         f"requires {required_bytes} bytes per GPU, exceeding vram_budget_bytes={vram_budget_bytes}."),
                details={"required_bytes": required_bytes, "vram_budget_bytes": vram_budget_bytes,
                         "source_tp_size": src_tp_size, "target_tp_size": target_tp_size, "target_dp_size": eff_dp_dst},
            ))

        out_holders = [
            RankWeightHolder(rank=r, tp_size=target_tp_size, dp_size=eff_dp_dst, ep_size=1,
                             quant_format=quant_fmt, moe_tp_size=target_tp_size)
            for r in range(world_size)
        ]
        source_mode = f"TP{src_tp_size}_DP{src_dp_size}" if src_tp_size > 1 else f"DP{src_dp_size}_TP1"
        target_mode = f"TP{target_tp_size}_DP{eff_dp_dst}" if target_tp_size > 1 else f"DP{eff_dp_dst}_TP1"
        if any(item.severity == "error" for item in findings):
            return out_holders, ReshardReport(
                source_mode, target_mode, world_size, 0, 0, 0, (time.perf_counter() - start_time) * 1000.0, findings
            )

        total_bytes = peak_scratch = 0
        replicated_kv_reported = False
        src_tp_group = holder_list[:src_tp_size]
        param_names = list(src_tp_group[0].tensors.keys())
        for name in param_names:
            spec = shard_specs.get(name, ShardSpec("replicated"))
            if canonical_weights is not None and name in canonical_weights:
                full_tensor = _clone_tensor(canonical_weights[name])
            elif isinstance(source_weights, dict):
                full_tensor = source_weights[name]
            else:
                full_tensor = self._gather_tp_parameter(name, src_tp_group, spec)
            layer_bytes = _tensor_nbytes(full_tensor)
            peak_scratch = max(peak_scratch, layer_bytes)
            total_bytes += layer_bytes * eff_dp_dst
            tp_shards, kv_finding = self._scatter_tp_parameter(full_tensor, target_tp_size, spec)
            if kv_finding is not None and not replicated_kv_reported:
                replicated_kv_reported = True
                findings.append(kv_finding)
            for global_rank in range(world_size):
                tp_rank = global_rank % target_tp_size
                out_holders[global_rank].tensors[name] = (
                    tp_shards[tp_rank] if global_rank < target_tp_size else _clone_tensor(tp_shards[tp_rank])
                )
        return out_holders, ReshardReport(
            source_mode, target_mode, world_size, len(param_names), total_bytes, peak_scratch,
            (time.perf_counter() - start_time) * 1000.0, findings,
        )

    def reshard_dp_to_tp(
        self, full_weights: dict[str, Any] | Sequence[RankWeightHolder], target_tp_size: int, shard_specs: dict[str, ShardSpec],
    ) -> tuple[list[RankWeightHolder], ReshardReport]:
        """Slice full ``DP (TP=1)`` weights into ``target_tp_size`` rank shards."""
        return self.reshard_tp_dp(full_weights, shard_specs, target_tp_size=target_tp_size, target_dp_size=1)

    def reshard_tp_to_dp(
        self, tp_holders: Sequence[RankWeightHolder], shard_specs: dict[str, ShardSpec],
        *, vram_budget_bytes: int | None = None, canonical_weights: dict[str, Any] | None = None,
    ) -> tuple[list[RankWeightHolder], ReshardReport]:
        """Reconstruct full ``DP (TP=1)`` replicas from ``TP`` rank shards."""
        if not tp_holders:
            raise ValueError("tp_holders cannot be empty")
        return self.reshard_tp_dp(
            tp_holders, shard_specs, target_tp_size=1, target_dp_size=len(tp_holders),
            vram_budget_bytes=vram_budget_bytes, canonical_weights=canonical_weights,
        )

    def reshard_moe_tp_ep(
        self, holders: Sequence[RankWeightHolder], shard_specs: dict[str, ShardSpec],
        *, target_moe_tp_size: int, target_ep_size: int | None = None, placement: str | None = None,
    ) -> tuple[list[RankWeightHolder], ReshardReport]:
        """Layer-by-layer MoE transpose between ANY ``(moe_tp_src, ep_src)`` and ``(moe_tp_dst, ep_dst)``."""
        start_time = time.perf_counter()
        if not holders:
            raise ValueError("holders cannot be empty")
        world_size = len(holders)
        if target_moe_tp_size < 1:
            raise ValueError(f"target_moe_tp_size must be >= 1, got {target_moe_tp_size}")
        eff_ep_dst = int(target_ep_size) if target_ep_size is not None else (world_size // target_moe_tp_size)
        if eff_ep_dst < 1 or target_moe_tp_size * eff_ep_dst != world_size:
            raise ValueError(f"target_moe_tp_size ({target_moe_tp_size}) * target_ep_size ({eff_ep_dst}) != world_size ({world_size})")
        src_ep_size, src_moe_tp_size = max(1, int(holders[0].ep_size)), holders[0].effective_moe_tp_size
        if src_moe_tp_size * src_ep_size != world_size:
            raise ValueError(f"Invalid source MoE mesh: moe_tp_size={src_moe_tp_size}, ep_size={src_ep_size}, world_size={world_size}")

        out_holders = [
            RankWeightHolder(rank=r, tp_size=holders[r].tp_size, dp_size=holders[r].dp_size,
                             ep_size=eff_ep_dst, quant_format=holders[r].quant_format, moe_tp_size=target_moe_tp_size)
            for r in range(world_size)
        ]
        total_bytes = peak_scratch = moe_count = 0
        for name in holders[0].tensors:
            spec = shard_specs.get(name, ShardSpec("replicated"))
            eff_placement = placement or spec.expert_placement
            if spec.kind not in ("moe_w13", "moe_w2"):
                for r in range(world_size):
                    out_holders[r].tensors[name] = _clone_tensor(holders[r].tensors[name])
                continue
            moe_count += 1
            num_experts = spec.num_experts or (_tensor_shape(holders[0].tensors[name])[0] * src_ep_size)
            if num_experts % src_ep_size != 0 or num_experts % eff_ep_dst != 0:
                raise ValueError(f"num_experts={num_experts} must be divisible by both src_ep_size={src_ep_size} and target_ep_size={eff_ep_dst}")
            layer_scratch = sum(_tensor_nbytes(h.tensors[name]) for h in holders)
            peak_scratch, total_bytes = max(peak_scratch, layer_scratch), total_bytes + layer_scratch
            dst_rank_slots: list[list[Any]] = [[None] * (num_experts // eff_ep_dst) for _ in range(world_size)]
            for global_exp in range(num_experts):
                src_ep_rank, src_slot = map_global_expert_to_ep(global_exp, num_experts, src_ep_size, eff_placement)
                dst_ep_rank, dst_slot = map_global_expert_to_ep(global_exp, num_experts, eff_ep_dst, eff_placement)
                exp_slices = [
                    _slice_dim(holders[src_ep_rank * src_moe_tp_size + tp_idx].tensors[name], 0, src_slot, src_slot + 1)
                    for tp_idx in range(src_moe_tp_size)
                ]
                if spec.kind == "moe_w13":
                    half = _tensor_shape(holders[0].tensors[name])[1] // 2
                    full_exp = _gather_fused(exp_slices, (half, half), dim=1)
                    full_half = _tensor_shape(full_exp)[1] // 2
                    dst_parts = _scatter_fused(full_exp, (full_half, full_half), target_moe_tp_size, dim=1)
                else:
                    dst_parts = _split_even_dim(_concat_dim(exp_slices, dim=2), target_moe_tp_size, 2)
                for dst_tp_rank, part in enumerate(dst_parts):
                    dst_rank_slots[dst_ep_rank * target_moe_tp_size + dst_tp_rank][dst_slot] = part
            for r in range(world_size):
                out_holders[r].tensors[name] = _concat_dim(dst_rank_slots[r], dim=0)

        return out_holders, ReshardReport(
            f"MoE_TP{src_moe_tp_size}_EP{src_ep_size}", f"MoE_TP{target_moe_tp_size}_EP{eff_ep_dst}",
            world_size, moe_count, total_bytes, peak_scratch, (time.perf_counter() - start_time) * 1000.0,
        )

    def reshard_moe_tp_to_ep(
        self, tp_holders: Sequence[RankWeightHolder], shard_specs: dict[str, ShardSpec], *, placement: str | None = None,
    ) -> tuple[list[RankWeightHolder], ReshardReport]:
        """Layer-by-layer MoE transpose from ``TP (EP=1)`` to ``EP (moe_tp=1)``."""
        if not tp_holders:
            raise ValueError("tp_holders cannot be empty")
        return self.reshard_moe_tp_ep(tp_holders, shard_specs, target_moe_tp_size=1, target_ep_size=len(tp_holders), placement=placement)

    def reshard_moe_ep_to_tp(
        self, ep_holders: Sequence[RankWeightHolder], shard_specs: dict[str, ShardSpec], *, placement: str | None = None,
    ) -> tuple[list[RankWeightHolder], ReshardReport]:
        """Layer-by-layer MoE transpose from ``EP (moe_tp=1)`` back to ``TP (EP=1)``."""
        if not ep_holders:
            raise ValueError("ep_holders cannot be empty")
        return self.reshard_moe_tp_ep(ep_holders, shard_specs, target_moe_tp_size=len(ep_holders), target_ep_size=1, placement=placement)
