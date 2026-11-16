"""CPU unit tests for ParallelismResharder (2D DP<->TP and MoE TP<->EP).

Runs in stdlib + pytest environments (GitHub Actions CPU runner) as well as
environments with PyTorch installed.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reshard import (  # noqa: E402
    ParallelismResharder,
    RankWeightHolder,
    ShardSpec,
    SimpleTensor,
    map_global_expert_to_ep,
)


def _make_2d_tensor(rows: int, cols: int, base: int = 1) -> SimpleTensor:
    matrix = [
        [float(base + row_idx * 100 + col_idx) for col_idx in range(cols)]
        for row_idx in range(rows)
    ]
    return SimpleTensor(matrix, dtype_bytes=2)


def _make_3d_tensor(planes: int, rows: int, cols: int, base: int = 1) -> SimpleTensor:
    volume = [
        [
            [float(base + plane_idx * 1000 + row_idx * 10 + col_idx) for col_idx in range(cols)]
            for row_idx in range(rows)
        ]
        for plane_idx in range(planes)
    ]
    return SimpleTensor(volume, dtype_bytes=2)


def _matmul_vec(weight_2d: list[list[float]], vec_1d: list[float]) -> list[float]:
    return [
        sum(weight_row[col_idx] * vec_1d[col_idx] for col_idx in range(len(vec_1d)))
        for weight_row in weight_2d
    ]


def _silu_and_mul(gate_up_vec: list[float]) -> list[float]:
    half = len(gate_up_vec) // 2
    gate_vec = gate_up_vec[:half]
    up_vec = gate_up_vec[half:]
    return [
        (gate_val / (1.0 + math.exp(-gate_val * 0.01))) * up_val
        for gate_val, up_val in zip(gate_vec, up_vec)
    ]


def _eval_tp_mlp_group(holders_group: list[RankWeightHolder], input_vec: list[float]) -> list[float]:
    hidden_size = len(input_vec)
    partial_outputs = []
    for holder in holders_group:
        local_gate_up = _matmul_vec(holder.tensors["model.layers.0.mlp.gate_up_proj.weight"].data, input_vec)
        local_act = _silu_and_mul(local_gate_up)
        local_out = _matmul_vec(holder.tensors["model.layers.0.mlp.down_proj.weight"].data, local_act)
        partial_outputs.append(local_out)
    return [
        sum(partial_outputs[rank_idx][dim_idx] for rank_idx in range(len(holders_group)))
        for dim_idx in range(hidden_size)
    ]


def _eval_moe_expert(
    holders: list[RankWeightHolder],
    global_exp: int,
    num_experts: int,
    placement: str,
    input_vec: list[float],
) -> list[float]:
    ep_size = holders[0].ep_size
    moe_tp_size = holders[0].effective_moe_tp_size
    ep_rank, local_slot = map_global_expert_to_ep(global_exp, num_experts, ep_size, placement)
    group_ranks = [ep_rank * moe_tp_size + tp_idx for tp_idx in range(moe_tp_size)]
    hidden_size = len(input_vec)
    partial_outputs = []
    for global_rank in group_ranks:
        w13_plane = holders[global_rank].tensors["layers.0.moe.w13_weight"].data[local_slot]
        w2_plane = holders[global_rank].tensors["layers.0.moe.w2_weight"].data[local_slot]
        local_w13 = _matmul_vec(w13_plane, input_vec)
        local_act = _silu_and_mul(local_w13)
        local_out = _matmul_vec(w2_plane, local_act)
        partial_outputs.append(local_out)
    return [
        sum(partial_outputs[tp_idx][dim_idx] for tp_idx in range(moe_tp_size))
        for dim_idx in range(hidden_size)
    ]


def test_dp4_to_tp4_and_back_merged_column_and_row_parallel_equivalence() -> None:
    resharder = ParallelismResharder()
    hidden_size = 8
    inter_size = 12
    vocab_size = 16
    tp_size = 4

    full_weights = {
        "model.embed_tokens.weight": _make_2d_tensor(vocab_size, hidden_size, base=10),
        "model.layers.0.input_layernorm.weight": SimpleTensor([1.0] * hidden_size, dtype_bytes=2),
        "model.layers.0.mlp.gate_up_proj.weight": _make_2d_tensor(2 * inter_size, hidden_size, base=1),
        "model.layers.0.mlp.down_proj.weight": _make_2d_tensor(hidden_size, inter_size, base=5),
    }
    shard_specs = {
        "model.embed_tokens.weight": ShardSpec(kind="vocab_parallel"),
        "model.layers.0.input_layernorm.weight": ShardSpec(kind="replicated"),
        "model.layers.0.mlp.gate_up_proj.weight": ShardSpec(
            kind="merged_col_parallel", sub_sizes=(inter_size, inter_size)
        ),
        "model.layers.0.mlp.down_proj.weight": ShardSpec(kind="row_parallel", partition_dim=1),
    }

    tp_holders, report_to_tp = resharder.reshard_dp_to_tp(full_weights, tp_size, shard_specs)
    assert report_to_tp.is_ok
    assert len(tp_holders) == tp_size
    for rank_idx, holder in enumerate(tp_holders):
        assert holder.rank == rank_idx
        assert holder.tp_size == tp_size
        assert holder.tensor_shapes() == {
            "model.embed_tokens.weight": (vocab_size // tp_size, hidden_size),
            "model.layers.0.input_layernorm.weight": (hidden_size,),
            "model.layers.0.mlp.gate_up_proj.weight": ((2 * inter_size) // tp_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": (hidden_size, inter_size // tp_size),
        }

    input_vec = [0.1 * (idx + 1) for idx in range(hidden_size)]
    full_gate_up = _matmul_vec(full_weights["model.layers.0.mlp.gate_up_proj.weight"].data, input_vec)
    full_act = _silu_and_mul(full_gate_up)
    full_out = _matmul_vec(full_weights["model.layers.0.mlp.down_proj.weight"].data, full_act)

    tp_allreduced_out = _eval_tp_mlp_group(tp_holders, input_vec)
    for expected_val, actual_val in zip(full_out, tp_allreduced_out):
        assert math.isclose(expected_val, actual_val, rel_tol=1e-9, abs_tol=1e-9)

    naive_concat_rows: list[list[float]] = []
    for holder in tp_holders:
        naive_concat_rows.extend(holder.tensors["model.layers.0.mlp.gate_up_proj.weight"].data)
    assert naive_concat_rows != full_weights["model.layers.0.mlp.gate_up_proj.weight"].data

    dp_holders, report_to_dp = resharder.reshard_tp_to_dp(tp_holders, shard_specs)
    assert report_to_dp.is_ok
    assert len(dp_holders) == tp_size
    for holder in dp_holders:
        assert holder.tp_size == 1
        assert holder.dp_size == tp_size
        for param_name, orig_tensor in full_weights.items():
            assert holder.tensors[param_name] == orig_tensor


def test_w2_tp2_dp2_and_moe_tp2_ep2_forward_equivalence_and_round_trip() -> None:
    """Explicitly verifies W=2 transitions: TP2 <-> DP2 and MoE TP2 <-> EP2."""
    resharder = ParallelismResharder()
    world_size = 2
    hidden_size = 4
    inter_size = 6
    num_experts = 4

    full_weights = {
        "model.layers.0.mlp.gate_up_proj.weight": _make_2d_tensor(2 * inter_size, hidden_size, base=2),
        "model.layers.0.mlp.down_proj.weight": _make_2d_tensor(hidden_size, inter_size, base=7),
        "layers.0.moe.w13_weight": _make_3d_tensor(num_experts, 2 * inter_size, hidden_size, base=15),
        "layers.0.moe.w2_weight": _make_3d_tensor(num_experts, hidden_size, inter_size, base=25),
    }
    shard_specs = {
        "model.layers.0.mlp.gate_up_proj.weight": ShardSpec(
            kind="merged_col_parallel", sub_sizes=(inter_size, inter_size)
        ),
        "model.layers.0.mlp.down_proj.weight": ShardSpec(kind="row_parallel", partition_dim=1),
        "layers.0.moe.w13_weight": ShardSpec(kind="moe_w13", num_experts=num_experts, expert_placement="round_robin"),
        "layers.0.moe.w2_weight": ShardSpec(kind="moe_w2", num_experts=num_experts, expert_placement="round_robin"),
    }

    # 1. DP2 (TP1) -> TP2 (DP1) -> DP2 (TP1)
    tp2_holders, rep_tp2 = resharder.reshard_tp_dp(
        full_weights, shard_specs, target_tp_size=2, target_dp_size=1
    )
    assert rep_tp2.is_ok
    assert len(tp2_holders) == world_size
    dp2_holders, rep_dp2 = resharder.reshard_tp_dp(
        tp2_holders, shard_specs, target_tp_size=1, target_dp_size=2
    )
    assert rep_dp2.is_ok
    input_vec = [0.2, -0.1, 0.3, 0.4]
    tp2_mlp_out = _eval_tp_mlp_group(tp2_holders, input_vec)
    dp2_mlp_out = _eval_tp_mlp_group([dp2_holders[0]], input_vec)
    for val_tp2, val_dp2 in zip(tp2_mlp_out, dp2_mlp_out):
        assert math.isclose(val_tp2, val_dp2, rel_tol=1e-9, abs_tol=1e-9)

    # 2. MoE TP2 (EP1) <-> EP2 (moe_tp=1)
    ep2_holders, rep_ep2 = resharder.reshard_moe_tp_ep(
        tp2_holders, shard_specs, target_moe_tp_size=1, target_ep_size=2
    )
    assert rep_ep2.is_ok
    assert ep2_holders[0].tensors["layers.0.moe.w13_weight"].shape == (
        num_experts // 2,
        2 * inter_size,
        hidden_size,
    )
    for exp_idx in range(num_experts):
        out_from_tp2 = _eval_moe_expert(tp2_holders, exp_idx, num_experts, "round_robin", input_vec)
        out_from_ep2 = _eval_moe_expert(ep2_holders, exp_idx, num_experts, "round_robin", input_vec)
        for val_a, val_b in zip(out_from_tp2, out_from_ep2):
            assert math.isclose(val_a, val_b, rel_tol=1e-9, abs_tol=1e-9)

    tp2_back, rep_back = resharder.reshard_moe_tp_ep(
        ep2_holders, shard_specs, target_moe_tp_size=2, target_ep_size=1
    )
    assert rep_back.is_ok
    for rank_idx in range(world_size):
        assert tp2_back[rank_idx].tensors == tp2_holders[rank_idx].tensors


def test_w4_hybrid_tp4_to_tp2_dp2_to_dp4_and_back_equivalence() -> None:
    """Verifies W=4 2D mesh transitions: TP4,DP1 <-> TP2,DP2 <-> TP1,DP4."""
    resharder = ParallelismResharder()
    hidden_size = 8
    inter_size = 12
    head_size = 2

    full_weights = {
        "qkv_proj": _make_2d_tensor((8 + 2 + 2) * head_size, hidden_size, base=50),
        "model.layers.0.mlp.gate_up_proj.weight": _make_2d_tensor(2 * inter_size, hidden_size, base=3),
        "model.layers.0.mlp.down_proj.weight": _make_2d_tensor(hidden_size, inter_size, base=9),
    }
    shard_specs = {
        "qkv_proj": ShardSpec(
            kind="qkv_parallel", total_num_heads=8, total_num_kv_heads=2, head_size=head_size
        ),
        "model.layers.0.mlp.gate_up_proj.weight": ShardSpec(
            kind="merged_col_parallel", sub_sizes=(inter_size, inter_size)
        ),
        "model.layers.0.mlp.down_proj.weight": ShardSpec(kind="row_parallel", partition_dim=1),
    }

    # Start at TP4, DP1
    tp4_holders, rep_tp4 = resharder.reshard_tp_dp(
        full_weights, shard_specs, target_tp_size=4, target_dp_size=1
    )
    assert rep_tp4.is_ok
    assert any(item.code == "reshard_kv_heads_replicated" for item in rep_tp4.findings)

    # Transition TP4, DP1 -> TP2, DP2
    tp2_dp2_holders, rep_tp2_dp2 = resharder.reshard_tp_dp(
        tp4_holders, shard_specs, target_tp_size=2, target_dp_size=2
    )
    assert rep_tp2_dp2.is_ok
    assert rep_tp2_dp2.source_mode == "TP4_DP1"
    assert rep_tp2_dp2.target_mode == "TP2_DP2"
    # At TP2 with total_num_kv_heads=2, KV heads are no longer replicated
    assert not any(item.code == "reshard_kv_heads_replicated" for item in rep_tp2_dp2.findings)
    # DP replica 0 (ranks 0, 1) and DP replica 1 (ranks 2, 3) hold identical TP2 shards
    assert tp2_dp2_holders[0].dp_rank == 0 and tp2_dp2_holders[0].tp_rank == 0
    assert tp2_dp2_holders[1].dp_rank == 0 and tp2_dp2_holders[1].tp_rank == 1
    assert tp2_dp2_holders[2].dp_rank == 1 and tp2_dp2_holders[2].tp_rank == 0
    assert tp2_dp2_holders[3].dp_rank == 1 and tp2_dp2_holders[3].tp_rank == 1
    assert tp2_dp2_holders[0].tensors == tp2_dp2_holders[2].tensors
    assert tp2_dp2_holders[1].tensors == tp2_dp2_holders[3].tensors

    # Transition TP2, DP2 -> TP1, DP4
    dp4_holders, rep_dp4 = resharder.reshard_tp_dp(
        tp2_dp2_holders, shard_specs, target_tp_size=1, target_dp_size=4
    )
    assert rep_dp4.is_ok
    for holder in dp4_holders:
        assert holder.tensors == full_weights

    # Verify MLP forward equivalence across TP4, TP2_DP2 (both replicas), and DP4
    input_vec = [0.05 * (idx + 1) for idx in range(hidden_size)]
    out_tp4 = _eval_tp_mlp_group(tp4_holders, input_vec)
    out_tp2_rep0 = _eval_tp_mlp_group(tp2_dp2_holders[0:2], input_vec)
    out_tp2_rep1 = _eval_tp_mlp_group(tp2_dp2_holders[2:4], input_vec)
    out_dp4 = _eval_tp_mlp_group([dp4_holders[0]], input_vec)
    for idx in range(hidden_size):
        assert math.isclose(out_tp4[idx], out_tp2_rep0[idx], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(out_tp4[idx], out_tp2_rep1[idx], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(out_tp4[idx], out_dp4[idx], rel_tol=1e-9, abs_tol=1e-9)

    # Transition TP1, DP4 -> TP2, DP2 -> TP4, DP1 and verify bit-exact match
    back_tp2_dp2, _ = resharder.reshard_tp_dp(dp4_holders, shard_specs, target_tp_size=2, target_dp_size=2)
    back_tp4, _ = resharder.reshard_tp_dp(back_tp2_dp2, shard_specs, target_tp_size=4, target_dp_size=1)
    for rank_idx in range(4):
        assert back_tp4[rank_idx].tensors == tp4_holders[rank_idx].tensors


@pytest.mark.parametrize("placement", ["linear", "round_robin"])
def test_w4_hybrid_moe_tp4_ep1_to_tp2_ep2_to_tp1_ep4_and_back(placement: str) -> None:
    """Verifies W=4 MoE 2D transitions: (TP4,EP1) <-> (TP2,EP2) <-> (TP1,EP4)."""
    resharder = ParallelismResharder()
    world_size = 4
    num_experts = 8
    inter_size = 8
    hidden_size = 4

    full_weights = {
        "layers.0.attn_norm": SimpleTensor([1.0] * hidden_size),
        "layers.0.moe.w13_weight": _make_3d_tensor(num_experts, 2 * inter_size, hidden_size, base=10),
        "layers.0.moe.w2_weight": _make_3d_tensor(num_experts, hidden_size, inter_size, base=20),
        "layers.1.moe.w13_weight": _make_3d_tensor(num_experts, 2 * inter_size, hidden_size, base=30),
        "layers.1.moe.w2_weight": _make_3d_tensor(num_experts, hidden_size, inter_size, base=40),
    }
    shard_specs = {
        "layers.0.attn_norm": ShardSpec(kind="replicated"),
        "layers.0.moe.w13_weight": ShardSpec(kind="moe_w13", num_experts=num_experts, expert_placement=placement),
        "layers.0.moe.w2_weight": ShardSpec(kind="moe_w2", num_experts=num_experts, expert_placement=placement),
        "layers.1.moe.w13_weight": ShardSpec(kind="moe_w13", num_experts=num_experts, expert_placement=placement),
        "layers.1.moe.w2_weight": ShardSpec(kind="moe_w2", num_experts=num_experts, expert_placement=placement),
    }

    # 1. Start at (moe_tp=4, ep=1)
    tp4_ep1, _ = resharder.reshard_dp_to_tp(full_weights, world_size, shard_specs)
    for holder in tp4_ep1:
        assert holder.tensors["layers.0.moe.w13_weight"].shape == (
            num_experts,
            (2 * inter_size) // 4,
            hidden_size,
        )

    # 2. Transition (moe_tp=4, ep=1) -> (moe_tp=2, ep=2)
    tp2_ep2, rep_2x2 = resharder.reshard_moe_tp_ep(
        tp4_ep1, shard_specs, target_moe_tp_size=2, target_ep_size=2, placement=placement
    )
    assert rep_2x2.is_ok
    assert rep_2x2.source_mode == "MoE_TP4_EP1"
    assert rep_2x2.target_mode == "MoE_TP2_EP2"
    for holder in tp2_ep2:
        assert holder.ep_size == 2
        assert holder.effective_moe_tp_size == 2
        assert holder.tensors["layers.0.moe.w13_weight"].shape == (
            num_experts // 2,
            (2 * inter_size) // 2,
            hidden_size,
        )
        assert holder.tensors["layers.0.moe.w2_weight"].shape == (
            num_experts // 2,
            hidden_size,
            inter_size // 2,
        )

    # 3. Transition (moe_tp=2, ep=2) -> (moe_tp=1, ep=4)
    tp1_ep4, rep_1x4 = resharder.reshard_moe_tp_ep(
        tp2_ep2, shard_specs, target_moe_tp_size=1, target_ep_size=4, placement=placement
    )
    assert rep_1x4.is_ok
    assert rep_1x4.source_mode == "MoE_TP2_EP2"
    assert rep_1x4.target_mode == "MoE_TP1_EP4"
    single_w13_layer_bytes = full_weights["layers.0.moe.w13_weight"].nbytes
    assert rep_2x2.peak_scratch_bytes == single_w13_layer_bytes
    assert rep_1x4.peak_scratch_bytes == single_w13_layer_bytes

    # Verify every expert's forward pass agrees across (4,1), (2,2), and (1,4)
    input_vec = [0.1, -0.2, 0.3, -0.1]
    for global_exp in range(num_experts):
        out_4x1 = _eval_moe_expert(tp4_ep1, global_exp, num_experts, placement, input_vec)
        out_2x2 = _eval_moe_expert(tp2_ep2, global_exp, num_experts, placement, input_vec)
        out_1x4 = _eval_moe_expert(tp1_ep4, global_exp, num_experts, placement, input_vec)
        for dim_idx in range(hidden_size):
            assert math.isclose(out_4x1[dim_idx], out_2x2[dim_idx], rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(out_4x1[dim_idx], out_1x4[dim_idx], rel_tol=1e-9, abs_tol=1e-9)

    # 4. Transition back (1,4) -> (2,2) -> (4,1) and also direct (1,4) -> (4,1)
    back_2x2, _ = resharder.reshard_moe_tp_ep(
        tp1_ep4, shard_specs, target_moe_tp_size=2, target_ep_size=2, placement=placement
    )
    back_4x1, _ = resharder.reshard_moe_tp_ep(
        back_2x2, shard_specs, target_moe_tp_size=4, target_ep_size=1, placement=placement
    )
    direct_4x1, _ = resharder.reshard_moe_ep_to_tp(tp1_ep4, shard_specs, placement=placement)
    for rank_idx in range(world_size):
        assert back_2x2[rank_idx].tensors == tp2_ep2[rank_idx].tensors
        assert back_4x1[rank_idx].tensors == tp4_ep1[rank_idx].tensors
        assert direct_4x1[rank_idx].tensors == tp4_ep1[rank_idx].tensors


def test_qkv_parallel_standard_and_replicated_kv_heads_round_trip() -> None:
    resharder = ParallelismResharder()
    tp_size = 4
    hidden_size = 8
    head_size = 2

    qkv_std_full = _make_2d_tensor((8 + 4 + 4) * head_size, hidden_size, base=100)
    specs_std = {
        "qkv_proj": ShardSpec(
            kind="qkv_parallel",
            total_num_heads=8,
            total_num_kv_heads=4,
            head_size=head_size,
        )
    }
    tp_std, rep_std = resharder.reshard_dp_to_tp({"qkv_proj": qkv_std_full}, tp_size, specs_std)
    assert rep_std.is_ok
    assert not rep_std.findings
    for holder in tp_std:
        assert holder.tensors["qkv_proj"].shape == (8, hidden_size)
    dp_std, _ = resharder.reshard_tp_to_dp(tp_std, specs_std)
    assert dp_std[0].tensors["qkv_proj"] == qkv_std_full

    qkv_rep_full = _make_2d_tensor((8 + 2 + 2) * head_size, hidden_size, base=200)
    specs_rep = {
        "qkv_proj": ShardSpec(
            kind="qkv_parallel",
            total_num_heads=8,
            total_num_kv_heads=2,
            head_size=head_size,
        )
    }
    tp_rep, rep_info = resharder.reshard_dp_to_tp({"qkv_proj": qkv_rep_full}, tp_size, specs_rep)
    assert rep_info.is_ok
    assert any(item.code == "reshard_kv_heads_replicated" for item in rep_info.findings)
    assert tp_rep[0].tensors["qkv_proj"].data[4:] == tp_rep[1].tensors["qkv_proj"].data[4:]
    assert tp_rep[2].tensors["qkv_proj"].data[4:] == tp_rep[3].tensors["qkv_proj"].data[4:]
    assert tp_rep[0].tensors["qkv_proj"].data[:4] != tp_rep[1].tensors["qkv_proj"].data[:4]

    dp_rep, rep_back = resharder.reshard_tp_to_dp(tp_rep, specs_rep)
    assert rep_back.is_ok
    for holder in dp_rep:
        assert holder.tensors["qkv_proj"] == qkv_rep_full


def test_vram_budget_and_tile_repacked_quantization_guards() -> None:
    resharder = ParallelismResharder()
    tp_size = 4
    full_weights = {"linear.weight": _make_2d_tensor(16, 8, base=1)}
    specs = {"linear.weight": ShardSpec(kind="col_parallel")}
    tp_holders, _ = resharder.reshard_dp_to_tp(full_weights, tp_size, specs)

    full_bytes = full_weights["linear.weight"].nbytes
    # Transitioning TP4 -> TP2+DP2 requires full_bytes // 2 per GPU
    _, bad_tp2_vram = resharder.reshard_tp_dp(
        tp_holders, specs, target_tp_size=2, target_dp_size=2, vram_budget_bytes=(full_bytes // 2) - 1
    )
    assert not bad_tp2_vram.is_ok
    assert any(item.code == "reshard_dp_vram_exceeded" for item in bad_tp2_vram.findings)

    _, bad_vram_report = resharder.reshard_tp_to_dp(
        tp_holders, specs, vram_budget_bytes=full_bytes - 1
    )
    assert not bad_vram_report.is_ok
    assert any(item.code == "reshard_dp_vram_exceeded" for item in bad_vram_report.findings)

    dp_ok, ok_vram_report = resharder.reshard_tp_to_dp(
        tp_holders, specs, vram_budget_bytes=full_bytes
    )
    assert ok_vram_report.is_ok
    assert dp_ok[0].tensors["linear.weight"] == full_weights["linear.weight"]

    marlin_specs = {"linear.weight": ShardSpec(kind="col_parallel", quant_format="awq_marlin")}
    _, bad_quant_report = resharder.reshard_tp_to_dp(tp_holders, marlin_specs)
    assert not bad_quant_report.is_ok
    assert any(item.code == "reshard_quantized_requires_canonical" for item in bad_quant_report.findings)

    dp_canon, ok_quant_report = resharder.reshard_tp_to_dp(
        tp_holders, marlin_specs, canonical_weights=full_weights
    )
    assert ok_quant_report.is_ok
    assert dp_canon[0].tensors["linear.weight"] == full_weights["linear.weight"]


def test_pytorch_tensor_compatibility_when_available() -> None:
    torch_mod: Any = pytest.importorskip("torch")
    resharder = ParallelismResharder()
    tp_size = 4
    full_weights = {
        "gate_up": torch_mod.arange(64, dtype=torch_mod.float32).reshape(16, 4),
        "moe_w13": torch_mod.arange(128, dtype=torch_mod.float32).reshape(4, 8, 4),
    }
    specs = {
        "gate_up": ShardSpec(kind="merged_col_parallel", sub_sizes=(8, 8)),
        "moe_w13": ShardSpec(kind="moe_w13", num_experts=4, expert_placement="round_robin"),
    }
    tp_holders, rep1 = resharder.reshard_dp_to_tp(full_weights, tp_size, specs)
    assert rep1.is_ok
    tp2_ep2, rep_mid = resharder.reshard_moe_tp_ep(tp_holders, specs, target_moe_tp_size=2, target_ep_size=2)
    assert rep_mid.is_ok
    ep_holders, rep2 = resharder.reshard_moe_tp_ep(tp2_ep2, specs, target_moe_tp_size=1, target_ep_size=4)
    assert rep2.is_ok
    tp_back, rep3 = resharder.reshard_moe_ep_to_tp(ep_holders, specs)
    assert rep3.is_ok
    tp2_dp2, rep_dp_mid = resharder.reshard_tp_dp(tp_back, specs, target_tp_size=2, target_dp_size=2)
    assert rep_dp_mid.is_ok
    dp_holders, rep4 = resharder.reshard_tp_to_dp(tp2_dp2, specs)
    assert rep4.is_ok
    assert torch_mod.equal(dp_holders[0].tensors["gate_up"], full_weights["gate_up"])
    assert torch_mod.equal(dp_holders[0].tensors["moe_w13"], full_weights["moe_w13"])


def test_tp_dp_reshard_refuses_moe_holders_sharded_by_expert() -> None:
    """Input: 4 experts on (tp=4, ep=4, moe_tp=1), each rank holding one whole expert, and the
    same experts on (tp=1, dp=4, ep=4). Action: reshard_tp_dp to TP1/DP4 and to TP4/DP1.
    Expect: an error finding `reshard_moe_mesh_unsupported` and no tensors written. Before: the
    TP gather concatenated four different experts into one expert of 4x the intermediate size
    (first case) or kept rank 0's single expert and dropped the other three (second case), and
    reported ok. Does not prove: an EP-aware TP<->DP path (none exists; transpose with
    reshard_moe_tp_ep to moe_tp == tp, ep == 1 first)."""
    resharder = ParallelismResharder()
    num_experts, inter_size, hidden_size = 4, 8, 4
    full = {
        "moe.w13": _make_3d_tensor(num_experts, 2 * inter_size, hidden_size, base=1),
        "moe.w2": _make_3d_tensor(num_experts, hidden_size, inter_size, base=2),
    }
    specs = {
        "moe.w13": ShardSpec(kind="moe_w13", num_experts=num_experts),
        "moe.w2": ShardSpec(kind="moe_w2", num_experts=num_experts),
    }
    tp4, _ = resharder.reshard_dp_to_tp(full, 4, specs)
    tp4_ep4, rep = resharder.reshard_moe_tp_to_ep(tp4, specs)
    assert rep.is_ok and tp4_ep4[0].tensors["moe.w13"].shape == (1, 2 * inter_size, hidden_size)
    dp4_ep4 = [
        RankWeightHolder(rank=rank, tp_size=1, dp_size=4, ep_size=4, moe_tp_size=1,
                         tensors=dict(tp4_ep4[rank].tensors))
        for rank in range(4)
    ]
    for holders, target_tp, target_dp in ((tp4_ep4, 1, 4), (dp4_ep4, 4, 1)):
        out, report = resharder.reshard_tp_dp(
            holders, specs, target_tp_size=target_tp, target_dp_size=target_dp)
        assert not report.is_ok, report.to_dict()
        assert [item.code for item in report.findings] == ["reshard_moe_mesh_unsupported"]
        assert all(not holder.tensors for holder in out)
