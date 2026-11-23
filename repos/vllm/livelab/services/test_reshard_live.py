"""Golden cases for reshard_live.py (live MoE TP <-> EP reshard between weight-holder groups).

CPU only. The plan cases need nothing; the transpose cases need torch; the layout cases need a
real vLLM (RSIBOX_VLLM_PYTHON, like test_vllm_args.py) because they run vLLM's own FlashInfer
TRT-LLM weight conversion on CPU. None of them maps a real holder: the GPU path (IPC map, in-place
write, byte comparison with a holder loaded from disk) is proved by the run in the PR receipt.

Run: RSIBOX_VLLM_PYTHON=<vLLM venv python> python3 -m pytest \
         repos/vllm/livelab/services/test_reshard_live.py -v
"""
import importlib.util, inspect, json, os, subprocess, sys, textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reshard_live as RL  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
VLLM_PY = os.environ.get("RSIBOX_VLLM_PYTHON") or \
    (sys.executable if importlib.util.find_spec("vllm") else None)


def meta(kind, *shape, dtype="torch.bfloat16"):
    return {"kind": kind, "shape": list(shape), "dtype": dtype}


# Qwen3.5-35B-A3B layer 0 shapes per rank: 256 experts, I = 512, H = 2048, 16 q heads, 2 kv heads.
def tp2_rank():
    return {
        "layers.3.self_attn.qkv_proj.weight": meta("param", (8 + 1 + 1) * 256, 2048),
        "layers.0.mlp.experts.w13_weight": meta("param", 256, 512, 2048),
        "layers.0.mlp.experts.w2_weight": meta("param", 256, 2048, 256),
        "layers.0.mlp.gate.weight": meta("param", 256, 2048),
        "rotary_emb.cos_sin_cache": meta("buffer", 8192, 64, dtype="torch.float32"),
    }


def ep2_rank():
    r = tp2_rank()
    r["layers.0.mlp.experts.w13_weight"] = meta("param", 128, 1024, 2048)
    r["layers.0.mlp.experts.w2_weight"] = meta("param", 128, 2048, 512)
    r["layers.0.mlp.experts._expert_map"] = meta("buffer", 256, dtype="torch.int32")
    return r


def test_plan_tp2_to_ep2_moves_only_experts():
    """TP2 -> TP2+EP2: attention, router and shared weights are the same shard on both sides
    (rank-local copy); the two expert tensors are transposed; the EP expert map is the target's
    own buffer (kept, never overwritten). Proves the classification; not the bytes."""
    p = RL.plan([tp2_rank(), tp2_rank()], [ep2_rank(), ep2_rank()])
    assert p["findings"] == []
    assert p["copy"] == ["layers.0.mlp.gate.weight", "layers.3.self_attn.qkv_proj.weight"]
    assert p["moe"] == {"layers.0.mlp.experts.w13_weight": "w13",
                        "layers.0.mlp.experts.w2_weight": "w2"}
    assert p["keep"] == ["layers.0.mlp.experts._expert_map", "rotary_emb.cos_sin_cache"]
    # and back: the source's expert map is a buffer the TP target does not have, ignored
    back = RL.plan([ep2_rank(), ep2_rank()], [tp2_rank(), tp2_rank()])
    assert back["findings"] == [] and back["ignored_src"] == ["layers.0.mlp.experts._expert_map"]


def test_plan_refuses_dense_shard_change():
    """TP2 -> DP2 (TP1 attention): qkv_proj is a different shard on each rank. A rank-local copy
    would write half a tensor into a whole one; the plan refuses before anything is written."""
    dp2 = tp2_rank()
    dp2["layers.3.self_attn.qkv_proj.weight"] = meta("param", (16 + 2 + 2) * 256, 2048)
    p = RL.plan([tp2_rank(), tp2_rank()], [dp2, dp2])
    assert [f["check"] for f in p["findings"]] == ["reshard_live_unmapped"]
    assert "qkv_proj" in p["findings"][0]["observed"]


def test_plan_refuses_dropped_param_and_bad_expert_mesh():
    """A source parameter the target lacks would be lost; experts that fit no (moe_tp, ep) mesh
    (here 3 per rank on 2 ranks from 256) cannot be transposed. Both refused."""
    src = tp2_rank()
    src["lm_head.weight"] = meta("param", 124160, 2048)
    p = RL.plan([src, src], [tp2_rank(), tp2_rank()])
    assert [f["check"] for f in p["findings"]] == ["reshard_live_dropped"]
    bad = ep2_rank()
    bad["layers.0.mlp.experts.w13_weight"] = meta("param", 3, 1024, 2048)
    p = RL.plan([tp2_rank(), tp2_rank()], [bad, bad])
    assert "reshard_live_moe_shape" in [f["check"] for f in p["findings"]]


def test_moe_layout_from_log_reads_vllm_backend_names(tmp_path):
    """The holder log names the backend by vLLM's enum value, which has spaces ("FlashInfer
    TRTLLM"); an unknown backend (here MoonEP) is refused, never treated as identity."""
    log = tmp_path / "holder.log"
    log.write_text("INFO x Using FlashInfer TRTLLM Unquantized MoE backend out of potential "
                   "backends: ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON'].\n")
    assert RL.moe_layout_from_log(str(log)) == "flashinfer_trtllm"
    log.write_text("INFO x Using MOONEP Unquantized MoE backend out of potential backends\n")
    with pytest.raises(ValueError, match="no layout codec"):
        RL.moe_layout_from_log(str(log))


# -- the transpose, in checkpoint layout ----------------------------------------------------

def shards(full13, full2, world, ep, placement="linear"):
    """Per-rank checkpoint-layout expert tensors as vLLM's FusedMoE weight loader builds them:
    rank r = ep rank r // moe_tp, moe-tp rank r % moe_tp; w13 rows are [gate slice; up slice]."""
    import torch
    E, two_i, _ = full13.shape
    i_full, moe_tp = two_i // 2, world // ep
    i_loc = i_full // moe_tp
    out13, out2 = [], []
    for r in range(world):
        ep_r, tp_r = divmod(r, moe_tp)
        ids = sorted((RL.map_global_expert_to_ep(g, E, ep, placement)[1], g)
                     for g in range(E) if RL.map_global_expert_to_ep(g, E, ep, placement)[0] == ep_r)
        ids = torch.tensor([g for _, g in ids])
        lo, hi = tp_r * i_loc, (tp_r + 1) * i_loc
        out13.append(torch.cat([full13[ids, lo:hi], full13[ids, i_full + lo:i_full + hi]], 1))
        out2.append(full2[ids, :, lo:hi].clone())
    return out13, out2


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
@pytest.mark.parametrize("world,src_ep,dst_ep,placement", [
    (2, 1, 2, "linear"),          # TP2 -> EP2 (the B200 run)
    (2, 2, 1, "linear"),          # EP2 -> TP2
    (4, 1, 2, "linear"),          # (moe_tp4, ep1) -> (moe_tp2, ep2)
    (4, 4, 2, "round_robin"),
])
def test_moe_transpose_matches_direct_sharding(world, src_ep, dst_ep, placement):
    """Experts sharded one way and transposed equal the same experts sharded the other way,
    bit for bit (pure index moves). Proves the expert and intermediate bookkeeping, incl. the
    gate/up halves of w13; not the kernel layout (next cases)."""
    import torch
    g = torch.Generator().manual_seed(0)
    full13 = torch.randn(8, 2 * 64, 32, generator=g).to(torch.bfloat16)
    full2 = torch.randn(8, 32, 64, generator=g).to(torch.bfloat16)
    s13, s2 = shards(full13, full2, world, src_ep, placement)
    d13, d2 = shards(full13, full2, world, dst_ep, placement)
    got13 = RL.moe_transpose(s13, "w13", src_ep, dst_ep, placement=placement)
    got2 = RL.moe_transpose(s2, "w2", src_ep, dst_ep, placement=placement)
    assert all(torch.equal(a, b) for a, b in zip(got13, d13))
    assert all(torch.equal(a, b) for a, b in zip(got2, d2))


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_moe_transpose_agrees_with_reshard_py():
    """The vectorised transpose agrees with reshard.py's per-expert reference (the existing
    reshard_moe_tp_to_ep) on a TP2 -> EP2 case."""
    import torch
    from reshard import ParallelismResharder, RankWeightHolder, ShardSpec
    g = torch.Generator().manual_seed(1)
    full13 = torch.randn(4, 2 * 16, 8, generator=g)
    full2 = torch.randn(4, 8, 16, generator=g)
    s13, s2 = shards(full13, full2, 2, 1)
    holders = [RankWeightHolder(rank=r, tp_size=2, tensors={"w13": s13[r], "w2": s2[r]})
               for r in range(2)]
    ref, rep = ParallelismResharder().reshard_moe_tp_to_ep(
        holders, {"w13": ShardSpec("moe_w13", num_experts=4), "w2": ShardSpec("moe_w2", num_experts=4)})
    assert rep.is_ok
    got13, got2 = RL.moe_transpose(s13, "w13", 1, 2), RL.moe_transpose(s2, "w2", 1, 2)
    for r in range(2):
        assert torch.equal(got13[r], ref[r].tensors["w13"])
        assert torch.equal(got2[r], ref[r].tensors["w2"])


# -- the kernel layout: vLLM's own FlashInfer TRT-LLM conversion, on CPU --------------------

LAYOUT_CASE = textwrap.dedent("""
    import json, sys, torch
    sys.path.insert(0, sys.argv[1])
    import reshard_live as RL
    exec(sys.argv[2])                       # shards(), this file's helper (no pytest there)
    try:
        RL._trtllm_convert(torch.zeros(1, 256, 128, dtype=torch.bfloat16),
                           torch.zeros(1, 128, 128, dtype=torch.bfloat16))
    except ImportError as e:
        print(json.dumps({"skip": f"this vLLM has no TRT-LLM BF16 MoE conversion: {e}"}))
        sys.exit(0)
    # Qwen3.5-35B-A3B expert shapes (I = 512, H = 2048), 4 experts instead of 256
    g = torch.Generator().manual_seed(0)
    full13 = torch.randn(4, 1024, 2048, generator=g).to(torch.bfloat16)
    full2 = torch.randn(4, 2048, 512, generator=g).to(torch.bfloat16)
    def stored(t13, t2):      # what a holder serves after process_weights_after_loading
        a, b = t13.clone(), t2.clone()
        RL._trtllm_convert(a, b)
        return a, b
    tp13, tp2 = zip(*[stored(a, b) for a, b in zip(*shards(full13, full2, 2, 1))])
    ep13, ep2 = zip(*[stored(a, b) for a, b in zip(*shards(full13, full2, 2, 2))])
    L = RL.MoeLayout("flashinfer_trtllm")
    def live(s13, s2, src_ep, dst_ep):
        o13 = RL.moe_transpose([L.decode(t, "w13") for t in s13], "w13", src_ep, dst_ep)
        o2 = RL.moe_transpose([L.decode(t, "w2") for t in s2], "w2", src_ep, dst_ep)
        return [L.encode(t, "w13") for t in o13], [L.encode(t, "w2") for t in o2]
    eq = lambda xs, ys: all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(xs, ys))
    to_ep, back = live(tp13, tp2, 1, 2), live(ep13, ep2, 2, 1)
    naive13 = RL.moe_transpose(list(tp13), "w13", 1, 2)
    naive2 = RL.moe_transpose(list(tp2), "w2", 1, 2)
    print(json.dumps({
        "stored_is_checkpoint_layout": eq(tp13, shards(full13, full2, 2, 1)[0]),
        "tp2_to_ep2": eq(to_ep[0], ep13) and eq(to_ep[1], ep2),
        "ep2_to_tp2": eq(back[0], tp13) and eq(back[1], tp2),
        "naive_w13_ok": eq(naive13, ep13), "naive_w2_ok": eq(naive2, ep2)}))
""")


@pytest.mark.skipif(not VLLM_PY, reason="needs vLLM: run with RSIBOX_VLLM_PYTHON set to a vLLM "
                    "venv's python (CPU CI has no vLLM)")
def test_trtllm_layout_reshard_is_byte_exact_and_naive_is_not():
    """Input: 4 Qwen3.5-shaped experts, stored as vLLM 0.33 stores them for the FlashInfer TRT-LLM
    BF16 kernel (the B200 default), at TP2 (I/rank = 256) and at EP2 (I/rank = 512).
    Action: decode -> transpose -> encode, both directions.
    Expected: byte-identical to what vLLM stores for the target mesh; and the naive transpose of
    the stored bytes (no codec) is wrong. That is the silent corruption this file exists to stop.
    Proves the layout codec against the installed FlashInfer; not the IPC path."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([VLLM_PY, "-c", LAYOUT_CASE, str(HERE), inspect.getsource(shards)], capture_output=True, text=True,
                       timeout=600, env=env, cwd=str(HERE))
    line = next((x for x in r.stdout.splitlines()[::-1] if x.startswith("{")), None)
    assert line, r.stderr[-3000:]
    got = json.loads(line)
    if "skip" in got:
        pytest.skip(got["skip"])
    assert got == {"stored_is_checkpoint_layout": False, "tp2_to_ep2": True, "ep2_to_tp2": True,
                   "naive_w13_ok": False, "naive_w2_ok": False}


# -- the launcher's route (engine.py reconfigure -> reshard_live.py ensure) ------------------

def test_reshardable_and_group_dir_round_trip():
    """Only an expert-sharding change of the weight key is reshardable (TP size, dtype, model
    are not); the target folder is deterministic, so EP2 -> TP2 lands on the TP2 group it came
    from (which still holds its weights: no reshard on the way back)."""
    assert RL.reshardable({"weight_key_changed": ["enable_expert_parallel"]})
    assert RL.reshardable({"weight_key_changed": ["enable_expert_parallel",
                                                  "expert_placement_strategy"]})
    assert not RL.reshardable({"weight_key_changed": ["enable_expert_parallel",
                                                      "tensor_parallel_size"]})
    assert not RL.reshardable({"weight_key_changed": []}) and not RL.reshardable(None)
    ep = RL.group_dir("/s/w", {"enable_expert_parallel": True,
                               "expert_placement_strategy": "linear"})
    assert ep == "/s/w.ep"
    assert RL.group_dir(ep, {"enable_expert_parallel": False}) == "/s/w"
    assert RL.group_dir("/s/w/", {"enable_expert_parallel": True,
                                  "expert_placement_strategy": "round_robin"}) == "/s/w.eprr"


def test_moe_ep_and_key_mesh_disambiguate_ep_swaps_and_4gpu_meshes():
    """Input: (a) 2-GPU EP2 linear -> EP2 round_robin (128x512 -> 128x512);
    (b) 4-GPU (TP4, EP2) -> (TP4, EP4) (128x256 -> 64x512).
    Expected: constrained `moe_ep` returns `(2, 2)` and `(2, 4)` rather than the ambiguous
    smaller divisor pairs `(1, 1)` and `(1, 2)`, and `_key_moe_mesh` extracts EP & placement."""
    from types import SimpleNamespace
    assert RL.moe_ep(128, 512, 128, 512, 2, src_ep=2, dst_ep=2) == (2, 2)
    assert RL.moe_ep(128, 256, 64, 512, 4, dst_ep=4) == (2, 4)
    assert RL.moe_ep(128, 256, 64, 512, 4, src_ep=2, dst_ep=4) == (2, 4)
    k_tp = [SimpleNamespace(enable_expert_parallel=False, expert_placement_strategy="linear")] * 2
    k_rr = [SimpleNamespace(enable_expert_parallel=True, expert_placement_strategy="round_robin")] * 2
    assert RL._key_moe_mesh(k_tp, 2) == (1, "linear")
    assert RL._key_moe_mesh(k_rr, 2) == (2, "round_robin")
    p = RL.plan([ep2_rank(), ep2_rank()], [ep2_rank(), ep2_rank()], src_ep=2, dst_ep=2)
    assert p["findings"] == [] and set(p["moe"]) == {
        "layers.0.mlp.experts.w13_weight", "layers.0.mlp.experts.w2_weight"
    }


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
@pytest.mark.parametrize("world,src_ep,dst_ep,src_place,dst_place", [
    (2, 1, 2, "linear", "round_robin"),       # TP2 -> EP2 round_robin (.eprr)
    (2, 2, 1, "round_robin", "linear"),       # EP2 round_robin -> TP2
    (2, 2, 2, "linear", "round_robin"),       # EP2 linear -> EP2 round_robin
    (2, 2, 2, "round_robin", "linear"),       # EP2 round_robin -> EP2 linear
    (4, 2, 4, "linear", "round_robin"),       # 4-GPU (TP4,EP2) -> (TP4,EP4) round_robin
])
def test_execute_and_transpose_honor_src_and_dst_expert_placement(
    world, src_ep, dst_ep, src_place, dst_place
):
    """Input: experts sharded under `(src_ep, src_place)` and transposed via `execute()` into
    `(dst_ep, dst_place)`. Expected: bit-exact match with direct target sharding on every rank
    and nonzero `moe_cross_gpu_bytes` even on same-EP placement swaps (`EP2 linear <-> rr`)."""
    import torch
    g = torch.Generator().manual_seed(7)
    full13 = torch.randn(8, 2 * 64, 32, generator=g).to(torch.bfloat16)
    full2 = torch.randn(8, 32, 64, generator=g).to(torch.bfloat16)
    s13, s2 = shards(full13, full2, world, src_ep, src_place)
    d13, d2 = shards(full13, full2, world, dst_ep, dst_place)
    src_group = [{"e.w13_weight": ("param", s13[r]), "e.w2_weight": ("param", s2[r])}
                 for r in range(world)]
    dst_group = [{"e.w13_weight": ("param", torch.zeros_like(d13[r])),
                  "e.w2_weight": ("param", torch.zeros_like(d2[r]))}
                 for r in range(world)]
    p = RL.plan(RL.metadata(src_group), RL.metadata(dst_group), src_ep=src_ep, dst_ep=dst_ep)
    assert p["findings"] == []
    id_layout = RL.MoeLayout("triton")
    rep = RL.execute(src_group, dst_group, p, id_layout, id_layout,
                     src_ep=src_ep, dst_ep=dst_ep, src_placement=src_place, dst_placement=dst_place)
    assert rep["moe_cross_gpu_bytes"] > 0
    for r in range(world):
        assert torch.equal(dst_group[r]["e.w13_weight"][1], d13[r])
        assert torch.equal(dst_group[r]["e.w2_weight"][1], d2[r])


HOLDER_ARGV_CASE = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import reshard_live as RL
    print(json.dumps(RL.holder_argv(json.loads(sys.argv[2]))))
""")


@pytest.mark.skipif(not VLLM_PY, reason="needs vLLM (RSIBOX_VLLM_PYTHON)")
def test_holder_argv_keeps_engine_args_only():
    """The target holder gets the engine's weight-relevant flags (by vLLM's EngineArgs parser),
    never its loader, worker class, memory budget or server flags: a holder started with
    --load-format ipc_cache would dial itself; --port is not an EngineArgs flag at all."""
    vargv = ["/m/Q", "--port", "8302", "--tensor-parallel-size", "2", "--enable-expert-parallel",
             "--gpu-memory-utilization", "0.3", "--max-model-len", "8192",
             "--load-format", "ipc_cache", "--worker-cls", "a.B"]
    r = subprocess.run([VLLM_PY, "-c", HOLDER_ARGV_CASE, str(HERE), json.dumps(vargv)],
                       capture_output=True, text=True, timeout=300, cwd=str(HERE),
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    line = next((x for x in r.stdout.splitlines()[::-1] if x.startswith("[")), None)
    assert line, r.stderr[-3000:]
    assert json.loads(line) == ["--model", "/m/Q", "--tensor-parallel-size", "2",
                                "--enable-expert-parallel", "--max-model-len", "8192"]
