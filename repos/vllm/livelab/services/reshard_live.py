"""Live MoE reshard between two weight-holder groups: TP2 <-> EP2 without reading the checkpoint.

A weight holder group is vLLM's weight cache daemon (`vllm preload`, weights.py), one process per
GPU, keyed by `WeightCacheKey` (tp size and rank, dp, `enable_expert_parallel`, MoE backend, ...).
An engine started with `--load-format ipc_cache` maps a group whose key matches its own. A
parallelism change is a different key, so it needs a different group. This tool fills the target
group from the running one, GPU to GPU:

1. The target group is started with `--load-format dummy` (no checkpoint read; it can be started
   ahead of time while the source engine still serves, since it only needs free HBM).
2. `run` maps both groups through CUDA IPC in one process (every rank of both groups), and for
   each tensor of the target:
   - a parameter with the same name and shape on the same rank of the source is copied
     rank-locally (dense weights: TP2 and TP2+EP2 shard attention and dense MLPs the same way);
   - a FusedMoE `w13_weight` / `w2_weight` is decoded from the source's kernel layout to the
     checkpoint layout (`MoeLayout`), transposed between (moe_tp, ep) meshes (`moe_transpose`),
     encoded into the target's kernel layout and written in place;
   - a buffer is kept: buffers are built by the target's own constructor from its own parallel
     config (expert maps, rotary caches), and the dummy loader does not touch non-persistent
     buffers. Buffers whose bytes differ from the source's are listed, never silently copied.
   Anything else (a target parameter with no source of that shape, a source parameter the
   target lacks) is refused before any byte is written (`plan`).
3. `--verify-socket-dir` maps a third group loaded from disk with the target's flags and compares
   every tensor byte for byte (the proof run; production skips it).

Why a layout codec: on CUDA, vLLM 0.33's unquantized-MoE oracle tries the FlashInfer TRT-LLM BF16
kernel first (the backend actually picked is read from each group's log), and its
`process_weights_after_loading` permutes every expert's bytes into a block layout that depends on
the per-rank intermediate size (256 at TP2, 512 at EP2). Slicing the stored bytes as if they were
checkpoint-shaped gives wrong weights silently. `MoeLayout` learns the permutation by running
vLLM's own conversion on index-encoded probes (two int16 planes), so it is exact for whatever the
installed FlashInfer does, and it refuses a conversion that pads (not a bijection).

Usage (vLLM venv python, CUDA_VISIBLE_DEVICES = the groups' GPUs, rank r on device r):
  python3 reshard_live.py run --model M --src-socket-dir A --src-args "-tp 2" \
      --dst-socket-dir B --dst-args "-tp 2 --enable-expert-parallel" \
      [--src-moe-layout L | --src-log holderA.log] [--dst-moe-layout L | --dst-log holderB.log] \
      [--verify-socket-dir C] [--vllm-args "--max-model-len 8192 --language-model-only"]
Prints one JSON line: plan counts, per-phase seconds, bytes moved, verification.
Exit 0 ok, 2 refused before writing (findings), 1 verification failed.

Not handled (refused by `plan`): dense TP size changes (TP2 <-> DP2 gathers attention and
linear-attention shards; Qwen3.5's GDN projections need per-module rules), quantized MoE (scales
and tile repacks), and any MoE backend not in LAYOUTS.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from reshard import map_global_expert_to_ep  # noqa: E402

MOE_SUFFIXES = {".w13_weight": "w13", ".w2_weight": "w2"}
BACKEND_LOG = re.compile(r"Using (.+?) Unquantized MoE backend")
# kernel layouts of unquantized (BF16) expert weights, by vLLM 0.33's UnquantizedMoeBackend value
# (what the log line names) -> the --moe-backend spelling used here
LOG_NAMES = {"TRITON": "triton", "BATCHED_TRITON": "batched_triton",
             "FlashInfer CUTLASS": "flashinfer_cutlass", "FlashInfer TRTLLM": "flashinfer_trtllm"}
LAYOUTS = tuple(LOG_NAMES.values())


# ---------------------------------------------------------------------------------------------
# Plan: metadata only (no torch), decides every tensor before anything is written
# ---------------------------------------------------------------------------------------------

def moe_kind(name, shape):
    for suffix, kind in MOE_SUFFIXES.items():
        if name.endswith(suffix) and len(shape) == 3:
            return kind
    return None


def plan(src, dst):
    """src, dst: one dict per rank {name: {"kind": "param"|"buffer", "shape": [...],
    "dtype": str}}. Returns {"copy": [names], "moe": {name: "w13"|"w2"}, "keep": [buffer names],
    "ignored_src": [src-only buffer names], "findings": [...]}; findings mean refuse."""
    found = []
    if len(src) != len(dst):
        found.append({"check": "reshard_live_world", "observed": f"{len(src)} source ranks, "
                      f"{len(dst)} target ranks", "expected": "the same GPUs, one rank each"})
        return {"copy": [], "moe": {}, "keep": [], "ignored_src": [], "findings": found}
    out = {"copy": [], "moe": {}, "keep": [], "ignored_src": [], "findings": found}
    names = sorted(set().union(*[set(d) for d in dst]))
    for name in names:
        metas_d = [d.get(name) for d in dst]
        metas_s = [s.get(name) for s in src]
        if any(m is None for m in metas_d):
            found.append({"check": "reshard_live_ranks", "observed": f"{name} missing on some "
                          "target ranks", "expected": "every rank holds the same tensor names"})
            continue
        if metas_d[0]["kind"] == "buffer":
            out["keep"].append(name)
            continue
        kind = moe_kind(name, metas_d[0]["shape"])
        if kind is not None:
            if any(m is None or moe_kind(name, m["shape"]) != kind for m in metas_s):
                found.append({"check": "reshard_live_unmapped", "observed": f"MoE {name} has no "
                              "3-D source on every rank", "expected": "the same FusedMoE layer"})
            elif any(m["dtype"] != metas_d[0]["dtype"] for m in metas_s):
                found.append({"check": "reshard_live_dtype", "observed": f"{name}: source "
                              f"{metas_s[0]['dtype']}, target {metas_d[0]['dtype']}",
                              "expected": "one dtype (BF16 MoE only)"})
            else:
                problem = moe_mesh_problem(kind, [m["shape"] for m in metas_s],
                                           [m["shape"] for m in metas_d])
                if problem:
                    found.append({"check": "reshard_live_moe_shape", "observed": f"{name}: "
                                  f"{problem}", "expected": "same experts and intermediate size"})
                else:
                    out["moe"][name] = kind
            continue
        same = all(s is not None and list(s["shape"]) == list(d["shape"])
                   and s["dtype"] == d["dtype"] for s, d in zip(metas_s, metas_d))
        if same:
            out["copy"].append(name)
        else:
            shapes = [None if s is None else list(s["shape"]) for s in metas_s]
            found.append({"check": "reshard_live_unmapped",
                          "observed": f"target parameter {name} {list(metas_d[0]['shape'])} has "
                                      f"source shapes {shapes}",
                          "expected": "a rank-local copy (same shard) or a FusedMoE expert tensor",
                          "why": "a dense shard change (e.g. TP2 -> DP2) needs per-module gather "
                                 "rules this tool does not have; writing it would corrupt it"})
    src_names = set().union(*[set(s) for s in src])
    for name in sorted(src_names - set(names)):
        if src[0].get(name, {}).get("kind") == "buffer":
            out["ignored_src"].append(name)
        else:
            found.append({"check": "reshard_live_dropped", "observed": f"source parameter {name} "
                          "has no target", "expected": "the same model on both sides"})
    return out


def moe_mesh(kind, shapes):
    """(experts_per_rank, local_intermediate, hidden) of per-rank expert tensor shapes."""
    e, a, b = shapes[0]
    return (e, a // 2, b) if kind == "w13" else (e, b, a)


def moe_mesh_problem(kind, src_shapes, dst_shapes):
    world = len(src_shapes)
    se, si, sh = moe_mesh(kind, src_shapes)
    de, di, dh = moe_mesh(kind, dst_shapes)
    if any(list(s) != list(src_shapes[0]) for s in src_shapes) or \
       any(list(d) != list(dst_shapes[0]) for d in dst_shapes):
        return "ranks hold different shapes"
    if sh != dh:
        return f"hidden {sh} vs {dh}"
    src_ep, dst_ep = moe_ep(se, si, de, di, world)
    if src_ep is None:
        return f"source {se}x{si}, target {de}x{di} on {world} ranks fit no (moe_tp, ep) mesh"
    return None


def moe_ep(src_e, src_i, dst_e, dst_i, world):
    """(src_ep, dst_ep) such that experts * ep and intermediate * moe_tp agree on both sides."""
    for src_ep in range(1, world + 1):
        if world % src_ep:
            continue
        for dst_ep in range(1, world + 1):
            if world % dst_ep:
                continue
            if (src_e * src_ep == dst_e * dst_ep and
                    src_i * (world // src_ep) == dst_i * (world // dst_ep)):
                return src_ep, dst_ep
    return None, None


# ---------------------------------------------------------------------------------------------
# Kernel layout codec and the MoE transpose (torch)
# ---------------------------------------------------------------------------------------------

def _probe(n, rows, cols, shift, torch):
    idx = torch.arange(n, dtype=torch.int64)
    v = ((idx >> shift) & 0xFFFF).to(torch.int32)
    v = torch.where(v >= 32768, v - 65536, v).to(torch.int16)
    return v.view(torch.bfloat16).view(1, rows, cols).clone()


def _unprobe(lo, hi, torch):
    lo = lo.reshape(-1).view(torch.int16).to(torch.int64) & 0xFFFF
    hi = hi.reshape(-1).view(torch.int16).to(torch.int64) & 0xFFFF
    return lo | (hi << 16)


def learn_permutation(convert, w13_shape, w2_shape):
    """Run `convert(w13, w2)` (in place, one expert) on index-encoded probes and return the flat
    permutations (P13, P2): encoded.flatten()[j] == canonical.flatten()[P[j]] per expert.
    Raises when the conversion is not a byte permutation of the same shape (e.g. it pads)."""
    import torch
    (r13, c13), (r2, c2) = w13_shape, w2_shape
    n13, n2 = r13 * c13, r2 * c2
    lo13, lo2 = _probe(n13, r13, c13, 0, torch), _probe(n2, r2, c2, 0, torch)
    hi13, hi2 = _probe(n13, r13, c13, 16, torch), _probe(n2, r2, c2, 16, torch)
    for w13, w2 in ((lo13, lo2), (hi13, hi2)):
        convert(w13, w2)
    perms = []
    for lo, hi, n, shape in ((lo13, hi13, n13, w13_shape), (lo2, hi2, n2, w2_shape)):
        if tuple(lo.shape[1:]) != tuple(shape):
            raise ValueError(f"layout conversion changed shape {shape} -> {tuple(lo.shape)}")
        perm = _unprobe(lo, hi, torch)
        if not torch.equal(torch.sort(perm).values, torch.arange(n)):
            raise ValueError(f"layout conversion of {shape} is not a permutation")
        perms.append(perm)
    return perms[0], perms[1]


def _trtllm_convert(w13, w2):
    import vllm.model_executor.layers.fused_moe  # noqa: F401  (import order: breaks a cycle)
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout)
    convert_moe_weights_to_flashinfer_trtllm_block_layout({}, w13, w2, is_gated_act_gemm=True)


class MoeLayout:
    """Encode/decode BF16 expert tensors [E, R, C] between the checkpoint layout and one MoE
    backend's stored layout. Permutation layouts are learned once per (backend, shape)."""
    _cache: dict = {}

    def __init__(self, backend, convert=None):
        if backend not in LAYOUTS and convert is None:
            raise ValueError(f"no layout codec for MoE backend {backend!r} (known: {LAYOUTS})")
        self.backend = backend
        self.convert = convert or (_trtllm_convert if backend == "flashinfer_trtllm" else None)

    def _perm(self, kind, shape, device):
        """(perm, inverse) on `device` for one expert of `shape`, or None for identity."""
        if self.convert is None:
            return None
        rows, cols = shape
        key = (self.backend, id(self.convert), kind, rows, cols)
        if key not in MoeLayout._cache:
            import torch
            # the w2 partner of a w13 [2I, H] is [H, I] and vice versa
            w13 = (rows, cols) if kind == "w13" else (2 * cols, rows)
            w2 = (rows, cols) if kind == "w2" else (cols, rows // 2)
            p13, p2 = learn_permutation(self.convert, w13, w2)
            perm = p13 if kind == "w13" else p2
            MoeLayout._cache[key] = (perm, torch.argsort(perm))
        perm, inv = MoeLayout._cache[key]
        return perm.to(device), inv.to(device)

    def _apply(self, x, kind, inverse):
        if self.backend == "flashinfer_cutlass" and kind == "w13":
            half = x.shape[1] // 2      # stored [w3; w1]: swapping halves is its own inverse
            return x.roll(half, dims=1)
        p = self._perm(kind, tuple(x.shape[1:]), x.device)
        if p is None:
            return x
        flat = x.reshape(x.shape[0], -1)
        return flat.index_select(1, p[1] if inverse else p[0]).view(x.shape)

    def decode(self, x, kind):
        return self._apply(x, kind, inverse=True)

    def encode(self, x, kind):
        return self._apply(x, kind, inverse=False)


def moe_transpose(src, kind, src_ep, dst_ep, devices=None, placement="linear"):
    """Checkpoint-layout expert tensors, one per source rank ([E/src_ep, 2*I/src_tp, H] for w13 with
    gate rows before up rows, [E/src_ep, H, I/src_tp] for w2) -> one per target rank for the
    (world/dst_ep, dst_ep) mesh. Rank r is ep rank r // moe_tp and moe-tp rank r % moe_tp (vLLM's
    order). `devices[r]` is where target rank r's tensor is built."""
    import torch
    world = len(src)
    src_tp, dst_tp = world // src_ep, world // dst_ep
    e_src = src[0].shape[0]
    num_experts = e_src * src_ep
    i_src = src[0].shape[1] // 2 if kind == "w13" else src[0].shape[2]
    i_dst = i_src * src_tp // dst_tp
    e_dst = num_experts // dst_ep
    out = []
    for r in range(world):
        dev = devices[r] if devices else src[r].device
        d_ep, d_tp = divmod(r, dst_tp)
        shape = ((e_dst, 2 * i_dst, src[0].shape[2]) if kind == "w13"
                 else (e_dst, src[0].shape[1], i_dst))
        buf = torch.empty(shape, dtype=src[0].dtype, device=dev)
        filled = torch.zeros(shape[0], dtype=torch.int64)
        slots = {}                                     # src ep rank -> [(dst slot, src slot)]
        for g in range(num_experts):
            ep_d, slot_d = map_global_expert_to_ep(g, num_experts, dst_ep, placement)
            if ep_d != d_ep:
                continue
            ep_s, slot_s = map_global_expert_to_ep(g, num_experts, src_ep, placement)
            slots.setdefault(ep_s, []).append((slot_d, slot_s))
        lo, hi = d_tp * i_dst, (d_tp + 1) * i_dst
        for ep_s, pairs in slots.items():
            for s_tp in range(src_tp):
                s_lo = s_tp * i_src
                o_lo, o_hi = max(lo, s_lo), min(hi, s_lo + i_src)
                if o_lo >= o_hi:
                    continue
                t = src[ep_s * src_tp + s_tp]
                src_idx = torch.tensor([b for _, b in pairs], device=t.device)
                dst_idx = torch.tensor([a for a, _ in pairs], device=dev)
                part = t.index_select(0, src_idx)
                if kind == "w13":
                    for half in (0, 1):
                        piece = part[:, half * i_src + o_lo - s_lo: half * i_src + o_hi - s_lo]
                        buf[dst_idx, half * i_dst + o_lo - lo: half * i_dst + o_hi - lo] = \
                            piece.to(dev)
                else:
                    buf[dst_idx, :, o_lo - lo: o_hi - lo] = part[:, :, o_lo - s_lo: o_hi - s_lo].to(dev)
                filled[dst_idx.cpu()] += o_hi - o_lo
        if not bool((filled == i_dst).all()):
            raise RuntimeError(f"moe_transpose left target rank {r} partly unfilled")
        out.append(buf)
    return out


def moe_layout_from_log(path):
    """The unquantized MoE backend a holder's log says it picked (its last such line)."""
    found = BACKEND_LOG.findall(open(path, errors="replace").read())
    if not found:
        raise ValueError(f"{path}: no 'Using <backend> Unquantized MoE backend' line")
    if found[-1] not in LOG_NAMES:
        raise ValueError(f"{path}: MoE backend {found[-1]!r} has no layout codec "
                         f"(known: {sorted(LOG_NAMES)})")
    return LOG_NAMES[found[-1]]


# ---------------------------------------------------------------------------------------------
# Holder groups over CUDA IPC (vLLM's own key and protocol, as an engine dials them)
# ---------------------------------------------------------------------------------------------

def engine_config(model, args):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    return EngineArgs.from_cli_args(parser.parse_args(["--model", model, *args])) \
        .create_engine_config()


def group_keys(cfg):
    """The WeightCacheKey of each rank of an engine with this config, in rank order (pp = 1)."""
    from vllm.model_executor.model_loader.weight_cache.daemon import get_engine_dp_placement
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey
    pc = cfg.parallel_config
    if pc.pipeline_parallel_size != 1:
        raise ValueError("pipeline parallelism is not handled")
    tp, dp = pc.tensor_parallel_size, pc.data_parallel_size
    keys = []
    for r in range(tp * dp):
        dp_rank, tp_rank = divmod(r, tp)
        dp_size_k, dp_rank_k = get_engine_dp_placement(cfg, dp_rank)
        keys.append(WeightCacheKey.from_model_config(
            cfg.model_config, tp_size=tp, tp_rank=tp_rank, cache_config=cfg.cache_config,
            dp_size=dp_size_k, dp_rank=dp_rank_k,
            enable_expert_parallel=pc.enable_expert_parallel,
            expert_placement_strategy=pc.expert_placement_strategy,
            kernel_config=cfg.kernel_config))
    return keys


def map_group(socket_dir, keys):
    """[{name: (kind, tensor)}] per rank: rank r's daemon (the one on device r) mapped through
    CUDA IPC. Raises with the daemon's answer on a key mismatch."""
    from vllm.model_executor.model_loader.weight_cache import protocol as P
    from vllm.platforms import current_platform
    ranks = []
    for r, key in enumerate(keys):
        path = P.get_socket_path(current_platform.get_device_uuid(r), socket_dir)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(path)
        try:
            P.send_msg(s, {"cmd": "get_state", "cache_key": key})
            reply = P.recv_msg(s)
        finally:
            s.close()
        if reply.get("status") != "ok":
            raise RuntimeError(f"holder rank {r} at {path} answered {reply}")
        ranks.append({n: (e.kind, e.rebuild(r)) for n, e in reply["entries"].items()
                      if e.ipc_args is not None})
    return ranks


def metadata(group):
    return [{n: {"kind": k, "shape": list(t.shape), "dtype": str(t.dtype)}
             for n, (k, t) in rank.items()} for rank in group]


def _bytes(t):
    return t.numel() * t.element_size()


def _same_bytes(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8))


def execute(src, dst, p, src_layout, dst_layout):
    """Write the target group in place following plan `p`. Returns byte counts and seconds."""
    import torch
    world = len(src)
    rep = {"copy_bytes": 0, "moe_bytes": 0, "moe_cross_gpu_bytes": 0}
    t0 = time.perf_counter()
    for name in p["copy"]:
        for r in range(world):
            d = dst[r][name][1]
            d.copy_(src[r][name][1])
            rep["copy_bytes"] += _bytes(d)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for name, kind in p["moe"].items():
        s_shapes = [list(src[r][name][1].shape) for r in range(world)]
        d_shapes = [list(dst[r][name][1].shape) for r in range(world)]
        se, si, _ = moe_mesh(kind, s_shapes)
        de, di, _ = moe_mesh(kind, d_shapes)
        src_ep, dst_ep = moe_ep(se, si, de, di, world)
        canon = [src_layout.decode(src[r][name][1], kind) for r in range(world)]
        outs = moe_transpose(canon, kind, src_ep, dst_ep,
                             devices=[dst[r][name][1].device for r in range(world)])
        for r in range(world):
            d = dst[r][name][1]
            d.copy_(dst_layout.encode(outs[r], kind))
            rep["moe_bytes"] += _bytes(d)
        # bytes whose source rank differs from their target rank (NVLink peer copies)
        rep["moe_cross_gpu_bytes"] += sum(_bytes(o) for o in outs) * (world - 1) // world \
            if src_ep != dst_ep else 0
        del canon, outs
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    rep.update({"copy_s": round(t1 - t0, 3), "moe_s": round(t2 - t1, 3)})
    return rep


def buffers_differ(src, dst, names):
    return sorted(n for n in names for r in range(len(dst))
                  if n in src[r] and not _same_bytes(src[r][n][1], dst[r][n][1]))


def verify(dst, ref):
    """Names of target tensors whose bytes differ from the reference group's, per rank."""
    differ, missing, n = [], [], 0
    for r in range(len(dst)):
        for name, (_, t) in dst[r].items():
            if name not in ref[r]:
                missing.append(name)
                continue
            n += 1
            if not _same_bytes(t, ref[r][name][1]):
                differ.append(f"rank{r}:{name}")
    return {"ok": not differ and not missing, "compared": n, "n_differ": len(differ),
            "differ": differ[:20], "missing": missing[:20]}


def run(a):
    import torch
    rep, t = {"phases_s": {}}, time.perf_counter()

    def phase(name):
        nonlocal t
        now = time.perf_counter()
        rep["phases_s"][name] = round(now - t, 3)
        t = now
    common = shlex.split(a.vllm_args)
    src_cfg = engine_config(a.model, shlex.split(a.src_args) + common)
    dst_cfg = engine_config(a.model, shlex.split(a.dst_args) + common)
    src_keys, dst_keys = group_keys(src_cfg), group_keys(dst_cfg)
    src_layout = MoeLayout(a.src_moe_layout or moe_layout_from_log(a.src_log))
    dst_layout = MoeLayout(a.dst_moe_layout or moe_layout_from_log(a.dst_log))
    rep["layouts"] = {"src": src_layout.backend, "dst": dst_layout.backend}
    phase("config")
    if len(src_keys) != len(dst_keys) or len(src_keys) > torch.cuda.device_count():
        rep["findings"] = [{"check": "reshard_live_world", "observed": f"{len(src_keys)} -> "
                            f"{len(dst_keys)} ranks on {torch.cuda.device_count()} GPUs"}]
        print(json.dumps(rep))
        return 2
    src, dst = map_group(a.src_socket_dir, src_keys), map_group(a.dst_socket_dir, dst_keys)
    phase("map")
    p = plan(metadata(src), metadata(dst))
    rep["plan"] = {k: len(v) for k, v in p.items() if k != "findings"}
    rep["ignored_src_buffers"] = p["ignored_src"]
    phase("plan")
    if p["findings"]:
        rep["findings"] = p["findings"][:20]
        rep["note"] = "refused before writing; target group unchanged"
        print(json.dumps(rep))
        return 2
    rep.update(execute(src, dst, p, src_layout, dst_layout))
    phase("write")
    rep["reshard_s"] = round(rep["phases_s"]["map"] + rep["phases_s"]["plan"]
                             + rep["phases_s"]["write"], 3)
    rep["buffers_differ_from_src"] = buffers_differ(src, dst, p["keep"])[:20]
    code = 0
    if a.verify_socket_dir:
        rep["verify"] = verify(dst, map_group(a.verify_socket_dir, dst_keys))
        phase("verify")
        code = 0 if rep["verify"]["ok"] else 1
    print(json.dumps(rep))
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--src-socket-dir", required=True)
    r.add_argument("--dst-socket-dir", required=True)
    r.add_argument("--src-args", required=True, help='vLLM flags of the source group, e.g. "-tp 2"')
    r.add_argument("--dst-args", required=True)
    r.add_argument("--vllm-args", default="", help="flags both groups share (max-model-len, ...)")
    for side in ("src", "dst"):
        g = r.add_mutually_exclusive_group(required=True)
        g.add_argument(f"--{side}-moe-layout", choices=LAYOUTS)
        g.add_argument(f"--{side}-log", help="the group's log, to read its MoE backend")
    r.add_argument("--verify-socket-dir", help="a group loaded from disk with --dst-args")
    a = ap.parse_args(argv)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
