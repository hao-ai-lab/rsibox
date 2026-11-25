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
import os
import re
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import corelab  # noqa: E402  (core/livelab on sys.path + shared helpers)
from rsibox import harness, ipc  # noqa: E402  (stdlib only: engine.py imports this module)

_finding = harness.plain_finding


def map_global_expert_to_ep(*args, **kwargs):
    """reshard.py's expert placement (imported on use: reshard.py imports torch, and engine.py
    imports this module on its launcher path)."""
    from reshard import map_global_expert_to_ep as place
    return place(*args, **kwargs)

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


def plan(src, dst, *, src_ep=None, dst_ep=None):
    """src, dst: one dict per rank {name: {"kind": "param"|"buffer", "shape": [...],
    "dtype": str}}. Returns {"copy": [names], "moe": {name: "w13"|"w2"}, "keep": [buffer names],
    "ignored_src": [src-only buffer names], "findings": [...]}; findings mean refuse."""
    found = []
    if len(src) != len(dst):
        found.append(_finding("reshard_live_world", f"{len(src)} source ranks, {len(dst)} target ranks",
                              "the same GPUs, one rank each"))
        return {"copy": [], "moe": {}, "keep": [], "ignored_src": [], "findings": found}
    out = {"copy": [], "moe": {}, "keep": [], "ignored_src": [], "findings": found}
    names = sorted(set().union(*[set(d) for d in dst]))
    for name in names:
        metas_d, metas_s = [d.get(name) for d in dst], [s.get(name) for s in src]
        if any(m is None for m in metas_d):
            found.append(_finding("reshard_live_ranks", f"{name} missing on some target ranks",
                                  "every rank holds the same tensor names"))
            continue
        if metas_d[0]["kind"] == "buffer":
            out["keep"].append(name)
            continue
        kind = moe_kind(name, metas_d[0]["shape"])
        if kind is not None:
            if any(m is None or moe_kind(name, m["shape"]) != kind for m in metas_s):
                found.append(_finding("reshard_live_unmapped", f"MoE {name} has no 3-D source on every rank",
                                      "the same FusedMoE layer"))
            elif any(m["dtype"] != metas_d[0]["dtype"] for m in metas_s):
                found.append(_finding("reshard_live_dtype", f"{name}: source {metas_s[0]['dtype']}, target {metas_d[0]['dtype']}",
                                      "one dtype (BF16 MoE only)"))
            elif (problem := moe_mesh_problem(kind, [m["shape"] for m in metas_s], [m["shape"] for m in metas_d],
                                              src_ep=src_ep, dst_ep=dst_ep)):
                found.append(_finding("reshard_live_moe_shape", f"{name}: {problem}",
                                      "same experts and intermediate size"))
            else:
                out["moe"][name] = kind
            continue
        if all(s is not None and list(s["shape"]) == list(d["shape"]) and s["dtype"] == d["dtype"]
               for s, d in zip(metas_s, metas_d)):
            out["copy"].append(name)
        else:
            shapes = [None if s is None else list(s["shape"]) for s in metas_s]
            found.append(_finding(
                "reshard_live_unmapped",
                f"target parameter {name} {list(metas_d[0]['shape'])} has source shapes {shapes}",
                "a rank-local copy (same shard) or a FusedMoE expert tensor",
                why="a dense shard change (e.g. TP2 -> DP2) needs per-module gather rules this tool does not have; writing it would corrupt it",
            ))
    for name in sorted(set().union(*[set(s) for s in src]) - set(names)):
        if src[0].get(name, {}).get("kind") == "buffer":
            out["ignored_src"].append(name)
        else:
            found.append(_finding("reshard_live_dropped", f"source parameter {name} has no target",
                                  "the same model on both sides"))
    return out


def _key_moe_mesh(keys, world):
    """Extract `(ep_size_or_None, placement_strategy)` from `WeightCacheKey`s."""
    if not keys:
        return None, "linear"
    k0 = keys[0]
    placement = getattr(k0, "expert_placement_strategy", None) or "linear"
    ep_flag = getattr(k0, "enable_expert_parallel", None)
    if ep_flag is False:
        return 1, placement
    if ep_flag is True:
        explicit = getattr(k0, "ep_size", None)
        return (int(explicit) if explicit else world), placement
    return None, placement


def moe_mesh(kind, shapes):
    """(experts_per_rank, local_intermediate, hidden) of per-rank expert tensor shapes."""
    e, a, b = shapes[0]
    return (e, a // 2, b) if kind == "w13" else (e, b, a)


def moe_mesh_problem(kind, src_shapes, dst_shapes, *, src_ep=None, dst_ep=None):
    world = len(src_shapes)
    se, si, sh = moe_mesh(kind, src_shapes)
    de, di, dh = moe_mesh(kind, dst_shapes)
    if any(list(s) != list(src_shapes[0]) for s in src_shapes) or \
       any(list(d) != list(dst_shapes[0]) for d in dst_shapes):
        return "ranks hold different shapes"
    if sh != dh:
        return f"hidden {sh} vs {dh}"
    r_src_ep, r_dst_ep = moe_ep(se, si, de, di, world, src_ep=src_ep, dst_ep=dst_ep)
    if r_src_ep is None:
        return f"source {se}x{si}, target {de}x{di} on {world} ranks fit no (moe_tp, ep) mesh"
    return None


def moe_ep(src_e, src_i, dst_e, dst_i, world, *, src_ep=None, dst_ep=None):
    """(src_ep, dst_ep) such that experts * ep and intermediate * moe_tp agree on both sides.

    When `src_ep` and/or `dst_ep` are provided (e.g. from `WeightCacheKey`), constrains the
    search to those values so EP->EP placement swaps and 4-GPU meshes do not pick an ambiguous
    smaller divisor pair.
    """
    src_candidates = [int(src_ep)] if src_ep is not None else list(range(1, world + 1))
    dst_candidates = [int(dst_ep)] if dst_ep is not None else list(range(1, world + 1))
    for s_ep in src_candidates:
        if s_ep < 1 or world % s_ep:
            continue
        for d_ep in dst_candidates:
            if d_ep < 1 or world % d_ep:
                continue
            if (src_e * s_ep == dst_e * d_ep and
                    src_i * (world // s_ep) == dst_i * (world // d_ep)):
                return s_ep, d_ep
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


def moe_transpose(src, kind, src_ep, dst_ep, devices=None, placement="linear", *,
                  src_placement=None, dst_placement=None):
    """Checkpoint-layout expert tensors, one per source rank ([E/src_ep, 2*I/src_tp, H] for w13 with
    gate rows before up rows, [E/src_ep, H, I/src_tp] for w2) -> one per target rank for the
    (world/dst_ep, dst_ep) mesh. Rank r is ep rank r // moe_tp and moe-tp rank r % moe_tp (vLLM's
    order). `devices[r]` is where target rank r's tensor is built."""
    import torch
    s_place = src_placement or placement
    d_place = dst_placement or placement
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
            ep_d, slot_d = map_global_expert_to_ep(g, num_experts, dst_ep, d_place)
            if ep_d != d_ep:
                continue
            ep_s, slot_s = map_global_expert_to_ep(g, num_experts, src_ep, s_place)
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
    found = BACKEND_LOG.findall(corelab.read_text(path, missing_ok=False))
    if not found:
        raise ValueError(f"{path}: no 'Using <backend> Unquantized MoE backend' line")
    if found[-1] not in LOG_NAMES:
        raise ValueError(f"{path}: MoE backend {found[-1]!r} has no layout codec "
                         f"(known: {sorted(LOG_NAMES)})")
    return LOG_NAMES[found[-1]]


# ---------------------------------------------------------------------------------------------
# Holder groups over CUDA IPC (vLLM's own key and protocol, as an engine dials them)
# ---------------------------------------------------------------------------------------------

def engine_config(argv):
    """vLLM's VllmConfig for a `vllm serve` argv (model positional or --model), parsed with the
    parser `vllm serve` uses (vllm_args.build) and built as `vllm serve` builds it."""
    import contextlib
    import vllm_args
    from vllm.engine.arg_utils import EngineArgs
    parser, _ = vllm_args.build()
    with contextlib.redirect_stdout(sys.stderr):
        ns = parser.parse_args(list(argv))
        if getattr(ns, "model_tag", None):
            ns.model = ns.model_tag
        return EngineArgs.from_cli_args(ns).create_engine_config()


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
        entries = corelab.weight_cache_get_state(path, key, label=f"holder rank {r}")
        ranks.append({n: (e.kind, e.rebuild(r)) for n, e in entries.items() if e.ipc_args is not None})
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


def _sync_all(torch):
    """Wait for every GPU: torch.cuda.synchronize() waits for the current device only."""
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def execute(src, dst, p, src_layout, dst_layout, *, src_ep=None, dst_ep=None,
            src_placement="linear", dst_placement="linear"):
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
    _sync_all(torch)
    t1 = time.perf_counter()
    for name, kind in p["moe"].items():
        s_shapes = [list(src[r][name][1].shape) for r in range(world)]
        d_shapes = [list(dst[r][name][1].shape) for r in range(world)]
        se, si, _ = moe_mesh(kind, s_shapes)
        de, di, _ = moe_mesh(kind, d_shapes)
        r_src_ep, r_dst_ep = moe_ep(se, si, de, di, world, src_ep=src_ep, dst_ep=dst_ep)
        if r_src_ep is None:
            raise ValueError(f"{name}: source {se}x{si}, target {de}x{di} on {world} ranks fit no "
                             f"(moe_tp, ep) mesh for src_ep={src_ep}, dst_ep={dst_ep}")
        canon = [src_layout.decode(src[r][name][1], kind) for r in range(world)]
        outs = moe_transpose(canon, kind, r_src_ep, r_dst_ep,
                             devices=[dst[r][name][1].device for r in range(world)],
                             src_placement=src_placement, dst_placement=dst_placement)
        for r in range(world):
            d = dst[r][name][1]
            d.copy_(dst_layout.encode(outs[r], kind))
            rep["moe_bytes"] += _bytes(d)
        # bytes whose source rank differs from their target rank (NVLink peer copies)
        cross_rank = r_src_ep != r_dst_ep or (r_src_ep > 1 and src_placement != dst_placement)
        rep["moe_cross_gpu_bytes"] += sum(_bytes(o) for o in outs) * (world - 1) // world \
            if cross_rank else 0
        del canon, outs
    _sync_all(torch)
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


class Phases:
    def __init__(self):
        self.seconds, self.t = {}, time.perf_counter()

    def __call__(self, name):
        now = time.perf_counter()
        self.seconds[name] = round(now - self.t, 3)
        self.t = now


def reshard_groups(src_dir, src_keys, dst_dir, dst_keys, src_layout, dst_layout,
                   verify_dir=None, phase=None):
    """Map both groups, plan, write the target in place, optionally verify. (code, report):
    code 0 ok, 2 refused before writing (report["findings"]), 1 verification failed."""
    import torch
    phase = phase or Phases()
    rep = {"layouts": {"src": src_layout.backend, "dst": dst_layout.backend}}
    if len(src_keys) != len(dst_keys) or len(src_keys) > torch.cuda.device_count():
        rep["findings"] = [_finding("reshard_live_world",
                                    f"{len(src_keys)} -> {len(dst_keys)} ranks on {torch.cuda.device_count()} GPUs",
                                    "the same GPUs, one rank each")]
        return 2, rep
    src_ep, src_placement = _key_moe_mesh(src_keys, len(src_keys))
    dst_ep, dst_placement = _key_moe_mesh(dst_keys, len(dst_keys))
    rep["placement"] = {"src": src_placement, "dst": dst_placement}
    src, dst = map_group(src_dir, src_keys), map_group(dst_dir, dst_keys)
    phase("map")
    p = plan(metadata(src), metadata(dst), src_ep=src_ep, dst_ep=dst_ep)
    rep["plan"] = {k: len(v) for k, v in p.items() if k != "findings"}
    rep["ignored_src_buffers"] = p["ignored_src"]
    phase("plan")
    if p["findings"]:
        rep["findings"] = p["findings"][:20]
        rep["note"] = "refused before writing; target group unchanged"
        return 2, rep
    rep.update(execute(src, dst, p, src_layout, dst_layout, src_ep=src_ep, dst_ep=dst_ep,
                       src_placement=src_placement, dst_placement=dst_placement))
    phase("write")
    rep["buffers_differ_from_src"] = buffers_differ(src, dst, p["keep"])[:20]
    if verify_dir:
        rep["verify"] = verify(dst, map_group(verify_dir, dst_keys))
        phase("verify")
        if not rep["verify"]["ok"]:
            return 1, rep
    return 0, rep


def run(a):
    """`run`: both groups already up (the lab path; flags given per group)."""
    phase = Phases()
    common = shlex.split(a.vllm_args)
    src_keys = group_keys(engine_config(["--model", a.model, *shlex.split(a.src_args), *common]))
    dst_keys = group_keys(engine_config(["--model", a.model, *shlex.split(a.dst_args), *common]))
    src_layout = MoeLayout(a.src_moe_layout or moe_layout_from_log(a.src_log))
    dst_layout = MoeLayout(a.dst_moe_layout or moe_layout_from_log(a.dst_log))
    phase("config")
    code, rep = reshard_groups(a.src_socket_dir, src_keys, a.dst_socket_dir, dst_keys, src_layout,
                               dst_layout, a.verify_socket_dir, phase)
    rep["phases_s"] = phase.seconds
    if code != 2:
        rep["reshard_s"] = round(sum(phase.seconds[k] for k in ("map", "plan", "write")), 3)
    print(json.dumps(rep))
    return code


# ---------------------------------------------------------------------------------------------
# ensure: the launcher's entry point (engine.py reconfigure). Make the holder group for a target
# `vllm serve` argv exist and hold the running engine's weights, without reading the checkpoint.
# ---------------------------------------------------------------------------------------------

# The weight-key fields a holder-to-holder reshard can change (vllm_args.RESHARD_FIELDS subset).
RESHARDABLE_FIELDS = ("enable_expert_parallel", "expert_placement_strategy")
# `vllm serve` flags a holder (`vllm preload`) must not get: the engine's loader, worker and
# memory budget, and the server's own flags (EngineArgs does not know the latter anyway).
NOT_HOLDER_FLAGS = {"--model", "--load-format", "--model-loader-extra-config", "--worker-cls",
                    "--worker-extension-cls", "--gpu-memory-utilization", "--kv-cache-memory-bytes",
                    "--weight-cache-socket-dir"}
EP_DIR = re.compile(r"\.ep(rr)?$")


def reshardable(reconfig):
    """True when the only WeightCacheKey fields a reconfiguration changes are the MoE expert
    sharding ones (TP <-> EP on the same GPUs); vllm_args.classify_reconfig's report."""
    changed = (reconfig or {}).get("weight_key_changed") or []
    return bool(changed) and set(changed) <= set(RESHARDABLE_FIELDS)


def group_dir(socket_dir, args):
    """Socket folder of the holder group for an engine with parsed `args`, next to a group of
    the same model: `<base>` without EP, `<base>.ep` (linear) or `<base>.eprr` (round robin)
    with it. Deterministic, so the way back finds the group it came from."""
    base = EP_DIR.sub("", socket_dir.rstrip("/"))
    if not args.get("enable_expert_parallel"):
        return base
    return base + (".eprr" if args.get("expert_placement_strategy") == "round_robin" else ".ep")


def holder_argv(vargv):
    """`vllm preload` flags for the engine argv `vargv`: the model and every EngineArgs flag the
    argv sets, minus NOT_HOLDER_FLAGS. vLLM's own parser decides what EngineArgs knows."""
    import vllm_args
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    known = set(EngineArgs.add_cli_args(FlexibleArgumentParser())._option_string_actions)
    model, items = vllm_args._tokenize_argv(vargv)
    out = ["--model", model]
    for opt, chunk in items:
        if opt is None or opt in NOT_HOLDER_FLAGS:
            continue
        raw = chunk[0].split("=", 1)[0]
        raw = ("--" + raw[2:].replace("_", "-")) if raw.startswith("--") else raw
        if raw in known:
            out += chunk
    return out


def _record_path(state_dir, socket_dir):
    import hashlib
    h = hashlib.sha1(os.path.abspath(socket_dir).encode()).hexdigest()[:12]
    return Path(state_dir) / f"group_{h}.json"


_read_record = ipc.read_json
_write_record = ipc.atomic_write_json


def stop_group(pid, grace=15.0):
    """Stop a holder group this module started (its pid leads its session)."""
    import weights
    return weights.stop_session(pid, grace)


def _probe_group(socket_dir, keys):
    """"ok" when every rank's daemon answers with these keys, "absent" when a socket does not
    answer, else the daemon's refusal (a group with other flags at that folder)."""
    try:
        map_group(socket_dir, keys)
        return "ok"
    except (FileNotFoundError, ConnectionRefusedError):
        return "absent"
    except RuntimeError as e:
        return str(e)


def ensure(a):
    import weights
    phase = Phases()
    rep = {"socket_dir": a.dst_socket_dir}
    src_vargv, dst_vargv = json.loads(a.src_vargv), json.loads(a.dst_vargv)
    src_keys = group_keys(engine_config(src_vargv))
    dst_keys = group_keys(engine_config(dst_vargv))
    phase("config")
    rec_path = _record_path(a.state_dir, a.dst_socket_dir)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec = _read_record(rec_path)
    state = _probe_group(a.dst_socket_dir, dst_keys)
    phase("probe_target")

    def done(code, **extra):
        rep.update(extra, phases_s=phase.seconds)
        print(json.dumps(rep))
        return code
    if state == "ok" and (rec is None or rec.get("filled")):
        # a group loaded from disk (not ours), or one we filled: it holds these weights
        return done(0, target="reused", pid=(rec or {}).get("pid"),
                    filled_from=(rec or {}).get("filled_from"))
    if state not in ("ok", "absent") and rec is None:
        return done(2, findings=[_finding(
            "weights_reshard_target", state, f"no group at {a.dst_socket_dir}, or one with the target flags",
            why="another group owns that folder; it is not ours to stop",
        )])
    if rec is not None and rec.get("pid"):
        stop_group(rec["pid"])
    if (finding := weights.long_socket_path_finding(a.dst_socket_dir)):
        return done(2, findings=[finding])
    os.makedirs(a.dst_socket_dir, exist_ok=True)
    log = str(rec_path.with_suffix(".log"))
    cmd = [a.vllm, "preload", *holder_argv(dst_vargv), "--weight-cache-socket-dir",
           a.dst_socket_dir, "--load-format", "dummy"]
    proc = corelab.spawn_logged(cmd, log)
    rec = {"pid": proc.pid, "socket_dir": a.dst_socket_dir, "cmd": cmd, "log": log,
           "filled": False, "started": time.time()}
    _write_record(rec_path, rec)
    ready = weights.wait_ready(log, proc.pid, timeout=a.timeout)
    phase("start_target")
    rep.update(pid=proc.pid, log=log, cmd=cmd)

    def fail(**extra):
        stop_group(proc.pid)
        rec_path.unlink(missing_ok=True)
        return done(2, stopped_target=True, **extra)
    if not ready["ready"]:
        return fail(findings=[_finding("weights_reshard_target", ready, "the target group READY", why="see its log")])
    try:
        src_layout, dst_layout = MoeLayout(moe_layout_from_log(a.src_log)), MoeLayout(moe_layout_from_log(log))
        code, res = reshard_groups(a.src_socket_dir, src_keys, a.dst_socket_dir, dst_keys,
                                   src_layout, dst_layout, a.verify_socket_dir, phase)
    except Exception as e:                       # noqa: BLE001 - reported, old engine untouched
        return fail(findings=[_finding("weights_reshard", f"{type(e).__name__}: {e}", "the target group filled")])
    rep["reshard"] = res
    if code != 0:
        return fail(findings=res.get("findings") or [{"check": "weights_reshard_verify", "observed": res.get("verify")}])
    rec.update(filled=True, filled_from=a.src_socket_dir)
    _write_record(rec_path, rec)
    return done(0, target="filled")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="reshard between two running groups (flags per group)")
    e = sub.add_parser("ensure", help="make the target engine's group exist and hold the source engine's weights")
    for sp in (r, e):
        for flag, kw in (("--src-socket-dir", dict(required=True)), ("--dst-socket-dir", dict(required=True)),
                         ("--verify-socket-dir", dict(help="a group loaded from disk with target flags"))):
            sp.add_argument(flag, **kw)
    for flag, kw in (("--model", dict(required=True)),
                     ("--src-args", dict(required=True, help='vLLM flags of the source group, e.g. "-tp 2"')),
                     ("--dst-args", dict(required=True)),
                     ("--vllm-args", dict(default="", help="flags both groups share (max-model-len, ...)"))):
        r.add_argument(flag, **kw)
    for side in ("src", "dst"):
        g = r.add_mutually_exclusive_group(required=True)
        g.add_argument(f"--{side}-moe-layout", choices=LAYOUTS)
        g.add_argument(f"--{side}-log", help="the group's log, to read its MoE backend")
    for flag, kw in (("--src-vargv", dict(required=True, help="JSON list: the running engine's argv")),
                     ("--src-log", dict(required=True, help="the running engine's log (MoE backend)")),
                     ("--dst-vargv", dict(required=True, help="JSON list: the target engine's argv")),
                     ("--state-dir", dict(required=True, help="where group records and logs go")),
                     ("--vllm", dict(default="vllm")), ("--timeout", dict(type=float, default=1800))):
        e.add_argument(flag, **kw)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else ensure(a)


if __name__ == "__main__":
    sys.exit(main())
