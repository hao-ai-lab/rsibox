"""Joint KernelService (S7) x KernelLab multi-slot B200 benchmark & receipt generator."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from adapter import kernel_ops
from adapter.lab import KernelLab
from rsibox import KernelService, PoolSlot, SandboxPool, harness


def _percentile(values: list[float], frac: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, math.ceil(frac * len(ordered)) - 1))])


def _resolve_gpu_routing(gpu_id: int) -> tuple[str, int]:
    """Map requested physical `gpu_id` against `CUDA_VISIBLE_DEVICES` -> `(torch_device, service_gpu_id)`."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None:
        return f"cuda:{gpu_id}", gpu_id
    tokens = [tok.strip() for tok in vis.split(",") if tok.strip()]
    if not tokens:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty")
    if str(gpu_id) in tokens:
        return f"cuda:{tokens.index(str(gpu_id))}", gpu_id
    if len(tokens) == 1:
        return "cuda:0", (int(tokens[0]) if tokens[0].isdigit() else gpu_id)
    raise ValueError(f"--gpu-id {gpu_id} is not exposed by CUDA_VISIBLE_DEVICES={vis!r}")


def _worker_main(cfg: dict[str, Any], barrier: Any, start_q: Any, out_q: Any) -> None:
    idx, backend, mode = int(cfg["idx"]), str(cfg["backend"]), str(cfg["mode"])
    state_dir, numel, bs, iters = str(cfg["state_dir"]), int(cfg["numel"]), int(cfg["block_size"]), int(cfg["iters"])
    gpu_id, evals, build_root = int(cfg["gpu_id"]), int(cfg["evals_per_worker"]), str(cfg["build_root"])
    cache_dir = os.path.join(state_dir, f"triton_cache_{idx}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir

    svc = KernelService(state_dir=state_dir, pin_cpus=True, cores_per_gpu=4, gpus_per_node=4) if mode == "kernel_service" else None
    lab: KernelLab | None = None
    try:
        cuda_device, svc_gpu_id = _resolve_gpu_routing(gpu_id)
        lab_cfg: dict[str, Any] = {"backend": backend, "numel": numel, "block_size": bs, "iters": iters, "device": cuda_device}
        if backend == "cuda":
            lab_cfg["build_root"] = build_root
        lab = KernelLab(lab_cfg)
        assert lab._worker is not None
        slot_id = f"slot-{idx}"
        if svc is not None:
            wpid = int(lab._worker["proc"].pid)
            svc.register_slot(svc_gpu_id, slot_id, pid=wpid, pgid=os.getpgid(wpid))

        import torch

        cap = torch.cuda.get_device_capability(cuda_device)
        arch_int, arch_str = (int(cap[0]) * 10) + int(cap[1]), f"{cap[0]}{cap[1]}"

        base_chk = lab.check({"numel": numel, "block_size": bs, "iters": 2})
        assert base_chk["ok"], base_chk
        if backend == "triton":
            import adapter.triton_kernels as tk
            import triton.compiler as tc
            from triton.backends.compiler import GPUTarget

            src = lab._base_source("adapter.triton_kernels")
            cand_src = harness.edit_once(src, "gain = 1.0", "gain = 1.0 + 0.0")
            patch = {"files": {"adapter.triton_kernels": cand_src}}
        else:
            src = lab._base_source("adapter.kernel_ops")
            patch = {"files": {"adapter.kernel_ops": harness.edit_once(src, "scale = 1.0", "scale = 1.0 + 0.0")}}
        assert lab.install(patch)["ok"]
        assert lab.check({"numel": numel, "block_size": bs, "iters": 2})["ok"]
        assert lab.uninstall()["ok"]

        barrier.wait(timeout=120)
        start_q.put(time.perf_counter())
        lats, diffs, freezes, cpus = [], [], set(), set()
        for it in range(evals):
            wl: dict[str, Any] = {"numel": numel, "block_size": bs, "iters": iters}
            if backend == "triton":
                iter_cache = os.path.join(cache_dir, f"cand_{it}")
                os.makedirs(iter_cache, exist_ok=True)
                os.environ["TRITON_CACHE_DIR"] = iter_cache
                ast_src = tc.ASTSource(  # type: ignore[possibly-unbound]
                    fn=tk.fused_affine_silu_kernel,  # type: ignore[possibly-unbound]
                    signature={"x_ptr": "*fp32", "bias_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32", "alpha": "fp32", "beta": "fp32", "BLOCK": "constexpr"},
                    constexprs={"BLOCK": bs},
                )
                tc.compile(ast_src, target=GPUTarget("cuda", arch_int, 32))  # type: ignore[possibly-unbound]
                shutil.rmtree(iter_cache, ignore_errors=True)
                os.environ["TRITON_CACHE_DIR"] = cache_dir
            else:
                cu_src = kernel_ops.cuda_kernel_source() + f"\n/* slot_{idx}_cand_{it} */"
                built = kernel_ops.compile_and_load_cuda(cu_src, build_root=build_root, arch=arch_str, use_cache=False)
                assert os.path.isfile(built["so"])
                wl["cuda_source"] = cu_src

            assert lab.install(patch)["ok"]
            if svc is not None:
                with svc.critical_region(svc_gpu_id, slot_id) as ticket:
                    chk = lab.check(wl)
                    freezes.add(ticket.freeze_mode)
                    if ticket.pinned_cpus:
                        cpus.add(ticket.pinned_cpus)
            else:
                chk = lab.check(wl)
            assert chk["ok"] and chk["proven"] and chk["witness"], chk
            diffs.append(float(chk["output"]["max_abs_diff"]))
            lats.append(float(chk["latency_ms"]))
            assert lab.uninstall()["ok"]

        out_q.put({"lats": lats, "diffs": diffs, "freeze": sorted(freezes), "cpus": [list(c) for c in sorted(cpus)], "end": time.perf_counter()})
    finally:
        if lab is not None:
            with contextlib.suppress(Exception):
                lab.close()
        if svc is not None:
            with contextlib.suppress(Exception):
                svc.close()


def _run_cell(backend: str, mode: str, k: int, numel: int, bs: int, iters: int, evals_per_worker: int, gpu_id: int, timeout_s: float = 180.0) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        build_root = os.path.join(tmp, "cu_build")
        os.makedirs(build_root, exist_ok=True)
        bar, sq, oq = ctx.Barrier(k), ctx.Queue(), ctx.Queue()
        procs = [
            ctx.Process(
                target=_worker_main,
                args=(
                    {
                        "idx": i, "backend": backend, "mode": mode, "state_dir": tmp, "build_root": build_root,
                        "numel": numel, "block_size": bs, "iters": iters, "gpu_id": gpu_id, "evals_per_worker": evals_per_worker,
                    },
                    bar, sq, oq,
                ),
            )
            for i in range(k)
        ]
        res: list[dict[str, Any]] = []
        starts: list[float] = []
        try:
            for p in procs:
                p.start()
            deadline = time.monotonic() + timeout_s
            while len(res) < k:
                while True:
                    try:
                        res.append(oq.get_nowait())
                    except queue.Empty:
                        break
                if len(res) >= k:
                    break
                for p in procs:
                    if not p.is_alive() and p.exitcode not in (None, 0):
                        raise RuntimeError(f"worker pid={p.pid} exited early with rc={p.exitcode}")
                if not any(p.is_alive() for p in procs):
                    while True:
                        try:
                            res.append(oq.get_nowait())
                        except queue.Empty:
                            break
                    if len(res) < k:
                        raise RuntimeError(f"workers exited before reporting all results ({len(res)}/{k})")
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for {k} worker results ({len(res)}/{k})")
                time.sleep(0.02)
            while len(starts) < k:
                starts.append(float(sq.get(timeout=5.0)))
            for p in procs:
                p.join(timeout=30.0)
                if p.exitcode != 0:
                    raise RuntimeError(f"worker pid={p.pid} exited with rc={p.exitcode}")
        finally:
            with contextlib.suppress(Exception):
                bar.abort()
            for p in procs:
                if p.is_alive():
                    with contextlib.suppress(Exception):
                        p.terminate()
                    with contextlib.suppress(Exception):
                        p.join(timeout=2.0)
                if p.is_alive():
                    with contextlib.suppress(Exception):
                        p.kill()
                    with contextlib.suppress(Exception):
                        p.join(timeout=2.0)
                with contextlib.suppress(Exception):
                    p.close()
            for q in (sq, oq):
                with contextlib.suppress(Exception):
                    q.close()
                with contextlib.suppress(Exception):
                    q.join_thread()
    wall = max(r["end"] for r in res) - min(starts)
    lats = [x for r in res for x in r["lats"]]
    diffs = [x for r in res for x in r["diffs"]]
    return {
        "backend": backend, "mode": mode, "k": k, "shape": f"{numel}x{bs}", "iters": iters,
        "evals": len(lats), "warm_wall_s": round(wall, 4), "evals_per_s": round(len(lats) / wall, 3),
        "candidates_per_min": round((len(lats) / wall) * 60.0, 2),
        "p50_ms": round(_percentile(lats, 0.50), 4), "p95_ms": round(_percentile(lats, 0.95), 4),
        "max_abs_diff": max(diffs), "witness_nonzero": True,
        "freeze_modes": sorted({f for r in res for f in r["freeze"]}),
        "pinned_cpus": sorted({tuple(c) for r in res for c in r["cpus"]}),
    }


def _verify_pool_and_replay(gpu_id: int) -> dict[str, Any]:
    """Verify SandboxPool(kernel_service=svc, slots_per_gpu=2) + fresh_check parity + cold/warm timing."""
    import torch

    cuda_device, svc_gpu_id = _resolve_gpu_routing(gpu_id)
    cap = torch.cuda.get_device_capability(cuda_device)
    arch_str = f"sm_{cap[0]}{cap[1]}"
    with tempfile.TemporaryDirectory() as tmp:
        svc = KernelService(state_dir=tmp, pin_cpus=True, cores_per_gpu=4, gpus_per_node=4)
        labs: list[KernelLab] = []
        try:
            t_cold = time.perf_counter()
            for _ in range(2):
                labs.append(KernelLab({"backend": "triton", "numel": 16384, "block_size": 256, "iters": 200, "device": cuda_device}))
            cold_s = time.perf_counter() - t_cold
            slots = []
            for i, lab in enumerate(labs):
                assert lab._worker is not None
                wpid = int(lab._worker["proc"].pid)
                slots.append(PoolSlot(f"pool-{i}", lab, role="eval", metadata={"gpu_id": svc_gpu_id, "pid": wpid, "pgid": os.getpgid(wpid)}))
            with SandboxPool(slots, kernel_service=svc, slots_per_gpu=2, close_labs=False) as pool:
                src = labs[0]._base_source("adapter.triton_kernels")
                vis_patch = {"files": {"adapter.triton_kernels": harness.edit_once(src, "gain = 1.0", "gain = 2.0")}}
                t_warm = time.perf_counter()
                with pool.lease("agent-0") as lease:
                    assert lease.lab.install(vis_patch)["ok"]
                    with lease.critical_region() as ticket:
                        live = lease.lab.check({"numel": 16384, "block_size": 256, "iters": 200})
                    warm_s = time.perf_counter() - t_warm
                    fresh = lease.lab.fresh_check(vis_patch, {"numel": 16384, "block_size": 256, "iters": 200})
                    assert live["ok"] and live["proven"] and live["witness"], live
                    assert fresh["ok"] and fresh["output"] == live["output"], (fresh, live)
                    assert lease.lab.uninstall()["ok"]
                    base = lease.lab.check({"numel": 16384, "block_size": 256, "iters": 200})
                    assert base["output"]["max_abs_diff"] == 0.0, base
            return {
                "torch_version": getattr(torch, "__version__", "unknown"),
                "gpu_name": torch.cuda.get_device_name(cuda_device),
                "arch": arch_str, "service_gpu_id": svc_gpu_id,
                "cold_s": round(cold_s, 4), "warm_s": round(warm_s, 4),
                "pool_freeze_mode": ticket.freeze_mode, "pool_pinned_cpus": list(ticket.pinned_cpus),
                "replay_verified": True, "witness_nonzero": True, "base_max_abs_diff": float(base["output"]["max_abs_diff"]),
            }
        finally:
            for lab in labs:
                with contextlib.suppress(Exception):
                    lab.close()
            with contextlib.suppress(Exception):
                svc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Joint KernelService (S7) x KernelLab B200 benchmark")
    ap.add_argument("--gpu-id", type=int, default=3)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    meta = _verify_pool_and_replay(args.gpu_id)
    regimes = [
        ("triton", 16384, 256, 400, 5),
        ("triton", 1048576, 512, 120, 3),
        ("cuda", 16384, 256, 400, 3),
    ]
    rows: list[dict[str, Any]] = []
    for backend, numel, bs, iters, epw in regimes:
        base_row = _run_cell(backend, "exclusive", 1, numel, bs, iters, epw, args.gpu_id)
        base_row["throughput_speedup"] = 1.0
        base_row["p95_inflation_pct"] = 0.0
        rows.append(base_row)
        for k in (2, 4):
            for mode in ("unisolated", "kernel_service"):
                row = _run_cell(backend, mode, k, numel, bs, iters, epw, args.gpu_id)
                row["throughput_speedup"] = round(row["evals_per_s"] / base_row["evals_per_s"], 3)
                row["p95_inflation_pct"] = round(((row["p95_ms"] / base_row["p95_ms"]) - 1.0) * 100.0, 2)
                rows.append(row)

    receipt = {
        "kind": "gpu_verification_receipt",
        "lane": "wavec_joint_s7_kernel",
        "service": "KernelService (S7) x KernelLab",
        "gpu_model": meta["gpu_name"],
        "arch": meta["arch"],
        "gpus": [meta["service_gpu_id"]],
        "torch_version": meta["torch_version"],
        "witness_nonzero": meta["witness_nonzero"],
        "replay_verified": meta["replay_verified"],
        "shapes": ["16384x256_triton", "1048576x512_triton", "16384x256_cuda"],
        "cold_s": meta["cold_s"],
        "warm_s": meta["warm_s"],
        "sandbox_pool_check": {
            "freeze_mode": meta["pool_freeze_mode"],
            "pinned_cpus": meta["pool_pinned_cpus"],
            "base_max_abs_diff": meta["base_max_abs_diff"],
        },
        "benchmarks": rows,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
