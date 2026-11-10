"""GPU check for snapshot_replay.py on a real vLLM V1 GPUModelRunner (audit row 5).

One process, one LLM (in-process engine core: VLLM_ENABLE_V1_MULTIPROCESSING=0), driven step by
step. The runner's execute_model / sample_tokens are wrapped (instance attributes) so that:

  step K (before it runs)   capture a tail snapshot (horizon H) and a GPU reference copy of
                            every block each layer's own KV-cache group will touch
  steps K..K+H-1            run normally: pass A (sampled tokens + logprobs recorded, and the
                            SchedulerOutputs that produced them)
  after step K+H-1          corrupt those blocks (noise from a private generator), restore the
                            snapshot, compare every layer's blocks with the reference (bytes),
                            re-execute the H recorded SchedulerOutputs on the runner: pass B
  then                      the engine continues to the end

Pass B must equal pass A token for token and logprob for logprob (sampling is random,
temperature 1, so this also needs the CUDA RNG restored), and the final completions must
equal a fresh run's (mode=fresh, same seed). The scheduler is never rewound: the replay feeds
the runner the SchedulerOutputs it already saw, which is the one sound use of a runner-level
snapshot (see snapshot_replay.py's status note).

Every time is time.perf_counter (monotonic) after torch.cuda.synchronize().

Usage:
  python3 gpu_check_snapshot_replay.py --model M --mode fresh  --out fresh.json
  python3 gpu_check_snapshot_replay.py --model M --mode replay --out replay.json \
      [--module PATH/TO/snapshot_replay.py]   # e.g. origin/main's copy, for the before row
"""
import argparse, copy, importlib.util, json, os, sys, time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve string annotations through it
    spec.loader.exec_module(mod)
    return mod


def prompts(n, approx_tokens):
    words = ("the keeper of the lighthouse wrote in his log every night about ships storms "
             "gulls tides lanterns and the long patient sea ").split()
    out = []
    for i in range(n):
        body = " ".join(words[(i * 7 + j) % len(words)] for j in range(approx_tokens))
        out.append(f"Story {i}. {body}\nContinue the story:")
    return out


def tolist(x):
    return x.tolist() if hasattr(x, "tolist") else [tolist(v) for v in x] if isinstance(x, (list, tuple)) else x


def step_record(out):
    lp = out.logprobs
    return {"req_ids": list(out.req_ids), "sampled": [tolist(t) for t in out.sampled_token_ids],
            "logprob_ids": tolist(lp.logprob_token_ids) if lp is not None else None,
            "logprobs": tolist(lp.logprobs) if lp is not None else None}


def attn_backend(runner):
    try:
        return sorted({g.backend.get_name() for gs in runner.attn_groups for g in gs})
    except Exception as e:   # noqa: BLE001 - informational only
        return f"unknown ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["fresh", "replay"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--module", default=str(HERE / "snapshot_replay.py"))
    ap.add_argument("--reqs", type=int, default=4)
    ap.add_argument("--prompt-words", type=int, default=1400)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--capture-step", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=32)   # FlashInfer refuses 16 at head 256
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(HERE))
    fixed = load_module(HERE / "snapshot_replay.py", "snapshot_replay_fixed")  # checker's group map
    sr = load_module(a.module, "snapshot_replay_under_test")                     # module under test

    def sync_ms():
        torch.cuda.synchronize()
        return time.perf_counter() * 1000.0

    llm = LLM(model=a.model, seed=0, max_model_len=4096, gpu_memory_utilization=0.4,
              enable_prefix_caching=False, async_scheduling=False,
              enforce_eager=a.enforce_eager, max_num_seqs=16, block_size=a.block_size)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    groups = [{"layers": list(g.layer_names), "spec": type(g.kv_cache_spec).__name__,
               "sliding_window": getattr(g.kv_cache_spec, "sliding_window", None),
               "block_size": g.kv_cache_spec.block_size}
              for g in runner.kv_cache_config.kv_cache_groups]
    res = {"model": a.model, "mode": a.mode, "module": a.module, "vllm": __import__("vllm").__version__,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "kv_cache_groups": groups,
           "num_kv_cache_tensors": len(runner.kv_cache_config.kv_cache_tensors),
           "attn_backend": attn_backend(runner),
           "clock": "time.perf_counter after torch.cuda.synchronize, ms", "capture_step": a.capture_step,
           "horizon": a.horizon}

    state = {"call": 0, "snap": None, "recorded": [], "passA": [], "ref": None, "t": {}}
    orig_exec, orig_sample = runner.execute_model, runner.sample_tokens
    snapper = sr.KVStateSnapshotter(kv_mode="tail", horizon_steps=a.horizon, max_step_tokens=1)
    K, H = a.capture_step, a.horizon

    def layer_blocks(snap):
        """[(key, tensor, axis, block ids of the layer's own group)] using the fixed map."""
        out = []
        for key, kv, g in fixed.kv_layer_groups(runner):
            ids = [b for b in snap.dirty_block_ids_per_group[g] if b > 0]
            out.append((key, kv, fixed._infer_block_axis(kv), ids, g))
        return out

    def run_one(so, grammar):
        out = orig_exec(so)
        if out is None:
            out = orig_sample(grammar)
        return out

    def exec_wrap(so, *rest):
        i = state["call"]
        if a.mode == "replay" and i == K:
            state["t"]["passA_start"] = sync_ms()
            t0 = sync_ms()
            snap = snapper.capture(runner)
            state["t"]["capture_ms"] = sync_ms() - t0
            state["snap"] = snap
            state["ref"] = [(key, kv.index_select(axis, torch.tensor(ids, device=kv.device)).clone())
                            for key, kv, axis, ids, _ in layer_blocks(snap)]
            state["t"]["passA_start"] = sync_ms()
        if a.mode == "replay" and K <= i < K + H:
            state["recorded"].append([copy.deepcopy(so), None])
        return orig_exec(so, *rest)

    def sample_wrap(grammar, *rest):
        i = state["call"]
        out = orig_sample(grammar, *rest)
        state["call"] += 1
        if a.mode == "replay" and K <= i < K + H:
            state["recorded"][i - K][1] = copy.deepcopy(grammar)
            state["passA"].append(step_record(out))
        if a.mode == "replay" and i == K + H - 1:
            state["t"]["passA_ms"] = sync_ms() - state["t"]["passA_start"]
            replay_check(out)
        return out

    def replay_check(_):
        snap = state["snap"]
        lb = layer_blocks(snap)
        gen = torch.Generator(device="cuda").manual_seed(1234)
        t0 = sync_ms()
        for key, kv, axis, ids, _ in lb:
            if ids:
                idx = torch.tensor(ids, device=kv.device)
                shape = list(kv.shape)
                shape[axis] = len(ids)
                noise = torch.randn(shape, generator=gen, device=kv.device, dtype=torch.float32).to(kv.dtype)
                kv.index_copy_(axis, idx, noise)
        state["t"]["corrupt_ms"] = sync_ms() - t0
        t0 = sync_ms()
        snapper.restore(runner, snap)
        state["t"]["restore_ms"] = sync_ms() - t0
        mism = []
        ref = dict((str(k), v) for k, v in state["ref"])
        for key, kv, axis, ids, g in lb:
            now = kv.index_select(axis, torch.tensor(ids, device=kv.device))
            if not torch.equal(now, ref[str(key)]):
                mism.append({"layer": str(key), "group": g, "blocks": len(ids)})
        res["blocks_check"] = {"layers": len(lb), "layers_mismatched": len(mism),
                               "mismatched": mism[:12],
                               "blocks_per_group": [len(x) for x in snap.dirty_block_ids_per_group]}
        t0 = sync_ms()
        passB, err = [], None
        try:
            for so, grammar in state["recorded"]:
                passB.append(step_record(run_one(copy.deepcopy(so), grammar)))
        except Exception as e:     # the before row may break here; record it
            err = f"{type(e).__name__}: {e}"
        state["t"]["replay_ms"] = sync_ms() - t0
        A = state["passA"]
        tok_eq = sum(1 for x, y in zip(A, passB) if x["sampled"] == y["sampled"])
        lp_eq = sum(1 for x, y in zip(A, passB) if x["logprobs"] == y["logprobs"])
        first_div = next((i for i, (x, y) in enumerate(zip(A, passB)) if x["sampled"] != y["sampled"]), None)
        res["replay_check"] = {"steps": H, "steps_replayed": len(passB), "error": err,
                               "steps_token_identical": tok_eq, "steps_logprob_equal": lp_eq,
                               "first_divergent_step": first_div,
                               "passA_tokens": [s["sampled"] for s in A],
                               "passB_tokens": [s["sampled"] for s in passB]}
        res["snapshot_summary"] = {k: v for k, v in snap.summary().items() if k != "findings"}
        res["timing_ms"] = {**{k: round(v, 4) for k, v in state["t"].items() if not k.endswith("_start")},
                            "module_timing_ms": {k: round(v, 4) for k, v in getattr(snap, "timing", {}).items()},
                            "passA_per_step_ms": round(state["t"]["passA_ms"] / H, 4),
                            "replay_per_step_ms": round(state["t"]["replay_ms"] / max(1, len(passB)), 4)}
        res["continue_engine"] = err is None and tok_eq == H
        if not res["continue_engine"]:
            raise SystemExit("replay diverged; the runner no longer matches the scheduler: stop")
        # Steady-state cost: capture + restore of the current state, repeated (a restore of a
        # snapshot taken this instant changes nothing, so the engine continues as before and
        # its final output must still equal the fresh run's).
        reps = []
        for _ in range(a.repeats):
            t0 = sync_ms()
            s2 = snapper.capture(runner)
            t1 = sync_ms()
            snapper.restore(runner, s2)
            t2 = sync_ms()
            reps.append({"capture_ms": round(t1 - t0, 4), "restore_ms": round(t2 - t1, 4),
                         **{k: round(v, 4) for k, v in getattr(s2, "timing", {}).items()},
                         "bytes": s2.total_backed_up_bytes})
        res["timing_ms"]["steady_state_repeats"] = reps

    runner.execute_model, runner.sample_tokens = exec_wrap, sample_wrap
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, ignore_eos=True,
                        logprobs=1)
    for i, p in enumerate(prompts(a.reqs, a.prompt_words)):
        engine.add_request(f"r{i}", p, sp)
    finals, t0 = {}, time.perf_counter()
    try:
        while engine.has_unfinished_requests():
            for o in engine.step():
                if o.finished:
                    c = o.outputs[0]
                    finals[o.request_id] = {"tokens": list(c.token_ids),
                                            "logprobs": [float(d[t].logprob) for d, t in zip(c.logprobs, c.token_ids)],
                                            "prompt_len": len(o.prompt_token_ids)}
    except SystemExit as e:
        res["stopped"] = str(e)
    res["wall_s"] = round(time.perf_counter() - t0, 3)
    res["engine_steps"] = state["call"]
    res["finals"] = finals
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k not in ("finals",)}, default=str)[:4000])


if __name__ == "__main__":
    main()
