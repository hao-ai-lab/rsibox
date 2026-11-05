"""Live golden tests of the kernel adapter against a running vLLM engine (GPU; results are receipts).

    python3 -m kernel_vllm.live_test --url http://127.0.0.1:<port> --dir <channel dir> \
        [--restart-s <warm restart seconds> --restart-source <how it was measured>] [--out <json>]

The engine must run with the host (FASTRSI_KERNEL_DIR) and the worker extension, greedy and
batch-invariant (VLLM_BATCH_INVARIANT=1). Standard library only; reads vLLM's base sources
from its install without importing it.

T0  base: one greedy request, kept for T3.
T1  while 6 streaming requests run, one transaction swaps a worker function
    (`PenaltiesState.apply` of the V2 model runner: ban the first token of "Paris" and
    " Paris"; `apply_all_penalties` never runs on vLLM 0.33's default runner) together with an
    engine-core function (`Scheduler.schedule`: a step counter). Checks: witnesses on both
    sides, no lost tokens (streamed tokens == usage == max_tokens, finish_reason "length"), no
    banned token streamed after the commit, prepare vs gap time, edit-to-run / edit-to-finish.
T2  a forced failure on the workers during validate: nobody commits, the engine keeps serving,
    the receipt says so.
T3  undo of T1: status undone, and T0's request gives the identical text again.
"""
import argparse
import importlib.util
import json
import os
import threading
import time
import urllib.request

from .client import send

PROMPTS = ["Write a long travel guide to Paris, France, district by district.",
           "Describe the history of Paris from the Romans to today.",
           "List twenty famous museums in Europe and say what each is known for.",
           "Explain how a city like Paris plans its public transport.",
           "Tell a long story about a cat who travels from Lyon to Paris.",
           "Explain the water cycle to a ten-year-old, in detail."]
SINGLE = "The capital of France is"


def vllm_src(rel):
    spec = importlib.util.find_spec("vllm")            # top level only: no import of vllm
    with open(os.path.join(os.path.dirname(spec.origin), rel)) as f:
        return f.read()


def line_of(src, needle):
    return next(i for i, s in enumerate(src.splitlines(), 1) if needle in s)


def post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def complete(url, model, prompt, max_tokens=48):
    r = post(url, "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                                      "temperature": 0.0, "repetition_penalty": 1.0001})
    return r["choices"][0]["text"]


def stream(url, model, prompt, max_tokens, out):
    """One streaming request; out gets (arrival time, token strings) per chunk, and the end."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "repetition_penalty": 1.0001, "stream": True, "logprobs": 0,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out.update(chunks=[], usage=None, finish=None, error=None)
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("usage"):
                    out["usage"] = d["usage"]
                for c in d.get("choices", []):
                    toks = (c.get("logprobs") or {}).get("tokens") or []
                    out["chunks"].append((time.time(), toks))
                    if c.get("finish_reason"):
                        out["finish"] = c["finish_reason"]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"


PEN = "v1/worker/gpu/sample/penalties.py"     # vLLM 0.33 runs the V2 model runner: its
PEN_MOD = "vllm.v1.worker.gpu.sample.penalties"  # penalties are PenaltiesState.apply, not
#                                                  vllm.v1.sample.ops.apply_all_penalties


def ban_patch(url, model, word):
    src = vllm_src(PEN)
    ids, strs = set(), set()
    for v in {word, " " + word}:
        toks = post(url, "/tokenize", {"model": model, "prompt": v,
                                       "add_special_tokens": False})["tokens"]
        ids.add(toks[0])
    for i in ids:
        strs.add(post(url, "/detokenize", {"model": model, "tokens": [i]})["prompt"])
    start = src.index("    def apply(self, logits")
    at = src.index("        return logits\n", src.index("        apply_penalties(", start))
    new = src[:at] + f'        logits[:, {sorted(ids)}] = float("-inf")  # fastrsi: ban {word!r}\n' \
        + src[at:]
    return src, new, sorted(ids), strs


def sched_patch():
    src = vllm_src("v1/core/sched/scheduler.py")
    old = "        self.current_step += 1\n"
    assert src.count(old) == 1
    new = src.replace(old, old + "        self.fastrsi_schedules = "
                                 "getattr(self, 'fastrsi_schedules', 0) + 1\n")
    return src, new


def text_hash(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--restart-s", type=float)
    ap.add_argument("--restart-source", default="")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out")
    ap.add_argument("--scan", default="auto", choices=["auto", "always"],
                    help="always: the pre-v2.2 locate (heap scans), for a before/after")
    a = ap.parse_args()
    url, m, res = a.url, a.model, {}
    baseline = ({"seconds": a.restart_s, "source": a.restart_source}
                if a.restart_s is not None else None)
    print("ping:", send(a.dir, {"op": "ping"})["ok"])

    # T0
    base_text = complete(url, m, SINGLE)
    print("T0 base:", repr(base_text))

    # T1
    p_src, p_new, ids, strs = ban_patch(url, m, "Paris")
    s_src, s_new = sched_patch()
    outs = [{} for _ in PROMPTS]
    ths = [threading.Thread(target=stream, args=(url, m, p, a.max_tokens, o))
           for p, o in zip(PROMPTS, outs)]
    for t in ths:
        t.start()
    time.sleep(3.0)                                    # streams are decoding now
    edit_time = time.time()
    plan = {"id": f"t1-{int(edit_time)}", "recipes": ["R0"],
            "predicted_resources": ["code objects"], "certainty": "L2",
            "modules": {
                PEN_MOD: {
                    "on": ["worker"], "source": p_new, "base_hash": text_hash(p_src), "scan": a.scan,
                    "targets": [["PenaltiesState.apply", line_of(p_src, "    def apply(self, logits")]]},
                "vllm.v1.core.sched.scheduler": {
                    "on": ["core"], "source": s_new, "base_hash": text_hash(s_src), "scan": a.scan,
                    "targets": [["Scheduler.schedule", line_of(s_src, "    def schedule(")]]}}}
    r1 = send(a.dir, {"op": "txn", "plan": plan, "edit_time": edit_time,
                      "baseline_restart": baseline})
    rec = r1.get("receipt") or {}
    for t in ths:
        t.join()
    pr = send(a.dir, {"op": "prove", "txn": plan["id"],
                      "golden": {"streams": len(outs), "banned_ids": ids}})
    commit_t = (rec.get("timeline") or {}).get("committed")
    lost, banned_after, max_gap = [], 0, 0.0
    for o in outs:
        n = sum(len(t) for _, t in o["chunks"])
        u = (o["usage"] or {}).get("completion_tokens")
        if o["error"] or n != u or u != a.max_tokens or o["finish"] != "length":
            lost.append({"streamed": n, "usage": u, "finish": o["finish"], "error": o["error"]})
        if commit_t:
            banned_after += sum(1 for ts, toks in o["chunks"] if ts > commit_t + 0.5
                                for tk in toks if tk in strs)
        times = [ts for ts, _ in o["chunks"]]
        max_gap = max([max_gap] + [b - x for x, b in zip(times, times[1:])])
    # chunk gaps while the transaction prepared (a stall shows as the gap ending in it)
    tl = rec.get("timeline") or {}
    ps, pd = tl.get("prepare_start"), tl.get("prepared")
    gaps_prep, gaps_before = [], []
    for o in outs:
        times = [ts for ts, _ in o["chunks"]]
        for x, b in zip(times, times[1:]):
            if ps and pd and ps <= b <= pd + 0.05:
                gaps_prep.append(b - x)
            elif ps and b < ps:
                gaps_before.append(b - x)
    gaps_before.sort()
    paris_before = sum(1 for o in outs for ts, toks in o["chunks"] if commit_t and ts < commit_t
                       for tk in toks if tk in strs)
    after_text = complete(url, m, SINGLE)
    res["T1"] = {
        "outcome": r1.get("outcome"), "edit_to_run_s": rec.get("edit_to_run_s"),
        "prepare_s": rec.get("prepare_s"), "gap_s": rec.get("gap_s"),
        "edit_to_finish_s": (pr.get("receipt") or {}).get("edit_to_finish_s"),
        "baseline_restart_s": baseline,
        "witness": (pr.get("receipt") or {}).get("witness"),
        "proof_findings": [(f.get("participant"), f["check"]) for f in
                           (pr.get("receipt") or {}).get("findings", [])],
        "streams": len(outs), "lost_token_streams": lost,
        "banned_tokens_streamed_before_commit": paris_before,
        "banned_tokens_streamed_after_commit": banned_after,
        "max_inter_chunk_gap_s": round(max_gap, 4),
        "scan": a.scan,
        "prepare_timings": {n: (p.get("prepare") or {}).get("timings") or [
            r.get("timings") for r in (p.get("prepare") or {}).get("ranks", [])]
            for n, p in rec.get("participants", {}).items()},
        "chunk_gap_median_before_s": round(gaps_before[len(gaps_before) // 2], 4)
        if gaps_before else None,
        "chunk_gap_max_during_prepare_s": round(max(gaps_prep), 4) if gaps_prep else None,
        "host_steps_prepare": r1.get("host", {}).get("steps_prepare"),
        "host_steps_before": r1.get("host", {}).get("steps_before"),
        "host_steps_during_txn": r1.get("host", {}).get("steps_during"),
        "held_steps": r1.get("host", {}).get("held_steps"),
        "single_after_commit": after_text, "banned_ids": ids, "banned_strs": sorted(strs),
        "commit_hashes_agree": (rec.get("commit_hashes") or {}).get("agree"),
        "findings": [(f.get("participant"), f["check"], f.get("accepted")) for f in
                     rec.get("findings", [])]}
    print("T1:", json.dumps(res["T1"], indent=1))

    # T2
    c_src = p_src
    old = "        do_penalty = use_penalty(sampling_params)\n"
    assert c_src.count(old) == 1
    c_new = c_src.replace(old, old + "        do_penalty = bool(do_penalty)\n")
    plan2 = {"id": f"t2-{int(time.time())}", "faults": {"worker": "validate"},
             "modules": {PEN_MOD: {
                 "on": ["worker"], "source": c_new, "base_hash": text_hash(c_src),
                 "targets": [["PenaltiesState.add_request",
                              line_of(c_src, "    def add_request(self, req_idx")]]}}}
    r2 = send(a.dir, {"op": "txn", "plan": plan2})
    st2 = send(a.dir, {"op": "status", "txn": plan2["id"]})
    serving = complete(url, m, SINGLE)
    rec2 = r2.get("receipt") or {}
    res["T2"] = {"outcome": r2.get("outcome"), "status": st2.get("state"),
                 "validate": {n: (p.get("validate") or {}).get("ok")
                              for n, p in rec2.get("participants", {}).items()},
                 "committed_by": [n for n, p in rec2.get("participants", {}).items()
                                  if "commit" in p],
                 "findings": [(f.get("participant"), f["check"]) for f in rec2.get("findings", [])],
                 "still_serving": serving == after_text, "gap_s": rec2.get("gap_s")}
    print("T2:", json.dumps(res["T2"], indent=1))

    # T3
    u = send(a.dir, {"op": "undo", "txn": plan["id"]})
    st3 = send(a.dir, {"op": "status", "txn": plan["id"]})
    again = complete(url, m, SINGLE)
    res["T3"] = {"undo_ok": u.get("ok"), "status": st3.get("state"),
                 "undo_s": (u.get("receipt") or {}).get("undo_s"),
                 "identical_to_base": again == base_text, "base": base_text, "after_undo": again}
    print("T3:", json.dumps(res["T3"], indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
