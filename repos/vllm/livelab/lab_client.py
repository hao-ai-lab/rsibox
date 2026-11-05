"""Measure a code change on a running `vllm serve`: time prefills on the original code, swap in
the change on every GPU worker, check that the new code ran, time again, then restore the original.

The server must run with VLLM_SERVER_DEV_MODE=1 and
`--worker-extension-cls fastrsi_lab_ext.LabExtension`, and listen on localhost only:
dev mode lets any caller run code in the engine.

Usage (one A/B on the warm server):
  python3 lab_client.py ab --url http://127.0.0.1:8000 --model <served-name> \
      --module vllm.model_executor.layers.sparse_attn_indexer --func sparse_attn_indexer \
      --patched <patched module source> --prefix 16384 --new 8192 --reps 5
Each step prints one JSON line prefixed with LAB. Standard library only.
"""
import argparse, json, random, statistics, time, urllib.request


def post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def rpc(url, method, *args):
    t = time.time()
    out = post(url, "/collective_rpc", {"method": method, "args": list(args)})
    return out, time.time() - t


def log(**kw):
    print("LAB " + json.dumps(kw), flush=True)


def prefill(url, model, tokens):
    """One request: prompt token ids, one output token. Returns seconds and the token."""
    t = time.time()
    out = post(url, "/v1/completions", {"model": model, "prompt": tokens, "max_tokens": 1,
                                        "temperature": 0.0, "logprobs": 1})
    return time.time() - t, out["choices"][0]["logprobs"]["tokens"][0]


def ab(a):
    t_exp = time.time()
    steps = {}

    def timed(name, fn, *args):
        t = time.time(); r = fn(*args); steps[name] = round(time.time() - t, 4); return r

    rng = random.Random(a.seed)
    prefix = [rng.randrange(1000, a.max_token_id) for _ in range(a.prefix)]
    news = [[rng.randrange(1000, a.max_token_id) for _ in range(a.new)] for _ in range(a.reps)]
    src = open(a.patched).read()

    def arm(name):
        # Reset the prefix cache and re-cache only the shared prefix, so every arm computes the
        # new tokens itself (otherwise later arms hit KV computed by earlier arms).
        post(a.url, "/reset_prefix_cache", {})
        prefill(a.url, a.model, prefix)
        times, toks = [], []
        for i, new in enumerate(news):          # the same prompts in every arm
            dt, tok = prefill(a.url, a.model, prefix + new)
            times.append(dt); toks.append(tok)
            log(event="prefill", arm=name, i=i, seconds=round(dt, 4), token=tok)
        return statistics.median(times), toks

    timed("reset_prefix_cache", post, a.url, "/reset_prefix_cache", {})
    timed("cache_prefix", prefill, a.url, a.model, prefix)
    warm = [rng.randrange(1000, a.max_token_id) for _ in range(a.new)]   # not used by any arm
    timed("warmup", lambda: [prefill(a.url, a.model, prefix + warm) for _ in range(2)])
    base, base_toks = timed("arm_base", arm, "base")
    out, dt = rpc(a.url, "fastrsi_install", a.module, a.func, src)
    log(event="install", seconds=round(dt, 4), ranks=out)
    try:
        log(event="witness_before", ranks=rpc(a.url, "fastrsi_witness", a.module)[0])
        edited, edit_toks = timed("arm_edited", arm, "edited")
        after = rpc(a.url, "fastrsi_witness", a.module)[0]
        log(event="witness_after", ranks=after)
        key = f"{a.module}.{a.func}"
        hits = [r["hits"].get(key, 0) for r in after["results"]]
        if min(hits) == 0:      # the edited arm did not run the new code on some rank
            raise SystemExit(f"witness: no calls of the new code on some rank: {hits}")
    finally:
        out, dt = rpc(a.url, "fastrsi_uninstall", a.module, a.func)
        log(event="uninstall", seconds=round(dt, 4), ranks=out)
    again, again_toks = timed("arm_base_again", arm, "base_again")
    log(event="summary", prefix=a.prefix, new=a.new, base_median=round(base, 4),
        edited_median=round(edited, 4), base_again_median=round(again, 4),
        speedup_pct=round(100 * (base / edited - 1), 1),
        same_tokens_edited_vs_base=sum(x == y for x, y in zip(base_toks, edit_toks)),
        same_tokens_base_again_vs_base=sum(x == y for x, y in zip(base_toks, again_toks)),
        reps=a.reps)
    # Per request, "seconds" in each prefill line is time to first token (max_tokens=1).
    log(event="timing", experiment_seconds=round(time.time() - t_exp, 2), steps=steps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ab"])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--func", required=True)
    p.add_argument("--patched", required=True)
    p.add_argument("--prefix", type=int, default=16384)
    p.add_argument("--new", type=int, default=8192)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-token-id", type=int, default=100000, help="below the model's vocab size")
    ab(p.parse_args())
