"""Drive a running `vllm serve` as a live lab: install, witness, uninstall, and time prefills.

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
    rng = random.Random(a.seed)
    prefix = [rng.randrange(1000, 100000) for _ in range(a.prefix)]
    news = [[rng.randrange(1000, 100000) for _ in range(a.new)] for _ in range(a.reps)]
    src = open(a.patched).read()

    def arm(name):
        times, toks = [], []
        for i, new in enumerate(news):          # the same prompts in every arm
            dt, tok = prefill(a.url, a.model, prefix + new)
            times.append(dt); toks.append(tok)
            log(event="prefill", arm=name, i=i, seconds=round(dt, 4), token=tok)
        return statistics.median(times), toks

    post(a.url, "/reset_prefix_cache", {})
    prefill(a.url, a.model, prefix)             # cache the prefix
    for _ in range(2):
        prefill(a.url, a.model, prefix + news[0])
    base, base_toks = arm("base")
    out, dt = rpc(a.url, "fastrsi_install", a.module, a.func, src)
    log(event="install", seconds=round(dt, 4), ranks=out)
    log(event="witness_before", ranks=rpc(a.url, "fastrsi_witness", a.module)[0])
    edited, edit_toks = arm("edited")
    log(event="witness_after", ranks=rpc(a.url, "fastrsi_witness", a.module)[0])
    out, dt = rpc(a.url, "fastrsi_uninstall", a.module, a.func)
    log(event="uninstall", seconds=round(dt, 4), ranks=out)
    again, again_toks = arm("base_again")
    log(event="summary", prefix=a.prefix, new=a.new, base_median=round(base, 4),
        edited_median=round(edited, 4), base_again_median=round(again, 4),
        speedup_pct=round(100 * (base / edited - 1), 1),
        same_tokens_edited_vs_base=sum(x == y for x, y in zip(base_toks, edit_toks)),
        same_tokens_base_again_vs_base=sum(x == y for x, y in zip(base_toks, again_toks)),
        reps=a.reps)


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
    ab(p.parse_args())
